#!/usr/bin/env python3
"""Build the continuous-prefix collector manifest for the geometry ROI gate.

The collector has a strict manifest schema.  Evaluation metadata therefore goes
to a separate ``evaluation_frames.json`` file and is never passed to it.

Only prefix frames from ``frame_id == 0`` through the largest selected
``sample_step`` are retained.  All frame and image hashes are checked against
the frozen parent collection before the output directory is created.  This
script is CPU-only and never imports the tracking model.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = ROOT / "results/geometry_roi_gate_manifest_20261024_v1/manifest.json"
DEFAULT_RUN = ROOT / "results/base_error_full_collection_20261023/full_collect_v1"
DEFAULT_OUT = ROOT / "results/geometry_roi_full_collect_manifest_20261024_v1"
DEFAULT_RUNNER = ROOT / "tmp_analysis/observe_geometry_roi_tokens_20261024.py"
DEFAULT_TRACE = ROOT / "tmp_analysis/roi_track_spatial_trace_20261024.py"
DEFAULT_CONFIG = DEFAULT_RUN / "collection_config.py"
DEFAULT_CHECKPOINT = (
    ROOT / "work/covtrack_tces_v2/saved_models/ctao_public_res/ctao_public.pth"
)

SCHEMA_VERSION = "OVMOT_GEOMETRY_ROI_TOKEN_MANIFEST_V1"
FROZEN_STATUS = "FROZEN_GEOMETRY_ROI_FULL_PREFIX_V1"
EXPECTED_FRAME_COUNT = 276
EXPECTED_EVALUATION_FRAME_COUNT = 72
EXPECTED_ANCHOR_COUNT = 122

COLLECTOR_KEYS = frozenset({
    "schema_version",
    "status",
    "runner_sha256",
    "trace_path",
    "GPU_UUID",
    "output_directory",
    "parent_collection_manifest",
    "code_root",
    "config_path",
    "checkpoint_path",
    "input_pins",
    "cache_member_allowlist",
    "token_dtype",
    "runtime",
    "frames",
})
FRAME_KEYS = frozenset({
    "source",
    "video_id",
    "image_id",
    "frame_id",
    "cache_path",
    "cache_sha256",
    "image_path",
    "image_sha256",
})
CACHE_MEMBER_ALLOWLIST = (
    "track_feats",
    "tracker_keep_mask",
    "pretracker_det_row",
    "det_bboxes",
    "pretracker_det_bboxes",
    "roi_proposal_row",
)
FORBIDDEN_COLLECTOR_KEYS = (
    "gt",
    "ground_truth",
    "annotation",
    "manual_anchor",
    "identity",
    "track_id",
    "category",
    "posterior",
    "matching",
)


class ManifestBuildError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ManifestBuildError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict),
                    f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def normalized_absolute(path: Path, field: str) -> str:
    path = Path(path)
    require(path.is_absolute(), f"{field} must be absolute: {path}")
    text = str(path)
    require(text == os.path.normpath(text),
            f"{field} must be normalized: {text}")
    return text


def integer(row: Mapping[str, Any], key: str, where: str) -> int:
    value = row.get(key)
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{where}: {key} must be an integer")
    return value


def text(row: Mapping[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    require(isinstance(value, str) and value,
            f"{where}: {key} must be a nonempty string")
    return value


def iter_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, key
            yield from iter_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_keys(child, f"{prefix}[{index}]")


def no_forbidden_keys(value: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for path, key in iter_keys(value):
        lowered = str(key).lower()
        if any(token in lowered for token in FORBIDDEN_COLLECTOR_KEYS):
            found.append(path)
    return found


def no_gate_input_leaks(gate_input: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject labels while allowing frozen model outputs.

    ``local_track_id`` is the pretracker row identity frozen in the parent
    collection.  It is not a GT identity and is retained only in the separate
    evaluation file.  ``pred_category_id`` is likewise a model output.
    """
    found: list[str] = []
    for index, row in enumerate(gate_input):
        for path, key in iter_keys(row, f"model_input[{index}]"):
            lowered = str(key).lower()
            if (lowered.startswith("gt_") or "ground_truth" in lowered or
                    "annotation" in lowered or "manual_anchor" in lowered or
                    "identity" in lowered or
                    (lowered in {"track_id", "global_track_id"})):
                found.append(path)
    return found


def validate_gate(gate: Mapping[str, Any]) -> None:
    require(gate.get("status") == "PASS_GEOMETRY_ROI_GATE_MANIFEST",
            "Unexpected gate manifest status")
    require(integer(gate, "schema_version", "gate") == 1,
            "Unexpected gate manifest version")
    sources = gate.get("selected_sources")
    folds = gate.get("folds")
    model_input = gate.get("model_input")
    labels = gate.get("evaluation_only_labels")
    require(isinstance(sources, list) and len(sources) == 6,
            "Gate manifest must select six sources")
    require(isinstance(folds, list) and len(folds) == 3,
            "Gate manifest must contain three folds")
    require(isinstance(model_input, list) and
            len(model_input) == EXPECTED_ANCHOR_COUNT,
            "Gate manifest must contain 122 anchors")
    require(isinstance(labels, list) and len(labels) == EXPECTED_ANCHOR_COUNT,
            "Gate manifest must contain 122 evaluation-only labels")
    require(no_gate_input_leaks(model_input) == [],
            "Gate model_input contains identity/category/GT fields")
    anchor_ids = [integer(row, "observation_id", "model_input") for row in model_input]
    label_ids = [integer(row, "observation_id", "evaluation_only_labels")
                 for row in labels]
    require(len(anchor_ids) == len(set(anchor_ids)),
            "Gate anchors repeat observation IDs")
    require(anchor_ids == label_ids,
            "Gate anchor and evaluation-label rows are not aligned")
    step_keys = {
        (text(row, "source_group", "model_input"),
         integer(row, "video_id", "model_input"),
         integer(row, "sampled_step", "model_input"))
        for row in model_input
    }
    require(len(step_keys) == EXPECTED_EVALUATION_FRAME_COUNT,
            "Gate anchors do not cover exactly 72 evaluation steps")
    pair_counts = gate.get("overall_pairs")
    require(isinstance(pair_counts, dict), "Gate pair support is missing")
    for key in (
        "known_same_gt_cross_step",
        "known_different_gt_cross_step",
        "known_different_gt_same_pred_category",
    ):
        require(integer(pair_counts, key, "overall_pairs") >= 1,
                f"No support for {key}")


def validate_parent_run(
    run_dir: Path,
    gate: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[str, dict[int, dict[str, Any]]],
]:
    execution_path = run_dir / "frame_execution.jsonl"
    inventory_path = run_dir / "dump_inventory.json"
    parent_manifest_path = run_dir / "run_manifest.json"
    require(execution_path.is_file(), f"Missing {execution_path}")
    require(inventory_path.is_file(), f"Missing {inventory_path}")
    require(parent_manifest_path.is_file(), f"Missing {parent_manifest_path}")
    rows = read_jsonl(execution_path)
    inventory = read_json(inventory_path)
    parent = read_json(parent_manifest_path)
    require(isinstance(parent.get("source_hashes"), dict) and
            parent["source_hashes"], "Parent run lacks frozen source hashes")
    for relative, digest in parent["source_hashes"].items():
        source_path = Path(parent["code_root"]) / relative
        require(source_path.is_file(), f"Frozen source is missing: {source_path}")
        require(sha256_path(source_path) == digest,
                f"Frozen source changed: {source_path}")

    by_video: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    image_ids: set[int] = set()
    for index, row in enumerate(rows):
        where = f"frame_execution[{index}]"
        source_video = integer(row, "video_id", where)
        frame_id = integer(row, "frame_id", where)
        image_id = integer(row, "image_id", where)
        require(image_id not in image_ids,
                f"Parent execution repeats image_id {image_id}")
        image_ids.add(image_id)
        require(frame_id not in by_video[str(source_video)],
                f"Parent execution repeats video/frame {source_video}/{frame_id}")
        by_video[str(source_video)][frame_id] = row
        npz_name = Path(text(row, "npz", where)).name
        npz_sha = text(row, "npz_sha256", where)
        require(inventory.get(npz_name) == npz_sha,
                f"Parent inventory disagrees with {npz_name}")
        require((run_dir / "dump" / npz_name).is_file(),
                f"Missing parent cache {npz_name}")

    # The gate references the same parent inventory through its own provenance.
    gate_pins = gate.get("input_pins")
    require(isinstance(gate_pins, dict), "Gate manifest lacks input pins")
    inventory_key = str(inventory_path.relative_to(ROOT))
    require(gate_pins.get(inventory_key) == sha256_path(inventory_path),
            "Gate inventory pin differs from the live parent inventory")
    return rows, inventory, by_video


def gate_video_plan(
    gate: Mapping[str, Any],
) -> list[tuple[str, int, list[int]]]:
    plan: list[tuple[str, int, list[int]]] = []
    for source in gate["selected_sources"]:
        source_group = text(source, "source_group", "selected_sources")
        videos = source.get("videos")
        require(isinstance(videos, list) and len(videos) == 2,
                f"{source_group} must have two selected videos")
        for video in videos:
            video_id = integer(video, "video_id", "selected_videos")
            steps = video.get("sample_steps")
            require(isinstance(steps, list) and len(steps) == 6 and
                    steps == sorted(set(steps)),
                    f"Invalid sample steps for {source_group}/{video_id}")
            require(all(isinstance(step, int) and not isinstance(step, bool)
                        and step >= 0 for step in steps),
                    "Sample steps must be nonnegative integers")
            plan.append((source_group, video_id, list(steps)))
    require(len(plan) == 12, "The gate must select 12 videos")
    require(len({video_id for _source, video_id, _steps in plan}) == 12,
            "Selected video IDs are not globally unique")
    return plan


def frame_receipt(
    run_dir: Path,
    inventory: Mapping[str, str],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    npz_path = Path(text(row, "npz", "frame_execution"))
    require(npz_path.is_absolute(), "Parent cache path is not absolute")
    name = npz_path.name
    receipt_path = run_dir / "dump" / (npz_path.stem + ".receipt.json")
    require(receipt_path.is_file(), f"Missing frame receipt {receipt_path}")
    require(inventory.get(receipt_path.name) == sha256_path(receipt_path),
            f"Receipt hash disagrees with inventory: {receipt_path.name}")
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "PASS_FRAME_DUMP",
            f"Invalid frame receipt status: {receipt_path}")
    require(receipt.get("file") == name, "Frame receipt filename mismatch")
    require(integer(receipt, "video_id", "frame_receipt") ==
            integer(row, "video_id", "frame_execution"),
            "Frame receipt video mismatch")
    require(integer(receipt, "image_id", "frame_receipt") ==
            integer(row, "image_id", "frame_execution"),
            "Frame receipt image mismatch")
    require(integer(receipt, "frame_id", "frame_receipt") ==
            integer(row, "frame_id", "frame_execution"),
            "Frame receipt frame mismatch")
    cache_sha = sha256_path(npz_path)
    require(cache_sha == text(row, "npz_sha256", "frame_execution"),
            f"Live cache hash differs from execution: {name}")
    require(receipt.get("sha256") == cache_sha,
            f"Live cache hash differs from receipt: {name}")
    image_path = Path(text(receipt, "filename", "frame_receipt"))
    require(image_path.is_absolute() and image_path.is_file(),
            f"Missing image: {image_path}")
    with np.load(npz_path, allow_pickle=False) as cache:
        require("tracker_keep_mask" in cache.files,
                f"Cache lacks tracker_keep_mask: {name}")
        keep = np.asarray(cache["tracker_keep_mask"])
        require(keep.dtype == np.dtype("bool") and keep.ndim == 1,
                f"Invalid tracker_keep_mask: {name}")
        retained_rows = int(np.count_nonzero(keep))
    record = {
        "source": "",
        "video_id": integer(row, "video_id", "frame_execution"),
        "image_id": integer(row, "image_id", "frame_execution"),
        "frame_id": integer(row, "frame_id", "frame_execution"),
        "cache_path": normalized_absolute(npz_path, "cache_path"),
        "cache_sha256": cache_sha,
        "image_path": normalized_absolute(image_path, "image_path"),
        "image_sha256": sha256_path(image_path),
    }
    return record, retained_rows


def build_prefix_frames(
    gate: Mapping[str, Any],
    run_dir: Path,
    inventory: Mapping[str, str],
    by_video: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    retained_rows = 0
    video_summary: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    for source_group, video_id, sample_steps in gate_video_plan(gate):
        maximum_step = max(sample_steps)
        source_rows = by_video.get(str(video_id))
        require(source_rows is not None, f"Parent cache has no video {video_id}")
        available = sorted(frame_id for frame_id in source_rows if frame_id <= maximum_step)
        expected = list(range(maximum_step + 1))
        require(available == expected,
                f"Video {video_id} prefix is not continuous: "
                f"missing={sorted(set(expected) - set(available))[:5]}, "
                f"extra={sorted(set(available) - set(expected))[:5]}")
        retained_in_video = 0
        for frame_id in expected:
            record, rows = frame_receipt(run_dir, inventory, source_rows[frame_id])
            record["source"] = source_group
            require(record["image_id"] not in seen_image_ids,
                    f"Prefix repeats globally unique image_id {record['image_id']}")
            seen_image_ids.add(record["image_id"])
            frames.append(record)
            retained_rows += rows
            retained_in_video += rows
        video_summary.append({
            "source": source_group,
            "video_id": video_id,
            "evaluation_steps": sample_steps,
            "prefix_max_frame_id": maximum_step,
            "prefix_frame_count": maximum_step + 1,
            "retained_rows": retained_in_video,
        })
    return frames, retained_rows, {
        "videos": video_summary,
        "frame_count": len(frames),
        "retained_rows": retained_rows,
    }


def evaluation_records(
    gate: Mapping[str, Any],
    prefix_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_frame = {
        (str(frame["source"]), int(frame["video_id"]), int(frame["frame_id"])): frame
        for frame in prefix_frames
    }
    by_image = {int(frame["image_id"]): frame for frame in prefix_frames}
    checks = gate["validation"]["frames"]["checks"]
    require(isinstance(checks, list) and len(checks) == EXPECTED_EVALUATION_FRAME_COUNT,
            "Gate validation must contain 72 evaluation frames")
    evaluation_frames: list[dict[str, Any]] = []
    evaluation_steps: dict[str, list[dict[str, Any]]] = {}
    check_by_step: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for item in checks:
        key = (
            text(item, "source_group", "gate evaluation frame"),
            integer(item, "video_id", "gate evaluation frame"),
            integer(item, "sampled_step", "gate evaluation frame"),
        )
        require(key not in check_by_step, "Gate repeats an evaluation step")
        image_id = integer(item, "image_id", "gate evaluation frame")
        frame = by_image.get(image_id)
        require(frame is not None, f"Evaluation image {image_id} is outside prefix")
        require((frame["source"], frame["video_id"], frame["frame_id"]) == key,
                "Evaluation image maps to the wrong source/video/frame")
        record = {
            "evaluation_index": len(evaluation_frames),
            "source": key[0],
            "video_id": key[1],
            "sample_step": key[2],
            "image_id": image_id,
            "frame_id": key[2],
            "cache_path": frame["cache_path"],
            "cache_sha256": frame["cache_sha256"],
            "image_path": frame["image_path"],
            "image_sha256": frame["image_sha256"],
        }
        evaluation_frames.append(record)
        check_by_step[key] = item
        video_key = f"{key[0]}:{key[1]}"
        evaluation_steps.setdefault(video_key, []).append({
            "sample_step": key[2],
            "image_id": image_id,
            "frame_id": key[2],
            "evaluation_index": record["evaluation_index"],
        })
    require(len(evaluation_frames) == EXPECTED_EVALUATION_FRAME_COUNT,
            "Evaluation frame count changed")
    require(len(evaluation_steps) == 12, "Evaluation step mapping must cover 12 videos")
    for values in evaluation_steps.values():
        require([value["sample_step"] for value in values] ==
                sorted(value["sample_step"] for value in values),
                "Evaluation steps are not sorted")

    anchors = copy.deepcopy(gate["model_input"])
    require(len(anchors) == EXPECTED_ANCHOR_COUNT, "Anchor count changed")
    anchor_steps: set[tuple[str, int, int]] = set()
    for anchor in anchors:
        key = (
            text(anchor, "source_group", "anchor"),
            integer(anchor, "video_id", "anchor"),
            integer(anchor, "sampled_step", "anchor"),
        )
        image_id = integer(anchor, "image_id", "anchor")
        frame = by_image.get(image_id)
        require(key in check_by_step and frame is not None,
                f"Anchor does not map to an evaluation frame: {key}")
        require((frame["source"], frame["video_id"], frame["frame_id"]) == key,
                "Anchor maps to the wrong evaluation frame")
        anchor_steps.add(key)
    require(anchor_steps == set(check_by_step),
            "Anchors do not cover all 72 evaluation frames")

    labels = copy.deepcopy(gate["evaluation_only_labels"])
    require(len(labels) == EXPECTED_ANCHOR_COUNT, "Label count changed")
    recomputed = recompute_observation_pairs(gate)
    generator_style = recompute_observation_pairs(gate, observed_rules=True)
    recorded = copy.deepcopy(gate["overall_pairs"])
    fold_recorded = {
        key: sum(integer(fold["pairs"], key, "fold pairs")
                 for fold in gate["folds"])
        for key in (
            "known_same_gt_cross_step",
            "known_different_gt_cross_step",
            "known_different_gt_same_pred_category",
            "unknown_identity_pairs_excluded",
        )
    }
    frozen_fold_sum_match = recorded == fold_recorded
    return {
        "status": "PASS_GEOMETRY_ROI_FULL_PREFIX_EVALUATION_MANIFEST",
        "schema_version": 1,
        "protocol": {
            "source": "frozen geometry ROI gate manifest",
            "evaluation_order": "gate validation frame order",
            "evaluation_steps_per_video": 6,
            "evaluation_frames": EXPECTED_EVALUATION_FRAME_COUNT,
            "anchors": EXPECTED_ANCHOR_COUNT,
            "collector_separation": (
                "This file is evaluation-only and must never be passed as the "
                "collector manifest."
            ),
            "unknown_identity_policy": "exclude; never a negative pair",
        },
        "pair_support": {
            "frozen_gate_recorded": recorded,
            "recomputed_cross_step_observation_pairs": recomputed,
            "recomputed_with_frozen_gate_constructor": generator_style,
            "frozen_fold_pair_sum": fold_recorded,
            "frozen_fold_sum_matches_overall": frozen_fold_sum_match,
            "interpretation": (
                "frozen_gate_recorded equals the frozen fold sums. The manifest "
                "builder's fold constructor applies cross-step gating to every "
                "pair but its top-level count applies it only to the same-"
                "identity branch, so different-identity and same-category "
                "counts are not stable. Use evaluation_frames.json's anchor "
                "rows for any source-OOF pair analysis and recompute the "
                "denominator under an explicit protocol."
            ),
        },
        "fold_pair_support": [
            {
                "fold": integer(fold, "fold", "fold"),
                "sources": list(fold["sources"]),
                "frozen_gate_recorded": copy.deepcopy(fold["pairs"]),
                "recomputed_cross_step_observation_pairs":
                    recompute_observation_pairs(gate, fold_sources=fold["sources"]),
                "recomputed_with_frozen_gate_constructor":
                    recompute_observation_pairs(
                        gate, fold_sources=fold["sources"], observed_rules=True),
            }
            for fold in gate["folds"]
        ],
        "evaluation_steps": evaluation_steps,
        "evaluation_frames": evaluation_frames,
        "anchors": anchors,
        "evaluation_only_labels": labels,
        "no_gt_in_collector_input": True,
    }


def recompute_observation_pairs(
    gate: Mapping[str, Any],
    *,
    fold_sources: Sequence[str] | None = None,
    observed_rules: bool = False,
) -> dict[str, int]:
    """Recompute known-identity cross-step observation pairs from frozen rows."""
    labels = {
        integer(row, "observation_id", "evaluation_only_labels"): row
        for row in gate["evaluation_only_labels"]
    }
    allowed_sources = None if fold_sources is None else set(fold_sources)
    by_video: dict[tuple[str, int], list[Mapping[str, Any]]] = \
        collections.defaultdict(list)
    for row in gate["model_input"]:
        source = text(row, "source_group", "model_input")
        if allowed_sources is not None and source not in allowed_sources:
            continue
        by_video[(source, integer(row, "video_id", "model_input"))].append(row)
    same = different = same_pred = unknown = 0
    for rows in by_video.values():
        for first, second in itertools.combinations(rows, 2):
            if (observed_rules is False and
                    integer(first, "sampled_step", "anchor") ==
                    integer(second, "sampled_step", "anchor")):
                continue
            a = labels[integer(first, "observation_id", "anchor")]["gt_track_id"]
            b = labels[integer(second, "observation_id", "anchor")]["gt_track_id"]
            if a is None or b is None:
                unknown += 1
            elif int(a) == int(b):
                same += 1
            else:
                different += 1
                if (observed_rules is False and
                        integer(first, "pred_category_id", "anchor") ==
                        integer(second, "pred_category_id", "anchor")):
                    same_pred += 1
    return {
        "known_same_gt_cross_step": same,
        "known_different_gt_cross_step": different,
        "known_different_gt_same_pred_category": same_pred,
        "unknown_identity_pairs_excluded": unknown,
    }


def estimate_disk(
    run_dir: Path,
    gate: Mapping[str, Any],
    prefix_retained_rows: int,
) -> dict[str, Any]:
    """Estimate token bytes from frozen prior collections without new inference."""
    candidates: list[tuple[Path, int, int]] = []
    for directory in sorted(Path("/raid/datasets/chc_data").glob(
            "geometry_roi_pilot_20261024_*")):
        receipt_path = directory / "collection_receipt.json"
        if not receipt_path.is_file():
            continue
        try:
            receipt = read_json(receipt_path)
            rows = integer(receipt, "retained_rows", "pilot receipt")
        except Exception:
            continue
        token_paths = list(directory.glob("*.npz"))
        token_bytes = sum(path.stat().st_size for path in token_paths)
        if token_bytes and rows:
            candidates.append((directory, rows, token_bytes))
    if candidates:
        directory, rows, token_bytes = max(
            candidates,
            key=lambda item: item[0].stat().st_mtime,
        )
        estimate = int(round(token_bytes / rows * prefix_retained_rows))
        baseline = {
            "source": str(directory),
            "retained_rows": rows,
            "token_npz_bytes": token_bytes,
            "bytes_per_retained_row": token_bytes / rows,
        }
    else:
        # Two float16 [256,7,7] arrays per retained row, before compression.
        bytes_per_row = 2 * 256 * 7 * 7 * 2
        estimate = bytes_per_row * prefix_retained_rows
        baseline = {
            "source": "dtype layout",
            "retained_rows": None,
            "token_npz_bytes": None,
            "bytes_per_retained_row": bytes_per_row,
            "note": "uncompressed lower-bound layout size",
        }
    return {
        "prefix_retained_rows": prefix_retained_rows,
        "estimated_token_npz_bytes": estimate,
        "estimated_token_npz_gib": estimate / (1024 ** 3),
        "estimate_basis": baseline,
        "estimate_excludes_temporary_peak_and_receipts": True,
    }


def build_collector_manifest(
    *,
    frames: Sequence[Mapping[str, Any]],
    runner: Path,
    trace: Path,
    run_manifest: Path,
    config: Path,
    checkpoint: Path,
    gate_manifest: Path,
    execution: Path,
    inventory_path: Path,
    output_directory: Path,
    gpu_uuid: str,
    python: Path,
    seed: int,
) -> dict[str, Any]:
    runner = runner.resolve()
    trace = trace.resolve()
    run_manifest = run_manifest.resolve()
    config = config.resolve()
    checkpoint = checkpoint.resolve()
    gate_manifest = gate_manifest.resolve()
    execution = execution.resolve()
    inventory_path = inventory_path.resolve()
    output_directory = output_directory.resolve()
    require(not output_directory.exists(),
            f"Collector output directory already exists: {output_directory}")
    require(output_directory.parent.is_dir(),
            f"Collector output parent is missing: {output_directory.parent}")
    require(gpu_uuid.startswith("GPU-"), "GPU UUID must be an nvidia-smi UUID")
    pinned_paths = (
        runner,
        trace,
        run_manifest,
        config,
        checkpoint,
        gate_manifest,
        execution,
        inventory_path,
    )
    for path in pinned_paths:
        require(path.is_file(), f"Pinned input is missing: {path}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": FROZEN_STATUS,
        "runner_sha256": sha256_path(runner),
        "trace_path": normalized_absolute(trace, "trace_path"),
        "GPU_UUID": gpu_uuid,
        "output_directory": normalized_absolute(output_directory, "output_directory"),
        "parent_collection_manifest": normalized_absolute(
            run_manifest, "parent_collection_manifest"),
        "code_root": read_json(run_manifest)["code_root"],
        "config_path": normalized_absolute(config, "config_path"),
        "checkpoint_path": normalized_absolute(checkpoint, "checkpoint_path"),
        "input_pins": {
            normalized_absolute(path, "input_pins key"): sha256_path(path)
            for path in pinned_paths
        },
        "cache_member_allowlist": list(CACHE_MEMBER_ALLOWLIST),
        "token_dtype": "float16",
        "runtime": {
            "python": normalized_absolute(python.resolve(), "runtime.python"),
            "seed": int(seed),
            "threads": 1,
        },
        "frames": [dict(frame) for frame in frames],
    }
    require(set(manifest) == COLLECTOR_KEYS, "Collector top-level schema changed")
    for index, frame in enumerate(manifest["frames"]):
        require(set(frame) == FRAME_KEYS,
                f"Collector frame {index} schema changed")
    require(no_forbidden_keys(manifest) == [],
            "Collector manifest contains forbidden GT/identity/class keys")
    return manifest


def validate_runner_contract(
    manifest: Mapping[str, Any],
    runner: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(runner.parent))
    try:
        import observe_geometry_roi_tokens_20261024 as collector
    finally:
        sys.path.pop(0)
    checked = collector.validate_manifest_structure(
        copy.deepcopy(manifest), check_runner=True)
    require(checked["schema_version"] == SCHEMA_VERSION,
            "Collector schema version mismatch")
    return {
        "status": "PASS_COLLECTOR_SCHEMA_AND_RUNNER",
        "runner_sha256": checked["runner_sha256"],
        "frame_count": len(checked["frames"]),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name("." + path.name + ".tmp")
    require(not path.exists() and not temporary.exists(),
            f"Refusing to overwrite {path}")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build(
    *,
    gate_path: Path = DEFAULT_GATE,
    run_dir: Path = DEFAULT_RUN,
    out_dir: Path = DEFAULT_OUT,
    runner_path: Path = DEFAULT_RUNNER,
    trace_path: Path = DEFAULT_TRACE,
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    collector_output: Path = Path(
        "/raid/datasets/chc_data/geometry_roi_full_20261024_v3"),
    gpu_uuid: str = "GPU-1b297aba-ae7e-e326-5903-476f1bb683d7",
    python: Path = Path(sys.executable),
    seed: int = 20261024,
) -> dict[str, Any]:
    started = time.monotonic()
    gate_path = gate_path.resolve()
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    require(not out_dir.exists(), f"Output directory already exists: {out_dir}")
    require(out_dir.parent.is_dir(), f"Output parent is missing: {out_dir.parent}")
    gate = read_json(gate_path)
    validate_gate(gate)
    _rows, inventory, by_video = validate_parent_run(run_dir, gate)
    frames, retained_rows, prefix_summary = build_prefix_frames(
        gate, run_dir, inventory, by_video)
    require(len(frames) == EXPECTED_FRAME_COUNT,
            f"Expected 276 prefix frames, got {len(frames)}")
    evaluation = evaluation_records(gate, frames)
    collector_manifest = build_collector_manifest(
        frames=frames,
        runner=runner_path,
        trace=trace_path,
        run_manifest=run_dir / "run_manifest.json",
        config=config_path,
        checkpoint=checkpoint_path,
        gate_manifest=gate_path,
        execution=run_dir / "frame_execution.jsonl",
        inventory_path=run_dir / "dump_inventory.json",
        output_directory=collector_output,
        gpu_uuid=gpu_uuid,
        python=python,
        seed=seed,
    )
    runner_check = validate_runner_contract(collector_manifest, runner_path)
    disk = estimate_disk(run_dir, gate, retained_rows)

    evaluation["collector_manifest_sha256"] = hashlib.sha256(
        json.dumps(collector_manifest, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    evaluation["input_pins"] = copy.deepcopy(collector_manifest["input_pins"])
    evaluation["prefix_summary"] = prefix_summary
    receipt = {
        "status": "PASS_GEOMETRY_ROI_FULL_COLLECTOR_MANIFEST",
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "prefix": "frame_id 0 through per-video maximum selected sample_step",
            "frame_order": "source order, video order, then increasing frame_id",
            "collector_frames": EXPECTED_FRAME_COUNT,
            "evaluation_frames": EXPECTED_EVALUATION_FRAME_COUNT,
            "anchors": EXPECTED_ANCHOR_COUNT,
            "existing_manifest_modified": False,
            "gpu_or_model_used": False,
        },
        "prefix_summary": prefix_summary,
        "pair_support": copy.deepcopy(evaluation["pair_support"]),
        "disk_estimate": disk,
        "runner_contract": runner_check,
        "collector_output_directory": collector_manifest["output_directory"],
        "limitations": [
            "This is a provenance and sampling artifact, not method-effect evidence.",
            "Disk is an estimate based on frozen token caches; collection is not run.",
            "Pair support counts are not independent source or video observations.",
        ],
        "no_gpu_used": True,
        "no_model_loaded": True,
        "no_training_started": True,
        "elapsed_seconds": time.monotonic() - started,
    }

    temporary_parent = Path(tempfile.mkdtemp(
        prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        atomic_write_json(temporary_parent / "collector_manifest.json",
                          collector_manifest)
        atomic_write_json(temporary_parent / "evaluation_frames.json",
                          evaluation)
        atomic_write_json(temporary_parent / "generation_receipt.json", receipt)
        checksum_path = temporary_parent / "SHA256SUMS"
        checksum_payload = "".join(
            f"{sha256_path(temporary_parent / name)}  {name}\n"
            for name in (
                "collector_manifest.json",
                "evaluation_frames.json",
                "generation_receipt.json",
            )
        )
        checksum_path.write_text(checksum_payload, encoding="utf-8")
        require(not out_dir.exists(),
                f"Output appeared during construction: {out_dir}")
        os.replace(temporary_parent, out_dir)
    finally:
        if temporary_parent.exists():
            for child in temporary_parent.iterdir():
                child.unlink()
            temporary_parent.rmdir()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--collector-output",
        type=Path,
        default=Path("/raid/datasets/chc_data/geometry_roi_full_20261024_v3"),
    )
    parser.add_argument(
        "--gpu-uuid",
        default="GPU-1b297aba-ae7e-e326-5903-476f1bb683d7",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=20261024)
    arguments = parser.parse_args(argv)
    try:
        receipt = build(
            gate_path=arguments.gate,
            run_dir=arguments.run_dir,
            out_dir=arguments.out,
            runner_path=arguments.runner,
            trace_path=arguments.trace,
            config_path=arguments.config,
            checkpoint_path=arguments.checkpoint,
            collector_output=arguments.collector_output,
            gpu_uuid=arguments.gpu_uuid,
            python=arguments.python,
            seed=arguments.seed,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL_GEOMETRY_ROI_FULL_COLLECTOR_MANIFEST",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": receipt["status"],
        "output": str(arguments.out.resolve()),
        "frame_count": receipt["prefix_summary"]["frame_count"],
        "retained_rows": receipt["prefix_summary"]["retained_rows"],
        "estimated_token_npz_gib": receipt["disk_estimate"]["estimated_token_npz_gib"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
