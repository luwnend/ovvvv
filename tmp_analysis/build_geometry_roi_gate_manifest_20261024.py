#!/usr/bin/env python3
"""Build a deterministic FIT-only manifest for the geometry ROI token gate.

The selector never reads accuracy, category correctness, identity similarity, or
feature outcomes.  GT track IDs are used only after selection to count the
required support pairs.  In particular, GT is never copied into model inputs and
unknown identities never enter a negative-pair denominator.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPORT = ROOT / "results/base_error_full_collection_20261023/support_v1"
DEFAULT_RUN = ROOT / "results/base_error_full_collection_20261023/full_collect_v1"
DEFAULT_OUT = ROOT / "results/geometry_roi_gate_manifest_20261024_v1"
SEED = 20261024
SOURCE_COUNT = 6
VIDEO_COUNT = 2
STEP_COUNT = 6
FOLD_COUNT = 3
MIN_PAIRS_PER_FOLD = 30
FORBIDDEN_MODEL_INPUT_KEYS = (
    "gt_",
    "ground_truth",
    "manual_anchor",
    "annotation",
    "match_iou",
    "soft_risk",
    "error",
)


class ManifestError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def stable_digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    require(path.is_file(), "Missing JSONL input: %s" % path)
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("%s:%d: %s" % (path, line_no, exc)) from exc
            require(isinstance(row, dict), "%s:%d is not an object" % (path, line_no))
            yield row


def require_int(row: Mapping[str, Any], key: str, where: str) -> int:
    value = row.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool),
        "%s: %s must be an integer, got %r" % (where, key, value),
    )
    return value


def require_text(row: Mapping[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    require(isinstance(value, str) and value, "%s: %s must be nonempty" % (where, key))
    return value


def load_eligible_supervision(path: Path) -> dict[str, dict[str, Any]]:
    """Return FIT eligible manual anchors keyed by unique observation ID."""
    eligible: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(read_jsonl(path), 1):
        where = "%s line %d" % (path.name, line_no)
        if row.get("split") != "fit" or row.get("eligible_for_parameter_fit") is not True:
            continue
        observation_id = require_int(row, "observation_id", where)
        video_id = require_int(row, "video_id", where)
        image_id = require_int(row, "image_id", where)
        step = require_int(row, "sampled_step", where)
        frame_index = require_int(row, "original_frame_index", where)
        local_track_id = require_int(row, "local_track_id", where)
        gt_track_id = require_int(row, "gt_track_id", where)
        manual_anchor_id = require_int(row, "manual_anchor_id", where)
        pred_category_id = require_int(row, "pred_category_id", where)
        source = require_text(row, "source_group", where)
        key = str(observation_id)
        require(key not in eligible, "Duplicate eligible observation_id %s" % key)
        eligible[key] = {
            "observation_id": observation_id,
            "source_group": source,
            "video_id": video_id,
            "image_id": image_id,
            "sampled_step": step,
            "original_frame_index": frame_index,
            "local_track_id": local_track_id,
            "pred_category_id": pred_category_id,
            # Selection code receives this mapping but never sorts by or filters
            # on these evaluation-only values.
            "gt_track_id": gt_track_id,
            "gt_category_id": row.get("gt_category_id"),
            "manual_anchor_id": manual_anchor_id,
        }
    require(eligible, "No FIT eligible manual anchors")
    return eligible


def join_gt_free(
    path: Path, eligible: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Keep only GT-free rows for eligible anchors and validate their structure."""
    joined: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(read_jsonl(path), 1):
        observation_id = row.get("observation_id")
        if not isinstance(observation_id, int) or isinstance(observation_id, bool):
            continue
        anchor = eligible.get(str(observation_id))
        if anchor is None:
            continue
        where = "%s line %d" % (path.name, line_no)
        source = require_text(row, "source_group", where)
        video_id = require_int(row, "video_id", where)
        image_id = require_int(row, "image_id", where)
        step = require_int(row, "sampled_step", where)
        frame_index = require_int(row, "original_frame_index", where)
        local_track_id = require_int(row, "local_track_id", where)
        pred_category_id = require_int(row, "pred_category_id", where)
        dump_file = require_text(row, "dump_file", where)
        dump_sha256 = require_text(row, "dump_sha256", where)
        dump_row = require_int(row, "dump_row", where)
        require(source == anchor["source_group"], "Source mismatch at " + where)
        require(video_id == anchor["video_id"], "Video mismatch at " + where)
        require(image_id == anchor["image_id"], "Image mismatch at " + where)
        require(step == anchor["sampled_step"], "Step mismatch at " + where)
        require(frame_index == anchor["original_frame_index"], "Frame mismatch at " + where)
        require(local_track_id == anchor["local_track_id"], "Track mismatch at " + where)
        require(pred_category_id == anchor["pred_category_id"], "Label mismatch at " + where)
        key = str(observation_id)
        require(key not in joined, "Duplicate GT-free observation_id %s" % key)
        joined[key] = {
            "observation_id": observation_id,
            "dump_file": dump_file,
            "dump_sha256": dump_sha256,
            "dump_row": dump_row,
        }
    missing = sorted(set(eligible) - set(joined))
    require(not missing, "Eligible observations missing GT-free locators: %s" % missing[:10])
    return joined


def evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    """Choose count sorted values including both ends without consulting outcomes."""
    ordered = sorted(set(values))
    require(len(ordered) >= count, "Need %d steps, found %d" % (count, len(ordered)))
    indices = [(i * (len(ordered) - 1)) // (count - 1) for i in range(count)]
    return [ordered[index] for index in indices]


def selection_plan(
    eligible: Mapping[str, Mapping[str, Any]], seed: int = SEED
) -> dict[str, Any]:
    """Hash-rank sources/videos and evenly select steps from structural metadata only."""
    by_source: dict[str, dict[int, list[Mapping[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in eligible.values():
        by_source[str(row["source_group"])][int(row["video_id"])].append(row)

    source_candidates: list[tuple[str, str, list[tuple[str, int, list[int]]]]] = []
    for source, videos in by_source.items():
        ordered_videos: list[tuple[str, int, list[int]]] = []
        for video_id, rows in videos.items():
            steps = sorted({int(row["sampled_step"]) for row in rows})
            if len(steps) >= STEP_COUNT:
                ordered_videos.append(
                    (stable_digest(seed, "video", source, video_id), video_id, evenly_spaced(steps, STEP_COUNT))
                )
        ordered_videos.sort()
        if len(ordered_videos) >= VIDEO_COUNT:
            source_candidates.append(
                (stable_digest(seed, "source", source), source, ordered_videos[:VIDEO_COUNT])
            )
    source_candidates.sort()
    return {
        "eligible_source_count": len(source_candidates),
        "source_order": [
            {"rank_digest": digest, "source_group": source, "videos": videos}
            for digest, source, videos in source_candidates
        ],
    }


def selected_rows(
    eligible: Mapping[str, Mapping[str, Any]], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in eligible.values():
        by_key[(str(row["source_group"]), int(row["video_id"]), int(row["sampled_step"]))].append(row)
    selected: list[dict[str, Any]] = []
    for item in plan["source_order"][:SOURCE_COUNT]:
        source = item["source_group"]
        for _digest, video_id, steps in item["videos"]:
            for step in steps:
                rows = by_key.get((source, video_id, step), [])
                require(rows, "Selected source/video/step has no eligible anchor")
                selected.extend(sorted(rows, key=lambda row: int(row["observation_id"])))
    return selected


def count_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count known-identity cross-step pairs; unknown is never a negative."""
    by_video: dict[tuple[str, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_video[(str(row["source_group"]), int(row["video_id"]))].append(row)
    same = different = same_pred = unknown = 0
    for group in by_video.values():
        for first, second in itertools.combinations(group, 2):
            if int(first["sampled_step"]) == int(second["sampled_step"]):
                continue
            a, b = first.get("gt_track_id"), second.get("gt_track_id")
            if a is None or b is None:
                unknown += 1
                continue
            if int(a) == int(b):
                same += 1
            else:
                different += 1
                if int(first["pred_category_id"]) == int(second["pred_category_id"]):
                    same_pred += 1
    return {
        "known_same_gt_cross_step": same,
        "known_different_gt_cross_step": different,
        "known_different_gt_same_pred_category": same_pred,
        "unknown_identity_pairs_excluded": unknown,
    }


def fold_plan(
    plan: Mapping[str, Any], eligible: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_sources = plan["source_order"][:SOURCE_COUNT]
    folds: list[dict[str, Any]] = []
    for fold_index in range(FOLD_COUNT):
        sources = selected_sources[fold_index * 2 : fold_index * 2 + 2]
        fold_rows: list[Mapping[str, Any]] = []
        for item in sources:
            source = item["source_group"]
            for _digest, video_id, steps in item["videos"]:
                wanted = set(steps)
                fold_rows.extend(
                    row
                    for row in eligible.values()
                    if row["source_group"] == source
                    and int(row["video_id"]) == int(video_id)
                    and int(row["sampled_step"]) in wanted
                )
        folds.append(
            {
                "fold": fold_index,
                "sources": [item["source_group"] for item in sources],
                "videos": [
                    {"source_group": item["source_group"], "video_id": video_id, "sample_steps": steps}
                    for item in sources
                    for _digest, video_id, steps in item["videos"]
                ],
                "pairs": count_pairs(fold_rows),
                "selected_rows": len(fold_rows),
            }
        )
    return folds, selected_sources


def sufficient(folds: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        fold["pairs"][name] >= MIN_PAIRS_PER_FOLD
        for fold in folds
        for name in (
            "known_same_gt_cross_step",
            "known_different_gt_cross_step",
            "known_different_gt_same_pred_category",
        )
    )


def find_window(
    plan: Mapping[str, Any], eligible: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Try every contiguous hash-ordered six-source window; retain all attempts."""
    order = plan["source_order"]
    attempts: list[dict[str, Any]] = []
    accepted: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None
    for start in range(len(order) - SOURCE_COUNT + 1):
        window_plan = dict(plan)
        window_plan["source_order"] = order[start : start + SOURCE_COUNT]
        folds, sources = fold_plan(window_plan, eligible)
        record = {
            "window_index": len(attempts),
            "start_rank": start,
            "sources": [item["source_group"] for item in window_plan["source_order"]],
            "folds": folds,
            "sufficient": sufficient(folds),
        }
        attempts.append(record)
        if record["sufficient"]:
            accepted = (folds, window_plan["source_order"])
            break
    require(accepted is not None, "No deterministic six-source window satisfies support thresholds")
    accepted_plan = dict(plan)
    accepted_plan["source_order"] = accepted[1]
    accepted_plan["attempt_count"] = len(attempts)
    accepted_plan["selected_start_rank"] = attempts[-1]["start_rank"]
    return accepted_plan, accepted[0], attempts


def validate_frame_sources(
    selected: Sequence[Mapping[str, Any]],
    gt_free: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, str],
    dump_dir: Path,
) -> dict[str, Any]:
    by_step: dict[tuple[str, int, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        by_step[(str(row["source_group"]), int(row["video_id"]), int(row["sampled_step"]))].append(row)
    checks: list[dict[str, Any]] = []
    digest_cache: dict[str, str] = {}
    image_ids: set[int] = set()
    for (source, video_id, step), rows in sorted(by_step.items()):
        locators = {str(gt_free[str(row["observation_id"])]["dump_file"]) for row in rows}
        require(len(locators) == 1, "Multiple dumps for one selected step: %s/%s/%s" % (source, video_id, step))
        dump_file = locators.pop()
        require(dump_file in inventory, "Dump absent from inventory: %s" % dump_file)
        dump_path = dump_dir / dump_file
        receipt_path = dump_path.with_suffix(".receipt.json")
        require(dump_path.is_file(), "Missing dump: %s" % dump_path)
        require(receipt_path.is_file(), "Missing receipt: %s" % receipt_path)
        if dump_file not in digest_cache:
            digest_cache[dump_file] = sha256_file(dump_path)
        actual_dump_sha = digest_cache[dump_file]
        require(actual_dump_sha == inventory[dump_file], "Dump SHA mismatch: %s" % dump_file)
        require(sha256_file(receipt_path) == inventory.get(receipt_path.name), "Receipt SHA mismatch: %s" % receipt_path.name)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt.get("status") == "PASS_FRAME_DUMP", "Invalid dump receipt: %s" % receipt_path)
        require(receipt.get("file") == dump_file, "Receipt filename mismatch: %s" % dump_file)
        image_path = Path(str(receipt.get("filename", "")))
        require(image_path.is_file(), "Missing image for %s: %s" % (dump_file, image_path))
        image_ids.add(int(receipt["image_id"]))
        for row in rows:
            locator = gt_free[str(row["observation_id"])]
            require(str(locator["dump_file"]) == dump_file, "Row dump mismatch")
            require(str(locator["dump_sha256"]) == actual_dump_sha, "Row dump SHA mismatch")
            require(receipt["image_id"] == row["image_id"], "Receipt image mismatch")
            require(receipt["video_id"] == row["video_id"], "Receipt video mismatch")
        checks.append(
            {
                "source_group": source,
                "video_id": video_id,
                "sampled_step": step,
                "image_id": int(rows[0]["image_id"]),
                "dump_file": dump_file,
                "dump_sha256": actual_dump_sha,
                "dump_row_min": min(int(gt_free[str(row["observation_id"])]["dump_row"]) for row in rows),
                "dump_row_max": max(int(gt_free[str(row["observation_id"])]["dump_row"]) for row in rows),
                "rows": len(rows),
                "image_exists": True,
                "dump_exists": True,
                "receipt_validated": True,
            }
        )
    return {
        "status": "PASS_SELECTED_IMAGES_AND_DUMPS",
        "selected_steps": len(checks),
        "unique_images": len(image_ids),
        "checks": checks,
    }


def validate_video_order(
    selected: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    for fold in folds:
        for video in fold["videos"]:
            steps = [int(value) for value in video["sample_steps"]]
            require(steps == sorted(set(steps)), "Fold sample steps are not increasing and unique")
            require(len(steps) == STEP_COUNT, "Fold video does not have exactly six steps")
            rows = [
                row
                for row in selected
                if row["source_group"] == video["source_group"]
                and int(row["video_id"]) == int(video["video_id"])
            ]
            require(len({int(row["sampled_step"]) for row in rows}) == STEP_COUNT, "Video step coverage mismatch")
            for step in steps:
                at_step = [row for row in rows if int(row["sampled_step"]) == step]
                require(at_step, "Missing selected rows for video/step")
                frames = {int(row["original_frame_index"]) for row in at_step}
                images = {int(row["image_id"]) for row in at_step}
                require(len(frames) == 1 and len(images) == 1, "Inconsistent selected step metadata")
    return {"status": "PASS_VIDEO_STEP_ORDER", "checked_videos": sum(len(f["videos"]) for f in folds)}


def forbidden_keys(value: Any, prefix: str = "model_input") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + "." + str(key)
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_MODEL_INPUT_KEYS):
                found.append(path)
            found.extend(forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, "%s[%d]" % (prefix, index)))
    return found


def build(
    support_dir: Path,
    run_dir: Path,
    out_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    started = time.monotonic()
    supervised_path = support_dir / "supervised_observations.jsonl"
    gt_free_path = support_dir / "gt_free_observations.jsonl"
    inventory_path = run_dir / "dump_inventory.json"
    dump_dir = run_dir / "dump"
    require(not out_dir.exists(), "Output directory already exists: %s" % out_dir)
    require(inventory_path.is_file(), "Missing dump inventory: %s" % inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(isinstance(inventory, dict) and inventory, "Invalid dump inventory")

    eligible = load_eligible_supervision(supervised_path)
    gt_free = join_gt_free(gt_free_path, eligible)
    initial_plan = selection_plan(eligible, seed)
    require(
        initial_plan["eligible_source_count"] >= SOURCE_COUNT,
        "Only %d sources have %d videos with %d steps"
        % (initial_plan["eligible_source_count"], VIDEO_COUNT, STEP_COUNT),
    )
    plan, folds, attempts = find_window(initial_plan, eligible)
    selected = selected_rows(eligible, plan)
    # Each selected structural step can contain several eligible anchors.  Require
    # every structural slot to be present instead of assuming exactly one row.
    require(
        len(
            {
                (row["source_group"], int(row["video_id"]), int(row["sampled_step"]))
                for row in selected
            }
        )
        == SOURCE_COUNT * VIDEO_COUNT * STEP_COUNT,
        "Selected structural cell count changed",
    )
    frame_checks = validate_frame_sources(selected, gt_free, inventory, dump_dir)
    order_check = validate_video_order(selected, folds)
    overall_pairs = count_pairs(selected)

    model_input = []
    evaluation_labels = []
    for row in selected:
        locator = gt_free[str(row["observation_id"])]
        model_input.append(
            {
                "observation_id": int(row["observation_id"]),
                "source_group": str(row["source_group"]),
                "video_id": int(row["video_id"]),
                "image_id": int(row["image_id"]),
                "sampled_step": int(row["sampled_step"]),
                "original_frame_index": int(row["original_frame_index"]),
                "local_track_id": int(row["local_track_id"]),
                "pred_category_id": int(row["pred_category_id"]),
                "dump_file": str(locator["dump_file"]),
                "dump_sha256": str(locator["dump_sha256"]),
                "dump_row": int(locator["dump_row"]),
            }
        )
        evaluation_labels.append(
            {
                "observation_id": int(row["observation_id"]),
                "gt_track_id": int(row["gt_track_id"]),
                "gt_category_id": row.get("gt_category_id"),
            }
        )
    leaked = forbidden_keys(model_input)
    require(not leaked, "GT or evaluation fields leaked into model input: %s" % leaked[:10])
    require(
        overall_pairs["unknown_identity_pairs_excluded"] == 0,
        "Selected FIT eligible anchors unexpectedly contain an unknown identity",
    )

    input_pins = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (supervised_path, gt_free_path, inventory_path)
    }
    manifest = {
        "status": "PASS_GEOMETRY_ROI_GATE_MANIFEST",
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "seed": seed,
            "selection_unit": "source -> video -> structural sampled_step",
            "source_selection": "stable SHA256 rank among FIT sources with >=2 eligible videos",
            "video_selection": "stable SHA256 rank among eligible videos with >=6 sampled_steps",
            "step_selection": "six evenly spaced unique sampled_steps including both endpoints",
            "outcome_selection_used": False,
            "identity_used_for_initial_selection": False,
            "folds": "three fixed source pairs in stable hash order",
            "pair_denominator": "all cross-step unordered row pairs within each selected video",
            "unknown_identity_policy": "exclude; never a negative pair",
        },
        "constants": {
            "source_count": SOURCE_COUNT,
            "videos_per_source": VIDEO_COUNT,
            "sample_steps_per_video": STEP_COUNT,
            "fold_count": FOLD_COUNT,
            "sources_per_fold": 2,
            "minimum_pairs_per_fold": MIN_PAIRS_PER_FOLD,
        },
        "input_pins": input_pins,
        "selection": {
            "eligible_anchor_count": len(eligible),
            "eligible_source_count": initial_plan["eligible_source_count"],
            "source_hash_order": [
                {"rank_digest": item["rank_digest"], "source_group": item["source_group"]}
                for item in initial_plan["source_order"]
            ],
            "attempts": attempts,
            "selected_start_rank": plan["selected_start_rank"],
            "attempt_count": plan["attempt_count"],
        },
        "folds": folds,
        "overall_pairs": overall_pairs,
        "selected_sources": [
            {
                "source_group": item["source_group"],
                "videos": [{"video_id": video_id, "sample_steps": steps} for _digest, video_id, steps in item["videos"]],
            }
            for item in plan["source_order"]
        ],
        "model_input": model_input,
        "evaluation_only_labels": evaluation_labels,
        "validation": {
            "frames": frame_checks,
            "video_step_order": order_check,
            "model_input_gt_leaks": leaked,
            "unknown_identity_negative_pairs": overall_pairs["unknown_identity_pairs_excluded"],
            "all_fold_support_thresholds_met": sufficient(folds),
        },
        "limitations": [
            "This is a sampling and provenance manifest, not identity accuracy or module benefit evidence.",
            "Pair support counts are not independent video or source observations.",
            "The FIT partition is exploratory parameter-fit data; it is not an independent test.",
            "Per-step dump rows can share a frame; image count is reported separately from row and pair counts.",
        ],
        "no_gpu_used": True,
        "no_model_loaded_or_training_started": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (out_dir / "SHA256SUMS").write_text(
        "%s  manifest.json\n" % sha256_file(manifest_path), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-dir", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        manifest = build(args.support_dir, args.run_dir, args.out, args.seed)
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.out),
                "selected_rows": len(manifest["model_input"]),
                "selected_steps": manifest["validation"]["frames"]["selected_steps"],
                "folds": [
                    {"fold": fold["fold"], "sources": fold["sources"], "pairs": fold["pairs"]}
                    for fold in manifest["folds"]
                ],
                "overall_pairs": manifest["overall_pairs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
