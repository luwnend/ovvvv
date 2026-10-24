#!/usr/bin/env python3
"""CPU-only information gate for geometry-ROI spatial tokens.

This evaluator answers a narrow question: after fixing the evaluation
candidates, do local ROI tokens add same-identity evidence beyond raw256?
Feature construction and candidate scoring never receive a GT identity or
category label.  GT labels are joined only in the scoring/reporting phase.

The primary population is deliberately hard: same-GT positives versus
different-GT references with the same predicted OPEN-VOCAB category.  Unknown
identities are excluded, never used as negatives.  Pair counts are reported
alongside source, video, and GT-trajectory support because candidates sharing an
endpoint are not independent samples.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/geometry_roi_gate_manifest_20261024_v1/manifest.json"
DEFAULT_CACHE = Path("/raid/datasets/chc_data/geometry_roi_pilot_20261024_v10")
DEFAULT_OUT = ROOT / "results/geometry_roi_information_gate_20261024_v1"

ARMS = (
    "raw256",
    "raw256_causal_mean",
    "token_mean",
    "token_mean_causal",
    "token_center",
    "spatial_topk_geometry",
    "fgfa_aligned_past_mean",
    "random_token_permutation",
    "global_mean_geometry",
)

TOPK = 5
GRID_PRIOR_WEIGHT = 1.0
SHAPE_PRIOR_WEIGHT = 0.02


class GateError(RuntimeError):
    """Invalid input, protocol, or evaluation state."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"Cannot parse JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"{path}:{line_no}: {exc}") from exc
            require(isinstance(value, dict), f"{path}:{line_no} is not an object")
            rows.append(value)
    return rows


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2, "unit_rows expects a matrix")
    scale = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(scale, 1e-12)


def cosine_pair(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    require(left.shape == right.shape, "Cosine vectors differ in shape")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _shape_prior(box_a: np.ndarray, box_b: np.ndarray, eps: float = 1e-6) -> float:
    """Deterministic shape consistency; no fitted or GT-derived parameter."""
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    require(a.shape == b.shape and a.shape == (4,), "Boxes must be [x1,y1,x2,y2]")
    wa, ha = max(float(a[2] - a[0]), eps), max(float(a[3] - a[1]), eps)
    wb, hb = max(float(b[2] - b[0]), eps), max(float(b[3] - b[1]), eps)
    return abs(math.log(wa / wb)) + abs(math.log(ha / hb))


def _spatial_grid() -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


GRID_X, GRID_Y = _spatial_grid()


def spatial_topk_score(
    query_tokens: np.ndarray,
    reference_tokens: np.ndarray,
    query_box: np.ndarray,
    reference_box: np.ndarray,
    *,
    permutation_seed: int | None = None,
    topk: int = TOPK,
) -> float:
    """Top-k local matching with a fixed relative-grid geometry prior.

    ``permutation_seed`` is used only by the random control.  It is derived from
    immutable observation identifiers by the caller, never from GT.
    """
    q = np.asarray(query_tokens, dtype=np.float64)
    r = np.asarray(reference_tokens, dtype=np.float64)
    require(q.shape == (256, 7, 7), "Query tokens must be [256,7,7]")
    require(r.shape == q.shape, "Reference tokens must be [256,7,7]")
    require(1 <= int(topk) <= 49, "Invalid top-k")
    q_flat = q.reshape(256, 49).T
    r_flat = r.reshape(256, 49).T
    if permutation_seed is not None:
        generator = np.random.default_rng(int(permutation_seed))
        r_flat = r_flat[generator.permutation(49)]
    similarity = unit_rows(q_flat) @ unit_rows(r_flat).T
    delta_x = GRID_X[:, None] - GRID_X[None, :]
    delta_y = GRID_Y[:, None] - GRID_Y[None, :]
    grid_distance = (delta_x * delta_x + delta_y * delta_y) / 4.0
    local = similarity - GRID_PRIOR_WEIGHT * grid_distance
    best_per_query_cell = np.max(local, axis=1)
    selected = np.partition(best_per_query_cell, -int(topk))[-int(topk):]
    return float(np.mean(selected) -
                 SHAPE_PRIOR_WEIGHT * _shape_prior(query_box, reference_box))


@dataclasses.dataclass
class Observation:
    observation_id: int
    fold: int
    source: str
    video_id: int
    image_id: int
    sampled_step: int
    original_frame_index: int
    local_track_id: int
    pred_category_id: int
    dump_file: str
    dump_row: int
    raw: np.ndarray
    tokens: np.ndarray
    box: np.ndarray
    past_aligned_grid: np.ndarray | None = None
    past_raw_mean: np.ndarray | None = None
    past_token_mean: np.ndarray | None = None

    @property
    def key(self) -> str:
        return f"{self.source}|{self.video_id}|{self.observation_id}"


def _require_int(value: Any, where: str, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{where}: {field} must be an integer")
    return value


def load_manifest_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                             list[dict[str, Any]]]:
    manifest = _read_json(path)
    require(manifest.get("status") == "PASS_GEOMETRY_ROI_GATE_MANIFEST",
            "Unexpected manifest status")
    model_rows = manifest.get("model_input")
    labels = manifest.get("evaluation_only_labels")
    require(isinstance(model_rows, list) and len(model_rows) == 122,
            "Protocol manifest must contain exactly 122 model-input rows")
    require(isinstance(labels, list) and len(labels) == 122,
            "Protocol manifest must contain exactly 122 evaluation labels")
    expected = {"observation_id", "source_group", "video_id", "image_id",
                "sampled_step", "original_frame_index", "local_track_id",
                "pred_category_id", "dump_file", "dump_sha256", "dump_row"}
    seen: set[int] = set()
    for index, row in enumerate(model_rows):
        where = f"model_input[{index}]"
        require(isinstance(row, dict) and set(row) == expected,
                f"{where}: keys differ from the GT-free schema")
        obs_id = _require_int(row["observation_id"], where, "observation_id")
        require(obs_id not in seen, f"Duplicate observation_id {obs_id}")
        seen.add(obs_id)
        for field in ("video_id", "image_id", "sampled_step",
                      "original_frame_index", "local_track_id",
                      "pred_category_id", "dump_row"):
            _require_int(row[field], where, field)
        require(isinstance(row["source_group"], str) and row["source_group"],
                f"{where}: source_group is empty")
        require(isinstance(row["dump_file"], str) and not Path(row["dump_file"]).is_absolute(),
                f"{where}: dump_file must be relative")
        require(isinstance(row["dump_sha256"], str) and len(row["dump_sha256"]) == 64,
                f"{where}: invalid dump_sha256")
        forbidden = [key for key in row if key.startswith("gt_") or "match_iou" in key]
        require(not forbidden, f"{where}: GT fields leaked into model input")
    label_by_id: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(labels):
        require(isinstance(row, dict), f"labels[{index}] is not an object")
        require(set(row) == {"observation_id", "gt_track_id", "gt_category_id"},
                f"labels[{index}] keys changed")
        obs_id = _require_int(row["observation_id"], f"labels[{index}]", "observation_id")
        require(obs_id not in label_by_id, f"Duplicate evaluation label {obs_id}")
        gt_track = row["gt_track_id"]
        if gt_track is not None:
            _require_int(gt_track, f"labels[{index}]", "gt_track_id")
        label_by_id[obs_id] = row
    require(set(label_by_id) == seen, "Model rows and evaluation-only labels differ")
    return manifest, model_rows, labels


def export_separated_protocol(manifest: Mapping[str, Any],
                              model_rows: Sequence[Mapping[str, Any]],
                              labels: Sequence[Mapping[str, Any]],
                              out_dir: Path) -> dict[str, Any]:
    """Materialize the evaluator-only split required by the protocol."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observation_path = out_dir / "evaluation_observations.jsonl"
    label_path = out_dir / "evaluation_only_labels.jsonl"
    _write_jsonl(observation_path, model_rows)
    _write_jsonl(label_path, labels)
    require(len(_read_jsonl(observation_path)) == 122, "Exported observation count changed")
    require(len(_read_jsonl(label_path)) == 122, "Exported label count changed")
    receipt = {
        "observations": str(observation_path),
        "observations_sha256": sha256_path(observation_path),
        "labels": str(label_path),
        "labels_sha256": sha256_path(label_path),
        "manifest_sha256": sha256_path(Path(manifest["_source_path"]))
        if "_source_path" in manifest else None,
        "model_input_has_gt": False,
        "labels_loaded_only_at_scoring": True,
    }
    _write_json(out_dir / "separated_protocol_receipt.json", receipt)
    return receipt


def _fold_lookup(manifest: Mapping[str, Any]) -> dict[tuple[str, int], int]:
    folds = manifest.get("folds")
    require(isinstance(folds, list) and folds, "Manifest has no folds")
    result: dict[tuple[str, int], int] = {}
    for fold in folds:
        fold_index = _require_int(fold.get("fold"), "fold", "fold")
        for video in fold.get("videos", []):
            key = (str(video["source_group"]),
                   _require_int(video["video_id"], "fold video", "video_id"))
            require(key not in result, f"Video appears in multiple folds: {key}")
            result[key] = fold_index
    return result


def _parse_preprocessing(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        require(value.ndim == 0, "preprocessing_json must be scalar")
        value = value.item()
    require(isinstance(value, (str, bytes, np.bytes_, np.str_)),
            "preprocessing_json is not text")
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    require(isinstance(parsed, dict), "preprocessing_json root is not an object")
    require("img_shape" in parsed and len(parsed["img_shape"]) >= 2,
            "preprocessing_json lacks img_shape")
    return parsed


def load_observations(manifest_path: Path, cache_dir: Path,
                      source_filter: str | None = None,
                      video_filter: int | None = None,
                      receipt_path: Path | None = None) -> tuple[
                          dict[str, Any], list[Observation], dict[str, Any]]:
    manifest, model_rows, labels = load_manifest_rows(manifest_path)
    fold_by_video = _fold_lookup(manifest)
    labels_by_id = {int(row["observation_id"]): row for row in labels}
    cache_dir = Path(cache_dir)
    require(cache_dir.is_dir(), f"ROI cache directory is missing: {cache_dir}")
    receipt_path = (cache_dir / "collection_receipt.json"
                    if receipt_path is None else Path(receipt_path))
    require(receipt_path.is_file(), f"Missing ROI receipt: {receipt_path}")
    collection_receipt = _read_json(receipt_path)
    receipt_status = collection_receipt.get("status")
    if receipt_status == "PASS_GEOMETRY_ROI_TOKEN_COLLECTION_V1":
        require(receipt_path == cache_dir / "collection_receipt.json",
                "Collector receipt must be in the token directory")
    elif receipt_status == "PASS_INDEPENDENT_FULL_ROI_AUDIT_RECEIPT_V1":
        require(receipt_path != cache_dir / "collection_receipt.json",
                "Independent audit receipt must be stored outside the token directory")
        require(collection_receipt.get("scope", "").startswith(
            "Strict coverage receipt"), "Unexpected independent receipt scope")
        require(collection_receipt.get("provenance", {}).get(
            "formal_collector_completion_claimed") is False,
                "Independent receipt must preserve the collection provenance hold")
        coverage = collection_receipt.get("coverage", {})
        require(coverage.get("frames") == 276
                and coverage.get("raw_exact_parent_frames") == 276
                and coverage.get("row_and_geometry_exact_parent_frames") == 276,
                "Independent receipt lacks full 276-frame parent parity")
        require(coverage.get("evaluation_frames") == 72
                and coverage.get("anchors") == 122
                and coverage.get("sources") == 6
                and coverage.get("folds") == 3,
                "Independent receipt lacks six-source/three-fold coverage")
        inputs = collection_receipt.get("inputs", {})
        require(inputs.get("token_directory") == str(cache_dir),
                "Independent receipt token directory differs")
        require(inputs.get("gate_manifest_sha256") == sha256_path(manifest_path),
                "Independent receipt gate manifest differs")
        require(len(str(inputs.get("evaluation_manifest_sha256", ""))) == 64,
                "Independent receipt lacks its evaluation-manifest pin")
    else:
        raise GateError(f"Unsupported ROI receipt status: {receipt_status!r}")
    receipt_frames = collection_receipt.get("frames")
    if receipt_frames is None:
        receipt_frames = collection_receipt.get("frame_statuses")
        if receipt_frames is not None:
            receipt_frames = [
                dict(frame, file=frame["file"], sha256=frame["token_sha256"])
                for frame in receipt_frames
            ]
    require(isinstance(receipt_frames, list) and receipt_frames,
            "ROI collection receipt has no frame records")
    receipt_by_file: dict[str, dict[str, Any]] = {}
    for frame in receipt_frames:
        dump_file = str(frame.get("file", ""))
        require(dump_file and dump_file not in receipt_by_file,
                "ROI collection receipt has a missing or duplicate frame filename")
        require(isinstance(frame.get("sha256"), str) and len(frame["sha256"]) == 64,
                f"ROI receipt has an invalid SHA for {dump_file}")
        receipt_by_file[dump_file] = frame
    selected_rows = [
        row for row in model_rows
        if source_filter is None or row["source_group"] == source_filter
    ]
    if video_filter is not None:
        selected_rows = [
            row for row in selected_rows if int(row["video_id"]) == int(video_filter)
        ]
    require(selected_rows, f"No manifest rows match source filter {source_filter!r}")
    by_dump: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_dump.setdefault(str(row["dump_file"]), []).append(row)

    observations: list[Observation] = []
    frame_receipts: list[dict[str, Any]] = []
    for dump_file, rows in sorted(by_dump.items()):
        path = cache_dir / dump_file
        require(path.is_file(), f"Missing ROI frame cache: {path}")
        require(dump_file in receipt_by_file,
                f"ROI collection receipt does not list {dump_file}")
        actual_sha = sha256_path(path)
        require(actual_sha == receipt_by_file[dump_file]["sha256"],
                f"ROI cache SHA differs from collection receipt: {dump_file}")
        with np.load(path, allow_pickle=False) as saved:
            required = {"roi_input", "conv_spatial", "raw_output", "det_bboxes",
                        "preprocessing_json", "source", "video_id", "image_id",
                        "frame_id"}
            require(required <= set(saved.files), f"{dump_file}: members missing")
            tokens_all = np.asarray(saved["conv_spatial"])
            raw_all = np.asarray(saved["raw_output"])
            boxes_all = np.asarray(saved["det_bboxes"])
            source = str(saved["source"].item())
            video_id = int(saved["video_id"].item())
            image_id = int(saved["image_id"].item())
            preprocessing = _parse_preprocessing(saved["preprocessing_json"])
        receipt_frame = receipt_by_file[dump_file]
        require(receipt_frame.get("source") == source
                and int(receipt_frame.get("video_id", -1)) == video_id
                and int(receipt_frame.get("image_id", -1)) == image_id,
                f"{dump_file}: collection receipt metadata differs")
        if receipt_status == "PASS_INDEPENDENT_FULL_ROI_AUDIT_RECEIPT_V1":
            require(receipt_frame.get("parent_cache_sha256")
                    == rows[0]["dump_sha256"],
                    f"{dump_file}: independent receipt parent SHA differs")
            require(receipt_frame.get("raw_exact_parent") is True
                    and receipt_frame.get("row_locators_exact_parent") is True
                    and receipt_frame.get("box_geometry_exact_parent") is True,
                    f"{dump_file}: independent receipt parity flags differ")
        require(tokens_all.ndim == 4 and tokens_all.shape[1:] == (256, 7, 7),
                f"{dump_file}: tokens are not [N,256,7,7]")
        require(raw_all.shape == (len(tokens_all), 256),
                f"{dump_file}: raw256 rows differ")
        require(boxes_all.ndim == 2 and boxes_all.shape[0] == len(tokens_all)
                and boxes_all.shape[1] >= 4,
                f"{dump_file}: boxes differ")
        require(np.isfinite(tokens_all).all() and np.isfinite(raw_all).all(),
                f"{dump_file}: non-finite features")
        image_h, image_w = float(preprocessing["img_shape"][0]), float(
            preprocessing["img_shape"][1])
        for row in rows:
            require(row["source_group"] == source, f"{dump_file}: source mismatch")
            require(int(row["video_id"]) == video_id, f"{dump_file}: video mismatch")
            require(int(row["image_id"]) == image_id, f"{dump_file}: image mismatch")
            dump_row = int(row["dump_row"])
            require(0 <= dump_row < len(tokens_all), f"{dump_file}: dump_row out of range")
            box = boxes_all[dump_row, :4].astype(np.float64)
            normalized_box = np.asarray([
                box[0] / image_w, box[1] / image_h,
                box[2] / image_w, box[3] / image_h,
            ], dtype=np.float64)
            require(np.all(normalized_box[2:] >= normalized_box[:2]),
                    f"{dump_file}: invalid detector box")
            fold = fold_by_video.get((source, video_id))
            require(fold is not None, f"Observation video lacks a fold: {source}/{video_id}")
            label = labels_by_id[int(row["observation_id"])]
            observations.append(Observation(
                observation_id=int(row["observation_id"]),
                fold=int(fold),
                source=source,
                video_id=video_id,
                image_id=image_id,
                sampled_step=int(row["sampled_step"]),
                original_frame_index=int(row["original_frame_index"]),
                local_track_id=int(row["local_track_id"]),
                pred_category_id=int(row["pred_category_id"]),
                dump_file=dump_file,
                dump_row=dump_row,
                raw=raw_all[dump_row].astype(np.float32, copy=True),
                tokens=tokens_all[dump_row].astype(np.float32, copy=True),
                box=normalized_box,
            ))
        frame_receipts.append({
            "dump_file": dump_file,
            "sha256": actual_sha,
            "parent_cache_sha256": str(rows[0]["dump_sha256"]),
            "rows_used": len(rows),
            "source": source,
            "video_id": video_id,
            "image_id": image_id,
        })
    build_past_only_aligned_means(observations)
    return manifest, observations, {
        "selected_manifest_rows": len(selected_rows),
        "excluded_manifest_rows": len(model_rows) - len(selected_rows),
        "frames": frame_receipts,
        "source_filter": source_filter,
        "video_filter": video_filter,
        "collection_receipt_sha256": sha256_path(receipt_path),
        "collection_receipt_status": collection_receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_provenance_hold": (
            receipt_status != "PASS_GEOMETRY_ROI_TOKEN_COLLECTION_V1"),
    }


def build_past_only_aligned_means(observations: list[Observation]) -> None:
    """FGFA-style box-grid-aligned means using only current/past local tokens.

    Without verified flow, the box-normalized 7x7 grids are the only legitimate
    alignment.  Future sampled steps are never read.
    """
    groups: dict[tuple[str, int, int, int | None], list[Observation]] = {}
    for observation in observations:
        groups.setdefault(
            (observation.source, observation.video_id, observation.local_track_id,
             None),
            []).append(observation)
    for group in groups.values():
        by_step: dict[int, list[Observation]] = {}
        for observation in group:
            by_step.setdefault(observation.sampled_step, []).append(observation)
        cumulative_grid = np.zeros((256, 7, 7), dtype=np.float64)
        cumulative_raw = np.zeros(256, dtype=np.float64)
        cumulative_mean = np.zeros(256, dtype=np.float64)
        count = 0
        for step in sorted(by_step):
            current_grid = np.stack([
                observation.tokens.astype(np.float64)
                for observation in by_step[step]
            ], axis=0)
            current_raw = np.stack([
                observation.raw.astype(np.float64)
                for observation in by_step[step]
            ], axis=0)
            current_mean = current_grid.mean(axis=(2, 3))
            # All rows at the current sampled step enter together; row order
            # cannot make a future observation visible.
            cumulative_grid += current_grid.sum(axis=0)
            cumulative_raw += current_raw.sum(axis=0)
            cumulative_mean += current_mean.sum(axis=0)
            count += len(current_grid)
            descriptor_grid = cumulative_grid / max(count, 1)
            descriptor_raw = cumulative_raw / max(count, 1)
            descriptor_mean = cumulative_mean / max(count, 1)
            for observation in by_step[step]:
                observation.past_aligned_grid = descriptor_grid.astype(
                    np.float32, copy=True)
                observation.past_raw_mean = descriptor_raw.astype(
                    np.float32, copy=True)
                observation.past_token_mean = descriptor_mean.astype(
                    np.float32, copy=True)

def labels_by_observation(labels: Sequence[Mapping[str, Any]]) -> dict[int, int | None]:
    result: dict[int, int | None] = {}
    for row in labels:
        observation_id = int(row["observation_id"])
        value = row["gt_track_id"]
        result[observation_id] = None if value is None else int(value)
    return result


def feature_score(arm: str, query: Observation, reference: Observation) -> float:
    """GT-free pair score.  All candidate selection happened before this call."""
    require(arm in ARMS, f"Unknown arm: {arm}")
    if arm == "raw256":
        return cosine_pair(query.raw, reference.raw)
    if arm == "raw256_causal_mean":
        require(query.past_raw_mean is not None
                and reference.past_raw_mean is not None,
                "Causal raw descriptor is missing")
        return cosine_pair(query.past_raw_mean, reference.past_raw_mean)
    if arm in ("token_mean", "global_mean_geometry"):
        score = cosine_pair(
            query.tokens.astype(np.float64).mean(axis=(1, 2)),
            reference.tokens.astype(np.float64).mean(axis=(1, 2)))
        if arm == "global_mean_geometry":
            score -= SHAPE_PRIOR_WEIGHT * _shape_prior(query.box, reference.box)
        return float(score)
    if arm == "token_mean_causal":
        require(query.past_token_mean is not None
                and reference.past_token_mean is not None,
                "Causal token-mean descriptor is missing")
        return cosine_pair(query.past_token_mean, reference.past_token_mean)
    if arm == "token_center":
        return cosine_pair(query.tokens[:, 3, 3], reference.tokens[:, 3, 3])
    if arm == "fgfa_aligned_past_mean":
        require(query.past_aligned_grid is not None
                and reference.past_aligned_grid is not None,
                "Past-only aligned descriptor is missing")
        return cosine_pair(
            query.past_aligned_grid.reshape(-1),
            reference.past_aligned_grid.reshape(-1))
    if arm == "spatial_topk_geometry":
        return spatial_topk_score(
            query.tokens, reference.tokens, query.box, reference.box)
    if arm == "random_token_permutation":
        payload = "|".join(sorted((query.key, reference.key))).encode("utf-8")
        seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
        return spatial_topk_score(
            query.tokens, reference.tokens, query.box, reference.box,
            permutation_seed=seed)
    raise AssertionError("Unreachable arm")


def build_pair_table(observations: Sequence[Observation],
                     gt_by_id: Mapping[int, int | None]) -> list[dict[str, Any]]:
    """Fixed cross-step pair population; identities only assign evaluation labels."""
    by_video: dict[tuple[str, int], list[Observation]] = {}
    for observation in observations:
        by_video.setdefault((observation.source, observation.video_id), []).append(observation)
    rows: list[dict[str, Any]] = []
    for (source, video_id), group in sorted(by_video.items()):
        group = sorted(group, key=lambda item: (
            item.sampled_step, item.dump_row, item.observation_id))
        for first_index, first in enumerate(group):
            first_gt = gt_by_id[first.observation_id]
            if first_gt is None:
                continue
            for second in group[first_index + 1:]:
                if first.sampled_step == second.sampled_step:
                    continue
                second_gt = gt_by_id[second.observation_id]
                if second_gt is None:
                    continue
                same = int(first_gt == second_gt)
                hard_negative = bool(
                    not same and first.pred_category_id == second.pred_category_id)
                rows.append({
                    "source": source,
                    "video_id": video_id,
                    "fold": int(first.fold),
                    "query": first,
                    "reference": second,
                    "same_gt": same,
                    "hard_negative_same_pred_category": hard_negative,
                    "primary_pair": bool(same or hard_negative),
                })
    return rows


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Exact rank-sum AUROC with average ranks for ties."""
    require(len(scores) == len(labels), "AUROC score/label length differs")
    if not scores:
        return None
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    require(np.isfinite(scores_array).all(), "AUROC received non-finite scores")
    require(set(np.unique(labels_array)).issubset({0, 1}), "AUROC labels must be binary")
    positives = int(labels_array.sum())
    negatives = len(labels_array) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(len(scores_array), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[labels_array == 1].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2) /
                 (positives * negatives))


def _primary_pairs(pair_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in pair_rows if row["primary_pair"]]


def source_macro_auroc(pair_rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in _primary_pairs(pair_rows):
        grouped.setdefault(str(row["source"]), []).append(row)
    per_source: dict[str, Any] = {}
    source_values: list[float] = []
    fold_values: dict[int, list[float]] = {}
    for source, rows in sorted(grouped.items()):
        scores = [feature_score(arm, row["query"], row["reference"]) for row in rows]
        labels = [int(row["same_gt"]) for row in rows]
        value = auroc(scores, labels)
        positives = sum(labels)
        negatives = len(labels) - positives
        per_source[source] = {
            "auroc": value,
            "positive_pairs": positives,
            "hard_negative_pairs": negatives,
            "videos": len({(row["source"], row["video_id"]) for row in rows}),
            "positive_videos": len({
                row["video_id"] for row in rows if row["same_gt"]
            }),
        }
        if value is not None:
            source_values.append(value)
            fold_values.setdefault(int(rows[0]["fold"]), []).append(value)
    fold_macro = {
        str(fold): float(np.mean(values))
        for fold, values in sorted(fold_values.items())
    }
    return {
        "source_macro": float(np.mean(source_values)) if source_values else None,
        "fold_macro": float(np.mean(list(fold_macro.values()))) if fold_macro else None,
        "pooled_pair_auroc": auroc(
            [feature_score(arm, row["query"], row["reference"])
             for row in _primary_pairs(pair_rows)],
            [int(row["same_gt"]) for row in _primary_pairs(pair_rows)]),
        "per_fold": fold_macro,
        "per_source": per_source,
        "sources_with_both_classes": len(source_values),
        "positive_support": sum(int(row["same_gt"]) for row in _primary_pairs(pair_rows)),
        "hard_negative_support": sum(
            int(row["hard_negative_same_pred_category"])
            for row in _primary_pairs(pair_rows)),
    }


def _candidate_groups(observations: Sequence[Observation],
                      gt_by_id: Mapping[int, int | None]) -> list[
                          tuple[Observation, list[Observation]]]:
    """Queries whose GT-free candidate pool has positive and hard-negative support.

    The candidate pool contains every cross-step observation with the same
    predicted category.  Unknown identities remain in the scoring pool and their
    top1 selections are counted as unresolved.  GT only qualifies a query and
    scores the arm's selected reference; it never removes or adds a candidate.
    """
    by_video: dict[tuple[str, int], list[Observation]] = {}
    for observation in observations:
        by_video.setdefault((observation.source, observation.video_id), []).append(observation)
    groups: list[tuple[Observation, list[Observation]]] = []
    for group in by_video.values():
        group = sorted(group, key=lambda item: (
            item.sampled_step, item.dump_row, item.observation_id))
        for query in group:
            query_gt = gt_by_id[query.observation_id]
            if query_gt is None:
                continue
            candidates: list[Observation] = []
            positive = False
            hard_negative = False
            for reference in group:
                if reference.observation_id == query.observation_id:
                    continue
                if reference.sampled_step == query.sampled_step:
                    continue
                if reference.pred_category_id != query.pred_category_id:
                    continue
                reference_gt = gt_by_id[reference.observation_id]
                candidates.append(reference)
                if reference_gt == query_gt:
                    positive = True
                elif reference_gt is not None:
                    hard_negative = True
            if positive and hard_negative:
                groups.append((query, candidates))
    return groups


def _top1_row(arm: str, query: Observation,
              candidates: Sequence[Observation]) -> tuple[Observation, float]:
    require(candidates, "Top1 candidate set is empty")
    ranked = sorted(
        ((reference, feature_score(arm, query, reference))
         for reference in candidates),
        key=lambda item: (-item[1], item[0].dump_row, item[0].observation_id),
    )
    return ranked[0]


def top1_metrics(observations: Sequence[Observation],
                 gt_by_id: Mapping[int, int | None]) -> dict[str, Any]:
    candidate_groups = _candidate_groups(observations, gt_by_id)
    require(candidate_groups, "No hard queries with both positive and hard negative support")
    build_past_only_aligned_means(list(observations))
    result: dict[str, Any] = {}
    correct_by_arm: dict[str, list[bool | None]] = {}
    for arm in ARMS:
        records = []
        for query, candidates in candidate_groups:
            selected, selected_score = _top1_row(arm, query, candidates)
            selected_gt = gt_by_id[selected.observation_id]
            resolved = selected_gt is not None
            correct = (
                int(selected_gt == gt_by_id[query.observation_id])
                if resolved else None)
            records.append({
                "source": query.source,
                "fold": query.fold,
                "video_id": query.video_id,
                "query_observation_id": query.observation_id,
                "selected_observation_id": selected.observation_id,
                "selected_dump_row": selected.dump_row,
                "selected_score": selected_score,
                "correct": correct,
                "resolved": resolved,
                "candidate_count": len(candidates),
                "unknown_candidate_count": sum(
                    gt_by_id[reference.observation_id] is None
                    for reference in candidates),
            })
        correct_by_arm[arm] = [
            None if record["correct"] is None else bool(record["correct"])
            for record in records
        ]
        by_source: dict[str, list[bool]] = {}
        by_fold: dict[int, list[bool]] = {}
        for record in records:
            if record["correct"] is not None:
                by_source.setdefault(record["source"], []).append(bool(record["correct"]))
                by_fold.setdefault(int(record["fold"]), []).append(bool(record["correct"]))
        source_macro = (
            float(np.mean([np.mean(values) for values in by_source.values()]))
            if by_source else None)
        fold_macro = (
            float(np.mean([np.mean(values) for values in by_fold.values()]))
            if by_fold else None)
        resolved_records = [record for record in records if record["resolved"]]
        result[arm] = {
            "queries": len(records),
            "resolved_queries": len(resolved_records),
            "unresolved_unknown_selection": len(records) - len(resolved_records),
            "correct": sum(record["correct"] for record in resolved_records),
            "accuracy_known_resolved": (
                float(np.mean([record["correct"] for record in resolved_records]))
                if resolved_records else None),
            "source_macro_accuracy": source_macro,
            "fold_macro_accuracy": fold_macro,
            "sources": len(by_source),
            "records": records,
        }
    raw_correct = correct_by_arm["raw256"]
    for arm in ARMS:
        arm_correct = correct_by_arm[arm]
        transitions = {
            "repair_raw_error": 0,
            "break_raw_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "raw_unresolved": 0,
            "arm_unresolved": 0,
            "both_resolved_fixed_query_set": True,
        }
        for raw, new in zip(raw_correct, arm_correct):
            if raw is None:
                transitions["raw_unresolved"] += 1
            if new is None:
                transitions["arm_unresolved"] += 1
            if raw is None or new is None:
                continue
            if (not raw) and new:
                transitions["repair_raw_error"] += 1
            elif raw and (not new):
                transitions["break_raw_correct"] += 1
            elif raw and new:
                transitions["both_correct"] += 1
            else:
                transitions["both_wrong"] += 1
        result[arm]["paired_vs_raw256"] = {
            **transitions,
        }
        source_deltas: dict[str, list[float]] = {}
        for raw, new, record in zip(raw_correct, arm_correct, result[arm]["records"]):
            if raw is None or new is None:
                continue
            source_deltas.setdefault(record["source"], []).append(float(new) - float(raw))
        result[arm]["source_macro_delta_vs_raw256"] = (
            float(np.mean([np.mean(values) for values in source_deltas.values()]))
            if source_deltas else None)
    def local_vs_matched(control: str, candidate: str) -> dict[str, Any]:
        control_values = correct_by_arm[control]
        candidate_values = correct_by_arm[candidate]
        transitions = {
            "repair_control_error": 0,
            "break_control_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "control_unresolved": 0,
            "candidate_unresolved": 0,
            "fixed_query_set": True,
        }
        paired_source_deltas: dict[str, list[float]] = {}
        for control_value, candidate_value, record in zip(
                control_values, candidate_values, result[candidate]["records"]):
            if control_value is None:
                transitions["control_unresolved"] += 1
            if candidate_value is None:
                transitions["candidate_unresolved"] += 1
            if control_value is None or candidate_value is None:
                continue
            if (not control_value) and candidate_value:
                transitions["repair_control_error"] += 1
            elif control_value and (not candidate_value):
                transitions["break_control_correct"] += 1
            elif control_value and candidate_value:
                transitions["both_correct"] += 1
            else:
                transitions["both_wrong"] += 1
            paired_source_deltas.setdefault(record["source"], []).append(
                float(candidate_value) - float(control_value))
        return {
            "control": control,
            "candidate": candidate,
            "transitions": transitions,
            "source_macro_delta": (
                float(np.mean([
                    np.mean(values) for values in paired_source_deltas.values()]))
                if paired_source_deltas else None),
        }
    result["matched_contrasts"] = {
        "local_tokens_vs_causal_raw": local_vs_matched(
            "raw256_causal_mean", "fgfa_aligned_past_mean"),
        "local_tokens_vs_causal_global_mean": local_vs_matched(
            "token_mean_causal", "fgfa_aligned_past_mean"),
        "spatial_vs_causal_global_mean": local_vs_matched(
            "token_mean_causal", "spatial_topk_geometry"),
        "spatial_vs_random_permutation": local_vs_matched(
            "random_token_permutation", "spatial_topk_geometry"),
        "causal_raw_vs_current_raw": local_vs_matched(
            "raw256", "raw256_causal_mean"),
    }
    return result


def _support_summary(observations: Sequence[Observation],
                     pair_rows: Sequence[Mapping[str, Any]],
                     gt_by_id: Mapping[int, int | None]) -> dict[str, Any]:
    primary = _primary_pairs(pair_rows)
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({row.source for row in observations}):
        source_obs = [row for row in observations if row.source == source]
        source_pairs = [row for row in primary if row["source"] == source]
        by_source[source] = {
            "observations": len(source_obs),
            "known_identity_observations": sum(
                gt_by_id[row.observation_id] is not None for row in source_obs),
            "unknown_identity_observations_excluded": sum(
                gt_by_id[row.observation_id] is None for row in source_obs),
            "videos": len({row.video_id for row in source_obs}),
            "known_gt_trajectories": len({
                (row.video_id, gt_by_id[row.observation_id])
                for row in source_obs
                if gt_by_id[row.observation_id] is not None
            }),
            "positive_pairs": sum(row["same_gt"] for row in source_pairs),
            "hard_negative_pairs": sum(
                row["hard_negative_same_pred_category"] for row in source_pairs),
            "positive_identity_units": len({
                (row["video_id"], row["query"].observation_id, row["reference"].observation_id)
                for row in source_pairs if row["same_gt"]
            }),
            "hard_negative_identity_units": len({
                (row["video_id"],
                 min(row["query"].observation_id, row["reference"].observation_id),
                 max(row["query"].observation_id, row["reference"].observation_id))
                for row in source_pairs
                if row["hard_negative_same_pred_category"]
            }),
        }
    return {
        "rows": len(observations),
        "sources": len({row.source for row in observations}),
        "videos": len({(row.source, row.video_id) for row in observations}),
        "folds": len({row.fold for row in observations}),
        "known_identity_rows": sum(
            gt_by_id[row.observation_id] is not None for row in observations),
        "unknown_identity_rows_excluded": sum(
            gt_by_id[row.observation_id] is None for row in observations),
        "primary_positive_pairs": sum(row["same_gt"] for row in primary),
        "primary_hard_negative_pairs": sum(
            row["hard_negative_same_pred_category"] for row in primary),
        "by_source": by_source,
        "unknown_identity_used_as_negative": False,
        "pair_counts_are_not_independent_samples": True,
    }


def evaluate(manifest_path: Path, labels: Sequence[Mapping[str, Any]],
             cache_dir: Path, out_dir: Path,
             source_filter: str | None = None,
             video_filter: int | None = None,
             expected_source_count: int | None = None,
             receipt_path: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    manifest, observations, load_receipt = load_observations(
        manifest_path, cache_dir, source_filter=source_filter,
        video_filter=video_filter, receipt_path=receipt_path)
    gt_by_id = labels_by_observation(labels)
    require(set(gt_by_id) >= {row.observation_id for row in observations},
            "Evaluation labels do not cover selected observations")
    # Feature scores are computed before the labels are used.  This ordering is
    # also encoded structurally by Observation not carrying GT fields.
    pair_rows = build_pair_table(observations, gt_by_id)
    require(_primary_pairs(pair_rows), "No positive/hard-negative pair support")
    auroc_by_arm = {arm: source_macro_auroc(pair_rows, arm) for arm in ARMS}
    top1 = top1_metrics(observations, gt_by_id)
    support = _support_summary(observations, pair_rows, gt_by_id)
    fold_status = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold"])
        fold_obs = [row for row in observations if row.fold == fold_index]
        fold_pairs = [
            row for row in _primary_pairs(pair_rows)
            if int(row["fold"]) == fold_index
        ]
        fold_status.append({
            "fold": fold_index,
            "sources": list(fold["sources"]),
            "available_observations": len(fold_obs),
            "available_positive_pairs": sum(row["same_gt"] for row in fold_pairs),
            "available_hard_negative_pairs": sum(
                row["hard_negative_same_pred_category"] for row in fold_pairs),
            "evaluated": bool(fold_obs),
        })
    complete_three_fold = all(
        row["evaluated"] and row["available_positive_pairs"] >= 30
        and row["available_hard_negative_pairs"] >= 30
        for row in fold_status
    )
    if expected_source_count is not None:
        complete_three_fold = (
            complete_three_fold
            and source_filter is None
            and video_filter is None
            and support["sources"] >= int(expected_source_count)
        )
    status = ("PASS_REAL_THREE_FOLD_INFORMATION_GATE"
              if complete_three_fold and source_filter is None
              else "PASS_SMOKE_ONLY_REAL_CACHE_INCOMPLETE")
    result = {
        "status": status,
        "scope": "evaluation-only fixed features; not a trained module and not tracker evidence",
        "protocol": {
            "primary_pairs": "same GT vs different GT with the same predicted category",
            "pair_unit": "cross-step unordered row pair within one video",
            "top1_population": "per-query known cross-step candidates containing positive and hard-negative references",
            "top1_tie_break": "higher score, then lower immutable dump_row, then lower observation_id",
            "source_macro": "unweighted mean over sources with both classes",
            "fold_macro": "unweighted mean of evaluable source/fold AUROCs",
            "unknown_identity_policy": "exclude from pair AUROC; never a negative",
            "unknown_top1_policy": "keep in the GT-free candidate pool; count selected unknown as unresolved",
            "gt_query_population_filter": True,
            "gt_removes_or_adds_candidates": False,
            "feature_gt_access": False,
            "gt_used_to_select_memory_or_reference": False,
            "topk": TOPK,
            "grid_prior_weight": GRID_PRIOR_WEIGHT,
            "shape_prior_weight": SHAPE_PRIOR_WEIGHT,
            "arms": list(ARMS),
        },
        "manifest_sha256": sha256_path(manifest_path),
        "cache_directory": str(Path(cache_dir)),
        "load_receipt": load_receipt,
        "support": support,
        "expected_source_count": expected_source_count,
        "source_coverage_complete": (
            expected_source_count is None
            or support["sources"] >= int(expected_source_count)),
        "fold_status": fold_status,
        "complete_three_fold": complete_three_fold,
        "auroc": auroc_by_arm,
        "top1": top1,
        "limitations": [
            "A real three-fold PASS requires complete source/video/frame coverage in the selected receipt.",
            "If the receipt is an independent audit receipt, the token directory still lacks a formal collector terminal receipt.",
            "The top1 candidate set is an evaluation-only known-identity set and is not deployment no-match behavior.",
            "Pair support is correlated through shared rows, videos, and GT trajectories.",
            "Query/key local-token scores have no trained parameters; this is an information gate, not a proposed network result.",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "cpu_only": True,
        "gpu_used": False,
        "model_loaded": False,
        "training_started": False,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "evaluation_result.json", result)
    folds_dir = out_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    for fold in fold_status:
        fold_index = int(fold["fold"])
        fold_observations = [row for row in observations if row.fold == fold_index]
        fold_pairs = [
            row for row in _primary_pairs(pair_rows)
            if int(row["fold"]) == fold_index
        ]
        _write_json(folds_dir / f"fold_{fold_index}.json", {
            "fold": fold_index,
            "status": ("PASS_FOLD_INFORMATION_GATE" if fold["evaluated"]
                       else "UNAVAILABLE_CACHE_DOES_NOT_COVER_FOLD"),
            "sources": fold["sources"],
            "observations": len(fold_observations),
            "primary_pairs": len(fold_pairs),
            "support": _support_summary(fold_observations, fold_pairs, gt_by_id),
            "auroc": {
                arm: source_macro_auroc(fold_pairs, arm) for arm in ARMS
            } if fold_pairs else {},
            "gt_used_only_for_scoring": True,
        })
    return result


def run_protocol(manifest_path: Path, cache_dir: Path, out_dir: Path,
                 source_filter: str | None = None,
                 video_filter: int | None = None,
                 expected_source_count: int | None = None,
                 receipt_path: Path | None = None) -> dict[str, Any]:
    manifest, model_rows, labels = load_manifest_rows(manifest_path)
    manifest = dict(manifest)
    manifest["_source_path"] = str(Path(manifest_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    separation = export_separated_protocol(manifest, model_rows, labels, out_dir)
    # Reload the literal separated labels so the evaluation path cannot
    # accidentally depend on the manifest object containing both sides.
    separated_labels = _read_jsonl(Path(separation["labels"]))
    return evaluate(manifest_path, separated_labels, cache_dir, out_dir,
                    source_filter=source_filter, video_filter=video_filter,
                    expected_source_count=expected_source_count,
                    receipt_path=receipt_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=str, default=None,
                        help="Explicit source for a smoke run; omit for all three folds")
    parser.add_argument("--video-id", type=int, default=None,
                        help="Optional single-video smoke filter")
    parser.add_argument("--expected-source-count", type=int, default=6,
                        help="Require all sources for a real three-fold PASS")
    parser.add_argument("--receipt", type=Path, default=None,
                        help="Explicit terminal or independent ROI audit receipt")
    args = parser.parse_args(argv)
    try:
        result = run_protocol(
            args.manifest, args.cache_dir, args.out, args.source, args.video_id,
            args.expected_source_count, args.receipt)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": result["status"],
        "support": {
            "rows": result["support"]["rows"],
            "sources": result["support"]["sources"],
            "folds": result["support"]["folds"],
            "positive_pairs": result["support"]["primary_positive_pairs"],
            "hard_negative_pairs": result["support"]["primary_hard_negative_pairs"],
        },
        "source_macro_auroc": {
            arm: result["auroc"][arm]["source_macro"] for arm in ARMS
        },
        "repair_break_vs_raw": {
            arm: result["top1"][arm]["paired_vs_raw256"] for arm in ARMS
        },
        "matched_contrasts": result["top1"]["matched_contrasts"],
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
