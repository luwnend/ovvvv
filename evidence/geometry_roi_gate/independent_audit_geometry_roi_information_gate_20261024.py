#!/usr/bin/env python3
"""Independent CPU audit of the geometry-ROI information gate.

This script does not import the evaluator.  It reparses the frozen manifest,
joins the evaluation labels by observation ID, reloads raw spatial arrays, and
reimplements the deterministic arms and top1 population from the written
protocol.  It then compares its result with the evaluator output.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/geometry_roi_gate_manifest_20261024_v1/manifest.json"
CACHE = Path("/raid/datasets/chc_data/geometry_roi_full_20261024_v1")
OUTPUT = ROOT / "results/geometry_roi_information_gate_full_20261024_v4/evaluation_result.json"
RECEIPT = ROOT / "results/geometry_roi_full_audit_receipt_20261024_v2/audit_receipt.json"
ARMS = (
    "raw256", "raw256_causal_mean", "token_mean", "token_mean_causal",
    "token_center", "spatial_topk_geometry", "fgfa_aligned_past_mean",
    "random_token_permutation", "global_mean_geometry",
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator < 1e-12 else float(np.clip(a @ b / denominator, -1, 1))


def auroc(scores: list[float], labels: list[int]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return (sum(rank for rank, label in zip(ranks, labels) if label)
            - positive * (positive + 1) / 2) / (positive * negative)


def shape_prior(a: np.ndarray, b: np.ndarray) -> float:
    wa, ha = max(float(a[2] - a[0]), 1e-6), max(float(a[3] - a[1]), 1e-6)
    wb, hb = max(float(b[2] - b[0]), 1e-6), max(float(b[3] - b[1]), 1e-6)
    return abs(math.log(wa / wb)) + abs(math.log(ha / hb))


def load() -> tuple[dict, list[dict], dict[int, int | None]]:
    manifest = json.loads(MANIFEST.read_text())
    labels = {int(row["observation_id"]): (
        None if row["gt_track_id"] is None else int(row["gt_track_id"]))
        for row in manifest["evaluation_only_labels"]}
    receipt = json.loads(RECEIPT.read_text())
    check(receipt["status"] == "PASS_INDEPENDENT_FULL_ROI_AUDIT_RECEIPT_V1",
          "Receipt status")
    check(receipt["inputs"]["token_directory"] == str(CACHE),
          "Receipt token directory")
    receipt_by_file = {row["file"]: row for row in receipt["frames"]}
    rows = [dict(row) for row in manifest["model_input"]]
    by_file: dict[str, list[dict]] = {}
    for row in rows:
        by_file.setdefault(row["dump_file"], []).append(row)
    loaded: dict[str, dict] = {}
    for filename, file_rows in by_file.items():
        path = CACHE / filename
        check(sha(path) == receipt_by_file[filename]["sha256"], "ROI cache hash")
        with np.load(path, allow_pickle=False) as z:
            tokens = np.asarray(z["conv_spatial"])
            raw = np.asarray(z["raw_output"])
            boxes = np.asarray(z["det_bboxes"])
            meta = json.loads(str(z["preprocessing_json"].item()))
            check(str(z["source"].item()) == file_rows[0]["source_group"]
                  and int(z["video_id"].item()) == int(file_rows[0]["video_id"]),
                  "Frame metadata")
        loaded[filename] = dict(tokens=tokens, raw=raw, boxes=boxes, meta=meta)
    observations: list[dict] = []
    for row in rows:
        filename = row["dump_file"]
        index = int(row["dump_row"])
        frame = loaded[filename]
        height, width = float(frame["meta"]["img_shape"][0]), float(frame["meta"]["img_shape"][1])
        box = np.asarray(frame["boxes"][index, :4], dtype=np.float64)
        observations.append(dict(
            id=int(row["observation_id"]),
            source=str(row["source_group"]),
            video_id=int(row["video_id"]),
            step=int(row["sampled_step"]),
            category=int(row["pred_category_id"]),
            local_track=int(row["local_track_id"]),
            dump_row=index,
            tokens=np.asarray(frame["tokens"][index], dtype=np.float64),
            raw=np.asarray(frame["raw"][index], dtype=np.float64),
            box=np.asarray([box[0]/width, box[1]/height, box[2]/width, box[3]/height]),
            gt=labels[int(row["observation_id"])],
        ))
    observations.sort(key=lambda row: (
        row["source"], row["video_id"], row["step"], row["dump_row"], row["id"]))
    # Past-aligned grid: cumulative mean of all rows in a local track up to and
    # including each sampled step.
    by_track: dict[tuple[str, int, int], list[dict]] = {}
    for row in observations:
        by_track.setdefault(
            (row["source"], row["video_id"], row["local_track"]), []).append(row)
    for group in by_track.values():
        cumulative = np.zeros((256, 7, 7), dtype=np.float64)
        cumulative_raw = np.zeros(256, dtype=np.float64)
        count = 0
        for step in sorted({row["step"] for row in group}):
            at_step = [row for row in group if row["step"] == step]
            cumulative += sum((row["tokens"] for row in at_step), np.zeros_like(cumulative))
            cumulative_raw += sum((row["raw"] for row in at_step), np.zeros_like(cumulative_raw))
            count += len(at_step)
            descriptor = cumulative / count
            raw_descriptor = cumulative_raw / count
            mean_descriptor = descriptor.mean(axis=(1, 2))
            for row in at_step:
                row["past"] = descriptor
                row["past_raw"] = raw_descriptor
                row["past_mean"] = mean_descriptor
    return manifest, observations, labels


def spatial(q: dict, r: dict, *, permuted: bool) -> float:
    q_flat = q["tokens"].reshape(256, 49).T
    r_flat = r["tokens"].reshape(256, 49).T
    if permuted:
        digest = hashlib.sha256("|".join(sorted((
            f"{q['source']}|{q['video_id']}|{q['id']}",
            f"{r['source']}|{r['video_id']}|{r['id']}"))).encode()).hexdigest()
        r_flat = r_flat[np.random.default_rng(int(digest[:16], 16)).permutation(49)]
    similarity = unit(q_flat) @ unit(r_flat).T
    coordinates = np.linspace(-1, 1, 7)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    distance = ((xx[:, None] - xx[None, :]) ** 2 +
                (yy[:, None] - yy[None, :]) ** 2) / 4
    best = np.max(similarity - distance, axis=1)
    return float(np.mean(np.sort(best)[-5:]) - 0.02 * shape_prior(q["box"], r["box"]))


def score(arm: str, q: dict, r: dict) -> float:
    if arm == "raw256":
        return cos(q["raw"], r["raw"])
    if arm == "raw256_causal_mean":
        return cos(q["past_raw"], r["past_raw"])
    if arm in ("token_mean", "global_mean_geometry"):
        value = cos(q["tokens"].mean(axis=(1, 2)), r["tokens"].mean(axis=(1, 2)))
        return value - (0.02 * shape_prior(q["box"], r["box"])
                        if arm == "global_mean_geometry" else 0.0)
    if arm == "token_center":
        return cos(q["tokens"][:, 3, 3], r["tokens"][:, 3, 3])
    if arm == "token_mean_causal":
        return cos(q["past_mean"], r["past_mean"])
    if arm == "fgfa_aligned_past_mean":
        return cos(q["past"].reshape(-1), r["past"].reshape(-1))
    if arm == "spatial_topk_geometry":
        return spatial(q, r, permuted=False)
    if arm == "random_token_permutation":
        return spatial(q, r, permuted=True)
    raise AssertionError(arm)


def main() -> int:
    manifest, observations, labels = load()
    check(len(observations) == 122, "Expected 122 full-gate observations")
    pairs = []
    for index, first in enumerate(observations):
        if first["gt"] is None:
            continue
        for second in observations[index + 1:]:
            if ((first["source"], first["video_id"]) !=
                    (second["source"], second["video_id"])):
                continue
            if first["step"] == second["step"] or second["gt"] is None:
                continue
            same = int(first["gt"] == second["gt"])
            hard = bool(not same and first["category"] == second["category"])
            if same or hard:
                pairs.append(dict(q=first, r=second, same=same, hard=hard))
    check(len(pairs) == 370, "Pair denominator differs")
    source_auroc = {}
    for arm in ARMS:
        source_auroc[arm] = auroc(
            [score(arm, pair["q"], pair["r"]) for pair in pairs],
            [pair["same"] for pair in pairs])
    groups = []
    for query in observations:
        if query["gt"] is None:
            continue
        candidates = [row for row in observations
                      if (row["source"], row["video_id"]) ==
                      (query["source"], query["video_id"])
                      and row["id"] != query["id"] and row["step"] != query["step"]
                      and row["category"] == query["category"]]
        if not candidates:
            continue
        has_positive = any(row["gt"] == query["gt"] for row in candidates)
        has_hard = any(row["gt"] is not None and row["gt"] != query["gt"]
                       for row in candidates)
        if has_positive and has_hard:
            groups.append((query, candidates))
    check(len(groups) > 0, "No top1 queries")
    top1 = {}
    for arm in ARMS:
        correct = []
        for query, candidates in groups:
            ranked = sorted(
                ((row, score(arm, query, row)) for row in candidates),
                key=lambda pair: (-pair[1], pair[0]["dump_row"], pair[0]["id"]))
            selected = ranked[0][0]
            correct.append(None if selected["gt"] is None else selected["gt"] == query["gt"])
        resolved = [value for value in correct if value is not None]
        top1[arm] = dict(
            queries=len(correct),
            resolved_queries=len(resolved),
            unresolved_unknown_selection=len(correct) - len(resolved),
            correct=sum(resolved),
            accuracy_known_resolved=float(np.mean(resolved)) if resolved else None,
        )
    output = json.loads(OUTPUT.read_text())
    source_groups = {}
    for pair in pairs:
        source_groups.setdefault(pair["q"]["source"], []).append(pair)
    source_auroc = {
        arm: float(np.mean([
            auroc(
                [score(arm, pair["q"], pair["r"]) for pair in group],
                [pair["same"] for pair in group])
            for group in source_groups.values()
        ]))
        for arm in ARMS
    }
    max_auroc_difference = max(
        abs(source_auroc[arm] - output["auroc"][arm]["source_macro"])
        for arm in ARMS)
    require = max_auroc_difference <= 1e-12
    check(require, "Evaluator and independent AUROC differ")
    for arm in ARMS:
        for key in ("queries", "resolved_queries", "unresolved_unknown_selection", "correct"):
            check(top1[arm][key] == output["top1"][arm][key],
                  f"{arm} top1 {key} differs")
        check(abs(top1[arm]["accuracy_known_resolved"] -
                  output["top1"][arm]["accuracy_known_resolved"]) <= 1e-12,
              f"{arm} top1 accuracy differs")
    result = {
        "status": "PASS_INDEPENDENT_GEOMETRY_ROI_GATE_FULL_AUDIT",
        "scope": "six-source three-fold fixed-feature information gate",
        "evaluator_output_sha256": sha(OUTPUT),
        "manifest_sha256": sha(MANIFEST),
        "observations": len(observations),
        "primary_pairs": len(pairs),
        "positive_pairs": sum(pair["same"] for pair in pairs),
        "hard_negative_pairs": sum(pair["hard"] for pair in pairs),
        "max_auroc_abs_difference": max_auroc_difference,
        "top1_queries": len(groups),
        "top1": top1,
        "independent_auroc": source_auroc,
        "unknown_identity_used_as_negative": False,
        "gt_used_to_select_candidate": False,
        "cpu_only": True,
        "gpu_used": False,
        "training_started": False,
    }
    path = OUTPUT.parent / "independent_audit_full.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
