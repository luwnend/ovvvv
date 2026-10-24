#!/usr/bin/env python3
"""Seal a strict evaluator receipt from the independent full-ROI audit.

This does not fabricate a collector completion receipt.  The missing terminal
receipt and the post-generation manifest path rewrite remain explicit
provenance holds.  The new receipt only attests that all 276 token pages have
been independently checked for schema, hashes, exact parent raw/row/geometry
parity, finite spatial tensors, timeline order, and evaluation coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "tmp_analysis/geometry_roi_full_independent_audit_20261024.json"
DEFAULT_GATE = ROOT / "results/geometry_roi_gate_manifest_20261024_v1/manifest.json"
DEFAULT_MANIFEST = (
    ROOT / "results/geometry_roi_full_collect_manifest_20261024_v1/collector_manifest.json")
DEFAULT_EVALUATION = (
    ROOT / "results/geometry_roi_full_collect_manifest_20261024_v1/evaluation_frames.json")
DEFAULT_CACHE = Path("/raid/datasets/chc_data/geometry_roi_full_20261024_v1")
DEFAULT_OUT = ROOT / "results/geometry_roi_full_audit_receipt_20261024_v1/audit_receipt.json"
EXPECTED_FRAMES = 276
EXPECTED_EVALUATION_FRAMES = 72
EXPECTED_ANCHORS = 122
EXPECTED_SOURCES = 6
EXPECTED_FOLDS = 3
EXPECTED_AUDIT_STATUS = "PASS_PAYLOAD_WITH_PROVENANCE_HOLD"
RECEIPT_STATUS = "PASS_INDEPENDENT_FULL_ROI_AUDIT_RECEIPT_V1"


class SealError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise SealError(message)


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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    require(not path.exists(), f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    require(not temporary.exists(), f"Temporary path exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(
    audit_path: Path,
    gate_path: Path,
    collector_manifest_path: Path,
    evaluation_path: Path,
    cache_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    audit_path = Path(audit_path).resolve()
    gate_path = Path(gate_path).resolve()
    collector_manifest_path = Path(collector_manifest_path).resolve()
    evaluation_path = Path(evaluation_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    require(audit_path.is_file(), "Independent audit JSON is missing")
    require(gate_path.is_file(), "Gate manifest is missing")
    require(collector_manifest_path.is_file(), "Collector manifest is missing")
    require(evaluation_path.is_file(), "Evaluation manifest is missing")
    require(cache_dir.is_dir(), "Token directory is missing")
    audit = read_json(audit_path)
    gate = read_json(gate_path)
    collector = read_json(collector_manifest_path)
    evaluation = read_json(evaluation_path)

    require(audit.get("status") == EXPECTED_AUDIT_STATUS,
            "Independent audit is not the expected payload/provenance hold")
    conclusion = audit.get("conclusion")
    require(isinstance(conclusion, dict), "Audit conclusion is missing")
    required_conclusion = {
        "payload_valid_frames": EXPECTED_FRAMES,
        "payload_total_frames": EXPECTED_FRAMES,
        "all_276_expected_frames_present": True,
        "raw_and_geometry_exact_to_parent": True,
        "all_spatial_tokens_finite_float16": True,
    }
    for key, expected in required_conclusion.items():
        require(conclusion.get(key) == expected,
                f"Audit conclusion {key} differs: {conclusion.get(key)!r}")
    require(conclusion.get("terminal_receipt_present") is False,
            "This seal is only for the independently audited missing-receipt case")
    rounds = audit.get("rounds")
    require(isinstance(rounds, list) and len(rounds) == 3,
            "Independent audit must contain three rounds")
    for round_value in rounds:
        require(isinstance(round_value.get("checks"), list)
                and len(round_value["checks"]) >= 2,
                "Each audit round must contain at least two checks")
        require(all(check.get("pass") is True for check in round_value["checks"]),
                f"Audit round {round_value.get('round')} did not pass")
    provenance = rounds[0]["evidence"].get("provenance", {})
    require(provenance.get("terminal_collection_receipt_present") is False,
            "Unexpected terminal receipt state")
    require(provenance.get("started_pid_alive") is False,
            "Collector may still be running")
    diff = provenance.get("current_diff_vs_runtime_manifest")
    require(isinstance(diff, dict) and set(diff) == {"output_directory"},
            "Manifest rewrite is not limited to output_directory")
    require(Path(provenance.get("actual_token_dir", "")).resolve() == cache_dir,
            "Audit token directory differs from requested cache")
    require(provenance.get("pins_all_live_exact") is True,
            "Audit input pins do not all match live values")
    require(provenance.get("sha256sums_all_match") is True,
            "Audit SHA256SUMS do not all match")
    require(provenance.get("trace_pin_match") is True
            and provenance.get("runner_pin_match") is True,
            "Collector code pins differ")

    collector_sha = sha256_path(collector_manifest_path)
    require(audit.get("inputs", {}).get("manifest_sha256") == collector_sha,
            "Audit was not computed on the current collector manifest")
    require(audit.get("inputs", {}).get("token_dir") == str(cache_dir),
            "Audit input token directory differs")
    manifest_output = Path(collector.get("output_directory", "")).resolve()
    if manifest_output != cache_dir:
        common_stem = cache_dir.name.rsplit("_v", 1)[0]
        suffix = manifest_output.name[len(common_stem):] \
            if manifest_output.name.startswith(common_stem) else None
        require(
            manifest_output.parent == cache_dir.parent
            and suffix in {"_v1", "_v2", "_v3"},
            "Collector manifest path differs from a known same-base rewrite")
        require(not manifest_output.exists(),
                "Manifest's rewritten output directory unexpectedly exists")
    frames = collector.get("frames")
    require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES,
            "Collector manifest must contain 276 frame records")
    manifest_by_file: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        require(isinstance(frame, dict), f"Collector frame {index} is not an object")
        filename = f"v{frame['video_id']}_i{frame['image_id']}_f{frame['frame_id']}.npz"
        require(filename not in manifest_by_file, f"Duplicate frame name {filename}")
        manifest_by_file[filename] = frame
    disk_files = sorted(path.name for path in cache_dir.glob("*.npz"))
    require(disk_files == sorted(manifest_by_file),
            "Disk token files differ from the collector manifest")
    require(len(disk_files) == EXPECTED_FRAMES,
            "Unexpected number of token files")

    frame_statuses = audit.get("frame_statuses")
    require(isinstance(frame_statuses, list) and len(frame_statuses) == EXPECTED_FRAMES,
            "Audit frame status list differs from 276")
    audit_by_file: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(frame_statuses):
        filename = row.get("file")
        require(isinstance(filename, str) and filename.endswith(".npz"),
                f"Audit frame {index} lacks a filename")
        require(filename not in audit_by_file, f"Audit repeats {filename}")
        audit_by_file[filename] = row
    require(set(audit_by_file) == set(manifest_by_file),
            "Audit frame set differs from the manifest")
    sealed_frames = []
    for filename in sorted(manifest_by_file):
        manifest_frame = manifest_by_file[filename]
        audited = audit_by_file[filename]
        path = cache_dir / filename
        require(path.is_file(), f"Token file is missing: {path}")
        current_sha = sha256_path(path)
        require(audited.get("status") == "PASS", f"Audit frame failed: {filename}")
        require(audited.get("token_sha256") == current_sha,
                f"Live token SHA differs from audit: {filename}")
        for key in ("parent_hash_match", "image_hash_match", "roi_finite",
                    "conv_finite", "raw_exact_parent", "rows_exact_parent",
                    "boxes_exact_parent", "proposals_exact_parent"):
            require(audited.get(key) is True, f"{filename}: {key} did not pass")
        require(audited.get("problems") == [], f"{filename}: audit problems remain")
        for field in ("source", "video_id", "image_id", "frame_id"):
            require(audited.get(field) == manifest_frame[field],
                    f"{filename}: {field} differs from manifest")
        require(isinstance(audited.get("retained_rows"), int)
                and audited["retained_rows"] >= 0,
                f"{filename}: invalid retained rows")
        sealed_frames.append({
            "file": filename,
            "sha256": current_sha,
            "source": manifest_frame["source"],
            "video_id": int(manifest_frame["video_id"]),
            "image_id": int(manifest_frame["image_id"]),
            "frame_id": int(manifest_frame["frame_id"]),
            "parent_cache_path": manifest_frame["cache_path"],
            "parent_cache_sha256": manifest_frame["cache_sha256"],
            "image_path": manifest_frame["image_path"],
            "image_sha256": manifest_frame["image_sha256"],
            "retained_rows": int(audited["retained_rows"]),
            "raw_exact_parent": True,
            "row_locators_exact_parent": True,
            "box_geometry_exact_parent": True,
        })

    require(evaluation.get("status")
            == "PASS_GEOMETRY_ROI_FULL_PREFIX_EVALUATION_MANIFEST",
            "Evaluation manifest status differs")
    eval_frames = evaluation.get("evaluation_frames")
    anchors = evaluation.get("anchors")
    labels = evaluation.get("evaluation_only_labels")
    require(isinstance(eval_frames, list)
            and len(eval_frames) == EXPECTED_EVALUATION_FRAMES,
            "Evaluation manifest frame count differs")
    require(isinstance(anchors, list) and len(anchors) == EXPECTED_ANCHORS,
            "Evaluation manifest anchor count differs")
    require(isinstance(labels, list) and len(labels) == EXPECTED_ANCHORS,
            "Evaluation-only label count differs")
    require(anchors == gate.get("model_input"),
            "Evaluation anchors differ from the frozen gate model input")
    require(labels == gate.get("evaluation_only_labels"),
            "Evaluation labels differ from the frozen gate labels")
    file_by_image = {
        int(frame["image_id"]):
        f"v{frame['video_id']}_i{frame['image_id']}_f{frame['frame_id']}.npz"
        for frame in eval_frames
    }
    eval_frame_by_image = {
        int(frame["image_id"]): frame for frame in eval_frames
    }
    require(len(file_by_image) == EXPECTED_EVALUATION_FRAMES,
            "Evaluation image IDs are not unique")
    anchor_locators = []
    for index, anchor in enumerate(anchors):
        image_id = int(anchor["image_id"])
        require(image_id in file_by_image, f"Anchor {index} lacks an evaluation frame")
        eval_frame = eval_frame_by_image[image_id]
        require(
            anchor["source_group"] == eval_frame["source"]
            and int(anchor["video_id"]) == int(eval_frame["video_id"])
            and int(anchor["sampled_step"]) == int(eval_frame["sample_step"]),
            f"Anchor {index} metadata differs from its evaluation frame")
        filename = file_by_image[image_id]
        dump_row = int(anchor["dump_row"])
        with np.load(cache_dir / filename, allow_pickle=False) as token_file:
            require("conv_spatial" in token_file.files,
                    f"{filename}: missing conv_spatial")
            row_count = int(token_file["conv_spatial"].shape[0])
        require(0 <= dump_row < row_count,
                f"Anchor {index} row is outside the token cache")
        anchor_locators.append({
            "observation_id": int(anchor["observation_id"]),
            "file": filename,
            "dump_row": dump_row,
            "row_count": row_count,
        })
    require(len({row["file"] for row in anchor_locators})
            == EXPECTED_EVALUATION_FRAMES,
            "Evaluation anchors do not cover all 72 frames")
    sources = {anchor["source_group"] for anchor in anchors}
    require(len(sources) == EXPECTED_SOURCES, "Gate does not cover six sources")
    require(len(gate.get("folds", [])) == EXPECTED_FOLDS,
            "Gate does not define three folds")

    receipt = {
        "schema_version": "OVMOT_GEOMETRY_ROI_AUDIT_RECEIPT_V1",
        "status": RECEIPT_STATUS,
        "scope": (
            "Strict coverage receipt for the independently audited token payload. "
            "It is not a collector terminal completion receipt."
        ),
        "inputs": {
            "independent_audit_path": str(audit_path),
            "independent_audit_sha256": sha256_path(audit_path),
            "gate_manifest_path": str(gate_path),
            "gate_manifest_sha256": sha256_path(gate_path),
            "collector_manifest_path": str(collector_manifest_path),
            "collector_manifest_sha256": collector_sha,
            "evaluation_manifest_path": str(evaluation_path),
            "evaluation_manifest_sha256": sha256_path(evaluation_path),
            "token_directory": str(cache_dir),
        },
        "provenance": {
            "independent_audit_status": audit["status"],
            "terminal_collection_receipt_present": False,
            "formal_collector_completion_claimed": False,
            "manifest_rewrite": diff,
            "audit_interpretation": conclusion["interpretation"],
            "source_code_pins_exact": True,
            "terminal_process_absent": True,
        },
        "coverage": {
            "frames": EXPECTED_FRAMES,
            "raw_exact_parent_frames": EXPECTED_FRAMES,
            "row_and_geometry_exact_parent_frames": EXPECTED_FRAMES,
            "evaluation_frames": EXPECTED_EVALUATION_FRAMES,
            "anchors": EXPECTED_ANCHORS,
            "sources": len(sources),
            "folds": len(gate["folds"]),
            "all_anchor_rows_within_token_pages": True,
        },
        "frames": sealed_frames,
        "anchor_locators": anchor_locators,
        "evaluation_labels_sha256": hashlib.sha256(
            json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cpu_only": True,
        "gpu_used": False,
        "model_loaded": False,
        "training_started": False,
    }
    require(len({frame["file"] for frame in sealed_frames}) == EXPECTED_FRAMES,
            "Receipt frame names are not unique")
    write_json(out_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--collector-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        receipt = seal(
            args.audit, args.gate, args.collector_manifest,
            args.evaluation_manifest, args.cache_dir, args.out)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": receipt["status"],
        "frames": receipt["coverage"]["frames"],
        "evaluation_frames": receipt["coverage"]["evaluation_frames"],
        "anchors": receipt["coverage"]["anchors"],
        "output": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
