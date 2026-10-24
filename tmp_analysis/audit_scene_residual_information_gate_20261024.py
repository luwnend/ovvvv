#!/usr/bin/env python3
"""Independent read-only audit of the completed scene-residual gate."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scene_residual_information_gate_20261024_v1"
PRIOR = ROOT / "results/spacetime_candidate_features_20261024_v1"
MODULE = ROOT / "tmp_analysis/all_candidate_scene_residual_20261024.py"
SELECTION = ROOT / "tmp_analysis/context_feature_memory_replay_protocol_20261024.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(v, msg):
    if not v:
        raise RuntimeError(msg)


def main():
    seal = json.loads((OUT / "GT_free_residual_sealed.json").read_text())
    require(seal["status"] == "SEALED_ALL_GT_FREE_SCENE_RESIDUALS_BEFORE_IDENTITY_READOUT", "seal status")
    receipts = json.loads((OUT / "pair_receipts.json").read_text())
    candidate = dict(np.load(PRIOR / "candidate_features.npz", allow_pickle=False))
    prior_receipts = [r for r in json.loads((PRIOR / "frame_receipts.json").read_text()) if int(r["candidate_rows"]) > 0]
    require(len(receipts) == len(prior_receipts) == 677, "receipt population")
    cursor = 0; total = 0; available = 0; max_gain_error = 0.0; max_raw_error = 0.0; files = 0
    for rec, prior in zip(receipts, prior_receipts):
        name = f"v{rec['video_id']}_f{rec['query_frame_id']}_r{rec['reference_frame_id']}.npz"
        path = OUT / "pairs" / name
        require(path.exists() and sha(path) == rec["output_sha256"] and path.stat().st_size == rec["output_bytes"], "output hash")
        with np.load(path, allow_pickle=False) as z:
            count = int(rec["candidate_count"])
            raw = np.asarray(z["raw_cosine"]).reshape(-1)
            static = np.asarray(z["static_abs_L1"]).reshape(-1)
            corrected = np.asarray(z["corrected_L1"]).reshape(-1)
            gain = np.asarray(z["gain"]).reshape(-1)
            avail = np.asarray(z["available"]).reshape(-1).astype(bool)
            anchors = np.asarray(z["anchor_count"]).reshape(-1)
            fit_index = np.asarray(z["fit_index"]).reshape(-1)
            require(len(raw) == count and np.isfinite(raw).all() and np.isfinite(static).all(), "core shape/finite")
            require(np.all(np.isfinite(corrected[avail])) and np.all(np.isnan(corrected[~avail])), "missing mask")
            require(np.all(np.isfinite(gain[avail])) and np.all(np.isnan(gain[~avail])), "gain mask")
            require(np.all(anchors[avail] >= 3) and np.all(anchors[~avail] < 3), "anchor mask")
            max_gain_error = max(max_gain_error, float(np.nanmax(np.abs(gain[avail] - (static[avail] - corrected[avail])))) if avail.any() else 0.0)
            prior_raw = candidate["scores"][cursor:cursor + count, 0]
            max_raw_error = max(max_raw_error, float(np.max(np.abs(raw - prior_raw))))
            require(rec["video_id"] == prior["video_id"] and rec["query_image_id"] == prior["image_id"] and count == prior["candidate_rows"], "row order")
            total += count; available += int(avail.sum()); cursor += count; files += 1
    require(cursor == len(candidate["scores"]) == 1210939 and total == 1210939 and available == 1207931, "totals")
    require(max_gain_error <= 2e-12 and max_raw_error <= 1e-12, "arithmetic/source parity")
    summary = json.loads((OUT / "summary.json").read_text())
    require(summary["status"] == "COMPLETE_GT_FREE_SCENE_RESIDUAL_INFORMATION_GATE", "summary status")
    require(summary["candidate_count"] == total and summary["available_candidates"] == available, "summary totals")
    result = {"status": "PASS_INDEPENDENT_SCENE_RESIDUAL_INFORMATION_OUTPUT_AUDIT",
              "pair_files": files, "candidate_count": total, "available_candidates": available,
              "max_gain_identity_error": max_gain_error, "max_raw_source_error": max_raw_error,
              "module_sha256": sha(MODULE), "selection_sha256": sha(SELECTION),
              "seal_sha256": sha(OUT / "GT_free_residual_sealed.json"),
              "summary_sha256": sha(OUT / "summary.json"), "unknown_not_negative": True,
              "read_only": True}
    (OUT / "independent_output_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
