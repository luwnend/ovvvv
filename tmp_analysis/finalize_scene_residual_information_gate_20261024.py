#!/usr/bin/env python3
"""Finalize an already completed GT-free scene-residual export."""
import hashlib
import json
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scene_residual_information_gate_20261024_v1"
PRIOR = ROOT / "results/spacetime_candidate_features_20261024_v1"
MODULE = ROOT / "tmp_analysis/all_candidate_scene_residual_20261024.py"
MODULE_SHA = "0824dbd527563496cf4d3275ec2de169295bf3d72c4eef145529ebd04cf5b7be"
SELECTION = ROOT / "tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
SELECTION_SHA = "740d6c782fdfe8240ddc8c4b924e566cebd2465a6a05233f02197b1e904aaf1b"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(v, msg):
    if not v:
        raise RuntimeError(msg)


def load_gate():
    path = ROOT / "tmp_analysis/diagnose_scene_residual_information_gate_20261024.py"
    spec = importlib.util.spec_from_file_location("scene_gate", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    require(sha(MODULE) == MODULE_SHA and sha(SELECTION) == SELECTION_SHA, "pinned source changed")
    seal_path = OUT / "GT_free_residual_sealed.json"
    seal = json.loads(seal_path.read_text())
    require(seal["status"] == "SEALED_ALL_GT_FREE_SCENE_RESIDUALS_BEFORE_IDENTITY_READOUT" and seal["unknown_identity_not_read"], "GT-free seal")
    receipts = json.loads((OUT / "pair_receipts.json").read_text())
    require(len(receipts) == 677, "pair receipt count")
    parts = []
    output_hashes = {}
    total = 0
    available = 0
    for rec in receipts:
        path = OUT / "pairs" / f"v{rec['video_id']}_f{rec['query_frame_id']}_r{rec['reference_frame_id']}.npz"
        require(path.exists() and sha(path) == rec["output_sha256"] and path.stat().st_size == rec["output_bytes"], "pair output hash")
        with np.load(path, allow_pickle=False) as z:
            count = int(rec["candidate_count"])
            require(z["corrected_L1"].shape == (count, count) or z["corrected_L1"].size == count, "pair residual shape")
            fields = {k: np.asarray(z[k]).reshape(-1) for k in ("corrected_L1", "gain", "static_abs_L1", "available")}
            require(len(fields["gain"]) == count and np.all(np.isfinite(fields["static_abs_L1"])), "pair values")
            require(np.all(np.isfinite(fields["corrected_L1"][fields["available"]])) and np.all(np.isnan(fields["corrected_L1"][~fields["available"]])), "missing residual contract")
            parts.append(fields)
            total += count; available += int(fields["available"].sum())
    require(total == 1210939 and available == 1207931, "population totals")
    residual = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    seal["pair_receipts_sha256"] = sha(OUT / "pair_receipts.json")
    seal["pair_output_sha256"] = {f"v{r['video_id']}_f{r['query_frame_id']}_r{r['reference_frame_id']}.npz": r["output_sha256"] for r in receipts}
    write(seal_path, seal)
    gate = load_gate()
    candidate = dict(np.load(PRIOR / "candidate_features.npz", allow_pickle=False))
    labels = dict(np.load(PRIOR / "fit_identity_labels.npz", allow_pickle=False))
    readout = gate.candidate_readout(candidate, labels, residual)
    summary = {"status": "COMPLETE_GT_FREE_SCENE_RESIDUAL_INFORMATION_GATE",
               "candidate_count": total, "pair_count": len(receipts), "available_candidates": available,
               "readout": readout,
               "interpretation": "Fixed exploratory FIT population. Unknown endpoints are excluded from labels, residual-supported ranking shares one denominator, and no tracker, training, independent generalization, physical camera, or novelty claim is made."}
    write(OUT / "summary.json", summary)
    write(OUT / "completion_receipt.json", {"status": "PASS_SCENE_RESIDUAL_INFORMATION_GATE",
        "summary_sha256": sha(OUT / "summary.json"), "seal_sha256": sha(seal_path),
        "pair_receipts_sha256": sha(OUT / "pair_receipts.json"), "pair_count": len(receipts),
        "candidate_count": total, "available_candidates": available, "actual_exit_code": 0,
        "GT_free_export_reused_after_finalizer_repair": True, "GPU": False, "training": False})


if __name__ == "__main__":
    main()
