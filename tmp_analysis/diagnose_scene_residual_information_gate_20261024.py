#!/usr/bin/env python3
"""Stream full candidate scene residuals and run a fixed information gate.

The GT-free residual export is complete before the previously sealed FIT identity
projection is read.  Each adjacent cached frame pair is processed independently;
only compact candidate arrays and hashes of the exclusion/anchor provenance are
stored.  The readout compares raw appearance with static displacement,
translation-corrected residual, gain, and a fixed sign-reversed placebo on the
same supported known candidates.  Unknown endpoints never become negatives.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/base_error_full_collection_20261023"
RUN = BASE / "full_collect_v1"
PRIOR = ROOT / "results/spacetime_candidate_features_20261024_v1"
SELECTION = ROOT / "tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
MODULE = ROOT / "tmp_analysis/all_candidate_scene_residual_20261024.py"
MODULE_SHA = "0824dbd527563496cf4d3275ec2de169295bf3d72c4eef145529ebd04cf5b7be"
SELECTION_SHA = "740d6c782fdfe8240ddc8c4b924e566cebd2465a6a05233f02197b1e904aaf1b"
OUT = ROOT / "results/scene_residual_information_gate_20261024_v1"
PROTOCOL = ROOT / "tmp_analysis/scene_residual_information_gate_protocol_20261024.json"
PROTOCOL_SHA = "__FILLED_AFTER_WRITE__"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    require(labels.shape == scores.shape and np.isfinite(scores).all(), "AUC input")
    p = int(np.count_nonzero(labels == 1)); n = int(np.count_nonzero(labels == 0))
    if not p or not n:
        return None
    order = np.argsort(scores, kind="stable")
    values, truth = scores[order], labels[order]
    total = 0.0; negatives_before = 0; start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        positives = int(np.count_nonzero(truth[start:end] == 1))
        ties = end - start - positives
        total += positives * (negatives_before + 0.5 * ties)
        negatives_before += ties
        start = end
    return float(total / (p * n))


def compact_digest(records):
    """Digest only exclusion keys and anchor pairs; do not persist all tuples."""
    h = hashlib.sha256()
    for record in records:
        value = {
            "q": list(record["excluded_query_rows"]),
            "r": list(record["excluded_reference_rows"]),
            "a": [list(x) for x in record["anchor_pairs"]],
            "available": bool(record["available"]),
            "count": int(record["anchor_count"]),
        }
        h.update(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def load_cache(record):
    path = Path(record["npz"])
    require(path.parent == RUN / "dump" and sha(path) == record["npz_sha256"], "cache pin")
    with np.load(path, allow_pickle=False) as z:
        require(int(z["video_id"]) == record["video_id"], "video locator")
        require(int(z["image_id"]) == record["image_id"], "image locator")
        require(int(z["frame_id"]) == record["frame_id"], "sample locator")
        require(int(z["frame_index"]) == record["frame_index"], "frame locator")
        boxes = np.ascontiguousarray(z["det_bboxes"][:, :4], dtype=np.float64)
        raw = np.ascontiguousarray(z["track_feats"], dtype=np.float64)
        proposals = np.ascontiguousarray(z["roi_proposal_row"], dtype=np.int64)
    require(boxes.shape[0] == record["observations"] and boxes.shape[1] == 4, "box shape")
    require(raw.shape == (len(boxes), 256) and proposals.shape == (len(boxes),), "feature shape")
    return boxes, raw, proposals


def fixed_protocol():
    require(sha(SELECTION) == SELECTION_SHA, "selection protocol changed")
    require(sha(MODULE) == MODULE_SHA, "residual module changed")
    protocol = json.loads(SELECTION.read_text())
    frames = [json.loads(line) for line in (RUN / "frame_execution.jsonl").open()]
    wanted = set(protocol["video_ids"])
    frames = [r for r in frames if r["video_id"] in wanted]
    require(len(frames) == 697 and len({r["video_id"] for r in frames}) == 20, "fixed frame population")
    inventory = json.loads((RUN / "dump_inventory.json").read_text())
    for r in frames:
        path = Path(r["npz"])
        require(path.name in inventory and inventory[path.name] == sha(path) == r["npz_sha256"], "frame inventory")
    prior_seal = json.loads((PRIOR / "GT_free_candidates_sealed.json").read_text())
    require(prior_seal["status"] == "SEALED_ALL_GT_FREE_CANDIDATES_BEFORE_SUPERVISION", "prior candidate seal")
    require(sha(PRIOR / "candidate_features.npz") == prior_seal["candidate_features_sha256"], "prior candidate hash")
    require(sha(PRIOR / "frame_receipts.json") == prior_seal["frame_receipts_sha256"], "prior receipt hash")
    return protocol, frames, prior_seal, inventory


def candidate_readout(candidate, labels, residual):
    raw = candidate["scores"][:, 0]
    source = candidate["source_index"].astype(np.int64)
    video = candidate["video_id"].astype(np.int64)
    qi = candidate["query_image_id"].astype(np.int64)
    qr = candidate["query_row"].astype(np.int64)
    rr = candidate["reference_row"].astype(np.int64)
    truth = labels["identity_label"].astype(np.int8)
    require(len(truth) == len(raw) == len(residual["corrected_L1"]), "candidate/residual alignment")
    support = residual["available"].astype(bool) & np.isfinite(residual["corrected_L1"])
    scores = {
        "raw_appearance_cosine": raw,
        "static_displacement": -candidate["scores"][:, 1] * 0.0,  # replaced below
        "scene_residual": -residual["corrected_L1"],
        "scene_gain": residual["gain"],
        "reversed_residual_placebo": residual["corrected_L1"],
    }
    # The static cue is the stored residual module's raw-pixel normalized L1.
    scores["static_displacement"] = -residual["static_abs_L1"]
    def report(mask, name):
        known = mask & (truth >= 0)
        result = {"candidates": int(mask.sum()), "known": int(known.sum()),
                  "positive": int(np.count_nonzero(truth[known] == 1)),
                  "negative": int(np.count_nonzero(truth[known] == 0)),
                  "unknown": int(np.count_nonzero(mask & (truth < 0))), "auc": {}}
        for key, value in scores.items():
            finite = known & np.isfinite(value)
            result["auc"][key] = auc(truth[finite], value[finite]) if finite.any() else None
            result.setdefault("support", {})[key] = int(finite.sum())
        return result
    result = {"all_candidates": report(np.ones(len(raw), dtype=bool), "all"),
              "residual_supported": report(support, "supported")}
    # Source-macro readout prevents a large source from hiding a failure on
    # other source groups.  The source index is metadata from the fixed
    # selection and is never used to construct a score.
    for scope_name, mask in (("all_candidates", np.ones(len(raw), dtype=bool)),
                             ("residual_supported", support)):
        macro = {}
        for key, value in scores.items():
            values = []
            for source_id in sorted(set(source.tolist())):
                finite = mask & (truth >= 0) & (source == source_id) & np.isfinite(value)
                if np.count_nonzero(truth[finite] == 1) and np.count_nonzero(truth[finite] == 0):
                    values.append(auc(truth[finite], value[finite]))
            macro[key] = {"mean": float(np.mean(values)) if values else None,
                          "supported_sources": len(values)}
        result[scope_name]["source_macro_auc"] = macro
    # Common residual-supported query population: raw and every residual score
    # use exactly the same known positive/negative candidate set.
    rows = defaultdict(list)
    selected = np.flatnonzero(support & (truth >= 0))
    for k in selected:
        rows[(int(video[k]), int(qi[k]), int(qr[k]))].append(int(k))
    transition = {key: {"raw_correct_score_correct": 0, "raw_correct_score_wrong": 0,
                         "raw_wrong_score_correct": 0, "raw_wrong_score_wrong": 0}
                  for key in scores if key != "raw_appearance_cosine"}
    query_coverage = {"queries": 0, "positive_negative_queries": 0,
                      "sources": set(), "videos": set()}
    for group in rows.values():
        group = np.asarray(group, dtype=np.int64)
        pos = truth[group] == 1; neg = truth[group] == 0
        if not pos.any() or not neg.any():
            continue
        query_coverage["positive_negative_queries"] += 1
        query_coverage["sources"].add(int(source[group[0]])); query_coverage["videos"].add(int(video[group[0]]))
        winners = {}
        for key, value in scores.items():
            finite = np.isfinite(value[group])
            require(finite.all(), "common ranking denominator")
            winners[key] = int(group[np.lexsort((rr[group], -value[group]))[0]])
        raw_ok = truth[winners["raw_appearance_cosine"]] == 1
        for key in transition:
            good = truth[winners[key]] == 1
            transition[key][("raw_correct_" if raw_ok else "raw_wrong_") + ("score_correct" if good else "score_wrong")] += 1
    query_coverage["queries"] = len(rows)
    query_coverage["sources"] = len(query_coverage["sources"]); query_coverage["videos"] = len(query_coverage["videos"])
    return {"population": result, "common_supported_query_ranking": {
        "coverage": query_coverage, "paired_top1": transition,
        "tie_rule": "highest score then lowest immutable reference_row",
        "unknown_not_negative": True,
        "interpretation": "known supported candidate ranking only; no deployment tracking metric"}}


def execute():
    started = time.monotonic()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be hidden")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(name) == "1", "single CPU thread")
    protocol, frames, prior_seal, inventory = fixed_protocol()
    require(not OUT.exists(), "no overwrite")
    OUT.mkdir(parents=True)
    pair_dir = OUT / "pairs"; pair_dir.mkdir()
    candidate = dict(np.load(PRIOR / "candidate_features.npz", allow_pickle=False))
    labels = dict(np.load(PRIOR / "fit_identity_labels.npz", allow_pickle=False))
    # The prior diagnostic keeps one receipt for the first frame of each video
    # with zero candidates; residual blocks begin at the first adjacent pair.
    prior_receipts = [r for r in json.loads((PRIOR / "frame_receipts.json").read_text())
                      if int(r["candidate_rows"]) > 0]
    module = load_module(MODULE, "frozen_all_candidate_scene_residual")
    by_video = defaultdict(list)
    for r in frames:
        by_video[int(r["video_id"])].append(r)
    source_index = {name: i for i, name in enumerate(protocol["source_groups"])}
    pair_receipts = []; global_parts = []
    cursor = 0
    for video in sorted(by_video):
        recs = sorted(by_video[video], key=lambda x: x["frame_id"])
        for query_rec, ref_rec in zip(recs[1:], recs[:-1]):
            qb, qf, qp = load_cache(query_rec); rb, rf, rp = load_cache(ref_rec)
            begun = time.monotonic()
            result = module.all_candidate_scene_residual(qb, rb, qf, rf, qp, rp, minimum_anchors=3, memoize=True)
            arrays = result["arrays"]
            count = int(result["candidate_count"])
            require(count == int(query_rec["observations"]) * int(ref_rec["observations"]), "full candidate count")
            require(cursor < len(prior_receipts), "prior receipt cursor")
            prior = prior_receipts[cursor]
            require(prior["video_id"] == video and prior["image_id"] == query_rec["image_id"] and prior["candidate_rows"] == count, "candidate order")
            keep = {k: arrays[k] for k in ("raw_cosine", "static_abs_L1", "corrected_L1", "gain", "available", "anchor_count", "fit_index", "residual_dxdy", "translation_xy", "residual_mad_xy")}
            path = pair_dir / f"v{video}_f{query_rec['frame_id']}_r{ref_rec['frame_id']}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, **keep)
            pair = dict(video_id=video, query_image_id=query_rec["image_id"], reference_image_id=ref_rec["image_id"],
                        query_frame_id=query_rec["frame_id"], reference_frame_id=ref_rec["frame_id"],
                        candidate_count=count, available=int(arrays["available"].sum()),
                        donor_fit_count=int(result["donor_fits_computed"]),
                        fit_provenance_digest=compact_digest(result["fit_records"]),
                        fit_index_sha256=hashlib.sha256(np.asarray(arrays["fit_index"], dtype=np.int64).tobytes()).hexdigest(),
                        output_sha256=sha(path), output_bytes=path.stat().st_size,
                        elapsed_seconds=time.monotonic()-begun,
                        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            pair_receipts.append(pair)
            # Pair matrices have different Q/M shapes; concatenate their
            # immutable row-major candidate order as one-dimensional blocks.
            global_parts.append({k: arrays[k].reshape(-1) for k in ("corrected_L1", "gain", "static_abs_L1", "available")})
            cursor += 1
            print(json.dumps(pair), flush=True)
    require(cursor == len(prior_receipts) and cursor == 677, "all adjacent pairs")
    residual = {k: np.concatenate([x[k] for x in global_parts]) for k in global_parts[0]}
    write_new(OUT / "GT_free_residual_sealed.json", {
        "status": "SEALED_ALL_GT_FREE_SCENE_RESIDUALS_BEFORE_IDENTITY_READOUT",
        "pair_count": cursor, "candidate_count": int(len(residual["gain"])),
        "pair_receipts_sha256": sha(OUT / "pair_receipts.json") if (OUT / "pair_receipts.json").exists() else None,
        "module_sha256": sha(MODULE), "selection_sha256": sha(SELECTION),
        "unknown_identity_not_read": True, "candidate_filtering": False,
    })
    write_new(OUT / "pair_receipts.json", pair_receipts)
    # The seal is rewritten only before any readout; use a fresh immutable receipt.
    seal_path = OUT / "GT_free_residual_sealed.json"
    seal = json.loads(seal_path.read_text()); seal["pair_receipts_sha256"] = sha(OUT / "pair_receipts.json")
    seal["pair_output_sha256"] = {p.name: p["output_sha256"] for p in pair_receipts}
    seal_path.unlink(); write_new(seal_path, seal)
    # Readout uses only the previously sealed, independently audited fit labels.
    readout = candidate_readout(candidate, labels, residual)
    summary = {"status": "COMPLETE_GT_FREE_SCENE_RESIDUAL_INFORMATION_GATE",
               "candidate_count": int(len(residual["gain"])), "pair_count": cursor,
               "available_candidates": int(residual["available"].sum()),
               "elapsed_seconds": time.monotonic()-started,
               "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "readout": readout,
               "interpretation": "Fixed exploratory FIT population; unknown endpoints are excluded from labels, residual supported ranking shares one denominator, and no tracker/training/novelty claim is made."}
    write_new(OUT / "summary.json", summary)
    write_new(OUT / "completion_receipt.json", {"status": "PASS_SCENE_RESIDUAL_INFORMATION_GATE",
        "summary_sha256": sha(OUT / "summary.json"), "seal_sha256": sha(seal_path),
        "pair_receipts_sha256": sha(OUT / "pair_receipts.json"),
        "candidate_count": int(len(residual["gain"])), "pair_count": cursor,
        "actual_exit_code": 0, "GPU": False, "training": False})


if __name__ == "__main__":
    execute()
