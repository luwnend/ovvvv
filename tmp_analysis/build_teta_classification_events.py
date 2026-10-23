"""Record installed TETA classification matches for fixed boxes and IDs.

This is a scoring cache, never an input to a method. The installed evaluator
runs unchanged; the subclass records its returned Hungarian matches. Replaying
events must reproduce every class/video/alpha TP,FN,FP count for five separately
evaluated readouts before this cache can be used for new readout diagnostics.
Scope: non-exhaustive TETA50, fixed detections and identities, one label/track.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np
import teta
from teta.eval import eval_sequence

ROOT = Path(__file__).resolve().parents[1]
DATASET = None
CLASS_NAMES = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4194304), b""):
            h.update(block)
    return h.hexdigest()


class EventTETA(teta.metrics.TETA):
    def __init__(self, names):
        super().__init__(exhaustive=False)
        self.class_index = {name: i for i, name in enumerate(names)}
        self.current = None
        self.events = []

    def eval_sequence_single_thr(self, data, cls, cid2clsname, cls_fp_thr, thr):
        assert thr == 50
        self.current = self.class_index[cls]
        try:
            return super().eval_sequence_single_thr(data, cls, cid2clsname, cls_fp_thr, thr)
        finally:
            self.current = None

    def compute_matches(self, data, t, ga_score, gt_ids, tk_ids, alpha):
        rows, cols = super().compute_matches(data, t, ga_score, gt_ids, tk_ids, alpha)
        if self.current is not None:
            assert isinstance(alpha, list) and len(alpha) == 20
            for a in range(10, 20):
                for tid in tk_ids[cols[a]]:
                    self.events.append((self.current, int(data["tk_id_map"][tid]), a - 10))
        return rows, cols


def one_sequence(item):
    i, name = item
    metric = EventTETA(CLASS_NAMES)
    result = eval_sequence(name, DATASET, "OVTrack", CLASS_NAMES, [metric], ["TETA"])
    counts = np.stack([np.stack([result[n]["TETA"][50][k]
                                for k in ("Cls_TP", "Cls_FN", "Cls_FP")])
                       for n in CLASS_NAMES]).astype(np.int32)
    events = np.asarray(metric.events, dtype=np.int64).reshape(-1, 3)
    seq = np.full(len(events), i, np.int32)
    return i, events, seq, counts


def replay_counts(event, global_track_id, labels, category_ids, n_videos):
    """Exactly replay non-exhaustive classification contributions of matches."""
    order = np.argsort(global_track_id)
    ids = global_track_id[order]
    indexes = np.searchsorted(ids, event["global_track_id"])
    assert np.all(indexes < len(ids)) and np.array_equal(ids[indexes], event["global_track_id"])
    pred = labels[order[indexes]]
    cat_order = np.argsort(category_ids)
    cat = category_ids[cat_order]
    pred_idx = np.searchsorted(cat, pred)
    assert np.all(pred_idx < len(cat)) and np.array_equal(cat[pred_idx], pred)
    pred_idx = cat_order[pred_idx]
    gt_idx, vid, alpha = event["gt_class_index"], event["sequence_index"], event["alpha_index"]
    correct = pred_idx == gt_idx
    size = len(category_ids) * n_videos * 10
    true_bin = (gt_idx * n_videos + vid) * 10 + alpha
    pred_bin = (pred_idx * n_videos + vid) * 10 + alpha
    tp = np.bincount(true_bin[correct], minlength=size).reshape(len(category_ids), n_videos, 10)
    fn = np.bincount(true_bin[~correct], minlength=size).reshape(len(category_ids), n_videos, 10)
    fp = np.bincount(pred_bin[~correct], minlength=size).reshape(len(category_ids), n_videos, 10)
    return np.stack([tp, fn, fp], axis=2).astype(np.int32)


def main():
    global DATASET, CLASS_NAMES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results/dependence_audit_20261023/teta_events")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / "receipt.json"
    if receipt_path.exists() or (args.out / "classification_events.npz").exists():
        raise FileExistsError("Preserve existing event run: " + str(args.out))
    started = time.monotonic()
    old = ROOT / "results/tces_audit_20261023"
    bm = json.load((old / "readouts/tcc/metrics.json").open())
    baseline_receipt = json.load((old / "readouts/tcc/receipt.json").open())
    dcfg = teta.config.get_default_dataset_config()
    dcfg.update(TRACKERS_TO_EVAL=["OVTrack"], GT_FOLDER=baseline_receipt["ann"],
                TRACKER_SUB_FOLDER=baseline_receipt["result"],
                OUTPUT_FOLDER=str(args.out), PRINT_CONFIG=False)
    DATASET = teta.datasets.TAO(dcfg)
    _, sequences, CLASS_NAMES = DATASET.get_eval_info()
    assert CLASS_NAMES == bm["class_names"]
    with np.load(old / "readouts/tcc/classification_counts.npz", allow_pickle=False) as z:
        expected_counts = z["counts"]
        assert np.array_equal(z["sequence_names"], sequences)
    category_ids = np.array([DATASET.cls_name2clsid[n] for n in CLASS_NAMES], np.int64)
    receipt = {"status": "RECORDING", "scope": "Scoring-only fixed-geometry non-exhaustive TETA50",
               "script_sha256": sha256(__file__), "baseline_receipt": baseline_receipt,
               "matching_proof": "TAO preprocessing selects overlap sets and global assignments using geometry and identities; predicted category affects only classification counts in non-exhaustive mode."}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    assert sha256(baseline_receipt["result"]) == baseline_receipt["result_sha256"]
    assert sha256(baseline_receipt["ann"]) == baseline_receipt["ann_sha256"]
    for p, h in baseline_receipt["evaluator_files"].items():
        assert sha256(p) == h
    results, seq_parts = [], []
    observed_counts = np.empty_like(expected_counts)
    with mp.get_context("fork").Pool(args.cores) as pool:
        for j, (i, events, seq, counts) in enumerate(pool.imap(one_sequence, enumerate(sequences)), 1):
            assert np.array_equal(counts, expected_counts[:, i]), "Recorded official baseline counts differ"
            observed_counts[:, i] = counts
            results.append(events)
            seq_parts.append(seq)
            if j % 50 == 0:
                print("videos", j, "/", len(sequences), "seconds", round(time.monotonic()-started, 1), flush=True)
    all_events = np.concatenate(results)
    event = {"gt_class_index": all_events[:, 0].astype(np.int32),
             "global_track_id": all_events[:, 1], "alpha_index": all_events[:, 2].astype(np.int32),
             "sequence_index": np.concatenate(seq_parts)}
    parity = {}
    cache = np.load(old / "track_cache.npz", allow_pickle=False)
    cache_cat = cache["category_ids"]
    # Official class order is not required to be numeric category-ID order.
    for rule, field in [("tcc", None), ("fused_sumprob", "p_fused_sumprob"),
                        ("fused_sumlog", "p_fused_sumlog"),
                        ("corrected_mask_tcc", "p_corrected_mask_argmax_counts"),
                        ("corrected_mask_sumprob", "p_corrected_mask_sumprob")]:
        labels = cache["tcc_category_id"] if field is None else cache_cat[np.argmax(cache[field][:, :len(cache_cat)], axis=1)]
        counts = replay_counts(event, cache["global_track_id"], labels, category_ids, len(sequences))
        with np.load(old / ("readouts/" + rule + "/classification_counts.npz"), allow_pickle=False) as z:
            assert np.array_equal(counts, z["counts"]), rule + ": official per-video count parity failed"
        parity[rule] = {"all_class_video_alpha_TP_FN_FP_bitwise_equal": True}
        print("independent official count parity", rule, "PASS", flush=True)
    path = args.out / "classification_events.npz"
    np.savez_compressed(path, **event, class_names=np.asarray(CLASS_NAMES),
                        category_ids=category_ids, sequence_names=np.asarray(sequences))
    receipt.update(status="PASS_EXACT_FIXED_GEOMETRY_CLASSIFICATION_EVENT_REPLAY",
                   event_count=len(all_events), sequences=len(sequences), classes=len(CLASS_NAMES),
                   parity=parity, output_sha256=sha256(path), elapsed_seconds=time.monotonic()-started,
                   limitation="Only unchanged box/score/order/IDs and constant per-track labels under this exact evaluator; not an evaluation of changed detection or association. New main readouts still receive separate full evaluator checks.")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print("PASS", path, flush=True)


if __name__ == "__main__":
    main()
