"""Paired attribution and video bootstrap from unrounded official TETA counts."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def classification(counts):
    # counts: classes x [TP,FN,FP] x classification alpha.
    return 100 * (counts[:, 0] / np.maximum(1, counts.sum(axis=1))).mean(axis=-1)


def paired_bootstrap(base, candidate, is_novel, reps, seed):
    """Sample videos jointly; keep the benchmark's fixed category universe.

    Recombine official counts before ratios and macro averaging. Never average
    stored per-video scores. This is a conditional validation-set diagnostic,
    not evidence of generalization or a pre-registered significance test.
    """
    assert base.shape == candidate.shape
    c, n_videos, _, _ = base.shape
    rng = np.random.RandomState(seed)
    flat_base = base.transpose(1, 0, 2, 3).reshape(n_videos, -1).astype(np.float64)
    flat_candidate = candidate.transpose(1, 0, 2, 3).reshape(n_videos, -1).astype(np.float64)
    delta = {"base": [], "novel": [], "combined": []}
    groups = {"base": ~is_novel, "novel": is_novel, "combined": np.ones(c, bool)}
    for start in range(0, reps, 50):
        weights = rng.multinomial(n_videos, np.full(n_videos, 1 / n_videos),
                                  size=min(50, reps - start)).astype(float)
        bs = (weights @ flat_base).reshape(-1, c, 3, 10)
        cs = (weights @ flat_candidate).reshape(-1, c, 3, 10)
        bscore = 100 * (bs[:, :, 0] / np.maximum(1, bs.sum(2))).mean(-1)
        cscore = 100 * (cs[:, :, 0] / np.maximum(1, cs.sum(2))).mean(-1)
        for name, mask in groups.items():
            delta[name].extend((cscore[:, mask] - bscore[:, mask]).mean(1).tolist())
    return {name: {"ClsA_delta_percentile_95": np.percentile(v, [2.5, 97.5]).tolist(),
                   "TETA_delta_percentile_95": (np.percentile(v, [2.5, 97.5]) / 3).tolist(),
                   "fraction_resamples_positive": float(np.mean(np.asarray(v) > 0))}
            for name, v in delta.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "results/tces_audit_20261023/readouts")
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--ann", type=Path, default=ROOT / "data_tao/annotations/validation_ours_v1.json")
    args = ap.parse_args()
    config = {"rules": args.rules, "reps": args.reps, "seed": args.seed}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = args.root.parent / ("paired_summary_" + run_id + ".json")
    if output.exists():
        raise FileExistsError("Preserve existing summary: " + str(output))
    baseline_dir = args.root / "tcc"
    baseline_receipt = json.load((baseline_dir / "receipt.json").open())
    assert baseline_receipt["rule"] == "tcc" and baseline_receipt["status"] == "COMPLETE"
    annotation_sha256 = sha256(args.ann)
    assert annotation_sha256 == baseline_receipt["ann_sha256"]
    for file, key in [("metrics.json", "metrics_sha256"),
                      ("classification_counts.npz", "counts_sha256"),
                      ("aggregate_raw.pkl", "aggregate_raw_sha256")]:
        assert sha256(baseline_dir / file) == baseline_receipt[key]
    bm = json.load((baseline_dir / "metrics.json").open())
    with (baseline_dir / "aggregate_raw.pkl").open("rb") as f:
        br = pickle.load(f)
    with np.load(baseline_dir / "classification_counts.npz", allow_pickle=False) as z:
        bcounts = z["counts"]
        names, videos, is_novel = z["class_names"], z["sequence_names"], z["is_novel"]
    bcls = classification(bcounts.sum(axis=1))
    assert np.allclose(bcls, np.asarray(bm["per_class"])[:, bm["fields"].index("ClsA")],
                       atol=1e-12, rtol=0)
    ann = json.load(args.ann.open())
    category = {c["id"]: c["name"] for c in ann["categories"]}
    image_video = {im["id"]: im["video_id"] for im in ann["images"]}
    gt_videos = {n: set() for n in names}
    for row in ann["annotations"]:
        gt_videos[category[row["category_id"]]].add(image_video[row["image_id"]])
    gt_video_support = np.array([len(gt_videos[n]) for n in names])
    cls_video_support = (bcounts[:, :, :2].sum((2, 3)) > 0).sum(1)
    support = {n: {"GT_videos": int(g), "localized_classification_videos": int(v)}
               for n, g, v in zip(names, gt_video_support, cls_video_support)}
    results = {}
    for rule in args.rules:
        d = args.root / rule
        m = json.load((d / "metrics.json").open())
        receipt = json.load((d / "receipt.json").open())
        assert receipt["status"] == "COMPLETE"
        assert receipt["rule"] == rule
        for key in ("baseline_sha256", "ann_sha256", "evaluator_files"):
            assert receipt[key] == baseline_receipt[key], rule + ": input lineage differs: " + key
        assert receipt["ann_sha256"] == annotation_sha256
        assert m["fields"] == bm["fields"]
        for file, key in [("metrics.json", "metrics_sha256"),
                          ("classification_counts.npz", "counts_sha256"),
                          ("aggregate_raw.pkl", "aggregate_raw_sha256")]:
            assert sha256(d / file) == receipt[key]
        assert m["class_names"] == bm["class_names"]
        with (d / "aggregate_raw.pkl").open("rb") as f:
            raw = pickle.load(f)
        invariant = ["Loc_TP", "Loc_FN", "Loc_FP", "LocA", "LocRe", "LocPr",
                     "AssocA", "AssocRe", "AssocPr"]
        assert all(np.array_equal(br[c][50][k], raw[c][50][k])
                   for c in br for k in invariant), rule + ": Loc/Assoc changed"
        with np.load(d / "classification_counts.npz", allow_pickle=False) as z:
            assert np.array_equal(names, z["class_names"])
            assert np.array_equal(videos, z["sequence_names"])
            assert np.array_equal(is_novel, z["is_novel"])
            ccounts = z["counts"]
        ccls = classification(ccounts.sum(axis=1))
        assert np.allclose(ccls, np.asarray(m["per_class"])[:, m["fields"].index("ClsA")],
                           atol=1e-12, rtol=0)
        assert np.array_equal(ccounts.sum(axis=1), np.stack([
            np.stack([raw[n][50][k] for k in ("Cls_TP", "Cls_FN", "Cls_FP")])
            for n in names
        ])), rule + ": video counts do not reproduce aggregate counts"
        delta = {s: {f: m["split"][s][f] - bm["split"][s][f]
                     for f in bm["fields"]} for s in bm["split"]}
        assert all(abs(v["TETA"] - v["ClsA"] / 3) < 1e-10 for v in delta.values())
        ci = paired_bootstrap(bcounts, ccounts, is_novel, args.reps, args.seed)
        # Delete one video at a time. This is influence analysis, not a CI.
        bminus = bcounts.sum(1)[:, None] - bcounts
        cminus = ccounts.sum(1)[:, None] - ccounts
        bscore = 100 * (bminus[:, :, 0] / np.maximum(1, bminus.sum(2))).mean(-1)
        cscore = 100 * (cminus[:, :, 0] / np.maximum(1, cminus.sum(2))).mean(-1)
        influence = {}
        for split, mask in [("base", ~is_novel), ("novel", is_novel),
                            ("combined", np.ones(len(names), bool))]:
            changes = (cscore[mask] - bscore[mask]).mean(0)
            j = int(np.argmax(np.abs(changes - delta[split]["ClsA"])))
            class_delta = ccls[mask] - bcls[mask]
            influence[split] = {
                "leave_one_video_out_ClsA_delta_range": [float(changes.min()), float(changes.max())],
                "most_influential_video": str(videos[j]),
                "ClsA_delta_without_that_video": float(changes[j]),
                "benefited_classes": int((class_delta > 1e-10).sum()),
                "harmed_classes": int((class_delta < -1e-10).sum()),
                "unchanged_classes": int((np.abs(class_delta) <= 1e-10).sum()),
            }
        results[rule] = {"split": m["split"], "delta_from_tcc": delta,
                         "paired_video_bootstrap": ci,
                         "influence_analysis": influence,
                         "raw_loc_assoc_all_arrays_identical": True,
                         "per_class_ClsA_delta": dict(zip(names.tolist(), (ccls - bcls).tolist())),
                         "receipt": receipt}
        print(json.dumps({"rule": rule, "delta": delta, "bootstrap": ci}), flush=True)
    report = {"baseline": bm["split"], "results": results,
              "per_class_support": support,
              "novel_classes_with_only_one_GT_video": int((gt_video_support[is_novel] == 1).sum()),
              "bootstrap": {"replicates": args.reps, "seed": args.seed,
                            "unit": "video, paired", "class_universe": "fixed 261 base / 35 novel",
                            "empty_resample_category": "absent classes contribute zero delta; unlike rerunning TAO class selection on resampled data",
                            "limitation": "26/35 novel classes occur in one GT video; about 10 novel classes are absent per ordinary video resample. Their deltas shrink toward zero. This interval cannot establish across-video class generalization or serve as a significance gate.",
                            "scope": "exploratory empirical video-composition perturbation interval; deterministic per-class deltas and support are primary evidence"}}
    report.update(run_id=run_id, script_sha256=sha256(__file__),
                  annotation_sha256=annotation_sha256)
    output.write_text(json.dumps(report, indent=2))
    print("SUMMARY", output, flush=True)


if __name__ == "__main__":
    main()
