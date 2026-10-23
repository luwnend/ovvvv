"""Evaluate label readouts with unchanged COVTrack detections and identities.

Run with the isolated ovtrack_ovmot Python. The cache is built from *all*
tracker observations, with no instance-GT matching or routing. The explicitly
named corrected-mask controls use public vocabulary frequency metadata.
TETA is called directly: no second majority vote is applied to the readout.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import teta


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr",
          "AssocRe", "AssocPr", "ClsRe", "ClsPr"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
    tmp.replace(path)


class RecordingTETA(teta.metrics.TETA):
    """Save unrounded official results without changing any metric operation."""
    @classmethod
    def get_name(cls):
        return "TETA"

    def __init__(self):
        super().__init__()
        self.aggregate = []
        self.sequence_counts = []
        self.sequence_names = None

    def combine_sequences(self, all_res):
        result = super().combine_sequences(all_res)
        self.aggregate.append(result)
        names = list(all_res)
        if self.sequence_names is None:
            self.sequence_names = names
        assert names == self.sequence_names
        self.sequence_counts.append(np.stack([
            np.stack([all_res[s][50][k] for k in ("Cls_TP", "Cls_FN", "Cls_FP")])
            for s in names
        ]).astype(np.int32))
        return result


def readout(cache, rule):
    if rule == "tcc":
        return cache["tcc_category_id"]
    field = "p_corrected_mask_argmax_counts" if rule == "corrected_mask_tcc" else "p_" + rule
    if field not in cache:
        raise KeyError(field)
    ids = cache["category_ids"]
    # Background is retained in cache for schema checks, never emitted as a class.
    return ids[np.argmax(cache[field][:, :len(ids)], axis=1)]


def build(args):
    out = args.out / args.rule
    out.mkdir(parents=True, exist_ok=True)
    result = out / "tao_track.json"
    if result.exists() or result.is_symlink():
        raise FileExistsError("Use --phase evaluate for an existing result; never overwrite: " + str(result))
    with np.load(args.cache, allow_pickle=False) as c:
        pred = readout(c, args.rule)
        gids = c["global_track_id"]
        keys = c["key"]
        labels = {int(g): int(p) for g, p in zip(gids, pred)}
        videos = {int(g): int(k[0]) for g, k in zip(gids, keys)}
        expected_obs = int(c["n_obs"].sum())
        changed_tracks = int(np.count_nonzero(pred != c["tcc_category_id"]))
    assert len(labels) == len(gids), "Global track ids must be unique"
    data = json.load(args.baseline.open())
    assert len(data) == expected_obs
    observed = set()
    changed_obs = 0
    for r in data:
        tid = int(r["track_id"])
        assert r["video_id"] == videos[tid]
        observed.add(tid)
        new = labels[tid]
        changed_obs += new != r["category_id"]
        r["category_id"] = new
    assert observed == set(labels)
    if args.rule == "tcc":
        assert changed_obs == 0
    # This preserves source order and every field except category_id.
    with result.open("w") as f:
        json.dump(data, f)
    receipt = {
        "rule": args.rule, "status": "PREDICTIONS_BUILT",
        "baseline": str(args.baseline), "baseline_sha256": sha256(args.baseline),
        "cache": str(args.cache), "cache_sha256": sha256(args.cache),
        "result": str(result), "result_sha256": sha256(result),
        "observations": len(data), "tracks": len(labels),
        "changed_tracks": changed_tracks, "changed_observations": changed_obs,
        "modified_fields": ["category_id"], "second_tcc": False,
        "aggregation_uses_gt": False,
    }
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def evaluate(args):
    out = args.out / args.rule
    result = out / "tao_track.json"
    receipt_path = out / "receipt.json"
    receipt = json.load(receipt_path.open())
    assert receipt["rule"] == args.rule
    if receipt["status"] not in ("PREDICTIONS_BUILT", "ORIGINAL_BASELINE"):
        raise FileExistsError("Use a new output directory to preserve an existing evaluation: " + str(out))
    assert sha256(result) == receipt["result_sha256"]
    ann = json.load(args.ann.open())
    category = {c["name"]: c for c in ann["categories"]}
    cfg = teta.config.get_default_eval_config()
    cfg.update(PRINT_ONLY_COMBINED=True, DISPLAY_LESS_PROGRESS=True,
               NUM_PARALLEL_CORES=args.cores, OUTPUT_TEM_RAW_DATA=True,
               OUTPUT_PER_SEQ_RES=True, PRINT_CONFIG=False)
    dcfg = teta.config.get_default_dataset_config()
    dcfg.update(TRACKERS_TO_EVAL=["OVTrack"], GT_FOLDER=str(args.ann),
                OUTPUT_FOLDER=str(out), TRACKER_SUB_FOLDER=str(result), PRINT_CONFIG=False)
    receipt.update(status="EVALUATING", ann=str(args.ann), ann_sha256=sha256(args.ann),
                   evaluator_files={str(Path(p)): sha256(p) for p in (
                       Path(teta.__file__).parent / "eval.py",
                       Path(teta.__file__).parent / "metrics/teta.py",
                       Path(teta.__file__).parent / "datasets/tao.py")})
    write_json(receipt_path, receipt)
    ds = teta.datasets.TAO(dcfg)
    class_list = ds.get_eval_info()[2]
    metric = RecordingTETA()
    _, messages = teta.Evaluator(cfg).evaluate([ds], [metric])
    assert messages["TAO"]["OVTrack"] == "Success", messages
    assert len(metric.aggregate) == len(class_list)
    vals = np.array([[100 * np.mean(r[50][f]) for f in FIELDS]
                     for r in metric.aggregate])
    is_novel = np.array([category[n]["frequency"] == "r" for n in class_list])
    split = {}
    for name, mask in [("combined", np.ones(len(class_list), bool)),
                       ("base", ~is_novel), ("novel", is_novel)]:
        split[name] = {"n_classes": int(mask.sum()),
                       **dict(zip(FIELDS, vals[mask].mean(0).tolist()))}
    metrics = {"rule": args.rule, "split": split,
               "category_averaging": "ann frequency split; equal class weights",
               "class_names": class_list, "per_class": vals.tolist(), "fields": FIELDS}
    write_json(out / "metrics.json", metrics)
    np.savez_compressed(out / "classification_counts.npz",
                        counts=np.stack(metric.sequence_counts),
                        class_names=np.array(class_list),
                        sequence_names=np.array(metric.sequence_names), is_novel=is_novel,
                        count_fields=np.array(["Cls_TP", "Cls_FN", "Cls_FP"]))
    # Preserve official unrounded alpha results before display formatting.
    with (out / "aggregate_raw.pkl").open("wb") as f:
        pickle.dump(dict(zip(class_list, metric.aggregate)), f)
    receipt.update(status="COMPLETE", metrics_sha256=sha256(out / "metrics.json"),
                   counts_sha256=sha256(out / "classification_counts.npz"),
                   aggregate_raw_sha256=sha256(out / "aggregate_raw.pkl"),
                   readout_script_sha256=sha256(__file__),
                   summary_sha256=sha256(out / "OVTrack/teta_summary_results.pth"))
    write_json(receipt_path, receipt)
    print(json.dumps({"rule": args.rule, "split": split}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rule", required=True, choices=["tcc", "fused_sumprob", "fused_sumlog",
                                                    "text_sumprob", "image_sumprob",
                                                    "corrected_mask_sumprob", "corrected_mask_sumlog",
                                                    "corrected_mask_tcc"])
    ap.add_argument("--phase", choices=["build", "evaluate", "both"], default="both")
    ap.add_argument("--cache", type=Path, default=ROOT / "results/tces_audit_20261023/track_cache.npz")
    ap.add_argument("--baseline", type=Path, default=ROOT / "results/cov_val_full_official_a100/tao_track.json")
    ap.add_argument("--ann", type=Path, default=ROOT / "data_tao/annotations/validation_ours_v1.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tces_audit_20261023/readouts")
    ap.add_argument("--cores", type=int, default=4)
    args = ap.parse_args()
    if args.phase in ("build", "both"):
        build(args)
    if args.phase in ("evaluate", "both"):
        evaluate(args)


if __name__ == "__main__":
    main()
