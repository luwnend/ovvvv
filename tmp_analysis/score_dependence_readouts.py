"""Score all frozen-geometry readouts with an independently verified TETA cache.

GT events and annotations are scoring-only inputs. They cannot affect the
already sealed GT-free labels. Main dependence outputs receive separate full
official evaluation; the remaining rows are exact classification-event replay.
"""
import argparse
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np

from build_teta_classification_events import replay_counts, sha256
from evaluate_track_readouts import FIELDS, write_json

ROOT = Path(__file__).resolve().parents[1]


def metrics_from_counts(counts, baseline):
    """Sum video counts, then use installed TETA's ten-alpha mean formula."""
    tp, fn, fp = counts.sum(axis=1, dtype=np.int64).transpose(1, 0, 2)
    values = np.asarray(baseline["per_class"], np.float64).copy()
    for name, den in (("ClsA", tp + fn + fp), ("ClsRe", tp + fn), ("ClsPr", tp + fp)):
        values[:, FIELDS.index(name)] = 100 * (tp / np.maximum(1, den)).mean(axis=1)
    values[:, FIELDS.index("TETA")] = values[:, [FIELDS.index(n) for n in ("LocA", "AssocA", "ClsA")]].mean(axis=1)
    return values


def split_metrics(values, novel):
    return {name: {"n_classes": int(mask.sum()), **dict(zip(FIELDS, values[mask].mean(axis=0).tolist()))}
            for name, mask in (("combined", np.ones(len(novel), bool)), ("base", ~novel), ("novel", novel))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "results/dependence_audit_20261023")
    args = ap.parse_args()
    stage = args.root
    out = stage / "scores"
    assert not out.exists(), "Never overwrite a completed or partial scoring run"
    out.mkdir()
    started = time.monotonic()
    old = ROOT / "results/tces_audit_20261023"
    manifest = json.load((stage / "manifest.json").open())
    event_receipt = json.load((stage / "teta_events/receipt.json").open())
    reference_manifest = json.load((old / "manifest.json").open())
    assert manifest["status"] == "PASS_DEPENDENCE_READOUTS_FULL"
    assert event_receipt["status"] == "PASS_EXACT_FIXED_GEOMETRY_CLASSIFICATION_EVENT_REPLAY"
    inputs = {}

    def bind(path, expected=None):
        digest = sha256(path)
        assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest
        return digest

    bind(stage / "manifest.json")
    bind(stage / "teta_events/receipt.json")
    bind(stage / "readout_cache.npz", manifest["output_sha256"])
    bind(stage / "teta_events/classification_events.npz", event_receipt["output_sha256"])
    bind(old / "track_cache.npz", reference_manifest["output_sha256"])
    for path, digest in manifest["inputs_sha256"].items():
        bind(Path(path), digest)
    bind(Path(event_receipt["baseline_receipt"]["ann"]), event_receipt["baseline_receipt"]["ann_sha256"])
    for path, digest in event_receipt["baseline_receipt"]["evaluator_files"].items():
        bind(Path(path), digest)
    baseline = json.load((old / "readouts/tcc/metrics.json").open())
    assert FIELDS == baseline["fields"]
    with np.load(stage / "teta_events/classification_events.npz", allow_pickle=False) as z:
        event = {k: z[k] for k in z.files}
    with np.load(stage / "readout_cache.npz", allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    with np.load(old / "track_cache.npz", allow_pickle=False) as z:
        assert np.array_equal(cache["key"], z["key"])
        assert np.array_equal(cache["global_track_id"], z["global_track_id"])
        assert np.array_equal(cache["n_obs"], z["n_obs"])
        assert np.array_equal(cache["category_ids"], z["category_ids"])
        reference_labels = {
            "tcc": z["tcc_category_id"],
            **{rule: z["category_ids"][z[field][:, :len(z["category_ids"])].argmax(axis=1)]
               for rule, field in (("fused_sumprob", "p_fused_sumprob"), ("fused_sumlog", "p_fused_sumlog"),
                                   ("corrected_mask_tcc", "p_corrected_mask_argmax_counts"),
                                   ("corrected_mask_sumprob", "p_corrected_mask_sumprob"),
                                   ("corrected_mask_sumlog", "p_corrected_mask_sumlog"))}}
    names = cache["method_names"].tolist()
    assert len(names) == len(set(names)) == 20
    assert np.array_equal(event["class_names"], baseline["class_names"])
    ann = json.load(open(event_receipt["baseline_receipt"]["ann"]))
    categories = {c["id"]: c for c in ann["categories"]}
    novel = np.asarray([categories[int(cid)]["frequency"] == "r" for cid in event["category_ids"]])
    assert (int(novel.sum()), int((~novel).sum())) == (35, 261)
    class_support = []
    for cid, name, is_novel in zip(event["category_ids"], event["class_names"], novel):
        gt = [a for a in ann["annotations"] if a["category_id"] == cid]
        class_support.append(dict(category_id=int(cid), name=str(name), novel=bool(is_novel),
                                  gt_videos=len({a["video_id"] for a in gt}),
                                  gt_tracks=len({a["track_id"] for a in gt}), gt_boxes=len(gt)))
    # Bind and verify each prior full run, including the sixth corrected-log
    # control. No assumption that event/cache/category orders are numerical.
    parity = {}
    for rule, labels in reference_labels.items():
        run = (stage if rule == "corrected_mask_sumlog" else old) / "readouts" / rule
        receipt = json.load((run / "receipt.json").open())
        assert receipt["status"] == "COMPLETE"
        bind(run / "receipt.json")
        bind(run / "classification_counts.npz", receipt["counts_sha256"])
        bind(run / "metrics.json", receipt["metrics_sha256"])
        with np.load(run / "classification_counts.npz", allow_pickle=False) as z:
            assert np.array_equal(z["class_names"], event["class_names"])
            assert np.array_equal(z["sequence_names"], event["sequence_names"])
            assert np.array_equal(z["is_novel"], novel)
            expected = z["counts"]
        actual = replay_counts(event, cache["global_track_id"], labels, event["category_ids"], len(event["sequence_names"]))
        assert np.array_equal(actual, expected), rule
        official = json.load((run / "metrics.json").open())
        error = float(np.abs(metrics_from_counts(actual, baseline) - official["per_class"]).max())
        assert error < 1e-10, (rule, error)
        parity[rule] = dict(all_class_video_alpha_counts_exact=True, per_class_metric_max_abs_error=error)
    methods = {}
    all_values = []
    for i, name in enumerate(names):
        labels = cache["pred_category_id"][i]
        counts = replay_counts(event, cache["global_track_id"], labels, event["category_ids"], len(event["sequence_names"]))
        values = metrics_from_counts(counts, baseline)
        all_values.append(values)
        path = out / (name + ".counts.npz")
        np.savez_compressed(path, counts=counts, class_names=event["class_names"],
                            sequence_names=event["sequence_names"], is_novel=novel)
        mask = name.split(".")[0]
        ref = cache["pred_category_id"][names.index(mask + ".full_sumlog")]
        methods[name] = dict(split=split_metrics(values, novel), counts_sha256=sha256(path),
                             changed_tracks_vs_released_tcc=int(np.count_nonzero(labels != reference_labels["tcc"])),
                             changed_tracks_vs_own_full_sumlog=int(np.count_nonzero(labels != ref)),
                             changed_observations_vs_own_full_sumlog=int(cache["n_obs"][labels != ref].sum()))
        print(name, json.dumps(methods[name]["split"]), flush=True)
    all_values = np.asarray(all_values)
    comparisons = {}
    for mask in ("released", "corrected"):
        ref = all_values[names.index(mask + ".full_sumlog")]
        for family in cache["method_families"]:
            name = mask + "." + family
            delta = all_values[names.index(name)] - ref
            comparisons[name + " - " + mask + ".full_sumlog"] = split_metrics(delta, novel)
    influence = {}
    for mask in ("released", "corrected"):
        delta = (all_values[names.index(mask + ".dep_inverse_density")] -
                 all_values[names.index(mask + ".full_sumlog")])[:, FIELDS.index("ClsA")]
        influence[mask] = {"improved_base": int((delta[~novel] > 1e-10).sum()),
                           "worsened_base": int((delta[~novel] < -1e-10).sum()),
                           "improved_novel": int((delta[novel] > 1e-10).sum()),
                           "worsened_novel": int((delta[novel] < -1e-10).sum()),
                           "per_class_delta_ClsA": delta.tolist()}
    output = out / "per_class_metrics.npz"
    np.savez_compressed(output, method_names=cache["method_names"], fields=np.asarray(FIELDS),
                        per_class=all_values, class_names=event["class_names"],
                        category_ids=event["category_ids"], is_novel=novel)
    summary = dict(status="PASS_20_FIXED_GEOMETRY_CLASSIFICATION_REPLAYS", script_sha256=sha256(__file__),
                   inputs_sha256=inputs, elapsed_seconds=time.monotonic()-started,
                   source="Exact frozen-geometry non-exhaustive TETA50 classification event replay, six independent official controls with bitwise count parity",
                   main_candidate_full_official_validation="Separate receipts required; this replay is not twenty independent full evaluator runs",
                   methods=methods, comparisons_vs_own_sumlog=comparisons, independent_control_parity=parity,
                   class_support=class_support, primary_influence=influence,
                   per_class_metrics_sha256=sha256(output), fields=FIELDS,
                   gt_in_method=False, unchanged_geometry_fields=["bbox", "track_id", "video_id", "image_id", "score", "order"],
                   limitations=["All observations retained in method aggregation; GT events used only after readout cache was sealed",
                                "Random seeds are sampling variation, not independent data replicates",
                                "Density, quality weighting and ordinary pooling are controls, not novelty",
                                "Sparse novel-class video support does not establish independent generalization"])
    write_json(out / "score_summary.json", summary)
    print("PASS", out / "score_summary.json", flush=True)


if __name__ == "__main__":
    main()
