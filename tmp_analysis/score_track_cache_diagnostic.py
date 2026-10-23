#!/usr/bin/env python3
"""GT-scored readout diagnostics on immutable, fully aggregated tracker output.

These are classification/purity diagnostics, NOT official TETA or ClsA. GT is
used only in this scoring script, never to select posterior observations or a
fusion mask. The geometry cache can be built before the posterior cache exists.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


GEOMETRY_VERSION = "classagnostic_max_total_iou_hungarian_then_threshold_v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".incomplete")
    with open(temp, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def iou_xywh(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    a_end = a[:, :2] + a[:, 2:]
    b_end = b[:, :2] + b[:, 2:]
    wh = np.maximum(0, np.minimum(a_end[:, None], b_end[None]) -
                    np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh.prod(2)
    area_a = np.maximum(a[:, 2:], 0).prod(1)
    area_b = np.maximum(b[:, 2:], 0).prod(1)
    union = area_a[:, None] + area_b[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def prepare_scoreinput(args):
    output = args.out.parent
    output.mkdir(parents=True, exist_ok=True)
    scoreinput = output / "scoreinput.npz"
    manifest_path = output / "scoreinput_manifest.json"
    source = {
        "prediction": {"path": str(args.pred), "sha256": sha256(args.pred)},
        "annotation": {"path": str(args.ann), "sha256": sha256(args.ann)},
        "matching": GEOMETRY_VERSION,
        "iou_threshold": args.iou,
    }
    if scoreinput.exists() and manifest_path.exists():
        manifest = json.load(open(manifest_path))
        if manifest["source"] != source:
            raise RuntimeError("Existing scoreinput has different sources or matching settings")
        if manifest["scoreinput_sha256"] != sha256(scoreinput):
            raise RuntimeError("scoreinput hash does not match its manifest")
        print("Reusing verified geometry cache", scoreinput, flush=True)
        return scoreinput, manifest_path

    started = time.monotonic()
    ann = json.load(open(args.ann))
    preds = json.load(open(args.pred))
    categories = {c["id"]: c for c in ann["categories"]}
    image_video = {x["id"]: x["video_id"] for x in ann["images"]}
    image_order = [x["id"] for x in ann["images"]]
    gt_by_image = defaultdict(list)
    gt_support = Counter()
    for gt in ann["annotations"]:
        gt_by_image[gt["image_id"]].append(gt)
        gt_support[int(gt["category_id"])] += 1

    tracks = {}
    track_keys, n_obs, tcc_categories = [], [], []
    pred_by_image = defaultdict(list)
    pred_track_rows = np.empty(len(preds), dtype=np.int32)
    for row, pred in enumerate(preds):
        iid = int(pred["image_id"])
        vid = int(pred["video_id"])
        if image_video.get(iid) != vid:
            raise RuntimeError("Prediction image/video identity is outside the annotation")
        key = (vid, int(pred["track_id"]))
        ti = tracks.get(key)
        if ti is None:
            ti = len(track_keys)
            tracks[key] = ti
            track_keys.append(key)
            n_obs.append(0)
            tcc_categories.append(int(pred["category_id"]))
        if tcc_categories[ti] != int(pred["category_id"]):
            raise RuntimeError("Formal baseline is not TCC-consistent within a track")
        n_obs[ti] += 1
        pred_track_rows[row] = ti
        pred_by_image[iid].append(row)

    fields = defaultdict(list)
    for fi, iid in enumerate(image_order):
        gs = gt_by_image.get(iid, [])
        ps = pred_by_image.get(iid, [])
        if gs and ps:
            sim = iou_xywh([g["bbox"] for g in gs], [preds[j]["bbox"] for j in ps])
            gr, pc = linear_sum_assignment(-sim)
            for gi, pi in zip(gr, pc):
                if sim[gi, pi] < args.iou:
                    continue
                g = gs[int(gi)]
                pred_row = ps[int(pi)]
                fields["pred_row_index"].append(pred_row)
                fields["track_row"].append(pred_track_rows[pred_row])
                fields["image_id"].append(iid)
                fields["gt_track_id"].append(int(g["track_id"]))
                fields["gt_category_id"].append(int(g["category_id"]))
                fields["iou"].append(sim[gi, pi])
        if (fi + 1) % 5000 == 0:
            print("Geometry frames %d/%d; matches %d" %
                  (fi + 1, len(image_order), len(fields["track_row"])), flush=True)

    support_ids = np.asarray(sorted(gt_support), dtype=np.int64)
    arrays = {key: np.asarray(value, dtype=np.float32 if key == "iou" else np.int64)
              for key, value in fields.items()}
    arrays.update(
        track_key=np.asarray(track_keys, dtype=np.int64),
        track_n_obs=np.asarray(n_obs, dtype=np.int64),
        track_tcc_category_id=np.asarray(tcc_categories, dtype=np.int64),
        gt_support_category_ids=support_ids,
        gt_support_n_annotations=np.asarray([gt_support[int(k)] for k in support_ids], dtype=np.int64),
        gt_support_frequency_rare=np.asarray([categories[int(k)]["frequency"] == "r" for k in support_ids]),
        gt_support_category_name=np.asarray([categories[int(k)]["name"] for k in support_ids]),
    )
    temp = scoreinput.with_name("scoreinput.incomplete.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, scoreinput)
    manifest = {
        "source": source,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "runtime": {"python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__},
        "scoreinput_sha256": sha256(scoreinput),
        "n_prediction_observations": len(preds),
        "n_prediction_tracks": len(track_keys),
        "n_annotation_images": len(image_order),
        "n_annotation_gt_observations": len(ann["annotations"]),
        "n_matched_observations": len(arrays["track_row"]),
        "n_matched_tracks": int(len(np.unique(arrays["track_row"]))),
        "n_gt_support_classes": len(support_ids),
        "n_gt_support_rare_classes": int(arrays["gt_support_frequency_rare"].sum()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "matching_protocol": (
            "For each image, maximize the sum of raw bbox IoUs with a class-agnostic "
            "one-to-one Hungarian assignment, then retain assigned pairs with IoU >= threshold. "
            "No class, posterior, score or GT identity enters the assignment. This is independent "
            "diagnostic matching, not TETA's temporal alignment assignment."
        ),
        "track_key_columns": ["video_id", "official_global_track_id"],
        "aggregation_scope": "Every original emitted tracker observation; matching only chooses scoring rows",
    }
    write_json(manifest_path, manifest)
    print("Prepared", scoreinput, "matches", manifest["n_matched_observations"], flush=True)
    return scoreinput, manifest_path


def group_metrics(hits, obs_class, select, category_ids, rare, gt_support_n):
    result = {}
    for group, class_select in (("all", np.ones(len(category_ids), dtype=bool)),
                                ("base", ~rare), ("novel_frequency_rare", rare)):
        mask = select & class_select[obs_class]
        support = np.bincount(obs_class[mask], minlength=len(category_ids))
        correct = np.stack([np.bincount(obs_class[mask], weights=h[mask],
                                       minlength=len(category_ids)) for h in hits]).astype(np.float64)
        accuracy = np.divide(correct, support[None], out=np.zeros_like(correct),
                             where=support[None] > 0)
        active = class_select & (support > 0)
        k_metrics = {}
        for ki, k in enumerate((1, 10, 50)):
            k_metrics["top%d" % k] = {
                "matched_observation_micro_percent": float(100 * correct[ki].sum() / int(mask.sum())) if mask.any() else None,
                "macro_over_matched_gt_classes_percent": float(100 * accuracy[ki, active].mean()) if active.any() else None,
                "macro_over_all_gt_support_classes_unmatched_as_zero_percent": float(100 * accuracy[ki, class_select].mean()) if class_select.any() else None,
                "correct_matched_observations": int(correct[ki].sum()),
            }
        result[group] = {
            "n_matched_observations": int(mask.sum()),
            "n_gt_support_classes": int(class_select.sum()),
            "n_matched_gt_classes": int(active.sum()),
            "n_unmatched_gt_classes": int((class_select & (support == 0)).sum()),
            "n_gt_observations_all_support_classes": int(gt_support_n[class_select].sum()),
            "metrics": k_metrics,
        }
    return result


def score(args, scoreinput, manifest_path):
    if not args.cache.exists():
        raise FileNotFoundError("Posterior cache is not ready; geometry cache is complete: " + str(args.cache))
    started = time.monotonic()
    sm = json.load(open(manifest_path))
    cache_manifest = args.cache.parent / "manifest.json"
    if not cache_manifest.exists():
        raise FileNotFoundError("Posterior cache audit manifest is not ready: " + str(cache_manifest))
    cm = json.load(open(cache_manifest))
    cache_hash = sha256(args.cache)
    if cm["status"] != "PASS_FULL_TRACKER_POSTERIOR_CACHE_AUDIT" or cm["output_sha256"] != cache_hash:
        raise RuntimeError("Posterior cache has not passed its manifest/hash audit")
    if (cm["input_sha256"]["annotation"] != sm["source"]["annotation"]["sha256"] or
            cm["input_sha256"]["official"] != sm["source"]["prediction"]["sha256"]):
        raise RuntimeError("Posterior cache and geometry cache have different source hashes")
    geometry = dict(np.load(scoreinput, allow_pickle=False))
    cache = np.load(args.cache, allow_pickle=False)
    cache_keys = np.column_stack([cache["key"][:, 0], cache["global_track_id"]])
    key_index = {tuple(k): i for i, k in enumerate(cache_keys.tolist())}
    if len(key_index) != len(cache_keys):
        raise RuntimeError("Posterior cache has duplicate video/global-track keys")
    track_key = geometry["track_key"]
    if len(track_key) != len(cache_keys):
        raise RuntimeError("Posterior cache and formal baseline do not have identical track support")
    try:
        cache_row_for_track = np.asarray([key_index[tuple(k)] for k in track_key.tolist()], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("Official baseline track is absent from posterior cache") from error
    if not np.array_equal(geometry["track_n_obs"], cache["n_obs"][cache_row_for_track]):
        raise RuntimeError("Per-track observation counts differ from fully aggregated cache")
    if not np.array_equal(geometry["track_tcc_category_id"], cache["tcc_category_id"][cache_row_for_track]):
        raise RuntimeError("Cached TCC does not reproduce the formal baseline")

    category_ids = cache["category_ids"]
    gt_category_ids = geometry["gt_support_category_ids"]
    if not np.array_equal(category_ids, gt_category_ids):
        raise RuntimeError("Posterior category columns and GT-support category set differ")
    rare = geometry["gt_support_frequency_rare"]
    if not np.array_equal(rare, cache["frequency_rare_mask"]):
        raise RuntimeError("Cache metric rare mask differs from annotation frequency")
    cat_index = {int(c): i for i, c in enumerate(category_ids)}
    obs_class = np.asarray([cat_index[int(c)] for c in geometry["gt_category_id"]], dtype=np.int64)
    obs_track = geometry["track_row"]
    scored_tracks, compact_obs_track = np.unique(obs_track, return_inverse=True)
    scored_cache_rows = cache_row_for_track[scored_tracks]
    matched_per_track = np.bincount(compact_obs_track, minlength=len(scored_tracks))
    identity_counts = [Counter() for _ in scored_tracks]
    for ti, gid in zip(compact_obs_track, geometry["gt_track_id"]):
        identity_counts[int(ti)][int(gid)] += 1
    n_gt_id = np.asarray([len(c) for c in identity_counts], dtype=np.int64)
    dominant_n = np.asarray([max(c.values()) for c in identity_counts], dtype=np.int64)
    purity = dominant_n / matched_per_track
    mixed_obs = n_gt_id[compact_obs_track] > 1
    strata = {
        "all_matched_tracks": np.ones(len(obs_class), dtype=bool),
        "single_gt_identity": ~mixed_obs,
        "mixed_gt_identity": mixed_obs,
    }

    readout_specs = [
        ("tcc", "majority_counts"),
        ("fused_sumprob", "p_fused_sumprob"),
        ("fused_sumlog", "p_fused_sumlog"),
        ("text_sumprob", "p_text_sumprob"),
        ("image_sumprob", "p_image_sumprob"),
        ("correctedmask_sumprob", "p_corrected_mask_sumprob"),
    ]
    readouts = {}
    hit_bank = []
    all_class_per_readout = {}
    for name, field in readout_specs:
        print("Scoring readout", name, flush=True)
        all_scores = cache[field]
        # Counts are uint32; cast before negation to avoid unsigned wraparound.
        scores = all_scores[scored_cache_rows, :len(category_ids)].astype(np.float64)
        del all_scores
        # category_ids are sorted; stable ties resolve to the lowest category ID.
        order = np.argsort(-scores, axis=1, kind="stable")[:, :50]
        del scores
        obs_order = order[compact_obs_track]
        hit = np.stack([(obs_order[:, :k] == obs_class[:, None]).any(1) for k in (1, 10, 50)])
        if name == "tcc":
            tcc_expected = geometry["track_tcc_category_id"][scored_tracks]
            if not np.array_equal(category_ids[order[:, 0]], tcc_expected):
                raise RuntimeError("TCC ranked counts do not reproduce formal baseline top1")
        readouts[name] = {
            "cache_field": field,
            "classification_scope": "same per-track readout for all its matched observations; all tracker observations were aggregated before GT scoring",
            "strata": {sn: group_metrics(hit, obs_class, sel, category_ids, rare,
                                        geometry["gt_support_n_annotations"])
                       for sn, sel in strata.items()},
        }
        per_class = {}
        for ci, cid in enumerate(category_ids):
            select = obs_class == ci
            n = int(select.sum())
            per_class[int(cid)] = {
                "n_matched_observations": n,
                "n_matched_tracks": int(len(np.unique(obs_track[select]))),
                "top1_percent": float(100 * hit[0, select].mean()) if n else None,
                "top10_percent": float(100 * hit[1, select].mean()) if n else None,
                "top50_percent": float(100 * hit[2, select].mean()) if n else None,
                "correct_counts_top1_top10_top50": hit[:, select].sum(1).astype(int).tolist(),
            }
        all_class_per_readout[name] = per_class
        hit_bank.append(hit)
        del order, obs_order

    hit_bank = np.stack(hit_bank)
    union_hits = hit_bank.any(0)
    candidate_hits_per_track = np.stack([
        np.bincount(compact_obs_track, weights=hit_bank[i, 0], minlength=len(scored_tracks))
        for i in range(len(readout_specs))
    ])
    chosen = candidate_hits_per_track.argmax(0)
    chosen_per_obs = chosen[compact_obs_track]
    track_oracle_hits = np.stack([
        hit_bank[chosen_per_obs, ki, np.arange(len(obs_class))]
        for ki in range(3)
    ])
    oracle = {
        "status": "GT-selected upper bound only; not a deployable method or official ClsA",
        "candidates": [name for name, _ in readout_specs],
        "per_track_gt_selected_candidate": {
            "selection": "For each track, GT chooses the candidate with most correct top1 matched observations; ties use listed order. Its same selected ranking is used for top10/top50.",
            "upper_bound_scope": "Optimizes global matched-observation top1 micro count only; a group or class macro need not dominate every individual candidate.",
            "n_tracks_selected_by_candidate": {name: int((chosen == i).sum())
                                               for i, (name, _) in enumerate(readout_specs)},
            "strata": {sn: group_metrics(track_oracle_hits, obs_class, sel, category_ids, rare,
                                        geometry["gt_support_n_annotations"])
                       for sn, sel in strata.items()},
        },
        "per_observation_candidate_union": {
            "selection": "GT may choose a different candidate separately for every observation and every k; deliberately loose upper bound.",
            "strata": {sn: group_metrics(union_hits, obs_class, sel, category_ids, rare,
                                        geometry["gt_support_n_annotations"])
                       for sn, sel in strata.items()},
        },
    }
    per_class = []
    for ci, cid in enumerate(category_ids):
        select = obs_class == ci
        per_class.append({
            "category_id": int(cid),
            "name": str(geometry["gt_support_category_name"][ci]),
            "frequency_group": "novel" if rare[ci] else "base",
            "n_gt_observations": int(geometry["gt_support_n_annotations"][ci]),
            "n_matched_observations": int(select.sum()),
            "n_matched_tracks": int(len(np.unique(obs_track[select]))),
            "n_single_identity_matched_observations": int((select & ~mixed_obs).sum()),
            "n_mixed_identity_matched_observations": int((select & mixed_obs).sum()),
            "readouts": {name: details[int(cid)] for name, details in all_class_per_readout.items()},
        })

    def purity_summary(which):
        val = purity[which]
        return {
            "n_tracks": int(which.sum()),
            "n_matched_observations": int(matched_per_track[which].sum()),
            "mean_track_purity": float(val.mean()) if len(val) else None,
            "median_track_purity": float(np.median(val)) if len(val) else None,
            "observation_weighted_purity": float(dominant_n[which].sum() / matched_per_track[which].sum()) if which.any() else None,
        }

    provenance = {
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "runtime": {"python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__},
        "geometry_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "geometry_cache": {"path": str(scoreinput), "sha256": sha256(scoreinput)},
        "posterior_cache": {"path": str(args.cache), "sha256": cache_hash},
        "source": sm["source"],
    }
    provenance["posterior_cache_manifest"] = {"path": str(cache_manifest), "sha256": sha256(cache_manifest)}
    output = {
        "status": "COMPLETE_NONOFFICIAL_CLASSIFICATION_DIAGNOSTIC",
        "metric_boundary": "Matched-observation top-k classification accuracy and GT-class macro accuracy; these are NOT official ClsA, TETA, or a replacement for a full evaluator run.",
        "provenance": provenance,
        "protocol": {
            "matching": sm["matching_protocol"],
            "iou_threshold": args.iou,
            "track_identity_key": ["video_id", "official_global_track_id"],
            "gt_identity_for_purity": ["video_id", "GT_track_id"],
            "posterior_aggregation": "All emitted tracker observations, including those with no diagnostic GT match. No GT used to select observations, choose a fusion mask or define an actual readout.",
            "macro_all_gt_support_zero_convention": "All GT-present classes retained (including all 35 rare classes); an unmatched class contributes zero only to the explicitly named zero-for-unmatched macro. Matched-class macro and unmatched class counts are separately reported.",
            "tcc_topk": "Top1 is exact formal-baseline TCC. Top10/50 rank per-track argmax-vote counts with ties resolved by lowest category ID; zero-count ties can fill remaining ranks.",
            "posterior_topk": "Descending accumulated score over 296 foreground columns; ties use lowest category ID. Background excluded only at readout.",
            "correctedmask": "Precomputed cache intervention using annotation-frequency mask; reported as a distinct model intervention, never silently substituted for the official fusion mask.",
            "purity_scope": "Identity purity uses only one-to-one GT-matched observations; unmatched observations were retained in posterior aggregation but cannot be assigned a GT identity.",
            "strata_macro_scope": "The all-GT-support-zero macro in a stratum includes the full GT class set even when a class has no matched observations in that stratum; prefer matched-class macro when describing within-stratum classification.",
        },
        "support": {
            "n_emitted_observations": int(geometry["track_n_obs"].sum()),
            "n_emitted_tracks": len(track_key),
            "n_matched_observations": len(obs_class),
            "n_matched_tracks": len(scored_tracks),
            "n_unmatched_tracks": int(len(track_key) - len(scored_tracks)),
            "n_gt_support_classes": len(category_ids),
            "n_gt_support_rare_classes": int(rare.sum()),
            "cache_full_observation_counts_equal_baseline": True,
            "cache_tcc_equals_formal_baseline": True,
        },
        "track_identity_purity": {
            "all_matched_tracks": purity_summary(np.ones(len(purity), dtype=bool)),
            "single_gt_identity": purity_summary(n_gt_id == 1),
            "mixed_gt_identity": purity_summary(n_gt_id > 1),
            "single_identity_fraction_of_matched_tracks": float((n_gt_id == 1).mean()),
        },
        "readouts": readouts,
        "oracle_upper_bounds": oracle,
        "per_gt_class": per_class,
        "elapsed_scoring_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.out, output)
    print("Diagnostic complete", args.out, "sha256", sha256(args.out), flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, default=root / "results/cov_val_full_official_a100/tao_track.json")
    ap.add_argument("--ann", type=Path, default=Path("/raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json"))
    ap.add_argument("--cache", type=Path, default=root / "results/tces_audit_20261023/track_cache.npz")
    ap.add_argument("--out", type=Path, default=root / "results/tces_audit_20261023/diagnostic.json")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    args.pred = args.pred.resolve()
    args.ann = args.ann.resolve()
    args.cache = args.cache.resolve()
    args.out = args.out.resolve()
    scoreinput, manifest = prepare_scoreinput(args)
    if not args.prepare_only:
        score(args, scoreinput, manifest)


if __name__ == "__main__":
    main()
