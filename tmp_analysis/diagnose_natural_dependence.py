#!/usr/bin/env python3
"""Descriptive dependence of repeated natural classification errors.

No tracker output, candidate label, model, old cache, or report is modified.
GT enters only the existing diagnostic matching and post-hoc scoring/strata.
Run the independent numerical and mapping preflights before --stage full.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

from verify_readout_artifacts import iter_json_array


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["same_wrong_category", "both_wrong", "endpoint_error_prevalence",
           "prediction_agreement", "cosine_similarity", "sampled_frame_id_lag",
           "detector_score_mean", "detector_score_abs_difference",
           "detector_score_midrank_mean", "detector_score_midrank_abs_difference",
           "posterior_max_mean", "posterior_max_abs_difference"]
OUTCOMES = METRICS[:4] + ["same_wrong_given_both_wrong"]
SCOPES = ["single_gt_identity_track", "mixed_track_same_gt_identity",
          "mixed_track_different_gt_identity_same_gt_class",
          "mixed_track_different_gt_identity_different_gt_class"]
VARIANTS = ["exact_lag_score_rank", "binned_lag_score_rank",
            "binned_lag_score_rank_posterior_max"]
FEATURES = ["track_feats", "cls_feats"]
PARTITIONS = ["all", "base_only", "novel_only", "base_novel"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    require(not path.exists(), "Guarded output already exists: " + str(path))
    temp = path.with_name(path.name + ".incomplete")
    require(not temp.exists(), "Incomplete output already exists: " + str(temp))
    with open(temp, "x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def write_npz(path, **arrays):
    require(not path.exists(), "Guarded output already exists: " + str(path))
    temp = path.with_name(path.stem + ".incomplete.npz")
    require(not temp.exists(), "Incomplete output already exists: " + str(temp))
    with open(temp, "xb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def canonical_filename(value):
    value = str(value).replace("\\", "/")
    marker = "data/tao/frames/"
    return value.split(marker, 1)[-1] if marker in value else value


def dump_name(image):
    return ("data/tao/frames/" + canonical_filename(image["file_name"])).replace("/", "-").replace(".jpg", "") + ".npz"


def midranks(scores):
    """Within-track (average 1-based rank - .5)/n, with exact score ties."""
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(scores)]
    ranks = np.empty(len(scores), dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:stop]] = (start + stop) / (2.0 * len(scores))
    return ranks


def near_far(cosine, indices):
    """Deterministic disjoint lower/upper thirds; minimum six pairs."""
    if len(indices) < 6:
        return None
    n = len(indices) // 3
    ordered = indices[np.argsort(cosine[indices], kind="stable")]
    return ordered[:n], ordered[-n:]


def pair_matrix(features, prediction, truth, frame, score, rank, confidence):
    i, j = np.triu_indices(len(prediction), 1)
    errors = prediction != truth
    same_pred = prediction[i] == prediction[j]
    both_wrong = errors[i] & errors[j]
    cosine = np.clip((features[i] * features[j]).sum(1, dtype=np.float64), -1, 1)
    matrix = np.column_stack([
        both_wrong & same_pred, both_wrong, (errors[i].astype(float) + errors[j]) / 2,
        same_pred, cosine, np.abs(frame[i] - frame[j]), (score[i] + score[j]) / 2,
        np.abs(score[i] - score[j]), (rank[i] + rank[j]) / 2,
        np.abs(rank[i] - rank[j]), (confidence[i] + confidence[j]) / 2,
        np.abs(confidence[i] - confidence[j]),
    ])
    return i, j, matrix


def numerical_preflight(args):
    features = np.asarray([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=float)
    prediction = np.asarray([7, 7, 9, 3])
    truth = np.full(4, 3)
    i, j, m = pair_matrix(features, prediction, truth, np.arange(4),
                          np.ones(4), np.full(4, .5), np.full(4, .8))
    require(len(m) == 6, "Hand pair count differs")
    require(m[:, 0].sum() == 1 and m[:, 1].sum() == 3, "Hand wrong-class counts differ")
    require(m[:, 2].mean() == .75 and m[:, 3].sum() == 1, "Hand baseline prevalence differs")
    require(np.array_equal(m[:, 4], [1, 0, -1, 0, -1, 0]), "Hand cosine values differ")
    far, near = near_far(np.asarray([-.9, -.8, -.1, .1, .8, .9]), np.arange(6))
    require(np.array_equal(far, [0, 1]) and np.array_equal(near, [4, 5]), "Hand quantile selection differs")
    require(near_far(np.zeros(5), np.arange(5)) is None, "Minimum pair guard failed")
    tied_far, tied_near = near_far(np.zeros(6), np.arange(6))
    require(not np.intersect1d(tied_far, tied_near).size, "Tied quantile groups overlap")
    ranks = midranks(np.asarray([1, 1, 3]))
    require(np.allclose(ranks, [1 / 3, 1 / 3, 5 / 6], rtol=0, atol=1e-15), "Tie midrank differs")
    # A mixed-identity/different-class pair can share a wrong third class.
    _, _, mixed = pair_matrix(np.asarray([[1., 0], [1., 0]]), np.asarray([7, 7]),
                              np.asarray([3, 9]), np.asarray([1, 2]), np.ones(2),
                              np.full(2, .5), np.full(2, .8))
    require(mixed[0, 0] == 1, "Different-GT-class shared wrong third class definition failed")
    # Independent hand check for denominator and aggregation plumbing.
    a = Accumulator(1, 1)
    hand_far, hand_near = near_far(m[:, 4], np.arange(6))
    classes = np.zeros(6, dtype=np.int64)
    for phase, pair_indices in ((0, np.arange(6)), (1, np.arange(6)), (2, hand_far), (3, hand_near)):
        a.add(phase, m, pair_indices, classes, classes, np.asarray([False]), 0)
    scored = a.result({"category_ids": np.asarray([3]), "category_names": np.asarray(["hand_class"]),
                       "category_frequency": np.asarray(["c"])}, np.asarray([False]))
    hand_group = scored["groups"]["all"]
    require(hand_group["all_available_pairs_baseline"]["n_pairs"] == 6, "All-pair baseline denominator changed")
    require(hand_group["appearance_far"]["n_pairs"] == hand_group["appearance_near"]["n_pairs"] == 2, "Selected pair denominators changed")
    require(hand_group["pooled_near_minus_far"]["same_wrong_category"] == .5, "Hand pooled difference changed")
    require(hand_group["equal_track_mean_differences"]["same_wrong_category"]["near_minus_far"] == .5, "Hand equal-track difference changed")
    require(hand_group["equal_supported_class_mean_differences"]["same_wrong_category"]["near_minus_far"] == .5, "Hand equal-class difference changed")
    require(scored["groups"]["novel_only"]["appearance_near"]["same_wrong_category"] is None, "Unsupported novel comparison is not null")
    out = {"status": "PASS_INDEPENDENT_HAND_WORKED_NUMERICAL_PREFLIGHT",
           "script_sha256": sha256(__file__), "n_unordered_pairs": 6,
           "same_wrong_pairs": 1, "both_wrong_pairs": 3,
           "endpoint_error_prevalence": .75, "prediction_agreement_pairs": 1,
           "cosines_in_upper_triangle_order": m[:, 4].tolist(),
           "pair_order": np.column_stack([i, j]).tolist(),
           "near_far_six_pair_check": {"far": far.tolist(), "near": near.tolist()},
           "tie_midrank_scores_1_1_3": ranks.tolist(),
           "minimum_six_and_disjoint_tie_selection": True,
           "different_gt_classes_shared_wrong_third_class": True,
           "hand_aggregation_check": {"all_pairs": 6, "far_pairs": 2, "near_pairs": 2,
                                      "same_wrong_pooled_equal_track_equal_class_difference": .5,
                                      "unsupported_novel_is_null": True}}
    path = args.out_dir / "natural_numerical_preflight.json"
    write_json(path, out)
    print("PASS numerical preflight", path, sha256(path), flush=True)


def load_common(args):
    started = time.monotonic()
    cm = json.load(open(args.audit / "manifest.json"))
    sm = json.load(open(args.audit / "scoreinput_manifest.json"))
    source_paths = {"annotation": args.ann, "baseline": args.baseline,
                    "track_cache": args.audit / "track_cache.npz", "scoreinput": args.audit / "scoreinput.npz"}
    hashes = {k: sha256(p) for k, p in source_paths.items()}
    require(cm["status"] == "PASS_FULL_TRACKER_POSTERIOR_CACHE_AUDIT", "Original cache audit incomplete")
    require(hashes["annotation"] == cm["input_sha256"]["annotation"] == sm["source"]["annotation"]["sha256"], "Annotation lineage mismatch")
    require(hashes["baseline"] == cm["input_sha256"]["official"] == sm["source"]["prediction"]["sha256"], "Baseline lineage mismatch")
    require(hashes["track_cache"] == cm["output_sha256"], "Track cache hash mismatch")
    require(hashes["scoreinput"] == sm["scoreinput_sha256"], "Geometry cache hash mismatch")
    geometry = dict(np.load(source_paths["scoreinput"], allow_pickle=False))
    with np.load(source_paths["track_cache"], allow_pickle=False) as data:
        cache = {k: data[k] for k in ("key", "global_track_id", "n_obs", "category_ids", "category_names", "category_frequency")}
    global_to_cache = {tuple(k): i for i, k in enumerate(np.column_stack([cache["key"][:, 0], cache["global_track_id"]]).tolist())}
    require(len(global_to_cache) == len(cache["key"]), "Duplicate official global track keys")
    track_to_cache = np.asarray([global_to_cache[tuple(k)] for k in geometry["track_key"].tolist()], dtype=np.int64)
    require(len(np.unique(track_to_cache)) == len(cache["key"]), "Track support is not bijective")
    require(np.array_equal(geometry["track_n_obs"], cache["n_obs"][track_to_cache]), "Track counts differ")
    ann = json.load(open(args.ann))
    images = {int(x["id"]): x for x in ann["images"]}
    require(len(images) == len(ann["images"]), "Duplicate annotation images")
    print("Verified source hashes and track bijection in %.1fs" % (time.monotonic() - started), flush=True)
    return cm, sm, geometry, cache, track_to_cache, ann, images, hashes, source_paths


def mapping_preflight(args):
    started = time.monotonic()
    cm, sm, g, c, track_to_cache, ann, images, hashes, source_paths = load_common(args)
    n = len(g["pred_row_index"])
    selected = {int(p): i for i, p in enumerate(g["pred_row_index"])}
    require(len(selected) == n, "Duplicate matched prediction rows")
    boxes = np.full((n, 4), np.nan)
    scores = np.full(n, np.nan)
    gt_boxes = np.full((n, 4), np.nan)
    gt_lookup = {}
    for row in ann["annotations"]:
        key = (int(row["image_id"]), int(row["track_id"]), int(row["category_id"]))
        require(key not in gt_lookup, "Duplicate GT identity/category in one image")
        gt_lookup[key] = row
    max_iou_error, seen = 0., 0
    for row_index, row in enumerate(iter_json_array(args.baseline)):
        mi = selected.get(row_index)
        if mi is None:
            continue
        ti = int(g["track_row"][mi])
        iid = int(g["image_id"][mi])
        require(int(row["image_id"]) == iid, "Geometry prediction-row image mismatch")
        require((int(row["video_id"]), int(row["track_id"])) == tuple(g["track_key"][ti]), "Geometry prediction-row track mismatch")
        require(int(row["video_id"]) == int(images[iid]["video_id"]), "Annotation video mismatch")
        key = (iid, int(g["gt_track_id"][mi]), int(g["gt_category_id"][mi]))
        gt = gt_lookup[key]
        boxes[mi] = row["bbox"]
        scores[mi] = row["score"]
        gt_boxes[mi] = gt["bbox"]
        p, q = boxes[mi], gt_boxes[mi]
        wh = np.maximum(0, np.minimum(p[:2] + p[2:], q[:2] + q[2:]) - np.maximum(p[:2], q[:2]))
        intersection = wh.prod()
        iou = intersection / (np.maximum(p[2:], 0).prod() + np.maximum(q[2:], 0).prod() - intersection)
        max_iou_error = max(max_iou_error, abs(iou - float(g["iou"][mi])))
        require(abs(iou - float(g["iou"][mi])) < 1e-6 and iou >= .5 - 1e-7, "Stored GT matching IoU mismatch")
        seen += 1
    require(row_index + 1 == sm["n_prediction_observations"] == cm["audit"]["observations"], "Baseline total count differs")
    require(seen == n and np.isfinite(boxes).all() and np.isfinite(scores).all(), "Matched baseline rows incomplete")
    chosen = np.sort(np.random.RandomState(args.seed).choice(n, min(256, n), replace=False))
    max_box_error, max_score_error = 0., 0.
    for mi in chosen:
        image = images[int(g["image_id"][mi])]
        ci = int(track_to_cache[int(g["track_row"][mi])])
        local_track_id = int(c["key"][ci, 1])
        with np.load(args.dump_dir / dump_name(image), allow_pickle=False) as data:
            require(canonical_filename(data["filename"].item()) == canonical_filename(image["file_name"]), "Sample filename mismatch")
            require(int(data["frame_id"]) == int(image["frame_id"]), "Sample frame_id mismatch")
            positions = np.flatnonzero(data["track_ids"] == local_track_id)
            require(len(positions) == 1, "Sample local track mapping is not unique")
            b = data["det_bboxes"][positions[0]].astype(float)
            xywh = np.r_[b[:2], b[2:4] - b[:2]]
            max_box_error = max(max_box_error, float(np.abs(xywh - boxes[mi]).max()))
            max_score_error = max(max_score_error, float(abs(b[4] - scores[mi])))
            require(np.allclose(xywh, boxes[mi], rtol=0, atol=1e-4), "Sample dump/baseline box mismatch")
            require(abs(b[4] - scores[mi]) < 1e-7, "Sample dump/baseline score mismatch")
    arrays_path = args.out_dir / "natural_mapping_preflight.npz"
    write_npz(arrays_path, bbox_xywh=boxes, detector_score=scores,
              matched_cache_row=track_to_cache[g["track_row"]], gt_bbox_xywh=gt_boxes)
    result = {"status": "PASS_INDEPENDENT_OBSERVATION_GT_MAPPING_PREFLIGHT",
              "script_sha256": sha256(__file__), "source_sha256": hashes,
              "all_matched_prediction_rows_identity_verified": seen,
              "all_matched_gt_annotation_identity_and_iou_verified": seen,
              "sample_seed": args.seed, "sample_observations": len(chosen),
              "sample_unique_images": len(np.unique(g["image_id"][chosen])),
              "sample_matched_indices": chosen.tolist(),
              "max_stored_iou_abs_error": max_iou_error,
              "sample_dump_baseline_box_max_abs_error": max_box_error,
              "sample_dump_baseline_score_max_abs_error": max_score_error,
              "track_support_bijection": True,
              "mapping_arrays_path": str(arrays_path), "mapping_arrays_sha256": sha256(arrays_path),
              "elapsed_seconds": time.monotonic() - started}
    path = args.out_dir / "natural_mapping_preflight.json"
    write_json(path, result)
    print("PASS mapping preflight", path, sha256(path), flush=True)


def metric_summary(total):
    n = float(total[0])
    out = {"n_pairs": int(n) if n.is_integer() else n}
    out.update({name: float(total[i + 1] / n) if n else None for i, name in enumerate(METRICS)})
    out["same_wrong_category_pairs"] = float(total[1])
    out["both_wrong_pairs"] = float(total[2])
    out["same_wrong_given_both_wrong"] = float(total[1] / total[2]) if total[2] else None
    return out


def outcome_vector(total):
    if not total[0]:
        return np.full(len(OUTCOMES), np.nan)
    return np.r_[total[1:5] / total[0], total[1] / total[2] if total[2] else np.nan]


class Accumulator:
    def __init__(self, nclass, ntrack):
        # all available, all eligible stratum pairs, far, near
        self.class_group_totals = np.zeros((4, 4, nclass, len(METRICS) + 1), dtype=float)
        self.class_totals = self.class_group_totals[:, 0]
        self.group_totals = np.zeros((4, 4, len(METRICS) + 1), dtype=float)
        self.track_totals = np.zeros((2, 4, ntrack, len(METRICS) + 1), dtype=float)
        self.available_tracks = set()
        self.eligible_tracks = set()
        self.n_strata = 0
        self.n_eligible_strata = 0
        self.n_tied_boundary_strata = 0
        self.n_flat_similarity_strata = 0
        self.available_class_tracks = defaultdict(set)
        self.eligible_class_tracks = defaultdict(set)

    def add(self, phase, matrix, pair_indices, left_class, right_class, rare, track):
        if not len(pair_indices):
            return
        m = np.column_stack([np.ones(len(pair_indices)), matrix[pair_indices]])
        a, b = left_class[pair_indices], right_class[pair_indices]
        np.add.at(self.class_totals[phase], a, .5 * m)
        np.add.at(self.class_totals[phase], b, .5 * m)
        self.group_totals[phase, 0] += m.sum(0)
        ra, rb = rare[a], rare[b]
        for gi, mask in ((1, ~ra & ~rb), (2, ra & rb), (3, ra != rb)):
            self.group_totals[phase, gi] += m[mask].sum(0)
            np.add.at(self.class_group_totals[phase, gi], a[mask], .5 * m[mask])
            np.add.at(self.class_group_totals[phase, gi], b[mask], .5 * m[mask])
            if phase in (2, 3):
                self.track_totals[phase - 2, gi, track] += m[mask].sum(0)
        if phase == 0:
            self.available_tracks.add(track)
            for ci in np.unique(np.r_[a, b]):
                self.available_class_tracks[int(ci)].add(track)
        elif phase == 1:
            self.eligible_tracks.add(track)
            for ci in np.unique(np.r_[a, b]):
                self.eligible_class_tracks[int(ci)].add(track)
        elif phase in (2, 3):
            self.track_totals[phase - 2, 0, track] += m.sum(0)

    def result(self, c, rare):
        groups = {}
        for gi, name in enumerate(PARTITIONS):
            far, near = self.group_totals[2, gi], self.group_totals[3, gi]
            require(far[0] == near[0], "Near/far counts differ")
            delta = outcome_vector(near) - outcome_vector(far)
            eligible_classes = np.flatnonzero(self.class_group_totals[2, gi, :, 0] > 0)
            equal_track, equal_class = {}, {}
            chosen_tracks = np.flatnonzero(self.track_totals[0, gi, :, 0] > 0)
            deltas = np.asarray([outcome_vector(self.track_totals[1, gi, ti]) - outcome_vector(self.track_totals[0, gi, ti]) for ti in chosen_tracks])
            for oi, outcome in enumerate(OUTCOMES):
                valid = np.isfinite(deltas[:, oi]) if len(deltas) else np.zeros(0, bool)
                equal_track[outcome] = {"near_minus_far": float(deltas[valid, oi].mean()) if valid.any() else None,
                                         "n_supported_tracks": int(valid.sum())}
            class_deltas = np.asarray([outcome_vector(self.class_group_totals[3, gi, ci]) - outcome_vector(self.class_group_totals[2, gi, ci]) for ci in eligible_classes])
            for oi, outcome in enumerate(OUTCOMES):
                valid = np.isfinite(class_deltas[:, oi]) if len(class_deltas) else np.zeros(0, bool)
                equal_class[outcome] = {"near_minus_far": float(class_deltas[valid, oi].mean()) if valid.any() else None,
                                         "n_supported_classes": int(valid.sum())}
            groups[name] = {"all_available_pairs_baseline": metric_summary(self.group_totals[0, gi]),
                            "all_pairs_in_eligible_strata_baseline": metric_summary(self.group_totals[1, gi]),
                            "appearance_far": metric_summary(far), "appearance_near": metric_summary(near),
                            "pooled_near_minus_far": {key: float(value) if np.isfinite(value) else None for key, value in zip(OUTCOMES, delta)},
                            "equal_track_mean_differences": equal_track or None,
                            "equal_supported_class_mean_differences": equal_class or None}
        perclass = []
        for ci, cid in enumerate(c["category_ids"]):
            far, near = self.class_totals[2, ci], self.class_totals[3, ci]
            df = outcome_vector(near) - outcome_vector(far)
            perclass.append({"category_id": int(cid), "name": str(c["category_names"][ci]),
                             "frequency": str(c["category_frequency"][ci]), "novel_frequency_rare": bool(rare[ci]),
                             "n_available_tracks": len(self.available_class_tracks[ci]),
                             "n_eligible_tracks": len(self.eligible_class_tracks[ci]),
                             "all_available_pair_endpoint_weight": float(self.class_totals[0, ci, 0]),
                             "eligible_pair_endpoint_weight": float(self.class_totals[1, ci, 0]),
                             "far_pair_endpoint_weight": float(far[0]), "near_pair_endpoint_weight": float(near[0]),
                             "all_available_baseline": {k: metric_summary(self.class_totals[0, ci])[k] for k in OUTCOMES},
                             "far": {k: metric_summary(far)[k] for k in OUTCOMES} if far[0] else None,
                             "near": {k: metric_summary(near)[k] for k in OUTCOMES} if near[0] else None,
                             "near_minus_far": {k: float(v) if np.isfinite(v) else None for k, v in zip(OUTCOMES, df)} if far[0] else None})
        return {"n_available_tracks": len(self.available_tracks), "n_eligible_tracks": len(self.eligible_tracks),
                "n_excluded_available_tracks_no_supported_stratum": len(self.available_tracks - self.eligible_tracks),
                "n_candidate_strata": self.n_strata, "n_eligible_strata": self.n_eligible_strata,
                "n_eligible_strata_tied_at_selected_boundary": self.n_tied_boundary_strata,
                "n_eligible_strata_flat_cosine": self.n_flat_similarity_strata,
                "unsupported_novel_category_ids": [int(c["category_ids"][i]) for i in np.flatnonzero(rare & (self.class_totals[2, :, 0] == 0))],
                "groups": groups, "per_class": perclass}


def source_provenance(args):
    paths = {"historical_dump_launch": args.audit / "original_dump_launch.json",
             "active_config": ROOT / "work/covtrack_tces_v2/configs/uncertainty-ovtrack-teta/ovtrack_r50_fullval.py",
             "mot_dump_code": ROOT / "work/covtrack_tces_v2/ovtrack/models/mot/ovtrack.py",
             "roi_feature_code": ROOT / "work/covtrack_tces_v2/ovtrack/models/roi_heads/ovtrack_roi_head.py",
             "tracker_code": ROOT / "work/covtrack_tces_v2/ovtrack/models/trackers/ovtracker.py"}
    return {"files": {k: {"path": str(v), "sha256": sha256(v)} for k, v in paths.items()},
            "active_roi_class": "OVTrackRoIHeadUncertainty", "active_tracker_class": "OVTrackerUncertainty",
            "track_feats": "Current active source: raw 256-D track-head vector from ROI uncertainty_simple_test, roi_head.py:2973, returned at 2992; ROI cls/location additions and ROI fusion are commented out at 2986-2990. Saved mot/ovtrack.py:261 after tracker keep mask. The tracker's fused local embeds rebinding does not appear to mutate this tensor in place.",
            "cls_feats": "512-D (text_head_feats + img_head_feats)/2 at roi_head.py:2952, returned as key_cls_feats, saved as cem_feats at mot/ovtrack.py:262.",
            "qualification": "track_feats is relatively independent of the classification vector only. Shared backbone/training, possible track-head initialization, and selection by a tracker using fused classification/location remain. Neither feature is independent noise. Inactive OVTrackRoISLAck additions are not attributed to this run. Source inspection describes current preserved code, not an instrumented historical execution trace."}


def protected_fingerprints(args):
    checks = {}
    protected = json.load(open(args.audit / "protected_before.json"))
    for path, expected in protected.items():
        if not isinstance(expected, dict) or "exists" not in expected:
            continue
        require(Path(path).exists() == expected["exists"], "Protected existence changed: " + path)
        value = sha256(path) if expected["exists"] else None
        require(value == expected["sha256"], "Protected file changed: " + path)
        checks[path] = {"exists": expected["exists"], "sha256": value}
    return checks


def full(args):
    started = time.monotonic()
    npf_path = args.out_dir / "natural_numerical_preflight.json"
    mpf_path = args.out_dir / "natural_mapping_preflight.json"
    npf, mpf = json.load(open(npf_path)), json.load(open(mpf_path))
    script_hash = sha256(__file__)
    require(npf["status"] == "PASS_INDEPENDENT_HAND_WORKED_NUMERICAL_PREFLIGHT", "Numerical preflight not passed")
    require(mpf["status"] == "PASS_INDEPENDENT_OBSERVATION_GT_MAPPING_PREFLIGHT", "Mapping preflight not passed")
    require(npf["script_sha256"] == mpf["script_sha256"] == script_hash, "Script changed after preflights")
    cm, sm, g, c, track_to_cache, ann, images, hashes, source_paths = load_common(args)
    require(hashes == mpf["source_sha256"], "Sources changed after mapping preflight")
    mapping_path = Path(mpf["mapping_arrays_path"])
    require(sha256(mapping_path) == mpf["mapping_arrays_sha256"], "Preflight mapping artifact changed")
    mapping = dict(np.load(mapping_path, allow_pickle=False))
    protected_before = protected_fingerprints(args)
    provenance = source_provenance(args)
    nmatch, ntrack = len(g["track_row"]), len(c["key"])
    matched_cache_row = track_to_cache[g["track_row"]]
    require(np.array_equal(matched_cache_row, mapping["matched_cache_row"]), "Mapping cache row mismatch")
    matched_lookup = {(int(iid), int(c["key"][ci, 1])): mi for mi, (iid, ci) in enumerate(zip(g["image_id"], matched_cache_row))}
    require(len(matched_lookup) == nmatch, "Matched local identity duplicate")
    local_index = {tuple(k): i for i, k in enumerate(c["key"].tolist())}
    require(len(local_index) == ntrack, "Duplicate local track keys")
    offsets = np.r_[0, np.cumsum(c["n_obs"])]
    all_scores = np.empty(int(offsets[-1]), dtype=np.float64)
    observation_counts = np.zeros(ntrack, dtype=np.int64)
    feats = {"track_feats": np.empty((nmatch, 256), dtype=np.float32),
             "cls_feats": np.empty((nmatch, 512), dtype=np.float32)}
    observed = np.zeros(nmatch, dtype=np.int64)
    prediction = np.empty(nmatch, dtype=np.int64)
    confidence = np.empty(nmatch, dtype=np.float64)
    frame = np.empty(nmatch, dtype=np.int64)
    matched_slot = np.empty(nmatch, dtype=np.int64)
    images_by_video = defaultdict(list)
    for im in ann["images"]:
        images_by_video[int(im["video_id"])].append(im)
    for rows in images_by_video.values():
        rows.sort(key=lambda x: int(x["frame_id"]))
    all_paths = {p.name: p for p in args.dump_dir.glob("*.npz")}
    require(len(all_paths) == cm["audit"]["dump_files"], "Dump count differs from audited source")
    inventory = hashlib.sha256()
    seen_names = set()
    scan = {"files": 0, "observations": 0, "matched_observations": 0,
            "max_matched_box_abs_error": 0., "max_matched_score_abs_error": 0.,
            "posterior_row_sum_max_abs_error": 0., "argmax_det_label_mismatches": 0,
            "feature_norm_ranges": {name: [float("inf"), 0.] for name in FEATURES}}
    scan_started = time.monotonic()
    for video in ann["videos"]:
        vid = int(video["id"])
        for im in images_by_video[vid]:
            name = dump_name(im)
            path = all_paths.get(name)
            if path is None:
                require(not any(iid == int(im["id"]) for iid, _ in matched_lookup), "Missing dump has matched observations")
                continue
            require(name not in seen_names, "Repeated dump filename")
            seen_names.add(name)
            inventory.update((name + "\t" + str(path.stat().st_size) + "\n").encode())
            with np.load(path, allow_pickle=False) as data:
                require(canonical_filename(data["filename"].item()) == canonical_filename(im["file_name"]), "Dump filename mismatch")
                require(int(data["frame_id"]) == int(im["frame_id"]), "Dump frame metadata mismatch")
                ids, boxes, p, labels = data["track_ids"], data["det_bboxes"], data["p_fused"], data["det_labels"]
                n = len(ids)
                require(ids.shape == (n,) and len(np.unique(ids)) == n, "Dump track IDs are not unique")
                require(boxes.shape == (n, 5) and np.isfinite(boxes).all(), "Dump box schema invalid")
                require(p.shape == (n, len(c["category_ids"]) + 1) and np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1 + 1e-6), "Posterior invalid")
                require(labels.shape == (n,), "det_labels schema invalid")
                rs_error = float(np.abs(p.sum(1, dtype=np.float64) - 1).max()) if n else 0.
                require(rs_error < 2e-6 and not np.count_nonzero(p[:, -1]), "Posterior normalization/background schema differs")
                scan["posterior_row_sum_max_abs_error"] = max(scan["posterior_row_sum_max_abs_error"], rs_error)
                # Compute label/confidence for EVERY emitted observation before GT selection.
                predicted = c["category_ids"][p[:, :-1].argmax(1)]
                pmax = p[:, :-1].max(1)
                scan["argmax_det_label_mismatches"] += int(np.count_nonzero(predicted != labels + 1))
                normalized = {}
                for feature in FEATURES:
                    x = data[feature].astype(np.float32)
                    require(x.shape == (n, feats[feature].shape[1]) and np.isfinite(x).all(), "Feature schema invalid: " + feature)
                    norms = np.linalg.norm(x, axis=1)
                    require(np.all(norms > 0), "Zero feature norm: " + feature)
                    if n:
                        scan["feature_norm_ranges"][feature][0] = min(scan["feature_norm_ranges"][feature][0], float(norms.min()))
                        scan["feature_norm_ranges"][feature][1] = max(scan["feature_norm_ranges"][feature][1], float(norms.max()))
                    normalized[feature] = x / norms[:, None]
                ci = np.asarray([local_index[(vid, int(tid))] for tid in ids], dtype=np.int64)
                require(np.all(observation_counts[ci] < c["n_obs"][ci]), "Track count exceeds audited source")
                positions = offsets[ci] + observation_counts[ci]
                all_scores[positions] = boxes[:, 4]
                observation_counts[ci] += 1
                for di, tid in enumerate(ids):
                    mi = matched_lookup.get((int(im["id"]), int(tid)))
                    if mi is None:
                        continue
                    require(ci[di] == matched_cache_row[mi], "Matched cache track differs")
                    require(observed[mi] == 0, "Matched observation filled twice")
                    for feature in FEATURES:
                        feats[feature][mi] = normalized[feature][di]
                    prediction[mi] = predicted[di]
                    confidence[mi] = pmax[di]
                    frame[mi] = int(im["frame_id"])
                    matched_slot[mi] = positions[di]
                    observed[mi] += 1
                    box = boxes[di, :4].astype(float)
                    xywh = np.r_[box[:2], box[2:] - box[:2]]
                    be = float(np.abs(xywh - mapping["bbox_xywh"][mi]).max())
                    se = float(abs(float(boxes[di, 4]) - mapping["detector_score"][mi]))
                    require(be < 1e-4 and se < 1e-7, "Full matched dump/baseline geometry mismatch")
                    scan["max_matched_box_abs_error"] = max(scan["max_matched_box_abs_error"], be)
                    scan["max_matched_score_abs_error"] = max(scan["max_matched_score_abs_error"], se)
                    scan["matched_observations"] += 1
                scan["files"] += 1
                scan["observations"] += n
            if scan["files"] % 500 == 0:
                print("Full natural-dependence scan files %d/%d observations %d matched %d elapsed %.1fs" %
                      (scan["files"], len(all_paths), scan["observations"], scan["matched_observations"], time.monotonic() - scan_started), flush=True)
    require(seen_names == set(all_paths), "Unprocessed dump files")
    require(inventory.hexdigest() == cm["dump_filename_size_inventory_sha256"], "Dump filename/size inventory changed")
    require(scan["observations"] == sm["n_prediction_observations"] and np.array_equal(observation_counts, c["n_obs"]), "Full observation/per-track counts differ")
    require(np.all(observed == 1) and scan["matched_observations"] == nmatch, "Full matched feature extraction incomplete")
    score_rank = np.empty(nmatch, dtype=np.float64)
    sort_order = np.argsort(matched_cache_row, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(matched_cache_row[sort_order][1:] != matched_cache_row[sort_order][:-1]) + 1, nmatch]
    track_rows = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        rows = sort_order[start:stop]
        ci = int(matched_cache_row[rows[0]])
        ranks = midranks(all_scores[offsets[ci]:offsets[ci + 1]])
        score_rank[rows] = ranks[matched_slot[rows] - offsets[ci]]
        track_rows.append(rows[np.argsort(frame[rows], kind="stable")])
    cat_index = {int(cid): i for i, cid in enumerate(c["category_ids"])}
    gt_class = np.asarray([cat_index[int(cid)] for cid in g["gt_category_id"]], dtype=np.int64)
    rare = c["category_frequency"] == "r"
    n_scored_tracks = len(track_rows)
    accum = {(scope, variant, feature): Accumulator(len(rare), n_scored_tracks)
             for scope in SCOPES for variant in VARIANTS for feature in FEATURES}
    scope_track_counts = dict.fromkeys(SCOPES, 0)
    scope_pair_counts = dict.fromkeys(SCOPES, 0)
    n_single, n_mixed, single_obs, mixed_obs = 0, 0, 0, 0
    for ti, rows in enumerate(track_rows):
        truth = g["gt_category_id"][rows]
        gid = g["gt_track_id"][rows]
        is_single = len(np.unique(gid)) == 1
        n_single += int(is_single)
        n_mixed += int(not is_single)
        single_obs += len(rows) * int(is_single)
        mixed_obs += len(rows) * int(not is_single)
        if len(rows) < 2:
            continue
        i, j, base_matrix = pair_matrix(feats["track_feats"][rows], prediction[rows], truth,
                                        frame[rows], mapping["detector_score"][rows], score_rank[rows], confidence[rows])
        same_identity = gid[i] == gid[j]
        same_class = truth[i] == truth[j]
        require(np.all(same_class[same_identity]), "One GT identity has different categories")
        masks = {SCOPES[0]: np.full(len(i), is_single, dtype=bool),
                 SCOPES[1]: ~np.full(len(i), is_single, dtype=bool) & same_identity,
                 SCOPES[2]: ~same_identity & same_class,
                 SCOPES[3]: ~same_identity & ~same_class}
        left_class, right_class = gt_class[rows][i], gt_class[rows][j]
        lag = base_matrix[:, 5].astype(int)
        require(np.all(lag > 0), "A track has duplicate sampled frame positions")
        lag_bin = np.digitize(lag, [3, 6, 11])
        score_mean_bin = np.minimum((base_matrix[:, 8] * 4).astype(int), 3)
        score_difference_bin = (base_matrix[:, 9] > .25).astype(int)
        confidence_mean_bin = np.minimum((base_matrix[:, 10] * 4).astype(int), 3)
        confidence_difference_bin = (base_matrix[:, 11] > .25).astype(int)
        lo_class = np.minimum(left_class, right_class)
        hi_class = np.maximum(left_class, right_class)
        for scope, mask in masks.items():
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            scope_track_counts[scope] += 1
            scope_pair_counts[scope] += len(indices)
            for variant in VARIANTS:
                strata = defaultdict(list)
                for pi in indices:
                    key = (int(lag[pi] if variant == VARIANTS[0] else lag_bin[pi]),
                           int(score_mean_bin[pi]), int(score_difference_bin[pi]),
                           int(lo_class[pi]), int(hi_class[pi]))
                    if variant == VARIANTS[2]:
                        key += (int(confidence_mean_bin[pi]), int(confidence_difference_bin[pi]))
                    strata[key].append(pi)
                for feature in FEATURES:
                    matrix = base_matrix.copy()
                    if feature == "cls_feats":
                        x = feats[feature][rows]
                        matrix[:, 4] = np.clip((x[i] * x[j]).sum(1, dtype=np.float64), -1, 1)
                    a = accum[(scope, variant, feature)]
                    a.add(0, matrix, indices, left_class, right_class, rare, ti)
                    a.n_strata += len(strata)
                    for pair_list in strata.values():
                        pair_indices = np.asarray(pair_list, dtype=np.int64)
                        selection = near_far(matrix[:, 4], pair_indices)
                        if selection is None:
                            continue
                        far, near = selection
                        a.n_eligible_strata += 1
                        a.n_flat_similarity_strata += int(np.ptp(matrix[pair_indices, 4]) == 0)
                        ordered_cosine = np.sort(matrix[pair_indices, 4])
                        k = len(far)
                        a.n_tied_boundary_strata += int(ordered_cosine[k - 1] == ordered_cosine[k] or ordered_cosine[-k] == ordered_cosine[-k - 1])
                        a.add(1, matrix, pair_indices, left_class, right_class, rare, ti)
                        a.add(2, matrix, far, left_class, right_class, rare, ti)
                        a.add(3, matrix, near, left_class, right_class, rare, ti)
        if (ti + 1) % 1000 == 0:
            print("Scored matched tracks %d/%d elapsed %.1fs" % (ti + 1, n_scored_tracks, time.monotonic() - started), flush=True)
    results = {scope: {variant: {feature: accum[(scope, variant, feature)].result(c, rare)
                                for feature in FEATURES} for variant in VARIANTS} for scope in SCOPES}
    for variant in VARIANTS:
        for feature in FEATURES:
            for scope in SCOPES:
                require(results[scope][variant][feature]["groups"]["all"]["all_available_pairs_baseline"]["n_pairs"] == scope_pair_counts[scope], "Scope pair denominator mismatch")
    protected_after = protected_fingerprints(args)
    require(protected_after == protected_before, "Protected sources changed during diagnosis")
    final_hashes = {k: sha256(v) for k, v in source_paths.items()}
    require(final_hashes == hashes and sha256(__file__) == script_hash, "Input or diagnostic script changed during run")
    result = {
        "status": "PASS_FULL_NATURAL_ERROR_DEPENDENCE_DESCRIPTIVE_AUDIT",
        "script": {"path": str(Path(__file__).resolve()), "sha256": script_hash},
        "runtime": {"python": sys.executable, "numpy": np.__version__, "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "source_sha256_before": hashes, "source_sha256_after": final_hashes,
        "source_paths": {k: str(v) for k, v in source_paths.items()},
        "preflights": {"numerical": {"path": str(npf_path), "sha256": sha256(npf_path), "status": npf["status"]},
                       "mapping": {"path": str(mpf_path), "sha256": sha256(mpf_path), "status": mpf["status"]}},
        "feature_origin": provenance,
        "scan": dict(scan, dump_filename_size_inventory_sha256=inventory.hexdigest(), all_track_counts_equal_original=True,
                     matched_features_filled_exactly_once=True, detector_score_range=[float(all_scores.min()), float(all_scores.max())]),
        "matched_track_support": {"total_tracks": n_scored_tracks, "total_observations": nmatch,
                                  "single_gt_identity_tracks": n_single, "single_gt_identity_observations": single_obs,
                                  "mixed_gt_identity_tracks": n_mixed, "mixed_gt_identity_observations": mixed_obs,
                                  "scope_tracks_with_at_least_one_pair": scope_track_counts, "scope_all_pair_counts": scope_pair_counts},
        "protocol": {
            "scoring_only": True, "official_teta_or_clsa": False,
            "geometry_matching": sm["matching_protocol"],
            "classification": "argmax of original p_fused foreground 296 columns for every emitted observation, before GT-based diagnostic selection; no candidate readout labels are generated",
            "pairs": "Every unordered pair of matched observations within each emitted predicted trajectory; no stochastic pair or track subsampling. Stable chronological pair enumeration resolves cosine ties deterministically.",
            "strata": "Same predicted trajectory, same identity-pair scope, unordered GT-class pair (diagnostic composition control), exact sampled frame_id lag for primary or lag bin for sensitivity, pair detector-score midrank-mean quartile and midrank absolute difference <=.25 versus >.25.",
            "sampled_frame_id_lag": "Difference in annotated sampled frame_id, not raw video frame number or seconds. Sensitivity bins 1-2, 3-5, 6-10, 11+.",
            "score_control": "Detector scores are tiny and are not treated as calibrated reliability. Midranks (average 1-based rank-.5)/n use ALL emitted observations of that trajectory, including unmatched observations. Pair mean-rank quartiles have cutpoints .25,.50,.75; rank difference cutpoint .25.",
            "posterior_sensitivity": "binned_lag_score_rank_posterior_max additionally bins pair mean of max original foreground p_fused into [0,.25),[.25,.5),[.5,.75),[.75,1] and absolute difference <=.25 versus >.25.",
            "selection": "Require >=6 pairs per stratum; select floor(n/3) smallest cosine as far and the same number largest cosine as near, independently for track_feats and cls_feats. All available pairs and all pairs in supported strata retain separate baseline denominators.",
            "same_wrong": "Both endpoints predict a class different from their own GT class AND predict the same category. For different GT classes this is a shared wrong third class, not merely an identity switch.",
            "baseline_error_prevalence": "Mean of the two endpoint error indicators over ALL pairs in the stated denominator; both-wrong and prediction agreement are also reported. same_wrong_given_both_wrong is conditional and uses its own denominator.",
            "per_class_weight": "Each pair contributes half weight to each endpoint GT class; same-class pairs therefore contribute one. Unsupported comparisons are null, never zero. Equal-class means use supported classes only.",
            "equal_track_and_class_means": "For every pair partition, first pool its selected far/near pairs within each supported predicted track or endpoint GT class, compute near-minus-far rates, then average those differences equally across supported units. Conditional same_wrong_given_both_wrong differences require both far and near both-wrong denominators to be nonzero and report their own support.",
            "uncertainty": "No pair-independent significance tests; pairs share trajectories. Differences and support are descriptive. Natural association is not causal and does not establish candidate efficacy.",
        },
        "results": results,
        "protected_before": protected_before, "protected_after": protected_after,
        "elapsed_seconds": time.monotonic() - started,
    }
    output = args.out_dir / "natural_error_dependence.json"
    write_json(output, result)
    print("PASS FULL", output, "sha256", sha256(output), "elapsed", result["elapsed_seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["preflight-numerical", "preflight-mapping", "full"])
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "results/cov_tces_v3_fullval")
    parser.add_argument("--ann", type=Path, default=Path("/raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/cov_val_full_official_a100/tao_track.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/tces_audit_20261023")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/dependence_audit_20261023")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    guarded = {"preflight-numerical": "natural_numerical_preflight.json", "preflight-mapping": "natural_mapping_preflight.json", "full": "natural_error_dependence.json"}
    require(not (args.out_dir / guarded[args.stage]).exists(), "Refusing to overwrite final stage artifact")
    if args.stage == "preflight-numerical":
        numerical_preflight(args)
    elif args.stage == "preflight-mapping":
        mapping_preflight(args)
    else:
        full(args)


if __name__ == "__main__":
    main()
