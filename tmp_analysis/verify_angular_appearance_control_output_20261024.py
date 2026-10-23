#!/usr/bin/env python3
"""Independent complete-FIT angular-control output verifier; bounded blocks.

The real path imports neither the angular producer/core nor its GT decoder.
The pinned previous independent audit supplies only utilities, GT-free
population metadata and an independently audited safe eligible-ID scanner.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sqlite3
import tempfile
import time

import numpy as np
import verify_fit_lag_identity_support_output_20261024 as base

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/base_error_full_collection_20261023"
RUN = STAGE / "full_collect_v1"
SUPPORT = STAGE / "support_v1"
PROTOCOL = ROOT / "tmp_analysis/angular_appearance_control_protocol_20261024.json"
PRODUCER = ROOT / "tmp_analysis/diagnose_angular_appearance_control_20261024.py"
PREFLIGHT = ROOT / "tmp_analysis/angular_appearance_control_independent_preflight_20261024.json"
PARENT = ROOT / "results/fit_lag_identity_support_20261024_v1"
FEATURES = ("raw_appearance_cosine", "angular_prediction_cosine")
LAGS = (1, 4)
SCOPES = ("all_candidates", "same_predicted_category")
POPULATIONS = ("all_known_population", "common_CV_covered_population")
SUBSETS = ("all_nonempty_known_queries", "positive_and_negative_available_queries")
FINAL_PRODUCER_SHA = "1e9fae320d2435a4560ffeeba3bfe441891f9d9db202296b269ba0419096828e"
FINAL_PROTOCOL_SHA = "44d8bce20de4ed3fd358a083665e6e9d8b2744d29d86dcdc9b2d9efa2f07eb69"
FINAL_PREFLIGHT_SHA = "fed48008427ce84c0a0732f9b6de25694cc2d199549dab84b2ad259e68b97824"
BASE_AUDITOR_SHA = "74a5c59c957439411acdb7139822c8d83c583584ab07e9e21957e731e648d3ce"
SCANNER_SHA = "862d31f63331a288804d808fb303e6704d6f4fdfa80f97daf13971f44c0f795d"
EXPECTED_ELIGIBLE_IDENTITIES = 21661
EXPECTED_TUNE_IDENTITIES = 2226
check, sha, load, same = base.check, base.sha, base.load, base.same


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def legal_anchors(video_source):
    """Independent split-first safe projection and immutable-row ID join."""
    manual = SUPPORT / "supervised_observations.jsonl"
    receipt = base.project((SUPPORT / "support_receipt.json").read_text(), {"output_files"})
    check(sha(manual) == receipt["output_files"][manual.name], "Manual payload hash changed")
    fields = {"split", "observation_id", "source_group", "video_id", "image_id", "sampled_step", "original_frame_index",
        "local_track_id", "gt_track_id", "manual_anchor_id", "eligible_for_parameter_fit", "pred_category_id"}
    identities, tune = {}, 0
    for line in manual.open():
        split = base.project(line, {"split"})["split"]
        check(split in ("fit", "tune"), "Unexpected identity split")
        if split == "tune":
            tune += 1
            continue
        row = base.project(line, fields)
        check(set(row) == fields and row["eligible_for_parameter_fit"] is True, "Ineligible manual identity")
        check(row["observation_id"] not in identities, "Duplicate eligible manual observation")
        identities[row["observation_id"]] = row
    check(len(identities) == EXPECTED_ELIGIBLE_IDENTITIES and tune == EXPECTED_TUNE_IDENTITIES, "Frozen eligible identity population changed")
    index = SUPPORT / "gt_free_observations.jsonl"
    check(sha(index) == load(SUPPORT / "gt_free_index_manifest.json")["files"][index.name], "Immutable observation index changed")
    anchors = {}
    for line in index.open():
        row = json.loads(line)
        if row["split"] != "fit":
            continue
        check(video_source[str(row["video_id"])] == row["source_group"], "Immutable index source mismatch")
        identity = identities.get(row["observation_id"])
        if identity is None:
            continue
        for key in fields - {"observation_id", "gt_track_id", "manual_anchor_id", "eligible_for_parameter_fit"}:
            check(row[key] == identity[key], "Immutable eligible identity provenance mismatch: " + key)
        key = (row["image_id"], row["dump_row"])
        check(key not in anchors, "Duplicate immutable eligible row")
        anchors[key] = (row["video_id"], identity["gt_track_id"])
    check(len(anchors) == len(identities), "Eligible identity missing immutable row")
    return anchors, tune


def independent_history(frame, history):
    """Snapshot the previous unique locator/raw/box before updating state."""
    ids = [int(x) for x in frame["released_ids"]]
    multiplicity = Counter(ids)
    n, width = frame["raw"].shape
    old_boxes = np.full((n, 4), np.nan)
    old_times = np.full(n, np.nan)
    old_raw = np.zeros((n, width), np.float64)
    locators, reasons = [], []
    for row, identity in enumerate(ids):
        locator = None
        if identity < 0:
            reason = "unassigned_id"
        elif multiplicity[identity] != 1:
            reason = "duplicate_current_id"
        elif identity not in history:
            reason = "no_previous_occurrence"
        elif history[identity] is None:
            reason = "previous_duplicate_invalid"
        else:
            previous = history[identity]
            check(previous["time"] < frame["original_frame_index"], "History must be strictly past")
            old_boxes[row] = previous["box"]
            old_times[row] = previous["time"]
            old_raw[row] = previous["raw"]
            locator = dict(image_id=previous["image_id"], dump_row=previous["dump_row"],
                           original_frame_index=previous["time"])
            reason = "available"
        locators.append(locator)
        reasons.append(reason)
    for row, identity in enumerate(ids):
        if identity < 0:
            continue
        history[identity] = None if multiplicity[identity] != 1 else dict(
            box=frame["boxes"][row].copy(), raw=frame["raw"][row].copy(),
            time=frame["original_frame_index"], image_id=frame["image_id"], dump_row=row)
    frame.update(previous_boxes=old_boxes, previous_times=old_times, previous_raw=old_raw,
                 previous_locators=locators, history_reason=reasons)
    return frame


def read_raw(record, history):
    # Read only declared cache fields. No posterior/class/GT targets are read.
    with np.load(record["npz"], allow_pickle=False) as saved:
        for key, field in (("video_id", "video_id"), ("image_id", "image_id"),
                           ("frame_id", "frame_id"), ("frame_index", "frame_index")):
            check(int(saved[key]) == record[field], "Raw frame metadata mismatch")
        check(saved["track_feats"].dtype == np.dtype("float16"), "Raw quantization changed")
        boxes = saved["det_bboxes"].astype(np.float64)
        raw = saved["track_feats"].astype(np.float64)
        ids = saved["track_ids"].astype(np.int64)
        categories = saved["det_labels"].astype(np.int32) + 1
    n = record["observations"]
    check(boxes.shape == (n, 5) and raw.shape == (n, 256) and ids.shape == categories.shape == (n,),
          "Raw retained row shape mismatch")
    check(np.isfinite(boxes).all() and np.isfinite(raw).all() and
          np.all(boxes[:, 2:4] > boxes[:, :2]), "Invalid retained input")
    frame = dict(video_id=record["video_id"], image_id=record["image_id"],
        sampled_step=record["frame_id"], original_frame_index=record["frame_index"],
        boxes=boxes[:, :4], raw=raw, released_ids=ids, predicted_categories=categories)
    return independent_history(frame, history)


def independent_prediction(reference, query_time):
    """Shortest great-circle forecast derived without calling the core.

    The forward tangent is (c*last - previous)/sin(theta). The epsilon
    belongs to the pinned core, not an estimated threshold or clipping gate.
    Zero last features remain exactly zero outside unit-vector arithmetic.
    """
    last = np.asarray(reference["raw"], np.float64)
    previous = np.asarray(reference["previous_raw"], np.float64)
    norms = np.sqrt(np.sum(last * last, axis=1))
    prior_norms = np.sqrt(np.sum(previous * previous, axis=1))
    nonzero = norms > 0
    valid = np.isfinite(reference["previous_times"]) & (prior_norms > 0) & nonzero
    check(query_time >= reference["original_frame_index"], "Query predates reference")
    check(np.all(reference["previous_times"][valid] < reference["original_frame_index"]),
          "Invalid angular history timeline")
    u = np.divide(last, norms[:, None], out=np.zeros_like(last), where=nonzero[:, None])
    p = np.divide(previous, prior_norms[:, None], out=np.zeros_like(previous), where=(prior_norms > 0)[:, None])
    c = np.clip(np.sum(p * u, axis=1), -1.0, 1.0)
    backward = p - c[:, None] * u
    sine = np.sqrt(np.sum(backward * backward, axis=1))
    directed = valid & (sine > 64 * np.finfo(np.float64).eps)
    coincident = valid & ~directed & (c >= 0)
    antipodal = valid & ~directed & (c < 0)
    rho = np.zeros(len(last), np.float64)
    theta = np.zeros(len(last), np.float64)
    rho[valid] = (query_time - reference["original_frame_index"]) / (
        reference["original_frame_index"] - reference["previous_times"][valid])
    theta[valid] = np.arctan2(sine[valid], c[valid])
    prediction = u.copy()
    phase = rho[directed] * theta[directed]
    forward = -backward[directed] / sine[directed, None]
    prediction[directed] = np.cos(phase)[:, None] * u[directed] + np.sin(phase)[:, None] * forward
    return prediction, dict(directed_arc=directed, coincident_static=coincident,
        antipodal_fallback=antipodal, missing_or_zero_history=nonzero & ~valid,
        last_zero_norm=~nonzero, previous_zero_norm=np.isfinite(reference["previous_times"]) & (prior_norms == 0),
        history_angle_radians=theta, extrapolation_ratio=rho)


def independent_scores(query, reference):
    def unit_floor(values):
        return values / np.maximum(np.sqrt(np.sum(values * values, axis=1))[:, None], 1e-12)
    prediction, diagnostic = independent_prediction(reference, query["original_frame_index"])
    # Prediction already has the exact core output convention, including zero.
    q = unit_floor(query["raw"])
    scores = np.stack((q @ unit_floor(reference["raw"]).T, q @ prediction.T), axis=2)
    return scores, diagnostic


def independent_row_metadata(frame):
    result = dict(video_id=frame["video_id"], image_id=frame["image_id"],
        sampled_step=frame["sampled_step"], original_frame_index=frame["original_frame_index"],
        boxes=frame["boxes"].tolist(), released_ids=frame["released_ids"].tolist(),
        predicted_categories=frame["predicted_categories"].tolist(), history_reason=frame["history_reason"],
        previous_boxes=[row.tolist() if np.isfinite(row).all() else None for row in frame["previous_boxes"]],
        previous_times=[int(value) if np.isfinite(value) else None for value in frame["previous_times"]],
        previous_locators=frame["previous_locators"])
    return result


def math_self_test():
    def frame(last, prior, prior_time, time=3):
        return dict(raw=np.asarray(last, float), previous_raw=np.asarray(prior, float),
            previous_times=np.asarray(prior_time, float), original_frame_index=time)
    thirty = np.pi / 6
    r = frame([[np.cos(thirty), np.sin(thirty)], [1, 0], [1, 0], [0, 0], [1, 0]],
              [[1, 0], [-1, 0], [1, 0], [0, 0], [0, 0]], [0, 0, 0, np.nan, 0])
    prediction, d = independent_prediction(r, 9)
    np.testing.assert_allclose(prediction[0], [0, 1], atol=1e-14)
    np.testing.assert_array_equal(prediction[1:], [[1, 0], [1, 0], [0, 0], [1, 0]])
    check(d["directed_arc"].tolist() == [True, False, False, False, False], "Directed angular oracle")
    check(d["antipodal_fallback"].tolist() == [False, True, False, False, False], "Antipodal oracle")
    check(d["coincident_static"].tolist() == [False, False, True, False, False], "Coincident oracle")
    check(d["last_zero_norm"].tolist() == [False, False, False, True, False], "Zero-last oracle")
    check(d["missing_or_zero_history"].tolist() == [False, False, False, False, True], "Missing/zero-prior oracle")
    long, _ = independent_prediction(frame([[0, 1]], [[1, 0]], [0], 1), 9)
    np.testing.assert_allclose(long, [[0, 1]], atol=1e-14)
    reverse, _ = independent_prediction(frame([[np.cos(thirty), -np.sin(thirty)]], [[1, 0]], [0]), 9)
    np.testing.assert_allclose(reverse, [[0, -1]], atol=1e-14)
    return 8


def database(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE query_support(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,image INTEGER,qrow INTEGER,qgt INTEGER,p INTEGER,n INTEGER,raw_correct INTEGER,correct_bits INTEGER,PRIMARY KEY(lag,scope,population,video,image,qrow))")
    return db


def fresh_cell():
    result = {key: 0 for key in ("candidate_pairs", "known_pairs", "positive_pairs", "negative_pairs", "unknown_pairs",
        "nonempty_known_queries", "positive_available_queries", "negative_available_queries", "hard_queries")}
    result["top1"] = {subset: {feature: {key: 0 for key in ("raw_wrong_feature_wrong", "raw_wrong_feature_correct",
        "raw_correct_feature_wrong", "raw_correct_feature_correct")} for feature in FEATURES} for subset in SUBSETS}
    result["pair_order"] = {feature: [[0] * 3 for _ in range(3)] for feature in FEATURES}
    # These independently verify all corresponding old raw support statistics.
    result["_trajectories"] = set()
    result["_identity_pairs"] = set()
    return result


def count_block(cell, scores, mask, qids, rids, source, lag, scope, population, video, image, db):
    """Literal enumeration with immutable reference-row ties; no label filling."""
    for qr, qgt in enumerate(qids):
        permitted = [row for row in range(len(rids)) if mask[qr, row]]
        known = [row for row in permitted if qgt >= 0 and rids[row] >= 0]
        positives = [row for row in known if rids[row] == qgt]
        negatives = [row for row in known if rids[row] != qgt]
        cell["candidate_pairs"] += len(permitted)
        cell["known_pairs"] += len(known)
        cell["positive_pairs"] += len(positives)
        cell["negative_pairs"] += len(negatives)
        cell["unknown_pairs"] += len(permitted) - len(known)
        if not known:
            continue
        check(np.isfinite(scores[qr, known]).all(), "Nonfinite fixed-population score")
        hard = bool(positives and negatives)
        cell["nonempty_known_queries"] += 1
        cell["positive_available_queries"] += bool(positives)
        cell["negative_available_queries"] += bool(negatives)
        cell["hard_queries"] += hard
        correct = [rids[max(known, key=lambda row: (float(scores[qr, row, col]), -row))] == qgt for col in range(2)]
        bits = sum(1 << col for col in range(2) if correct[col])
        db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
            source, lag, scope, population, video, image, qr, int(qgt), len(positives), len(negatives), int(correct[0]), bits))
        cell["_trajectories"].update((video, int(g)) for g in [qgt] + [rids[row] for row in known])
        cell["_identity_pairs"].update((video, int(qgt), int(rids[row])) for row in known)
        for subset in SUBSETS if hard else SUBSETS[:1]:
            for col, feature in enumerate(FEATURES):
                key = "raw_" + ("correct" if correct[0] else "wrong") + "_feature_" + ("correct" if correct[col] else "wrong")
                cell["top1"][subset][feature][key] += 1
        for positive in positives:
            for negative in negatives:
                order = [0 if scores[qr, positive, col] < scores[qr, negative, col] else
                         2 if scores[qr, positive, col] > scores[qr, negative, col] else 1 for col in range(2)]
                for col, feature in enumerate(FEATURES):
                    cell["pair_order"][feature][order[0]][order[col]] += 1


def final_cell(cell):
    result = {key: int(value) for key, value in cell.items() if key not in ("top1", "pair_order", "_trajectories", "_identity_pairs")}
    result["top1"] = {}
    for subset in SUBSETS:
        result["top1"][subset] = {}
        for feature in FEATURES:
            counts = dict(cell["top1"][subset][feature])
            counts.update(repair_raw_error=counts["raw_wrong_feature_correct"], break_raw_correct=counts["raw_correct_feature_wrong"],
                raw_correct=counts["raw_correct_feature_wrong"] + counts["raw_correct_feature_correct"],
                feature_correct=counts["raw_wrong_feature_correct"] + counts["raw_correct_feature_correct"])
            result["top1"][subset][feature] = counts
    result["positive_negative_pair_order"] = {feature: dict(comparisons=sum(map(sum, cell["pair_order"][feature])),
        raw_to_feature_loss_tie_win_matrix=cell["pair_order"][feature]) for feature in FEATURES}
    return result


def combine_cells(values):
    result = fresh_cell()
    for cell in values:
        for key in result:
            if key == "top1":
                for subset in SUBSETS:
                    for feature in FEATURES:
                        for count in result[key][subset][feature]:
                            result[key][subset][feature][count] += cell[key][subset][feature][count]
            elif key == "pair_order":
                for feature in FEATURES:
                    for a in range(3):
                        for b in range(3):
                            result[key][feature][a][b] += cell[key][feature][a][b]
            elif key.startswith("_"):
                result[key].update(cell[key])
            else:
                result[key] += cell[key]
    return result


def query_rows(db, source, lag, scope, population):
    where, args = "lag=? AND scope=? AND population=?", [lag, scope, population]
    if source is not None:
        where += " AND source=?"
        args.append(source)
    return db.execute("SELECT source,video,image,qrow,qgt,p,n,raw_correct,correct_bits FROM query_support WHERE " + where + " ORDER BY source,video,image,qrow", args)


def query_metrics_subset(db, source, lag, scope, population, hard):
    trajectories = defaultdict(lambda: [0, 0, 0])
    for s, video, image, row, gt, p, n, raw, bits in query_rows(db, source, lag, scope, population):
        if not hard or (p and n):
            value = trajectories[s, video, gt]
            value[0] += 1
            value[1] += not raw
            value[2] += not (bits & 2)
    rows = [dict(source=s, video_id=v, gt_track_id=g, hard_queries=value[0], raw_errors=value[1], angular_errors=value[2])
            for (s, v, g), value in sorted(trajectories.items())]
    bysource = defaultdict(list)
    for row in rows:
        bysource[row["source"]].append(row)
    result = dict(query_GT_error_distribution=rows, query_GT_trajectories=len(rows), hard_query_sources=len(bysource))
    for name, column in (("raw", "raw_errors"), ("angular", "angular_errors")):
        errors = sum(row[column] for row in rows)
        source_errors = [sum(row[column] for row in group) for group in bysource.values()]
        result[name] = dict(errors=errors, error_sources=sum(value > 0 for value in source_errors),
            error_GT_trajectories=sum(row[column] > 0 for row in rows),
            largest_source_error_share=max(source_errors, default=0) / errors if errors else None,
            largest_GTtrajectory_error_share=max((row[column] for row in rows), default=0) / errors if errors else None,
            GTtrajectory_macro_correctness=float(np.mean([1 - row[column] / row["hard_queries"] for row in rows])) if rows else None,
            source_GTtrajectory_query_macro_correctness=float(np.mean([np.mean([1 - row[column] / row["hard_queries"] for row in group]) for group in bysource.values()])) if bysource else None,
            source_query_macro_correctness=float(np.mean([1 - sum(row[column] for row in group) / sum(row["hard_queries"] for row in group) for group in bysource.values()])) if bysource else None)
    return result


def query_metrics(db, source, lag, scope, population):
    return {subset: query_metrics_subset(db, source, lag, scope, population, index == 1)
            for index, subset in enumerate(SUBSETS)}


def cross_lag(db, source, scope, population):
    bylag = {}
    for lag in LAGS:
        bylag[lag] = {(v, image, row): (bool(raw), bool(bits & 2))
            for s, v, image, row, g, p, n, raw, bits in query_rows(db, source, lag, scope, population) if p and n}
    shared = set(bylag[1]) & set(bylag[4])
    result = dict(shared_hard_queries=len(shared))
    for col, name in enumerate(("raw", "angular")):
        result[name + "_error_lag1"] = sum(not bylag[1][key][col] for key in shared)
        result[name + "_error_lag4"] = sum(not bylag[4][key][col] for key in shared)
        result[name + "_error_both"] = sum(not bylag[1][key][col] and not bylag[4][key][col] for key in shared)
    return result


BOUNDARIES = {"rho": [0, 1, 2, 4, 8, 16, 32, 64, 128],
    "history_angle_radians": [0, .01, .05, .1, .25, .5, 1, 2, float(np.pi)]}
MASK_NAMES = ("directed_arc", "coincident_static", "antipodal_fallback", "missing_or_zero_history", "last_zero_norm", "previous_zero_norm")


def empty_forecast():
    return dict(reference_exposures=0, candidate_weight=0, masks={}, candidate_weighted_masks={},
        distributions={key: dict(boundaries=edges, bins=[0] * (len(edges) + 1), count=0, total=0., minimum=None, maximum=None)
                       for key, edges in BOUNDARIES.items()})


def count_forecast(stats, key, diagnostics, nq):
    cell = stats.setdefault(key, empty_forecast())
    cell["reference_exposures"] += len(diagnostics["last_zero_norm"])
    cell["candidate_weight"] += nq * len(diagnostics["last_zero_norm"])
    for name in MASK_NAMES:
        count = sum(bool(value) for value in diagnostics[name])
        cell["masks"][name] = cell["masks"].get(name, 0) + count
        cell["candidate_weighted_masks"][name] = cell["candidate_weighted_masks"].get(name, 0) + nq * count
    valid = ~(diagnostics["missing_or_zero_history"] | diagnostics["last_zero_norm"])
    for name, diagname in (("rho", "extrapolation_ratio"), ("history_angle_radians", "history_angle_radians")):
        values = diagnostics[diagname][valid]
        dist = cell["distributions"][name]
        if not len(values):
            continue
        # Literal half-open-bin oracle, independent of producer searchsorted.
        for value in values:
            index = sum(value >= edge for edge in BOUNDARIES[name])
            dist["bins"][index] += 1
        dist["count"] += len(values)
        dist["total"] += float(values.sum())
        dist["minimum"] = float(values.min()) if dist["minimum"] is None else min(dist["minimum"], float(values.min()))
        dist["maximum"] = float(values.max()) if dist["maximum"] is None else max(dist["maximum"], float(values.max()))


def empty_triples():
    return dict(counts=Counter(), row_triplets=set(), GT_ID_triplets=set(), videos=set(), trajectories=set())


def count_triples(tables, key, mask, qids, rids, q, r, anchors, video):
    cell = tables.setdefault(key, empty_triples())
    reference_by_id = defaultdict(list)
    for rr, rgt in enumerate(rids):
        if rgt >= 0:
            reference_by_id[int(rgt)].append(rr)
    for qr, qgt in enumerate(qids):
        if qgt < 0:
            continue
        for rr in reference_by_id[int(qgt)]:
            rgt = rids[rr]
            if not mask[qr, rr]:
                continue
            cell["counts"]["endpoint_known_same_candidates"] += 1
            loc = r["previous_locators"][rr]
            previous = None if loc is None else anchors.get((loc["image_id"], loc["dump_row"]))
            check(previous is None or previous[0] == video, "Past eligible identity crosses video")
            category = "past_unknown" if previous is None else "past_known_same" if previous[1] == rgt else "past_known_different"
            cell["counts"][category] += 1
            if loc is not None:
                cell["counts"]["available_locator_endpoint_same_candidates"] += 1
                cell["row_triplets"].add((video, loc["image_id"], loc["dump_row"], r["image_id"], rr, q["image_id"], qr))
            if previous is not None:
                cell["GT_ID_triplets"].add((video, previous[1], int(rgt), int(qgt)))
                cell["videos"].add(video)
                cell["trajectories"].add((video, int(qgt)))


def final_triples(cell):
    result = {key: int(cell["counts"][key]) for key in ("endpoint_known_same_candidates", "past_unknown", "past_known_same",
        "past_known_different", "available_locator_endpoint_same_candidates")}
    result.update(distinct_immutable_row_triplets=len(cell["row_triplets"]), distinct_video_GT_ID_triplets=len(cell["GT_ID_triplets"]),
        fully_known_triplet_videos=len(cell["videos"]), fully_known_endpoint_GT_trajectories=len(cell["trajectories"]))
    return result


def report_results(db, cells, roles, history, gaps, forecast, triples):
    sources = sorted(roles)
    reports, union = {}, {}
    for source in sources:
        report = dict(role=roles[source]["role"], dataset_block=roles[source]["block"], lags={}, cross_lag={})
        for lag in LAGS:
            report["lags"][str(lag)] = {}
            for scope in SCOPES:
                report["lags"][str(lag)][scope] = {}
                for pop in POPULATIONS:
                    key = (source, lag, scope, pop)
                    cell = final_cell(cells[key])
                    cell["query_macros_and_error_concentration"] = query_metrics(db, source, lag, scope, pop)
                    cell["history_truth_support"] = final_triples(triples.get(key, empty_triples()))
                    report["lags"][str(lag)][scope][pop] = cell
            report["lags"][str(lag)]["reference_history_reason_counts"] = dict(history[source, lag])
            report["lags"][str(lag)]["original_frame_gap_per_frame_pair_counter"] = {str(k): v for k, v in sorted(gaps[source, lag].items())}
            report["lags"][str(lag)]["forecast_diagnostics"] = forecast.get((source, lag), empty_forecast())
        report["cross_lag"] = {scope: {pop: cross_lag(db, source, scope, pop) for pop in POPULATIONS} for scope in SCOPES}
        reports[source] = report
    for lag in LAGS:
        union[str(lag)] = {}
        for scope in SCOPES:
            union[str(lag)][scope] = {}
            for pop in POPULATIONS:
                combined = combine_cells([cells[source, lag, scope, pop] for source in sources])
                result = final_cell(combined)
                result.update(sources_with_known_pairs=sum(cells[s, lag, scope, pop]["known_pairs"] > 0 for s in sources),
                    sources_with_hard_queries=sum(cells[s, lag, scope, pop]["hard_queries"] > 0 for s in sources),
                    sources_with_raw_hard_errors=sum(cells[s, lag, scope, pop]["top1"][SUBSETS[1]][FEATURES[0]]["raw_wrong_feature_wrong"] > 0 for s in sources))
                result["query_macros_and_error_concentration"] = query_metrics(db, None, lag, scope, pop)
                triple = empty_triples()
                for source in sources:
                    value = triples.get((source, lag, scope, pop), empty_triples())
                    triple["counts"].update(value["counts"])
                    for key in ("row_triplets", "GT_ID_triplets", "videos", "trajectories"):
                        triple[key].update(value[key])
                result["history_truth_support"] = dict(final_triples(triple), fully_known_triplet_sources=sum(
                    bool(triples.get((s, lag, scope, pop), empty_triples())["GT_ID_triplets"]) for s in sources))
                union[str(lag)][scope][pop] = result
    return dict(micro_union=union, per_source=reports,
        cross_lag={scope: {pop: cross_lag(db, None, scope, pop) for pop in POPULATIONS} for scope in SCOPES},
        training_status="NO_TRAINING_OR_PARAMETER_SELECTION",
        history_truth_units="Candidates are endpoint-known-same QxM rows; distinct row triplets require sealed past locator; GT_ID triplets require all three eligible identities. Unknown stays unknown.")


def compare_databases(expected, actual):
    for db in (expected, actual):
        check(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQLite integrity check failed")
        check(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("query_support",)], "Unexpected SQLite tables")
    same(actual.execute("PRAGMA table_info(query_support)").fetchall(), expected.execute("PRAGMA table_info(query_support)").fetchall(), "Query table schema")
    rows = 0
    order = ",".join(str(i) for i in range(1, 13))
    for left, right in itertools.zip_longest(actual.execute("SELECT * FROM query_support ORDER BY " + order),
                                            expected.execute("SELECT * FROM query_support ORDER BY " + order)):
        check(left == right, "Independent query row differs")
        rows += 1
    return dict(query_support=rows)


def verify_baseline(results, cells, parent):
    keys = ("candidate_pairs", "known_pairs", "positive_pairs", "negative_pairs", "unknown_pairs", "nonempty_known_queries",
            "positive_available_queries", "negative_available_queries", "hard_queries")
    checked = 0
    for source in [None] + sorted(results["per_source"]):
        actual = results["micro_union"] if source is None else results["per_source"][source]["lags"]
        old = parent["results"]["micro_union"] if source is None else parent["results"]["per_source"][source]["lags"]
        for lag in LAGS:
            for scope in SCOPES:
                for pop in POPULATIONS:
                    a, b = actual[str(lag)][scope][pop], old[str(lag)][scope][pop]
                    same({key: a[key] for key in keys}, {key: b[key] for key in keys}, "Every raw population counter")
                    for subset in SUBSETS:
                        same(a["top1"][subset][FEATURES[0]], b["top1"][subset][FEATURES[0]], "Every raw top1 statistic")
                    same(a["positive_negative_pair_order"][FEATURES[0]], b["positive_negative_pair_order"][FEATURES[0]], "Every raw pair-order statistic")
                    cell = combine_cells([cells[s, lag, scope, pop] for s in sorted(results["per_source"])]) if source is None else cells[source, lag, scope, pop]
                    check(len(cell["_trajectories"]) == b["official_GT_trajectories"], "Raw trajectory support differs")
                    if source is not None:
                        check(len(cell["_identity_pairs"]) == b["distinct_ordered_identity_pairs"], "Raw distinct ordered identity pairs differ")
                        distribution = [{key: row[key] for key in ("video_id", "gt_track_id", "hard_queries", "raw_errors")}
                            for row in a["query_macros_and_error_concentration"][SUBSETS[1]]["query_GT_error_distribution"]]
                        same(distribution, b["query_GT_error_distribution"], "Every raw GT error distribution")
                    checked += 1
        if source is not None:
            for scope in SCOPES:
                for pop in POPULATIONS:
                    values = results["per_source"][source]["cross_lag"][scope][pop]
                    same({key: values[key] for key in parent["results"]["per_source"][source]["cross_lag"][scope][pop]},
                        parent["results"]["per_source"][source]["cross_lag"][scope][pop], "Every original cross-lag raw query intersection")
    primary = results["cross_lag"][SCOPES[1]][POPULATIONS[1]]["shared_hard_queries"]
    return dict(status="PASS_EXACT_PARENT_RAW_SCORES_COUNTS_TOP1", source_union_cells=checked, primary_shared_hard_queries=primary)


def inspect_scores(out, grouped, video_source, sources, anchors, db, bounds, parent_out, loader=read_raw):
    cells = {(s, lag, scope, pop): fresh_cell() for s in sources for lag in LAGS for scope in SCOPES for pop in POPULATIONS}
    history_counts, gap_counts = defaultdict(Counter), defaultdict(Counter)
    forecast, triples = {}, {}
    errors = np.zeros(2)
    candidates = Counter()
    maxblock = maxring = receipt_count = parityblocks = 0
    expected_files = {"frame_pairs.jsonl"}
    with (out / "frame_pairs.jsonl").open() as stream:
        for video, records in sorted(grouped.items()):
            source = video_source[str(video)]
            history, ring, offsets = {}, deque(maxlen=5), Counter()
            rowpath = out / "shards" / ("v%d.rows.json" % video)
            metadata = load(rowpath)
            same({k: metadata[k] for k in ("video_id", "source_group")}, dict(video_id=video, source_group=source), "Rowmap header")
            check(len(metadata["frames"]) == len(records), "Rowmap frame count")
            expected_files.add(str(rowpath.relative_to(out)))
            shards, parent_shards = {}, {}
            for lag in LAGS:
                relative = "shards/v%d_lag%d.scores.npy" % (video, lag)
                expected_files.add(relative)
                shards[lag] = np.load(out / relative, mmap_mode="r", allow_pickle=False)
                parent_shards[lag] = np.load(parent_out / relative, mmap_mode="r", allow_pickle=False)
                n = sum(records[i]["observations"] * records[i-lag]["observations"] for i in range(lag, len(records)))
                check(isinstance(shards[lag], np.memmap) and shards[lag].shape == (n, 2) and shards[lag].dtype == np.dtype("float64"), "Angular score shard dtype/shape")
                check(isinstance(parent_shards[lag], np.memmap) and parent_shards[lag].shape == (n, 6), "Parent score shard shape")
            for record in records:
                q = loader(record, history)
                ring.append(q)
                maxring = max(maxring, len(ring))
                step = record["frame_id"]
                same(metadata["frames"][step], independent_row_metadata(q), "Independent locator/raw-history row mapping")
                nq = len(q["boxes"])
                for lag in LAGS:
                    r = ring[-lag-1] if step >= lag else None
                    nr = 0 if r is None else len(r["boxes"])
                    count = nq * nr
                    if r is not None:
                        truth, diagnostics = independent_scores(q, r)
                        serialized = {key: value.tolist() for key, value in diagnostics.items()}
                    else:
                        serialized = None
                    expected = dict(video_id=video, source_group=source, lag=lag, query_step=step, query_image_id=q["image_id"],
                        reference_step=None if r is None else r["sampled_step"], reference_image_id=None if r is None else r["image_id"],
                        query_rows=nq, reference_rows=nr, original_frame_gap=None if r is None else q["original_frame_index"] - r["original_frame_index"],
                        offset=offsets[lag], candidates=count, score_file="shards/v%d_lag%d.scores.npy" % (video, lag), forecast=serialized)
                    line = next(stream, None)
                    check(line is not None, "Missing complete frame-pair receipt")
                    same(json.loads(line), expected, "Independent exact-lag receipt and all forecast masks/rho/theta")
                    receipt_count += 1
                    candidates[lag] += count
                    maxblock = max(maxblock, count)
                    if r is not None:
                        block = shards[lag][offsets[lag]:offsets[lag]+count].copy().reshape(nq, nr, 2)
                        np.testing.assert_allclose(block, truth, rtol=1e-12, atol=1e-12)
                        for col in range(2):
                            if count:
                                errors[col] = max(errors[col], float(np.max(np.abs(block[:, :, col] - truth[:, :, col]))))
                        parentraw = parent_shards[lag][offsets[lag]:offsets[lag]+count, 0].reshape(nq, nr)
                        check(np.array_equal(block[:, :, 0], parentraw), "Raw scores differ from sealed original baseline")
                        parityblocks += 1
                        history_counts[source, lag].update(r["history_reason"])
                        gap_counts[source, lag][expected["original_frame_gap"]] += 1
                        count_forecast(forecast, (source, lag), diagnostics, nq)
                        def ids(frame):
                            values = []
                            for row in range(len(frame["boxes"])):
                                value = anchors.get((frame["image_id"], row))
                                check(value is None or value[0] == video, "Manual identity crosses video")
                                values.append(-1 if value is None else value[1])
                            return np.asarray(values, np.int64)
                        qids, rids = ids(q), ids(r)
                        cv = np.isfinite(r["previous_times"])
                        categories = q["predicted_categories"][:, None] == r["predicted_categories"][None, :]
                        for scope, mask in zip(SCOPES, (np.ones((nq, nr), bool), categories)):
                            for pop, selected in zip(POPULATIONS, (mask, mask & cv[None, :])):
                                key = (source, lag, scope, pop)
                                count_block(cells[key], block, selected, qids, rids, source, lag, scope, pop, video, q["image_id"], db)
                                count_triples(triples, key, selected, qids, rids, q, r, anchors, video)
                        db.commit()
                    offsets[lag] += count
                check(maxring <= bounds["generation_ring_frames"] and maxblock <= bounds["max_candidates_per_frame_pair"], "Independent bounded block/ring violated")
            for lag in LAGS:
                check(offsets[lag] == len(shards[lag]), "Unverified score rows remain")
            del shards, parent_shards
            if len(expected_files) % 60 == 1:
                print(json.dumps(dict(phase="independent_angular_score_and_parent_recomputation", video_id=video, candidates=sum(candidates.values()))), flush=True)
        check(next(stream, None) is None, "Extra frame-pair receipt")
    checks = dict(files=sorted(expected_files), candidates_by_lag={str(k): candidates[k] for k in LAGS},
        frame_pair_receipts=receipt_count, maximum_candidate_block=maxblock, maximum_feature_ring_frames=maxring,
        exact_parent_raw_score_blocks=parityblocks, feature_maximum_absolute_error=dict(zip(FEATURES, errors.tolist())))
    return cells, history_counts, gap_counts, forecast, triples, checks


def audit(out):
    started = time.monotonic()
    target = out / "independent_output_audit.json"
    check(not target.exists(), "Preserve existing independent output audit")
    check(all((FINAL_PRODUCER_SHA, FINAL_PROTOCOL_SHA, FINAL_PREFLIGHT_SHA)), "Independent final SHA gates absent")
    for path, digest in ((PRODUCER, FINAL_PRODUCER_SHA), (PROTOCOL, FINAL_PROTOCOL_SHA), (PREFLIGHT, FINAL_PREFLIGHT_SHA),
        (Path(base.__file__), BASE_AUDITOR_SHA), (ROOT / "tmp_analysis/verify_fit_identity_edge_support_output.py", SCANNER_SHA)):
        check(sha(path) == digest, "Pinned independent input changed: " + str(path))
    execution = load(out / "producer_execution_receipt.json")
    check(execution["actual_tool_execution"]["exit_code"] == 0, "Producer actual exit0 absent")
    for relative, info in execution.get("evidence", {}).items():
        path = ROOT / relative
        check(sha(path) == info["sha256"] and path.stat().st_size == info["bytes"], "Producer execution evidence changed")
    protocol, preflight = load(PROTOCOL), load(PREFLIGHT)
    summary, seal = load(out / "summary.json"), load(out / "GT_free_candidates_sealed.json")
    check(preflight["status"] == "PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_IMPLEMENTATION_PREFLIGHT", "Independent preflight absent")
    check(preflight["script_sha256"] == FINAL_PRODUCER_SHA and preflight["protocol_sha256"] == FINAL_PROTOCOL_SHA, "Preflight final implementation mismatch")
    check(summary["status"] == "COMPLETE_ALL_SEALED_FIT_ANGULAR_CONTROL_PENDING_INDEPENDENT_AUDIT", "Producer not complete")
    check(summary["script_sha256"] == seal["script_sha256"] == FINAL_PRODUCER_SHA and summary["protocol_sha256"] == seal["protocol_sha256"] == FINAL_PROTOCOL_SHA, "Producer/protocol mismatch")
    check(seal["status"] == "SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION" and seal["supervised_payload_or_helper_opened"] is False and seal["unknowns_or_old_IDs_filtered"] is False, "Pre-supervision global seal missing")
    # Complete global candidate seal verification is deliberately before ID IO.
    for relative, digest in seal["files_sha256"].items():
        check(sha(out / relative) == digest, "Globally sealed angular candidate member changed")
    same(summary["generated"]["files_sha256"], seal["files_sha256"], "Seal/member hash report")
    for relative, digest in summary["output_files_sha256"].items():
        check(sha(out / relative) == digest, "Producer final output hash changed")
    for path, digest in summary["input_sha256"].items():
        check(sha(path) == digest, "Producer protected input hash changed")
    for path, digest in protocol["input_pins"].items():
        check(summary["input_sha256"][path] == digest, "Protocol input pin omitted")
    for path, digest in seal["input_sha256"].items():
        check(summary["input_sha256"][path] == digest, "Seal input pin omitted")
    for flag in ("training_started", "parameters_or_thresholds_selected", "GPU_used", "tune_test_or_classheldout_targets_decoded", "unknown_is_negative"):
        check(summary[flag] is False, "Forbidden producer scope flag: " + flag)
    check(summary["sealed_files_verified_before_supervised_import_and_after"] is True and summary["all_input_hashes_verified_before_after"] is True, "Required verification boundary absent")
    roles, video_source, frames, grouped, population = base.metadata_population(protocol)
    parent_summary, parent_seal = load(PARENT / "summary.json"), load(PARENT / "GT_free_candidates_sealed.json")
    for relative, digest in parent_seal["files_sha256"].items():
        check(sha(PARENT / relative) == digest, "Pinned original raw candidate member changed")
    watched = dict(summary["input_sha256"])
    watched.update({str(out / name): digest for name, digest in seal["files_sha256"].items()})
    watched.update({str(PARENT / name): digest for name, digest in parent_seal["files_sha256"].items()})
    for path in (Path(__file__).resolve(), out / "summary.json", out / "producer_execution_receipt.json", out / "GT_free_candidates_sealed.json", out / "support.sqlite", PARENT / "summary.json"):
        watched[str(path)] = sha(path)
    anchors, tune = legal_anchors(video_source)
    for frame in frames:
        check(sha(frame["npz"]) == frame["npz_sha256"] == summary["input_sha256"][frame["npz"]], "Selected raw cache hash changed")
    with tempfile.TemporaryDirectory(prefix="independent_angular_sqlite_", dir=out) as temporary:
        expected = database(Path(temporary) / "expected.sqlite")
        cells, history, gaps, forecast, triples, checks = inspect_scores(out, grouped, video_source, sorted(roles), anchors, expected, protocol["resource_bounds"], PARENT)
        results = report_results(expected, cells, roles, history, gaps, forecast, triples)
        same(summary["results"], results, "All source/lag/scope/population paired ranking, macro, concentration, history support")
        parity = verify_baseline(results, cells, parent_summary)
        parity["exact_full_score_blocks"] = checks["exact_parent_raw_score_blocks"]
        same(summary["baseline_parity"], parity, "Independent complete raw baseline parity")
        actual = sqlite3.connect((out / "support.sqlite").as_uri() + "?mode=ro", uri=True)
        counts = compare_databases(expected, actual)
        actual.close()
        expected.close()
    check(set(checks["files"]) == set(seal["files_sha256"]), "Global seal incomplete/extra files")
    for key in ("candidates_by_lag", "frame_pair_receipts", "maximum_candidate_block", "maximum_feature_ring_frames"):
        same(summary["generated"][key], checks[key], "Independent generation count/resource: " + key)
    check(summary["maximum_score_rows_read"] == checks["maximum_candidate_block"], "Producer read block bound differs")
    check(summary["eligible_FIT_manual_anchors"] == len(anchors), "Eligible FIT manual identity count differs")
    same(summary["population"], protocol["population"], "Fixed complete FIT population")
    omitted = sorted(set(map(int, video_source)) - set(grouped))
    check(protocol["population"]["uncollected_allocation_video_ids"] == omitted, "Uncollected allocated videos differ")
    same(summary["uncollected_allocation_videos"], [dict(video_id=video, source_group=video_source[str(video)], cached_frames=0,
        source_in_sealed_roles=video_source[str(video)] in roles) for video in omitted], "Uncollected allocations report")
    for path, info in load(RUN / "protected_before.json").items():
        check(Path(path).exists() == info["exists"], "Protected baseline existence changed")
        if info["exists"]:
            check(sha(path) == info["sha256"], "Protected baseline hash changed")
    for path, digest in watched.items():
        check(sha(path) == digest, "Input/output/verifier changed during independent audit")
    report = dict(status="PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_OUTPUT_AUDIT", auditor_sha256=sha(__file__),
        producer_script_sha256=FINAL_PRODUCER_SHA, protocol_sha256=FINAL_PROTOCOL_SHA, preflight_sha256=FINAL_PREFLIGHT_SHA,
        independent_base_auditor_sha256=BASE_AUDITOR_SHA, independent_scanner_sha256=SCANNER_SHA,
        summary_sha256=sha(out / "summary.json"), producer_execution_receipt_sha256=sha(out / "producer_execution_receipt.json"),
        support_SQLite_sha256=sha(out / "support.sqlite"), global_GT_free_seal_sha256=sha(out / "GT_free_candidates_sealed.json"),
        population=population, uncollected_allocated_FIT_video_ids=omitted, independently_recomputed=checks,
        independently_reconstructed_SQLite_table_rows=counts, eligible_FIT_manual_anchors=len(anchors), tune_rows_split_only_skipped=tune,
        all_source_scope_lag_and_identity_distributions_verified=True, all_original_raw_scores_counts_statistics_and_cross_lag_verified=True,
        forecast_masks_distributions_and_three_anchor_history_support_verified=True,
        all_inputs_outputs_and_protected_baseline_hashes_verified=True, producer_or_angular_core_or_original_supervised_decoder_imported=False,
        forbidden_correctness_values_decoded=False, unknown_is_negative=False, training=False, GPU=False,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, elapsed_seconds=time.monotonic()-started,
        execution_observation="Producer actual exit0 is bound to supervisor receipt; auditor execution receipt is supervisor-owned",
        interpretation="Ordinary fixed angular appearance control and descriptive support audit only; no training, novelty, assignment or dynamics-validity claim")
    write(target, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def synthetic_fixture():
    """Independent quantized cache fixtures, including empty/short sources."""
    rng = np.random.RandomState(77411)
    frames, records, anchors = {}, [], {}
    times = [0, 5, 9, 31, 50, 52, 90, 131, 145, 190]
    for video, counts in ((1, [2, 3, 0, 2, 2, 2, 3, 2, 2, 2]),
                          (2, [2, 2, 1, 3, 2, 2, 2, 2, 2, 2]), (3, [0, 0])):
        for step, n in enumerate(counts):
            raw = rng.randn(n, 256)
            if step % 4 == 0:
                raw[:] = 0
            elif step % 4 == 1:
                raw[:] = 0
                for row in range(n):
                    raw[row, row] = 1
            elif step % 4 == 2:
                raw[:] = 0
                for row in range(n):
                    raw[row, row] = -1
            else:
                raw[:] = 0
                for row in range(n):
                    raw[row, row] = .5
                    raw[row, (row + 1) % 256] = .866
            boxes = np.asarray([[10 + times[step] + row*21, 10 + video, 22 + times[step] + row*21, 30 + video]
                                for row in range(n)], float).reshape(n, 4)
            ids = np.asarray([0, 0, -1] if video == 1 and step == 1 else list(range(n)), np.int64)
            if n > 2 and step != 1:
                ids[-1] = -1
            categories = np.ones(n, np.int32)
            if n > 1 and step % 3 == 0:
                categories[-1] = 2
            image = video*1000 + step
            frame = dict(video_id=video, image_id=image, sampled_step=step, original_frame_index=times[step], boxes=boxes,
                raw=raw.astype(np.float16).astype(np.float64), released_ids=ids, predicted_categories=categories,
                semantic=rng.randn(n, 512).astype(np.float16).astype(np.float64))
            frames[video, step] = frame
            records.append(dict(video_id=video, frame_id=step, observations=n))
            for row in range(min(2, n)):
                if video == 2 and step == 5 and row == 1:
                    continue
                anchors[image, row] = (video, 9 if row == 0 and step == 8 else row + 1)
    return frames, records, anchors


def temporary_full_entry_test():
    """Only here import subject code, solely to construct temporary outputs."""
    import importlib.util
    def module(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    producer = module(PRODUCER, "only_synthetic_angular_fixture_producer")
    core = module(producer.CORE, "only_synthetic_angular_fixture_core")
    old_producer = module(ROOT / "tmp_analysis/audit_fit_lag_identity_support_20261024.py", "only_synthetic_parent_fixture_producer")
    geometry = module(old_producer.GEOMETRY, "only_synthetic_parent_fixture_geometry")
    keys = ("STAGE", "RUN", "SUPPORT", "PROTOCOL", "PREFLIGHT", "PARENT", "FINAL_PROTOCOL_SHA", "FINAL_PREFLIGHT_SHA",
            "EXPECTED_ELIGIBLE_IDENTITIES", "EXPECTED_TUNE_IDENTITIES")
    saved = {key: globals()[key] for key in keys}
    base_saved = {key: getattr(base, key) for key in ("STAGE", "RUN")}
    try:
        with tempfile.TemporaryDirectory(prefix="independent_angular_complete_entry_") as temporary:
            folder = Path(temporary)
            stage, out, parent = folder / "stage", folder / "output", folder / "parent"
            run, support, context = stage / "full_collect_v1", stage / "support_v1", stage / "pair_context_v1"
            for path in (run, support, context, out, parent):
                path.mkdir(parents=True)
            snapshots, records, anchors = synthetic_fixture()
            video_source = {"1": "S", "2": "T", "3": "ZERO", "4": "uncollected_no_role"}
            roles = {s: dict(split="fit", role="mean_role" if s == "S" else "pair_role", block="synthetic")
                     for s in ("S", "T", "ZERO", "UNUSED")}
            for record in records:
                frame = snapshots[record["video_id"], record["frame_id"]]
                path = run / ("v%d_f%d.npz" % (record["video_id"], record["frame_id"]))
                np.savez(path, video_id=frame["video_id"], image_id=frame["image_id"], frame_id=frame["sampled_step"],
                    frame_index=frame["original_frame_index"], det_bboxes=np.concatenate((frame["boxes"], np.ones((len(frame["boxes"]), 1))), axis=1),
                    track_feats=frame["raw"].astype(np.float16), cls_feats=frame["semantic"].astype(np.float16),
                    track_ids=frame["released_ids"], det_labels=frame["predicted_categories"]-1,
                    posterior_poison=np.asarray(["FORBIDDEN_CACHE_POSTERIOR"]))
                record.update(image_id=frame["image_id"], frame_index=frame["original_frame_index"], npz=str(path), npz_sha256=sha(path))
            write(context / "source_roles.json", dict(groups=roles))
            write(stage / "full_collection_plan.json", dict(video_records=[dict(video_id=int(v), source_group=s, split="fit") for v, s in video_source.items()]))
            (run / "frame_execution.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
            write(run / "protected_before.json", {})
            # Build manual payload with forbidden fields and a tune poison. The
            # full real legal_anchors function must skip these lexically.
            identities, indices = [], []
            frame_by_image = {f["image_id"]: f for f in snapshots.values()}
            for observation, ((image, row), (video, gt)) in enumerate(sorted(anchors.items())):
                frame = frame_by_image[image]
                identity = dict(split="fit", observation_id=observation, source_group=video_source[str(video)], video_id=video,
                    image_id=image, sampled_step=frame["sampled_step"], original_frame_index=frame["original_frame_index"],
                    local_track_id=int(frame["released_ids"][row]), gt_track_id=gt, manual_anchor_id=observation,
                    eligible_for_parameter_fit=True, pred_category_id=int(frame["predicted_categories"][row]))
                identities.append(dict(identity, gt_category_id="POISON_CLASS", error="POISON_ERROR", soft_risk="POISON_RISK", posterior_argmax="POISON_POSTERIOR"))
                indices.append(dict(identity, dump_row=row))
            manual, index = support / "supervised_observations.jsonl", support / "gt_free_observations.jsonl"
            manual.write_text("".join(json.dumps(row)+"\n" for row in identities)+json.dumps(dict(split="tune", gt_track_id="POISON_TUNE"))+"\n")
            index.write_text("".join(json.dumps(row)+"\n" for row in indices))
            write(support / "support_receipt.json", dict(output_files={manual.name: sha(manual)}))
            write(support / "gt_free_index_manifest.json", dict(files={index.name: sha(index)}))
            bounds = dict(max_frame_rows=3, max_video_frames=10, max_candidates_per_frame_pair=9, generation_ring_frames=5,
                          candidates_by_lag={"1": 69, "4": 46})
            population = dict(sealed_fit_source_groups=4, allocated_fit_videos=4, selected_fit_videos=3, selected_frames=len(records),
                retained_observations=sum(r["observations"] for r in records), uncollected_allocation_video_ids=[4])
            inputs = {str(p): sha(p) for p in (context / "source_roles.json", stage / "full_collection_plan.json", run / "frame_execution.jsonl")}
            protocol_path, preflight_path = folder / "protocol.json", folder / "preflight.json"
            write(protocol_path, dict(population=population, resource_bounds=bounds, input_pins=inputs))
            protocol_sha = sha(protocol_path)
            write(preflight_path, dict(status="PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_IMPLEMENTATION_PREFLIGHT",
                script_sha256=FINAL_PRODUCER_SHA, protocol_sha256=protocol_sha))
            inputs.update({str(protocol_path): protocol_sha, str(preflight_path): sha(preflight_path)})
            inputs.update({r["npz"]: r["npz_sha256"] for r in records})
            generated = producer.generate(out, records, video_source, sorted(roles), core, producer.load_frame, bounds)
            write(out / "GT_free_candidates_sealed.json", dict(status="SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION", **generated,
                script_sha256=FINAL_PRODUCER_SHA, protocol_sha256=protocol_sha, input_sha256=inputs,
                features=list(FEATURES), supervised_payload_or_helper_opened=False, unknowns_or_old_IDs_filtered=False))
            # A real six-column legacy producer supplies temp baseline scores
            # and legacy statistics independently of the new angular producer.
            pg = old_producer.generate(parent, records, video_source, sorted(roles), geometry, old_producer.load_frame, bounds)
            write(parent / "GT_free_candidates_sealed.json", dict(status="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION", **pg))
            pdb, pcells, ph, pga, _ = old_producer.score_sealed(parent, anchors, sorted(roles))
            parent_results = old_producer.summarize(parent, pdb, pcells, roles, ph, pga)
            pdb.commit(); pdb.close()
            write(parent / "summary.json", dict(results=parent_results))
            db, cells, h, g, maximum, forecasts, triplets, parityblocks = producer.score_sealed(out, anchors, sorted(roles), parent)
            results = producer.summarize(out, db, cells, roles, h, g, forecasts, triplets)
            db.commit(); db.close()
            parity = dict(status="PASS_EXACT_PARENT_RAW_SCORES_COUNTS_TOP1", source_union_cells=40,
                primary_shared_hard_queries=results["cross_lag"][SCOPES[1]][POPULATIONS[1]]["shared_hard_queries"], exact_full_score_blocks=parityblocks)
            summary = dict(status="COMPLETE_ALL_SEALED_FIT_ANGULAR_CONTROL_PENDING_INDEPENDENT_AUDIT", results=results,
                baseline_parity=parity, script_sha256=FINAL_PRODUCER_SHA, protocol_sha256=protocol_sha, input_sha256=inputs,
                all_input_hashes_verified_before_after=True, sealed_files_verified_before_supervised_import_and_after=True,
                generated=generated, maximum_score_rows_read=maximum, eligible_FIT_manual_anchors=len(anchors), population=population,
                uncollected_allocation_videos=[dict(video_id=4, source_group="uncollected_no_role", cached_frames=0, source_in_sealed_roles=False)],
                output_files_sha256={name: sha(out / name) for name in ("GT_free_candidates_sealed.json", "support.sqlite")},
                training_started=False, parameters_or_thresholds_selected=False, GPU_used=False,
                tune_test_or_classheldout_targets_decoded=False, unknown_is_negative=False)
            write(out / "summary.json", summary)
            write(out / "producer_execution_receipt.json", dict(actual_tool_execution=dict(exit_code=0,
                observed_by="Synthetic fixture only; no real producer process claim"), evidence={str(out / "summary.json"): dict(
                    sha256=sha(out / "summary.json"), bytes=(out / "summary.json").stat().st_size)}))
            globals().update(STAGE=stage, RUN=run, SUPPORT=support, PROTOCOL=protocol_path, PREFLIGHT=preflight_path, PARENT=parent,
                FINAL_PROTOCOL_SHA=protocol_sha, FINAL_PREFLIGHT_SHA=sha(preflight_path), EXPECTED_ELIGIBLE_IDENTITIES=len(anchors), EXPECTED_TUNE_IDENTITIES=1)
            base.STAGE, base.RUN = stage, run
            saved_loads = json.loads
            def guarded(text, *args, **kwargs):
                check("POISON_" not in text and "FORBIDDEN_CACHE_POSTERIOR" not in text, "Forbidden synthetic outcome payload decoded")
                return saved_loads(text, *args, **kwargs)
            json.loads = guarded
            try:
                receipt = audit(out)
            finally:
                json.loads = saved_loads
            check(receipt["status"] == "PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_OUTPUT_AUDIT", "Complete temporary real audit entry failed")
            check(receipt["independently_recomputed"]["candidates_by_lag"] == {"1": 69, "4": 46}, "Synthetic complete candidate counts")
            # Global seal corruption must fail before the identity function.
            saved_anchor_fn = globals()["legal_anchors"]
            def forbid_identity(*args, **kwargs):
                raise RuntimeError("Independent verifier touched identities before rejecting corrupt global seal")
            globals()["legal_anchors"] = forbid_identity
            (out / "independent_output_audit.json").unlink()
            tested = []
            try:
                for relative in ("shards/v1.rows.json", "shards/v1_lag4.scores.npy", "frame_pairs.jsonl"):
                    path = out / relative
                    original = path.read_bytes()
                    path.write_bytes(original+b"corrupt")
                    try:
                        audit(out)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("Corrupted global member accepted")
                    path.write_bytes(original)
                    tested.append(relative)
                path = out / "shards/v2_lag1.scores.npy"
                backup = out / "only_synthetic_backup.npy"
                path.rename(backup)
                try:
                    audit(out)
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError("Missing global member accepted")
                backup.rename(path)
            finally:
                globals()["legal_anchors"] = saved_anchor_fn
            return dict(candidates_by_lag={"1": 69, "4": 46}, SQLite_table_rows=receipt["independently_reconstructed_SQLite_table_rows"],
                original_parent_raw_statistics_verified=True, poisoned_payload_projection_verified=True,
                global_seal_tampered_types_rejected_before_identity=tested, missing_sealed_member_rejected_before_identity=True)
    finally:
        globals().update(saved)
        for key, value in base_saved.items():
            setattr(base, key, value)


def self_test():
    checks = math_self_test()
    integration = temporary_full_entry_test()
    result = dict(status="PASS_INDEPENDENT_ANGULAR_APPEARANCE_OUTPUT_AUDITOR_SYNTHETIC", verifier_sha256=sha(__file__),
        producer_sha256=FINAL_PRODUCER_SHA, protocol_sha256=FINAL_PROTOCOL_SHA, preflight_sha256=FINAL_PREFLIGHT_SHA,
        math_oracle_checks=checks, full_temporary_audit_entry=True, integration=integration,
        real_cache_or_supervision_opened=False, real_model_opened=False, GPU=False,
        synthetic_fixture_producer_and_core_used_only_to_create_subject_output=True,
        synthetic_authority_overrides=["temporary plan/roles/frame execution/float16 cache/protocol/preflight/parent outputs",
            "temporary eligible-identity population counts only; real safe scanner and immutable-row join run unmodified"])
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-out", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        if args.self_test_out:
            write(args.self_test_out, result)
    else:
        audit(args.out or ROOT / "results/angular_appearance_control_20261024_v1")


