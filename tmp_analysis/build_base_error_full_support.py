#!/usr/bin/env python3
"""Full fit/tune scope: seal all-emitted indices before natural-error support.

Requires a complete independently verified collection. Test outcomes are never
matched or scored. Uses all official GT for geometry assignment before any
base/fit-category eligibility filtering. No model import, GPU, or fitting.
Copied from the frozen pilot builder; scope and provenance guards are adapted
in this separate file. Real support requires explicit --execute after review.
--preflight reads only sealed metadata/audits, never parses supervision JSONs.
"""

import argparse
import ast
from collections import Counter, defaultdict
import copy
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLD = 0.5
OPEN_SPLITS = {"fit", "tune"}
PILOT_BUILDER = ROOT / "tmp_analysis/build_base_error_support.py"
PILOT_BUILDER_SHA256 = "64e702efa4e0956b19e9f9a0991ecfff8022c527e3e8bfb3a50710bbf4090186"
PARENT_PLAN_SHA256 = "f37f1ecc52734953ea5269407ab0aa8013e650e97bed4dde3cdd399a11e13ba5"


def adaptation_contract():
    return dict(parent_pilot_builder=str(PILOT_BUILDER), parent_pilot_builder_sha256=PILOT_BUILDER_SHA256,
                unchanged_semantics=["Class-blind max-cardinality then summed-IoU Hungarian matching over ALL official GT",
                                     "Inclusive IoU>=0.5; ambiguity excluded only after global assignment; unmatched stays unknown",
                                     "All-emitted GT-free observation/track seal precedes parsing supervision source JSONs",
                                     "Exact is_ori manual anchors, unique synset bridge and source/official bbox equality",
                                     "Only fit/tune and readout-fit target classes reach correctness/soft-risk outcomes",
                                     "Actual emitted det_labels+1 error is primary; posterior argmax separate",
                                     "Same predicted track AND same GT identity defines descriptive pair scope",
                                     "Oracle identity diagnostic is never a training mask/weight/feature or reference selector"],
                adapted_guards=["full_inference.json and separately sealed full_collection_plan.json replace pilot-only scope",
                                "Require independent full metadata and full runtime audit statuses plus dump verification",
                                "Full allocation has 450 videos, 448 nonempty videos, 383 source groups, 15283 frames",
                                "Full targets have explicit official old-vocabulary annotation/image/category bridges and tune eligibility",
                                "No test frames/dumps in this collection; all-emitted index still preserves every fit/tune prediction",
                                "Fresh output inside full stage; --execute gate; metadata-only --preflight mode",
                                "Per-image eligible anchor counts indexed once instead of scanning all anchors per frame"],
                reference_policy="This builder does not select references. Full all-emitted GT-free tracks remain the only admissible reference pool; GT correctness, identity, holdout and oracle purity never select references.",
                closed_outcome_boundary="Source annotation JSONs contain original training labels including readout-test/heldout labels; only selected fit/tune coordinates are mapped, and heldout correctness is never computed.")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def line(handle, value):
    handle.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")


def iou_matrix(predicted_xyxy, truth_xyxy):
    p = np.asarray(predicted_xyxy, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(truth_xyxy, dtype=np.float64).reshape(-1, 4)
    require(np.isfinite(p).all() and np.isfinite(g).all(), "Nonfinite geometry")
    lo = np.maximum(p[:, None, :2], g[None, :, :2])
    hi = np.minimum(p[:, None, 2:], g[None, :, 2:])
    intersection = np.maximum(0, hi - lo).prod(axis=2)
    pa = np.maximum(0, p[:, 2:] - p[:, :2]).prod(axis=1)
    ga = np.maximum(0, g[:, 2:] - g[:, :2]).prod(axis=1)
    union = pa[:, None] + ga[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def geometry_match(predicted_xyxy, truth_xyxy):
    """No labels accepted. Max cardinality above gate, then max summed IoU.

    All emitted predictions and all official GT participate, including
    predictions ambiguous against multiple GT. Ambiguity is excluded AFTER
    assignment so it cannot silently alter the class-blind matching pool.
    """
    overlaps = iou_matrix(predicted_xyxy, truth_xyxy)
    ambiguous = (overlaps >= IOU_THRESHOLD).sum(axis=1) > 1
    if not overlaps.shape[0] or not overlaps.shape[1]:
        return {}, ambiguous, overlaps
    valid = overlaps >= IOU_THRESHOLD
    bonus = max(overlaps.shape) + 1.0
    cost = np.where(valid, -(bonus + overlaps), 0.0)
    rows, columns = linear_sum_assignment(cost)
    matches = {int(i): int(j) for i, j in zip(rows, columns) if valid[i, j]}
    return matches, ambiguous, overlaps


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    require(w > 0 and h > 0, "Nonpositive official GT box")
    return [x, y, x + w, y + h]


def emitted_rows(data):
    tids = np.asarray(data["track_ids"], dtype=np.int64)
    mask = np.asarray(data["emitted_track_mask"], dtype=bool)
    require(np.array_equal(mask, tids >= 0), "Emitted mask does not equal local_id>=0")
    require(np.all(tids >= -1), "Unknown negative local-track sentinel")
    return np.flatnonzero(mask)


def track_key(video, local):
    require(int(local) >= 0, "Do not aggregate unassigned id=-1")
    return (int(video), int(local))


def canonical_frame_name(value):
    """Normalize harmless prefixes/separators, never discard split identity."""
    parts = str(value).replace("\\", "/").split("/")
    require(".." not in parts, "Parent traversal in frame filename")
    starts = [k for k, part in enumerate(parts) if part in ("train", "val", "test")]
    require(len(starts) == 1, "Frame filename needs one unambiguous dataset split")
    return "/".join(parts[starts[0]:])


def image_indices(rows, description):
    by_id, by_coordinate = {}, {}
    for image in rows:
        iid = int(image["id"])
        coordinate = (int(image["video_id"]), int(image["frame_index"]))
        require(iid not in by_id, "Duplicate %s image ID" % description)
        require(coordinate not in by_coordinate, "Duplicate %s (video_id,frame_index)" % description)
        canonical_frame_name(image["file_name"])
        by_id[iid] = image; by_coordinate[coordinate] = image
    return by_id, by_coordinate


def require_same_frame(left, right, description):
    require((int(left["video_id"]), int(left["frame_index"])) ==
            (int(right["video_id"]), int(right["frame_index"])), description+" coordinate mismatch")
    require(canonical_frame_name(left["file_name"]) == canonical_frame_name(right["file_name"]),
            description+" canonical filename mismatch")
    for field in ("width", "height"):
        if field in left and field in right:
            require(int(left[field]) == int(right[field]), description+" image dimensions mismatch")


def fixed_lag_bin(lag, kind):
    if kind == "sample_step":
        return "1" if lag == 1 else "2-3" if lag <= 3 else "4-7" if lag <= 7 else "8+"
    return "1-30" if lag <= 30 else "31-60" if lag <= 60 else "61-150" if lag <= 150 else "151+"


class ObservationStats:
    def __init__(self):
        self.n = self.correct = self.argmax_correct = self.label_argmax_differences = 0
        self.risks = []
        self.groups, self.videos, self.gt_tracks, self.pred_tracks, self.categories = (set() for _ in range(5))

    def add(self, row):
        self.n += 1
        self.correct += int(not row["error"])
        self.argmax_correct += int(row["posterior_argmax_category_id"] == row["gt_category_id"])
        self.label_argmax_differences += int(row["pred_category_id"] != row["posterior_argmax_category_id"])
        self.risks.append(row["soft_risk"])
        self.groups.add(row["source_group"]); self.videos.add(row["video_id"])
        self.gt_tracks.add((row["video_id"], row["gt_track_id"]))
        self.pred_tracks.add(track_key(row["video_id"], row["local_track_id"]))
        self.categories.add(row["gt_category_id"])

    def result(self):
        risks = np.asarray(self.risks, np.float64)
        distribution = {"mean": float(risks.mean()), "min": float(risks.min()), "max": float(risks.max()),
                        "q10": float(np.quantile(risks, .1)), "median": float(np.median(risks)),
                        "q90": float(np.quantile(risks, .9))} if self.n else None
        return dict(observations=self.n, correct=self.correct, errors=self.n-self.correct,
                    error_rate=(self.n-self.correct)/self.n if self.n else None,
                    posterior_argmax_correct=self.argmax_correct,
                    emitted_label_argmax_disagreements=self.label_argmax_differences,
                    soft_risk=distribution, source_groups=len(self.groups), videos=len(self.videos),
                    gt_trajectories=len(self.gt_tracks), predicted_tracks=len(self.pred_tracks),
                    target_categories=len(self.categories), target_category_ids=sorted(self.categories))


class PairStats:
    def __init__(self):
        self.counts = Counter()
        self.groups, self.videos, self.gt_tracks, self.pred_tracks, self.categories = (set() for _ in range(5))
        self.soft_product = self.iou_sum = 0.0
        self.by_source = defaultdict(Counter)
        self.by_trajectory = defaultdict(Counter)

    def add(self, a, b):
        require(a["image_id"] != b["image_id"], "Pairs must use distinct frames")
        require((a["split"], a["video_id"], a["local_track_id"], a["gt_track_id"], a["gt_category_id"])
                == (b["split"], b["video_id"], b["local_track_id"], b["gt_track_id"], b["gt_category_id"]),
                "Invalid same-identity semantic pair")
        agree = a["pred_category_id"] == b["pred_category_id"]
        both = bool(a["error"] and b["error"])
        self.counts["pairs"] += 1
        self.counts["agreement_pairs"] += int(agree)
        self.counts["both_correct"] += int(not a["error"] and not b["error"])
        self.counts["exactly_one_wrong"] += int(bool(a["error"]) != bool(b["error"]))
        self.counts["both_wrong"] += int(both)
        self.counts["same_wrong_class"] += int(agree and both)
        self.counts["different_wrong_class"] += int(not agree and both)
        self.soft_product += a["soft_risk"] * b["soft_risk"]
        self.iou_sum += min(a["match_iou"], b["match_iou"])
        self.groups.add(a["source_group"]); self.videos.add(a["video_id"])
        self.gt_tracks.add((a["video_id"], a["gt_track_id"]))
        self.pred_tracks.add(track_key(a["video_id"], a["local_track_id"]))
        self.categories.add(a["gt_category_id"])
        for counter in (self.by_source[a["source_group"]], self.by_trajectory[(a["video_id"], a["gt_track_id"])]):
            counter["pairs"] += 1
            counter["both_wrong"] += int(both)
            counter["same_wrong_class"] += int(agree and both)

    def result(self):
        n = self.counts["pairs"]
        counts = {k: int(self.counts[k]) for k in ("pairs", "agreement_pairs", "both_correct", "exactly_one_wrong", "both_wrong", "same_wrong_class", "different_wrong_class")}
        concentration = {}
        for event in ("pairs", "both_wrong", "same_wrong_class"):
            total = counts[event]
            concentration["max_source_"+event+"_share"] = max((v[event] for v in self.by_source.values()), default=0)/total if total else None
            concentration["max_gt_trajectory_"+event+"_share"] = max((v[event] for v in self.by_trajectory.values()), default=0)/total if total else None
        return dict(**counts, both_wrong_rate=counts["both_wrong"]/n if n else None,
                    same_wrong_class_rate=counts["same_wrong_class"]/n if n else None,
                    mean_endpoint_soft_risk_product=self.soft_product/n if n else None,
                    source_groups=len(self.groups), videos=len(self.videos), gt_trajectories=len(self.gt_tracks),
                    predicted_tracks=len(self.pred_tracks), target_categories=len(self.categories),
                    target_category_ids=sorted(self.categories), concentration=concentration,
                    per_source_event_counts={group: {event: counter[event] for event in ("pairs", "both_wrong", "same_wrong_class")}
                                             for group, counter in sorted(self.by_source.items())})


def full_scope_contract(preparation, plan, full_plan, manifest, metadata, metadata_audit, stage):
    """Pure full-scope metadata contract; no supervision annotations accepted."""
    require(preparation["collection_scope"] == "FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES", "Wrong full-stage scope")
    require(preparation["plan_sha256"] == full_plan["parent_plan_sha256"] == PARENT_PLAN_SHA256, "Parent plan lineage changed")
    require(full_plan["status"] == "SEALED_FULL_FIT_TUNE_METADATA_ONLY_NO_COLLECTION_RUN", "Unsealed full plan")
    require(full_plan["source_group_splits_inherited"] is True and full_plan["category_splits_inherited"] is True, "Full splits not inherited")
    require(full_plan["test_video_ids_selected"] == [] and full_plan["test_gt_frame_maps_built"] is False and full_plan["test_prediction_gt_joins"] is False, "Full plan opens test")
    require(Path(manifest["annotation"]).resolve() == stage / "full_inference.json", "Requires full input, not pilot/smoke")
    require(metadata["annotations"] == [] and metadata["tracks"] == [], "Inference contains GT")
    require([c["id"] for c in metadata["categories"]] == list(range(1, 1204)), "Wrong canonical full vocabulary")
    require(full_plan["readout_fit_category_ids"] == plan["vocabulary"]["readout_fit_category_ids"] and
            full_plan["readout_category_holdout_ids"] == plan["vocabulary"]["readout_category_holdout_ids"], "Readout class gates changed")
    fit_classes, holdout = set(full_plan["readout_fit_category_ids"]), set(full_plan["readout_category_holdout_ids"])
    require(len(fit_classes) == 146 and len(holdout) == 36 and not fit_classes.intersection(holdout), "Invalid class partition")
    records = {int(v["video_id"]): v for v in full_plan["video_records"]}
    require(len(records) == len(full_plan["video_records"]) == full_plan["allocation_videos"] == 450, "Allocation changed")
    parent_records = {int(v["video_id"]): v for v in plan["full_metadata_split"]["videos"]}
    require(records == {v: r for v, r in parent_records.items() if r["split"] in OPEN_SPLITS}, "Full allocation differs from sealed parent")
    require(all(r["split"] in OPEN_SPLITS for r in records.values()), "Test video selected")
    group_splits = defaultdict(set)
    for record in records.values():
        group_splits[record["source_group"]].add(record["split"])
    require(all(len(parts) == 1 for parts in group_splits.values()), "Source groups overlap splits")
    image_ids = [int(im["id"]) for im in metadata["images"]]
    require(len(image_ids) == len(set(image_ids)) == full_plan["total_images"] == preparation["images"] == 15283, "Full frame scope changed")
    require(full_plan["images"] == [dict(image_id=im["id"], **{k: v for k, v in im.items() if k != "id"}) for im in metadata["images"]], "Full frame plan differs")
    selected_videos = {int(im["video_id"]) for im in metadata["images"]}
    require(selected_videos <= set(records), "Full input contains unallocated video")
    require(len(selected_videos) == full_plan["selected_nonempty_videos"] == preparation["videos"] == 448, "Full video count changed")
    require(sorted(set(records)-selected_videos) == full_plan["omitted_zero_anchor_video_ids"] == [1720, 1974], "Empty-video exclusions changed")
    require({int(v["id"]) for v in metadata["videos"]} == selected_videos, "Inference video/image scope differs")
    require(len({records[v]["source_group"] for v in selected_videos}) == preparation["source_groups"] == 383, "Source group coverage changed")
    require(len(set(full_plan["manual_anchor_ctao_annotation_ids"])) == len(full_plan["manual_anchor_ctao_annotation_ids"]) == full_plan["total_manual_targets"] == preparation["manual_targets"] == 46908, "Full target IDs not unique")
    require(metadata_audit["status"] == "PASS_INDEPENDENT_FULL_COLLECTION_METADATA_AUDIT" and
            metadata_audit["images"] == 15283 and metadata_audit["videos"] == 448 and
            metadata_audit["no_test_images_or_supervised_targets_selected"] is True and
            metadata_audit["source_group_splits_disjoint"] is True, "Independent full metadata audit absent")
    require(manifest["expected_images"] == 15283 and manifest["expected_videos"] == 448, "Run scope differs")
    return records


def verified_inputs(stage, run, verify_dump_bytes=True):
    require(sha(PILOT_BUILDER) == PILOT_BUILDER_SHA256, "Frozen pilot builder changed")
    require(run.parent == stage, "Full run must be inside its stage")
    preparation = load(stage / "preparation_manifest.json")
    require(preparation["status"] == "PREPARED_NO_INFERENCE_RUN", "Unsealed preparation")
    for name, value in preparation["files"].items():
        require(sha(stage / name) == value, "Prepared input changed: " + name)
    plan_path = Path(preparation["plan"])
    require(sha(plan_path) == preparation["plan_sha256"], "Metadata plan changed")
    plan = load(plan_path)
    full_plan_path = stage / "full_collection_plan.json"
    require(Path(preparation["full_collection_plan"]).resolve() == full_plan_path and
            sha(full_plan_path) == preparation["full_collection_plan_sha256"], "Full plan pointer/hash changed")
    full_plan = load(full_plan_path)
    manifest = load(run / "run_manifest.json")
    annotation = Path(manifest["annotation"]).resolve()
    require(annotation == stage / "full_inference.json", "Support requires full scope, not smoke/pilot")
    require(manifest["annotation_sha256"] == sha(annotation) == preparation["files"][annotation.name], "Inference annotation hash mismatch")
    metadata = load(annotation)
    metadata_audit = load(stage / "independent_metadata_audit.json")
    require(metadata_audit["preparation_manifest_sha256"] == sha(stage / "preparation_manifest.json") and
            metadata_audit["annotation_sha256"] == manifest["annotation_sha256"], "Independent metadata audit binding mismatch")
    full_scope_contract(preparation, plan, full_plan, manifest, metadata, metadata_audit, stage)
    receipt = load(run / "collection_receipt.json")
    require(receipt["status"] == "COMPLETE_GT_FREE_COLLECTION_PENDING_INDEPENDENT_AUDIT", "Collection not complete")
    for key, filename in (("run_manifest_sha256", "run_manifest.json"), ("dump_inventory_sha256", "dump_inventory.json"),
                          ("events_sha256", "frame_execution.jsonl")):
        require(receipt[key] == sha(run / filename), "Collection receipt binding mismatch: " + filename)
    require(receipt["protected_sources_unchanged"] is True and receipt["source_code_unchanged"] is True and
            receipt["training_started"] is False, "Collection protected source/training failure")
    require(manifest["model_training"] is False and manifest["new_parameter_fit"] is False and
            manifest["supervision_file_opened"] is False, "Collection was not GT-free frozen inference")
    dump_check = load(run / "dump_verification.json")
    require(dump_check["status"] == "PASS_BASE_ERROR_DUMP", "Full dump verification has not passed")
    require(dump_check["annotation_sha256"] == manifest["annotation_sha256"] and not dump_check["missing_images"], "Wrong or partial dump verification")
    independent = load(run / "independent_collection_audit.json")
    require(independent["status"] == "PASS_INDEPENDENT_FULL_GT_FREE_COLLECTION_AUDIT", "Independent full runtime audit absent")
    require(independent["annotation_sha256"] == manifest["annotation_sha256"] and
            independent["run_manifest_sha256"] == sha(run / "run_manifest.json") and
            independent["collection_receipt_sha256"] == sha(run / "collection_receipt.json"), "Independent audit binding mismatch")
    require(independent["actual_emitted_outputs_exact"] is True and independent["all_npz_and_receipt_hashes_verified"] is True and
            independent["real_tracker_object_resets_verified"] == 448, "Full emitted outputs or tracker resets not independently verified")
    inventory = load(run / "dump_inventory.json")
    expected_names = {"v%d_i%d_f%d%s" % (im["video_id"], im["id"], im["frame_id"], suffix)
                      for im in metadata["images"] for suffix in (".npz", ".receipt.json")}
    require(set(inventory) == expected_names == {p.name for p in (run / "dump").iterdir()}, "Dump file inventory changed")
    if verify_dump_bytes:
        for name, value in inventory.items():
            require(sha(run / "dump" / name) == value, "Dump changed after audit: " + name)
    expected = {row["file"]: row["sha256"] for row in independent["frame_files"]}
    require(len(expected) == len(independent["frame_files"]) == 15283, "Duplicate/missing independent frame binding")
    require(expected == {name: value for name, value in inventory.items() if name.endswith(".npz")}, "Independent frame hash inventory mismatch")
    require({row["file"]: row["sha256"] for row in dump_check["dump_files"]} == expected and
            len(dump_check["dump_files"]) == 15283, "Dump verifier frame inventory mismatch")
    require(len(metadata["images"]) == receipt["completed_images"] == dump_check["frames"] == independent["frames"], "Frame scope mismatch")
    require(receipt["completed_videos"] == independent["videos"] == 448, "Video scope mismatch")
    require(receipt["emitted_observations"] == dump_check["emitted_track_observations"] == independent["emitted_observations"] == 681602,
            "All-emitted count differs between audits")
    return preparation, plan, full_plan, manifest, metadata, receipt, inventory


def seal_gt_free_index(out, run, metadata, records, inventory):
    """This function never opens or accepts a supervision annotation file."""
    tracks = defaultdict(list)
    frames = {}
    structure = defaultdict(Counter)
    next_observation = 0
    vocab = None
    with (out / "gt_free_observations.jsonl").open("x") as observations:
        for image in metadata["images"]:
            iid, vid, step = int(image["id"]), int(image["video_id"]), int(image["frame_id"])
            record = records[vid]; split = record["split"]
            path = run / "dump" / ("v%d_i%d_f%d.npz" % (vid, iid, step))
            with np.load(path, allow_pickle=False) as saved:
                tids = saved["track_ids"].astype(np.int64)
                rows = emitted_rows(saved)
                ids = saved["posterior_category_ids"]
                require(int(saved["image_id"]) == iid and int(saved["video_id"]) == vid and int(saved["frame_id"]) == step, "Dump/frame mapping mismatch")
                require(int(saved["frame_index"]) == image["frame_index"], "Original frame index changed")
                require(np.array_equal(ids, np.r_[np.arange(1, 1204), -1]), "Unsupported vocabulary regime")
                if vocab is None:
                    vocab = ids.copy()
                else:
                    require(np.array_equal(vocab, ids), "Vocabulary changed between frames")
                live_counts = Counter(map(int, tids[rows]))
                structure[split]["frames"] += 1
                structure[split]["retained_rows"] += len(tids)
                structure[split]["unassigned_rows"] += int((tids < 0).sum())
                structure[split]["emitted_rows"] += len(rows)
                structure[split]["zero_emitted_frames"] += int(len(rows) == 0)
                structure[split]["duplicate_local_id_frame_extra_rows"] += sum(n-1 for n in live_counts.values())
                row_ids = {}
                for row in rows:
                    tid = int(tids[row]); key = track_key(vid, tid)
                    item = dict(observation_id=next_observation, video_id=vid, local_track_id=tid,
                                image_id=iid, sampled_step=step, original_frame_index=int(image["frame_index"]),
                                split=split, source_group=record["source_group"],
                                dump_file=path.name, dump_sha256=inventory[path.name], dump_row=int(row),
                                pred_category_id=int(saved["det_labels"][row])+1,
                                score=float(saved["det_bboxes"][row, -1]))
                    line(observations, item)
                    tracks[key].append(next_observation); row_ids[int(row)] = next_observation
                    next_observation += 1
                frames[iid] = dict(path=path, row_ids=row_ids, split=split, image=image)
    with (out / "gt_free_tracks.jsonl").open("x") as handle:
        for (vid, tid), observation_ids in sorted(tracks.items()):
            line(handle, dict(video_id=vid, local_track_id=tid, split=records[vid]["split"],
                              source_group=records[vid]["source_group"], n_observations=len(observation_ids),
                              observation_ids=observation_ids))
    sealed = dict(status="SEALED_GT_FREE_ALL_EMITTED_INDEX_BEFORE_SUPERVISION_OPEN",
                  sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  aggregation_key=["video_id", "local_track_id"], observations=next_observation, tracks=len(tracks),
                  negative_local_ids_aggregated=False, gt_filtering=False, supervision_opened=False,
                  posterior_category_ids=vocab.tolist(),
                  split_structure={k: dict(v) for k, v in sorted(structure.items())},
                  files={name: sha(out / name) for name in ("gt_free_observations.jsonl", "gt_free_tracks.jsonl")},
                  readout_rule="Every emitted observation belongs to this immutable index; later supervision subsets must never replace this index for readout.")
    write(out / "gt_free_index_manifest.json", sealed)
    return frames, tracks, sealed


def validate_target_provenance(plan, metadata, records, truth, targets, source, full_plan):
    """Pure annotation-contract check: no predictions or test outcomes accepted.

    Source JSONs contain the original training benchmark's complete annotations;
    only selected fit/tune coordinates enter GT maps or manual category/box
    joins. Closed-label sentinels elsewhere in the source must remain unused.
    """
    require(targets["status"] == "TARGETS_ONLY_NOT_MODEL_INPUT", "Invalid target provenance")
    fit_categories = set(targets["readout_fit_category_ids"])
    require(fit_categories == set(plan["vocabulary"]["readout_fit_category_ids"]), "Fit-class split changed")
    holdout = set(targets["readout_category_holdout_ids"])
    require(holdout == set(plan["vocabulary"]["readout_category_holdout_ids"]) and
            not fit_categories.intersection(holdout), "Holdout split changed")
    require(fit_categories == set(full_plan["readout_fit_category_ids"]) and
            holdout == set(full_plan["readout_category_holdout_ids"]), "Full-plan category partition changed")
    images, _ = image_indices(metadata["images"], "full inference")
    official_images, official_coordinates = image_indices(truth["images"], "official TAO")
    source_images, _ = image_indices(source["images"], "C-TAO")
    planned_images, _ = image_indices([dict(row, id=row["image_id"]) for row in full_plan["images"]], "planned full inference")
    require(set(images) == set(planned_images), "Full image IDs differ from sealed plan")
    planned_target_ids = set(full_plan["manual_anchor_ctao_annotation_ids"])
    require(len(planned_target_ids) == len(full_plan["manual_anchor_ctao_annotation_ids"]), "Duplicate planned manual ID")
    target_ids = [int(a["id"]) for a in targets["annotations"]]
    require(len(target_ids) == len(set(target_ids)) and set(target_ids) == planned_target_ids,
            "Manual target IDs do not uniquely equal sealed C-TAO annotation IDs")
    source_targets = {}
    for annotation in source["annotations"]:
        aid = int(annotation["id"])
        if aid in planned_target_ids:
            require(aid not in source_targets, "Duplicate selected C-TAO annotation ID")
            source_targets[aid] = annotation
    require(set(source_targets) == planned_target_ids, "Planned manual ID absent from C-TAO source")
    open_coordinates = {}
    for iid, image in images.items():
        vid = int(image["video_id"])
        require(vid in records and records[vid]["split"] in OPEN_SPLITS, "Full inference contains test/unknown split")
        require(iid in source_images, "Full image ID absent from C-TAO")
        original = source_images[iid]
        require_same_frame(image, planned_images[iid], "Full/plan")
        require_same_frame(image, original, "Full/C-TAO")
        require(original.get("is_ori") is True, "Manual target frame is not C-TAO is_ori=True")
        require(int(image["source_frame_id"]) == int(original["frame_id"]), "Original C-TAO frame_id provenance mismatch")
        coordinate = (vid, int(image["frame_index"]))
        require(coordinate in official_coordinates, "Manual frame lacks official TAO coordinate")
        require_same_frame(image, official_coordinates[coordinate], "Full/official TAO")
        open_coordinates[coordinate] = iid
    canonical = {c["id"]: c for c in targets["categories"]}
    source_categories = {c["id"]: c for c in source["categories"]}
    require(len(canonical) == len(targets["categories"]) and set(canonical) == set(source_categories), "Canonical category IDs changed")
    for cid, category in canonical.items():
        require(category.get("synset") == source_categories[cid].get("synset") and
                category.get("frequency") == source_categories[cid].get("frequency"), "C-TAO category metadata bridge changed")
    synsets = defaultdict(list)
    for category in canonical.values():
        if category.get("synset"):
            synsets[category["synset"]].append(category["id"])
    old_categories = {c["id"]: c for c in truth["categories"]}
    require(len(old_categories) == len(truth["categories"]), "Duplicate official TAO category ID")
    category_mapping = {cid: synsets.get(c.get("synset"), []) if c.get("synset") else [] for cid, c in old_categories.items()}
    by_image = defaultdict(list)
    official_anchor_keys = set()
    for annotation in truth["annotations"]:
        # No predicted category, posterior, or test outcome is inspected here.
        image = official_images[annotation["image_id"]]
        coordinate = (image["video_id"], image["frame_index"])
        if coordinate not in open_coordinates:
            continue
        stage_id = open_coordinates[coordinate]
        require(int(annotation["video_id"]) == int(image["video_id"]), "Official annotation/image video mismatch")
        key = (stage_id, int(annotation["track_id"]))
        require(key not in official_anchor_keys, "Duplicate official identity/frame annotation")
        official_anchor_keys.add(key)
        mapped = category_mapping[annotation["category_id"]]
        by_image[stage_id].append(dict(official_annotation_id=annotation["id"], official_image_id=annotation["image_id"],
                                      gt_track_id=annotation["track_id"], old_category_id=annotation["category_id"],
                                      category_id=mapped[0] if len(mapped) == 1 else None,
                                      bbox=annotation["bbox"], xyxy=xywh_to_xyxy(annotation["bbox"]),
                                      iscrowd=bool(annotation.get("iscrowd", 0)), ignore=bool(annotation.get("ignore", 0))))
    manual = {}
    for anchor in targets["annotations"]:
        iid = int(anchor["image_id"])
        require(iid in images, "Target image outside selected full scope")
        image = images[iid]; split = records[image["video_id"]]["split"]
        require(anchor["readout_split"] == split, "Target split mismatch")
        require(split in OPEN_SPLITS, "Test/unknown target in full scope")
        aid = int(anchor["id"])
        original = source_targets[aid]
        for field in ("id", "image_id", "video_id", "track_id", "category_id"):
            require(int(anchor[field]) == int(original[field]), "Manual C-TAO source annotation field mismatch: "+field)
        require(np.allclose(anchor["bbox"], original["bbox"], rtol=0, atol=1e-6), "Manual/C-TAO source bbox mismatch")
        require(anchor["video_id"] == image["video_id"], "Manual video mismatch")
        require(anchor["supervision_source"] == "exact original-TAO manual anchor intersection", "Nonmanual target")
        key = (iid, int(anchor["track_id"]))
        require(key not in manual, "Duplicate manual track/frame anchor")
        eligible = anchor["category_id"] in fit_categories
        require(canonical[anchor["category_id"]]["frequency"] in ("f", "c"), "Nonbase manual target entered open supervision")
        require(anchor["readout_class_holdout"] == (not eligible), "Class holdout metadata mismatch")
        require(anchor["eligible_for_parameter_fit"] == (split == "fit" and eligible), "Parameter eligibility mismatch")
        require(anchor["eligible_for_tuning"] == (split == "tune" and eligible), "Tune eligibility mismatch")
        matching = [gt for gt in by_image[iid] if gt["gt_track_id"] == anchor["track_id"]]
        require(len(matching) == 1, "Manual anchor lacks unique full official GT identity")
        gt = matching[0]
        require(gt["category_id"] == anchor["category_id"] and np.allclose(gt["bbox"], anchor["bbox"], rtol=0, atol=1e-6), "Manual/official category or box mismatch")
        require((anchor["official_tao_annotation_id"], anchor["official_tao_image_id"], anchor["official_tao_category_id"]) ==
                (gt["official_annotation_id"], gt["official_image_id"], gt["old_category_id"]), "Explicit old-TAO provenance bridge changed")
        manual[key] = dict(anchor_id=int(anchor["id"]), category_id=int(anchor["category_id"]), eligible=eligible)
    require(not any(records[images[iid]["video_id"]]["split"] == "test" for iid in by_image), "Test GT was prepared for matching")
    provenance = dict(status="PASS_FIT_TUNE_MANUAL_SOURCE_CHAIN", open_frames=len(open_coordinates),
                      checked_manual_anchors=len(manual), official_coordinates_unique=True,
                      canonical_filenames_and_dimensions_equal=True, ctao_is_ori_true=True,
                      source_frame_ids_preserved=True, sealed_manual_annotation_ids_equal=True,
                      old_to_new_unique_synset_bridge=True, manual_source_and_official_boxes_equal_atol=1e-6,
                      explicit_old_tao_annotation_image_category_bridges=True, eligible_for_tuning_verified=True,
                      test_gt_frame_maps_built=False, test_outcomes_evaluated=False,
                      source_files_contain_test_partition_labels=True,
                      interpretation="Full source JSONs were parsed; test prediction/GT joins and outcomes remain closed.")
    return by_image, manual, fit_categories, provenance


def full_gt_and_manual(plan, stage, metadata, records, full_plan):
    """Only build official-GT frame maps for fit/tune. Test errors stay closed."""
    truth_path = Path(plan["inputs"]["tao_train"]["path"])
    source_path = Path(plan["inputs"]["ctao_base"]["path"])
    require(sha(truth_path) == plan["inputs"]["tao_train"]["sha256"], "Official TAO train changed")
    require(sha(source_path) == plan["inputs"]["ctao_base"]["sha256"], "C-TAO source changed")
    truth = load(truth_path)
    source = load(source_path)
    targets = load(stage / "manual_anchor_targets.json")
    return validate_target_provenance(plan, metadata, records, truth, targets, source, full_plan)


def score_fit_tune_support(out, metadata, records, frames, by_image, manual,
                           fit_categories, gt_free, index_manifest_sha):
    """Score only eligible rows; references are never selected here."""
    require(all(records[im["video_id"]]["split"] in OPEN_SPLITS for im in metadata["images"]),
            "Test/unknown frame reached full support scoring")
    counts = {part: Counter() for part in sorted(OPEN_SPLITS)}
    eligible_rows = []
    eligible_anchor_count = Counter(iid for (iid, _), anchor in manual.items() if anchor["eligible"])
    matched_identities = defaultdict(set)
    observation_stats = {part: ObservationStats() for part in sorted(OPEN_SPLITS)}
    strata = {part: defaultdict(ObservationStats) for part in sorted(OPEN_SPLITS)}
    with (out / "supervised_observations.jsonl").open("x") as rows_file, (out / "frame_matching_receipts.jsonl").open("x") as frame_file:
        for image in metadata["images"]:
            iid, vid = int(image["id"]), int(image["video_id"])
            record = records[vid]; split = record["split"]
            if split not in OPEN_SPLITS:
                continue  # no test dump/GT join and no test class correctness
            frame = frames[iid]; gt = by_image.get(iid, [])
            with np.load(frame["path"], allow_pickle=False) as saved:
                rows = emitted_rows(saved); boxes = saved["det_bboxes"][rows, :4]
                matches, ambiguous, overlaps = geometry_match(boxes, [x["xyxy"] for x in gt])
                frame_counts = Counter(frames=1, emitted_predictions=len(rows), full_official_gt=len(gt),
                                       official_gt_unmapped_category=sum(g["category_id"] is None for g in gt),
                                       geometry_matched=len(matches), geometry_unmatched_unknown=len(rows)-len(matches),
                                       ambiguous_multi_gt_predictions=int(ambiguous.sum()),
                                       official_gt_not_assigned=len(gt)-len(matches))
                seen_eligible = set()
                for localrow, rawrow in enumerate(rows):
                    if localrow not in matches:
                        continue
                    target = gt[matches[localrow]]
                    pred_track = track_key(vid, int(saved["track_ids"][rawrow]))
                    if ambiguous[localrow]:
                        frame_counts["ambiguous_matched_excluded"] += 1
                        continue
                    matched_identities[pred_track].add(int(target["gt_track_id"]))
                    if target["iscrowd"] or target["ignore"]:
                        frame_counts["crowd_or_ignore_match_excluded"] += 1
                        continue
                    anchor = manual.get((iid, int(target["gt_track_id"])))
                    if anchor is None:
                        frame_counts["nonmanual_intersection_match_excluded"] += 1
                        continue
                    if not anchor["eligible"]:
                        frame_counts["class_holdout_match_closed"] += 1
                        continue
                    cid = anchor["category_id"]
                    require(cid in fit_categories and target["category_id"] == cid, "Non-fit class reached outcomes")
                    ids = saved["posterior_category_ids"]
                    columns = np.flatnonzero(ids > 0)
                    truth_column = np.flatnonzero(ids == cid)
                    require(len(truth_column) == 1, "GT class absent or duplicated in vocabulary")
                    p = saved["p_fused"][rawrow, columns].astype(np.float64)
                    require(np.isfinite(p).all() and np.all(p >= 0) and p.sum() > 0, "Invalid foreground posterior")
                    p /= p.sum()
                    column_map = {int(ids[col]): j for j, col in enumerate(columns)}
                    pred = int(saved["det_labels"][rawrow]) + 1
                    require(pred in column_map, "Emitted prediction category outside posterior vocabulary")
                    item = dict(observation_id=frame["row_ids"][int(rawrow)], split=split,
                                source_group=record["source_group"], video_id=vid, image_id=iid,
                                sampled_step=int(image["frame_id"]), original_frame_index=int(image["frame_index"]),
                                local_track_id=pred_track[1], gt_track_id=int(target["gt_track_id"]),
                                gt_category_id=cid, pred_category_id=pred, error=int(pred != cid),
                                posterior_argmax_category_id=int(ids[columns[int(p.argmax())]]),
                                soft_risk=float(1-p[column_map[cid]]),
                                match_iou=float(overlaps[localrow, matches[localrow]]), manual_anchor_id=anchor["anchor_id"],
                                eligible_for_parameter_fit=split == "fit")
                    line(rows_file, item); eligible_rows.append(item)
                    observation_stats[split].add(item)
                    for name, value in (("source_group", record["source_group"]), ("target_category", str(cid))):
                        strata[split][name + "/" + value].add(item)
                    seen_eligible.add(anchor["anchor_id"])
                    frame_counts["eligible_fit_category_supervised"] += 1
                frame_counts["eligible_manual_anchors"] = eligible_anchor_count[iid]
                frame_counts["eligible_manual_anchors_without_supervision"] = frame_counts["eligible_manual_anchors"]-len(seen_eligible)
                counts[split].update(frame_counts)
                line(frame_file, dict(image_id=iid, video_id=vid, split=split, **dict(frame_counts)))
    groups = defaultdict(list)
    for row in eligible_rows:
        groups[(row["split"], row["video_id"], row["local_track_id"], row["gt_track_id"])].append(row)
    pair_stats = {part: defaultdict(PairStats) for part in sorted(OPEN_SPLITS)}
    pair_count = 0
    with (out / "supervised_pairs.jsonl").open("x") as pair_file:
        for (split, vid, tid, gt_id), observations in sorted(groups.items()):
            observations.sort(key=lambda x: (x["sampled_step"], x["observation_id"]))
            for a, b in itertools.combinations(observations, 2):
                if a["image_id"] == b["image_id"]:
                    counts[split]["same_frame_duplicate_track_pair_excluded"] += 1
                    continue
                step_lag = b["sampled_step"]-a["sampled_step"]
                raw_lag = b["original_frame_index"]-a["original_frame_index"]
                require(step_lag > 0 and raw_lag > 0, "Pair timeline not strictly increasing")
                observed_single = len(matched_identities[(vid, tid)]) == 1
                agree = a["pred_category_id"] == b["pred_category_id"]
                scopes = ["all_same_identity_subpairs"]
                if observed_single:
                    scopes.append("diagnostic_oracle_observed_single_matched_identity_tracks")
                for scope in scopes:
                    for suffix in ("all", "agreement/"+str(agree).lower(),
                                   "sample_step_lag/"+fixed_lag_bin(step_lag, "sample_step"),
                                   "original_frame_lag/"+fixed_lag_bin(raw_lag, "original_frame"),
                                   "step_lag_and_agreement/"+fixed_lag_bin(step_lag, "sample_step")+"/"+str(agree).lower()):
                        pair_stats[split][scope+"/"+suffix].add(a, b)
                line(pair_file, dict(split=split, video_id=vid, local_track_id=tid, gt_track_id=gt_id,
                                     observation_i=a["observation_id"], observation_j=b["observation_id"],
                                     sample_step_lag=step_lag, original_frame_lag=raw_lag,
                                     agreement=agree, both_wrong=bool(a["error"] and b["error"]),
                                     same_wrong_class=bool(agree and a["error"] and b["error"])))
                pair_count += 1
    require(sha(out / "gt_free_index_manifest.json") == index_manifest_sha, "GT-free seal changed")
    for name, digest in gt_free["files"].items():
        require(sha(out / name) == digest, "GT-free index was GT-filtered or mutated")
    by_split = {}
    for part in sorted(OPEN_SPLITS):
        predtracks = {track_key(r["video_id"], r["local_track_id"]) for r in eligible_rows if r["split"] == part}
        by_split[part] = dict(matching_counts=dict(counts[part]), observations=observation_stats[part].result(),
                              diagnostic_oracle_identity=dict(
                                  observed_multi_identity_predicted_tracks=sum(len(matched_identities[k]) > 1 for k in predtracks),
                                  observed_single_identity_predicted_tracks=sum(len(matched_identities[k]) == 1 for k in predtracks),
                                  labels_used="All unambiguous official matched identities, including heldout/nonmanual targets",
                                  training_or_tuning_use_permitted=False),
                              pair_scopes={name: stats.result() for name, stats in sorted(pair_stats[part].items())},
                              observation_strata={name: stats.result() for name, stats in sorted(strata[part].items())})
    return eligible_rows, pair_count, by_split


def build_support(stage, run, out):
    require(not out.exists(), "Refuse to overwrite existing support directory")
    require(out.parent == stage and out.name.startswith("support_"), "Use a fresh support_* directory inside this full stage")
    preparation, plan, full_plan, manifest, metadata, collection, inventory = verified_inputs(stage, run)
    input_paths = [stage / "preparation_manifest.json", stage / "manual_anchor_targets.json", stage / "full_collection_plan.json",
                   stage / "full_inference.json", stage / "independent_metadata_audit.json", run / "run_manifest.json",
                   run / "collection_receipt.json", run / "dump_inventory.json", run / "dump_verification.json",
                   run / "independent_collection_audit.json", Path(preparation["plan"]),
                   Path(plan["inputs"]["tao_train"]["path"]), Path(plan["inputs"]["ctao_base"]["path"]),
                   Path(__file__).resolve(), PILOT_BUILDER]
    input_snapshot = {str(p): sha(p) for p in input_paths}
    require(not out.exists(), "Refuse to overwrite existing support directory")
    out.mkdir(parents=True)
    records = {v["video_id"]: v for v in full_plan["video_records"]}
    frames, tracks, gt_free = seal_gt_free_index(out, run, metadata, records, inventory)
    require(gt_free["observations"] == collection["emitted_observations"], "All-emitted index count mismatch")
    require(set(gt_free["split_structure"]) == OPEN_SPLITS, "Closed split reached full GT-free index")
    index_manifest_sha = sha(out / "gt_free_index_manifest.json")
    print(json.dumps(dict(status="SEALED_GT_FREE_INDEX", observations=gt_free["observations"], tracks=gt_free["tracks"], sha256=index_manifest_sha)), flush=True)
    # Supervision is opened only below this seal, after BOTH collection audits.
    require(sha(out / "gt_free_index_manifest.json") == index_manifest_sha and
            gt_free["supervision_opened"] is False and gt_free["gt_filtering"] is False, "Missing pre-supervision all-emitted seal")
    by_image, manual, fit_categories, target_provenance = full_gt_and_manual(plan, stage, metadata, records, full_plan)
    eligible_rows, pair_count, by_split = score_fit_tune_support(
        out, metadata, records, frames, by_image, manual, fit_categories, gt_free, index_manifest_sha)
    output_files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    receipt = dict(status="PASS_FIT_TUNE_SIGNAL_SUPPORT_AUDIT", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   stage=str(stage), run=str(run), out=str(out), inputs=input_snapshot,
                   gt_free_index_manifest_sha256=index_manifest_sha, output_files=output_files,
                   collection_scope="FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES", scope_adaptation=adaptation_contract(),
                   supervision_seal_boundary="Annotation source bytes are SHA-hashed before sealing; supervision JSON payloads are parsed only after the all-emitted index seal.",
                   target_provenance=target_provenance,
                   supervised_observations=len(eligible_rows), supervised_pairs=pair_count,
                   fit_tune=by_split,
                   test=dict(status="CLOSED_NO_GT_MATCHING_NO_ERROR_OUTCOMES", **gt_free["split_structure"].get("test", {})),
                   method=dict(geometry="Hungarian: max valid IoU>=.5 cardinality, then total IoU; no predicted/GT category used",
                               matching_pool="All emitted predictions vs ALL official original-TAO GT including rare, holdout and unmapped categories",
                               ambiguity="Prediction with >=2 official GT at IoU>=.5 excluded after global assignment",
                               supervision="Exact manual-anchor intersection and fit target class; fit or tune videos only",
                               correctness="Primary emitted det_labels+1 vs GT; posterior argmax correctness separately recorded",
                               soft_risk="1-p_fused[GT] after foreground-only row normalization",
                               pairs="Same predicted (video,local_id) AND same official GT identity; distinct images; both endpoints independently eligible",
                               pair_weighting="Exhaustive pairs are descriptive diagnostics, not independent samples or training weights; report source/GT-trajectory concentration",
                               oracle_identity_diagnostic="Secondary oracle scope uses heldout/nonmanual identity state; never a training/selection mask, weight or feature; omitted from supervised pair rows",
                               readout="All emitted observations remain in sealed GT-free index; supervised subsets never replace readout indices",
                               lag="Ordinal sampled steps and preserved original frame indices separately; no elapsed seconds inferred"),
                   unmatched_is_unknown=True, class_holdout_outcomes_computed=False, test_outcomes_computed=False,
                   test_dump_files_opened=False, references_selected=False, reference_selection_uses_GT=False,
                   oracle_identity_diagnostics_used_for_training=False,
                   training_started=False, model_imported=False, gpu_used=False, training_readiness_established=False,
                   limitations=["Counts/pairs are descriptive support, not independent sample counts, covariance or efficacy evidence",
                                "Observed single matched identity does not establish whole-track purity",
                                "Sparse manual-anchor timeline and current full1203 vocabulary regime",
                                "Frozen detector may have seen base classes/videos; class holdout is readout pseudo-novel only"])
    # Recheck immutable inputs after scoring; no source/run/stage write occurs.
    for path, digest in receipt["inputs"].items():
        require(sha(path) == digest, "Input changed while building support: "+path)
    write(out / "support_receipt.json", receipt)
    summary = dict(status=receipt["status"], stage=str(stage), run=str(run),
                   support_receipt_sha256=sha(out / "support_receipt.json"),
                   gt_free_index_manifest_sha256=index_manifest_sha, supervised_observations=len(eligible_rows),
                   supervised_pairs=pair_count,
                   fit_tune={part: {"matching_counts": value["matching_counts"], "observations": value["observations"],
                                    "diagnostic_oracle_identity": value["diagnostic_oracle_identity"],
                                    "pair_scopes": {name: stats for name, stats in value["pair_scopes"].items() if name.endswith("/all")}}
                             for part, value in by_split.items()}, test=receipt["test"],
                   training_started=False, training_readiness_established=False,
                   oracle_identity_diagnostics_used_for_training=False,
                   raw_indices_and_supervision_rows="Local-only artifacts; do not Git-sync raw matched rows or indices",
                   interpretations=receipt["limitations"])
    write(out / "support_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return receipt


def synthetic_provenance_arguments():
    images = [dict(id=iid, video_id=vid, frame_index=frame, frame_id=step, source_frame_id=frame+1,
                   width=100, height=100, file_name="train/demo%d/frame%04d.jpg" % (vid, frame))
              for iid, vid, frame, step in ((501,7,10,0),(503,7,20,1),(504,9,30,0))]
    closed = dict(id=502, video_id=8, frame_index=40, frame_id=0, source_frame_id=41,
                  width=100, height=100, file_name="train/closed/frame0040.jpg")
    metadata = dict(images=copy.deepcopy(images))
    records = {7:dict(split="fit", source_group="synthetic_A"), 9:dict(split="tune", source_group="synthetic_B")}
    categories = [dict(id=3,synset="demo.n.01",frequency="f"),dict(id=4,synset="heldout.n.01",frequency="c")]
    anchors = [dict(id=700,image_id=501,video_id=7,track_id=9,category_id=3,bbox=[1.,2.,5.,6.]),
               dict(id=701,image_id=501,video_id=7,track_id=10,category_id=4,bbox=[11.,2.,5.,6.]),
               dict(id=702,image_id=503,video_id=7,track_id=9,category_id=3,bbox=[1.,2.,5.,6.]),
               dict(id=703,image_id=504,video_id=9,track_id=90,category_id=3,bbox=[1.,2.,5.,6.])]
    source = dict(categories=copy.deepcopy(categories),
                  images=[dict(im,frame_id=im["source_frame_id"],is_ori=True) for im in images+[closed]],
                  annotations=copy.deepcopy(anchors)+[dict(id=704,category_id="UNUSED_CLOSED_SENTINEL")])
    original_image_ids = {501:101,503:103,504:104,502:102}
    old_categories = {3:999,4:998}
    targets = dict(status="TARGETS_ONLY_NOT_MODEL_INPUT",categories=copy.deepcopy(categories),
                   readout_fit_category_ids=[3],readout_category_holdout_ids=[4],annotations=[])
    for a in anchors:
        split=records[a["video_id"]]["split"];eligible=a["category_id"]==3
        targets["annotations"].append(dict(a,readout_split=split,readout_class_holdout=not eligible,
                                           eligible_for_parameter_fit=split=="fit" and eligible,
                                           eligible_for_tuning=split=="tune" and eligible,
                                           supervision_source="exact original-TAO manual anchor intersection",
                                           official_tao_annotation_id=a["id"]-699,
                                           official_tao_image_id=original_image_ids[a["image_id"]],
                                           official_tao_category_id=old_categories[a["category_id"]]))
    full_plan = dict(readout_fit_category_ids=[3],readout_category_holdout_ids=[4],
                     images=[dict(image_id=im["id"],**{k:v for k,v in im.items() if k!="id"}) for im in images],
                     manual_anchor_ctao_annotation_ids=[a["id"] for a in anchors])
    plan = dict(vocabulary=dict(readout_fit_category_ids=[3],readout_category_holdout_ids=[4]))
    truth = dict(categories=[dict(id=999,synset="demo.n.01"),dict(id=998,synset="heldout.n.01"),dict(id=888,synset="unmapped.n.01")],
                 images=[dict(im,id=original_image_ids[im["id"]]) for im in images+[closed]],
                 annotations=[dict(a,id=a["id"]-699,image_id=original_image_ids[a["image_id"]],category_id=old_categories[a["category_id"]]) for a in anchors])
    truth["annotations"] += [dict(id=8,image_id=101,video_id=7,track_id=11,category_id=888,bbox=[21.,2.,5.,6.]),
                              dict(id=9,image_id=101,video_id=7,track_id=12,category_id=999,bbox=[31.,2.,5.,6.]),
                              dict(id=10,image_id=102,category_id="CLOSED_INVALID_SENTINEL")]
    return dict(plan=plan,metadata=metadata,records=records,truth=truth,targets=targets,source=source,full_plan=full_plan)


def provenance_self_test():
    """Full-scope provenance, explicit old IDs and test-label noninterference."""
    arguments=synthetic_provenance_arguments()
    result=validate_target_provenance(**arguments)
    require(set(result[0])=={501,503,504} and len(result[0][501])==4,"All official GT/test boundary failed")
    require(any(gt["category_id"] is None for gt in result[0][501]),"Unmapped official GT removed")
    require(set(result[1])=={(501,9),(501,10),(503,9),(504,90)},"Exact selected manual anchors changed")
    require(result[1][(501,10)]["eligible"] is False,"Heldout target became eligible")
    aliases=copy.deepcopy(arguments);aliases["truth"]["images"][0]["file_name"]="/data/tao/frames/train/demo7/frame0010.jpg"
    validate_target_provenance(**aliases)
    rejected={}
    def reject(name,mutate):
        altered=copy.deepcopy(arguments);mutate(altered)
        try:validate_target_provenance(**altered)
        except (AssertionError,KeyError,ValueError):rejected[name]="PASS_REJECTED"
        else:raise AssertionError("Accepted corrupt full provenance: "+name)
    reject("duplicate_official_coordinate",lambda a:a["truth"]["images"].append(dict(a["truth"]["images"][0],id=999)))
    reject("duplicate_official_image_id",lambda a:a["truth"]["images"].append(dict(a["truth"]["images"][0],frame_index=999)))
    reject("official_filename_mismatch",lambda a:a["truth"]["images"][0].update(file_name="train/demo/WRONG.jpg"))
    reject("official_split_prefix_mismatch",lambda a:a["truth"]["images"][0].update(file_name="val/demo7/frame0010.jpg"))
    reject("ctao_is_ori_false",lambda a:a["source"]["images"][0].update(is_ori=False))
    reject("ctao_source_frame_id_mismatch",lambda a:a["source"]["images"][0].update(frame_id=77))
    reject("ctao_filename_mismatch",lambda a:a["source"]["images"][0].update(file_name="train/demo/WRONG.jpg"))
    reject("manual_source_id_missing",lambda a:a["source"]["annotations"][0].update(id=999))
    reject("manual_source_bbox_changed",lambda a:a["source"]["annotations"][0].update(bbox=[2.,2.,5.,6.]))
    reject("manual_source_track_id_changed",lambda a:a["source"]["annotations"][0].update(track_id=999))
    reject("plan_manual_ids_changed",lambda a:a["full_plan"].update(manual_anchor_ctao_annotation_ids=[700]))
    reject("official_bbox_changed",lambda a:a["truth"]["annotations"][0].update(bbox=[2.,2.,5.,6.]))
    reject("official_category_bridge_changed",lambda a:a["truth"]["categories"][0].update(synset="unmapped.n.01"))
    reject("duplicate_official_identity_at_frame",lambda a:a["truth"]["annotations"].append(dict(a["truth"]["annotations"][0],id=999)))
    reject("explicit_old_annotation_id_changed",lambda a:a["targets"]["annotations"][0].update(official_tao_annotation_id=777))
    reject("explicit_old_image_id_changed",lambda a:a["targets"]["annotations"][0].update(official_tao_image_id=777))
    reject("explicit_old_category_id_changed",lambda a:a["targets"]["annotations"][0].update(official_tao_category_id=3))
    reject("tune_eligibility_changed",lambda a:a["targets"]["annotations"][3].update(eligible_for_tuning=False))
    reject("holdout_eligibility_changed",lambda a:a["targets"]["annotations"][1].update(eligible_for_parameter_fit=True))
    reject("test_record_selected",lambda a:a["records"][7].update(split="test"))
    reject("duplicated_canonical_synset",lambda a:a["targets"]["categories"][1].update(synset="demo.n.01"))
    return dict(status="PASS_SYNTHETIC_FULL_MANUAL_SOURCE_CHAIN",old_999_to_new_3_by_synset_not_numeric_id=True,
                unmapped_and_holdout_official_gt_retained=True,invalid_test_labels_not_evaluated=True,
                explicit_old_tao_annotation_image_category_bridges=True,eligible_for_tuning=True,rejected_mutations=rejected)


def integrated_synthetic_test():
    """Actual sealing/scoring functions on synthetic dumps, with poisoned holdout rows."""
    arguments=synthetic_provenance_arguments();metadata=arguments["metadata"];records=arguments["records"]
    by_image,manual,fit_categories,_=validate_target_provenance(**arguments)
    with tempfile.TemporaryDirectory(prefix="full_support_contract_") as temporary:
        root=Path(temporary);run=root/"run";dump=run/"dump";dump.mkdir(parents=True);out=root/"support";out.mkdir()
        inventory={}
        for image in metadata["images"]:
            iid,vid,step=image["id"],image["video_id"],image["frame_id"]
            # Frame 501: eligible, heldout, unmapped, nonmanual, unmatched, unassigned.
            boxes=([[1,2,6,8,.7],[11,2,16,8,.8],[21,2,26,8,.8],[31,2,36,8,.8],
                    [71,72,76,78,.6],[81,82,86,88,.5]] if iid==501 else [[1,2,6,8,.7]])
            tids=np.asarray([7,8,9,10,11,-1] if iid==501 else [7],np.int64)
            labels=np.asarray([7,3,2,2,2,2] if iid==501 else [2],np.int64)
            p=np.zeros((len(tids),1204),np.float64);p[:,2]=1
            if iid==501:
                p[0]=0;p[0,2]=.8;p[0,7]=.2
                # Heldout/unmapped/nonmanual/unmatched p rows must never be read for outcomes.
                p[1:]=np.nan
            name="v%d_i%d_f%d.npz"%(vid,iid,step);path=dump/name
            np.savez_compressed(path,track_ids=tids,emitted_track_mask=tids>=0,
                                posterior_category_ids=np.r_[np.arange(1,1204),-1],p_fused=p,
                                video_id=vid,image_id=iid,frame_id=step,frame_index=image["frame_index"],
                                det_bboxes=np.asarray(boxes),det_labels=labels)
            inventory[name]=sha(path)
        frames,tracks,sealed=seal_gt_free_index(out,run,metadata,records,inventory)
        require(sealed["observations"]==7 and sealed["tracks"]==6,"All-emitted index lost ineligible/unknown rows")
        seal_hash=sha(out/"gt_free_index_manifest.json")
        before={name:sha(out/name) for name in ("gt_free_index_manifest.json","gt_free_observations.jsonl","gt_free_tracks.jsonl")}
        observed,pair_count,summary=score_fit_tune_support(out,metadata,records,frames,by_image,manual,fit_categories,sealed,seal_hash)
        require(len(observed)==3 and pair_count==1,"Supervised eligibility/pair fixture differs")
        require(observed[0]["pred_category_id"]==8 and observed[0]["error"]==1 and
                observed[0]["posterior_argmax_category_id"]==3,"Posterior argmax silently replaced emitted error")
        require(sum(r["error"] for r in observed)==1 and all(r["gt_category_id"]==3 for r in observed),"Heldout correctness leaked")
        fit_counts=summary["fit"]["matching_counts"]
        require(fit_counts["class_holdout_match_closed"]==1 and fit_counts["geometry_unmatched_unknown"]==1 and
                fit_counts["nonmanual_intersection_match_excluded"]==2,"Full GT filtering/unknown counts differ")
        require({name:sha(out/name) for name in before}==before,"GT-free seal mutated after supervision")
        all_rows=[json.loads(line) for line in (out/"gt_free_observations.jsonl").read_text().splitlines()]
        require(len(all_rows)==7 and all("error" not in row and "gt_track_id" not in row for row in all_rows),"GT metadata in reference pool")
        require(all(r["observation_id"] in {o["observation_id"] for o in all_rows} for r in observed),"Supervision replaced readout index")
    return dict(status="PASS_SYNTHETIC_FULL_SEAL_AND_SUPERVISION",all_emitted=7,supervised=3,pairs=1,
                poisoned_holdout_unmapped_nonmanual_unmatched_posteriors_not_scored=True,
                emitted_error_differs_from_posterior_argmax_preserved=True,
                all_emitted_index_preserved_after_GT_layer=True,reference_selection_performed=False)


def full_scope_self_test():
    """Fixed full-scope size contract without opening any actual dataset."""
    stage=Path("/synthetic/full_stage")
    records=[]
    for vid in list(range(448))+[1720,1974]:
        split="fit" if vid<403 or vid in (1720,1974) else "tune"
        group=("fit_%d"%(vid%341)) if split=="fit" else ("tune_%d"%((vid-403)%42))
        records.append(dict(video_id=vid,split=split,source_group=group))
    classes=dict(readout_fit_category_ids=list(range(1,147)),readout_category_holdout_ids=list(range(147,183)))
    metadata=dict(images=[dict(id=i,video_id=i%448,frame_id=i//448,frame_index=i//448) for i in range(15283)],
                  videos=[dict(id=i) for i in range(448)],categories=[dict(id=i) for i in range(1,1204)],annotations=[],tracks=[])
    full_plan=dict(status="SEALED_FULL_FIT_TUNE_METADATA_ONLY_NO_COLLECTION_RUN",parent_plan_sha256=PARENT_PLAN_SHA256,
                   source_group_splits_inherited=True,category_splits_inherited=True,test_video_ids_selected=[],
                   test_gt_frame_maps_built=False,test_prediction_gt_joins=False,video_records=records,
                   allocation_videos=450,total_images=15283,selected_nonempty_videos=448,
                   omitted_zero_anchor_video_ids=[1720,1974],total_manual_targets=46908,
                   manual_anchor_ctao_annotation_ids=list(range(46908)),
                   images=[dict(image_id=im["id"],**{k:v for k,v in im.items() if k!="id"}) for im in metadata["images"]],**classes)
    arguments=dict(preparation=dict(collection_scope="FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES",plan_sha256=PARENT_PLAN_SHA256,
                                    images=15283,videos=448,source_groups=383,manual_targets=46908),
                   plan=dict(vocabulary=classes,full_metadata_split=dict(videos=copy.deepcopy(records))),full_plan=full_plan,
                   manifest=dict(annotation=str(stage/"full_inference.json"),expected_images=15283,expected_videos=448),
                   metadata=metadata,metadata_audit=dict(status="PASS_INDEPENDENT_FULL_COLLECTION_METADATA_AUDIT",images=15283,
                                                        videos=448,no_test_images_or_supervised_targets_selected=True,
                                                        source_group_splits_disjoint=True),stage=stage)
    full_scope_contract(**arguments)
    mutations={}
    def reject(name,mutate):
        changed=copy.deepcopy(arguments);mutate(changed)
        try:full_scope_contract(**changed)
        except (AssertionError,KeyError,ValueError):mutations[name]="PASS_REJECTED"
        else:raise AssertionError("Accepted wrong full metadata scope: "+name)
    reject("pilot_annotation",lambda a:a["manifest"].update(annotation=str(stage/"pilot_inference.json")))
    reject("smoke_annotation",lambda a:a["manifest"].update(annotation=str(stage/"smoke_inference.json")))
    reject("test_allocation",lambda a:a["full_plan"]["video_records"][0].update(split="test"))
    reject("duplicate_frame",lambda a:a["metadata"]["images"][1].update(id=0))
    reject("partial_frame_scope",lambda a:a["metadata"]["images"].pop())
    reject("GT_in_inference",lambda a:a["metadata"]["annotations"].append(dict(id=1)))
    reject("class_partition_changed",lambda a:a["full_plan"]["readout_fit_category_ids"].append(147))
    reject("metadata_audit_missing",lambda a:a["metadata_audit"].update(status="PENDING"))
    reject("new_source_split",lambda a:a["full_plan"]["video_records"][0].update(source_group="wrong"))
    return dict(status="PASS_SYNTHETIC_FULL_SCOPE_GUARDS",rejected_mutations=mutations)


def frozen_function_parity():
    require(sha(PILOT_BUILDER)==PILOT_BUILDER_SHA256,"Frozen pilot builder hash changed")
    source=ast.parse(PILOT_BUILDER.read_text());current=ast.parse(Path(__file__).read_text())
    original={n.name:ast.dump(n,include_attributes=False) for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newer={n.name:ast.dump(n,include_attributes=False) for n in current.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    unchanged=["iou_matrix","geometry_match","xywh_to_xyxy","emitted_rows","track_key","canonical_frame_name",
               "image_indices","require_same_frame","fixed_lag_bin","ObservationStats","PairStats","seal_gt_free_index"]
    require(all(original[name]==newer[name] for name in unchanged),"Copied core implementation drifted")
    return dict(status="PASS_FROZEN_PILOT_CORE_AST_PARITY",unchanged=unchanged,pilot_sha256=PILOT_BUILDER_SHA256)


def self_test():
    # Holdout/non-fit geometry is in the matching pool; matching has no class argument.
    matches, ambiguous, _ = geometry_match([[20, 0, 30, 10]], [[0, 0, 10, 10], [20, 0, 30, 10]])
    require(matches == {0: 1} and not ambiguous[0], "Full-GT assignment failed")
    matches, ambiguous, _ = geometry_match([[0, 0, 10, 10]], [[0, 0, 10, 10], [1, 0, 11, 10]])
    require(bool(ambiguous[0]) and len(matches) == 1, "Multi-GT ambiguity not flagged")
    matches, _, overlaps = geometry_match([[1, 0, 4, 1]], [[0, 0, 3, 1]])
    require(matches == {0: 0} and overlaps[0, 0] == .5, "Inclusive IoU=.5 gate failed")
    matches, _, _ = geometry_match([[100, 100, 110, 110]], [[0, 0, 3, 1]])
    require(matches == {}, "Unmatched unknown was assigned")
    rows = emitted_rows(dict(track_ids=np.array([-1, 7, 7]), emitted_track_mask=np.array([False, True, True])))
    require(rows.tolist() == [1, 2], "Unassigned rows included")
    keys = {track_key(1, 7), track_key(2, 7)}
    require(len(keys) == 2, "Local IDs from different videos merged")
    try:
        track_key(1, -1)
    except AssertionError:
        pass
    else:
        raise AssertionError("Unassigned ID formed a track")
    require("test" not in OPEN_SPLITS, "Test outcomes opened")
    support = PairStats()
    shared = dict(split="fit", source_group="A", video_id=1, local_track_id=7, gt_track_id=9,
                  gt_category_id=3, pred_category_id=8, error=1, soft_risk=.8, match_iou=.9)
    rows_a = [dict(shared, image_id=i) for i in (1, 2, 3)]
    for a, b in itertools.combinations(rows_a, 2):
        support.add(a, b)
    correct = dict(shared, source_group="B", video_id=2, pred_category_id=3, error=0, soft_risk=.1)
    support.add(dict(correct, image_id=4), dict(correct, image_id=5))
    measured = support.result()
    require(measured["pairs"] == 4 and measured["both_wrong"] == measured["same_wrong_class"] == 3,
            "Pair concentration fixture event counts failed")
    require(measured["concentration"]["max_source_pairs_share"] == .75 and
            measured["concentration"]["max_gt_trajectory_pairs_share"] == .75 and
            measured["concentration"]["max_source_both_wrong_share"] == 1.,
            "Source/trajectory pair concentration failed")
    return dict(status="PASS_CPU_FULL_SUPPORT_CONTRACTS", full_gt_before_eligibility=True,
                category_independent_geometry=True, multi_gt_ambiguity_excluded=True,
                inclusive_iou_half=True, unmatched_unknown=True,
                repeated_local_id_across_videos_separate=True, negative_id_not_aggregated=True,
                test_outcome_split_closed=True, no_dataset_or_model_loaded=True,
                pair_source_and_gt_trajectory_concentration=True,
                manual_source_provenance=provenance_self_test(), full_scope=full_scope_self_test(),
                integrated_seal_and_supervision=integrated_synthetic_test(), frozen_core_parity=frozen_function_parity(),
                real_supervision_payloads_parsed=False, real_support_executed=False,
                script_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Verify sealed metadata/audit bindings only; no supervision JSON parsing or support output")
    parser.add_argument("--execute", action="store_true", help="Only after independent review GO; permits real support construction")
    parser.add_argument("--write-contract", type=Path)
    args = parser.parse_args()
    require(not (args.self_test and (args.preflight or args.execute)), "Self-test cannot execute real support")
    require(not (args.preflight and args.execute), "Preflight cannot execute real support")
    if args.write_contract:
        write(args.write_contract,dict(status="PREDECLARED_FULL_SUPPORT_ADAPTATION",script_sha256=sha(__file__),**adaptation_contract()))
    if args.self_test:
        print(json.dumps(self_test(), indent=2), flush=True)
        return
    require(args.stage and args.run, "Provide --stage/--run, or --self-test")
    if args.preflight:
        preparation,plan,full_plan,manifest,metadata,collection,inventory=verified_inputs(args.stage.resolve(),args.run.resolve(),verify_dump_bytes=False)
        print(json.dumps(dict(status="PASS_FULL_SUPPORT_METADATA_PREFLIGHT_NOT_REAL_SUPPORT",stage=str(args.stage.resolve()),
                              run=str(args.run.resolve()),frames=len(metadata["images"]),videos=len(metadata["videos"]),
                              all_emitted_observations=collection["emitted_observations"],dump_inventory_files=len(inventory),
                              annotation_sha256=manifest["annotation_sha256"],full_plan_sha256=preparation["full_collection_plan_sha256"],
                              pilot_builder_sha256=sha(PILOT_BUILDER),script_sha256=sha(__file__),
                              real_supervision_payloads_parsed=False,raw_dump_payloads_opened=False,
                              raw_dump_bytes_rehashed=False,raw_dump_integrity_source="Bound complete dump verifier and independent full collection audit; actual execution rehashes all dump/receipt bytes before seal",
                              support_output_created=False,real_support_executed=False),indent=2),flush=True)
        return
    require(args.execute,"Real support is disabled until explicit independent-review GO and --execute")
    require(args.out is not None,"Provide a fresh --out support_* inside the full stage")
    build_support(args.stage.resolve(),args.run.resolve(),args.out.resolve())


if __name__ == "__main__":
    main()
