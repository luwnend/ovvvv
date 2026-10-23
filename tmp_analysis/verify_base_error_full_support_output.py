#!/usr/bin/env python3
"""Independent full-support audit: all index/label/pair rows and all-frame matching.

Does not import the producer or a tracker; old pilot verifier stays frozen.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import numpy as np
from scipy.optimize import linear_sum_assignment


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def overlaps(p, g):
    p = np.asarray(p, np.float64).reshape(-1, 4)
    g = np.asarray(g, np.float64).reshape(-1, 4)
    low = np.maximum(p[:, None, :2], g[None, :, :2])
    high = np.minimum(p[:, None, 2:], g[None, :, 2:])
    area = np.maximum(high-low, 0).prod(-1)
    union = np.maximum(p[:, 2:]-p[:, :2], 0).prod(-1)[:, None] + np.maximum(g[:, 2:]-g[:, :2], 0).prod(-1)[None, :] - area
    return np.divide(area, union, out=np.zeros_like(area), where=union > 0)


def xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x+w, y+h]


def filename(value):
    parts = str(value).replace("\\", "/").split("/")
    starts = [i for i, p in enumerate(parts) if p in ("train", "val", "test")]
    check(len(starts) == 1 and ".." not in parts, "Ambiguous filename split")
    return "/".join(parts[starts[0]:])


def primary_stats(observations, pairs):
    source = defaultdict(Counter); trajectory = defaultdict(Counter)
    values = Counter()
    product = 0.
    for a, b in pairs:
        agree = a["pred_category_id"] == b["pred_category_id"]
        both = bool(a["error"] and b["error"])
        events = dict(pairs=1, agreement_pairs=int(agree), both_correct=int(not a["error"] and not b["error"]),
                      exactly_one_wrong=int(bool(a["error"]) != bool(b["error"])), both_wrong=int(both),
                      same_wrong_class=int(agree and both), different_wrong_class=int(not agree and both))
        values.update(events); source[a["source_group"]].update(events)
        trajectory[(a["video_id"], a["gt_track_id"])].update(events)
        product += a["soft_risk"]*b["soft_risk"]
    n = values["pairs"]
    concentration = {}
    for event in ("pairs", "both_wrong", "same_wrong_class"):
        for name, by in (("source", source), ("gt_trajectory", trajectory)):
            concentration["max_"+name+"_"+event+"_share"] = max((v[event] for v in by.values()), default=0)/values[event] if values[event] else None
    return values, concentration, product/n if n else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); out = args.support.resolve(); started = time.time()
    destination = args.output or out / "independent_support_audit.json"
    check(not destination.exists(), "Refuse to overwrite independent audit")
    receipt_start_sha = sha(out / "support_receipt.json")
    summary_start_sha = sha(out / "support_summary.json")
    verifier_start_sha = sha(__file__)
    receipt = load(out / "support_receipt.json"); summary = load(out / "support_summary.json")
    check(receipt["status"] == summary["status"] == "PASS_FIT_TUNE_SIGNAL_SUPPORT_AUDIT", "Unfinished support")
    check(summary["support_receipt_sha256"] == sha(out / "support_receipt.json"), "Summary receipt SHA")
    check(not receipt["test_outcomes_computed"] and not receipt["class_holdout_outcomes_computed"] and not receipt["training_started"], "Closed-outcome/training flags")
    check(not receipt["oracle_identity_diagnostics_used_for_training"], "Oracle training flag")
    for path, digest in receipt["inputs"].items():
        check(sha(path) == digest, "Support input SHA: " + path)
    for name, digest in receipt["output_files"].items():
        check(sha(out / name) == digest, "Support output SHA: " + name)
    stage = Path(receipt["stage"]); run = Path(receipt["run"])
    seal = load(out / "gt_free_index_manifest.json")
    check(sha(out / "gt_free_index_manifest.json") == receipt["gt_free_index_manifest_sha256"] == summary["gt_free_index_manifest_sha256"], "Index seal SHA")
    check(seal["status"] == "SEALED_GT_FREE_ALL_EMITTED_INDEX_BEFORE_SUPERVISION_OPEN" and not seal["gt_filtering"] and not seal["supervision_opened"] and not seal["negative_local_ids_aggregated"], "GT-free seal flags")
    for name, digest in seal["files"].items():
        check(sha(out / name) == digest, "GT-free sealed file SHA")
    meta = load(stage / "full_inference.json"); preparation = load(stage / "preparation_manifest.json")
    plan = load(preparation["plan"]); records = {v["video_id"]: v for v in plan["full_metadata_split"]["videos"]}
    images = {im["id"]: im for im in meta["images"]}
    full_plan = load(stage / "full_collection_plan.json")
    check(receipt["collection_scope"] == preparation["collection_scope"] == "FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES", "Wrong full scope")
    check(full_plan["parent_plan_sha256"] == preparation["plan_sha256"] == sha(preparation["plan"]), "Parent plan binding")
    check(preparation["full_collection_plan_sha256"] == sha(stage / "full_collection_plan.json"), "Full plan binding")
    check(meta["annotations"] == [] and meta["tracks"] == [], "GT in full inference")
    check(len(meta["images"]) == 15283 and len(meta["videos"]) == 448, "Wrong full frame/video count")
    check(len(images) == len(meta["images"]) and set(images) == {r["image_id"] for r in full_plan["images"]}, "Full frame scope/duplicates")
    check([c["id"] for c in meta["categories"]] == list(range(1,1204)), "Full vocabulary")
    check(all(records[im["video_id"]]["split"] in ("fit","tune") for im in meta["images"]), "Closed video selected")
    check(len({records[im["video_id"]]["source_group"] for im in meta["images"]}) == 383, "Full source coverage")
    group_parts = defaultdict(set)
    for r in records.values(): group_parts[r["source_group"]].add(r["split"])
    check(all(len(p)==1 for p in group_parts.values()), "Source leakage")
    collection = load(run / "collection_receipt.json"); inventory = load(run / "dump_inventory.json")
    expected_files = {"v%d_i%d_f%d%s" % (im["video_id"],im["id"],im["frame_id"],suffix)
                      for im in meta["images"] for suffix in (".npz",".receipt.json")}
    check(set(inventory) == expected_files == {p.name for p in (run/"dump").iterdir()}, "Full raw inventory scope")
    collected_audit = load(run / "independent_collection_audit.json")
    execution = {e["image_id"]: e for e in rows(run / "frame_execution.jsonl")}
    check(len(execution) == len(rows(run / "frame_execution.jsonl")) == len(images) and set(execution) == set(images), "Runtime event scope/duplicates")
    check(collected_audit["status"] == "PASS_INDEPENDENT_FULL_GT_FREE_COLLECTION_AUDIT", "Full collection audit missing")
    index = rows(out / "gt_free_observations.jsonl"); tracks = rows(out / "gt_free_tracks.jsonl")
    check(len(index) == seal["observations"] == collection["emitted_observations"] == 681602, "Incomplete all-emitted index")
    check([x["observation_id"] for x in index] == list(range(len(index))), "Observation IDs not unique contiguous")
    check(len(tracks) == seal["tracks"] and len(tracks) > 0, "Track index count")
    by_frame = defaultdict(list); by_track = defaultdict(list)
    index_fields = {"observation_id", "video_id", "local_track_id", "image_id", "sampled_step", "original_frame_index", "split", "source_group", "dump_file", "dump_sha256", "dump_row", "pred_category_id", "score"}
    for x in index:
        check(set(x) == index_fields and x["local_track_id"] >= 0, "GT field or negative track in index")
        im = images[x["image_id"]]; rec = records[im["video_id"]]
        check((x["video_id"], x["sampled_step"], x["original_frame_index"], x["split"], x["source_group"]) == (im["video_id"], im["frame_id"], im["frame_index"], rec["split"], rec["source_group"]), "Index metadata mapping")
        check(x["dump_sha256"] == inventory[x["dump_file"]], "Index/raw dump SHA binding")
        by_frame[x["image_id"]].append(x); by_track[(x["video_id"], x["local_track_id"])].append(x["observation_id"])
    seen_tracks = set()
    for t in tracks:
        key = (t["video_id"], t["local_track_id"])
        check(key not in seen_tracks and key in by_track and t["local_track_id"] >= 0, "Duplicate/cross-video/negative track")
        check(t["observation_ids"] == by_track[key] and t["n_observations"] == len(by_track[key]), "GT-filtered trajectory membership")
        check(t["split"] == records[key[0]]["split"] and t["source_group"] == records[key[0]]["source_group"], "Track split/source")
        seen_tracks.add(key)
    check(seen_tracks == set(by_track), "Missing complete trajectory")
    supervised = rows(out / "supervised_observations.jsonl"); pair_rows = rows(out / "supervised_pairs.jsonl")
    check(len(supervised) == receipt["supervised_observations"] == summary["supervised_observations"], "Supervised observation count")
    check(len(pair_rows) == receipt["supervised_pairs"] == summary["supervised_pairs"], "Pair count")
    supervised_by_id = {x["observation_id"]: x for x in supervised}
    check(len(supervised_by_id) == len(supervised), "Duplicate supervised observation")
    supervised_by_frame = defaultdict(list)
    for x in supervised:
        supervised_by_frame[x["image_id"]].append(x)
    targets = load(stage / "manual_anchor_targets.json"); anchors = {a["id"]: a for a in targets["annotations"]}
    fit_categories = set(targets["readout_fit_category_ids"])
    check(fit_categories == set(full_plan["readout_fit_category_ids"]) == set(plan["vocabulary"]["readout_fit_category_ids"]) and len(fit_categories) == 146, "Fit class partition")
    holdout = set(targets["readout_category_holdout_ids"])
    check(holdout == set(full_plan["readout_category_holdout_ids"]) == set(plan["vocabulary"]["readout_category_holdout_ids"]) and len(holdout) == 36 and not holdout & fit_categories, "Heldout class partition")
    official = load(plan["inputs"]["tao_train"]["path"])
    original_images = {im["id"]: im for im in official["images"]}
    official_coordinates = {(im["video_id"], im["frame_index"]): im for im in official["images"]}
    check(len(original_images) == len(official_coordinates) == len(official["images"]), "Official duplicate image/coordinate")
    open_coordinates = {(im["video_id"], im["frame_index"]): im["id"] for im in meta["images"] if records[im["video_id"]]["split"] in ("fit", "tune")}
    synsets = defaultdict(list)
    for c in targets["categories"]:
        if c.get("synset"):
            synsets[c["synset"]].append(c["id"])
    old_categories = {c["id"]: c for c in official["categories"]}
    truth = defaultdict(list); official_by_key = {}
    for a in official["annotations"]:
        im = original_images[a["image_id"]]; coordinate = (im["video_id"], im["frame_index"])
        if coordinate not in open_coordinates:
            continue
        iid = open_coordinates[coordinate]; mapped = synsets.get(old_categories[a["category_id"]].get("synset"), [])
        record = dict(a, canonical_category=mapped[0] if len(mapped) == 1 else None, xyxy=xyxy(a["bbox"]))
        truth[iid].append(record); key = (iid, a["track_id"])
        check(key not in official_by_key, "Duplicate official anchor identity")
        official_by_key[key] = record
    del official
    used_anchor_ids = set(anchors)
    check(len(anchors) == len(targets["annotations"]) == 46908 and used_anchor_ids == set(full_plan["manual_anchor_ctao_annotation_ids"]), "Full manual target scope")
    source = load(plan["inputs"]["ctao_base"]["path"])
    source_images = {im["id"]: im for im in source["images"]}
    check(len(source_images) == len(source["images"]), "Source duplicate image ID")
    for iid,im in images.items():
        sim=source_images[iid]; oim=official_coordinates[(im["video_id"],im["frame_index"])]
        for other in (sim,oim):
            check(all(im[k] == other[k] for k in ("video_id","frame_index","width","height")) and filename(im["file_name"]) == filename(other["file_name"]), "Full official/source frame provenance")
        check(sim.get("is_ori") is True and im["source_frame_id"] == sim["frame_id"], "Full exact manual source timeline")
    source_anchors = {a["id"]: a for a in source["annotations"] if a["id"] in used_anchor_ids}
    check(set(source_anchors) == used_anchor_ids, "Missing original C-TAO anchors")
    check(len(source_anchors) == sum(a["id"] in used_anchor_ids for a in source["annotations"]), "Duplicate original C-TAO anchor")
    canonical = {c["id"]: c for c in targets["categories"]}
    source_categories = {c["id"]: c for c in source["categories"]}
    check(set(canonical) == set(source_categories) == set(range(1,1204)), "Canonical taxonomy coverage")
    for cid,c in canonical.items():
        check(c.get("synset") == source_categories[cid].get("synset") and c.get("frequency") == source_categories[cid].get("frequency"), "Category source bridge")
    for aid,a in anchors.items():
        original=source_anchors[aid]; im=images[a["image_id"]]; sim=source_images[a["image_id"]]
        gt=official_by_key[(a["image_id"],a["track_id"])]; part=records[im["video_id"]]["split"]
        check(all(a[k] == original[k] for k in ("id","image_id","video_id","track_id","category_id","bbox")), "Full C-TAO manual source")
        check(sim.get("is_ori") is True and im["source_frame_id"]==sim["frame_id"], "Manual source frame ID")
        check(a["video_id"]==im["video_id"] and a["readout_split"]==part, "Full manual split/video")
        check(a["readout_class_holdout"]==(a["category_id"] not in fit_categories), "Manual class eligibility")
        check(a["eligible_for_parameter_fit"]==(part=="fit" and a["category_id"] in fit_categories) and a["eligible_for_tuning"]==(part=="tune" and a["category_id"] in fit_categories), "Manual fit/tune eligibility")
        check(canonical[a["category_id"]]["frequency"] in ("f","c"), "Nonbase target")
        check(gt["canonical_category"]==a["category_id"] and np.allclose(gt["bbox"],a["bbox"],rtol=0,atol=1e-6), "Full official synset/box bridge")
        check((a["official_tao_annotation_id"],a["official_tao_image_id"],a["official_tao_category_id"])==(gt["id"],gt["image_id"],gt["category_id"]), "Explicit old TAO annotation/image/category bridge")
        for k in ("video_id","frame_index","width","height"):
            check(im[k]==sim[k], "Full C-TAO frame metadata")
        check(filename(im["file_name"]) == filename(sim["file_name"]), "Full C-TAO filename")
    anchor_by_key={(a["image_id"],a["track_id"]):a for a in anchors.values()}
    eligible_anchor_count=Counter(a["image_id"] for a in anchors.values() if a["category_id"] in fit_categories)
    check(len(anchor_by_key)==len(anchors), "Duplicate manual identity/frame")
    for x in supervised:
        oid = x["observation_id"]; observed = index[oid]; a = anchors[x["manual_anchor_id"]]
        check(x["split"] in ("fit", "tune") and x["gt_category_id"] in fit_categories, "Forbidden class/split outcome")
        check(x["eligible_for_parameter_fit"] == (x["split"] == "fit"), "Fit/tune parameter eligibility")
        for field in ("image_id", "video_id", "local_track_id", "split", "source_group", "sampled_step", "original_frame_index", "pred_category_id"):
            check(x[field] == observed[field], "Supervision/index mapping: " + field)
        check((a["image_id"], a["video_id"], a["track_id"], a["category_id"], a["readout_split"]) == (x["image_id"], x["video_id"], x["gt_track_id"], x["gt_category_id"], x["split"]), "Supervised manual target mismatch")
        check(not a["readout_class_holdout"] and a["eligible_for_parameter_fit"] == (x["split"] == "fit"), "Heldout target leaked")
        original = source_anchors[a["id"]]
        check(all(a[k] == original[k] for k in ("id", "image_id", "video_id", "track_id", "category_id", "bbox")), "C-TAO exact manual source")
        check(source_images[a["image_id"]].get("is_ori") is True, "Pseudo-frame supervised")
        gt = official_by_key[(a["image_id"], a["track_id"])]
        check(gt["canonical_category"] == a["category_id"] and np.allclose(gt["bbox"], a["bbox"], rtol=0, atol=1e-6), "Official synset/manual box source")
        check(x["error"] == int(x["pred_category_id"] != x["gt_category_id"]), "Error label mismatch")
    del source, source_images
    frame_receipts = rows(out / "frame_matching_receipts.jsonl")
    f_receipts = {x["image_id"]: x for x in frame_receipts}
    check(len(f_receipts) == len(frame_receipts) == 15283 and set(f_receipts) == set(open_coordinates.values()), "Test joined or fit/tune frame omitted")
    sampled = set()
    for split in ("fit", "tune"):
        candidates = [f for f in frame_receipts if f["split"] == split]
        ranked = sorted(candidates, key=lambda f: hashlib.sha256(("independent-support-v1/"+str(f["image_id"])).encode()).hexdigest())
        sampled.update(f["image_id"] for f in ranked[:6])
        for field in ("ambiguous_multi_gt_predictions", "nonmanual_intersection_match_excluded", "class_holdout_match_closed"):
            edge = [f for f in ranked if f.get(field, 0)]
            if edge:
                sampled.add(edge[0]["image_id"])
    structure = defaultdict(Counter); sample_results = []; duplicate_supervised_gt = set()
    matched_identities = defaultdict(set)
    for frame_number, im in enumerate(meta["images"],1):
        iid, vid, step = im["id"], im["video_id"], im["frame_id"]
        part = records[vid]["split"]; path = run / "dump" / ("v%d_i%d_f%d.npz" % (vid, iid, step))
        check(sha(path) == inventory[path.name], "Live NPZ changed: " + path.name)
        receipt_file = path.with_suffix(".receipt.json")
        check(sha(receipt_file) == inventory[receipt_file.name], "Live frame receipt changed: " + receipt_file.name)
        with np.load(path, allow_pickle=False) as data:
            tids, mask, labels, boxes = data["track_ids"], data["emitted_track_mask"], data["det_labels"], data["det_bboxes"]
            emitted = np.flatnonzero(tids >= 0)
            check(np.array_equal(mask, tids >= 0), "Emitted mask")
            obs = by_frame[iid]
            check([x["dump_row"] for x in obs] == emitted.tolist(), "GT-selected or omitted emitted row")
            check(len(obs) == execution[iid]["emitted_observations"], "Runtime/index count mismatch")
            for x in obs:
                row = x["dump_row"]
                check(x["dump_file"] == path.name and x["local_track_id"] == int(tids[row]) and x["pred_category_id"] == int(labels[row])+1 and x["score"] == float(boxes[row, -1]), "Raw/index ID-label-score mapping")
            structure[part].update(frames=1, retained_rows=len(tids), unassigned_rows=int((tids < 0).sum()), emitted_rows=len(emitted), zero_emitted_frames=int(len(emitted) == 0), duplicate_local_id_frame_extra_rows=len(emitted)-len(set(tids[emitted].tolist())))
            if iid in supervised_by_frame:
                probabilities = data["p_fused"].astype(np.float64); category_ids = data["posterior_category_ids"]
                check(np.array_equal(category_ids, np.r_[np.arange(1, 1204), -1]), "Posterior category mapping")
                for x in supervised_by_frame[iid]:
                    row = index[x["observation_id"]]["dump_row"]; p = probabilities[row, :-1]; p = p/p.sum()
                    check(abs(x["soft_risk"]-(1-p[x["gt_category_id"]-1])) < 1e-12 and x["posterior_argmax_category_id"] == int(p.argmax())+1, "Posterior risk/argmax mapping")
                    matrix = overlaps(boxes[row:row+1, :4], [g["xyxy"] for g in truth[iid]])[0]
                    target_column = next(j for j, g in enumerate(truth[iid]) if g["track_id"] == x["gt_track_id"])
                    check(int((matrix >= .5).sum()) == 1 and matrix[target_column] >= .5 and abs(matrix[target_column]-x["match_iou"]) < 1e-12, "Ambiguous or wrong supervised geometry")
                    key = (iid, x["gt_track_id"]); check(key not in duplicate_supervised_gt, "GT assigned twice")
                    duplicate_supervised_gt.add(key)
            if True:  # Independently recompute every full-scope frame.
                gt = truth[iid]; matrix = overlaps(boxes[emitted, :4], [g["xyxy"] for g in gt]); valid = matrix >= .5
                r, c = linear_sum_assignment(np.where(valid, -(max(matrix.shape)+1.+matrix), 0.))
                matches = {int(a): int(b) for a, b in zip(r, c) if valid[a, b]}
                ambiguity = valid.sum(1) > 1; expected_ids = set(); seen_anchors = set()
                fc = Counter(frames=1, emitted_predictions=len(emitted), full_official_gt=len(gt),
                             official_gt_unmapped_category=sum(g["canonical_category"] is None for g in gt),
                             geometry_matched=len(matches), geometry_unmatched_unknown=len(emitted)-len(matches),
                             ambiguous_multi_gt_predictions=int(ambiguity.sum()), official_gt_not_assigned=len(gt)-len(matches))
                for pr, gr in matches.items():
                    target=gt[gr]
                    if ambiguity[pr]: fc["ambiguous_matched_excluded"]+=1; continue
                    matched_identities[(vid,int(tids[emitted[pr]]))].add(target["track_id"])
                    if target.get("iscrowd",0) or target.get("ignore",0): fc["crowd_or_ignore_match_excluded"]+=1; continue
                    anchor=anchor_by_key.get((iid,target["track_id"]))
                    if anchor is None: fc["nonmanual_intersection_match_excluded"]+=1; continue
                    if anchor["category_id"] not in fit_categories: fc["class_holdout_match_closed"]+=1; continue
                    expected_ids.add(obs[pr]["observation_id"]);seen_anchors.add(anchor["id"]);fc["eligible_fit_category_supervised"]+=1
                fc["eligible_manual_anchors"]=eligible_anchor_count[iid]
                fc["eligible_manual_anchors_without_supervision"]=fc["eligible_manual_anchors"]-len(seen_anchors)
                check(f_receipts[iid]==dict(image_id=iid,video_id=vid,split=part,**dict(fc)), "All-frame matching/exclusion receipt mismatch")
                check(expected_ids == {x["observation_id"] for x in supervised_by_frame[iid]}, "Sampled full-GT Hungarian/eligibility disagreement")
                fr = f_receipts[iid]
                check(fr["full_official_gt"] == len(gt) and fr["geometry_matched"] == len(matches) and fr["geometry_unmatched_unknown"] == len(emitted)-len(matches) and fr["ambiguous_multi_gt_predictions"] == int(ambiguity.sum()), "Sampled frame matching counters")
                sample_results.append(dict(image_id=iid, split=part, predictions=len(emitted), all_official_gt=len(gt), matched=len(matches), supervised=len(expected_ids)))
        if frame_number % 1000 == 0:
            print(json.dumps(dict(status="INDEPENDENT_FULL_SUPPORT_FRAME_AUDIT_PROGRESS",frames=frame_number,total=15283,elapsed_seconds=time.time()-started)),flush=True)
    check({k: dict(v) for k, v in structure.items()} == seal["split_structure"], "All-emitted split structure")
    check(receipt["test"] == dict(status="CLOSED_NO_GT_MATCHING_NO_ERROR_OUTCOMES", **dict(structure["test"])) and summary["test"] == receipt["test"], "Test outcomes or structure")
    pair_fields = {"split", "video_id", "local_track_id", "gt_track_id", "observation_i", "observation_j", "sample_step_lag", "original_frame_lag", "agreement", "both_wrong", "same_wrong_class"}
    actual_pairs = set(); split_pairs = defaultdict(list); expected_groups = defaultdict(list)
    for x in supervised:
        expected_groups[(x["split"], x["video_id"], x["local_track_id"], x["gt_track_id"])].append(x)
    for pair in pair_rows:
        check(set(pair) == pair_fields, "Oracle/unknown field in primary pair")
        a = supervised_by_id[pair["observation_i"]]; b = supervised_by_id[pair["observation_j"]]
        check(a["split"] == b["split"] == pair["split"] in ("fit", "tune") and a["gt_category_id"] == b["gt_category_id"] in fit_categories, "Ineligible pair endpoint")
        for field in ("video_id", "local_track_id", "gt_track_id"):
            check(pair[field] == a[field] == b[field], "Pair identity mismatch")
        check(a["image_id"] != b["image_id"] and pair["sample_step_lag"] == b["sampled_step"]-a["sampled_step"] > 0 and pair["original_frame_lag"] == b["original_frame_index"]-a["original_frame_index"] > 0, "Pair lag")
        agreement = a["pred_category_id"] == b["pred_category_id"]; both = bool(a["error"] and b["error"])
        check(pair["agreement"] == agreement and pair["both_wrong"] == both and pair["same_wrong_class"] == bool(agreement and both), "Pair event labels")
        key = (a["observation_id"], b["observation_id"]); check(key not in actual_pairs, "Duplicate pair")
        actual_pairs.add(key); split_pairs[pair["split"]].append((a, b))
    expected_pairs = {(a["observation_id"], b["observation_id"]) for group in expected_groups.values() for a, b in itertools.combinations(sorted(group, key=lambda x: x["sampled_step"]), 2) if a["image_id"] != b["image_id"]}
    check(actual_pairs == expected_pairs, "Pairs selected by extra oracle condition")
    split_results = {}
    for part in ("fit", "tune"):
        selected = [x for x in supervised if x["split"] == part]
        stats = receipt["fit_tune"][part]["observations"]
        check(stats["observations"] == len(selected) and stats["errors"] == sum(x["error"] for x in selected) and stats["correct"] == sum(not x["error"] for x in selected), "Observation outcome totals")
        check(stats["source_groups"] == len({x["source_group"] for x in selected}) and stats["gt_trajectories"] == len({(x["video_id"], x["gt_track_id"]) for x in selected}) and stats["predicted_tracks"] == len({(x["video_id"], x["local_track_id"]) for x in selected}), "Observation grouping counts")
        counted = Counter()
        for fr in frame_receipts:
            if fr["split"] == part:
                counted.update({k: v for k, v in fr.items() if k not in ("image_id", "video_id", "split")})
        excluded = sum(a["image_id"] == b["image_id"] for key,group in expected_groups.items() if key[0] == part for a,b in itertools.combinations(group,2))
        if excluded: counted["same_frame_duplicate_track_pair_excluded"] += excluded
        check(dict(counted) == receipt["fit_tune"][part]["matching_counts"], "Frame receipt aggregate")
        events, concentration, softproduct = primary_stats(selected, split_pairs[part])
        primary = receipt["fit_tune"][part]["pair_scopes"]["all_same_identity_subpairs/all"]
        check(all(primary[k] == v for k, v in events.items()) and primary["concentration"] == concentration and abs(primary["mean_endpoint_soft_risk_product"]-softproduct) < 1e-12, "Primary pair receipt stats")
        endpoint_groups={a["source_group"] for a,b in split_pairs[part]}
        endpoint_gt={(a["video_id"],a["gt_track_id"]) for a,b in split_pairs[part]}
        endpoint_pred={(a["video_id"],a["local_track_id"]) for a,b in split_pairs[part]}
        endpoint_videos={a["video_id"] for a,b in split_pairs[part]}
        endpoint_cats={a["gt_category_id"] for a,b in split_pairs[part]}
        check((primary["source_groups"],primary["gt_trajectories"],primary["predicted_tracks"],primary["videos"],primary["target_categories"],primary["target_category_ids"]) == (len(endpoint_groups),len(endpoint_gt),len(endpoint_pred),len(endpoint_videos),len(endpoint_cats),sorted(endpoint_cats)), "Primary pair grouping counts")
        by_source_events={group:dict(pairs=0,both_wrong=0,same_wrong_class=0) for group in endpoint_groups}
        for a,b in split_pairs[part]:
            c=by_source_events[a["source_group"]];c["pairs"]+=1;c["both_wrong"]+=int(bool(a["error"] and b["error"]));c["same_wrong_class"]+=int(bool(a["error"] and b["error"] and a["pred_category_id"]==b["pred_category_id"]))
        check(primary["per_source_event_counts"] == by_source_events, "Primary per-source event counts")
        check(receipt["fit_tune"][part]["diagnostic_oracle_identity"]["training_or_tuning_use_permitted"] is False, "Oracle diagnostic used for training")
        split_results[part] = dict(observations=len(selected), pairs=len(split_pairs[part]), target_categories=len({x["gt_category_id"] for x in selected}), errors=stats["errors"], source_groups=stats["source_groups"])
    check(len(sample_results)==15283, "Partial independent full matching")
    for part in ("fit","tune"):
        predtracks={(x["video_id"],x["local_track_id"]) for x in supervised if x["split"]==part}
        oracle=receipt["fit_tune"][part]["diagnostic_oracle_identity"]
        check(oracle["observed_multi_identity_predicted_tracks"]==sum(len(matched_identities[k])>1 for k in predtracks) and oracle["observed_single_identity_predicted_tracks"]==sum(len(matched_identities[k])==1 for k in predtracks), "Independent oracle descriptive counts")
    for name, digest in receipt["output_files"].items():
        check(sha(out / name) == digest, "Support output changed during independent audit")
    for path, digest in receipt["inputs"].items():
        check(sha(path) == digest, "Support input changed during independent audit: " + path)
    check(sha(out / "support_receipt.json") == receipt_start_sha and sha(out / "support_summary.json") == summary_start_sha and sha(__file__) == verifier_start_sha, "Receipt/summary/verifier changed during independent audit")
    report = dict(status="PASS_INDEPENDENT_FULL_SUPPORT_OUTPUT_AUDIT", support=str(out),
                  support_receipt_sha256=sha(out / "support_receipt.json"), support_builder_sha256=receipt["inputs"][str(Path(__file__).resolve().parent / "build_base_error_full_support.py")],
                  gt_free_index_manifest_sha256=sha(out / "gt_free_index_manifest.json"),
                  all_emitted_observations=len(index), all_emitted_tracks=len(tracks),
                  all_frame_dump_hashes_and_index_rows_verified=len(images), all_raw_dump_and_frame_receipt_hashes_verified=2*len(images), input_hashes_verified=len(receipt["inputs"]), output_hashes_verified=len(receipt["output_files"]),
                  supervised_observations_verified=len(supervised), primary_pairs_verified=len(pair_rows),
                  exact_manual_source_and_posterior_mapping_verified=True, all_manual_target_provenance_verified=len(anchors), primary_pairs_equal_all_eligible_endpoint_combinations=True,
                  oracle_fields_absent_from_primary_pairs=True, test_prediction_gt_joins_or_outcomes=False,
                  fit_tune=split_results, test_structure_only=dict(structure["test"]),
                  independently_recomputed_full_gt_matching_frames=len(sample_results), sampled_matching_results=[r for r in sample_results if r["image_id"] in sampled],
                  matching_scope="All15283 full fit/tune frames; sample listing is display-only",
                  allowed_next_step="Predeclared CPU full nested B0/B1/B2/B3 hard-error mean diagnostics after hash-bound implementation preflight; no pair training, tracker training or test opening",
                  training_readiness_established=False, method_efficacy_established=False,
                  elapsed_seconds=time.time()-started,
                  limitations=["Raw posterior probability semantics inherit the sealed full collection audit; this verifier independently checks all supervised soft risks and argmax mappings", "Whole-track identity oracle scopes remain diagnostics only", "No detector-unseen class/video, centered covariance or efficacy claim"],
                  independent_verifier_sha256=sha(__file__))
    with destination.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
