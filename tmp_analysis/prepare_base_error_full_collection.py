#!/usr/bin/env python3
"""Prepare full fit/tune sparse-anchor collection metadata; never infer or fit.

The parent plan's source-group and category splits stay sealed. Only exact
C-TAO/original-TAO manual-anchor intersections in fit/tune videos select frames.
Inference receives whitelisted metadata, empty annotations and empty tracks;
manual boxes, identities, classes and source-annotation IDs stay in another file.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
PARENT_PLAN_SHA = "f37f1ecc52734953ea5269407ab0aa8013e650e97bed4dde3cdd399a11e13ba5"
IMAGE_FIELDS = ("id", "video_id", "file_name", "width", "height", "frame_id", "frame_index")
VIDEO_FIELDS = ("id", "name", "width", "height", "fps")
CATEGORY_FIELDS = ("id", "name", "synonyms", "synset", "def", "frequency")
TARGET_FIELDS = ("id", "image_id", "video_id", "track_id", "category_id", "bbox", "area", "iscrowd")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, key, label):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, "Duplicate " + label + ": " + str(k))
        result[k] = row
    return result


def canonical_filename(value):
    parts = str(value).replace("\\", "/").split("/")
    require(".." not in parts, "Parent path component in filename")
    positions = [i for i, p in enumerate(parts) if p in ("train", "val", "test")]
    require(len(positions) == 1, "Ambiguous split prefix in filename: " + str(value))
    result = "/".join(parts[positions[0]:])
    require(result.startswith("train/"), "Full collection must use original TAO train frames")
    return result


def write(path, obj):
    with path.open("x") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def counts(anchors, images, records, fit_categories):
    result = {}
    for split in ("fit", "tune"):
        anns = [a for a in anchors if a["readout_split"] == split]
        ims = [im for im in images if records[im["video_id"]]["split"] == split]
        vids = {im["video_id"] for im in ims}
        fit = [a for a in anns if a["category_id"] in fit_categories]
        per_class = defaultdict(list)
        for a in fit:
            per_class[a["category_id"]].append(a)
        result[split] = dict(
            allocation_videos=sum(v["split"] == split for v in records.values()),
            selected_nonempty_videos=len(vids),
            source_groups=len({records[v]["source_group"] for v in vids}),
            anchor_images=len(ims), manual_anchor_boxes=len(anns),
            fit_category_boxes=len(fit),
            heldout_category_boxes=len(anns) - len(fit),
            fit_categories_with_boxes=sorted(per_class),
            fit_category_support={str(cid): dict(
                manual_boxes=len(xs),
                videos=len({x["video_id"] for x in xs}),
                source_groups=len({records[x["video_id"]]["source_group"] for x in xs}),
                gt_tracks=len({(x["video_id"], x["track_id"]) for x in xs}))
                for cid, xs in sorted(per_class.items())},
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "tmp_analysis/base_error_learning_plan_20261023.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/base_error_full_collection_20261023")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.out = args.out.resolve()
    require(not args.out.exists(), "Preserve existing stage; choose a new --out")
    require(sha(args.plan) == PARENT_PLAN_SHA, "Unexpected parent plan; splits must remain sealed")
    plan = json.loads(args.plan.read_text())
    require(plan["metadata_validation"]["status"] == "PASS_METADATA_ONLY", "Parent metadata validation missing")
    source = Path(plan["inputs"]["ctao_base"]["path"])
    official_source = Path(plan["inputs"]["tao_train"]["path"])
    protected_paths = [args.plan, source, official_source, Path(__file__).resolve()]
    protected_paths += [ROOT / "tmp_analysis" / name for name in (
        "build_base_error_learning_plan.py", "prepare_base_error_collection.py",
        "collect_base_error_pilot.py", "verify_base_error_collection.py", "verify_base_error_dump.py",
        "build_base_error_support.py")]
    pilot = ROOT / "results/base_error_collection_20261023_v3"
    protected_paths += [pilot / "preparation_manifest.json", pilot / "pilot_inference.json",
                        pilot / "manual_anchor_targets.json"]
    protected_before = {str(path): sha(path) for path in protected_paths}
    require(protected_before[str(source)] == plan["inputs"]["ctao_base"]["sha256"], "C-TAO changed")
    require(protected_before[str(official_source)] == plan["inputs"]["tao_train"]["sha256"], "Original TAO changed")
    records_all = unique(plan["full_metadata_split"]["videos"], lambda x: x["video_id"], "plan video ID")
    records = {vid: row for vid, row in records_all.items() if row["split"] in ("fit", "tune")}
    require(len(records) == 450, "Full fit/tune allocation changed")
    require(not ({x["source_group"] for x in records.values() if x["split"] == "fit"} &
                 {x["source_group"] for x in records.values() if x["split"] == "tune"}), "Source group split leakage")
    fit_categories = set(plan["vocabulary"]["readout_fit_category_ids"])
    holdout_categories = set(plan["vocabulary"]["readout_category_holdout_ids"])
    require(len(fit_categories) == 146 and len(holdout_categories) == 36 and
            not fit_categories & holdout_categories, "Category split changed")

    print(json.dumps(dict(status="READING_METADATA_ONLY", source=str(source))), flush=True)
    ctao = json.loads(source.read_text())
    official = json.loads(official_source.read_text())
    categories_by_id = unique(ctao["categories"], lambda x: x["id"], "canonical category ID")
    require(sorted(categories_by_id) == list(range(1, 1204)), "Wrong canonical full vocabulary")
    categories = [{k: categories_by_id[cid][k] for k in CATEGORY_FIELDS if k in categories_by_id[cid]}
                  for cid in range(1, 1204)]
    synsets = defaultdict(list)
    for row in categories:
        if row.get("synset"):
            synsets[row["synset"]].append(row["id"])
    old_categories = unique(official["categories"], lambda x: x["id"], "official category ID")
    bridge = {cid: synsets.get(row.get("synset"), []) for cid, row in old_categories.items()}
    source_images = unique(ctao["images"], lambda x: x["id"], "C-TAO image ID")
    source_videos = unique(ctao["videos"], lambda x: x["id"], "C-TAO video ID")
    official_images = unique((im for im in official["images"] if im["video_id"] in records),
                             lambda x: x["id"], "fit/tune official image ID")
    official_coordinates = unique(official_images.values(),
                                  lambda x: (x["video_id"], x["frame_index"]), "fit/tune official coordinate")
    manual_by_key = unique((a for a in official["annotations"] if a["video_id"] in records),
                           lambda a: (a["video_id"], official_images[a["image_id"]]["frame_index"], a["track_id"]),
                           "fit/tune official identity/frame")
    anchors = []
    anchor_ids = set()
    excluded = Counter()
    for row in ctao["annotations"]:
        vid = row["video_id"]
        if vid not in records:
            continue  # Never join test-partition predictions/labels or build its GT frame map.
        im = source_images[row["image_id"]]
        if not im.get("is_ori"):
            continue
        key = (vid, im["frame_index"], row["track_id"])
        original = manual_by_key.get(key)
        if original is None:
            excluded["no_official_identity_frame"] += 1
            continue
        if bridge[original["category_id"]] != [row["category_id"]]:
            excluded["not_unique_equal_synset_bridge"] += 1
            continue
        require(len(row["bbox"]) == len(original["bbox"]) == 4, "Invalid manual bbox")
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(row["bbox"], original["bbox"])):
            excluded["different_bbox"] += 1
            continue
        require(categories_by_id[row["category_id"]]["frequency"] in ("f", "c"), "Novel GT in base source")
        require(row["category_id"] in fit_categories | holdout_categories, "Unallocated anchor category")
        require(row["id"] not in anchor_ids, "Duplicate manual source annotation ID")
        anchor_ids.add(row["id"])
        old_im = official_coordinates[(vid, im["frame_index"])]
        require(old_im["id"] == original["image_id"], "Official image/annotation mismatch")
        require(canonical_filename(im["file_name"]) == canonical_filename(old_im["file_name"]), "Manual filename mismatch")
        require((im["width"], im["height"]) == (old_im["width"], old_im["height"]), "Manual dimensions mismatch")
        target = {k: row[k] for k in TARGET_FIELDS if k in row}
        target.update(
            readout_split=records[vid]["split"], readout_class_holdout=row["category_id"] in holdout_categories,
            eligible_for_parameter_fit=records[vid]["split"] == "fit" and row["category_id"] in fit_categories,
            eligible_for_tuning=records[vid]["split"] == "tune" and row["category_id"] in fit_categories,
            supervision_source="exact original-TAO manual anchor intersection",
            official_tao_annotation_id=original["id"], official_tao_image_id=original["image_id"],
            official_tao_category_id=original["category_id"],
        )
        anchors.append(target)
    anchors.sort(key=lambda x: x["id"])
    by_video = defaultdict(list)
    for anchor in anchors:
        by_video[anchor["video_id"]].append(anchor)
    for vid, rec in records.items():
        rows = by_video[vid]
        require(len(rows) == rec["manual_anchor_boxes"], "Per-video exact-anchor count changed: " + str(vid))
        require(len({x["image_id"] for x in rows}) == rec["manual_anchor_images"], "Per-video anchor image count changed")
    selected_ids = {a["image_id"] for a in anchors}
    grouped = defaultdict(list)
    for iid in selected_ids:
        original = source_images[iid]
        require(original["video_id"] in records and original.get("is_ori") is True, "Invalid selected frame")
        grouped[original["video_id"]].append(original)
    images = []
    for vid in sorted(grouped):
        frames = sorted(grouped[vid], key=lambda im: (im["frame_index"], im["id"]))
        require(len({im["frame_index"] for im in frames}) == len(frames), "Duplicate selected original coordinate")
        for step, original in enumerate(frames):
            im = {k: original[k] for k in IMAGE_FIELDS if k in original}
            require(set(IMAGE_FIELDS) <= set(im), "Missing required image field")
            im.update(source_frame_id=original["frame_id"], frame_id=step, sample_step=step)
            require(canonical_filename(im["file_name"]) == im["file_name"], "Unexpected source relative path")
            require((ROOT / "data_tao/frames" / im["file_name"]).is_file(), "Missing selected frame: " + im["file_name"])
            images.append(im)
    selected_video_ids = set(grouped)
    omitted_video_ids = sorted(set(records) - selected_video_ids)
    require(len(images) == 15283 and len(anchors) == 46908, "Full fit/tune scope changed")
    require(len(selected_video_ids) == 448 and omitted_video_ids == [1720, 1974], "Unexpected empty allocated videos")
    summary = counts(anchors, images, records, fit_categories)
    require(summary["fit"]["anchor_images"] == 13791 and summary["tune"]["anchor_images"] == 1492, "Split frame counts changed")
    require(summary["fit"]["fit_category_boxes"] == 39903 and summary["tune"]["fit_category_boxes"] == 4557, "Split category gates changed")
    info = dict(
        collection_protocol="sparse manual-anchor timeline, GT-free predictions on every selected fit/tune image",
        frame_id="sampled-step index starting at 0 separately for each video",
        frame_index="preserved original frame index; never interpret sample steps as seconds",
        vocabulary="canonical LVIS v1 1203 foreground + background; full-vocabulary rare-column fusion",
        gt_annotations_passed_to_model=False,
        inference_image_field_allowlist=list(IMAGE_FIELDS) + ["sample_step", "source_frame_id"],
        inference_video_field_allowlist=list(VIDEO_FIELDS), inference_category_field_allowlist=list(CATEGORY_FIELDS),
        frequency_prior="fixed public LVIS-v1 frequency, not collection annotation counts",
    )
    base = dict(categories=categories, annotations=[], tracks=[], info=info)
    videos = [{k: source_videos[vid][k] for k in VIDEO_FIELDS if k in source_videos[vid]} for vid in sorted(grouped)]
    inference = dict(base, videos=videos, images=images)
    fit_videos = sorted((vid for vid in grouped if records[vid]["split"] == "fit" and len(grouped[vid]) >= 2),
                        key=lambda vid: hashlib.sha256(("schema-smoke-v1/" + str(vid)).encode()).hexdigest())
    smoke_video_ids = fit_videos[:2]
    smoke = dict(base, videos=[v for v in videos if v["id"] in smoke_video_ids],
                 images=[im for im in images if im["video_id"] in smoke_video_ids and im["sample_step"] < 2])
    require(len(smoke["images"]) == 4, "Smoke requires two within-video transitions and two resets")
    # Reuse only the generic field/timeline contract, never the pilot stage PASS.
    for collection in (inference, smoke):
        require(set(collection) == {"categories", "annotations", "tracks", "info", "images", "videos"}, "Unexpected top-level input")
        require(collection["annotations"] == [] and collection["tracks"] == [], "GT leaked into inference")
        for key, allowed in (("images", info["inference_image_field_allowlist"]),
                             ("videos", VIDEO_FIELDS), ("categories", CATEGORY_FIELDS)):
            require(all(set(row) <= set(allowed) for row in collection[key]), "Inference sidechannel in " + key)
    planned_images = [dict(image_id=im["id"], **{k: v for k, v in im.items() if k != "id"}) for im in images]
    pilot_image_ids = {im["image_id"] for im in plan["minimal_pilot"]["images"] if im["split"] in ("fit", "tune")}
    pilot_anchors = set(plan["minimal_pilot"]["manual_anchor_ctao_annotation_ids"])
    pilot_ids = set(plan["minimal_pilot"]["video_ids"])
    require(pilot_image_ids <= selected_ids, "Pilot fit/tune frames absent from full selection")
    added_videos = selected_video_ids - pilot_ids
    added_anchors = [a for a in anchors if a["id"] not in pilot_anchors]
    selected_source_groups = {records[vid]["source_group"] for vid in selected_video_ids}
    pilot_source_groups = {records[vid]["source_group"] for vid in selected_video_ids & pilot_ids}
    full_plan = dict(
        status="SEALED_FULL_FIT_TUNE_METADATA_ONLY_NO_COLLECTION_RUN",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_plan=str(args.plan), parent_plan_sha256=protected_before[str(args.plan)],
        inputs={"ctao_base": dict(path=str(source), sha256=protected_before[str(source)]),
                "tao_train": dict(path=str(official_source), sha256=protected_before[str(official_source)])},
        selection_rule="All exact original-manual-anchor intersections in the sealed full fit/tune video allocation; no prediction outcomes or test-partition supervision used for selection",
        video_records=[records[vid] for vid in sorted(records)],
        source_group_splits_inherited=True, category_splits_inherited=True,
        readout_fit_category_ids=sorted(fit_categories), readout_category_holdout_ids=sorted(holdout_categories),
        allocation_videos=len(records), selected_nonempty_videos=len(selected_video_ids),
        omitted_zero_anchor_video_ids=omitted_video_ids,
        images=planned_images, manual_anchor_ctao_annotation_ids=sorted(anchor_ids),
        counts=summary, total_images=len(images), total_manual_targets=len(anchors),
        added_vs_pilot_fit_tune=dict(
            images=len(selected_ids - pilot_image_ids), selected_nonempty_videos=len(added_videos),
            source_groups=len(selected_source_groups - pilot_source_groups), manual_targets=len(added_anchors),
            fit_category_targets=sum(a["category_id"] in fit_categories for a in added_anchors)),
        test_video_ids_selected=[], test_gt_frame_maps_built=False, test_prediction_gt_joins=False,
        original_source_jsons_contain_test_partition_labels=True,
        annotation_source_ids_in_inference=False,
        excluded_original_frame_boxes_fit_tune=dict(excluded),
        interpretation="Metadata support is not matched observations, natural-error outcomes, independent sample counts or efficacy evidence",
    )
    targets = dict(
        status="TARGETS_ONLY_NOT_MODEL_INPUT", categories=categories, annotations=anchors,
        readout_fit_category_ids=sorted(fit_categories), readout_category_holdout_ids=sorted(holdout_categories),
        leakage_rule="Only fit-video AND fit-category endpoints may fit parameters; tune-video AND fit-category endpoints may tune; class holdout outcomes stay closed; unmatched observations remain unknown",
        source_provenance="C-TAO annotation ID/image ID/track ID and canonical class remain separate from official old-vocabulary annotation ID/image ID/category ID; unique synset and exact bbox bridge only",
    )
    del ctao, official
    protected_after = {str(path): sha(path) for path in protected_paths}
    require(protected_after == protected_before, "Protected input/source changed during preparation")
    check = dict(
        status="PASS_FULL_FIT_TUNE_METADATA_PREPARATION_SELF_CHECK",
        exact_manual_intersection_recomputed=True, per_video_parent_counts_equal=True,
        image_paths_checked=len(images), metadata_sidechannels_absent=True,
        source_group_splits_disjoint=True, category_splits_disjoint=True,
        test_frames_selected=0, test_gt_maps_built=False, prediction_gt_joins_performed=False,
        original_source_jsons_contain_test_partition_labels=True,
        inference_started=False, training_started=False, model_imported=False, gpu_used=False,
        protected_inputs_before=protected_before, protected_inputs_after=protected_after,
        counts=summary, zero_anchor_allocated_videos=omitted_video_ids,
        independent_full_stage_audit="PENDING_SEPARATE_REVIEWER",
    )
    args.out.mkdir(parents=True)
    for filename, payload in (("full_collection_plan.json", full_plan), ("full_inference.json", inference),
                              ("smoke_inference.json", smoke), ("manual_anchor_targets.json", targets),
                              ("preparation_self_check.json", check)):
        write(args.out / filename, payload)
    executable = "/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python"
    command = [executable, str(ROOT / "tmp_analysis/collect_base_error_pilot.py"),
               "--annotation", str(args.out / "full_inference.json"), "--out", str(args.out / "full_collect_v1")]
    readme = (
        "# Full fit/tune sparse-anchor collection\n\n"
        "Prepared metadata only; GPU inference and new fitting have not started. "
        "The sealed parent allocation has 450 videos; 448 contain selected anchors. "
        "Videos 1720 and 1974 have zero exact anchors and are excluded from inference.\n\n"
        "Inference schema: full_inference.json and smoke_inference.json contain only whitelisted "
        "image/video/category fields, empty annotations and empty tracks. Local image and video IDs "
        "stay original; frame_id and sample_step count sparse selected steps separately per video. "
        "frame_index and source_frame_id preserve source timing and are not seconds. "
        "All emitted predictions remain eligible for raw cache storage.\n\n"
        "Supervision schema: manual_anchor_targets.json contains C-TAO manual annotation/image/track/category "
        "IDs, bbox, source split/class gates and explicit official_tao_annotation_id, official_tao_image_id "
        "and official_tao_category_id. The official category ID uses the old vocabulary; compare by the "
        "unique synset bridge, never direct numeric equality. None of these supervision fields reach the model.\n\n"
        "full_collection_plan.json records source hashes, inherited splits, selected frame/source IDs and "
        "manual-anchor IDs. preparation_self_check.json is the producer's metadata check, not an independent "
        "collection audit. The existing pilot stage verifier and support builder hardcode the old pilot "
        "scope; use separately reviewed full-stage adapters. Test outcomes remain closed.\n\n"
        "Reviewable collection command (not executed by preparation):\n\n```bash\n" + shlex.join(command) + "\n```\n"
    )
    with (args.out / "README.md").open("x") as handle:
        handle.write(readme)
    files = {p.name: sha(p) for p in sorted(args.out.iterdir())}
    manifest = dict(
        status="PREPARED_NO_INFERENCE_RUN", collection_scope="FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES",
        plan=str(args.plan), plan_sha256=protected_before[str(args.plan)],
        full_collection_plan=str(args.out / "full_collection_plan.json"),
        full_collection_plan_sha256=files["full_collection_plan.json"],
        source=str(source), source_sha256=protected_before[str(source)],
        official_source=str(official_source), official_source_sha256=protected_before[str(official_source)],
        script=str(Path(__file__).resolve()), script_sha256=protected_before[str(Path(__file__).resolve())],
        allocation_videos=len(records), images=len(images), videos=len(selected_video_ids),
        source_groups=len(selected_source_groups), manual_targets=len(anchors),
        fit_eligible_manual_targets=39903, tune_eligible_manual_targets=4557,
        smoke_videos=sorted(smoke_video_ids), timeline=info, files=files,
        independent_full_stage_audit="PENDING_SEPARATE_REVIEWER", inference_started=False, training_started=False,
        collection_command_argv=command,
    )
    write(args.out / "preparation_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("timeline", "files")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
