#!/usr/bin/env python3
"""Independent metadata and completed-run audits for full fit/tune collection.

Reconstruct manual anchors from the sealed original sources. The historical
pilot stage verifier remains frozen; its pure helpers are reused, not its
pilot-size stage check. No model import, GPU, fitting, or prediction/GT scoring.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import runpy

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "work/covtrack_base_error_20261023"
UTILITY = COPY / "ovtrack/base_error_dump.py"
PILOT_VERIFIER = ROOT / "tmp_analysis/verify_base_error_collection.py"
PILOT_VERIFIER_SHA = "872d9ade3f416025fff07cfbcc0efc63334379093f4c066cf6fad9e84258ca5c"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    return json.loads(Path(path).read_text())


require(sha(PILOT_VERIFIER) == PILOT_VERIFIER_SHA, "Frozen pilot verifier changed")
_pilot = runpy.run_path(str(PILOT_VERIFIER))
check_inference, assignment = _pilot["check_inference"], _pilot["assignment"]
METADATA_FIELDS = _pilot["METADATA_FIELDS"]


def reconstruct_full_anchors(plan, source, official):
    """Independent source join; old taxonomy numeric IDs are never compared."""
    records = {int(v["video_id"]): v for v in plan["full_metadata_split"]["videos"]}
    require(len(records) == len(plan["full_metadata_split"]["videos"]), "Duplicate planned video")
    source_images = {int(i["id"]): i for i in source["images"]}
    official_images = {int(i["id"]): i for i in official["images"]}
    source_categories = {int(c["id"]): c for c in source["categories"]}
    require(set(source_categories) == set(range(1, 1204)), "Canonical source vocabulary changed")
    synsets = defaultdict(list)
    for cid, c in source_categories.items():
        require(bool(c.get("synset")), "Missing canonical synset")
        synsets[c["synset"]].append(cid)
    old_categories = {int(c["id"]): c for c in official["categories"]}
    category_map = {cid: synsets.get(c.get("synset"), []) if c.get("synset") else [] for cid, c in old_categories.items()}
    official_coordinates = {}; original = {}
    for image in official_images.values():
        key = (int(image["video_id"]), int(image["frame_index"]))
        require(key not in official_coordinates, "Duplicate original official frame coordinate")
        official_coordinates[key] = image
    for annotation in official["annotations"]:
        image = official_images[int(annotation["image_id"])]
        require(int(annotation["video_id"]) == int(image["video_id"]), "Original annotation/video mismatch")
        key = (int(annotation["video_id"]), int(image["frame_index"]), int(annotation["track_id"]))
        require(key not in original, "Duplicate original official track/frame anchor")
        original[key] = annotation
    all_anchors = []; selected = []
    for annotation in source["annotations"]:
        image = source_images[int(annotation["image_id"])]
        if image.get("is_ori") is not True:
            continue
        key = (int(annotation["video_id"]), int(image["frame_index"]), int(annotation["track_id"]))
        old = original.get(key)
        if old is None or category_map[int(old["category_id"])] != [int(annotation["category_id"])]:
            continue
        if len(annotation["bbox"]) != 4 or not np.allclose(annotation["bbox"], old["bbox"], rtol=0, atol=1e-6):
            continue
        require(int(annotation["video_id"]) == int(image["video_id"]), "Source annotation/video mismatch")
        require(source_categories[int(annotation["category_id"])]["frequency"] in ("f", "c"), "Nonbase exact manual target")
        all_anchors.append(annotation)
        if records[int(annotation["video_id"])]["split"] in ("fit", "tune"):
            original_image = official_coordinates[(int(image["video_id"]), int(image["frame_index"]))]
            require(image["file_name"].replace("\\", "/") == original_image["file_name"].replace("\\", "/"), "Original filename differs for selected source anchor")
            require(all(image[k] == original_image[k] for k in ("width", "height")), "Original dimensions differ")
            selected.append(annotation)
    require(len(all_anchors) == 52911, "Full source manual-anchor reconstruction changed")
    require(len(selected) == 46908, "Full fit/tune manual target reconstruction changed")
    return records, source_images, all_anchors, selected, original


def verify_full_stage(stage):
    manifest = load(stage / "preparation_manifest.json")
    require(manifest["status"] == "PREPARED_NO_INFERENCE_RUN", "Unexpected full-stage status")
    require(manifest["plan_sha256"] == "f37f1ecc52734953ea5269407ab0aa8013e650e97bed4dde3cdd399a11e13ba5", "Original sealed plan pointer changed")
    for path_key, hash_key in (("plan", "plan_sha256"), ("source", "source_sha256"), ("official_source", "official_source_sha256")):
        require(sha(manifest[path_key]) == manifest[hash_key], "Full-stage source SHA mismatch: " + path_key)
    require(sha(ROOT / "tmp_analysis/prepare_base_error_full_collection.py") == manifest["script_sha256"], "Full preparation script changed")
    for name, digest in manifest["files"].items():
        require(sha(stage / name) == digest, "Full-stage file SHA mismatch: " + name)
    require({"full_collection_plan.json", "full_inference.json", "smoke_inference.json", "manual_anchor_targets.json", "preparation_self_check.json", "README.md"} == set(manifest["files"]), "Full-stage file set changed")
    full, images = check_inference(stage / "full_inference.json")
    smoke, smoke_images = check_inference(stage / "smoke_inference.json")
    require(len(images) == manifest["images"] == 15283 and len(full["videos"]) == manifest["videos"] == 448, "Full fit/tune scope mismatch")
    require(len(smoke_images) == 4 and len(smoke["videos"]) == 2 and all(im == images[iid] for iid, im in smoke_images.items()), "Smoke not a two-video subset of full input")
    plan = load(manifest["plan"])
    require(manifest["source_sha256"] == plan["inputs"]["ctao_base"]["sha256"] and manifest["official_source_sha256"] == plan["inputs"]["tao_train"]["sha256"], "Original sealed source lineage changed")
    source = load(manifest["source"]); official = load(manifest["official_source"])
    category_fields = {"id", "name", "synonyms", "synset", "def", "frequency"}
    expected_categories = [{k: v for k, v in c.items() if k in category_fields} for c in source["categories"]]
    require(full["categories"] == smoke["categories"] == expected_categories, "Full inference vocabulary metadata differs from source")
    source_videos = {v["id"]: v for v in source["videos"]}
    video_fields = {"id", "name", "width", "height", "fps"}
    require(full["videos"] == [{k: v for k, v in source_videos[vid].items() if k in video_fields} for vid in sorted({im["video_id"] for im in images.values()})], "Full video metadata differs from source")
    require(smoke["videos"] == [v for v in full["videos"] if v["id"] in manifest["smoke_videos"]], "Full smoke video metadata differs")
    records, source_images, all_anchors, selected, original_anchors = reconstruct_full_anchors(plan, source, official)
    selected_ids = {int(a["id"]) for a in selected}; selected_images = {int(a["image_id"]) for a in selected}
    require(len(selected_ids) == len(selected) and set(images) == selected_images, "Full input not exact manual-anchor image union")
    selected_videos = {int(images[iid]["video_id"]) for iid in images}
    require(selected_videos == {int(a["video_id"]) for a in selected}, "Full input video set mismatch")
    grouped = defaultdict(list)
    for iid, im in images.items():
        original = source_images[iid]
        expected = {k: original[k] for k in ("id", "video_id", "file_name", "width", "height", "frame_index") if k in original}
        expected["source_frame_id"] = original["frame_id"]
        expected["frame_id"] = im["frame_id"]; expected["sample_step"] = im["sample_step"]
        require(im == expected and original.get("is_ori") is True, "Full input original geometry/timeline identity changed")
        require(records[im["video_id"]]["split"] in ("fit", "tune"), "Test video in full inference")
        grouped[im["video_id"]].append(im)
    for vid, ims in grouped.items():
        require(ims == sorted(ims, key=lambda im: (im["frame_index"], im["id"])), "Full source timeline reordered")
    allocated = {vid for vid, r in records.items() if r["split"] in ("fit", "tune")}
    empty = sorted(allocated-selected_videos)
    require(len(allocated) == 450 and empty == [1720, 1974], "Anchor-empty allocated video exclusions changed")
    groups = defaultdict(set)
    for r in records.values():
        groups[r["source_group"]].add(r["split"])
    require(all(len(x) == 1 for x in groups.values()), "Source groups cross sealed splits")
    fit_categories = set(plan["vocabulary"]["readout_fit_category_ids"]); holdout = set(plan["vocabulary"]["readout_category_holdout_ids"])
    require(not fit_categories & holdout and len(fit_categories) == 146 and len(holdout) == 36, "Readout class holdout changed")
    targets = load(stage / "manual_anchor_targets.json")
    require(targets["status"] == "TARGETS_ONLY_NOT_MODEL_INPUT", "Targets not separated")
    require(targets["categories"] == expected_categories, "Full target category bridge metadata differs")
    require(set(targets["readout_fit_category_ids"]) == fit_categories and set(targets["readout_category_holdout_ids"]) == holdout, "Full target category split mismatch")
    source_anchors = {int(a["id"]): a for a in selected}
    require(len(targets["annotations"]) == len(selected_ids) and {a["id"] for a in targets["annotations"]} == selected_ids, "Full target manual IDs differ from independent reconstruction")
    counts = defaultdict(Counter)
    for a in targets["annotations"]:
        original = source_anchors[a["id"]]; part = records[a["video_id"]]["split"]
        target_fields = {"id", "image_id", "video_id", "track_id", "category_id", "bbox", "area", "iscrowd"}
        expected_source = {k: v for k, v in original.items() if k in target_fields}
        require(all(a[k] == v for k, v in expected_source.items()), "Full target source annotation changed")
        require(set(a) == set(expected_source) | {"readout_split", "readout_class_holdout", "eligible_for_parameter_fit", "eligible_for_tuning", "supervision_source", "official_tao_annotation_id", "official_tao_image_id", "official_tao_category_id"}, "Unreviewed target field set")
        original_image = source_images[a["image_id"]]
        old = original_anchors[(a["video_id"], original_image["frame_index"], a["track_id"])]
        require((a["official_tao_annotation_id"], a["official_tao_image_id"], a["official_tao_category_id"]) == (old["id"], old["image_id"], old["category_id"]), "Explicit old-TAO source bridge incorrect")
        eligible = a["category_id"] in fit_categories
        require(a["readout_split"] == part and a["readout_class_holdout"] == (not eligible) and a["eligible_for_parameter_fit"] == (part == "fit" and eligible), "Full target eligibility changed")
        require(a["eligible_for_tuning"] == (part == "tune" and eligible) and a["supervision_source"] == "exact original-TAO manual anchor intersection", "Full tuning/manual provenance gate")
        counts[part]["manual_targets"] += 1; counts[part]["eligible_targets"] += int(eligible)
    require(counts["fit"]["manual_targets"] == 42020 and counts["tune"]["manual_targets"] == 4888 and counts["fit"]["eligible_targets"] == manifest["fit_eligible_manual_targets"] == 39903 and counts["tune"]["eligible_targets"] == manifest["tune_eligible_manual_targets"] == 4557, "Full supervised support counts changed")
    full_plan = load(stage / "full_collection_plan.json")
    require(sha(stage / "full_collection_plan.json") == manifest["full_collection_plan_sha256"], "Full plan SHA binding")
    require(Path(manifest["full_collection_plan"]).resolve() == stage / "full_collection_plan.json", "Wrong full plan pointer")
    require(full_plan["status"] == "SEALED_FULL_FIT_TUNE_METADATA_ONLY_NO_COLLECTION_RUN" and full_plan["parent_plan_sha256"] == manifest["plan_sha256"], "Full plan parent/status")
    require(full_plan["inputs"] == {k: plan["inputs"][k] for k in ("ctao_base", "tao_train")}, "Full plan original inputs")
    require(full_plan["video_records"] == [records[vid] for vid in sorted(allocated)], "Full plan split records altered")
    require(manifest["source_groups"] == len({records[v]["source_group"] for v in selected_videos}) == 383, "Full source-group coverage changed")
    require(full_plan["readout_fit_category_ids"] == sorted(fit_categories) and full_plan["readout_category_holdout_ids"] == sorted(holdout), "Full plan category split")
    require(full_plan["images"] == [dict(image_id=im["id"], **{k: v for k, v in im.items() if k != "id"}) for im in full["images"]], "Full plan/inference timeline mismatch")
    require(full_plan["manual_anchor_ctao_annotation_ids"] == sorted(selected_ids), "Full plan anchor IDs")
    require(full_plan["allocation_videos"] == manifest["allocation_videos"] == 450 and full_plan["selected_nonempty_videos"] == 448 and full_plan["omitted_zero_anchor_video_ids"] == empty and full_plan["total_images"] == 15283 and full_plan["total_manual_targets"] == manifest["manual_targets"] == 46908, "Full plan totals")
    require(full_plan["test_video_ids_selected"] == [] and full_plan["test_gt_frame_maps_built"] is False and full_plan["test_prediction_gt_joins"] is False, "Full plan test boundary")
    require(manifest["inference_started"] is False and manifest["training_started"] is False, "Preparation inference/training flags")
    reconstructed_counts = {}
    for part in ("fit", "tune"):
        anns = [a for a in selected if records[a["video_id"]]["split"] == part]
        ims = [im for im in images.values() if records[im["video_id"]]["split"] == part]
        vids = {im["video_id"] for im in ims}; per_class = defaultdict(list)
        for a in anns:
            if a["category_id"] in fit_categories:
                per_class[a["category_id"]].append(a)
        reconstructed_counts[part] = dict(allocation_videos=sum(records[v]["split"] == part for v in allocated), selected_nonempty_videos=len(vids), source_groups=len({records[v]["source_group"] for v in vids}), anchor_images=len(ims), manual_anchor_boxes=len(anns), fit_category_boxes=sum(map(len, per_class.values())), heldout_category_boxes=len(anns)-sum(map(len, per_class.values())), fit_categories_with_boxes=sorted(per_class), fit_category_support={str(cid): dict(manual_boxes=len(xs), videos=len({a["video_id"] for a in xs}), source_groups=len({records[a["video_id"]]["source_group"] for a in xs}), gt_tracks=len({(a["video_id"], a["track_id"]) for a in xs})) for cid, xs in sorted(per_class.items())})
        for vid in allocated:
            if records[vid]["split"] != part:
                continue
            va = [a for a in anns if a["video_id"] == vid]
            require(len(va) == records[vid]["manual_anchor_boxes"] and len({a["image_id"] for a in va}) == records[vid]["manual_anchor_images"], "Parent per-video anchor counts")
    require(full_plan["counts"] == reconstructed_counts, "Full plan class/source support counts")
    self_check = load(stage / "preparation_self_check.json")
    require(self_check["status"] == "PASS_FULL_FIT_TUNE_METADATA_PREPARATION_SELF_CHECK" and self_check["counts"] == reconstructed_counts, "Producer self-check mismatch")
    require(self_check["protected_inputs_before"] == self_check["protected_inputs_after"], "Preparation protected inputs changed")
    for path, digest in self_check["protected_inputs_after"].items():
        require(sha(path) == digest, "Preparation protected input changed: " + path)
    require(self_check["image_paths_checked"] == 15283 and self_check["test_frames_selected"] == 0 and self_check["prediction_gt_joins_performed"] is False, "Producer path/test claims")
    for im in images.values():
        require((ROOT / "data_tao/frames" / im["file_name"]).is_file(), "Full selected image missing")
    return dict(status="PASS_INDEPENDENT_FULL_COLLECTION_METADATA_AUDIT", images=len(images), videos=len(selected_videos),
                allocation_videos=len(allocated), allocated_videos_without_manual_anchors=empty,
                reconstructed_all_source_manual_anchors=len(all_anchors), selected_fit_tune_manual_targets=len(selected),
                manual_targets_by_split={k: dict(v) for k, v in counts.items()},
                per_split_image_counts={k: v["anchor_images"] for k, v in reconstructed_counts.items()},
                no_test_images_or_supervised_targets_selected=True, source_group_splits_disjoint=True,
                canonical_full_vocabulary_and_gt_free_whitelists=True, sampled_step_timeline_original_identity_verified=True,
                annotation=str(stage / "full_inference.json"), annotation_sha256=sha(stage / "full_inference.json"),
                preparation_manifest_sha256=sha(stage / "preparation_manifest.json")), manifest


# Runtime implementation copied from frozen pilot verifier SHA 872d9ade3f416025fff07cfbcc0efc63334379093f4c066cf6fad9e84258ca5c.
# Only accepted inference filename changes; full stage has its own audit above.
def verify_run(stage, run, preparation):
    manifest = load(run / "run_manifest.json")
    annotation = Path(manifest["annotation"]).resolve()
    require(annotation.parent == stage and annotation.name in ("full_inference.json", "smoke_inference.json"), "Run annotation outside audited stage")
    require(sha(annotation) == manifest["annotation_sha256"] == preparation["files"][annotation.name], "Run annotation SHA mismatch")
    metadata, images = check_inference(annotation)
    require(manifest["expected_images"] == len(images) and manifest["expected_videos"] == len(metadata["videos"]), "Run declared scope mismatch")
    require(manifest["model_training"] is False and manifest["new_parameter_fit"] is False and manifest["supervision_file_opened"] is False, "Training/GT input during collection")
    require(manifest["aggregation_key"] == ["video_id", "local_track_id"], "Track IDs need video-local keys")
    require(sha(manifest["runner"]) == manifest["runner_sha256"], "Runner source changed")
    require(sha(manifest["source_config"]) == manifest["source_config_sha256"], "Original config changed")
    require(sha(run / "collection_config.py") == manifest["config_sha256"], "Effective config changed")
    require(sha(manifest["checkpoint"]) == manifest["checkpoint_sha256"] == "e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c", "Frozen checkpoint changed")
    code_root = Path(manifest["code_root"]).resolve()
    require(code_root == COPY, "Wrong isolated source")
    current_code = {str(p.relative_to(code_root)): sha(p) for p in sorted((code_root / "ovtrack").rglob("*.py"))}
    require(current_code == manifest["source_hashes"], "Isolated code changed during/after run")
    require(sha(COPY / "ovtrack/models/roi_heads/class_name.py") == sha(ROOT / "work/covtrack_base/ovtrack/models/roi_heads/class_name.py"), "Canonical class source changed")
    model = assignment(run / "collection_config.py", "model")
    roi = model["roi_head"]
    require(roi["type"] == "OVTrackRoIHeadUncertainty" and roi["bbox_head"]["num_classes"] == 1203, "Inactive/wrong ROI head or vocabulary")
    for name, expected in (("only_validation_categories", False), ("only_test_categories", False), ("custom_classes", False),
                           ("ensemble", True), ("fix_bg", False), ("use_special_prompt", False), ("record_probs_info", False),
                           ("use_cls_static_ratio", True), ("use_motion_static_ratio", True)):
        require(roi[name] == expected, "Wrong active config: " + name)
    require(roi.get("fixed_lambda") is None and model["tracker"]["vis"] is False and model["tracker"]["memo_frames"] == 50, "Fusion/tracker config changed")
    mask = np.zeros(1204, dtype=bool)
    mask[np.asarray(assignment(COPY / "ovtrack/models/roi_heads/class_name.py", "lvis_novel_label_ids"), np.int64)] = True
    require(int(mask.sum()) == 337 and not mask[-1], "Invalid canonical full-LVIS mask")
    powers = np.where(mask, 1. / 3, 2. / 3)
    ready = load(run / "model_ready.json")
    require(ready["actual_novel_mask_count"] == 337 and ready["text_features_shape"] == [1203, 512], "Real model vocabulary readiness mismatch")
    receipt = load(run / "collection_receipt.json")
    require(receipt["status"] == "COMPLETE_GT_FREE_COLLECTION_PENDING_INDEPENDENT_AUDIT", "Run has not completed")
    for key, file in (("run_manifest_sha256", "run_manifest.json"), ("events_sha256", "frame_execution.jsonl"), ("dump_inventory_sha256", "dump_inventory.json")):
        require(receipt[key] == sha(run / file), "Sealed run SHA mismatch: " + file)
    require(receipt["protected_sources_unchanged"] and receipt["source_code_unchanged"] and receipt["training_started"] is False, "Protection/training receipt failed")
    before, after = load(run / "protected_before.json"), load(run / "protected_after.json")
    require(before == after and before, "Protected before/after mismatch")
    for path, spec in after.items():
        p = Path(path)
        require(p.exists() == spec["exists"] and (sha(p) if p.exists() else None) == spec["sha256"], "Protected file changed: " + path)
    inventory = load(run / "dump_inventory.json")
    actual_inventory = {p.name: sha(p) for p in sorted((run / "dump").iterdir())}
    require(actual_inventory == inventory, "Dump inventory SHA/file set mismatch")
    require(len(inventory) == 2 * len(images), "Missing frame/receipt or unexpected dump file")
    events = [json.loads(line) for line in (run / "frame_execution.jsonl").read_text().splitlines()]
    require(len(events) == len(images), "Missing real execution record")
    api = runpy.run_path(str(UTILITY))
    observations = emitted = empty = resets = 0; max_error = 0.; frame_files = []; last_totals = {}
    for index, (im, event) in enumerate(zip(metadata["images"], events)):
        iid, vid, step = im["id"], im["video_id"], im["frame_id"]
        require(event["index"] == index and all(event[k] == im[k] for k in ("video_id", "frame_id", "frame_index")), "Execution order/identity mismatch")
        require(event["image_id"] == iid and event["emitted_output_exact"] is True, "Actual model output equality failed")
        require(event["tracker_reset_observed"] == (step == 0), "Real tracker object reset mismatch")
        if step != 0:
            require(event["previous_tracker_total"] == last_totals[vid], "Within-video tracker state mismatch")
        last_totals[vid] = event["tracker_num_tracklets"]
        key = "v%d_i%d_f%d" % (vid, iid, step)
        npz = run / "dump" / (key + ".npz"); frame_receipt_path = run / "dump" / (key + ".receipt.json")
        require(Path(event["npz"]).resolve() == npz.resolve() and event["npz_sha256"] == inventory[npz.name], "Execution NPZ binding mismatch")
        with np.load(npz, allow_pickle=False) as saved:
            data = dict(saved)
        checked = api["validate_payload"](data)
        require(np.array_equal(data["posterior_category_ids"], np.r_[np.arange(1, 1204), -1]), "Noncanonical posterior column IDs")
        require(np.array_equal(data["actual_fusion_mask"], mask) and np.array_equal(data["fusion_text_exponents"], powers), "Noncanonical actual full fusion mask/exponents")
        require(data["fusion_mode"].item() == "asymmetric_actual_mask" and data["vocabulary_name"].item() == "LVIS_1203_one_based_category_ids", "Wrong fusion mode/vocabulary")
        require(data["fusion_regime"].item() == "canonical_full_LVIS1203_zero_based_novel_ids_bg_false", "Wrong fusion regime")
        require(not bool(data["only_validation_categories"]) and not bool(data["only_test_categories"]) and bool(data["ensemble"]), "Subset/text-only run")
        require(int(data["image_id"]) == iid and int(data["video_id"]) == vid and int(data["frame_id"]) == step and int(data["frame_index"]) == im["frame_index"], "Dump frame identity mismatch")
        require(data["annotation_sha256"].item() == manifest["annotation_sha256"] and Path(data["annotation_path"].item()).resolve() == annotation, "Dump annotation binding mismatch")
        identity = json.loads(data["image_metadata_json"].item())
        require(set(identity) <= METADATA_FIELDS and identity == {k: v for k, v in im.items() if k in METADATA_FIELDS}, "Dump image metadata sidechannel/provenance mismatch")
        require(data["canonical_filename"].item() == api["canonical_filename"](im["file_name"]), "Filename mismatch")
        require(np.all(data["p_image"][:, -1] == 0) and np.all(data["p_fused"][:, -1] == 0), "Image/fused background must be zero in actual regime")
        tids = data["track_ids"]; live_ids = np.sort(tids[tids >= 0])
        require(np.all(tids >= -1), "Unknown negative track-ID sentinel")
        if step == 0:
            require(np.array_equal(live_ids, np.arange(event["tracker_num_tracklets"])), "First-frame IDs do not restart from fresh tracker")
            resets += 1
        r = load(frame_receipt_path)
        require(r["status"] == "PASS_FRAME_DUMP" and r["schema_version"] == 2 and r["file"] == npz.name, "Frame receipt status/schema mismatch")
        require(r["sha256"] == inventory[npz.name] and r["bytes"] == npz.stat().st_size, "Frame NPZ SHA/byte mismatch")
        require(r["annotation_sha256"] == manifest["annotation_sha256"] and r["filename"] == data["filename"].item(), "Frame receipt annotation/filename mismatch")
        require((r["video_id"], r["image_id"], r["frame_id"]) == (vid, iid, step), "Frame receipt identity mismatch")
        for field, value in checked.items():
            require(r[field] == value and event[field] == value, "Schema/receipt/execution mismatch: " + field)
        observations += checked["observations"]; emitted += checked["emitted_observations"]; empty += int(checked["empty_frame"])
        max_error = max(max_error, checked["fusion_max_abs_error"])
        frame_files.append(dict(file=npz.name, sha256=inventory[npz.name], observations=checked["observations"]))
    require(resets == len(metadata["videos"]) >= 2, "Need at least two real video resets")
    require(receipt["completed_images"] == len(images) and receipt["completed_videos"] == resets and receipt["retained_observations"] == observations and receipt["emitted_observations"] == emitted, "Collection totals mismatch")
    return dict(status="PASS_INDEPENDENT_GT_FREE_COLLECTION_AUDIT", frames=len(images), videos=resets,
                real_tracker_object_resets_verified=resets, actual_emitted_outputs_exact=True,
                retained_observations=observations, emitted_observations=emitted,
                unassigned_observations_retained=observations-emitted, zero_observation_frames=empty,
                posterior_columns=1204, canonical_fusion_mask_true_count=337,
                fusion_max_abs_error=max_error, all_npz_and_receipt_hashes_verified=True,
                protected_files_verified=len(after), isolated_code_hashes_verified=len(current_code),
                annotation=str(annotation), annotation_sha256=manifest["annotation_sha256"],
                run_manifest_sha256=sha(run / "run_manifest.json"), collection_receipt_sha256=sha(run / "collection_receipt.json"),
                frame_files=frame_files, limitations=["Sparse selected-anchor timeline; memo50 counts selected steps, not dense raw frames", "Unmatched targets remain unknown", "Readout class holdout is not detector class/video independence", "No learner, target matcher, calibration or efficacy is validated by this collection audit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); stage = args.stage.resolve()
    report, preparation = verify_full_stage(stage)
    if args.run:
        run = args.run.resolve()
        if run.is_file():
            require(run.name == "run_manifest.json", "--run file must be run_manifest.json")
            run = run.parent
        runtime = verify_run(stage, run, preparation)
        runtime["status"] = "PASS_INDEPENDENT_FULL_GT_FREE_COLLECTION_AUDIT"
        report = dict(stage=report, **runtime)
    report["independent_validator_sha256"] = sha(__file__)
    report["frozen_pilot_helper_sha256"] = PILOT_VERIFIER_SHA
    report["shared_schema_validator_sha256"] = sha(UTILITY)
    if args.output:
        require(not args.output.exists(), "Refuse to overwrite full audit report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
