#!/usr/bin/env python3
"""Independent CPU audit of a prepared stage and completed GT-free collection.

Use --stage DIR alone for preparation audit, and add --run DIR for a sealed
real-run audit. This program never imports the model, uses a GPU, matches GT,
fits parameters, or writes any source/preparation/run file.
"""
import argparse
import ast
from collections import defaultdict
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
IMAGE_FIELDS = {"id", "video_id", "file_name", "width", "height", "frame_id",
                "frame_index", "sample_step", "source_frame_id"}
VIDEO_FIELDS = {"id", "name", "width", "height", "fps"}
CATEGORY_FIELDS = {"id", "name", "synonyms", "synset", "def", "frequency"}
METADATA_FIELDS = IMAGE_FIELDS - {"sample_step"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def literal(node):
    """Read mmcv's exported literal config without executing Python code."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        require(not node.args and all(k.arg is not None for k in node.keywords), "Nonliteral config dict")
        return {k.arg: literal(k.value) for k in node.keywords}
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [literal(x) for x in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def assignment(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return literal(node.value)
    raise AssertionError("Missing assignment: " + name)


def check_inference(path):
    data = load(path)
    require(set(data) == {"images", "videos", "categories", "annotations", "tracks", "info"}, "Unexpected inference top-level field")
    require(data["annotations"] == [] and data["tracks"] == [], "GT annotations/tracks in inference JSON")
    require(set(data["info"]["inference_image_field_allowlist"]) == IMAGE_FIELDS, "Unreviewed image allowlist")
    require(set(data["info"]["inference_video_field_allowlist"]) == VIDEO_FIELDS, "Unreviewed video allowlist")
    require(set(data["info"]["inference_category_field_allowlist"]) == CATEGORY_FIELDS, "Unreviewed category allowlist")
    require(data["info"]["gt_annotations_passed_to_model"] is False, "GT input flag")
    require([c["id"] for c in data["categories"]] == list(range(1, 1204)), "Not canonical full category order")
    for category in data["categories"]:
        require(set(category) <= CATEGORY_FIELDS, "GT aggregate or unreviewed category field: " + str(set(category) - CATEGORY_FIELDS))
    videos = {v["id"]: v for v in data["videos"]}
    require(len(videos) == len(data["videos"]), "Duplicate video ID")
    require(all(set(v) <= VIDEO_FIELDS for v in videos.values()), "GT/unreviewed video metadata")
    images = {}; previous = None; completed = set(); steps = defaultdict(int)
    for image in data["images"]:
        require(set(image) <= IMAGE_FIELDS, "GT/unreviewed image metadata")
        iid, vid, step = image["id"], image["video_id"], image["frame_id"]
        require(iid not in images and vid in videos, "Duplicate image or unknown video")
        require(step == image["sample_step"] == steps[vid], "Noncontiguous sampled-step timeline")
        if vid != previous:
            require(vid not in completed and step == 0, "Video interleaving or missing frame0")
            if previous is not None:
                completed.add(previous)
        previous = vid; steps[vid] += 1; images[iid] = image
    require(set(steps) == set(videos) and len(images) > 0, "Missing video support")
    return data, images


def verify_stage(stage):
    manifest = load(stage / "preparation_manifest.json")
    require(manifest["status"] == "PREPARED_NO_INFERENCE_RUN", "Unexpected preparation status")
    require(sha(manifest["plan"]) == manifest["plan_sha256"], "Plan SHA changed")
    require(sha(manifest["source"]) == manifest["source_sha256"], "Original C-TAO source SHA changed")
    require(sha(ROOT / "tmp_analysis/prepare_base_error_collection.py") == manifest["script_sha256"], "Preparation script changed")
    require(set(manifest["files"]) == {"pilot_inference.json", "smoke_inference.json", "manual_anchor_targets.json"}, "Unreviewed stage file set")
    for name, digest in manifest["files"].items():
        require(sha(stage / name) == digest, "Stage SHA mismatch: " + name)
    pilot, images = check_inference(stage / "pilot_inference.json")
    smoke, smoke_images = check_inference(stage / "smoke_inference.json")
    require(len(images) == manifest["images"] == 1299 and len(pilot["videos"]) == manifest["videos"] == 41, "Pilot scope changed")
    require(len(smoke_images) == 4 and len(smoke["videos"]) == 2, "Smoke must exercise two videos and two transitions")
    require({v["id"] for v in smoke["videos"]} == set(manifest["smoke_videos"]), "Smoke video mismatch")
    require(all(im == images[iid] and im["sample_step"] < 2 for iid, im in smoke_images.items()), "Smoke image mismatch")
    plan = load(manifest["plan"])
    require(plan["metadata_validation"]["status"] == "PASS_METADATA_ONLY", "Unvalidated plan")
    planned_images = {im["image_id"]: im for im in plan["minimal_pilot"]["images"]}
    require(set(planned_images) == set(images), "Plan/pilot image set mismatch")
    for iid, im in images.items():
        original = planned_images[iid]
        require(all(im[k] == original[k] for k in ("video_id", "frame_index", "source_frame_id", "file_name")), "Original timeline/provenance changed")
    grouped = defaultdict(set)
    video_records = {v["video_id"]: v for v in plan["full_metadata_split"]["videos"]}
    for video in video_records.values():
        grouped[video["source_group"]].add(video["split"])
    require(all(len(s) == 1 for s in grouped.values()), "Source group crosses readout split")
    fit = set(plan["vocabulary"]["readout_fit_category_ids"])
    holdout = set(plan["vocabulary"]["readout_category_holdout_ids"])
    require(not fit & holdout and len(fit) == 146 and len(holdout) == 36, "Category holdout changed")
    targets = load(stage / "manual_anchor_targets.json")
    require(set(targets["readout_fit_category_ids"]) == fit and set(targets["readout_category_holdout_ids"]) == holdout, "Target category gate mismatch")
    require(targets["status"] == "TARGETS_ONLY_NOT_MODEL_INPUT", "Targets not separated")
    annotations = targets["annotations"]
    require({a["id"] for a in annotations} == set(plan["minimal_pilot"]["manual_anchor_ctao_annotation_ids"]), "Manual anchor set mismatch")
    require(len(annotations) == manifest["manual_targets"] == 3953, "Target count mismatch")
    require({a["image_id"] for a in annotations} == set(images), "Target image mismatch")
    for row in annotations:
        part = video_records[row["video_id"]]["split"]
        eligible = part == "fit" and row["category_id"] in fit
        require(row["readout_split"] == part and row["readout_class_holdout"] == (row["category_id"] in holdout), "Target split/holdout gate mismatch")
        require(row["eligible_for_parameter_fit"] == eligible, "Ineligible class/video admitted for fit")
    eligible = sum(a["eligible_for_parameter_fit"] for a in annotations)
    require(eligible == manifest["fit_eligible_manual_targets"] == 2138, "Eligible target count mismatch")
    return dict(status="PASS_METADATA_ONLY_COLLECTION_STAGE", images=len(images), videos=len(pilot["videos"]),
                smoke_images=len(smoke_images), manual_targets=len(annotations), fit_eligible_manual_targets=eligible,
                image_video_category_sidechannels_absent=True, source_group_splits_disjoint=True,
                fit_holdout_categories_disjoint=True, stage_manifest_sha256=sha(stage / "preparation_manifest.json")), manifest


def verify_run(stage, run, preparation):
    manifest = load(run / "run_manifest.json")
    annotation = Path(manifest["annotation"]).resolve()
    require(annotation.parent == stage and annotation.name in ("pilot_inference.json", "smoke_inference.json"), "Run annotation outside audited stage")
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
    report, preparation = verify_stage(stage)
    if args.run:
        run = args.run.resolve()
        if run.is_file():
            require(run.name == "run_manifest.json", "--run file must be run_manifest.json")
            run = run.parent
        report = dict(stage=report, **verify_run(stage, run, preparation))
    report["independent_validator_sha256"] = sha(__file__)
    report["shared_schema_validator_sha256"] = sha(UTILITY)
    if args.output:
        require(not args.output.exists(), "Refuse to overwrite audit report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
