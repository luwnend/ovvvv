"""Prepare isolated inference metadata and separate manual-anchor supervision.

Frames follow the sealed sparse-anchor pilot. Reindexing frame_id establishes
sampled-step tracker semantics, not elapsed physical video time. Ground-truth
annotations are absent from every inference JSON and stored separately.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_FIELDS = ("id", "video_id", "file_name", "width", "height", "frame_id", "frame_index")
VIDEO_FIELDS = ("id", "name", "width", "height", "fps")
CATEGORY_FIELDS = ("id", "name", "synonyms", "synset", "def", "frequency")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=ROOT / "tmp_analysis/base_error_learning_plan_20261023.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/base_error_collection_20261023_v3")
    args = ap.parse_args()
    assert not args.out.exists(), "Preserve existing stage and use a new output"
    plan = json.loads(args.plan.read_text())
    assert plan["status"] == "METADATA_ONLY_COLLECTION_NOT_RUN"
    assert plan["metadata_validation"]["status"] == "PASS_METADATA_ONLY"
    source = Path(plan["inputs"]["ctao_base"]["path"])
    assert sha(source) == plan["inputs"]["ctao_base"]["sha256"]
    ann = json.loads(source.read_text())
    videos = {v["id"]: {k: v[k] for k in VIDEO_FIELDS if k in v} for v in ann["videos"]}
    categories = [{k: c[k] for k in CATEGORY_FIELDS if k in c} for c in ann["categories"]]
    images = {im["id"]: im for im in ann["images"]}
    pilot = plan["minimal_pilot"]
    records = {v["video_id"]: v for v in plan["full_metadata_split"]["videos"]}
    grouped = defaultdict(list)
    for row in pilot["images"]:
        assert row["video_id"] in pilot["video_ids"]
        original = images[row["image_id"]]
        assert (original["video_id"], original["frame_index"], original["file_name"]) == (row["video_id"], row["frame_index"], row["file_name"])
        grouped[row["video_id"]].append({k: original[k] for k in IMAGE_FIELDS if k in original})
    result = []
    for vid in sorted(grouped):
        frames = sorted(grouped[vid], key=lambda im: (im["frame_index"], im["id"]))
        assert len({im["frame_index"] for im in frames}) == len(frames)
        for step, im in enumerate(frames):
            im["source_frame_id"] = im.get("frame_id", -1)
            im["frame_id"] = step
            im["sample_step"] = step
            assert (ROOT / "data_tao/frames" / im["file_name"]).is_file()
            result.append(im)
    assert len(result) == 1299 and len(grouped) == 41
    targets = set(pilot["manual_anchor_ctao_annotation_ids"])
    anchors = [a for a in ann["annotations"] if a["id"] in targets]
    assert len(anchors) == len(targets) == 3953
    assert {a["image_id"] for a in anchors} == {im["id"] for im in result}
    fit_categories = set(plan["vocabulary"]["readout_fit_category_ids"])
    for a in anchors:
        part = records[a["video_id"]]["split"]
        a["readout_split"] = part
        a["readout_class_holdout"] = a["category_id"] not in fit_categories
        a["eligible_for_parameter_fit"] = part == "fit" and a["category_id"] in fit_categories
        a["supervision_source"] = "exact original-TAO manual anchor intersection"
    assert sum(a["eligible_for_parameter_fit"] for a in anchors) == 2138
    base = {"categories": categories, "annotations": [], "tracks": [],
            "info": {"collection_protocol": "sparse manual-anchor timeline, GT-free predictions on every selected image",
                     "frame_id": "sampled-step index starting at 0 separately for each video",
                     "frame_index": "preserved original frame index; never interpret sample steps as seconds",
                     "vocabulary": "canonical LVIS v1 1203 foreground + background; full-vocabulary rare-column fusion",
                     "gt_annotations_passed_to_model": False,
                     "inference_image_field_allowlist": list(IMAGE_FIELDS) + ["sample_step", "source_frame_id"],
                     "inference_video_field_allowlist": list(VIDEO_FIELDS),
                     "inference_category_field_allowlist": list(CATEGORY_FIELDS),
                     "frequency_prior": "fixed public LVIS-v1 frequency, not pilot annotation counts"}}
    args.out.mkdir(parents=True)
    inference = dict(base, videos=[videos[v] for v in sorted(grouped)], images=result)
    write(args.out / "pilot_inference.json", inference)
    fit_videos = [v for v in grouped if records[v]["split"] == "fit"]
    fit_videos.sort(key=lambda v: hashlib.sha256(("schema-smoke-v1/" + str(v)).encode()).hexdigest())
    # Four real images check two within-video transitions and a video reset.
    chosen = fit_videos[:2]
    smoke_images = [im for vid in chosen for im in result if im["video_id"] == vid and im["sample_step"] < 2]
    assert len(smoke_images) == 4
    write(args.out / "smoke_inference.json", dict(base, videos=[videos[v] for v in chosen], images=smoke_images))
    write(args.out / "manual_anchor_targets.json", {
        "status": "TARGETS_ONLY_NOT_MODEL_INPUT", "categories": categories,
        "annotations": anchors, "readout_fit_category_ids": sorted(fit_categories),
        "readout_category_holdout_ids": plan["vocabulary"]["readout_category_holdout_ids"],
        "leakage_rule": "No heldout-class or non-fit-video labels enter parameter fitting, including pair targets; unmatched is unknown, never an automatic negative"})
    manifest = {"status": "PREPARED_NO_INFERENCE_RUN", "plan": str(args.plan), "plan_sha256": sha(args.plan),
                "source": str(source), "source_sha256": sha(source), "script_sha256": sha(__file__),
                "images": len(result), "videos": len(grouped), "manual_targets": len(anchors),
                "fit_eligible_manual_targets": 2138, "smoke_videos": chosen,
                "timeline": base["info"], "files": {p.name: sha(p) for p in args.out.glob("*.json")}}
    write(args.out / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
