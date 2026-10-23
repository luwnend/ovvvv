#!/usr/bin/env python3
"""Build metadata-only independent base-error collection plan. Never infer/train.

For byte-identical regeneration, pass the original plan's created_utc via
--created-utc and write to a fresh --output path. Inputs are read-only.
"""

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(root, created_utc):
    r = root
    paths = {
        "ctao_base": r / "work/covtrack_base/saved_models/ctao_dataset/ctao_base.json",
        "tao_train": r / "data_tao/annotations/train.json",
        "validation": r / "data_tao/annotations/validation_ours_v1.json",
        "training_config": r / "work/covtrack_base/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py",
        "class_name": r / "work/covtrack_base/ovtrack/models/roi_heads/class_name.py",
        "frozen_released_checkpoint": r / "work/covtrack_base/saved_models/ctao_public_res/ctao_public.pth",
        "dump_roi_head": r / "work/covtrack_tces_v2/ovtrack/models/roi_heads/ovtrack_roi_head.py",
        "dump_mot": r / "work/covtrack_tces_v2/ovtrack/models/mot/ovtrack.py",
        "val_cache_builder": r / "tmp_analysis/build_track_posterior_cache.py",
    }
    a = json.loads(paths["ctao_base"].read_text())
    b = json.loads(paths["tao_train"].read_text())
    val = json.loads(paths["validation"].read_text())
    cats = {c["id"]: c for c in a["categories"]}
    ai = {i["id"]: i for i in a["images"]}
    bi = {i["id"]: i for i in b["images"]}
    vids = {v["id"]: v for v in a["videos"]}
    syn = collections.defaultdict(list)
    for c in cats.values():
        syn[c.get("synset")].append(c["id"])
    oldcats = {c["id"]: c for c in b["categories"]}
    cross = {k: syn.get(c.get("synset"), []) for k, c in oldcats.items()}

    def anchor_key(x, im):
        return (x["video_id"], im[x["image_id"]]["frame_index"], x["track_id"])

    orig = {anchor_key(x, bi): x for x in b["annotations"]}
    assert len(orig) == len(b["annotations"])
    anchors = []
    for x in a["annotations"]:
        if not ai[x["image_id"]].get("is_ori"):
            continue
        y = orig.get(anchor_key(x, ai))
        if (
            y is not None
            and cross[y["category_id"]] == [x["category_id"]]
            and all(abs(float(m) - float(n)) <= 1e-6 for m, n in zip(x["bbox"], y["bbox"]))
        ):
            assert cats[x["category_id"]]["frequency"] in ("f", "c")
            anchors.append(x)
    assert len(anchors) == 52911
    seed = "20261023-base-natural-error-metadata-v1"

    def rank(kind, key):
        return hashlib.sha256((seed + "|" + kind + "|" + str(key)).encode()).hexdigest()

    def source(v):
        m = v.get("metadata", {})
        ds = m.get("dataset", v["name"].split("/")[1])
        name = v["name"].split("/", 2)[-1]
        if ds == "AVA":
            key = "YouTube/" + m.get("movie", name.split("_scene_")[0])
        elif ds == "HACS":
            key = "YouTube/" + m.get("youtube_id", name)
        elif ds == "ArgoVerse":
            key = "ArgoVerse/" + m.get("sequence", name)
        else:
            key = ds + "/" + name
        return ds, key

    train_groups = collections.defaultdict(list)
    ds_groups = collections.defaultdict(set)
    for v in a["videos"]:
        ds, g = source(v)
        train_groups[g].append(v["id"])
        ds_groups[ds].add(g)
    valgroups = {source(v)[1] for v in val["videos"]}
    assert not (set(train_groups) & valgroups)
    video_id_overlap = len(set(vids) & {v["id"] for v in val["videos"]})
    video_names = {v["name"].replace("\\", "/").split("/", 1)[-1] for v in a["videos"]}
    val_video_names = {v["name"].replace("\\", "/").split("/", 1)[-1] for v in val["videos"]}
    image_names = {i["file_name"].replace("\\", "/").split("/", 1)[-1] for i in a["images"]}
    val_image_names = {i["file_name"].replace("\\", "/").split("/", 1)[-1] for i in val["images"]}
    assert video_id_overlap == 0
    assert not (video_names & val_video_names)
    assert not (image_names & val_image_names)
    group_split = {}
    byds = {}
    for ds, groups in sorted(ds_groups.items()):
        gs = sorted(groups, key=lambda g: rank("group", g))
        nfit = int(0.8 * len(gs))
        ntune = (len(gs) - nfit) // 2
        split = {"fit": gs[:nfit], "tune": gs[nfit:nfit + ntune], "test": gs[nfit + ntune:]}
        for part, gg in split.items():
            for g in gg:
                assert g not in group_split
                group_split[g] = part
        byds[ds] = {
            part: {"groups": len(gg), "videos": sum(len(train_groups[g]) for g in gg)}
            for part, gg in split.items()
        }
    video_split = {v: group_split[g] for g, vs in train_groups.items() for v in vs}
    assert len(video_split) == 500
    obs = sorted({x["category_id"] for x in anchors})
    fitcats, holdcats = [], []
    for freq in ("f", "c"):
        ks = sorted((k for k in obs if cats[k]["frequency"] == freq), key=lambda k: rank("class", k))
        n = int(0.2 * len(ks))
        holdcats += ks[:n]
        fitcats += ks[n:]
    assert len(obs) == 182 and len(fitcats) == 146 and len(holdcats) == 36
    byvid = collections.defaultdict(list)
    for x in anchors:
        byvid[x["video_id"]].append(x)

    def stat(vid_ids):
        xs = [x for v in vid_ids for x in byvid[v]]
        tr = collections.Counter((x["video_id"], x["track_id"]) for x in xs)
        return {
            "videos": len(vid_ids),
            "source_groups": len({source(vids[v])[1] for v in vid_ids}),
            "anchor_images": len({x["image_id"] for x in xs}),
            "manual_anchor_boxes": len(xs),
            "gt_tracks": len(tr),
            "gt_tracks_at_least_6_anchors": sum(n >= 6 for n in tr.values()),
            "observed_categories": len({x["category_id"] for x in xs}),
            "fit_category_boxes": sum(x["category_id"] in fitcats for x in xs),
            "heldout_category_boxes": sum(x["category_id"] in holdcats for x in xs),
            "heldout_categories_with_boxes": sorted({x["category_id"] for x in xs if x["category_id"] in holdcats}),
        }

    pilot_groups = {}
    for ds, groups in sorted(ds_groups.items()):
        for part, need in [("fit", 2), ("tune", 1), ("test", 1)]:
            eligible = [
                g for g in groups
                if group_split[g] == part and stat(train_groups[g])["gt_tracks_at_least_6_anchors"] >= 1
            ]
            eligible.sort(key=lambda g: rank("pilot", g))
            assert len(eligible) >= need, (ds, part, len(eligible))
            for g in eligible[:need]:
                pilot_groups[g] = part
    pilot_vids = sorted(v for g in pilot_groups for v in train_groups[g])
    pilot_vidset = set(pilot_vids)
    pilotanchors = [x for x in anchors if x["video_id"] in pilot_vidset]
    records = []
    for vid in sorted(vids):
        v = vids[vid]
        ds, g = source(v)
        xs = byvid[vid]
        tr = collections.Counter(x["track_id"] for x in xs)
        records.append({
            "video_id": vid, "video_name": v["name"], "dataset": ds,
            "source_group": g, "split": video_split[vid], "pilot": vid in pilot_vidset,
            "manual_anchor_images": len({x["image_id"] for x in xs}),
            "manual_anchor_boxes": len(xs), "manual_anchor_tracks": len(tr),
            "tracks_at_least_6_anchors": sum(n >= 6 for n in tr.values()),
        })
    image_recs = []
    for imid in sorted({x["image_id"] for x in pilotanchors}):
        i = ai[imid]
        image_recs.append({
            "image_id": imid, "video_id": i["video_id"], "frame_index": i["frame_index"],
            "source_frame_id": i.get("frame_id"), "file_name": i["file_name"],
            "split": video_split[i["video_id"]],
        })
    plan = {
        "status": "METADATA_ONLY_COLLECTION_NOT_RUN",
        "created_utc": created_utc,
        "seed": seed,
        "purpose": "Independent base-training natural-error readout pilot; no validation examples used as fitting/tuning data.",
        "inputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
        "annotation_facts": {
            "ctao_images": len(a["images"]), "ctao_videos": len(a["videos"]),
            "ctao_boxes": len(a["annotations"]), "ctao_tracks": len(a["tracks"]),
            "ctao_observed_base_categories": 182,
            "ctao_original_images": sum(bool(i.get("is_ori")) for i in a["images"]),
            "manual_anchor_intersection_boxes": len(anchors),
            "ctao_ori_boxes_not_in_official_gt_key": 352,
            "manual_anchor_rule": "C-TAO is_ori=true AND unique (video_id, original frame_index, track_id) in official TAO train AND unique old-synset-to-v1 category mapping equals C-TAO category AND bbox equal within 1e-6. Other C-TAO boxes excluded from supervised targets.",
        },
        "dedup": {
            "train_video_ids": len(vids), "val_video_ids": len(val["videos"]), "video_id_overlap": video_id_overlap,
            "normalized_video_and_image_path_overlap": 0,
            "train_source_groups": len(train_groups), "val_source_groups": len(valgroups),
            "source_group_overlap": 0,
            "source_group_rule": "YouTube/movie for AVA; YouTube/youtube_id for HACS; ArgoVerse/sequence across cameras; dataset/full clip name otherwise.",
            "limitations": "Metadata dedup only. No image content dedup, Charades-subject holdout or YFCC-uploader holdout was performed.",
        },
        "vocabulary": {
            "canonical": "LVIS v1 category_id 1..1203",
            "base_vocabulary_ids": sorted(c["id"] for c in cats.values() if c["frequency"] in ("f", "c")),
            "base_vocabulary_count": 866, "actual_observed_base_categories": obs,
            "readout_fit_category_ids": sorted(fitcats), "readout_category_holdout_ids": sorted(holdcats),
            "category_holdout_meaning": "Pseudo-novel only to this readout learner. Base detector/CLIP may have seen these classes; this is not detector-unseen-category evidence.",
            "proposed_collection_foreground_ids": list(range(1, 1204)),
            "proposed_background_sentinel": -1,
            "proposed_config_overrides": {
                "model.roi_head.only_validation_categories": False,
                "model.roi_head.only_test_categories": False,
                "model.roi_head.custom_classes": False,
            },
            "proposed_posterior_shape_columns": 1204,
            "required_before_collection": [
                "Export explicit posterior_category_ids and actual fusion mask in each run manifest/dump; check each branch column count and mapping.",
                "Generalize cache builder away from val296 mapping and entropy denominator log296.",
                "CPU schema or isolated smoke must prove returned det_labels+1 mapping and complete 1203+background support; no smoke was run here.",
            ],
            "name_alias": {
                "category_id": 655, "class_name": "speaker_(stereo_equipment)",
                "annotation_name": "speaker_(stero_equipment)",
                "mapping_rule": "Preserve numeric category IDs and record this typo alias; do not reorder by display name.",
            },
        },
        "full_metadata_split": {
            "unit": "source group; video trajectories remain nested repeated observations",
            "allocation": "Within each source dataset, SHA256-ranked source groups: floor(80%) fit, remaining half tune and half test.",
            "counts_by_dataset": byds,
            "counts": {part: stat(sorted(v for v, p in video_split.items() if p == part)) for part in ["fit", "tune", "test"]},
            "videos": records,
        },
        "minimal_pilot": {
            "scope": "Metadata-selected independent schema/learnability pilot. Full source groups kept together; no performance or power guarantee.",
            "allocation": "Within each source dataset and metadata split select SHA256-ranked eligible source groups: 2 fit, 1 tune, 1 test; eligibility is at least one GT track with >=6 manual anchors.",
            "timeline": "Only frames having at least one matched manual anchor are listed. This is a sparse TAO-anchor protocol, not dense continuous-frame evidence. Optional dense context collection requires a separate manifest.",
            "video_ids": pilot_vids, "source_groups": {g: p for g, p in sorted(pilot_groups.items())},
            "counts": {part: stat(sorted(v for v in pilot_vids if video_split[v] == part)) for part in ["fit", "tune", "test"]},
            "images": image_recs,
            "manual_anchor_ctao_annotation_ids": sorted(x["id"] for x in pilotanchors),
            "no_data_collection_started": True,
        },
        "training_use_policy": {
            "fit": "Fit only targets with fit video split AND fit category; inference observations contain no GT filtering.",
            "tune": "Tune only tune videos and fit categories. Category-holdout labels excluded from optimization and threshold selection.",
            "test": "Readout video test and pseudo-novel category holdout opened only after method specification; no claim of statistical power.",
            "independence": "Bootstrap/report by source group or video/trajectory clusters, never treat frame or pair counts as independent samples.",
            "residual_target": "GT-free frozen inference first. Match only manual anchors in a separate supervised layer; unmatched observations unknown, not automatically negative.",
            "public_validation": "Existing val cache and approved diagnostics remain external development evidence and are never training examples.",
        },
        "missing_before_executable_collection": [
            "An isolated inference copy and immutable collection config for the 1203 foreground vocabulary.",
            "Self-describing dump schema, generalized cache builder, zero-observation frame receipts, and vocabulary parity smoke.",
            "A sealed pilot annotation/config file preserving listed image/video/category IDs and track temporal order.",
            "Class-blind GT matching and residual target construction with original/manual versus pseudo provenance.",
            "A learnable mechanism and predeclared loss/control/readout evaluation; this metadata plan does not establish novelty or efficacy.",
        ],
        "actions_taken": {
            "gpu_used": False, "training_started": False, "inference_started": False,
            "active_training_modified": False, "frozen_repo_modified": False,
        },
    }
    missing = [i["file_name"] for i in image_recs if not (r / "data_tao/frames" / i["file_name"]).is_file()]
    assert not missing, missing[:10]
    assert len({i["image_id"] for i in image_recs}) == len(image_recs) == 1299
    parts = collections.defaultdict(set)
    for v in records:
        parts[v["split"]].add(v["source_group"])
    assert all(not (parts[x] & parts[y]) for x, y in [("fit", "tune"), ("fit", "test"), ("tune", "test")])
    assert sum(v["manual_anchor_boxes"] for v in plan["full_metadata_split"]["counts"].values()) == 52911
    assert not (set(fitcats) & set(holdcats))
    plan["metadata_validation"] = {
        "status": "PASS_METADATA_ONLY", "pilot_image_file_paths_checked": 1299,
        "pilot_missing_file_paths": 0, "video_source_group_splits_disjoint": True,
        "category_fit_holdout_sets_disjoint": True, "full_manual_anchor_box_sum": 52911,
        "image_content_integrity_checked": False, "inference_or_model_schema_test_run": False,
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("tmp_analysis/base_error_learning_plan_20261023.json"))
    parser.add_argument("--created-utc", default=None)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    if output.exists():
        parser.error("Refusing to overwrite existing plan; choose a fresh --output path")
    created = args.created_utc or datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan = build(args.root.resolve(), created)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "status": plan["status"]}))


if __name__ == "__main__":
    main()
