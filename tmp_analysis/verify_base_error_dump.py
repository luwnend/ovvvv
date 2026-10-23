#!/usr/bin/env python3
"""Verify isolated base-error dumps, or run CPU-only synthetic schema tests.

No model import, GPU, ground-truth matching, training or original file writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "work/covtrack_base_error_20261023"
UTILITY = COPY / "ovtrack/base_error_dump.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def self_test(api):
    """Exercise 1203+background, zero frames, row maps and a sentinel -1 ID."""
    with tempfile.TemporaryDirectory(prefix="base_error_schema_cpu_") as task_temp:
        temp = Path(task_temp)
        ann = temp / "images.json"
        ann.write_text(json.dumps(dict(images=[dict(id=10, video_id=2, frame_id=0, frame_index=30,
                                                   file_name="train/demo/frame0031.jpg"),
                                              dict(id=11, video_id=2, frame_id=1, frame_index=60,
                                                   file_name="train/demo/frame0061.jpg")],
                                        categories=[], annotations=[], tracks=[], videos=[])))
        c = 1203
        ids = np.r_[np.arange(1, c + 1, dtype=np.int64), -1]
        class_names = runpy.run_path(str(COPY / "ovtrack/models/roi_heads/class_name.py"))
        mask = np.zeros(c + 1, dtype=bool)
        mask[np.asarray(class_names["lvis_novel_label_ids"], np.int64)] = True
        assert mask.sum() == 337 and not mask[-1]
        schema = dict(posterior_category_ids=ids, posterior_background_index=np.int64(c),
                      actual_fusion_mask=mask, fusion_text_exponents=np.where(mask, 1. / 3, 2. / 3),
                      fusion_mode=np.asarray("asymmetric_actual_mask"),
                      vocabulary_name=np.asarray("LVIS_1203_one_based_category_ids"),
                      only_validation_categories=np.bool_(False), only_test_categories=np.bool_(False),
                      ensemble=np.bool_(True))
        rng = np.random.default_rng(3)
        pt = rng.random((4, c + 1)); pt /= pt.sum(1, keepdims=True)
        pi = rng.random((4, c + 1)); pi[:, -1] = 0; pi /= pi.sum(1, keepdims=True)
        pf = pt ** schema["fusion_text_exponents"] * pi ** (1 - schema["fusion_text_exponents"])
        pf /= pf.sum(1, keepdims=True)
        cache = dict(schema=schema, p_text=pt.astype(np.float32), p_image=pi.astype(np.float32),
                     p_fused=pf.astype(np.float32), det_inds=[np.asarray([0 * c + 9, 3 * c + 79, 1 * c + 6])],
                     filename="data/tao/frames/train/demo/frame0031.jpg")
        pre = np.asarray([[0,0,5,6,.2], [2,1,10,11,.3], [1,0,6,7,.4]], np.float32)
        labels = np.asarray([9,79,6], np.int64)
        keep = np.asarray([False, True, True])
        post = pre[keep].copy(); post[:, -1] *= .9
        payload = api["build_payload"](cache, dict(filename=cache["filename"], ori_shape=[30,40,3]),
                                       ann, pre, labels, post, labels[keep], np.asarray([-1,7]),
                                       np.ones((3,256),np.float32), np.ones((3,512),np.float32), keep, 0)
        assert np.array_equal(payload["roi_proposal_row"], [3,1])
        assert np.array_equal(payload["pretracker_det_row"], [1,2])
        assert np.array_equal(payload["emitted_track_mask"], [False,True])
        assert payload["p_fused"].shape == (2,1204)
        saved = api["write_frame_dump"](temp / "dump", payload)
        with np.load(saved, allow_pickle=False) as persisted:
            check = api["validate_payload"](dict(persisted))
        assert check["observations"] == 2 and check["emitted_observations"] == 1
        zero_cache = dict(cache, det_inds=[np.zeros(0,np.int64)], filename="data/tao/frames/train/demo/frame0061.jpg")
        zero = api["build_payload"](zero_cache, dict(filename=zero_cache["filename"],ori_shape=[30,40,3]),
                                    ann,np.zeros((0,5),np.float32),np.zeros(0,np.int64),None,None,None,None,None,None,1)
        zero_saved = api["write_frame_dump"](temp / "dump",zero)
        with np.load(zero_saved,allow_pickle=False) as persisted:
            zero_check=api["validate_payload"](dict(persisted))
        assert zero_check["empty_frame"] and zero["p_fused"].shape==(0,1204)
        assert zero_saved.with_suffix(".receipt.json").exists()
        rejected = {}
        for name, mutated in (("wrong_row_map",dict(payload,roi_proposal_row=np.asarray([1,3]))),
                              ("wrong_class_ids",dict(payload,pred_category_id=np.asarray([1,2]))),
                              ("nonfinite_features",dict(payload,track_feats=np.full((2,256),np.nan,np.float16)))):
            try:
                api["validate_payload"](mutated)
            except (AssertionError, ValueError):
                rejected[name]="PASS"
            else:
                raise AssertionError("Validator accepted corrupt case: "+name)
        try:
            api["write_frame_dump"](temp / "dump", payload)
        except FileExistsError:
            rejected["no_overwrite"]="PASS"
        else:
            raise AssertionError("Frame overwrite accepted")
        return dict(status="PASS_CPU_SYNTHETIC_SCHEMA",full_1203_background=True,
                    canonical_mask_true_count=int(mask.sum()),posttracker_negative_id_retained_and_flagged=True,
                    detector_to_proposal_and_tracker_row_maps=True,zero_observation_npz_and_receipt=True,
                    train_filename_supported=True,fusion_max_abs_error=check["fusion_max_abs_error"],
                    malformed_cases_rejected=rejected)


def verify(args, api):
    annotation_sha=sha256(args.annotation)
    raw=json.load(args.annotation.open())
    images={int(im["id"]):{k:im[k] for k in ("id","video_id","frame_id","file_name")} for im in raw["images"]}
    del raw
    seen=set();obs=emitted=empty=0;error=0.;vocab=None;mask=None;files=[]
    paths=sorted(args.dump_dir.glob("*.npz"))
    assert paths,"No frame dumps found"
    for path in paths:
        assert ".incomplete." not in path.name,"Incomplete frame dump"
        with np.load(path,allow_pickle=False) as saved:
            data={k:saved[k] for k in saved.files}
        result=api["validate_payload"](data)
        image_id=int(data["image_id"]);assert image_id in images and image_id not in seen
        im=images[image_id];assert int(data["video_id"])==int(im["video_id"])
        assert int(data["frame_id"])==int(im["frame_id"])
        assert data["canonical_filename"].item()==api["canonical_filename"](im["file_name"])
        assert data["annotation_sha256"].item()==annotation_sha
        ids=data["posterior_category_ids"]
        if args.expected_foreground_classes:
            assert len(ids)==args.expected_foreground_classes+1
        if vocab is None:vocab=ids;mask=data["actual_fusion_mask"]
        else:assert np.array_equal(vocab,ids) and np.array_equal(mask,data["actual_fusion_mask"])
        receipt_path=path.with_suffix(".receipt.json");receipt=json.load(receipt_path.open())
        assert receipt["status"]=="PASS_FRAME_DUMP" and receipt["sha256"]==sha256(path)
        assert receipt["observations"]==result["observations"] and receipt["emitted_observations"]==result["emitted_observations"]
        assert receipt["image_id"]==image_id and receipt["file"]==path.name
        seen.add(image_id);obs+=result["observations"];emitted+=result["emitted_observations"]
        empty+=int(result["empty_frame"]);error=max(error,result["fusion_max_abs_error"])
        files.append(dict(file=path.name,sha256=receipt["sha256"],image_id=image_id))
    assert len(list(args.dump_dir.glob("*.receipt.json")))==len(paths)
    if not args.allow_partial:assert seen==set(images),"Collection is incomplete, including zero-observation frame receipts"
    return dict(status="PASS_BASE_ERROR_DUMP" if seen==set(images) else "PASS_PARTIAL_BASE_ERROR_DUMP",
                frames=len(seen),expected_frames=len(images),missing_images=sorted(set(images)-seen),
                observations=obs,emitted_track_observations=emitted,unassigned_id_observations=obs-emitted,
                zero_observation_frames=empty,posterior_foreground_classes=len(vocab)-1,
                actual_fusion_mask_true_count=int(mask.sum()),fusion_max_abs_error=error,
                annotation_sha256=annotation_sha,dump_files=files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--dump-dir",type=Path)
    parser.add_argument("--annotation",type=Path)
    parser.add_argument("--expected-foreground-classes",type=int,default=1203)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();api=runpy.run_path(str(UTILITY))
    if args.self_test:report=self_test(api)
    else:
        assert args.dump_dir and args.annotation,"Provide --dump-dir and --annotation or --self-test"
        report=verify(args,api)
    report["validator_sha256"]=sha256(__file__);report["schema_utility_sha256"]=sha256(UTILITY)
    if args.output:
        assert not args.output.exists(),"Refuse to overwrite validation report"
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temp=args.output.with_suffix(args.output.suffix+".tmp");temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(args.output)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
