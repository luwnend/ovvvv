"""Collect frozen full-vocabulary observations without reading supervision.

Run with the pinned ovtrack_ovmot interpreter. The runner only opens a prepared
metadata-only inference JSON. Every image must produce a schema-checked NPZ,
including empty frames. Raw dumps are sealed before any later GT matching.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "work/covtrack_base_error_20261023"
GPU = "GPU-2a55197e-2c19-45ff-cf33-13031951f9d0"
CHECKPOINT = ROOT / "work/covtrack_tces_v2/saved_models/ctao_public_res/ctao_public.pth"
CHECKPOINT_SHA = "e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c"
CONFIG = ROOT / "work/covtrack_tces_v2/configs/uncertainty-ovtrack-teta/ovtrack_r50_fullval.py"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_guard():
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    row = [r.split(", ") for r in info.splitlines() if r.startswith(GPU + ",")]
    assert len(row) == 1 and int(row[0][2]) > 4096 and int(row[0][3]) >= 15000, row
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    processes = []
    for line in apps.splitlines():
        uuid, pid, used = [s.strip() for s in line.split(",")]
        if uuid != GPU:
            continue
        owner = subprocess.check_output(["ps", "-o", "user=", "-p", pid], text=True).strip()
        assert owner and owner != "chenhc", "Protected chenhc process on collection GPU: " + line
        processes.append(dict(pid=int(pid), owner=owner, used_memory_mib=used))
    return dict(checked_utc=utc(), uuid=GPU, name=row[0][1], total_mib=int(row[0][2]), free_mib=int(row[0][3]), existing_processes=processes,
                authorization="non-4GB GPU with no pre-existing chenhc process; other owners allowed")


def protected_snapshot():
    old = json.loads((ROOT / "results/tces_audit_20261023/protected_before.json").read_text())
    now = {}
    for path, spec in old.items():
        if not isinstance(spec, dict) or "exists" not in spec:
            continue
        p = Path(path)
        assert p.exists() == spec["exists"], path
        digest = sha(p) if p.exists() else None
        assert digest == spec["sha256"], path
        now[path] = dict(exists=p.exists(), sha256=digest)
    return now


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.annotation = args.annotation.resolve()
    assert not args.out.exists(), "Refuse overwrite; use a new run directory"
    metadata = json.loads(args.annotation.read_text())
    assert metadata["annotations"] == [] and metadata["tracks"] == []
    allowed_images = set(metadata["info"]["inference_image_field_allowlist"])
    allowed_videos = set(metadata["info"]["inference_video_field_allowlist"])
    allowed_categories = set(metadata["info"]["inference_category_field_allowlist"])
    assert all(set(im) <= allowed_images for im in metadata["images"])
    assert all(set(v) <= allowed_videos for v in metadata["videos"])
    assert all(set(c) <= allowed_categories for c in metadata["categories"])
    assert [c["id"] for c in metadata["categories"]] == list(range(1, 1204))
    frames = metadata["images"]
    assert len({im["id"] for im in frames}) == len(frames) > 0
    previous = None
    last_steps = {}
    completed_videos = set()
    for im in frames:
        vid, step = im["video_id"], im["frame_id"]
        assert step == im["sample_step"] == last_steps.get(vid, -1) + 1
        if vid != previous:
            assert vid not in completed_videos and step == 0
            if previous is not None:
                completed_videos.add(previous)
        previous = vid
        last_steps[vid] = step
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    guard = gpu_guard()
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["OVMOT_BASE_ERROR_ANNOTATION"] = str(args.annotation)
    os.environ["OVMOT_TCES_DUMP_DIR"] = str(args.out.resolve() / "dump")
    sys.path.insert(0, str(COPY))
    import numpy as np
    import torch
    import mmcv
    import ovtrack
    from ovtrack.apis import init_model, inference_model
    from ovtrack.base_error_dump import validate_payload
    from mmdet.datasets.builder import PIPELINES

    @PIPELINES.register_module()
    class CollectionFrameStep:
        """Bridge inference_model's img_info step into VideoCollect metadata."""
        def __call__(self, data):
            data["frame_id"] = int(data["img_info"]["frame_id"])
            return data
    assert Path(ovtrack.__file__).resolve().is_relative_to(COPY) if hasattr(Path(), "is_relative_to") else str(Path(ovtrack.__file__).resolve()).startswith(str(COPY) + "/")
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    np.random.seed(20261023)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = mmcv.Config.fromfile(str(CONFIG))
    cfg.model.backbone.init_cfg = None
    roi = cfg.model.roi_head
    roi.update(only_validation_categories=False, only_test_categories=False,
               custom_classes=False, ensemble=True, fix_bg=False,
               use_special_prompt=False, record_probs_info=False,
               prompt_path=str(ROOT / "work/covtrack_tces_v2/saved_models/pretrained_models/detpro_prompt.pt"))
    roi.feature_fusion_head.max_fusion_ratio = 2.0
    cfg.model.tracker.update(match_score_thr=.37, memo_frames=50, momentum_embed=.4,
                             confused_features=True, vis=False)
    cfg.model.test_cfg.rcnn.max_per_img = 80
    cfg.data.test.pipeline.insert(0, dict(type="CollectionFrameStep"))
    cfg.data = dict(test=dict(pipeline=cfg.data.test.pipeline))
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    before = protected_snapshot()
    source_hashes = {str(p.relative_to(COPY)): sha(p) for p in sorted((COPY / "ovtrack").rglob("*.py"))}
    args.out.mkdir(parents=True)
    cfg.dump(str(args.out / "collection_config.py"))
    manifest = dict(status="RUNNING_GT_FREE_COLLECTION", started_utc=utc(), pid=os.getpid(), annotation=str(args.annotation), annotation_sha256=sha(args.annotation),
                    expected_images=len(frames), expected_videos=len(last_steps), timeline=metadata["info"], device=guard,
                    checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
                    source_config=str(CONFIG), source_config_sha256=sha(CONFIG), config_sha256=sha(args.out / "collection_config.py"),
                    runner=str(Path(__file__).resolve()), runner_sha256=sha(__file__), code_root=str(COPY), source_hashes=source_hashes,
                    model_training=False, new_parameter_fit=False, supervision_file_opened=False,
                    aggregation_key=["video_id", "local_track_id"], row_selection="all model outputs; no GT selection; id=-1 recorded, emitted_track_mask used for tracks",
                    runtime=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda, mmcv=mmcv.__version__, seed=20261023, threads=1))
    write(args.out / "run_manifest.json", manifest)
    write(args.out / "protected_before.json", before)
    started = time.monotonic()
    model = init_model(cfg, str(CHECKPOINT), device="cuda:0")
    model.requires_grad_(False)
    assert not model.training
    assert model.roi_head.num_classes == 1203 and not model.roi_head.fix_bg
    write(args.out / "model_ready.json", dict(ready_utc=utc(), load_seconds=time.monotonic() - started,
          actual_novel_mask_count=int(model.roi_head.novel_index.sum()), text_features_shape=list(model.roi_head.text_features_for_classes.shape)))
    n_observations = n_emitted = 0
    with (args.out / "frame_execution.jsonl").open("x") as log:
        for k, im in enumerate(frames):
            path = ROOT / "data_tao/frames" / im["file_name"]
            assert path.is_file()
            prev_tracker = getattr(model, "tracker", None)
            prev_total = int(getattr(prev_tracker, "num_tracklets", 0))
            tick = time.monotonic()
            result = inference_model(model, str(path), im["frame_id"])
            torch.cuda.synchronize()
            reset = model.tracker is not prev_tracker
            assert reset == (im["frame_id"] == 0)
            assert not model.tracker.vis and not hasattr(model.tracker, "filename2ann")
            key = "v%d_i%d_f%d" % (im["video_id"], im["id"], im["frame_id"])
            npz_path = args.out / "dump" / (key + ".npz")
            with np.load(npz_path, allow_pickle=False) as archive:
                data = dict(archive)
            checked = validate_payload(data)
            assert int(data["video_id"]) == im["video_id"] and int(data["image_id"]) == im["id"]
            assert checked["posterior_columns"] == 1204
            actual = []
            for label, values in enumerate(result["track_results"]):
                for row in values:
                    actual.append((label, int(row[0]), *map(float, row[1:])))
            expected = [(int(label), int(tid), *map(float, box)) for label, tid, box in zip(data["det_labels"], data["track_ids"], data["det_bboxes"]) if tid >= 0]
            assert sorted(actual) == sorted(expected), "Saved emitted observations differ from actual model output"
            n_observations += checked["observations"]
            n_emitted += checked["emitted_observations"]
            event = dict(index=k, image_id=im["id"], video_id=im["video_id"], frame_id=im["frame_id"], frame_index=im["frame_index"],
                         tracker_reset_observed=reset, previous_tracker_total=prev_total,
                         tracker_num_tracklets=int(model.tracker.num_tracklets), seconds=time.monotonic() - tick,
                         elapsed_seconds=time.monotonic() - started, emitted_output_exact=True, **checked,
                         npz=str(npz_path.resolve()), npz_sha256=sha(npz_path))
            log.write(json.dumps(event) + "\n"); log.flush()
            if k < 4 or (k + 1) % 25 == 0 or k + 1 == len(frames):
                print(json.dumps(dict(status="COLLECTING", completed=k+1, total=len(frames), **event)), flush=True)
    after = protected_snapshot()
    assert before == after
    assert source_hashes == {str(p.relative_to(COPY)): sha(p) for p in sorted((COPY / "ovtrack").rglob("*.py"))}
    assert sha(args.annotation) == manifest["annotation_sha256"]
    write(args.out / "protected_after.json", after)
    inventory = {p.name: sha(p) for p in sorted((args.out / "dump").iterdir())}
    assert sum(n.endswith(".npz") for n in inventory) == len(frames)
    assert sum(n.endswith(".receipt.json") for n in inventory) == len(frames)
    write(args.out / "dump_inventory.json", inventory)
    receipt = dict(status="COMPLETE_GT_FREE_COLLECTION_PENDING_INDEPENDENT_AUDIT", completed_utc=utc(), elapsed_seconds=time.monotonic() - started,
                   completed_images=len(frames), completed_videos=len(last_steps), retained_observations=n_observations, emitted_observations=n_emitted,
                   run_manifest_sha256=sha(args.out / "run_manifest.json"), events_sha256=sha(args.out / "frame_execution.jsonl"),
                   dump_inventory_sha256=sha(args.out / "dump_inventory.json"), protected_sources_unchanged=True, source_code_unchanged=True,
                   research_goal_complete=False, training_started=False)
    write(args.out / "collection_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
