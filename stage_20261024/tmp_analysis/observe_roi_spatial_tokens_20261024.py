"""Four-frame, untrained spatial observation with exact frozen-cache parity."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def authenticate(protocol):
    spec = json.loads(protocol.read_text())
    require(spec['status'] == 'FROZEN_FOUR_FRAME_ROI_SPATIAL_OBSERVATION_V1', 'Protocol not frozen')
    require(sha(__file__) == spec['runner_sha256'], 'Runner changed')
    for path, digest in spec['input_pins'].items():
        require(sha(path) == digest, 'Input pin changed: ' + path)
    parent = json.loads(Path(spec['parent_collection_manifest']).read_text())
    for path, digest in parent['source_hashes'].items():
        require(sha(Path(spec['code_root']) / path) == digest, 'Original model code changed: ' + path)
    authority = json.loads(Path(spec['protected_authority_path']).read_text())
    require(len(authority['protected_files']) == 7, 'Protected scope changed')
    for path, record in authority['protected_files'].items():
        require(Path(path).exists() == record['exists'] and
                (sha(path) if record['exists'] else None) == record['sha256'], 'Protected bytes changed')
    return spec


def gpu_guard(uuid):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == uuid, 'Device visibility differs from frozen UUID')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).splitlines()
    row = [line.split(', ') for line in rows if line.startswith(uuid + ',')]
    require(len(row) == 1 and int(row[0][2]) > 4096 and int(row[0][3]) >= 15000, 'GPU capacity/free memory')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    owners = []
    for line in apps.splitlines():
        device, pid, used = [part.strip() for part in line.split(',')]
        if device != uuid:
            continue
        owner = subprocess.check_output(['ps', '-o', 'user=', '-p', pid], text=True).strip()
        require(owner and owner != 'chenhc', 'Existing chenhc process on selected GPU')
        owners.append(dict(pid=int(pid), owner=owner, memory_mib=int(used)))
    return dict(uuid=uuid, name=row[0][1], free_mib=int(row[0][3]), preexisting_processes=owners)


def execute(protocol):
    started = time.monotonic(); spec = authenticate(protocol)
    out = Path(spec['output_directory'])
    require(not out.exists(), 'Never overwrite a pilot or retry it automatically')
    require(len(spec['frames']) == 4, 'Fixed four-frame scope changed')
    device = gpu_guard(spec['GPU_UUID'])
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        require(os.environ.get(name) == '1', 'Single CPU thread environment required')
    require(not os.environ.get('OVMOT_TCES_DUMP_DIR'), 'Original cache writer must be disabled')
    sys.path.insert(0, spec['code_root'])
    import numpy as np
    import torch
    import mmcv
    import ovtrack
    from mmdet.datasets.builder import PIPELINES
    from ovtrack.apis import init_model, inference_model
    require(str(Path(ovtrack.__file__).resolve()).startswith(spec['code_root'] + '/'), 'Wrong model import')
    @PIPELINES.register_module()
    class CollectionFrameStep:
        def __call__(self, data):
            data['frame_id'] = int(data['img_info']['frame_id'])
            return data
    torch.set_num_threads(1); torch.manual_seed(20261023); np.random.seed(20261023)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    cfg = mmcv.Config.fromfile(spec['config_path'])
    cfg.model.backbone.init_cfg = None
    require(cfg.model.roi_head.type == 'OVTrackRoIHeadUncertainty', 'Wrong active model')
    require(not cfg.model.roi_head.init_track_head_by_bbox_head and not cfg.model.tracker.vis, 'Unexpected mutable/GT-assisted inference')
    out.mkdir()
    write(out / 'started.json', dict(status='RUNNING_ORDINARY_SPATIAL_OBSERVATION',
        protocol_sha256=sha(protocol), pid=os.getpid(), device=device, training=False,
        GPU_minutes_planned=[1, 5], no_identity_class_or_new_experiment_outputs_loaded=True))
    model = init_model(cfg, spec['checkpoint_path'], device='cuda:0')
    model.requires_grad_(False)
    loader = importlib.util.spec_from_file_location('roi_spatial_trace', spec['trace_path'])
    trace_module = importlib.util.module_from_spec(loader); loader.loader.exec_module(trace_module)
    head = model.roi_head.track_head
    require(head.num_convs == 4 and head.num_fcs == 1, 'Frozen track-head structure changed')
    trace = trace_module.ROITrackSpatialTrace(head)
    original_track_forward = model.roi_head._track_forward
    original_simple_test = model.simple_test
    observed_boxes, observed_meta = [], []
    def track_forward(x, boxes):
        observed_boxes.append([box.detach().clone() for box in boxes])
        return original_track_forward(x, boxes)
    def simple_test(img, img_metas, rescale=False):
        require(rescale is True and len(img_metas) == 1, 'Expected single augmented image and rescaled detector boxes')
        observed_meta.append(dict(scale_factor=np.asarray(img_metas[0]['scale_factor']).tolist(),
            ori_shape=list(img_metas[0]['ori_shape']), img_shape=list(img_metas[0]['img_shape']),
            pad_shape=list(img_metas[0]['pad_shape'])))
        return original_simple_test(img, img_metas, rescale=rescale)
    model.roi_head._track_forward = track_forward
    model.simple_test = simple_test
    receipts = []
    try:
        for frame in spec['frames']:
            require(sha(frame['cache_path']) == frame['cache_sha256'] and sha(frame['image_path']) == frame['image_sha256'], 'Frame input changed')
            with np.load(frame['cache_path'], allow_pickle=False) as saved:
                # This separate observation protocol allows only these fields;
                # never invoke the collector's label/posterior validator.
                expected = {key: saved[key].copy() for key in spec['cache_member_allowlist']}
            if hasattr(model, 'tracker'):
                model.tracker.keep_mask = None
            previous_tracker = getattr(model, 'tracker', None)
            observed_boxes.clear(); observed_meta.clear()
            trace.begin()
            inference_model(model, frame['image_path'], frame['frame_id'])
            torch.cuda.synchronize()
            require((model.tracker is not previous_tracker) == (frame['frame_id'] == 0), 'Tracker reset differs from original timeline')
            keep = model.tracker.keep_mask.detach().cpu().numpy()
            require(keep.dtype == np.dtype('bool') and np.array_equal(keep, expected['tracker_keep_mask']), 'Retained row selection changed')
            captured = trace.take(keep)
            require(len(observed_boxes) == len(observed_meta) == 1 and len(observed_boxes[0]) == 1,
                    'Unexpected track ROI/preprocessing call count')
            actual_boxes = observed_boxes[0][0]
            scale = torch.tensor(observed_meta[0]['scale_factor'], device=actual_boxes.device, dtype=actual_boxes.dtype)
            target_boxes = torch.as_tensor(expected['pretracker_det_bboxes'][:, :4], device=actual_boxes.device) * scale
            require(torch.equal(actual_boxes, target_boxes) and len(actual_boxes) == len(captured['full_raw_output']),
                    'Pretracker geometry/row order differs from parent cache')
            raw = captured['raw_output'].cpu().numpy()
            rows = captured['pretracker_det_row'].cpu().numpy()
            require(np.array_equal(rows, expected['pretracker_det_row']), 'Track ROI/detection row locator mismatch')
            full_half = np.asarray(captured['full_raw_output'].cpu().numpy(), dtype=np.float16)
            require(raw.shape == expected['track_feats'].shape and
                    np.array_equal(full_half[keep], expected['track_feats']), 'Actual raw head output differs from frozen cache')
            require(tuple(captured['roi_input'].shape[1:]) == (256, 7, 7), 'Spatial token shape changed')
            target = out / ('v%d_f%d.npz' % (frame['video_id'], frame['frame_id']))
            arrays = dict(roi_input=captured['roi_input'].cpu().numpy(),
                conv_spatial=captured['conv_spatial'].cpu().numpy(),
                raw_output=raw, pretracker_det_row=rows,
                roi_proposal_row=expected['roi_proposal_row'], det_bboxes=expected['det_bboxes'][:, :4])
            with target.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            receipt = dict(video_id=frame['video_id'], frame_id=frame['frame_id'],
                image_id=frame['image_id'], retained_rows=len(raw),
                spatial_shape=list(arrays['roi_input'].shape), native_full_batch_FC_replay_exact=True,
                cached_float16_raw_bitwise_exact=True, retained_row_locator_exact=True,
                saved_token_dtype=str(arrays['conv_spatial'].dtype), layout='NCHW',
                native_raw_dtype=str(raw.dtype), preprocessing=observed_meta[0],
                saved_retained_subset_FC_replay_not_yet_verified=True,
                file=target.name, sha256=sha(target), elapsed_seconds=time.monotonic()-started)
            receipts.append(receipt); print(json.dumps(receipt), flush=True)
    finally:
        trace.close()
        model.roi_head._track_forward = original_track_forward
        model.simple_test = original_simple_test
    authenticate(protocol)
    value = dict(status='PASS_FOUR_FRAME_SPATIAL_OBSERVATION_AND_RAW_CACHE_PARITY',
        protocol_sha256=sha(protocol), frames=receipts, total_retained_rows=sum(r['retained_rows'] for r in receipts),
        elapsed_seconds=time.monotonic()-started, device=device,
        no_training_or_method_effectiveness_or_novelty_established=True,
        all_input_pins_and_protected7_unchanged=True,
        raw_token_parity_scope='native full-batch FC replay; actual raw CPU NumPy castfloat16 equals parent cache; saved NCHW float32 tokens are retained subset, whose separate FC replay has not yet been verified')
    write(out / 'observation_receipt.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(execute(arguments.protocol), indent=2))
