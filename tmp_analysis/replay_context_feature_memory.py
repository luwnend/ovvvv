#!/usr/bin/env python3
"""FIT-only CPU smoke of frozen feature-memory interventions on quantized cache.

All predictions are sealed before manual identities are loaded. This evaluates
new ID partitions on fixed geometrically matched anchors, never old-ID pairs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'results/base_error_full_collection_20261023'
RUN = STAGE / 'full_collect_v1'
WORK = ROOT / 'work/covtrack_base_error_20261023'
PROTOCOL = ROOT / 'tmp_analysis/context_feature_memory_replay_protocol_20261024.json'
PROTOCOL_SHA = '740d6c782fdfe8240ddc8c4b924e566cebd2465a6a05233f02197b1e904aaf1b'
PREFLIGHT = ROOT / 'tmp_analysis/context_feature_memory_independent_preflight_20261024.json'
sys.path.insert(0, str(WORK))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def canonical_partition(values):
    mapping = {}
    result = []
    for i, value in enumerate(values):
        key = ('unassigned', i) if value < 0 else ('id', int(value))
        if key not in mapping:
            mapping[key] = len(mapping)
        result.append(mapping[key])
    return result


def bcubed(rows):
    """Equal source -> official GT trajectory -> unique observation weights."""
    by_source = defaultdict(list)
    for row in rows:
        by_source[row['source_group']].append(row)
    sources = {}
    for source, values in sorted(by_source.items()):
        gt = Counter((r['video_id'], r['gt_track_id']) for r in values)
        pred = Counter((r['video_id'], r['new_id']) if r['new_id'] >= 0
                       else ('singleton', r['observation_id']) for r in values)
        cell = Counter(((r['video_id'], r['gt_track_id']),
                        (r['video_id'], r['new_id']) if r['new_id'] >= 0
                        else ('singleton', r['observation_id'])) for r in values)
        precision = recall = 0.
        for r in values:
            y = (r['video_id'], r['gt_track_id'])
            z = (r['video_id'], r['new_id']) if r['new_id'] >= 0 else ('singleton', r['observation_id'])
            weight = 1. / len(gt) / gt[y]
            precision += weight * cell[y, z] / pred[z]
            recall += weight * cell[y, z] / gt[y]
        sources[source] = dict(observations=len(values), official_trajectories=len(gt),
                               precision=precision, recall=recall,
                               f1=2*precision*recall/(precision+recall),
                               unassigned=sum(r['new_id'] < 0 for r in values))
    if not sources:
        return dict(sources=0, observations=0, precision=None, recall=None, f1=None, per_source={})
    p = sum(v['precision'] for v in sources.values()) / len(sources)
    r = sum(v['recall'] for v in sources.values()) / len(sources)
    return dict(sources=len(sources), observations=len(rows), precision=p, recall=r,
                f1=2*p*r/(p+r), per_source=sources)


def score_sealed_predictions(out, protocol, frames):
    helper = ROOT / 'tmp_analysis/build_fit_identity_edge_support.py'
    require(sha(helper) == 'e68f35ca48ba6f32d519efbdb33a711fc201cbf1a2f65c50f1c828a767b1d2b5', 'Reviewed fit-only decoder changed')
    from build_fit_identity_edge_support import load_fit_identities, project_object
    support = STAGE / 'support_v1'
    receipt = project_object((support / 'support_receipt.json').read_text(), {'output_files'})
    require(sha(support / 'supervised_observations.jsonl') == receipt['output_files']['supervised_observations.jsonl'],
            'Eligible manual source changed')
    pair_audit_path = STAGE / 'pair_context_v1/independent_pair_context_audit.json'
    require(sha(pair_audit_path) == 'df7c386a8f01d3dbfce928e58758f9c455f685e8485dbfa8b19a0cd4ac936e81', 'Role provenance changed')
    pair_audit = json.loads(pair_audit_path.read_text())
    roles_path = STAGE / 'pair_context_v1/source_roles.json'
    require(sha(roles_path) == pair_audit['output_files_sha256']['source_roles.json'], 'Sealed roles changed')
    roles = json.loads(roles_path.read_text())['groups']
    identities = load_fit_identities(support / 'supervised_observations.jsonl', roles)
    manifest_path = support / 'gt_free_index_manifest.json'
    require(sha(manifest_path) == '7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667', 'Index seal changed')
    manifest = json.loads(manifest_path.read_text())
    index = support / 'gt_free_observations.jsonl'
    require(sha(index) == manifest['files'][index.name], 'GT-free index changed')
    anchors = {}
    for line in index.open():
        row = json.loads(line)
        if row['video_id'] not in protocol['video_ids']:
            continue
        identity = identities.get(row['observation_id'])
        if identity is None:
            continue
        for key in ('split', 'video_id', 'image_id', 'source_group', 'local_track_id'):
            require(row[key] == identity[key], 'Original observation provenance differs')
        require(row['split'] == 'fit', 'Non-fit anchor')
        key = (row['image_id'], row['dump_row'])
        require(key not in anchors, 'Duplicate manual anchor observation')
        anchors[key] = dict(observation_id=row['observation_id'], video_id=row['video_id'],
                            source_group=row['source_group'], gt_track_id=identity['gt_track_id'])
    results = {}
    for mode in protocol['modes']:
        rows = []
        for line in (out / (mode + '.jsonl')).open():
            record = json.loads(line)
            for j, new_id in enumerate(record['ids']):
                anchor = anchors.get((record['image_id'], j))
                if anchor is not None:
                    rows.append(dict(anchor, new_id=new_id))
        require(len(rows) == len(anchors), 'New ID scoring changed fixed eligible denominator')
        results[mode] = bcubed(rows)
    scored_hashes = {str(p): sha(p) for p in (helper, support / 'supervised_observations.jsonl',
                    support / 'support_receipt.json', roles_path, pair_audit_path, manifest_path, index)}
    require(scored_hashes[str(support / 'supervised_observations.jsonl')] == receipt['output_files']['supervised_observations.jsonl'],
            'Manual identity source changed during scoring')
    require(scored_hashes[str(index)] == manifest['files'][index.name], 'GT-free index changed during scoring')
    return dict(metrics=results, eligible_anchor_observations=len(anchors),
                selected_retained_observations=sum(r['observations'] for r in frames),
                selected_sources_without_eligible_anchors=sorted(set(protocol['source_groups']) -
                    {r['source_group'] for r in anchors.values()}),
                unknown_identity_is_negative=False,
                old_predicted_track_pairs_used=False,
                category_correctness_decoded=False,
                scoring_input_sha256=scored_hashes,
                supervised_source_sha256=sha(support / 'supervised_observations.jsonl'))


def execute():
    started = time.monotonic()
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'Predeclared protocol changed')
    protocol = json.loads(PROTOCOL.read_text())
    out = Path(protocol['output_directory'])
    require(not out.exists(), 'Preserve previous output')
    review = json.loads(PREFLIGHT.read_text())
    require(review['status'] == 'PASS_INDEPENDENT_CONTEXT_FEATURE_MEMORY_IMPLEMENTATION_PREFLIGHT', 'Missing independent preflight')
    require(review['runner_sha256'] == sha(__file__) and
            review['module_sha256'] == sha(ROOT / 'tmp_analysis/context_feature_memory.py'), 'Reviewed code changed')
    inputs = dict(protocol['input_pins'])
    for path, expected in inputs.items():
        require(sha(path) == expected, 'Input changed: ' + path)
    inventory = json.loads((RUN / 'dump_inventory.json').read_text())
    frames = [json.loads(line) for line in (RUN / 'frame_execution.jsonl').open()]
    frames = [r for r in frames if r['video_id'] in protocol['video_ids']]
    frames.sort(key=lambda r: (r['video_id'], r['frame_id']))
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame['video_id']].append(frame)
        path = Path(frame['npz'])
        require(path.parent == RUN / 'dump', 'Unexpected dump root')
        expected = inventory[path.name]
        require(frame['npz_sha256'] == expected and sha(path) == expected, 'Cached frame changed')
        inputs[str(path)] = expected
    require(set(grouped) == set(protocol['video_ids']), 'Selected video missing')
    for video in grouped.values():
        require([r['frame_id'] for r in video] == list(range(len(video))), 'Sparse sampled-step gap')
    from context_feature_memory import OVTrackerContextMemory
    from ovtrack.models.roi_heads.ovtrack_roi_head import FeatureFusionModule
    from ovtrack.models.losses.cyc_loss import CycleLoss
    run_manifest_path = RUN / 'run_manifest.json'
    require(sha(run_manifest_path) == '8a99676a2a45a61d3c76d423480f59381cfe4ab80e0a54df671417a5599198ab', 'Accepted run manifest changed')
    inputs[str(run_manifest_path)] = sha(run_manifest_path)
    manifest = json.loads(run_manifest_path.read_text())
    checkpoint = Path(manifest['checkpoint'])
    require(sha(checkpoint) == manifest['checkpoint_sha256'], 'Frozen checkpoint changed')
    inputs[str(checkpoint)] = manifest['checkpoint_sha256']
    state = torch.load(str(checkpoint), map_location='cpu')
    state = state.get('state_dict', state)
    prefix = 'roi_head.fusion_head.'
    fusion_cfg = dict(protocol['fusion']); fusion_cfg.pop('type')
    fusion = FeatureFusionModule(**fusion_cfg).eval()
    fusion.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    cycle = CycleLoss()
    torch.set_num_threads(1)
    out.mkdir()
    write_new(out / 'protocol.json', protocol)
    inputs[str(PROTOCOL)] = sha(PROTOCOL)
    inputs[str(PREFLIGHT)] = sha(PREFLIGHT)
    inputs[str(ROOT / 'tmp_analysis/context_feature_memory.py')] = sha(ROOT / 'tmp_analysis/context_feature_memory.py')
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    parity = {}
    prediction_hashes = {}
    # Complete all GT-free predictions before opening the supervision source.
    for mode in protocol['modes']:
        mode_parity = []
        output = out / (mode + '.jsonl')
        with output.open('x') as handle, torch.no_grad():
            for vid, video in sorted(grouped.items()):
                tracker_cfg = dict(protocol['tracker']); tracker_cfg.pop('type')
                tracker = OVTrackerContextMemory(memory_mode=mode, **tracker_cfg)
                tracker.set_fusion_head(fusion, cycle)
                original, replay = [], []
                for frame in video:
                    with np.load(frame['npz'], allow_pickle=False) as saved:
                        require(int(saved['image_id']) == frame['image_id'] and int(saved['video_id']) == vid, 'Frame identity mismatch')
                        boxes = torch.from_numpy(saved['det_bboxes'].copy()).float()
                        labels = torch.from_numpy(saved['det_labels'].copy()).long()
                        raw = torch.from_numpy(saved['track_feats'].astype(np.float32))
                        cls = torch.from_numpy(saved['cls_feats'].astype(np.float32))
                        original_ids = saved['track_ids'].tolist()
                    before_boxes, before_labels = boxes.clone(), labels.clone()
                    b, l, ids = tracker.match(boxes, labels, raw, cls, frame_id=frame['frame_id'],
                                               method='ovtrack-teta', pre_filtered=True)
                    require(torch.equal(b, before_boxes) and torch.equal(l, before_labels), 'Replay changed fixed geometry/labels/order/score')
                    require(len(ids) == frame['observations'], 'Replay dropped retained rows')
                    record = dict(video_id=vid, image_id=frame['image_id'], sampled_step=frame['frame_id'],
                                  original_frame_index=frame['frame_index'], ids=ids.tolist())
                    handle.write(json.dumps(record, separators=(',', ':')) + '\n')
                    original.extend(original_ids); replay.extend(ids.tolist())
                mode_parity.append(dict(video_id=vid, observations=len(original),
                    row_id_differences=sum(a != b for a, b in zip(original, replay)),
                    canonical_partition_equal=canonical_partition(original) == canonical_partition(replay)))
        prediction_hashes[output.name] = sha(output)
        parity[mode] = mode_parity
        print(json.dumps(dict(status='MODE_PREDICTIONS_SEALED', mode=mode, frames=len(frames))), flush=True)
    write_new(out / 'predictions_sealed.json', dict(status='ALL_FIVE_GT_FREE_MODE_PREDICTIONS_SEALED_BEFORE_SUPERVISION',
        output_files_sha256=prediction_hashes, protocol_sha256=PROTOCOL_SHA))
    scores = score_sealed_predictions(out, protocol, frames)
    inputs.update(scores['scoring_input_sha256'])
    for path, expected in inputs.items():
        require(sha(path) == expected, 'Input changed during replay: ' + path)
    for name, expected in prediction_hashes.items():
        require(sha(out / name) == expected, 'Predictions changed during scoring')
    write_new(out / 'input_hashes.json', inputs)
    summary = dict(status='COMPLETE_QUANTIZED_FIT_FEATURE_MEMORY_SMOKE_PENDING_INDEPENDENT_AUDIT',
        modes=protocol['modes'], primary=protocol['primary'], selected_sources=len(protocol['source_groups']),
        selected_videos=len(grouped), selected_frames=len(frames), scoring=scores,
        parity_with_original_GPU_cache=parity, elapsed_seconds=time.monotonic()-started,
        predictions_sealed_before_manual_identity_join=True, GT_used_in_predictions=False,
        category_correctness_decoded=False, tune_test_or_class_holdout_correctness_opened=False,
        GPU_used=False, training_started=False, research_goal_complete=False,
        protocol_sha256=PROTOCOL_SHA, output_files_sha256=prediction_hashes,
        input_hashes_sha256=sha(out / 'input_hashes.json'),
        scope='Sparse fit146 manual-anchor identity subset on same quantized fixed detections; BCubed diagnostics, no official TETA or efficacy claim')
    write_new(out / 'summary.json', summary)
    print(json.dumps(dict(status=summary['status'], elapsed_seconds=summary['elapsed_seconds'],
                         metrics={k:{m:v[m] for m in ('precision','recall','f1')} for k,v in scores['metrics'].items()})), flush=True)


def self_test():
    require(canonical_partition([4, 4, 10, -1, -1]) == [0, 0, 1, 2, 3], 'ID relabel/singleton error')
    def rows(ids):
        return [dict(source_group='s', video_id=1, gt_track_id=y,
                     observation_id=i, new_id=z) for i, (y, z) in enumerate(zip([5, 5, 6, 6], ids))]
    perfect = bcubed(rows([0, 0, 1, 1])); split = bcubed(rows([-1, -1, -1, -1])); merged = bcubed(rows([0, 0, 0, 0]))
    require(perfect['precision'] == perfect['recall'] == 1., 'Perfect partition wrong')
    require(split['precision'] == 1. and split['recall'] == .5, 'Singleton denominator changed')
    require(merged['precision'] == .5 and merged['recall'] == 1., 'Known different identity metric wrong')
    weighted = rows([0,0,1,1]) + [dict(source_group='other', video_id=2, gt_track_id=7, observation_id=9, new_id=-1)]
    require(bcubed(weighted)['precision'] == bcubed(weighted)['recall'] == 1., 'Source equal weighting wrong')
    print(json.dumps(dict(status='PASS_REPLAY_SCORING_SYNTHETIC', real_cache_opened=False,
                         real_supervision_opened=False, GPU_used=False, runner_sha256=sha(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    require(args.execute != args.self_test, 'Choose exactly one action')
    self_test() if args.self_test else execute()
