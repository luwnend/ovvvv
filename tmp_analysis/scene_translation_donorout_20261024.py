#!/usr/bin/env python3
"""GT-free donor-out predictive diagnostic; independent of the running OOF."""
import argparse
from collections import defaultdict, deque
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'tmp_analysis/scene_translation_donorout_protocol_20261024.json'
ARMS = ('fullpool', 'row_excluded', 'proposal_overlap_excluded')
FIT_SLOTS = ('available', 'anchor_count', 'translation_x', 'translation_y',
             'MAD_x', 'MAD_y', 'fixed_donor_LOO_max', 'innovation_x',
             'innovation_y', 'corrected_normalized_L1', 'gain')
COLUMNS = ('raw_cosine', 'query_margin', 'reference_margin', 'static_normalized_L1') + tuple(
    arm + '/' + slot for arm in ARMS for slot in FIT_SLOTS)
LOADER_MEMBERS = ('det_bboxes', 'track_feats', 'roi_proposal_row',
                  'video_id', 'image_id', 'frame_id', 'frame_index')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest_array(value):
    return np.frombuffer(hashlib.sha256(canonical(value)).digest(), dtype=np.uint8)


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def component(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate(protocol):
    spec = json.loads(Path(protocol).read_text())
    require(spec['status'] == 'FROZEN_GTFREE_SCENE_TRANSLATION_DONOROUT_V1', 'Not frozen')
    require(spec['lags'] == [1, 4] and spec['minimum_anchors'] == 3, 'Fixed design changed')
    require(spec['arms'] == list(ARMS) and spec['columns'] == list(COLUMNS), 'Schema changed')
    for path, digest in spec['input_pins'].items():
        require(sha(path) == digest, 'Input changed: ' + path)
    require(spec['producer_path'] == str(Path(__file__).resolve()), 'Producer path changed')
    require(sha(__file__) == spec['producer_sha256'], 'Producer changed')
    authority = json.loads(Path(spec['protected_authority_path']).read_text())
    require(len(authority['protected_files']) == 7, 'Protected scope changed')
    for path, record in authority['protected_files'].items():
        require(Path(path).exists() == record['exists'], 'Protected existence changed')
        require((sha(path) if Path(path).exists() else None) == record['sha256'], 'Protected bytes changed')
    return spec


def prepare_metadata(spec):
    scanner = component(spec['metadata_projection_helper'], 'donorout_metadata_projection')
    roles = json.loads(Path(spec['roles_path']).read_text())
    roles = roles.get('groups', roles)
    sources = sorted(s for s, row in roles.items() if row['split'] == 'fit')
    require(len(sources) == 341 and all(roles[s]['role'] in ('mean_role', 'pair_role') for s in sources), 'FIT metadata population changed')
    # Only video_records are decoded; counts in these metadata records are never
    # used for selection, prediction, weights or diagnostic target construction.
    plan = scanner.project_object(Path(spec['plan_path']).read_text(), {'video_records'})
    video_sources = {int(row['video_id']): row['source_group'] for row in plan['video_records']
                     if row['split'] == 'fit' and row['source_group'] in sources}
    inventory = json.loads(Path(spec['inventory_path']).read_text())
    videos = defaultdict(list)
    with Path(spec['frame_execution_path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if int(row['video_id']) not in video_sources:
                continue
            require(row['npz_sha256'] == inventory[Path(row['npz']).name], 'Inventory mismatch')
            videos[int(row['video_id'])].append({key: row[key] for key in
                ('video_id', 'image_id', 'frame_id', 'frame_index', 'observations', 'npz', 'npz_sha256')})
    for video, records in videos.items():
        records.sort(key=lambda row: row['frame_id'])
        require([r['frame_id'] for r in records] == list(range(len(records))), 'Incomplete sampled video')
        require(len(records) <= 41 and all(r['observations'] <= 77 for r in records), 'Frozen resource bound exceeded')
        require(all(a['frame_index'] < b['frame_index'] for a, b in zip(records, records[1:])), 'Timeline changed')
    require(len(videos) == 403 and sum(map(len, videos.values())) == 13791, 'Frame population changed')
    require(sum(r['observations'] for records in videos.values() for r in records) == 613536, 'Rows changed')
    require(set(video_sources[v] for v in videos) == set(sources), 'Source coverage changed')
    return sources, video_sources, dict(videos)


def load_frame(record):
    require(sha(record['npz']) == record['npz_sha256'], 'Raw cache changed')
    with np.load(record['npz'], allow_pickle=False) as saved:
        # Explicit allowlist: no category, released ID, posterior or GT value.
        for key, field in (('video_id', 'video_id'), ('image_id', 'image_id'),
                           ('frame_id', 'frame_id'), ('frame_index', 'frame_index')):
            require(int(saved[key]) == record[field], 'Frame provenance changed')
        boxes = np.ascontiguousarray(saved['det_bboxes'], dtype=np.float64)
        raw = np.ascontiguousarray(saved['track_feats'], dtype=np.float64)
        proposals = saved['roi_proposal_row'].copy()
    n = record['observations']
    require(boxes.shape == (n, 5) and raw.shape == (n, 256), 'Unexpected retained shapes')
    require(proposals.shape == (n,) and proposals.dtype.kind in 'iu' and np.all(proposals >= 0), 'Invalid proposal locators')
    return dict(record=record, boxes=np.ascontiguousarray(boxes[:, :4]), raw=raw,
                proposals=np.ascontiguousarray(proposals, dtype=np.int64))


def strict_exclusions(frame, row):
    boxes = frame['boxes']
    overlap = np.all(np.minimum(boxes[:, 2:], boxes[row, 2:]) >
                     np.maximum(boxes[:, :2], boxes[row, :2]), axis=1)
    return np.flatnonzero(overlap | (frame['proposals'] == frame['proposals'][row])).tolist()


def encode_fit(fit, query_box, reference_box):
    values = np.full(len(FIT_SLOTS), np.nan)
    values[:2] = [float(fit['available']), fit['anchor_count']]
    if not fit['available']:
        return values
    scale = np.sqrt(query_box[2:] - query_box[:2]) * np.sqrt(reference_box[2:] - reference_box[:2])
    qc = query_box[:2] / 2 + query_box[2:] / 2
    rc = reference_box[:2] / 2 + reference_box[2:] / 2
    displacement = qc - rc
    innovation = (displacement - np.asarray(fit['translation'])) / scale
    static = np.abs(displacement / scale).sum()
    corrected = np.abs(innovation).sum()
    values[2:] = (fit['translation'] + fit['residual_mad_xy'] +
                  [fit['max_leave_one_anchor_shift']] + innovation.tolist() +
                  [float(corrected), float(static - corrected)])
    require(np.isfinite(values).all(), 'Nonfinite available geometry')
    return values


def predict_frame_pair(query, reference, engine_class):
    engine = engine_class(query['boxes'], reference['boxes'], query['raw'], reference['raw'])
    holdouts = engine.fullpool_mnn()
    full = engine.fit(minimum_anchors=3)
    n = len(holdouts)
    values = np.full((n, len(COLUMNS)), np.nan)
    locators = np.empty((n, 2), np.int64)
    anchors = np.empty((n, 3, 32), np.uint8)
    exclusions = np.empty((n, 3, 32), np.uint8)
    for index, h in enumerate(holdouts):
        qi, ri = h['query_row'], h['reference_row']
        qbox, rbox = query['boxes'][qi], reference['boxes'][ri]
        scale = np.sqrt(qbox[2:] - qbox[:2]) * np.sqrt(rbox[2:] - rbox[:2])
        qc, rc = qbox[:2] / 2 + qbox[2:] / 2, rbox[:2] / 2 + rbox[2:] / 2
        static = np.abs((qc - rc) / scale).sum()
        require(np.isfinite(static), 'Nonfinite held-out static geometry')
        values[index, :4] = [h['raw_cosine'], np.nan if h['query_margin'] is None else h['query_margin'],
                             np.nan if h['reference_margin'] is None else h['reference_margin'], static]
        locators[index] = [qi, ri]
        sets = [([], []), ([qi], [ri]),
                (strict_exclusions(query, qi), strict_exclusions(reference, ri))]
        for arm, (qe, re) in enumerate(sets):
            fit = full if arm == 0 else engine.fit(excluded_query_rows=qe, excluded_reference_rows=re, minimum_anchors=3)
            values[index, 4 + arm * 11:4 + (arm + 1) * 11] = encode_fit(fit, qbox, rbox)
            anchors[index, arm] = digest_array([[a['query_row'], a['reference_row']] for a in fit['anchors']])
            exclusions[index, arm] = digest_array(dict(query=qe, reference=re))
    return values, locators, anchors, exclusions


def predict_video(records, engine_class, loader=load_frame):
    """Shared full-video path for synthetic fixtures and the real producer."""
    ring=deque(maxlen=5);receipts=[];parts=[];locs=[];ah=[];eh=[];offset=0
    candidates={1:0,4:0}
    for record in records:
        query=loader(record);ring.append(query)
        for lag in (1,4):
            reference=ring[-lag-1] if record['frame_id']>=lag else None
            item=dict(lag=lag,query_step=record['frame_id'],query_image_id=record['image_id'],
                query_original_frame=record['frame_index'],query_rows=len(query['boxes']),
                reference_step=None if reference is None else reference['record']['frame_id'],
                reference_image_id=None if reference is None else reference['record']['image_id'],
                reference_original_frame=None if reference is None else reference['record']['frame_index'],
                reference_rows=0 if reference is None else len(reference['boxes']),
                offset=offset,holdouts=0,
                reason='missing_reference_prefix' if reference is None else 'reference_available')
            candidates[lag]+=item['query_rows']*item['reference_rows']
            if reference is not None:
                values,locations,anchors,exclusions=predict_frame_pair(query,reference,engine_class)
                item['holdouts']=len(values)
                parts.append(values);locs.append(np.column_stack((np.full(len(values),len(receipts),np.int64),locations)))
                ah.append(anchors);eh.append(exclusions);offset+=len(values)
            receipts.append(item)
    arrays=dict(values=np.concatenate(parts) if parts else np.empty((0,37)),
        rowloc=np.concatenate(locs) if locs else np.empty((0,3),np.int64),
        anchor_hashes=np.concatenate(ah) if ah else np.empty((0,3,32),np.uint8),
        exclusion_hashes=np.concatenate(eh) if eh else np.empty((0,3,32),np.uint8))
    return arrays,receipts,candidates


def summarize(sources, video_sources, video_outputs, root):
    # No semantic/identity labels are loaded. Every saved holdout, including
    # unavailable fits, remains represented in coverage and frame receipts.
    selectors = {arm: (i,) for i, arm in enumerate(ARMS)}
    selectors['common_row_strict'] = (1, 2)
    selectors['common_all_three'] = (0, 1, 2)
    rows = {name: {lag: {s: [] for s in sources} for lag in (1, 4)} for name in selectors}
    source_holdouts = {lag: dict.fromkeys(sources, 0) for lag in (1, 4)}
    frame_counts = {lag: 0 for lag in (1, 4)}
    for video, output in sorted(video_outputs.items()):
        source = video_sources[video]
        receipt = json.loads((root / output['receipt']).read_text())
        with np.load(root / output['npz'], allow_pickle=False) as saved:
            values = saved['values']
            for frame in receipt['frame_pairs']:
                lag = frame['lag'];frame_counts[lag] += 1
                block = values[frame['offset']:frame['offset'] + frame['holdouts']]
                source_holdouts[lag][source] += len(block)
                for name, arms in selectors.items():
                    valid = np.ones(len(block), bool)
                    for arm in arms:
                        valid &= block[:, 4 + 11 * arm] == 1
                    if not valid.any():
                        continue
                    rows[name][lag][source].append(dict(
                        query_step=frame['query_step'], video_id=video,
                        holdouts=int(valid.sum()), gains={ARMS[a]:float(np.median(block[valid, 4 + 11 * a + 10])) for a in arms}))
    report = dict(status='GTFREE_PREDICTABILITY_ONLY_NOT_IDENTITY_OR_TRACKING_EVALUATION',
                  sources=sources, frame_receipts_by_lag=frame_counts, readouts={})
    for name, per_lag in rows.items():
        result = {}
        for lag, per_source in per_lag.items():
            arms = selectors[name];details = {}
            for source, frames in per_source.items():
                details[source] = dict(fullpool_holdouts=source_holdouts[lag][source],
                    supported_frame_pairs=len(frames), valid_holdouts=sum(f['holdouts'] for f in frames),
                    source_mean_frame_median_gain={ARMS[a]: None if not frames else float(np.mean([f['gains'][ARMS[a]] for f in frames])) for a in arms})
            supported = [s for s in sources if details[s]['supported_frame_pairs']]
            result[str(lag)] = dict(supported_sources=len(supported),
                source_macro_gain={ARMS[a]: None if not supported else float(np.mean([details[s]['source_mean_frame_median_gain'][ARMS[a]] for s in supported])) for a in arms},
                by_source=details)
        common = [s for s in sources if all(result[str(l)]['by_source'][s]['supported_frame_pairs'] for l in (1, 4))]
        result['common_supported_sources_across_lags'] = dict(sources=common,
            source_macro_gain_by_lag={str(l): {ARMS[a]: None if not common else float(np.mean([result[str(l)]['by_source'][s]['source_mean_frame_median_gain'][ARMS[a]] for s in common])) for a in selectors[name]} for l in (1, 4)})
        report['readouts'][name] = result
    main = report['readouts'][ARMS[2]]
    passed = all(main[str(l)]['source_macro_gain'][ARMS[2]] is not None and main[str(l)]['source_macro_gain'][ARMS[2]] > 0 for l in (1, 4))
    report['minimal_gate'] = 'PASS_MINIMAL_GTFREE_DONOR_PREDICTABILITY' if passed else 'NOT_PASSED_MINIMAL_GTFREE_DONOR_PREDICTABILITY'
    report['method_novelty_or_identity_effectiveness_established'] = False
    return report


def execute(protocol):
    started = time.monotonic();spec = authenticate(protocol)
    out = Path(spec['output_directory'])
    require(not out.exists(), 'Never overwrite or resume this diagnostic output')
    sources, video_sources, videos = prepare_metadata(spec)
    engine_class = component(spec['cached_module_path'], 'donorout_cached_engine').CachedSceneMotion
    out.mkdir();(out / 'videos').mkdir()
    write_new(out / 'metadata_manifest.json', dict(sources=sources, video_sources=video_sources,
        videos={str(v):r for v,r in videos.items()}, input_pins=spec['input_pins'],
        allowed_NPZ_members=list(LOADER_MEMBERS), identity_or_category_values_decoded=False))
    outputs = {};candidate_counts = {1:0, 4:0};holdout_count=0
    for index, (video, records) in enumerate(sorted(videos.items())):
        arrays,receipts,candidates=predict_video(records,engine_class)
        for lag in (1,4):candidate_counts[lag]+=candidates[lag]
        offset=len(arrays['values'])
        npz_name='videos/v%d.npz'%video;receipt_name='videos/v%d.json'%video
        with (out / npz_name).open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        write_new(out / receipt_name,dict(video_id=video, source=video_sources[video],
            columns=list(COLUMNS), arms=list(ARMS), frame_pairs=receipts,holdouts=offset,
            npz=npz_name,npz_sha256=sha(out / npz_name)))
        outputs[video]=dict(npz=npz_name,npz_sha256=sha(out / npz_name),
            receipt=receipt_name,receipt_sha256=sha(out / receipt_name))
        holdout_count+=offset
        if index%20==0 or index==len(videos)-1:
            print(json.dumps(dict(completed_videos=index+1,total_videos=len(videos),
                holdouts=holdout_count,elapsed_seconds=time.monotonic()-started)),flush=True)
    require(candidate_counts == {1:29096237,4:26255927},'Full frame-pair coverage changed')
    seal=dict(status='SEALED_COMPLETE_GTFREE_DONOR_PREDICTIONS',protocol_sha256=sha(protocol),
        candidate_pairs_by_lag=candidate_counts,total_holdouts=holdout_count,video_outputs=outputs,
        metadata_manifest_sha256=sha(out / 'metadata_manifest.json'),input_pins=spec['input_pins'])
    write_new(out / 'GT_free_predictions_sealed.json',seal)
    summary=summarize(sources,video_sources,outputs,out)
    write_new(out / 'summary.json',summary)
    authenticate(protocol)
    write_new(out / 'producer_receipt.json',dict(status='COMPLETE_GTFREE_DONOR_PREDICTION_AND_DESCRIPTIVE_READOUT',
        elapsed_seconds=time.monotonic()-started,seal_sha256=sha(out/'GT_free_predictions_sealed.json'),
        summary_sha256=sha(out/'summary.json'),no_identity_labels_or_new_OOF_outputs_loaded=True))
    return dict(output_directory=str(out),holdouts=holdout_count,elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    args=parser.parse_args()
    print(json.dumps(execute(args.protocol),indent=2))
