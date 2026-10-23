#!/usr/bin/env python3
"""Independent scalar oracle and synthetic fixtures for donor-out translation.

The synthetic entrypoint does not read real cache, labels, identity joins, OOF
outputs. The frozen-schema --verify entrypoint independently reconstructs every
saved frame/heldout and readout from its allowed raw inputs. It imports neither
the producer nor its cached reducer. Pure synthetic QA can import their APIs.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parents[1]
CACHED_MODULE = ROOT / 'tmp_analysis/cached_endpoint_scene_motion_20261024.py'
CACHED_SHA256 = '5eaf00fa1bd52ab0c5a3430fcdf90fd9609d4c17fdc3d81db06cf25bc41b0f32'
ARMS = ('fullpool', 'row_excluded', 'proposal_overlap_excluded')
FIT_SLOTS = ('available', 'anchor_count', 'translation_x', 'translation_y',
             'MAD_x', 'MAD_y', 'fixed_donor_LOO_max', 'innovation_x',
             'innovation_y', 'corrected_normalized_L1', 'gain')
COLUMNS = ('raw_cosine', 'query_margin', 'reference_margin', 'static_normalized_L1') + tuple(
    arm + '/' + slot for arm in ARMS for slot in FIT_SLOTS)
LOADER_MEMBERS = ('det_bboxes', 'track_feats', 'roi_proposal_row',
                  'video_id', 'image_id', 'frame_id', 'frame_index')
EXPECTED_POPULATION = dict(sources=341, videos=403, frames=13791, rows=613536,
    candidates={1: 29096237, 4: 26255927}, prefix={1: 403, 4: 1606}, frame_receipts=27582)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def list_hash(value):
    return hashlib.sha256(canonical_json(value)).digest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def median(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Median of empty values')
    size = len(ordered)
    return (ordered[size // 2] if size % 2 else
            ordered[size // 2 - 1] / 2 + ordered[size // 2] / 2)


def normalize_scalar(vector):
    vector = [float(value) for value in vector]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError('Features must be finite nonempty vectors')
    peak = max(abs(value) for value in vector)
    if not peak:
        return None
    scaled = [value / peak for value in vector]
    norm = math.sqrt(sum(value * value for value in scaled))
    return [value / norm for value in scaled]


def validate_side(boxes, features):
    if len(boxes) != len(features):
        raise ValueError('Box and feature rows must align')
    dimensions = {len(feature) for feature in features}
    if len(dimensions) > 1:
        raise ValueError('Feature dimensions must agree')
    for box in boxes:
        if (len(box) != 4 or not all(math.isfinite(float(x)) for x in box)
                or not box[2] > box[0] or not box[3] > box[1]
                or not math.isfinite(box[2] - box[0])
                or not math.isfinite(box[3] - box[1])):
            raise ValueError('Finite positive xyxy boxes required')
    return [normalize_scalar(feature) for feature in features]


def validate_exclusions(excluded, count):
    excluded = list(excluded)
    if any(type(row) is not int or not 0 <= row < count for row in excluded):
        raise ValueError('Exclusions must be integer row locators in range')
    return set(excluded)


def scalar_mnn(query_boxes, reference_boxes, query_features, reference_features,
               excluded_query_rows=(), excluded_reference_rows=(), canonical_similarity=None):
    qnorm = validate_side(query_boxes, query_features)
    rnorm = validate_side(reference_boxes, reference_features)
    dimensions = {len(feature) for feature in query_features + reference_features}
    if len(dimensions) > 1:
        raise ValueError('Query and reference feature dimensions differ')
    qe = validate_exclusions(excluded_query_rows, len(query_boxes))
    re = validate_exclusions(excluded_reference_rows, len(reference_boxes))
    qs = [i for i, value in enumerate(qnorm) if i not in qe and value is not None]
    rs = [j for j, value in enumerate(rnorm) if j not in re and value is not None]
    if canonical_similarity is None:
        similarities = {(i, j): sum(x * y for x, y in zip(qnorm[i], rnorm[j]))
                        for i in qs for j in rs}
    else:
        if (len(canonical_similarity) != len(query_boxes)
                or any(len(row) != len(reference_boxes) for row in canonical_similarity)):
            raise ValueError('Canonical similarity shape mismatch')
        similarities = {(i, j): float(canonical_similarity[i][j]) for i in qs for j in rs}
    anchors = []
    for i in qs:
        for j in rs:
            value = similarities[i, j]
            # Strict comparisons independently express both unique maxima.
            if not (value > 0
                    and all(value > similarities[i, k] for k in rs if k != j)
                    and all(value > similarities[k, j] for k in qs if k != i)):
                continue
            qa = [similarities[i, k] for k in rs if k != j]
            ra = [similarities[k, j] for k in qs if k != i]
            anchors.append(dict(query_row=i, reference_row=j,
                raw_cosine=value,
                query_margin=None if not qa else value - max(qa),
                reference_margin=None if not ra else value - max(ra)))
    return anchors


def center(box):
    return [box[axis] / 2 + box[axis + 2] / 2 for axis in (0, 1)]


def scalar_fit(query_boxes, reference_boxes, query_features, reference_features,
               excluded_query_rows=(), excluded_reference_rows=(), canonical_similarity=None):
    qe = validate_exclusions(excluded_query_rows, len(query_boxes))
    re = validate_exclusions(excluded_reference_rows, len(reference_boxes))
    anchors = scalar_mnn(query_boxes, reference_boxes, query_features,
                         reference_features, sorted(qe), sorted(re), canonical_similarity)
    result = dict(available=len(anchors) >= 3, anchor_count=len(anchors),
                  anchors=anchors, translation=None, residual_mad_xy=None,
                  max_leave_one_anchor_shift=None,
                  excluded_query_rows=sorted(qe), excluded_reference_rows=sorted(re))
    if not result['available']:
        return result
    displacement = []
    for anchor in anchors:
        qc = center(query_boxes[anchor['query_row']])
        rc = center(reference_boxes[anchor['reference_row']])
        displacement.append([qc[axis] - rc[axis] for axis in (0, 1)])
    translation = [median([item[axis] for item in displacement]) for axis in (0, 1)]
    mad = [median([abs(item[axis] - translation[axis]) for item in displacement])
           for axis in (0, 1)]
    leave_shifts = []
    for held in range(len(displacement)):
        remaining = [item for i, item in enumerate(displacement) if i != held]
        leave = [median([item[axis] for item in remaining]) for axis in (0, 1)]
        leave_shifts.append(math.hypot(leave[0] - translation[0],
                                      leave[1] - translation[1]))
    result.update(translation=translation, residual_mad_xy=mad,
                  max_leave_one_anchor_shift=max(leave_shifts))
    if not all(math.isfinite(value) for value in
               translation + mad + [max(leave_shifts)]):
        raise ValueError('Unrepresentable finite geometry')
    return result


def positive_area_overlap(left, right):
    return (min(left[2], right[2]) > max(left[0], right[0])
            and min(left[3], right[3]) > max(left[1], right[1]))


def strict_exclusions(boxes, proposal_rows, endpoint_row):
    if len(boxes) != len(proposal_rows):
        raise ValueError('Proposal locators must align with retained rows')
    validate_exclusions([endpoint_row], len(boxes))
    return [row for row, box in enumerate(boxes)
            if row == endpoint_row
            or proposal_rows[row] == proposal_rows[endpoint_row]
            or positive_area_overlap(box, boxes[endpoint_row])]


def scalar_metrics(query_box, reference_box, fit):
    qc, rc = center(query_box), center(reference_box)
    scale = [math.sqrt(query_box[axis + 2] - query_box[axis])
             * math.sqrt(reference_box[axis + 2] - reference_box[axis])
             for axis in (0, 1)]
    displacement = [qc[axis] - rc[axis] for axis in (0, 1)]
    static = sum(abs(displacement[axis] / scale[axis]) for axis in (0, 1))
    if not fit['available']:
        return dict(static_l1=static, innovation=None, corrected_l1=None, gain=None)
    innovation = [(displacement[axis] - fit['translation'][axis]) / scale[axis]
                  for axis in (0, 1)]
    corrected = sum(abs(value) for value in innovation)
    values = [static, corrected, static - corrected] + innovation
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Unrepresentable normalized residual')
    return dict(static_l1=static, innovation=innovation, corrected_l1=corrected,
                gain=static - corrected)


def scalar_frame(query_boxes, reference_boxes, query_features, reference_features,
                 query_proposal_rows, reference_proposal_rows, canonical_similarity=None):
    """All full-pool held-out donors and all arms, retaining unsupported entries."""
    full_fit = scalar_fit(query_boxes, reference_boxes, query_features, reference_features,
                          canonical_similarity=canonical_similarity)
    output = []
    for held in full_fit['anchors']:
        i, j = held['query_row'], held['reference_row']
        masks = dict(fullpool=([], []), row=([i], [j]), strict=(
            strict_exclusions(query_boxes, query_proposal_rows, i),
            strict_exclusions(reference_boxes, reference_proposal_rows, j)))
        arms = {}
        for name, (qe, re) in masks.items():
            fitted = full_fit if name == 'fullpool' else scalar_fit(
                query_boxes, reference_boxes, query_features, reference_features, qe, re,
                canonical_similarity)
            arms[name] = dict(fit=fitted, metrics=scalar_metrics(
                query_boxes[i], reference_boxes[j], fitted))
        output.append(dict(heldout=held, arms=arms))
    return output


def canonical_matrix(query_features, reference_features, dimension=None):
    """Protocol's full-matrix numerical definition, independent of subject code.

    BLAS dot-product roundoff can change exact ties. Scalar scans operate on
    this one full matrix; independent scalar dots are only a numeric comparator.
    """
    import numpy as np
    dimensions = {len(row) for row in query_features + reference_features}
    dimension = dimension or (next(iter(dimensions)) if dimensions else 1)
    def normalize(rows):
        array = np.asarray(rows, dtype=np.float64).reshape(-1, dimension)
        peak = np.max(np.abs(array), axis=1, keepdims=True)
        scaled = np.zeros_like(array)
        np.divide(array, peak, out=scaled, where=peak > 0)
        length = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
        normalized = np.zeros_like(array)
        np.divide(scaled, length, out=normalized, where=length > 0)
        return normalized
    return (normalize(query_features) @ normalize(reference_features).T).tolist()


def fast_frame_oracle(query, reference):
    """Independent canonical-matrix reducer; never imports subject reducers.

    Returns every fullpool h in query-row order, 37 independently recomputed
    numeric columns, and hashes of independently reconstructed locator lists.
    """
    import numpy as np
    qb, rb = np.ascontiguousarray(query['boxes'], np.float64), np.ascontiguousarray(reference['boxes'], np.float64)
    qf, rf = np.ascontiguousarray(query['raw'], np.float64), np.ascontiguousarray(reference['raw'], np.float64)
    for boxes, feats, proposals in ((qb, qf, query['proposals']), (rb, rf, reference['proposals'])):
        check(boxes.ndim == 2 and boxes.shape[1] == 4, 'Invalid xyxy shape')
        check(feats.ndim == 2 and len(feats) == len(boxes) and feats.shape[1] > 0, 'Feature shape mismatch')
        check(np.isfinite(boxes).all() and np.isfinite(feats).all(), 'Nonfinite input')
        check(np.all(boxes[:, 2:] > boxes[:, :2]) and np.isfinite(boxes[:, 2:] - boxes[:, :2]).all(), 'Invalid box dimensions')
        check(len(proposals) == len(boxes), 'Missing proposal alias locators')
    check(qf.shape[1] == rf.shape[1], 'Raw feature dimensions differ')
    def normalize(feats):
        peak = np.max(np.abs(feats), axis=1, keepdims=True)
        scaled = np.zeros_like(feats)
        np.divide(feats, peak, out=scaled, where=peak > 0)
        lengths = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
        norm = np.zeros_like(feats)
        np.divide(scaled, lengths, out=norm, where=lengths > 0)
        return norm, peak[:, 0] > 0
    qn, qvalid = normalize(qf)
    rn, rvalid = normalize(rf)
    similarity = qn @ rn.T
    qc, rc = qb[:, :2] / 2 + qb[:, 2:] / 2, rb[:, :2] / 2 + rb[:, 2:] / 2

    def match(qe, re, with_margins=False):
        qmask, rmask = qvalid.copy(), rvalid.copy()
        qmask[qe], rmask[re] = False, False
        qi, ri = np.flatnonzero(qmask), np.flatnonzero(rmask)
        if not len(qi) or not len(ri):
            return np.empty((0, 2), np.int64), np.empty((0, 3), np.float64)
        block = similarity[np.ix_(qi, ri)]
        rmax, cmax = block.max(axis=1), block.max(axis=0)
        rbest, cbest = block.argmax(axis=1), block.argmax(axis=0)
        runique = (block == rmax[:, None]).sum(axis=1) == 1
        cunique = (block == cmax[None, :]).sum(axis=0) == 1
        keep = np.flatnonzero((rmax > 0) & runique & cunique[rbest]
                             & (cbest[rbest] == np.arange(len(qi))))
        pairs = np.column_stack((qi[keep], ri[rbest[keep]]))
        if not with_margins:
            return pairs.astype(np.int64), None
        metrics = []
        for i in keep:
            j = rbest[i]
            qa, ra = np.delete(block[i], j), np.delete(block[:, j], i)
            metrics.append([block[i, j], np.nan if not len(qa) else block[i, j] - qa.max(),
                            np.nan if not len(ra) else block[i, j] - ra.max()])
        return pairs.astype(np.int64), np.asarray(metrics, np.float64).reshape(-1, 3)

    def vector_median(values):
        ordered = np.sort(values, axis=0)
        size = len(ordered)
        return (ordered[size // 2].copy() if size % 2 else
                ordered[size // 2 - 1] / 2 + ordered[size // 2] / 2)

    def fit(pairs):
        size = len(pairs)
        if size < 3:
            return None
        displacement = qc[pairs[:, 0]] - rc[pairs[:, 1]]
        translation = vector_median(displacement)
        mad = vector_median(np.abs(displacement - translation))
        # Derive deleted medians from coordinate ranks, independently of the
        # subject implementation's fit. Stable rank handles equal-valued data.
        ranks = np.argsort(np.argsort(displacement, axis=0, kind='stable'), axis=0, kind='stable')
        ordered = np.sort(displacement, axis=0)
        k = size // 2
        if size % 2:
            # Below/at/above the middle removed item changes retained centres.
            leave = np.where(ranks < k, ordered[k] / 2 + ordered[k + 1] / 2,
                    np.where(ranks == k, ordered[k - 1] / 2 + ordered[k + 1] / 2,
                             ordered[k - 1] / 2 + ordered[k] / 2))
        else:
            leave = np.where(ranks < k, ordered[k], ordered[k - 1])
        shifts = leave - translation
        largest = float(np.max(np.hypot(shifts[:, 0], shifts[:, 1])))
        check(np.isfinite(displacement).all() and np.isfinite(translation).all()
              and np.isfinite(mad).all() and math.isfinite(largest), 'Geometry overflow')
        return translation, mad, largest

    full_pairs, full_metrics = match([], [], with_margins=True)
    n = len(full_pairs)
    values = np.full((n, 37), np.nan)
    anchor_hashes = np.empty((n, 3, 32), np.uint8)
    exclusion_hashes = np.empty((n, 3, 32), np.uint8)
    full_fit = fit(full_pairs)
    qproposal, rproposal = np.asarray(query['proposals']), np.asarray(reference['proposals'])
    for h, (i, j) in enumerate(full_pairs):
        qbox, rbox = qb[i], rb[j]
        scale = np.sqrt(qbox[2:] - qbox[:2]) * np.sqrt(rbox[2:] - rbox[:2])
        displacement = qc[i] - rc[j]
        static = float(np.sum(np.abs(displacement / scale)))
        values[h, :3], values[h, 3] = full_metrics[h], static
        qoverlap = (np.minimum(qb[:, 2], qbox[2]) > np.maximum(qb[:, 0], qbox[0])) & (np.minimum(qb[:, 3], qbox[3]) > np.maximum(qb[:, 1], qbox[1]))
        roverlap = (np.minimum(rb[:, 2], rbox[2]) > np.maximum(rb[:, 0], rbox[0])) & (np.minimum(rb[:, 3], rbox[3]) > np.maximum(rb[:, 1], rbox[1]))
        masks = [([], []), ([int(i)], [int(j)]),
                 (np.flatnonzero(qoverlap | (qproposal == qproposal[i])).tolist(),
                  np.flatnonzero(roverlap | (rproposal == rproposal[j])).tolist())]
        for arm, (qe, re) in enumerate(masks):
            pairs = full_pairs if arm == 0 else match(qe, re)[0]
            fitted = full_fit if arm == 0 else fit(pairs)
            start = 4 + arm * 11
            values[h, start:start + 2] = [float(fitted is not None), len(pairs)]
            if fitted is not None:
                translation, mad, largest = fitted
                innovation = (displacement - translation) / scale
                corrected = float(np.sum(np.abs(innovation)))
                values[h, start + 2:start + 11] = list(translation) + list(mad) + [largest] + list(innovation) + [corrected, static - corrected]
                check(np.isfinite(values[h, start:start + 11]).all(), 'Nonfinite available fit encoding')
            anchor_hashes[h, arm] = np.frombuffer(list_hash(pairs.tolist()), np.uint8)
            exclusion_hashes[h, arm] = np.frombuffer(list_hash(dict(query=qe, reference=re)), np.uint8)
    return values, full_pairs, anchor_hashes, exclusion_hashes, similarity


def compare_arrays(actual, expected, message, tolerance=2e-12):
    import numpy as np
    check(actual.shape == expected.shape, message + ': shape differs')
    check(np.array_equal(np.isnan(actual), np.isnan(expected)), message + ': missingness differs')
    check(not np.isinf(actual).any(), message + ': infinite values')
    finite = np.isfinite(expected)
    check(np.allclose(actual[finite], expected[finite], rtol=tolerance, atol=tolerance),
          message + ': finite values differ')
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.


def project_video_records(text):
    """Decode only the named top-level plan projection; skip other JSON values."""
    decoder = json.JSONDecoder()
    def space(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position
    def skip(position):
        position = space(position)
        stack, quoted, escaped = [], False, False
        for cursor in range(position, len(text)):
            char = text[cursor]
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
                    if not stack:
                        return cursor + 1
            elif char == '"':
                quoted = True
            elif char in '[{':
                stack.append(char)
            elif char in ']}':
                if not stack:
                    return cursor
                check((stack.pop(), char) in (('[', ']'), ('{', '}')), 'Malformed skipped plan JSON')
                if not stack:
                    return cursor + 1
            elif char == ',' and not stack:
                return cursor
        raise ValueError('Unterminated skipped plan JSON')
    position = space(0)
    check(text[position] == '{', 'Plan must be a top-level object')
    position += 1
    result = None
    while True:
        position = space(position)
        if text[position] == '}':
            break
        key, position = decoder.raw_decode(text, position)
        check(isinstance(key, str), 'Plan key must be text')
        position = space(position)
        check(text[position] == ':', 'Malformed plan key separator')
        position = space(position + 1)
        if key == 'video_records':
            check(result is None, 'Duplicate video_records key')
            result, position = decoder.raw_decode(text, position)
        else:
            position = skip(position)
        position = space(position)
        if text[position] == '}':
            break
        check(text[position] == ',', 'Malformed top-level plan separator')
        position += 1
    check(isinstance(result, list), 'Missing allowed video_records projection')
    return result


def independent_metadata(spec):
    roles = json.loads(Path(spec['roles_path']).read_text())
    roles = roles.get('groups', roles)
    sources = sorted(source for source, row in roles.items() if row['split'] == 'fit')
    check(len(sources) == EXPECTED_POPULATION['sources'] and all(roles[source]['role'] in ('mean_role', 'pair_role')
                                    for source in sources), 'FIT source population changed')
    projected = project_video_records(Path(spec['plan_path']).read_text())
    video_sources = {int(row['video_id']): row['source_group'] for row in projected
                     if row['split'] == 'fit' and row['source_group'] in sources}
    inventory = json.loads(Path(spec['inventory_path']).read_text())
    videos = defaultdict(list)
    with Path(spec['frame_execution_path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if int(row['video_id']) not in video_sources:
                continue
            check(row['npz_sha256'] == inventory[Path(row['npz']).name], 'Raw inventory mismatch')
            videos[int(row['video_id'])].append({key: row[key] for key in (
                'video_id', 'image_id', 'frame_id', 'frame_index', 'observations', 'npz', 'npz_sha256')})
    for video, rows in videos.items():
        rows.sort(key=lambda row: row['frame_id'])
        check([row['frame_id'] for row in rows] == list(range(len(rows))), 'Missing/repeated sampled frame')
        check(len(rows) <= 41 and all(0 <= row['observations'] <= 77 for row in rows), 'Frozen bounds changed')
        check(all(left['frame_index'] < right['frame_index'] for left, right in zip(rows, rows[1:])), 'Original timeline changed')
    check(len(videos) == EXPECTED_POPULATION['videos'] and sum(len(rows) for rows in videos.values()) == EXPECTED_POPULATION['frames'], 'FIT frame coverage changed')
    check(sum(row['observations'] for rows in videos.values() for row in rows) == EXPECTED_POPULATION['rows'], 'Retained-row count changed')
    check({video_sources[video] for video in videos} == set(sources), 'FIT source coverage changed')
    return sources, video_sources, dict(videos)


def read_allowed_frame(record):
    import numpy as np
    check(file_sha(record['npz']) == record['npz_sha256'], 'Raw cache SHA changed')
    with np.load(record['npz'], allow_pickle=False) as saved:
        provenance = {key: int(saved[key]) for key in ('video_id', 'image_id', 'frame_id', 'frame_index')}
        boxes = np.ascontiguousarray(saved['det_bboxes'], dtype=np.float64)
        raw = np.ascontiguousarray(saved['track_feats'], dtype=np.float64)
        proposals = saved['roi_proposal_row'].copy()
    check(all(provenance[key] == record[key] for key in provenance), 'Raw provenance mismatch')
    n = record['observations']
    check(boxes.shape == (n, 5) and raw.shape == (n, 256), 'Frozen raw shapes changed')
    check(proposals.shape == (n,) and proposals.dtype.kind in 'iu' and (proposals >= 0).all(), 'Invalid proposal row locators')
    return dict(record=record, boxes=boxes[:, :4], raw=raw, proposals=proposals)


def compare_json(actual, expected, path='output'):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), path + ': keys differ')
        for key in expected:
            compare_json(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), path + ': length differs')
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare_json(left, right, path + '[%d]' % i)
    elif isinstance(expected, float):
        check(isinstance(actual, (int, float)) and not isinstance(actual, bool)
              and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), path + ': numeric value differs')
    else:
        check(type(actual) is type(expected) and actual == expected, path + ': exact value differs')


def independent_summary(sources, frame_blocks):
    """Rebuild each readout without importing the producer's summarizer."""
    import numpy as np
    selectors = dict((arm, (i,)) for i, arm in enumerate(ARMS))
    selectors.update(common_row_strict=(1, 2), common_all_three=(0, 1, 2))
    rows = {name: {lag: {source: [] for source in sources} for lag in (1, 4)} for name in selectors}
    target_counts = {lag: dict.fromkeys(sources, 0) for lag in (1, 4)}
    frame_counts = {1: 0, 4: 0}
    for source, lag, block in frame_blocks:
        target_counts[lag][source] += len(block)
        frame_counts[lag] += 1
        for name, arms in selectors.items():
            common = [i for i in range(len(block)) if all(block[i, 4 + arm * 11] == 1 for arm in arms)]
            if common:
                rows[name][lag][source].append(dict(count=len(common),
                    gains={arm: median([float(block[i, 14 + arm * 11]) for i in common]) for arm in arms}))
    output = dict(status='GTFREE_PREDICTABILITY_ONLY_NOT_IDENTITY_OR_TRACKING_EVALUATION',
                  sources=sources, frame_receipts_by_lag={str(key): value for key, value in frame_counts.items()}, readouts={})
    for name, lag_rows in rows.items():
        arms, results = selectors[name], {}
        for lag in (1, 4):
            details = {}
            for source in sources:
                frames = lag_rows[lag][source]
                details[source] = dict(fullpool_holdouts=target_counts[lag][source],
                    supported_frame_pairs=len(frames), valid_holdouts=sum(frame['count'] for frame in frames),
                    source_mean_frame_median_gain={ARMS[arm]: (sum(frame['gains'][arm] for frame in frames) / len(frames) if frames else None) for arm in arms})
            supported = [source for source in sources if details[source]['supported_frame_pairs']]
            results[str(lag)] = dict(supported_sources=len(supported),
                source_macro_gain={ARMS[arm]: (sum(details[source]['source_mean_frame_median_gain'][ARMS[arm]] for source in supported) / len(supported) if supported else None) for arm in arms},
                by_source=details)
        common = [source for source in sources if all(results[str(lag)]['by_source'][source]['supported_frame_pairs'] for lag in (1, 4))]
        results['common_supported_sources_across_lags'] = dict(sources=common,
            source_macro_gain_by_lag={str(lag): {ARMS[arm]: (sum(results[str(lag)]['by_source'][source]['source_mean_frame_median_gain'][ARMS[arm]] for source in common) / len(common) if common else None) for arm in arms} for lag in (1, 4)})
        output['readouts'][name] = results
    main = output['readouts'][ARMS[2]]
    passed = all(main[str(lag)]['source_macro_gain'][ARMS[2]] is not None
                 and main[str(lag)]['source_macro_gain'][ARMS[2]] > 0 for lag in (1, 4))
    output['minimal_gate'] = ('PASS_MINIMAL_GTFREE_DONOR_PREDICTABILITY' if passed
                              else 'NOT_PASSED_MINIMAL_GTFREE_DONOR_PREDICTABILITY')
    output['method_novelty_or_identity_effectiveness_established'] = False
    return output


def verify_selected_scalar(query, reference, values, locations, ah, eh, similarity, selector):
    """Fixed hash-selected raw frame subset: scalar math and exact membership."""
    import numpy as np
    qb, rb = query['boxes'].tolist(), reference['boxes'].tolist()
    qf, rf = query['raw'].tolist(), reference['raw'].tolist()
    qp, rp = query['proposals'].tolist(), reference['proposals'].tolist()
    sim = similarity.tolist()
    full = scalar_fit(qb, rb, qf, rf, canonical_similarity=sim)
    pairs = [(anchor['query_row'], anchor['reference_row']) for anchor in full['anchors']]
    check(pairs == [tuple(row) for row in locations.tolist()], 'Selected scalar fullpool membership differs')
    if not len(locations):
        return 0.
    h = selector % len(locations)
    i, j = map(int, locations[h])
    masks = [([], []), ([i], [j]), (strict_exclusions(qb, qp, i), strict_exclusions(rb, rp, j))]
    for arm, (qe, re) in enumerate(masks):
        fit = full if arm == 0 else scalar_fit(qb, rb, qf, rf, qe, re, sim)
        metric = scalar_metrics(qb[i], rb[j], fit)
        encoded = [float(fit['available']), float(fit['anchor_count'])] + [math.nan] * 9
        if fit['available']:
            encoded[2:] = fit['translation'] + fit['residual_mad_xy'] + [fit['max_leave_one_anchor_shift']] + metric['innovation'] + [metric['corrected_l1'], metric['gain']]
        compare_arrays(values[h, 4 + arm * 11:4 + (arm + 1) * 11], np.asarray(encoded), 'Fixed subset scalar fit')
        check(list_hash([[a['query_row'], a['reference_row']] for a in fit['anchors']]) == ah[h, arm].tobytes(), 'Selected scalar anchor hash differs')
        check(list_hash(dict(query=qe, reference=re)) == eh[h, arm].tobytes(), 'Selected scalar exclusion hash differs')
    # Scalar dot is a numeric comparison only; canonical ties remain exact.
    qn, rn = normalize_scalar(qf[i]), normalize_scalar(rf[j])
    scalar_cosine = sum(x * y for x, y in zip(qn, rn))
    check(math.isclose(scalar_cosine, float(similarity[i, j]), rel_tol=2e-12, abs_tol=2e-12), 'Scalar cosine arithmetic differs')
    return abs(scalar_cosine - float(similarity[i, j]))


def authenticate_audit_protocol(protocol_path):
    spec = json.loads(Path(protocol_path).read_text())
    check(spec['status'] == 'FROZEN_GTFREE_SCENE_TRANSLATION_DONOROUT_V1', 'Protocol is not frozen')
    check(spec['lags'] == [1, 4] and spec['minimum_anchors'] == 3, 'Frozen lag/minimum changed')
    check(spec['arms'] == list(ARMS) and spec['columns'] == list(COLUMNS), 'Frozen column schema changed')
    for path, expected in spec['input_pins'].items():
        check(file_sha(path) == expected, 'Pinned input changed: ' + path)
    check(file_sha(spec['producer_path']) == spec['producer_sha256'], 'Producer changed after freeze')
    authority = json.loads(Path(spec['protected_authority_path']).read_text())
    check(len(authority['protected_files']) == 7, 'Protected authority scope changed')
    for path, record in authority['protected_files'].items():
        exists = Path(path).exists()
        check(exists == record['exists'] and (file_sha(path) if exists else None) == record['sha256'], 'Protected artifact changed: ' + path)
    return spec


def verify_outputs(protocol_path):
    """Full raw-to-receipt audit, importing neither producer nor cached code."""
    import numpy as np
    started = time.monotonic()
    spec = authenticate_audit_protocol(protocol_path)
    output = Path(spec['output_directory'])
    audit_path = output / 'independent_output_audit.json'
    check(not audit_path.exists(), 'Independent receipt already exists; never overwrite')
    seal_path, summary_path = output / 'GT_free_predictions_sealed.json', output / 'summary.json'
    seal = json.loads(seal_path.read_text())
    check(seal['status'] == 'SEALED_COMPLETE_GTFREE_DONOR_PREDICTIONS', 'Predictions are not sealed complete')
    check(seal['protocol_sha256'] == file_sha(protocol_path), 'Seal protocol SHA differs')
    compare_json(seal['input_pins'], spec['input_pins'], 'seal.input_pins')
    sources, video_sources, videos = independent_metadata(spec)
    manifest_path = output / 'metadata_manifest.json'
    check(file_sha(manifest_path) == seal['metadata_manifest_sha256'], 'Metadata manifest SHA differs')
    manifest = json.loads(manifest_path.read_text())
    compare_json(manifest, dict(sources=sources,
        video_sources={str(k): v for k, v in video_sources.items()},
        videos={str(k): v for k, v in videos.items()}, input_pins=spec['input_pins'],
        allowed_NPZ_members=list(LOADER_MEMBERS), identity_or_category_values_decoded=False), 'metadata_manifest')
    check(set(seal['video_outputs']) == {str(video) for video in videos}, 'Sealed video coverage differs')
    frame_blocks, candidates, prefix, empty = [], {1: 0, 4: 0}, {1: 0, 4: 0}, {1: 0, 4: 0}
    targets, frames_checked, scalar_frames, scalar_dot_maxabs, maxabs = 0, 0, 0, 0., 0.
    for number, (video, records) in enumerate(sorted(videos.items())):
        sealed = seal['video_outputs'][str(video)]
        check(sealed['npz'] == 'videos/v%d.npz' % video and sealed['receipt'] == 'videos/v%d.json' % video, 'Unexpected sealed output paths')
        npz_path, receipt_path = output / sealed['npz'], output / sealed['receipt']
        check(file_sha(npz_path) == sealed['npz_sha256'] and file_sha(receipt_path) == sealed['receipt_sha256'], 'Video output SHA changed')
        receipt = json.loads(receipt_path.read_text())
        check(receipt['video_id'] == video and receipt['source'] == video_sources[video], 'Video/source locator mismatch')
        check(receipt['npz'] == sealed['npz'] and receipt['npz_sha256'] == sealed['npz_sha256'], 'Receipt NPZ binding differs')
        check(receipt['columns'] == list(COLUMNS) and receipt['arms'] == list(ARMS), 'Receipt schema differs')
        check(len(receipt['frame_pairs']) == 2 * len(records), 'Missing prefix/empty frame receipt')
        with np.load(npz_path, allow_pickle=False) as saved:
            check(set(saved.files) == {'values', 'rowloc', 'anchor_hashes', 'exclusion_hashes'}, 'Unexpected prediction NPZ arrays')
            values, rowloc = saved['values'], saved['rowloc']
            ah, eh = saved['anchor_hashes'], saved['exclusion_hashes']
        check(values.dtype == np.dtype('float64') and values.shape == (receipt['holdouts'], 37), 'Values dtype/shape differs')
        check(rowloc.dtype == np.dtype('int64') and rowloc.shape == (len(values), 3), 'Row locator dtype/shape differs')
        check(ah.dtype == np.dtype('uint8') and eh.dtype == np.dtype('uint8') and ah.shape == eh.shape == (len(values), 3, 32), 'List hash dtype/shape differs')
        offset, ring, receipt_index = 0, deque(maxlen=5), 0
        for record in records:
            query = read_allowed_frame(record)
            ring.append(query)
            for lag in (1, 4):
                reference = ring[-lag - 1] if record['frame_id'] >= lag else None
                item = receipt['frame_pairs'][receipt_index]
                base = dict(lag=lag, query_step=record['frame_id'], query_image_id=record['image_id'],
                    query_original_frame=record['frame_index'], query_rows=len(query['boxes']),
                    reference_step=None if reference is None else reference['record']['frame_id'],
                    reference_image_id=None if reference is None else reference['record']['image_id'],
                    reference_original_frame=None if reference is None else reference['record']['frame_index'],
                    reference_rows=0 if reference is None else len(reference['boxes']), offset=offset,
                    holdouts=0, reason='missing_reference_prefix' if reference is None else 'reference_available')
                candidates[lag] += base['query_rows'] * base['reference_rows']
                if reference is None:
                    prefix[lag] += 1
                    block = np.empty((0, 37), np.float64)
                else:
                    expected, locations, expected_ah, expected_eh, similarity = fast_frame_oracle(query, reference)
                    base['holdouts'] = len(expected)
                    end = offset + len(expected)
                    block = values[offset:end]
                    check(end <= len(values), 'Frame block exceeds complete output')
                    maxabs = max(maxabs, compare_arrays(block, expected, 'video %d receipt %d values' % (video, receipt_index)))
                    expected_loc = np.column_stack((np.full(len(expected), receipt_index, np.int64), locations))
                    check(np.array_equal(rowloc[offset:end], expected_loc), 'Fullpool h membership or frame-offset changed')
                    check(np.array_equal(ah[offset:end], expected_ah), 'Rematched anchor membership/hash differs')
                    check(np.array_equal(eh[offset:end], expected_eh), 'Endpoint/proposal/overlap exclusion hash differs')
                    for arm in range(3):
                        available, counts = block[:, 4 + arm * 11], block[:, 5 + arm * 11]
                        check(np.isin(available, [0., 1.]).all() and (counts >= 0).all() and (counts == np.floor(counts)).all(), 'Invalid availability/count encoding')
                        check(np.array_equal(available == 1, counts >= 3), 'Unavailable reason is not exactly count<3')
                        check(np.isnan(block[available == 0, 6 + arm * 11:15 + arm * 11]).all(), 'Unsupported fit contains fabricated numeric fallback')
                    selector = int.from_bytes(hashlib.sha256(('%d:%d:%d' % (video, record['frame_id'], lag)).encode()).digest()[:8], 'big')
                    if selector % 257 == 0:
                        scalar_dot_maxabs = max(scalar_dot_maxabs, verify_selected_scalar(query, reference, expected, locations, expected_ah, expected_eh, similarity, selector))
                        scalar_frames += 1
                    if not len(query['boxes']) or not len(reference['boxes']):
                        empty[lag] += 1
                    offset = end
                compare_json(item, base, 'video %d framepair receipt %d' % (video, receipt_index))
                frame_blocks.append((video_sources[video], lag, block.copy()))
                receipt_index += 1
                frames_checked += 1
        check(offset == len(values) == receipt['holdouts'], 'Trailing/omitted heldout records')
        targets += len(values)
        if number % 20 == 0 or number == len(videos) - 1:
            print(json.dumps(dict(audited_videos=number + 1, total_videos=len(videos),
                frame_receipts=frames_checked, heldouts=targets, elapsed_seconds=time.monotonic() - started)), flush=True)
    check(candidates == EXPECTED_POPULATION['candidates'], 'Full exact-lag candidate coverage differs')
    compare_json(seal['candidate_pairs_by_lag'], {str(k): v for k, v in candidates.items()}, 'seal candidate counts')
    check(targets == seal['total_holdouts'], 'All-h population differs from seal')
    check(prefix == EXPECTED_POPULATION['prefix'] and frames_checked == EXPECTED_POPULATION['frame_receipts'], 'Prefix/full-frame receipt coverage differs')
    summary = json.loads(summary_path.read_text())
    compare_json(summary, independent_summary(sources, frame_blocks), 'source-balanced summary')
    producer_receipt = json.loads((output / 'producer_receipt.json').read_text())
    check(producer_receipt['status'] == 'COMPLETE_GTFREE_DONOR_PREDICTION_AND_DESCRIPTIVE_READOUT'
          and producer_receipt['seal_sha256'] == file_sha(seal_path)
          and producer_receipt['summary_sha256'] == file_sha(summary_path), 'Producer completion binding differs')
    authenticate_audit_protocol(protocol_path)
    result = dict(status='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_OUTPUT_AUDIT',
        protocol_sha256=file_sha(protocol_path), seal_sha256=file_sha(seal_path), summary_sha256=file_sha(summary_path),
        verifier_sha256=file_sha(__file__), full_raw_reconstruction=True,
        exact_fullpool_h_and_rematched_anchor_exclusion_hashes=True,
        all_37_columns_and_hierarchical_common_support_summaries_verified=True,
        sources=len(sources), videos=len(videos), frame_pair_receipts=frames_checked,
        prefix_by_lag={str(k): v for k, v in prefix.items()}, empty_nonprefix_by_lag={str(k): v for k, v in empty.items()},
        total_holdouts=targets, max_abs_numeric_difference=maxabs,
        fixed_hash_selected_scalar_frames=scalar_frames, scalar_dot_max_abs_difference=scalar_dot_maxabs,
        raw_member_allowlist=list(LOADER_MEMBERS), identity_or_OOF_evaluation_opened=False,
        physical_camera_motion_or_identity_or_novelty_established=False,
        elapsed_seconds=time.monotonic() - started)
    with audit_path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def aggregate_scalar(frame_records, source_universe, required_arms=('strict',)):
    """Common h first, then frame median, source frame mean, source mean."""
    source_frames = {source: [] for source in source_universe}
    selected = []
    for record in frame_records:
        if record['source'] not in source_frames:
            raise ValueError('Frame outside frozen source universe')
        held = [row for row in record['heldouts'] if all(
            row['arms'][arm]['fit']['available'] for arm in required_arms)]
        medians = {arm: (median([row['arms'][arm]['metrics']['gain'] for row in held])
                         if held else None) for arm in required_arms}
        selected.append(dict(locator=record['locator'], common_h=len(held), medians=medians))
        if held:
            source_frames[record['source']].append(medians)
    sources = {}
    for source, frames in source_frames.items():
        sources[source] = dict(valid_frames=len(frames), means={
            arm: (sum(frame[arm] for frame in frames) / len(frames) if frames else None)
            for arm in required_arms})
    supported = [source for source in source_universe if sources[source]['valid_frames']]
    macro = {arm: (sum(sources[source]['means'][arm] for source in supported) / len(supported)
                   if supported else None) for arm in required_arms}
    return dict(frames=selected, sources=sources, supported_sources=supported, macro=macro)


def box(x, y, width=2., height=2.):
    return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]


def identity(size):
    return [[float(i == j) for j in range(size)] for i in range(size)]


def translated_scene(size=5, dx=7., dy=-3.):
    reference = [box(i * 20., i * 10.) for i in range(size)]
    query = [[b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy] for b in reference]
    return query, reference, identity(size), identity(size)


def synthetic_selftest():
    checks = []
    qb, rb, qf, rf = translated_scene()
    records = scalar_frame(qb, rb, qf, rf, list(range(5)), list(range(5)))
    check(len(records) == 5, 'Must retain all fullpool heldouts')
    for held in records:
        for arm in ('fullpool', 'row', 'strict'):
            result = held['arms'][arm]
            check(result['fit']['translation'] == [7., -3.], 'Known translation changed')
            check(result['metrics']['corrected_l1'] == 0., 'Heldout prediction wrong')
            check(math.isclose(result['metrics']['gain'], 5., abs_tol=1e-14),
                  'Wrong normalization or L1 gain')
    checks.append('known_translation_all_fullpool_donors_and_three_arms')

    qb, rb, qf, rf = translated_scene(3)
    records = scalar_frame(qb, rb, qf, rf, list(range(3)), list(range(3)))
    check(len(records) == 3, 'Missing heldouts silently filtered')
    check(all(row['arms']['fullpool']['fit']['available'] for row in records), 'Control needs min3')
    check(all(not row['arms']['row']['fit']['available']
              and row['arms']['row']['metrics']['gain'] is None for row in records),
          'Insufficient support became identity or zero gain')
    checks.append('all_h_including_missing_and_in_sample_control_distinction')

    qb, rb, qf, rf = translated_scene(4)
    qf[1] = [.8, .6, 0., 0.]
    full = scalar_mnn(qb, rb, qf, rf)
    after = [(a['query_row'], a['reference_row']) for a in full
             if a['query_row'] != 0 and a['reference_row'] != 0]
    fit = scalar_fit(qb, rb, qf, rf, [0], [0])
    check(len(after) == 2 and fit['available'] and fit['anchor_count'] == 3,
          'Re-matching after deletion must recover a new reciprocal pair')
    check((1, 1) in [(a['query_row'], a['reference_row']) for a in fit['anchors']],
          'Recovered competing match missing')
    checks.append('match_then_delete_is_not_donorout_rematching')

    qb, rb, qf, rf = translated_scene(4)
    rb[1] = box(.5, .5)
    qb[1] = [rb[1][0] + 7, rb[1][1] - 3, rb[1][2] + 7, rb[1][3] - 3]
    record = scalar_frame(qb, rb, qf, rf, list(range(4)), list(range(4)))[0]
    check(record['arms']['row']['fit']['available'], 'Row control unexpectedly unavailable')
    check(not record['arms']['strict']['fit']['available'], 'Overlap aliases not removed')
    check(strict_exclusions([box(0, 0), box(2, 0), box(1.9, 0)], [0, 1, 2], 0) == [0, 2],
          'Boundary touching has no positive area')
    check(strict_exclusions([box(0, 0), box(20, 0)], [8, 8], 0) == [0, 1],
          'Same proposal must be removed without overlap')
    checks.append('strict_overlap_proposal_aliases_and_zeroarea_boundary_contact')

    qe, re = record['arms']['strict']['fit']['excluded_query_rows'], record['arms']['strict']['fit']['excluded_reference_rows']
    before = scalar_fit(qb, rb, qf, rf, qe, re)
    qmut, rmut, qfm, rfm = copy.deepcopy(qb), copy.deepcopy(rb), copy.deepcopy(qf), copy.deepcopy(rf)
    qmut[0], rmut[0], qfm[0], rfm[0] = box(1000, 900), box(-800, -600), [0., 1., 0., 0.], [0., 0., 1., 0.]
    check(before == scalar_fit(qmut, rmut, qfm, rfm, qe, re), 'Fixedmask endpoint mutation changed fit')
    check(strict_exclusions(qmut, list(range(4)), 0) != qe,
          'Fixture must show recomputed overlap mask can change')
    checks.append('fixedmask_only_endpoint_mutation_invariance')

    qb, rb, qf, rf = translated_scene(5)
    qf[0] = [0.] * 5
    qf[1], rf[1] = rf[2][:], rf[2][:]
    anchors = scalar_mnn(qb, rb, qf, rf)
    check(all(a['query_row'] not in (0, 1, 2) for a in anchors), 'Zero or exact ties matched')
    check(scalar_mnn([box(0, 0)], [box(0, 0)], [[1., 0.]], [[0., 1.]]) == [], 'Zero cosine is not positive')
    check(scalar_mnn([box(0, 0)], [box(0, 0)], [[1., 0.]], [[-1., 0.]]) == [], 'Negative cosine accepted')
    single = scalar_mnn([box(0, 0)], [box(0, 0)], [[1., 0.]], [[1., 0.]])[0]
    check(single['query_margin'] is None and single['reference_margin'] is None, 'No competitor implies missing margin')
    check(scalar_mnn([], [], [], []) == [], 'Empty frame must remain empty')
    checks.append('zero_features_strict_ties_nonpositive_cosine_missingmargin_empty_frames')

    qb, rb, qf, rf = translated_scene(4)
    shifts = [[0., 0.], [2., 4.], [6., 8.], [20., 30.]]
    qb = [[b[0] + d[0], b[1] + d[1], b[2] + d[0], b[3] + d[1]] for b, d in zip(rb, shifts)]
    fit = scalar_fit(qb, rb, qf, rf)
    check(fit['translation'] == [4., 6.] and fit['residual_mad_xy'] == [3., 4.], 'Even coordinate median/MAD wrong')
    metric = scalar_metrics(box(10, 20, 4, 9), box(0, 0, 1, 4), dict(available=True, translation=[2., 8.]))
    check(metric['innovation'] == [4., 2.] and metric['static_l1'] == 25. / 3., 'Endpoint geometric scale wrong')
    checks.append('even_coordinate_medians_MAD_and_endpoint_geometricmean_scale')

    def held(gains):
        return dict(arms={name: dict(fit=dict(available=value is not None), metrics=dict(gain=value))
                          for name, value in gains.items()})
    frames = [dict(source='a', locator='a/video1/frame1', heldouts=[held(dict(strict=x, row=x, fullpool=x)) for x in (0., 100., 100.)]),
              dict(source='a', locator='a/video2/frame1', heldouts=[held(dict(strict=0., row=0., fullpool=0.))]),
              dict(source='a', locator='a/video2/frame2', heldouts=[held(dict(strict=None, row=10., fullpool=10.))]),
              dict(source='b', locator='b/frame1', heldouts=[held(dict(strict=-10., row=-10., fullpool=-10.))])]
    agg = aggregate_scalar(frames, ['a', 'b', 'c'])
    check(agg['sources']['a']['means']['strict'] == 50. and agg['macro']['strict'] == 20., 'Aggregation must be median then frame/source means')
    check(agg['sources']['c']['means']['strict'] is None and agg['sources']['a']['valid_frames'] == 2, 'Missing source/frame padded with zero')
    common = aggregate_scalar(frames, ['a', 'b', 'c'], ('row', 'strict'))
    check(common['frames'][2]['common_h'] == 0 and common['macro'] == dict(row=20., strict=20.), 'Common h must precede aggregation')
    check(aggregate_scalar([], ['a'])['macro']['strict'] is None, 'Empty support must remain missing')
    checks.append('same_h_intersection_frame_median_source_frame_mean_source_equal_macro_nozero_padding')
    return dict(status='PASS_SCALAR_DONOROUT_SYNTHETIC_FIXTURES', checks=checks,
                fixture_groups=len(checks), real_cache_opened=False,
                producer_schema_assumed=False, identity_or_OOF_evaluation_opened=False,
                independent_math_uses='Python scalar arithmetic and full enumeration',
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def producer_synthetic_check():
    """Import producer only for toys; the real-output auditor never imports it."""
    import numpy as np
    producer_path = ROOT / 'tmp_analysis/scene_translation_donorout_20261024.py'
    check(file_sha(CACHED_MODULE) == CACHED_SHA256, 'Cached module pin changed')
    modules = []
    for path, name in ((producer_path, 'toy_only_donorout_producer'),
                       (CACHED_MODULE, 'toy_only_donorout_cached')):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    producer, cached = modules
    check(tuple(producer.COLUMNS) == COLUMNS and tuple(producer.ARMS) == ARMS, 'Toy producer schema changed')
    def frame(boxes, feats, proposals=None, dimension=None):
        dimensions = {len(row) for row in feats}
        dimension = dimension or (next(iter(dimensions)) if dimensions else 1)
        return dict(boxes=np.asarray(boxes, np.float64).reshape(-1, 4),
                    raw=np.asarray(feats, np.float64).reshape(-1, dimension),
                    proposals=np.asarray(list(range(len(boxes))) if proposals is None else proposals, np.int64))
    rng = random.Random(20261024)
    cases = []
    for count in range(1, 10):
        qb, rb, qf, rf = translated_scene(count)
        cases.append((frame(qb, qf), frame(rb, rf)))
        if count > 3:
            qf[1] = qf[0][:]
            rf[2] = rf[1][:]
            cases.append((frame(qb, qf), frame(rb, rf)))
    qb, rb, qf, rf = translated_scene(5)
    cases.append((frame([], [], dimension=5), frame(rb, rf)))
    cases.append((frame(qb, qf), frame([], [], dimension=5)))
    cases.append((frame([], [], dimension=5), frame([], [], dimension=5)))
    for repetition in range(16):
        nq, nr, dim = rng.randrange(2, 11), rng.randrange(2, 11), 8
        rf = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(nr)]
        qf = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(nq)]
        for i in range(min(nq, nr)):
            qf[i] = [value + rng.gauss(0, .01) for value in rf[i]]
        qb = [box(rng.randrange(20), rng.randrange(20), 4, 6) for _ in range(nq)]
        rb = [box(rng.randrange(20), rng.randrange(20), 4, 6) for _ in range(nr)]
        if repetition % 3 == 0:
            qf[0] = [0.] * dim
        cases.append((frame(qb, qf, [i // 2 for i in range(nq)]),
                      frame(rb, rf, [i // 2 for i in range(nr)])))
    # Canonical C-order math must be independent of caller storage layout.
    baseq, baser = cases[-1]
    for order in ('F', 'strided'):
        layouts = []
        for original in (baseq, baser):
            changed = dict(original)
            if order == 'F':
                changed['raw'] = np.asfortranarray(original['raw'])
                changed['boxes'] = np.asfortranarray(original['boxes'])
            else:
                backing = np.empty((len(original['raw']), original['raw'].shape[1] * 2))
                backing[:, ::2] = original['raw']
                changed['raw'] = backing[:, ::2]
                backing_box = np.empty((len(original['boxes']), 8))
                backing_box[:, ::2] = original['boxes']
                changed['boxes'] = backing_box[:, ::2]
            layouts.append(changed)
        cases.append(tuple(layouts))
        plain, altered = fast_frame_oracle(baseq, baser), fast_frame_oracle(*layouts)
        check(all(np.array_equal(left, right, equal_nan=True) for left, right in zip(plain, altered)), 'Caller F/strided layout changed canonical outputs')
    maxabs, targets = 0., 0
    for query, reference in cases:
        expected, locations, ah, eh, sim = fast_frame_oracle(query, reference)
        actual = producer.predict_frame_pair(query, reference, cached.CachedSceneMotion)
        maxabs = max(maxabs, compare_arrays(actual[0], expected, 'Toy 37-column encoding'))
        check(np.array_equal(actual[1], locations), 'Toy fullpool h membership/order changed')
        check(np.array_equal(actual[2], ah) and np.array_equal(actual[3], eh), 'Toy anchor or exclusion list hash changed')
        scalar = scalar_frame(query['boxes'].tolist(), reference['boxes'].tolist(),
            query['raw'].tolist(), reference['raw'].tolist(), query['proposals'].tolist(),
            reference['proposals'].tolist(), sim.tolist())
        check([(row['heldout']['query_row'], row['heldout']['reference_row']) for row in scalar]
              == [tuple(row) for row in locations.tolist()], 'Scalar fullpool membership differs')
        for h, row in enumerate(scalar):
            for arm, scalar_name in enumerate(('fullpool', 'row', 'strict')):
                fitted, metrics = row['arms'][scalar_name]['fit'], row['arms'][scalar_name]['metrics']
                encoded = [float(fitted['available']), float(fitted['anchor_count'])] + [math.nan] * 9
                if fitted['available']:
                    encoded[2:] = fitted['translation'] + fitted['residual_mad_xy'] + [fitted['max_leave_one_anchor_shift']] + metrics['innovation'] + [metrics['corrected_l1'], metrics['gain']]
                maxabs = max(maxabs, compare_arrays(expected[h, 4 + arm * 11:4 + (arm + 1) * 11], np.asarray(encoded), 'Scalar versus independent fast fit'))
                check(list_hash([[a['query_row'], a['reference_row']] for a in fitted['anchors']]) == ah[h, arm].tobytes(), 'Scalar donor membership differs')
        targets += len(locations)
    # Worst observed row bound, with all 77 fullpool h and nonoverlapping boxes.
    qb, rb, qf, rf = translated_scene(77)
    query, reference = frame(qb, qf), frame(rb, rf)
    times = []
    for _ in range(3):
        started = time.monotonic()
        values, locations, *_ = fast_frame_oracle(query, reference)
        times.append(time.monotonic() - started)
        check(len(values) == 77 and len(locations) == 77, 'Worst-bound toy h count changed')
    return dict(status='PASS_INDEPENDENT_DONOROUT_PRODUCER_TOYS',
                cases=len(cases), heldout_targets=targets, max_abs_numeric_difference=maxabs,
                fullpool_membership_and_two_locator_hashes_exact=True,
                scalar_canonical_submatrix_membership_exact=True,
                C_F_strided_layout_membership_and_hashes_identical=True,
                worst_77x77_all77heldouts_seconds=times,
                conservative_27582_frames_seconds=max(times) * 27582,
                producer_sha256=file_sha(producer_path), cached_sha256=CACHED_SHA256,
                verifier_sha256=file_sha(__file__), real_cache_opened=False,
                identity_or_OOF_evaluation_opened=False)


def temporary_output_cli_check():
    """Exercise the real CLI on isolated toys, with test-only population patch.

    The production --verify CLI has no population override. This temporary
    wrapper imports this file in a child and replaces only fixed workload
    constants before calling its unchanged CLI. No real cache is opened.
    """
    import numpy as np
    import shutil
    producer_path = ROOT / 'tmp_analysis/scene_translation_donorout_20261024.py'
    modules = []
    for path, name in ((producer_path, 'fixture_only_producer'), (CACHED_MODULE, 'fixture_only_cached')):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    producer, cached = modules
    def write(path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    with tempfile.TemporaryDirectory(prefix='independent_donorout_cli_') as folder:
        root = Path(folder)
        rawroot, out = root / 'raw', root / 'output'
        rawroot.mkdir(); out.mkdir(); (out / 'videos').mkdir()
        sources, video_sources = ['a', 'b', 'c'], {11: 'a', 22: 'b', 33: 'c'}
        sizes = {11: [4, 4, 4], 22: [5, 4, 0, 3, 5, 4], 33: [3]}
        videos, inventory, executions = {}, {}, []
        for video, counts in sizes.items():
            rows = []
            for step, count in enumerate(counts):
                filename = rawroot / ('v%d_f%d.npz' % (video, step))
                boxes = np.asarray([box(i * 20 + step * 3, i * 10 - step, 4, 6) + [.9]
                                    for i in range(count)], np.float64).reshape(-1, 5)
                feats = np.zeros((count, 256), np.float16)
                for i in range(count):
                    feats[i, i] = 1.
                proposals = np.arange(count, dtype=np.int64)
                if video == 22 and count > 3:
                    proposals[1] = proposals[0]
                # Forbidden object arrays raise under allow_pickle=False if
                # accidentally decoded. Both readers must ignore these keys.
                np.savez_compressed(filename, det_bboxes=boxes, track_feats=feats,
                    roi_proposal_row=proposals, video_id=np.int64(video),
                    image_id=np.int64(video * 100 + step), frame_id=np.int64(step),
                    frame_index=np.int64(15 + step * 30),
                    track_ids=np.asarray([object()], dtype=object),
                    category_logits=np.asarray([object()], dtype=object),
                    identity_gt=np.asarray([object()], dtype=object))
                record = dict(video_id=video, image_id=video * 100 + step, frame_id=step,
                    frame_index=15 + step * 30, observations=count,
                    npz=str(filename), npz_sha256=file_sha(filename))
                rows.append(record); executions.append(record); inventory[filename.name] = record['npz_sha256']
            videos[video] = rows
        role_path, plan_path, inventory_path, execution_path = [root / name for name in ('roles.json', 'plan.json', 'inventory.json', 'frame_execution.jsonl')]
        write(role_path, dict(groups={source: dict(split='fit', role='mean_role') for source in sources}))
        write(plan_path, dict(never_decode_this_value=dict(synthetic_marker='escaped ] } quote \\"'), video_records=[dict(video_id=video, source_group=source, split='fit') for video, source in video_sources.items()]))
        write(inventory_path, inventory)
        execution_path.write_text(''.join(json.dumps(row) + '\n' for row in executions))
        authority = {}
        for i in range(7):
            path = root / ('protected%d.txt' % i); path.write_text('synthetic protected pin %d\n' % i)
            authority[str(path)] = dict(exists=True, sha256=file_sha(path))
        authority_path = root / 'authority.json'; write(authority_path, dict(protected_files=authority))
        pins = {str(path): file_sha(path) for path in (role_path, plan_path, inventory_path,
                execution_path, authority_path, CACHED_MODULE, Path(__file__))}
        protocol = dict(status='FROZEN_GTFREE_SCENE_TRANSLATION_DONOROUT_V1',
            lags=[1, 4], minimum_anchors=3, arms=list(ARMS), columns=list(COLUMNS),
            input_pins=pins, producer_path=str(producer_path), producer_sha256=file_sha(producer_path),
            protected_authority_path=str(authority_path), output_directory=str(out),
            roles_path=str(role_path), plan_path=str(plan_path), inventory_path=str(inventory_path),
            frame_execution_path=str(execution_path))
        protocol_path = root / 'protocol.json'; write(protocol_path, protocol)
        write(out / 'metadata_manifest.json', dict(sources=sources,
            video_sources=video_sources, videos=videos, input_pins=pins,
            allowed_NPZ_members=list(LOADER_MEMBERS), identity_or_category_values_decoded=False))
        outputs, candidates, targets = {}, {1: 0, 4: 0}, 0
        for video, records in videos.items():
            arrays, receipts, count = producer.predict_video(records, cached.CachedSceneMotion)
            npzname, receiptname = 'videos/v%d.npz' % video, 'videos/v%d.json' % video
            np.savez_compressed(out / npzname, **arrays)
            write(out / receiptname, dict(video_id=video, source=video_sources[video],
                columns=list(COLUMNS), arms=list(ARMS), frame_pairs=receipts,
                holdouts=len(arrays['values']), npz=npzname, npz_sha256=file_sha(out / npzname)))
            outputs[video] = dict(npz=npzname, receipt=receiptname,
                npz_sha256=file_sha(out / npzname), receipt_sha256=file_sha(out / receiptname))
            targets += len(arrays['values'])
            for lag in (1, 4): candidates[lag] += count[lag]
        seal = dict(status='SEALED_COMPLETE_GTFREE_DONOR_PREDICTIONS', protocol_sha256=file_sha(protocol_path),
            candidate_pairs_by_lag=candidates, total_holdouts=targets, video_outputs=outputs,
            metadata_manifest_sha256=file_sha(out / 'metadata_manifest.json'), input_pins=pins)
        write(out / 'GT_free_predictions_sealed.json', seal)
        write(out / 'summary.json', producer.summarize(sources, video_sources, outputs, out))
        def completion(directory):
            write(directory / 'producer_receipt.json', dict(status='COMPLETE_GTFREE_DONOR_PREDICTION_AND_DESCRIPTIVE_READOUT',
                seal_sha256=file_sha(directory / 'GT_free_predictions_sealed.json'),
                summary_sha256=file_sha(directory / 'summary.json')))
        completion(out)
        expectation = dict(sources=3, videos=3, frames=len(executions), rows=sum(row['observations'] for row in executions),
            candidates=candidates, prefix={lag: sum(min(lag, len(rows)) for rows in videos.values()) for lag in (1, 4)},
            frame_receipts=2 * len(executions))
        wrapper = root / 'fixture_cli.py'
        wrapper.write_text('import importlib.util,sys\n'
            's=importlib.util.spec_from_file_location("independent_fixture_auditor",%r)\n' % str(Path(__file__).resolve()) +
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n' +
            'm.EXPECTED_POPULATION=%r\n' % expectation +
            'm.main()\n')
        def invoke(protocol_file):
            return subprocess.run([sys.executable, str(wrapper), '--verify', '--protocol', str(protocol_file)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        positive = invoke(protocol_path)
        check(positive.returncode == 0, 'Complete temporary CLI failed: ' + positive.stderr[-3000:])
        positive_receipt = json.loads((out / 'independent_output_audit.json').read_text())
        check(positive_receipt['status'] == 'PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_OUTPUT_AUDIT', 'CLI audit did not seal')
        rejected = []
        for kind in ('numeric_gain', 'anchor_hash', 'exclusion_hash', 'prefix_receipt', 'source_summary'):
            changed = root / kind; shutil.copytree(out, changed)
            (changed / 'independent_output_audit.json').unlink()
            changed_protocol = dict(protocol, output_directory=str(changed))
            changed_protocol_path = root / (kind + '_protocol.json'); write(changed_protocol_path, changed_protocol)
            changed_seal = json.loads((changed / 'GT_free_predictions_sealed.json').read_text())
            changed_seal['protocol_sha256'] = file_sha(changed_protocol_path)
            if kind in ('numeric_gain', 'anchor_hash', 'exclusion_hash'):
                target = changed / 'videos/v11.npz'
                with np.load(target, allow_pickle=False) as saved: payload = {key: saved[key] for key in saved.files}
                if kind == 'numeric_gain':
                    payload['values'][0, 14] += 1.
                else:
                    key = 'anchor_hashes' if kind == 'anchor_hash' else 'exclusion_hashes'
                    payload[key][0, 1, 0] ^= np.uint8(1)
                np.savez_compressed(target, **payload)
                path = changed / 'videos/v11.json'; record = json.loads(path.read_text())
                record['npz_sha256'] = file_sha(target); write(path, record)
                changed_seal['video_outputs']['11']['npz_sha256'] = file_sha(target)
                changed_seal['video_outputs']['11']['receipt_sha256'] = file_sha(path)
            elif kind == 'prefix_receipt':
                path = changed / 'videos/v11.json'; record = json.loads(path.read_text())
                record['frame_pairs'].pop(0); write(path, record)
                changed_seal['video_outputs']['11']['receipt_sha256'] = file_sha(path)
            else:
                path = changed / 'summary.json'; record = json.loads(path.read_text())
                record['readouts']['fullpool']['1']['by_source']['a']['valid_holdouts'] += 1; write(path, record)
            write(changed / 'GT_free_predictions_sealed.json', changed_seal); completion(changed)
            negative = invoke(changed_protocol_path)
            check(negative.returncode != 0 and not (changed / 'independent_output_audit.json').exists(), 'Resealed corruption was accepted: ' + kind)
            rejected.append(dict(case=kind, actual_returncode=negative.returncode))
        return dict(status='PASS_TEMPORARY_DONOROUT_OUTPUT_CLI_QA',
            real_cache_opened=False, synthetic_population_override_only_in_temporary_wrapper=True,
            verifier_sha256=file_sha(__file__), producer_sha256=file_sha(producer_path),
            positive_actual_returncode=positive.returncode, positive_audit=positive_receipt,
            resealed_rejection_cases=rejected, forbidden_object_NPZ_members_ignored=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--producer-synthetic', action='store_true')
    modes.add_argument('--output-cli-self-test', action='store_true')
    modes.add_argument('--verify', action='store_true')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    if arguments.verify:
        check(arguments.protocol is not None and arguments.output is None, '--verify needs --protocol and writes its fixed output receipt')
        result = verify_outputs(arguments.protocol)
    elif arguments.output_cli_self_test:
        result = temporary_output_cli_check()
    else:
        result = producer_synthetic_check() if arguments.producer_synthetic else synthetic_selftest()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if arguments.output:
        with arguments.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
