"""Cache raw cosine once, then recompute endpoint-excluded reciprocal matches.

Synthetic/reference use only. No cache reader, GT, category, tracker ID,
training, model score, camera interpretation, or real-data entrypoint exists.
Exclusions are supplied by the caller; this module cannot identify physical
aliases. It snapshots its inputs and normalizes each row independently.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


REFERENCE_SHA256 = '4ba1517f42d7b60e109db1675c9c4b1d54948cc3a45aeec5031cf40cf9ed730b'


def _validate(boxes, features):
    # Canonical C layout is part of the strict floating tie protocol. Equal
    # values in C/F/strided inputs must enter the same normalization kernel.
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    features = np.ascontiguousarray(features, dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or features.ndim != 2
            or len(boxes) != len(features) or features.shape[1] < 1
            or not np.isfinite(boxes).all()
            or not np.isfinite(features).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('Finite positive Nx4 boxes and aligned NxD features required')
    with np.errstate(over='ignore', invalid='ignore'):
        widths = boxes[:, 2:] - boxes[:, :2]
    if not np.isfinite(widths).all():
        raise ValueError('Box dimensions exceed finite float64 geometry')
    return boxes, features


def _finite_geometry(value):
    if not np.isfinite(value).all():
        raise ValueError('Geometry arithmetic exceeded finite float64 representation')
    return value


def _median_xy(values):
    ordered = np.sort(values, axis=0)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return ordered[n // 2 - 1] / 2 + ordered[n // 2] / 2


def _normalized(features):
    peak = np.max(np.abs(features), axis=1, keepdims=True)
    scaled = np.divide(features, peak, out=np.zeros_like(features), where=peak > 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0), peak[:, 0] > 0


def _exclusions(rows, count):
    rows = tuple(rows)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or i < 0 or i >= count for i in rows):
        raise ValueError('Excluded rows must be valid integer indices')
    return set(int(i) for i in rows)


def _unique_mutual(similarity, query_rows, reference_rows):
    if not similarity.size:
        return []
    row_best = np.argmax(similarity, axis=1)
    col_best = np.argmax(similarity, axis=0)
    row_values = similarity[np.arange(len(query_rows)), row_best]
    col_values = similarity[col_best, np.arange(len(reference_rows))]
    row_unique = np.count_nonzero(similarity == row_values[:, None], axis=1) == 1
    col_unique = np.count_nonzero(similarity == col_values[None, :], axis=0) == 1
    accepted = ((row_values > 0) & row_unique & col_unique[row_best]
                & (col_best[row_best] == np.arange(len(query_rows))))
    row_second = None if len(reference_rows) < 2 else np.partition(similarity, -2, axis=1)[:, -2]
    col_second = None if len(query_rows) < 2 else np.partition(similarity, -2, axis=0)[-2, :]
    anchors = []
    for qi in np.flatnonzero(accepted):
        ri = row_best[qi]
        value = row_values[qi]
        anchors.append(dict(
            query_row=int(query_rows[qi]), reference_row=int(reference_rows[ri]),
            raw_cosine=float(value),
            query_margin=None if row_second is None else float(value-row_second[qi]),
            reference_margin=None if col_second is None else float(value-col_second[ri])))
    return anchors


def _leave_one_medians(displacements):
    """Exact safe coordinate medians after each single deletion, one sort."""
    n = len(displacements)
    order = np.argsort(displacements, axis=0, kind='stable')
    ordered = np.take_along_axis(displacements, order, axis=0)
    ranks = np.empty_like(order)
    axes = np.arange(2)
    ranks[order, axes] = np.arange(n)[:, None]
    if n % 2 == 0:
        middle = n // 2 - 1
        return ordered[middle + (ranks <= middle), axes]
    lower, upper = n // 2 - 1, n // 2
    return (ordered[lower + (ranks <= lower), axes] / 2
            + ordered[upper + (ranks <= upper), axes] / 2)


def _snapshot(value):
    value = np.array(value, copy=True, order='C')
    value.setflags(write=False)
    return value


class CachedSceneMotion:
    """One immutable frame pair with exactly one normalized cosine product.

    ``fit`` matches the reference common_translation output schema. Strict tie,
    positivity, order, missing, median/MAD and fixed-anchor LOO rules are the
    same. Floating dot values can differ at roundoff from a differently shaped
    BLAS multiplication; no tolerance is used to redefine ties or positivity.
    """

    def __init__(self, query_boxes, reference_boxes, query_features, reference_features):
        qb, qf = _validate(query_boxes, query_features)
        rb, rf = _validate(reference_boxes, reference_features)
        if qf.shape[1] != rf.shape[1]:
            raise ValueError('Feature dimensions differ')
        self._qb, self._rb = _snapshot(qb), _snapshot(rb)
        qn, qvalid = _normalized(qf)
        rn, rvalid = _normalized(rf)
        self._qn, self._rn = _snapshot(qn), _snapshot(rn)
        self._qvalid, self._rvalid = _snapshot(qvalid), _snapshot(rvalid)
        self._similarity = _snapshot(qn @ rn.T)
        self._qc = _snapshot(qb[:, :2] / 2 + qb[:, 2:] / 2)
        self._rc = _snapshot(rb[:, :2] / 2 + rb[:, 2:] / 2)
        self.cosine_products_built = 1

    @property
    def query_count(self):
        return len(self._qb)

    @property
    def reference_count(self):
        return len(self._rb)

    def _pool(self, excluded_query_rows, excluded_reference_rows):
        qe = _exclusions(excluded_query_rows, self.query_count)
        re = _exclusions(excluded_reference_rows, self.reference_count)
        qi = np.asarray([i for i in range(self.query_count)
                         if i not in qe and self._qvalid[i]], dtype=np.int64)
        ri = np.asarray([i for i in range(self.reference_count)
                         if i not in re and self._rvalid[i]], dtype=np.int64)
        # Fresh directed maxima on the reduced pool, never filter fullpool MNN.
        anchors = _unique_mutual(self._similarity[np.ix_(qi, ri)], qi, ri)
        return qe, re, anchors

    def fullpool_mnn(self, *, excluded_query_rows=(), excluded_reference_rows=()):
        """Return ordered raw-only anchors; optional exclusions precede MNN.

        With defaults this is the full valid-row pool. Future donor holdout
        callers must invoke again with heldout endpoints excluded.
        """
        return self._pool(excluded_query_rows, excluded_reference_rows)[2]

    def fit(self, *, excluded_query_rows=(), excluded_reference_rows=(), minimum_anchors=3):
        if (isinstance(minimum_anchors, (bool, np.bool_))
                or not isinstance(minimum_anchors, (int, np.integer))
                or minimum_anchors < 3):
            raise ValueError('At least three anchors required by this reference contract')
        qe, re, anchors = self._pool(excluded_query_rows, excluded_reference_rows)
        result = dict(available=False, reason='insufficient_reciprocal_anchors',
                      minimum_anchors=int(minimum_anchors), anchor_count=len(anchors),
                      anchors=anchors, translation=None, residual_mad_xy=None,
                      max_leave_one_anchor_shift=None, units='original_image_pixels',
                      physical_camera_motion_established=False,
                      excluded_query_rows=sorted(qe), excluded_reference_rows=sorted(re))
        if len(anchors) < minimum_anchors:
            return result
        with np.errstate(over='ignore', invalid='ignore'):
            displacements = _finite_geometry(np.asarray([
                self._qc[a['query_row']] - self._rc[a['reference_row']] for a in anchors]))
            translation = _median_xy(displacements)
            residual_mad = _median_xy(_finite_geometry(np.abs(displacements-translation)))
            leave_one = _leave_one_medians(displacements)
            leave_difference = _finite_geometry(leave_one-translation)
            max_shift = _finite_geometry(np.hypot(leave_difference[:, 0], leave_difference[:, 1])).max()
        result.update(available=True, reason='sufficient_reciprocal_anchors',
                      translation=translation.tolist(), residual_mad_xy=residual_mad.tolist(),
                      max_leave_one_anchor_shift=float(max_shift))
        return result

    def candidate(self, query_row, reference_row, *, extra_query_exclusions=(),
                  extra_reference_exclusions=(), minimum_anchors=3):
        _exclusions((query_row,), self.query_count)
        _exclusions((reference_row,), self.reference_count)
        fit = self.fit(excluded_query_rows=(query_row,) + tuple(extra_query_exclusions),
                       excluded_reference_rows=(reference_row,) + tuple(extra_reference_exclusions),
                       minimum_anchors=minimum_anchors)
        innovation = None
        if fit['available']:
            q, r = self._qb[query_row], self._rb[reference_row]
            scale = np.sqrt(q[2:]-q[:2]) * np.sqrt(r[2:]-r[:2])
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                innovation = _finite_geometry((q[:2] / 2 + q[2:] / 2
                    - (r[:2] / 2 + r[2:] / 2) - np.asarray(fit['translation'])) / scale).tolist()
        return dict(fit=fit, normalized_innovation=innovation,
                    innovation_available=fit['available'], score_or_gate_produced=False)


def _reference():
    path = Path(__file__).with_name('endpoint_excluded_scene_motion.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Frozen reference changed')
    spec = importlib.util.spec_from_file_location('scene_motion_frozen_reference', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _equal(actual, expected, path='output'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            _equal(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _equal(a, e, path+'[%d]' % i)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13,
                                   equal_nan=True, err_msg=path)
    else:
        assert type(actual) is type(expected) and actual == expected, path


def _fixture(rng, n=12, m=None, dim=256):
    m = n if m is None else m
    qxy, rxy = rng.uniform(0, 100, (n, 2)), rng.uniform(0, 100, (m, 2))
    qb = np.column_stack((qxy, qxy+rng.uniform(5, 30, (n, 2))))
    rb = np.column_stack((rxy, rxy+rng.uniform(5, 30, (m, 2))))
    rf = rng.normal(size=(m, dim))
    qf = rng.normal(size=(n, dim))
    k = min(n, m)
    qf[:k] = rf[:k] + rng.normal(scale=.025, size=(k, dim))
    return qb, rb, qf, rf


def synthetic_self_test(round_number=1):
    ref = _reference()
    rng = np.random.RandomState(20261024 + int(round_number))
    checks, comparisons = [], 0
    qb, rb, qf, rf = _fixture(rng)
    cached = CachedSceneMotion(qb, rb, qf, rf)
    patterns = [((), ()), ((0,), (1,)), ((0, 2, 4), (1, 3, 5)),
                (tuple(range(12)), ()), ((), tuple(range(12))), ((0, 0), (1, 1))]
    patterns += [((i,), (j,)) for i in range(12) for j in range(12)]
    for qe, re in patterns:
        for minimum in (3, 4, 12, 13):
            kwargs = dict(excluded_query_rows=qe, excluded_reference_rows=re, minimum_anchors=minimum)
            expected = ref.common_translation(qb, rb, qf, rf, **kwargs)
            _equal(cached.fit(**kwargs), expected)
            _equal(cached.fullpool_mnn(excluded_query_rows=qe, excluded_reference_rows=re), expected['anchors'])
            comparisons += 2
    for i, j in ((0, 1), (3, 3), (11, 2)):
        kwargs = dict(extra_query_exclusions=(2, 4), extra_reference_exclusions=(0, 5))
        _equal(cached.candidate(i, j, **kwargs),
               ref.candidate_translation_evidence(qb, rb, qf, rf, i, j, **kwargs))
        comparisons += 1
    assert cached.cosine_products_built == 1
    checks.append('full_fit_anchor_margin_missing_LOO_candidate_schema_reference_parity')

    eye = np.eye(256)[:12].copy()
    tieq, tier = eye.copy(), eye.copy()
    tieq[1] = tieq[0]; tier[3] = tier[2]; tieq[4] = 0; tier[5] = 0
    for a, b, c, d in ((qb, rb, tieq, tier),
                        (qb[:0], rb, qf[:0], rf), (qb, rb[:0], qf, rf[:0]),
                        (qb[:0], rb[:0], qf[:0], rf[:0]),
                        (qb[:1], rb[:1], qf[:1], qf[:1]),
                        (qb, rb, np.zeros_like(qf), np.zeros_like(rf))):
        obj = CachedSceneMotion(a, b, c, d)
        _equal(obj.fit(), ref.common_translation(a, b, c, d)); comparisons += 1
    _equal(CachedSceneMotion(qb, rb, tieq, tier).fit(excluded_query_rows=(0, 1, 6), excluded_reference_rows=(2, 3, 7)),
           ref.common_translation(qb, rb, tieq, tier, excluded_query_rows=(0, 1, 6), excluded_reference_rows=(2, 3, 7)))
    comparisons += 1
    checks.append('strict_ties_zero_empty_singleton_alias_exclusions_preserve_missing')

    # Deleting a competitor can CREATE a reciprocal match: filtering fullpool
    # matches after exclusion is observably wrong.
    cfq = np.zeros((4, 256)); cfr = np.zeros((4, 256))
    cfq[0, :2] = (1, 0); cfq[1, :2] = (.9, .1)
    cfr[0, :2] = (1, 0); cfr[1, :2] = (0, 1)
    cfq[2, 2] = cfr[2, 2] = 1; cfq[3, 3] = cfr[3, 3] = 1
    obj = CachedSceneMotion(qb[:4], rb[:4], cfq, cfr)
    before = {(a['query_row'], a['reference_row']) for a in obj.fullpool_mnn()}
    after = {(a['query_row'], a['reference_row']) for a in obj.fullpool_mnn(excluded_query_rows=(0,))}
    assert (1, 0) not in before and (1, 0) in after
    _equal(obj.fit(excluded_query_rows=(0,)),
           ref.common_translation(qb[:4], rb[:4], cfq, cfr, excluded_query_rows=(0,)))
    comparisons += 1
    checks.append('deleted_competitor_recomputes_MNN_and_creates_new_anchor')

    qe, re = (0, 2, 4), (1, 3, 5)
    baseline = cached.fit(excluded_query_rows=qe, excluded_reference_rows=re)
    aq, ar, af, ag = qb.copy(), rb.copy(), qf.copy(), rf.copy()
    aq[list(qe)] += 1e6; ar[list(re)] -= 1e6
    af[list(qe)] = rng.normal(size=(len(qe), 256))*1e150
    ag[list(re)] = rng.normal(size=(len(re), 256))*1e150
    _equal(CachedSceneMotion(aq, ar, af, ag).fit(excluded_query_rows=qe, excluded_reference_rows=re), baseline)
    qb[:] += 10000; rb[:] -= 20000; qf[:] = 0; rf[:] = 0
    _equal(cached.fit(excluded_query_rows=qe, excluded_reference_rows=re), baseline)
    checks.append('fixed_exclusion_value_mutation_and_caller_input_mutation_invariance')

    # Most coherent donors may all move: parity must not invent camera truth
    # or discard the majority because their translation is large.
    rbox = np.tile([0., 0., 10., 10.], (9, 1))
    qbox = rbox.copy(); qbox[:3] += [1, 2, 1, 2]; qbox[3:] += [20, -30, 20, -30]
    feats = np.eye(256)[:9]
    result = CachedSceneMotion(qbox, rbox, feats*1e300, feats*1e-300).fit()
    _equal(result, ref.common_translation(qbox, rbox, feats*1e300, feats*1e-300))
    assert result['translation'] == [20., -30.] and not result['physical_camera_motion_established']
    for bad in (float('nan'), float('inf')):
        broken = feats.copy(); broken[0, 0] = bad
        for fn in (lambda: CachedSceneMotion(qbox, rbox, broken, feats),
                   lambda: ref.common_translation(qbox, rbox, broken, feats)):
            try: fn()
            except ValueError: pass
            else: raise AssertionError('Nonfinite input accepted')
    invalids = [dict(minimum_anchors=2), dict(minimum_anchors=True),
                dict(excluded_query_rows=(True,)), dict(excluded_reference_rows=(9,))]
    finite_obj = CachedSceneMotion(qbox, rbox, feats, feats)
    for kwargs in invalids:
        for fn in (lambda: finite_obj.fit(**kwargs),
                   lambda: ref.common_translation(qbox, rbox, feats, feats, **kwargs)):
            try: fn()
            except ValueError: pass
            else: raise AssertionError('Invalid contract accepted')
    checks.append('coherent_majority_outliers_extreme_finite_features_nonfinite_and_invalid_contracts')
    return dict(status='PASS_CACHED_ENDPOINT_SCENE_MOTION_SYNTHETIC',
                round=int(round_number), checks=checks, full_output_comparisons=comparisons,
                numeric_rtol=2e-13, numeric_atol=2e-13,
                structure_anchor_order_tie_missing_keys_compared_exactly=True,
                reference_sha256=REFERENCE_SHA256, real_cache_opened=False,
                new_oof_readout_opened=False, trained=False)


def synthetic_benchmark():
    """77x77, 30 independently generated frame pairs, every fullpool MNN LOO."""
    ref = _reference(); rng = np.random.RandomState(771030)
    fixtures = [_fixture(rng, 77) for _ in range(30)]
    tick = time.perf_counter()
    objects = [CachedSceneMotion(*fixture) for fixture in fixtures]
    build_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    fullpools = [obj.fullpool_mnn() for obj in objects]
    mnn_seconds = time.perf_counter()-tick
    tick = time.perf_counter(); cached_outputs = []
    for obj, anchors in zip(objects, fullpools):
        cached_outputs.append([obj.candidate(a['query_row'], a['reference_row']) for a in anchors])
    cached_leaveout_seconds = time.perf_counter()-tick
    tick = time.perf_counter(); reference_outputs = []
    for fixture, anchors in zip(fixtures, fullpools):
        reference_outputs.append([ref.candidate_translation_evidence(*fixture, a['query_row'], a['reference_row']) for a in anchors])
    reference_leaveout_seconds = time.perf_counter()-tick
    _equal(cached_outputs, reference_outputs)
    assert all(obj.cosine_products_built == 1 for obj in objects)
    return dict(status='PASS_SYNTHETIC_77X77_30_PAIRS_FULLPOOL_MNN_LEAVEOUT_BENCHMARK',
                query_rows=77, reference_rows=77, dimension=256, frame_pairs=30,
                fullpool_anchor_counts=[len(x) for x in fullpools],
                heldout_endpoint_pairs=sum(map(len, fullpools)),
                cosine_products_built=sum(x.cosine_products_built for x in objects),
                build_seconds=build_seconds, fullpool_mnn_seconds=mnn_seconds,
                cached_leaveout_seconds=cached_leaveout_seconds,
                reference_leaveout_seconds=reference_leaveout_seconds,
                cached_total_seconds=build_seconds+mnn_seconds+cached_leaveout_seconds,
                reference_full_output_parity=True, synthetic_only=True,
                workload='all fullpool MNN donor leaveouts, not all 5929 candidates per frame',
                runtime_is_not_full_cache_estimate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-round', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = synthetic_self_test(args.synthetic_round)
    if args.benchmark:
        result['benchmark'] = synthetic_benchmark()
    result['module_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as handle:
            handle.write(payload)
    print(payload, end='')
