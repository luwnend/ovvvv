"""Independent slow all-candidate oracle; synthetic arrays only.

No producer exclusion, matching, fit, or reduction helper is imported. The
canonical full-matrix floating convention is rebuilt before scalar scans.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / 'tmp_analysis/all_candidate_scene_residual_20261024.py'
FLOAT_KEYS = ('raw_cosine', 'pair_scale_xy', 'static_abs_L1', 'translation_xy',
              'residual_mad_xy', 'residual_dxdy', 'corrected_L1', 'gain')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_subject():
    spec = importlib.util.spec_from_file_location('independent_allcandidate_subject', SUBJECT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(raw):
    values = np.ascontiguousarray(raw, dtype=np.float64)
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = np.zeros_like(values)
    np.divide(values, maximum, out=scaled, where=maximum > 0)
    length = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    normalized = np.zeros_like(values)
    np.divide(scaled, length, out=normalized, where=length > 0)
    return normalized, [bool(x) for x in (maximum[:, 0] > 0)]


def exclusion_rows(boxes, proposals, endpoint):
    a = boxes[endpoint]
    result = []
    for row, b in enumerate(boxes):
        overlap = min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
        if row == endpoint or proposals[row] == proposals[endpoint] or overlap:
            result.append(row)
    return tuple(result)


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else values[n // 2 - 1] / 2 + values[n // 2] / 2


def oracle(qb, rb, qf, rf, qp, rp, minimum=3):
    qb, rb = np.asarray(qb, dtype=np.float64), np.asarray(rb, dtype=np.float64)
    qn, qvalid = canonical(qf)
    rn, rvalid = canonical(rf)
    similarity = qn @ rn.T
    n, m = len(qb), len(rb)
    qe = tuple(exclusion_rows(qb, qp, i) for i in range(n))
    re = tuple(exclusion_rows(rb, rp, j) for j in range(m))
    records = {}
    for i in range(n):
        for j in range(m):
            qs = [x for x in range(n) if x not in qe[i] and qvalid[x]]
            rs = [y for y in range(m) if y not in re[j] and rvalid[y]]
            anchors = []
            for x in qs:
                for y in rs:
                    value = similarity[x, y]
                    if value > 0 and all(value > similarity[x, z] for z in rs if z != y) and all(value > similarity[z, y] for z in qs if z != x):
                        anchors.append((x, y))
            scale = [float(np.sqrt(qb[i, k + 2] - qb[i, k]) * np.sqrt(rb[j, k + 2] - rb[j, k])) for k in (0, 1)]
            displacement = [float(qb[i, k] / 2 + qb[i, k + 2] / 2 - (rb[j, k] / 2 + rb[j, k + 2] / 2)) for k in (0, 1)]
            static = sum(abs(displacement[k] / scale[k]) for k in (0, 1))
            z = dict(available=len(anchors) >= minimum, anchor_count=len(anchors),
                     anchor_pairs=tuple(anchors), excluded_query_rows=qe[i], excluded_reference_rows=re[j],
                     raw_cosine=float(similarity[i, j]), pair_scale_xy=scale, static_abs_L1=static, translation_xy=None, residual_mad_xy=None,
                     residual_dxdy=None, corrected_L1=None, gain=None)
            if z['available']:
                ds = [[float(qb[x, k] / 2 + qb[x, k + 2] / 2 - (rb[y, k] / 2 + rb[y, k + 2] / 2)) for k in (0, 1)] for x, y in anchors]
                t = [median([v[k] for v in ds]) for k in (0, 1)]
                mad = [median([abs(v[k] - t[k]) for v in ds]) for k in (0, 1)]
                innovation = [(displacement[k] - t[k]) / scale[k] for k in (0, 1)]
                corrected = sum(abs(v) for v in innovation)
                z.update(translation_xy=t, residual_mad_xy=mad, residual_dxdy=innovation,
                         corrected_L1=corrected, gain=static - corrected)
            records[i, j] = z
    return qe, re, records


def compare(module, fixture, memoize=True):
    actual = module.all_candidate_scene_residual(*fixture, memoize=memoize)
    qe, re, expected = oracle(*fixture)
    n, m = len(fixture[0]), len(fixture[1])
    assert actual['candidate_shape'] == (n, m) and actual['candidate_count'] == n * m
    assert actual['query_exclusions'] == qe and actual['reference_exclusions'] == re
    assert actual['cosine_products_built'] == 1 and not actual['fullpool_h_used_to_filter_candidates']
    for (i, j), e in expected.items():
        a = actual['arrays']
        r = actual['fit_records'][a['fit_index'][i, j]]
        for key in ('anchor_pairs', 'excluded_query_rows', 'excluded_reference_rows', 'available', 'anchor_count'):
            assert r[key] == e[key], (i, j, key, r[key], e[key])
        for key in ('available', 'anchor_count'):
            assert a[key][i, j] == e[key]
        for key in FLOAT_KEYS:
            if key == 'raw_cosine':
                assert a[key][i, j] == e[key], ('canonical_fullC_raw_cosine', i, j)
            elif e[key] is None:
                assert np.isnan(a[key][i, j]).all(), (i, j, key)
            else:
                np.testing.assert_allclose(a[key][i, j], e[key], rtol=3e-13, atol=3e-13)
    return actual


def fixture(n=8, m=9, dimension=12, seed=20261024):
    rng = np.random.RandomState(seed)
    qxy, rxy = rng.uniform(-10, 20, (n, 2)), rng.uniform(-10, 20, (m, 2))
    qb = np.column_stack((qxy, qxy + rng.uniform(.2, 2, (n, 2))))
    rb = np.column_stack((rxy, rxy + rng.uniform(.2, 2, (m, 2))))
    qf, rf = rng.normal(size=(n, dimension)), rng.normal(size=(m, dimension))
    k = min(n, m)
    qf[:k] = rf[:k] + .01 * rng.normal(size=(k, dimension))
    return qb, rb, qf, rf, np.arange(n, dtype=np.int64), np.arange(m, dtype=np.int64)


def same_arrays(left, right):
    assert set(left['arrays']) == set(right['arrays'])
    for key in left['arrays']:
        if key == 'fit_index':
            continue
        np.testing.assert_array_equal(left['arrays'][key], right['arrays'][key])
    for i in range(left['candidate_shape'][0]):
        for j in range(left['candidate_shape'][1]):
            a = left['fit_records'][left['arrays']['fit_index'][i, j]]
            b = right['fit_records'][right['arrays']['fit_index'][i, j]]
            assert a == b


def run():
    begin = time.monotonic()
    module = load_subject()
    rounds = {}
    # Round 1: every candidate has an independent scalar oracle, then a fixture
    # where deleting a competitor creates a donor absent from fullpool MNN.
    candidates = 0
    for seed in range(20261024, 20261030):
        f = fixture(seed=seed)
        f[4][1] = f[4][0]
        f[5][2] = f[5][0]
        compare(module, f)
        candidates += len(f[0]) * len(f[1])
    f = list(fixture(n=6, m=6, dimension=8))
    f[0] = np.array([[i * 10., 0., i * 10 + 2., 2.] for i in range(6)])
    f[1] = f[0] - [1., 0., 1., 0.]
    f[2] = np.eye(8)[:6].copy()
    f[3] = f[2].copy()
    f[2][1, :2] = [.9, .1]
    a = compare(module, f)
    record = a['fit_records'][a['arrays']['fit_index'][0, 1]]
    assert (1, 0) in record['anchor_pairs'] and a['candidate_count'] == 36
    rounds['1'] = dict(checks=['independent_scalar_oracle_all_QxM_random_alias_pools_and_exact_canonical_raw_cosine',
                              'nonfullpool_target_and_exclusion_creates_new_reciprocal_donor'],
                       oracle_candidates=candidates + 36)
    # Round 2: direct, nontransitive positive overlap and distant proposal alias;
    # exact ties, zero/negative appearance targets, empty shapes and bad locators.
    boxes = np.array([[0., 0., 2., 2.], [1., 0., 3., 2.], [2., 0., 4., 2.],
                      [100., 0., 102., 2.], [120., 0., 122., 2.], [140., 0., 142., 2.]])
    proposals = np.array([0, 1, 2, 0, 4, 5])
    raw = np.eye(8)[:6]
    f = (boxes, boxes + [0., 5., 0., 5.], raw, raw, proposals, proposals)
    a = compare(module, f)
    assert a['query_exclusions'][0] == (0, 1, 3) and 2 not in a['query_exclusions'][0]
    f = list(fixture(n=7, m=8, dimension=10))
    f[2][0] = 0
    f[3][0] = 0
    f[2][2] = f[2][1]
    f[3][3] = f[3][2]
    f[2][4] = -f[3][4]
    a = compare(module, f)
    assert a['candidate_count'] == 56 and a['arrays']['available'].shape == (7, 8)
    for n, m in [(0, 4), (4, 0), (0, 0), (1, 1)]:
        compare(module, fixture(n=n, m=m))
    f = fixture()
    for value in [np.zeros(len(f[0]), dtype=bool), np.full(len(f[0]), -1, dtype=np.int64), np.full(len(f[0]), 2**64-1, dtype=np.uint64)]:
        changed = list(f)
        changed[4] = value
        try:
            module.all_candidate_scene_residual(*changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal proposal locator accepted')
    rounds['2'] = dict(checks=['direct_overlap_no_transitive_closure_touch_and_distant_proposal_alias',
                              'ties_zero_negative_targets_complete_missing_empty_and_invalid_proposals'])
    # Round 3: complete memoization semantics, exact C/F/strided input invariance,
    # no caller mutation, and dimensionless residual covariance to geometry units.
    f = list(fixture())
    f[4][:3] = 0
    f[5][:4] = 0
    snapshots = [x.copy() for x in f]
    a, b = compare(module, f, True), compare(module, f, False)
    same_arrays(a, b)
    assert a['donor_fits_computed'] < b['donor_fits_computed'] == len(f[0]) * len(f[1])
    for x, y in zip(f, snapshots):
        np.testing.assert_array_equal(x, y)
    ff = [np.asfortranarray(x) if x.ndim > 1 else x.copy() for x in f]
    c = compare(module, ff)
    same_arrays(a, c)
    strided = []
    for x in f:
        if x.ndim == 2:
            large = np.empty((len(x), x.shape[1]*2), dtype=x.dtype)
            large[:, ::2] = x
            strided.append(large[:, ::2])
        else:
            strided.append(x.copy())
    same_arrays(a, compare(module, strided))
    shifted = [x.copy() for x in f]
    shifted[0] += [30., -40., 30., -40.]
    shifted[1] += [30., -40., 30., -40.]
    moved = compare(module, shifted)
    scaled = [x.copy() for x in f]
    scaled[0] *= 4
    scaled[1] *= 4
    resized = compare(module, scaled)
    for key in ('static_abs_L1', 'residual_dxdy', 'corrected_L1', 'gain'):
        np.testing.assert_allclose(a['arrays'][key], moved['arrays'][key], rtol=5e-13, atol=5e-13, equal_nan=True)
        np.testing.assert_allclose(a['arrays'][key], resized['arrays'][key], rtol=5e-13, atol=5e-13, equal_nan=True)
    rounds['3'] = dict(checks=['memoize_on_off_per_candidate_arrays_and_exact_donor_provenance_equal',
                              'C_F_strided_layout_input_immutability_and_shared_geometry_unit_invariance'])
    return dict(status='PASS_INDEPENDENT_ALL_CANDIDATE_SCALAR_ORACLE_THREE_ROUNDS',
                subject_path=str(SUBJECT), subject_sha256=sha(SUBJECT),
                auditor_path=str(Path(__file__).resolve()), auditor_sha256=sha(__file__),
                frozen_cached_sha256=module.CACHED_SHA256, rounds=rounds,
                oracle_imports_subject_matching_or_exclusion_or_fit=False,
                actual_data_or_labels_read=False, training_or_full_FIT_started=False,
                elapsed_seconds=time.monotonic()-begin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    result = run()
    if args.output:
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))
