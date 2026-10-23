#!/usr/bin/env python3
"""Synthetic-only independent canonical-matrix oracle; never opens real caches."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location('independent_cached_subject', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(raw):
    # Deliberately separate implementation; canonical C float64 operations.
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    maximum = abs(raw).max(axis=1)
    scaled = np.zeros_like(raw)
    nonzero = maximum != 0
    scaled[nonzero] = raw[nonzero] / maximum[nonzero, None]
    lengths = np.linalg.norm(scaled, axis=1)
    scaled[nonzero] /= lengths[nonzero, None]
    return scaled, nonzero


def scalar_matches(matrix, qrows, rrows):
    """Scalar competition scan: no argmax/count reducer from subject."""
    answers = []
    for qi in qrows:
        values = [float(matrix[qi, rj]) for rj in rrows]
        if not values:
            continue
        best = max(values)
        winners = [rj for rj, value in zip(rrows, values) if value == best]
        if best <= 0 or len(winners) != 1:
            continue
        ri = winners[0]
        inverse = [qj for qj in qrows if matrix[qj, ri] == max(float(matrix[qk, ri]) for qk in qrows)]
        if inverse != [qi]:
            continue
        qalt = [float(matrix[qi, rj]) for rj in rrows if rj != ri]
        ralt = [float(matrix[qj, ri]) for qj in qrows if qj != qi]
        answers.append(dict(query_row=int(qi), reference_row=int(ri), raw_cosine=best,
            query_margin=None if not qalt else best-max(qalt),
            reference_margin=None if not ralt else best-max(ralt)))
    return answers


def oracle(qb, rb, qf, rf, qe=(), re=(), minimum=3):
    qn, qgood = normalized(qf)
    rn, rgood = normalized(rf)
    sim = qn @ rn.T
    qrows = [i for i in range(len(qb)) if qgood[i] and i not in qe]
    rrows = [i for i in range(len(rb)) if rgood[i] and i not in re]
    anchors = scalar_matches(sim, qrows, rrows)
    result = dict(available=False, reason='insufficient_reciprocal_anchors',
        minimum_anchors=int(minimum), anchor_count=len(anchors), anchors=anchors,
        translation=None, residual_mad_xy=None, max_leave_one_anchor_shift=None,
        units='original_image_pixels', physical_camera_motion_established=False,
        excluded_query_rows=sorted(set(qe)), excluded_reference_rows=sorted(set(re)))
    if len(anchors) < minimum:
        return result
    qc = qb[:, :2]/2+qb[:, 2:]/2
    rc = rb[:, :2]/2+rb[:, 2:]/2
    shifts = np.asarray([qc[a['query_row']]-rc[a['reference_row']] for a in anchors])
    center = np.median(shifts, axis=0)
    mad = np.median(abs(shifts-center), axis=0)
    # Brute leave-one medians are independent of the subject's optimized ranks.
    loo = np.asarray([np.median(shifts[[j for j in range(len(shifts)) if j != i]], axis=0)
                      for i in range(len(shifts))])
    result.update(available=True, reason='sufficient_reciprocal_anchors',
        translation=center.tolist(), residual_mad_xy=mad.tolist(),
        max_leave_one_anchor_shift=float(np.max(np.linalg.norm(loo-center, axis=1))))
    return result


def exact_anchors(actual, expected):
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected):
        assert set(a) == set(e)
        for key in ('query_row', 'reference_row'):
            assert type(a[key]) is int and a[key] == e[key], (actual, expected)
        for key in ('raw_cosine', 'query_margin', 'reference_margin'):
            assert a[key] == e[key], (key, a, e)


def equal(actual, expected):
    assert set(actual) == set(expected)
    exact_anchors(actual['anchors'], expected['anchors'])
    for key in ('available', 'reason', 'minimum_anchors', 'anchor_count', 'units',
                'physical_camera_motion_established', 'excluded_query_rows', 'excluded_reference_rows'):
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    for key in ('translation', 'residual_mad_xy', 'max_leave_one_anchor_shift'):
        if expected[key] is None:
            assert actual[key] is None
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=2e-13)


def boxes(rng, n):
    xy = rng.uniform(-100, 100, (n, 2))
    return np.ascontiguousarray(np.column_stack((xy, xy+rng.uniform(1, 40, (n, 2)))))


def must_raise(fn):
    try:
        fn()
    except (ValueError, TypeError, IndexError):
        return
    raise AssertionError('Expected validation failure')


def run(module, round_number):
    rng = np.random.RandomState(951240+round_number)
    cls = module.CachedSceneMotion
    checks = []
    comparisons = 0
    for n, m, dim in ((0, 0, 1), (0, 5, 2), (5, 0, 2), (1, 1, 1),
                      (4, 4, 8), (5, 7, 17), (9, 9, 256), (12, 10, 256), (77, 77, 256)):
        qb, rb = boxes(rng, n), boxes(rng, m)
        qf, rf = rng.normal(size=(n, dim)), rng.normal(size=(m, dim))
        k = min(n, m)
        qf[:k] = rf[:k]+rng.normal(scale=.01, size=(k, dim))
        if n > 4:
            qf[-1] = 0
        if m > 4:
            rf[-1] = 0
        qf, rf = np.ascontiguousarray(qf), np.ascontiguousarray(rf)
        cached = cls(qb, rb, qf, rf)
        patterns = [((), ()), (tuple(range(n)), ()), ((), tuple(range(m)))]
        if 0 < n <= 4 and 0 < m <= 4:
            qp = [tuple(i for i in range(n) if mask & (1 << i)) for mask in range(1 << n)]
            rp = [tuple(i for i in range(m) if mask & (1 << i)) for mask in range(1 << m)]
            patterns += list(itertools.product(qp, rp))
        else:
            patterns += [(tuple(sorted(set(rng.randint(0, n, min(n, 3)).tolist()))) if n else (),
                          tuple(sorted(set(rng.randint(0, m, min(m, 3)).tolist()))) if m else ())
                         for _ in range(16)]
            patterns += [((a['query_row'],), (a['reference_row'],)) for a in cached.fullpool_mnn()]
        for qe, re in patterns:
            for minimum in (3, 4):
                expected = oracle(qb, rb, qf, rf, qe, re, minimum)
                equal(cached.fit(excluded_query_rows=qe, excluded_reference_rows=re, minimum_anchors=minimum), expected)
                exact_anchors(cached.fullpool_mnn(excluded_query_rows=qe, excluded_reference_rows=re), expected['anchors'])
                comparisons += 2
        assert cached.cosine_products_built == 1
    checks.append('scalar_extrema_exact_membership_margin_and_missing_fullmatrix_oracle')

    # Duplicate values, strict ties, zero and negative rows are not tie-broken.
    qb, rb = boxes(rng, 8), boxes(rng, 8)
    qf = np.eye(8);rf=np.eye(8)
    qf[1] = qf[0];rf[3] = rf[2];qf[4] = 0;rf[5] = 0;qf[7] = -rf[7]
    obj = cls(qb, rb, qf, rf)
    for qe, re in (((), ()), ((0,), ()), ((1, 4), (2, 5)), ((0, 1, 4), (2, 3, 5))):
        equal(obj.fit(excluded_query_rows=qe, excluded_reference_rows=re), oracle(qb, rb, qf, rf, qe, re))
        comparisons += 1
    # Explicit match-after-exclusion counterexample.
    qf=np.zeros((4, 8));rf=np.zeros((4, 8))
    qf[0,:2]=[1,0];qf[1,:2]=[.9,.1];rf[0,:2]=[1,0];rf[1,:2]=[0,1]
    qf[2,2]=rf[2,2]=1;qf[3,3]=rf[3,3]=1
    obj=cls(qb[:4],rb[:4],qf,rf)
    before={(a['query_row'],a['reference_row']) for a in obj.fullpool_mnn()}
    after={(a['query_row'],a['reference_row']) for a in obj.fullpool_mnn(excluded_query_rows=(0,))}
    assert (1,0) not in before and (1,0) in after
    equal(obj.fit(excluded_query_rows=(0,)), oracle(qb[:4],rb[:4],qf,rf,(0,),()))
    checks.append('exact_ties_zero_negative_and_new_reciprocal_after_competitor_removal')

    # Check every donor removal at odd/even counts and duplicated displacement ranks.
    for n in range(3, 18):
        rb=np.tile([0.,0.,10.,10.],(n,1))
        shifts=rng.randint(-3,4,(n,2)).astype(float)
        qb=rb+np.column_stack((shifts,shifts))
        raw=np.eye(n)
        equal(cls(qb,rb,raw,raw).fit(),oracle(qb,rb,raw,raw))
        comparisons+=1
    # Heavy coherent moving majority remains ordinary image translation.
    rb=np.tile([0.,0.,10.,10.],(9,1));qb=rb.copy()
    qb[:3]+=[1,2,1,2];qb[3:]+=[20,-30,20,-30]
    raw=np.eye(9)
    fit=cls(qb,rb,raw*1e300,raw*1e-300).fit()
    assert fit['translation']==[20.,-30.] and not fit['physical_camera_motion_established']
    checks.append('brute_leaveone_median_rank_parity_odd_even_and_coherent_foreground_counterexample')

    # Mutate only fixed excluded endpoints, then mutate original caller arrays.
    qb,rb=boxes(rng,12),boxes(rng,12);rf=rng.normal(size=(12,256));qf=rf+.01*rng.normal(size=(12,256))
    obj=cls(qb,rb,qf,rf);qe,re=(0,2,4),(1,3,5)
    baseline=obj.fit(excluded_query_rows=qe,excluded_reference_rows=re)
    for _ in range(24):
        aq,ar,af,ag=qb.copy(),rb.copy(),qf.copy(),rf.copy()
        aq[list(qe)]+=rng.uniform(-1e7,1e7);ar[list(re)]+=rng.uniform(-1e7,1e7)
        af[list(qe)]=rng.normal(size=(3,256))*1e100
        ag[list(re)]=rng.normal(size=(3,256))*1e-100
        equal(cls(aq,ar,af,ag).fit(excluded_query_rows=qe,excluded_reference_rows=re),baseline)
        comparisons+=1
    expected=oracle(qb,rb,qf,rf,(0,),(1,))
    candidate=obj.candidate(0,1)
    equal(candidate['fit'],expected)
    if expected['available']:
        scale=np.sqrt(qb[0,2:]-qb[0,:2])*np.sqrt(rb[1,2:]-rb[1,:2])
        d=(qb[0,:2]/2+qb[0,2:]/2)-(rb[1,:2]/2+rb[1,2:]/2)
        np.testing.assert_allclose(candidate['normalized_innovation'],(d-np.asarray(expected['translation']))/scale,rtol=2e-13,atol=2e-13)
    assert candidate['score_or_gate_produced'] is False
    qb[:]+=1e5;rb[:]-=1e5;qf[:]=0;rf[:]=0
    equal(obj.fit(excluded_query_rows=qe,excluded_reference_rows=re),baseline)
    checks.append('fixed_exclusion_admissible_endpoint_mutation_snapshot_and_candidate_scale_sign')

    # Permuting well-separated candidates changes row locators only.
    n=10;qb,rb=boxes(rng,n),boxes(rng,n);raw=np.eye(n)
    base=cls(qb,rb,raw,raw).fit()
    qp,rp=rng.permutation(n),rng.permutation(n)
    transformed=cls(qb[qp],rb[rp],raw[qp],raw[rp]).fit()
    assert {(qp[a['query_row']],rp[a['reference_row']]) for a in transformed['anchors']} == {(i,i) for i in range(n)}
    for key in ('translation','residual_mad_xy','max_leave_one_anchor_shift'):
        np.testing.assert_allclose(transformed[key],base[key],rtol=2e-13,atol=2e-13)
    checks.append('independent_row_permutation_well_separated_membership_and_geometry')

    qb,rb=boxes(rng,4),boxes(rng,4);raw=np.eye(4);obj=cls(qb,rb,raw,raw)
    for bad in ((True,),(-1,),(4,),(1.0,),('1',),(np.bool_(True),)):
        must_raise(lambda bad=bad:obj.fit(excluded_query_rows=bad))
        must_raise(lambda bad=bad:obj.fit(excluded_reference_rows=bad))
    for minimum in (True,False,0,1,2,3.0,'3'):
        must_raise(lambda minimum=minimum:obj.fit(minimum_anchors=minimum))
    bad=qb.copy();bad[0,2]=bad[0,0]
    must_raise(lambda:cls(bad,rb,raw,raw))
    bad=qb.copy();bad[0,0]=np.nan
    must_raise(lambda:cls(bad,rb,raw,raw))
    must_raise(lambda:cls(qb,rb,raw,np.ones((4,3))))
    must_raise(lambda:cls(qb,rb,np.full((4,4),np.inf),raw))
    huge=np.tile([-1e308,0.,1e308,1.],(4,1))
    must_raise(lambda:cls(huge,rb,raw,raw))
    equal(obj.fit(excluded_query_rows=(0,0),excluded_reference_rows=(1,1)),oracle(qb,rb,raw,raw,(0,0),(1,1)))
    checks.append('invalid_shape_nonfinite_geometry_ranges_bool_and_duplicate_exclusion_contract')
    return checks,comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    assert sha(args.module)==args.expected_sha256,'Module pin mismatch'
    module=load(args.module)
    checks,comparisons=run(module,args.round)
    assert sha(args.module)==args.expected_sha256,'Module changed during audit'
    receipt=dict(status='PASS_INDEPENDENT_CACHED_SCENE_MOTION_CANONICAL_MATRIX_SYNTHETIC_AUDIT',
        round=args.round,module_sha256=args.expected_sha256,auditor_sha256=sha(__file__),
        checks=checks,full_output_comparisons=comparisons,
        discrete_membership_and_margins_compared_exactly=True,
        geometry_numeric_rtol=2e-13,geometry_numeric_atol=2e-13,
        sliced_reference_universal_bitwise_parity_claimed=False,
        real_cache_opened=False,new_OOF_outputs_opened=False,GT_or_class_or_track_ids_decoded=False,
        python=sys.executable,numpy_version=np.__version__,
        thread_environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        elapsed_seconds=time.monotonic()-started)
    with args.output.open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
