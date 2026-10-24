"""Synthetic-only independent scalar all-candidate oracle and invariance QA."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np


BASE = Path(__file__).resolve().parent
SUBJECT = BASE / 'all_candidate_scene_residual_20261024.py'
FLOAT_FIELDS = ('raw_cosine','pair_scale_xy','static_abs_L1','translation_xy','residual_mad_xy',
                'residual_dxdy','corrected_L1','gain')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def canonical_matrix(q, r):
    # Independent construction from the canonical arithmetic contract, without
    # importing subject/frozen engine normalization, matcher or fit helpers.
    def norm(x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        peaks = np.max(np.abs(x), axis=1, keepdims=True)
        x = np.divide(x, peaks, out=np.zeros_like(x), where=peaks>0)
        sizes = np.sqrt(np.add.reduce(x*x, axis=1, keepdims=True))
        return np.divide(x, sizes, out=np.zeros_like(x), where=sizes>0), peaks[:,0]>0
    a, av = norm(q)
    b, bv = norm(r)
    return a @ b.T, av, bv


def scalar_median(values):
    s = sorted(float(v) for v in values)
    n = len(s)
    return s[n//2] if n%2 else s[n//2-1]/2+s[n//2]/2


def exclusions(boxes, proposals, endpoint):
    result = []
    e = boxes[endpoint]
    for i, b in enumerate(boxes):
        direct = (min(float(b[2]),float(e[2])) > max(float(b[0]),float(e[0]))
                  and min(float(b[3]),float(e[3])) > max(float(b[1]),float(e[1])))
        if direct or int(proposals[i]) == int(proposals[endpoint]):
            result.append(i)
    return tuple(result)


def strict_scan(similarity, qrows, rrows):
    accepted = []
    for q in qrows:
        vals = [float(similarity[q,r]) for r in rrows]
        if not vals:
            continue
        largest = max(vals)
        if largest <= 0 or vals.count(largest) != 1:
            continue
        r = rrows[vals.index(largest)]
        column = [float(similarity[k,r]) for k in qrows]
        best = max(column)
        if column.count(best) == 1 and qrows[column.index(best)] == q:
            accepted.append((q,r))
    return tuple(accepted)


def slow_oracle(qb, rb, qf, rf, qp, rp, minimum=3):
    qb, rb = np.asarray(qb,dtype=np.float64), np.asarray(rb,dtype=np.float64)
    matrix, qvalid, rvalid = canonical_matrix(qf,rf)
    result = {}
    for i in range(len(qb)):
        for j in range(len(rb)):
            qe, re = exclusions(qb,qp,i), exclusions(rb,rp,j)
            qr = [k for k in range(len(qb)) if k not in qe and qvalid[k]]
            rr = [k for k in range(len(rb)) if k not in re and rvalid[k]]
            anchors = strict_scan(matrix,qr,rr)
            centres = lambda b: (float(b[0])/2+float(b[2])/2,float(b[1])/2+float(b[3])/2)
            qc, rc = centres(qb[i]), centres(rb[j])
            scale = [math.sqrt(float(qb[i,k+2]-qb[i,k]))*math.sqrt(float(rb[j,k+2]-rb[j,k])) for k in range(2)]
            displacement = [qc[k]-rc[k] for k in range(2)]
            static = sum(abs(displacement[k]/scale[k]) for k in range(2))
            item = dict(excluded_query_rows=qe,excluded_reference_rows=re,anchor_pairs=anchors,
                        available=len(anchors)>=minimum,anchor_count=len(anchors),
                        raw_cosine=float(matrix[i,j]),pair_scale_xy=scale,static_abs_L1=static,
                        translation_xy=[np.nan,np.nan],residual_mad_xy=[np.nan,np.nan],
                        residual_dxdy=[np.nan,np.nan],corrected_L1=np.nan,gain=np.nan)
            if item['available']:
                donor = [[centres(qb[q])[k]-centres(rb[r])[k] for k in range(2)] for q,r in anchors]
                t = [scalar_median([d[k] for d in donor]) for k in range(2)]
                mad = [scalar_median([abs(d[k]-t[k]) for d in donor]) for k in range(2)]
                residual = [(displacement[k]-t[k])/scale[k] for k in range(2)]
                corrected = sum(abs(x) for x in residual)
                item.update(translation_xy=t,residual_mad_xy=mad,residual_dxdy=residual,
                            corrected_L1=corrected,gain=static-corrected)
            result[i,j] = item
    return result, matrix, qvalid, rvalid


def check_all(subject, fixture, counter, minimum=3):
    snapshots = [np.array(x,copy=True) for x in fixture]
    actual = subject(*fixture,minimum_anchors=minimum)
    oracle, _, _, _ = slow_oracle(*fixture,minimum=minimum)
    n,m = len(fixture[0]),len(fixture[1])
    require(actual['candidate_shape']==(n,m) and actual['candidate_count']==n*m,'Complete QxM')
    require(actual['cosine_products_built']==1 and actual['fullpool_h_used_to_filter_candidates'] is False,'One full canonical matrix/no h filter')
    arrays=actual['arrays']
    for field in FLOAT_FIELDS:
        require(arrays[field].shape == ((n,m,2) if field.endswith('_xy') or field=='residual_dxdy' else (n,m)),field+' full shape')
        require(arrays[field].dtype == np.float64,field+' float64')
    require(arrays['available'].dtype==np.bool_ and arrays['anchor_count'].dtype==np.int64 and arrays['fit_index'].dtype==np.int64,'Discrete dtypes')
    for (i,j), expected in oracle.items():
        fit=actual['fit_records'][arrays['fit_index'][i,j]]
        for field in ('excluded_query_rows','excluded_reference_rows','anchor_pairs','available','anchor_count'):
            require(fit[field]==expected[field],'%s exact for %s'%(field,(i,j)))
            counter['exact_membership_fields']+=1
        require(bool(arrays['available'][i,j])==expected['available'],'Availability')
        require(int(arrays['anchor_count'][i,j])==expected['anchor_count'],'Anchor count')
        for field in FLOAT_FIELDS:
            np.testing.assert_allclose(arrays[field][i,j],expected[field],atol=2e-12,rtol=2e-12,equal_nan=True,err_msg=field+str((i,j)))
            counter['numeric_fields']+=1
        counter['candidates']+=1
    for before, after in zip(snapshots,fixture):
        require(before.dtype==after.dtype and before.shape==after.shape and before.tobytes()==after.tobytes(),'Caller input unchanged')
    return actual


def spaced(n,shift=(0,0)):
    xy=np.column_stack((np.arange(n,dtype=np.float64)*32+shift[0],np.arange(n,dtype=np.float64)*4+shift[1]))
    return np.column_stack((xy,xy+np.array([8.,8.])))


def run(round_number):
    begin=time.monotonic()
    s=importlib.util.spec_from_file_location('all_candidate_subject',SUBJECT);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    subject=module.all_candidate_scene_residual
    rng=np.random.RandomState(20261024+int(round_number))
    checks=[];counter=dict(candidates=0,exact_membership_fields=0,numeric_fields=0)
    for n,m,dim in [(8,7,17),(7,9,33),(6,6,16)]:
        qb,rb=spaced(n,(3,-2)),spaced(m)
        qf,rf=rng.normal(size=(n,dim)),rng.normal(size=(m,dim))
        qf[:min(n,m)]=rf[:min(n,m)]+rng.normal(scale=.03,size=(min(n,m),dim))
        qf[-1]=0
        qp,rp=np.arange(n,dtype=np.int64),np.arange(m,dtype=np.int64)
        qp[-1]=qp[0]
        check_all(subject,(qb,rb,qf,rf,qp,rp),counter)
    checks.append('all_candidates_independent_scalar_oracle_exact_anchors_exclusions_numeric_input_immutability')

    boxes=spaced(8);eye=np.eye(16)[:8];props=np.arange(8,dtype=np.int64)
    fixture=(boxes+np.array([3.,-2.,3.,-2.]),boxes,eye.copy(),eye.copy(),props.copy(),props.copy())
    a=check_all(subject,fixture,counter)
    require(a['arrays']['available'][0,1],'Non-fullpool-MNN zero cosine target retained with donors')
    neg=list(fixture);neg[2]=eye.copy();neg[2][0]=-eye[1]
    b=check_all(subject,tuple(neg),counter)
    require(b['candidate_count']==64 and b['arrays']['available'][0,1],'Negative raw cosine target retained')
    # Removing a competitor creates (1,0), which is absent in fullpool MNN.
    cfq=np.eye(16)[:6].copy();cfr=cfq.copy();cfq[0,:2]=(1,0);cfq[1,:2]=(.9,.1);cfr[1,:2]=(0,1)
    created=check_all(subject,(spaced(6),spaced(6),cfq,cfr,np.arange(6),np.arange(6)),counter)
    mat,qv,rv=canonical_matrix(cfq,cfr)
    before=strict_scan(mat,list(np.flatnonzero(qv)),list(np.flatnonzero(rv)))
    fit=created['fit_records'][created['arrays']['fit_index'][0,5]]
    require((1,0) not in before and (1,0) in fit['anchor_pairs'],'Fresh MNN creates donor after endpoint exclusion')
    checks.append('non_MNN_zero_negative_cosine_candidates_retained_and_deleted_competitor_rematches')

    qb=spaced(8);rb=spaced(8)
    qb[:4]=np.array([[0.,0.,10.,10.],[10.,0.,20.,10.],[19.,0.,29.,10.],[28.,0.,38.,10.]])
    qp=props.copy();qp[3]=qp[0]
    aliases=check_all(subject,(qb,rb,eye,eye,qp,props),counter)
    require(aliases['query_exclusions'][0]==(0,3),'Disjoint same-proposal alias excluded, touching row1 retained')
    require(aliases['query_exclusions'][1]==(1,2),'Direct chain overlap excludes B/C but no transitive closure')
    qp=props.copy();rp=props.copy();qp[1]=qp[0];rp[1]=rp[0]
    mf=(spaced(8),spaced(8),eye,eye,qp,rp)
    on=check_all(subject,mf,counter);off=subject(*mf,memoize=False)
    require(on['donor_fits_computed']==49 and off['donor_fits_computed']==64,'Equivalent sets memoized once')
    for k in on['arrays']:
        if k!='fit_index':np.testing.assert_array_equal(on['arrays'][k],off['arrays'][k])
    for i in range(8):
        for j in range(8):
            x=on['fit_records'][on['arrays']['fit_index'][i,j]]
            y=off['fit_records'][off['arrays']['fit_index'][i,j]]
            require(x==y,'Memoization exact fit provenance')
    checks.append('strict_alias_touching_nontransitive_overlap_and_memo_on_off_exact_candidate_provenance')

    tq,tr=eye.copy(),eye.copy();tq[1]=tq[0];tr[3]=tr[2];tq[4]=0
    check_all(subject,(spaced(8),spaced(8),tq,tr,props,props),counter)
    for n,m in [(0,0),(0,4),(4,0),(1,1),(3,3)]:
        low=check_all(subject,(spaced(n),spaced(m),np.eye(16)[:n],np.eye(16)[:m],np.arange(n),np.arange(m)),counter)
        require(not low['arrays']['available'].any(),'Low support stays unavailable')
        for k in ('translation_xy','residual_mad_xy','residual_dxdy','corrected_L1','gain'):
            require(np.isnan(low['arrays'][k]).all(),'Missing donors preserve NaN:'+k)
    check_all(subject,(spaced(8),spaced(8),np.zeros((8,16)),np.zeros((8,16)),props,props),counter)
    checks.append('strict_ties_zero_raw_empty_query_reference_low_support_keep_full_shape_and_NaNs')

    # Endpoint perturbation is conditional on unchanged exclusion sets.
    base=subject(*fixture);changed=[x.copy() for x in fixture]
    changed[0][0]+=np.array([1.,-1.,1.,-1.]);changed[1][1]+=np.array([-2.,1.,-2.,1.])
    changed[2][0]=rng.normal(size=16)*1e100;changed[3][1]=rng.normal(size=16)*1e-100
    after=check_all(subject,tuple(changed),counter)
    require(base['query_exclusions'][0]==after['query_exclusions'][0] and base['reference_exclusions'][1]==after['reference_exclusions'][1],'Fixed exclusion condition')
    old=base['fit_records'][base['arrays']['fit_index'][0,1]];new=after['fit_records'][after['arrays']['fit_index'][0,1]]
    require(old==new,'Excluded endpoint values do not alter own donor estimate under fixed sets')
    for field in FLOAT_FIELDS:
        transformed=[x.copy() for x in fixture]
        transformed[0]+=np.array([64.,-128.,64.,-128.]);transformed[1]+=np.array([64.,-128.,64.,-128.])
        shifted=subject(*transformed)
        np.testing.assert_allclose(base['arrays'][field],shifted['arrays'][field],atol=2e-12,rtol=2e-12,equal_nan=True)
    scaled=[x.copy() for x in fixture];factor=np.array([2.,4.,2.,4.]);scaled[0]*=factor;scaled[1]*=factor
    scaled=check_all(subject,tuple(scaled),counter)
    for field in ('residual_dxdy','static_abs_L1','corrected_L1','gain'):
        np.testing.assert_allclose(base['arrays'][field],scaled['arrays'][field],atol=2e-12,rtol=2e-12,equal_nan=True)
    for field in ('pair_scale_xy','translation_xy','residual_mad_xy'):
        np.testing.assert_allclose(base['arrays'][field]*factor[:2],scaled['arrays'][field],atol=2e-12,rtol=2e-12,equal_nan=True)
    checks.append('fixed_exclusion_endpoint_perturbation_and_common_translation_axis_scale_geometry_invariance')

    # Equal values in C/F/strided arrays must enter the same full matrix kernel.
    variants=[]
    for layout in ('F','stride'):
        parts=[]
        for x in fixture:
            if x.ndim==2:
                if layout=='F':y=np.asfortranarray(x)
                else:
                    storage=np.empty((len(x),2*x.shape[1]),dtype=x.dtype);storage[:,::2]=x;y=storage[:,::2]
            else:y=x.copy()
            parts.append(y)
        variants.append(subject(*parts))
    for other in variants:
        for k in base['arrays']:np.testing.assert_array_equal(base['arrays'][k],other['arrays'][k])
        require(base['fit_records']==other['fit_records'],'Canonical input layout provenance')
    bad=[]
    for proposals in (np.ones(8,dtype=bool),np.full(8,-1,dtype=np.int64),np.full(8,2**63,dtype=np.uint64),np.zeros(8,dtype=float)):
        args=list(fixture);args[4]=proposals;bad.append((args,{}))
    args=[x.copy() for x in fixture];args[2][0,0]=np.nan;bad.append((args,{}))
    args=[x.copy() for x in fixture];args[0][0,2]=args[0][0,0];bad.append((args,{}))
    args=[x.copy() for x in fixture];args[0][:]=[-1e308,0.,-9e307,8.];args[1][:]=[9e307,0.,1e308,8.];bad.append((args,{}))
    bad.extend([(fixture,{'minimum_anchors':2}),(fixture,{'minimum_anchors':True}),(fixture,{'memoize':1})])
    for args,kwargs in bad:
        try:subject(*args,**kwargs)
        except ValueError:pass
        else:raise AssertionError('Invalid/overflow input accepted')
    checks.append('C_F_strided_canonical_parity_invalid_boolean_overflow_proposal_and_geometry_fail_closed')
    require(len(checks)>=2,'At least two independent checks')
    return dict(status='PASS_INDEPENDENT_ALL_CANDIDATE_SCENE_RESIDUAL_SYNTHETIC_QA',round=int(round_number),
        subject_sha256=sha(SUBJECT),cached_sha256=sha(module.CACHED_PATH),auditor_sha256=sha(__file__),
        seed=20261024+int(round_number),checks=checks,comparisons=counter,numeric_atol=2e-12,numeric_rtol=2e-12,
        oracle='Independent canonical full C matrix and explicit per-row/per-column strict scan, scalar medians, direct scalar overlap, scalar candidate geometry; no subject/frozen fit or matcher helper used.',
        endpoint_independence='Only conditional on unchanged explicit endpoint/proposal/direct-overlap exclusion sets.',
        real_cache_or_new_results_opened=False,GPU_or_training_or_method_effectiveness_or_novelty_established=False,
        elapsed_seconds=time.monotonic()-begin)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.round)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
