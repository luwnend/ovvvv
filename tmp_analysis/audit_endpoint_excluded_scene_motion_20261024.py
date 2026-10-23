#!/usr/bin/env python3
"""Independent synthetic audit; no real cache, identity or OOF result is read."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/"tmp_analysis/endpoint_excluded_scene_motion.py"
EXPECTED_SHA="4ba1517f42d7b60e109db1675c9c4b1d54948cc3a45aeec5031cf40cf9ed730b"


def check(value,message):
    if not value:raise AssertionError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boxes(centers):
    centers=np.asarray(centers,float).reshape(-1,2);return np.c_[centers-1,centers+1]


def equal(left,right):
    check(json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False),"Expected exact endpoint-excluded evidence invariance")


def assert_finite(value):
    json.dumps(value,allow_nan=False)


def reject(function,*args,**kwargs):
    try:function(*args,**kwargs)
    except ValueError:return
    raise AssertionError("Invalid input was accepted")


def scalar_median(values):
    values=sorted(float(v) for v in values);n=len(values)
    return values[n//2] if n%2 else values[n//2-1]/2+values[n//2]/2


def oracle(qb,rb,qf,rf,qe=(),re=()):
    # Independently enumerate strict row/column maxima from scalar cosine.
    qs=[i for i in range(len(qb)) if i not in qe and np.any(qf[i])];rs=[j for j in range(len(rb)) if j not in re and np.any(rf[j])]
    cosine={}
    for i in qs:
        for j in rs:
            x=qf[i]/max(map(abs,qf[i]));y=rf[j]/max(map(abs,rf[j]));xn=math.sqrt(sum(float(a)*a for a in x));yn=math.sqrt(sum(float(b)*b for b in y))
            cosine[i,j]=sum(float(a)*b for a,b in zip(x,y))/xn/yn
    pairs=[]
    for i in qs:
        for j in rs:
            score=cosine[i,j]
            if score>0 and all(score>cosine[i,k] for k in rs if k!=j) and all(score>cosine[k,j] for k in qs if k!=i):pairs.append((i,j))
    displacement=np.array([(qb[i,:2]/2+qb[i,2:]/2)-(rb[j,:2]/2+rb[j,2:]/2) for i,j in pairs],float).reshape(-1,2)
    if len(pairs)<3:return pairs,None,None,None
    translation=np.array([scalar_median(displacement[:,c]) for c in (0,1)])
    mad=np.array([scalar_median(abs(displacement[:,c]-translation[c])) for c in (0,1)])
    shift=max(math.hypot(*(np.array([scalar_median([d[c] for k,d in enumerate(displacement) if k!=i]) for c in (0,1)])-translation)) for i in range(len(pairs)))
    return pairs,translation,mad,shift


def round_one(m):
    checks=[];r=boxes([[k*10,k] for k in range(6)]);q=r+np.tile([7,-3,7,-3],(6,1));features=np.eye(6)
    initial=m.candidate_translation_evidence(q,r,features,features,0,0)
    check(initial["fit"]["anchor_count"]==5 and initial["fit"]["translation"]==[7.,-3.],"Known translation changed")
    np.testing.assert_array_equal(initial["normalized_innovation"],[0.,0.]);check(initial["score_or_gate_produced"] is False and initial["fit"]["physical_camera_motion_established"] is False,"Prototype overstated semantics")
    for seed in range(24):
        rng=np.random.RandomState(seed);qm=q.copy();rm=r.copy();qf=features.copy();rf=features.copy()
        qm[0]=boxes(rng.randn(1,2)*100)[0];rm[0]=boxes(rng.randn(1,2)*100)[0];qf[0]=rng.randn(6)*10**(seed%6);rf[0]=rng.randn(6)*10**(seed%5)
        changed=m.candidate_translation_evidence(qm,rm,qf,rf,0,0);equal(initial["fit"],changed["fit"])
        check(changed["normalized_innovation"]!=initial["normalized_innovation"],"Scored endpoint residual incorrectly ignored endpoints")
    checks.append("24_legal_endpoint_value_mutations_preserve_fit_support_matches_but_change_candidate_residual")
    alias=m.candidate_translation_evidence(q,r,features,features,0,0,extra_query_exclusions=(1,),extra_reference_exclusions=(1,))
    qm=q.copy();qf=features.copy();qm[1]+=100;qf[1]=np.arange(6)
    equal(alias["fit"],m.candidate_translation_evidence(qm,r,qf,features,0,0,extra_query_exclusions=(1,),extra_reference_exclusions=(1,))["fit"])
    retained=m.candidate_translation_evidence(q,r,features,features,0,0)["fit"]["anchors"]
    check(any(a["query_row"]==1 and a["reference_row"]==1 for a in retained),"Row exclusion falsely established physical alias closure")
    checks.append("fixed_extra_alias_exclusion_mutation_invariance_and_nonexcluded_physical_alias_limitation")
    qf=np.eye(6)[:5];rf=np.eye(6)[:5];qf[0]=np.eye(6)[0];qf[1]=.8*np.eye(6)[0]+.6*np.eye(6)[1];rf[0]=np.eye(6)[5];rf[1]=np.eye(6)[0]
    rb=boxes([[i*10,0] for i in range(5)]);qb=rb.copy()
    correct=m.common_translation(qb,rb,qf,rf,excluded_query_rows=(0,),excluded_reference_rows=(0,))
    full=m.common_translation(qb,rb,qf,rf);post=[a for a in full["anchors"] if a["query_row"]!=0 and a["reference_row"]!=0]
    check(correct["anchor_count"]==4 and len(post)==3 and (1,1) in {(a["query_row"],a["reference_row"]) for a in correct["anchors"]},"Before-matching exclusion failed reciprocal competition counterexample")
    checks.append("explicit_competition_counterexample_proves_match_then_delete_is_different")
    return checks


def round_two(m):
    checks=[];rng=np.random.RandomState(141024);tested=0
    for count in range(0,9):
        for repetition in range(3):
            qf=rng.randn(count,12);rf=rng.randn(count+1,12);qb=boxes(rng.randn(count,2)*10);rb=boxes(rng.randn(count+1,2)*10)
            if count:qf[-1]=0
            if count>3:rf[:count-1]=qf[:count-1]
            qe=() if count<2 else (1,);re=() if count<2 else (0,)
            actual=m.common_translation(qb,rb,qf,rf,excluded_query_rows=qe,excluded_reference_rows=re);pairs,tr,mad,shift=oracle(qb,rb,qf,rf,qe,re)
            check([(a["query_row"],a["reference_row"]) for a in actual["anchors"]]==pairs,"Independent scalar reciprocal match oracle differs")
            check(actual["available"]==(tr is not None),"Numerical support availability differs")
            if tr is not None:
                np.testing.assert_allclose(actual["translation"],tr,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(actual["residual_mad_xy"],mad,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(actual["max_leave_one_anchor_shift"],shift,atol=1e-12,rtol=1e-12)
            assert_finite(actual);tested+=1
    checks.append("27_empty_zero_raw_and_random_frames_match_independent_scalar_cosine_median_MAD_leaveone_oracle")
    f=np.eye(5);r=boxes([[0,0],[3,3],[6,6],[9,9],[12,12]]);q=r+np.array([4,7,4,7]);f[1]=f[0]
    tied=m.common_translation(q,r,f,f);check(tied["anchor_count"]==3 and all(a["query_row"]>=2 for a in tied["anchors"]),"Exact appearance tie received favourable row-order match")
    zeros=m.common_translation(q,r,np.zeros_like(f),np.zeros_like(f));check(not zeros["available"] and zeros["anchor_count"]==0 and zeros["translation"] is None,"Zero or unsupported rows became zero-motion fallback")
    one=m.common_translation(r[:1],r[:1],np.ones((1,2)),np.ones((1,2)));check(one["anchors"][0]["query_margin"] is None and one["anchors"][0]["reference_margin"] is None,"No competitor became artificial certainty")
    checks.append("strict_ties_zero_rows_empty_support_and_missing_competitor_margin_remain_explicit")
    f=np.eye(6);r=boxes([[k*10,k*2] for k in range(6)]);q=r+np.array([3,-4,3,-4]);base=m.common_translation(q,r,f,f,excluded_query_rows=(0,),excluded_reference_rows=(0,))
    pq=np.array([3,1,5,0,4,2]);pr=np.array([2,0,4,5,1,3]);qp=int(np.flatnonzero(pq==0)[0]);rp=int(np.flatnonzero(pr==0)[0]);permuted=m.common_translation(q[pq],r[pr],f[pq],f[pr],excluded_query_rows=(qp,),excluded_reference_rows=(rp,))
    check({(pq[a["query_row"]],pr[a["reference_row"]]) for a in permuted["anchors"]}=={(a["query_row"],a["reference_row"]) for a in base["anchors"]},"Row permutation changed physical anchor matching")
    for key in ("translation","residual_mad_xy","max_leave_one_anchor_shift"):np.testing.assert_allclose(permuted[key],base[key],atol=0,rtol=0)
    checks.append("independent_query_reference_permutations_preserve_translation_MAD_leaveone_and_mapped_anchors")
    return checks


def round_three(m):
    checks=[];r=boxes(np.zeros((5,2)));q=boxes([[5,-2],[5,-2],[5,-2],[1000,300],[-500,-600]]);f=np.eye(5)
    fit=m.common_translation(q,r,f,f);np.testing.assert_array_equal(fit["translation"],[5,-2]);np.testing.assert_array_equal(fit["residual_mad_xy"],[0,0]);assert_finite(fit)
    coherent=m.common_translation(boxes(np.tile([80,-30],(5,1))),r,f,f);check(coherent["translation"]==[80.,-30.] and coherent["residual_mad_xy"]==[0.,0.] and coherent["max_leave_one_anchor_shift"]==0 and coherent["physical_camera_motion_established"] is False,"Coherent object movement was labelled camera truth")
    checks.append("strict_majority_outlier_robustness_and_zeroMAD_common_objectmotion_counterexample")
    for bad in ((True,),(-1,),(5,),(.5,)):
        reject(m.common_translation,q,r,f,f,excluded_query_rows=bad)
    for minimum in (True,0,2,2.5):reject(m.common_translation,q,r,f,f,minimum_anchors=minimum)
    reject(m.candidate_translation_evidence,q,r,f,f,True,0);reject(m.candidate_translation_evidence,q,r,f,f,0,99)
    for badbox in (np.full((5,4),np.nan),np.zeros((5,4)),np.array([[-1e308,0,1e308,1]]*5)):
        reject(m.common_translation,badbox,r,f,f)
    reject(m.common_translation,q,r,f,np.ones((5,6)))
    checks.append("boolean_fractional_range_shape_nonfinite_and_overflow_box_inputs_fail_closed")
    huge=np.tile([9e307,9e307,1e308,1e308],(3,1));large=m.common_translation(huge,huge,np.eye(3),np.eye(3));check(large["available"] and large["translation"]==[0,0] and large["max_leave_one_anchor_shift"]==0,"Huge finite equal boxes overflowed midpoint")
    assert_finite(large)
    positive=np.tile([9e307,9e307,1e308,1e308],(3,1));negative=np.tile([-1e308,-1e308,-9e307,-9e307],(3,1))
    reject(m.common_translation,positive,negative,np.eye(3),np.eye(3))
    tiny=np.nextafter(0.,1.);small=np.tile([0,0,tiny,tiny],(4,1));out=m.candidate_translation_evidence(small,small,np.eye(4),np.eye(4),0,0);assert_finite(out);check(out["normalized_innovation"]==[0,0],"Small finite box residual is not finite zero")
    checks.append("large_equal_finite_midpoints_and_subnormal_sizes_safe_unrepresentable_displacements_rejected")
    return checks


def run(number):
    check(sha(MODULE)==EXPECTED_SHA,"Subject module changed")
    spec=importlib.util.spec_from_file_location("independent_common_motion_subject",MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    checks={1:round_one,2:round_two,3:round_three}[number](m)
    check(sha(MODULE)==EXPECTED_SHA,"Subject module changed during test")
    return dict(status="PASS_INDEPENDENT_ENDPOINT_EXCLUDED_SCENE_MOTION_SYNTHETIC_AUDIT",round=number,checks=checks,module_sha256=EXPECTED_SHA,script_sha256=sha(__file__),real_cache_or_identity_or_new_OOF_evaluation_opened=False,physical_camera_identification_established=False,full_population_compute_or_tracking_effectiveness_audited=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--round",type=int,choices=(1,2,3),required=True);p.add_argument("--out",type=Path,required=True);args=p.parse_args();result=run(args.round)
    with args.out.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(result,indent=2),flush=True)
