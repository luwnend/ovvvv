"""Independent CPU synthetic preflight; never opens real cache/GT/model."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SUBJECT=ROOT/"tmp_analysis/diagnose_spacetime_candidate_features_20261024.py"


def import_path(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def features(query,reference):
    def norm(x):return x/np.maximum(np.sqrt((x*x).sum(1))[:,None],1e-12)
    q,m=query["boxes"],reference["boxes"]
    raw=norm(query["raw"])@norm(reference["raw"]).T
    semantic=norm(query["semantic"])@norm(reference["semantic"]).T
    absolute=norm(q)@norm(m).T
    qc,mc=(q[:,:2]+q[:,2:])/2,(m[:,:2]+m[:,2:])/2
    qs,ms=q[:,2:]-q[:,:2],m[:,2:]-m[:,:2]
    scale=np.sqrt(qs[:,None,:]*ms[None,:,:])
    displacement=-np.sqrt((((qc[:,None,:]-mc[None,:,:])/scale)**2).sum(2))
    lo=np.maximum(q[:,None,:2],m[None,:,:2]);hi=np.minimum(q[:,None,2:],m[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(2)
    overlap=intersection/(qs.prod(1)[:,None]+ms.prod(1)[None,:]-intersection)
    innovation=np.full((len(q),len(m)),np.nan)
    for j in range(len(m)):
        t=reference["previous_times"][j]
        if np.isfinite(t):
            prev=reference["previous_boxes"][j];pc=(prev[:2]+prev[2:])/2
            velocity=(mc[j]-pc)/(reference["original_frame_index"]-t)
            predicted=mc[j]+velocity*(query["original_frame_index"]-reference["original_frame_index"])
            innovation[:,j]=-np.sqrt((((qc-predicted)/scale[:,j,:])**2).sum(1))
    return np.stack((raw,absolute,overlap,displacement,innovation,semantic),2).reshape(-1,6)


def independent_ranking(arrays,truth,selected,names):
    """Literal per-query/per-pair enumeration; no producer statistics helpers."""
    groups={}
    for row in selected:
        key=tuple(int(arrays[field][row]) for field in ("video_id","query_image_id","query_row"))
        groups.setdefault(key,[]).append(int(row))
    coverage=dict(nonempty_known_queries=len(groups),positive_available_queries=0,negative_available_queries=0,
                  positive_and_negative_available_queries=0,no_positive_queries=0,no_negative_queries=0,
                  CV_valid_known_candidates=len(selected))
    targets={key:dict(queries=0,sources=set(),videos=set(),positive_candidates=0,negative_candidates=0,
                     top1={name:{"raw_wrong_feature_wrong":0,"raw_wrong_feature_correct":0,
                                 "raw_correct_feature_wrong":0,"raw_correct_feature_correct":0} for name in names})
             for key in ("all_nonempty_known_queries","positive_and_negative_available_queries")}
    matrices={name:[[0]*3 for _ in range(3)] for name in names}
    for key,rows in groups.items():
        positives=[row for row in rows if truth[row]==1];negatives=[row for row in rows if truth[row]==0]
        assert len(positives)+len(negatives)==len(rows)
        for prefix,present in (("positive",bool(positives)),("negative",bool(negatives))):
            coverage[prefix+"_available_queries"]+=int(present)
            coverage["no_"+prefix+"_queries"]+=int(not present)
        both=bool(positives and negatives);coverage["positive_and_negative_available_queries"]+=int(both)
        winners=[max(rows,key=lambda row:(float(arrays["scores"][row,column]),-int(arrays["reference_row"][row])))
                 for column in range(6)]
        correct=[truth[row]==1 for row in winners]
        for target_name in ("all_nonempty_known_queries",)+( ("positive_and_negative_available_queries",) if both else () ):
            target=targets[target_name];target["queries"]+=1
            target["sources"].add(int(arrays["source_index"][rows[0]]));target["videos"].add(key[0])
            target["positive_candidates"]+=len(positives);target["negative_candidates"]+=len(negatives)
            for column,name in enumerate(names):
                label="raw_"+("correct" if correct[0] else "wrong")+"_feature_"+("correct" if correct[column] else "wrong")
                target["top1"][name][label]+=1
        for p in positives:
            for n in negatives:
                signs=[]
                for column in range(6):
                    a,b=arrays["scores"][p,column],arrays["scores"][n,column]
                    signs.append(0 if a<b else 2 if a>b else 1)
                for column,name in enumerate(names):matrices[name][signs[0]][signs[column]]+=1
    for target in targets.values():
        for field in ("sources","videos"):target[field]=len(target[field])
        for counts in target["top1"].values():
            counts.update(raw_correct=counts["raw_correct_feature_wrong"]+counts["raw_correct_feature_correct"],
                          feature_correct=counts["raw_wrong_feature_correct"]+counts["raw_correct_feature_correct"],
                          repair_raw_error=counts["raw_wrong_feature_correct"],break_raw_correct=counts["raw_correct_feature_wrong"])
    orders={}
    for name,table in matrices.items():
        orders[name]=dict(positive_negative_comparisons=sum(map(sum,table)),feature_loss=sum(row[0] for row in table),
                         feature_tie=sum(row[1] for row in table),feature_win=sum(row[2] for row in table),
                         raw_to_feature_loss_tie_win_matrix=table,repair_raw_loss_to_win=table[0][2],break_raw_win_to_loss=table[2][0])
    coverage["positive_available_fraction"]=coverage["positive_available_queries"]/len(groups) if groups else None
    return dict(coverage=coverage,subsets=targets,positive_negative_pair_order=orders)


def main():
    subject=import_path(SUBJECT,"independent_subject_spacetime")
    geometry=import_path(subject.GEOMETRY,"independent_geometry_spacetime")
    script_sha=subject.sha(SUBJECT);protocol_sha=subject.sha(subject.PROTOCOL)
    assert subject.PROTOCOL_SHA==protocol_sha
    rng=np.random.RandomState(20261024)
    times=[0,4,20,29,50];counts=[2,2,3,0,1];raw_frames=[]
    for step,(time,count) in enumerate(zip(times,counts)):
        boxes=np.asarray([[10+time+40*k,10,20+time+40*k,30] for k in range(count)],float).reshape(count,4)
        raw_frames.append(dict(video_id=9,image_id=70+step,sampled_step=step,original_frame_index=time,boxes=boxes,
                               raw=rng.randn(count,256),semantic=rng.randn(count,512),
                               released_ids=np.asarray([0]+[-1]*(count-1),np.int64) if count else np.zeros(0,np.int64),
                               predicted_categories=np.arange(count,dtype=np.int32)%2+1))
    snapshot=copy.deepcopy(raw_frames)
    records=[dict(video_id=9,frame_id=k) for k in range(len(raw_frames))]
    arrays,receipts=subject.generate(records,["S"],{"9":"S"},lambda row:copy.deepcopy(raw_frames[row["frame_id"]]),geometry)
    assert [r["candidate_rows"] for r in receipts]==[0,4,6,0,0]
    assert arrays["scores"].shape==(10,6)
    # Manual independent past-position map. Never query producer history helper.
    history={};independent=[];expected_cv=[]
    for frame in raw_frames:
        frame["previous_boxes"]=np.full((len(frame["boxes"]),4),np.nan)
        frame["previous_times"]=np.full(len(frame["boxes"]),np.nan)
        for j,k in enumerate(frame["released_ids"]):
            if k>=0 and int(k) in history:
                frame["previous_boxes"][j],frame["previous_times"][j]=history[int(k)]
        if independent:
            previous=raw_frames[len(independent)-1]
        independent.append(frame)
        for j,k in enumerate(frame["released_ids"]):
            if k>=0:history[int(k)]=(frame["boxes"][j].copy(),frame["original_frame_index"])
    expected=np.concatenate([features(raw_frames[k],raw_frames[k-1]) for k in range(1,len(raw_frames))])
    np.testing.assert_allclose(arrays["scores"],expected,rtol=1e-13,atol=1e-13,equal_nan=True)
    assert arrays["query_row"].tolist()==[0,0,1,1,0,0,1,1,2,2]
    assert arrays["reference_row"].tolist()==[0,1,0,1,0,1,0,1,0,1]
    assert arrays["cv_valid"].tolist()==[False]*4+[True,False]*3
    for a,b in zip(raw_frames,snapshot):
        for field in ("boxes","raw","semantic","released_ids","predicted_categories"):
            np.testing.assert_array_equal(a[field],b[field])

    # Duplicate-ID history must not be used by a later frame, and id=-1 never
    # becomes a common history key. Next unique frame re-establishes history.
    history={}
    a=copy.deepcopy(snapshot[0]);a["released_ids"]=np.array([4,4])
    subject.attach_history(a,history)
    b=copy.deepcopy(snapshot[1]);b["released_ids"]=np.array([4,-1])
    subject.attach_history(b,history);assert np.isnan(b["previous_times"]).all()
    c=copy.deepcopy(snapshot[2]);c["released_ids"]=np.array([4,-1,-1])
    subject.attach_history(c,history)
    assert c["previous_times"][0]==4 and np.isnan(c["previous_times"][1:]).all()
    history={}
    unique=copy.deepcopy(snapshot[0]);unique["released_ids"]=np.array([4,-1]);subject.attach_history(unique,history)
    ambiguous=copy.deepcopy(snapshot[1]);ambiguous["released_ids"]=np.array([4,4]);subject.attach_history(ambiguous,history)
    assert np.isnan(ambiguous["previous_times"]).all(),"Current duplicated occurrences inherited old CV"
    recovered=copy.deepcopy(snapshot[2]);recovered["released_ids"]=np.array([4,-1,-1]);subject.attach_history(recovered,history)
    assert np.isnan(recovered["previous_times"]).all(),"History jumped across an ambiguous occurrence"
    next_unique=copy.deepcopy(snapshot[4]);next_unique["released_ids"]=np.array([4]);subject.attach_history(next_unique,history)
    assert next_unique["previous_times"].tolist()==[20.]

    for n in range(1,31):
        labels=rng.randint(0,2,n);scores=rng.randint(-2,3,n).astype(float)
        positive=scores[labels==1];negative=scores[labels==0]
        brute=None if not len(positive) or not len(negative) else float(np.mean([(a>b)+.5*(a==b) for a in positive for b in negative]))
        result=subject.auc(labels,scores)
        assert result is brute is None or abs(result-brute)<1e-14

    # Independent ranks include missing positive/negative groups, unknown labels,
    # score ties, permuted storage order, two videos and differing source scopes.
    matrices_seen=np.zeros((3,3),np.int64)
    for trial in range(80):
        sizes=rng.randint(1,7,8);n=int(sizes.sum());q=np.repeat(np.arange(8),sizes)
        arrays=dict(video_id=q//4+10,query_image_id=q+100,query_row=q%2,source_index=q//3,
                    reference_row=np.concatenate([rng.permutation(size) for size in sizes]),
                    cv_valid=rng.rand(n)>.2,same_predicted_category=rng.rand(n)>.35,
                    scores=rng.randint(-2,3,(n,6)).astype(float))
        truth=rng.choice([-1,0,1],n,p=[.15,.45,.40])
        for scope in (np.ones(n,bool),arrays["same_predicted_category"]):
            selected=np.flatnonzero(scope&arrays["cv_valid"]&(truth>=0));rng.shuffle(selected)
            actual=subject.query_ranking_statistics(arrays,{"identity_label":truth},selected)
            expected=independent_ranking(arrays,truth,selected,subject.FEATURES)
            for key,value in expected.items():assert actual[key]==value,(trial,key)
            matrices_seen+=np.asarray(actual["positive_negative_pair_order"][subject.FEATURES[1]]["raw_to_feature_loss_tie_win_matrix"])
    assert (matrices_seen>0).all(),"Synthetic checks failed to exercise all 3x3 order transitions"
    # Explicit adverse tie: lowest immutable row is negative despite array order.
    tied=dict(video_id=np.ones(2,int),query_image_id=np.ones(2,int),query_row=np.zeros(2,int),
              source_index=np.zeros(2,int),reference_row=np.array([9,2]),cv_valid=np.ones(2,bool),scores=np.zeros((2,6)))
    result=subject.query_ranking_statistics(tied,{"identity_label":np.array([1,0])},np.array([0,1]))
    assert all(v["feature_correct"]==0 for v in result["subsets"]["positive_and_negative_available_queries"]["top1"].values())
    empty=subject.query_ranking_statistics(tied,{"identity_label":np.array([1,0])},np.array([],int))
    assert empty["coverage"]["positive_available_fraction"] is None and empty["coverage"]["nonempty_known_queries"]==0

    # Required seal check is tested before any real helper or source is opened.
    with tempfile.TemporaryDirectory(prefix="independent_spacetime_seal_") as temp:
        try:subject.join_fit_identity(Path(temp),{}, {}, {})
        except FileNotFoundError:pass
        else:raise AssertionError("Identity join accepted an unsealed candidate set")
    assert script_sha==subject.sha(SUBJECT) and protocol_sha==subject.sha(subject.PROTOCOL)
    receipt=dict(status="PASS_INDEPENDENT_SPACETIME_CANDIDATE_FEATURES_IMPLEMENTATION_PREFLIGHT",
                 script_sha256=script_sha,protocol_sha256=protocol_sha,
                 geometry_module_sha256=subject.sha(subject.GEOMETRY),selection_protocol_sha256=subject.sha(subject.SELECTION),
                 checks=["independent_all_QxM_float64_feature_and_irregular_CV_recomputation_unequal_counts",
                         "original_row_mapping_unknown_ID_inclusion_empty_previous_not_skipped",
                         "current_duplicate_and_future_ID_history_invalidation_unique_recovery_without_jump",
                         "30_bruteforce_tie_AUC_and_absent_class_cases",
                         "160_independently_enumerated_common_CV_known_ranking_scopes_all_3x3_order_cells",
                         "immutable_reference_row_tie_wrong_GT_control_empty_and_no_positive_query_coverage",
                         "unsealed_identity_join_rejected_before_any_payload"],
                 real_cache_opened=False,real_supervision_opened=False,real_model_opened=False,GPU=False,
                 script_and_protocol_unchanged=True,limits=["Synthetic implementation preflight only; no efficacy evidence"])
    destination=ROOT/"tmp_analysis/spacetime_candidate_features_independent_preflight_20261024.json"
    with destination.open("x") as stream:stream.write(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
