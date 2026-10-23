#!/usr/bin/env python3
"""Seal full adjacent-frame GT-free candidate features, then FIT identity AUC.

No learned weight, matching, GT-based candidate selection or class correctness.
Real execution is root-owned after independent implementation preflight.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
RUN=STAGE/"full_collect_v1"
SUPPORT=STAGE/"support_v1"
PROTOCOL=ROOT/"tmp_analysis/spacetime_candidate_features_protocol_20261024.json"
PROTOCOL_SHA="d38beac2839a6473ff6a973e41a7c194febb818e40e433df63793e3349f086a1"
SELECTION=ROOT/"tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
GEOMETRY=ROOT/"tmp_analysis/relative_track_geometry.py"
HELPER=ROOT/"tmp_analysis/build_fit_identity_edge_support.py"
PREFLIGHT=ROOT/"tmp_analysis/spacetime_candidate_features_independent_preflight_20261024.json"
FEATURES=("raw_appearance_cosine","absolute_xyxy_cosine","box_iou","negative_normalized_displacement_norm",
          "negative_normalized_CV_innovation_norm","semantic_embedding_cosine")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):result.update(block)
    return result.hexdigest()


def write(path,value):
    with Path(path).open("x") as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def normalized(values):
    values=np.asarray(values,dtype=np.float64)
    require(values.ndim==2 and np.isfinite(values).all(),"Expected finite feature matrix")
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)


def iou_matrix(query,reference):
    lo=np.maximum(query[:,None,:2],reference[None,:,:2]);hi=np.minimum(query[:,None,2:],reference[None,:,2:])
    area=np.prod(np.maximum(hi-lo,0),axis=2)
    qarea=np.prod(query[:,2:]-query[:,:2],axis=1);rarea=np.prod(reference[:,2:]-reference[:,:2],axis=1)
    return area/(qarea[:,None]+rarea[None,:]-area)


def attach_history(frame,history):
    """Past-only released-ID locator; it never labels or filters candidates."""
    n=len(frame["boxes"]);previous=np.full((n,4),np.nan);times=np.full(n,np.nan)
    counts=Counter(int(identity) for identity in frame["released_ids"] if identity>=0)
    for row,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity<0 or counts[identity]>1:continue
        old=history.get(identity)
        if old is not None:
            box,t=old
            require(t<frame["original_frame_index"],"History must precede reference frame")
            previous[row]=box;times[row]=t
    frame["previous_boxes"]=previous;frame["previous_times"]=times
    for row,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity<0:continue
        if counts[identity]>1:
            # A duplicate-ID frame has no uniquely observed position. Do not
            # silently choose a row or jump across that ambiguous occurrence.
            history[identity]=None
        else:history[identity]=(frame["boxes"][row].copy(),frame["original_frame_index"])
    return frame


def candidate_block(query,reference,source_index,geometry):
    nq,nm=len(query["boxes"]),len(reference["boxes"])
    require(query["video_id"]==reference["video_id"] and query["sampled_step"]==reference["sampled_step"]+1 and
            query["original_frame_index"]>reference["original_frame_index"],"Nonadjacent/invalid original timeline")
    geom=geometry.pair_geometry(query["boxes"],reference["boxes"],query["original_frame_index"],
        np.full(nm,reference["original_frame_index"]),reference["previous_boxes"],reference["previous_times"])
    raw=normalized(query["raw"])@normalized(reference["raw"]).T
    absolute=normalized(query["boxes"])@normalized(reference["boxes"]).T
    semantic=normalized(query["semantic"])@normalized(reference["semantic"]).T
    overlap=iou_matrix(query["boxes"],reference["boxes"])
    displacement=-np.linalg.norm(geom["normalized_displacement"],axis=2)
    innovation=-np.linalg.norm(geom["normalized_CV_innovation"],axis=2)
    scores=np.stack((raw,absolute,overlap,displacement,innovation,semantic),axis=2).reshape(-1,len(FEATURES))
    cv=np.broadcast_to(geom["velocity_history_available"],(nq,nm)).reshape(-1).copy()
    require(np.isfinite(scores[:,[0,1,2,3,5]]).all() and np.isfinite(scores[cv,4]).all() and np.isnan(scores[~cv,4]).all(),"Unexpected score/missing-history state")
    qr=np.repeat(np.arange(nq,dtype=np.int32),nm);rr=np.tile(np.arange(nm,dtype=np.int32),nq)
    n=len(qr)
    arrays=dict(scores=scores,cv_valid=cv,same_predicted_category=(query["predicted_categories"][qr]==reference["predicted_categories"][rr]),
        source_index=np.full(n,source_index,np.int16),video_id=np.full(n,query["video_id"],np.int32),
        query_image_id=np.full(n,query["image_id"],np.int32),reference_image_id=np.full(n,reference["image_id"],np.int32),
        query_row=qr,reference_row=rr,query_original_frame=np.full(n,query["original_frame_index"],np.int64),
        reference_original_frame=np.full(n,reference["original_frame_index"],np.int64),
        reference_released_id=reference["released_ids"][rr].astype(np.int64),
        history_original_frame=reference["previous_times"][rr].copy())
    return arrays


def auc(labels,scores):
    """Mann-Whitney AUC, half credit for exact ties; no finite-value selection."""
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=np.float64)
    require(labels.shape==scores.shape and np.all((labels==0)|(labels==1)) and np.isfinite(scores).all(),"AUC input/denominator invalid")
    positive=int(np.count_nonzero(labels==1));negative=len(labels)-positive
    if not positive or not negative:return None
    order=np.argsort(scores,kind="stable");values=scores[order];truth=labels[order]
    total=0.;negative_before=0;start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[end]==values[start]:end+=1
        npair=int(np.count_nonzero(truth[start:end]==1));nnegative=end-start-npair
        total+=npair*(negative_before+.5*nnegative);negative_before+=nnegative;start=end
    return float(total/(positive*negative))


def query_ranking_statistics(arrays,joined,selected):
    """Conditional known-subset ranks; one common CV mask for all six scores.

    No candidate feature or score depends on identities. Labels are used here,
    after sealing, only to score precomputed candidates and report coverage.
    """
    truth=joined["identity_label"]
    groups=defaultdict(list)
    for row in selected:
        groups[(int(arrays["video_id"][row]),int(arrays["query_image_id"][row]),int(arrays["query_row"][row]))].append(row)
    subsets={name:dict(queries=0,source_indexes=set(),videos=set(),positive_candidates=0,negative_candidates=0,
        top1={feature:dict(raw_wrong_feature_wrong=0,raw_wrong_feature_correct=0,
            raw_correct_feature_wrong=0,raw_correct_feature_correct=0) for feature in FEATURES})
        for name in ("all_nonempty_known_queries","positive_and_negative_available_queries")}
    orders={feature:np.zeros((3,3),np.int64) for feature in FEATURES}
    coverage=dict(nonempty_known_queries=len(groups),positive_available_queries=0,negative_available_queries=0,
        positive_and_negative_available_queries=0,no_positive_queries=0,no_negative_queries=0,
        CV_valid_known_candidates=len(selected))
    for rows in groups.values():
        rows=np.asarray(rows,np.int64);labels=truth[rows];positive=labels==1;negative=labels==0
        require(np.all(positive|negative) and arrays["cv_valid"][rows].all(),"Ranking denominator is not common CV-valid known candidates")
        require(len(set(arrays["reference_row"][rows].tolist()))==len(rows),"Duplicate immutable reference row within query")
        has_positive=bool(positive.any());has_negative=bool(negative.any())
        coverage["positive_available_queries"]+=int(has_positive);coverage["negative_available_queries"]+=int(has_negative)
        coverage["positive_and_negative_available_queries"]+=int(has_positive and has_negative)
        coverage["no_positive_queries"]+=int(not has_positive);coverage["no_negative_queries"]+=int(not has_negative)
        scores=arrays["scores"][rows];correct=[]
        for column in range(len(FEATURES)):
            # Stable immutable dump_row tie resolution is declared before labels.
            winner=np.lexsort((arrays["reference_row"][rows],-scores[:,column]))[0]
            correct.append(bool(labels[winner]==1))
        for name in ("all_nonempty_known_queries",)+( ("positive_and_negative_available_queries",) if has_positive and has_negative else () ):
            target=subsets[name];target["queries"]+=1
            target["source_indexes"].add(int(arrays["source_index"][rows[0]]));target["videos"].add(int(arrays["video_id"][rows[0]]))
            target["positive_candidates"]+=int(positive.sum());target["negative_candidates"]+=int(negative.sum())
            for feature,good in zip(FEATURES,correct):
                target["top1"][feature]["raw_"+("correct" if correct[0] else "wrong")+"_feature_"+("correct" if good else "wrong")]+=1
        if has_positive and has_negative:
            signs=np.sign(scores[positive,None,:]-scores[None,negative,:]).astype(np.int8).reshape(-1,len(FEATURES))+1
            for column,feature in enumerate(FEATURES):
                orders[feature]+=np.bincount(3*signs[:,0]+signs[:,column],minlength=9).reshape(3,3)
    for target in subsets.values():
        target["sources"]=len(target.pop("source_indexes"));target["videos"]=len(target["videos"])
        for counts in target["top1"].values():
            counts["raw_correct"]=counts["raw_correct_feature_wrong"]+counts["raw_correct_feature_correct"]
            counts["feature_correct"]=counts["raw_wrong_feature_correct"]+counts["raw_correct_feature_correct"]
            counts["repair_raw_error"]=counts["raw_wrong_feature_correct"]
            counts["break_raw_correct"]=counts["raw_correct_feature_wrong"]
    pair_orders={}
    for feature,matrix in orders.items():
        pair_orders[feature]=dict(positive_negative_comparisons=int(matrix.sum()),
            feature_loss=int(matrix[:,0].sum()),feature_tie=int(matrix[:,1].sum()),feature_win=int(matrix[:,2].sum()),
            raw_to_feature_loss_tie_win_matrix=matrix.tolist(),repair_raw_loss_to_win=int(matrix[0,2]),
            break_raw_win_to_loss=int(matrix[2,0]))
    coverage["positive_available_fraction"]=coverage["positive_available_queries"]/len(groups) if groups else None
    return dict(coverage=coverage,subsets=subsets,positive_negative_pair_order=pair_orders,
        pair_order_matrix_axis=["loss","tie","win"],top1_tie_rule="highest score then lowest immutable reference dump_row",
        primary_top1_subset="positive_and_negative_available_queries",
        interpretation="Conditional ranking within CV-valid known-identity reference subset only; complete sealed candidate set still includes unknowns. No deployment top1 or assignment benefit is established.")


def score_statistics(arrays,joined,mask,include_cv):
    truth=joined["identity_label"];known=mask&(truth>=0);selected=np.flatnonzero(known)
    positives=truth[selected]==1
    ids=set()
    for side in ("query","reference"):
        ids.update(zip(arrays["video_id"][selected].tolist(),joined[side+"_gt_identity"][selected].tolist()))
    result=dict(candidates=int(np.count_nonzero(mask)),known_candidates=len(selected),
        positive_pairs=int(positives.sum()),negative_pairs=int(len(selected)-positives.sum()),
        unknown_candidates=int(np.count_nonzero(mask&(truth<0))),
        sources=int(len(set(arrays["source_index"][selected].tolist()))),videos=int(len(set(arrays["video_id"][selected].tolist()))),
        official_GT_trajectories=len(ids),
        query_unique_observations=len(set(zip(arrays["query_image_id"][selected].tolist(),arrays["query_row"][selected].tolist()))),
        reference_unique_observations=len(set(zip(arrays["reference_image_id"][selected].tolist(),arrays["reference_row"][selected].tolist()))),
        CV_known_candidates=int(np.count_nonzero(known&arrays["cv_valid"])),
        auc={})
    columns=range(len(FEATURES)) if include_cv else (0,1,2,3,5)
    for column in columns:result["auc"][FEATURES[column]]=auc(truth[selected],arrays["scores"][selected,column])
    if include_cv:result["paired_query_ranking"]=query_ranking_statistics(arrays,joined,selected)
    return result


def describe(arrays,joined,sources):
    result={}
    for name,hard in (("all_candidates",np.ones(len(arrays["scores"]),dtype=bool)),("same_predicted_category",arrays["same_predicted_category"])):
        scope={}
        for cvname,cvonly in (("all_known_population",False),("common_CV_covered_population",True)):
            mask=hard&(arrays["cv_valid"] if cvonly else True)
            overall=score_statistics(arrays,joined,mask,cvonly)
            per_source={source:score_statistics(arrays,joined,mask&(arrays["source_index"]==index),cvonly) for index,source in enumerate(sources)}
            macro={}
            for feature in overall["auc"]:
                values=[stats["auc"][feature] for stats in per_source.values() if stats["auc"][feature] is not None]
                macro[feature]=dict(mean=float(np.mean(values)) if values else None,supported_sources=len(values))
            scope[cvname]=dict(micro_union=overall,source_macro_AUROC=macro,per_source=per_source)
        result[name]=scope
    return result


def load_frame(record):
    with np.load(record["npz"],allow_pickle=False) as saved:
        require(all(int(saved[key])==record[field] for key,field in (("video_id","video_id"),("image_id","image_id"),("frame_id","frame_id"),("frame_index","frame_index"))),"Cached frame provenance changed")
        boxes=saved["det_bboxes"].astype(np.float64);raw=saved["track_feats"].astype(np.float64)
        semantic=saved["cls_feats"].astype(np.float64);ids=saved["track_ids"].copy();categories=saved["det_labels"].copy()+1
    n=len(boxes)
    require(boxes.shape==(n,5) and raw.shape==(n,256) and semantic.shape==(n,512) and ids.shape==categories.shape==(n,),"Retained row shape mismatch")
    require(n==record["observations"] and np.isfinite(boxes).all() and np.all(boxes[:,2:4]>boxes[:,:2]),"Dropped/invalid retained box row")
    return dict(video_id=record["video_id"],image_id=record["image_id"],sampled_step=record["frame_id"],
        original_frame_index=record["frame_index"],boxes=boxes[:,:4],raw=raw,semantic=semantic,
        released_ids=ids.astype(np.int64),predicted_categories=categories.astype(np.int32))


def generate(frames,sources,video_source,loader,geometry):
    """No supervision argument or file IO; complete chronological frame stream."""
    grouped=defaultdict(list)
    for record in frames:grouped[record["video_id"]].append(record)
    blocks=[];receipts=[]
    for video,records in sorted(grouped.items()):
        history={};previous=None
        records.sort(key=lambda row:row["frame_id"])
        require([r["frame_id"] for r in records]==list(range(len(records))),"Missing selected sampled frame")
        for record in records:
            frame=attach_history(loader(record),history)
            require(frame["video_id"]==video,"Loader changed video")
            count=0
            if previous is not None:
                block=candidate_block(frame,previous,sources.index(video_source[str(video)]),geometry)
                blocks.append(block);count=len(block["scores"])
            receipts.append(dict(video_id=video,image_id=frame["image_id"],sampled_step=frame["sampled_step"],
                original_frame_index=frame["original_frame_index"],query_rows=len(frame["boxes"]),
                reference_rows=0 if previous is None else len(previous["boxes"]),candidate_rows=count,
                adjacent_original_frame_gap=None if previous is None else frame["original_frame_index"]-previous["original_frame_index"],
                reference_history_original_frame_gaps=[] if previous is None else
                    (previous["original_frame_index"]-previous["previous_times"][np.isfinite(previous["previous_times"])]).astype(np.int64).tolist(),
                retained_released_unassigned_rows=int(np.count_nonzero(frame["released_ids"]<0))))
            previous=frame
    require(blocks,"No adjacent selected frame pairs")
    combined={key:np.concatenate([b[key] for b in blocks],axis=0) for key in blocks[0]}
    require(len(combined["scores"])==sum(r["candidate_rows"] for r in receipts),"Candidate/frame count mismatch")
    return combined,receipts


def authenticate():
    require(sha(PROTOCOL)==PROTOCOL_SHA,"Predeclared candidate protocol changed")
    protocol=json.loads(PROTOCOL.read_text())
    require(sha(SELECTION)==protocol["selection_protocol_sha256"] and sha(GEOMETRY)==protocol["geometry_module_sha256"] and
            sha(HELPER)==protocol["fit_identity_helper_sha256"],"Frozen selection/geometry/decoder changed")
    review=json.loads(PREFLIGHT.read_text())
    require(review["status"]=="PASS_INDEPENDENT_SPACETIME_CANDIDATE_FEATURES_IMPLEMENTATION_PREFLIGHT" and
            review["script_sha256"]==sha(__file__) and review["protocol_sha256"]==PROTOCOL_SHA,"Independent implementation review absent/stale")
    selection=json.loads(SELECTION.read_text());inputs=dict(selection["input_pins"])
    for path,digest in inputs.items():require(sha(path)==digest,"Fixed input changed: "+path)
    extra={PROTOCOL:PROTOCOL_SHA,SELECTION:protocol["selection_protocol_sha256"],GEOMETRY:protocol["geometry_module_sha256"],
        HELPER:protocol["fit_identity_helper_sha256"],SUPPORT/"independent_support_audit.json":protocol["support_audit_sha256"],
        SUPPORT/"support_receipt.json":protocol["support_receipt_sha256"],STAGE/"pair_context_v1/independent_pair_context_audit.json":protocol["pair_audit_sha256"],
        SUPPORT/"gt_free_index_manifest.json":protocol["gt_free_manifest_sha256"],PREFLIGHT:sha(PREFLIGHT),Path(__file__).resolve():sha(__file__)}
    for path,digest in extra.items():require(sha(path)==digest,"Pinned source changed");inputs[str(path)]=digest
    frames=[json.loads(line) for line in (RUN/"frame_execution.jsonl").open()]
    frames=[r for r in frames if r["video_id"] in selection["video_ids"]]
    require(len(frames)==697 and len(set(r["video_id"] for r in frames))==20 and len(selection["source_groups"])==12,"Fixed smoke population changed")
    inventory=json.loads((RUN/"dump_inventory.json").read_text())
    for frame in frames:
        path=Path(frame["npz"]);require(path.parent==RUN/"dump" and frame["npz_sha256"]==inventory[path.name]==sha(path),"Cached retained frame changed")
        inputs[str(path)]=inventory[path.name]
    return protocol,selection,frames,inputs


def join_fit_identity(out,arrays,selection,inputs):
    seal_path=out/"GT_free_candidates_sealed.json"
    seal=json.loads(seal_path.read_text())
    require(seal["status"]=="SEALED_ALL_GT_FREE_CANDIDATES_BEFORE_SUPERVISION" and
            sha(out/"candidate_features.npz")==seal["candidate_features_sha256"],"Supervision join before GT-free seal")
    # This is the first import of any helper capable of decoding supervised rows.
    helper=load_module(HELPER,"candidate_fit_identity_decoder")
    receipt=helper.project_object((SUPPORT/"support_receipt.json").read_text(),{"output_files"})
    manual=SUPPORT/"supervised_observations.jsonl"
    require(sha(manual)==receipt["output_files"][manual.name],"Fit manual identity source changed")
    roles_path=STAGE/"pair_context_v1/source_roles.json"
    audit=json.loads((STAGE/"pair_context_v1/independent_pair_context_audit.json").read_text())
    require(sha(roles_path)==audit["output_files_sha256"][roles_path.name],"Source roles changed")
    roles=json.loads(roles_path.read_text())["groups"]
    identities=helper.load_fit_identities(manual,roles)  # never decodes GT category/error
    manifest=json.loads((SUPPORT/"gt_free_index_manifest.json").read_text())
    index=SUPPORT/"gt_free_observations.jsonl"
    require(sha(index)==manifest["files"][index.name],"Immutable GT-free observation index changed")
    anchors={}
    with index.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row["video_id"] not in selection["video_ids"]:continue
            target=identities.get(row["observation_id"])
            if target is None:continue
            require(row["split"]=="fit" and all(row[k]==target[k] for k in ("video_id","source_group","local_track_id","image_id","sampled_step","original_frame_index","pred_category_id")),"Fit anchor/index mismatch")
            key=(row["image_id"],row["dump_row"]);require(key not in anchors,"Duplicate manual row anchor")
            anchors[key]=(row["video_id"],target["gt_track_id"])
    n=len(arrays["scores"]);gtq=np.full(n,-1,np.int64);gtr=np.full(n,-1,np.int64)
    for side,target in (("query",gtq),("reference",gtr)):
        for k,(image,row,video) in enumerate(zip(arrays[side+"_image_id"],arrays[side+"_row"],arrays["video_id"])):
            anchor=anchors.get((int(image),int(row)))
            if anchor is not None:
                require(anchor[0]==video,"GT identity video scope changed");target[k]=anchor[1]
    labels=np.full(n,-1,np.int8);known=(gtq>=0)&(gtr>=0);labels[known]=(gtq[known]==gtr[known]).astype(np.int8)
    for path in (manual,roles_path,index,seal_path):inputs[str(path)]=sha(path)
    return dict(identity_label=labels,query_gt_identity=gtq,reference_gt_identity=gtr),len(anchors)


def execute():
    started=time.monotonic();protocol,selection,frames,inputs=authenticate()
    out=Path(protocol["output_directory"]);require(not out.exists(),"Preserve existing diagnostic output")
    geometry=load_module(GEOMETRY,"candidate_relative_geometry")
    arrays,frame_receipts=generate(frames,selection["source_groups"],selection["video_source"],load_frame,geometry)
    out.mkdir();np.savez_compressed(out/"candidate_features.npz",**arrays)
    write(out/"frame_receipts.json",frame_receipts)
    write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_GT_FREE_CANDIDATES_BEFORE_SUPERVISION",
        candidate_features_sha256=sha(out/"candidate_features.npz"),frame_receipts_sha256=sha(out/"frame_receipts.json"),
        script_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,input_sha256=inputs,
        selected_frames=len(frame_receipts),candidates=len(arrays["scores"]),feature_names=list(FEATURES),
        GT_or_supervised_payload_opened=False,old_ID_used_as_truth_or_candidate_filter=False))
    joined,anchors=join_fit_identity(out,arrays,selection,inputs)
    np.savez_compressed(out/"fit_identity_labels.npz",**joined)
    results=describe(arrays,joined,selection["source_groups"])
    require(inputs=={path:sha(path) for path in inputs},"Protected input changed during diagnostic")
    seal=json.loads((out/"GT_free_candidates_sealed.json").read_text())
    require(sha(out/"candidate_features.npz")==seal["candidate_features_sha256"] and
            sha(out/"frame_receipts.json")==seal["frame_receipts_sha256"],"GT-free sealed output changed after identity join")
    def gap_statistics(gaps):
        return dict(count=len(gaps),minimum=min(gaps) if gaps else None,median=float(np.median(gaps)) if gaps else None,
            maximum=max(gaps) if gaps else None,counter={str(k):v for k,v in sorted(Counter(gaps).items())})
    sampling=dict(adjacent_original_frame_gap_per_frame_pair=gap_statistics([r["adjacent_original_frame_gap"] for r in frame_receipts if r["adjacent_original_frame_gap"] is not None]),
        history_original_frame_gap_per_reference_observation=gap_statistics([gap for r in frame_receipts for gap in r["reference_history_original_frame_gaps"]]))
    summary=dict(status="COMPLETE_FIXED_FIT_SPACETIME_CANDIDATE_DIAGNOSTIC_PENDING_INDEPENDENT_AUDIT",
        selected_sources=12,selected_videos=20,selected_frames=len(frame_receipts),retained_observations=sum(r["query_rows"] for r in frame_receipts),
        all_candidates=len(arrays["scores"]),eligible_fit_manual_anchor_observations=anchors,
        feature_names=list(FEATURES),results=results,sampling_gap_coverage=sampling,input_sha256=inputs,all_input_hashes_verified_before_after=True,
        output_files_sha256={name:sha(out/name) for name in ("candidate_features.npz","frame_receipts.json","GT_free_candidates_sealed.json","fit_identity_labels.npz")},
        candidates_sealed_before_fit_identity_join=True,unknown_is_negative=False,GT_category_or_error_decoded=False,
        old_ID_used_for_history_locator_only=True,no_ego_compensation=True,time_unit="original_frames",
        tune_test_class_holdout_correctness_opened=False,training_started=False,GPU_used=False,network_or_readout_changed=False,
        protocol_sha256=PROTOCOL_SHA,script_sha256=sha(__file__),interpretation=protocol["interpretation"],
        elapsed_seconds=time.monotonic()-started)
    write(out/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("status","all_candidates","selected_frames","elapsed_seconds")},indent=2),flush=True)


def fixture(frame,n=2,ids=None):
    boxes=np.array([[10+10*frame+40*k,10,20+10*frame+40*k,20] for k in range(n)],float).reshape(n,4)
    raw=np.zeros((n,256));sem=np.zeros((n,512))
    for k in range(n):raw[k,k]=1;sem[k,k]=1
    return dict(video_id=1,image_id=100+frame,sampled_step=frame,original_frame_index=[0,10,35,40][frame],boxes=boxes,
        raw=raw,semantic=sem,released_ids=np.array(ids if ids is not None else list(range(n)),np.int64),predicted_categories=np.ones(n,np.int32))


def synthetic(round_number):
    require(sha(GEOMETRY)=="c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a","Geometry unit changed")
    geometry=load_module(GEOMETRY,"synthetic_candidate_geometry");checks=[]
    if round_number==1:
        history={};a=attach_history(fixture(0,ids=[0,-1]),history);b=attach_history(fixture(1,ids=[0,-1]),history)
        c=attach_history(fixture(2,ids=[0,-1]),history);block=candidate_block(c,b,0,geometry)
        require(block["scores"].shape==(4,6) and block["reference_released_id"].tolist()==[0,-1,0,-1] and
            block["cv_valid"].tolist()==[True,False,True,False],"Full QxM/id=-1/history coverage differs")
        require(np.all(block["query_original_frame"]-block["reference_original_frame"]==25) and
            block["history_original_frame"][0]==0,"Irregular original-frame gap ignored")
        checks.append("full_QxM_including_unassigned_and_irregular_original_frame_CV")
        records=[dict(video_id=1,frame_id=k) for k in range(4)]
        def loader(r):return fixture(r["frame_id"],0 if r["frame_id"]==1 else 2)
        arrays,receipts=generate(records,["S"],{"1":"S"},loader,geometry)
        require([r["candidate_rows"] for r in receipts]==[0,0,0,4] and len(arrays["scores"])==4,"Empty previous frame was skipped")
        checks.append("complete_zero_candidate_frame_receipts_previous_empty_not_skipped")
        history={};attach_history(fixture(0,ids=[0,0]),history);b=attach_history(fixture(1,ids=[0,-1]),history)
        require(np.isnan(b["previous_times"]).all(),"Ambiguous/no-ID history invented")
        c=attach_history(fixture(2,ids=[0,-1]),history)
        require(c["previous_times"][0]==10 and np.isnan(c["previous_times"][1]),"Nearest unique past history incorrect")
        history={};attach_history(fixture(0),history);ambiguous=attach_history(fixture(1,ids=[0,0]),history)
        require(np.isnan(ambiguous["previous_times"]).all(),"Current duplicate rows inherited ambiguous CV history")
        next_unique=attach_history(fixture(2,ids=[0,-1]),history)
        require(np.isnan(next_unique["previous_times"]).all(),"History jumped across ambiguous duplicate occurrence")
        checks.append("duplicate_old_ID_history_invalidated_until_unique_observation")
    elif round_number==2:
        hist={};a=attach_history(fixture(0),hist);b=attach_history(fixture(1),hist);c=attach_history(fixture(2),hist)
        block=candidate_block(c,b,0,geometry)
        require(block["scores"][0,0]==1 and block["scores"][1,0]==0 and block["scores"][0,5]==1,"Exact cosine control differs")
        require(block["scores"][0,3]<0 and block["scores"][0,4]<0,"Negative norm score direction differs")
        shifted=fixture(2);shifted["boxes"]+=np.array([1000,2000,1000,2000]);ref=fixture(1);ref["boxes"]+=np.array([1000,2000,1000,2000])
        ref["previous_boxes"]=b["previous_boxes"]+np.array([1000,2000,1000,2000]);ref["previous_times"]=b["previous_times"]
        new=candidate_block(shifted,ref,0,geometry)
        require(np.allclose(block["scores"][:,3:5],new["scores"][:,3:5]) and not np.allclose(block["scores"][:,1],new["scores"][:,1]),"Absolute and relative geometry conflated")
        checks.append("feature_sign_exact_cosine_translation_relative_vs_absolute")
        truth=np.array([1,0,1,0,0]);scores=np.array([.5,.5,1.,0.,1.])
        brute=np.mean([float(x>y)+.5*float(x==y) for x in scores[truth==1] for y in scores[truth==0]])
        require(auc(truth,scores)==brute and auc([1,1],[0,1]) is None,"Tie-aware/null AUC differs")
        checks.append("AUROC_exact_ties_bruteforce_and_absent_class_null")
        joined=dict(identity_label=np.array([1,0,0,1],np.int8),query_gt_identity=np.array([1,1,2,2]),reference_gt_identity=np.array([1,2,1,2]))
        block["cv_valid"]=np.array([True,False,True,False]);block["scores"][~block["cv_valid"],4]=np.nan
        desc=describe(block,joined,["S"])
        common=desc["all_candidates"]["common_CV_covered_population"]["micro_union"]
        require(common["known_candidates"]==2 and len(common["auc"])==6,"CV comparisons changed per-score denominator")
        checks.append("all_six_scores_share_one_CV_covered_denominator")
        # Three fixed queries: raw error repaired, raw correct broken, and no
        # known positive. Row-order intentionally differs from dump_row order.
        rank=dict(source_index=np.zeros(6,np.int16),video_id=np.ones(6,np.int32),
            query_image_id=np.array([11,11,12,12,13,13]),query_row=np.zeros(6,np.int32),
            reference_row=np.array([1,0,1,0,1,0]),cv_valid=np.ones(6,bool),
            scores=np.repeat(np.array([[0.,1.],[1.,0.],[1.,0.],[0.,1.],[0.,0.],[0.,0.]]),3,axis=1))
        # Columns raw and geometry must differ; all other fixed scores copy the
        # geometry column, to exercise identical masks and deterministic ties.
        rank["scores"][:,0]=[0,1,1,0,0,0]
        rank["scores"][:,1:]=np.array([1,0,0,1,0,0])[:,None]
        labels=dict(identity_label=np.array([1,0,1,0,0,0]))
        ranking=query_ranking_statistics(rank,labels,np.arange(6))
        primary=ranking["subsets"]["positive_and_negative_available_queries"]
        transitions=primary["top1"][FEATURES[1]]
        require(primary["queries"]==2 and transitions["repair_raw_error"]==transitions["break_raw_correct"]==1 and
            ranking["coverage"]["no_positive_queries"]==1 and
            ranking["subsets"]["all_nonempty_known_queries"]["queries"]==3,"Paired top1 transition/positive coverage changed")
        order=ranking["positive_negative_pair_order"][FEATURES[1]]
        require(order["repair_raw_loss_to_win"]==order["break_raw_win_to_loss"]==1,"Within-query pair-order transition differs")
        # Tie must follow immutable dump_row even when the later array row is
        # negative; GT must not choose the tied winner.
        rank["scores"][:2]=0
        tied=query_ranking_statistics(rank,labels,np.arange(2))
        require(tied["subsets"]["positive_and_negative_available_queries"]["top1"][FEATURES[0]]["feature_correct"]==0,
            "Top1 tie selected favorable GT instead of lowest dump_row")
        checks.append("paired_top1_repair_break_pair_order_fixed_tie_and_no_positive_coverage")
    elif round_number==3:
        helper=load_module(HELPER,"synthetic_only_identity_projection")
        roles={"S":dict(split="fit",role="mean_role")}
        identity=dict(observation_id=1,split="fit",source_group="S",video_id=1,image_id=1,sampled_step=1,original_frame_index=10,
            local_track_id=0,gt_track_id=5,pred_category_id=1,manual_anchor_id=9,eligible_for_parameter_fit=True)
        with tempfile.TemporaryDirectory(prefix="spacetime_projection_") as temporary:
            path=Path(temporary)/"synthetic_manual.jsonl"
            path.write_text(json.dumps(dict(identity,gt_category_id="CLASS_POISON",error="ERROR_POISON",soft_risk="RISK_POISON"))+"\n"+
                json.dumps(dict(split="tune",gt_track_id="TUNE_POISON"))+"\n")
            original=json.loads
            def guarded(text,*a,**kw):
                require(not any(word in text for word in ("CLASS_POISON","ERROR_POISON","RISK_POISON","TUNE_POISON")),"Forbidden value decoded")
                return original(text,*a,**kw)
            json.loads=guarded
            try:result=helper.load_fit_identities(path,roles)
            finally:json.loads=original
            require(list(result)==[1],"FIT projection scope changed")
        checks.append("fit_only_identity_decoder_poison_no_class_error_tune_identity")
        history={};a=attach_history(fixture(0),history);b=attach_history(fixture(1),history)
        original=json.dumps([a["boxes"].tolist(),b["boxes"].tolist()]);block=candidate_block(b,a,0,geometry)
        permutation=np.array([1,0]);permuted=dict(b)
        for key in ("boxes","raw","semantic","released_ids","predicted_categories"):permuted[key]=b[key][permutation]
        new=candidate_block(permuted,a,0,geometry)
        require(np.array_equal(new["scores"].reshape(2,2,6),block["scores"].reshape(2,2,6)[permutation],equal_nan=True) and
            original==json.dumps([a["boxes"].tolist(),b["boxes"].tolist()]),"Permutation mapping/input immutability failed")
        checks.append("full_candidate_permutation_row_mapping_and_inputs_immutable")
        with tempfile.TemporaryDirectory(prefix="spacetime_seal_guard_") as temporary:
            try:join_fit_identity(Path(temporary),{}, {}, {})
            except FileNotFoundError:pass
            else:raise AssertionError("Accepted identity join without GT-free seal")
        checks.append("supervision_join_rejects_missing_GT_free_seal")
        joined=dict(identity_label=np.array([-1,0,1,-1],np.int8),query_gt_identity=np.array([-1,1,1,-1]),reference_gt_identity=np.array([2,2,1,-1]))
        stats=score_statistics(block,joined,np.ones(4,dtype=bool),False)
        require(stats["known_candidates"]==2 and stats["positive_pairs"]==stats["negative_pairs"]==1 and stats["unknown_candidates"]==2,"Unknown became negative")
        checks.append("unknown_endpoints_excluded_from_positive_negative_labels")
    else:raise ValueError("Exactly three synthetic rounds")
    require(len(checks)>=2,"Each round requires at least two checks")
    return dict(status="PASS_SPACETIME_CANDIDATE_FEATURES_SYNTHETIC_ROUND",round=round_number,checks=checks,
        script_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,real_cache_or_supervised_payload_opened=False,
        training_started=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--synthetic-round",type=int,choices=(1,2,3))
    parser.add_argument("--synthetic-out",type=Path);parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    require(not(args.synthetic_round and args.execute),"Synthetic and real modes exclusive")
    if args.synthetic_round:
        result=synthetic(args.synthetic_round)
        if args.synthetic_out:write(args.synthetic_out,result)
        print(json.dumps(result,indent=2))
    else:
        require(args.execute,"Real execution requires --execute after independent preflight")
        execute()


if __name__=="__main__":main()
