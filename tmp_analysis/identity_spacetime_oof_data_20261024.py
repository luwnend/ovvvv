#!/usr/bin/env python3
"""Source-OOF data primitives; synthetic CLI only, no training or real execution.

Identity decoding order is lexical split -> source -> allowed eligibility ->
allowlisted eligible identity. Never load all FIT identities and then filter.
GT-free frame/history/Gperm generation has no identity input or return field.
"""
import argparse
from collections import Counter,defaultdict,deque
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
SEED="identity_spacetime_source_oof_v1"
HELPER=ROOT/"tmp_analysis/build_fit_identity_edge_support.py"
GEOMETRY=ROOT/"tmp_analysis/relative_track_geometry.py"
MODULES=ROOT/"tmp_analysis/identity_spacetime_modules_20261024.py"
PINS={HELPER:"e68f35ca48ba6f32d519efbdb33a711fc201cbf1a2f65c50f1c828a767b1d2b5",
      GEOMETRY:"c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a",
      MODULES:"2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3"}
IDENTITY_FIELDS={"observation_id","split","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id","gt_track_id","pred_category_id","manual_anchor_id","eligible_for_parameter_fit"}
INDEX_FIELDS={"observation_id","split","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id","pred_category_id","dump_row"}
BOUNDS=dict(max_frame_rows=77,max_video_frames=41,max_candidates_per_frame_pair=5929,generation_ring_frames=5)


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(8*1024*1024),b""):h.update(data)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=3)
def component(path):
    path=Path(path);require(path in PINS and sha(path)==PINS[path],"Frozen component changed")
    name="oof_data_"+path.stem
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    # dataclasses resolves module metadata from sys.modules on some interpreters.
    import sys
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def groups(roles):
    return roles["groups"] if "groups" in roles and isinstance(roles["groups"],dict) else roles


def build_fold_manifest(roles):
    roles=groups(roles);strata=defaultdict(list)
    for source,row in sorted(roles.items()):
        if row["split"]!="fit":continue
        require(row["role"] in ("mean_role","pair_role") and row.get("block")==source.split("/",1)[0],"Invalid original FIT source metadata")
        strata[(row["block"],row["role"])].append(source)
    assignments={};stratum_reports={}
    for (block,role),sources in sorted(strata.items()):
        ordered=sorted(sources,key=lambda source:(hashlib.sha256((SEED+"\0"+source).encode()).hexdigest(),source))
        stratum_reports[block+"/"+role]=ordered
        for i,source in enumerate(ordered):assignments[source]=dict(fold=i%5,role=role,dataset_block=block,stratum=[block,role])
    sources=sorted(assignments)
    folds={str(fold):dict(held_out_sources=[s for s in sources if assignments[s]["fold"]==fold],
        train_sources=[s for s in sources if assignments[s]["fold"]!=fold]) for fold in range(5)}
    return dict(seed=SEED,number_of_folds=5,groups={s:assignments[s] for s in sources},folds=folds,
        strata_order=stratum_reports,source_count=len(sources),roles_are_metadata_only=True,labels_used=False)


def prepare_inputs(protocol):
    """Authenticate caller's pinned metadata/authority; never decode identities."""
    protocol=json.loads(Path(protocol).read_text()) if isinstance(protocol,(str,Path)) else protocol
    inputs=dict(protocol["input_pins"])
    for path,digest in inputs.items():require(sha(path)==digest,"Input pin changed: "+str(path))
    roles_path=Path(protocol.get("source_roles_path",STAGE/"pair_context_v1/source_roles.json"))
    plan_path=Path(protocol.get("collection_plan_path",STAGE/"full_collection_plan.json"))
    frames_path=Path(protocol.get("frame_execution_path",STAGE/"full_collect_v1/frame_execution.jsonl"))
    inventory_path=Path(protocol.get("dump_inventory_path",STAGE/"full_collect_v1/dump_inventory.json"))
    for path in (roles_path,plan_path,frames_path,inventory_path,*PINS):require(str(path) in inputs,"Required GT-free metadata/component must be pinned: "+str(path))
    roles=groups(json.loads(roles_path.read_text()));manifest=build_fold_manifest(roles)
    if protocol.get("fold_manifest_path"):
        path=Path(protocol["fold_manifest_path"]);require(str(path) in inputs and json.loads(path.read_text())==manifest,"Pinned fold manifest differs from fixed metadata rule")
    plan=json.loads(plan_path.read_text());video_sources={str(row["video_id"]):row["source_group"] for row in plan["video_records"] if row["split"]=="fit"}
    inventory=json.loads(inventory_path.read_text());frames=[]
    with frames_path.open() as f:
        for line in f:
            row=json.loads(line)
            if str(row["video_id"]) not in video_sources:continue
            require(video_sources[str(row["video_id"])] in manifest["groups"],"Cached FIT source is outside original sealed roles")
            require(row["npz_sha256"]==inventory[Path(row["npz"]).name],"GT-free cache inventory changed")
            frames.append(row)
    if "identity_authority" in protocol:authority=dict(protocol["identity_authority"])
    else:
        receipt_path=STAGE/"support_v1/support_receipt.json";index_manifest_path=STAGE/"support_v1/gt_free_index_manifest.json"
        require(str(receipt_path) in inputs and str(index_manifest_path) in inputs,"Identity authority metadata must be pinned")
        scanner=component(HELPER);receipt=scanner.project_object(receipt_path.read_text(),{"output_files"});index_manifest=json.loads(index_manifest_path.read_text())
        manual=STAGE/"support_v1/supervised_observations.jsonl";index=STAGE/"support_v1/gt_free_observations.jsonl"
        authority=dict(manual_path=str(manual),manual_sha256=receipt["output_files"][manual.name],index_path=str(index),index_sha256=index_manifest["files"][index.name])
    for key in ("manual","index"):
        path=Path(authority[key+"_path"]);digest=authority[key+"_sha256"]
        require(sha(path)==digest,"Identity authority bytes changed")
        inputs[str(path)]=digest  # Hashing bytes is not identity-value decoding.
    return dict(roles=roles,manifest=manifest,frame_records=frames,video_sources=video_sources,authority=authority,input_sha256=inputs)


def load_train_anchors(allowed_sources,roles,video_sources,authority):
    """Decode ONLY allowed FIT eligible labels; skip held-out values lexically."""
    roles=groups(roles);allowed=set(allowed_sources)
    require(all(s in roles and roles[s]["split"]=="fit" and roles[s]["role"] in ("mean_role","pair_role") for s in allowed),"Invalid train-source authority")
    scanner=component(HELPER);manual=Path(authority["manual_path"]);index=Path(authority["index_path"])
    require(sha(manual)==authority["manual_sha256"] and sha(index)==authority["index_sha256"],"Manual/index authority changed")
    identities={};counts=Counter();source_counts=Counter()
    with manual.open() as stream:
        for line in stream:
            split=scanner.project_object(line,{"split"}).get("split")
            require(split in ("fit","tune"),"Unsupported supervised split")
            if split!="fit":counts["skipped_nonfit_rows"]+=1;continue
            source=scanner.project_object(line,{"source_group"}).get("source_group")
            if source not in allowed:counts["skipped_unallowed_source_rows"]+=1;continue
            eligible=scanner.project_object(line,{"eligible_for_parameter_fit"}).get("eligible_for_parameter_fit")
            require(isinstance(eligible,bool),"Missing eligibility authority")
            if not eligible:counts["skipped_ineligible_train_rows"]+=1;continue
            row=scanner.project_object(line,IDENTITY_FIELDS);require(set(row)==IDENTITY_FIELDS,"Missing eligible identity field")
            for field in IDENTITY_FIELDS-{"split","source_group","eligible_for_parameter_fit"}:
                require(scanner.integer(row[field]) and row[field]>=0,"Invalid eligible identity field: "+field)
            require(row["pred_category_id"]>0 and row["source_group"]==source and video_sources[str(row["video_id"])]==source,"Identity source/video mismatch")
            require(row["observation_id"] not in identities,"Duplicate allowed observation identity")
            identities[row["observation_id"]]=row;source_counts[source]+=1
    anchors={};joined=set()
    with index.open() as stream:
        for line in stream:
            split=scanner.project_object(line,{"split"}).get("split")
            if split!="fit":continue
            source=scanner.project_object(line,{"source_group"}).get("source_group")
            if source not in allowed:continue
            row=scanner.project_object(line,INDEX_FIELDS);require(set(row)==INDEX_FIELDS,"Missing GT-free immutable index field")
            identity=identities.get(row["observation_id"])
            if identity is None:continue
            require(video_sources[str(row["video_id"])]==source,"Immutable index source/video mismatch")
            for field in INDEX_FIELDS-{"dump_row"}:
                require(row[field]==identity[field],"Identity/index immutable locator mismatch: "+field)
            require(scanner.integer(row["dump_row"]) and row["dump_row"]>=0,"Invalid immutable dump row")
            key=(row["image_id"],row["dump_row"]);require(key not in anchors and row["observation_id"] not in joined,"Duplicate immutable anchor")
            anchors[key]=(row["video_id"],identity["gt_track_id"]);joined.add(row["observation_id"])
    require(joined==set(identities),"Eligible train identity lacks immutable row")
    projection=[identities[k] for k in sorted(identities)]
    canonical_anchors=[dict(image_id=k[0],dump_row=k[1],video_id=v[0],gt_track_id=v[1]) for k,v in sorted(anchors.items())]
    receipt=dict(allowed_sources=sorted(allowed),eligible_train_anchors=len(anchors),eligible_identity_counts_by_source={s:source_counts[s] for s in sorted(allowed)},
        train_label_projection_sha256=canonical_sha(projection),immutable_anchor_sha256=canonical_sha(canonical_anchors),
        input_sha256={str(manual):authority["manual_sha256"],str(index):authority["index_sha256"],str(HELPER):PINS[HELPER]},
        counts=dict(counts),decoding_order=["split","source_group","train_eligibility","allowlisted_eligible_identity"],heldout_identity_values_decoded=False,forbidden_correctness_fields_decoded=False)
    return anchors,receipt


def load_frame(record):
    require(sha(record["npz"])==record["npz_sha256"],"GT-free cache file changed")
    with np.load(record["npz"],allow_pickle=False) as stored:
        require(all(int(stored[key])==record[field] for key,field in (("video_id","video_id"),("image_id","image_id"),("frame_id","frame_id"),("frame_index","frame_index"))),"GT-free frame provenance changed")
        boxes=stored["det_bboxes"].astype(np.float64);raw=stored["track_feats"].astype(np.float64)
        ids=stored["track_ids"].astype(np.int64);categories=stored["det_labels"].astype(np.int32)+1
    n=record["observations"]
    require(boxes.shape==(n,5) and raw.shape==(n,256) and ids.shape==categories.shape==(n,) and np.isfinite(raw).all() and np.isfinite(boxes).all() and np.all(boxes[:,2:4]>boxes[:,:2]),"Invalid retained GT-free rows")
    return dict(video_id=record["video_id"],image_id=record["image_id"],sampled_step=record["frame_id"],original_frame_index=record["frame_index"],
        boxes=boxes[:,:4],raw=raw,released_ids=ids,predicted_categories=categories)


def attach_history(frame,history):
    n=len(frame["boxes"]);counts=Counter(int(x) for x in frame["released_ids"] if x>=0)
    previous=np.full((n,4),np.nan);times=np.full(n,np.nan);locators=[None]*n;reasons=[]
    for row,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity<0:reasons.append("unassigned_id");continue
        if counts[identity]>1:reasons.append("duplicate_current_id");continue
        if identity not in history:reasons.append("no_previous_occurrence");continue
        if history[identity] is None:reasons.append("previous_duplicate_invalid");continue
        box,time,image,dump_row=history[identity];require(time<frame["original_frame_index"],"History must precede reference")
        previous[row]=box;times[row]=time;locators[row]=dict(image_id=int(image),dump_row=int(dump_row),original_frame_index=int(time));reasons.append("available")
    for row,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity>=0:history[identity]=None if counts[identity]>1 else (frame["boxes"][row].copy(),frame["original_frame_index"],frame["image_id"],row)
    frame.update(previous_boxes=previous,previous_times=times,previous_locators=locators,history_reason=reasons)
    return frame


def iter_gtfree_frame_pairs(video_ids,lags=(1,4),*,frame_records=None,video_sources=None,loader=None,bounds=None):
    """Yield every exact sampled lag's full QxM, including prefix/empty receipts."""
    require(tuple(lags)==(1,4),"Fixed sampled lags are exactly (1,4)")
    require(frame_records is not None and video_sources is not None,"Call prepare_inputs once and pass its authenticated GT-free metadata")
    bounds=BOUNDS if bounds is None else bounds;loader=load_frame if loader is None else loader
    require(bounds.get("generation_ring_frames",5)==5,"Five-frame ring is fixed")
    selected=set(int(v) for v in video_ids);grouped=defaultdict(list)
    for record in frame_records:
        if int(record["video_id"]) in selected:grouped[int(record["video_id"])].append(record)
    geometry=component(GEOMETRY);modules=component(MODULES)
    for video,records in sorted(grouped.items()):
        records=sorted(records,key=lambda row:row["frame_id"])
        require([r["frame_id"] for r in records]==list(range(len(records))) and len(records)<=bounds["max_video_frames"],"Incomplete sampled-step video or bound exceeded")
        source=video_sources[str(video)];history={};ring=deque(maxlen=5);last_time=None;offsets=Counter()
        for record in records:
            frame=loader(record);nq=len(frame["boxes"])
            require(nq==record["observations"] and nq<=bounds["max_frame_rows"] and frame["video_id"]==video and frame["sampled_step"]==record["frame_id"],"Loader provenance or frame bound differs")
            require(last_time is None or frame["original_frame_index"]>last_time,"Nonincreasing original-frame timeline")
            last_time=frame["original_frame_index"];attach_history(frame,history);ring.append(frame)
            for lag in lags:
                reference=ring[-lag-1] if frame["sampled_step"]>=lag else None;nm=0 if reference is None else len(reference["boxes"])
                require(nq*nm<=bounds["max_candidates_per_frame_pair"],"Full candidate block bound exceeded")
                if reference is None:
                    values=np.empty((nq,0,9));masks=np.zeros((nq,0,9),bool)
                    plan=modules.gperm_donors([],[],[])
                else:
                    require(frame["sampled_step"]-reference["sampled_step"]==lag,"Skipped an empty sampled frame")
                    geom=geometry.pair_geometry(frame["boxes"],reference["boxes"],frame["original_frame_index"],np.full(nm,reference["original_frame_index"]),reference["previous_boxes"],reference["previous_times"])
                    values,masks=modules.geometry_slots(geom,np.full(nm,reference["original_frame_index"]),reference["previous_times"])
                    plan=modules.gperm_donors(reference["predicted_categories"],np.isfinite(reference["previous_times"]),np.arange(nm))
                receipt=dict(video_id=video,source_group=source,lag=lag,query_step=frame["sampled_step"],query_image_id=frame["image_id"],
                    reference_step=None if reference is None else reference["sampled_step"],reference_image_id=None if reference is None else reference["image_id"],
                    query_rows=nq,reference_rows=nm,candidates=nq*nm,offset=offsets[lag],original_frame_gap=None if reference is None else frame["original_frame_index"]-reference["original_frame_index"],
                    feature_ring_frames=len(ring),row_order="query_major_then_reference_dump_row")
                offsets[lag]+=nq*nm
                yield dict(source=source,video_id=video,lag=lag,query=frame,reference=reference,geometry_values=values,geometry_masks=masks,gperm_plan=plan,receipt=receipt)


def build_train_examples(pair_stream,anchors,allowed_sources,coverage=None):
    """Loss projection only; every example still carries all retained refs."""
    allowed=set(allowed_sources);coverage=Counter() if coverage is None else coverage;modules=component(MODULES)
    for pair in pair_stream:
        source=pair["source"];video=pair["video_id"]
        require(source in allowed,"Held-out source frame pair entered training")
        q=pair["query"];r=pair["reference"];coverage["frame_pair_receipts"]+=1;coverage["retained_candidate_pairs"]+=pair["receipt"]["candidates"]
        if r is None:coverage["missing_prefix_receipts"]+=1;continue
        def identities(frame):
            result=[]
            for row in range(len(frame["boxes"])):
                value=anchors.get((frame["image_id"],row));require(value is None or value[0]==video,"Identity video mismatch")
                result.append(-1 if value is None else value[1])
            return np.asarray(result,np.int64)
        qids=identities(q);rids=identities(r);nm=len(rids)
        for qr,identity in enumerate(qids):
            coverage["query_lag_rows"]+=1
            if identity<0:coverage["unknown_query_rows"]+=1;continue
            projection=modules.training_projection(int(identity),rids,q["predicted_categories"][qr],r["predicted_categories"])
            coverage["same_category_unknown_reference_exposures"]+=int(np.count_nonzero((rids<0)&(r["predicted_categories"]==q["predicted_categories"][qr])))
            if not len(projection["positive_rows"]):coverage["known_query_no_positive_coverage_only"]+=1
            if not projection["hard"]:coverage["nonhard_known_queries"]+=1;continue
            coverage["hard_query_lags"]+=1;coverage["positive_negative_comparisons"]+=len(projection["positive_rows"])*len(projection["negative_rows"])
            record=dict(source=source,video_id=video,qgt=int(identity),image_id=q["image_id"],qrow=qr,lag=pair["lag"],positive_rows=projection["positive_rows"],negative_rows=projection["negative_rows"])
            yield dict(record=record,query_raw=q["raw"][qr:qr+1],reference_raw=r["raw"],values=pair["geometry_values"][qr],masks=pair["geometry_masks"][qr],gperm_plan=pair["gperm_plan"],
                reference_predicted_categories=r["predicted_categories"],reference_history_valid=np.isfinite(r["previous_times"]),reference_dump_rows=np.arange(nm),reference_image_id=r["image_id"])


def fixture(step,n=3,ids=None,video=1):
    raw=np.zeros((n,256));boxes=np.array([[step*3+k*20,0,step*3+k*20+10,10] for k in range(n)],float).reshape(n,4)
    for i in range(n):raw[i,i]=1
    return dict(video_id=video,image_id=video*100+step,sampled_step=step,original_frame_index=[0,10,35,40,90,100][step],boxes=boxes,raw=raw,
        released_ids=np.array(list(range(n)) if ids is None else ids,np.int64),predicted_categories=np.ones(n,np.int32))


def synthetic(number):
    checks=[];scanner=component(HELPER)
    if number==1:
        roles={f"D/{i}":dict(split="fit",role="mean_role" if i%2 else "pair_role",block="D") for i in range(15)}
        roles["D/tune"]=dict(split="tune",role="tune_evaluation")
        first=build_fold_manifest(roles);second=build_fold_manifest(dict(reversed(list(roles.items()))))
        require(first==second and first["source_count"]==15 and all(len(first["folds"][str(f)]["held_out_sources"])>=2 for f in range(5)),"Metadata folds depend on input order/support")
        for key,ordered in first["strata_order"].items():
            require([first["groups"][s]["fold"] for s in ordered]==[i%5 for i in range(len(ordered))],"Stratum SHA/NUL roundrobin differs")
        checks.append("fixed_stratified_metadata_SHA_NUL_roundrobin_all_sources_and_input_order_invariance")
        with tempfile.TemporaryDirectory(prefix="oof_train_identity_") as temp:
            temp=Path(temp);manual=temp/"manual.jsonl";index=temp/"index.jsonl"
            good=dict(observation_id=1,split="fit",source_group="D/train",video_id=1,image_id=100,sampled_step=0,original_frame_index=0,local_track_id=0,gt_track_id=7,pred_category_id=1,manual_anchor_id=1,eligible_for_parameter_fit=True)
            identity=dict(good,gt_category_id="CLASS_POISON",error="ERROR_POISON",soft_risk="RISK_POISON",posterior_argmax_category_id="POST_POISON")
            held=dict(good,source_group="D/held",gt_track_id={"HELD_POISON":["not","an","integer"]},eligible_for_parameter_fit="HELD_ELIGIBILITY_POISON")
            tune=dict(split="tune",source_group="D/tune",gt_track_id="TUNE_POISON")
            ineligible=dict(good,observation_id=2,gt_track_id="INELIGIBLE_POISON",eligible_for_parameter_fit=False)
            manual.write_text("".join(json.dumps(x)+"\n" for x in (identity,held,tune,ineligible)))
            idx={k:good[k] for k in INDEX_FIELDS if k!="dump_row"};idx["dump_row"]=0
            index.write_text(json.dumps(idx)+"\n"+json.dumps(dict(split="fit",source_group="D/held",observation_id="HELD_INDEX_POISON"))+"\n")
            authority=dict(manual_path=str(manual),manual_sha256=sha(manual),index_path=str(index),index_sha256=sha(index))
            roles={s:dict(split="fit",role="mean_role",block="D") for s in ("D/train","D/held")};original=json.loads
            def guarded(text,*args,**kwargs):
                require("POISON" not in text,"Forbidden held-out/correctness value decoded")
                return original(text,*args,**kwargs)
            json.loads=guarded
            try:anchors,receipt=load_train_anchors(["D/train"],roles,{"1":"D/train"},authority)
            finally:json.loads=original
            require(anchors=={(100,0):(1,7)} and receipt["eligible_train_anchors"]==1,"Train-only anchor projection differs")
            before=receipt["train_label_projection_sha256"];held["gt_track_id"]={"OTHER_HELD_POISON":False}
            manual.write_text("".join(json.dumps(x)+"\n" for x in (identity,held,tune,ineligible)));authority["manual_sha256"]=sha(manual)
            anchors,receipt=load_train_anchors(["D/train"],roles,{"1":"D/train"},authority)
            require(receipt["train_label_projection_sha256"]==before,"Held-out mutation altered train functional projection")
            identity["gt_track_id"]=8;manual.write_text("".join(json.dumps(x)+"\n" for x in (identity,held,tune,ineligible)));authority["manual_sha256"]=sha(manual)
            anchors,receipt=load_train_anchors(["D/train"],roles,{"1":"D/train"},authority)
            require(receipt["train_label_projection_sha256"]!=before and anchors[(100,0)][1]==8,"Legal train mutation had no effect")
        checks.append("lexical_split_source_eligibility_before_identity_poison_skip_heldout_invariance_train_mutation_effect")
    elif number==2:
        bounds=dict(max_frame_rows=3,max_video_frames=6,max_candidates_per_frame_pair=9,generation_ring_frames=5)
        records=[dict(video_id=1,frame_id=k,observations=0 if k==1 else 3) for k in range(6)]
        pairs=list(iter_gtfree_frame_pairs([1],frame_records=records,video_sources={"1":"D/S"},loader=lambda r:fixture(r["frame_id"],r["observations"],[] if r["observations"]==0 else [0,-1,2]),bounds=bounds))
        require(len(pairs)==12 and sum(p["receipt"]["candidates"] for p in pairs if p["lag"]==1)==27 and sum(p["receipt"]["candidates"] for p in pairs if p["lag"]==4)==9 and max(p["receipt"]["feature_ring_frames"] for p in pairs)==5,"Exact full lag QxM/prefix/empty/ring differs")
        empty=next(p for p in pairs if p["lag"]==1 and p["query"]["sampled_step"]==2)
        require(empty["receipt"]["reference_rows"]==0 and empty["geometry_values"].shape==(3,0,9),"Empty reference skipped")
        lag4=next(p for p in pairs if p["lag"]==4 and p["query"]["sampled_step"]==4)
        require(lag4["receipt"]["original_frame_gap"]==90 and not lag4["geometry_masks"][:,:,5:].any(),"Sampled lag converted to original frame units or CV faked")
        checks.append("full_lag1_lag4_QxM_prefix_empty_unknown_original_time_ringfive_and_CVmask")
        history={};first=attach_history(fixture(0,2),history);second=attach_history(fixture(1,2),history);snapshot=second["previous_boxes"].copy();first["boxes"][:]=999
        duplicate=attach_history(fixture(2,2,ids=[0,0]),history);following=attach_history(fixture(3,2,ids=[0,-1]),history);resumed=attach_history(fixture(4,2,ids=[0,-1]),history)
        require(np.array_equal(second["previous_boxes"],snapshot) and duplicate["history_reason"]==["duplicate_current_id"]*2 and following["history_reason"]==["previous_duplicate_invalid","unassigned_id"] and resumed["previous_locators"][0]["image_id"]==103,"Unique duplicate no-ID locator snapshot contract differs")
        records=[dict(video_id=v,frame_id=k,observations=1) for v in (1,2) for k in range(2)]
        reset=list(iter_gtfree_frame_pairs([1,2],frame_records=records,video_sources={"1":"D/A","2":"D/B"},loader=lambda r:fixture(r["frame_id"],1,video=r["video_id"]),bounds=bounds))
        require(all(p["query"]["previous_times"].tolist()==[np.nan] or p["query"]["history_reason"]==["no_previous_occurrence"] for p in reset if p["query"]["sampled_step"]==0),"History crossed videos")
        checks.append("history_duplicate_invalidates_next_unique_unassigned_snapshots_and_video_reset")
    elif number==3:
        bounds=dict(max_frame_rows=3,max_video_frames=6,max_candidates_per_frame_pair=9,generation_ring_frames=5)
        records=[dict(video_id=1,frame_id=k,observations=3) for k in range(6)]
        pairs=list(iter_gtfree_frame_pairs([1],frame_records=records,video_sources={"1":"D/S"},loader=lambda r:fixture(r["frame_id"]),bounds=bounds))
        anchors={(100+k,row):(1,7 if row in (0,1) else 8) for k in range(6) for row in range(3)}
        # Unknown ref is retained as a GT-free donor but not negative.
        del anchors[(100,1)];coverage=Counter();examples=list(build_train_examples(pairs,anchors,["D/S"],coverage))
        example=next(e for e in examples if e["record"]["image_id"]==101 and e["record"]["qrow"]==0 and e["record"]["lag"]==1)
        require(example["reference_raw"].shape==(3,256) and example["record"]["positive_rows"].tolist()==[0] and example["record"]["negative_rows"].tolist()==[2] and example["gperm_plan"].donor_rows.tolist()==[2,0,1],"Known projection altered retained candidates or Gperm donors")
        require(not example["masks"][:,5:].any(),"Training wrongly required commonCV support")
        require(coverage["same_category_unknown_reference_exposures"]>0 and all("gt" not in key.lower() for p in pairs for key in p["query"]),"Unknown lost or GT fields leaked into frame iterator")
        weights,report=component(MODULES).hierarchical_weights([e["record"] for e in examples],["D/S","D/ZERO"],["D/S","D/ZERO"])
        require(abs(weights.sum()-1)<1e-12 and report["zero_support_train_sources"]==["D/ZERO"],"Data records do not support exact hierarchy")
        checks.append("training_samepred_allknown_hard_fullPxN_unknown_notnegative_allrefs_Gperm_and_noCVselection")
        try:list(build_train_examples(pairs,anchors,["D/held"]))
        except ValueError:pass
        else:raise AssertionError("Held-out frame source entered training")
        noanchor=list(build_train_examples(pairs,{},["D/S"]))
        require(noanchor==[],"Unknown observations generated null/no-match training targets")
        checks.append("training_rejects_unallowed_source_and_no_identity_yields_coverage_only")
        empty_records=[dict(video_id=1,frame_id=k,observations=0) for k in range(2)]
        empty=list(iter_gtfree_frame_pairs([1],frame_records=empty_records,video_sources={"1":"D/S"},loader=lambda r:fixture(r["frame_id"],0),bounds=bounds))
        require(len(empty)==4 and all(p["geometry_values"].shape==(0,0,9) for p in empty),"All-empty short video receipt failed")
        checks.append("all_empty_short_video_prefix_and_zero_candidate_geometry_stream")
    else:raise ValueError("Exactly three synthetic rounds")
    return dict(status="PASS_IDENTITY_SPACETIME_OOF_DATA_SYNTHETIC",round=number,checks=checks,script_sha256=sha(__file__),
        component_sha256={str(k):v for k,v in PINS.items()},real_supervision_or_cache_opened=False,training_started=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--synthetic-round",type=int,choices=(1,2,3),required=True);parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    result=synthetic(args.synthetic_round)
    if args.synthetic_out:
        with args.synthetic_out.open("x") as f:f.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
