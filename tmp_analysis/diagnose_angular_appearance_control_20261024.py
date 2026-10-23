#!/usr/bin/env python3
"""Streaming full-sealed-FIT fixed angular appearance control. No training or tuning.

Real execution belongs to root after independent preflight. Synthetic mode
opens only temporary synthetic data and the frozen pure appearance core.
"""
import argparse
from collections import Counter,defaultdict,deque
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sqlite3
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
RUN=STAGE/"full_collect_v1"
SUPPORT=STAGE/"support_v1"
CONTEXT=STAGE/"pair_context_v1"
PROTOCOL=ROOT/"tmp_analysis/angular_appearance_control_protocol_20261024.json"
PROTOCOL_SHA="44d8bce20de4ed3fd358a083665e6e9d8b2744d29d86dcdc9b2d9efa2f07eb69"
PREFLIGHT=ROOT/"tmp_analysis/angular_appearance_control_independent_preflight_20261024.json"
GEOMETRY=ROOT/"tmp_analysis/relative_track_geometry.py"
HELPER=ROOT/"tmp_analysis/build_fit_identity_edge_support.py"
CORE=ROOT/"tmp_analysis/causal_appearance_transport.py"
CORE_SHA="8b534c5f133880165e3e46abbd93e49584ebf8e20bec9c58b46ab425fa8f5924"
PARENT=ROOT/"results/fit_lag_identity_support_20261024_v1"
FEATURES=("raw_appearance_cosine","angular_prediction_cosine")
SCOPES=("all_candidates","same_predicted_category")
POPS=("all_known_population","common_CV_covered_population")
BOUNDARIES={"rho":[0,1,2,4,8,16,32,64,128],"history_angle_radians":[0,.01,.05,.1,.25,.5,1,2,float(np.pi)]}
LAGS=(1,4)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open("x") as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def normalized(values):
    values=np.asarray(values,np.float64)
    require(values.ndim==2 and np.isfinite(values).all(),"Invalid feature matrix")
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)


def attach_history(frame,history):
    """Unique released IDs locate past snapshots only; never truth/filter."""
    n=len(frame["boxes"]);counts=Counter(int(x) for x in frame["released_ids"] if x>=0)
    previous=np.full((n,4),np.nan);times=np.full(n,np.nan)
    raw=np.zeros_like(frame["raw"],dtype=np.float64);locators=[None]*n;reason=[]
    for k,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity<0:reason.append("unassigned_id");continue
        if counts[identity]>1:reason.append("duplicate_current_id");continue
        if identity not in history:reason.append("no_previous_occurrence");continue
        if history[identity] is None:reason.append("previous_duplicate_invalid");continue
        box,t,appearance,image,row=history[identity]
        require(t<frame["original_frame_index"],"Nonpast history")
        previous[k]=box;times[k]=t;raw[k]=appearance
        locators[k]=dict(image_id=int(image),dump_row=int(row),original_frame_index=int(t))
        reason.append("available")
    for k,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity>=0:
            history[identity]=None if counts[identity]>1 else (frame["boxes"][k].copy(),frame["original_frame_index"],
                frame["raw"][k].copy(),frame["image_id"],k)
    frame.update(previous_boxes=previous,previous_times=times,previous_raw=raw,previous_locators=locators,history_reason=reason)
    return frame

def score_block(query,reference,core):
    """Predict from past+reference only, then compare with query raw appearance."""
    nq,nm=len(query["boxes"]),len(reference["boxes"])
    require(query["video_id"]==reference["video_id"] and query["sampled_step"]>reference["sampled_step"] and
        query["original_frame_index"]>reference["original_frame_index"],"Invalid frame-pair timeline")
    last=reference["raw"];norm=np.linalg.norm(last,axis=1);nonzero=norm>0
    prediction=np.zeros_like(last,dtype=np.float64)
    diag={k:np.zeros(nm,dtype=bool) for k in ("directed_arc","coincident_static","antipodal_fallback","missing_or_zero_history")}
    diag.update(history_angle_radians=np.zeros(nm),extrapolation_ratio=np.zeros(nm))
    if np.any(nonzero):
        result=core.angular_velocity_prediction(reference["previous_raw"][nonzero],last[nonzero],
            reference["previous_times"][nonzero],np.full(int(nonzero.sum()),reference["original_frame_index"]),
            np.full(int(nonzero.sum()),query["original_frame_index"]),np.isfinite(reference["previous_times"])[nonzero])
        prediction[nonzero]=result["prediction"]
        for key in diag:diag[key][nonzero]=result[key]
    diag["last_zero_norm"]=~nonzero
    diag["previous_zero_norm"]=np.isfinite(reference["previous_times"])&(np.linalg.norm(reference["previous_raw"],axis=1)==0)
    # raw column exactly retains the parent normalization and GEMM expression.
    scores=np.stack((normalized(query["raw"])@normalized(last).T,
        normalized(query["raw"])@prediction.T),axis=2).reshape(nq*nm,2)
    require(np.isfinite(scores).all() and np.isfinite(prediction).all(),"Nonfinite angular scores")
    require(np.array_equal(prediction[~nonzero],np.zeros_like(prediction[~nonzero])),"Zero-last must remain exact zero")
    return scores,{k:v.tolist() for k,v in diag.items()}

def frame_metadata(frame):
    return dict(video_id=int(frame["video_id"]),image_id=int(frame["image_id"]),sampled_step=int(frame["sampled_step"]),
        original_frame_index=int(frame["original_frame_index"]),boxes=frame["boxes"].tolist(),
        released_ids=frame["released_ids"].tolist(),predicted_categories=frame["predicted_categories"].tolist(),
        previous_boxes=[[float(x) for x in row] if np.isfinite(row).all() else None for row in frame["previous_boxes"]],
        previous_times=[int(x) if np.isfinite(x) else None for x in frame["previous_times"]],
        previous_locators=frame["previous_locators"],history_reason=frame["history_reason"])


def load_frame(record):
    # Load only declared small fields; posterior payloads are not materialized.
    with np.load(record["npz"],allow_pickle=False) as stored:
        require(all(int(stored[key])==record[field] for key,field in
            (("video_id","video_id"),("image_id","image_id"),("frame_id","frame_id"),("frame_index","frame_index"))),"Frame provenance changed")
        box=stored["det_bboxes"].astype(np.float64);raw=stored["track_feats"].astype(np.float64)
        ids=stored["track_ids"].astype(np.int64)
        categories=stored["det_labels"].astype(np.int32)+1
    n=len(box)
    require(box.shape==(n,5) and raw.shape==(n,256) and ids.shape==categories.shape==(n,) and
        n==record["observations"] and np.isfinite(box).all() and np.all(box[:,2:4]>box[:,:2]),"Invalid retained rows")
    return dict(video_id=record["video_id"],image_id=record["image_id"],sampled_step=record["frame_id"],
        original_frame_index=record["frame_index"],boxes=box[:,:4],raw=raw,
        released_ids=ids,predicted_categories=categories)


def generate(out,frames,video_source,sources,core,loader,bounds):
    """GT-free phase; two virtual memmaps, five feature frames, one video map."""
    (out/"shards").mkdir();grouped=defaultdict(list)
    for row in frames:grouped[row["video_id"]].append(row)
    files={};candidate_counts=Counter();maximum_block=0;maximum_ring=0;pair_rows=0
    receipts_path=out/"frame_pairs.jsonl"
    with receipts_path.open("x") as receipts:
        for video,records in sorted(grouped.items()):
            records.sort(key=lambda r:r["frame_id"])
            require([r["frame_id"] for r in records]==list(range(len(records))),"Missing cached sampled frame")
            require(len(records)<=bounds["max_video_frames"] and all(r["observations"]<=bounds["max_frame_rows"] for r in records),"Frame bound exceeded")
            sizes={lag:sum(records[t]["observations"]*records[t-lag]["observations"] for t in range(lag,len(records))) for lag in LAGS}
            paths={lag:out/"shards"/f"v{video}_lag{lag}.scores.npy" for lag in LAGS}
            maps={lag:np.lib.format.open_memmap(paths[lag],mode="w+",dtype=np.float64,shape=(sizes[lag],2)) for lag in LAGS}
            offsets=Counter();history={};ring=deque(maxlen=5);metadata=[]
            for record in records:
                frame=attach_history(loader(record),history);ring.append(frame);maximum_ring=max(maximum_ring,len(ring))
                require(frame["video_id"]==video and frame["sampled_step"]==record["frame_id"],"Loader mapping changed")
                metadata.append(frame_metadata(frame))
                for lag in LAGS:
                    reference=ring[-lag-1] if frame["sampled_step"]>=lag else None
                    nq=len(frame["boxes"]);nm=0 if reference is None else len(reference["boxes"]);count=nq*nm
                    require(count<=bounds["max_candidates_per_frame_pair"],"Candidate block bound exceeded")
                    if reference is not None:
                        require(frame["sampled_step"]-reference["sampled_step"]==lag,"Ring skipped an empty frame")
                        block,diagnostics=score_block(frame,reference,core);require(block.shape==(count,2),"Incomplete QxM")
                        maps[lag][offsets[lag]:offsets[lag]+count]=block
                    receipt=dict(video_id=video,source_group=video_source[str(video)],lag=lag,query_step=frame["sampled_step"],
                        query_image_id=frame["image_id"],reference_step=None if reference is None else reference["sampled_step"],
                        reference_image_id=None if reference is None else reference["image_id"],query_rows=nq,reference_rows=nm,
                        original_frame_gap=None if reference is None else frame["original_frame_index"]-reference["original_frame_index"],
                        offset=offsets[lag],candidates=count,score_file=str(paths[lag].relative_to(out)),
                        forecast=None if reference is None else diagnostics)
                    receipts.write(json.dumps(receipt,separators=(",",":"))+"\n");pair_rows+=1
                    offsets[lag]+=count;candidate_counts[lag]+=count;maximum_block=max(maximum_block,count)
            rowpath=out/"shards"/f"v{video}.rows.json";write(rowpath,dict(video_id=video,source_group=video_source[str(video)],frames=metadata))
            files[str(rowpath.relative_to(out))]=sha(rowpath)
            for lag in LAGS:
                require(offsets[lag]==sizes[lag],"Unwritten score rows");maps[lag].flush()
                del maps[lag];files[str(paths[lag].relative_to(out))]=sha(paths[lag])
            if len(files)%60==0:print(json.dumps(dict(phase="GT_free_generation",video_id=video,candidates=sum(candidate_counts.values()))),flush=True)
    files[receipts_path.name]=sha(receipts_path)
    return dict(files_sha256=files,candidates_by_lag={str(k):candidate_counts[k] for k in LAGS},
        maximum_candidate_block=maximum_block,maximum_feature_ring_frames=maximum_ring,frame_pair_receipts=pair_rows,
        videos=len(grouped),source_groups=len(sources))


def verify_seal(out):
    seal=json.loads((out/"GT_free_candidates_sealed.json").read_text())
    require(seal["status"]=="SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION","No global GT-free seal")
    for relative,digest in seal["files_sha256"].items():
        require(sha(out/relative)==digest,"Sealed score/rowmap/receipt changed: "+relative)
    return seal


def join_identities(out,roles,video_source,inputs):
    verify_seal(out)  # first full sealed-output pass BEFORE supervised import
    helper=module(HELPER,"fit_lag_identity_projector")
    receipt=helper.project_object((SUPPORT/"support_receipt.json").read_text(),{"output_files"})
    manual=SUPPORT/"supervised_observations.jsonl"
    require(sha(manual)==receipt["output_files"][manual.name],"Manual identity source changed")
    identities=helper.load_fit_identities(manual,roles)
    require(len(identities)<=25000,"Identity-memory bound exceeded")
    manifest=json.loads((SUPPORT/"gt_free_index_manifest.json").read_text());index=SUPPORT/"gt_free_observations.jsonl"
    require(sha(index)==manifest["files"][index.name],"Immutable GT-free index changed")
    anchors={}
    with index.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row["split"]!="fit":continue
            require(str(row["video_id"]) in video_source and row["source_group"]==video_source[str(row["video_id"])] ,"FIT index video/source differs")
            identity=identities.get(row["observation_id"])
            if identity is None:continue
            require(all(row[k]==identity[k] for k in ("video_id","source_group","local_track_id","image_id","sampled_step","original_frame_index","pred_category_id")),"Manual anchor/index mismatch")
            key=(row["image_id"],row["dump_row"]);require(key not in anchors,"Duplicate immutable row anchor")
            anchors[key]=(row["video_id"],identity["gt_track_id"])
    require(len(anchors)==len(identities),"Not every eligible FIT identity has an immutable row")
    for p in (manual,index):inputs[str(p)]=sha(p)
    return anchors


def open_database(path):
    db=sqlite3.connect(path);db.execute("PRAGMA temp_store=FILE");db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE query_support(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,image INTEGER,qrow INTEGER,qgt INTEGER,p INTEGER,n INTEGER,raw_correct INTEGER,correct_bits INTEGER,PRIMARY KEY(lag,scope,population,video,image,qrow))")
    db.execute("CREATE INDEX query_source ON query_support(source,lag,scope,population,video,qgt)")
    return db

def fresh_cell():
    return dict(candidate_pairs=0,known_pairs=0,positive_pairs=0,negative_pairs=0,unknown_pairs=0,
        nonempty_known_queries=0,positive_available_queries=0,negative_available_queries=0,hard_queries=0,
        top1={subset:{f:dict(raw_wrong_feature_wrong=0,raw_wrong_feature_correct=0,raw_correct_feature_wrong=0,raw_correct_feature_correct=0)
            for f in FEATURES} for subset in ("all_nonempty_known_queries","positive_and_negative_available_queries")},
        pair_order={f:np.zeros((3,3),np.int64) for f in FEATURES})


def update_cell(cell,scores,mask,qids,rids,reference_rows,source,lag,scope,population,video,image,db):
    """One frame pair; bounded by max_frame_rows^2, labels only score fixed rows."""
    nq,nm=mask.shape;labels=np.full((nq,nm),-1,np.int8)
    known=(qids[:,None]>=0)&(rids[None,:]>=0);labels[known]=(np.broadcast_to(qids[:,None],(nq,nm))[known]==np.broadcast_to(rids[None,:],(nq,nm))[known]).astype(np.int8)
    selected=mask&known
    cell["candidate_pairs"]+=int(mask.sum());cell["known_pairs"]+=int(selected.sum())
    cell["positive_pairs"]+=int(np.count_nonzero(selected&(labels==1)));cell["negative_pairs"]+=int(np.count_nonzero(selected&(labels==0)))
    cell["unknown_pairs"]+=int(np.count_nonzero(mask&~known))
    columns=range(2)
    for qr in range(nq):
        refs=np.flatnonzero(selected[qr])
        if not len(refs):continue
        truth=labels[qr,refs];value=scores[qr,refs];p=int(np.count_nonzero(truth==1));n=len(refs)-p
        require(np.isfinite(value[:,list(columns)]).all(),"Unexpected finite-score selection")
        correct={c:bool(truth[np.lexsort((reference_rows[refs],-value[:,c]))[0]]==1) for c in columns}
        cell["nonempty_known_queries"]+=1;cell["positive_available_queries"]+=int(p>0);cell["negative_available_queries"]+=int(n>0)
        cell["hard_queries"]+=int(p>0 and n>0)
        bits=sum((1<<c)*int(correct[c]) for c in columns)
        db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(source,lag,scope,population,video,image,qr,int(qids[qr]),p,n,int(correct[0]),bits))
        for subset in ("all_nonempty_known_queries",)+( ("positive_and_negative_available_queries",) if p and n else () ):
            for c in columns:
                key="raw_"+("correct" if correct[0] else "wrong")+"_feature_"+("correct" if correct[c] else "wrong")
                cell["top1"][subset][FEATURES[c]][key]+=1
        if p and n:
            signs=np.sign(value[truth==1,None,:][:,:,list(columns)]-value[None,truth==0,:][:,:,list(columns)]).astype(np.int8)+1
            signs=signs.reshape(-1,len(list(columns)))
            for k,c in enumerate(columns):cell["pair_order"][FEATURES[c]]+=np.bincount(3*signs[:,0]+signs[:,k],minlength=9).reshape(3,3)


def score_sealed(out,anchors,sources,parent_out=None):
    """One metadata video and <=5929 score rows at a time; details in SQLite."""
    db=open_database(out/"support.sqlite");cells={};history_stats=defaultdict(Counter);gap_stats=defaultdict(Counter);forecast_stats={};triples={};raw_parity_blocks=0
    current_video=None;metadata=None;maximum_read=0
    with (out/"frame_pairs.jsonl").open() as stream:
        for line in stream:
            receipt=json.loads(line);video=receipt["video_id"];lag=receipt["lag"];source=receipt["source_group"]
            if video!=current_video:
                metadata=json.loads((out/"shards"/f"v{video}.rows.json").read_text())["frames"];current_video=video
            q=metadata[receipt["query_step"]];nq=receipt["query_rows"];nm=receipt["reference_rows"]
            require(len(q["boxes"])==nq and q["image_id"]==receipt["query_image_id"],"Sealed query mapping differs")
            for scope in ("all_candidates","same_predicted_category"):
                for pop in ("all_known_population","common_CV_covered_population"):
                    key=(source,lag,scope,pop)
                    if key not in cells:cells[key]=fresh_cell()
            if receipt["reference_step"] is None:continue
            r=metadata[receipt["reference_step"]]
            require(len(r["boxes"])==nm and r["image_id"]==receipt["reference_image_id"] and receipt["candidates"]==nq*nm,"Sealed reference mapping differs")
            gap_stats[(source,lag)][receipt["original_frame_gap"]]+=1
            history_stats[(source,lag)].update(r["history_reason"])
            path=out/receipt["score_file"];saved=np.load(path,mmap_mode="r",allow_pickle=False)
            block=saved[receipt["offset"]:receipt["offset"]+receipt["candidates"]].copy().reshape(nq,nm,2);del saved
            if parent_out is not None:
                parent=np.load(parent_out/receipt["score_file"],mmap_mode="r",allow_pickle=False)
                expected=parent[receipt["offset"]:receipt["offset"]+receipt["candidates"],0].reshape(nq,nm)
                require(np.array_equal(block[:,:,0],expected),"Full raw score block differs from pinned parent")
                del parent;raw_parity_blocks+=1
            accumulate_forecast(forecast_stats,(source,lag),receipt["forecast"],nq)
            maximum_read=max(maximum_read,nq*nm)
            def ids(frame):
                result=[]
                for row in range(len(frame["boxes"])):
                    value=anchors.get((frame["image_id"],row))
                    require(value is None or value[0]==video,"Manual identity video scope changed")
                    result.append(-1 if value is None else value[1])
                return np.asarray(result,np.int64)
            qids=ids(q);rids=ids(r);cv=np.array([x is not None for x in r["previous_times"]],bool)
            same=np.asarray(q["predicted_categories"])[:,None]==np.asarray(r["predicted_categories"])[None,:]
            for scope,hard in (("all_candidates",np.ones((nq,nm),bool)),("same_predicted_category",same)):
                for pop,mask in (("all_known_population",hard),("common_CV_covered_population",hard&cv[None,:])):
                    update_cell(cells[(source,lag,scope,pop)],block,mask,qids,rids,np.arange(nm),source,lag,scope,pop,video,q["image_id"],db)
                    update_triples(triples,(source,lag,scope,pop),mask,qids,rids,q,r,anchors,video)
            db.commit()
    # Retain zero-support original source/lag/scope cells explicitly.
    for source in sources:
        for lag in LAGS:
            for scope in ("all_candidates","same_predicted_category"):
                for pop in ("all_known_population","common_CV_covered_population"):
                    key=(source,lag,scope,pop)
                    if key not in cells:cells[key]=fresh_cell()
    return db,cells,history_stats,gap_stats,maximum_read,forecast_stats,triples,raw_parity_blocks


def finalized_cell(cell,population):
    result={k:v for k,v in cell.items() if k not in ("top1","pair_order")}
    features=FEATURES
    result["top1"]={}
    for subset,table in cell["top1"].items():
        result["top1"][subset]={}
        for feature in features:
            counts=dict(table[feature]);counts.update(repair_raw_error=counts["raw_wrong_feature_correct"],break_raw_correct=counts["raw_correct_feature_wrong"],
                raw_correct=counts["raw_correct_feature_wrong"]+counts["raw_correct_feature_correct"],
                feature_correct=counts["raw_wrong_feature_correct"]+counts["raw_correct_feature_correct"])
            result["top1"][subset][feature]=counts
    result["positive_negative_pair_order"]={feature:dict(comparisons=int(cell["pair_order"][feature].sum()),
        raw_to_feature_loss_tie_win_matrix=cell["pair_order"][feature].tolist()) for feature in features}
    return result


def cross_lag(db,source,scope,population):
    where="a.scope=? AND a.population=? AND a.lag=1 AND b.lag=4 AND a.p>0 AND a.n>0 AND b.p>0 AND b.n>0"
    args=[scope,population]
    if source is not None:where+=" AND a.source=?";args.append(source)
    row=db.execute("SELECT COUNT(*),SUM(a.raw_correct=0),SUM(b.raw_correct=0),SUM(a.raw_correct=0 AND b.raw_correct=0),SUM((a.correct_bits&2)=0),SUM((b.correct_bits&2)=0),SUM((a.correct_bits&2)=0 AND (b.correct_bits&2)=0) FROM query_support a JOIN query_support b ON a.video=b.video AND a.image=b.image AND a.qrow=b.qrow AND a.scope=b.scope AND a.population=b.population WHERE "+where,args).fetchone()
    return dict(zip(("shared_hard_queries","raw_error_lag1","raw_error_lag4","raw_error_both","angular_error_lag1","angular_error_lag4","angular_error_both"),[int(x or 0) for x in row]))

def empty_forecast():
    return dict(reference_exposures=0,candidate_weight=0,masks=Counter(),candidate_weighted_masks=Counter(),
        distributions={key:dict(boundaries=edges,bins=[0]*(len(edges)+1),count=0,total=0.,minimum=None,maximum=None)
            for key,edges in BOUNDARIES.items()})


def accumulate_forecast(stats,key,diag,nq):
    cell=stats.setdefault(key,empty_forecast());n=len(diag["last_zero_norm"])
    cell["reference_exposures"]+=n;cell["candidate_weight"]+=n*nq
    for name,values in diag.items():
        if name in ("history_angle_radians","extrapolation_ratio"):continue
        count=int(np.count_nonzero(values));cell["masks"][name]+=count;cell["candidate_weighted_masks"][name]+=nq*count
    # rho/angle include valid geometric-history nonzero-previous rows even antipodal;
    # missing/zero-last fallback zero placeholders are not artificial observations.
    valid=~np.asarray(diag["missing_or_zero_history"],bool)&~np.asarray(diag["last_zero_norm"],bool)
    for key,name in (("rho","extrapolation_ratio"),("history_angle_radians","history_angle_radians")):
        values=np.asarray(diag[name])[valid];dist=cell["distributions"][key]
        if not len(values):continue
        counts=np.bincount(np.searchsorted(dist["boundaries"],values,side="right"),minlength=len(dist["bins"]))
        dist["bins"]=[a+int(b) for a,b in zip(dist["bins"],counts)]
        dist["count"]+=len(values);dist["total"]+=float(values.sum())
        dist["minimum"]=float(values.min()) if dist["minimum"] is None else min(dist["minimum"],float(values.min()))
        dist["maximum"]=float(values.max()) if dist["maximum"] is None else max(dist["maximum"],float(values.max()))


def new_triples():
    return dict(counts=Counter(),row_triplets=set(),GT_ID_triplets=set(),videos=set(),trajectories=set())


def update_triples(tables,key,mask,qids,rids,q,r,anchors,video):
    cell=tables.setdefault(key,new_triples())
    positive=mask&(qids[:,None]>=0)&(rids[None,:]>=0)&(qids[:,None]==rids[None,:])
    for qi,ri in zip(*np.nonzero(positive)):
        cell["counts"]["endpoint_known_same_candidates"]+=1
        loc=r["previous_locators"][ri];past=None if loc is None else anchors.get((loc["image_id"],loc["dump_row"]))
        require(past is None or past[0]==video,"Past anchor video differs")
        category="past_unknown" if past is None else ("past_known_same" if past[1]==rids[ri] else "past_known_different")
        cell["counts"][category]+=1
        if loc is not None:
            cell["row_triplets"].add((video,loc["image_id"],loc["dump_row"],r["image_id"],int(ri),q["image_id"],int(qi)))
            cell["counts"]["available_locator_endpoint_same_candidates"]+=1
        if past is not None:
            cell["GT_ID_triplets"].add((video,int(past[1]),int(rids[ri]),int(qids[qi])))
            cell["videos"].add(video);cell["trajectories"].add((video,int(qids[qi])))


def finalized_triples(cell):
    return dict(**{name:int(cell["counts"][name]) for name in ("endpoint_known_same_candidates","past_unknown","past_known_same","past_known_different","available_locator_endpoint_same_candidates")},
        distinct_immutable_row_triplets=len(cell["row_triplets"]),distinct_video_GT_ID_triplets=len(cell["GT_ID_triplets"]),
        fully_known_triplet_videos=len(cell["videos"]),fully_known_endpoint_GT_trajectories=len(cell["trajectories"]))


def query_metrics_subset(db,source,lag,scope,pop,hard):
    args=[lag,scope,pop];where="lag=? AND scope=? AND population=?"+ (" AND p>0 AND n>0" if hard else "")
    if source is not None:where+=" AND source=?";args.append(source)
    rows=[dict(source=s,video_id=v,gt_track_id=g,hard_queries=n,raw_errors=e,angular_errors=a) for s,v,g,n,e,a in
        db.execute("SELECT source,video,qgt,COUNT(*),SUM(raw_correct=0),SUM((correct_bits&2)=0) FROM query_support WHERE "+where+" GROUP BY source,video,qgt ORDER BY source,video,qgt",args)]
    bysource=defaultdict(list)
    for row in rows:bysource[row["source"]].append(row)
    result=dict(query_GT_error_distribution=rows,query_GT_trajectories=len(rows),hard_query_sources=len(bysource))
    for name,col in (("raw","raw_errors"),("angular","angular_errors")):
        total=sum(r[col] for r in rows);source_errors=[sum(r[col] for r in group) for group in bysource.values()]
        result[name]=dict(errors=total,error_sources=sum(x>0 for x in source_errors),error_GT_trajectories=sum(r[col]>0 for r in rows),
            largest_source_error_share=max(source_errors,default=0)/total if total else None,
            largest_GTtrajectory_error_share=max((r[col] for r in rows),default=0)/total if total else None,
            GTtrajectory_macro_correctness=float(np.mean([1-r[col]/r["hard_queries"] for r in rows])) if rows else None,
            source_GTtrajectory_query_macro_correctness=float(np.mean([np.mean([1-r[col]/r["hard_queries"] for r in group]) for group in bysource.values()])) if bysource else None,
            source_query_macro_correctness=float(np.mean([1-sum(r[col] for r in group)/sum(r["hard_queries"] for r in group) for group in bysource.values()])) if bysource else None)
    return result


def query_metrics(db,source,lag,scope,pop):
    return {subset:query_metrics_subset(db,source,lag,scope,pop,hard) for subset,hard in
        (("all_nonempty_known_queries",False),("positive_and_negative_available_queries",True))}


def summarize(out,db,cells,roles,history,gaps,forecast,triples):
    sources=sorted(s for s,v in roles.items() if v["split"]=="fit");reports={};union={}
    for source in sources:
        report=dict(role=roles[source]["role"],dataset_block=roles[source]["block"],lags={},cross_lag={})
        for lag in LAGS:
            report["lags"][str(lag)]={}
            for scope in SCOPES:
                report["lags"][str(lag)][scope]={}
                for pop in POPS:
                    cell=finalized_cell(cells[(source,lag,scope,pop)],pop)
                    cell["query_macros_and_error_concentration"]=query_metrics(db,source,lag,scope,pop)
                    cell["history_truth_support"]=finalized_triples(triples.get((source,lag,scope,pop),new_triples()))
                    report["lags"][str(lag)][scope][pop]=cell
            report["lags"][str(lag)]["reference_history_reason_counts"]=dict(history[(source,lag)])
            report["lags"][str(lag)]["original_frame_gap_per_frame_pair_counter"]={str(k):v for k,v in sorted(gaps[(source,lag)].items())}
            report["lags"][str(lag)]["forecast_diagnostics"]=forecast.get((source,lag),empty_forecast())
        report["cross_lag"]={scope:{pop:cross_lag(db,source,scope,pop) for pop in POPS} for scope in SCOPES}
        reports[source]=report
    for lag in LAGS:
        union[str(lag)]={}
        for scope in SCOPES:
            union[str(lag)][scope]={}
            for pop in POPS:
                combined=fresh_cell();triple=new_triples()
                for source in sources:
                    cell=cells[(source,lag,scope,pop)]
                    for key in combined:
                        if key=="top1":
                            for subset in combined[key]:
                                for feature in FEATURES:
                                    for count in combined[key][subset][feature]:combined[key][subset][feature][count]+=cell[key][subset][feature][count]
                        elif key=="pair_order":
                            for feature in FEATURES:combined[key][feature]+=cell[key][feature]
                        else:combined[key]+=cell[key]
                    st=triples.get((source,lag,scope,pop),new_triples())
                    triple["counts"].update(st["counts"])
                    for k in ("row_triplets","GT_ID_triplets","videos","trajectories"):triple[k].update(st[k])
                result=finalized_cell(combined,pop)
                result["sources_with_known_pairs"]=sum(cells[(s,lag,scope,pop)]["known_pairs"]>0 for s in sources)
                result["sources_with_hard_queries"]=sum(cells[(s,lag,scope,pop)]["hard_queries"]>0 for s in sources)
                result["sources_with_raw_hard_errors"]=sum(cells[(s,lag,scope,pop)]["top1"]["positive_and_negative_available_queries"][FEATURES[0]]["raw_wrong_feature_wrong"]>0 for s in sources)
                result["query_macros_and_error_concentration"]=query_metrics(db,None,lag,scope,pop)
                result["history_truth_support"]=dict(finalized_triples(triple),fully_known_triplet_sources=sum(bool(triples.get((s,lag,scope,pop),new_triples())["GT_ID_triplets"]) for s in sources))
                union[str(lag)][scope][pop]=result
    return dict(micro_union=union,per_source=reports,cross_lag={scope:{pop:cross_lag(db,None,scope,pop) for pop in POPS} for scope in SCOPES},
        training_status="NO_TRAINING_OR_PARAMETER_SELECTION",history_truth_units="Candidates are endpoint-known-same QxM rows; distinct row triplets require sealed past locator; GT_ID triplets require all three eligible identities. Unknown stays unknown.")


def baseline_parity(results,parent):
    count_keys=("candidate_pairs","known_pairs","positive_pairs","negative_pairs","unknown_pairs","nonempty_known_queries","positive_available_queries","negative_available_queries","hard_queries")
    checked=0
    for source in [None]+sorted(results["per_source"]):
        a=results["micro_union"] if source is None else results["per_source"][source]["lags"]
        b=parent["results"]["micro_union"] if source is None else parent["results"]["per_source"][source]["lags"]
        for lag in LAGS:
            for scope in SCOPES:
                for pop in POPS:
                    x=a[str(lag)][scope][pop];y=b[str(lag)][scope][pop]
                    require(all(x[k]==y[k] for k in count_keys),"Parent raw population count differs")
                    for subset in x["top1"]:
                        require(x["top1"][subset][FEATURES[0]]==y["top1"][subset][FEATURES[0]],"Parent raw top1 differs")
                    require(x["positive_negative_pair_order"][FEATURES[0]]==y["positive_negative_pair_order"][FEATURES[0]],"Parent raw pair order differs")
                    checked+=1
    require(results["cross_lag"]["same_predicted_category"]["common_CV_covered_population"]["shared_hard_queries"]==1331,"Primary common-query support changed")
    return dict(status="PASS_EXACT_PARENT_RAW_SCORES_COUNTS_TOP1",source_union_cells=checked,primary_shared_hard_queries=1331)


def validate_population(roles,video_source,frames,population):
    fit_sources=sorted(k for k,v in roles.items() if v["split"]=="fit")
    selected=set(row["video_id"] for row in frames)
    omitted=sorted(int(video) for video in video_source if int(video) not in selected)
    require(all(video_source[str(video)] in fit_sources for video in selected),"Cached FIT source missing from original sealed roles")
    require(len(fit_sources)==population["sealed_fit_source_groups"] and len(video_source)==population["allocated_fit_videos"] and
        len(frames)==population["selected_frames"] and len(selected)==population["selected_fit_videos"] and
        sum(r["observations"] for r in frames)==population["retained_observations"] and
        omitted==population["uncollected_allocation_video_ids"],"Frozen FIT population changed")
    return [dict(video_id=video,source_group=video_source[str(video)],cached_frames=0,
        source_in_sealed_roles=video_source[str(video)] in fit_sources) for video in omitted]


def authenticate():
    require(sha(PROTOCOL)==PROTOCOL_SHA,"Fixed support protocol changed")
    protocol=json.loads(PROTOCOL.read_text());inputs=dict(protocol["input_pins"])
    for p,digest in inputs.items():require(sha(p)==digest,"Protected input changed: "+p)
    preflight=json.loads(PREFLIGHT.read_text())
    require(preflight["status"]=="PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_IMPLEMENTATION_PREFLIGHT" and
        preflight["script_sha256"]==sha(__file__) and preflight["protocol_sha256"]==PROTOCOL_SHA,"Independent preflight absent/stale")
    roles=json.loads((CONTEXT/"source_roles.json").read_text())["groups"]
    fit_sources=sorted(k for k,v in roles.items() if v["split"]=="fit")
    require(all(roles[source]["role"] in ("mean_role","pair_role") for source in fit_sources),"Unexpected FIT role")
    plan=json.loads((STAGE/"full_collection_plan.json").read_text())
    video_source={str(row["video_id"]):row["source_group"] for row in plan["video_records"] if row["split"]=="fit"}
    inventory=json.loads((RUN/"dump_inventory.json").read_text());frames=[]
    with (RUN/"frame_execution.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line)
            if str(row["video_id"]) not in video_source:continue
            p=Path(row["npz"]);require(p.parent==RUN/"dump" and row["npz_sha256"]==inventory[p.name]==sha(p),"FIT cache changed")
            inputs[str(p)]=inventory[p.name];frames.append(row)
    population=protocol["population"]
    validate_population(roles,video_source,frames,population)
    inputs[str(PROTOCOL)]=PROTOCOL_SHA;inputs[str(PREFLIGHT)]=sha(PREFLIGHT);inputs[str(Path(__file__).resolve())]=sha(__file__)
    return protocol,roles,fit_sources,video_source,frames,inputs


def execute():
    started=time.monotonic();protocol,roles,sources,video_source,frames,inputs=authenticate();authenticated=time.monotonic()
    out=Path(protocol["output_directory"]);require(not out.exists(),"Preserve existing angular output");out.mkdir()
    core=module(CORE,"angular_control_core")
    generated=generate(out,frames,video_source,sources,core,load_frame,protocol["resource_bounds"])
    require(generated["candidates_by_lag"]==protocol["resource_bounds"]["candidates_by_lag"],"Fixed candidate counts changed")
    write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION",**generated,
        script_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,input_sha256=inputs,
        features=list(FEATURES),supervised_payload_or_helper_opened=False,unknowns_or_old_IDs_filtered=False))
    sealed=time.monotonic();anchors=join_identities(out,roles,video_source,inputs);joined=time.monotonic()
    db,cells,history,gaps,maxread,forecast,triples,parityblocks=score_sealed(out,anchors,sources,PARENT)
    results=summarize(out,db,cells,roles,history,gaps,forecast,triples);db.commit();db.close();scored=time.monotonic()
    parity=baseline_parity(results,json.loads((PARENT/"summary.json").read_text()))
    parity["exact_full_score_blocks"]=parityblocks
    verify_seal(out);require(all(sha(p)==digest for p,digest in inputs.items()),"Protected inputs changed during audit")
    summary=dict(status="COMPLETE_ALL_SEALED_FIT_ANGULAR_CONTROL_PENDING_INDEPENDENT_AUDIT",results=results,baseline_parity=parity,
        protocol_sha256=PROTOCOL_SHA,script_sha256=sha(__file__),input_sha256=inputs,all_input_hashes_verified_before_after=True,
        sealed_files_verified_before_supervised_import_and_after=True,eligible_FIT_manual_anchors=len(anchors),population=protocol["population"],
        generated=generated,maximum_score_rows_read=maxread,uncollected_allocation_videos=validate_population(roles,video_source,frames,protocol["population"]),
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-started,
        stage_seconds=dict(authentication=authenticated-started,generation=sealed-authenticated,seal_verification_and_join=joined-sealed,scoring=scored-joined,final_verification=time.monotonic()-scored),
        output_files_sha256={name:sha(out/name) for name in ("GT_free_candidates_sealed.json","support.sqlite")},
        training_started=False,parameters_or_thresholds_selected=False,GPU_used=False,tune_test_or_classheldout_targets_decoded=False,unknown_is_negative=False,
        interpretation="Fixed ordinary dynamic appearance control only; no novelty, causal, assignment or training-readiness claim. Missing eligible positives do not certify unmatched targets.")
    write(out/"summary.json",summary)
    print(json.dumps(dict(status=summary["status"],candidates=sum(generated["candidates_by_lag"].values()),elapsed_seconds=summary["elapsed_seconds"],baseline_parity=parity)),flush=True)


def fixture(step,n=2,ids=None):
    boxes=np.array([[10+10*step+40*k,10,20+10*step+40*k,20] for k in range(n)],float).reshape(n,4)
    raw=np.zeros((n,256))
    for k in range(n):raw[k,k]=1
    return dict(video_id=1,image_id=100+step,sampled_step=step,original_frame_index=[0,10,35,40,90,100][step],boxes=boxes,
        raw=raw,released_ids=np.array(ids if ids is not None else list(range(n)),np.int64),predicted_categories=np.ones(n,np.int32))


def seal_fixture(out,generated):
    write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION",**generated,
        features=list(FEATURES),supervised_payload_or_helper_opened=False))


def synthetic(round_number):
    require(sha(CORE)==CORE_SHA,"Frozen angular core changed")
    core=module(CORE,"synthetic_angular_core");checks=[]
    bounds=dict(max_frame_rows=8,max_video_frames=6,max_candidates_per_frame_pair=64)
    if round_number==1:
        # Fixed injected 0 -> pi/4 -> pi/2 arc with irregular original times.
        history={};prior=fixture(0,1);prior["raw"][0,:2]=[1,0];attach_history(prior,history)
        last=fixture(1,1);last["raw"][0,:2]=np.sqrt(.5);attach_history(last,history)
        query=fixture(2,1);query["original_frame_index"]=20;query["raw"][0,:2]=[0,1]
        scores,diag=score_block(query,last,core)
        require(abs(scores[0,1]-1)<1e-14 and abs(scores[0,0]-np.sqrt(.5))<1e-14 and diag["directed_arc"]==[True] and diag["extrapolation_ratio"]==[1.],"Injected directional forecast failed")
        # Wrong history with same last and time reverses the predicted turn.
        wrong=last.copy();wrong["previous_raw"]=last["previous_raw"].copy();wrong["previous_raw"][0,:2]=[0,1]
        wrongscore,_=score_block(query,wrong,core)
        require(abs(wrongscore[0,1])<1e-14,"Wrong-history challenge did not alter forecast")
        poison=query.copy();poison["raw"]=np.zeros_like(query["raw"]);poison["boxes"]=np.full_like(query["boxes"],999)
        _,again=score_block(poison,last,core);require(again==diag,"Query appearance/box changed prediction diagnostics")
        checks.append("injected_angular_drift_correct_and_wrong_history_original_time_ratio_query_poison")
        history={};first=fixture(0,5);first["raw"][:]=0;first["raw"][:,0]=1;first["raw"][2]=0;attach_history(first,history)
        second=fixture(1,5);second["raw"][:]=0;second["raw"][0,0]=-1;second["raw"][1,0]=2;second["raw"][2,0]=1;second["raw"][4,0]=1
        second["released_ids"][4]=-1;attach_history(second,history)
        q=fixture(2,1);score,diag=score_block(q,second,core)
        require(diag["antipodal_fallback"]==[True,False,False,False,False] and diag["coincident_static"][1] and diag["previous_zero_norm"][2] and diag["last_zero_norm"][3] and diag["missing_or_zero_history"][4],"Degenerate masks differ")
        require(score[3,1]==0 and np.isfinite(score).all(),"Zero-last forecast not exact zero")
        checks.append("antipodal_coincident_zero_previous_last_and_missing_history_fallback")
        history={};f0=attach_history(fixture(0),history);f1=attach_history(fixture(1),history)
        loc=f1["previous_locators"][0].copy();raw=f1["previous_raw"].copy();f0["raw"][:]=99
        duplicate=attach_history(fixture(2,ids=[0,0]),history);following=attach_history(fixture(3,ids=[0,-1]),history);resume=attach_history(fixture(4,ids=[0,-1]),history)
        require(f1["previous_locators"][0]==loc and np.array_equal(f1["previous_raw"],raw) and loc==dict(image_id=100,dump_row=0,original_frame_index=0),"Past snapshot mutated")
        require(duplicate["history_reason"]==["duplicate_current_id"]*2 and following["history_reason"]==["previous_duplicate_invalid","unassigned_id"] and resume["previous_times"][0]==40,"Unique duplicate/noID resume rule differs")
        checks.append("past_locator_raw_snapshot_future_mutation_duplicate_invalidates_next_unique_then_resumes")
        with tempfile.TemporaryDirectory(prefix="angular_empty_") as temp:
            records=[dict(video_id=1,frame_id=k,observations=0 if k==1 else 2) for k in range(6)]
            out=Path(temp);gen=generate(out,records,{"1":"S"},["S"],core,lambda r:fixture(r["frame_id"],r["observations"],[] if r["observations"]==0 else [0,-1]),bounds)
            receipts=[json.loads(x) for x in (out/"frame_pairs.jsonl").open()]
            require(gen["candidates_by_lag"]=={"1":12,"4":4} and len(receipts)==12 and gen["maximum_feature_ring_frames"]==5 and next(r for r in receipts if r["query_step"]==2 and r["lag"]==1)["candidates"]==0,"Exact lag QxM prefix/empty failed")
        with tempfile.TemporaryDirectory(prefix="angular_allempty_") as temp:
            records=[dict(video_id=1,frame_id=k,observations=0) for k in range(2)];out=Path(temp)
            gen=generate(out,records,{"1":"S"},["S"],core,lambda r:fixture(r["frame_id"],0),bounds)
            require(gen["candidates_by_lag"]=={"1":0,"4":0} and np.load(out/"shards/v1_lag4.scores.npy",mmap_mode="r").shape==(0,2),"All-empty/short zero shard failed")
        checks.append("full_lag_QxM_empty_prefix_unassigned_short_all_empty_zero_length_shard")
    elif round_number==2:
        with tempfile.TemporaryDirectory(prefix="angular_paired_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell()
            scores=np.array([[[.1,.9],[.9,.1]],[[.1,.9],[.9,.1]]]);mask=np.ones((2,2),bool);q=np.array([1,2]);r=np.array([1,2])
            for lag in LAGS:update_cell(cell if lag==1 else fresh_cell(),scores,mask,q,r,np.arange(2),"S",lag,SCOPES[1],POPS[1],1,10,db)
            stats=finalized_cell(cell,POPS[1]);top=stats["top1"]["positive_and_negative_available_queries"][FEATURES[1]]
            require(stats["known_pairs"]==4 and stats["positive_pairs"]==stats["negative_pairs"]==2 and top["repair_raw_error"]==top["break_raw_correct"]==1,"Paired repair/break oracle failed")
            unknown=fresh_cell();update_cell(unknown,scores,mask,np.array([-1,2]),r,np.arange(2),"T",1,SCOPES[0],POPS[0],2,20,db)
            require(unknown["known_pairs"]==2 and unknown["unknown_pairs"]==2 and unknown["negative_pairs"]==1,"Unknown became negative")
            require(cross_lag(db,"S",SCOPES[1],POPS[1])["shared_hard_queries"]==2,"Cross-lag same-query join differs")
            metrics=query_metrics(db,"S",1,SCOPES[1],POPS[1])["positive_and_negative_available_queries"]
            require(metrics["raw"]["source_GTtrajectory_query_macro_correctness"]==.5 and metrics["angular"]["source_GTtrajectory_query_macro_correctness"]==.5,"Source trajectory macro oracle failed")
            require([x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]==["query_support"],"SQLite stores candidate/extra tables")
            db.close()
        checks.append("repair_break_source_GTtrajectory_macro_crosslag_unknown_not_negative_query_only_SQLite")
        history={};previous=attach_history(fixture(0,3),history);reference=attach_history(fixture(1,3),history);query=fixture(2,1)
        # Geometric history remains available even when angular is zero/antipodal.
        reference["raw"][0]=0;reference["raw"][1]=-reference["previous_raw"][1]
        block,diag=score_block(query,reference,core);require(np.isfinite(reference["previous_times"]).all() and diag["last_zero_norm"][0] and diag["antipodal_fallback"][1],"Degeneracy narrowed CV availability")
        qmeta=frame_metadata(attach_history(query,history));rmeta=frame_metadata(reference)
        anchors={(100,0):(1,7),(100,1):(1,9)};triples={}
        update_triples(triples,("S",1,SCOPES[0],POPS[1]),np.ones((1,3),bool),np.array([7]),np.array([7,7,7]),qmeta,rmeta,anchors,1)
        triple=finalized_triples(next(iter(triples.values())))
        require([triple[k] for k in ("endpoint_known_same_candidates","past_known_same","past_known_different","past_unknown","distinct_immutable_row_triplets","distinct_video_GT_ID_triplets")]==[3,1,1,1,3,2],"Triple support unit/count oracle differs")
        checks.append("common_CV_not_narrowed_by_angular_degeneracy_and_three_anchor_same_different_unknown_units")
        with tempfile.TemporaryDirectory(prefix="angular_tie_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell();scores=np.zeros((1,2,2))
            update_cell(cell,scores,np.ones((1,2),bool),np.array([1]),np.array([1,2]),np.array([1,0]),"S",1,SCOPES[0],POPS[0],1,1,db)
            require(cell["top1"]["positive_and_negative_available_queries"][FEATURES[0]]["raw_wrong_feature_wrong"]==1,"Tie was GT favorable")
            db.close()
        checks.append("immutable_lowest_reference_dump_row_tie")
    elif round_number==3:
        roles={"S":dict(split="fit",role="mean_role",block="synthetic"),"ZERO":dict(split="fit",role="pair_role",block="synthetic")}
        with tempfile.TemporaryDirectory(prefix="angular_endtoend_") as temp:
            out=Path(temp)/"out";out.mkdir();local=Path(temp)/"support";local.mkdir()
            records=[dict(video_id=1,frame_id=k,observations=2) for k in range(6)]
            generated=generate(out,records,{"1":"S"},["S","ZERO"],core,lambda r:fixture(r["frame_id"]),bounds);seal_fixture(out,generated)
            identities=[];indices=[]
            for k in range(6):
                for row in range(2):
                    item=dict(observation_id=k*2+row,split="fit",source_group="S",video_id=1,image_id=100+k,sampled_step=k,original_frame_index=[0,10,35,40,90,100][k],local_track_id=row,gt_track_id=row+1,pred_category_id=1,manual_anchor_id=k*2+row,eligible_for_parameter_fit=True)
                    identities.append(dict(item,gt_category_id="CLASS_POISON",error="ERROR_POISON",soft_risk="RISK_POISON",posterior_argmax_category_id="POST_POISON"));indices.append(dict(item,dump_row=row))
            manual=local/"supervised_observations.jsonl";manual.write_text("".join(json.dumps(x)+"\n" for x in identities)+json.dumps(dict(split="tune",gt_track_id="TUNE_POISON"))+"\n")
            index=local/"gt_free_observations.jsonl";index.write_text("".join(json.dumps(x)+"\n" for x in indices))
            write(local/"support_receipt.json",dict(output_files={manual.name:sha(manual)}));write(local/"gt_free_index_manifest.json",dict(files={index.name:sha(index)}))
            global SUPPORT
            saved_support=SUPPORT;saved_loads=json.loads;inputs={}
            def guarded(text,*args,**kwargs):
                require(not any(x in text for x in ("CLASS_POISON","ERROR_POISON","RISK_POISON","POST_POISON","TUNE_POISON")),"Forbidden payload decoded")
                return saved_loads(text,*args,**kwargs)
            SUPPORT=local;json.loads=guarded
            try:anchors=join_identities(out,roles,{"1":"S"},inputs)
            finally:SUPPORT=saved_support;json.loads=saved_loads
            db,cells,history,gaps,maxread,forecast,triples,blocks=score_sealed(out,anchors,["S","ZERO"])
            result=summarize(out,db,cells,roles,history,gaps,forecast,triples);db.close();verify_seal(out)
            require(len(anchors)==12 and maxread==4 and result["micro_union"]["1"][SCOPES[0]][POPS[0]]["candidate_pairs"]==20 and result["per_source"]["ZERO"]["lags"]["4"][SCOPES[1]][POPS[1]]["known_pairs"]==0 and result["cross_lag"][SCOPES[1]][POPS[1]]["shared_hard_queries"]==2,"End-to-end sealed known projection/report differs")
            checks.append("full_temp_generation_global_seal_poison_safe_FIT_join_score_summary_zero_support_source")
            target=out/next(iter(generated["files_sha256"]));target.write_bytes(b"tampered")
            try:join_identities(out,{}, {}, {})
            except ValueError:pass
            else:raise AssertionError("Supervision preceded failed seal")
            target.unlink()
            try:verify_seal(out)
            except FileNotFoundError:pass
            else:raise AssertionError("Accepted missing sealed file")
            checks.append("tampered_and_missing_sealed_file_block_supervision_before_helper_import")
        with tempfile.TemporaryDirectory(prefix="angular_parity_") as temp:
            out=Path(temp)/"new";out.mkdir();parent=Path(temp)/"parent";parent.mkdir();(parent/"shards").mkdir()
            records=[dict(video_id=1,frame_id=k,observations=2) for k in range(6)]
            gen=generate(out,records,{"1":"S"},["S"],core,lambda r:fixture(r["frame_id"]),bounds)
            for lag in LAGS:
                data=np.load(out/"shards"/f"v1_lag{lag}.scores.npy");legacy=np.zeros((len(data),6));legacy[:,0]=data[:,0];np.save(parent/"shards"/f"v1_lag{lag}.scores.npy",legacy)
            result=score_sealed(out,{(100+k,row):(1,row+1) for k in range(6) for row in range(2)},["S"],parent);result[0].close()
            require(result[-1]==7,"Raw parent full block exact parity count differs")
            array=np.load(parent/"shards/v1_lag1.scores.npy",mmap_mode="r+");array[0,0]+=1;array.flush();del array
            # New report directory/database to exercise rejected mismatched raw baseline.
            (out/"support.sqlite").unlink()
            try:score_sealed(out,{},["S"],parent)
            except ValueError:pass
            else:raise AssertionError("Accepted parent raw mismatch")
        checks.append("raw_parent_all_frame_block_exact_parity_and_mismatch_rejection")
        roles={"S":dict(split="fit",role="mean_role")};video_source={"1":"S","2":"uncollected/no_role"}
        population=dict(sealed_fit_source_groups=1,allocated_fit_videos=2,selected_frames=1,selected_fit_videos=1,retained_observations=2,uncollected_allocation_video_ids=[2])
        require(validate_population(roles,video_source,[dict(video_id=1,observations=2)],population)[0]["source_in_sealed_roles"] is False,"Uncollected allocation role gate differs")
        with tempfile.TemporaryDirectory(prefix="angular_bound_") as temp:
            try:generate(Path(temp),[dict(video_id=1,frame_id=0,observations=2)],{"1":"S"},["S"],core,lambda r:fixture(0),dict(bounds,max_frame_rows=1))
            except ValueError:pass
            else:raise AssertionError("Resource bound ignored")
        checks.append("frozen_sealed_population_and_declared_resource_bound_rejection")
    else:raise ValueError("Exactly three rounds")
    require(len(checks)>=2,"Each round needs two independent checks")
    return dict(status="PASS_ANGULAR_APPEARANCE_CONTROL_SYNTHETIC_ROUND",round=round_number,checks=checks,
        script_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,core_sha256=sha(CORE),real_cache_or_supervised_payload_opened=False,
        training_started=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--synthetic-round",type=int,choices=(1,2,3));parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    require(not(args.execute and args.synthetic_round),"Synthetic and real modes exclusive")
    if args.synthetic_round:
        result=synthetic(args.synthetic_round)
        if args.synthetic_out:write(args.synthetic_out,result)
        print(json.dumps(result,indent=2))
    else:require(args.execute,"Use explicit --execute only after independent preflight");execute()


if __name__=="__main__":main()
