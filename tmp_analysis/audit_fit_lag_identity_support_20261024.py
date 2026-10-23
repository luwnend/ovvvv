#!/usr/bin/env python3
"""Streaming all-sealed-FIT lag{1,4} candidate support. No training or tuning.

Real execution belongs to root after independent preflight. Synthetic mode
opens only temporary synthetic data and the reviewed pure geometry module.
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
PROTOCOL=ROOT/"tmp_analysis/fit_lag_identity_support_protocol_20261024.json"
PROTOCOL_SHA="dd84da8fd028af36c223f42a30d19f9f23725eb790e60aaa668700c80a0806ad"
PREFLIGHT=ROOT/"tmp_analysis/fit_lag_identity_support_independent_preflight_20261024.json"
GEOMETRY=ROOT/"tmp_analysis/relative_track_geometry.py"
HELPER=ROOT/"tmp_analysis/build_fit_identity_edge_support.py"
FEATURES=("raw_appearance_cosine","absolute_xyxy_cosine","box_iou","negative_normalized_displacement_norm",
          "negative_normalized_CV_innovation_norm","semantic_embedding_cosine")
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
    n=len(frame["boxes"]);counts=Counter(int(x) for x in frame["released_ids"] if x>=0)
    previous=np.full((n,4),np.nan);times=np.full(n,np.nan);reason=[]
    for k,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity<0:reason.append("unassigned_id");continue
        if counts[identity]>1:reason.append("duplicate_current_id");continue
        if identity not in history:reason.append("no_previous_occurrence");continue
        if history[identity] is None:reason.append("previous_duplicate_invalid");continue
        box,t=history[identity]
        require(t<frame["original_frame_index"],"Nonpast history")
        previous[k]=box;times[k]=t;reason.append("available")
    for k,identity in enumerate(frame["released_ids"]):
        identity=int(identity)
        if identity>=0:history[identity]=None if counts[identity]>1 else (frame["boxes"][k].copy(),frame["original_frame_index"])
    frame.update(previous_boxes=previous,previous_times=times,history_reason=reason)
    return frame


def score_block(query,reference,geometry):
    nq,nm=len(query["boxes"]),len(reference["boxes"])
    require(query["video_id"]==reference["video_id"] and query["sampled_step"]>reference["sampled_step"] and
        query["original_frame_index"]>reference["original_frame_index"],"Invalid frame-pair timeline")
    geom=geometry.pair_geometry(query["boxes"],reference["boxes"],query["original_frame_index"],
        np.full(nm,reference["original_frame_index"]),reference["previous_boxes"],reference["previous_times"])
    lower=np.maximum(query["boxes"][:,None,:2],reference["boxes"][None,:,:2])
    upper=np.minimum(query["boxes"][:,None,2:],reference["boxes"][None,:,2:])
    intersection=np.prod(np.maximum(upper-lower,0),axis=2)
    qarea=np.prod(query["boxes"][:,2:]-query["boxes"][:,:2],axis=1)
    rarea=np.prod(reference["boxes"][:,2:]-reference["boxes"][:,:2],axis=1)
    scores=np.stack((normalized(query["raw"])@normalized(reference["raw"]).T,
        normalized(query["boxes"])@normalized(reference["boxes"]).T,
        intersection/(qarea[:,None]+rarea[None,:]-intersection),
        -np.linalg.norm(geom["normalized_displacement"],axis=2),
        -np.linalg.norm(geom["normalized_CV_innovation"],axis=2),
        normalized(query["semantic"])@normalized(reference["semantic"]).T),axis=2).reshape(nq*nm,6)
    valid=np.isfinite(reference["previous_times"])
    require(np.isfinite(scores[:,[0,1,2,3,5]]).all() and
        np.isfinite(scores.reshape(nq,nm,6)[:,valid,4]).all() and
        np.isnan(scores.reshape(nq,nm,6)[:,~valid,4]).all(),"CV score/mask disagrees")
    return scores


def frame_metadata(frame):
    return dict(video_id=int(frame["video_id"]),image_id=int(frame["image_id"]),sampled_step=int(frame["sampled_step"]),
        original_frame_index=int(frame["original_frame_index"]),boxes=frame["boxes"].tolist(),
        released_ids=frame["released_ids"].tolist(),predicted_categories=frame["predicted_categories"].tolist(),
        previous_boxes=[[float(x) for x in row] if np.isfinite(row).all() else None for row in frame["previous_boxes"]],
        previous_times=[int(x) if np.isfinite(x) else None for x in frame["previous_times"]],history_reason=frame["history_reason"])


def load_frame(record):
    # Load only declared small fields; posterior payloads are not materialized.
    with np.load(record["npz"],allow_pickle=False) as stored:
        require(all(int(stored[key])==record[field] for key,field in
            (("video_id","video_id"),("image_id","image_id"),("frame_id","frame_id"),("frame_index","frame_index"))),"Frame provenance changed")
        box=stored["det_bboxes"].astype(np.float64);raw=stored["track_feats"].astype(np.float64)
        semantic=stored["cls_feats"].astype(np.float64);ids=stored["track_ids"].astype(np.int64)
        categories=stored["det_labels"].astype(np.int32)+1
    n=len(box)
    require(box.shape==(n,5) and raw.shape==(n,256) and semantic.shape==(n,512) and ids.shape==categories.shape==(n,) and
        n==record["observations"] and np.isfinite(box).all() and np.all(box[:,2:4]>box[:,:2]),"Invalid retained rows")
    return dict(video_id=record["video_id"],image_id=record["image_id"],sampled_step=record["frame_id"],
        original_frame_index=record["frame_index"],boxes=box[:,:4],raw=raw,semantic=semantic,
        released_ids=ids,predicted_categories=categories)


def generate(out,frames,video_source,sources,geometry,loader,bounds):
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
            maps={lag:np.lib.format.open_memmap(paths[lag],mode="w+",dtype=np.float64,shape=(sizes[lag],6)) for lag in LAGS}
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
                        block=score_block(frame,reference,geometry);require(block.shape==(count,6),"Incomplete QxM")
                        maps[lag][offsets[lag]:offsets[lag]+count]=block
                    receipt=dict(video_id=video,source_group=video_source[str(video)],lag=lag,query_step=frame["sampled_step"],
                        query_image_id=frame["image_id"],reference_step=None if reference is None else reference["sampled_step"],
                        reference_image_id=None if reference is None else reference["image_id"],query_rows=nq,reference_rows=nm,
                        original_frame_gap=None if reference is None else frame["original_frame_index"]-reference["original_frame_index"],
                        offset=offsets[lag],candidates=count,score_file=str(paths[lag].relative_to(out)))
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
    require(seal["status"]=="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION","No global GT-free seal")
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
    db.execute("CREATE TABLE trajectories(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,gt INTEGER,PRIMARY KEY(lag,scope,population,video,gt))")
    db.execute("CREATE TABLE identity_pairs(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,qgt INTEGER,rgt INTEGER,pairs INTEGER,PRIMARY KEY(lag,scope,population,video,qgt,rgt))")
    db.execute("CREATE INDEX query_source ON query_support(source,lag,scope,population,video,qgt)")
    db.execute("CREATE INDEX trajectory_source ON trajectories(source,lag,scope,population)")
    db.execute("CREATE INDEX identity_pair_source ON identity_pairs(source,lag,scope,population)")
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
    columns=range(6) if population=="common_CV_covered_population" else (0,1,2,3,5)
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
        identities=set([int(qids[qr])]+rids[refs].tolist())
        db.executemany("INSERT OR IGNORE INTO trajectories VALUES(?,?,?,?,?,?)",[(source,lag,scope,population,video,identity) for identity in identities])
        pair_counts=Counter(int(x) for x in rids[refs])
        db.executemany("INSERT INTO identity_pairs VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(lag,scope,population,video,qgt,rgt) DO UPDATE SET pairs=pairs+excluded.pairs",
            [(source,lag,scope,population,video,int(qids[qr]),rid,count) for rid,count in pair_counts.items()])
        for subset in ("all_nonempty_known_queries",)+( ("positive_and_negative_available_queries",) if p and n else () ):
            for c in columns:
                key="raw_"+("correct" if correct[0] else "wrong")+"_feature_"+("correct" if correct[c] else "wrong")
                cell["top1"][subset][FEATURES[c]][key]+=1
        if p and n:
            signs=np.sign(value[truth==1,None,:][:,:,list(columns)]-value[None,truth==0,:][:,:,list(columns)]).astype(np.int8)+1
            signs=signs.reshape(-1,len(list(columns)))
            for k,c in enumerate(columns):cell["pair_order"][FEATURES[c]]+=np.bincount(3*signs[:,0]+signs[:,k],minlength=9).reshape(3,3)


def score_sealed(out,anchors,sources):
    """One metadata video and <=5929 score rows at a time; details in SQLite."""
    db=open_database(out/"support.sqlite");cells={};history_stats=defaultdict(Counter);gap_stats=defaultdict(Counter)
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
            block=saved[receipt["offset"]:receipt["offset"]+receipt["candidates"]].copy().reshape(nq,nm,6);del saved
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
            db.commit()
    # Retain zero-support original source/lag/scope cells explicitly.
    for source in sources:
        for lag in LAGS:
            for scope in ("all_candidates","same_predicted_category"):
                for pop in ("all_known_population","common_CV_covered_population"):
                    key=(source,lag,scope,pop)
                    if key not in cells:cells[key]=fresh_cell()
    return db,cells,history_stats,gap_stats,maximum_read


def finalized_cell(cell,population):
    result={k:v for k,v in cell.items() if k not in ("top1","pair_order")}
    features=FEATURES if population=="common_CV_covered_population" else tuple(f for k,f in enumerate(FEATURES) if k!=4)
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
    row=db.execute("SELECT COUNT(*),SUM(a.raw_correct=0),SUM(b.raw_correct=0),SUM(a.raw_correct=0 AND b.raw_correct=0) FROM query_support a JOIN query_support b ON a.video=b.video AND a.image=b.image AND a.qrow=b.qrow AND a.scope=b.scope AND a.population=b.population WHERE a.source=? AND a.scope=? AND a.population=? AND a.lag=1 AND b.lag=4 AND a.p>0 AND a.n>0 AND b.p>0 AND b.n>0",(source,scope,population)).fetchone()
    return dict(shared_hard_queries=int(row[0]),raw_error_lag1=int(row[1] or 0),raw_error_lag4=int(row[2] or 0),raw_error_both=int(row[3] or 0))


def summarize(out,db,cells,roles,history,gaps):
    sources=sorted(source for source,value in roles.items() if value["split"]=="fit")
    reports={};folds={};hard_sources=set();error_sources=set();cross_sources=set()
    primary=("same_predicted_category","common_CV_covered_population")
    for source in sources:
        report=dict(role=roles[source]["role"],dataset_block=roles[source]["block"],lags={},cross_lag={})
        for lag in LAGS:
            report["lags"][str(lag)]={}
            for scope in ("all_candidates","same_predicted_category"):
                report["lags"][str(lag)][scope]={}
                for pop in ("all_known_population","common_CV_covered_population"):
                    stats=finalized_cell(cells[(source,lag,scope,pop)],pop)
                    stats["official_GT_trajectories"]=db.execute("SELECT COUNT(*) FROM trajectories WHERE source=? AND lag=? AND scope=? AND population=?",(source,lag,scope,pop)).fetchone()[0]
                    stats["distinct_ordered_identity_pairs"]=db.execute("SELECT COUNT(*) FROM identity_pairs WHERE source=? AND lag=? AND scope=? AND population=?",(source,lag,scope,pop)).fetchone()[0]
                    stats["query_GT_error_distribution"]=[dict(video_id=v,gt_track_id=g,hard_queries=n,raw_errors=e) for v,g,n,e in
                        db.execute("SELECT video,qgt,COUNT(*),SUM(raw_correct=0) FROM query_support WHERE source=? AND lag=? AND scope=? AND population=? AND p>0 AND n>0 GROUP BY video,qgt ORDER BY video,qgt",(source,lag,scope,pop))]
                    report["lags"][str(lag)][scope][pop]=stats
            report["lags"][str(lag)]["reference_history_reason_counts"]=dict(history[(source,lag)])
            report["lags"][str(lag)]["original_frame_gap_per_frame_pair_counter"]={str(k):v for k,v in sorted(gaps[(source,lag)].items())}
        for scope in ("all_candidates","same_predicted_category"):
            report["cross_lag"][scope]={pop:cross_lag(db,source,scope,pop) for pop in ("all_known_population","common_CV_covered_population")}
        reports[source]=report
        if any(cells[(source,lag,*primary)]["hard_queries"]>0 for lag in LAGS):hard_sources.add(source)
        if any(cells[(source,lag,*primary)]["top1"]["positive_and_negative_available_queries"][FEATURES[0]]["raw_wrong_feature_wrong"]>0 for lag in LAGS):error_sources.add(source)
        if report["cross_lag"][primary[0]][primary[1]]["shared_hard_queries"]>0:cross_sources.add(source)
    for source in sources:
        folds[source]={}
        for lag in LAGS:
            target=cells[(source,lag,*primary)];others=[cells[(other,lag,*primary)] for other in sources if other!=source]
            train=dict(positive_pairs=sum(c["positive_pairs"] for c in others),negative_pairs=sum(c["negative_pairs"] for c in others),hard_queries=sum(c["hard_queries"] for c in others))
            evaluation={k:target[k] for k in ("positive_pairs","negative_pairs","hard_queries")}
            folds[source][str(lag)]=dict(training_support=train,evaluation_support=evaluation,
                necessary_positive_negative_support=all(x[k]>0 for x in (train,evaluation) for k in ("positive_pairs","negative_pairs","hard_queries")))
    union={}
    for lag in LAGS:
        union[str(lag)]={}
        for scope in ("all_candidates","same_predicted_category"):
            union[str(lag)][scope]={}
            for pop in ("all_known_population","common_CV_covered_population"):
                combined=fresh_cell()
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
                result=finalized_cell(combined,pop)
                result["sources_with_known_pairs"]=sum(cells[(source,lag,scope,pop)]["known_pairs"]>0 for source in sources)
                result["sources_with_hard_queries"]=sum(cells[(source,lag,scope,pop)]["hard_queries"]>0 for source in sources)
                result["sources_with_raw_hard_errors"]=sum(cells[(source,lag,scope,pop)]["top1"]["positive_and_negative_available_queries"][FEATURES[0]]["raw_wrong_feature_wrong"]>0 for source in sources)
                result["official_GT_trajectories"]=db.execute("SELECT COUNT(*) FROM trajectories WHERE lag=? AND scope=? AND population=?",(lag,scope,pop)).fetchone()[0]
                union[str(lag)][scope][pop]=result
    conditions=dict(at_least_two_hard_query_sources=len(hard_sources)>=2,at_least_two_raw_error_sources=len(error_sources)>=2,
        cross_lag_shared_hard_query_exists=bool(cross_sources),
        two_lag_source_blocked_positive_negative_fold_exists=any(all(v["necessary_positive_negative_support"] for v in f.values()) for f in folds.values()))
    return dict(micro_union=union,per_source=reports,source_held_out_support=folds,
        necessary_support_gate=dict(status="HOLD_PENDING_SEPARATE_TRAINING_PROTOCOL",necessary_conditions=conditions,
            all_necessary_conditions_met=all(conditions.values()),blockers=[k for k,v in conditions.items() if not v],
            hard_query_sources=sorted(hard_sources),raw_error_sources=sorted(error_sources),cross_lag_shared_hard_sources=sorted(cross_sources),
            sufficient_training_support_established=False,training_protocol_frozen=False))


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
    require(preflight["status"]=="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_IMPLEMENTATION_PREFLIGHT" and
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
    out=Path(protocol["output_directory"]);require(not out.exists(),"Preserve existing support output");out.mkdir()
    geometry=module(GEOMETRY,"fit_lag_relative_geometry")
    generated=generate(out,frames,video_source,sources,geometry,load_frame,protocol["resource_bounds"])
    require(generated["candidates_by_lag"]==protocol["resource_bounds"]["candidates_by_lag"],"Fixed candidate counts changed")
    write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION",**generated,
        script_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,input_sha256=inputs,
        features=list(FEATURES),supervised_payload_or_helper_opened=False,unknowns_or_old_IDs_filtered=False))
    sealed=time.monotonic();anchors=join_identities(out,roles,video_source,inputs);joined=time.monotonic()
    db,cells,history,gaps,maximum_read=score_sealed(out,anchors,sources)
    results=summarize(out,db,cells,roles,history,gaps);db.commit();db.close();scored=time.monotonic()
    verify_seal(out)
    require(all(sha(p)==digest for p,digest in inputs.items()),"Protected inputs changed during audit")
    summary=dict(status="COMPLETE_ALL_SEALED_FIT_LAG_SUPPORT_PENDING_INDEPENDENT_AUDIT",results=results,
        protocol_sha256=PROTOCOL_SHA,script_sha256=sha(__file__),input_sha256=inputs,all_input_hashes_verified_before_after=True,
        sealed_files_verified_before_supervised_import_and_after=True,eligible_FIT_manual_anchors=len(anchors),
        population=protocol["population"],generated=generated,maximum_score_rows_read=maximum_read,
        uncollected_allocation_videos=validate_population(roles,video_source,frames,protocol["population"]),
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_seconds=time.monotonic()-started,stage_seconds=dict(authentication=authenticated-started,generation=sealed-authenticated,
            seal_verification_and_join=joined-sealed,scoring=scored-joined,final_verification=time.monotonic()-scored),
        output_files_sha256={name:sha(out/name) for name in ("GT_free_candidates_sealed.json","support.sqlite")},
        training_started=False,parameters_or_thresholds_selected=False,GPU_used=False,tune_test_or_classheldout_targets_decoded=False,
        unknown_is_negative=False,interpretation="Necessary support audit only; sparse-known conditional ranking, source/identity-correlated observations. No training readiness, assignment improvement, novelty or five-mode rescue.")
    write(out/"summary.json",summary)
    print(json.dumps(dict(status=summary["status"],candidates=sum(map(int,generated["candidates_by_lag"].values())),elapsed_seconds=summary["elapsed_seconds"],support_gate=results["necessary_support_gate"])),flush=True)


def fixture(step,n=2,ids=None):
    boxes=np.array([[10+10*step+40*k,10,20+10*step+40*k,20] for k in range(n)],float).reshape(n,4)
    raw=np.zeros((n,256));semantic=np.zeros((n,512))
    for k in range(n):raw[k,k]=1;semantic[k,k]=1
    return dict(video_id=1,image_id=100+step,sampled_step=step,original_frame_index=[0,10,35,40,90,100][step],boxes=boxes,
        raw=raw,semantic=semantic,released_ids=np.array(ids if ids is not None else list(range(n)),np.int64),predicted_categories=np.ones(n,np.int32))


def synthetic(round_number):
    require(sha(GEOMETRY)=="c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a","Geometry changed")
    geometry=module(GEOMETRY,"synthetic_fit_lag_geometry");checks=[]
    bounds=dict(max_frame_rows=2,max_video_frames=6,max_candidates_per_frame_pair=4)
    if round_number==1:
        records=[dict(video_id=1,frame_id=k,observations=0 if k==1 else 2) for k in range(6)]
        with tempfile.TemporaryDirectory(prefix="fit_lag_generation_") as temp:
            out=Path(temp);generated=generate(out,records,{"1":"S"},["S"],geometry,
                lambda r:fixture(r["frame_id"],r["observations"],ids=[] if not r["observations"] else [0,-1]),bounds)
            receipts=[json.loads(line) for line in (out/"frame_pairs.jsonl").open()]
            require(generated["candidates_by_lag"]=={"1":12,"4":4} and len(receipts)==12 and
                next(r for r in receipts if r["query_step"]==2 and r["lag"]==1)["candidates"]==0 and
                next(r for r in receipts if r["query_step"]==4 and r["lag"]==4)["original_frame_gap"]==90,"Lag/prefix/empty/full candidate mapping differs")
            require(generated["maximum_feature_ring_frames"]==5 and generated["maximum_candidate_block"]==4,"Bounded generation differs")
        checks.append("full_lag1_lag4_QxM_prefix_empty_unassigned_irregular_gap_and_resource_bounds")
        with tempfile.TemporaryDirectory(prefix="fit_lag_short_empty_") as temp:
            short=[dict(video_id=1,frame_id=k,observations=0) for k in range(2)]
            out=Path(temp);empty=generate(out,short,{"1":"S"},["S"],geometry,lambda r:fixture(r["frame_id"],0),bounds)
            require(empty["candidates_by_lag"]=={"1":0,"4":0} and empty["frame_pair_receipts"]==4 and
                np.load(out/"shards/v1_lag4.scores.npy",mmap_mode="r").shape==(0,6),"Short/empty zero-length shard failed")
        checks.append("short_video_all_empty_zero_length_lag4_shard_and_complete_receipts")
        history={};attach_history(fixture(0),history);duplicate=attach_history(fixture(1,ids=[0,0]),history)
        following=attach_history(fixture(2,ids=[0,-1]),history);resumed=attach_history(fixture(3,ids=[0,-1]),history)
        require(duplicate["history_reason"]==["duplicate_current_id"]*2 and following["history_reason"]==["previous_duplicate_invalid","unassigned_id"] and
            resumed["previous_times"][0]==35 and np.isnan(resumed["previous_times"][1]),"Past-only duplicate/no-ID recovery differs")
        checks.append("duplicate_current_history_next_unique_invalid_and_later_resume_no_ID_never_history")
    elif round_number==2:
        with tempfile.TemporaryDirectory(prefix="fit_lag_ranking_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell()
            scores=np.zeros((2,2,6));scores[0,:,0]=[1,0];scores[0,:,1:]=np.array([0,1])[:,None]
            scores[1,:,0]=[0,1];scores[1,:,1:]=np.array([1,0])[:,None]
            q=np.array([1,2]);r=np.array([1,2]);mask=np.ones((2,2),bool)
            update_cell(cell,scores,mask,q,r,np.arange(2),"S",1,"same_predicted_category","common_CV_covered_population",1,10,db)
            stats=finalized_cell(cell,"common_CV_covered_population");top=stats["top1"]["positive_and_negative_available_queries"][FEATURES[1]]
            require(stats["known_pairs"]==4 and stats["positive_pairs"]==stats["negative_pairs"]==2 and top["break_raw_correct"]==2 and top["repair_raw_error"]==0,"Paired query counts differ")
            unknown=fresh_cell();update_cell(unknown,scores,mask,np.array([-1,2]),r,np.arange(2),"T",1,"all_candidates","common_CV_covered_population",2,20,db)
            require(unknown["known_pairs"]==2 and unknown["unknown_pairs"]==2 and unknown["negative_pairs"]==1,"Unknown became negative")
            update_cell(fresh_cell(),scores,mask,q,r,np.arange(2),"S",4,"same_predicted_category","common_CV_covered_population",1,10,db)
            require(cross_lag(db,"S","same_predicted_category","common_CV_covered_population")["shared_hard_queries"]==2,"Same-query cross-lag mismatch")
            db.close()
        checks.append("paired_query_source_identity_counts_unknown_not_negative_and_cross_lag_join")
        with tempfile.TemporaryDirectory(prefix="fit_lag_ties_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell();scores=np.zeros((1,2,6));scores[:,:,4]=np.nan
            update_cell(cell,scores,np.ones((1,2),bool),np.array([1]),np.array([1,2]),np.array([1,0]),"S",1,"all_candidates","all_known_population",1,1,db)
            require(cell["top1"]["positive_and_negative_available_queries"][FEATURES[0]]["raw_wrong_feature_wrong"]==1,"Tie used favorable GT instead of immutable row")
            db.close()
        checks.append("immutable_reference_row_tie_and_nonCV_score_mask_no_finite_value_selection")
        with tempfile.TemporaryDirectory(prefix="fit_lag_streamed_support_") as temp:
            out=Path(temp);records=[dict(video_id=1,frame_id=k,observations=2) for k in range(6)]
            generated=generate(out,records,{"1":"S"},["S","ZERO"],geometry,lambda r:fixture(r["frame_id"]),bounds)
            anchors={(100+k,row):(1,row+1) for k in range(6) for row in range(2)}
            db,cells,history,gaps,maxread=score_sealed(out,anchors,["S","ZERO"])
            result=summarize(out,db,cells,{"S":dict(split="fit",role="mean_role",block="synthetic"),
                "ZERO":dict(split="fit",role="pair_role",block="synthetic")},history,gaps)
            require(maxread==4 and result["micro_union"]["1"]["all_candidates"]["all_known_population"]["candidate_pairs"]==20 and
                result["per_source"]["ZERO"]["lags"]["4"]["same_predicted_category"]["common_CV_covered_population"]["known_pairs"]==0 and
                "ZERO" in result["source_held_out_support"] and not result["necessary_support_gate"]["all_necessary_conditions_met"],
                "Streamed support/zero-source/union/fold integration differs")
            db.close()
        checks.append("streamed_generation_scoring_summary_union_zero_source_and_all_fold_integration")
    elif round_number==3:
        with tempfile.TemporaryDirectory(prefix="fit_lag_seal_") as temp:
            out=Path(temp);part=out/"synthetic.npy";np.save(part,np.zeros((4,6)))
            write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION",files_sha256={part.name:sha(part)}))
            verify_seal(out);part.write_bytes(b"changed")
            try:verify_seal(out)
            except ValueError:pass
            else:raise AssertionError("Accepted altered global shard")
            # Must reject before opening real helper/support files.
            try:join_identities(out,{}, {}, {})
            except ValueError:pass
            else:raise AssertionError("Identity import/join preceded seal validation")
        checks.append("global_seal_rejects_altered_shard_before_supervised_helper_import")
        helper=module(HELPER,"synthetic_fit_lag_projection");roles={"S":dict(split="fit",role="mean_role")}
        identity=dict(observation_id=1,split="fit",source_group="S",video_id=1,image_id=1,sampled_step=1,original_frame_index=10,
            local_track_id=0,gt_track_id=5,pred_category_id=1,manual_anchor_id=9,eligible_for_parameter_fit=True)
        with tempfile.TemporaryDirectory(prefix="fit_lag_poison_") as temp:
            p=Path(temp)/"synthetic.jsonl";p.write_text(json.dumps(dict(identity,gt_category_id="CLASS_POISON",error="ERROR_POISON",soft_risk="RISK_POISON"))+"\n"+json.dumps(dict(split="tune",gt_track_id="TUNE_POISON"))+"\n")
            original=json.loads
            def guarded(text,*args,**kwargs):
                require(not any(x in text for x in ("CLASS_POISON","ERROR_POISON","RISK_POISON","TUNE_POISON")),"Forbidden value decoded")
                return original(text,*args,**kwargs)
            json.loads=guarded
            try:result=helper.load_fit_identities(p,roles)
            finally:json.loads=original
            require(list(result)==[1],"FIT-only identity projection changed")
        checks.append("FIT_identity_projection_tune_class_error_risk_poison_not_decoded")
        records=[dict(video_id=1,frame_id=k,observations=2) for k in range(6)]
        with tempfile.TemporaryDirectory(prefix="fit_lag_overbound_") as temp:
            try:generate(Path(temp),records,{"1":"S"},["S"],geometry,lambda r:fixture(r["frame_id"]),dict(bounds,max_frame_rows=1))
            except ValueError:pass
            else:raise AssertionError("Ignored per-frame resource bound")
        checks.append("generation_rejects_declared_resource_bound_violation")
        roles={"S":dict(split="fit",role="mean_role")};video_source={"1":"S","2":"uncollected/no_role"}
        frames=[dict(video_id=1,observations=2)]
        population=dict(sealed_fit_source_groups=1,allocated_fit_videos=2,selected_frames=1,
            selected_fit_videos=1,retained_observations=2,uncollected_allocation_video_ids=[2])
        omitted=validate_population(roles,video_source,frames,population)
        require(omitted==[dict(video_id=2,source_group="uncollected/no_role",cached_frames=0,source_in_sealed_roles=False)],"Uncollected allocation changed sealed source population")
        try:validate_population(roles,video_source,[dict(video_id=2,observations=2)],population)
        except ValueError:pass
        else:raise AssertionError("Accepted cached source absent from sealed roles")
        checks.append("uncollected_allocation_may_lack_role_but_every_cached_source_must_have_original_sealed_FIT_role")
    else:raise ValueError("Exactly three rounds")
    require(len(checks)>=2,"Each round needs at least two checks")
    return dict(status="PASS_FIT_LAG_IDENTITY_SUPPORT_SYNTHETIC_ROUND",round=round_number,checks=checks,
        script_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,real_cache_or_supervised_payload_opened=False,
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
