#!/usr/bin/env python3
"""Independent bounded-memory full FIT lag support output verification."""
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import argparse
from collections import Counter,defaultdict,deque
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sqlite3
import tempfile
import time

import numpy as np

from verify_fit_identity_edge_support_output import project

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
RUN=STAGE/"full_collect_v1"
SUPPORT=STAGE/"support_v1"
PROTOCOL=ROOT/"tmp_analysis/fit_lag_identity_support_protocol_20261024.json"
PRODUCER=ROOT/"tmp_analysis/audit_fit_lag_identity_support_20261024.py"
PREFLIGHT=ROOT/"tmp_analysis/fit_lag_identity_support_independent_preflight_20261024.json"
FEATURES=("raw_appearance_cosine","absolute_xyxy_cosine","box_iou","negative_normalized_displacement_norm",
          "negative_normalized_CV_innovation_norm","semantic_embedding_cosine")
LAGS=(1,4)
SCOPES=("all_candidates","same_predicted_category")
POPULATIONS=("all_known_population","common_CV_covered_population")
SUBSETS=("all_nonempty_known_queries","positive_and_negative_available_queries")
# Filled only after independent final implementation preflight.
FINAL_PRODUCER_SHA="9beb8faf932d13f417d491ddd8f853de001d17a414eac0d59d016b0ffcf3af3b"
FINAL_PROTOCOL_SHA="dd84da8fd028af36c223f42a30d19f9f23725eb790e60aaa668700c80a0806ad"
FINAL_PREFLIGHT_SHA="0344310cdf2ca904ac00cb64dd250db4a0b1b71f7f94904fb56f38d94ed36b0d"
SCANNER_SHA="862d31f63331a288804d808fb303e6704d6f4fdfa80f97daf13971f44c0f795d"


def check(value,message):
    if not value:raise AssertionError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def same(actual,expected,where=""):
    if isinstance(expected,dict):
        check(set(actual)==set(expected),"Dictionary keys differ: "+where)
        for key,value in expected.items():same(actual[key],value,where+"/"+str(key))
    elif isinstance(expected,list):
        check(len(actual)==len(expected),"List length differs: "+where)
        for index,value in enumerate(expected):same(actual[index],value,where+"/"+str(index))
    elif isinstance(expected,float):check(actual is not None and abs(actual-expected)<=1e-12,"Float differs: "+where)
    else:check(actual==expected,"Value differs: "+where)


def metadata_population(protocol):
    roles=load(STAGE/"pair_context_v1/source_roles.json")["groups"]
    fit={source:role for source,role in roles.items() if role["split"]=="fit"}
    records=load(STAGE/"full_collection_plan.json")["video_records"]
    videos={str(row["video_id"]):row["source_group"] for row in records if row["split"]=="fit"}
    frames=[json.loads(line) for line in (RUN/"frame_execution.jsonl").open()]
    frames=sorted((r for r in frames if str(r["video_id"]) in videos),key=lambda r:(r["video_id"],r["frame_id"]))
    grouped=defaultdict(list)
    for row in frames:grouped[row["video_id"]].append(row)
    check(all(videos[str(video)] in fit for video in grouped),"Cached FIT video/role mismatch")
    expected=dict(sealed_fit_source_groups=len(fit),allocated_fit_videos=len(videos),selected_fit_videos=len(grouped),
                  selected_frames=len(frames),retained_observations=sum(row["observations"] for row in frames))
    for key,value in expected.items():check(protocol["population"][key]==value,"Fixed population differs: "+key)
    counts={lag:0 for lag in LAGS}
    for rows in grouped.values():
        check([r["frame_id"] for r in rows]==list(range(len(rows))),"Missing exact sampled frame")
        for lag in LAGS:counts[lag]+=sum(rows[i]["observations"]*rows[i-lag]["observations"] for i in range(lag,len(rows)))
    same({str(k):v for k,v in counts.items()},protocol["resource_bounds"]["candidates_by_lag"],"Metadata QxM counts")
    return fit,videos,frames,grouped,expected


def read_raw(record,history):
    """Independent unique past-ID snapshot; later frames cannot mutate it."""
    with np.load(record["npz"],allow_pickle=False) as saved:
        for key,field in (("video_id","video_id"),("image_id","image_id"),("frame_id","frame_id"),("frame_index","frame_index")):
            check(int(saved[key])==record[field],"Raw frame provenance changed")
        check(saved["track_feats"].dtype==saved["cls_feats"].dtype==np.dtype("float16"),"Raw cache quantization changed")
        boxes=saved["det_bboxes"].astype(float)
        frame=dict(video_id=record["video_id"],image_id=record["image_id"],sampled_step=record["frame_id"],
            original_frame_index=record["frame_index"],boxes=boxes[:,:4],raw=saved["track_feats"].astype(float),
            semantic=saved["cls_feats"].astype(float),released_ids=saved["track_ids"].astype(np.int64),
            predicted_categories=saved["det_labels"].astype(np.int32)+1)
    n=len(boxes);check(n==record["observations"] and boxes.shape==(n,5),"Raw retained rows changed")
    check(frame["raw"].shape==(n,256) and frame["semantic"].shape==(n,512),"Raw cache feature width changed")
    check(np.isfinite(boxes).all() and np.all(boxes[:,2:4]>boxes[:,:2]),"Invalid retained boxes")
    return independent_history(frame,history)


def independent_history(frame,history):
    identities=frame["released_ids"].tolist();counts=Counter(identities);n=len(identities)
    old_boxes=np.full((n,4),np.nan);old_times=np.full(n,np.nan);reasons=[]
    for row,identity in enumerate(identities):
        if identity<0:reason="unassigned_id"
        elif counts[identity]>1:reason="duplicate_current_id"
        elif identity not in history:reason="no_previous_occurrence"
        elif history[identity] is None:reason="previous_duplicate_invalid"
        else:
            box,t=history[identity];check(t<frame["original_frame_index"],"History is not past")
            old_boxes[row]=box;old_times[row]=t;reason="available"
        reasons.append(reason)
    for row,identity in enumerate(identities):
        if identity>=0:history[identity]=None if counts[identity]>1 else (frame["boxes"][row].copy(),frame["original_frame_index"])
    frame.update(previous_boxes=old_boxes,previous_times=old_times,history_reason=reasons)
    return frame


def row_metadata(frame):
    return dict(video_id=frame["video_id"],image_id=frame["image_id"],sampled_step=frame["sampled_step"],
        original_frame_index=frame["original_frame_index"],boxes=frame["boxes"].tolist(),released_ids=frame["released_ids"].tolist(),
        predicted_categories=frame["predicted_categories"].tolist(),history_reason=frame["history_reason"],
        previous_boxes=[row.tolist() if np.isfinite(row).all() else None for row in frame["previous_boxes"]],
        previous_times=[int(t) if np.isfinite(t) else None for t in frame["previous_times"]])


def independent_scores(q,r):
    def cosine(a,b):
        qa=a/np.maximum(np.sqrt((a*a).sum(1))[:,None],1e-12)
        rb=b/np.maximum(np.sqrt((b*b).sum(1))[:,None],1e-12)
        return qa@rb.T
    qb,rb=q["boxes"],r["boxes"];nq,nr=len(qb),len(rb)
    qc=(qb[:,:2]+qb[:,2:])/2;rc=(rb[:,:2]+rb[:,2:])/2
    qs=qb[:,2:]-qb[:,:2];rs=rb[:,2:]-rb[:,:2];scale=np.sqrt(qs[:,None,:]*rs[None,:,:])
    intersection=np.maximum(np.minimum(qb[:,None,2:],rb[None,:,2:])-np.maximum(qb[:,None,:2],rb[None,:,:2]),0).prod(2)
    iou=intersection/(qs.prod(1)[:,None]+rs.prod(1)[None,:]-intersection)
    displacement=-np.sqrt((((qc[:,None,:]-rc[None,:,:])/scale)**2).sum(2))
    innovation=np.full((nq,nr),np.nan)
    for row,t in enumerate(r["previous_times"]):
        if not np.isfinite(t):continue
        box=r["previous_boxes"][row];past=(box[:2]+box[2:])/2
        v=(rc[row]-past)/(r["original_frame_index"]-t)
        predicted=rc[row]+v*(q["original_frame_index"]-r["original_frame_index"])
        innovation[:,row]=-np.sqrt((((qc-predicted)/scale[:,row,:])**2).sum(1))
    return np.stack((cosine(q["raw"],r["raw"]),cosine(qb,rb),iou,displacement,innovation,
                     cosine(q["semantic"],r["semantic"])),2).reshape(nq,nr,6)


def legal_anchors(video_source):
    manual=SUPPORT/"supervised_observations.jsonl"
    receipt=project((SUPPORT/"support_receipt.json").read_text(),{"output_files"})
    check(sha(manual)==receipt["output_files"][manual.name],"Manual identity payload hash changed")
    fields={"split","observation_id","source_group","video_id","image_id","sampled_step","original_frame_index",
            "local_track_id","gt_track_id","manual_anchor_id","eligible_for_parameter_fit","pred_category_id"}
    identities={};tune=0
    for line in manual.open():
        split=project(line,{"split"})["split"];check(split in ("fit","tune"),"Unexpected manual split")
        if split=="tune":tune+=1;continue
        row=project(line,fields);check(set(row)==fields and row["eligible_for_parameter_fit"] is True,"Ineligible manual identity")
        check(row["observation_id"] not in identities,"Duplicate manual identity")
        identities[row["observation_id"]]=row
    check(len(identities)==21661 and tune==2226,"Frozen identity support changed")
    manifest=load(SUPPORT/"gt_free_index_manifest.json");index=SUPPORT/"gt_free_observations.jsonl"
    check(sha(index)==manifest["files"][index.name],"Immutable observation index hash changed")
    anchors={}
    for line in index.open():
        row=json.loads(line)
        if row["split"]!="fit":continue
        check(video_source[str(row["video_id"])]==row["source_group"],"Identity index population changed")
        identity=identities.get(row["observation_id"])
        if identity is None:continue
        for key in fields-{"observation_id","gt_track_id","manual_anchor_id","eligible_for_parameter_fit"}:
            check(row[key]==identity[key],"Manual/index provenance mismatch: "+key)
        key=(row["image_id"],row["dump_row"]);check(key not in anchors,"Duplicate immutable manual row")
        anchors[key]=(row["video_id"],identity["gt_track_id"])
    check(len(anchors)==len(identities),"Eligible manual identities missing immutable row")
    return anchors,tune


def database(path):
    db=sqlite3.connect(path);db.execute("PRAGMA temp_store=FILE");db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE query_support(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,image INTEGER,qrow INTEGER,qgt INTEGER,p INTEGER,n INTEGER,raw_correct INTEGER,correct_bits INTEGER,PRIMARY KEY(lag,scope,population,video,image,qrow))")
    db.execute("CREATE TABLE trajectories(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,gt INTEGER,PRIMARY KEY(lag,scope,population,video,gt))")
    db.execute("CREATE TABLE identity_pairs(source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,qgt INTEGER,rgt INTEGER,pairs INTEGER,PRIMARY KEY(lag,scope,population,video,qgt,rgt))")
    return db


def fresh_cell():
    return dict(candidate_pairs=0,known_pairs=0,positive_pairs=0,negative_pairs=0,unknown_pairs=0,
        nonempty_known_queries=0,positive_available_queries=0,negative_available_queries=0,hard_queries=0,
        top1={subset:{f:{"raw_wrong_feature_wrong":0,"raw_wrong_feature_correct":0,"raw_correct_feature_wrong":0,"raw_correct_feature_correct":0}
                      for f in FEATURES} for subset in SUBSETS},orders={f:[[0]*3 for _ in range(3)] for f in FEATURES})


def count_block(cell,scores,mask,qids,rids,source,lag,scope,population,video,image,db):
    """Literal per-query winner and positive-negative pair enumeration."""
    columns=range(6) if population==POPULATIONS[1] else (0,1,2,3,5)
    for qr,qgt in enumerate(qids):
        permitted=[j for j in range(len(rids)) if mask[qr,j]]
        references=[j for j in permitted if qgt>=0 and rids[j]>=0]
        positives=[j for j in references if rids[j]==qgt];negatives=[j for j in references if rids[j]!=qgt]
        cell["candidate_pairs"]+=len(permitted);cell["known_pairs"]+=len(references)
        cell["positive_pairs"]+=len(positives);cell["negative_pairs"]+=len(negatives)
        cell["unknown_pairs"]+=len(permitted)-len(references)
        if not references:continue
        check(np.isfinite(scores[qr,references][:,list(columns)]).all(),"Nonfinite score in fixed known scope")
        cell["nonempty_known_queries"]+=1;cell["positive_available_queries"]+=bool(positives);cell["negative_available_queries"]+=bool(negatives)
        hard=bool(positives and negatives);cell["hard_queries"]+=hard
        correct={c:rids[max(references,key=lambda j:(float(scores[qr,j,c]),-j))]==qgt for c in columns}
        bits=sum((1<<c) for c in columns if correct[c])
        db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(source,lag,scope,population,video,image,qr,int(qgt),len(positives),len(negatives),int(correct[0]),bits))
        for gt in set([int(qgt)]+[int(rids[j]) for j in references]):
            db.execute("INSERT OR IGNORE INTO trajectories VALUES(?,?,?,?,?,?)",(source,lag,scope,population,video,gt))
        for rgt,count in Counter(int(rids[j]) for j in references).items():
            db.execute("INSERT INTO identity_pairs VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(lag,scope,population,video,qgt,rgt) DO UPDATE SET pairs=pairs+excluded.pairs",(source,lag,scope,population,video,int(qgt),rgt,count))
        for subset in SUBSETS if hard else SUBSETS[:1]:
            for column in columns:
                name="raw_"+("correct" if correct[0] else "wrong")+"_feature_"+("correct" if correct[column] else "wrong")
                cell["top1"][subset][FEATURES[column]][name]+=1
        for p in positives:
            for n in negatives:
                order={c:0 if scores[qr,p,c]<scores[qr,n,c] else 2 if scores[qr,p,c]>scores[qr,n,c] else 1 for c in columns}
                for column in columns:cell["orders"][FEATURES[column]][order[0]][order[column]]+=1


def final_cell(cell,population):
    result={k:int(v) for k,v in cell.items() if k not in ("top1","orders")}
    names=FEATURES if population==POPULATIONS[1] else tuple(f for i,f in enumerate(FEATURES) if i!=4)
    result["top1"]={}
    for subset in SUBSETS:
        result["top1"][subset]={}
        for name in names:
            counts=dict(cell["top1"][subset][name]);counts.update(repair_raw_error=counts["raw_wrong_feature_correct"],
                break_raw_correct=counts["raw_correct_feature_wrong"],raw_correct=counts["raw_correct_feature_wrong"]+counts["raw_correct_feature_correct"],
                feature_correct=counts["raw_wrong_feature_correct"]+counts["raw_correct_feature_correct"])
            result["top1"][subset][name]=counts
    result["positive_negative_pair_order"]={name:dict(comparisons=sum(map(sum,cell["orders"][name])),raw_to_feature_loss_tie_win_matrix=cell["orders"][name]) for name in names}
    return result


def report_results(db,cells,roles,history,gaps):
    reports={};hard_sources=[];error_sources=[];cross_sources=[];sources=sorted(roles)
    primary=(SCOPES[1],POPULATIONS[1]);folds={}
    for source in sources:
        item=dict(role=roles[source]["role"],dataset_block=roles[source]["block"],lags={},cross_lag={})
        for lag in LAGS:
            item["lags"][str(lag)]={}
            for scope in SCOPES:
                item["lags"][str(lag)][scope]={}
                for pop in POPULATIONS:
                    key=(source,lag,scope,pop);stats=final_cell(cells[key],pop)
                    stats["official_GT_trajectories"]=db.execute("SELECT COUNT(*) FROM trajectories WHERE source=? AND lag=? AND scope=? AND population=?",key).fetchone()[0]
                    stats["distinct_ordered_identity_pairs"]=db.execute("SELECT COUNT(*) FROM identity_pairs WHERE source=? AND lag=? AND scope=? AND population=?",key).fetchone()[0]
                    counts=Counter();errors=Counter()
                    for v,gt,p,n,correct in db.execute("SELECT video,qgt,p,n,raw_correct FROM query_support WHERE source=? AND lag=? AND scope=? AND population=?",key):
                        if p>0 and n>0:counts[v,gt]+=1;errors[v,gt]+=int(not correct)
                    stats["query_GT_error_distribution"]=[dict(video_id=v,gt_track_id=g,hard_queries=counts[v,g],raw_errors=errors[v,g]) for v,g in sorted(counts)]
                    item["lags"][str(lag)][scope][pop]=stats
            item["lags"][str(lag)]["reference_history_reason_counts"]=dict(history[source,lag])
            item["lags"][str(lag)]["original_frame_gap_per_frame_pair_counter"]={str(k):v for k,v in sorted(gaps[source,lag].items())}
        for scope in SCOPES:
            item["cross_lag"][scope]={}
            for pop in POPULATIONS:
                lagsets={}
                for lag in LAGS:
                    lagsets[lag]={(v,im,row):correct for v,im,row,p,n,correct in db.execute("SELECT video,image,qrow,p,n,raw_correct FROM query_support WHERE source=? AND lag=? AND scope=? AND population=?",(source,lag,scope,pop)) if p>0 and n>0}
                shared=set(lagsets[1])&set(lagsets[4])
                item["cross_lag"][scope][pop]=dict(shared_hard_queries=len(shared),raw_error_lag1=sum(not lagsets[1][k] for k in shared),
                    raw_error_lag4=sum(not lagsets[4][k] for k in shared),raw_error_both=sum(not lagsets[1][k] and not lagsets[4][k] for k in shared))
        reports[source]=item
        if any(cells[(source,lag,*primary)]["hard_queries"] for lag in LAGS):hard_sources.append(source)
        if any(cells[(source,lag,*primary)]["top1"][SUBSETS[1]][FEATURES[0]]["raw_wrong_feature_wrong"] for lag in LAGS):error_sources.append(source)
        if item["cross_lag"][primary[0]][primary[1]]["shared_hard_queries"]:cross_sources.append(source)
    for source in sources:
        folds[source]={}
        for lag in LAGS:
            evaluation={k:int(cells[(source,lag,*primary)][k]) for k in ("positive_pairs","negative_pairs","hard_queries")}
            train={k:sum(int(cells[(s,lag,*primary)][k]) for s in sources if s!=source) for k in evaluation}
            folds[source][str(lag)]=dict(training_support=train,evaluation_support=evaluation,
                necessary_positive_negative_support=all(v>0 for d in (train,evaluation) for v in d.values()))
    def summed(values):
        if isinstance(values[0],dict):return {key:summed([value[key] for value in values]) for key in values[0]}
        if isinstance(values[0],list):return [summed([value[index] for value in values]) for index in range(len(values[0]))]
        return sum(values)
    union={}
    for lag in LAGS:
        union[str(lag)]={}
        for scope in SCOPES:
            union[str(lag)][scope]={}
            for pop in POPULATIONS:
                values=[cells[source,lag,scope,pop] for source in sources];combined=final_cell(summed(values),pop)
                combined.update(sources_with_known_pairs=sum(value["known_pairs"]>0 for value in values),
                    sources_with_hard_queries=sum(value["hard_queries"]>0 for value in values),
                    sources_with_raw_hard_errors=sum(value["top1"][SUBSETS[1]][FEATURES[0]]["raw_wrong_feature_wrong"]>0 for value in values),
                    official_GT_trajectories=db.execute("SELECT COUNT(*) FROM trajectories WHERE lag=? AND scope=? AND population=?",(lag,scope,pop)).fetchone()[0])
                union[str(lag)][scope][pop]=combined
    conditions=dict(at_least_two_hard_query_sources=len(hard_sources)>=2,at_least_two_raw_error_sources=len(error_sources)>=2,
        cross_lag_shared_hard_query_exists=bool(cross_sources),two_lag_source_blocked_positive_negative_fold_exists=any(all(f["necessary_positive_negative_support"] for f in row.values()) for row in folds.values()))
    return dict(micro_union=union,per_source=reports,source_held_out_support=folds,necessary_support_gate=dict(status="HOLD_PENDING_SEPARATE_TRAINING_PROTOCOL",
        necessary_conditions=conditions,all_necessary_conditions_met=all(conditions.values()),blockers=[k for k,v in conditions.items() if not v],
        hard_query_sources=hard_sources,raw_error_sources=error_sources,cross_lag_shared_hard_sources=cross_sources,
        sufficient_training_support_established=False,training_protocol_frozen=False))


def inspect_scores(out,grouped,video_source,sources,anchors,db,bounds,loader=read_raw):
    cells={(s,lag,scope,pop):fresh_cell() for s in sources for lag in LAGS for scope in SCOPES for pop in POPULATIONS}
    history_counts=defaultdict(Counter);gap_counts=defaultdict(Counter);errors=np.zeros(6);candidate_counts=Counter()
    maxblock=0;maxring=0;receipt_count=0;expected_files={"frame_pairs.jsonl"}
    with (out/"frame_pairs.jsonl").open() as receipt_stream:
        for video,records in sorted(grouped.items()):
            source=video_source[str(video)];history={};ring=deque(maxlen=5);offsets=Counter()
            rowpath=out/"shards"/f"v{video}.rows.json";metadata=load(rowpath)
            same({k:metadata[k] for k in ("video_id","source_group")},dict(video_id=video,source_group=source),"Video row metadata header")
            check(len(metadata["frames"])==len(records),"Rowmap frame count differs")
            expected_files.add(str(rowpath.relative_to(out)));shards={}
            for lag in LAGS:
                path=out/"shards"/f"v{video}_lag{lag}.scores.npy";expected_files.add(str(path.relative_to(out)))
                saved=np.load(path,mmap_mode="r",allow_pickle=False);n=sum(records[i]["observations"]*records[i-lag]["observations"] for i in range(lag,len(records)))
                check(isinstance(saved,np.memmap) and saved.shape==(n,6) and saved.dtype==np.dtype("float64"),"Shard memmap/dtype/shape changed")
                shards[lag]=saved
            for record in records:
                q=loader(record,history);ring.append(q);maxring=max(maxring,len(ring));step=record["frame_id"]
                same(metadata["frames"][step],row_metadata(q),"Independently rebuilt rowmap/history")
                nq=len(q["boxes"])
                for lag in LAGS:
                    r=ring[-lag-1] if step>=lag else None;nr=0 if r is None else len(r["boxes"]);count=nq*nr
                    expected=dict(video_id=video,source_group=source,lag=lag,query_step=step,query_image_id=q["image_id"],reference_step=None if r is None else r["sampled_step"],
                        reference_image_id=None if r is None else r["image_id"],query_rows=nq,reference_rows=nr,original_frame_gap=None if r is None else q["original_frame_index"]-r["original_frame_index"],
                        offset=offsets[lag],candidates=count,score_file=f"shards/v{video}_lag{lag}.scores.npy")
                    line=next(receipt_stream,None);check(line is not None,"Missing frame receipt");same(json.loads(line),expected,"Independent exact-lag frame receipt")
                    receipt_count+=1;candidate_counts[lag]+=count;maxblock=max(maxblock,count)
                    if r is not None:
                        block=shards[lag][offsets[lag]:offsets[lag]+count].copy().reshape(nq,nr,6);truth=independent_scores(q,r)
                        np.testing.assert_allclose(block,truth,rtol=1e-12,atol=1e-12,equal_nan=True)
                        for column in range(6):
                            finite=np.isfinite(truth[:,:,column])
                            if finite.any():errors[column]=max(errors[column],float(np.abs(block[:,:,column][finite]-truth[:,:,column][finite]).max()))
                        gap_counts[source,lag][expected["original_frame_gap"]]+=1;history_counts[source,lag].update(r["history_reason"])
                        def ids(frame):
                            result=[]
                            for row in range(len(frame["boxes"])):
                                value=anchors.get((frame["image_id"],row));check(value is None or value[0]==video,"Manual identity crosses video")
                                result.append(-1 if value is None else value[1])
                            return result
                        qids,rids=ids(q),ids(r);cv=np.isfinite(r["previous_times"])
                        masks=(np.ones((nq,nr),bool),q["predicted_categories"][:,None]==r["predicted_categories"][None,:])
                        for scope,hard in zip(SCOPES,masks):
                            for pop,mask in zip(POPULATIONS,(hard,hard&cv[None,:])):
                                count_block(cells[source,lag,scope,pop],block,mask,qids,rids,source,lag,scope,pop,video,q["image_id"],db)
                        db.commit()
                    offsets[lag]+=count
                check(len(ring)<=bounds["generation_ring_frames"] and maxblock<=bounds["max_candidates_per_frame_pair"],"Independent resource bound exceeded")
            for lag,saved in shards.items():check(offsets[lag]==len(saved),"Unverified shard rows remain")
            del shards
            if len(expected_files)%60==1:print(json.dumps(dict(phase="independent_full_feature_recomputation",video_id=video,candidates=sum(candidate_counts.values()))),flush=True)
        check(next(receipt_stream,None) is None,"Extra frame receipt")
    return cells,history_counts,gap_counts,dict(files=sorted(expected_files),candidates_by_lag={str(k):candidate_counts[k] for k in LAGS},
        frame_pair_receipts=receipt_count,maximum_candidate_block=maxblock,maximum_feature_ring_frames=maxring,feature_maximum_absolute_error=dict(zip(FEATURES,errors.tolist())))


def compare_databases(expected,actual):
    check(actual.execute("PRAGMA integrity_check").fetchall()==[("ok",)],"Producer SQLite integrity failure")
    check(expected.execute("PRAGMA integrity_check").fetchall()==[("ok",)],"Independent SQLite integrity failure")
    counts={}
    for table in ("query_support","trajectories","identity_pairs"):
        ecols=expected.execute("PRAGMA table_info("+table+")").fetchall();acols=actual.execute("PRAGMA table_info("+table+")").fetchall()
        same(acols,ecols,"SQLite table schema "+table)
        order=",".join(str(i) for i in range(1,len(ecols)+1));n=0
        for a,b in itertools.zip_longest(actual.execute("SELECT * FROM "+table+" ORDER BY "+order),expected.execute("SELECT * FROM "+table+" ORDER BY "+order)):
            check(a==b,"Independently reconstructed SQLite row differs: "+table);n+=1
        counts[table]=n
    check({row[0] for row in actual.execute("SELECT name FROM sqlite_master WHERE type='table'")}==set(counts),"Extra producer SQLite tables")
    return counts


def audit(out):
    started=time.monotonic();target=out/"independent_output_audit.json";check(not target.exists(),"Preserve sealed independent audit")
    check(all((FINAL_PRODUCER_SHA,FINAL_PROTOCOL_SHA,FINAL_PREFLIGHT_SHA)),"Independent final SHA gate not yet frozen")
    for path,digest in ((PRODUCER,FINAL_PRODUCER_SHA),(PROTOCOL,FINAL_PROTOCOL_SHA),(PREFLIGHT,FINAL_PREFLIGHT_SHA),(ROOT/"tmp_analysis/verify_fit_identity_edge_support_output.py",SCANNER_SHA)):
        check(sha(path)==digest,"Frozen verifier input changed: "+str(path))
    execution=load(out/"producer_execution_receipt.json");check(execution["actual_tool_execution"]["exit_code"]==0,"Producer actual exit0 receipt absent")
    for relative,info in execution.get("evidence",{}).items():
        path=ROOT/relative;check(sha(path)==info["sha256"] and path.stat().st_size==info["bytes"],"Producer execution evidence changed")
    protocol=load(PROTOCOL);summary=load(out/"summary.json");seal=load(out/"GT_free_candidates_sealed.json");preflight=load(PREFLIGHT)
    check(preflight["status"]=="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_IMPLEMENTATION_PREFLIGHT","Independent implementation gate missing")
    check(summary["status"]=="COMPLETE_ALL_SEALED_FIT_LAG_SUPPORT_PENDING_INDEPENDENT_AUDIT","Producer incomplete")
    check(summary["script_sha256"]==seal["script_sha256"]==FINAL_PRODUCER_SHA and summary["protocol_sha256"]==seal["protocol_sha256"]==FINAL_PROTOCOL_SHA,"Wrong producer/protocol")
    check(seal["status"]=="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION" and seal["supervised_payload_or_helper_opened"] is False and seal["unknowns_or_old_IDs_filtered"] is False,"Global pre-supervision seal missing")
    for relative,digest in seal["files_sha256"].items():check(sha(out/relative)==digest,"Globally sealed shard/rowmap/receipt changed")
    same(summary["generated"]["files_sha256"],seal["files_sha256"],"Summary/global seal member hashes")
    for relative,digest in summary["output_files_sha256"].items():check(sha(out/relative)==digest,"Producer final output changed")
    for path,digest in summary["input_sha256"].items():check(sha(path)==digest,"Producer protected input changed")
    for path,digest in protocol["input_pins"].items():check(summary["input_sha256"][path]==digest,"Protocol input pin omitted")
    for path,digest in seal["input_sha256"].items():check(summary["input_sha256"][path]==digest,"Pre-supervision input omitted")
    for flag in ("training_started","parameters_or_thresholds_selected","GPU_used","tune_test_or_classheldout_targets_decoded","unknown_is_negative"):
        check(summary[flag] is False,"Forbidden producer scope change: "+flag)
    check(summary["sealed_files_verified_before_supervised_import_and_after"] is True and summary["all_input_hashes_verified_before_after"] is True,"Missing producer verification boundary")
    roles,video_source,frames,grouped,population=metadata_population(protocol)
    watched={**summary["input_sha256"],**{str(out/name):digest for name,digest in seal["files_sha256"].items()},
             **{str(path):sha(path) for path in (Path(__file__).resolve(),out/"summary.json",out/"producer_execution_receipt.json",out/"GT_free_candidates_sealed.json",out/"support.sqlite")}}
    # All candidate shards have been globally verified before identity IO here.
    anchors,tune=legal_anchors(video_source)
    for frame in frames:check(sha(frame["npz"])==frame["npz_sha256"]==summary["input_sha256"][frame["npz"]],"Raw selected frame hash changed")
    with tempfile.TemporaryDirectory(prefix="independent_fit_lag_sqlite_",dir=out) as temporary:
        expected=database(Path(temporary)/"expected.sqlite")
        cells,history,gaps,checks=inspect_scores(out,grouped,video_source,sorted(roles),anchors,expected,protocol["resource_bounds"])
        results=report_results(expected,cells,roles,history,gaps)
        same(summary["results"],results,"Every source/scope/lag ranking distribution and support gate")
        actual=sqlite3.connect((out/"support.sqlite").as_uri()+"?mode=ro",uri=True)
        actual.execute("PRAGMA temp_store=FILE");actual.execute("PRAGMA cache_size=-16384")
        counts=compare_databases(expected,actual);actual.close();expected.close()
    check(set(checks["files"])==set(seal["files_sha256"]),"Global seal omitted/added candidate files")
    for key in ("candidates_by_lag","frame_pair_receipts","maximum_candidate_block","maximum_feature_ring_frames"):
        same(summary["generated"][key],checks[key],"Independent generation resource/count "+key)
    check(summary["maximum_score_rows_read"]==checks["maximum_candidate_block"],"Producer score read block differs")
    check(summary["eligible_FIT_manual_anchors"]==len(anchors),"Eligible full FIT manual anchors differ")
    same(summary["population"],protocol["population"],"Fixed population report")
    omitted=sorted(set(map(int,video_source))-set(grouped))
    check(protocol["population"]["uncollected_allocation_video_ids"]==omitted,"Uncollected allocation differs")
    same(summary["uncollected_allocation_videos"],[dict(video_id=video,source_group=video_source[str(video)],cached_frames=0,
        source_in_sealed_roles=video_source[str(video)] in roles) for video in omitted],"Uncollected allocation report")
    protected=load(RUN/"protected_before.json")
    for path,info in protected.items():
        check(Path(path).exists()==info["exists"],"Protected baseline existence changed")
        if info["exists"]:check(sha(path)==info["sha256"],"Protected baseline hash changed")
    for path,digest in watched.items():check(sha(path)==digest,"Input/output/auditor changed during independent audit")
    report=dict(status="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_OUTPUT_AUDIT",auditor_sha256=sha(__file__),
        producer_script_sha256=FINAL_PRODUCER_SHA,protocol_sha256=FINAL_PROTOCOL_SHA,preflight_sha256=FINAL_PREFLIGHT_SHA,
        summary_sha256=sha(out/"summary.json"),producer_execution_receipt_sha256=sha(out/"producer_execution_receipt.json"),
        support_SQLite_sha256=sha(out/"support.sqlite"),global_GT_free_seal_sha256=sha(out/"GT_free_candidates_sealed.json"),
        population=population,uncollected_allocated_FIT_video_ids=sorted(set(map(int,video_source))-set(grouped)),
        independently_recomputed=checks,independently_reconstructed_SQLite_table_rows=counts,eligible_FIT_manual_anchors=len(anchors),
        tune_rows_split_only_skipped=tune,all_source_scope_lag_and_identity_distributions_verified=True,source_held_out_and_cross_lag_support_verified=True,
        necessary_support_gate=results["necessary_support_gate"],all_inputs_outputs_and_protected_baseline_hashes_verified=True,
        producer_or_geometry_or_supervised_decoder_imported=False,forbidden_correctness_values_decoded=False,unknown_is_negative=False,
        training=False,GPU=False,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_seconds=time.monotonic()-started,execution_observation="Producer actual exit0 is bound to supervisor receipt; auditor execution receipt is supervisor-owned",
        estimated_CPU_minutes=[15,45],estimate_is_not_measurement=True,
        interpretation="Independent descriptive fixed full-FIT support verification only; all necessary gates remain HOLD and no training, assignment efficacy or novelty is established.")
    with target.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(status=report["status"],elapsed_seconds=report["elapsed_seconds"],candidates=sum(checks["candidates_by_lag"].values()),necessary_support_gate=report["necessary_support_gate"])),flush=True)


def full_temporary_self_test():
    """Exercise actual audit entry with synthetic metadata/cache authorities."""
    import copy
    import importlib.util
    import audit_fit_lag_identity_support_implementation_20261024 as fixture_source
    spec=importlib.util.spec_from_file_location("only_synthetic_fixture_producer",PRODUCER)
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    geometry_spec=importlib.util.spec_from_file_location("only_synthetic_fixture_geometry",producer.GEOMETRY)
    geometry=importlib.util.module_from_spec(geometry_spec);geometry_spec.loader.exec_module(geometry)
    saved_globals={key:globals()[key] for key in ("STAGE","RUN","PROTOCOL","PREFLIGHT","FINAL_PROTOCOL_SHA","FINAL_PREFLIGHT_SHA","legal_anchors")}
    try:
        with tempfile.TemporaryDirectory(prefix="independent_fit_lag_full_entry_") as temporary:
            base=Path(temporary);stage=base/"stage";run=stage/"full_collect_v1";context=stage/"pair_context_v1"
            run.mkdir(parents=True);context.mkdir();out=base/"output";out.mkdir()
            snapshots,records,anchors=fixture_source.fixtures();video_source={"1":"S","2":"T","3":"ZERO","4":"uncollected_no_role"}
            roles={s:dict(split="fit",role="mean_role" if s=="S" else "pair_role",block="synthetic") for s in ("S","T","ZERO","UNUSED")}
            def save(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
            for record in records:
                frame=snapshots[record["video_id"],record["frame_id"]];path=run/f"v{record['video_id']}_f{record['frame_id']}.npz"
                np.savez(path,video_id=frame["video_id"],image_id=frame["image_id"],frame_id=frame["sampled_step"],frame_index=frame["original_frame_index"],
                    det_bboxes=np.concatenate((frame["boxes"],np.ones((len(frame["boxes"]),1))),axis=1),track_feats=frame["raw"].astype(np.float16),
                    cls_feats=frame["semantic"].astype(np.float16),track_ids=frame["released_ids"],det_labels=frame["predicted_categories"]-1)
                record.update(image_id=frame["image_id"],frame_index=frame["original_frame_index"],npz=str(path),npz_sha256=sha(path))
            save(context/"source_roles.json",{"groups":roles});save(stage/"full_collection_plan.json",dict(video_records=[dict(video_id=int(video),source_group=source,split="fit") for video,source in video_source.items()]))
            (run/"frame_execution.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records));save(run/"protected_before.json",{})
            bounds=dict(max_frame_rows=3,max_video_frames=10,max_candidates_per_frame_pair=9,generation_ring_frames=5,candidates_by_lag={"1":69,"4":46})
            population=dict(sealed_fit_source_groups=4,allocated_fit_videos=4,selected_fit_videos=3,selected_frames=22,retained_observations=sum(r["observations"] for r in records),uncollected_allocation_video_ids=[4])
            protocol_path=base/"protocol.json";preflight_path=base/"preflight.json"
            inputs={str(p):sha(p) for p in (context/"source_roles.json",stage/"full_collection_plan.json",run/"frame_execution.jsonl")}
            save(protocol_path,dict(population=population,resource_bounds=bounds,input_pins=inputs));protocol_sha=sha(protocol_path)
            save(preflight_path,dict(status="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_IMPLEMENTATION_PREFLIGHT"))
            inputs={**inputs,str(protocol_path):protocol_sha,str(preflight_path):sha(preflight_path),**{r["npz"]:r["npz_sha256"] for r in records}}
            generated=producer.generate(out,records,video_source,sorted(roles),geometry,producer.load_frame,bounds)
            sealpath=out/"GT_free_candidates_sealed.json";save(sealpath,dict(status="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION",**generated,
                script_sha256=FINAL_PRODUCER_SHA,protocol_sha256=protocol_sha,input_sha256=inputs,features=list(FEATURES),supervised_payload_or_helper_opened=False,unknowns_or_old_IDs_filtered=False))
            db,cells,h,g,maxread=producer.score_sealed(out,anchors,sorted(roles));results=producer.summarize(out,db,cells,roles,h,g);db.commit();db.close()
            summary=dict(status="COMPLETE_ALL_SEALED_FIT_LAG_SUPPORT_PENDING_INDEPENDENT_AUDIT",results=results,script_sha256=FINAL_PRODUCER_SHA,protocol_sha256=protocol_sha,
                input_sha256=inputs,all_input_hashes_verified_before_after=True,sealed_files_verified_before_supervised_import_and_after=True,
                generated=generated,maximum_score_rows_read=maxread,eligible_FIT_manual_anchors=len(anchors),population=population,
                uncollected_allocation_videos=[dict(video_id=4,source_group="uncollected_no_role",cached_frames=0,source_in_sealed_roles=False)],
                output_files_sha256={name:sha(out/name) for name in ("GT_free_candidates_sealed.json","support.sqlite")},
                training_started=False,parameters_or_thresholds_selected=False,GPU_used=False,tune_test_or_classheldout_targets_decoded=False,unknown_is_negative=False)
            save(out/"summary.json",summary);save(out/"producer_execution_receipt.json",dict(actual_tool_execution=dict(exit_code=0,observed_by="synthetic fixture only, no real process claim"),
                evidence={str(out/"summary.json"):dict(sha256=sha(out/"summary.json"),bytes=(out/"summary.json").stat().st_size)}))
            globals().update(STAGE=stage,RUN=run,PROTOCOL=protocol_path,PREFLIGHT=preflight_path,FINAL_PROTOCOL_SHA=protocol_sha,FINAL_PREFLIGHT_SHA=sha(preflight_path),legal_anchors=lambda _: (anchors,0))
            audit(out)
            result=load(out/"independent_output_audit.json")
            check(result["status"]=="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_OUTPUT_AUDIT" and result["independently_recomputed"]["candidates_by_lag"]=={"1":69,"4":46},"Complete temporary audit entry failed")
    finally:globals().update(saved_globals)


def self_test():
    # Identity labels and rows intentionally disagree with immutable row order.
    with tempfile.TemporaryDirectory(prefix="independent_fit_lag_self_") as temp:
        db=database(Path(temp)/"expected.sqlite");cell=fresh_cell();scores=np.zeros((3,3,6))
        scores[0,:,0]=[1,1,0];scores[0,:,1:]=np.array([0,2,1])[:,None]
        mask=np.ones((3,3),bool)
        count_block(cell,scores,mask,[8,-1,9],[4,8,9],"S",1,SCOPES[1],POPULATIONS[1],1,100,db)
        check(cell["known_pairs"]==6 and cell["unknown_pairs"]==3 and cell["positive_pairs"]==2 and cell["negative_pairs"]==4,"Literal known/unknown support count")
        check(cell["top1"][SUBSETS[1]][FEATURES[1]]["raw_wrong_feature_correct"]==1,"Literal top1 repair/tie check")
        tables=compare_databases(db,db);check(tables["query_support"]==2,"SQLite row verification")
        db.close()
    history={}
    def f(step,ids):return dict(video_id=1,image_id=step,sampled_step=step,original_frame_index=step*7,boxes=np.array([[x,0,x+1,1] for x in range(len(ids))],float).reshape(-1,4),released_ids=np.array(ids),predicted_categories=np.ones(len(ids),int),raw=np.zeros((len(ids),256)),semantic=np.zeros((len(ids),512)))
    a=independent_history(f(0,[4,-1]),history);b=independent_history(f(1,[4,4]),history);c=independent_history(f(2,[4]),history);d=independent_history(f(3,[4]),history)
    check(b["history_reason"]==["duplicate_current_id"]*2 and c["history_reason"]==["previous_duplicate_invalid"] and d["previous_times"].tolist()==[14.],"Independent duplicate history")
    check(independent_scores(d,c).shape==(1,1,6) and np.isnan(independent_scores(c,b)[:,:,4]).all(),"Independent feature/CV missing score")
    check(project('{"split":"tune","gt_track_id":{"poison":1},"error":666}',{"split"})=={"split":"tune"},"Split-only poison projection")
    full_temporary_self_test()
    print(json.dumps(dict(status="PASS_INDEPENDENT_FIT_LAG_OUTPUT_AUDITOR_SYNTHETIC",real_cache_or_supervision_opened=False,checks=8,
        verifier_sha256=sha(__file__),producer_sha256=FINAL_PRODUCER_SHA,protocol_sha256=FINAL_PROTOCOL_SHA,preflight_sha256=FINAL_PREFLIGHT_SHA,
        full_temporary_audit_entry=True,synthetic_authority_overrides=["temporary plan/roles/frame execution/cache and protocol/preflight","direct synthetic legal anchor map; real identity payload never opened"],
        synthetic_fixture_producer_used_only_to_create_subject_output=True)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");parser.add_argument("--out",type=Path)
    arguments=parser.parse_args()
    if arguments.self_test:self_test()
    else:audit(arguments.out or ROOT/"results/fit_lag_identity_support_20261024_v1")
