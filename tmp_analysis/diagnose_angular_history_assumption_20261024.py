#!/usr/bin/env python3
"""Post-seal, descriptive angular-history diagnostics; never changes predictions.

A minimum-row eligible positive is a fixed diagnostic representative only.
Clean requires EVERY known positive to have an eligible known-same past anchor.
All positive rows and mixed signatures are retained; unknown is never negative.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"tmp_analysis/angular_appearance_control_protocol_20261024.json"
PROTOCOL_SHA="44d8bce20de4ed3fd358a083665e6e9d8b2744d29d86dcdc9b2d9efa2f07eb69"
HELPER=ROOT/"tmp_analysis/build_fit_identity_edge_support.py"
HELPER_SHA="e68f35ca48ba6f32d519efbdb33a711fc201cbf1a2f65c50f1c828a767b1d2b5"
SUPPORT=ROOT/"results/base_error_full_collection_20261023/support_v1"
STAGE=ROOT/"results/base_error_full_collection_20261023"
SCOPE="same_predicted_category"
POP="common_CV_covered_population"
RHO=[0,1,2,4,8,16,32,64,128]
ANGLE=[0,.01,.05,.1,.25,.5,1,2,float(np.pi)]
STATUSES=("known_same","known_different","unknown_with_locator","missing_locator")


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open("x") as f:f.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def module(path):
    spec=importlib.util.spec_from_file_location("postseal_eligible_identity_projector",path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def verify_seal(out):
    seal=json.loads((out/"GT_free_candidates_sealed.json").read_text())
    require(seal["status"]=="SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION","Missing global angular seal")
    for name,digest in seal["files_sha256"].items():require(sha(out/name)==digest,"Changed sealed file: "+name)
    return seal


def bin_index(value,edges):
    # 0=underflow, i=left-closed interval [edges[i-1], edges[i]), last=overflow.
    return None if value is None else int(np.searchsorted(edges,value,side="right"))


def angle(cosine):
    return float(np.arccos(np.clip(cosine,-1.,1.)))


def past_status(reference,row,anchors,video,identity):
    locator=reference["previous_locators"][row]
    if locator is None:return "missing_locator"
    value=anchors.get((locator["image_id"],locator["dump_row"]))
    require(value is None or value[0]==video,"Past identity crossed video")
    if value is None:return "unknown_with_locator"
    return "known_same" if value[1]==identity else "known_different"


def history_signature(positives):
    counts={status:sum(row["past_status"]==status for row in positives) for status in STATUSES}
    require(sum(counts.values())==len(positives) and len(positives)>0,"Bad positive history signature")
    if counts["known_same"]==len(positives):label="clean_all_known_same"
    elif counts["known_different"]==len(positives):label="all_known_different"
    elif counts["missing_locator"]==len(positives):label="all_missing_locator"
    elif counts["unknown_with_locator"]==len(positives):label="all_unknown_with_locator"
    else:label="mixed"
    return dict(label=label,positive_reference_counts=counts,presence_signature="|".join(s for s in STATUSES if counts[s]))


def positive_record(reference,row,query_row,block,diag,anchors,video,qgt,query_zero):
    valid=not query_zero and not diag["last_zero_norm"][row]
    angular_history_valid=not diag["last_zero_norm"][row] and not diag["missing_or_zero_history"][row]
    rho=float(diag["extrapolation_ratio"][row]) if angular_history_valid else None
    theta=float(diag["history_angle_radians"][row]) if angular_history_valid else None
    residual=angle(block[query_row,row,1]) if valid else None
    raw_residual=angle(block[query_row,row,0]) if valid else None
    hypothetical_turn=None if rho is None else rho*theta
    # The sealed core falls back to normalized last on an antipodal arc. Its
    # actual turn is zero even though hypothetical rho*theta may equal pi.
    turn=None if diag["last_zero_norm"][row] else (hypothetical_turn if diag["directed_arc"][row] else 0.)
    return dict(reference_dump_row=int(row),reference_image_id=reference["image_id"],
        past_locator=reference["previous_locators"][row],past_status=past_status(reference,row,anchors,video,qgt),
        rho=rho,history_angle_radians=theta,rho_bin=bin_index(rho,RHO),history_angle_bin=bin_index(theta,ANGLE),
        angular_to_query_residual_radians=residual,raw_to_query_residual_radians=raw_residual,
        residual_bin=bin_index(residual,ANGLE),unwrapped_forecast_turn_radians=turn,
        hypothetical_rho_theta_radians=hypothetical_turn,
        principal_forecast_turn_radians=None if turn is None else abs(float(np.arctan2(np.sin(turn),np.cos(turn)))),
        angular_residual_valid=valid,angular_history_valid=angular_history_valid,
        query_zero_norm=bool(query_zero),last_zero_norm=bool(diag["last_zero_norm"][row]),
        directed_arc=bool(diag["directed_arc"][row]),coincident_static=bool(diag["coincident_static"][row]),
        antipodal_fallback=bool(diag["antipodal_fallback"][row]),previous_zero_norm=bool(diag["previous_zero_norm"][row]))


def make_query(source,lag,video,query,reference,qr,qids,rids,block,diag,anchors,query_zero):
    same=np.asarray(reference["predicted_categories"])==query["predicted_categories"][qr]
    cv=np.array([x is not None for x in reference["previous_times"]],bool)
    known=same&cv&(rids>=0)&(qids[qr]>=0);refs=np.flatnonzero(known)
    positive=refs[rids[refs]==qids[qr]];negative=refs[rids[refs]!=qids[qr]]
    if not len(positive) or not len(negative):return None,[]
    winners=[int(refs[np.lexsort((refs,-block[qr,refs,column]))[0]]) for column in range(2)]
    correct=[bool(rids[row]==qids[qr]) for row in winners]
    positives=[positive_record(reference,int(row),qr,block,diag,anchors,video,int(qids[qr]),query_zero) for row in positive]
    signature=history_signature(positives)
    representative=positives[0]  # refs sorted by immutable dump_row, no score-dependent choice.
    row=dict(source=source,lag=lag,video_id=video,query_image_id=query["image_id"],query_dump_row=qr,query_gt=int(qids[qr]),
        positive_count=len(positive),negative_count=len(negative),raw_correct=correct[0],angular_correct=correct[1],
        repair=bool(not correct[0] and correct[1]),break_raw=bool(correct[0] and not correct[1]),
        raw_winner_dump_row=winners[0],angular_winner_dump_row=winners[1],
        raw_winner_past_status=past_status(reference,winners[0],anchors,video,int(rids[winners[0]])),
        angular_winner_past_status=past_status(reference,winners[1],anchors,video,int(rids[winners[1]])),
        history_signature=signature,representative_positive=representative,
        all_positive_mean_angular_residual=float(np.mean([p["angular_to_query_residual_radians"] for p in positives if p["angular_residual_valid"]])) if any(p["angular_residual_valid"] for p in positives) else None,
        all_positive_mean_raw_residual=float(np.mean([p["raw_to_query_residual_radians"] for p in positives if p["angular_residual_valid"]])) if any(p["angular_residual_valid"] for p in positives) else None)
    return row,positives


def outcome(rows):
    total=len(rows)
    result=dict(queries=total,raw_correct=sum(r["raw_correct"] for r in rows),angular_correct=sum(r["angular_correct"] for r in rows),
        repair=sum(r["repair"] for r in rows),break_raw=sum(r["break_raw"] for r in rows),sources=len({r["source"] for r in rows}),
        query_GT_trajectories=len({(r["video_id"],r["query_gt"]) for r in rows}))
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[r["source"]][(r["video_id"],r["query_gt"])].append(r)
    for name,col in (("raw","raw_correct"),("angular","angular_correct"),("all_positive_angular_residual","all_positive_mean_angular_residual"),("all_positive_raw_residual","all_positive_mean_raw_residual")):
        bysource=[]
        for trajectories in groups.values():
            values=[]
            for queries in trajectories.values():
                present=[r[col] for r in queries if r[col] is not None]
                if present:values.append(float(np.mean(present)))
            if values:bysource.append(float(np.mean(values)))
        result[name+"_source_GTtrajectory_query_macro"]=float(np.mean(bysource)) if bysource else None
    return result


def summarize(rows,sources):
    result={}
    for lag in (1,4):
        selected=[r for r in rows if r["lag"]==lag]
        strata={label:outcome([r for r in selected if r["history_signature"]["label"]==label]) for label in
            ("clean_all_known_same","all_known_different","all_missing_locator","all_unknown_with_locator","mixed")}
        signatures=sorted({r["history_signature"]["presence_signature"] for r in selected})
        bins={}
        for field,edges in (("rho_bin",RHO),("history_angle_bin",ANGLE),("residual_bin",ANGLE)):
            bins[field]=dict(boundaries=edges,representative_only=True,
                bins={str(i):outcome([r for r in selected if r["representative_positive"][field]==i]) for i in range(len(edges)+1)},
                unavailable=outcome([r for r in selected if r["representative_positive"][field] is None]))
        result[str(lag)]=dict(all_hard=outcome(selected),history_strata=strata,
            mixed_presence_signatures={key:outcome([r for r in selected if r["history_signature"]["presence_signature"]==key]) for key in signatures},
            representative_bins=bins,per_source={source:outcome([r for r in selected if r["source"]==source]) for source in sources})
    a={(r["video_id"],r["query_image_id"],r["query_dump_row"]):r for r in rows if r["lag"]==1}
    b={(r["video_id"],r["query_image_id"],r["query_dump_row"]):r for r in rows if r["lag"]==4}
    common=sorted(a.keys()&b.keys())
    result["cross_lag_shared"]=dict(queries=len(common),sources=len({a[k]["source"] for k in common}),
        lag1=outcome([a[k] for k in common]),lag4=outcome([b[k] for k in common]))
    return result


def positive_summary(rows):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in rows:
        if row["angular_residual_valid"]:
            groups[row["source"]][(row["video_id"],row["query_gt"])][(row["query_image_id"],row["query_dump_row"])].append(row)
    result=dict(positive_rows=len(rows),valid_residual_rows=sum(r["angular_residual_valid"] for r in rows),
        queries=len({(r["video_id"],r["query_image_id"],r["query_dump_row"]) for r in rows}),
        sources=len({r["source"] for r in rows}),query_GT_trajectories=len({(r["video_id"],r["query_gt"]) for r in rows}))
    for name in ("angular_to_query_residual_radians","raw_to_query_residual_radians"):
        bysource=[]
        for trajectories in groups.values():
            bytrajectory=[]
            for queries in trajectories.values():
                bytrajectory.append(float(np.mean([np.mean([r[name] for r in positives]) for positives in queries.values()])))
            bysource.append(float(np.mean(bytrajectory)))
        result[name+"_source_GTtrajectory_query_positive_macro"]=float(np.mean(bysource)) if bysource else None
    return result


def direction_and_speed(previous,last,query,previous_time,last_time,query_time):
    """Post-seal three-anchor diagnosis in the reference tangent space."""
    arrays=[np.asarray(x,np.float64) for x in (previous,last,query)]
    require(all(x.ndim==1 and x.shape==arrays[0].shape and np.isfinite(x).all() for x in arrays),"Invalid diagnostic raw features")
    norms=[np.linalg.norm(x) for x in arrays]
    for name,norm in zip(("previous","last","query"),norms):
        if norm==0:return dict(valid=False,reason="zero_"+name)
    require(previous_time<last_time<query_time,"Noncausal diagnostic timestamps")
    p,m,q=[x/norm for x,norm in zip(arrays,norms)]
    cp=float(np.clip(p@m,-1.,1.));cq=float(np.clip(q@m,-1.,1.))
    backward=p-cp*m;observed=q-cq*m;sp=np.linalg.norm(backward);sq=np.linalg.norm(observed)
    eps=64*np.finfo(np.float64).eps
    if sp<=eps:return dict(valid=False,reason="undefined_past_tangent")
    if sq<=eps:return dict(valid=False,reason="undefined_observed_tangent")
    past_angle=float(np.arctan2(sp,cp));observed_angle=float(np.arctan2(sq,cq))
    past_speed=past_angle/(last_time-previous_time);observed_speed=observed_angle/(query_time-last_time)
    return dict(valid=True,reason="valid",forward_observed_tangent_dot=float(np.clip((-backward/sp)@(observed/sq),-1.,1.)),
        past_original_frame_angular_speed=past_speed,observed_original_frame_angular_speed=observed_speed,
        observed_over_past_speed_ratio=observed_speed/past_speed,
        signed_angular_speed_error=observed_speed-past_speed,absolute_angular_speed_error=abs(observed_speed-past_speed))


def directional_summary(rows):
    clean=[r for r in rows if r.get("query_allpositives_clean",False)]
    valid=[r for r in clean if r.get("directional",{}).get("valid",False)]
    result=dict(all_positive_rows=len(rows),known_same_past_rows=sum(r["past_status"]=="known_same" for r in rows),
        strict_clean_query_positive_rows=len(clean),valid_clean_triples=len(valid),invalid_clean_triples=len(clean)-len(valid),
        invalid_clean_reasons=dict(Counter(r.get("directional",{}).get("reason","not_collected") for r in clean if not r.get("directional",{}).get("valid",False))),
        clean_rule="Every known positive in this query has a locator and eligible known-same past identity; direction additionally requires all three raw vectors nonzero and both tangents nondegenerate.")
    for field in ("forward_observed_tangent_dot","past_original_frame_angular_speed","observed_original_frame_angular_speed","observed_over_past_speed_ratio","signed_angular_speed_error","absolute_angular_speed_error"):
        groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for r in valid:groups[r["source"]][(r["video_id"],r["query_gt"])][(r["query_image_id"],r["query_dump_row"])].append(r["directional"][field])
        means=[np.mean([np.mean([np.mean(v) for v in queries.values()]) for queries in trajectories.values()]) for trajectories in groups.values()]
        result[field+"_source_GTtrajectory_query_positive_macro"]=float(np.mean(means)) if means else None
    return result


def all_positive_summary(rows):
    result={}
    for lag in (1,4):
        selected=[r for r in rows if r["lag"]==lag]
        bins={}
        for field,edges in (("rho_bin",RHO),("history_angle_bin",ANGLE),("residual_bin",ANGLE)):
            bins[field]=dict(boundaries=edges,bins={str(i):positive_summary([r for r in selected if r[field]==i]) for i in range(len(edges)+1)},
                unavailable=positive_summary([r for r in selected if r[field] is None]))
        grouped=defaultdict(list)
        for row in selected:grouped[(row["past_status"],row["rho_bin"],row["history_angle_bin"])].append(row)
        cross={status:{str(rb):{str(ab):positive_summary(grouped[(status,rb,ab)]) for ab in [None]+list(range(len(ANGLE)+1))}
            for rb in [None]+list(range(len(RHO)+1))} for status in STATUSES}
        result[str(lag)]=dict(all_positives=positive_summary(selected),
            past_status={status:positive_summary([r for r in selected if r["past_status"]==status]) for status in STATUSES},bins=bins,
            past_status_rho_history_angle_full_cross=cross,strict_clean_direction_and_speed=directional_summary(selected))
    return result


def load_anchors(out,roles,video_sources):
    verify_seal(out)  # Must precede helper import or any identity payload decode.
    require(sha(HELPER)==HELPER_SHA,"Identity projector changed")
    helper=module(HELPER);manual=SUPPORT/"supervised_observations.jsonl"
    receipt=helper.project_object((SUPPORT/"support_receipt.json").read_text(),{"output_files"})
    require(sha(manual)==receipt["output_files"][manual.name],"Manual identities changed")
    identities=helper.load_fit_identities(manual,roles)
    manifest=json.loads((SUPPORT/"gt_free_index_manifest.json").read_text());index=SUPPORT/"gt_free_observations.jsonl"
    require(sha(index)==manifest["files"][index.name],"GTfree index changed")
    anchors={}
    with index.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row["split"]!="fit":continue
            identity=identities.get(row["observation_id"])
            if identity is None:continue
            require(video_sources[str(row["video_id"])]==row["source_group"],"Source/video mismatch")
            require(all(row[k]==identity[k] for k in ("video_id","source_group","local_track_id","image_id","sampled_step","original_frame_index","pred_category_id")),"Identity immutable row mismatch")
            key=(row["image_id"],row["dump_row"]);require(key not in anchors,"Duplicate anchor")
            anchors[key]=(row["video_id"],identity["gt_track_id"])
    require(len(anchors)==len(identities),"Missing anchors")
    return anchors,{str(manual):sha(manual),str(index):sha(index)}


def diagnose(out,dest,anchors,sources,query_zero_loader,raw_loader=None):
    rows=[];positive_rows=[];video=None;metadata=None;max_block=0
    db=sqlite3.connect(f"file:{out/'support.sqlite'}?mode=ro",uri=True)
    with (dest/"queries.jsonl").open("x") as qstream,(dest/"positive_history.jsonl").open("x") as pstream,(out/"frame_pairs.jsonl").open() as stream:
        for line in stream:
            receipt=json.loads(line)
            if receipt["reference_step"] is None:continue
            if video!=receipt["video_id"]:
                video=receipt["video_id"];metadata=json.loads((out/"shards"/f"v{video}.rows.json").read_text())["frames"]
            q=metadata[receipt["query_step"]];r=metadata[receipt["reference_step"]]
            nq,nm=receipt["query_rows"],receipt["reference_rows"];max_block=max(max_block,nq*nm)
            scores=np.load(out/receipt["score_file"],mmap_mode="r",allow_pickle=False)
            block=scores[receipt["offset"]:receipt["offset"]+receipt["candidates"]].copy().reshape(nq,nm,2);del scores
            qids=np.array([anchors.get((q["image_id"],i),(video,-1))[1] for i in range(nq)])
            rids=np.array([anchors.get((r["image_id"],i),(video,-1))[1] for i in range(nm)])
            queryzero=query_zero_loader(q)
            for qr in range(nq):
                row,positives=make_query(receipt["source_group"],receipt["lag"],video,q,r,qr,qids,rids,block,receipt["forecast"],anchors,bool(queryzero[qr]))
                if row is None:continue
                expected=db.execute("SELECT p,n,raw_correct,correct_bits FROM query_support WHERE lag=? AND scope=? AND population=? AND video=? AND image=? AND qrow=?",(receipt["lag"],SCOPE,POP,video,q["image_id"],qr)).fetchone()
                require(expected==(row["positive_count"],row["negative_count"],int(row["raw_correct"]),int(row["raw_correct"])+2*int(row["angular_correct"])),"Query differs from sealed producer query report")
                rows.append(row);qstream.write(json.dumps(row,separators=(",",":"))+"\n")
                for positive in positives:
                    payload=dict(source=row["source"],lag=row["lag"],video_id=video,query_image_id=q["image_id"],query_dump_row=qr,query_gt=int(qids[qr]),**positive)
                    payload["query_allpositives_clean"]=row["history_signature"]["label"]=="clean_all_known_same"
                    payload["directional"]=dict(valid=False,reason="not_known_same_past")
                    if positive["past_status"]=="known_same":
                        if raw_loader is None:payload["directional"]=dict(valid=False,reason="not_collected")
                        else:
                            previous,pt=raw_loader(positive["past_locator"]["image_id"])
                            last,mt=raw_loader(r["image_id"]);queryraw,qt=raw_loader(q["image_id"])
                            require(pt==positive["past_locator"]["original_frame_index"] and mt==r["original_frame_index"] and qt==q["original_frame_index"],"Raw diagnostic time differs from sealed locator")
                            payload["directional"]=direction_and_speed(previous[positive["past_locator"]["dump_row"]],last[positive["reference_dump_row"]],queryraw[qr],pt,mt,qt)
                    positive_rows.append(payload);pstream.write(json.dumps(payload,separators=(",",":"))+"\n")
    expected=db.execute("SELECT COUNT(*) FROM query_support WHERE scope=? AND population=? AND p>0 AND n>0",(SCOPE,POP)).fetchone()[0]
    db.close();require(len(rows)==expected,"Hard query population was dropped")
    results=summarize(rows,sources);results["all_positive_descriptive_support"]=all_positive_summary(positive_rows)
    return results,max_block


def execute(out,dest):
    require(sha(PROTOCOL)==PROTOCOL_SHA,"Angular protocol changed");protocol=json.loads(PROTOCOL.read_text())
    for p,digest in protocol["input_pins"].items():require(sha(p)==digest,"Protected input changed")
    seal=verify_seal(out);summary=json.loads((out/"summary.json").read_text())
    require(summary["status"]=="COMPLETE_ALL_SEALED_FIT_ANGULAR_CONTROL_PENDING_INDEPENDENT_AUDIT","Incomplete angular output")
    audit=json.loads((out/"independent_output_audit.json").read_text())
    require(audit["status"]=="PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_OUTPUT_AUDIT","Independent output audit required")
    require(seal["protocol_sha256"]==summary["protocol_sha256"]==PROTOCOL_SHA,"Output protocol mismatch")
    require(audit["protocol_sha256"]==PROTOCOL_SHA and audit["summary_sha256"]==sha(out/"summary.json") and
        audit["producer_script_sha256"]==summary["script_sha256"] and audit["support_SQLite_sha256"]==sha(out/"support.sqlite") and
        audit["global_GT_free_seal_sha256"]==sha(out/"GT_free_candidates_sealed.json"),"Independent audit does not bind these exact outputs")
    require(not dest.exists(),"Preserve existing diagnostic output")
    protected={str(out/"summary.json"):sha(out/"summary.json"),str(out/"independent_output_audit.json"):sha(out/"independent_output_audit.json"),str(out/"support.sqlite"):sha(out/"support.sqlite"),str(PROTOCOL):sha(PROTOCOL),str(Path(__file__).resolve()):sha(__file__)}
    roles=json.loads((STAGE/"pair_context_v1/source_roles.json").read_text())["groups"]
    sources=sorted(s for s,v in roles.items() if v["split"]=="fit")
    plan=json.loads((STAGE/"full_collection_plan.json").read_text());videos={str(r["video_id"]):r["source_group"] for r in plan["video_records"] if r["split"]=="fit"}
    anchors,identpins=load_anchors(out,roles,videos);protected.update(identpins)
    caches={}
    with (STAGE/"full_collect_v1/frame_execution.jsonl").open() as f:
        for line in f:
            row=json.loads(line)
            if str(row["video_id"]) in videos:caches[row["image_id"]]=row
    @lru_cache(maxsize=5)
    def loadraw(image):
        record=caches[image];require(sha(record["npz"])==record["npz_sha256"],"Cache changed")
        with np.load(record["npz"],allow_pickle=False) as stored:raw=stored["track_feats"].astype(np.float64)
        return raw,record["frame_index"]
    def queryzeros(frame):return np.linalg.norm(loadraw(frame["image_id"])[0],axis=1)==0
    dest.mkdir();results,maxblock=diagnose(out,dest,anchors,sources,queryzeros,loadraw)
    require(results["cross_lag_shared"]["queries"]==1331 and results["1"]["all_hard"]["queries"]==2949 and results["4"]["all_hard"]["queries"]==2142,"Fixed primary support changed")
    verify_seal(out);require(all(sha(p)==digest for p,digest in protected.items()),"Protected input changed during diagnostics")
    write(dest/"summary.json",dict(status="COMPLETE_READ_ONLY_ANGULAR_HISTORY_ASSUMPTION_DIAGNOSTIC",results=results,
        scope=SCOPE,population=POP,query_subset="positive_and_negative_available_known_queries",
        scope_limit="Only the fixed primary scope/population hard queries were diagnosed; other scopes/populations are not covered by this output.",
        input_sha256=protected,script_sha256=sha(__file__),maximum_candidate_block_read=maxblock,
        representative_rule="Minimum immutable dump_row among all eligible known positive references; diagnostic only, never a selected prediction.",
        clean_rule="All known positive reference rows have a locator and eligible past identity equal to endpoint GT; mixed and unknown are separate. Unlabelled retained candidates remain unknown.",
        residual_macro="Per query mean over every valid known positive residual, then query mean within GTtrajectory, trajectory mean within source, source mean. Representative bins are not all-positive expectations.",
        training_started=False,predictions_modified=False,unknown_is_negative=False,
        files_sha256={name:sha(dest/name) for name in ("queries.jsonl","positive_history.jsonl")}))
    print(json.dumps(dict(status="COMPLETE_READ_ONLY_ANGULAR_HISTORY_ASSUMPTION_DIAGNOSTIC",out=str(dest),queries=sum(results[str(l)]["all_hard"]["queries"] for l in (1,4)))))


def toy():
    reference=dict(image_id=10,predicted_categories=[1]*3,previous_times=[0,0,0],previous_locators=[dict(image_id=1,dump_row=i,original_frame_index=0) for i in range(3)])
    query=dict(image_id=20,predicted_categories=[1])
    diag={key:[False]*3 for key in ("last_zero_norm","missing_or_zero_history","directed_arc","coincident_static","antipodal_fallback","previous_zero_norm")}
    diag.update(extrapolation_ratio=[1.,4.,2.],history_angle_radians=[.1,.5,1.]);diag["directed_arc"]=[True]*3
    return query,reference,diag


def synthetic(number):
    checks=[]
    if number==1:
        q,r,d=toy();scores=np.array([[[.1,.9],[.9,.1],[.8,.2]]]);anchors={(1,0):(1,7),(1,1):(1,9)}
        row,positives=make_query("S",1,1,q,r,0,np.array([7]),np.array([7,7,9]),scores,d,anchors,False)
        require(row["history_signature"]["label"]=="mixed" and row["history_signature"]["positive_reference_counts"]["known_same"]==row["history_signature"]["positive_reference_counts"]["known_different"]==1,"Mixed positive histories collapsed")
        require(row["representative_positive"]["reference_dump_row"]==0 and row["raw_winner_dump_row"]==1 and row["angular_winner_dump_row"]==0 and len(positives)==2,"Representative confused with winning candidate")
        checks.append("all_positive_mixed_clean_signature_and_fixed_representative_separate_from_winners")
        anchors={};row,positives=make_query("S",1,1,q,r,0,np.array([7]),np.array([7,-1,9]),scores,d,anchors,False)
        require(row["positive_count"]==row["negative_count"]==1 and row["history_signature"]["label"]=="all_unknown_with_locator","Unknown endpoint/history became negative/clean")
        require(make_query("S",1,1,q,r,0,np.array([7]),np.array([7,-1,-1]),scores,d,anchors,False)==(None,[]),"Nonhard query admitted")
        checks.append("unknown_endpoint_never_negative_and_unknown_past_never_clean_nonhard_excluded")
    elif number==2:
        require([bin_index(x,RHO) for x in (-1,0,1,128,129)]==[0,1,2,9,9] and bin_index(None,RHO) is None,"Fixed bin boundary rule differs")
        q,r,d=toy();scores=np.zeros((1,3,2));scores[0,0]=[0.,1.];anchors={(1,0):(1,7)}
        p=positive_record(r,0,0,scores,d,anchors,1,7,False)
        require(p["angular_to_query_residual_radians"]==0. and abs(p["raw_to_query_residual_radians"]-np.pi/2)<1e-14,"Injected exact forecast residual wrong")
        d["extrapolation_ratio"][0]=80.;d["history_angle_radians"][0]=np.pi/4
        p=positive_record(r,0,0,scores,d,anchors,1,7,False)
        require(p["unwrapped_forecast_turn_radians"]>2*np.pi and p["principal_forecast_turn_radians"]<1e-13,"Multi-turn diagnostics silently clipped")
        checks.append("predeclared_bin_boundaries_injected_exact_direction_and_unclipped_multiturn")
        d["directed_arc"][0]=False;d["antipodal_fallback"][0]=True
        d["extrapolation_ratio"][0]=1.;d["history_angle_radians"][0]=np.pi
        p=positive_record(r,0,0,scores,d,anchors,1,7,False)
        require(p["unwrapped_forecast_turn_radians"]==p["principal_forecast_turn_radians"]==0. and p["hypothetical_rho_theta_radians"]==np.pi,"Antipodal fallback falsely reported a pi forecast turn")
        d["antipodal_fallback"][0]=False;d["missing_or_zero_history"][0]=True
        p=positive_record(r,0,0,scores,d,anchors,1,7,False)
        require(p["unwrapped_forecast_turn_radians"]==0. and p["hypothetical_rho_theta_radians"] is None,"Missing-history fallback actual turn differs")
        checks.append("antipodal_and_missing_history_actual_zero_turn_separate_from_hypothetical_rho_theta")
        d["last_zero_norm"][0]=True;p=positive_record(r,0,0,scores,d,anchors,1,7,False)
        require(p["angular_to_query_residual_radians"] is None and p["rho"] is None,"Zero-last angle treated as real")
        d["last_zero_norm"][0]=False;p=positive_record(r,0,0,scores,d,anchors,1,7,True)
        require(p["angular_to_query_residual_radians"] is None,"Zero-query angle treated as real")
        checks.append("zero_last_query_residual_invalid_and_missing_angular_history_not_fake_zero_bin")
        diag=direction_and_speed([1.,0.],np.sqrt(.5)*np.ones(2),[0.,1.],0,10,20)
        reverse=direction_and_speed([1.,0.],np.sqrt(.5)*np.ones(2),[1.,0.],0,10,20)
        require(diag["valid"] and abs(diag["forward_observed_tangent_dot"]-1)<1e-14 and abs(diag["observed_over_past_speed_ratio"]-1)<1e-14 and abs(reverse["forward_observed_tangent_dot"]+1)<1e-14,"Three-anchor forward tangent/speed oracle differs")
        require(not direction_and_speed([1.,0.],[-1.,0.],[0.,1.],0,10,20)["valid"] and direction_and_speed([1.,0.],[0.,1.],[0.,0.],0,10,20)["reason"]=="zero_query","Degenerate triple denominator invalidity differs")
        checks.append("clean_three_anchor_tangent_direction_original_frame_speed_and_degenerate_denominators")
    elif number==3:
        with tempfile.TemporaryDirectory(prefix="angular_history_seal_") as temp:
            out=Path(temp);part=out/"scores.npy";np.save(part,np.zeros((1,2)))
            write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION",files_sha256={part.name:sha(part)}))
            verify_seal(out);part.write_bytes(b"tampered")
            try:load_anchors(out,{}, {})
            except ValueError:pass
            else:raise AssertionError("Identity loading preceded seal failure")
        checks.append("tampered_global_seal_blocks_identity_helper_import")
        q,r,d=toy();scores=np.array([[[.1,.9],[.9,.1],[.8,.2]]]);anchors={(1,0):(1,7),(1,1):(1,7)};rows=[]
        for lag in (1,4):
            row,positives=make_query("S",lag,1,q,r,0,np.array([7]),np.array([7,7,9]),scores,d,anchors,False);rows.append(row)
        result=summarize(rows,["S","ZERO"])
        require(result["cross_lag_shared"]["queries"]==1 and result["1"]["history_strata"]["clean_all_known_same"]["queries"]==1 and result["4"]["per_source"]["ZERO"]["queries"]==0,"Crosslag/zero-source/clean support differs")
        require(result["1"]["all_hard"]["all_positive_angular_residual_source_GTtrajectory_query_macro"]==rows[0]["all_positive_mean_angular_residual"],"All-positive residual macro replaced by representative")
        require(len(result["1"]["representative_bins"]["rho_bin"]["bins"])==len(RHO)+1,"Empty fixed bins dropped")
        payloads=[]
        for positive in positives:payloads.append(dict(source="S",lag=1,video_id=1,query_image_id=20,query_dump_row=0,query_gt=7,query_allpositives_clean=True,directional=dict(valid=False,reason="undefined_observed_tangent"),**positive))
        cross=all_positive_summary(payloads)["1"]["past_status_rho_history_angle_full_cross"]
        require(len(cross)==4 and len(cross["known_same"])==len(RHO)+2 and len(cross["known_same"]["None"])==len(ANGLE)+2 and sum(x["positive_rows"] for status in cross.values() for rb in status.values() for x in rb.values())==2,"Complete all-positive history x rho x theta cross lost zero cells or rows")
        checks.append("endtoend_query_strata_crosslag_zero_source_allpositive_macro_and_emptybins")
        with tempfile.TemporaryDirectory(prefix="angular_history_endtoend_") as temp:
            out=Path(temp)/"out";out.mkdir();(out/"shards").mkdir();dest=Path(temp)/"diagnostic";dest.mkdir()
            q,r,d=toy();r2=dict(r,image_id=11);metadata=[dict(),r,dict(),dict(),r2,q]
            write(out/"shards/v1.rows.json",dict(frames=metadata));receipts=[]
            for lag,step,refimage in ((1,4,11),(4,1,10)):
                filename=f"shards/v1_lag{lag}.scores.npy";np.save(out/filename,scores.reshape(3,2))
                receipts.append(dict(video_id=1,source_group="S",lag=lag,reference_step=step,query_step=5,query_rows=1,reference_rows=3,offset=0,candidates=3,score_file=filename,forecast=d))
            (out/"frame_pairs.jsonl").write_text("".join(json.dumps(x)+"\n" for x in receipts))
            db=sqlite3.connect(out/"support.sqlite")
            db.execute("CREATE TABLE query_support(lag INTEGER,scope TEXT,population TEXT,video INTEGER,image INTEGER,qrow INTEGER,p INTEGER,n INTEGER,raw_correct INTEGER,correct_bits INTEGER)")
            db.executemany("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?)",[(lag,SCOPE,POP,1,20,0,2,1,1,3) for lag in (1,4)]);db.commit();db.close()
            anchors={(1,0):(1,7),(1,1):(1,7),(20,0):(1,7)}
            for image in (10,11):
                for row,identity in enumerate((7,7,9)):anchors[(image,row)]=(1,identity)
            result,maxblock=diagnose(out,dest,anchors,["S","ZERO"],lambda _:np.array([False]))
            require(maxblock==3 and result["cross_lag_shared"]["queries"]==1 and result["1"]["all_hard"]["queries"]==1 and
                result["all_positive_descriptive_support"]["4"]["all_positives"]["positive_rows"]==2 and
                len((dest/"queries.jsonl").read_text().splitlines())==2 and len((dest/"positive_history.jsonl").read_text().splitlines())==4,"Full disk stream query/all-positive reconstruction differs")
        checks.append("full_temporary_metadata_scores_query_SQLite_stream_exact_support_allpositive_outputs")
    else:raise ValueError("Three rounds only")
    return dict(status="PASS_ANGULAR_HISTORY_ASSUMPTION_SYNTHETIC",round=number,checks=checks,script_sha256=sha(__file__),real_payload_opened=False,predictions_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--out",type=Path)
    parser.add_argument("--diagnostic-out",type=Path);parser.add_argument("--synthetic-round",type=int,choices=(1,2,3));parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    require(not(args.execute and args.synthetic_round),"Exclusive real/synthetic modes")
    if args.synthetic_round:
        receipt=synthetic(args.synthetic_round)
        if args.synthetic_out:write(args.synthetic_out,receipt)
        print(json.dumps(receipt,indent=2))
    else:require(args.execute and args.out and args.diagnostic_out,"Explicit execute/out/diagnostic-out required");execute(args.out,args.diagnostic_out)


if __name__=="__main__":main()
