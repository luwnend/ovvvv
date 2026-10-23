"""Fresh independent output audit. No producer imports, model or outcome IO."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from verify_fit_identity_edge_support_output import project

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
RUN=STAGE/"full_collect_v1"
PROTOCOL=ROOT/"tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
PREFLIGHT=ROOT/"tmp_analysis/context_feature_memory_independent_preflight_20261024.json"
MODES=("released","raw_confidence","raw_only","raw_store_one_sided","raw_store_symmetric")
PINS={PROTOCOL:"740d6c782fdfe8240ddc8c4b924e566cebd2465a6a05233f02197b1e904aaf1b",
      ROOT/"tmp_analysis/replay_context_feature_memory.py":"35e635ac66ee27da7561cac8c4b723d67776a2534f4b56faa5da5fa80a6d208b",
      ROOT/"tmp_analysis/context_feature_memory.py":"4885de8169a5bf5177d9153907d18faa401811549681f535c59f320fd52abaa4",
      ROOT/"tmp_analysis/verify_fit_identity_edge_support_output.py":"862d31f63331a288804d808fb303e6704d6f4fdfa80f97daf13971f44c0f795d",
      STAGE/"support_v1/support_receipt.json":"e206051d0809b8e20b24529254eacece96f997c72b7155e8264197e7c07aae1e",
      STAGE/"support_v1/gt_free_index_manifest.json":"7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667",
      RUN/"run_manifest.json":"8a99676a2a45a61d3c76d423480f59381cfe4ab80e0a54df671417a5599198ab"}


def check(value,message):
    if not value:raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def close(actual,expected,where=""):
    if isinstance(expected,dict):
        check(set(actual)==set(expected),"Keys differ: "+where)
        for k,v in expected.items():close(actual[k],v,where+"/"+k)
    elif isinstance(expected,list):
        check(len(actual)==len(expected),"List length differs: "+where)
        for i,v in enumerate(expected):close(actual[i],v,where+"/"+str(i))
    elif isinstance(expected,float):
        check(abs(actual-expected)<=1e-12,"Metric differs: "+where)
    else:check(actual==expected,"Value differs: "+where)


def partition(values):
    # Independent pair-equivalence canonicalization, with unique negative IDs.
    groups={};result=[]
    for index,value in enumerate(values):
        key=("negative",index) if value<0 else value
        result.append(groups.setdefault(key,len(groups)))
    return result


def metrics(rows):
    """Contingency matrices; no producer helper, no old-ID pair restriction."""
    sources={}
    for source in sorted({r["source_group"] for r in rows}):
        subset=[r for r in rows if r["source_group"]==source]
        labels=sorted({(r["video_id"],r["gt_track_id"]) for r in subset})
        outputs=[]
        for r in subset:
            outputs.append(("id",r["video_id"],r["new_id"]) if r["new_id"]>=0
                           else ("singleton",r["observation_id"]))
        clusters=sorted(set(outputs),key=str)
        ymap={v:i for i,v in enumerate(labels)};zmap={v:i for i,v in enumerate(clusters)}
        table=np.zeros((len(labels),len(clusters)),np.int64)
        for r,z in zip(subset,outputs):table[ymap[r["video_id"],r["gt_track_id"]],zmap[z]]+=1
        gt_counts=table.sum(1);pred_counts=table.sum(0)
        p=rscore=0.
        for y in range(len(labels)):
            # Within a GT trajectory every one of its observations has 1/ngt
            # mass, then trajectories share a source's total mass equally.
            for z in np.flatnonzero(table[y]):
                count=int(table[y,z]);mass=count/int(gt_counts[y])/len(labels)
                p+=mass*count/int(pred_counts[z]);rscore+=mass*count/int(gt_counts[y])
        sources[source]=dict(observations=len(subset),official_trajectories=len(labels),precision=p,recall=rscore,
                             f1=2*p*rscore/(p+rscore),unassigned=sum(r["new_id"]<0 for r in subset))
    if not sources:return dict(sources=0,observations=0,precision=None,recall=None,f1=None,per_source={})
    p=sum(r["precision"] for r in sources.values())/len(sources)
    r=sum(r["recall"] for r in sources.values())/len(sources)
    return dict(sources=len(sources),observations=len(rows),precision=p,recall=r,f1=2*p*r/(p+r),per_source=sources)


def feature_headers(path):
    result={}
    with zipfile.ZipFile(path) as archive:
        for key,width in (("track_feats",256),("cls_feats",512)):
            with archive.open(key+".npy") as stream:
                version=np.lib.format.read_magic(stream)
                shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
            check(len(shape)==2 and shape[1]==width and dtype==np.dtype("float16"),"Cache feature precision/dimension changed")
            result[key]=dict(shape=list(shape),dtype=str(dtype))
    return result


def audit(out):
    started=time.monotonic();destination=out/"independent_output_audit.json"
    check(not destination.exists(),"Preserve prior independent audit")
    for path,digest in PINS.items():check(sha(path)==digest,"Pinned input changed: "+str(path))
    spec=load(PROTOCOL);summary=load(out/"summary.json");preflight=load(PREFLIGHT)
    check(summary["status"]=="COMPLETE_QUANTIZED_FIT_FEATURE_MEMORY_SMOKE_PENDING_INDEPENDENT_AUDIT","Unfinished producer")
    check(preflight["status"]=="PASS_INDEPENDENT_CONTEXT_FEATURE_MEMORY_IMPLEMENTATION_PREFLIGHT","Missing implementation gate")
    check(preflight["runner_sha256"]==PINS[ROOT/"tmp_analysis/replay_context_feature_memory.py"] and
          preflight["module_sha256"]==PINS[ROOT/"tmp_analysis/context_feature_memory.py"],"Stale implementation gate")
    close(load(out/"protocol.json"),spec,"protocol copy")
    close(summary["modes"],list(MODES),"mode scope");check(summary["primary"]=="raw_confidence","Primary changed")
    for flag in ("GT_used_in_predictions","category_correctness_decoded","tune_test_or_class_holdout_correctness_opened", "GPU_used","training_started","research_goal_complete"):
        check(summary[flag] is False,"Scope flag changed: "+flag)
    check(summary["predictions_sealed_before_manual_identity_join"] is True,"Prediction seal flag absent")
    inputs=load(out/"input_hashes.json")
    check(sha(out/"input_hashes.json")==summary["input_hashes_sha256"],"Input hash manifest changed")
    for path,digest in inputs.items():check(sha(path)==digest,"Producer recorded input changed: "+path)
    for path,digest in spec["input_pins"].items():check(inputs[path]==digest,"Protocol input missing")
    plan=load(STAGE/"full_collection_plan.json")
    video_records=plan["video_records"]
    fit_sources=sorted({v["source_group"] for v in video_records if v["split"]=="fit"},
                       key=lambda s:hashlib.sha256(("context_feature_memory_smoke_v1\0"+s).encode()).hexdigest())
    close(spec["source_groups"],fit_sources[:12],"GT-free source selection")
    all_frames=[json.loads(line) for line in (RUN/"frame_execution.jsonl").open()]
    selected=[f for f in all_frames if f["video_id"] in spec["video_ids"]]
    selected.sort(key=lambda f:(f["video_id"],f["frame_id"]))
    valid_videos={v["video_id"] for v in video_records if v["split"]=="fit" and v["source_group"] in spec["source_groups"]}
    nonempty_videos={f["video_id"] for f in all_frames}&valid_videos
    close(spec["video_ids"],sorted(nonempty_videos),"Complete selected videos")
    grouped=defaultdict(list)
    for frame in selected:grouped[frame["video_id"]].append(frame)
    for vid,frames in grouped.items():close([f["frame_id"] for f in frames],list(range(len(frames))),"Selected full timeline")
    check(summary["selected_frames"]==len(selected) and summary["selected_videos"]==len(grouped) and summary["selected_sources"]==12,"Frame/video/source counts")
    seal=load(out/"predictions_sealed.json")
    check(seal["status"]=="ALL_FIVE_GT_FREE_MODE_PREDICTIONS_SEALED_BEFORE_SUPERVISION","Missing five-mode seal")
    close(seal["output_files_sha256"],summary["output_files_sha256"],"Prediction output hashes")
    check(seal["protocol_sha256"]==PINS[PROTOCOL],"Prediction seal protocol")
    expected_names={m+".jsonl" for m in MODES}
    check(set(seal["output_files_sha256"])==expected_names,"Prediction mode files")
    watched={**inputs,**{str(p):sha(p) for p in (*PINS,PREFLIGHT,out/"summary.json",out/"predictions_sealed.json",out/"input_hashes.json",Path(__file__).resolve())}}
    predictions={};parity={};first_differences={};headers={};total_rows=0
    original_by_frame={}
    for frame in selected:
        path=Path(frame["npz"])
        check(inputs[str(path)]==frame["npz_sha256"]==sha(path),"Selected cache bytes changed")
        headers[path.name]=feature_headers(path)
        with np.load(path,allow_pickle=False) as saved:
            check(int(saved["image_id"])==frame["image_id"] and int(saved["video_id"])==frame["video_id"],"Cache frame identity")
            original=saved["track_ids"].tolist()
        check(len(original)==frame["observations"],"Retained observation count")
        original_by_frame[frame["image_id"]]=original;total_rows+=len(original)
    for mode in MODES:
        path=out/(mode+".jsonl");check(sha(path)==seal["output_files_sha256"][path.name],"Prediction bytes changed")
        watched[str(path)]=sha(path)
        records=[json.loads(line) for line in path.open()]
        check(len(records)==len(selected),"Mode frame count")
        by_video=defaultdict(lambda:[[],[]]);seen=set();predictions[mode]={};first_differences[mode]={}
        for record,frame in zip(records,selected):
            close({k:record[k] for k in ("video_id","image_id","sampled_step","original_frame_index")},
                  dict(video_id=frame["video_id"],image_id=frame["image_id"],sampled_step=frame["frame_id"],original_frame_index=frame["frame_index"]),"Prediction frame ordering")
            check(record["image_id"] not in seen,"Duplicate prediction frame");seen.add(record["image_id"])
            ids=record["ids"];original=original_by_frame[frame["image_id"]]
            check(len(ids)==len(original) and all(type(i) is int and i>=-1 for i in ids),"Dropped/invalid prediction row")
            positive=[i for i in ids if i>=0];check(len(positive)==len(set(positive)),"Duplicate ID in one frame")
            predictions[mode][record["image_id"]]=ids
            by_video[frame["video_id"]][0].extend(original);by_video[frame["video_id"]][1].extend(ids)
            if frame["video_id"] not in first_differences[mode]:
                diff=next((j for j,(a,b) in enumerate(zip(original,ids)) if a!=b),None)
                if diff is not None:first_differences[mode][frame["video_id"]]=dict(image_id=frame["image_id"],sampled_step=frame["frame_id"],dump_row=diff,original_id=original[diff],new_id=ids[diff])
        parity[mode]=[dict(video_id=vid,observations=len(original),row_id_differences=sum(a!=b for a,b in zip(original,new)),
                           canonical_partition_equal=partition(original)==partition(new)) for vid,(original,new) in sorted(by_video.items())]
        close(summary["parity_with_original_GPU_cache"][mode],parity[mode],"Original-versus-quantized replay parity")

    # Only identity fields are decoded, with split first; correctness/category
    # outcomes remain opaque. Existing immutable gate/source audits establish
    # eligibility; no new GT matching or old-track pair selection occurs.
    support=STAGE/"support_v1";receipt=project((support/"support_receipt.json").read_text(),{"output_files"})
    identity_path=support/"supervised_observations.jsonl"
    check(sha(identity_path)==receipt["output_files"][identity_path.name],"Manual source hash")
    identities={};tune_skipped=0;fields={"split","observation_id","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id","gt_track_id","manual_anchor_id","eligible_for_parameter_fit"}
    for line in identity_path.open():
        split=project(line,{"split"})["split"];check(split in ("fit","tune"),"Closed support split")
        if split=="tune":tune_skipped+=1;continue
        row=project(line,fields);check(set(row)==fields and row["eligible_for_parameter_fit"] is True,"Illegal manual identity")
        check(row["observation_id"] not in identities,"Duplicate identity observation")
        identities[row["observation_id"]]=row
    check(len(identities)==21661 and tune_skipped==2226,"Independent source scope changed")
    manifest=load(support/"gt_free_index_manifest.json");index=support/"gt_free_observations.jsonl"
    check(sha(index)==manifest["files"][index.name],"Immutable observation index")
    anchors={}
    for line in index.open():
        row=json.loads(line)
        if row["video_id"] not in grouped or row["observation_id"] not in identities:continue
        identity=identities[row["observation_id"]]
        for key in ("split","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id"):
            check(row[key]==identity[key],"Independent identity/observation provenance mismatch")
        check(row["split"]=="fit","Nonfit scoring anchor")
        key=(row["image_id"],row["dump_row"]);check(key not in anchors,"Duplicate anchor row")
        anchors[key]=dict(observation_id=row["observation_id"],source_group=row["source_group"],video_id=row["video_id"],gt_track_id=identity["gt_track_id"])
    measured={}
    for mode in MODES:
        rows=[]
        for (iid,row),anchor in anchors.items():
            check(row<len(predictions[mode][iid]),"Lost fixed scoring anchor")
            rows.append(dict(anchor,new_id=predictions[mode][iid][row]))
        measured[mode]=metrics(rows)
    intervention_difference={}
    for mode in MODES:
        video_details=[]
        differing_frames=0
        for vid,frames in sorted(grouped.items()):
            baseline=[];changed=[];changed_frames=0
            for frame in frames:
                a=predictions["released"][frame["image_id"]];b=predictions[mode][frame["image_id"]]
                baseline.extend(a);changed.extend(b)
                changed_frames+=int(a!=b)
            differing_frames+=changed_frames
            video_details.append(dict(video_id=vid,observations=len(baseline),frames=len(frames),
                frames_with_row_ID_difference=changed_frames,row_ID_differences=sum(a!=b for a,b in zip(baseline,changed)),
                canonical_partition_equal=partition(baseline)==partition(changed)))
        changed_anchors=sum(predictions["released"][iid][row]!=predictions[mode][iid][row] for iid,row in anchors)
        intervention_difference[mode]=dict(retained_observations=total_rows,
            row_ID_differences=sum(d["row_ID_differences"] for d in video_details),
            frames_with_row_ID_difference=differing_frames,videos_with_row_ID_difference=sum(d["row_ID_differences"]>0 for d in video_details),
            videos_with_partition_difference=sum(not d["canonical_partition_equal"] for d in video_details),
            eligible_anchors_with_row_ID_difference=changed_anchors,per_video=video_details)
    close(summary["scoring"]["metrics"],measured,"Independent new-ID B3 metrics")
    check(summary["scoring"]["eligible_anchor_observations"]==len(anchors),"Fixed anchor denominator")
    check(summary["scoring"]["selected_retained_observations"]==total_rows,"Retained denominator")
    close(summary["scoring"]["selected_sources_without_eligible_anchors"],sorted(set(spec["source_groups"])-{r["source_group"] for r in anchors.values()}),"Zero-anchor source reporting")
    for key in ("unknown_identity_is_negative","old_predicted_track_pairs_used","category_correctness_decoded"):
        check(summary["scoring"][key] is False,"Scoring boundary changed")
    baseline_protected=load(RUN/"protected_before.json")
    for path,info in baseline_protected.items():
        check(Path(path).exists()==info["exists"],"Protected file existence changed: "+path)
        if info["exists"]:check(sha(path)==info["sha256"],"Protected baseline file changed: "+path)
    for path,digest in preflight["protected_source_sha256"].items():check(sha(path)==digest,"Frozen scientific source changed")
    for path,digest in watched.items():check(sha(path)==digest,"Input/output/auditor changed during audit: "+path)
    report=dict(status="PASS_INDEPENDENT_CONTEXT_FEATURE_MEMORY_OUTPUT_AUDIT",summary_sha256=sha(out/"summary.json"),
                runner_sha256=PINS[ROOT/"tmp_analysis/replay_context_feature_memory.py"],module_sha256=PINS[ROOT/"tmp_analysis/context_feature_memory.py"],
                protocol_sha256=PINS[PROTOCOL],prediction_hashes=seal["output_files_sha256"],
                selected_sources=len(spec["source_groups"]),selected_videos=len(grouped),selected_frames=len(selected),retained_observations=total_rows,
                independently_recomputed_eligible_anchor_observations=len(anchors),metrics=measured,
                parity_with_original_GPU_cache=parity,first_row_id_divergence=first_differences,
                intervention_prediction_difference_vs_quantized_released=intervention_difference,
                float16_feature_headers_verified=len(headers),tune_rows_split_only_skipped=tune_skipped,
                forbidden_correctness_values_decoded=False,old_predicted_track_pairs_used=False,unknown_is_negative=False,
                protected_baseline_files_verified=True,frozen_scientific_sources_verified=True,all_recorded_inputs_and_outputs_unchanged=True,
                seal_before_join_basis="Audited bound sequential producer plus all-five prediction seal hashes; no independent wall-clock instrumentation of identity reads",
                real_model_or_fusion_inference_rerun=False,GT_rematching=False,training=False,GPU=False,
                scope="Conditional eligible fit146 manual-anchor B3 on quantized fixed retained detections; no official metric, generalization or efficacy claim",
                elapsed_seconds=time.monotonic()-started,auditor_sha256=sha(__file__))
    with destination.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(status=report["status"],frames=len(selected),anchors=len(anchors),metrics={k:{m:v[m] for m in ("precision","recall","f1")} for k,v in measured.items()},primary_prediction_differences={k:v for k,v in intervention_difference["raw_confidence"].items() if k!="per_video"},elapsed_seconds=report["elapsed_seconds"])))


def self_test():
    rows=[dict(source_group="s",video_id=1,gt_track_id=y,observation_id=i,new_id=z) for i,(y,z) in enumerate(zip((1,1,2),(7,7,7)))]
    result=metrics(rows);check(abs(result["precision"]-.5)<1e-15 and result["recall"]==1.,"Unequal GT weights")
    for row in rows:row["new_id"]=-1
    result=metrics(rows);check(result["precision"]==1. and abs(result["recall"]-.75)<1e-15,"Singleton denominator")
    poison='{"split":"tune","gt_track_id":{"never":"decode"},"error":{"never":"decode"}}'
    check(project(poison,{"split"})=={"split":"tune"},"Split-only projection")
    check(partition([7,7,-1,-1,9])==[0,0,1,2,3],"Canonical singleton partition")
    print(json.dumps(dict(status="PASS_INDEPENDENT_CONTEXT_MEMORY_OUTPUT_AUDITOR_SYNTHETIC",checks=4,real_cache_or_supervision_opened=False)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:audit(args.out or ROOT/"results/context_feature_memory_smoke_20261024_v1")
