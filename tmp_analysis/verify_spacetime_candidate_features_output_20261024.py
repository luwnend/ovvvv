"""Fresh CPU output auditor; never imports producer, geometry or GT decoder."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_spacetime_candidate_implementation_20261024 import independent_ranking
from verify_fit_identity_edge_support_output import project

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
RUN=STAGE/"full_collect_v1"
SUPPORT=STAGE/"support_v1"
PROTOCOL=ROOT/"tmp_analysis/spacetime_candidate_features_protocol_20261024.json"
SELECTION=ROOT/"tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
PREFLIGHT=ROOT/"tmp_analysis/spacetime_candidate_features_independent_preflight_20261024.json"
SUBJECT=ROOT/"tmp_analysis/diagnose_spacetime_candidate_features_20261024.py"
FEATURES=("raw_appearance_cosine","absolute_xyxy_cosine","box_iou","negative_normalized_displacement_norm",
          "negative_normalized_CV_innovation_norm","semantic_embedding_cosine")
PINS={PROTOCOL:"d38beac2839a6473ff6a973e41a7c194febb818e40e433df63793e3349f086a1",
      SELECTION:"740d6c782fdfe8240ddc8c4b924e566cebd2465a6a05233f02197b1e904aaf1b",
      SUBJECT:"c6fef39442dd476f918c326210ace407cdcb4e8b4700544f34454e29ba7665aa",
      PREFLIGHT:"952a45839bbefbf55ae4372b38cd90d259de3d1421e40c93d08ed405480b8214",
      ROOT/"tmp_analysis/audit_spacetime_candidate_implementation_20261024.py":"b5a094ff01ee29b44022e381890682c83b94de35d167e47970858e35e45d6b73",
      ROOT/"tmp_analysis/verify_fit_identity_edge_support_output.py":"862d31f63331a288804d808fb303e6704d6f4fdfa80f97daf13971f44c0f795d",
      ROOT/"tmp_analysis/relative_track_geometry.py":"c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a",
      SUPPORT/"support_receipt.json":"e206051d0809b8e20b24529254eacece96f997c72b7155e8264197e7c07aae1e",
      SUPPORT/"gt_free_index_manifest.json":"7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667",
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
        for key,value in expected.items():close(actual[key],value,where+"/"+str(key))
    elif isinstance(expected,list):
        check(len(actual)==len(expected),"List length differs: "+where)
        for index,value in enumerate(expected):close(actual[index],value,where+"/"+str(index))
    elif isinstance(expected,float):check(actual is not None and abs(actual-expected)<=1e-12,"Metric differs: "+where)
    else:check(actual==expected,"Value differs: "+where)


def rank_auc(truth,score):
    """Average-rank U statistic, independently derived from tie-group ranks."""
    positives=int((truth==1).sum());negatives=int((truth==0).sum())
    if not positives or not negatives:return None
    check(np.isfinite(score).all(),"Nonfinite AUC score")
    _,inverse,counts=np.unique(score,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts);means=(ends-counts+1+ends)/2.
    rank_sum=float(means[inverse[truth==1]].sum())
    return (rank_sum-positives*(positives+1)/2)/(positives*negatives)


def statistics(arrays,labels,mask,common_cv):
    truth=labels["identity_label"];known=mask&(truth>=0);selected=np.flatnonzero(known)
    identities=set()
    for side in ("query","reference"):
        identities.update(zip(arrays["video_id"][selected].tolist(),labels[side+"_gt_identity"][selected].tolist()))
    result=dict(candidates=int(mask.sum()),known_candidates=len(selected),positive_pairs=int((truth[selected]==1).sum()),
                negative_pairs=int((truth[selected]==0).sum()),unknown_candidates=int((mask&(truth<0)).sum()),
                sources=len(set(arrays["source_index"][selected].tolist())),videos=len(set(arrays["video_id"][selected].tolist())),
                official_GT_trajectories=len(identities),query_unique_observations=len(set(zip(arrays["query_image_id"][selected].tolist(),arrays["query_row"][selected].tolist()))),
                reference_unique_observations=len(set(zip(arrays["reference_image_id"][selected].tolist(),arrays["reference_row"][selected].tolist()))),
                CV_known_candidates=int((known&arrays["cv_valid"]).sum()),auc={})
    columns=range(6) if common_cv else (0,1,2,3,5)
    for column in columns:result["auc"][FEATURES[column]]=rank_auc(truth[selected],arrays["scores"][selected,column])
    if common_cv:
        rank=independent_ranking(arrays,truth,selected,FEATURES)
        rank.update(pair_order_matrix_axis=["loss","tie","win"],top1_tie_rule="highest score then lowest immutable reference dump_row",
                    primary_top1_subset="positive_and_negative_available_queries",
                    interpretation="Conditional ranking within CV-valid known-identity reference subset only; complete sealed candidate set still includes unknowns. No deployment top1 or assignment benefit is established.")
        result["paired_query_ranking"]=rank
    return result


def recompute_statistics(arrays,labels,sources):
    result={}
    for scope,hard in (("all_candidates",np.ones(len(arrays["scores"]),bool)),("same_predicted_category",arrays["same_predicted_category"])):
        result[scope]={}
        for population,common_cv in (("all_known_population",False),("common_CV_covered_population",True)):
            mask=hard&(arrays["cv_valid"] if common_cv else True)
            overall=statistics(arrays,labels,mask,common_cv)
            per_source={source:statistics(arrays,labels,mask&(arrays["source_index"]==i),common_cv) for i,source in enumerate(sources)}
            macro={}
            for feature in overall["auc"]:
                supported=[value["auc"][feature] for value in per_source.values() if value["auc"][feature] is not None]
                macro[feature]=dict(mean=sum(supported)/len(supported) if supported else None,supported_sources=len(supported))
            result[scope][population]=dict(micro_union=overall,source_macro_AUROC=macro,per_source=per_source)
    return result


def scores_from_raw(query,reference):
    """Direct box/vector algebra, no geometry or candidate producer imports."""
    def cosine(a,b):
        return (a/np.maximum(np.sqrt((a*a).sum(1))[:,None],1e-12))@(b/np.maximum(np.sqrt((b*b).sum(1))[:,None],1e-12)).T
    q,m=query["boxes"],reference["boxes"];nq,nm=len(q),len(m)
    qc=(q[:,:2]+q[:,2:])/2;mc=(m[:,:2]+m[:,2:])/2
    qs=q[:,2:]-q[:,:2];ms=m[:,2:]-m[:,:2]
    scales=np.sqrt(qs[:,None,:]*ms[None,:,:])
    intersections=np.maximum(np.minimum(q[:,None,2:],m[None,:,2:])-np.maximum(q[:,None,:2],m[None,:,:2]),0).prod(2)
    iou=intersections/(qs.prod(1)[:,None]+ms.prod(1)[None,:]-intersections)
    displacement=-np.sqrt((((qc[:,None,:]-mc[None,:,:])/scales)**2).sum(2))
    innovation=np.full((nq,nm),np.nan)
    for j in range(nm):
        t=reference["history_times"][j]
        if not np.isfinite(t):continue
        old=reference["history_boxes"][j];pc=(old[:2]+old[2:])/2
        speed=(mc[j]-pc)/(reference["time"]-t)
        predicted=mc[j]+speed*(query["time"]-reference["time"])
        innovation[:,j]=-np.sqrt((((qc-predicted)/scales[:,j,:])**2).sum(1))
    return np.stack((cosine(query["raw"],reference["raw"]),cosine(q,m),iou,displacement,innovation,
                     cosine(query["semantic"],reference["semantic"])),2).reshape(-1,6)


def frame_from_cache(record,history):
    with np.load(record["npz"],allow_pickle=False) as saved:
        for key,field in (("video_id","video_id"),("image_id","image_id"),("frame_id","frame_id"),("frame_index","frame_index")):
            check(int(saved[key])==record[field],"Cache provenance differs")
        raw=saved["track_feats"];semantic=saved["cls_feats"]
        check(raw.dtype==semantic.dtype==np.dtype("float16"),"Cache feature precision changed")
        boxes=saved["det_bboxes"].astype(float)[:,:4];ids=saved["track_ids"].astype(np.int64)
        frame=dict(boxes=boxes,raw=raw.astype(float),semantic=semantic.astype(float),ids=ids,
                   predicted=saved["det_labels"].astype(np.int32)+1,time=record["frame_index"],record=record)
    n=len(boxes);check(n==record["observations"],"Retained cache rows dropped")
    check(frame["raw"].shape==(n,256) and frame["semantic"].shape==(n,512),"Cache feature width changed")
    frame["history_times"]=np.full(n,np.nan);frame["history_boxes"]=np.full((n,4),np.nan)
    counts=Counter(ids.tolist())
    for j,identity in enumerate(ids):
        if identity<0 or counts[identity]!=1:continue
        old=history.get(int(identity))
        if old is not None:
            check(old[1]<frame["time"],"Nonpast history")
            frame["history_boxes"][j],frame["history_times"][j]=old
    for j,identity in enumerate(ids):
        if identity<0:continue
        history[int(identity)]=(boxes[j].copy(),frame["time"]) if counts[identity]==1 else None
    return frame


def reconstruct_candidates(arrays,frames,spec):
    expected_receipts=[];offset=0;retained=0;maximum_error=np.zeros(6);groups=defaultdict(list)
    for record in frames:groups[record["video_id"]].append(record)
    for video,records in sorted(groups.items()):
        previous=None;history={}
        check([r["frame_id"] for r in records]==list(range(len(records))),"Missing selected sampled frame")
        for record in records:
            query=frame_from_cache(record,history);nq=len(query["boxes"]);nm=0 if previous is None else len(previous["boxes"])
            n=nq*nm if previous is not None else 0;retained+=nq
            expected_receipts.append(dict(video_id=video,image_id=record["image_id"],sampled_step=record["frame_id"],original_frame_index=record["frame_index"],
                query_rows=nq,reference_rows=nm,candidate_rows=n,adjacent_original_frame_gap=None if previous is None else query["time"]-previous["time"],
                reference_history_original_frame_gaps=[] if previous is None else [int(previous["time"]-t) for t in previous["history_times"] if np.isfinite(t)],
                retained_released_unassigned_rows=int((query["ids"]<0).sum())))
            if previous is not None:
                rr=np.tile(np.arange(nm),nq);qr=np.repeat(np.arange(nq),nm);block=slice(offset,offset+n)
                metadata=dict(cv_valid=np.tile(np.isfinite(previous["history_times"]),nq),same_predicted_category=query["predicted"][qr]==previous["predicted"][rr],
                    source_index=np.full(n,spec["source_groups"].index(spec["video_source"][str(video)])),video_id=np.full(n,video),
                    query_image_id=np.full(n,record["image_id"]),reference_image_id=np.full(n,previous["record"]["image_id"]),query_row=qr,reference_row=rr,
                    query_original_frame=np.full(n,query["time"]),reference_original_frame=np.full(n,previous["time"]),reference_released_id=previous["ids"][rr],
                    history_original_frame=previous["history_times"][rr])
                check(set(arrays)==set(metadata)|{"scores"},"Candidate schema changed")
                for key,value in metadata.items():check(np.array_equal(arrays[key][block],value,equal_nan=True),"Candidate mapping/history differs: "+key)
                expected=scores_from_raw(query,previous)
                np.testing.assert_allclose(arrays["scores"][block],expected,rtol=1e-12,atol=1e-12,equal_nan=True)
                if n:
                    for column in range(6):
                        finite=np.isfinite(expected[:,column])
                        if finite.any():maximum_error[column]=max(maximum_error[column],float(np.max(np.abs(arrays["scores"][block,column][finite]-expected[:,column][finite]))))
                offset+=n
            previous=query
    check(offset==len(arrays["scores"]),"Extra/dropped full QxM candidates")
    return expected_receipts,retained,maximum_error.tolist()


def independent_identities(arrays,spec):
    # This scanner skips forbidden JSON values without decoding them. Split is
    # decoded first; tune identity payloads are never parsed.
    manual=SUPPORT/"supervised_observations.jsonl"
    receipt=project((SUPPORT/"support_receipt.json").read_text(),{"output_files"})
    check(sha(manual)==receipt["output_files"][manual.name],"Manual identity bytes changed")
    roles_path=STAGE/"pair_context_v1/source_roles.json";role_audit=load(STAGE/"pair_context_v1/independent_pair_context_audit.json")
    check(sha(roles_path)==role_audit["output_files_sha256"][roles_path.name],"Frozen source roles changed")
    roles=load(roles_path)["groups"];identities={};tune=0
    fields={"split","observation_id","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id","gt_track_id","manual_anchor_id","eligible_for_parameter_fit"}
    for line in manual.open():
        split=project(line,{"split"})["split"];check(split in ("fit","tune"),"Unexpected manual split")
        if split=="tune":tune+=1;continue
        row=project(line,fields)
        check(set(row)==fields and row["eligible_for_parameter_fit"] is True and roles[row["source_group"]]["split"]=="fit","Ineligible identity")
        check(row["observation_id"] not in identities,"Duplicate identity observation")
        identities[row["observation_id"]]=row
    check(len(identities)==21661 and tune==2226,"Frozen FIT/tune scope changed")
    manifest=load(SUPPORT/"gt_free_index_manifest.json");index=SUPPORT/"gt_free_observations.jsonl"
    check(sha(index)==manifest["files"][index.name],"GT-free observation index bytes changed")
    anchors={}
    for line in index.open():
        row=json.loads(line)
        if row["video_id"] not in spec["video_ids"]:continue
        identity=identities.get(row["observation_id"])
        if identity is None:continue
        for key in ("split","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id"):
            check(row[key]==identity[key],"FIT identity/index provenance differs")
        key=(row["image_id"],row["dump_row"]);check(key not in anchors,"Duplicate manual anchor row")
        anchors[key]=(row["video_id"],identity["gt_track_id"])
    labels={}
    for side in ("query","reference"):
        values=[]
        for image,row,video in zip(arrays[side+"_image_id"],arrays[side+"_row"],arrays["video_id"]):
            anchor=anchors.get((int(image),int(row)))
            check(anchor is None or anchor[0]==video,"Identity escaped video scope")
            values.append(-1 if anchor is None else anchor[1])
        labels[side+"_gt_identity"]=np.asarray(values,np.int64)
    q,r=labels["query_gt_identity"],labels["reference_gt_identity"]
    labels["identity_label"]=np.asarray([-1 if a<0 or b<0 else int(a==b) for a,b in zip(q,r)],np.int8)
    return labels,len(anchors),tune


def gaps(values):
    return dict(count=len(values),minimum=min(values) if values else None,median=float(np.median(values)) if values else None,
                maximum=max(values) if values else None,counter={str(k):v for k,v in sorted(Counter(values).items())})


def audit(out):
    started=time.monotonic();destination=out/"independent_output_audit.json"
    check(not destination.exists(),"Preserve prior independent audit")
    for path,digest in PINS.items():check(sha(path)==digest,"Pinned implementation/input changed: "+str(path))
    producer=load(out/"producer_execution_receipt.json")
    check(producer["actual_tool_execution"]["exit_code"]==0,"No root-observed actual producer exit 0")
    for path,info in producer["evidence"].items():
        check(sha(ROOT/path)==info["sha256"] and (ROOT/path).stat().st_size==info["bytes"],"Producer execution evidence changed: "+path)
    spec=load(SELECTION);protocol=load(PROTOCOL);summary=load(out/"summary.json");preflight=load(PREFLIGHT);seal=load(out/"GT_free_candidates_sealed.json")
    check(preflight["status"]=="PASS_INDEPENDENT_SPACETIME_CANDIDATE_FEATURES_IMPLEMENTATION_PREFLIGHT","No preflight")
    check(summary["status"]=="COMPLETE_FIXED_FIT_SPACETIME_CANDIDATE_DIAGNOSTIC_PENDING_INDEPENDENT_AUDIT","Unfinished producer")
    check(summary["script_sha256"]==seal["script_sha256"]==PINS[SUBJECT] and summary["protocol_sha256"]==seal["protocol_sha256"]==PINS[PROTOCOL],"Wrong frozen producer/protocol")
    check(seal["status"]=="SEALED_ALL_GT_FREE_CANDIDATES_BEFORE_SUPERVISION","Missing GT-free candidate seal")
    check(seal["GT_or_supervised_payload_opened"] is False and seal["old_ID_used_as_truth_or_candidate_filter"] is False,"Seal scope changed")
    for filename,key in (("candidate_features.npz","candidate_features_sha256"),("frame_receipts.json","frame_receipts_sha256")):
        check(sha(out/filename)==seal[key],"GT-free sealed artifact changed")
    for filename,digest in summary["output_files_sha256"].items():check(sha(out/filename)==digest,"Producer output changed")
    for path,digest in summary["input_sha256"].items():check(sha(path)==digest,"Producer input changed: "+path)
    for path,digest in seal["input_sha256"].items():check(summary["input_sha256"][path]==digest,"Preseal input omitted")
    check(summary["candidates_sealed_before_fit_identity_join"] is True and summary["all_input_hashes_verified_before_after"] is True,"Missing execution boundary checks")
    for flag in ("unknown_is_negative","GT_category_or_error_decoded","tune_test_class_holdout_correctness_opened","training_started","GPU_used","network_or_readout_changed"):
        check(summary[flag] is False,"Scope flag changed: "+flag)
    close(summary["feature_names"],list(FEATURES));close(seal["feature_names"],list(FEATURES));close(summary["interpretation"],protocol["interpretation"])
    watched={**summary["input_sha256"],**{str(path):sha(path) for path in (*PINS,Path(__file__).resolve(),out/"summary.json",out/"producer_execution_receipt.json",*(out/name for name in summary["output_files_sha256"]))}}
    frames=[json.loads(line) for line in (RUN/"frame_execution.jsonl").open()]
    frames=sorted((r for r in frames if r["video_id"] in spec["video_ids"]),key=lambda r:(r["video_id"],r["frame_id"]))
    check(len(frames)==697 and len({r["video_id"] for r in frames})==20 and len(spec["source_groups"])==12,"Fixed population changed")
    for record in frames:check(sha(record["npz"])==record["npz_sha256"]==summary["input_sha256"][record["npz"]],"Selected cache bytes changed")
    with np.load(out/"candidate_features.npz",allow_pickle=False) as saved:arrays={key:saved[key] for key in saved.files}
    receipts,retained,max_errors=reconstruct_candidates(arrays,frames,spec)
    close(load(out/"frame_receipts.json"),receipts,"Complete independent frame receipts")
    close(summary["sampling_gap_coverage"],dict(adjacent_original_frame_gap_per_frame_pair=gaps([r["adjacent_original_frame_gap"] for r in receipts if r["adjacent_original_frame_gap"] is not None]),
          history_original_frame_gap_per_reference_observation=gaps([g for r in receipts for g in r["reference_history_original_frame_gaps"]])),"Independent unweighted gap statistics")
    labels,anchors,tune=independent_identities(arrays,spec)
    with np.load(out/"fit_identity_labels.npz",allow_pickle=False) as saved:
        check(set(saved.files)==set(labels),"Label schema changed")
        for key,expected in labels.items():check(np.array_equal(saved[key],expected),"Independent FIT/unknown labels differ: "+key)
    check(anchors==924==summary["eligible_fit_manual_anchor_observations"],"Fixed manual anchor coverage changed")
    check(retained==28082==summary["retained_observations"],"Retained population changed")
    check(summary["selected_frames"]==seal["selected_frames"]==len(frames) and summary["selected_sources"]==12 and summary["selected_videos"]==20,"Summary population mismatch")
    check(summary["all_candidates"]==seal["candidates"]==len(arrays["scores"]),"Candidate denominator changed")
    results=recompute_statistics(arrays,labels,spec["source_groups"])
    close(summary["results"],results,"All independent AUROC/coverage/ranking/3x3 tables")
    protected=load(RUN/"protected_before.json")
    for path,info in protected.items():
        check(Path(path).exists()==info["exists"],"Protected baseline existence changed")
        if info["exists"]:check(sha(path)==info["sha256"],"Protected baseline changed: "+path)
    for path,digest in watched.items():check(sha(path)==digest,"Input/output/auditor changed during audit: "+path)
    report=dict(status="PASS_INDEPENDENT_SPACETIME_CANDIDATE_FEATURES_OUTPUT_AUDIT",auditor_sha256=sha(__file__),
        summary_sha256=sha(out/"summary.json"),producer_execution_receipt_sha256=sha(out/"producer_execution_receipt.json"),
        script_sha256=PINS[SUBJECT],protocol_sha256=PINS[PROTOCOL],selected_sources=12,selected_videos=20,selected_frames=len(frames),
        retained_observations=retained,all_candidates=len(arrays["scores"]),eligible_fit_manual_anchor_observations=anchors,
        all_six_features_and_candidate_mappings_independently_recomputed=True,feature_maximum_absolute_recomputation_error=dict(zip(FEATURES,max_errors)),
        all_identity_labels_and_unknowns_independently_recomputed=True,all_AUROC_and_query_top1_and_3x3_order_tables_independently_recomputed=True,
        sampling_gaps_independently_recomputed_per_frame_pair_or_reference_observation=True,results=results,tune_rows_split_only_skipped=tune,
        forbidden_correctness_values_decoded=False,unknown_is_negative=False,producer_or_geometry_or_GT_decoder_imported=False,
        input_output_and_protected_baseline_hashes_verified=True,training=False,GPU=False,network_or_readout_changed=False,
        seal_before_join_basis="Audited bound sequential producer and GT-free feature/frame hashes; no independent wall-clock instrumentation of supervised reads",
        scope=protocol["interpretation"],elapsed_seconds=time.monotonic()-started)
    with destination.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","all_candidates","selected_frames","eligible_fit_manual_anchor_observations","elapsed_seconds")},indent=2))


def self_test():
    random=np.random.RandomState(411)
    for n in range(1,50):
        truth=random.randint(0,2,n);score=random.randint(-2,3,n).astype(float)
        p=score[truth==1];m=score[truth==0]
        brute=None if not len(p) or not len(m) else sum(float(a>b)+.5*float(a==b) for a in p for b in m)/(len(p)*len(m))
        result=rank_auc(truth,score)
        check(result is brute is None or abs(result-brute)<1e-14,"Independent rank-AUC brute force mismatch")
    check(project('{"split":"tune","gt_track_id":{"poison":1},"error":666}',{"split"})=={"split":"tune"},"Split-only projection failed")
    print(json.dumps(dict(status="PASS_INDEPENDENT_SPACETIME_OUTPUT_AUDITOR_SYNTHETIC",bruteforce_AUC_cases=49,real_cache_or_supervision_opened=False)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:audit(args.out or ROOT/"results/spacetime_candidate_features_20261024_v1")
