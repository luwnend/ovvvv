#!/usr/bin/env python3
"""Independent fit-only endpoint support audit; no producer imports or model IO."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"results/base_error_full_collection_20261023"
SUPPORT=STAGE/"support_v1"
CONTEXT=STAGE/"pair_context_v1"
TAG="base_fit_identity_edge_support_20261023"
PRODUCER=ROOT/"tmp_analysis/build_fit_identity_edge_support.py"
PRODUCER_SHA="e68f35ca48ba6f32d519efbdb33a711fc201cbf1a2f65c50f1c828a767b1d2b5"
PREFLIGHT=ROOT/("tmp_analysis/"+TAG+"_independent_preflight.json")
PINS={SUPPORT/"independent_support_audit.json":"92174aaa954f6e66cd0e7f520f2f883f7b9e0017c84fbf9bf4509e734ba714e8",
      SUPPORT/"support_receipt.json":"e206051d0809b8e20b24529254eacece96f997c72b7155e8264197e7c07aae1e",
      SUPPORT/"gt_free_index_manifest.json":"7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667",
      CONTEXT/"independent_pair_context_audit.json":"df7c386a8f01d3dbfce928e58758f9c455f685e8485dbfa8b19a0cd4ac936e81",
      CONTEXT/"pair_context_receipt.json":"ae9b5fd9a122533733e14208e81b0c875ca7bbb41ed1397ff4b804b25901dd2c",
      PRODUCER:PRODUCER_SHA}
ROLES=("mean_role","pair_role")
LABELS=("same_gt","different_gt","one_endpoint_unknown","both_endpoints_unknown")
AGREEMENTS=("same_predicted_category","different_predicted_category","unavailable")
IDENTITY_FIELDS={"observation_id","split","source_group","video_id","image_id","sampled_step",
                 "original_frame_index","local_track_id","gt_track_id","pred_category_id",
                 "manual_anchor_id","eligible_for_parameter_fit"}
COPIED_FIELDS=("edge_id","video_id","local_track_id","source_group","role","observation_i","observation_j",
               "sample_step_lag","original_frame_lag","lag_bin","n_track_observations","n_reference_observations",
               "reference_observation_ids_sha256")
OUTPUT_FIELDS=set(COPIED_FIELDS)|{"split","identity_status","predicted_category_agreement","gt_track_id_i",
                                    "gt_track_id_j","manual_anchor_id_i","manual_anchor_id_j"}


def check(value,message):
    if not value: raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def exact(a,b,message):
    check(json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False)==
          json.dumps(b,sort_keys=True,separators=(",",":"),allow_nan=False),message)


def end_of_value(text,start):
    """Independent boundary scan; skipped values are never JSON decoded."""
    quote=False;escaped=False;stack=[]
    for pos in range(start,len(text)):
        c=text[pos]
        if quote:
            if escaped: escaped=False
            elif c=="\\": escaped=True
            elif c=='"': quote=False
            continue
        if c=='"': quote=True
        elif c in "[{": stack.append(c)
        elif c in "]}":
            if not stack: return pos
            expected="[" if c=="]" else "{"
            check(stack.pop()==expected,"Mismatched skipped JSON container")
        elif c=="," and not stack: return pos
    raise AssertionError("Unterminated projected JSON value")


def project(text,wanted,with_keys=False):
    check(not wanted&{"error","soft_risk","gt_category_id","posterior_argmax_category_id"},"Forbidden outcome projection")
    decoder=json.JSONDecoder();pos=0
    while pos<len(text) and text[pos].isspace():pos+=1
    check(text[pos]=="{","Expected projected object");pos+=1;keys=set();result={}
    while True:
        while pos<len(text) and text[pos].isspace():pos+=1
        if text[pos]=="}":pos+=1;break
        key,used=decoder.raw_decode(text,pos);check(type(key) is str and key not in keys,"Duplicate/nonstring projected key")
        keys.add(key);pos=used
        while text[pos].isspace():pos+=1
        check(text[pos]==":","Missing projected colon");pos+=1
        while text[pos].isspace():pos+=1
        end=end_of_value(text,pos)
        if key in wanted:
            value,used=decoder.raw_decode(text,pos)
            check(not text[used:end].strip(),"Projected value has trailing data")
            result[key]=value
        pos=end
        if text[pos]=="}":pos+=1;break
        check(text[pos]==",","Missing projected delimiter");pos+=1
    check(not text[pos:].strip(),"Projected object has trailing data")
    return (result,keys) if with_keys else result


def projected_file(path,wanted):
    return project(Path(path).read_text(),set(wanted))


def fit_identities(path,roles):
    result={};tune_rows=0
    with path.open() as stream:
        for line in stream:
            split=project(line,{"split"}).get("split")
            check(split in ("fit","tune"),"Test/unknown source row")
            if split=="tune":tune_rows+=1;continue
            row=project(line,IDENTITY_FIELDS)
            check(set(row)==IDENTITY_FIELDS and row["eligible_for_parameter_fit"] is True,"Invalid fit manual identity")
            oid=row["observation_id"];check(type(oid) is int and oid not in result,"Duplicate/invalid fit observation")
            check(row["source_group"] in roles and roles[row["source_group"]]["split"]=="fit","Fit identity source role")
            result[oid]=row
    return result,tune_rows


def expected_row(edge,identities,roles):
    check(edge["split"]=="fit" and roles[edge["source_group"]]["role"]==edge["role"] in ROLES,"Fit edge role mismatch")
    a=identities.get(edge["observation_i"]);b=identities.get(edge["observation_j"])
    for endpoint,row in (("i",a),("j",b)):
        if row is not None:
            exact((row["split"],row["source_group"],row["video_id"],row["local_track_id"],row["image_id"],row["sampled_step"]),
                  ("fit",edge["source_group"],edge["video_id"],edge["local_track_id"],edge["image_"+endpoint],edge["sampled_step_"+endpoint]),"Endpoint identity provenance mismatch")
    if a is None or b is None:
        label="both_endpoints_unknown" if a is None and b is None else "one_endpoint_unknown";agree="unavailable"
    else:
        label="same_gt" if a["gt_track_id"]==b["gt_track_id"] else "different_gt"
        agree="same_predicted_category" if a["pred_category_id"]==b["pred_category_id"] else "different_predicted_category"
        check(b["original_frame_index"]-a["original_frame_index"]==edge["original_frame_lag"],"Original endpoint timeline mismatch")
    result={k:edge[k] for k in COPIED_FIELDS}
    result.update(split="fit",identity_status=label,predicted_category_agreement=agree,
                  gt_track_id_i=None if a is None else a["gt_track_id"],gt_track_id_j=None if b is None else b["gt_track_id"],
                  manual_anchor_id_i=None if a is None else a["manual_anchor_id"],manual_anchor_id_j=None if b is None else b["manual_anchor_id"])
    return result


def concentration(values):
    total=sum(values);ordered=sorted(values,reverse=True);squares=sum(v*v for v in ordered)
    return dict(events=total,nonempty_units=len(ordered),max_unit_events=ordered[0] if ordered else 0,
                max_unit_share=ordered[0]/total if total else None,top5_share=sum(ordered[:5])/total if total else None,
                top10_share=sum(ordered[:10])/total if total else None,hhi=squares/(total*total) if total else None,
                effective_units=total*total/squares if total else None,events_after_deleting_largest_unit=total-(ordered[0] if ordered else 0))


def statistics(rows):
    # Rows are compact tuples (status, source, video, local, GT_i, GT_j).
    events=Counter(r[0] for r in rows);sources={};tracks={};details={}
    for label in LABELS:
        subset=[r for r in rows if r[0]==label]
        src=Counter(r[1] for r in subset);trk=Counter((r[2],r[3]) for r in subset)
        gt=Counter((r[2],*sorted((r[4],r[5]))) for r in subset) if label in ("same_gt","different_gt") else Counter()
        trajectories={(r[2],t) for r in subset for t in (r[4],r[5])} if label in ("same_gt","different_gt") else set()
        sources[label]=set(src);tracks[label]=set(trk)
        details[label]=dict(source=concentration(list(src.values())),predicted_track=concentration(list(trk.values())),
                            official_gt_identity_pair=concentration(list(gt.values())),official_gt_trajectories=len(trajectories),
                            sources_with_ge_5_edges=sum(n>=5 for n in src.values()),sources_with_ge_10_edges=sum(n>=10 for n in src.values()))
    return dict(retained_fit_edges=len(rows),identity_counts={label:events[label] for label in LABELS},
                both_identity_known=events["same_gt"]+events["different_gt"],sources_with_same_gt=len(sources["same_gt"]),
                sources_with_different_gt=len(sources["different_gt"]),sources_with_both_identity_statuses=len(sources["same_gt"]&sources["different_gt"]),
                tracks_with_both_identity_statuses=len(tracks["same_gt"]&tracks["different_gt"]),per_identity_concentration=details)


def numerical_equal(actual,expected,where):
    if isinstance(expected,dict):
        check(set(actual)==set(expected),"Statistic keys differ: "+where)
        for k,v in expected.items():numerical_equal(actual[k],v,where+"/"+k)
    elif isinstance(expected,list):
        check(len(actual)==len(expected),"Statistic list scope differs: "+where)
        for i,v in enumerate(expected):numerical_equal(actual[i],v,where+"/"+str(i))
    elif type(expected) is float:
        check(type(actual) in (int,float) and abs(actual-expected)<=1e-12*max(1.,abs(expected)),"Numerical statistic differs: "+where)
    else: exact(actual,expected,"Statistic differs: "+where)


def verify(out):
    started=time.monotonic();destination=out/"independent_identity_support_audit.json";check(not destination.exists(),"Preserve independent audit")
    summary_path=out/"identity_support_summary.json"
    fields={"status","fit_eligible_manual_identity_observations","fit_role_source_counts","support","correctness_fields_decoded",
            "fit_identity_fields_decoded","input_sha256","output_files_sha256","all_input_hashes_verified_before_after"}
    flags={"tune_identity_joined","tune_results_reopened","test_opened","class_holdout_correctness_opened","classification_correctness_computed",
           "training_started","model_imported","GPU_used","selected_edges_replaced","all_emitted_references_changed","whole_track_purity_used",
           "unknown_is_negative","pair_training","new_hypothesis_efficacy","training_readiness_established"}
    summary=projected_file(summary_path,fields|flags)
    check(summary["status"]=="COMPLETE_FIT_IDENTITY_EDGE_STRUCTURE_DIAGNOSTIC_PENDING_INDEPENDENT_AUDIT","Unfinished identity support")
    for flag in flags:check(summary[flag] is False,"Scope boundary violated: "+flag)
    exact(summary["correctness_fields_decoded"],[],"Correctness decoder opened")
    exact(summary["fit_identity_fields_decoded"],sorted(IDENTITY_FIELDS),"Identity allowlist changed")
    check(summary["all_input_hashes_verified_before_after"] is True,"Input sealing absent")
    review=projected_file(PREFLIGHT,{"status","script_sha256"})
    exact(review,dict(status="PASS_INDEPENDENT_FIT_IDENTITY_EDGE_SUPPORT_IMPLEMENTATION_PREFLIGHT",script_sha256=PRODUCER_SHA),"Stale independent implementation gate")
    for path,digest in PINS.items():check(sha(path)==digest,"Fixed upstream source changed: "+str(path))
    expected_paths={PRODUCER,PREFLIGHT,*PINS,SUPPORT/"supervised_observations.jsonl",CONTEXT/"source_roles.json",CONTEXT/"edges.jsonl",CONTEXT/"track_contexts.jsonl"}
    input_hashes={str(path):sha(path) for path in expected_paths}
    exact(summary["input_sha256"],input_hashes,"Identity input hash set differs")
    support_receipt=projected_file(SUPPORT/"support_receipt.json",{"output_files"})
    check(input_hashes[str(SUPPORT/"supervised_observations.jsonl")]==support_receipt["output_files"]["supervised_observations.jsonl"],"Manual identity source bytes changed")
    pair_audit=projected_file(CONTEXT/"independent_pair_context_audit.json",{"output_files_sha256"})
    for name,digest in pair_audit["output_files_sha256"].items():check(input_hashes[str(CONTEXT/name)]==digest,"GT-free edge/role/reference input changed")
    row_path=out/"fit_edge_identity_rows.jsonl"
    exact(summary["output_files_sha256"],{row_path.name:sha(row_path)},"Identity sidecar bytes changed")
    watched={**input_hashes,str(row_path):sha(row_path),str(summary_path):sha(summary_path),str(Path(__file__).resolve()):sha(__file__)}
    roles=json.loads((CONTEXT/"source_roles.json").read_text())["groups"]
    identities,tune_rows=fit_identities(SUPPORT/"supervised_observations.jsonl",roles)
    check(len(identities)==summary["fit_eligible_manual_identity_observations"],"Fit eligible identity count differs")
    buckets=defaultdict(list)
    for role in ROLES:buckets[("role",role)]=[]
    for group,r in roles.items():
        if r["split"]=="fit":buckets[("source",group)]=[]
    for role in ROLES:
        for lag in range(4):
            for agree in AGREEMENTS:buckets[("marginal",role,lag,agree)]=[]
    fit_edges=0;tune_edges=0;seen=set()
    with (CONTEXT/"edges.jsonl").open() as source,row_path.open() as target:
        for line in source:
            edge=json.loads(line)  # independently sealed GT-free metadata only
            check(edge["split"] in ("fit","tune"),"Closed/test selected edge")
            if edge["split"]=="tune":tune_edges+=1;continue
            check(edge["edge_id"] not in seen,"Duplicate selected fit edge");seen.add(edge["edge_id"])
            expected=expected_row(edge,identities,roles);actual_line=next(target,None);check(actual_line is not None,"Missing selected fit edge sidecar")
            actual,keys=project(actual_line,OUTPUT_FIELDS,with_keys=True)
            check(keys==OUTPUT_FIELDS,"Unexpected identity output field; possible outcome/selection sidechannel")
            exact(actual,expected,"Endpoint/unknown/edge/reference/role sidecar mismatch")
            compact=(expected["identity_status"],expected["source_group"],expected["video_id"],expected["local_track_id"],expected["gt_track_id_i"],expected["gt_track_id_j"])
            group=expected["source_group"];role=expected["role"];lag=expected["lag_bin"];agree=expected["predicted_category_agreement"]
            for key in (("all",),("role",role),("source",group),("marginal",role,lag,agree),("sourcecell",group,lag,agree)):
                buckets[key].append(compact)
            fit_edges+=1
            if fit_edges%100000==0:print(json.dumps(dict(status="INDEPENDENT_IDENTITY_EDGE_PROGRESS",fit_edges=fit_edges,elapsed_seconds=time.monotonic()-started)),flush=True)
        check(next(target,None) is None,"Extra/duplicate/nonfit identity output row")
    derived=dict(all_fit=statistics(buckets[("all",)]),per_role={role:statistics(buckets[("role",role)]) for role in ROLES},
        per_source={g:dict(role=roles[g]["role"],**statistics(buckets[("source",g)])) for g in sorted(roles) if roles[g]["split"]=="fit"},
        role_lag_agreement=[dict(role=key[1],lag_bin=key[2],predicted_category_agreement=key[3],**statistics(value)) for key,value in sorted(buckets.items()) if key[0]=="marginal"],
        source_lag_agreement=[dict(source_group=key[1],role=roles[key[1]]["role"],lag_bin=key[2],predicted_category_agreement=key[3],**statistics(value)) for key,value in sorted(buckets.items()) if key[0]=="sourcecell"])
    numerical_equal(summary["support"],derived,"support")
    exact(summary["fit_role_source_counts"],dict(Counter(r["role"] for r in roles.values() if r["split"]=="fit")),"Source roles adjusted by support")
    for path,digest in watched.items():check(sha(path)==digest,"Input/output/verifier changed during audit: "+path)
    report=dict(status="PASS_INDEPENDENT_FIT_IDENTITY_EDGE_SUPPORT_OUTPUT_AUDIT",output_directory=str(out),
        summary_sha256=sha(summary_path),identity_support_summary_sha256=sha(summary_path),
        fit_edge_identity_rows_sha256=sha(row_path),output_files_sha256={row_path.name:sha(row_path)},producer_sha256=PRODUCER_SHA,
        independent_verifier_sha256=sha(__file__),implementation_preflight_sha256=sha(PREFLIGHT),
        fit_eligible_identity_observations_verified=len(identities),all_fit_selected_edges_verified=fit_edges,
        tune_rows_split_only_decoded=tune_rows,tune_GT_free_edges_skipped=tune_edges,
        all_fit_endpoint_manual_identity_status_and_reference_hashes_verified=True,
        all_global_role_source_fixed24cells_sourcecells_and_concentrations_recomputed=True,
        global_support=derived["all_fit"],per_role_support=derived["per_role"],
        classification_outcome_values_decoded=False,tune_identity_values_decoded=False,real_NPZ_opened=False,
        training_or_model_imported=False,GPU_used=False,elapsed_seconds=time.monotonic()-started,
        scope="All selected fit edges and all fit eligible manual identity rows; no sampling. Mixed source bytes scanned, only fit identity fields decoded.",
        limitations=["Manual-source provenance inherits the fixed independently audited support rows; no raw GT rematching occurs here.",
                     "Unknown endpoints stay separate; different-GT labels cover observed endpoints only, not references or whole-track purity.",
                     "Descriptive identity support is not a statistical/mechanism, support-sufficiency, training-readiness or efficacy gate."])
    with destination.open("x") as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"],audit=str(destination),audit_sha256=sha(destination),fit_edges=fit_edges,
                         identity_counts=derived["all_fit"]["identity_counts"],elapsed_seconds=report["elapsed_seconds"]),indent=2),flush=True)


def self_test():
    original=json.JSONDecoder.raw_decode
    def guard(self,text,*args,**kwargs):
        start=args[0] if args else kwargs.get("idx",0)
        fragment=text[start:]
        check(not fragment.startswith('"TUNE_SECRET"') and not fragment.startswith('"OUTCOME_SECRET"'),"Forbidden decoder call")
        return original(self,text,*args,**kwargs)
    with tempfile.TemporaryDirectory(prefix="independent_identity_verifier_") as folder:
        path=Path(folder)/"mixed.jsonl"
        fit=dict(observation_id=1,split="fit",source_group="A",video_id=1,image_id=2,sampled_step=1,original_frame_index=7,
                 local_track_id=1,gt_track_id=9,pred_category_id=2,manual_anchor_id=3,eligible_for_parameter_fit=True,error="OUTCOME_SECRET")
        path.write_text(json.dumps(dict(split="tune",gt_track_id="TUNE_SECRET",error="OUTCOME_SECRET"))+"\n"+json.dumps(fit)+"\n")
        json.JSONDecoder.raw_decode=guard
        try:identities,tune=fit_identities(path,{"A":dict(split="fit",role="mean_role")})
        finally:json.JSONDecoder.raw_decode=original
        check(len(identities)==1 and tune==1 and "error" not in identities[1],"Independent selective fit projection")
    rows=[("different_gt","a",1,2,3,4)]*3+[("different_gt","b",2,2,3,4)]
    stats=statistics(rows);c=stats["per_identity_concentration"]["different_gt"]["source"]
    check(c["max_unit_share"]==.75 and c["effective_units"]==1.6 and c["events_after_deleting_largest_unit"]==1,"Independent concentration fixture")
    print(json.dumps(dict(status="PASS_INDEPENDENT_FIT_IDENTITY_OUTPUT_VERIFIER_SYNTHETIC",outcome_and_tune_identity_decoder_poison_passed=True,
                         concentration_verified=True,producer_imported=False,real_data_opened=False,verifier_sha256=sha(__file__)),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--out",type=Path,default=STAGE/TAG);args=parser.parse_args()
    if args.self_test:self_test()
    else:verify(args.out.resolve())


if __name__=="__main__":main()
