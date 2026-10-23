#!/usr/bin/env python3
"""Finalize only a supervised, independently audited complete OOF execution.

Reads hashes and metric summaries only; no label/cache decoder or report edits.
The finalizer cannot attest its own future process exit. The supervisor records
that separately in execution/finalization.exit.json after this process returns.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import uuid

ROOT=Path(__file__).resolve().parents[1]
SUCCESS="PASS_IDENTITY_SPACETIME_OOF_EXECUTION_AND_INDEPENDENT_AUDIT"
AUDIT="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT"
GLOBAL="GT_free_OOF_predictions_sealed.json"
FEATURES=("raw_appearance_cosine","adapter_appearance_cosine","A0G0","A0G1","A0Gmask","A0Gperm","A1G0","A1G1","A1Gmask","A1Gperm")
SCOPE="same_predicted_category";POP="common_CV_covered_population";SUBSET="positive_and_negative_available_queries"


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.parent/("."+path.name+"."+uuid.uuid4().hex)
    try:
        with temporary.open("x") as f:
            f.write(json.dumps(value,indent=2,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
        os.link(str(temporary),str(path))
        fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temporary.exists():temporary.unlink()


def check_pins(pins):
    for path,digest in pins.items():require(Path(path).is_file() and sha(path)==digest,"Pinned artifact changed: "+str(path))


def require_exit(record,where):
    require(record.get("actual_exit_code")==0 and isinstance(record.get("pid"),int) and isinstance(record.get("waited_by_parent_pid"),int) and
        record.get("command") and record.get("started_utc") and record.get("finished_utc"),"Missing actual successful parent-wait exit: "+where)


def require_audit_bindings(out,execution,audit,spec):
    """Accept only documented exact audit binding keys; no status-only PASS."""
    expected={"global_GT_free_seal_sha256":sha(out/GLOBAL),"summary_sha256":sha(out/"evaluation/summary.json"),
        "support_SQLite_sha256":sha(out/"evaluation/query_support.sqlite")}
    aliases={"global_GT_free_seal_sha256":("global_GT_free_seal_sha256","global_seal_sha256"),
        "summary_sha256":("summary_sha256","evaluation_summary_sha256"),
        "support_SQLite_sha256":("support_SQLite_sha256","query_database_sha256","evaluation_SQLite_sha256")}
    for key,names in aliases.items():
        matched=[audit[name] for name in names if name in audit]
        require(matched and all(value==expected[key] for value in matched),"Independent audit does not bind exact "+key)
    require(audit.get("protocol_sha256")==sha(read(execution/"launch_spec.json")["protocol"]),"Audit protocol binding differs")
    for key in ("trainer","evaluator","verifier"):
        names=(key+"_sha256",key+"_script_sha256","auditor_sha256") if key=="verifier" else (key+"_sha256",key+"_script_sha256")
        values=[audit[name] for name in names if name in audit]
        if values:require(all(v==spec[key+"_sha256"] for v in values),"Audit module binding differs")
    require(audit.get("coordinator_receipt_sha256")==sha(out/"coordinator_receipt.json"),"Audit does not bind actual model/evaluation exit chain")


def control_brief(summary):
    reports={}
    for seed,report in summary["results"]["per_seed"].items():
        lags={}
        for lag in ("1","4"):
            cell=report["micro_union"][lag][SCOPE][POP];stats=cell["query_statistics"][SUBSET]
            features={feature:dict(micro=stats["features"][feature]["micro_correctness"],
                source_GTtrajectory_query_macro=stats["features"][feature]["source_GTtrajectory_query_macro_correctness"],
                correct=stats["features"][feature]["correct"],errors=stats["features"][feature]["errors"],
                repair_raw_error=cell["top1"][SUBSET][feature]["repair_raw_error"],break_raw_correct=cell["top1"][SUBSET][feature]["break_raw_correct"]) for feature in FEATURES}
            zero=[source for source,row in report["per_source"].items() if row["lags"][lag][SCOPE][POP]["hard_queries"]==0]
            lags[lag]=dict(hard_queries=cell["hard_queries"],known_positive_pairs=cell["positive_pairs"],known_negative_pairs=cell["negative_pairs"],
                hard_query_sources=stats["sources"],zero_hard_query_sources=zero,features=features,comparisons=cell["comparisons"][SUBSET])
        reports[seed]=dict(lags=lags,cross_lag_shared=report["cross_lag"][SCOPE][POP])
    return dict(status="FINAL_AUDITED_IDENTITY_SPACETIME_OOF_CONTROL_BRIEF",scope=SCOPE,population=POP,query_subset=SUBSET,
        coverage_note="This brief shows the fixed primary hard-query readout. The complete evaluation retains both scopes, both populations, all-nonempty and hard subsets, all seeds and zero-support sources.",
        per_seed=reports,seed_robustness=summary["results"]["seed_robustness"],
        interpretation="Exploratory source OOF on already examined FIT; ordinary adapter/head controls, no official tracking, novelty, backbone improvement or independent generalization claim.")


def finalize(out,execution):
    out=Path(out);execution=Path(execution);require(not (out/"completion_receipt.json").exists() and not (out/"control_brief.json").exists(),"Preserve existing finalization outputs")
    receipt=read(execution/"execution_receipt.json");require(receipt.get("status")==SUCCESS,"Supervisor execution is not PASS")
    for key in ("coordinator","independent_auditor"):require_exit(receipt[key],key)
    launch_path=execution/"launch_spec.json";require(receipt["launch_sha256"]==sha(launch_path),"Launch authority changed")
    launch=read(launch_path);check_pins(launch["file_pins"]);protocol=Path(launch["protocol"]);spec=read(protocol)
    require(Path(spec["output_directory"])==out and Path(spec["execution_directory"])==execution,"Output/execution authority mismatch")
    require(spec["finalizer_path"]==str(Path(__file__).resolve()) and spec["finalizer_sha256"]==sha(__file__),"Finalizer bytes not pinned")
    audit=read(out/"independent_output_audit.json");require(audit.get("status")==AUDIT,"Independent output audit is not PASS")
    require(receipt["global_seal_sha256"]==sha(out/GLOBAL) and receipt["independent_audit_sha256"]==sha(out/"independent_output_audit.json"),"Execution/audit/global hash chain differs")
    require_audit_bindings(out,execution,audit,spec)
    coordinator_path=Path(spec["coordinator_path"]);require(sha(coordinator_path)==spec["coordinator_sha256"],"Coordinator bytes changed")
    module_spec=importlib.util.spec_from_file_location("identity_oof_finalization_global_gate",coordinator_path);coordinator=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(coordinator)
    coordinator.check_pins(coordinator.all_pins(spec));coordinator.verify_global_seal(out,spec)
    protected=read(spec["protected_authority_path"])["protected_files"]
    require(len(protected)==7 and coordinator.snapshot(protected)==protected,"Seven protected artifacts changed")
    process_chain=read(out/"coordinator_receipt.json")
    require(len(process_chain["actual_model_processes"])==15,"Missing actual model process exit chain")
    for i,record in enumerate(process_chain["actual_model_processes"]):require_exit(record,"model "+str(i))
    require_exit(process_chain["evaluation_process"],"unified_evaluation")
    require(process_chain["protocol_sha256"]==sha(protocol) and process_chain["global_seal_sha256"]==sha(out/GLOBAL),"Coordinator receipt lineage mismatch")
    summary=read(out/"evaluation/summary.json")
    require(summary["status"]=="COMPLETE_UNIFIED_ALL_SEALED_SOURCE_OOF_EVALUATION_PENDING_INDEPENDENT_AUDIT" and summary["query_database_sha256"]==sha(out/"evaluation/query_support.sqlite"),"Evaluation incomplete or query database changed")
    brief=control_brief(summary);brief["evaluation_summary_sha256"]=sha(out/"evaluation/summary.json");write(out/"control_brief.json",brief)
    bound={"execution_receipt":execution/"execution_receipt.json","launch_spec":launch_path,"coordinator_receipt":out/"coordinator_receipt.json",
        "global_GT_free_seal":out/GLOBAL,"independent_output_audit":out/"independent_output_audit.json","evaluation_summary":out/"evaluation/summary.json",
        "query_database":out/"evaluation/query_support.sqlite","control_brief":out/"control_brief.json","protocol":protocol,"protected_authority":Path(spec["protected_authority_path"])}
    completion=dict(status="COMPLETE_AUDITED_IDENTITY_SPACETIME_SOURCE_OOF_STAGE",finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files_sha256={name:dict(path=str(path),sha256=sha(path)) for name,path in bound.items()},
        actual_child_exit_chain=dict(coordinator=receipt["coordinator"],independent_auditor=receipt["independent_auditor"],model_processes=process_chain["actual_model_processes"],unified_evaluation=process_chain["evaluation_process"]),
        seven_protected_files_unchanged=protected,script_sha256=sha(__file__),
        finalizer_actual_exit_recorded_later_by_supervisor=str(execution/"finalization.exit.json"),finalizer_actual_exit_not_claimed_here=True,
        GPU_used=False,unknown_is_negative=False,training_started_by_finalizer=False)
    check_pins(launch["file_pins"]);require(coordinator.snapshot(protected)==protected,"Protected files changed during finalization")
    write(out/"completion_receipt.json",completion);return completion


def fixture(root):
    out=root/"out";execution=root/"execution";out.mkdir();execution.mkdir();(out/"evaluation").mkdir()
    checker=root/"checker.py";checker.write_text("import hashlib\nfrom pathlib import Path\ndef all_pins(spec):return spec['input_pins']\ndef check_pins(pins):\n for p,d in pins.items():\n  assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d\ndef verify_global_seal(out,spec):return {}\ndef snapshot(expected):\n return {p:dict(exists=Path(p).exists(),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None) for p in expected}\n")
    protected={}
    for i in range(7):
        path=root/f"protected{i}";path.write_text(str(i));protected[str(path)]=dict(exists=True,sha256=sha(path))
    authority=root/"protected.json";write(authority,dict(protected_files=protected))
    protocol=root/"protocol.json";spec=dict(output_directory=str(out),execution_directory=str(execution),finalizer_path=str(Path(__file__).resolve()),finalizer_sha256=sha(__file__),coordinator_path=str(checker),coordinator_sha256=sha(checker),protected_authority_path=str(authority),input_pins={str(authority):sha(authority)})
    write(protocol,spec);write(execution/"launch_spec.json",dict(protocol=str(protocol),file_pins={str(protocol):sha(protocol),str(checker):sha(checker)}));write(out/GLOBAL,dict(status="sealed_fixture"))
    db=out/"evaluation/query_support.sqlite";db.write_bytes(b"synthetic sqlite hash standin")
    feature={f:dict(correct=1,errors=0,micro_correctness=1.,source_GTtrajectory_query_macro_correctness=1.) for f in FEATURES};top={f:dict(repair_raw_error=0,break_raw_correct=0) for f in FEATURES}
    cell=dict(hard_queries=1,positive_pairs=1,negative_pairs=1,query_statistics={SUBSET:dict(sources=1,features=feature)},top1={SUBSET:top},comparisons={SUBSET:dict()})
    zero=dict(hard_queries=0);source={"S":dict(lags={lag:{SCOPE:{POP:cell}} for lag in ("1","4")}),"ZERO":dict(lags={lag:{SCOPE:{POP:zero}} for lag in ("1","4")})}
    perseed={str(s):dict(micro_union={lag:{SCOPE:{POP:cell}} for lag in ("1","4")},per_source=source,cross_lag={SCOPE:{POP:dict(shared_hard_queries=1)}}) for s in (20261024,20261025,20261026)}
    write(out/"evaluation/summary.json",dict(status="COMPLETE_UNIFIED_ALL_SEALED_SOURCE_OOF_EVALUATION_PENDING_INDEPENDENT_AUDIT",query_database_sha256=sha(db),results=dict(per_seed=perseed,seed_robustness={})))
    process=dict(actual_exit_code=0,pid=1,waited_by_parent_pid=2,command=["synthetic"],started_utc="start",finished_utc="end")
    write(out/"coordinator_receipt.json",dict(actual_model_processes=[dict(process) for _ in range(15)],evaluation_process=process,protocol_sha256=sha(protocol),global_seal_sha256=sha(out/GLOBAL)))
    audit=dict(status=AUDIT,protocol_sha256=sha(protocol),global_GT_free_seal_sha256=sha(out/GLOBAL),summary_sha256=sha(out/"evaluation/summary.json"),support_SQLite_sha256=sha(db),coordinator_receipt_sha256=sha(out/"coordinator_receipt.json"));write(out/"independent_output_audit.json",audit)
    write(execution/"execution_receipt.json",dict(status=SUCCESS,coordinator=process,independent_auditor=process,launch_sha256=sha(execution/"launch_spec.json"),global_seal_sha256=sha(out/GLOBAL),independent_audit_sha256=sha(out/"independent_output_audit.json")))
    return out,execution


def synthetic(number):
    checks=[]
    if number==1:
        with tempfile.TemporaryDirectory(prefix="oof_finalize_success_") as temp:
            out,execution=fixture(Path(temp));result=finalize(out,execution)
            require(result["status"]=="COMPLETE_AUDITED_IDENTITY_SPACETIME_SOURCE_OOF_STAGE" and result["finalizer_actual_exit_not_claimed_here"],"Completion claims future self exit")
            brief=read(out/"control_brief.json");require(brief["per_seed"]["20261024"]["lags"]["1"]["zero_hard_query_sources"]==["ZERO"] and len(brief["per_seed"]["20261024"]["lags"]["4"]["features"])==10,"Primary brief dropped source/column")
        checks.append("complete_success_exit_chain_exact_hashes_seven_protections_primary_brief_tenfeatures_zero_sources")
        with tempfile.TemporaryDirectory(prefix="oof_finalize_reentry_") as temp:
            out,execution=fixture(Path(temp));finalize(out,execution)
            try:finalize(out,execution)
            except ValueError:pass
            else:raise AssertionError("Overwrote finalization output")
        checks.append("exclusive_completion_no_overwrite_and_no_claim_of_finalizer_future_actual_exit")
    elif number==2:
        for code in (137,-9,1):
            with tempfile.TemporaryDirectory(prefix="oof_finalize_badexit_") as temp:
                out,execution=fixture(Path(temp));path=execution/"execution_receipt.json";record=read(path);record["coordinator"]["actual_exit_code"]=code;path.write_text(json.dumps(record))
                try:finalize(out,execution)
                except ValueError:pass
                else:raise AssertionError("Accepted failed actual child exit")
        checks.append("exit137_signal_and_nonzero_actual_child_rejected_despite_PASS_status")
        with tempfile.TemporaryDirectory(prefix="oof_finalize_badaudit_") as temp:
            out,execution=fixture(Path(temp));path=out/"independent_output_audit.json";record=read(path);record["status"]="PENDING";path.write_text(json.dumps(record))
            try:finalize(out,execution)
            except ValueError:pass
            else:raise AssertionError("Accepted pending independent audit")
        checks.append("independent_output_audit_requires_PASS_before_artifact_publication")
    elif number==3:
        with tempfile.TemporaryDirectory(prefix="oof_finalize_changeddb_") as temp:
            out,execution=fixture(Path(temp));(out/"evaluation/query_support.sqlite").write_bytes(b"changed")
            try:finalize(out,execution)
            except ValueError:pass
            else:raise AssertionError("Accepted changed audited query database")
        checks.append("independent_audit_exact_summary_global_query_SQLite_hash_chain_rejects_mutation")
        with tempfile.TemporaryDirectory(prefix="oof_finalize_protected_") as temp:
            out,execution=fixture(Path(temp));(Path(temp)/"protected3").write_text("changed")
            try:finalize(out,execution)
            except ValueError:pass
            else:raise AssertionError("Accepted changed protected input")
        checks.append("all_seven_protected_file_existence_and_SHA_verified_before_completion")
    else:raise ValueError("Three rounds only")
    return dict(status="PASS_IDENTITY_SPACETIME_OOF_FINALIZER_SYNTHETIC",round=number,checks=checks,script_sha256=sha(__file__),real_payload_opened=False,training_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path);parser.add_argument("--execution",type=Path);parser.add_argument("--synthetic-round",type=int,choices=(1,2,3));parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    if args.synthetic_round:
        result=synthetic(args.synthetic_round)
        if args.synthetic_out:write(args.synthetic_out,result)
    else:require(args.out and args.execution,"Explicit out/execution or synthetic round required");result=finalize(args.out,args.execution)
    print(json.dumps(dict(status=result["status"],script_sha256=sha(__file__))))


if __name__=="__main__":main()
