#!/usr/bin/env python3
"""Derive a lightweight support brief and final receipt from sealed outputs."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/fit_lag_identity_support_20261024_v1"
EXECUTION=ROOT/"results/fit_lag_identity_support_execution_20261024_v1"
AUTHORITY=ROOT/"tmp_analysis/feature_spacetime_protected_before_20261024.json"
FEATURES=("raw_appearance_cosine","absolute_xyxy_cosine","box_iou","negative_normalized_displacement_norm",
          "negative_normalized_CV_innovation_norm","semantic_embedding_cosine")
SCOPES=("all_candidates","same_predicted_category")
POPULATIONS=("all_known_population","common_CV_covered_population")
PRIMARY=(SCOPES[1],POPULATIONS[1])
SUPPORT_FIELDS=("candidate_pairs","known_pairs","positive_pairs","negative_pairs","unknown_pairs",
    "nonempty_known_queries","positive_available_queries","negative_available_queries","hard_queries",
    "official_GT_trajectories","distinct_ordered_identity_pairs","raw_hard_errors")
CROSS_FIELDS=("shared_hard_queries","raw_error_lag1","raw_error_lag4","raw_error_both")


def check(value,message):
    if not value:raise AssertionError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def evidence(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)


def names(population):return FEATURES if population==POPULATIONS[1] else tuple(f for i,f in enumerate(FEATURES) if i!=4)


def compact_cell(cell,population):
    counts={key:cell[key] for key in SUPPORT_FIELDS if key!="raw_hard_errors"}
    counts["raw_hard_errors"]=cell["top1"]["positive_and_negative_available_queries"][FEATURES[0]]["raw_wrong_feature_wrong"]
    check(counts["known_pairs"]==counts["positive_pairs"]+counts["negative_pairs"],"Known identity pair arithmetic")
    check(counts["candidate_pairs"]==counts["known_pairs"]+counts["unknown_pairs"],"Candidate/unknown arithmetic")
    check(counts["positive_available_queries"]+counts["negative_available_queries"]-counts["hard_queries"]==counts["nonempty_known_queries"],"Known-query support arithmetic")
    correct={}
    for subset in ("all_nonempty_known_queries","positive_and_negative_available_queries"):
        correct[subset]=[]
        total=counts["nonempty_known_queries"] if subset=="all_nonempty_known_queries" else counts["hard_queries"]
        for feature in names(population):
            row=cell["top1"][subset][feature]
            check(sum(row[k] for k in ("raw_wrong_feature_wrong","raw_wrong_feature_correct","raw_correct_feature_wrong","raw_correct_feature_correct"))==total,"Top1 denominator arithmetic")
            check(row["feature_correct"]==row["raw_correct"]+row["repair_raw_error"]-row["break_raw_correct"],"Top1 transition arithmetic")
            correct[subset].append([row["feature_correct"],row["repair_raw_error"],row["break_raw_correct"]])
    return dict(support=[counts[key] for key in SUPPORT_FIELDS],correct_repair_break=correct)


def derive_brief(summary,audit,bindings):
    results=summary["results"];per_source={};sources=sorted(results["per_source"])
    cross_union={scope:{pop:{key:0 for key in CROSS_FIELDS} for pop in POPULATIONS} for scope in SCOPES}
    fold_pass=Counter();both_pass=0
    for source in sources:
        original=results["per_source"][source];item=dict(role=original["role"],dataset_block=original["dataset_block"],lags={},cross_lag={},source_held_out_support={})
        for lag in ("1","4"):
            item["lags"][lag]={scope:{pop:compact_cell(original["lags"][lag][scope][pop],pop) for pop in POPULATIONS} for scope in SCOPES}
            fold=results["source_held_out_support"][source][lag];keys=("positive_pairs","negative_pairs","hard_queries")
            item["source_held_out_support"][lag]=dict(training=[fold["training_support"][k] for k in keys],evaluation=[fold["evaluation_support"][k] for k in keys],
                necessary_positive_negative_support=fold["necessary_positive_negative_support"])
            check(fold["necessary_positive_negative_support"]==all(fold[side][key]>0 for side in ("training_support","evaluation_support") for key in keys),"Source-held-out necessary support arithmetic")
            fold_pass[lag]+=int(fold["necessary_positive_negative_support"])
        both=all(item["source_held_out_support"][lag]["necessary_positive_negative_support"] for lag in ("1","4"))
        item["both_lags_necessary_positive_negative_support"]=both;both_pass+=int(both)
        for scope in SCOPES:
            item["cross_lag"][scope]={}
            for pop in POPULATIONS:
                cross=original["cross_lag"][scope][pop]
                item["cross_lag"][scope][pop]=[cross[key] for key in CROSS_FIELDS]
                for key in CROSS_FIELDS:cross_union[scope][pop][key]+=cross[key]
        per_source[source]=item
    check(len(sources)==341 and set(results["source_held_out_support"])==set(sources),"Complete 341-source support/fold coverage")
    primary={}
    for lag in ("1","4"):
        value=results["micro_union"][lag][PRIMARY[0]][PRIMARY[1]]
        primary[lag]=dict(candidate_pairs=value["candidate_pairs"],known_pairs=value["known_pairs"],positive_pairs=value["positive_pairs"],negative_pairs=value["negative_pairs"],unknown_pairs=value["unknown_pairs"],
            nonempty_known_queries=value["nonempty_known_queries"],positive_available_queries=value["positive_available_queries"],negative_available_queries=value["negative_available_queries"],hard_queries=value["hard_queries"],
            no_known_positive_queries=value["nonempty_known_queries"]-value["positive_available_queries"],no_known_negative_queries=value["nonempty_known_queries"]-value["negative_available_queries"],
            official_GT_trajectories=value["official_GT_trajectories"],sources_with_known_pairs=value["sources_with_known_pairs"],sources_with_hard_queries=value["sources_with_hard_queries"],sources_with_raw_hard_errors=value["sources_with_raw_hard_errors"],
            paired_top1=value["top1"],positive_negative_pair_order=value["positive_negative_pair_order"])
        for scope in SCOPES:
            for pop in POPULATIONS:
                union=results["micro_union"][lag][scope][pop]
                for key in SUPPORT_FIELDS[:9]:check(union[key]==sum(results["per_source"][source]["lags"][lag][scope][pop][key] for source in sources),"Union/source total mismatch: "+key)
    gate=results["necessary_support_gate"]
    check(gate["status"]=="HOLD_PENDING_SEPARATE_TRAINING_PROTOCOL" and gate["training_protocol_frozen"] is False and gate["sufficient_training_support_established"] is False,"Support gate must remain HOLD")
    return dict(status="DERIVED_INDEPENDENTLY_AUDITED_FULL_FIT_LAG_IDENTITY_SUPPORT_BRIEF",schema="FIT_LAG_IDENTITY_SUPPORT_BRIEF_V1",split="FIT_ONLY_ELIGIBLE_MANUAL_IDENTITY",lags=[1,4],lag_unit="sampled_steps",time_unit="original_frames",ego_compensation=False,
        population=summary["population"],uncollected_allocation_videos=summary["uncollected_allocation_videos"],eligible_FIT_manual_anchors=summary["eligible_FIT_manual_anchors"],
        complete_candidate_counts_by_lag=summary["generated"]["candidates_by_lag"],primary_scope=dict(scope=PRIMARY[0],population=PRIMARY[1]),primary_two_lag=primary,
        micro_union_all_scopes=results["micro_union"],source_support_vector_fields=list(SUPPORT_FIELDS),
        feature_names_by_population={pop:list(names(pop)) for pop in POPULATIONS},correct_repair_break_vector_fields=["feature_correct","repair_raw_error","break_raw_correct"],
        cross_lag_vector_fields=list(CROSS_FIELDS),source_held_out_support_vector_fields=["positive_pairs","negative_pairs","hard_queries"],
        all_341_sources=per_source,cross_lag_union=cross_union,
        source_held_out_fold_counts=dict(total_sources=len(sources),necessary_support_by_lag=dict(fold_pass),both_lags_necessary_support=both_pass,
            both_lags_without_necessary_support=len(sources)-both_pass,unsupported_folds_retained=True),
        necessary_support_gate=gate,source_counts=dict(hard_query_sources=len(gate["hard_query_sources"]),raw_error_sources=len(gate["raw_error_sources"]),cross_lag_shared_hard_sources=len(gate["cross_lag_shared_hard_sources"])),
        independent_audit=dict(status=audit["status"],candidate_score_rows=sum(audit["independently_recomputed"]["candidates_by_lag"].values()),
            SQLite_table_rows=audit["independently_reconstructed_SQLite_table_rows"],feature_maximum_absolute_error=audit["independently_recomputed"]["feature_maximum_absolute_error"],
            elapsed_seconds=audit["elapsed_seconds"],peak_RSS_KiB=audit["peak_RSS_KiB"]),
        producer_resources=dict(elapsed_seconds=summary["elapsed_seconds"],peak_RSS_KiB=summary["peak_RSS_KiB"],maximum_score_rows=summary["maximum_score_rows_read"],maximum_feature_ring_frames=summary["generated"]["maximum_feature_ring_frames"]),
        bindings=bindings,unknown_is_negative=False,known_no_positive_is_certified_unmatched=False,training_started=False,GPU_used=False,parameters_or_thresholds_selected=False,
        interpretation="Necessary descriptive support only. Known-subset top1 is conditional on sparse eligible identities; unknown references can contain the true identity. Candidates and queries share source, video and GT trajectories. No complete-candidate assignment, training readiness, generalization or novelty claim is established.")


def main():
    started=time.monotonic();brief_path=OUT/"support_brief.json";complete_path=OUT/"completion_receipt.json"
    check(not brief_path.exists() and not complete_path.exists(),"Preserve existing final stage artifacts")
    summary=read(OUT/"summary.json");audit=read(OUT/"independent_output_audit.json");seal=read(OUT/"GT_free_candidates_sealed.json")
    execution=read(EXECUTION/"execution_receipt.json");spec=read(EXECUTION/"launch_spec.json")
    check(execution["status"]=="PASS_FIT_LAG_SUPPORT_EXECUTION_AND_INDEPENDENT_AUDIT","Supervisor execution incomplete or failed")
    check(execution["seven_protected_unchanged"] is True and execution["all_bound_file_pins_unchanged"] is True,"Supervisor final checks failed")
    check(audit["status"]=="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_OUTPUT_AUDIT","Independent full output audit is not PASS")
    check(summary["status"]=="COMPLETE_ALL_SEALED_FIT_LAG_SUPPORT_PENDING_INDEPENDENT_AUDIT","Unexpected sealed producer status")
    check(seal["status"]=="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION","Missing global GT-free seal")
    for path,digest in spec["file_pins"].items():check(sha(path)==digest,"Final pinned file changed: "+path)
    for path,digest in summary["input_sha256"].items():check(sha(path)==digest,"Recorded producer input changed: "+path)
    for relative,digest in seal["files_sha256"].items():check(sha(OUT/relative)==digest,"Globally sealed output member changed: "+relative)
    for relative,digest in summary["output_files_sha256"].items():check(sha(OUT/relative)==digest,"Final producer output changed: "+relative)
    check(seal["files_sha256"]==summary["generated"]["files_sha256"],"Summary/global seal hashes differ")
    check(set(seal["files_sha256"])==set(audit["independently_recomputed"]["files"]),"Independent audit/global seal member set differs")
    for key,name in (("summary_sha256","summary.json"),("support_SQLite_sha256","support.sqlite"),("global_GT_free_seal_sha256","GT_free_candidates_sealed.json"),("producer_execution_receipt_sha256","producer_execution_receipt.json")):
        check(audit[key]==sha(OUT/name),"Independent audit binding differs: "+key)
    check(execution["independent_output_audit"]["sha256"]==sha(OUT/"independent_output_audit.json"),"Supervisor/auditor hash binding differs")
    check(execution["launch_spec_sha256"]==sha(EXECUTION/"launch_spec.json"),"Supervisor launch specification changed")
    stages={}
    for index,name in enumerate(("producer","verifier")):
        exit_record=read(EXECUTION/(name+"_exit.json"));process=read(EXECUTION/(name+"_process.json"))
        check(exit_record["actual_tool_execution"]["exit_code"]==0 and exit_record["actual_tool_execution"]["subprocess"] is True,"Actual child exit0 absent: "+name)
        check(exit_record["actual_tool_execution"]["pid"]==process["pid"] and exit_record["process"]==process["process"],"Actual child process identity differs: "+name)
        check(exit_record["launch_spec_sha256"]==sha(EXECUTION/"launch_spec.json") and exit_record==execution["stages"][index],"Execution stage/launch chain differs: "+name)
        stages[name]=exit_record
    check(sha(OUT/"producer_execution_receipt.json")==sha(EXECUTION/"producer_exit.json"),"Producer receipt copies differ")
    protected_before=read(EXECUTION/"protected_before.json");protected_after=read(EXECUTION/"protected_after.json");authority=read(AUTHORITY)
    check(protected_before["status"]=="PASS_SEVEN_PROTECTED_BEFORE" and protected_after["status"]=="PASS_SEVEN_PROTECTED_AFTER","Seven-file protection statuses invalid")
    check(len(protected_before["protected_files"])==7 and protected_before["protected_files"]==protected_after["protected_files"]==authority["protected_files"]==spec["protected_expected"],"Seven protected files differ across authority/before/after")
    for path,info in protected_after["protected_files"].items():
        check(Path(path).exists()==info["exists"],"Protected file existence changed")
        if info["exists"]:check(sha(path)==info["sha256"],"Protected file bytes changed")
    for log in execution["logs"].values():check(sha(log["path"])==log["sha256"] and Path(log["path"]).stat().st_size==log["bytes"],"Final child log changed")
    check(all(audit[key] is True for key in ("all_source_scope_lag_and_identity_distributions_verified","source_held_out_and_cross_lag_support_verified","all_inputs_outputs_and_protected_baseline_hashes_verified")),"Incomplete independent coverage")
    for flag in ("training_started","parameters_or_thresholds_selected","GPU_used","tune_test_or_classheldout_targets_decoded","unknown_is_negative"):
        check(summary[flag] is False,"Forbidden producer scope: "+flag)
    files=[OUT/"summary.json",OUT/"support.sqlite",OUT/"GT_free_candidates_sealed.json",OUT/"independent_output_audit.json",OUT/"producer_execution_receipt.json",
        EXECUTION/"launch_spec.json",EXECUTION/"launch_receipt.json",EXECUTION/"supervisor_ready.json",EXECUTION/"producer_process.json",EXECUTION/"producer_exit.json",
        EXECUTION/"verifier_process.json",EXECUTION/"verifier_exit.json",EXECUTION/"execution_receipt.json",EXECUTION/"protected_before.json",EXECUTION/"protected_after.json",AUTHORITY,
        Path(spec["producer"]),Path(spec["protocol"]),Path(spec["preflight"]),Path(spec["verifier"]),Path(__file__).resolve()]
    bindings={str(path.relative_to(ROOT)):evidence(path) for path in files}
    brief=derive_brief(summary,audit,bindings)
    payload=json.dumps(brief,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n"
    check(len(payload.encode())<=2_000_000,"Lightweight brief exceeds 2MB")
    with brief_path.open("x") as stream:stream.write(payload)
    member_counts=dict(total=len(seal["files_sha256"]),score_shards=sum(name.endswith(".scores.npy") for name in seal["files_sha256"]),rowmaps=sum(name.endswith(".rows.json") for name in seal["files_sha256"]),frame_pair_receipt_files=1)
    check(member_counts==dict(total=1210,score_shards=806,rowmaps=403,frame_pair_receipt_files=1),"Unexpected complete GT-free member counts")
    receipt=dict(status="PASS_FIT_LAG_IDENTITY_SUPPORT_AND_INDEPENDENT_OUTPUT_AUDIT",recorded_utc=datetime.now(timezone.utc).isoformat(),stage_complete=True,overall_research_goal_complete=False,
        producer=dict(actual_exit_code=0,actual_tool_execution=stages["producer"]["actual_tool_execution"],process=stages["producer"]["process"],receipt=bindings[str((EXECUTION/"producer_exit.json").relative_to(ROOT))]),
        independent_audit=dict(actual_exit_code=0,status=audit["status"],actual_tool_execution=stages["verifier"]["actual_tool_execution"],process=stages["verifier"]["process"],receipt=bindings[str((EXECUTION/"verifier_exit.json").relative_to(ROOT))],audit=bindings[str((OUT/"independent_output_audit.json").relative_to(ROOT))]),
        support_brief=evidence(brief_path),bindings=bindings,supervisor_execution_status=execution["status"],all_bound_file_pins_unchanged=True,
        protected_files=dict(count=7,authority_before_after_and_current_identical=True,before=bindings[str((EXECUTION/"protected_before.json").relative_to(ROOT))],after=bindings[str((EXECUTION/"protected_after.json").relative_to(ROOT))]),
        global_GT_free_seal=dict(status=seal["status"],evidence=bindings[str((OUT/"GT_free_candidates_sealed.json").relative_to(ROOT))],member_counts=member_counts,
            candidates_by_lag=seal["candidates_by_lag"],frame_pair_receipts=seal["frame_pair_receipts"],all_member_hashes_reverified=True,
            verification_basis="Bound sequential producer source and global GT-free hashes before supervised join; no independent instrumentation of payload read times"),
        independent_full_recomputation=dict(candidate_score_rows=sum(audit["independently_recomputed"]["candidates_by_lag"].values()),SQLite_table_rows=audit["independently_reconstructed_SQLite_table_rows"],
            feature_maximum_absolute_error=audit["independently_recomputed"]["feature_maximum_absolute_error"]),
        final_child_logs=execution["logs"],necessary_support_gate=brief["necessary_support_gate"],training_started=False,GPU_used=False,unknown_is_negative=False,parameters_or_thresholds_selected=False,
        builder_sha256=sha(__file__),builder_elapsed_seconds=time.monotonic()-started,derivation="Exact read-only derivation and hash/arithmetic checks over existing sealed producer/auditor/execution outputs; no scientific run repeated")
    with complete_path.open("x") as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
    check(read(complete_path)["status"]=="PASS_FIT_LAG_IDENTITY_SUPPORT_AND_INDEPENDENT_OUTPUT_AUDIT","Final receipt serialization failed")
    print(json.dumps(dict(status=receipt["status"],brief=evidence(brief_path),completion=evidence(complete_path),builder_sha256=sha(__file__),builder_elapsed_seconds=receipt["builder_elapsed_seconds"],source_held_out_fold_counts=brief["source_held_out_fold_counts"]),indent=2))


if __name__=="__main__":main()
