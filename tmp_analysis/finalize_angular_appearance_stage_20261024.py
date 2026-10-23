#!/usr/bin/env python3
"""Read-only validation and lightweight derivation from a completed angular run."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/angular_appearance_control_20261024_v1"
EXECUTION = ROOT / "results/angular_appearance_control_execution_20261024_v1"
AUTHORITY = ROOT / "tmp_analysis/feature_spacetime_protected_before_20261024.json"
SCOPES = ("all_candidates", "same_predicted_category")
POPS = ("all_known_population", "common_CV_covered_population")
FEATURES = ("raw_appearance_cosine", "angular_prediction_cosine")
SUBSETS = ("all_nonempty_known_queries", "positive_and_negative_available_queries")
TRIPLE_COUNTS = ("endpoint_known_same_candidates", "past_unknown", "past_known_same", "past_known_different",
                 "available_locator_endpoint_same_candidates", "distinct_immutable_row_triplets",
                 "distinct_video_GT_ID_triplets", "fully_known_triplet_videos", "fully_known_endpoint_GT_trajectories")
COUNTS = ("candidate_pairs", "known_pairs", "positive_pairs", "negative_pairs", "unknown_pairs",
          "nonempty_known_queries", "positive_available_queries", "negative_available_queries", "hard_queries")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evidence(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def compact(cell):
    result = {k: cell[k] for k in COUNTS}
    check(result["known_pairs"] == result["positive_pairs"] + result["negative_pairs"], "Known-pair arithmetic")
    check(result["candidate_pairs"] == result["known_pairs"] + result["unknown_pairs"], "Unknown-pair arithmetic")
    check(result["positive_available_queries"] + result["negative_available_queries"] - result["hard_queries"] == result["nonempty_known_queries"], "Query support arithmetic")
    result["top1"] = cell["top1"]
    for subset, total in (("all_nonempty_known_queries", result["nonempty_known_queries"]),
                          ("positive_and_negative_available_queries", result["hard_queries"])):
        for feature in FEATURES:
            row = result["top1"][subset][feature]
            keys = ("raw_wrong_feature_wrong", "raw_wrong_feature_correct", "raw_correct_feature_wrong", "raw_correct_feature_correct")
            check(sum(row[k] for k in keys) == total, "Paired top1 denominator")
            check(row["repair_raw_error"] == row[keys[1]] and row["break_raw_correct"] == row[keys[2]], "Paired repair/break cells")
            check(row["raw_correct"] == row[keys[2]] + row[keys[3]] and row["feature_correct"] == row[keys[1]] + row[keys[3]], "Paired correctness cells")
            check(row["feature_correct"] == row["raw_correct"] + row["repair_raw_error"] - row["break_raw_correct"], "Repair/break arithmetic")
    result["history_truth_support"] = cell["history_truth_support"]
    result["query_macros_and_error_concentration"] = {
        subset: {k: v for k, v in metrics.items() if k != "query_GT_error_distribution"}
        for subset, metrics in cell["query_macros_and_error_concentration"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-incomplete", action="store_true", help="Reuse an existing byte-identical derived brief only; never overwrite final receipts")
    args = parser.parse_args()
    brief_path, complete_path = OUT / "control_brief.json", OUT / "completion_receipt.json"
    check(not complete_path.exists() and (not brief_path.exists() or args.recover_incomplete), "Preserve existing final artifacts")
    summary, audit, seal = (read(OUT / name) for name in ("summary.json", "independent_output_audit.json", "GT_free_candidates_sealed.json"))
    execution, spec = (read(EXECUTION / name) for name in ("execution_receipt.json", "launch_spec.json"))
    check(execution["status"] == "PASS_ANGULAR_APPEARANCE_CONTROL_EXECUTION_AND_INDEPENDENT_AUDIT", "Execution incomplete")
    check(execution["seven_protected_unchanged"] and execution["all_bound_file_pins_unchanged"], "Final execution checks")
    check(audit["status"] == "PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_OUTPUT_AUDIT", "Independent audit incomplete")
    check(summary["status"] == "COMPLETE_ALL_SEALED_FIT_ANGULAR_CONTROL_PENDING_INDEPENDENT_AUDIT", "Producer incomplete")
    check(seal["status"] == "SEALED_ALL_FIT_ANGULAR_CANDIDATES_BEFORE_SUPERVISION", "Missing global seal")
    for records in (spec["file_pins"], summary["input_sha256"]):
        for path, digest in records.items():
            check(sha(path) == digest, "Changed input: " + path)
    for records in (seal["files_sha256"], summary["output_files_sha256"]):
        for relative, digest in records.items():
            check(sha(OUT / relative) == digest, "Changed output: " + relative)
    check(seal["files_sha256"] == summary["generated"]["files_sha256"], "Producer seal binding")
    check(set(seal["files_sha256"]) == set(audit["independently_recomputed"]["files"]), "Independent full member coverage")
    for key, name in (("summary_sha256", "summary.json"), ("support_SQLite_sha256", "support.sqlite"),
                      ("global_GT_free_seal_sha256", "GT_free_candidates_sealed.json"),
                      ("producer_execution_receipt_sha256", "producer_execution_receipt.json")):
        check(audit[key] == sha(OUT / name), "Auditor binding: " + key)
    check(execution["independent_output_audit"]["sha256"] == sha(OUT / "independent_output_audit.json"), "Supervisor audit binding")
    check(execution["launch_spec_sha256"] == sha(EXECUTION / "launch_spec.json"), "Supervisor specification binding")
    stages = {}
    for i, name in enumerate(("producer", "verifier")):
        record, process = (read(EXECUTION / (name + suffix)) for suffix in ("_exit.json", "_process.json"))
        check(record["actual_tool_execution"]["exit_code"] == 0 and record["actual_tool_execution"]["subprocess"], "Actual child exit absent: " + name)
        check(record["actual_tool_execution"]["pid"] == process["pid"] and record["process"] == process["process"], "Child process binding")
        check(record == execution["stages"][i] and record["launch_spec_sha256"] == sha(EXECUTION / "launch_spec.json"), "Stage binding")
        stages[name] = record
    check(sha(OUT / "producer_execution_receipt.json") == sha(EXECUTION / "producer_exit.json"), "Producer receipt copies")
    before, after, authority = (read(path) for path in (EXECUTION / "protected_before.json", EXECUTION / "protected_after.json", AUTHORITY))
    check(before["status"] == "PASS_SEVEN_PROTECTED_BEFORE" and after["status"] == "PASS_SEVEN_PROTECTED_AFTER", "Protected checks")
    check(len(before["protected_files"]) == 7 and before["protected_files"] == after["protected_files"] == authority["protected_files"] == spec["protected_expected"], "Seven protected bindings")
    for path, info in after["protected_files"].items():
        check(Path(path).exists() == info["exists"], "Protected existence")
        if info["exists"]:
            check(sha(path) == info["sha256"], "Protected content")
    for log in execution["logs"].values():
        check(sha(log["path"]) == log["sha256"] and Path(log["path"]).stat().st_size == log["bytes"], "Child log binding")
    for key in ("all_source_scope_lag_and_identity_distributions_verified", "all_original_raw_scores_counts_statistics_and_cross_lag_verified",
                "forecast_masks_distributions_and_three_anchor_history_support_verified", "all_inputs_outputs_and_protected_baseline_hashes_verified"):
        check(audit[key] is True, "Incomplete audit: " + key)
    for key in ("training_started", "parameters_or_thresholds_selected", "GPU_used", "tune_test_or_classheldout_targets_decoded", "unknown_is_negative"):
        check(summary[key] is False, "Forbidden scope: " + key)
    results = summary["results"]
    check(len(results["per_source"]) == 341, "Original source population")
    union = {lag: {scope: {pop: compact(results["micro_union"][lag][scope][pop]) for pop in POPS} for scope in SCOPES} for lag in ("1", "4")}
    per_source = {}
    for source, item in results["per_source"].items():
        per_source[source] = dict(role=item["role"], dataset_block=item["dataset_block"],
            lags={}, cross_lag=item["cross_lag"])
        for lag in ("1", "4"):
            per_source[source]["lags"][lag] = {}
            for scope in SCOPES:
                per_source[source]["lags"][lag][scope] = {}
                for pop in POPS:
                    cell = item["lags"][lag][scope][pop]
                    compact(cell)  # Check every source arithmetic as well as the union.
                    per_source[source]["lags"][lag][scope][pop] = dict(
                        support=[cell[k] for k in COUNTS],
                        correct_repair_break={subset: [[cell["top1"][subset][f][k] for k in ("feature_correct", "repair_raw_error", "break_raw_correct")] for f in FEATURES] for subset in SUBSETS},
                        history_triplets=[cell["history_truth_support"][k] for k in TRIPLE_COUNTS])
    for lag in ("1", "4"):
        for scope in SCOPES:
            for pop in POPS:
                for key in COUNTS:
                    check(union[lag][scope][pop][key] == sum(item["lags"][lag][scope][pop]["support"][COUNTS.index(key)] for item in per_source.values()), "Source/union arithmetic")
                for subset in SUBSETS:
                    for feature in FEATURES:
                        for key, count in union[lag][scope][pop]["top1"][subset][feature].items():
                            check(count == sum(item["lags"][lag][scope][pop]["top1"][subset][feature][key] for item in results["per_source"].values()), "Source/union paired top1 arithmetic")
    paths = [OUT / name for name in ("summary.json", "support.sqlite", "GT_free_candidates_sealed.json", "independent_output_audit.json", "producer_execution_receipt.json")]
    paths += [EXECUTION / name for name in ("launch_spec.json", "launch_receipt.json", "supervisor_ready.json", "producer_process.json", "producer_exit.json", "verifier_process.json", "verifier_exit.json", "execution_receipt.json", "protected_before.json", "protected_after.json")]
    paths += [AUTHORITY, Path(__file__).resolve(), *[Path(spec[k]) for k in ("producer", "protocol", "preflight", "verifier")]]
    bindings = {str(path.relative_to(ROOT)): evidence(path) for path in paths}
    brief = dict(status="DERIVED_INDEPENDENTLY_AUDITED_FULL_FIT_ANGULAR_CONTROL_BRIEF", population=summary["population"],
        eligible_FIT_manual_anchors=summary["eligible_FIT_manual_anchors"], candidates_by_lag=seal["candidates_by_lag"],
        micro_union=union, all_341_sources=per_source, cross_lag=results["cross_lag"], baseline_parity=summary["baseline_parity"],
        source_support_vector_fields=list(COUNTS), source_history_triplet_vector_fields=list(TRIPLE_COUNTS), feature_order=list(FEATURES), correct_repair_break_vector_fields=["feature_correct", "repair_raw_error", "break_raw_correct"],
        resources=dict(producer_seconds=summary["elapsed_seconds"], verifier_seconds=audit["elapsed_seconds"], producer_peak_RSS_KiB=summary["peak_RSS_KiB"], verifier_peak_RSS_KiB=audit["peak_RSS_KiB"]),
        bindings=bindings, interpretation="Sparse known-identity conditional ranking only; ordinary untrained control. No official tracking gain, novelty, training readiness or causal time-effect claim.")
    payload = json.dumps(brief, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    check(len(payload.encode()) < 2_000_000, "Lightweight brief exceeds 2MB")
    if brief_path.exists():
        check(brief_path.read_bytes() == payload.encode(), "Incomplete brief differs from exact current derivation; preserve it")
    else:
        with brief_path.open("x") as stream:
            stream.write(payload)
    receipt = dict(status="PASS_ANGULAR_APPEARANCE_CONTROL_AND_INDEPENDENT_OUTPUT_AUDIT", recorded_utc=datetime.now(timezone.utc).isoformat(),
        stage_complete=True, overall_research_goal_complete=False, supervisor_execution_status=execution["status"],
        stages=stages, bindings=bindings, control_brief=evidence(brief_path), all_bound_file_pins_unchanged=True,
        seven_protected_unchanged=True, all_sealed_member_hashes_reverified=True, globally_sealed_members=len(seal["files_sha256"]),
        candidates_by_lag=seal["candidates_by_lag"], independently_reconstructed_SQLite_table_rows=audit["independently_reconstructed_SQLite_table_rows"],
        feature_maximum_absolute_error=audit["independently_recomputed"]["feature_maximum_absolute_error"],
        training_started=False, GPU_used=False, unknown_is_negative=False,
        verification_basis="Sequential pinned producer and complete sealed hashes; actual child wait receipts owned by detached supervisor; no independent instrumentation of payload read times")
    with complete_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=receipt["status"], brief=evidence(brief_path), completion=evidence(complete_path)), indent=2))


if __name__ == "__main__":
    main()
