"""Copy an explicit, bounded research-stage snapshot to the managed Git repo.

No data, model, prediction JSON, raw posterior, NPZ, PDF or log is copied.
This helper does not commit or push: inspect the Git diff after synchronization.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ["项目进度.md", "OVTrack_收尾总结.md", "OVMOT_方法调研笔记.md",
           "COVTrack_改进方案.md", "COVTrack_论文级改进设计.md",
           "COVTrack_完整val轨迹读出_20261023.md", "COVTrack_自然依赖核验_20261023.md"]
SCRIPTS = ["build_track_posterior_cache.py", "evaluate_track_readouts.py",
           "summarize_track_readouts.py", "score_track_cache_diagnostic.py",
           "build_dependence_readouts.py", "build_teta_classification_events.py",
           "score_dependence_readouts.py", "diagnose_natural_dependence.py",
           "diagnose_natural_dependence_posterior01.py",
           "evaluate_dependence_readout.py", "verify_dependence_stage.py", "sync_research_stage.py",
           "novelty_audit_20261023.md", "covtrackpp_code_boundary_20261023.md",
           "dependence_independent_review_20261023.md", "base_error_learning_readiness_20261023.md",
           "build_base_error_learning_plan.py", "base_error_learning_plan_20261023.json"]
DEPENDENCE_JSON = ["design.json", "manifest.json", "completion_receipt.json", "protected_after.json",
                   "teta_events/receipt.json", "scores/score_summary.json",
                   "readback_verification.json", "natural_numerical_preflight.json", "natural_mapping_preflight.json",
                   "natural_error_dependence_summary.json", "natural_execution_receipt.json", "official_execution_receipt.json",
                   "natural_protocol_deviation_receipt.json", "posterior01/natural_numerical_preflight.json",
                   "posterior01/natural_mapping_preflight.json", "posterior01/natural_error_dependence_summary.json",
                   "posterior01/natural_execution_receipt.json", "posterior01/boundary_contract.json",
                   "posterior01/primary_reproduction_check.json",
                   "readouts/corrected_mask_sumlog/receipt.json", "readouts/corrected_mask_sumlog/metrics.json",
                   "readouts/released.dep_inverse_density/receipt.json", "readouts/released.dep_inverse_density/metrics.json",
                   "readouts/corrected.dep_inverse_density/receipt.json", "readouts/corrected.dep_inverse_density/metrics.json"]
BASE_SCRIPTS = ["prepare_base_error_collection.py", "collect_base_error_pilot.py",
                "verify_base_error_dump.py", "verify_base_error_collection.py",
                "base_error_learning_objective_20261023.md", "check_base_error_objective_synthetic.py",
                "base_error_objective_synthetic_20261023.json", "base_error_pipeline_cpu_validation_20261023.json",
                "base_collection_protocol_review_20261023.md", "base_collection_verifier_rejection_checks_20261023.json",
                "build_base_error_support.py", "base_error_collection_deviations_20261023.json",
                "finish_base_error_collection.py", "base_support_review_handoff_20261023.json"]
BASE_SCRIPTS += ["base_error_support_cpu_check_20261023.json", "base_error_support_handoff_20261023.md"]
BASE_FINAL_SCRIPTS = ["base_error_support_cpu_check_20261023_v2.json",
                      "base_support_independent_metadata_provenance_20261023.json", "base_support_contract_audit_20261023.json",
                      "base_support_independent_review_20261023.md", "base_pilot_support_interpretation_20261023.md",
                      "diagnose_base_mean_quality.py", "base_mean_quality_predeclared_spec_20261023_v2.json",
                      "base_mean_quality_synthetic_receipt_20261023_v2.json",
                      "prepare_base_error_full_collection.py", "verify_base_error_full_collection.py",
                      "run_full_collection_background.py", "base_error_pause_handoff_20261023.md"]
BASE_JSON = ["preparation_manifest.json", "smoke_v2/collection_receipt.json",
             "smoke_v2/run_manifest.json", "smoke_v2/dump_verification.json", "smoke_v2/independent_collection_audit.json",
             "pilot_collect_v1/collection_receipt.json", "pilot_collect_v1/run_manifest.json",
             "pilot_collect_v1/protected_after.json", "pilot_collect_v1/dump_verification.json",
             "pilot_collect_v1/independent_collection_audit.json", "execution_receipt.json",
             "support_v1/support_receipt.json", "support_v1/support_summary.json"]
BASE_JSON += ["pilot_collect_v1/background_completion_audit.json", "support_v1/independent_support_audit.json",
              "mean_quality_v1/diagnostic_summary.json", "mean_quality_v1/independent_mean_quality_audit.json",
              "fit_tune_execution_receipt.json"]
FULL_SCRIPTS = ["build_base_error_full_support.py", "base_error_full_support_adaptation_contract_20261023.json",
                "base_error_full_support_cpu_execution_20261023.json", "full_support_independent_preflight_20261023.json",
                "full_support_independent_preflight_20261023.md", "verify_base_error_full_support_output.py",
                "diagnose_full_mean_quality.py", "full_mean_implementation_spec_20261023.json",
                "full_mean_synthetic_receipt_20261023_v2.json", "full_mean_independent_synthetic_audit_20261023.json",
                "full_mean_independent_preflight_20261023.json", "full_mean_pair_gate_protocol_20261023.json",
                "full_mean_pair_gate_protocol_20261023.md", "full_stage_handoff_20261023.md",
                "audit_full_mean_quality.py", "full_mean_output_auditor_synthetic_receipt_20261023.json"]
FULL_CONTEXT_SCRIPTS = ["build_full_pair_context_cache.py", "verify_full_pair_context_output.py",
                       "full_pair_context_sealer_independent_preflight_20261023.json",
                       "full_pair_context_sealer_independent_preflight_20261023.md",
                       "pair_conditioning_identifiability_audit_20261023.md",
                       "check_pair_conditioning_mismatch_20261023.py", "pair_conditioning_mismatch_synthetic_20261023.json",
                       "semantic_uncertainty_2609_22351_fulltext_audit_20261023.md",
                       "semantic_uncertainty_2609_22351_source_receipt_20261023.json"]
FULL_CONTEXT_JSON = ["pair_context_execution_receipt.json", "independent_pair_context_audit_execution_receipt.json",
                     "pair_context_v1/pair_context_receipt.json", "pair_context_v1/source_roles.json",
                     "pair_context_v1/independent_pair_context_audit.json", "pair_route_decision.json"]
FULL_IDENTITY_SCRIPTS = ["build_fit_identity_edge_support.py", "verify_fit_identity_edge_support_output.py",
                         "base_fit_identity_edge_support_20261023_synthetic_v2.json",
                         "base_fit_identity_edge_support_20261023_independent_preflight.json",
                         "base_fit_identity_edge_support_20261023_independent_preflight.md",
                         "base_fit_identity_edge_support_20261023_output_verifier_synthetic.json",
                         "base_fit_identity_edge_support_20261023_oracle_design.md"]
FULL_IDENTITY_JSON = ["fit_identity_execution_receipt.json",
                      "independent_fit_identity_edge_support_audit_execution_receipt.json",
                      "base_fit_identity_edge_support_20261023/identity_support_brief.json",
                      "base_fit_identity_edge_support_20261023/independent_identity_support_audit.json"]
FULL_JSON = ["background_execution_receipt.json", "independent_completion_review.json",
             "full_collect_v1/collection_receipt.json", "full_collect_v1/run_manifest.json",
             "full_collect_v1/model_ready.json", "full_collect_v1/protected_before.json", "full_collect_v1/protected_after.json",
             "support_v1/support_summary.json", "support_v1/independent_support_audit.json",
             "independent_support_audit_execution_receipt.json",
             "mean_quality_v1/diagnostic_summary.json", "mean_quality_v1/independent_mean_quality_audit.json",
             "fit_tune_execution_receipt.json"]
FULL_PAUSE_SCRIPTS = FULL_SCRIPTS + [
                "evaluator_audit_handoff_20261023.json",
                "check_full_mean_pair_gate_synthetic.py", "full_mean_pair_gate_synthetic_20261023.json",
                "build_full_pair_context_cache.py", "full_pair_context_cache_synthetic_20261023.json"]
# This checkpoint records incomplete work explicitly; it never substitutes for final audits.
# The 2.6 MB full support receipt stays local and is hash-bound by the pause receipt.
FULL_PAUSE_JSON = ["background_execution_receipt.json", "independent_completion_review.json",
                  "full_collect_v1/collection_receipt.json", "full_collect_v1/run_manifest.json",
                  "full_collect_v1/model_ready.json", "full_collect_v1/protected_before.json",
                  "full_collect_v1/protected_after.json", "support_v1/support_summary.json",
                  "support_v1/independent_support_audit.json",
                  "independent_support_audit_execution_receipt.json",
                  "dialogue_pause_receipt.json"]

# Independent feature stage: do not inherit historical dependence/base evidence.
# Raw cache/prediction rows and upstream papers remain local and hash-bound only.
FEATURE_SPACETIME_REPORTS = ["项目进度.md", "COVTrack_特征与时空改进_20261024.md"]
FEATURE_SPACETIME_SCRIPTS = [
    "sync_research_stage.py", "feature_spacetime_handoff_20261024.md",
    "full_stage_handoff_20261023.md", "covtrack_feature_spacetime_design_20261024.md",
    "feature_spacetime_protected_before_20261024.json",
    "feature_spacetime_stage_completion_20261024.json",
    "audit_covtrack_feature_memory_path_20261024.py",
    "covtrack_feature_memory_path_independent_review_20261024.md",
    "covtrack_feature_memory_path_independent_review_20261024.json",
    "audit_covtrack_geometry_time_math_20261024.py",
    "covtrack_geometry_time_math_independent_review_20261024.json",
    "context_feature_memory.py", "replay_context_feature_memory.py",
    "context_feature_memory_protocol_20261024.json",
    "context_feature_memory_replay_protocol_20261024.json",
    "context_feature_memory_synthetic_round1_20261024_v2.json",
    "context_feature_memory_synthetic_round2_20261024_v2.json",
    "context_feature_memory_synthetic_round3_20261024_v2.json",
    "audit_context_feature_memory_implementation_20261024.py",
    "context_feature_memory_20261024_independent_preflight.json",
    "audit_context_feature_memory_runner_20261024.py",
    "context_feature_memory_independent_preflight_20261024.json",
    "verify_context_feature_memory_output_20261024.py",
    "supplement_context_memory_partition_20261024.py",
    "relative_track_geometry.py", "relative_track_geometry_synthetic_20261024.json",
    "relative_track_geometry_independent_review_20261024_v2.json",
    "identity_memory_direction_audit_20261023.md",
    "identity_memory_direction_source_receipt_20261023.json",
    # Explicit lightweight dependencies used by the replay/audit identity decoder.
    "build_fit_identity_edge_support.py", "verify_fit_identity_edge_support_output.py",
]
FEATURE_MEMORY_JSON = [
    "summary.json", "protocol.json", "input_hashes.json", "predictions_sealed.json",
    "producer_execution_receipt.json", "completion_receipt.json",
    "independent_output_audit.json", "independent_audit_execution_receipt.json",
    "independent_primary_partition_supplement.json",
]
# Final candidate v2 sources; real-output receipts are added by exact name only.
FEATURE_SPACETIME_CANDIDATE_SCRIPTS = [
    "diagnose_spacetime_candidate_features_20261024.py",
    "spacetime_candidate_features_protocol_20261024.json",
    "spacetime_candidate_features_synthetic_round1_20261024_v2.json",
    "spacetime_candidate_features_synthetic_round2_20261024_v2.json",
    "spacetime_candidate_features_synthetic_round3_20261024_v2.json",
    "spacetime_candidate_features_independent_preflight_20261024.json",
    "audit_spacetime_candidate_implementation_20261024.py",
    "verify_spacetime_candidate_features_output_20261024.py",
]
FEATURE_SPACETIME_CANDIDATE_JSON = [
    "spacetime_candidate_features_20261024_v1/summary.json",
    "spacetime_candidate_features_20261024_v1/GT_free_candidates_sealed.json",
    "spacetime_candidate_features_20261024_v1/producer_execution_receipt.json",
    "spacetime_candidate_features_20261024_v1/independent_output_audit.json",
    "spacetime_candidate_features_20261024_v1/independent_audit_execution_receipt.json",
    "spacetime_candidate_features_20261024_v1/completion_receipt.json",
]


def feature_spacetime_mapping():
    """Return the exact lightweight allowlist without reading or copying files."""
    mapping = [(ROOT / "report" / name, Path("report") / name)
               for name in FEATURE_SPACETIME_REPORTS]
    mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name)
                for name in FEATURE_SPACETIME_SCRIPTS + FEATURE_SPACETIME_CANDIDATE_SCRIPTS]
    memory = Path("context_feature_memory_smoke_20261024_v1")
    mapping += [(ROOT / "results" / memory / name, Path("evidence") / memory / name)
                for name in FEATURE_MEMORY_JSON]
    mapping += [(ROOT / "results" / source, Path("evidence") / source)
                for source in FEATURE_SPACETIME_CANDIDATE_JSON]
    return mapping


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "ovvvv")
    ap.add_argument("--stage", choices=["dependence_20261023", "base_error_interface_20261023", "base_error_20261023", "base_full_20261023", "base_full_pause_20261023", "feature_spacetime_20261024"], required=True)
    ap.add_argument("--diagnostic-json", nargs="*", default=[],
                    help="Exact relative JSON paths under dependence_audit_20261023; no glob")
    args = ap.parse_args()
    assert (args.repo / ".git").is_dir(), "Use the existing separate managed repository"
    if args.stage == "feature_spacetime_20261024":
        assert not args.diagnostic_json, "Feature stage uses its explicit allowlist only"
        mapping = feature_spacetime_mapping()
    else:
        mapping = [(ROOT / "report" / p, Path("report") / p) for p in REPORTS]
        mapping += [(ROOT / "tmp_analysis" / p, Path("tmp_analysis") / p) for p in SCRIPTS]
        stage = ROOT / "results/dependence_audit_20261023"
        json_paths = DEPENDENCE_JSON + args.diagnostic_json
        for name in json_paths:
            p = Path(name)
            assert not p.is_absolute() and ".." not in p.parts and p.suffix == ".json", name
            assert p.name not in ("tao_track.json", "validation_ours_v1.json"), name
            mapping.append((stage / p, Path("evidence/dependence_audit_20261023") / p))
        mapping += [(ROOT / "tmp_analysis/covtrackpp_live_sources_20261023" / name,
                     Path("evidence/covtrackpp_code_audit_20261023") / name)
                    for name in ("manifest.json", "audit_verification.json")]
    if args.stage in ("base_error_interface_20261023", "base_error_20261023", "base_full_20261023", "base_full_pause_20261023"):
        mapping.append((ROOT / "report/COVTrack_base观测采集_20261023.md", Path("report/COVTrack_base观测采集_20261023.md")))
        mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name) for name in BASE_SCRIPTS]
        base = ROOT / "results/base_error_collection_20261023_v3"
        selected_base = BASE_JSON[:5] if args.stage == "base_error_interface_20261023" else BASE_JSON
        mapping += [(base / name, Path("evidence/base_error_collection_20261023") / name) for name in selected_base]
        copy = ROOT / "work/covtrack_base_error_20261023"
        mapping += [(copy / name, Path("evidence/base_error_collection_20261023/schema") / name)
                    for name in ("SOURCE_COPY_MANIFEST.json", "SOURCE_PATCH_MANIFEST.json", "CPU_SCHEMA_SELF_TEST_FINAL.json",
                                 "ovtrack_models_mot_ovtrack.py.patch", "ovtrack_models_roi_heads_ovtrack_roi_head.py.patch")]
        mapping.append((copy / "ovtrack/base_error_dump.py", Path("evidence/base_error_collection_20261023/schema/base_error_dump.py")))
        if args.stage in ("base_error_20261023", "base_full_20261023", "base_full_pause_20261023"):
            mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name) for name in BASE_FINAL_SCRIPTS]
            full = ROOT / "results/base_error_full_collection_20261023"
            full_names = ["preparation_manifest.json", "independent_metadata_audit.json"]
            if (full / "background_launch_receipt.json").is_file():
                full_names += ["background_launch_receipt.json", "background_pause_handoff.json"]
            else:
                pause = json.loads((full / "pause_handoff.json").read_text())
                assert pause["status"] == "USER_PAUSED_BEFORE_FULL_COLLECTION_LAUNCH"
                assert pause["full_collection_launched"] is False
                full_names.append("pause_handoff.json")
            mapping += [(full / name, Path("evidence/base_error_full_collection_20261023") / name)
                        for name in full_names]
            if args.stage == "base_full_20261023":
                mapping.append((ROOT / "report/COVTrack_full质量控制_20261023.md", Path("report/COVTrack_full质量控制_20261023.md")))
                mapping.append((ROOT / "report/COVTrack_fit身份混入支持_20261023.md", Path("report/COVTrack_fit身份混入支持_20261023.md")))
                mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name) for name in FULL_SCRIPTS]
                mapping += [(full / name, Path("evidence/base_error_full_collection_20261023") / name) for name in FULL_JSON]
                mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name) for name in FULL_CONTEXT_SCRIPTS]
                mapping += [(full / name, Path("evidence/base_error_full_collection_20261023") / name) for name in FULL_CONTEXT_JSON]
                mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name) for name in FULL_IDENTITY_SCRIPTS]
                mapping += [(full / name, Path("evidence/base_error_full_collection_20261023") / name) for name in FULL_IDENTITY_JSON]
            if args.stage == "base_full_pause_20261023":
                mapping += [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name) for name in FULL_PAUSE_SCRIPTS]
                mapping += [(full / name, Path("evidence/base_error_full_collection_20261023") / name) for name in FULL_PAUSE_JSON]
    # Read and validate the entire allowlist before writing any destination.
    payloads = []
    destinations = set()
    for source, rel in mapping:
        assert not rel.is_absolute() and ".." not in rel.parts, str(rel)
        assert rel not in destinations, "Duplicate destination: " + str(rel)
        destinations.add(rel)
        assert source.is_file() and not source.is_symlink(), str(source)
        data = source.read_bytes()
        assert len(data) <= 2_000_000, "Only lightweight stage files: " + str(source)
        assert rel.suffix in (".md", ".py", ".json", ".patch")
        if rel.suffix == ".json":
            obj = json.loads(data)
            assert isinstance(obj, dict), "Evidence must be a summary/receipt object"
        payloads.append((source, rel, data))
    assert sum(len(data) for _, _, data in payloads) <= 20_000_000
    manifest = {"stage": args.stage, "source_root": str(ROOT), "files": [],
                "excludes": ["datasets", "checkpoints", "prediction rows", "raw posterior caches", "NPZ", "JSONL", "logs", "PDFs"],
                "meaning": "Source artifact paths remain local; evidence JSON is a copied receipt, not the underlying data"}
    for source, rel, data in payloads:
        destination = args.repo / rel
        assert not destination.is_symlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".sync-tmp")
        temp.write_bytes(data)
        temp.replace(destination)
        assert destination.read_bytes() == data
        manifest["files"].append({"source": str(source), "destination": str(rel),
                                  "bytes": len(data), "sha256": sha256(data)})
    snapshot_date = "20261024" if args.stage == "feature_spacetime_20261024" else "20261023"
    snapshot = args.repo / "evidence" / ("snapshot_" + snapshot_date + "_" + args.stage + ".json")
    snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"copied_files": len(payloads), "bytes": sum(len(x[2]) for x in payloads),
                      "snapshot": str(snapshot)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
