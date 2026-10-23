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
BASE_JSON = ["preparation_manifest.json", "smoke_v2/collection_receipt.json",
             "smoke_v2/run_manifest.json", "smoke_v2/dump_verification.json", "smoke_v2/independent_collection_audit.json",
             "pilot_collect_v1/collection_receipt.json", "pilot_collect_v1/run_manifest.json",
             "pilot_collect_v1/protected_after.json", "pilot_collect_v1/dump_verification.json",
             "pilot_collect_v1/independent_collection_audit.json", "execution_receipt.json",
             "support_v1/support_receipt.json", "support_v1/support_summary.json"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "ovvvv")
    ap.add_argument("--stage", choices=["dependence_20261023", "base_error_interface_20261023", "base_error_20261023"], required=True)
    ap.add_argument("--diagnostic-json", nargs="*", default=[],
                    help="Exact relative JSON paths under dependence_audit_20261023; no glob")
    args = ap.parse_args()
    assert (args.repo / ".git").is_dir(), "Use the existing separate managed repository"
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
    if args.stage in ("base_error_interface_20261023", "base_error_20261023"):
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
    # Read and validate the entire allowlist before writing any destination.
    payloads = []
    for source, rel in mapping:
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
                "excludes": ["datasets", "checkpoints", "prediction rows", "raw posterior caches", "logs", "PDFs"],
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
    snapshot = args.repo / "evidence" / ("snapshot_20261023_" + args.stage + ".json")
    snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"copied_files": len(payloads), "bytes": sum(len(x[2]) for x in payloads),
                      "snapshot": str(snapshot)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
