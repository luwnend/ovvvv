"""Copy an exact lightweight FIT lag-support checkpoint; never launch or push.

Preparation and running snapshots describe unfinished work explicitly. Raw
scores, per-frame rows, SQLite, JSONL and full summaries stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "sync_fit_lag_support_stage_20261024.py",
    "identity_spacetime_factorial_design_20261024.md",
    "fit_lag_identity_support_protocol_20261024.json",
    "audit_fit_lag_identity_support_20261024.py",
    "fit_lag_identity_support_synthetic_round1_20261024_v3.json",
    "fit_lag_identity_support_synthetic_round2_20261024_v3.json",
    "fit_lag_identity_support_synthetic_round3_20261024_v3.json",
    "fit_lag_identity_support_independent_preflight_20261024.json",
    "audit_fit_lag_identity_support_implementation_20261024.py",
    "verify_fit_lag_identity_support_output_20261024.py",
    "fit_lag_identity_support_output_verifier_synthetic_20261024.json",
    "run_fit_lag_support_background_20261024.py",
    "fit_lag_support_background_synthetic_20261024_v2.json",
    "feature_spacetime_git_publication_20261024.json",
    "feature_spacetime_handoff_20261024.md",
    "fit_lag_identity_support_handoff_20261024.md",
]
REPORTS = ["项目进度.md", "COVTrack_特征与时空改进_20261024.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=("preparation", "running", "complete"))
    parser.add_argument("--repo", type=Path, default=ROOT.parent / "ovvvv")
    parser.add_argument("--receipt", action="append", default=[],
                        help="Exact dict JSON path under results, no glob")
    args = parser.parse_args()
    assert (args.repo / ".git").is_dir()
    mapping = [(ROOT / "tmp_analysis" / name, Path("tmp_analysis") / name)
               for name in SCRIPTS]
    mapping += [(ROOT / "report" / name, Path("report") / name)
                for name in REPORTS]
    allowed_prefixes = {"fit_lag_identity_support_20261024_v1",
                        "fit_lag_identity_support_execution_20261024_v1"}
    # Exact receipt paths are recorded in the manifest; data files cannot pass
    # these path/type/size restrictions or the explicit filename allowlist.
    allowed_names = {
        "launch_receipt.json", "supervisor_ready.json", "background_completion.json",
        "producer_execution_receipt.json", "independent_audit_execution_receipt.json",
        "independent_output_audit.json", "completion_receipt.json",
        "GT_free_candidates_sealed.json", "support_brief.json", "progress_receipt.json",
        "protected_before.json", "protected_after.json", "synthetic_receipt.json",
        "launch_spec.json", "supervisor_process.json", "producer_process.json",
        "producer_exit.json", "verifier_process.json", "verifier_exit.json",
        "execution_receipt.json",
    }
    if args.phase == "running":
        assert any(Path(x).name == "launch_receipt.json" for x in args.receipt)
        assert any(Path(x).name == "supervisor_ready.json" for x in args.receipt)
    if args.phase == "complete":
        complete = "fit_lag_identity_support_20261024_v1/completion_receipt.json"
        assert complete in args.receipt
        receipt = json.loads((ROOT / "results" / complete).read_text())
        assert receipt["status"] == "PASS_FIT_LAG_IDENTITY_SUPPORT_AND_INDEPENDENT_OUTPUT_AUDIT"
        assert receipt["producer"]["actual_exit_code"] == 0
        assert receipt["independent_audit"]["actual_exit_code"] == 0
    for value in args.receipt:
        rel = Path(value)
        assert not rel.is_absolute() and ".." not in rel.parts
        assert len(rel.parts) == 2 and rel.parts[0] in allowed_prefixes
        assert rel.name in allowed_names
        mapping.append((ROOT / "results" / rel, Path("evidence") / rel))
    payloads = []
    destinations = set()
    for source, destination in mapping:
        assert destination not in destinations
        destinations.add(destination)
        assert source.is_file() and not source.is_symlink(), str(source)
        data = source.read_bytes()
        assert len(data) <= 2_000_000, str(source)
        assert destination.suffix in (".py", ".md", ".json")
        if destination.suffix == ".json":
            assert isinstance(json.loads(data), dict)
        payloads.append((source, destination, data))
    assert sum(len(data) for _, _, data in payloads) <= 20_000_000
    manifest = {
        "stage": "fit_lag_identity_support_20261024", "phase": args.phase,
        "phase_is_completion": args.phase == "complete",
        "source_root": str(ROOT), "files": [],
        "excludes": ["NPZ", "NPY scores", "JSONL", "SQLite", "full summary",
                     "checkpoints", "logs", "datasets"],
    }
    for source, destination, data in payloads:
        target = args.repo / destination
        assert not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".sync-tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        assert target.read_bytes() == data
        manifest["files"].append({
            "source": str(source), "destination": str(destination),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        })
    snapshot = args.repo / "evidence" / (
        "snapshot_20261024_fit_lag_support_" + args.phase + ".json")
    snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(payloads), "phase": args.phase,
                      "bytes": sum(len(x[2]) for x in payloads),
                      "snapshot": str(snapshot)}))


if __name__ == "__main__":
    main()
