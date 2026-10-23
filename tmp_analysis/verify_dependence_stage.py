"""Seal this diagnostic stage after full evaluation, replay and source checks.

This stage receipt makes no claim that a new research method is complete.
Training output weights may progress in another session; only its immutable
config/launcher are checked, without touching that session's processes.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
from evaluate_track_readouts import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/dependence_audit_20261023"


def main():
    target = STAGE / "completion_receipt.json"
    assert not target.exists() and not (STAGE / "protected_after.json").exists()
    started = time.monotonic()
    bound = {}

    def bind(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = (ROOT if path.parts[0] in ("results", "tmp_analysis", "repos", "work", "data_tao", "scripts") else STAGE) / path
        digest = sha256(path)
        assert expected is None or digest == expected, str(path)
        bound[str(path)] = digest
        return digest

    def read(path):
        path = Path(path)
        if not path.is_absolute():
            path = STAGE / path
        bind(path)
        return json.load(path.open())

    manifest = read("manifest.json")
    scores = read("scores/score_summary.json")
    natural = read("natural_error_dependence.json")
    natural_summary = read("natural_error_dependence_summary.json")
    natural_execution = read("natural_execution_receipt.json")
    correction = read("posterior01/natural_error_dependence.json")
    correction_summary = read("posterior01/natural_error_dependence_summary.json")
    correction_execution = read("posterior01/natural_execution_receipt.json")
    deviation = read("natural_protocol_deviation_receipt.json")
    official_execution = read("official_execution_receipt.json")
    assert official_execution["status"] == "PASS_FULL_OFFICIAL_EXECUTION"
    for execution in official_execution["observed_execution_steps"]:
        assert execution["observed_exit_code"] == 0 and execution["observed_by"] == "tools.write_stdin"
        if "receipt" in execution:
            bind(execution["receipt"], execution["receipt_sha256"])
        bind(execution["log"], execution["log_sha256"])
    bind(ROOT / "tmp_analysis/evaluate_dependence_readout.py", official_execution["wrapper_script_sha256"])
    bind(ROOT / "tmp_analysis/evaluate_track_readouts.py", official_execution["shared_evaluator_script_sha256"])
    assert natural_execution["status"] == "COMPLETE_FULL_NATURAL_DEPENDENCE_DIAGNOSTIC_WITH_OBSERVED_TERMINAL_EXITS"
    for execution in natural_execution["stages"].values():
        assert execution["terminal_exit_code"] == 0 and execution["terminal_exit_observed"]
        bind(execution["artifact_path"], execution["artifact_sha256"])
    assert correction_execution["status"] == "COMPLETE_APPROVED_POSTERIOR_BOUNDARY_CORRECTION_WITH_OBSERVED_TERMINAL_EXITS"
    for execution in correction_execution["stages"].values():
        assert execution["terminal_exit_code"] == 0 and execution["terminal_exit_observed"]
        bind(execution["artifact_path"], execution["artifact_sha256"])
    assert deviation["status"] == "DEVIATION_DISCLOSED_AND_INDEPENDENT_CORRECTION_COMPLETE"
    for item in deviation.values():
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            bind(item["path"], item["sha256"])
    for scope, controls in natural["results"].items():
        for control in ("exact_lag_score_rank", "binned_lag_score_rank"):
            assert controls[control] == correction["results"][scope][control]
    assert correction_execution["approved_posterior_control"] == {"mean_edges": [0, .1, .3, .6, 1], "absolute_difference_cutpoint": .1}
    for summary in (natural_summary, correction_summary):
        bind(summary["full_artifact"]["path"], summary["full_artifact"]["sha256"])
    assert manifest["status"] == "PASS_DEPENDENCE_READOUTS_FULL"
    assert scores["status"] == "PASS_20_FIXED_GEOMETRY_CLASSIFICATION_REPLAYS"
    assert natural["status"] == "PASS_FULL_NATURAL_ERROR_DEPENDENCE_DESCRIPTIVE_AUDIT"
    bind(ROOT / "tmp_analysis/build_dependence_readouts.py", manifest["script_sha256"])
    event_receipt = read("teta_events/receipt.json")
    bind(ROOT / "tmp_analysis/build_teta_classification_events.py", event_receipt["script_sha256"])
    bind("readout_cache.npz", manifest["output_sha256"])
    bind("scores/per_class_metrics.npz", scores["per_class_metrics_sha256"])
    for path, digest in scores["inputs_sha256"].items():
        bind(path, digest)
    bind(ROOT / "tmp_analysis/score_dependence_readouts.py", scores["script_sha256"])
    bind(natural["script"]["path"], natural["script"]["sha256"])
    assert natural["source_sha256_before"] == natural["source_sha256_after"]
    for key, path in natural["source_paths"].items():
        bind(path, natural["source_sha256_after"][key])
    assert natural["scan"]["observations"] == manifest["audit"]["observations"] == 1529908
    assert natural["scan"]["files"] == manifest["audit"]["dump_files"] == 36371
    assert natural["scan"]["dump_filename_size_inventory_sha256"] == read(ROOT / "results/tces_audit_20261023/manifest.json")["dump_filename_size_inventory_sha256"]
    assert natural["matched_track_support"]["scope_all_pair_counts"] == {
        "single_gt_identity_track": 425198, "mixed_track_same_gt_identity": 95849,
        "mixed_track_different_gt_identity_same_gt_class": 40420,
        "mixed_track_different_gt_identity_different_gt_class": 8856}
    old = ROOT / "results/tces_audit_20261023"
    original_receipt = read(old / "readouts/tcc/receipt.json")
    bind(old / "readouts/tcc/aggregate_raw.pkl", original_receipt["aggregate_raw_sha256"])
    original_raw = pickle.load((old / "readouts/tcc/aggregate_raw.pkl").open("rb"))
    raw_fields = ["Loc_TP", "Loc_FN", "Loc_FP", "LocA", "LocRe", "LocPr", "AssocA", "AssocRe", "AssocPr"]
    parity = {}
    for rule in ("corrected_mask_sumlog", "released.dep_inverse_density", "corrected.dep_inverse_density"):
        run = STAGE / "readouts" / rule
        receipt = read(run / "receipt.json")
        metrics = read(run / "metrics.json")
        assert receipt["status"] == "COMPLETE"
        bind(receipt["result"], receipt["result_sha256"])
        bind(run / "classification_counts.npz", receipt["counts_sha256"])
        bind(run / "metrics.json", receipt["metrics_sha256"])
        bind(run / "aggregate_raw.pkl", receipt["aggregate_raw_sha256"])
        bind(run / "OVTrack/teta_summary_results.pth", receipt["summary_sha256"])
        raw = pickle.load((run / "aggregate_raw.pkl").open("rb"))
        assert list(raw) == list(original_raw)
        for name in raw:
            for field in raw_fields:
                assert np.array_equal(raw[name][50][field], original_raw[name][50][field]), (rule, name, field)
        replay_name = "corrected.full_sumlog" if rule == "corrected_mask_sumlog" else rule
        replay_path = STAGE / "scores" / (replay_name + ".counts.npz")
        bind(replay_path, scores["methods"][replay_name]["counts_sha256"])
        with np.load(run / "classification_counts.npz", allow_pickle=False) as a, np.load(replay_path, allow_pickle=False) as b:
            for field in ("counts", "class_names", "sequence_names", "is_novel"):
                assert np.array_equal(a[field], b[field]), (rule, field)
        for split, values in scores["methods"][replay_name]["split"].items():
            for field, value in values.items():
                assert abs(metrics["split"][split][field] - value) < 1e-10, (rule, split, field)
        if ".dep_" in rule:
            assert receipt["parity"]["all_noncategorical_fields_equal"]
            assert receipt["parity"]["all_rows_checked"] == 1529908
            assert receipt["protected_sources_unchanged_after_evaluation"]
        parity[rule] = {"all_class_video_alpha_TP_FN_FP_exact": True, "raw_geometry_association_arrays_exact": True,
                        "all_replayed_split_metrics_abs_error_below": 1e-10}
    # All 20 output count files remain bound to the replay summary.
    for name, item in scores["methods"].items():
        bind("scores/" + name + ".counts.npz", item["counts_sha256"])
    protected = read(old / "protected_before.json")
    after = {}
    for path, before in protected.items():
        if not isinstance(before, dict) or "exists" not in before:
            continue
        assert Path(path).exists() == before["exists"], path
        digest = bind(path, before["sha256"]) if before["exists"] else None
        after[path] = {"exists": before["exists"], "sha256": digest}
    checkpoint = read(old / "checkpoint_receipt.json")
    bind(checkpoint["checkpoint"], checkpoint["sha256"])
    upstream = ROOT / "repos/COVTrack"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(upstream), text=True).strip()
    assert head == "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b"
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=str(upstream), check=True)
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=str(upstream)).split(b"\0")
    n_blobs = 0
    for entry in entries:
        if not entry:
            continue
        info, path = entry.split(b"\t", 1)
        mode, expected, index_stage = info.split()
        assert index_stage == b"0"
        full = upstream / path.decode()
        data = os.readlink(full).encode() if mode == b"120000" else full.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert digest == expected.decode(), str(full)
        n_blobs += 1
    assert n_blobs == 248
    # Agent execution receipts record actual terminal polls; preserve their
    # contents verbatim rather than guessing process success from output files.
    write_json(STAGE / "protected_after.json", {"files": after, "checkpoint_sha256": checkpoint["sha256"],
                                                "upstream_head": head, "all_tracked_blobs_equal_index": n_blobs})
    receipt = {"status": "PASS_DEPENDENCE_DIAGNOSTIC_STAGE", "research_goal_complete": False,
               "input_output_sha256": bound, "official_vs_replay": parity,
               "actual_execution_receipts": {"natural": natural_execution, "posterior_boundary_correction": correction_execution, "official": official_execution},
               "root_actual_terminal_polls": {"classification_event_cache_session_1308_exit": 0, "twenty_readouts_scoring_session_56085_exit": 0},
               "natural_summary_sha256": sha256(STAGE / "natural_error_dependence_summary.json"),
               "protected_after_sha256": sha256(STAGE / "protected_after.json"),
               "verifier_sha256": sha256(__file__), "elapsed_seconds": time.monotonic()-started,
               "conclusion_scope": "Current fixed appearance inverse-density readout is dominated by ordinary score weighting in both masks; natural same-error concentration is descriptive and leaves learned conditional error-process hypothesis unproven",
               "no_new_training_this_stage": True}
    write_json(target, receipt)
    print("PASS", target, flush=True)


if __name__ == "__main__":
    main()
