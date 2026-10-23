#!/usr/bin/env python3
"""Independent synthetic and GT-free-metadata angular implementation gate."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import verify_angular_appearance_control_output_20261024 as independent

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "tmp_analysis/diagnose_angular_appearance_control_20261024.py"
PROTOCOL = ROOT / "tmp_analysis/angular_appearance_control_protocol_20261024.json"
EXPECTED_SCRIPT = "1e9fae320d2435a4560ffeeba3bfe441891f9d9db202296b269ba0419096828e"
EXPECTED_PROTOCOL = "44d8bce20de4ed3fd358a083665e6e9d8b2744d29d86dcdc9b2d9efa2f07eb69"
CORE_SHA = "8b534c5f133880165e3e46abbd93e49584ebf8e20bec9c58b46ab425fa8f5924"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    check, sha, same = independent.check, independent.sha, independent.same
    check(sha(SUBJECT) == EXPECTED_SCRIPT and sha(PROTOCOL) == EXPECTED_PROTOCOL, "Final implementation/protocol changed")
    subject = module(SUBJECT, "independent_angular_preflight_subject")
    check(sha(subject.CORE) == CORE_SHA, "Angular primitive changed")
    core = module(subject.CORE, "independent_angular_preflight_subject_core")
    receipts = {}
    for number in (1, 2, 3):
        path = ROOT / ("tmp_analysis/angular_appearance_control_synthetic_round%d_20261024_v2.json" % number)
        receipt = independent.load(path)
        check(receipt["status"] == "PASS_ANGULAR_APPEARANCE_CONTROL_SYNTHETIC_ROUND" and receipt["round"] == number,
              "Missing reviewed synthetic round")
        check(receipt["script_sha256"] == EXPECTED_SCRIPT and receipt["protocol_sha256"] == EXPECTED_PROTOCOL and receipt["core_sha256"] == CORE_SHA,
              "Synthetic round final hash mismatch")
        check(len(receipt["checks"]) >= 2, "Synthetic round requires independent checks")
        receipts[str(path)] = sha(path)
    protocol = independent.load(PROTOCOL)
    roles, videos, metadata, grouped, population = independent.base.metadata_population(protocol)
    same(population, dict(sealed_fit_source_groups=341, allocated_fit_videos=405, selected_fit_videos=403,
        selected_frames=13791, retained_observations=613536), "Independent complete FIT GT-free metadata")
    omitted = subject.validate_population(roles, videos, metadata, protocol["population"])
    check([row["video_id"] for row in omitted] == [1720, 1974] and all(not row["source_in_sealed_roles"] for row in omitted), "Uncollected allocation policy")
    wrong = copy.deepcopy(metadata)
    wrong[0]["video_id"] = 1720
    try:
        subject.validate_population(roles, videos, wrong, protocol["population"])
    except ValueError:
        pass
    else:
        raise AssertionError("Cached missing-role allocation accepted")
    math_checks = independent.math_self_test()
    random = np.random.RandomState(14411)
    for trial in range(120):
        n = int(random.randint(0, 6))
        last, previous = random.randn(n, 256), random.randn(n, 256)
        if n > 0:
            last[0] = 0
        if n > 1:
            previous[1] = 0
        if n > 2:
            previous[2] = last[2] * -1
        if n > 3:
            previous[3] = last[3] * 3
        qn = int(random.randint(0, 5))
        query = dict(video_id=1, sampled_step=5, original_frame_index=trial + 500,
            boxes=np.ones((qn, 4)), raw=random.randn(qn, 256))
        times = random.randint(0, 200, n).astype(float)
        if n:
            times[-1] = np.nan
        reference = dict(video_id=1, sampled_step=1, original_frame_index=300,
            boxes=np.ones((n, 4)), raw=last, previous_raw=previous, previous_times=times)
        actual, diagnostics = subject.score_block(query, reference, core)
        expected, oracle_diagnostics = independent.independent_scores(query, reference)
        np.testing.assert_allclose(actual.reshape(qn, n, 2), expected, atol=1e-12, rtol=1e-12)
        same(diagnostics, {key: value.tolist() for key, value in oracle_diagnostics.items()}, "Independent random angular/degenerate masks and rho/theta")
        poison = dict(query, raw=np.zeros_like(query["raw"]), boxes=np.full_like(query["boxes"], -999))
        _, poisoned_diagnostics = subject.score_block(poison, reference, core)
        same(poisoned_diagnostics, diagnostics, "Query input cannot affect forecast")
    # Literal oracle for every 3x3 pair-order transition and paired top1 tie.
    seen = np.zeros((3, 3), int)
    with tempfile.TemporaryDirectory(prefix="independent_angular_rank_oracle_") as temporary:
        actual = subject.open_database(Path(temporary) / "actual.sqlite")
        expected = independent.database(Path(temporary) / "expected.sqlite")
        for trial in range(150):
            nq, nr = int(random.randint(1, 5)), int(random.randint(1, 6))
            scores = random.randint(-2, 3, (nq, nr, 2)).astype(float)
            qids, rids = random.choice([-1, 1, 2, 3], nq), random.choice([-1, 1, 2, 3], nr)
            mask = random.rand(nq, nr) > .2
            a, b = subject.fresh_cell(), independent.fresh_cell()
            pop = independent.POPULATIONS[trial % 2]
            subject.update_cell(a, scores, mask, qids, rids, np.arange(nr), "S", 1, "all_candidates", pop, 1, trial, actual)
            independent.count_block(b, scores, mask, qids, rids, "S", 1, "all_candidates", pop, 1, trial, expected)
            same(subject.finalized_cell(a, pop), independent.final_cell(b), "Literal random ranking and pair-order")
            seen += np.asarray(b["pair_order"][independent.FEATURES[1]])
        check((seen > 0).all(), "Not every pair-order transition tested")
        rows = independent.compare_databases(expected, actual)
        for pop in independent.POPULATIONS:
            same(subject.query_metrics(actual, "S", 1, "all_candidates", pop),
                 independent.query_metrics(expected, "S", 1, "all_candidates", pop), "Independent both-subset source trajectory macro/errors")
        actual.close(); expected.close()
    # History snapshots compared before and after future updates, including
    # duplicate/next-unique invalidation and missing ID exclusion.
    snapshots, records, _ = independent.synthetic_fixture()
    producer_history, oracle_history = {}, {}
    held = None
    for record in [row for row in records if row["video_id"] == 1]:
        frame = snapshots[record["video_id"], record["frame_id"]]
        a = subject.attach_history(copy.deepcopy(frame), producer_history)
        b = independent.independent_history(copy.deepcopy(frame), oracle_history)
        same(subject.frame_metadata(a), independent.independent_row_metadata(b), "Independent unique history locator snapshot")
        np.testing.assert_array_equal(a["previous_raw"], b["previous_raw"])
        if record["frame_id"] == 3:
            held = (a, a["previous_raw"].copy(), copy.deepcopy(a["previous_locators"]))
    np.testing.assert_array_equal(held[0]["previous_raw"], held[1])
    same(held[0]["previous_locators"], held[2], "Future cannot mutate retained reference snapshot")
    integration = independent.temporary_full_entry_test()
    check(sha(SUBJECT) == EXPECTED_SCRIPT and sha(PROTOCOL) == EXPECTED_PROTOCOL, "Subject changed during independent preflight")
    result = dict(status="PASS_INDEPENDENT_ANGULAR_APPEARANCE_CONTROL_IMPLEMENTATION_PREFLIGHT",
        script_sha256=EXPECTED_SCRIPT, protocol_sha256=EXPECTED_PROTOCOL, core_sha256=CORE_SHA,
        checker_sha256=sha(__file__), reviewed_synthetic_round_files_sha256=receipts,
        GT_free_metadata_population=population, uncollected_allocated_FIT_video_ids=[1720, 1974],
        synthetic_full_entry=integration, random_angular_oracle_trials=120, random_ranking_oracle_trials=150,
        random_ranking_SQLite_table_rows=rows, hand_math_oracle_checks=math_checks,
        checks=["complete_GT_free_metadata_original_roles_population_exact_lag_QxM_counts",
            "independent_shortest_arc_original_time_sign_irregular_gap_reverse_long_wrap_and_all_degeneracies",
            "zero_last_exact_zero_previous_zero_fallback_common_CV_never_shrunk",
            "unique_duplicate_next_unique_noID_and_future_invariant_raw_locator_snapshots",
            "literal_paired_top1_lowest_row_ties_unknown_notnegative_and_all_3x3_pair_orders",
            "both_nonempty_and_hard_query_source_GTtrajectory_macro_and_error_concentration",
            "full_temp_float16_NPZ_generation_global_seal_safe_manual_join_output_audit",
            "legacy_parent_sixcolumn_raw_exact_full_blocks_and_all_corresponding_statistics",
            "poisoned_class_error_risk_posterior_and_tune_identity_never_decoded",
            "three_tampered_candidate_member_types_and_missing_file_rejected_before_ID_IO",
            "source_scopes_lags_populations_crosslag_and_same_different_unknown_threeanchor_support",
            "all_three_final_v2_synthetic_receipts_bound_to_frozen_script_protocol_core"],
        real_cached_features_opened=False, real_supervised_payload_opened=False, real_model_opened=False, GPU=False,
        limits=["Synthetic implementation and GT-free metadata preflight only; real producer and independent output audit remain unfinished"])
    destination = ROOT / "tmp_analysis/angular_appearance_control_independent_preflight_20261024.json"
    independent.write(destination, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
