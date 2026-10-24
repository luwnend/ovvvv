#!/usr/bin/env python3
"""Unit and contract tests for the fixed-feature geometry-ROI gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp_analysis"))
import geometry_roi_information_gate_20261024 as subject  # noqa: E402


def _tokens(seed: int, signal: str | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(256, 7, 7)).astype(np.float32)
    if signal is not None:
        values[17, :, :] += 6.0
        if signal == "negative":
            values[17, :, :] = -6.0
    return values


def _observation(
    *,
    observation_id: int,
    gt_track_id: int | None,
    category: int,
    step: int,
    fold: int = 0,
    source: str = "Synthetic/source_a",
    video_id: int = 10,
    local_track_id: int = 1,
    row: int = 0,
    tokens: np.ndarray | None = None,
    raw: np.ndarray | None = None,
) -> subject.Observation:
    return subject.Observation(
        observation_id=observation_id,
        fold=fold,
        source=source,
        video_id=video_id,
        image_id=observation_id,
        sampled_step=step,
        original_frame_index=step * 30,
        local_track_id=local_track_id,
        pred_category_id=category,
        dump_file=f"frame_{observation_id}.npz",
        dump_row=row,
        raw=np.zeros(256, dtype=np.float32) if raw is None else raw,
        tokens=_tokens(observation_id) if tokens is None else tokens,
        box=np.asarray([0.2, 0.2, 0.6, 0.6], dtype=np.float64),
    )


class BasicMathTests(unittest.TestCase):
    def test_auroc_ties_and_absent_classes(self) -> None:
        self.assertEqual(subject.auroc([1, 1, 1, 1], [1, 0, 1, 0]), 0.5)
        self.assertAlmostEqual(
            subject.auroc([0.1, 0.2, 0.8, 0.9], [0, 1, 1, 0]), 0.5)
        self.assertAlmostEqual(
            subject.auroc([0.1, 0.2, 0.8, 0.9], [0, 1, 1, 1]), 1.0)
        self.assertIsNone(subject.auroc([0.1, 0.2], [0, 0]))
        self.assertIsNone(subject.auroc([], []))

    def test_cosine_roundoff_is_clipped(self) -> None:
        value = subject.cosine_pair(
            np.asarray([1.0 + 1e-12, 2.0]),
            np.asarray([1.0 + 1e-12, 2.0]))
        self.assertAlmostEqual(value, 1.0)
        self.assertLessEqual(value, 1.0)

    def test_spatial_prior_ignores_absolute_screen_position(self) -> None:
        tokens = _tokens(3)
        left = np.asarray([0.1, 0.1, 0.4, 0.5])
        middle_right = np.asarray([0.6, 0.5, 0.9, 0.9])
        translated = middle_right + np.asarray([-0.5, -0.4, -0.5, -0.4])
        self.assertAlmostEqual(
            subject.spatial_topk_score(tokens, tokens, left, middle_right),
            subject.spatial_topk_score(tokens, tokens, left, translated))

    def test_random_permutation_is_deterministic(self) -> None:
        first, second = _tokens(1), _tokens(2)
        box = np.asarray([0.1, 0.2, 0.7, 0.8])
        a = subject.spatial_topk_score(first, second, box, box, permutation_seed=17)
        b = subject.spatial_topk_score(first, second, box, box, permutation_seed=17)
        self.assertEqual(a, b)


class PopulationTests(unittest.TestCase):
    def test_unknown_is_excluded_and_same_category_is_primary(self) -> None:
        a = _observation(observation_id=1, gt_track_id=7, category=5, step=0)
        b = _observation(observation_id=2, gt_track_id=7, category=5, step=1)
        c = _observation(observation_id=3, gt_track_id=8, category=5, step=2)
        u = _observation(observation_id=4, gt_track_id=None, category=5, step=3)
        gt = {1: 7, 2: 7, 3: 8, 4: None}
        pairs = subject.build_pair_table([a, b, c, u], gt)
        primary = [row for row in pairs if row["primary_pair"]]
        self.assertEqual(len(primary), 3)
        self.assertEqual(sum(row["same_gt"] for row in primary), 1)
        self.assertEqual(
            sum(row["hard_negative_same_pred_category"] for row in primary), 2)
        self.assertFalse(any(
            row["query"].observation_id == 4 or row["reference"].observation_id == 4
            for row in pairs))
        support = subject._support_summary([a, b, c, u], pairs, gt)
        self.assertEqual(support["unknown_identity_rows_excluded"], 1)
        self.assertFalse(support["unknown_identity_used_as_negative"])

    def test_same_step_pairs_are_not_used(self) -> None:
        a = _observation(observation_id=1, gt_track_id=2, category=3, step=0, row=0)
        b = _observation(observation_id=2, gt_track_id=2, category=3, step=0, row=1)
        pairs = subject.build_pair_table([a, b], {1: 2, 2: 2})
        self.assertEqual(pairs, [])

    def test_past_aligned_mean_does_not_see_future_step(self) -> None:
        early = _observation(
            observation_id=1, gt_track_id=1, category=2, step=0,
            tokens=np.ones((256, 7, 7), dtype=np.float32))
        future = _observation(
            observation_id=2, gt_track_id=1, category=2, step=9,
            tokens=np.full((256, 7, 7), 999.0, dtype=np.float32))
        subject.build_past_only_aligned_means([early, future])
        self.assertTrue(np.allclose(early.past_aligned_grid, 1.0))
        self.assertTrue(np.allclose(future.past_aligned_grid, 500.0))
        self.assertIsNot(early.past_aligned_grid, future.past_aligned_grid)
        causal_raw = np.random.default_rng(7).normal(size=256).astype(np.float32)
        early.raw = causal_raw.copy()
        future.raw = np.full(256, 999.0, dtype=np.float32)
        subject.build_past_only_aligned_means([early, future])
        self.assertTrue(np.allclose(early.past_raw_mean, causal_raw))
        self.assertTrue(np.allclose(future.past_raw_mean, (causal_raw + 999) / 2))

    def test_pair_groups_require_positive_and_hard_negative(self) -> None:
        a = _observation(observation_id=1, gt_track_id=1, category=3, step=0)
        b = _observation(observation_id=2, gt_track_id=1, category=3, step=1)
        c = _observation(observation_id=3, gt_track_id=2, category=3, step=2)
        d = _observation(observation_id=4, gt_track_id=3, category=9, step=3)
        groups = subject._candidate_groups(
            [a, b, c, d], {1: 1, 2: 1, 3: 2, 4: 3})
        self.assertEqual([query.observation_id for query, _ in groups], [1, 2])
        self.assertEqual(
            {reference.observation_id for reference in groups[0][1]}, {2, 3})

    def test_unknown_candidate_stays_in_pool_but_is_not_a_negative(self) -> None:
        a = _observation(observation_id=1, gt_track_id=1, category=3, step=0)
        b = _observation(observation_id=2, gt_track_id=1, category=3, step=1)
        c = _observation(observation_id=3, gt_track_id=2, category=3, step=2)
        u = _observation(observation_id=4, gt_track_id=None, category=3, step=3)
        groups = subject._candidate_groups(
            [a, b, c, u], {1: 1, 2: 1, 3: 2, 4: None})
        self.assertEqual(len(groups), 2)
        self.assertEqual(
            {reference.observation_id for reference in groups[0][1]}, {2, 3, 4})
        pairs = subject.build_pair_table([a, b, c, u], {1: 1, 2: 1, 3: 2, 4: None})
        support = subject._support_summary(
            [a, b, c, u], pairs, {1: 1, 2: 1, 3: 2, 4: None})
        self.assertEqual(support["unknown_identity_rows_excluded"], 1)
        self.assertEqual(support["primary_hard_negative_pairs"], 2)


class ContractTests(unittest.TestCase):
    def test_label_file_does_not_contain_gt_fields(self) -> None:
        # The protocol exporter is the only bridge between GT-free model rows
        # and scoring labels.  Keep their schemas disjoint.
        manifest = subject.DEFAULT_MANIFEST
        if not manifest.is_file():
            self.skipTest("Protocol manifest is not present")
        _, rows, labels = subject.load_manifest_rows(manifest)
        self.assertEqual(len(rows), 122)
        self.assertTrue(all(not any(key.startswith("gt_") for key in row)
                            for row in rows))
        self.assertTrue(all(set(row) == {
            "observation_id", "gt_track_id", "gt_category_id"} for row in labels))

    def test_manifest_rejects_gt_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads(subject.DEFAULT_MANIFEST.read_text())
            manifest["model_input"][0]["gt_track_id"] = 11
            path = root / "bad.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(subject.GateError):
                subject.load_manifest_rows(path)

    def test_explicit_export_hashes_are_stable(self) -> None:
        _, rows, labels = subject.load_manifest_rows(subject.DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory() as temporary:
            first = subject.export_separated_protocol(
                {"_source_path": str(subject.DEFAULT_MANIFEST)},
                rows, labels, Path(temporary) / "a")
            second = subject.export_separated_protocol(
                {"_source_path": str(subject.DEFAULT_MANIFEST)},
                rows, labels, Path(temporary) / "b")
            self.assertEqual(
                first["observations_sha256"], second["observations_sha256"])
            self.assertEqual(first["labels_sha256"], second["labels_sha256"])


class SyntheticGateTests(unittest.TestCase):
    def test_local_signal_has_an_identifiable_oracle(self) -> None:
        """Local tokens carry the only identity signal; raw is distractor noise.

        This is an implementation oracle, not a claim that real ROI tokens have
        this signal.  Both spatial matching and token-mean should discover it.
        """
        raw_rng = np.random.default_rng(41)
        observations = []
        gt = {}
        for step in range(4):
            for track in range(2):
                observation_id = 1 + track * 10 + step
                signal = f"positive_{track}" if track == 0 else "negative"
                tokens = _tokens(100 + track * 10 + step, signal=signal)
                raw = raw_rng.normal(size=256).astype(np.float32)
                observation = _observation(
                    observation_id=observation_id,
                    gt_track_id=track,
                    category=4,
                    step=step,
                    local_track_id=track,
                    row=observation_id,
                    tokens=tokens,
                    raw=raw,
                )
                observations.append(observation)
                gt[observation_id] = track
        pairs = subject.build_pair_table(observations, gt)
        for arm in ("token_mean", "spatial_topk_geometry"):
            metrics = subject.source_macro_auroc(pairs, arm)
            self.assertEqual(metrics["source_macro"], 1.0, arm)
        top1 = subject.top1_metrics(observations, gt)
        self.assertEqual(
            top1["spatial_topk_geometry"]["accuracy_known_resolved"], 1.0)
        self.assertEqual(
            top1["spatial_topk_geometry"]["unresolved_unknown_selection"], 0)
        self.assertIn("local_tokens_vs_causal_raw", top1["matched_contrasts"])

    def test_fold_status_reports_missing_real_coverage(self) -> None:
        if not subject.DEFAULT_CACHE.is_dir():
            self.skipTest("ROI pilot cache is not present")
        with tempfile.TemporaryDirectory() as temporary:
            result = subject.run_protocol(
                subject.DEFAULT_MANIFEST, subject.DEFAULT_CACHE,
                Path(temporary), source_filter="ArgoVerse/bae67a44-0f30-30c1-8999-06fc1c7ab80a",
                video_filter=2583, expected_source_count=6)
            self.assertEqual(
                result["status"], "PASS_SMOKE_ONLY_REAL_CACHE_INCOMPLETE")
            self.assertFalse(result["complete_three_fold"])
            self.assertEqual(result["support"]["sources"], 1)
            self.assertTrue(result["protocol"]["gt_query_population_filter"])
            self.assertFalse(result["protocol"]["gt_removes_or_adds_candidates"])
            self.assertEqual(
                result["support"]["primary_hard_negative_pairs"],
                sum(row["hard_negative_pairs"]
                    for row in result["auroc"]["raw256"]["per_source"].values()))
            self.assertTrue(all(
                "unresolved_unknown_selection" in metrics
                for arm, metrics in result["top1"].items()
                if arm != "matched_contrasts"))
            self.assertFalse(result["gpu_used"])
            self.assertFalse(result["training_started"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
