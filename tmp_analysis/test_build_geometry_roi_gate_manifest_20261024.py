#!/usr/bin/env python3
"""Three-round CPU tests for the geometry ROI gate manifest.

Run with ``--round N``.  Each round includes two synthetic checks and two
checks against the real generated manifest and source inputs.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/geometry_roi_gate_manifest_20261024_v1"
SUPPORT = ROOT / "results/base_error_full_collection_20261023/support_v1"
RUN = ROOT / "results/base_error_full_collection_20261023/full_collect_v1"
MODULE_PATH = Path(__file__).with_name("build_geometry_roi_gate_manifest_20261024.py")
SPEC = importlib.util.spec_from_file_location("roi_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def synthetic_eligible() -> dict[str, dict]:
    rows = {}
    oid = 0
    for source in ("S3", "S1", "S2"):
        for video in (11, 10):
            for step in (0, 2, 4, 6, 8, 10, 12):
                rows[str(oid)] = {
                    "observation_id": oid,
                    "source_group": source,
                    "video_id": video,
                    "image_id": video * 100 + step,
                    "sampled_step": step,
                    "original_frame_index": step * 30,
                    "local_track_id": 0,
                    "pred_category_id": 1,
                    "gt_track_id": 7,
                }
                oid += 1
    return rows


class SyntheticChecks(unittest.TestCase):
    def test_synthetic_hash_plan_is_input_order_invariant(self):
        rows = synthetic_eligible()
        first = module.selection_plan(rows, 91)
        reversed_rows = dict(reversed(list(rows.items())))
        second = module.selection_plan(reversed_rows, 91)
        self.assertEqual(first, second)
        self.assertEqual(len(first["source_order"]), 3)

    def test_synthetic_pairs_exclude_unknown_and_count_three_denominators(self):
        rows = []
        for step, identity, pred in ((1, 4, 9), (2, 4, 8), (3, 5, 9)):
            rows.append(
                dict(
                    source_group="S",
                    video_id=1,
                    sampled_step=step,
                    gt_track_id=identity,
                    pred_category_id=pred,
                )
            )
        rows.append(
            dict(
                source_group="S",
                video_id=1,
                sampled_step=4,
                gt_track_id=None,
                pred_category_id=9,
            )
        )
        counts = module.count_pairs(rows)
        self.assertEqual(counts["known_same_gt_cross_step"], 1)
        self.assertEqual(counts["known_different_gt_cross_step"], 2)
        self.assertEqual(counts["known_different_gt_same_pred_category"], 1)
        self.assertEqual(counts["unknown_identity_pairs_excluded"], 3)


class RealChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = OUT / "manifest.json"
        if not path.is_file():
            raise unittest.SkipTest("Run the manifest builder first")
        cls.manifest = json.loads(path.read_text(encoding="utf-8"))

    def test_real_structure_steps_folds_and_pair_recomputation(self):
        manifest = self.manifest
        self.assertEqual(manifest["status"], "PASS_GEOMETRY_ROI_GATE_MANIFEST")
        self.assertEqual(len(manifest["selected_sources"]), 6)
        self.assertEqual(len(manifest["folds"]), 3)
        for source in manifest["selected_sources"]:
            self.assertEqual(len(source["videos"]), 2)
            for video in source["videos"]:
                self.assertEqual(len(video["sample_steps"]), 6)
                self.assertEqual(video["sample_steps"], sorted(set(video["sample_steps"])))
        for fold in manifest["folds"]:
            self.assertEqual(len(fold["sources"]), 2)
            for key in (
                "known_same_gt_cross_step",
                "known_different_gt_cross_step",
                "known_different_gt_same_pred_category",
            ):
                self.assertGreaterEqual(fold["pairs"][key], 30)
        labels = {
            row["observation_id"]: row for row in manifest["evaluation_only_labels"]
        }
        joined = []
        for row in manifest["model_input"]:
            joined.append(dict(row, **{k: labels[row["observation_id"]][k] for k in ("gt_track_id",)}))
        self.assertEqual(module.count_pairs(joined), manifest["overall_pairs"])

    def test_real_model_input_has_no_gt_and_images_dumps_exist(self):
        manifest = self.manifest
        self.assertEqual(module.forbidden_keys(manifest["model_input"]), [])
        self.assertEqual(manifest["validation"]["model_input_gt_leaks"], [])
        for row in manifest["model_input"]:
            self.assertTrue((RUN / "dump" / row["dump_file"]).is_file())
            self.assertFalse(any("gt" in key.lower() for key in row))
        for check in manifest["validation"]["frames"]["checks"]:
            self.assertTrue(check["image_exists"])
            self.assertTrue(check["dump_exists"])
            self.assertTrue(check["receipt_validated"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=(1, 2, 3))
    args, remaining = parser.parse_known_args()
    print("ROUND", args.round, file=sys.stderr)
    suite = unittest.TestSuite()
    for case in (SyntheticChecks("test_synthetic_hash_plan_is_input_order_invariant"),
                 SyntheticChecks("test_synthetic_pairs_exclude_unknown_and_count_three_denominators"),
                 RealChecks("test_real_structure_steps_folds_and_pair_recomputation"),
                 RealChecks("test_real_model_input_has_no_gt_and_images_dumps_exist")):
        suite.addTest(case)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
