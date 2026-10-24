#!/usr/bin/env python3
"""CPU-only synthetic checks for the geometry ROI token collector.

Exactly two checks are assigned to each of three rounds:

1. row locator/keep mask; NCHW axis order
2. frozen manifest parameters; empty rows plus atomic NPZ write
3. source/path validation plus no-execute gate; GPU guard rejection
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import observe_geometry_roi_tokens_20261024 as collector


ROUNDS = {
    1: (
        "test_row_locator_matches_keep_mask",
        "test_nchw_axis_order_is_enforced",
    ),
    2: (
        "test_frozen_manifest_parameters",
        "test_multiple_video_runs",
    ),
    3: (
        "test_source_paths_and_no_execute_gate",
        "test_gpu_guard_rejects_4gb_and_chenhc",
    ),
}


def valid_manifest(root: Path) -> dict:
    runner = Path(collector.__file__).resolve()
    trace = root / "trace.py"
    parent = root / "parent_manifest.json"
    config = root / "config.py"
    checkpoint = root / "checkpoint.pth"
    pins = {
        str(runner): collector.sha256_path(runner),
        str(trace): "1" * 64,
        str(parent): "2" * 64,
        str(config): "3" * 64,
        str(checkpoint): "4" * 64,
    }
    return {
        "schema_version": collector.SCHEMA_VERSION,
        "status": "FROZEN_GEOMETRY_ROI_TOKEN_COLLECTION_SYNTHETIC",
        "runner_sha256": collector.sha256_path(runner),
        "trace_path": str(trace),
        "GPU_UUID": "GPU-00000000-0000-0000-0000-000000000001",
        "output_directory": str(root / "never_created"),
        "parent_collection_manifest": str(parent),
        "code_root": str(root / "code"),
        "config_path": str(config),
        "checkpoint_path": str(checkpoint),
        "input_pins": pins,
        "cache_member_allowlist": list(collector.CACHE_MEMBER_ALLOWLIST),
        "token_dtype": "float16",
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "seed": 20261024,
            "threads": 1,
        },
        "frames": [{
            "source": "Synthetic/source",
            "video_id": 7,
            "image_id": 11,
            "frame_id": 0,
            "cache_path": str(root / "v7_f0.npz"),
            "cache_sha256": "a" * 64,
            "image_path": str(root / "v7_f0.jpg"),
            "image_sha256": "b" * 64,
        }],
    }


class GeometryROITokenCollectorTest(unittest.TestCase):
    def test_row_locator_matches_keep_mask(self):
        keep = np.asarray([True, False, True, False, False, False, True])
        rows = collector.retained_row_locators(
            keep, np.asarray([0, 2, 6]), len(keep))
        self.assertEqual(rows.tolist(), [0, 2, 6])
        with self.assertRaises(collector.CollectorError):
            collector.retained_row_locators(
                keep, np.asarray([0, 3, 6]), len(keep))
        with self.assertRaises(collector.CollectorError):
            collector.retained_row_locators(
                keep.astype(np.int64), np.asarray([0, 2, 6]), len(keep))
        with self.assertRaises(collector.CollectorError):
            collector.retained_row_locators(
                keep, np.asarray([0, 2]), len(keep))

    def test_nchw_axis_order_is_enforced(self):
        roi = np.arange(2 * 256 * 7 * 7, dtype=np.float32).reshape(
            2, 256, 7, 7)
        conv = roi + 1
        collector.validate_spatial_tokens(roi, conv, 2)
        with self.assertRaises(collector.CollectorError):
            collector.validate_spatial_tokens(roi.transpose(0, 3, 2, 1), conv, 2)
        with self.assertRaises(collector.CollectorError):
            collector.validate_spatial_tokens(roi, conv[:, :, :6], 2)

    def test_frozen_manifest_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = valid_manifest(Path(temporary).resolve())
            checked = collector.validate_manifest_structure(manifest)
            self.assertEqual(tuple(checked["cache_member_allowlist"]),
                             collector.CACHE_MEMBER_ALLOWLIST)
            self.assertEqual(checked["runtime"]["threads"], 1)
            wrong_allowlist = copy.deepcopy(manifest)
            wrong_allowlist["cache_member_allowlist"].append("track_ids")
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(wrong_allowlist)
            wrong_dtype = copy.deepcopy(manifest)
            wrong_dtype["token_dtype"] = "float64"
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(wrong_dtype)
            wrong_threads = copy.deepcopy(manifest)
            wrong_threads["runtime"]["threads"] = 2
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(wrong_threads)

    def test_empty_rows_and_atomic_npz(self):
        arrays = collector.build_frame_arrays(
            roi_input=np.zeros((0, 256, 7, 7), dtype=np.float32),
            conv_spatial=np.zeros((0, 256, 7, 7), dtype=np.float32),
            raw_output=np.zeros((0, 256), dtype=np.float32),
            keep_mask=np.zeros(0, dtype=bool),
            raw_pretracker_det_row=np.zeros(0, dtype=np.int64),
            roi_proposal_row=np.zeros(0, dtype=np.int64),
            det_bboxes=np.zeros((0, 5), dtype=np.float32),
            pretracker_det_bboxes=np.zeros((0, 5), dtype=np.float32),
            preprocessing={
                "scale_factor": [1.0, 1.0],
                "ori_shape": [100, 200, 3],
                "img_shape": [100, 200, 3],
                "pad_shape": [100, 200, 3],
            },
            source="Synthetic/empty",
            video_id=7,
            image_id=11,
            frame_id=0,
            image_sha256="b" * 64,
            cache_sha256="a" * 64,
            token_dtype="float16",
        )
        self.assertEqual(arrays["roi_input"].shape, (0, 256, 7, 7))
        self.assertEqual(arrays["conv_spatial"].shape, (0, 256, 7, 7))
        self.assertEqual(arrays["raw_output"].shape, (0, 256))
        self.assertEqual(arrays["pretracker_det_row"].shape, (0,))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "empty.npz"
            collector.atomic_write_npz(target, arrays)
            with np.load(target, allow_pickle=False) as saved:
                self.assertEqual(tuple(saved.files), collector.FRAME_NPZ_MEMBERS)
                self.assertEqual(saved["roi_input"].shape, (0, 256, 7, 7))
            with self.assertRaises(collector.CollectorError):
                collector.atomic_write_npz(target, arrays)

    def test_multiple_video_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = valid_manifest(Path(temporary).resolve())
            first = copy.deepcopy(manifest["frames"][0])
            first["video_id"] = 7
            first["image_id"] = 11
            first["frame_id"] = 0
            second = copy.deepcopy(first)
            second["image_id"] = 12
            second["frame_id"] = 1
            third = copy.deepcopy(first)
            third["video_id"] = 8
            third["image_id"] = 13
            third["frame_id"] = 0
            manifest["frames"] = [first, second, third]
            collector.validate_manifest_structure(manifest)
            split_run = copy.deepcopy(manifest)
            split_run["frames"] = [first, third, second]
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(split_run)
            wrong_origin = copy.deepcopy(manifest)
            wrong_origin["frames"][2]["frame_id"] = 1
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(wrong_origin)

    def test_source_paths_and_no_execute_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = valid_manifest(root)
            relative_cache = copy.deepcopy(manifest)
            relative_cache["frames"][0]["cache_path"] = "relative.npz"
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(relative_cache)
            blank_source = copy.deepcopy(manifest)
            blank_source["frames"][0]["source"] = " "
            with self.assertRaises(collector.CollectorError):
                collector.validate_manifest_structure(blank_source)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            result = collector.run(path, execute=False)
            self.assertEqual(result["status"],
                             "PASS_GEOMETRY_ROI_MANIFEST_PREFLIGHT_ONLY")
            self.assertFalse(Path(manifest["output_directory"]).exists())
            self.assertFalse(result["GPU_or_model_touched"])

    def test_gpu_guard_rejects_4gb_and_chenhc(self):
        uuid = "GPU-00000000-0000-0000-0000-000000000001"
        environment = {"CUDA_VISIBLE_DEVICES": uuid}

        def runner(command, **_kwargs):
            joined = " ".join(command)
            if "--query-gpu" in joined and "memory.total" in joined:
                if CURRENT_CASE["total"] == 4096:
                    return f"{uuid}, NVIDIA A100-SXM4-4GB, 4096, 4096\n"
                return f"{uuid}, NVIDIA A100-SXM4-40GB, 40960, 30000\n"
            if "--query-compute-apps" in joined:
                if CURRENT_CASE["owner"] is None:
                    return ""
                return f"{uuid}, 12345, 1024\n"
            if command[:2] == ["ps", "-o"]:
                return CURRENT_CASE["owner"] + "\n"
            raise AssertionError(f"Unexpected guard command: {command}")

        CURRENT_CASE = {"total": 4096, "owner": None}
        with self.assertRaises(collector.CollectorError):
            collector.gpu_guard(uuid, runner=runner, environment=environment)
        CURRENT_CASE = {"total": 40960, "owner": "chenhc"}
        with self.assertRaises(collector.CollectorError):
            collector.gpu_guard(uuid, runner=runner, environment=environment)
        CURRENT_CASE = {"total": 40960, "owner": "alice"}
        result = collector.gpu_guard(uuid, runner=runner, environment=environment)
        self.assertEqual(result["name"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual(result["preexisting_non_chenhc_processes"][0]["owner"],
                         "alice")
        with self.assertRaises(collector.CollectorError):
            collector.gpu_guard(uuid, runner=runner,
                                environment={"CUDA_VISIBLE_DEVICES": "2"})


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=tuple(ROUNDS), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    suite = unittest.TestSuite(
        GeometryROITokenCollectorTest(name) for name in ROUNDS[arguments.round])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "round": arguments.round,
        "checks": list(ROUNDS[arguments.round]),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "actual_GPU_or_model_used": False,
        "module_sha256": collector.sha256_path(
            Path(collector.__file__).resolve()),
        "test_sha256": collector.sha256_path(Path(__file__).resolve()),
    }
    with arguments.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
