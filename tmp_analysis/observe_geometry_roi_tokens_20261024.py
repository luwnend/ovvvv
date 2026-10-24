#!/usr/bin/env python3
"""Read-only collector for retained geometry-ROI spatial tokens.

The collector is deliberately stricter than the four-frame observation pilot.
It consumes a future-frozen JSON manifest and never launches the model unless
``--execute`` is supplied.  Without that flag it performs manifest-shape
validation only: no GPU query, model import, directory creation, or inference.

Manifest contract (``OVMOT_GEOMETRY_ROI_TOKEN_MANIFEST_V1``)
-----------------------------------------------------------
Required top-level keys, with no unknown keys:

* ``schema_version`` and ``status``: exact strings shown in the constants below.
* ``runner_sha256``: SHA-256 of this file at freeze time.
* ``trace_path``: absolute path to ``roi_track_spatial_trace_20261024.py``.
* ``GPU_UUID``: exact value that must also be in ``CUDA_VISIBLE_DEVICES``.
* ``output_directory``: absolute, nonexistent destination directory.
* ``parent_collection_manifest``: absolute path to the frozen parent manifest.
* ``code_root``, ``config_path`` and ``checkpoint_path``: absolute model inputs.
* ``input_pins``: mapping from every required absolute input path above, including
  this runner and the parent manifest, to its lowercase SHA-256.  Additional
  frozen inputs may be pinned; all pins are rechecked after collection.
* ``cache_member_allowlist``: exactly ``CACHE_MEMBER_ALLOWLIST`` below.  The
  parent NPZ is opened without pickle and only these members are copied.  GT,
  identity, class/posterior, or matching fields are never accessed.
* ``token_dtype``: ``"float16"`` or ``"float32"`` for both saved spatial tensors.
* ``runtime``: exactly ``python`` (absolute interpreter path), ``seed`` (integer),
  and ``threads`` (must be 1).
* ``frames``: nonempty list of frame records.  Each record has exactly
  ``source``, ``video_id``, ``image_id``, ``frame_id``, ``cache_path``,
  ``cache_sha256``, ``image_path`` and ``image_sha256``.  Paths are absolute.
  Within each video, records are contiguous from ``frame_id == 0``.  This
  preserves the tracker timeline and the model's video-reset semantics.

Each frame NPZ has exactly the members in ``FRAME_NPZ_MEMBERS``.  Spatial arrays
are retained-row NCHW tensors with shape ``[N, 256, 7, 7]``.  ``raw_output`` is
the unmodified native track-head output with shape ``[N, 256]``.  Row maps are
``pretracker_det_row`` (index into detector rows) and ``roi_proposal_row``
(index into pre-NMS proposal rows).  Per-page receipt writing uses a same-
directory temporary file followed by ``os.replace``.  A failure exits nonzero
and is never retried.  An incomplete output directory has no final
``collection_receipt.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "OVMOT_GEOMETRY_ROI_TOKEN_MANIFEST_V1"
RUNNING_STATUS = "RUNNING_GEOMETRY_ROI_TOKEN_COLLECTION_V1"
COMPLETE_STATUS = "PASS_GEOMETRY_ROI_TOKEN_COLLECTION_V1"

CACHE_MEMBER_ALLOWLIST = (
    "track_feats",
    "tracker_keep_mask",
    "pretracker_det_row",
    "det_bboxes",
    "pretracker_det_bboxes",
    "roi_proposal_row",
)

FRAME_NPZ_MEMBERS = (
    "roi_input",
    "conv_spatial",
    "raw_output",
    "pretracker_det_row",
    "roi_proposal_row",
    "det_bboxes",
    "pretracker_det_bboxes",
    "preprocessing_json",
    "source",
    "video_id",
    "image_id",
    "frame_id",
    "image_sha256",
    "cache_sha256",
)

TOP_LEVEL_KEYS = frozenset({
    "schema_version",
    "status",
    "runner_sha256",
    "trace_path",
    "GPU_UUID",
    "output_directory",
    "parent_collection_manifest",
    "code_root",
    "config_path",
    "checkpoint_path",
    "input_pins",
    "cache_member_allowlist",
    "token_dtype",
    "runtime",
    "frames",
})

FRAME_KEYS = frozenset({
    "source",
    "video_id",
    "image_id",
    "frame_id",
    "cache_path",
    "cache_sha256",
    "image_path",
    "image_sha256",
})

RUNTIME_KEYS = frozenset({"python", "seed", "threads"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN_DTYPES = {"float16": np.dtype("float16"), "float32": np.dtype("float32")}


class CollectorError(RuntimeError):
    """A deterministic preflight, integrity, or collection failure."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _absolute_path(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value and "\x00" not in value,
            f"{field} must be a nonempty string")
    path = Path(value)
    require(path.is_absolute(), f"{field} must be absolute")
    require(path == Path(os.path.normpath(value)), f"{field} must be normalized")
    return path


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CollectorError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(),
                           object_pairs_hook=_no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectorError(f"Cannot parse JSON manifest {path}: {exc}") from exc
    require(isinstance(value, dict), "Manifest root must be a JSON object")
    return value


def validate_manifest_structure(spec: Mapping[str, Any],
                                *, check_runner: bool = True) -> dict[str, Any]:
    """Validate schema, frame order, allowlist, and runner binding.

    This function deliberately does not hash external inputs.  The execution
    path calls ``authenticate_manifest`` before touching the GPU.
    """
    require(isinstance(spec, Mapping), "Manifest must be a mapping")
    spec = dict(spec)
    require(set(spec) == TOP_LEVEL_KEYS,
            "Manifest keys changed: "
            f"missing={sorted(TOP_LEVEL_KEYS - set(spec))}, "
            f"unknown={sorted(set(spec) - TOP_LEVEL_KEYS)}")
    require(spec["schema_version"] == SCHEMA_VERSION,
            "Manifest schema_version changed")
    require(spec["status"].startswith("FROZEN_"),
            "Manifest must be explicitly frozen")
    require(_is_sha256(spec["runner_sha256"]), "Invalid runner_sha256")
    if check_runner:
        require(spec["runner_sha256"] == sha256_path(Path(__file__).resolve()),
                "Collector runner differs from the frozen manifest")
    require(isinstance(spec["GPU_UUID"], str) and
            spec["GPU_UUID"].startswith("GPU-"),
            "GPU_UUID must be an nvidia-smi GPU UUID")

    path_fields = {
        "trace_path": spec["trace_path"],
        "output_directory": spec["output_directory"],
        "parent_collection_manifest": spec["parent_collection_manifest"],
        "code_root": spec["code_root"],
        "config_path": spec["config_path"],
        "checkpoint_path": spec["checkpoint_path"],
    }
    for field, value in path_fields.items():
        _absolute_path(value, field)

    require(tuple(spec["cache_member_allowlist"]) == CACHE_MEMBER_ALLOWLIST,
            "Parent-cache member allowlist changed")
    require(spec["token_dtype"] in TOKEN_DTYPES,
            "token_dtype must be float16 or float32")

    runtime = spec["runtime"]
    require(isinstance(runtime, dict) and set(runtime) == RUNTIME_KEYS,
            "runtime keys changed")
    _absolute_path(runtime["python"], "runtime.python")
    require(isinstance(runtime["seed"], int) and not isinstance(runtime["seed"], bool),
            "runtime.seed must be an integer")
    require(runtime["threads"] == 1 and not isinstance(runtime["threads"], bool),
            "runtime.threads must be exactly 1")

    pins = spec["input_pins"]
    require(isinstance(pins, dict) and pins, "input_pins must be a nonempty mapping")
    for path_text, digest in pins.items():
        _absolute_path(path_text, "input_pins key")
        require(_is_sha256(digest), f"Invalid SHA-256 pin for {path_text}")
    required_pins = {
        str(Path(__file__).resolve()),
        str(Path(spec["trace_path"])),
        str(Path(spec["parent_collection_manifest"])),
        str(Path(spec["config_path"])),
        str(Path(spec["checkpoint_path"])),
    }
    require(required_pins <= set(pins),
            "input_pins misses required runner/model/parent inputs")
    require(pins[str(Path(__file__).resolve())] == spec["runner_sha256"],
            "input_pins runner digest differs from runner_sha256")

    frames = spec["frames"]
    require(isinstance(frames, list) and frames,
            "frames must be a nonempty list")
    seen_image_ids: set[int] = set()
    video_runs: list[tuple[int, list[int]]] = []
    for index, frame in enumerate(frames):
        require(isinstance(frame, dict) and set(frame) == FRAME_KEYS,
                f"frames[{index}] keys changed")
        source = frame["source"]
        require(isinstance(source, str) and source and source.strip() == source,
                f"frames[{index}].source is empty or padded")
        for field in ("video_id", "image_id", "frame_id"):
            value = frame[field]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f"frames[{index}].{field} must be a nonnegative integer")
        require(frame["image_id"] not in seen_image_ids,
                f"Duplicate image_id: {frame['image_id']}")
        seen_image_ids.add(frame["image_id"])
        for field in ("cache_path", "image_path"):
            _absolute_path(frame[field], f"frames[{index}].{field}")
        for field in ("cache_sha256", "image_sha256"):
            require(_is_sha256(frame[field]),
                    f"frames[{index}].{field} is not lowercase SHA-256")
        if not video_runs or video_runs[-1][0] != frame["video_id"]:
            video_runs.append((frame["video_id"], []))
        video_runs[-1][1].append(frame["frame_id"])

    seen_video_ids: set[int] = set()
    for video_id, ordinals in video_runs:
        require(video_id not in seen_video_ids,
                f"video {video_id} must form exactly one contiguous run")
        seen_video_ids.add(video_id)
        require(ordinals == list(range(len(ordinals))),
                f"video {video_id} must provide contiguous frames from frame_id 0")
    return spec


def load_manifest(path: Path, *, check_runner: bool = True) -> dict[str, Any]:
    path = Path(path)
    require(path.is_absolute(), "Manifest path must be absolute")
    require(path.is_file(), f"Manifest does not exist: {path}")
    return validate_manifest_structure(_json_object(path),
                                       check_runner=check_runner)


def authenticate_manifest(path: Path, spec: Mapping[str, Any],
                          *, check_inputs: bool = True) -> dict[str, Any]:
    """Pin the manifest, every declared input, and frozen model source files."""
    manifest_path = Path(path)
    pre_hash = sha256_path(manifest_path)
    require(pre_hash == sha256_path(manifest_path),
            "Manifest changed while being authenticated")
    if not check_inputs:
        return {"manifest_sha256": pre_hash}
    for path_text, expected in spec["input_pins"].items():
        pinned = Path(path_text)
        require(pinned.is_file(), f"Pinned input is missing: {pinned}")
        require(sha256_path(pinned) == expected,
                f"Pinned input changed: {pinned}")
    parent = _json_object(Path(spec["parent_collection_manifest"]))
    source_hashes = parent.get("source_hashes")
    require(isinstance(source_hashes, dict) and source_hashes,
            "Parent manifest has no frozen source_hashes")
    code_root = Path(spec["code_root"])
    for relative, expected in source_hashes.items():
        require(isinstance(relative, str) and relative and
                _is_sha256(expected),
                "Malformed parent source pin")
        source_path = code_root / relative
        require(source_path.is_file(), f"Frozen model source is missing: {source_path}")
        require(sha256_path(source_path) == expected,
                f"Frozen model source changed: {source_path}")
    for frame in spec["frames"]:
        require(sha256_path(Path(frame["cache_path"])) == frame["cache_sha256"],
                f"Parent cache changed: {frame['cache_path']}")
        require(sha256_path(Path(frame["image_path"])) == frame["image_sha256"],
                f"Image changed: {frame['image_path']}")
    require(pre_hash == sha256_path(manifest_path),
            "Manifest changed during authentication")
    return {
        "manifest_sha256": pre_hash,
        "verified_input_pins": len(spec["input_pins"]),
        "verified_model_sources": len(source_hashes),
        "verified_frames": len(spec["frames"]),
    }


def retained_row_locators(keep_mask: Any, pretracker_det_row: Any,
                          n_pre: int) -> np.ndarray:
    """Return detector-row locators and enforce keep-mask/row-map agreement."""
    keep = np.asarray(keep_mask)
    rows = np.asarray(pretracker_det_row)
    require(keep.dtype == np.dtype("bool"), "keep mask must be boolean")
    require(rows.dtype.kind in "iu", "row locators must be integer")
    require(keep.shape == (int(n_pre),), "keep mask has the wrong row axis")
    require(rows.shape == (int(np.count_nonzero(keep)),),
            "row locator count differs from retained rows")
    expected = np.flatnonzero(keep).astype(np.int64, copy=False)
    require(np.array_equal(rows.astype(np.int64, copy=False), expected),
            "row locator does not equal the keep-mask positions")
    return expected


def validate_spatial_tokens(roi_input: Any, conv_spatial: Any,
                            n_pre: int) -> None:
    """Require retained-row NCHW layout ``[N, 256, 7, 7]``."""
    roi = np.asarray(roi_input)
    conv = np.asarray(conv_spatial)
    expected = (int(n_pre), 256, 7, 7)
    require(roi.shape == expected, "roi_input is not [N,256,7,7]")
    require(conv.shape == expected, "conv_spatial is not [N,256,7,7]")
    require(np.isfinite(roi).all(), "roi_input contains non-finite values")
    require(np.isfinite(conv).all(), "conv_spatial contains non-finite values")


def preprocessing_record(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only deployment geometry, never image metadata or annotations."""
    record: dict[str, Any] = {}
    for field in ("scale_factor", "ori_shape", "img_shape", "pad_shape"):
        require(field in meta, f"Missing preprocessing field: {field}")
        value = np.asarray(meta[field]).tolist()
        require(isinstance(value, list), f"Preprocessing {field} must be a vector")
        record[field] = value
    require(len(record["scale_factor"]) in (2, 4),
            "scale_factor must have 2 or 4 entries")
    for field in ("ori_shape", "img_shape", "pad_shape"):
        require(len(record[field]) >= 2,
                f"{field} must contain height and width")
    return record


def build_frame_arrays(
    *,
    roi_input: Any,
    conv_spatial: Any,
    raw_output: Any,
    keep_mask: Any,
    raw_pretracker_det_row: Any,
    roi_proposal_row: Any,
    det_bboxes: Any,
    pretracker_det_bboxes: Any,
    preprocessing: Mapping[str, Any],
    source: str,
    video_id: int,
    image_id: int,
    frame_id: int,
    image_sha256: str,
    cache_sha256: str,
    token_dtype: str,
) -> dict[str, np.ndarray]:
    """Build one retained-row NPZ payload with an exact member schema."""
    require(token_dtype in TOKEN_DTYPES, "Unknown token dtype")
    roi = np.asarray(roi_input)
    conv = np.asarray(conv_spatial)
    keep = np.asarray(keep_mask)
    require(keep.dtype == np.dtype("bool"),
            "keep mask must be boolean")
    n_pre = len(keep)
    require(roi.shape == conv.shape and
            roi.shape[1:] == (256, 7, 7) and
            roi.shape[0] == int(np.count_nonzero(keep)),
            "spatial tensors must be retained-row [N,256,7,7]")
    require(np.isfinite(roi).all() and np.isfinite(conv).all(),
            "spatial tensors contain non-finite values")
    rows = retained_row_locators(keep, raw_pretracker_det_row, n_pre)
    n = len(rows)
    raw = np.asarray(raw_output)
    require(raw.ndim == 2 and raw.shape == (n, 256),
            "raw_output must be retained-row [N,256]")
    proposals = np.asarray(roi_proposal_row)
    require(proposals.ndim == 1 and proposals.dtype.kind in "iu" and
            proposals.shape[0] == n,
            "roi_proposal_row must provide one integer locator per retained row")
    require(np.all(proposals >= 0), "roi_proposal_row must be nonnegative")
    boxes = np.asarray(det_bboxes)
    pre_boxes = np.asarray(pretracker_det_bboxes)
    require(boxes.ndim == 2 and boxes.shape[0] == n and boxes.shape[1] >= 4,
            "det_bboxes must have one retained row and at least four columns")
    require(pre_boxes.ndim == 2 and pre_boxes.shape[0] == n_pre and
            pre_boxes.shape[1] >= 4,
            "pretracker_det_bboxes must have one detector row")
    require(np.isfinite(boxes).all() and np.isfinite(pre_boxes).all(),
            "Detector boxes contain non-finite values")
    require(np.isfinite(raw).all(), "raw_output contains non-finite values")
    dtype = TOKEN_DTYPES[token_dtype]
    arrays = {
        "roi_input": np.asarray(roi, dtype=dtype),
        "conv_spatial": np.asarray(conv, dtype=dtype),
        "raw_output": raw,
        "pretracker_det_row": rows.astype(np.int64, copy=False),
        "roi_proposal_row": proposals.astype(np.int64, copy=False),
        "det_bboxes": boxes,
        "pretracker_det_bboxes": pre_boxes[keep_mask],
        "preprocessing_json": np.asarray(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":"))),
        "source": np.asarray(source),
        "video_id": np.asarray(video_id, dtype=np.int64),
        "image_id": np.asarray(image_id, dtype=np.int64),
        "frame_id": np.asarray(frame_id, dtype=np.int64),
        "image_sha256": np.asarray(image_sha256),
        "cache_sha256": np.asarray(cache_sha256),
    }
    require(tuple(arrays) == FRAME_NPZ_MEMBERS,
            "Internal NPZ member order changed")
    return arrays


def _atomic_write(stream_writer: Callable[[Any], None], path: Path) -> None:
    path = Path(path)
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream_writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), f"Output appeared during write: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    require(tuple(arrays) == FRAME_NPZ_MEMBERS,
            "Refusing to write a noncanonical frame NPZ")
    _atomic_write(lambda stream: np.savez_compressed(stream, **arrays), path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) +
               "\n").encode("utf-8")
    _atomic_write(lambda stream: stream.write(encoded), path)


def gpu_guard(
    gpu_uuid: str,
    *,
    runner: Callable[..., str] = subprocess.check_output,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Require an allowed A100 and reject every chenhc compute process."""
    environment = os.environ if environment is None else environment
    require(environment.get("CUDA_VISIBLE_DEVICES") == gpu_uuid,
            "CUDA_VISIBLE_DEVICES must exactly equal GPU_UUID")
    device_rows = runner(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    matched = []
    for line in device_rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and parts[0] == gpu_uuid:
            matched.append(parts)
    require(len(matched) == 1, "Selected GPU UUID was not found exactly once")
    _, name, total_mib, free_mib = matched[0]
    total_mib, free_mib = int(total_mib), int(free_mib)
    require("A100" in name, "Selected GPU is not an A100")
    require(total_mib > 4096, "Refusing a 4GB-class GPU")
    require(total_mib >= 39000 and free_mib >= 15000,
            "Selected A100 lacks the required free memory")

    application_rows = runner(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    preexisting = []
    for line in application_rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[0] != gpu_uuid:
            continue
        _, pid_text, used_text = parts
        require(pid_text.isdigit(), "Malformed compute-process PID")
        owner = runner(
            ["ps", "-o", "user=", "-p", pid_text],
            text=True,
        ).strip()
        require(owner and owner != "chenhc",
                f"Selected GPU has a chenhc compute process: pid={pid_text}")
        preexisting.append({
            "pid": int(pid_text),
            "owner": owner,
            "memory_mib": int(used_text),
        })
    return {
        "uuid": gpu_uuid,
        "name": name,
        "total_mib": total_mib,
        "free_mib": free_mib,
        "preexisting_non_chenhc_processes": preexisting,
    }


def _copy_allowed_parent(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        missing = [name for name in CACHE_MEMBER_ALLOWLIST if name not in saved.files]
        require(not missing, f"Parent cache misses allowlisted members: {missing}")
        return {name: saved[name].copy() for name in CACHE_MEMBER_ALLOWLIST}


def _validate_parent_members(expected: Mapping[str, np.ndarray]) -> tuple[np.ndarray, int, int]:
    keep = np.asarray(expected["tracker_keep_mask"])
    pre_rows = np.asarray(expected["pretracker_det_row"])
    boxes = np.asarray(expected["det_bboxes"])
    pre_boxes = np.asarray(expected["pretracker_det_bboxes"])
    raw = np.asarray(expected["track_feats"])
    proposals = np.asarray(expected["roi_proposal_row"])
    require(pre_boxes.ndim == 2 and pre_boxes.shape[1] >= 4,
            "Parent pretracker_det_bboxes layout changed")
    n_pre = len(pre_boxes)
    rows = retained_row_locators(keep, pre_rows, n_pre)
    require(boxes.ndim == 2 and boxes.shape[0] == len(rows),
            "Parent det_bboxes row count differs from keep mask")
    require(raw.ndim == 2 and raw.shape == (len(rows), 256),
            "Parent track_feats layout changed")
    require(proposals.shape == (len(rows),) and proposals.dtype.kind in "iu",
            "Parent roi_proposal_row layout changed")
    require(np.array_equal(np.asarray(pre_boxes[keep, :4]),
                           np.asarray(boxes[:, :4])),
            "Parent retained detector geometry differs from pretracker geometry")
    return rows, n_pre, len(rows)


def _load_trace_module(path: Path):
    spec = importlib.util.spec_from_file_location("geometry_roi_track_trace", path)
    require(spec is not None and spec.loader is not None,
            "Cannot load the frozen track-head trace module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(hasattr(module, "ROITrackSpatialTrace"),
            "Trace module lacks ROITrackSpatialTrace")
    return module


def execute_collection(manifest_path: Path) -> dict[str, Any]:
    """Execute exactly once.  Any failure is raised and exits nonzero."""
    started = time.monotonic()
    manifest_path = Path(manifest_path)
    spec = load_manifest(manifest_path)
    preflight = authenticate_manifest(manifest_path, spec, check_inputs=True)
    require(Path(sys.executable).resolve() == Path(spec["runtime"]["python"]).resolve(),
            "Current Python interpreter differs from runtime.python")
    out = Path(spec["output_directory"])
    require(not out.exists(), "Output directory already exists; never overwrite")
    require(out.parent.is_dir(), "Output directory parent must already exist")
    require(not os.environ.get("OVMOT_TCES_DUMP_DIR"),
            "Original posterior cache writer must be disabled")
    device = gpu_guard(spec["GPU_UUID"])

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, spec["code_root"])
    try:
        import mmcv
        import torch
        import ovtrack
        from mmdet.datasets.builder import PIPELINES
        from ovtrack.apis import init_model, inference_model
    except Exception as exc:
        raise CollectorError(f"Frozen model import failed: {exc}") from exc

    imported = Path(ovtrack.__file__).resolve()
    code_root = Path(spec["code_root"]).resolve()
    require(imported == code_root or code_root in imported.parents,
            "Imported ovtrack is outside code_root")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(spec["runtime"]["seed"])
    np.random.seed(spec["runtime"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    @PIPELINES.register_module(force=True)
    class CollectionFrameStep:
        """Match the frozen collection config's pipeline node name exactly."""

        def __call__(self, data):
            data["frame_id"] = int(data["img_info"]["frame_id"])
            return data

    cfg = mmcv.Config.fromfile(spec["config_path"])
    cfg.model.backbone.init_cfg = None
    require(cfg.model.roi_head.type == "OVTrackRoIHeadUncertainty",
            "Unexpected active ROI-head type")
    require(not cfg.model.roi_head.init_track_head_by_bbox_head,
            "Track head must not mutate its initial weights in inference")
    require(not cfg.model.tracker.vis, "Tracker visualization/GT path must be disabled")

    out.mkdir(parents=False, exist_ok=False)
    atomic_write_json(out / "started.json", {
        "status": RUNNING_STATUS,
        "pid": os.getpid(),
        "started_unix": time.time(),
        "manifest_sha256": preflight["manifest_sha256"],
        "runner_sha256": spec["runner_sha256"],
        "device": device,
        "training": False,
        "automatic_retry": False,
        "access_scope": "only cache_member_allowlist; no GT/identity/class/posterior/matching",
    })

    model = init_model(cfg, spec["checkpoint_path"], device="cuda:0")
    model.requires_grad_(False)
    model.eval()
    # Frozen collection config requires an explicit initial tracker.  The
    # first selected frame resets it through the normal frame_id==0 path.
    model.init_tracker()
    require(getattr(model, "tracker", None) is not None,
            "Explicit tracker initialization failed")
    trace_module = _load_trace_module(Path(spec["trace_path"]))
    head = model.roi_head.track_head
    require(head.num_convs == 4 and head.num_fcs == 1,
            "Frozen track-head structure changed")
    trace = trace_module.ROITrackSpatialTrace(head)
    original_track_forward = model.roi_head._track_forward
    original_simple_test = model.simple_test
    observed_boxes: list[list[Any]] = []
    observed_meta: list[dict[str, Any]] = []

    def track_forward(x, boxes):
        observed_boxes.append([box.detach().clone() for box in boxes])
        return original_track_forward(x, boxes)

    def simple_test(img, img_metas, rescale=False):
        require(rescale is True and len(img_metas) == 1,
                "Expected one rescaled image per geometry-ROI frame")
        observed_meta.append(preprocessing_record(img_metas[0]))
        return original_simple_test(img, img_metas, rescale=rescale)

    model.roi_head._track_forward = track_forward
    model.simple_test = simple_test
    frame_receipts: list[dict[str, Any]] = []
    try:
        for ordinal, frame in enumerate(spec["frames"], start=1):
            expected = _copy_allowed_parent(Path(frame["cache_path"]))
            parent_rows, n_pre, n = _validate_parent_members(expected)
            require(sha256_path(Path(frame["cache_path"])) == frame["cache_sha256"],
                    "Parent cache changed immediately before inference")
            require(sha256_path(Path(frame["image_path"])) == frame["image_sha256"],
                    "Image changed immediately before inference")
            model.tracker.keep_mask = None
            previous_tracker = getattr(model, "tracker", None)
            observed_boxes.clear()
            observed_meta.clear()
            if n_pre:
                trace.begin()
            try:
                inference_model(model, frame["image_path"], frame["frame_id"])
                torch.cuda.synchronize()
            except Exception as exc:
                raise CollectorError(
                    f"Frozen inference failed at frame {frame['video_id']}:"
                    f"{frame['frame_id']}: {type(exc).__name__}: {exc}") from exc
            require(len(observed_meta) == 1,
                    "Track-head preprocessing hook did not run exactly once")
            require((model.tracker is not previous_tracker) ==
                    (frame["frame_id"] == 0),
                    "Tracker reset differs from manifest timeline")
            expected_keep = np.asarray(expected["tracker_keep_mask"], dtype=bool)
            actual_keep = (np.zeros(0, dtype=bool)
                           if getattr(model.tracker, "keep_mask", None) is None
                           else model.tracker.keep_mask.detach().cpu().numpy())
            require(actual_keep.dtype == np.dtype("bool") and
                    np.array_equal(actual_keep, expected_keep),
                    "Tracker keep mask differs from parent cache")

            if n_pre == 0:
                require(not observed_boxes,
                        "Track head ran on an empty detector frame")
                empty_spatial = np.zeros((0, 256, 7, 7), dtype=np.float32)
                empty_raw = np.zeros((0, 256), dtype=np.float32)
                arrays = build_frame_arrays(
                    roi_input=empty_spatial,
                    conv_spatial=empty_spatial,
                    raw_output=empty_raw,
                    keep_mask=expected_keep,
                    raw_pretracker_det_row=parent_rows,
                    roi_proposal_row=np.asarray(expected["roi_proposal_row"]),
                    det_bboxes=np.asarray(expected["det_bboxes"]),
                    pretracker_det_bboxes=np.asarray(
                        expected["pretracker_det_bboxes"]),
                    preprocessing=observed_meta[0],
                    source=frame["source"],
                    video_id=frame["video_id"],
                    image_id=frame["image_id"],
                    frame_id=frame["frame_id"],
                    image_sha256=frame["image_sha256"],
                    cache_sha256=frame["cache_sha256"],
                    token_dtype=spec["token_dtype"],
                )
                captured = None
            else:
                require(len(observed_boxes) == 1 and
                        len(observed_boxes[0]) == 1,
                        "Unexpected track ROI call count")
                actual_boxes = observed_boxes[0][0]
                scale = torch.as_tensor(
                    observed_meta[0]["scale_factor"],
                    device=actual_boxes.device, dtype=actual_boxes.dtype)
                target_boxes = torch.as_tensor(
                    np.asarray(expected["pretracker_det_bboxes"])[:, :4],
                    device=actual_boxes.device, dtype=actual_boxes.dtype) * scale
                require(torch.equal(actual_boxes, target_boxes),
                        "Track ROI geometry/order differs from parent cache")
                captured = trace.take(expected_keep)
                traced_rows = captured["pretracker_det_row"].cpu().numpy()
                require(np.array_equal(traced_rows, parent_rows),
                        "Trace row locators differ from parent cache")
                full_raw = captured["full_raw_output"].detach().cpu().numpy()
                parent_raw = np.asarray(expected["track_feats"])
                require(np.array_equal(
                    full_raw.astype(parent_raw.dtype, copy=False)[expected_keep],
                    parent_raw),
                    "Native track-head output differs from parent cache")
                arrays = build_frame_arrays(
                    roi_input=captured["roi_input"].detach().cpu().numpy(),
                    conv_spatial=captured["conv_spatial"].detach().cpu().numpy(),
                    # The parent cache stores float16 immediately after the
                    # native head output. Compare the same dtype, rather than
                    # requiring float32 arithmetic to be bitwise reproduced.
                    raw_output=full_raw.astype(parent_raw.dtype, copy=False)[
                        expected_keep],
                    keep_mask=expected_keep,
                    raw_pretracker_det_row=parent_rows,
                    roi_proposal_row=np.asarray(expected["roi_proposal_row"]),
                    det_bboxes=np.asarray(expected["det_bboxes"]),
                    pretracker_det_bboxes=np.asarray(
                        expected["pretracker_det_bboxes"]),
                    preprocessing=observed_meta[0],
                    source=frame["source"],
                    video_id=frame["video_id"],
                    image_id=frame["image_id"],
                    frame_id=frame["frame_id"],
                    image_sha256=frame["image_sha256"],
                    cache_sha256=frame["cache_sha256"],
                    token_dtype=spec["token_dtype"],
                )
                require(np.array_equal(arrays["raw_output"],
                                       parent_raw),
                        "Retained raw output differs from parent cache")

            target = out / (f"v{frame['video_id']}_i{frame['image_id']}"
                            f"_f{frame['frame_id']}.npz")
            atomic_write_npz(target, arrays)
            with np.load(target, allow_pickle=False) as saved:
                require(tuple(saved.files) == FRAME_NPZ_MEMBERS,
                        "Saved frame schema changed")
                require(saved["roi_input"].shape == (n, 256, 7, 7),
                        "Saved roi_input axes changed")
                require(saved["conv_spatial"].shape == (n, 256, 7, 7),
                        "Saved conv_spatial axes changed")
            receipt = {
                "ordinal": ordinal,
                "source": frame["source"],
                "video_id": frame["video_id"],
                "image_id": frame["image_id"],
                "frame_id": frame["frame_id"],
                "detector_observation_count": n_pre,
                "retained_rows": n,
                "roi_input_shape": list(arrays["roi_input"].shape),
                "conv_spatial_shape": list(arrays["conv_spatial"].shape),
                "token_dtype": spec["token_dtype"],
                "native_raw_dtype": str(arrays["raw_output"].dtype),
                "row_locator_exact": True,
                "parent_raw_parity_exact": True,
                "preprocessing": observed_meta[0],
                "file": target.name,
                "sha256": sha256_path(target),
            }
            frame_receipts.append(receipt)
            print(json.dumps(receipt, ensure_ascii=False), flush=True)
    finally:
        trace.close()
        model.roi_head._track_forward = original_track_forward
        model.simple_test = original_simple_test

    postflight = authenticate_manifest(manifest_path, spec, check_inputs=True)
    require(postflight["manifest_sha256"] == preflight["manifest_sha256"],
            "Manifest changed during collection")
    result = {
        "status": COMPLETE_STATUS,
        "manifest_sha256": preflight["manifest_sha256"],
        "runner_sha256": spec["runner_sha256"],
        "device": device,
        "frames": frame_receipts,
        "frame_count": len(frame_receipts),
        "retained_rows": sum(row["retained_rows"] for row in frame_receipts),
        "elapsed_seconds": time.monotonic() - started,
        "training": False,
        "automatic_retry": False,
        "saved_scope": "retained rows only",
        "access_scope": "only cache_member_allowlist; no GT/identity/class/posterior/matching",
    }
    atomic_write_json(out / "collection_receipt.json", result)
    return result


def run(manifest_path: Path, *, execute: bool) -> dict[str, Any]:
    spec = load_manifest(Path(manifest_path))
    if not execute:
        return {
            "status": "PASS_GEOMETRY_ROI_MANIFEST_PREFLIGHT_ONLY",
            "manifest_sha256": sha256_path(Path(manifest_path)),
            "runner_sha256": spec["runner_sha256"],
            "frame_count": len(spec["frames"]),
            "output_directory_exists": Path(spec["output_directory"]).exists(),
            "GPU_or_model_touched": False,
            "execute_requires_explicit_flag": True,
        }
    return execute_collection(Path(manifest_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true",
        help="Run the frozen GPU collector. Omit for manifest-only preflight.")
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments.manifest, execute=arguments.execute)
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL_GEOMETRY_ROI_TOKEN_COLLECTION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "automatic_retry": False,
        }, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
