#!/usr/bin/env python3
"""Read-only resource preflight for the streaming full-candidate contract.

The command reads sealed metadata and file hashes only.  It does not open any
NPZ payload, run the residual module, create the planned output directory, or
start a worker.  A disk or runtime planning failure is a useful result: it
must be resolved before a full FIT launch is authorized.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begun = __import__("time").monotonic()
    protocol = read(args.protocol)
    require(protocol["status"] ==
            "DRAFT_READONLY_METADATA_STREAMING_FULL_CANDIDATE_SCENE_RESIDUAL_V1",
            "Unexpected draft protocol status")
    require(sys.executable == protocol["runtime"]["python"], "Wrong Python runtime")
    require(protocol["runtime"]["CPU_only"] is True, "CPU-only preflight required")
    require(protocol["runtime"]["CUDA_VISIBLE_DEVICES"] == "", "CUDA must be disabled")
    require(not Path(protocol["streaming_and_recovery"]["output_directory"]).exists(),
            "Planned output directory already exists; no overwrite/resume from an unknown run")
    require(not Path(protocol["streaming_and_recovery"]["execution_directory"]).exists(),
            "Planned execution directory already exists; no overwrite/resume from an unknown run")

    pin_checks = {}
    for key, path_key in (
        ("plan_sha256", "plan_path"),
        ("frame_execution_sha256", "frame_execution_path"),
        ("inventory_sha256", "inventory_path"),
        ("scene_residual_module_sha256", "scene_residual_module_path"),
        ("cached_motion_sha256", "cached_motion_path"),
        ("pilot_audit_sha256", "pilot_audit_path"),
    ):
        path = Path(protocol["inputs"][path_key])
        actual = sha(path)
        require(actual == protocol["inputs"][key], "Input pin changed: " + str(path))
        pin_checks[str(path)] = {"sha256": actual, "size_bytes": path.stat().st_size}

    plan = read(protocol["inputs"]["plan_path"])
    require(plan["status"] == "SEALED_FULL_FIT_TUNE_METADATA_ONLY_NO_COLLECTION_RUN",
            "Full collection plan is not sealed metadata-only")
    fit_records = [r for r in plan["video_records"] if r.get("split") == "fit"]
    selected = {int(r["video_id"]) for r in fit_records
                if int(r.get("manual_anchor_images", 0)) > 0}
    require(len(selected) == protocol["population"]["videos_expected"],
            "Unexpected selected FIT video count")
    require(len({r["source_group"] for r in fit_records if int(r["video_id"]) in selected}) ==
            protocol["population"]["source_groups_expected"], "Unexpected source count")

    # frame_execution is metadata only.  In particular, npz paths and hashes
    # are recorded, but their payload bytes are never opened here.
    frames = defaultdict(dict)
    line_count = 0
    with Path(protocol["inputs"]["frame_execution_path"]).open() as stream:
        for raw in stream:
            line_count += 1
            record = json.loads(raw)
            video = int(record["video_id"])
            if video not in selected:
                continue
            frame = int(record["frame_id"])
            require(frame not in frames[video], "Duplicate selected frame locator")
            observations = int(record["observations"])
            require(observations >= 0 and record["emitted_observations"] == observations,
                    "Observation/emitted count mismatch")
            require(Path(record["npz"]).is_absolute() and len(record["npz_sha256"]) == 64,
                    "Missing raw-cache locator/hash")
            frames[video][frame] = dict(
                video_id=video, image_id=int(record["image_id"]), frame_id=frame,
                frame_index=int(record["frame_index"]), observations=observations,
                npz=record["npz"], npz_sha256=record["npz_sha256"])
    require(sum(len(v) for v in frames.values()) == protocol["population"]["frames_expected"],
            "Unexpected selected frame count")
    require(len(frames) == len(selected), "A selected FIT video has no frames")

    # The sealed collection is sampled contiguously by frame_id.  Verify that
    # fact before counting lag pairs; no pair is inferred across a gap.
    for video, records in frames.items():
        ids = sorted(records)
        require(ids == list(range(len(ids))), "Selected frame_id sequence has a gap")

    lag_stats = {}
    total_candidates = 0
    total_pairs = 0
    total_similarity_entries = 0
    maximum_chunk = 0
    maximum_chunk_locator = None
    by_video = {}
    for lag in protocol["population"]["lags"]:
        pair_count = candidate_count = similarity_entries = 0
        per_video = {}
        for video in sorted(frames):
            records = frames[video]
            per_video[video] = {"frame_pairs": 0, "candidates": 0,
                                "similarity_entries": 0}
            for query_frame in sorted(records):
                reference_frame = query_frame - int(lag)
                if reference_frame not in records:
                    continue
                q, m = records[query_frame]["observations"], records[reference_frame]["observations"]
                candidates = q * m
                pair_count += 1
                candidate_count += candidates
                similarity_entries += candidates * candidates
                per_video[video]["frame_pairs"] += 1
                per_video[video]["candidates"] += candidates
                per_video[video]["similarity_entries"] += candidates * candidates
                if candidates > maximum_chunk:
                    maximum_chunk = candidates
                    maximum_chunk_locator = dict(video_id=video, lag=int(lag),
                                                query_frame_id=query_frame,
                                                reference_frame_id=reference_frame,
                                                query_rows=q, reference_rows=m)
        expected_pairs = int(protocol["population"]["expected_frame_pairs"][str(lag)])
        expected_candidates = int(protocol["population"]["expected_candidates"][str(lag)])
        require(pair_count == expected_pairs, "Lag pair count differs from frozen protocol")
        require(candidate_count == expected_candidates, "Lag candidate count differs from frozen protocol")
        lag_stats[str(lag)] = dict(frame_pairs=pair_count, candidates=candidate_count,
                                   similarity_entries=similarity_entries,
                                   per_video=per_video)
        total_pairs += pair_count
        total_candidates += candidate_count
        total_similarity_entries += similarity_entries
    require(total_candidates == int(protocol["population"]["expected_candidates"]["total"]),
            "Total candidate count differs from frozen protocol")
    require(maximum_chunk == protocol["population"]["maximum_candidates_per_chunk"],
            "Maximum chunk differs from frozen protocol")

    # Pilot scaling: one arm, the pinned pilot's measured similarity-entry
    # work, and an explicit factor for three arms plus a planning margin.  This
    # is an estimate, never a completion claim or a runtime reservation.
    pilot = read(protocol["inputs"]["pilot_audit_path"])
    require(pilot["status"] == "PASS_INDEPENDENT_FOUR_FRAME_ALL_CANDIDATE_COMPLETION_AUDIT",
            "Pilot completion audit is not PASS")
    pilot_entries = sum(int((p["shape"][0] * p["shape"][1]) ** 2) for p in pilot["pairs"])
    require(pilot_entries == protocol["resource_planning"]["pilot_similarity_entry_work"],
            "Pilot operation count differs from frozen protocol")
    pilot_elapsed = sum(float(p["compute_seconds"]) for p in pilot["pairs"])
    require(abs(pilot_elapsed - protocol["resource_planning"]["pilot_elapsed_seconds"]) < 1e-12,
            "Pilot timing differs from frozen protocol")
    seconds_per_entry = pilot_elapsed / pilot_entries
    arm_factor = int(protocol["resource_planning"]["arms_to_project"])
    raw_estimate = total_similarity_entries * seconds_per_entry * arm_factor
    planning_estimate = raw_estimate * float(protocol["resource_planning"]["planning_safety_factor"])

    # Mandatory edge schema is deliberately fixed-width.  NPZ headers and
    # manifest lines get a conservative reserve, while two explicit safety
    # reserves are reported separately so nobody mistakes them for a disk
    # reservation.
    per_candidate = int(protocol["edge_output"]["edge_bytes_uncompressed"]["total_per_candidate"])
    edge_bytes = total_candidates * per_candidate
    chunk_header_reserve = total_pairs * 8192
    output_overhead = int(protocol["resource_planning"]["output_overhead_reserve_bytes"])
    extra_reserve = int(protocol["resource_planning"]["extra_safety_reserve_bytes"])
    planning_bytes = edge_bytes + chunk_header_reserve + output_overhead
    free_bytes = shutil.disk_usage(Path(protocol["streaming_and_recovery"]["output_directory"]).parent).free
    minimum_free = int(protocol["resource_planning"]["minimum_free_disk_bytes"])
    after_planned = free_bytes - planning_bytes - extra_reserve
    disk_gate = after_planned >= minimum_free
    # A full candidate run is deliberately held if the planning estimate is
    # longer than a workday.  This is a policy signal, not a failed benchmark.
    runtime_gate = planning_estimate <= 12 * 3600
    status = "PASS_METADATA_STREAMING_RESOURCE_PREFLIGHT"
    if not disk_gate or not runtime_gate:
        status = "HOLD_METADATA_STREAMING_RESOURCE_PREFLIGHT"
    result = dict(
        status=status,
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(args.protocol),
        preflight_sha256=sha(__file__),
        input_pins=pin_checks,
        line_count_total=line_count,
        selected_fit_videos=len(selected),
        selected_fit_frames=sum(len(v) for v in frames.values()),
        lag_stats=lag_stats,
        totals=dict(frame_pairs=total_pairs, candidates=total_candidates,
                    similarity_entries=total_similarity_entries,
                    maximum_chunk_candidates=maximum_chunk,
                    maximum_chunk_locator=maximum_chunk_locator),
        runtime_estimate=dict(
            pilot_elapsed_seconds=pilot_elapsed,
            pilot_similarity_entries=pilot_entries,
            seconds_per_similarity_entry=seconds_per_entry,
            projected_arms=arm_factor,
            raw_projected_seconds=raw_estimate,
            planning_safety_factor=float(protocol["resource_planning"]["planning_safety_factor"]),
            planning_projected_seconds=planning_estimate,
            planning_projected_hours=planning_estimate / 3600.0,
            runtime_gate_12h=runtime_gate,
            estimate_is_not_runtime_measurement=True),
        storage_estimate=dict(
            edge_bytes_uncompressed=edge_bytes,
            edge_bytes_uncompressed_gib=edge_bytes / 2**30,
            chunk_header_reserve_bytes=chunk_header_reserve,
            output_overhead_reserve_bytes=output_overhead,
            planning_output_bytes=planning_bytes,
            extra_safety_reserve_bytes=extra_reserve,
            free_bytes_at_snapshot=free_bytes,
            free_bytes_after_planning_and_extra_reserve=after_planned,
            minimum_free_bytes_after_run=minimum_free,
            disk_gate= disk_gate,
            estimates_are_not_disk_reservations=True),
        opened_npz_payloads=False,
        launched_worker=False,
        output_directory_created=False,
        labels_identity_gt_decoded=False,
        no_results_or_tracking_evaluation=True,
        interpretation=("Metadata-only resource estimate. HOLD means the full FIT worker must not be launched "
                        "until the disk/runtime gate is re-evaluated after an explicit resource decision."))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_pins", "lag_stats")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
