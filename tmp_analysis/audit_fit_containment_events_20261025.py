#!/usr/bin/env python3
"""Budget-only audit of GT-free FIT parent-child containment events.

Reads only detector boxes, labels and released IDs from the sealed FIT cache.
No GT class, identity, correctness or future frame is used. The output is a
sampling frame for a later visual-mask pilot, not a causal result.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/chenhc/ov/OVMOT_RESEARCH")
CACHE = ROOT / "results/base_error_full_collection_20261023/full_collect_v1"
OUT = ROOT / "results/fit_containment_event_budget_20261025_v4"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ioc(parent, child):
    intersection = max(
        0.0, min(parent[2], child[2]) - max(parent[0], child[0])
    ) * max(0.0, min(parent[3], child[3]) - max(parent[1], child[1]))
    child_area = (child[2] - child[0]) * (child[3] - child[1])
    if child_area <= 0:
        return 0.0
    centre_x = (child[0] + child[2]) / 2
    centre_y = (child[1] + child[3]) / 2
    inside = (
        parent[0] <= centre_x <= parent[2]
        and parent[1] <= centre_y <= parent[3]
    )
    return intersection / child_area * float(inside)


def main():
    if OUT.exists():
        raise FileExistsError(str(OUT))
    receipts = [
        json.loads(line)
        for line in (CACHE / "frame_execution.jsonl").open()
    ]
    assert len(receipts) == 15283
    video_source = {}
    with (ROOT / "results/base_error_full_collection_20261023/support_v1/gt_free_observations.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            existing = video_source.get(int(row["video_id"]))
            if existing is not None:
                assert existing == row["source_group"]
            video_source[int(row["video_id"])] = str(row["source_group"])
    assert len(video_source) == 448, len(video_source)
    plan = json.loads(
        (ROOT / "results/base_error_full_collection_20261023/full_collection_plan.json").read_text()
    )
    fit_videos = {
        int(row["video_id"])
        for row in plan["video_records"]
        if row["split"] == "fit"
    }
    assert len(fit_videos) == 405
    receipts = [
        row for row in receipts if int(row["video_id"]) in fit_videos
    ]
    assert len(receipts) == 13791
    for receipt in receipts:
        if int(receipt["video_id"]) not in video_source:
            raise KeyError(
                f"source map lacks video {receipt['video_id']} from {receipt['npz']}; "
                f"loaded videos={len(video_source)}"
            )
    edge_area_bin = Counter()
    edges = []
    for receipt in receipts:
        with np.load(receipt["npz"], allow_pickle=False) as data:
            boxes = data["det_bboxes"][:, :4].astype(np.float64)
            labels = data["det_labels"].astype(np.int64)
            ids = data["track_ids"].astype(np.int64)
            emitted = data["emitted_track_mask"].astype(bool)
        valid = emitted & (ids >= 0)
        indices = np.flatnonzero(valid)
        for parent_row in indices:
            parent_box = boxes[parent_row]
            parent_area = (
                parent_box[2] - parent_box[0]
            ) * (parent_box[3] - parent_box[1])
            for child_row in indices:
                if parent_row == child_row:
                    continue
                child_box = boxes[child_row]
                child_area = (
                    child_box[2] - child_box[0]
                ) * (child_box[3] - child_box[1])
                if parent_area <= 0 or child_area <= 0:
                    continue
                ratio = child_area / parent_area
                if ratio < 0.01 or ratio > 0.8:
                    continue
                overlap = ioc(parent_box, child_box)
                if overlap <= 0.8:
                    continue
                # The edge is anchored to two different released tracks in the
                # same frame. Repeated observations are not yet independent
                # events; track-level dedup happens before any pilot sampling.
                key = (
                    int(receipt["video_id"]),
                    int(receipt["image_id"]),
                    int(ids[parent_row]),
                    int(ids[child_row]),
                )
                edges.append(
                    dict(
                        key=list(key),
                        frame_index=int(receipt["frame_index"]),
                        parent_pred=int(labels[parent_row]) + 1,
                        child_pred=int(labels[child_row]) + 1,
                        ioc=float(overlap),
                        area_ratio=float(ratio),
                        parent_box=[float(v) for v in parent_box],
                        child_box=[float(v) for v in child_box],
                        npz=receipt["npz"],
                        source_group=video_source[int(receipt["video_id"])],
                    )
                )
                bin_index = (
                    "lt10pct"
                    if ratio < 0.1
                    else "10to30pct"
                    if ratio < 0.3
                    else "30to60pct"
                    if ratio < 0.6
                    else "60to80pct"
                )
                edge_area_bin[bin_index] += 1
    tracks = defaultdict(set)
    for edge in edges:
        parent_track = (edge["key"][0], edge["key"][2])
        tracks[parent_track].add(edge["key"][3])
    # Deduplicate one representative frame per (parent track, child track).
    # Repeated observations are not independent events.
    dedup = {}
    for edge in edges:
        key = (edge["key"][0], edge["key"][2], edge["key"][3])
        if key not in dedup or edge["frame_index"] < dedup[key]["frame_index"]:
            dedup[key] = edge
    unique_edges = sorted(dedup.values(), key=lambda e: (
        e["key"][0], e["frame_index"], e["key"][2], e["key"][3]
    ))
    unique_sources = defaultdict(int)
    for edge in unique_edges:
        unique_sources[edge["source_group"]] += 1
    prominent_sources = {
        source: count
        for source, count in sorted(
            unique_sources.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    }
    OUT.mkdir()
    (OUT / "GT_free_edges.jsonl").write_text(
        "".join(json.dumps(edge, separators=(",", ":")) + "\n" for edge in unique_edges)
    )
    summary = {
        "status": "GT_FREE_FIT_CONTAINMENT_BUDGET_ONLY",
        "frames": len(receipts),
        "edges": len(edges),
        "unique_parent_child_edges": len(unique_edges),
        "unique_parent_frames": len({(e["key"][0], e["key"][1], e["key"][2]) for e in edges}),
        "unique_parent_tracks": len(tracks),
        "unique_child_tracks": len({(e["key"][0], e["key"][3]) for e in edges}),
        "area_ratio_bins": dict(edge_area_bin),
        "unique_sources": len(unique_sources),
        "sources_with_edges": sum(bool(v) for v in unique_sources.values()),
        "top20_sources_by_unique_edges": prominent_sources,
        "top_source_share_of_unique_edges": (
            max(unique_sources.values()) / len(unique_edges)
            if unique_edges else None
        ),
        "source_mapping_sha256": sha(
            ROOT / "results/base_error_full_collection_20261023/support_v1/gt_free_observations.jsonl"
        ),
        "gt_used": False,
        "training": False,
        "GPU": False,
        "interpretation": (
            "Candidate relationship budget only. Repeated track observations "
            "are not independent events. No claim about parent class error, "
            "occlusion causality or tracking performance."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = {
        "status": "PASS_GT_FREE_BUDGET_AUDIT",
        "script_sha256": sha(__file__),
        "frame_execution_sha256": sha(CACHE / "frame_execution.jsonl"),
        "edges_jsonl_sha256": sha(OUT / "GT_free_edges.jsonl"),
        "summary_sha256": sha(OUT / "summary.json"),
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
