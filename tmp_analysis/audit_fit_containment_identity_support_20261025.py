#!/usr/bin/env python3
"""Describe identity support of the already sealed FIT containment edges.

No training, model choice or threshold selection. GT identities are joined
after the GT-free edge list exists. Unknown is not a negative pair.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/chenhc/ov/OVMOT_RESEARCH")
SUPPORT = ROOT / "results/base_error_full_collection_20261023/support_v1"
EDGES = ROOT / "results/fit_containment_event_budget_20261025_v4/GT_free_edges.jsonl"
OUT = ROOT / "results/fit_containment_identity_support_20261025_v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if OUT.exists():
    raise FileExistsError(str(OUT))

observations = {}
for line in (SUPPORT / "gt_free_observations.jsonl").open():
    row = json.loads(line)
    if row["split"] == "fit":
        observation_id = int(row["observation_id"])
        assert observation_id not in observations
        observations[observation_id] = row

identity_rows = {}
for line in (SUPPORT / "supervised_observations.jsonl").open():
    row = json.loads(line)
    if row["split"] == "fit" and row["eligible_for_parameter_fit"] is True:
        identity_rows[int(row["observation_id"])] = row

track_identities = defaultdict(set)
for observation_id, truth in identity_rows.items():
    observation = observations.get(observation_id)
    if observation is None:
        continue
    track = (int(observation["video_id"]), int(observation["local_track_id"]))
    track_identities[track].add(int(truth["gt_track_id"]))

status = Counter()
pair = Counter()
source_status = defaultdict(Counter)
parents = set()
children = set()
sources = set()
for line in EDGES.open():
    edge = json.loads(line)
    video_id, _, parent_track_id, child_track_id = edge["key"]
    parent = (video_id, parent_track_id)
    child = (video_id, child_track_id)
    parents.add(parent)
    children.add(child)
    sources.add(edge["source_group"])
    parent_gt = track_identities.get(parent, set())
    child_gt = track_identities.get(child, set())
    if parent_gt and child_gt:
        label = "both_known"
        pair["same_gt" if parent_gt & child_gt else "different_gt"] += 1
    elif parent_gt:
        label = "parent_known"
    elif child_gt:
        label = "child_known"
    else:
        label = "unknown"
    status[label] += 1
    source_status[edge["source_group"]][label] += 1

known_sources = {
    source: counts
    for source, counts in source_status.items()
    if counts["both_known"] or counts["parent_known"]
}
OUT.mkdir()
summary = {
    "status": "FIT_CONTAINMENT_IDENTITY_SUPPORT_DESCRIPTION",
    "edges": sum(status.values()),
    "unique_parent_tracks": len(parents),
    "unique_child_tracks": len(children),
    "sources": len(sources),
    "unknown_not_negative": True,
    "identity_status": dict(status),
    "known_endpoint_pair_identity": dict(pair),
    "sources_with_any_known_endpoint": len(known_sources),
    "interpretation": (
        "A relation is not an independent sample. Both-known endpoints are "
        "manual anchors, not full-track purity. Same/different GT labels are "
        "for later scoring, never for selecting parent, child, relation, "
        "memory or thresholds."
    ),
}
receipt = {
    "status": "PASS_DESCRIPTIVE_SUPPORT_AUDIT",
    "edges_sha256": sha(EDGES),
    "gt_free_observations_sha256": sha(SUPPORT / "gt_free_observations.jsonl"),
    "supervised_observations_sha256": sha(SUPPORT / "supervised_observations.jsonl"),
    "script_sha256": sha(__file__),
    "training": False,
    "GPU": False,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(summary, indent=2))
