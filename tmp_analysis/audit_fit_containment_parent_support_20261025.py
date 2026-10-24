#!/usr/bin/env python3
"""Describe parent-track support of the sealed FIT containment graph.

The edge file already deduplicates observations. This audit reports how many
distinct parent trajectories and class contexts the future pilot can sample,
without reading correctness or selecting candidates by outcomes.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/chenhc/ov/OVMOT_RESEARCH')
EDGES = ROOT / 'results/fit_containment_event_budget_20261025_v4/GT_free_edges.jsonl'
OUT = ROOT / 'results/fit_containment_parent_support_20261025_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if OUT.exists():
    raise FileExistsError(str(OUT))
parents = defaultdict(set)
children = set()
sources = defaultdict(set)
windows = defaultdict(set)
parent_child_classes = Counter()
parent_classes = Counter()
parent_rows = 0
for line in EDGES.open():
    edge = json.loads(line)
    video_id, image_id, parent_id, child_id = edge['key']
    parent = (video_id, parent_id)
    parents[parent].add(child_id)
    children.add((video_id, child_id))
    sources[edge['source_group']].add(parent)
    windows[parent].add(int(edge['frame_index']))
    parent_child_classes[(edge['parent_pred'], edge['child_pred'])] += 1
    parent_classes[edge['parent_pred']] += 1
    parent_rows += 1
OUT.mkdir()
summary = {
    'status': 'FIT_CONTAINMENT_PARENT_SUPPORT_DESCRIPTION',
    'edges': parent_rows,
    'unique_parent_track_contexts': len(parents),
    'unique_child_track_contexts': len(children),
    'sources': len(sources),
    'parent_child_class_pairs': len(parent_child_classes),
    'parent_classes': len(parent_classes),
    'top_parent_child_pairs': {
        str(k): v for k, v in parent_child_classes.most_common(20)
    },
    'source_parent_support': {
        'min': min(map(len, sources.values())),
        'median': sorted(map(len, sources.values()))[len(sources) // 2],
        'max': max(map(len, sources.values())),
        'sources_at_least_5_parents': sum(len(v) >= 5 for v in sources.values()),
        'largest_source': max(sources, key=lambda s: len(sources[s])),
    },
    'observation_frames_per_parent': {
        'min': min(map(len, windows.values())),
        'median': sorted(map(len, windows.values()))[len(windows) // 2],
        'max': max(map(len, windows.values())),
    },
    'unknown_not_negative': True,
    'gt_used': False,
    'training': False,
    'GPU': False,
    'interpretation': (
        'Sampling support only. A parent track with multiple child classes '
        'must not yield duplicate independent events; the same parent is one '
        'cluster. This does not show class error, occlusion causality or '
        'tracking performance.'
    ),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
receipt = {
    'status': 'PASS_FIT_CONTAINMENT_PARENT_SUPPORT_DESCRIPTION',
    'edges_sha256': sha(EDGES),
    'script_sha256': sha(__file__),
    'training': False,
    'GPU': False,
}
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(summary, indent=2))
