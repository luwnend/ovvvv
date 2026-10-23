"""CPU-only synthetic path audit; reads source, never real cache/model/GT."""
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp_analysis/covtrack_feature_memory_path_independent_review_20261024.json"
FILES = [
    "repos/COVTrack/ovtrack/models/trackers/ovtracker.py",
    "repos/COVTrack/ovtrack/models/roi_heads/ovtrack_roi_head.py",
    "repos/COVTrack/ovtrack/models/roi_heads/track_heads/quasi_dense_embed_head.py",
    "repos/COVTrack/ovtrack/models/losses/cyc_loss.py",
    "work/covtrack_base_error_20261023/ovtrack/models/trackers/ovtracker.py",
    "work/covtrack_base_error_20261023/ovtrack/base_error_dump.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def bbox_overlaps(a, b):
    lo = torch.maximum(a[:, None, :2], b[None, :, :2])
    hi = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (hi - lo).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-12)


def main():
    torch.set_num_threads(1)
    before = {p: digest(ROOT / p) for p in FILES}
    namespace = dict(torch=torch, F=F, f=F, np=np, os=os,
                     defaultdict=defaultdict, bbox_overlaps=bbox_overlaps)
    namespace["nn"] = torch.nn
    similarity = load_definition(ROOT / "repos/COVTrack/ovtrack/core/track/similarity.py", "cal_similarity", namespace)
    Tracker = load_definition(ROOT / FILES[0], "OVTrackerUncertainty", namespace)
    Cycle = load_definition(ROOT / FILES[3], "CycleLoss", namespace)

    class RecordCycle:
        def __init__(self):
            self.inner, self.calls = Cycle(), []

        def get_pair_consistency(self, values):
            self.calls.append(tuple(v.detach().clone() for v in values))
            return self.inner.get_pair_consistency(values)

    class AddFusion:
        """Observable deterministic residual to audit tensor plumbing only."""
        def __init__(self):
            self.calls = []

        def __call__(self, appearance, boxes, semantic, *confidence):
            self.calls.append(appearance.detach().clone())
            return appearance + appearance.new_tensor([[.25, .1], [-.1, .3]])

    tracker = Tracker(momentum_embed=.4, match_score_thr=.1,
                      confused_features=True, vis=False)
    fusion, cycle = AddFusion(), RecordCycle()
    tracker.set_fusion_head(fusion, cycle)
    boxes = torch.tensor([[0., 0., 10., 10., .9], [30., 0., 40., 10., .8]])
    labels = torch.tensor([0, 1])
    raw = torch.eye(2)
    semantic = torch.eye(2)
    tracker.match(boxes.clone(), labels.clone(), raw.clone(), semantic.clone(), 0)
    assert not fusion.calls and not cycle.calls
    assert torch.equal(tracker.memo[2], raw)
    tracker.match(boxes.clone(), labels.clone(), raw.clone(), semantic.clone(), 1)
    assert len(fusion.calls) == 1 and len(cycle.calls) == 3
    assert torch.equal(cycle.calls[0][0], raw)
    assert torch.equal(cycle.calls[0][1], raw)
    expected = raw + .4 * raw.new_tensor([[.25, .1], [-.1, .3]])
    assert torch.allclose(tracker.memo[2], expected)
    tracker.match(boxes.clone(), labels.clone(), raw.clone(), semantic.clone(), 2)
    assert len(fusion.calls) == 2 and len(cycle.calls) == 6
    assert torch.equal(cycle.calls[3][0], raw)
    assert torch.allclose(cycle.calls[3][1], expected)
    assert not torch.allclose(cycle.calls[3][1], raw)
    plumbing = dict(status="PASS", first_frame_memory="raw", fusion_calls_per_nonempty_memo_frame=1,
                    third_frame_raw_key=cycle.calls[3][0].tolist(),
                    third_frame_context_conditioned_memo=cycle.calls[3][1].tolist(),
                    caveat="Deterministic stub proves source tensor routing, not learned fusion magnitude or real-data efficacy")

    # Reapplying the released after-filter input is idempotent, including the
    # non-transitive case where a removed predecessor originally suppresses C.
    test_tracker = Tracker(vis=False)
    chain = torch.tensor([[0., 0., 10., 10., .4], [4., 0., 14., 10., .3], [8., 0., 18., 10., .2]])
    retained = test_tracker.remove_distractor(chain, torch.zeros(3, dtype=torch.long),
                                             torch.eye(3), torch.eye(3))
    assert len(retained[0]) == 1
    repeated = test_tracker.remove_distractor(*retained[:2], track_feats=retained[2], cls_feats=retained[3])
    assert all(torch.equal(a, b) for a, b in zip(retained[:4], repeated[:4]))
    generator = torch.Generator().manual_seed(20261024)
    for count in range(1, 101):
        low = torch.rand((count, 2), generator=generator) * 20
        high = low + torch.rand((count, 2), generator=generator) * 15 + .1
        values = torch.cat([low, high, torch.rand((count, 1), generator=generator)], dim=1)
        cls = torch.arange(count) % 3
        features = torch.rand((count, 4), generator=generator)
        first = test_tracker.remove_distractor(values, cls, features, features)
        second = test_tracker.remove_distractor(first[0], first[1], first[2], first[3])
        assert all(torch.equal(a, b) for a, b in zip(first[:4], second[:4]))

    # Whole scene rows affect column normalization even when a row will not emit.
    def scores(queries):
        ref = torch.tensor([[1., 0.]])
        exp = torch.exp(similarity(queries, ref))
        return ((exp / (exp.sum(1, keepdim=True) + 1e-6)
                 + exp / (exp.sum(0, keepdim=True) + 1e-6)) / 2
                + similarity(queries, ref, method="cosine")) / 2
    two, one = scores(torch.tensor([[1., 0.], [1., 0.]])), scores(torch.tensor([[1., 0.]]))
    assert not torch.allclose(two[0], one[0])

    # Empty frames must still advance memo expiry. No actual detector is run.
    expiry = Tracker(memo_frames=2, vis=False)
    expiry.match(boxes[:1].clone(), labels[:1].clone(), raw[:1].clone(), semantic[:1].clone(), 0)
    assert not expiry.empty
    expiry.match(torch.empty((0, 5)), torch.empty(0, dtype=torch.long),
                 torch.empty((0, 2)), torch.empty((0, 2)), 2)
    assert expiry.empty

    # Cache precision cannot reconstruct the exact original raw float32 tensors.
    values = np.asarray([.12345678, .23456789, .99987656], np.float32)
    reconstructed = values.astype(np.float16).astype(np.float32)
    assert not np.array_equal(values, reconstructed)
    after = {p: digest(ROOT / p) for p in FILES}
    assert before == after
    result = dict(
        status="PASS_INDEPENDENT_COVTRACK_FEATURE_MEMORY_PATH_SYNTHETIC_AUDIT",
        scope="CPU source and synthetic only; no checkpoint, real NPZ, GT, classification outcomes, training or GPU",
        source_hashes_before=before, source_hashes_after=after,
        checks=dict(memory_tensor_routing=plumbing,
                    after_filter_idempotence=dict(status="PASS", random_cases=100, nontransitive_chain=True),
                    retained_nonemitted_rows_affect_scores=dict(status="PASS", with_other_row=float(two[0]), alone=float(one[0])),
                    empty_frame_expiry=dict(status="PASS"),
                    cache_precision_loss=dict(status="PASS", max_abs_error=float(np.abs(values-reconstructed).max()),
                                              implication="Exact original float32 replay cannot be guaranteed from float16 saved features")),
        cuda_initialized=torch.cuda.is_initialized(),
        limitations=["No baseline replay or efficacy measured", "After-filter idempotence assumes identical retained row order, boxes, scores, labels and thresholds", "Synthetic fusion is a routing probe, not the released trained module"])
    assert not result["cuda_initialized"]
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], checks=list(result["checks"]), artifact=str(OUT))))


if __name__ == "__main__":
    main()
