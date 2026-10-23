"""Independent CPU synthetic preflight of the five-mode prototype only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp_analysis"))
import context_feature_memory as subject
from ovtrack.models.losses.cyc_loss import CycleLoss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OffsetFusion(torch.nn.Module):
    def __init__(self, zero=False):
        super().__init__()
        self.zero = zero

    def forward(self, raw, boxes, cls, appearance_conf, box_conf, semantic_conf):
        if self.zero:
            return raw.clone()
        return raw + .015 * (1-appearance_conf) + .0002 * boxes[:, :1]


def fixture(frame, count=3):
    raw = torch.zeros((count, 256))
    cls = torch.zeros((count, 512))
    for k in range(count):
        raw[k, k] = 2. + frame*.03
        cls[k, k] = 1.
        cls[k, 511] = .05*frame
    boxes = torch.tensor([[10.+k*40+frame, 10., 25.+k*40+frame, 30., .9]
                          for k in range(count)], dtype=torch.float32).reshape(count, 5)
    return boxes, torch.arange(count, dtype=torch.long), raw, cls


def clone_args(args):
    return [x.clone() for x in args]


def tracker(mode, zero=False, **kwargs):
    cfg = dict(confused_features=True, vis=False, match_score_thr=.37,
               momentum_embed=.4, memo_frames=3)
    cfg.update(kwargs)
    if mode == "original":
        result = subject.OVTrackerUncertainty(**cfg)
    else:
        result = subject.OVTrackerContextFeatureMemory(feature_memory_mode=mode,
                                                       record_context=True, **cfg)
    result.set_fusion_head(OffsetFusion(zero), CycleLoss())
    return result


def compare_state(base, other):
    assert base.num_tracklets == other.num_tracklets
    assert list(base.tracklets) == list(other.tracklets)
    for identity in base.tracklets:
        a, b = base.tracklets[identity], other.tracklets[identity]
        assert a["frame_ids"] == b["frame_ids"]
        for key in ("bboxes", "labels"):
            assert len(a[key]) == len(b[key])
            assert all(torch.equal(x, y) for x, y in zip(a[key], b[key]))
        assert torch.equal(a["embeds"], b["embeds"])
        assert torch.equal(a["cls_embeds"], b["cls_embeds"])


def main():
    torch.set_num_threads(1)
    module_sha = sha(subject.__file__)
    protocol = ROOT / "tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
    protected = {str(subject.WORK_COPY/k): sha(subject.WORK_COPY/k) for k in subject.SOURCE_PINS}
    checks = []

    # Original scientific class is an independent behavioral baseline.
    for zero in (False, True):
        base = tracker("original", zero)
        modes = subject.MODES if zero else ("released",)
        others = {mode: tracker(mode, zero) for mode in modes}
        for frame, count in ((0,3),(1,3),(2,2),(3,0),(7,0),(8,2)):
            args = fixture(frame, count)
            expected = base.match(*clone_args(args), frame_id=frame)
            for mode, obj in others.items():
                got = obj.match(*clone_args(args), frame_id=frame)
                assert all(torch.equal(a,b) for a,b in zip(expected, got)), (zero, mode, frame)
                compare_state(base, obj)
    checks.append("released_exact_class_parity_and_all_modes_zero_residual_reduction")

    # Independently reconstruct raw EMA from observed immutable raw inputs and
    # final assigned IDs, without importing producer bank/confidence helpers.
    primary = tracker("raw_confidence")
    expected_raw = {}
    for frame in range(4):
        args = fixture(frame)
        before_raw = {k:v.clone() for k,v in expected_raw.items()}
        before_memo = None if primary.empty else tuple(x.clone() for x in primary.memo)
        got = primary.match(*clone_args(args), frame_id=frame)
        for identity, vector in zip(got[2], args[2][primary.keep_mask]):
            identity = int(identity)
            if identity < 0:
                continue
            expected_raw[identity] = .6*expected_raw[identity] + .4*vector if identity in expected_raw else vector.clone()
        expected_raw = {k:v for k,v in expected_raw.items() if k in primary.tracklets}
        assert set(expected_raw) == set(primary.raw_appearance)
        for k,v in expected_raw.items():
            assert torch.equal(v, primary.raw_appearance[k]["embeds"])
        if before_memo is not None:
            context = primary.last_context
            assert torch.equal(context["scoring_reference"], before_memo[2])
            raw_reference = torch.stack([before_raw[int(k)] for k in before_memo[4]])
            assert torch.equal(context["raw_reference"], raw_reference)
            for index, (query, reference) in enumerate(((args[2],raw_reference),
                                                       (args[0][:,:4],before_memo[0][:,:4]),
                                                       (args[3],before_memo[3]))):
                expected_conf = CycleLoss().get_pair_consistency([query,reference])[1]
                assert torch.equal(context["key_confidences"][index].flatten(), expected_conf)
            if frame >= 2:
                assert not torch.equal(raw_reference, before_memo[2])
                legacy_conf = CycleLoss().get_pair_consistency([args[2],before_memo[2]])[1]
                assert not torch.equal(context["key_confidences"][0].flatten(), legacy_conf)
    checks.append("independent_raw_EMA_reconstruction_primary_changes_only_appearance_confidence_memory")

    symmetric = tracker("raw_store_symmetric")
    symmetric.match(*fixture(0), frame_id=0)
    previous = tuple(x.clone() for x in symmetric.memo)
    args = fixture(1,2)
    symmetric.match(*clone_args(args), frame_id=1)
    context = symmetric.last_context
    for i,(key,ref) in enumerate(((args[2],previous[2]),(args[0][:,:4],previous[0][:,:4]),(args[3],previous[3]))):
        forward = CycleLoss().get_pair_consistency([key,ref])[1]
        reverse = CycleLoss().get_pair_consistency([ref,key])[1]
        assert torch.equal(context["key_confidences"][i].flatten(),forward)
        assert torch.equal(context["reference_confidences"][i].flatten(),reverse)
    assert context["scoring_reference"].shape == (3,256)
    assert context["scoring_key"].shape == (2,256)
    checks.append("independent_bidirectional_same_raw_snapshot_unequal_scene_sizes")

    for mode in subject.MODES:
        obj = tracker(mode, init_score_thr=.5, obj_score_thr=.5)
        args = list(fixture(0))
        args[0][1] = args[0][0]
        args[0][1,-1] = .1
        args[0][2,-1] = .2
        original = clone_args(args)
        got = obj.match(*clone_args(args), frame_id=0)
        assert obj.keep_mask.tolist() == [True,False,True]
        assert got[2].tolist() == [0,-1]
        assert list(obj.raw_appearance) == [0]
        assert torch.equal(obj.raw_appearance[0]["embeds"],original[2][0])
        assert all(torch.equal(a,b) for a,b in zip(args,original))
        obj.match(*fixture(5,0),frame_id=5)
        assert not obj.tracklets and not obj.raw_appearance
    checks.append("all_mode_original_filter_row_alignment_unassigned_context_and_empty_expiry")

    assert module_sha == sha(subject.__file__)
    assert protected == {p:sha(p) for p in protected}
    assert not torch.cuda.is_initialized()
    result = dict(status="PASS_INDEPENDENT_CONTEXT_FEATURE_MEMORY_IMPLEMENTATION_PREFLIGHT",
                  module_sha256=module_sha, protocol_sha256=sha(protocol),
                  checks=checks, protected_source_sha256=protected,
                  actual_runtime=dict(torch=torch.__version__,python=sys.executable,CPU_threads=1,GPU_initialized=False),
                  real_cache_or_GT_opened=False,real_checkpoint_loaded=False,
                  limitations=["Synthetic implementation preflight only", "No exact original float32 GPU replay or real-data benefit established", "Scoring runner requires separate preflight"])
    out = ROOT / "tmp_analysis/context_feature_memory_20261024_independent_preflight.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
