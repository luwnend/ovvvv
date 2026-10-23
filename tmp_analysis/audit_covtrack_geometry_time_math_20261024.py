"""Independent mathematical examples only: no real data/model/GT."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repos/COVTrack/ovtrack/models/losses/cyc_loss.py"
OUT = ROOT / "tmp_analysis/covtrack_geometry_time_math_independent_review_20261024.json"


def pairwise_cos(x, y):
    return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T


def relative_geometry(x, y):
    """All-pairs center delta / reference width-height and log scale ratios."""
    cx, cy = (x[:, :2] + x[:, 2:]) / 2, (y[:, :2] + y[:, 2:]) / 2
    sx, sy = x[:, 2:] - x[:, :2], y[:, 2:] - y[:, :2]
    return torch.cat(((cx[:, None] - cy[None]) / sy[None],
                      torch.log(sx[:, None] / sy[None])), dim=-1)


def main():
    torch.set_num_threads(1)
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if getattr(n, "name", "") == "CycleLoss")
    node.decorator_list = []
    namespace = dict(torch=torch, nn=torch.nn, np=np, f=F, F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)
    cycle = namespace["CycleLoss"]()
    memo = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]], dtype=torch.float64)
    key = torch.tensor([[2., 0., 12., 10.], [18., 0., 28., 10.]], dtype=torch.float64)
    translate = torch.tensor([1000., 1000., 1000., 1000.], dtype=torch.float64)
    original_cos, moved_cos = pairwise_cos(key, memo), pairwise_cos(key+translate, memo+translate)
    original_cycle, moved_cycle = cycle.get_pair_consistency([key, memo])[1], cycle.get_pair_consistency([key+translate, memo+translate])[1]
    assert not torch.allclose(original_cos, moved_cos)
    assert not torch.allclose(original_cycle, moved_cycle)
    assert torch.allclose(relative_geometry(key, memo), relative_geometry(key+translate, memo+translate))
    assert torch.allclose(pairwise_cos(3*key, 3*memo), original_cos)
    assert torch.allclose(relative_geometry(3*key, 3*memo), relative_geometry(key, memo))
    anisotropic = torch.tensor([3., .5, 3., .5], dtype=torch.float64)
    assert not torch.allclose(pairwise_cos(key*anisotropic, memo*anisotropic), original_cos)
    assert torch.allclose(relative_geometry(key*anisotropic, memo*anisotropic), relative_geometry(key, memo))
    radial_a = torch.tensor([[10., 10., 20., 20.]], dtype=torch.float64)
    radial_b = radial_a * 2
    assert torch.allclose(pairwise_cos(radial_a, radial_b), torch.ones((1, 1), dtype=torch.float64))
    radial_intersection = (torch.minimum(radial_a[:, 2:], radial_b[:, 2:]) - torch.maximum(radial_a[:, :2], radial_b[:, :2])).clamp_min(0).prod()
    assert float(radial_intersection) == 0.
    singleton_conf = cycle.get_pair_consistency([radial_a, radial_b])[1]
    assert singleton_conf.tolist() == [1.]
    orthogonal_conf = cycle.get_pair_consistency([torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])])[1]
    assert orthogonal_conf.tolist() == [1.]

    # The same constant-velocity path, sampled at unequal original-frame gaps.
    times, centers = np.asarray([0., 10., 40.]), np.asarray([0., 10., 40.])
    sample_times = np.arange(3, dtype=float)
    actual_velocity = (centers[1]-centers[0]) / (times[1]-times[0])
    timed_prediction = centers[1] + actual_velocity * (times[2]-times[1])
    sample_velocity = (centers[1]-centers[0]) / (sample_times[1]-sample_times[0])
    step_prediction = centers[1] + sample_velocity * (sample_times[2]-sample_times[1])
    assert timed_prediction == centers[2] and step_prediction != centers[2]
    assert before == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result = dict(status="PASS_INDEPENDENT_COVTRACK_GEOMETRY_TIME_MATHEMATICAL_CHECK",
                  scope="Synthetic mathematical examples only; no dataset, model/checkpoint, learning, GPU or efficacy evaluation",
                  source_sha256=before,
                  absolute_xyxy_cosine=dict(original=original_cos.tolist(), translated=moved_cos.tolist(),
                                            original_cycle_confidence=original_cycle.tolist(), translated_cycle_confidence=moved_cycle.tolist(),
                                            common_translation_changes_cosine=True, relative_geometry_unchanged=True,
                                            common_uniform_scale_preserves_cosine=True,
                                            common_anisotropic_resize_changes_cosine=True,
                                            radial_scaling_can_give_cosine_one_with_zero_overlap=True),
                  cycle_singleton_degeneracy=dict(disjoint_box_confidence=singleton_conf.tolist(), orthogonal_appearance_confidence=orthogonal_conf.tolist(),
                                                   explanation="One candidate on each side makes both softmax matrices [1]; this is cycle assignment ambiguity, not absolute match quality"),
                  unequal_time_cv=dict(original_frame_index=times.tolist(), sampled_step=sample_times.tolist(), centers=centers.tolist(),
                                       correct_original_frame_prediction=float(timed_prediction), wrong_unit_step_prediction=float(step_prediction), target_center=float(centers[2])),
                  limits=["No measured frequency or benefit on real data", "Do not replace the pretrained bbox_transform coordinate units", "Relative motion is a separate control with original-frame timing; lifecycle remains in the released sampled-step units unless separately specified", "Original frame index is not seconds without verified FPS"])
    assert not torch.cuda.is_initialized()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], artifact=str(OUT), original_cycle_confidence=original_cycle.tolist(), translated_cycle_confidence=moved_cycle.tolist(), dt_predictions=[float(timed_prediction),float(step_prediction)])))


if __name__ == "__main__":
    main()
