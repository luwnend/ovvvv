"""Independent synthetic spot audit of untrained transport primitives only.

Imports the primitive, never the producer's checker, cached data or labels.
Independent closed-form rotation oracles check direction, units and gradients.
This script does not certify a learned predictor, novelty or tracking utility.
"""
import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

CORE = Path(__file__).resolve().with_name("causal_appearance_transport.py")
EXPECTED_CORE_SHA256 = "8b534c5f133880165e3e46abbd93e49584ebf8e20bec9c58b46ab425fa8f5924"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    if sha(CORE) != EXPECTED_CORE_SHA256:
        raise RuntimeError("Audit is bound to a different primitive revision")
    spec = importlib.util.spec_from_file_location("independent_transport_subject", CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    checks = []

    def check(value, name):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    def rejected(function, name):
        try:
            function()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError("Expected rejection: " + name)

    a = math.pi / 6
    previous = np.array([[1., 0.]])
    last = np.array([[math.cos(a), math.sin(a)]])
    valid = np.array([True])
    args = [previous, last, np.array([2.]), np.array([5.]), np.array([11.]), valid]
    originals = [value.copy() for value in args]
    prediction = core.angular_velocity_prediction(*args)
    check(np.allclose(prediction["prediction"], [[0., 1.]], atol=1e-14, rtol=0),
          "hand_direction_30deg_per_3_original_frames_then_6_frames_yields_90deg")
    check(prediction["extrapolation_ratio"].tolist() == [2.], "original_frame_ratio_is_two_not_sample_step_one")
    check(all(np.array_equal(x, y) for x, y in zip(args, originals)), "angular_inputs_not_mutated")
    reverse = core.angular_velocity_prediction(last, previous, args[2], args[3], args[4], valid)
    check(np.allclose(reverse["prediction"], [[.5, -math.sqrt(3) / 2]], atol=1e-14, rtol=0),
          "reverse_past_arc_reverses_future_direction")
    fallback = core.angular_velocity_prediction(np.array([[np.nan, np.nan], [-1., 0.], [1., 0.], [0., 0.]]),
        np.array([[1., 0.]] * 4), np.array([np.nan, 0., 0., 0.]), np.ones(4), np.full(4, 100.),
        np.array([False, True, True, True]))
    check(np.array_equal(fallback["prediction"], np.array([[1., 0.]] * 4)),
          "masked_poison_antipodal_coincident_zero_previous_return_last")
    check(fallback["missing_or_zero_history"].tolist() == [True, False, False, True]
          and fallback["antipodal_fallback"].tolist() == [False, True, False, False]
          and fallback["coincident_static"].tolist() == [False, False, True, False],
          "missing_zero_coincident_antipodal_reason_masks_are_distinct")
    for before, now, future in ((2., 1., 3.), (0., 1., 0.)):
        rejected(lambda: core.angular_velocity_prediction(previous, last, np.array([before]), np.array([now]),
                 np.array([future]), valid), "invalid_time_order_rejected_" + str((before, now, future)))
    long = core.angular_velocity_prediction(previous, last, args[2], args[3], np.array([77.]), valid)
    check(np.allclose(long["prediction"], last, atol=1e-13, rtol=0),
          "long_gap_extra_4pi_turn_exposes_periodicity_without_hidden_clipping")
    empty = core.angular_velocity_prediction(np.empty((0, 2)), np.empty((0, 2)), np.empty(0), np.empty(0),
                                             np.empty(0), np.empty(0, dtype=bool))
    check(empty["prediction"].shape == (0, 2) and empty["directed_arc"].shape == (0,), "empty_angular_shapes")
    params = list(inspect.signature(core.angular_velocity_prediction).parameters)
    check(params == ["previous", "last", "previous_time", "last_time", "query_time", "history_valid"],
          "angular_api_has_no_query_appearance_or_query_box_input")
    # Unique-but-wrong history remains marked valid and forecasts the wrong identity.
    contaminated = core.angular_velocity_prediction(np.array([[0., 1.]]), previous, np.array([0.]),
                        np.array([1.]), np.array([2.]), valid)
    check(contaminated["directed_arc"].item() and np.allclose(contaminated["prediction"], [[0., -1.]], atol=1e-14),
          "wrong_unique_history_cannot_be_detected_by_the_primitive")

    dtype = torch.float64
    basis = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=dtype)
    generator = torch.tensor([[[0., -1.], [1., 0.]]], dtype=dtype)
    features = torch.tensor([[1., 0., 2.]], dtype=dtype)
    mask = torch.tensor([True])
    gap = torch.tensor([math.pi / 2], dtype=dtype)
    tensors = (features, basis, generator, gap, mask)
    originals = [value.clone() for value in tensors]
    result = core.low_rank_orthogonal_transport(*tensors)
    check(torch.allclose(result, torch.tensor([[0., 1., 2.]], dtype=dtype), atol=1e-13, rtol=0),
          "closed_form_rotation_sends_e1_to_e2_and_preserves_orthogonal_complement")
    check(all(torch.equal(x, y) for x, y in zip(tensors, originals)), "transport_inputs_not_mutated")
    second = core.low_rank_orthogonal_transport(result, basis, generator, gap, mask)
    direct = core.low_rank_orthogonal_transport(features, basis, generator, 2 * gap, mask)
    check(torch.allclose(second, direct, atol=1e-13, rtol=0), "fixed_generator_semigroup_is_only_an_algebraic_property")
    identity = core.low_rank_orthogonal_transport(features, basis, generator, torch.zeros_like(gap), mask)
    check(torch.equal(identity, features), "zero_gap_exact_identity")
    poison = torch.full((1, 2, 2), float("nan"), dtype=dtype)
    ignored = core.low_rank_orthogonal_transport(features, basis, poison, gap, ~mask)
    check(torch.equal(ignored, features), "invalid_poison_generator_exact_fallback")
    different = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=dtype)
    operators = torch.cat((torch.zeros_like(generator), -generator))
    collapsed = core.low_rank_orthogonal_transport(different, basis, operators, gap.repeat(2), torch.tensor([True, True]))
    check(torch.allclose(collapsed[0], collapsed[1], atol=1e-13, rtol=0)
          and not torch.allclose(different[0], different[1]),
          "counterexample_different_orthogonal_operators_collapse_distinct_identities")
    for dtype in (torch.float32, torch.float64):
        empty = core.low_rank_orthogonal_transport(torch.empty((0, 3), dtype=dtype), basis.to(dtype),
                torch.empty((0, 2, 2), dtype=dtype), torch.empty(0, dtype=dtype), torch.empty(0, dtype=torch.bool))
        check(empty.shape == (0, 3) and empty.dtype == dtype, "empty_transport_" + str(dtype))
        theta = .73
        output = core.low_rank_orthogonal_transport(features.to(dtype), basis.to(dtype), generator.to(dtype),
                        torch.tensor([theta], dtype=dtype), mask)
        expected = torch.tensor([[math.cos(theta), math.sin(theta), 2.]], dtype=dtype)
        tolerance = 3e-6 if dtype == torch.float32 else 1e-13
        check(torch.allclose(output, expected, atol=tolerance, rtol=0)
              and torch.allclose(output.norm(dim=1), features.to(dtype).norm(dim=1), atol=tolerance, rtol=0),
              "closed_form_rotation_and_norm_tolerance_" + str(dtype))
    # A scalar rate yields an exact hand derivative; do not finite-difference the implementation.
    rate = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    differentiable_features = features.clone().requires_grad_()
    rate_generator = generator * rate
    differentiable = core.low_rank_orthogonal_transport(differentiable_features, basis, rate_generator,
                            torch.tensor([2.], dtype=torch.float64), mask)
    loss = differentiable[0, 0]
    loss.backward()
    check(abs(rate.grad.item() + 2 * math.sin(.8)) < 1e-12
          and torch.allclose(differentiable_features.grad,
                            torch.tensor([[math.cos(.8), -math.sin(.8), 0.]], dtype=torch.float64), atol=1e-12, rtol=0),
          "autograd_scalar_generator_rate_and_feature_gradients_match_hand_derivatives")
    rejected(lambda: core.low_rank_orthogonal_transport(features, basis, torch.ones_like(generator), gap, mask),
             "non_skew_generator_rejected")
    rejected(lambda: core.low_rank_orthogonal_transport(features, 2 * basis, generator, gap, mask),
             "non_orthonormal_basis_rejected")
    check(sha(CORE) == EXPECTED_CORE_SHA256, "core_hash_unchanged_after_synthetic_audit")
    return dict(status="PASS_INDEPENDENT_CAUSAL_APPEARANCE_TRANSPORT_SYNTHETIC_AUDIT",
                recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                core_path=str(CORE), core_sha256=sha(CORE), auditor_sha256=sha(__file__),
                checks=checks, checks_passed=len(checks), device="CPU", synthetic_only=True,
                real_features_or_supervision_opened=False, training_started=False,
                limits=["Mathematical synthetic checks only; no real FIT dynamic-control result",
                        "No learned generator or future-leakage audit of an external history encoder",
                        "Orthogonality does not guarantee cross-identity separation",
                        "Fixed-generator semigroup is structural, not learned evidence",
                        "Unique-but-contaminated history remains undetectable inside this API"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
