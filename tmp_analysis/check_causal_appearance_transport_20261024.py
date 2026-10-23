"""Three rounds of falsifiable checks for the untrained transport prototype."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from causal_appearance_transport import (
    angular_velocity_prediction, low_rank_orthogonal_transport,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(function):
    try:
        function()
    except ValueError:
        return True
    return False


def arc(a):
    return np.array([np.cos(a), np.sin(a)])


def angular(previous, last, before, now, future, valid=True):
    return angular_velocity_prediction(np.array([previous]), np.array([last]),
                                      np.array([before]), np.array([now]),
                                      np.array([future]), np.array([valid]))


def run(round_number):
    checks = []
    dtype = torch.float64
    if round_number == 1:
        for sign in (-1, 1):
            out = angular(arc(0), arc(sign * .3), 10, 40, 100)
            require(np.allclose(out['prediction'][0], arc(sign * .9), atol=1e-13),
                    'Original-frame intervals or forward arc direction wrong')
        checks.append('positive_negative_angular_velocity_irregular_frame_intervals')
        out = angular(arc(0), arc(.3), 10, 40, 1000)
        require(np.allclose(out['prediction'][0], arc(9.9), atol=1e-12),
                'Long-gap periodic extrapolation differs')
        checks.append('long_gap_rotation_remains_periodic_without_adaptive_clipping')
        out = angular(arc(0), arc(0), 0, 10, 20)
        require(out['coincident_static'][0] and np.array_equal(out['prediction'][0], arc(0)),
                'Coincident history is not explicitly static')
        out = angular(arc(0), -arc(0), 0, 10, 20)
        require(out['antipodal_fallback'][0] and np.array_equal(out['prediction'][0], -arc(0)),
                'Antipodal ambiguity must not invent a direction')
        checks.append('coincident_static_and_antipodal_fallback_are_distinct')
        out = angular([np.nan, np.nan], arc(.4), np.nan, 10, 20, False)
        require(out['missing_or_zero_history'][0] and np.allclose(out['prediction'][0], arc(.4)),
                'Masked missing history leaked into prediction')
        require(rejected(lambda: angular(arc(0), arc(.3), 10, 10, 20)), 'Duplicate time accepted')
        require(rejected(lambda: angular(arc(0), arc(.3), 0, 10, 9)), 'Future reference accepted')
        checks.append('missing_nan_history_fallback_and_time_rejection')
    elif round_number == 2:
        torch.manual_seed(20261024)
        basis = torch.linalg.qr(torch.randn(7, 3, dtype=dtype))[0]
        a = torch.randn(4, 3, 3, dtype=dtype)
        a = a - a.transpose(-1, -2)
        x = torch.randn(4, 7, dtype=dtype)
        valid = torch.ones(4, dtype=torch.bool)
        gap = torch.tensor([0., .5, 1., 2.], dtype=dtype)
        pred = low_rank_orthogonal_transport(x, basis, a, gap, valid)
        require(torch.allclose(pred.norm(dim=1), x.norm(dim=1), atol=1e-12), 'Norm changed')
        require(torch.allclose(pred - (pred @ basis) @ basis.T,
                               x - (x @ basis) @ basis.T, atol=1e-12), 'Orthogonal complement changed')
        checks.append('norm_and_identity_on_orthogonal_complement')
        direct = low_rank_orthogonal_transport(x, basis, a, gap + .3, valid)
        composition = low_rank_orthogonal_transport(pred, basis, a, torch.full_like(gap, .3), valid)
        require(torch.allclose(direct, composition, atol=2e-12), 'Fixed-generator composition failed')
        checks.append('fixed_generator_semigroup_is_structural_only')
        a[1] = float('nan'); valid[1] = False
        masked = low_rank_orthogonal_transport(x, basis, a, gap, valid)
        require(torch.equal(masked[1], x[1]) and torch.isfinite(masked).all(), 'Invalid generator propagated')
        checks.append('invalid_history_exact_fallback_even_with_nan_generator')
        require(rejected(lambda: low_rank_orthogonal_transport(x, basis * 2, a, gap, valid)),
                'Nonorthonormal basis accepted')
        require(rejected(lambda: low_rank_orthogonal_transport(x, basis, torch.ones_like(a), gap, valid)),
                'Non-skew generator accepted')
        checks.append('basis_and_skew_contract_rejection')
    elif round_number == 3:
        # Orthogonal transforms preserve one vector norm, but track-specific
        # transforms can map distinct identities to exactly the same vector.
        basis = torch.eye(2, dtype=dtype)
        x = torch.eye(2, dtype=dtype)
        generator = torch.zeros(2, 2, 2, dtype=dtype)
        generator[1] = torch.tensor([[0., np.pi / 2], [-np.pi / 2, 0.]], dtype=dtype)
        pred = low_rank_orthogonal_transport(x, basis, generator,
                                             torch.ones(2, dtype=dtype),
                                             torch.ones(2, dtype=torch.bool))
        require(abs(float(pred[0] @ pred[1]) - 1) < 1e-12, 'Collision counterexample disappeared')
        checks.append('orthogonality_does_not_preserve_between_track_identity_separation')
        # A perfectly unique but wrong history can make a deterministic predictor
        # confidently extrapolate to a distractor. This core cannot know truth.
        forecast = angular(arc(0), arc(.4), 0, 10, 20)['prediction'][0]
        truth = arc(.4); distractor = arc(.8)
        require(truth @ distractor < truth @ truth and forecast @ distractor > forecast @ truth,
                'Wrong-history failure challenge did not expose a false improvement')
        checks.append('wrong_history_can_convert_correct_last_feature_to_distractor')
        raw = torch.tensor([[.2, .7]], dtype=dtype, requires_grad=True)
        rate = torch.tensor(.4, dtype=dtype, requires_grad=True)
        z = rate * torch.tensor([[[0., -1.], [1., 0.]]], dtype=dtype)
        result = low_rank_orthogonal_transport(raw, basis, z, torch.ones(1, dtype=dtype),
                                               torch.ones(1, dtype=torch.bool))
        result[0, 0].backward()
        require(torch.isfinite(raw.grad).all() and torch.isfinite(rate.grad) and abs(float(rate.grad)) > .01,
                'No finite nonzero optimization path')
        checks.append('appearance_and_skew_rate_have_finite_nonzero_gradient')
        t = torch.zeros(1, dtype=dtype)
        require(torch.allclose(low_rank_orthogonal_transport(raw.detach(), basis, z.detach(), t,
                                     torch.ones(1, dtype=torch.bool)), raw.detach()), 'Zero gap not identity')
        require(angular_velocity_prediction(np.empty((0, 2)), np.empty((0, 2)),
                     np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool))['prediction'].shape == (0, 2),
                'Empty bank shape changed')
        checks.append('zero_gap_identity_and_empty_memory')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    checks = run(args.round)
    root = Path(__file__).resolve().parent
    receipt = dict(status='PASS_SYNTHETIC_TRANSPORT_PRIMITIVES', round=args.round,
                   checks=checks, checks_passed=len(checks), training_started=False,
                   real_data_accessed=False, performance_or_novelty_established=False,
                   code_sha256=hashlib.sha256((root / 'causal_appearance_transport.py').read_bytes()).hexdigest(),
                   checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = root / ('causal_appearance_transport_synthetic_round%d_20261024.json' % args.round)
    with out.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
