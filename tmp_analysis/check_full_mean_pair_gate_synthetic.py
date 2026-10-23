#!/usr/bin/env python3
"""Exact finite-support algebra for the full mean/pair protocol; no data access."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, description):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
        raise AssertionError((description, actual, expected))


def wrong(p, y):
    result = np.array(p, dtype=np.float64, copy=True)
    result[y] = 0
    return result


def null_trace(p, q, pi, mui, muj):
    g0 = float(p @ q - np.sum(pi * p * q))
    expected_wi, expected_wj = p * (1 - pi), q * (1 - pi)
    return float(g0 - expected_wi @ muj - mui @ expected_wj + mui @ muj)


def run():
    # Independent endpoints with one common unknown label. The true coarsened
    # mean is known exactly; positive residual is not a mean-fitting failure.
    p_values = np.asarray([[.9, .1], [.5, .5]])
    q_values = np.asarray([[.7, .3], [.1, .9]])
    pi = np.asarray([.6, .4])
    mi, mj = p_values.mean(0), q_values.mean(0)
    mui, muj = mi * (1 - pi), mj * (1 - pi)
    trace, conditional_y_trace = 0., 0.
    for a, b, y in itertools.product(range(2), range(2), range(2)):
        wi, wj = wrong(p_values[a], y), wrong(q_values[b], y)
        weight = .25 * pi[y]
        trace += weight * float((wi - mui) @ (wj - muj))
        conditional_y_trace += weight * float((wi - wrong(mi, y)) @ (wj - wrong(mj, y)))
    latent = float(np.sum(mi * mj * pi * (1 - pi)))
    close(trace, latent, "positive exact latent-label term")
    close(conditional_y_trace, 0., "Y-conditional mean removes the latent-label term")
    assert trace > 0

    # General law of total covariance when endpoint distributions depend on Y.
    p_cond = np.asarray([[[.9, .1], [.5, .5]], [[.4, .6], [.2, .8]]])
    q_cond = np.asarray([[[.8, .2], [.4, .6]], [[.6, .4], [.2, .8]]])
    conditional_mui = np.asarray([wrong(p_cond[y].mean(0), y) for y in range(2)])
    conditional_muj = np.asarray([wrong(q_cond[y].mean(0), y) for y in range(2)])
    marginal_mui, marginal_muj = pi @ conditional_mui, pi @ conditional_muj
    total, within, between = 0., 0., 0.
    for y, a, b in itertools.product(range(2), range(2), range(2)):
        wi, wj = wrong(p_cond[y, a], y), wrong(q_cond[y, b], y)
        total += .25 * pi[y] * float((wi - marginal_mui) @ (wj - marginal_muj))
        within += .25 * pi[y] * float((wi - conditional_mui[y]) @ (wj - conditional_muj[y]))
    for y in range(2):
        between += pi[y] * float((conditional_mui[y] - marginal_mui) @
                                (conditional_muj[y] - marginal_muj))
    close(total, within + between, "law of total covariance")
    close(within, 0., "conditional endpoint independence")

    rng = np.random.default_rng(20261023)
    null_cases = permutation_cases = 0
    max_expected_excess = 0.
    for _ in range(100):
        p, q, probability = rng.dirichlet(np.ones(5), size=3)
        mui = rng.dirichlet(np.ones(5)) * rng.random()
        muj = rng.dirichlet(np.ones(5)) * rng.random()
        expected = sum(probability[y] * float((wrong(p, y) - mui) @ (wrong(q, y) - muj))
                       for y in range(5))
        t0 = null_trace(p, q, probability, mui, muj)
        close(expected, t0, "operational shared-label null includes all cross terms")
        close(t0, null_trace(q, p, probability, muj, mui), "endpoint symmetry")
        permutation = rng.permutation(5)
        close(t0, null_trace(p[permutation], q[permutation], probability[permutation],
                            mui[permutation], muj[permutation]), "candidate permutation equivariance")
        excess = sum(probability[y] * (float((wrong(p, y) - mui) @ (wrong(q, y) - muj)) - t0)
                     for y in range(5))
        close(excess, 0., "a true full-conditional pi leaves no predictable t-t0 mean")
        max_expected_excess = max(max_expected_excess, abs(excess))
        null_cases += 1
        permutation_cases += 1

    # An imperfect label posterior produces an apparent correction despite fixed
    # endpoints and no possible view-coupling mechanism in their generation.
    p, q = np.asarray([.9, .1]), np.asarray([.7, .3])
    true_pi, fitted_pi = np.asarray([.6, .4]), np.asarray([.5, .5])
    mui, muj = p * (1 - true_pi), q * (1 - true_pi)
    exact = null_trace(p, q, true_pi, mui, muj)
    fitted = null_trace(p, q, fitted_pi, mui, muj)
    posterior_misspecification_excess = exact - fitted
    assert abs(posterior_misspecification_excess) > .001

    # Noisy estimates of two means share an estimation error when based on the
    # same reference. Independent reference errors remove this artifact here.
    shared_reference = independent_reference = 0.
    for xi, xj, reference in itertools.product([-.1, .1], [-.1, .1], [-.2, .2]):
        shared_reference += (xi - reference) * (xj - reference) / 8
    for xi, xj, ri, rj in itertools.product([-.1, .1], [-.1, .1], [-.2, .2], [-.2, .2]):
        independent_reference += (xi - ri) * (xj - rj) / 16
    close(shared_reference, .04, "shared-reference nuisance estimation error")
    close(independent_reference, 0., "disjoint-reference estimation errors")

    return dict(
        status="PASS_EXACT_COMMON_LABEL_AND_REFERENCE_SYNTHETIC_ONLY",
        independent_endpoint_shared_label=dict(
            exact_centered_trace=trace, latent_label_term=latent,
            label_conditional_trace=conditional_y_trace,
            interpretation="Positive ideal-centered trace survives perfect marginal means; common Y alone suffices"),
        general_total_covariance=dict(total_trace=total, within_Y=within, between_Y=between),
        exact_operational_null_cases=null_cases, class_permutation_cases=permutation_cases,
        max_absolute_true_pi_expected_excess=max_expected_excess,
        misspecified_pi=dict(apparent_excess=posterior_misspecification_excess,
                             interpretation="A learnable correction can be label-posterior misspecification alone"),
        shared_reference_estimation=dict(shared_error_trace=shared_reference,
                                         disjoint_error_trace=independent_reference),
        full_dataset_opened=False, test_opened=False, heldout_correctness_opened=False,
        real_model_fitted=False, GPU_used=False,
        limitations="Exact constructed distributions only; no data sufficiency, efficacy, causal mechanism or novelty evidence",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = ROOT / "tmp_analysis/full_mean_pair_gate_protocol_20261023.json"
    assert protocol.is_file(), "Protocol must predate synthetic execution"
    assert not args.output.exists(), "Preserve receipts"
    result = run()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
