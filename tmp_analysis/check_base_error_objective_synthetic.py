#!/usr/bin/env python3
"""Exact CPU-only finite-support checks for the proposed error objectives.

No dataset, validation artifact, model, GPU or optimization is accessed.
"""

import argparse
import hashlib
import json
from pathlib import Path


def expect(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def covariance_cases():
    cases = {}
    for name, probs in {
        "independent": [0.25, 0.25, 0.25, 0.25],
        "positive": [0.5, 0.0, 0.0, 0.5],
        "negative": [0.0, 0.5, 0.5, 0.0],
    }.items():
        states = [(0, 0), (0, 1), (1, 0), (1, 1)]
        bi = expect([x for x, _ in states], probs)
        bj = expect([y for _, y in states], probs)
        joint = expect([x * y for x, y in states], probs)
        centered = expect([(x - bi) * (y - bj) for x, y in states], probs)
        assert abs(centered - (joint - bi * bj)) < 1e-12
        cases[name] = {"b_i": bi, "b_j": bj, "product_of_marginal_risks": bi * bj,
                       "joint_wrong_probability": joint, "binary_error_covariance": centered}
    assert {x["product_of_marginal_risks"] for x in cases.values()} == {0.25}
    assert [cases[k]["binary_error_covariance"] for k in ["independent", "positive", "negative"]] == [0.0, 0.25, -0.25]
    return cases


def same_wrong_direction_cases():
    # Ground-truth class is zero. Prediction -1 below means correct (zero).
    states = [(False, False), (False, True), (True, False), (True, True)]
    probs = [0.25] * 4
    output = {}
    for name, wrong_j in [("same_wrong_class", 1), ("different_wrong_class", 2)]:
        wi = [[0, int(ei), 0] for ei, _ in states]
        wj = [[0, int(ej) * int(wrong_j == 1), int(ej) * int(wrong_j == 2)] for _, ej in states]
        mui = [expect([row[c] for row in wi], probs) for c in range(3)]
        muj = [expect([row[c] for row in wj], probs) for c in range(3)]
        same = expect([dot(x, y) for x, y in zip(wi, wj)], probs)
        joint = expect([int(x) * int(y) for x, y in states], probs)
        output[name] = {"joint_wrong_probability": joint, "same_wrong_class_probability": same,
                        "independent_wrong_direction_overlap": dot(mui, muj),
                        "diagonal_wrong_vector_excess": same - dot(mui, muj)}
    assert output["same_wrong_class"]["joint_wrong_probability"] == output["different_wrong_class"]["joint_wrong_probability"] == 0.25
    assert output["same_wrong_class"]["same_wrong_class_probability"] == 0.25
    assert output["different_wrong_class"]["same_wrong_class_probability"] == 0.0
    # Cross-category dependence can be nonzero while diagonal same-error excess is zero.
    # Exactly one of (correct, correct) / (wrong-class1, wrong-class2) occurs.
    left = [[0, 0, 0], [0, 1, 0]]
    right = [[0, 0, 0], [0, 0, 1]]
    probabilities = [0.5, 0.5]
    left_mean = [expect([row[c] for row in left], probabilities) for c in range(3)]
    right_mean = [expect([row[c] for row in right], probabilities) for c in range(3)]
    cross_covariance = [
        [expect([(x[c] - left_mean[c]) * (y[d] - right_mean[d])
                 for x, y in zip(left, right)], probabilities) for d in range(3)]
        for c in range(3)
    ]
    trace = sum(cross_covariance[c][c] for c in range(3))
    binary_covariance = sum(sum(row) for row in cross_covariance)
    assert binary_covariance == cross_covariance[1][2] == 0.25
    assert trace == 0.0
    output["cross_class_covariance_hidden_by_diagonal"] = {
        "both_wrong_covariance": binary_covariance, "diagonal_same_wrong_excess": trace,
        "nonzero_off_diagonal_covariance_C12": cross_covariance[1][2],
    }
    return output


def conditional_mean_mismatch():
    # Each group is conditionally independent. Marginal risk mixes easy/hard groups.
    risks = [0.1, 0.9]
    marginal = sum(risks) / 2
    pooled_centered = sum((p - marginal) ** 2 for p in risks) / 2
    assert abs(pooled_centered - 0.16) < 1e-12
    return {"within_group_covariance": 0.0, "pooled_marginal_centering_excess": pooled_centered,
            "meaning": "Uncontrolled quality composition creates excess despite within-group independence."}


def same_identity_hard_event_constraint():
    for y in range(3):
        for hi in range(3):
            for hj in range(3):
                ei, ej = int(hi != y), int(hj != y)
                same = int(hi == hj and hi != y)
                assert same == int(hi == hj) * ei * ej
                if hi == hj:
                    assert ei == ej
                else:
                    assert ei + ej >= 1
                    assert ei * ej == ei + ej - 1
    return {"enumerated_states": 27, "status": "PASS",
            "meaning": "Hard same-wrong has an exact argmax-agreement gate for a common true label."}


def soft_centering_identity():
    pi = [0.6, 0.3, 0.1]
    pj = [0.2, 0.7, 0.1]
    mean_label = [0.4, 0.5, 0.1]
    # Both observations share the same random true label, drawn from mean_label.
    wi = [[pi[c] * int(c != y) for c in range(3)] for y in range(3)]
    wj = [[pj[c] * int(c != y) for c in range(3)] for y in range(3)]
    mui = [expect([row[c] for row in wi], mean_label) for c in range(3)]
    muj = [expect([row[c] for row in wj], mean_label) for c in range(3)]
    raw = expect([dot(x, y) for x, y in zip(wi, wj)], mean_label)
    centered = expect([dot([x[c] - mui[c] for c in range(3)],
                           [y[c] - muj[c] for c in range(3)]) for x, y in zip(wi, wj)], mean_label)
    assert abs(centered - (raw - dot(mui, muj))) < 1e-12
    y = 0
    actual_centered = dot([wi[y][c] - mui[c] for c in range(3)],
                          [wj[y][c] - muj[c] for c in range(3)])
    shortcut = dot(wi[y], wj[y]) - dot(mui, muj)
    assert abs(actual_centered - shortcut) > 1e-6
    perm = [2, 0, 1]
    assert abs(dot(wi[y], wj[y]) - dot([wi[y][c] for c in perm], [wj[y][c] for c in perm])) < 1e-12
    return {"expected_centered_trace": centered, "expected_raw_minus_mean_product": raw - dot(mui, muj),
            "single_pair_centered_target": actual_centered, "incorrect_single_pair_shortcut": shortcut,
            "class_permutation_invariance": "PASS",
            "meaning": "Expectation identity requires correct conditional means; individual targets need both cross terms."}


def plugin_sign_trap_and_direct_mean():
    vectors = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.2, 0.5]]
    count = 0
    for truth in range(3):
        for pleft in vectors:
            for pright in vectors:
                for qleft in vectors:
                    for qright in vectors:
                        left = [pleft[c] * (qleft[c] - int(c == truth)) for c in range(3)]
                        right = [pright[c] * (qright[c] - int(c == truth)) for c in range(3)]
                        assert all(x * y >= 0 for x, y in zip(left, right))
                        count += 1
    states = [(0.1, 0.1), (0.1, 0.5), (0.5, 0.1), (0.5, 0.5)]
    results = {}
    for name, probabilities in {"independent": [0.25] * 4,
                                "positive": [0.5, 0, 0, 0.5],
                                "negative": [0, 0.5, 0.5, 0]}.items():
        # Common true class zero; wrong evidence is in class one.
        left_mean = expect([x for x, _ in states], probabilities)
        right_mean = expect([y for _, y in states], probabilities)
        residual_product = expect([(x - left_mean) * (y - right_mean) for x, y in states], probabilities)
        results[name] = residual_product
    assert abs(results["independent"]) < 1e-12
    assert abs(results["positive"] - 0.04) < 1e-12
    assert abs(results["negative"] + 0.04) < 1e-12
    return {"plugin_same_identity_nonnegative_states_checked": count,
            "direct_coarsened_mean_residual_trace": results,
            "meaning": "The current-posterior plug-in has a sign degeneracy; direct coarsened means allow positive and negative natural-view residuals."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"status": "PASS_EXACT_SYNTHETIC_CONSISTENCY_ONLY", "uses_dataset": False,
              "uses_validation": False, "uses_gpu": False, "training_run": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "binary_covariance": covariance_cases(), "wrong_direction": same_wrong_direction_cases(),
              "quality_mixture_confound": conditional_mean_mismatch(),
              "same_identity_hard_constraint": same_identity_hard_event_constraint(),
              "soft_centering": soft_centering_identity(),
              "plugin_sign_trap_and_direct_mean": plugin_sign_trap_and_direct_mean()}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("Refusing to overwrite an existing receipt")
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
