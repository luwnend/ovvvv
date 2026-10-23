#!/usr/bin/env python3
"""Independent mathematical synthetic audit of fixed identity modules.

This file never reads real caches, labels, checkpoints or training outputs.
The independent oracles below use explicit enumeration, not subject helpers.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "tmp_analysis/identity_spacetime_modules_20261024.py"
RECOMMENDATION = ROOT / "tmp_analysis/identity_spacetime_followup_recommendation_20261024.md"
FINAL_SUBJECT_SHA = "2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3"
RECOMMENDATION_SHA = "23a1361129af172e201e10c0fefa82cf3e76c78ce640d824329652785f40f173"


def check(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def oracle_hierarchical_weights(records):
    """Equal supported source, trajectory, observation, then available lag."""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    unique = set()
    for index, row in enumerate(records):
        check(row["p"] > 0 and row["n"] > 0, "Oracle only accepts fixed p+n query-lags")
        key = (row["source"], row["video"], row["gt"], row["image"], row["qrow"], row["lag"])
        check(key not in unique, "Duplicate immutable query-lag")
        unique.add(key)
        tree[row["source"]][row["video"], row["gt"]][row["image"], row["qrow"]].append(index)
    weights = np.zeros(len(records), float)
    for trajectories in tree.values():
        for observations in trajectories.values():
            for lags in observations.values():
                value = 1 / (len(tree) * len(trajectories) * len(observations) * len(lags))
                for index in lags:
                    weights[index] = value
    return weights


def hierarchy_fixture():
    def row(source, video, gt, image, qrow, lag, p, n):
        return dict(source=source, video=video, gt=gt, image=image, qrow=qrow, lag=lag, p=p, n=n)
    return [row("A", 1, 7, 11, 0, 1, 1, 3), row("A", 1, 7, 11, 0, 4, 3, 1),
            row("A", 1, 7, 12, 0, 1, 2, 2), row("A", 1, 9, 13, 0, 4, 1, 1),
            row("B", 2, 7, 21, 0, 1, 4, 1)]


def oracle_normalizer(blocks):
    """Exact positive/negative edge marginals, then global per-slot validity."""
    observations = [[] for _ in range(9)]
    for block in blocks:
        positive, negative = list(block["positive"]), list(block["negative"])
        check(positive and negative, "Normalizer population must be p+n")
        for rows in (positive, negative):
            marginal = block["weight"] / (2 * len(rows))
            for row in rows:
                for slot in range(9):
                    if block["valid"][row, slot]:
                        observations[slot].append((float(block["values"][row, slot]), marginal))
    means, stds, masses = np.zeros(9), np.ones(9), np.zeros(9)
    for slot, values in enumerate(observations):
        if not values:
            continue
        mass = sum(weight for _, weight in values)
        mean = sum(value * weight for value, weight in values) / mass
        variance = sum((value - mean) ** 2 * weight for value, weight in values) / mass
        means[slot], stds[slot], masses[slot] = mean, max(np.sqrt(variance), 1e-6), mass
    return means, stds, masses


def normalizer_fixture():
    a = np.tile(np.array([0, 2, 4, 6], float)[:, None], (1, 9))
    b = np.tile(np.array([10, 14, 20], float)[:, None], (1, 9))
    av, bv = np.ones_like(a, bool), np.ones_like(b, bool)
    # Slot1 keeps one of three negatives, so valid entries retain their
    # original 1/3 negative marginal before global renormalization.
    av[2:, 1] = False
    bv[0, 1] = False
    # Slot2 globally absent; slot3 valid constant => independent std floor.
    av[:, 2] = False
    bv[:, 2] = False
    a[:, 3], b[:, 3] = 3, 3
    a[~av] = np.nan
    b[~bv] = np.nan
    return [dict(weight=.25, values=a, valid=av, positive=[0], negative=[1, 2, 3]),
            dict(weight=.75, values=b, valid=bv, positive=[0, 1], negative=[2])]


def oracle_donors(categories, history_valid, dump_rows):
    groups = defaultdict(list)
    for index, (category, valid) in enumerate(zip(categories, history_valid)):
        groups[int(category), bool(valid)].append(index)
    donors = np.arange(len(categories))
    for group in groups.values():
        ordered = sorted(group, key=lambda index: int(dump_rows[index]))
        if len(ordered) >= 2:
            for position, index in enumerate(ordered):
                donors[index] = ordered[(position - 1) % len(ordered)]
    return donors


def independent_oracle_self_test():
    weights = oracle_hierarchical_weights(hierarchy_fixture())
    np.testing.assert_array_equal(weights, [1/16, 1/16, 1/8, 1/4, 1/2])
    means, stds, masses = oracle_normalizer(normalizer_fixture())
    # Overall dim0 uses P/N halves: query1 mean2, query2 mean16, then .25/.75.
    check(abs(means[0] - 12.5) < 1e-14 and masses[0] == 1, "Marginal moment hand oracle")
    check(means[2] == 0 and stds[2] == 1 and stds[3] == 1e-6, "Missing/constant slot oracle")
    check(abs(masses[1] - (1/8 + 1/24 + 3/16 + 3/8)) < 1e-14, "Per-slot validity original marginal mass")
    return dict(hierarchy_weights=weights.tolist(), normalizer_means=means.tolist(), normalizer_stds=stds.tolist(), valid_marginal_mass=masses.tolist())


def subject_record(row):
    return dict(source=row["source"], video_id=row["video"], qgt=row["gt"], image_id=row["image"], qrow=row["qrow"], lag=row["lag"],
        positive_rows=np.arange(row["p"]), negative_rows=np.arange(row["p"], row["p"] + row["n"]))


def subject_blocks(blocks):
    return [dict(source="A" if index == 0 else "B", values=row["values"].copy(), masks=row["valid"].copy(),
        positive_rows=np.asarray(row["positive"]), negative_rows=np.asarray(row["negative"]), query_lag_weight=row["weight"])
        for index, row in enumerate(blocks)]


def audit_weights_and_normalizer(subject):
    rows = hierarchy_fixture()
    actual, coverage = subject.hierarchical_weights([subject_record(row) for row in rows], ["A", "B", "ZERO", "HELD"], ["A", "B", "ZERO"])
    expected = oracle_hierarchical_weights(rows)
    np.testing.assert_array_equal(actual, expected)
    check(coverage["supported_sources"] == ["A", "B"] and coverage["zero_support_train_sources"] == ["ZERO"], "Coverage omitted zero-support source")
    check(coverage["all_source_coverage"] == dict(A=4, B=1, ZERO=0, HELD=0), "All-source coverage changed")
    mutated = [subject_record(dict(row, p=row["p"]*3, n=row["n"]*5)) for row in rows]
    np.testing.assert_array_equal(subject.hierarchical_weights(mutated, ["A", "B", "ZERO"], ["A", "B", "ZERO"])[0], expected)
    duplicate = [subject_record(row) for row in rows] + [subject_record(rows[0])]
    try:
        subject.hierarchical_weights(duplicate, ["A", "B"], ["A", "B"])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate immutable query-lag multiplied loss")
    for forbidden in (dict(subject_record(rows[0]), source="HELD"), dict(subject_record(rows[0]), lag=2)):
        try:
            subject.hierarchical_weights([forbidden], ["A", "HELD"], ["A"])
        except ValueError:
            pass
        else:
            raise AssertionError("Held-out source/undeclared lag admitted")
    blocks = normalizer_fixture()
    means, stds, mass = oracle_normalizer(blocks)
    actual = subject.fit_geometry_normalizer(subject_blocks(blocks), ["A", "B"])
    np.testing.assert_allclose(actual.mean, means, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(actual.std, stds, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(actual.valid_marginal_weight, mass, atol=1e-14, rtol=1e-14)
    check(not actual.mean.flags.writeable and not actual.std.flags.writeable, "Train normalizer arrays can mutate")
    for block in subject_blocks(blocks):
        transformed = actual.transform(block["values"], block["masks"])
        check(np.equal(transformed[~block["masks"]], 0).all(), "Invalid normalized slots are not exact zero")
        np.testing.assert_allclose(transformed[block["masks"]], ((block["values"]-means)/stds)[block["masks"]], rtol=1e-14, atol=1e-14)
    # Unknown retained edge has arbitrarily large geometry and a valid mask;
    # its marginal must remain exactly zero and never alter moments.
    amended = subject_blocks(blocks)
    amended[0]["values"] = np.concatenate((amended[0]["values"], np.full((1, 9), 1e12)))
    amended[0]["masks"] = np.concatenate((amended[0]["masks"], np.ones((1, 9), bool)))
    comparison = subject.fit_geometry_normalizer(amended, ["A", "B"])
    np.testing.assert_array_equal(comparison.mean, actual.mean)
    np.testing.assert_array_equal(comparison.std, actual.std)
    try:
        subject.fit_geometry_normalizer(subject_blocks(blocks), ["A"])
    except ValueError:
        pass
    else:
        raise AssertionError("Held-out source normalized")
    calls = [0]
    def changing_stream():
        calls[0] += 1
        result = subject_blocks(blocks)
        if calls[0] == 2:
            result[0]["values"][0, 0] += 7
        return iter(result)
    try:
        subject.fit_geometry_normalizer(changing_stream, ["A", "B"])
    except ValueError:
        pass
    else:
        raise AssertionError("Normalizer silently accepted value drift between passes")
    return dict(hierarchical_records=len(rows), supported_sources=2, all_sources=4,
        imbalanced_PN_marginal_population_moments=True, dimensionwise_valid_mass_verified=True,
        duplicate_and_heldout_rejected=True, unknown_edges_weight_zero=True, normalizer_stream_mutation_rejected=True)


def audit_batch_gradient(subject):
    weights = [.9, .05, .05]
    slopes = [2., 4., 8.]
    def gradient(indices):
        theta = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
        losses = torch.stack([theta * slopes[index] for index in indices])
        value = subject.weighted_batch_loss(losses, [weights[index] for index in indices], 3)
        value.backward()
        return float(theta.grad)
    check(abs(gradient([1, 2]) - .9) < 1e-14, "Batch gradient used local weight normalization")
    check(abs(gradient([0]) - 5.4) < 1e-14, "Last short batch B used configured capacity")
    check(abs(gradient([0, 1, 2]) - 2.4) < 1e-14, "Full population gradient differs")
    for size in (1, 2):
        expected = np.mean([gradient(indices) for indices in itertools.combinations(range(3), size)])
        check(abs(expected - 2.4) < 1e-14, "Uniform batches are not an unbiased global weighted gradient")
    return dict(batch_last_two_gradient=.9, last_short_batch_gradient=5.4, full_population_gradient=2.4,
        uniform_subset_gradient_expectation=2.4, gradients_observed_by_autograd=True)


def audit_adapter_and_loss(subject):
    torch.manual_seed(91)
    raw = torch.randn((4, 256), dtype=torch.float64)
    raw[3] = 0
    adapter = subject.IdentityAdapter256(20261024)
    unit = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # The declared adapter normalizes its unit input again. Its mathematical
    # identity initialization therefore permits that second-normalization
    # rounding (unlike the required exactly-zero original-zero rows).
    torch.testing.assert_close(adapter(raw), unit, rtol=1e-14, atol=1e-15)
    check(adapter.first.weight.shape == (64, 256) and adapter.last.weight.shape == (256, 64), "Adapter architecture differs")
    check(torch.count_nonzero(adapter.first.bias) == 0 and torch.count_nonzero(adapter.last.weight) == 0 and torch.count_nonzero(adapter.last.bias) == 0, "Adapter initialization differs")
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.copy_(torch.randn_like(parameter))
    zero = torch.zeros((2, 256), dtype=torch.float64, requires_grad=True)
    outputs = adapter(zero)
    check(torch.equal(outputs, torch.zeros_like(outputs)), "Learned biases made zero raw nonzero")
    outputs.sum().backward()
    check(torch.equal(zero.grad, torch.zeros_like(zero)), "Zero input leaked adapter gradient")
    check(all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0 for parameter in adapter.parameters()), "Zero input leaked parameter gradient")
    # Produce actual stale gradients before freezing, matching staged use.
    adapter.zero_grad(set_to_none=True)
    adapter(raw).sum().backward()
    check(any(parameter.grad is not None for parameter in adapter.parameters()), "Stale-gradient test did not exercise staged adapter")
    subject.freeze_adapter(adapter)
    check(all(not parameter.requires_grad and parameter.grad is None for parameter in adapter.parameters()), "Freeze did not clear old gradient state")
    state = {key: value.clone() for key, value in adapter.state_dict().items()}
    cosine = subject.frozen_appearance_cosine(raw, raw, adapter)
    check(not cosine.requires_grad and cosine.grad_fn is None, "Geometry cosine is still connected to adapter")
    norm = subject.GeometryNormalizer(np.zeros(9), np.ones(9), np.ones(9))
    values, masks = np.zeros((4, 4, 9)), np.ones((4, 4, 9), bool)
    for mode in ("G0", "G1", "Gmask", "Gperm"):
        plan = subject.gperm_donors([1]*4, [True]*4, [0, 1, 2, 3]) if mode == "Gperm" else None
        inputs, _ = subject.head_inputs(cosine, values, masks, norm, mode, plan)
        head = subject.ResidualAssociationHead19(20261024)
        check(head.first.weight.shape == (32, 19) and head.last.weight.shape == (1, 32), "Head architecture differs")
        torch.testing.assert_close(head(inputs), cosine/.1, rtol=0, atol=0)
        head(inputs).sum().backward()
        check(all(parameter.grad is None for parameter in adapter.parameters()), "Geometry backprop reached frozen adapter")
    check(all(torch.equal(value, state[key]) for key, value in adapter.state_dict().items()), "Geometry arm changed same frozen adapter")
    try:
        subject.frozen_appearance_cosine(raw, raw, subject.IdentityAdapter256(2))
    except ValueError:
        pass
    else:
        raise AssertionError("Unfrozen adapter admitted geometry cosine")
    scores = torch.tensor([.1, .8, -.3, .2, 999., -999.], dtype=torch.float64, requires_grad=True)
    projection = subject.training_projection(7, [7, 7, 8, 9, -1, 10], 1, [1, 1, 1, 1, 1, 2])
    np.testing.assert_array_equal(projection["positive_rows"], [0, 1])
    np.testing.assert_array_equal(projection["negative_rows"], [2, 3])
    loss = subject.pairwise_loss(scores, projection["positive_rows"], projection["negative_rows"], .1)
    expected = sum(torch.nn.functional.softplus((scores[n]-scores[p])/.1) for p in [0, 1] for n in [2, 3])/4
    torch.testing.assert_close(loss, expected, rtol=1e-14, atol=1e-14)
    actual_gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, scores)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-14, atol=1e-14)
    check(actual_gradient[4] == 0 and actual_gradient[5] == 0, "Unknown/other-category row used in loss")
    check(not subject.training_projection(7, [-1, 8], 1, [1, 1])["hard"], "Missing-positive coverage became no-match label")
    return dict(identity_initialization_to_roundoff=True, learned_bias_zero_input_and_gradient_exact_zero=True,
        freeze_clears_stale_gradients=True, all_four_G_arms_frozen_adapter_gradient_block=True,
        head_zero_residual_exact=True, all_PxN_loss_and_gradient_oracle=True, unknown_and_other_category_gradient_zero=True)


def audit_gperm_and_slots(subject):
    categories, history, dump = [1, 1, 1, 2, 2, 3], [True, True, True, False, True, False], [20, 10, 30, 40, 50, 60]
    plan = subject.gperm_donors(categories, history, dump)
    expected = oracle_donors(categories, history, dump)
    np.testing.assert_array_equal(plan.donor_rows, expected)
    np.testing.assert_array_equal(expected, [1, 2, 0, 3, 4, 5])
    check(not plan.donor_rows.flags.writeable, "Permutation plan can mutate")
    values = np.arange(2*6*9, dtype=float).reshape(2, 6, 9)
    masks = np.ones_like(values, bool)
    masks[:, np.logical_not(history), 5:] = False
    values[~masks] = np.nan
    norm = subject.GeometryNormalizer(np.arange(9, dtype=float)/10, np.arange(1, 10, dtype=float), np.ones(9))
    s = torch.arange(12, dtype=torch.float64).reshape(2, 6)/13
    normalized = np.where(masks, (values-norm.mean)/norm.std, 0)
    position = [0, 1, 2, 3, 5, 6, 7]
    for mode in ("G0", "G1", "Gmask", "Gperm"):
        inputs, coverage = subject.head_inputs(s, values, masks, norm, mode, plan if mode == "Gperm" else None)
        np.testing.assert_array_equal(inputs[:, :, 0].numpy(), s.numpy())
        actual, actual_masks = inputs[:, :, 1:10].numpy(), inputs[:, :, 10:].numpy()
        if mode == "G0":
            check(np.equal(actual, 0).all() and np.equal(actual_masks, 0).all(), "G0 retained hidden time/history/mask")
        else:
            np.testing.assert_array_equal(actual_masks, masks.astype(float))
            np.testing.assert_array_equal(actual[:, :, [4, 8]], normalized[:, :, [4, 8]])
            truth = normalized.copy()
            if mode == "Gmask":
                truth[:, :, position] = 0
            elif mode == "Gperm":
                truth[:, :, position] = normalized[:, expected, :][:, :, position]
                # Reference row1 is UNKNOWN to loss. It must remain donor of
                # known row0; a known-first projection would instead use row2.
                check(actual[0, 0, 0] == normalized[0, 1, 0] and actual[0, 0, 0] != normalized[0, 2, 0],
                    "Unknown retained donor was removed before intervention")
                check(coverage == dict(mapping_nonidentity_references=3, mapping_nonidentity_candidates=6,
                    geometry_values_changed_candidates=6), "GT-free mapping/value coverage differs")
            np.testing.assert_array_equal(actual, truth)
    duplicate_values = np.ones_like(values)
    duplicate_masks = np.ones_like(masks)
    _, _, coverage = subject.intervene_geometry(duplicate_values, duplicate_masks, "Gperm", plan)
    check(coverage["mapping_nonidentity_candidates"] == 6 and coverage["geometry_values_changed_candidates"] == 0,
          "Mapping movement falsely asserted value or identity destruction")
    # A legal same-identity duplicate donor can remain same GT; Gperm must not
    # consult GT to avoid that case or advertise guaranteed wrong identities.
    equal_gt = [7, 7, 7, 9, 9, -1]
    check(all(equal_gt[index] == equal_gt[expected[index]] for index in range(3)), "Same-GT counterexample missing")
    geometry_path = ROOT / "tmp_analysis/relative_track_geometry.py"
    check(sha(geometry_path) == "c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a", "Reviewed geometry changed")
    geometry_module = module(geometry_path, "independent_pure_synthetic_reviewed_geometry")
    query_boxes = np.array([[4., 0., 6., 2.], [1., 1., 5., 5.]])
    reference_boxes = np.array([[0., 0., 2., 2.], [1., 1., 5., 5.]])
    previous = np.array([[-2., 0., 0., 2.], [np.nan]*4])
    geom = geometry_module.pair_geometry(query_boxes, reference_boxes, 20, np.array([10., 10.]), previous, np.array([0., np.nan]))
    slots, valid = subject.geometry_slots(geom, [10., 10.], [0., np.nan])
    check(slots.shape == valid.shape == (2, 2, 9), "Nine-value geometry shape")
    np.testing.assert_allclose(slots[0, 0], [2, 0, 0, 0, np.log(11), 1, 0, 0, np.log(11)], atol=1e-14, rtol=1e-14)
    check(valid[:, 0].all() and valid[:, 1, :5].all() and not valid[:, 1, 5:].any(), "History mask changed denominator")
    check(geom["predicted_box_iou"][1, 1] == 1 and np.isnan(slots[1, 1, 7]) and not valid[1, 1, 7], "Static overlap masqueraded as CV-IoU")
    check(np.array_equal(np.isfinite(slots), valid), "Geometry NaN/mask contract differs")
    return dict(full_retained_unknown_donor_preserved=True, immutable_row_cyclic_mapping_exact=True,
        own_time_and_masks_preserved=True, shared_normalizer_precedes_all_G_interventions=True,
        mapping_and_actual_value_coverage_separate=True, same_GT_donor_counterexample_retained=True,
        nine_geometry_slots_hand_checked=True, original_frame_timing=True, static_overlap_never_CV_IoU=True)


def main():
    torch.set_num_threads(1)
    check(FINAL_SUBJECT_SHA and sha(SUBJECT) == FINAL_SUBJECT_SHA, "Final modules SHA missing/changed")
    check(sha(RECOMMENDATION) == RECOMMENDATION_SHA, "Fixed recommendation changed")
    subject = module(SUBJECT, "independent_identity_spacetime_modules_subject")
    check(subject.CONTRACT_SHA == RECOMMENDATION_SHA, "Module contract mismatch")
    synthetic = {}
    for number in (1, 2, 3):
        path = ROOT / ("tmp_analysis/identity_spacetime_modules_synthetic_round%d_20261024_v2.json" % number)
        receipt = json.loads(path.read_text())
        check(receipt["status"] == "PASS_IDENTITY_SPACETIME_MODULES_SYNTHETIC" and receipt["round"] == number,
              "Missing synthetic round")
        check(receipt["script_sha256"] == FINAL_SUBJECT_SHA and receipt["contract_sha256"] == RECOMMENDATION_SHA and len(receipt["checks"]) >= 2,
              "Final synthetic round binding mismatch")
        synthetic[str(path)] = sha(path)
    oracle = independent_oracle_self_test()
    checks = dict(weight_and_normalizer=audit_weights_and_normalizer(subject), batch_gradient=audit_batch_gradient(subject),
        adapter_and_loss=audit_adapter_and_loss(subject), Gperm_and_geometry=audit_gperm_and_slots(subject))
    check(sha(SUBJECT) == FINAL_SUBJECT_SHA and sha(RECOMMENDATION) == RECOMMENDATION_SHA, "Inputs changed during audit")
    result = dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_MODULES_SYNTHETIC_AUDIT", checker_sha256=sha(__file__),
        module_sha256=FINAL_SUBJECT_SHA, recommendation_sha256=RECOMMENDATION_SHA,
        reviewed_subject_synthetic_round_files_sha256=synthetic, independent_hand_oracles=oracle, independent_checks=checks,
        real_cache_or_identity_opened=False, real_model_or_checkpoint_opened=False, optimizer_steps=0,
        training_runner_implemented=False, real_training_run=False, GPU_used=False,
        limits=["Pure synthetic module audit only; no runner, source-OOF supervision loader, real training or evaluation is authorized by this PASS"])
    output = ROOT / "tmp_analysis/identity_spacetime_modules_independent_audit_20261024.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()


