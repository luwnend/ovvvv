#!/usr/bin/env python3
"""Three synthetic challenge rounds; no cache, labels or training are opened."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

MODULE = Path(__file__).with_name('endpoint_excluded_scene_motion.py')
spec = importlib.util.spec_from_file_location('excluded_translation', MODULE)
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


def scene():
    xy = np.asarray([[0., 0.], [20., 10.], [40., 40.], [60., 15.], [80., 60.]])
    rb = np.concatenate((xy, xy + [10., 20.]), axis=1)
    qb = rb + [30., -7., 30., -7.]
    return qb, rb, np.eye(5), np.eye(5)


def evidence(q, r, qf, rf):
    return motion.candidate_translation_evidence(q, r, qf, rf, 0, 0)


def assert_missing(result):
    assert not result['fit']['available']
    assert result['fit']['translation'] is None
    assert result['normalized_innovation'] is None


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted')


def run(round_number):
    q, r, qf, rf = scene()
    checks = []
    original = evidence(q, r, qf, rf)
    if round_number == 1:
        assert original['fit']['anchor_count'] == 4
        np.testing.assert_allclose(original['fit']['translation'], [30., -7.])
        np.testing.assert_allclose(original['normalized_innovation'], [0., 0.])
        checks.append('heldout_endpoint_known_common_translation')

        q2, r2, qf2, rf2 = q.copy(), r.copy(), qf.copy(), rf.copy()
        q2[0] += [900., -200., 900., -200.]
        r2[0] += [-500., 100., -500., 100.]
        qf2[0], rf2[0] = [-8., 2., 3., 7., 0.], [4., 0., 1., -1., 9.]
        changed = evidence(q2, r2, qf2, rf2)
        assert original['fit'] == changed['fit']
        assert changed['normalized_innovation'] != original['normalized_innovation']
        checks.append('endpoint_feature_and_geometry_mutation_cannot_change_fit')

        q3 = q.copy()
        q3[1:] += [12., 6., 12., 6.]
        assert evidence(q3, r, qf, rf)['fit']['translation'] != original['fit']['translation']
        checks.append('other_object_motion_changes_estimate')

        # A scored endpoint can steal another row's best appearance neighbor.
        # Removing only selected endpoint edges after matching is insufficient.
        a = np.eye(4)
        a[1] = [.9, .1, 0., 0.]
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.eye(4)
        full = motion._unique_mutual(a @ b.T)
        after = [(i, j) for i, j, *_ in full if i != 0 and j != 0]
        before = [(i + 1, j + 1) for i, j, *_ in motion._unique_mutual(a[1:] @ b[1:].T)]
        assert (1, 1) not in after and (1, 1) in before
        checks.append('match_then_delete_counterexample')
    elif round_number == 2:
        empty = motion.common_translation(np.empty((0, 4)), r, np.empty((0, 5)), rf)
        assert not empty['available'] and empty['anchors'] == []
        checks.append('empty_support_has_no_identity_warp')

        assert_missing(evidence(q, r, qf * 0, rf))
        tied = evidence(q, r, np.ones_like(qf), np.ones_like(rf))
        assert_missing(tied)
        checks.append('zero_features_and_tied_matches_are_not_confident_anchors')

        short = evidence(q[:3], r[:3], qf[:3], rf[:3])
        assert_missing(short)
        assert short['fit']['anchor_count'] == 2
        checks.append('two_remaining_matches_are_insufficient')

        reject(lambda: motion.common_translation(q, r, qf, rf, excluded_query_rows=(True,)))
        reject(lambda: motion.common_translation(q, r, qf, rf, minimum_anchors=2))
        reject(lambda: motion.common_translation(q, r, qf, rf[:, :4]))
        reject(lambda: motion.common_translation(q[:, :3], r, qf, rf))
        reject(lambda: motion.common_translation(q, r, qf, rf, excluded_reference_rows=(5,)))
        checks.append('invalid_exclusions_shape_or_anchor_minimum_rejected')
    elif round_number == 3:
        offset = np.asarray([1000., -900., 1000., -900.])
        moved = evidence(q + offset, r + offset, qf, rf)
        assert moved['fit'] == original['fit']
        np.testing.assert_allclose(moved['normalized_innovation'], original['normalized_innovation'])
        scale = np.asarray([3., 2., 3., 2.])
        resized = evidence(q * scale, r * scale, qf, rf)
        np.testing.assert_allclose(resized['fit']['translation'], [90., -14.])
        np.testing.assert_allclose(resized['normalized_innovation'], original['normalized_innovation'])
        checks.append('coordinate_origin_and_axis_scale_equivariance')

        perm = np.asarray([0, 4, 2, 1, 3])
        shuffled = evidence(q[perm], r[perm], qf[perm], rf[perm])
        np.testing.assert_allclose(shuffled['fit']['translation'], original['fit']['translation'])
        checks.append('support_permutation_does_not_change_estimated_motion')

        outlier = q.copy()
        outlier[4] += [1e5, -1e5, 1e5, -1e5]
        robust = evidence(outlier, r, qf, rf)
        np.testing.assert_allclose(robust['fit']['translation'], [30., -7.])
        assert robust['fit']['anchor_count'] == 4
        checks.append('one_arbitrary_displacement_outlier_does_not_move_median')

        # Exactly the same observations admit a camera translation world and
        # a static-camera/co-moving-objects world. Stable fit cannot choose one.
        assert original['fit']['residual_mad_xy'] == [0., 0.]
        assert original['fit']['max_leave_one_anchor_shift'] == 0.
        assert original['fit']['physical_camera_motion_established'] is False
        checks.append('perfect_support_is_not_camera_or_identity_verification')

        huge = evidence(q, r, qf * 1e300, rf * 1e300)
        np.testing.assert_allclose(huge['fit']['translation'], original['fit']['translation'])
        checks.append('finite_large_features_normalize_without_overflow')

        extreme_boxes = np.tile([9e307, 9e307, 1e308, 1e308], (5, 1))
        extreme = evidence(extreme_boxes, extreme_boxes, qf, rf)
        assert extreme['fit']['translation'] == [0., 0.]
        assert extreme['normalized_innovation'] == [0., 0.]
        unrepresentable = np.tile([-1e308, 0., 1e308, 1.], (5, 1))
        reject(lambda: evidence(unrepresentable, r, qf, rf))
        checks.append('extreme_finite_geometry_remains_finite_or_explicitly_rejected')
    else:
        raise ValueError('Unknown synthetic round')
    return dict(status='PASS_ENDPOINT_EXCLUDED_SCENE_MOTION_SYNTHETIC',
                round=round_number, checks=checks,
                module_sha256=hashlib.sha256(MODULE.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                real_cache_opened=False, identity_values_opened=False,
                training_or_tracker_replay=False, efficacy_or_novelty_established=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.round)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
