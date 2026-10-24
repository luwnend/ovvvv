"""Ordinary raw-only endpoint-excluded scene residual for every Q x M pair.

No cache reader, classifier, identity, training, score/gate, or evaluation lives
here. Every candidate is retained, including non-MNN/zero/negative-cosine pairs.
One frozen CachedSceneMotion instance builds the canonical full float64 cosine
matrix. For each endpoint pair, same-proposal and directly positive-overlap
rows are excluded before fresh strict-positive unique reciprocal matching.

Translation/MAD use original-image pixels. Candidate scale is the coordinatewise
sqrt(query width/height) * sqrt(reference width/height). residual_dxdy is
(query centre - reference centre - donor translation) / candidate scale.
static_abs_L1 is defined even without donors; the donor-dependent quantities
are NaN when fewer than minimum_anchors survive. This is ordinary geometric
evidence and establishes no physical camera motion, identity, or novelty.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


CACHED_PATH = Path(__file__).with_name('cached_endpoint_scene_motion_20261024.py')
CACHED_SHA256 = '5eaf00fa1bd52ab0c5a3430fcdf90fd9609d4c17fdc3d81db06cf25bc41b0f32'


def _cached_class():
    if hashlib.sha256(CACHED_PATH.read_bytes()).hexdigest() != CACHED_SHA256:
        raise ValueError('Frozen canonical CachedSceneMotion changed')
    spec = importlib.util.spec_from_file_location('all_candidate_frozen_cached_scene', CACHED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CachedSceneMotion


def _proposals(value, count):
    value = np.asarray(value)
    if (value.shape != (count,) or value.dtype.kind not in 'iu'
            or np.any(value < 0) or np.any(value > np.iinfo(np.int64).max)):
        raise ValueError('Aligned nonnegative integer proposal locators required')
    result = np.array(value, dtype=np.int64, order='C', copy=True)
    result.setflags(write=False)
    return result


def strict_endpoint_exclusions(boxes, proposals):
    """Canonical sorted row sets, with direct overlaps only, no closure.

    Endpoint inclusion follows from positive own area and identical proposal.
    Touching boundaries do not overlap. Inputs must already be validated.
    """
    result = []
    for row in range(len(boxes)):
        overlap = np.all(np.minimum(boxes[:, 2:], boxes[row, 2:]) >
                         np.maximum(boxes[:, :2], boxes[row, :2]), axis=1)
        result.append(tuple(int(i) for i in np.flatnonzero(
            overlap | (proposals == proposals[row]))))
    return tuple(result)


def _finite(value, label, positive=False):
    if not np.isfinite(value).all() or (positive and np.any(value <= 0)):
        raise ValueError('Nonfinite or unrepresentable candidate geometry: ' + label)
    return value


def all_candidate_scene_residual(query_boxes, reference_boxes,
                                 query_raw, reference_raw,
                                 query_proposals, reference_proposals,
                                 *, minimum_anchors=3, memoize=True):
    """Return full candidate arrays and exact donor/exclusion audit locators.

    ``arrays`` contain [Q,M] scalars or [Q,M,2] vectors. ``fit_index`` locates
    one of ``fit_records``; equivalent exclusion sets share a fit when memoized.
    These records include exact rematched anchor pairs for audit, never labels.
    Input arrays are snapshotted by the frozen engine and never mutated.
    ``static_abs_L1`` and ``pair_scale_xy`` remain defined without donor support;
    available/count/fit_index remain explicit and all donor quantities are NaN.
    """
    if (isinstance(minimum_anchors, (bool, np.bool_))
            or not isinstance(minimum_anchors, (int, np.integer))
            or minimum_anchors < 3):
        raise ValueError('At least three donor anchors required')
    if not isinstance(memoize, (bool, np.bool_)):
        raise ValueError('memoize must be boolean')
    engine = _cached_class()(query_boxes, reference_boxes, query_raw, reference_raw)
    n, m = engine.query_count, engine.reference_count
    qp, rp = _proposals(query_proposals, n), _proposals(reference_proposals, m)
    # These are immutable float64 snapshots created by the pinned engine.
    qb, rb = engine._qb, engine._rb
    qe, re = strict_endpoint_exclusions(qb, qp), strict_endpoint_exclusions(rb, rp)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        displacement = _finite(engine._qc[:, None, :] - engine._rc[None, :, :], 'displacement')
        scale = _finite(np.sqrt(qb[:, 2:] - qb[:, :2])[:, None, :] *
                        np.sqrt(rb[:, 2:] - rb[:, :2])[None, :, :], 'scale', positive=True)
        static = _finite(np.abs(displacement / scale).sum(axis=2), 'static_abs_L1')
    arrays = dict(raw_cosine=engine._similarity.copy(),
                  available=np.zeros((n, m), dtype=np.bool_),
                  anchor_count=np.zeros((n, m), dtype=np.int64),
                  fit_index=np.empty((n, m), dtype=np.int64),
                  pair_scale_xy=scale, static_abs_L1=static,
                  translation_xy=np.full((n, m, 2), np.nan, dtype=np.float64),
                  residual_mad_xy=np.full((n, m, 2), np.nan, dtype=np.float64),
                  residual_dxdy=np.full((n, m, 2), np.nan, dtype=np.float64),
                  corrected_L1=np.full((n, m), np.nan, dtype=np.float64),
                  gain=np.full((n, m), np.nan, dtype=np.float64))
    memo, records = {}, []
    for i in range(n):
        for j in range(m):
            key = (qe[i], re[j])
            if memoize and key in memo:
                index = memo[key]
            else:
                # fit slices the ONE canonical full matrix, then recomputes
                # strict MNN on all remaining valid rows. No fullpool-h filter.
                fit = engine.fit(excluded_query_rows=qe[i],
                                 excluded_reference_rows=re[j],
                                 minimum_anchors=int(minimum_anchors))
                record = dict(excluded_query_rows=qe[i], excluded_reference_rows=re[j],
                              anchor_pairs=tuple((a['query_row'], a['reference_row']) for a in fit['anchors']),
                              available=fit['available'], anchor_count=fit['anchor_count'],
                              translation_xy=fit['translation'], residual_mad_xy=fit['residual_mad_xy'])
                index = len(records)
                records.append(record)
                if memoize:
                    memo[key] = index
            record = records[index]
            arrays['fit_index'][i, j] = index
            arrays['anchor_count'][i, j] = record['anchor_count']
            arrays['available'][i, j] = record['available']
            if record['available']:
                translation = np.asarray(record['translation_xy'], dtype=np.float64)
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    residual = _finite((displacement[i, j] - translation) / scale[i, j], 'residual_dxdy')
                    corrected = _finite(np.abs(residual).sum(), 'corrected_L1')
                    gain = _finite(static[i, j] - corrected, 'gain')
                arrays['translation_xy'][i, j] = translation
                arrays['residual_mad_xy'][i, j] = record['residual_mad_xy']
                arrays['residual_dxdy'][i, j] = residual
                arrays['corrected_L1'][i, j] = corrected
                arrays['gain'][i, j] = gain
    return dict(arrays=arrays, fit_records=tuple(records),
                query_exclusions=qe, reference_exclusions=re,
                candidate_shape=(n, m), candidate_count=n*m,
                donor_fits_computed=len(records), cosine_products_built=engine.cosine_products_built,
                minimum_anchors=int(minimum_anchors), memoized=bool(memoize),
                fullpool_h_used_to_filter_candidates=False,
                raw_boxes_and_proposals_only=True, score_or_gate_produced=False,
                physical_camera_motion_identity_or_novelty_established=False)
