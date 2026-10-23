"""Reference-only common image-plane translation from other detections.

No GT, learned parameters, tracker IDs, semantic labels, image decoding or
real-data entrypoint. This is an ordinary geometry control, not a camera-motion
estimator or a new tracking method. The candidate's two rows are removed BEFORE
appearance matching, fitting and support diagnostics. Other detections may
still depict the same physical object; row exclusion cannot establish that
their identities, errors or motions are independent.

The intentionally simple implementation handles one candidate at a time.
Do not run it over the full cache without a separately frozen runtime design.
"""
import numpy as np


def _validate(boxes, features):
    boxes = np.asarray(boxes, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or features.ndim != 2
            or len(boxes) != len(features) or features.shape[1] < 1
            or not np.isfinite(boxes).all()
            or not np.isfinite(features).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('Finite positive Nx4 boxes and aligned NxD features required')
    with np.errstate(over='ignore', invalid='ignore'):
        widths = boxes[:, 2:] - boxes[:, :2]
    if not np.isfinite(widths).all():
        raise ValueError('Box dimensions exceed finite float64 geometry')
    return boxes, features


def _finite_geometry(value):
    if not np.isfinite(value).all():
        raise ValueError('Geometry arithmetic exceeded finite float64 representation')
    return value


def _median_xy(values):
    # Average the middle pair without overflowing their sum.
    ordered = np.sort(values, axis=0)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return ordered[n // 2 - 1] / 2 + ordered[n // 2] / 2


def _normalized(features):
    # Rescale first to avoid overflow for finite but large feature values.
    peak = np.max(np.abs(features), axis=1, keepdims=True)
    scaled = np.divide(features, peak, out=np.zeros_like(features), where=peak > 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0), peak[:, 0] > 0


def _other_rows(count, excluded):
    return np.asarray([i for i in range(count) if i not in excluded], dtype=np.int64)


def _exclusions(rows, count):
    rows = tuple(rows)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or i < 0 or i >= count for i in rows):
        raise ValueError('Excluded rows must be valid integer indices')
    return set(int(i) for i in rows)


def _unique_mutual(similarity):
    """Strict unique positive reciprocal maxima; never break a tie by row order."""
    if not similarity.size:
        return []
    row_best = np.argmax(similarity, axis=1)
    col_best = np.argmax(similarity, axis=0)
    result = []
    for qi, ri in enumerate(row_best):
        value = similarity[qi, ri]
        if (value <= 0 or col_best[ri] != qi
                or np.count_nonzero(similarity[qi] == value) != 1
                or np.count_nonzero(similarity[:, ri] == value) != 1):
            continue
        row_alternatives = np.delete(similarity[qi], ri)
        col_alternatives = np.delete(similarity[:, ri], qi)
        # A missing competitor is missing, not evidence of infinite certainty.
        result.append((qi, int(ri), float(value),
                       None if not len(row_alternatives) else float(value - row_alternatives.max()),
                       None if not len(col_alternatives) else float(value - col_alternatives.max())))
    return result


def common_translation(query_boxes, reference_boxes, query_features,
                       reference_features, *, excluded_query_rows=(),
                       excluded_reference_rows=(), minimum_anchors=3):
    """Fit componentwise median query-minus-reference center displacement.

    Exclusions must be selected by immutable row locators before this call.
    Neither excluded endpoints nor their values enter matching or diagnostics.
    ``available`` means sufficient reciprocal matches for this numerical rule,
    never physical correctness, calibrated uncertainty or verified identity.
    Two-dimensional MAD and leave-one-anchor-out shifts describe support only.
    No residual trimming, geometric filtering or fallback identity warp occurs.
    """
    qb, qf = _validate(query_boxes, query_features)
    rb, rf = _validate(reference_boxes, reference_features)
    if qf.shape[1] != rf.shape[1]:
        raise ValueError('Feature dimensions differ')
    if (isinstance(minimum_anchors, (bool, np.bool_))
            or not isinstance(minimum_anchors, (int, np.integer))
            or minimum_anchors < 3):
        raise ValueError('At least three anchors required by this reference contract')
    qe = _exclusions(excluded_query_rows, len(qb))
    re = _exclusions(excluded_reference_rows, len(rb))
    qi, ri = _other_rows(len(qb), qe), _other_rows(len(rb), re)
    qn, qvalid = _normalized(qf[qi])
    rn, rvalid = _normalized(rf[ri])
    qi, ri, qn, rn = qi[qvalid], ri[rvalid], qn[qvalid], rn[rvalid]
    matches = _unique_mutual(qn @ rn.T)
    anchors = [dict(query_row=int(qi[i]), reference_row=int(ri[j]),
                    raw_cosine=cos, query_margin=qm, reference_margin=rm)
               for i, j, cos, qm, rm in matches]
    result = dict(available=False, reason='insufficient_reciprocal_anchors',
                  minimum_anchors=int(minimum_anchors), anchor_count=len(anchors),
                  anchors=anchors, translation=None, residual_mad_xy=None,
                  max_leave_one_anchor_shift=None, units='original_image_pixels',
                  physical_camera_motion_established=False,
                  excluded_query_rows=sorted(qe), excluded_reference_rows=sorted(re))
    if len(anchors) < minimum_anchors:
        return result
    qc = qb[:, :2] / 2 + qb[:, 2:] / 2
    rc = rb[:, :2] / 2 + rb[:, 2:] / 2
    with np.errstate(over='ignore', invalid='ignore'):
        displacements = _finite_geometry(np.asarray([
            qc[a['query_row']] - rc[a['reference_row']] for a in anchors]))
        translation = _median_xy(displacements)
        residual_mad = _median_xy(_finite_geometry(np.abs(displacements - translation)))
        leave_one = np.asarray([_median_xy(np.delete(displacements, i, axis=0))
                                for i in range(len(anchors))])
        leave_difference = _finite_geometry(leave_one - translation)
        max_shift = _finite_geometry(np.hypot(leave_difference[:, 0], leave_difference[:, 1])).max()
    result.update(available=True, reason='sufficient_reciprocal_anchors',
                  translation=translation.tolist(), residual_mad_xy=residual_mad.tolist(),
                  max_leave_one_anchor_shift=float(max_shift))
    return result


def candidate_translation_evidence(query_boxes, reference_boxes, query_features,
                                   reference_features, query_row, reference_row,
                                   *, extra_query_exclusions=(),
                                   extra_reference_exclusions=(), minimum_anchors=3):
    """Compute endpoint-excluded support and an object-size-normalized residual.

    The final residual legitimately uses the scored endpoints, whereas its
    fitted translation and support diagnostics exclude them. Extra exclusions
    can remove GT-free duplicate/history aliases under an external contract;
    no complete physical-identity exclusion can be inferred here.
    """
    qb, qf = _validate(query_boxes, query_features)
    rb, rf = _validate(reference_boxes, reference_features)
    _exclusions((query_row,), len(qb))
    _exclusions((reference_row,), len(rb))
    fit = common_translation(qb, rb, qf, rf,
        excluded_query_rows=(query_row,) + tuple(extra_query_exclusions),
        excluded_reference_rows=(reference_row,) + tuple(extra_reference_exclusions),
        minimum_anchors=minimum_anchors)
    innovation = None
    if fit['available']:
        q, r = qb[query_row], rb[reference_row]
        scale = np.sqrt(q[2:] - q[:2]) * np.sqrt(r[2:] - r[:2])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            innovation = _finite_geometry((q[:2] / 2 + q[2:] / 2
                - (r[:2] / 2 + r[2:] / 2) - np.asarray(fit['translation'])) / scale).tolist()
    return dict(fit=fit, normalized_innovation=innovation,
                innovation_available=fit['available'], score_or_gate_produced=False)
