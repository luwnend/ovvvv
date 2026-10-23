"""Untrained appearance dynamics primitives; no tracker or label access.

The angular predictor is a fixed control, not a proposed novel method. The
low-rank operator is a mathematical building block, not a trained predictor.
Neither orthogonality nor a good synthetic forecast guarantees identity
preservation when different tracks receive different operators.
"""
import numpy as np
import torch


def angular_velocity_prediction(previous, last, previous_time, last_time,
                                query_time, history_valid):
    """Extrapolate each past feature arc using ORIGINAL frame-index intervals.

    All arrays except features have shape [M]; features are [M,D]. The caller
    must locate history without truth or future observations. `query_time` is
    the known timestamp; there is deliberately no query appearance/box input.
    No-history, zero previous feature, and numerically undefined antipodal arcs
    fall back to normalized `last`. Unassigned/ambiguous IDs must be masked by
    the caller. This function cannot detect a wrong but unique past ID.
    """
    previous = np.asarray(previous, dtype=np.float64)
    last = np.asarray(last, dtype=np.float64)
    if last.ndim != 2 or previous.shape != last.shape:
        raise ValueError("Expected matching [M,D] past and last feature arrays")
    size = len(last)
    values = [np.asarray(x) for x in
              (previous_time, last_time, query_time, history_valid)]
    if any(x.shape != (size,) for x in values):
        raise ValueError("Times and history mask must each have shape [M]")
    if values[-1].dtype != np.bool_:
        raise ValueError("History mask must be boolean")
    before, now, future = [x.astype(np.float64) for x in values[:3]]
    valid = values[-1].copy()
    norm = np.linalg.norm(last, axis=1)
    if not np.isfinite(last).all() or np.any(norm == 0):
        raise ValueError("Every last appearance must be finite and nonzero")
    if not np.isfinite(now).all() or not np.isfinite(future).all() or np.any(future < now):
        raise ValueError("Query timestamp must be finite and no earlier than last")
    if np.any(valid & (~np.isfinite(before) | (before >= now))):
        raise ValueError("A valid history must be strictly before the reference")
    if not np.isfinite(previous[valid]).all():
        raise ValueError("A valid previous appearance must be finite")
    current = last / norm[:, None]
    prior = np.zeros_like(last)
    pnorm = np.linalg.norm(previous[valid], axis=1)
    indices = np.flatnonzero(valid)
    zero = indices[pnorm == 0]
    valid[zero] = False
    nonzero = indices[pnorm > 0]
    prior[nonzero] = previous[nonzero] / pnorm[pnorm > 0, None]
    cosine = np.clip(np.sum(prior * current, axis=1), -1.0, 1.0)
    backwards_tangent = prior - cosine[:, None] * current
    sine = np.linalg.norm(backwards_tangent, axis=1)
    # Only a floating-point singularity criterion, never a learned motion gate.
    epsilon = 64 * np.finfo(np.float64).eps
    directed = valid & (sine > epsilon)
    coincident = valid & ~directed & (cosine >= 0)
    antipodal = valid & ~directed & ~coincident
    result = current.copy()
    ratio = np.zeros(size, dtype=np.float64)
    angle = np.zeros(size, dtype=np.float64)
    ratio[valid] = (future[valid] - now[valid]) / (now[valid] - before[valid])
    angle[valid] = np.arctan2(sine[valid], cosine[valid])
    turn = ratio[directed] * angle[directed]
    tangent = -backwards_tangent[directed] / sine[directed, None]
    result[directed] = (np.cos(turn)[:, None] * current[directed]
                        + np.sin(turn)[:, None] * tangent)
    return {
        "prediction": result,
        "directed_arc": directed,
        "coincident_static": coincident,
        "antipodal_fallback": antipodal,
        "missing_or_zero_history": ~valid,
        "history_angle_radians": angle,
        "extrapolation_ratio": ratio,
    }


def low_rank_orthogonal_transport(features, basis, generator, time_gap, valid):
    """Apply I + U (exp(dt A) - I) U^T to each row feature.

    `basis` is [D,R] with orthonormal columns, `generator` [M,R,R] is
    skew-symmetric, `time_gap` [M] is in the explicitly chosen time unit, and
    `valid` [M] is boolean. Invalid rows return the input exactly. A learned
    history encoder must produce A BEFORE seeing query features or positions;
    this primitive alone does not enforce that external dataflow contract.
    Time gaps are linear: log-time would break the fixed-generator semigroup.
    """
    if features.ndim != 2:
        raise ValueError("Features must have shape [M,D]")
    m, d = features.shape
    if basis.ndim != 2 or basis.shape[0] != d or not 0 < basis.shape[1] <= d:
        raise ValueError("Invalid [D,R] basis")
    r = basis.shape[1]
    if generator.shape != (m, r, r) or time_gap.shape != (m,) or valid.shape != (m,):
        raise ValueError("Generator, gaps or mask have incompatible shapes")
    if valid.dtype != torch.bool or features.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32/float64 features and a boolean validity mask")
    if any(t.device != features.device for t in (basis, generator, time_gap, valid)):
        raise ValueError("All inputs must share a device")
    if any(t.dtype != features.dtype for t in (basis, generator, time_gap)):
        raise ValueError("Floating inputs must share a dtype")
    if not torch.isfinite(features).all() or not torch.isfinite(basis).all():
        raise ValueError("Nonfinite appearance or basis")
    if not torch.isfinite(time_gap).all() or bool((time_gap < 0).any()):
        raise ValueError("Use finite nonnegative time gaps")
    active = generator[valid]
    tolerance = 128 * torch.finfo(features.dtype).eps * max(1, r)
    eye = torch.eye(r, dtype=features.dtype, device=features.device)
    if not torch.allclose(basis.T @ basis, eye, atol=tolerance, rtol=tolerance):
        raise ValueError("Basis columns must be orthonormal")
    if not torch.isfinite(active).all() or not torch.allclose(
            active + active.transpose(-1, -2), torch.zeros_like(active),
            atol=tolerance, rtol=tolerance):
        raise ValueError("Valid generators must be finite and skew-symmetric")
    safe = torch.where(valid[:, None, None], generator, torch.zeros_like(generator))
    rotation = torch.matrix_exp(time_gap[:, None, None] * safe)
    coordinates = features @ basis
    rotated = torch.bmm(rotation, coordinates.unsqueeze(-1)).squeeze(-1)
    moved = features + (rotated - coordinates) @ basis.T
    return torch.where(valid[:, None], moved, features)
