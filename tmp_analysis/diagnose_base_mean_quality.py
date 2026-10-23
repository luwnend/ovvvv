#!/usr/bin/env python3
"""Predeclared CPU mean-error baselines, fit-group crossfit and tune evaluation.

Real fitting requires an explicit --execute flag. This is an observation-level
mean-error diagnostic, not a pair-dependence or covariance model. Test and class
holdout correctness outcomes are never read or fitted; test dumps are never
opened. Fit/tune frame NPZs can also contain class-holdout object predictions,
but features are extracted only for eligible supervised rows. Synthetic tests
require no support.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6
POSTERIOR_EPS = 1e-12
RIDGE = 1.0
MAX_FOLDS = 5
FOLD_SEED = "mean_quality_source_group_v1"
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPLICATES = 1000
FEATURE_NAMES = [branch + "_" + feature for branch in ("fused", "text", "image")
                 for feature in ("foreground_entropy_normalized", "foreground_top1", "foreground_margin")]
FEATURE_NAMES += ["text_image_JS_normalized", "raw_detector_score", "box_normalized_area",
                  "box_log_aspect", "fused_background_probability"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def specification():
    return dict(status="PREDECLARED_NOT_REAL_FIT", features=FEATURE_NAMES,
                target="Primary error field: actual emitted det_labels+1 differs from eligible manual-anchor GT category; posterior argmax never replaces this target",
                B0="Training-subset hierarchical-weight mean emitted error, clipped to [1e-6,1-1e-6]",
                B1="One class-shared ridge-logistic model, weighted mean BCE + 1/2*lambda*sum(feature_coefficient**2); unpenalized intercept",
                ridge_lambda=RIDGE, feature_standardization="Training-subset hierarchical weighted mean/std; std<1e-12 replaced by 1; no tune statistics",
                training_weight="Equal source group -> equal eligible official GT trajectory (video_id,gt_track_id) inside group -> equal observation inside trajectory. Recomputed in every fold training subset.",
                feature_exclusions=["GT/predicted class IDs", "GT boxes/IoU", "GT track IDs", "source group IDs", "video/track IDs", "manual anchor IDs", "error/soft_risk"],
                metadata_use="Source group only for split/weights/bootstrap; eligible GT trajectory only for training/evaluation weights, never a numeric predictor",
                raw_detector_score="pretracker_det_bboxes[pretracker_det_row[dump_row],4], before tracker momentum smoothing; never substitute det_bboxes[dump_row,4]",
                feature_transforms="Foreground-only normalization before entropy/top1/margin; JS(text,image)/log(2); area clip(width*height/(original_image_width*height),0,1); log aspect clipped [-12,12]; original fused background probability retained",
                fit_crossfit="At most 5 deterministic source-group folds: fixed-seed SHA256 group order, round-robin assignment. Each group's predictions use only other fit groups. Fit/tune group overlap rejected.",
                folds_max=MAX_FOLDS, fold_seed=FOLD_SEED,
                tune="Predictions from final all-fit model, evaluation only; never select lambda, features, threshold, calibration or model based on tune",
                degenerate_strategy="If training outcomes contain one class, B1 falls back to B0 with zero feature coefficients; if fewer than 2 fit source groups, stop instead of producing leaking crossfit",
                evaluation="Brier and BCE plus calibration-in-the-large intercept and joint logistic calibration intercept/slope; micro and identical hierarchical source/trajectory/observation weighting",
                calibration="Descriptive outcome regression only; never recalibrates predictions. Constant logit -> joint intercept/slope not identifiable; one outcome class -> calibration NA. Failed/unbounded joint fit -> NA.",
                bootstrap=dict(unit="source_group", replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                               weights="Sample source groups with replacement. In weighted metrics each sampled group copy has equal mass, then equal eligible GT trajectories, then equal observations. Micro metric uses repeated raw observations.",
                               interpretation="Percentile interval conditional on already fitted crossfit/final-model predictions; models are not refit in bootstrap. Not a training uncertainty interval."),
                scope="Mean observation error only, not pair covariance, independent samples, causal dependence or generalization evidence",
                control_capacity_limit="Fixed lambda=1 B1 is a low-capacity mean-quality sanity control. No gain does not establish that quality is unpredictable, and cannot qualify a pair model. Stronger quality controls with inner source-group CV remain missing.",
                test="CLOSED; no test dump or error outcome is read", class_holdout="CLOSED; only support-certified readout-fit classes accepted")


def quality_features(saved, row):
    """Takes only inference-visible arrays and a raw row; no target argument."""
    ids = np.asarray(saved["posterior_category_ids"])
    fg = np.flatnonzero(ids > 0)
    require(len(fg) > 1 and ids[-1] == -1, "Unsupported posterior background schema")
    values = []
    probabilities = {}
    for branch in ("fused", "text", "image"):
        p = np.asarray(saved["p_" + branch][row, fg], np.float64)
        require(np.isfinite(p).all() and np.all(p >= 0) and p.sum() > POSTERIOR_EPS, "Invalid foreground probability")
        p = p / p.sum()
        probabilities[branch] = p
        top = np.partition(p, -2)[-2:]
        entropy = -float(np.sum(p * np.log(np.clip(p, POSTERIOR_EPS, None)))) / np.log(len(p))
        values.extend([entropy, float(top.max()), float(top.max() - top.min())])
    a, b = probabilities["text"], probabilities["image"]
    m = (a + b) / 2
    js = .5 * (np.sum(a * np.log(np.maximum(a, POSTERIOR_EPS) / np.maximum(m, POSTERIOR_EPS))) +
                np.sum(b * np.log(np.maximum(b, POSTERIOR_EPS) / np.maximum(m, POSTERIOR_EPS)))) / np.log(2)
    localrow = int(saved["pretracker_det_row"][row])
    require(localrow >= 0 and localrow < len(saved["pretracker_det_bboxes"]), "Invalid raw detector-score row mapping")
    raw_score = float(saved["pretracker_det_bboxes"][localrow, 4])
    require(np.isfinite(raw_score) and 0 <= raw_score <= 1, "Invalid raw detector score")
    box = np.asarray(saved["det_bboxes"][row, :4], np.float64)
    width, height = max(float(box[2] - box[0]), 0.), max(float(box[3] - box[1]), 0.)
    shape = np.asarray(saved["ori_shape"])
    require(shape[0] > 0 and shape[1] > 0, "Invalid original image shape")
    area = float(np.clip(width * height / (float(shape[0]) * float(shape[1])), 0, 1))
    aspect = float(np.clip(np.log(max(width, POSTERIOR_EPS) / max(height, POSTERIOR_EPS)), -12, 12))
    background = float(saved["p_fused"][row, -1])
    require(np.isfinite(background) and 0 <= background <= 1, "Invalid fused background probability")
    values.extend([float(js), raw_score, area, aspect, background])
    result = np.asarray(values, np.float64)
    require(result.shape == (len(FEATURE_NAMES),) and np.isfinite(result).all(), "Nonfinite quality summary")
    return result


def hierarchy_weights(groups, trajectories):
    """IDs affect weighting only; no outcome/class or feature-dependent weights."""
    require(len(groups) == len(trajectories) > 0, "Empty weighting subset")
    group_tracks = defaultdict(lambda: defaultdict(list))
    for i, (group, track) in enumerate(zip(groups, trajectories)):
        group_tracks[str(group)][str(track)].append(i)
    weights = np.zeros(len(groups), np.float64)
    for tracks in group_tracks.values():
        for rows in tracks.values():
            weights[rows] = 1. / len(group_tracks) / len(tracks) / len(rows)
    require(np.isclose(weights.sum(), 1) and np.all(weights > 0), "Invalid hierarchical weights")
    return weights


def fixed_group_folds(groups):
    unique = sorted(set(map(str, groups)), key=lambda g: hashlib.sha256((FOLD_SEED + "\0" + g).encode()).hexdigest())
    require(len(unique) >= 2, "Need at least two distinct fit source groups for honest group crossfit")
    k = min(MAX_FOLDS, len(unique))
    mapping = {group: i % k for i, group in enumerate(unique)}
    return np.asarray([mapping[str(group)] for group in groups], np.int64), mapping


def weighting_diagnostics(groups, trajectories):
    w = hierarchy_weights(groups, trajectories)
    group_mass, trajectory_mass = defaultdict(float), defaultdict(float)
    for group, trajectory, weight in zip(groups, trajectories, w):
        group_mass[str(group)] += float(weight)
        trajectory_mass[(str(group), str(trajectory))] += float(weight)
    return dict(source_groups=len(group_mass), eligible_GT_trajectories=len(trajectory_mass),
                Kish_effective_source_groups=float(1 / sum(v * v for v in group_mass.values())),
                Kish_effective_trajectories=float(1 / sum(v * v for v in trajectory_mass.values())),
                Kish_effective_observations=float(1 / (w @ w)),
                interpretation="Weight concentration summaries only; these are not counts of statistically independent samples")


def fit_mean_model(x, y, groups, trajectories):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    w = hierarchy_weights(groups, trajectories)
    require(x.ndim == 2 and x.shape[1] == len(FEATURE_NAMES) and y.shape == (len(x),), "Bad training array")
    require(np.isin(y, [0, 1]).all() and np.isfinite(x).all(), "Bad training outcomes/features")
    p0 = float(np.clip(w @ y, EPS, 1 - EPS))
    mean = w @ x
    scale = np.sqrt(w @ ((x - mean) ** 2))
    constant = scale < 1e-12
    scale[constant] = 1.
    z = (x - mean) / scale
    coef = np.zeros(len(FEATURE_NAMES) + 1, np.float64)
    coef[0] = float(logit(p0))
    if len(np.unique(y)) == 1:
        optimizer = dict(status="DEGENERATE_OUTCOME_B0_FALLBACK", iterations=0, success=True)
    else:
        design = np.c_[np.ones(len(z)), z]
        def objective(beta):
            eta = design @ beta
            loss = float(w @ (np.logaddexp(0, eta) - y * eta) + .5 * RIDGE * (beta[1:] @ beta[1:]))
            gradient = design.T @ (w * (expit(eta) - y))
            gradient[1:] += RIDGE * beta[1:]
            return loss, gradient
        opt = minimize(objective, coef, method="L-BFGS-B", jac=True,
                       options=dict(maxiter=200, ftol=1e-12, gtol=1e-9, maxls=40))
        require(opt.success and np.isfinite(opt.x).all(), "Fixed ridge-logistic optimizer failed: " + str(opt.message))
        coef = opt.x
        optimizer = dict(status="FIT_FIXED_RIDGE_LOGISTIC", iterations=int(opt.nit), success=bool(opt.success),
                         loss=float(opt.fun), max_abs_gradient=float(np.abs(opt.jac).max()))
    return dict(mean=mean, scale=scale, coefficient=coef, B0=p0,
                constant_features=[FEATURE_NAMES[j] for j in np.flatnonzero(constant)], optimizer=optimizer,
                training_observations=len(y), training_source_groups=len(set(map(str, groups))),
                training_eligible_GT_trajectories=len(set(zip(map(str, groups), map(str, trajectories)))),
                training_errors=int(y.sum()), training_correct=int(len(y) - y.sum()),
                training_single_outcome_class=bool(len(np.unique(y)) == 1),
                training_weighting=weighting_diagnostics(groups, trajectories))


def predict(model, x):
    x = np.asarray(x, np.float64)
    z = (x - model["mean"]) / model["scale"]
    return np.clip(expit(model["coefficient"][0] + z @ model["coefficient"][1:]), EPS, 1 - EPS)


def crossfit(x, y, groups, trajectories):
    fold, mapping = fixed_group_folds(groups)
    predictions = np.full((len(y), 2), np.nan)
    models = []
    groups = np.asarray(groups); trajectories = np.asarray(trajectories)
    for f in sorted(set(fold.tolist())):
        train, validation = fold != f, fold == f
        require(not set(groups[train]).intersection(groups[validation]), "Source group leaked between crossfit training and validation")
        model = fit_mean_model(x[train], y[train], groups[train], trajectories[train])
        predictions[validation, 0] = model["B0"]
        predictions[validation, 1] = predict(model, x[validation])
        models.append(dict(fold=int(f), train_groups=sorted(set(groups[train].tolist())),
                           validation_groups=sorted(set(groups[validation].tolist())),
                           validation_observations=int(validation.sum()), validation_errors=int(y[validation].sum()),
                           validation_correct=int(validation.sum() - y[validation].sum()),
                           validation_single_outcome_class=bool(len(np.unique(y[validation])) == 1),
                           validation_weighting=weighting_diagnostics(groups[validation], trajectories[validation]), model=model))
    require(np.isfinite(predictions).all(), "Crossfit failed to predict every fit observation")
    return predictions, fold, mapping, models


def calibrations(y, p, weight):
    y, p, w = np.asarray(y, np.float64), np.clip(p, EPS, 1 - EPS), np.asarray(weight, np.float64)
    w = w / w.sum()
    if len(np.unique(y[w > 0])) < 2:
        return dict(calibration_in_the_large_intercept=None, calibration_intercept=None, calibration_slope=None,
                    calibration_status="ONE_OUTCOME_CLASS_NOT_IDENTIFIABLE")
    eta = logit(p)
    # Offset-logit calibration-in-the-large (slope fixed at one).
    low, high = -60., 60.
    for _ in range(80):
        middle = (low + high) / 2
        if w @ expit(eta + middle) > w @ y:
            high = middle
        else:
            low = middle
    intercept_offset = (low + high) / 2
    if float(w @ ((eta - w @ eta) ** 2)) < 1e-12:
        return dict(calibration_in_the_large_intercept=float(intercept_offset), calibration_intercept=None,
                    calibration_slope=None, calibration_status="CONSTANT_LOGIT_JOINT_CALIBRATION_NOT_IDENTIFIABLE")
    design = np.c_[np.ones(len(y)), eta]
    def objective(beta):
        v = design @ beta
        return float(w @ (np.logaddexp(0, v) - y * v)), design.T @ (w * (expit(v) - y))
    opt = minimize(objective, np.asarray([0., 1.]), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=150, ftol=1e-11, gtol=1e-8, maxls=30))
    if not opt.success or not np.isfinite(opt.x).all() or np.max(np.abs(opt.x)) > 50:
        return dict(calibration_in_the_large_intercept=float(intercept_offset), calibration_intercept=None,
                    calibration_slope=None, calibration_status="JOINT_CALIBRATION_FAILED_OR_UNBOUNDED")
    return dict(calibration_in_the_large_intercept=float(intercept_offset), calibration_intercept=float(opt.x[0]),
                calibration_slope=float(opt.x[1]), calibration_status="OK")


def metrics(y, p, weights):
    weights = np.asarray(weights, np.float64); weights /= weights.sum()
    p = np.clip(p, EPS, 1 - EPS)
    return dict(observed_mean_error=float(weights @ y), predicted_mean_error=float(weights @ p),
                Brier=float(weights @ ((p - y) ** 2)),
                BCE=float(weights @ (-y * np.log(p) - (1 - y) * np.log1p(-p))),
                **calibrations(y, p, weights))


def summarize_predictions(y, predictions, groups, trajectories, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    names = ["B0_global_mean", "B1_shared_quality"]
    hierarchy = hierarchy_weights(groups, trajectories)
    point = {kind: {name: metrics(y, predictions[:, j], weights) for j, name in enumerate(names)}
             for kind, weights in (("micro", np.ones(len(y)) / len(y)), ("source_trajectory_observation_weighted", hierarchy))}
    for kind in point:
        point[kind]["B1_minus_B0"] = {key: point[kind][names[1]][key] - point[kind][names[0]][key]
                                             for key in ("Brier", "BCE")}
    unique = sorted(set(map(str, groups)))
    group_rows = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scalar_keys = ["observed_mean_error", "predicted_mean_error", "Brier", "BCE",
                   "calibration_in_the_large_intercept", "calibration_intercept", "calibration_slope"]
    draws = {kind: {name: defaultdict(list) for name in names + ["B1_minus_B0"]} for kind in point}
    for _ in range(bootstrap_replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        # Duplicated groups remain separate equal-mass bootstrap copies.
        selected = np.concatenate([group_rows[unique[g]] for g in sampled])
        weighted = np.concatenate([hierarchy[group_rows[unique[g]]] /
                                   hierarchy[group_rows[unique[g]]].sum() / len(sampled) for g in sampled])
        for kind, weights in (("micro", np.ones(len(selected)) / len(selected)),
                              ("source_trajectory_observation_weighted", weighted)):
            computed = []
            for j, name in enumerate(names):
                value = metrics(y[selected], predictions[selected, j], weights)
                computed.append(value)
                for key in scalar_keys:
                    if value[key] is not None:
                        draws[kind][name][key].append(value[key])
            for key in ("Brier", "BCE"):
                draws[kind]["B1_minus_B0"][key].append(computed[1][key] - computed[0][key])
    interval = {kind: {name: {key: dict(low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                                             valid_draws=len(values)) for key, values in values_by_key.items() if values}
                       for name, values_by_key in names_to_values.items()} for kind, names_to_values in draws.items()}
    return dict(point=point, source_group_bootstrap_95_percentile=interval,
                bootstrap_replicates=bootstrap_replicates, bootstrap_source_groups=len(unique),
                bootstrap_interpretation="Descriptive percentile intervals conditional on already fitted predictions; no baseline refitting, no training uncertainty or independence claim")


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def synthetic_test():
    rng = np.random.default_rng(55)
    n = 240
    groups = np.asarray(["synthetic_group%d" % (i // 30) for i in range(n)])
    trajectories = np.asarray(["video0:track%d" % (i // 5) for i in range(n)])
    x = rng.normal(size=(n, len(FEATURE_NAMES))); x[:, -1] = 0
    y = (rng.random(n) < expit(.2 + .9 * x[:, 0] - .7 * x[:, 2])).astype(float)
    predicted, fold, mapping, models = crossfit(x, y, groups, trajectories)
    held = fold == 0
    changed = y.copy(); changed[held] = 1 - changed[held]
    changed_predictions, _, _, _ = crossfit(x, changed, groups, trajectories)
    require(np.array_equal(predicted[held], changed_predictions[held]), "Held-group outcomes leaked into their crossfit predictions")
    w = hierarchy_weights(groups, trajectories)
    require(all(np.isclose(w[groups == g].sum(), 1 / len(set(groups))) for g in set(groups)), "Groups not equal mass")
    unequal_groups = np.asarray(["A", "B", "B", "B", "B", "B", "B"])
    unequal_trajectories = np.asarray(["a1", "b1", "b2", "b2", "b2", "b2", "b2"])
    unequal_weights = hierarchy_weights(unequal_groups, unequal_trajectories)
    require(np.allclose(unequal_weights, [.5, .25, .05, .05, .05, .05, .05]),
            "Unequal source/trajectory lengths broke hierarchical weights")
    unequal_diagnostics = weighting_diagnostics(unequal_groups, unequal_trajectories)
    require(np.isclose(unequal_diagnostics["Kish_effective_source_groups"], 2) and
            np.isclose(unequal_diagnostics["Kish_effective_trajectories"], 1 / .375),
            "Effective-weight summaries failed")
    fallback = fit_mean_model(x[:20], np.zeros(20), groups[:20], trajectories[:20])
    require(np.allclose(predict(fallback, x[:20]), EPS) and fallback["optimizer"]["status"].startswith("DEGENERATE"), "One-class fallback failed")
    require(FEATURE_NAMES[-1] in models[0]["model"]["constant_features"], "Constant predictor not recorded")
    c = 4; p = rng.random((2, c + 1)); p /= p.sum(1, keepdims=True)
    image = rng.random((2, c + 1)); image[:, -1] = 0; image /= image.sum(1, keepdims=True)
    fused = p * image; fused /= fused.sum(1, keepdims=True)
    saved = dict(posterior_category_ids=np.r_[np.arange(1,c+1),-1], p_text=p, p_image=image, p_fused=fused,
                 det_bboxes=np.asarray([[0,0,20,30,.9],[5,6,22,33,.8]]),
                 pretracker_det_bboxes=np.asarray([[0,0,20,30,.01],[5,6,22,33,.02]]),
                 pretracker_det_row=np.asarray([1,0]), ori_shape=np.asarray([100,200,3]))
    feature = quality_features(saved, 0)
    require(feature[FEATURE_NAMES.index("raw_detector_score")] == .02, "Used smoothed tracker score instead of mapped raw detector score")
    modified = dict(saved, det_labels=np.asarray([999,888]), track_ids=np.asarray([777,666]),
                    gt_category_id=99, error=0, source_group="forbidden_numeric_input")
    require(np.array_equal(feature, quality_features(modified, 0)), "Forbidden label/identity fields changed predictors")
    summary = summarize_predictions(y, predicted, groups, trajectories, bootstrap_replicates=12)
    require(summary["bootstrap_source_groups"] == 8 and np.isfinite(predicted).all(), "Synthetic summary failed")
    return dict(status="PASS_CPU_SYNTHETIC_MEAN_QUALITY_CONTRACTS", real_support_opened=False, model_or_GPU_used=False,
                held_group_outcome_flip_leaves_own_predictions_bitwise_equal=True,
                equal_group_trajectory_observation_weights=True, unequal_group_and_trajectory_lengths=True,
                weighting_effective_count_summaries=True, one_class_fallback=True,
                constant_feature_report=True, raw_detector_score_mapping=True, forbidden_identity_label_fields_ignored=True,
                fixed_source_group_bootstrap_contract=True, real_data_fitting_started=False)


def load_support(support):
    receipt_path = support / "support_receipt.json"
    receipt = json.load(receipt_path.open())
    require(receipt["status"] == "PASS_FIT_TUNE_SIGNAL_SUPPORT_AUDIT", "Support is not independently sealed")
    require(receipt["test_outcomes_computed"] is False and receipt["class_holdout_outcomes_computed"] is False, "Closed outcomes opened by support")
    source_hashes = {str(receipt_path): sha256(receipt_path)}
    for name in ("supervised_observations.jsonl", "gt_free_observations.jsonl"):
        path = support / name
        require(sha256(path) == receipt["output_files"][name], "Support row file changed: " + name)
        source_hashes[str(path)] = receipt["output_files"][name]
    run = Path(receipt["run"])
    inventory_path = run / "dump_inventory.json"
    require(str(inventory_path) in receipt["inputs"] and sha256(inventory_path) == receipt["inputs"][str(inventory_path)], "Sealed dump inventory changed")
    source_hashes[str(inventory_path)] = sha256(inventory_path)
    inventory = json.load(inventory_path.open())
    rows = []
    with (support / "supervised_observations.jsonl").open() as handle:
        for line in handle:
            raw = json.loads(line)
            require(raw["split"] in ("fit", "tune"), "Test/holdout supervision is forbidden")
            require(raw["eligible_for_parameter_fit"] == (raw["split"] == "fit"), "Parameter-fit eligibility mismatch")
            require(raw["error"] in (0,1) and raw["error"] == int(raw["pred_category_id"] != raw["gt_category_id"]), "Primary target is not emitted-label error")
            # Class identity is used only for this target-integrity/mismatch audit,
            # then dropped. It never enters the predictor array or fitting code.
            row = {k:raw[k] for k in ("observation_id","split","source_group","video_id","image_id","local_track_id","gt_track_id","error")}
            row["emitted_category_id_for_integrity_only"] = raw["pred_category_id"]
            row["label_argmax_disagrees"] = raw["pred_category_id"] != raw["posterior_argmax_category_id"]
            row["primary_vs_argmax_error_disagrees"] = bool(raw["error"]) != (raw["posterior_argmax_category_id"] != raw["gt_category_id"])
            rows.append(row)
    require(rows and len({r["observation_id"] for r in rows}) == len(rows), "Missing or duplicate supervised observations")
    required = {r["observation_id"]:r for r in rows}
    # This pre-existing all-emitted file includes test *indices*, but only open
    # supervised observation IDs are retained; no test dump is ever opened.
    with (support / "gt_free_observations.jsonl").open() as handle:
        for line in handle:
            raw = json.loads(line)
            if raw["observation_id"] not in required:
                continue
            row = required[raw["observation_id"]]
            for key in ("video_id","image_id","local_track_id","split","source_group"):
                require(raw[key] == row[key], "GT-free/supervised row identity mismatch")
            require(raw["dump_sha256"] == inventory[raw["dump_file"]], "Observation dump hash does not match sealed inventory")
            row.update(dump_file=raw["dump_file"],dump_row=int(raw["dump_row"]))
    fit_groups = {r["source_group"] for r in rows if r["split"] == "fit"}
    tune_groups = {r["source_group"] for r in rows if r["split"] == "tune"}
    require(fit_groups and tune_groups and not fit_groups.intersection(tune_groups), "Fit/tune source groups overlap or are empty")
    by_file = defaultdict(list)
    for i,row in enumerate(rows):
        require("dump_file" in row, "Supervised observation absent from GT-free index")
        by_file[row["dump_file"]].append((i,row))
    x = np.empty((len(rows),len(FEATURE_NAMES)),np.float64)
    for filename, indexed in by_file.items():
        path = run / "dump" / filename
        require(sha256(path) == inventory[filename], "Open fit/tune dump changed after support sealing")
        source_hashes[str(path)] = inventory[filename]
        with np.load(path,allow_pickle=False) as saved:
            for i,row in indexed:
                j=row["dump_row"]
                require(bool(saved["emitted_track_mask"][j]) and int(saved["track_ids"][j]) == row["local_track_id"], "Raw support row mismatch")
                require(int(saved["video_id"]) == row["video_id"] and int(saved["image_id"]) == row["image_id"], "Raw support frame mismatch")
                require(int(saved["det_labels"][j]) + 1 == row["emitted_category_id_for_integrity_only"], "Raw emitted-label/support target mismatch")
                x[i]=quality_features(saved,j)
    return rows,x,source_hashes


def run_real(args):
    require(args.execute,"Real support fitting is disabled by default; root must explicitly authorize --execute")
    require(not args.out.exists(),"Use a new output directory")
    started=time.time();rows,x,sources=load_support(args.support.resolve())
    y=np.asarray([r["error"] for r in rows],np.float64)
    split=np.asarray([r["split"] for r in rows]);groups=np.asarray([str(r["source_group"]) for r in rows])
    trajectories=np.asarray(["%d:%d"%(r["video_id"],r["gt_track_id"]) for r in rows])
    fit,tune=split=="fit",split=="tune"
    fit_pred,fold,mapping,models=crossfit(x[fit],y[fit],groups[fit],trajectories[fit])
    final=fit_mean_model(x[fit],y[fit],groups[fit],trajectories[fit])
    tune_pred=np.c_[np.full(int(tune.sum()),final["B0"]),predict(final,x[tune])]
    summaries={"fit_source_group_crossfit":summarize_predictions(y[fit],fit_pred,groups[fit],trajectories[fit]),
               "tune_evaluation_only":summarize_predictions(y[tune],tune_pred,groups[tune],trajectories[tune])}
    support_counts={}
    for name,mask in (("fit",fit),("tune",tune)):
        support_counts[name]=dict(observations=int(mask.sum()),source_groups=len(set(groups[mask])),
                                  eligible_GT_trajectories=len(set(zip(groups[mask],trajectories[mask]))),
                                  errors=int(y[mask].sum()),correct=int(mask.sum()-y[mask].sum()),
                                  micro_mean_error=float(y[mask].mean()),hierarchical_mean_error=float(hierarchy_weights(groups[mask],trajectories[mask])@y[mask]),
                                  emitted_label_vs_posterior_argmax_disagreements=sum(rows[i]["label_argmax_disagrees"] for i in np.flatnonzero(mask)),
                                  primary_error_vs_argmax_error_disagreements=sum(rows[i]["primary_vs_argmax_error_disagrees"] for i in np.flatnonzero(mask)))
        support_counts[name]["single_outcome_class"] = bool(len(np.unique(y[mask])) == 1)
        support_counts[name]["weighting"] = weighting_diagnostics(groups[mask], trajectories[mask])
    for path,digest in sources.items():require(sha256(path)==digest,"Protected diagnostic input changed: "+path)
    args.out.mkdir(parents=True)
    prediction=np.empty((len(rows),2),np.float64);prediction[fit]=fit_pred;prediction[tune]=tune_pred
    allfold=np.full(len(rows),-1,np.int64);allfold[fit]=fold
    np.savez_compressed(args.out/'predictions.npz',observation_id=np.asarray([r['observation_id'] for r in rows],np.int64),
                        split=split,error=y,fold=allfold,p_B0=prediction[:,0],p_B1=prediction[:,1],feature_names=np.asarray(FEATURE_NAMES))
    report=dict(status='COMPLETE_MEAN_QUALITY_DIAGNOSTIC',specification=specification(),support_counts=support_counts,
                feature_ranges={name:dict(fit_min=float(x[fit,j].min()),fit_max=float(x[fit,j].max()),
                                          tune_min=float(x[tune,j].min()),tune_max=float(x[tune,j].max())) for j,name in enumerate(FEATURE_NAMES)},
                results=summaries,fit_source_group_fold_mapping=mapping,crossfit_models=models,all_fit_final_model=final,
                input_sha256=sources,script_sha256=sha256(__file__),predictions_sha256=sha256(args.out/'predictions.npz'),
                elapsed_seconds=time.time()-started,tune_used_for_selection=False,test_outcomes_opened=False,
                test_dumps_opened=False,class_holdout_outcomes_opened=False,real_parameter_fit_scope='CPU mean-quality diagnostic only; no tracking/detector weights changed')
    report["limitations"] = dict(low_capacity_control_only=True, pair_model_gate_satisfied=False,
                                no_gain_does_not_establish_quality_unpredictability=True,
                                stronger_quality_control_with_inner_source_group_CV_still_missing=True,
                                bootstrap_intervals_descriptive_only=True)
    (args.out/'diagnostic.json').write_text(json.dumps(serialize(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],support_counts=support_counts,elapsed_seconds=report['elapsed_seconds']),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--write-spec',type=Path)
    parser.add_argument('--support',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--execute',action='store_true',help='Only after explicit authorization to fit real support')
    args=parser.parse_args()
    if args.write_spec:
        require(not args.write_spec.exists(),'Refuse to overwrite a specification')
        args.write_spec.parent.mkdir(parents=True,exist_ok=True)
        args.write_spec.write_text(json.dumps(specification(),indent=2)+'\n')
    if args.self_test:
        print(json.dumps(synthetic_test(),indent=2),flush=True)
    elif args.support:
        require(args.out is not None,'Provide --out for real diagnostic')
        run_real(args)
    elif not args.write_spec:
        parser.error('Use --self-test, --write-spec, or explicitly authorized --support/--out/--execute')


if __name__=='__main__':main()
