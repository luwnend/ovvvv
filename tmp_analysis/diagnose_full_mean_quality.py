#!/usr/bin/env python3
"""Nested source-group selection of CPU mean-error controls, never a pair model.

Frozen pilot helper supplies inference-only features, sealing checks and
hierarchical weights. All choices are made inside fit source groups. The tune
partition is evaluated once, after the final all-fit pipeline is frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn

import diagnose_base_mean_quality as base

ROOT = Path(__file__).resolve().parents[1]
EPS = base.EPS
SEED = 20261023
HELPER_SHA = "2458c403d4f7938ef0f649f9be419027aa5ce8168d011af1572675f5bdd4a84b"
MODEL_NAMES = ["B0_mean", "B1_fixed_ridge_1", "B2_nested_raw", "B3_nested_Platt"]


def require(value, message):
    if not value:
        raise AssertionError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(base.serialize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def candidate_grid():
    # Order is also the declared deterministic simplicity/tie preference.
    return ([dict(name="mean", kind="mean")] +
            [dict(name="ridge_" + str(lam), kind="ridge", ridge=lam)
             for lam in (10., 1., .1, .01, .001, .0001)] +
            [dict(name="HGB_leaf%d_l2%d" % (leaves, l2), kind="HGB", leaves=leaves, l2=l2)
             for leaves in (7, 15) for l2 in (10, 1)])


def specification():
    return dict(status="PREDECLARED_FULL_MEAN_ONLY", candidates=candidate_grid(),
                features=base.FEATURE_NAMES, frozen_feature_helper_sha256=HELPER_SHA,
                target="Eligible emitted-label error; no GT/category/identity predictor",
                hierarchy="Equal source -> equal eligible official GT trajectory -> equal observation; recompute for each training subset",
                selection="5 deterministic source-group outer folds; 3 inner folds per outer train; select concatenated inner OOF hierarchical Brier. Candidates within 1e-5 of minimum use declared grid order.",
                fold_seeds=dict(outer="full_mean_outer_v1", inner="full_mean_inner_v1"),
                ridge="Weighted mean BCE + .5*lambda*||beta_features||^2; train-only weighted standardization; unpenalized intercept",
                HGB=dict(max_iter=100, learning_rate=.05, min_samples_leaf=40, max_bins=128,
                         early_stopping=False, random_state=SEED, sample_weight="n_train * hierarchy_weights; l2 uses that scale"),
                calibrator="Fixed Platt on selected candidate's inner OOF predictions, weighted BCE + .5*1e-4*(intercept^2+(slope-1)^2); intercept [-20,20], slope [0,10]. Refit base on whole training subset. Outer/tune outcomes never calibrate their own predictions.",
                outputs=MODEL_NAMES, primary="B3_nested_Platt; always report raw B2 too, never choose between them on outer/tune",
                final="All-fit 3 inner source-group folds choose candidate and fit OOF calibrator; refit all fit, then predict tune once",
                bootstrap=dict(replicates=1000, seed=SEED, unit="source_group", interpretation="Conditional on frozen predictions; no model refit or training-uncertainty claim"),
                gate="Hard mean-risk control only. Does not qualify wrong-evidence vector centering, pair-dependence prediction, final tracking efficacy or novelty.",
                tune_selection=False, test_outcomes=False, heldout_class_correctness=False, GPU=False)


def folds(groups, maximum, seed):
    unique = sorted(set(map(str, groups)), key=lambda g: hashlib.sha256((seed + "\0" + g).encode()).hexdigest())
    require(len(unique) >= maximum, "Insufficient source groups for declared nested folds")
    mapping = {group: i % maximum for i, group in enumerate(unique)}
    return np.asarray([mapping[str(group)] for group in groups]), mapping


def train_model(x, y, groups, trajectories, candidate):
    w = base.hierarchy_weights(groups, trajectories)
    p0 = float(np.clip(w @ y, EPS, 1 - EPS))
    model = dict(candidate=candidate, p0=p0)
    if candidate["kind"] == "mean" or len(np.unique(y)) == 1:
        model["kind"] = "constant"
        return model
    if candidate["kind"] == "ridge":
        mean = w @ x
        scale = np.sqrt(w @ ((x - mean) ** 2))
        scale[scale < 1e-12] = 1.
        design = np.c_[np.ones(len(x)), (x - mean) / scale]
        coef = np.r_[base.logit(p0), np.zeros(x.shape[1])]
        lam = candidate["ridge"]
        def objective(beta):
            eta = design @ beta
            value = w @ (np.logaddexp(0, eta) - y * eta) + .5 * lam * (beta[1:] @ beta[1:])
            gradient = design.T @ (w * (expit(eta) - y))
            gradient[1:] += lam * beta[1:]
            return float(value), gradient
        opt = minimize(objective, coef, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=300, ftol=1e-12, gtol=1e-8, maxls=40))
        require(opt.success and np.isfinite(opt.x).all(), "Ridge fit failed: " + str(opt.message))
        model.update(kind="ridge", mean=mean, scale=scale, coefficient=opt.x,
                     optimizer=dict(success=True, iterations=int(opt.nit), loss=float(opt.fun)))
    else:
        estimator = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05, min_samples_leaf=40,
                    max_leaf_nodes=candidate["leaves"], l2_regularization=candidate["l2"], max_bins=128,
                    early_stopping=False, random_state=SEED)
        estimator.fit(x, y, sample_weight=w * len(y))
        model.update(kind="HGB", estimator=estimator)
    return model


def predict(model, x):
    if model["kind"] == "constant":
        result = np.full(len(x), model["p0"])
    elif model["kind"] == "ridge":
        result = expit(model["coefficient"][0] + ((x - model["mean"]) / model["scale"]) @ model["coefficient"][1:])
    else:
        result = model["estimator"].predict_proba(x)[:, 1]
    return np.clip(result, EPS, 1 - EPS)


def calibrator_fit(p, y, groups, trajectories):
    w = base.hierarchy_weights(groups, trajectories)
    eta = base.logit(p)
    design = np.c_[np.ones(len(y)), eta]
    def objective(beta):
        v = design @ beta
        delta = beta - np.asarray([0., 1.])
        return float(w @ (np.logaddexp(0, v) - y * v) + .5e-4 * (delta @ delta)), \
            design.T @ (w * (expit(v) - y)) + 1e-4 * delta
    opt = minimize(objective, np.asarray([0., 1.]), jac=True, method="L-BFGS-B",
                   bounds=[(-20., 20.), (0., 10.)], options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    require(opt.success and np.isfinite(opt.x).all(), "Inner OOF Platt fitting failed: " + str(opt.message))
    return dict(intercept=float(opt.x[0]), slope=float(opt.x[1]), loss=float(opt.fun),
                boundary=bool(abs(opt.x[0]) >= 19.999 or opt.x[1] <= 1e-6 or opt.x[1] >= 9.999),
                trained_on="selected candidate inner OOF predictions of training groups only")


def calibrated(cal, p):
    return np.clip(expit(cal["intercept"] + cal["slope"] * base.logit(p)), EPS, 1 - EPS)


def fit_pipeline(x, y, groups, trajectories, grid=None):
    grid = candidate_grid() if grid is None else grid
    fold, mapping = folds(groups, 3, "full_mean_inner_v1")
    w = base.hierarchy_weights(groups, trajectories)
    oof = np.full((len(y), len(grid)), np.nan)
    inner_models = []
    for f in range(3):
        train, valid = fold != f, fold == f
        require(not set(groups[train]) & set(groups[valid]), "Inner source leakage")
        fold_models = []
        for j, candidate in enumerate(grid):
            model = train_model(x[train], y[train], groups[train], trajectories[train], candidate)
            oof[valid, j] = predict(model, x[valid])
            fold_models.append(model)
        inner_models.append(dict(fold=f, training_rows=np.flatnonzero(train),
                                 validation_rows=np.flatnonzero(valid), models=fold_models))
    require(np.isfinite(oof).all(), "Inner predictions incomplete")
    scores = w @ ((oof - y[:, None]) ** 2)
    minimum = float(scores.min())
    selected = next(j for j, value in enumerate(scores) if value <= minimum + 1e-5)
    cal = calibrator_fit(oof[:, selected], y, groups, trajectories)
    final = train_model(x, y, groups, trajectories, grid[selected])
    info = dict(selected_candidate=grid[selected], inner_candidate_Brier={c["name"]:float(v) for c,v in zip(grid,scores)},
                inner_fold_mapping=mapping, calibrator=cal, training_groups=sorted(set(groups.tolist())),
                training_rows=len(y), selection_scores_are_unbiased_estimates=False)
    info['_audit'] = dict(inner_fold=fold, inner_oof_predictions=oof, inner_models=inner_models,
                          grid=grid, selected_index=selected)
    return final, cal, info


def prediction_columns(x, baseline, fixed, selected, cal):
    raw = predict(selected, x)
    return np.c_[predict(baseline, x), predict(fixed, x), raw, calibrated(cal, raw)]


def run_nested(x, y, groups, trajectories, grid=None, progress=True):
    fold, mapping = folds(groups, 5, "full_mean_outer_v1")
    predictions = np.full((len(y), 4), np.nan)
    receipts, audits = [], []
    for f in range(5):
        train, valid = fold != f, fold == f
        require(not set(groups[train]) & set(groups[valid]), "Outer source leakage")
        selected, cal, info = fit_pipeline(x[train], y[train], groups[train], trajectories[train], grid)
        inner_audit = info.pop('_audit')
        baseline = train_model(x[train], y[train], groups[train], trajectories[train], dict(name="mean",kind="mean"))
        fixed = train_model(x[train], y[train], groups[train], trajectories[train], dict(name="ridge_1.0",kind="ridge",ridge=1.))
        predictions[valid] = prediction_columns(x[valid], baseline, fixed, selected, cal)
        receipts.append(dict(outer_fold=f, validation_groups=sorted(set(groups[valid].tolist())),
                             validation_rows=int(valid.sum()), **info))
        audits.append(dict(outer_fold=f, training_rows_in_fit=np.flatnonzero(train),
                           validation_rows_in_fit=np.flatnonzero(valid), inner=inner_audit,
                           baseline=baseline, fixed=fixed, selected=selected, calibrator=cal))
        if progress:
            print(json.dumps(dict(status="OUTER_FOLD_COMPLETE", fold=f, rows=int(valid.sum()), selected=info["selected_candidate"], calibrator=cal)),flush=True)
    require(np.isfinite(predictions).all(), "Outer predictions incomplete")
    return predictions, fold, mapping, receipts, audits


def summarize(y, p, groups, trajectories, replicates=1000):
    hierarchy = base.hierarchy_weights(groups, trajectories)
    point = {name:{model:base.metrics(y,p[:,j],w) for j,model in enumerate(MODEL_NAMES)}
             for name,w in (("micro",np.ones(len(y))/len(y)),("source_trajectory_observation_weighted",hierarchy))}
    # Aggregate losses by source first; exactly equivalent to copying whole
    # source groups, while avoiding thousands of needless calibration fits.
    unique = sorted(set(groups))
    group_loss = {kind:[] for kind in ("micro", "source_trajectory_observation_weighted")}
    count = []
    losses = np.stack([(p-y[:,None])**2, -y[:,None]*np.log(p)-(1-y[:,None])*np.log1p(-p)],axis=2)
    for group in unique:
        mask = groups==group; count.append(int(mask.sum()))
        group_loss["micro"].append(losses[mask].sum(0))
        weights=hierarchy[mask]/hierarchy[mask].sum()
        group_loss["source_trajectory_observation_weighted"].append(np.einsum('n,njk->jk',weights,losses[mask]))
    rng=np.random.default_rng(SEED)
    multiplicities=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=replicates)
    ci={}
    for kind,values in group_loss.items():
        sums=np.einsum('rg,gjk->rjk',multiplicities,np.asarray(values))
        denominator=multiplicities@np.asarray(count) if kind=='micro' else np.full(replicates,len(unique))
        means=sums/denominator[:,None,None]
        ci[kind]={}
        for j,model in enumerate(MODEL_NAMES):
            ci[kind][model]={metric:dict(low=float(np.quantile(means[:,j,k],.025)),high=float(np.quantile(means[:,j,k],.975))) for k,metric in enumerate(('Brier','BCE'))}
        for a,b in ((3,0),(3,1),(3,2),(2,1)):
            ci[kind][MODEL_NAMES[a]+'_minus_'+MODEL_NAMES[b]]={metric:dict(low=float(np.quantile(means[:,a,k]-means[:,b,k],.025)),high=float(np.quantile(means[:,a,k]-means[:,b,k],.975))) for k,metric in enumerate(('Brier','BCE'))}
    bins={}
    for j,model in enumerate(MODEL_NAMES):
        cells=[]
        which=np.minimum((p[:,j]*10).astype(int),9)
        for b in range(10):
            mask=which==b
            if not mask.any(): continue
            w=hierarchy[mask]/hierarchy[mask].sum()
            cells.append(dict(probability_bin=[b/10,(b+1)/10],observations=int(mask.sum()),source_groups=len(set(groups[mask])),
                              hierarchical_mass=float(hierarchy[mask].sum()),weighted_prediction=float(w@p[mask,j]),
                              weighted_error=float(w@y[mask]),weighted_residual=float(w@(y[mask]-p[mask,j])),
                              interpretation='Diagnostic cell only; no bin-specific calibration fitted'))
        bins[model]=cells
    return dict(point=point,source_group_bootstrap_95_percentile=ci,reliability_fixed_probability_bins=bins,
                bootstrap_replicates=replicates,source_groups=len(unique),bootstrap_interpretation=specification()['bootstrap']['interpretation'])


def self_test():
    rng=np.random.default_rng(SEED); groups=np.repeat(np.asarray(['g%d'%i for i in range(15)]),12)
    trajectories=np.asarray([g+':'+str(i%3) for i,g in enumerate(groups)])
    x=rng.normal(size=(len(groups),len(base.FEATURE_NAMES))); x[:,-1]=0
    y=(rng.random(len(x))<expit(x[:,0]+.7*x[:,1]*x[:,2])).astype(float)
    small=[dict(name='mean',kind='mean'),dict(name='ridge_0.1',kind='ridge',ridge=.1)]
    p,fold,_,_,_=run_nested(x,y,groups,trajectories,small,False)
    # Flip every validation outcome for one outer fold, then refit the entire
    # nested pipeline. Its own predictions must not change, including selection
    # and calibration; other outer folds legitimately may change.
    changed=y.copy(); held=fold==0; changed[held]=1-changed[held]
    q,_,_,_,_=run_nested(x,changed,groups,trajectories,small,False)
    require(np.array_equal(p[held],q[held]),'Outer outcomes leaked into selection or calibration')
    # Independent nonlinear estimator serialization round-trip is checked in
    # the real path; this checks its zero-leak train-only invocation and scale.
    tree=train_model(x,y,groups,trajectories,dict(name='tinyHGB',kind='HGB',leaves=7,l2=10))
    require(np.isfinite(predict(tree,x)).all(),'Nonlinear training failed')
    cal=calibrator_fit(np.linspace(.1,.9,len(y)),y,groups,trajectories)
    require(0<=cal['slope']<=10,'Calibrator slope constraints failed')
    summary=summarize(y,p,groups,trajectories,replicates=12)
    require(summary['source_groups']==15,'Group resampling lost sources')
    return dict(status='PASS_SYNTHETIC_NESTED_SELECTION_CALIBRATION_CONTRACTS',outer_outcome_flip_invariance=True,
                inner_and_outer_source_disjoint=True,tree_weight_scale=True,positive_bounded_Platt=True,
                full_grid_frozen_separately=True,real_support_opened=False,GPU_used=False)


def execute(args):
    require(args.execute,'Use explicit --execute after implementation review')
    require(not args.out.exists(),'Refuse overwrite')
    require(base.sha256(Path(base.__file__))==HELPER_SHA,'Frozen pilot helper changed')
    spec_path=ROOT/'tmp_analysis/full_mean_pair_gate_protocol_20261023.json'
    require(spec_path.is_file(),'Freeze independent protocol first')
    frozen=json.loads(spec_path.read_text())
    require(frozen.get('mean_control',{}).get('candidate_grid')==candidate_grid(),'Implementation differs from frozen candidate grid')
    receipt_path=args.support/'support_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    require('base_error_full_collection_20261023' in receipt['run'],'Use full collection support only')
    independent=args.support/'independent_support_audit.json'
    require(independent.is_file(),'Independent full support audit missing')
    audit=json.loads(independent.read_text())
    require(audit['status']=='PASS_INDEPENDENT_FULL_SUPPORT_OUTPUT_AUDIT','Independent full support audit not PASS')
    require(audit['support_receipt_sha256']==base.sha256(receipt_path),'Independent support audit is stale')
    preflight_path=ROOT/'tmp_analysis/full_mean_independent_preflight_20261023.json'
    preflight=json.loads(preflight_path.read_text())
    require(preflight['status']=='PASS_INDEPENDENT_FULL_MEAN_IMPLEMENTATION_PREFLIGHT','Mean implementation preflight missing')
    require(preflight['script_sha256']==base.sha256(__file__),'Mean script changed after independent review')
    require(preflight['helper_sha256']==HELPER_SHA,'Preflight helper mismatch')
    require(preflight['protocol_sha256']==base.sha256(spec_path),'Protocol changed after independent review')
    start=time.time(); rows,x,sources=base.load_support(args.support.resolve())
    y=np.asarray([v['error'] for v in rows],float); split=np.asarray([v['split'] for v in rows])
    groups=np.asarray([v['source_group'] for v in rows]); trajectories=np.asarray(['%d:%d'%(v['video_id'],v['gt_track_id']) for v in rows])
    fit,tune=split=='fit',split=='tune'
    print(json.dumps(dict(status='FEATURES_EXTRACTED',rows=len(rows),fit_rows=int(fit.sum()),tune_rows=int(tune.sum()),elapsed_seconds=time.time()-start)),flush=True)
    oof,fold,mapping,fold_info,nested_audits=run_nested(x[fit],y[fit],groups[fit],trajectories[fit])
    selected,cal,info=fit_pipeline(x[fit],y[fit],groups[fit],trajectories[fit])
    final_inner_audit=info.pop('_audit')
    baseline=train_model(x[fit],y[fit],groups[fit],trajectories[fit],dict(name='mean',kind='mean'))
    fixed=train_model(x[fit],y[fit],groups[fit],trajectories[fit],dict(name='ridge_1.0',kind='ridge',ridge=1.))
    prediction=np.empty((len(y),4)); prediction[fit]=oof; prediction[tune]=prediction_columns(x[tune],baseline,fixed,selected,cal)
    print(json.dumps(dict(status='FINAL_PIPELINE_FROZEN_AND_TUNE_PREDICTED',selected=info['selected_candidate'])),flush=True)
    result={name:summarize(y[mask],prediction[mask],groups[mask],trajectories[mask]) for name,mask in (('fit_nested_source_group_crossfit',fit),('tune_evaluation_only',tune))}
    for path,digest in sources.items(): require(base.sha256(path)==digest,'Diagnostic input changed: '+path)
    args.out.mkdir(parents=True)
    allfold=np.full(len(y),-1); allfold[fit]=fold
    np.savez_compressed(args.out/'features_predictions.npz',x=x,error=y,split=split,source_group=groups,trajectory=trajectories,
                        observation_id=np.asarray([v['observation_id'] for v in rows],np.int64),outer_fold=allfold,
                        feature_names=np.asarray(base.FEATURE_NAMES),prediction=prediction,model_names=np.asarray(MODEL_NAMES))
    import joblib
    fitted=dict(selected=selected,calibrator=cal,baseline=baseline,fixed=fixed,feature_names=base.FEATURE_NAMES)
    joblib.dump(fitted,args.out/'final_fit_pipeline.joblib')
    joblib.dump(dict(fit_rows_in_full=np.flatnonzero(fit), tune_rows_in_full=np.flatnonzero(tune),
                     outer=nested_audits, final_inner=final_inner_audit,
                     note='Local-only verification objects; never numeric model inputs'),args.out/'nested_audit.joblib')
    loaded=joblib.load(args.out/'final_fit_pipeline.joblib')
    require(np.array_equal(prediction[tune],prediction_columns(x[tune],loaded['baseline'],loaded['fixed'],loaded['selected'],loaded['calibrator'])),'Saved pipeline predictions differ')
    source_manifest=dict(status='SEALED_FULL_MEAN_INPUT_HASHES',inputs=sources)
    write(args.out/'input_hashes.json',source_manifest)
    summary=dict(status='COMPLETE_FULL_NESTED_MEAN_QUALITY_DIAGNOSTIC_PENDING_INDEPENDENT_REVIEW',specification=specification(),
                 support_counts={name:dict(rows=int(mask.sum()),errors=int(y[mask].sum()),source_groups=len(set(groups[mask])),
                                          GT_trajectories=len(set(zip(groups[mask],trajectories[mask])))) for name,mask in (('fit',fit),('tune',tune))},
                 results=result,outer_fold_mapping=mapping,outer_training_receipts=fold_info,final_all_fit_pipeline=info,
                 sklearn_version=sklearn.__version__,script_sha256=base.sha256(__file__),
                 protocol_sha256=base.sha256(spec_path),independent_support_audit_sha256=base.sha256(independent),
                 independent_implementation_preflight_sha256=base.sha256(preflight_path),
                 input_hash_manifest_sha256=base.sha256(args.out/'input_hashes.json'),
                 output_hashes={n:base.sha256(args.out/n) for n in ('features_predictions.npz','final_fit_pipeline.joblib','nested_audit.joblib')},
                 elapsed_seconds=time.time()-start,tune_used_for_selection=False,test_outcomes_opened=False,
                 class_holdout_correctness_opened=False,detector_or_tracker_training=False,pair_method_gate_satisfied=False,
                 limitations=['Hard mean-error diagnostic only, not wrong-evidence vector calibration or pair-dependence proof',
                              'Bootstrap conditions on fitted predictions; source trajectory repetitions are not independent samples',
                              'Readout-fit data can have been seen by the released detector; no unseen-detector generalization claim'])
    write(args.out/'diagnostic_summary.json',summary)
    print(json.dumps(dict(status=summary['status'],support_counts=summary['support_counts'],elapsed_seconds=summary['elapsed_seconds'])),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true'); ap.add_argument('--spec-output',type=Path)
    ap.add_argument('--support',type=Path); ap.add_argument('--out',type=Path); ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if args.spec_output: write(args.spec_output,specification())
    if args.self_test: print(json.dumps(self_test(),indent=2),flush=True)
    elif args.support:
        require(args.out is not None,'Provide new output directory'); execute(args)
    elif not args.spec_output: ap.error('Use --self-test or --support/--out/--execute')


if __name__=='__main__': main()
