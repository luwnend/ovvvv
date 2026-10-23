#!/usr/bin/env python3
"""Independent full mean output audit, without any candidate baseline refitting.

Reads eligible fit/tune rows only. Rebuilds features, targets, train weights,
predictions, selection/calibration scores and source-bootstrap loss intervals.
Candidate inner OOF arrays must be persisted by the reviewed producer.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq, minimize
import joblib

import audit_base_mean_quality as independent

EPS = 1e-6
NAMES = ["B0_mean", "B1_fixed_ridge_1", "B2_nested_raw", "B3_nested_Platt"]
ROOT = Path(__file__).resolve().parents[1]
check, digest, load = independent.check, independent.digest, independent.load


def write(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def independent_folds(groups, maximum, seed):
    ordered = sorted(set(map(str, groups)), key=lambda group: hashlib.sha256((seed+"\0"+group).encode()).hexdigest())
    mapping = {group: i % maximum for i,group in enumerate(ordered)}
    return np.asarray([mapping[str(group)] for group in groups]), mapping


def predict(model, x):
    if model["kind"] == "constant":
        return np.full(len(x), model["p0"])
    if model["kind"] == "ridge":
        eta = model["coefficient"][0] + ((x-model["mean"])/model["scale"]) @ model["coefficient"][1:]
        return np.clip(expit(eta), EPS, 1-EPS)
    check(model["kind"] == "HGB", "Unknown persisted model kind")
    return np.clip(model["estimator"].predict_proba(x)[:,1], EPS, 1-EPS)


def model_contract(model, x, y, rows, candidate):
    check(model["candidate"] == candidate, "Persisted model has wrong candidate")
    w = independent.weights(rows)
    p0 = float(np.clip(w @ y, EPS, 1-EPS))
    check(np.isclose(model["p0"],p0,atol=1e-12),"Stored training mean differs from independent hierarchy")
    result = dict(kind=model["kind"],train_rows=len(y),train_sources=len({r["source_group"] for r in rows}),
                  train_trajectories=len({(r["source_group"],r["video_id"],r["gt_track_id"]) for r in rows}),
                  single_outcome_class=bool(len(np.unique(y))==1))
    if model["kind"] == "ridge":
        mean = w @ x; scale = np.sqrt(w @ ((x-mean)**2)); constant=scale<1e-12;scale[constant]=1
        check(np.allclose(model["mean"],mean,atol=1e-12,rtol=1e-12) and
              np.allclose(model["scale"],scale,atol=1e-12,rtol=1e-12),"Stored scaler is not training-only weighted scaler")
        beta=np.asarray(model["coefficient"]);z=(x-mean)/scale
        grad=np.column_stack((np.ones(len(x)),z)).T@(w*(expit(beta[0]+z@beta[1:])-y))
        grad[1:]+=candidate["ridge"]*beta[1:]
        check(np.max(np.abs(grad))<2e-5,"Stored ridge gradient is not stationary")
        result.update(gradient_max_abs=float(np.max(np.abs(grad))),constant_features=[independent.FEATURES[j] for j in np.flatnonzero(constant)])
    elif model["kind"] == "HGB":
        params=model["estimator"].get_params()
        for key,value in dict(max_iter=100,learning_rate=.05,min_samples_leaf=40,max_leaf_nodes=candidate["leaves"],
                              l2_regularization=candidate["l2"],max_bins=128,early_stopping=False,random_state=20261023).items():
            check(params[key]==value,"Persisted HGB parameter differs: "+key)
        check(model["estimator"].n_features_in_==len(independent.FEATURES),"HGB feature dimension differs")
        check(np.isclose(float(expit(model["estimator"]._baseline_prediction[0,0])),p0,atol=1e-12,rtol=1e-12),
              "HGB initial logit differs from hierarchical weighted training prevalence")
        result["HGB_sample_weight_verification"]="Producer code contract audited; persisted estimator does not expose historical sample weights. No HGB refit performed."
    else:
        check(candidate["kind"]=="mean" or len(np.unique(y))==1,"Unexpected constant fallback")
    return result


def check_platt(cal, raw_oof, y, rows):
    check(-20<=cal["intercept"]<=20 and 0<=cal["slope"]<=10,"Platt bounds changed")
    w=independent.weights(rows);eta=np.log(np.clip(raw_oof,EPS,1-EPS))-np.log1p(-np.clip(raw_oof,EPS,1-EPS))
    beta=np.asarray([cal["intercept"],cal["slope"]]);design=np.column_stack((np.ones(len(y)),eta))
    fitted=design@beta;delta=beta-np.asarray([0.,1.])
    loss=float(w@(np.logaddexp(0,fitted)-y*fitted)+.5e-4*(delta@delta))
    gradient=design.T@(w*(expit(fitted)-y))+1e-4*delta
    projected=gradient.copy()
    for j,(low,high) in enumerate([(-20,20),(0,10)]):
        if beta[j]<=low+1e-7 and gradient[j]>0:projected[j]=0
        if beta[j]>=high-1e-7 and gradient[j]<0:projected[j]=0
    check(np.isclose(loss,cal["loss"],atol=1e-10,rtol=1e-10),"Platt objective differs")
    check(np.max(np.abs(projected))<2e-5,"Platt projected gradient not stationary")
    return dict(loss=loss,projected_gradient_max_abs=float(np.max(np.abs(projected))),boundary=cal["boundary"])


def calibration_metrics(y,p,w):
    p=np.clip(p,EPS,1-EPS);w=w/w.sum();eta=np.log(p)-np.log1p(-p)
    result=dict(observed_mean_error=float(w@y),predicted_mean_error=float(w@p),Brier=float(w@((p-y)**2)),
                BCE=float(w@(-y*np.log(p)-(1-y)*np.log1p(-p))))
    if len(np.unique(y[w>0]))<2:
        return dict(result,calibration_in_the_large_intercept=None,calibration_intercept=None,calibration_slope=None)
    result["calibration_in_the_large_intercept"]=float(brentq(lambda b:w@(expit(eta+b)-y),-60,60))
    if float(w@((eta-w@eta)**2))<1e-12:
        return dict(result,calibration_intercept=None,calibration_slope=None)
    design=np.column_stack((np.ones(len(y)),eta))
    def objective(beta):
        v=design@beta
        return float(w@(np.logaddexp(0,v)-y*v)),design.T@(w*(expit(v)-y))
    opt=minimize(objective,np.asarray([0.,1.]),jac=True,method="BFGS",options=dict(gtol=1e-10,maxiter=300))
    if np.max(np.abs(objective(opt.x)[1]))>1e-7 or np.max(np.abs(opt.x))>50:
        return dict(result,calibration_intercept=None,calibration_slope=None)
    return dict(result,calibration_intercept=float(opt.x[0]),calibration_slope=float(opt.x[1]))


def check_summary(y,p,groups,rows,summary):
    hierarchy=independent.weights(rows)
    max_point_delta=0.
    for kind,w in (("micro",np.ones(len(y))/len(y)),("source_trajectory_observation_weighted",hierarchy)):
        for j,name in enumerate(NAMES):
            computed=calibration_metrics(y,p[:,j],w);reported=summary["point"][kind][name]
            for key,value in computed.items():
                if value is None:
                    check(reported[key] is None,"Descriptive calibration identifiability differs")
                else:
                    tolerance=3e-4 if key in ("calibration_intercept","calibration_slope") else 1e-10
                    check(np.isclose(value,reported[key],atol=tolerance,rtol=tolerance),"Point metric differs: "+key)
                    max_point_delta=max(max_point_delta,abs(value-reported[key]))
    unique=sorted(set(groups));loss=np.stack(((p-y[:,None])**2,-y[:,None]*np.log(p)-(1-y[:,None])*np.log1p(-p)),axis=2)
    counts=[];micro=[];weighted=[]
    for group in unique:
        mask=groups==group;counts.append(int(mask.sum()));micro.append(loss[mask].sum(axis=0))
        w=hierarchy[mask]/hierarchy[mask].sum();weighted.append(np.tensordot(w,loss[mask],axes=1))
    rng=np.random.default_rng(20261023);draws=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=1000)
    max_ci_delta=0.
    for kind,source_values in (("micro",micro),("source_trajectory_observation_weighted",weighted)):
        numerator=np.tensordot(draws,np.asarray(source_values),axes=([1],[0]))
        denominator=draws@np.asarray(counts) if kind=="micro" else np.full(1000,len(unique))
        values=numerator/denominator[:,None,None]
        expected={name:values[:,j,:] for j,name in enumerate(NAMES)}
        for a,b in ((3,0),(3,1),(3,2),(2,1)):expected[NAMES[a]+"_minus_"+NAMES[b]]=values[:,a,:]-values[:,b,:]
        for name,v in expected.items():
            for k,metric in enumerate(("Brier","BCE")):
                ci=np.quantile(v[:,k],[.025,.975]);reported=summary["source_group_bootstrap_95_percentile"][kind][name][metric]
                check(np.allclose(ci,[reported["low"],reported["high"]],atol=1e-12,rtol=1e-12),"Source-bootstrap CI differs")
                max_ci_delta=max(max_ci_delta,float(np.max(np.abs(ci-[reported["low"],reported["high"]]))))
    for j,name in enumerate(NAMES):
        bins=summary["reliability_fixed_probability_bins"][name];which=np.minimum((p[:,j]*10).astype(int),9)
        expected_bins=[]
        for b in range(10):
            mask=which==b
            if not mask.any():continue
            w=hierarchy[mask]/hierarchy[mask].sum()
            expected_bins.append(dict(probability_bin=[b/10,(b+1)/10],observations=int(mask.sum()),source_groups=len(set(groups[mask])),
                                      hierarchical_mass=float(hierarchy[mask].sum()),weighted_prediction=float(w@p[mask,j]),
                                      weighted_error=float(w@y[mask]),weighted_residual=float(w@(y[mask]-p[mask,j]))))
        check(len(bins)==len(expected_bins),"Reliability bin count differs")
        for a,b in zip(bins,expected_bins):
            for key,value in b.items():
                check(np.allclose(a[key],value,atol=1e-12,rtol=1e-12),"Reliability bin differs: "+key)
    return dict(point_metrics_max_absolute_difference=max_point_delta,bootstrap_1000_loss_CI_max_absolute_difference=max_ci_delta,
                fixed_probability_reliability_bins_recomputed=True)


def read_eligible_features(support, input_hashes):
    receipt=load(support/"support_receipt.json");run=Path(receipt["run"])
    rows=[json.loads(line) for line in (support/"supervised_observations.jsonl").open()]
    plan=load(ROOT/"tmp_analysis/base_error_learning_plan_20261023.json")
    fit_classes=set(plan["vocabulary"]["readout_fit_category_ids"]);heldout=set(plan["vocabulary"]["readout_category_holdout_ids"])
    selected={r["observation_id"]:r for r in rows};index={};file_splits=defaultdict(set)
    check(len(selected)==len(rows),"Duplicate supervised observation")
    for line in (support/"gt_free_observations.jsonl").open():
        raw=json.loads(line);file_splits[raw["dump_file"]].add(raw["split"])
        if raw["observation_id"] in selected:index[raw["observation_id"]]=raw
    files=defaultdict(list)
    for i,row in enumerate(rows):
        check(row["split"] in ("fit","tune") and row["gt_category_id"] in fit_classes and row["gt_category_id"] not in heldout,
              "Closed split/class target in full mean")
        check(row["eligible_for_parameter_fit"] is (row["split"]=="fit"),"Fit eligibility mismatch")
        check(row["error"]==int(row["pred_category_id"]!=row["gt_category_id"]),"Primary target not emitted error")
        raw=index[row["observation_id"]]
        for key in ("video_id","image_id","local_track_id","split","source_group","pred_category_id"):
            check(row[key]==raw[key],"Supervised/index identity mismatch")
        check(file_splits[raw["dump_file"]]=={row["split"]},"Used dump mixes splits")
        files[raw["dump_file"]].append((i,row,raw))
    recorded={Path(path).name for path in input_hashes if path.endswith(".npz")}
    check(recorded==set(files),"Mean raw dump input set differs from eligible features")
    check(not any("test" in file_splits[name] for name in files),"Test dump in mean features")
    x=np.empty((len(rows),len(independent.FEATURES)),np.float64)
    for name,indexed in files.items():
        with np.load(run/"dump"/name,allow_pickle=False) as saved:
            data={key:saved[key] for key in ("posterior_category_ids","p_fused","p_text","p_image","pretracker_det_row", "pretracker_det_bboxes","det_bboxes","ori_shape")}
            tids=saved["track_ids"];labels=saved["det_labels"];emitted=saved["emitted_track_mask"]
            vid=int(saved["video_id"]);iid=int(saved["image_id"])
            for i,row,raw in indexed:
                j=int(raw["dump_row"])
                check(bool(emitted[j]) and int(tids[j])==row["local_track_id"],"Raw emitted track differs")
                check(vid==row["video_id"] and iid==row["image_id"],"Raw frame differs")
                check(int(labels[j])+1==row["pred_category_id"],"Raw emitted target label differs")
                x[i]=independent.independent_features(data,j)
    return rows,x,len(files)


def check_inner(audit,info,x,y,groups,rows,grid):
    expected_fold,mapping=independent_folds(groups,3,"full_mean_inner_v1")
    check(audit["grid"]==grid,"Persisted inner grid differs from frozen protocol")
    check(np.array_equal(audit["inner_fold"],expected_fold) and info["inner_fold_mapping"]==mapping,"Inner group folds differ")
    check(info["training_rows"]==len(y) and info["training_groups"]==sorted(set(groups)),"Pipeline training metadata differs")
    oof=np.full((len(y),len(grid)),np.nan);models_checked=[]
    check(len(audit["inner_models"])==3,"Missing inner fold models")
    for record in audit["inner_models"]:
        f=record["fold"];training=np.flatnonzero(expected_fold!=f);validation=np.flatnonzero(expected_fold==f)
        check(np.array_equal(record["training_rows"],training) and np.array_equal(record["validation_rows"],validation),
              "Persisted inner training/validation rows differ")
        check(not set(groups[training]).intersection(groups[validation]),"Inner source overlap")
        check(len(record["models"])==len(grid),"Missing inner candidate models")
        train_rows=[rows[i] for i in training]
        for j,(model,candidate) in enumerate(zip(record["models"],grid)):
            model_check=model_contract(model,x[training],y[training],train_rows,candidate)
            models_checked.append(model_check)
            oof[validation,j]=predict(model,x[validation])
    check(np.allclose(oof,audit["inner_oof_predictions"],atol=1e-12,rtol=1e-12),"Inner OOF predictions differ from persisted held-source models")
    w=independent.weights(rows);scores=w@((oof-y[:,None])**2)
    for candidate,value in zip(grid,scores):
        check(np.isclose(value,info["inner_candidate_Brier"][candidate["name"]],atol=1e-12,rtol=1e-12),"Inner candidate score differs")
    selected=next(i for i,v in enumerate(scores) if v<=scores.min()+1e-5)
    check(audit["selected_index"]==selected and info["selected_candidate"]==grid[selected],"Nested selection/tie order differs")
    cal=check_platt(info["calibrator"],oof[:,selected],y,rows)
    return dict(selected_candidate=grid[selected],inner_candidate_models_checked=len(models_checked),
                ridge_max_gradient=max([m.get("gradient_max_abs",0.) for m in models_checked]),
                single_class_inner_models=sum(m["single_outcome_class"] for m in models_checked),
                inner_source_fold_disjoint=True,inner_OOF_predictions_reconstructed=True,
                selection_scores_and_tie_rule_recomputed=True,calibrator=cal)


def check_pipeline_predictions(bundle,calibrator,x,p):
    raw=predict(bundle["selected"],x)
    eta=np.log(raw)-np.log1p(-raw)
    reconstructed=np.column_stack((predict(bundle["baseline"],x),predict(bundle["fixed"],x),raw,
                                   np.clip(expit(calibrator["intercept"]+calibrator["slope"]*eta),EPS,1-EPS)))
    check(np.allclose(reconstructed,p,atol=1e-12,rtol=1e-12),"Saved outer/tune predictions differ from persisted pipeline")
    return float(np.max(np.abs(reconstructed-p)))


def self_test():
    import diagnose_full_mean_quality as producer
    rng=np.random.default_rng(205);groups=np.repeat(np.asarray(["g%d"%i for i in range(15)]),16)
    trajectories=np.asarray(["%d:%d"%(i//16,i%3) for i in range(len(groups))])
    x=rng.normal(size=(len(groups),14));x[:,-1]=0
    y=(rng.random(len(x))<expit(x[:,0]+.5*x[:,1]*x[:,2])).astype(float)
    rows=[dict(source_group=g,video_id=i//16,gt_track_id=i%3) for i,g in enumerate(groups)]
    selected,cal,info=producer.fit_pipeline(x,y,groups,trajectories);inner=info.pop("_audit")
    checked=check_inner(inner,info,x,y,groups,rows,producer.candidate_grid())
    baseline=producer.train_model(x,y,groups,trajectories,dict(name="mean",kind="mean"))
    fixed=producer.train_model(x,y,groups,trajectories,dict(name="ridge_1.0",kind="ridge",ridge=1.))
    p=producer.prediction_columns(x,baseline,fixed,selected,cal)
    metrics=check_summary(y,p,groups,rows,producer.summarize(y,p,groups,trajectories))
    return dict(status="PASS_SYNTHETIC_FULL_MEAN_INDEPENDENT_AUDITOR",inner=checked,metrics=metrics,
                real_support_opened=False,baseline_models_refitted_only_on_synthetic_data=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2),flush=True)
        return
    check(args.diagnostic is not None,"Provide --diagnostic or --self-test")
    out=args.diagnostic.resolve();audit_path=out/"independent_mean_quality_audit.json"
    check(not audit_path.exists(),"Refuse overwrite independent audit")
    started=time.time();summary_path=out/"diagnostic_summary.json";summary_sha_before=digest(summary_path);summary=load(summary_path)
    check(summary["status"]=="COMPLETE_FULL_NESTED_MEAN_QUALITY_DIAGNOSTIC_PENDING_INDEPENDENT_REVIEW","Full mean incomplete")
    check(digest(ROOT/"tmp_analysis/diagnose_full_mean_quality.py")==summary["script_sha256"],"Reviewed mean producer changed")
    check(digest(ROOT/"tmp_analysis/full_mean_pair_gate_protocol_20261023.json")==summary["protocol_sha256"],"Frozen protocol changed")
    protocol=load(ROOT/"tmp_analysis/full_mean_pair_gate_protocol_20261023.json");grid=protocol["mean_control"]["candidate_grid"]
    check(summary["specification"]["candidates"]==grid and summary["specification"]["features"]==independent.FEATURES,"Producer specification changed")
    for flag in ("tune_used_for_selection","test_outcomes_opened","class_holdout_correctness_opened","detector_or_tracker_training","pair_method_gate_satisfied"):
        check(summary[flag] is False,"Invalid mean scope flag: "+flag)
    for filename,value in summary["output_hashes"].items():check(digest(out/filename)==value,"Mean output changed: "+filename)
    check(digest(out/"input_hashes.json")==summary["input_hash_manifest_sha256"],"Input hash manifest changed")
    inputs=load(out/"input_hashes.json")["inputs"]
    for path,value in inputs.items():check(digest(path)==value,"Mean input changed: "+path)
    support_paths=[Path(p).parent for p in inputs if p.endswith("support_receipt.json")]
    check(len(support_paths)==1,"Ambiguous support input")
    support=support_paths[0];support_receipt=load(support/"support_receipt.json")
    for path,value in support_receipt["inputs"].items():check(digest(path)==value,"Full support source changed: "+path)
    for filename,value in support_receipt["output_files"].items():check(digest(support/filename)==value,"Full support rows/index changed")
    check(support_receipt["status"]=="PASS_FIT_TUNE_SIGNAL_SUPPORT_AUDIT" and
          support_receipt["collection_scope"]=="FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES","Wrong support scope")
    check(support_receipt["test_outcomes_computed"] is False and support_receipt["class_holdout_outcomes_computed"] is False,"Support closed outcomes open")
    support_audit=load(support/"independent_support_audit.json")
    check(support_audit["status"]=="PASS_INDEPENDENT_FULL_SUPPORT_OUTPUT_AUDIT" and
          support_audit["support_receipt_sha256"]==digest(support/"support_receipt.json"),"Independent support audit stale")
    check(digest(support/"independent_support_audit.json")==summary["independent_support_audit_sha256"],"Support audit changed")
    check(digest(ROOT/"tmp_analysis/full_mean_independent_preflight_20261023.json")==summary["independent_implementation_preflight_sha256"],"Implementation preflight changed")
    preflight=load(ROOT/"tmp_analysis/full_mean_independent_preflight_20261023.json")
    check(preflight["status"]=="PASS_INDEPENDENT_FULL_MEAN_IMPLEMENTATION_PREFLIGHT" and
          preflight["script_sha256"]==summary["script_sha256"] and preflight["protocol_sha256"]==summary["protocol_sha256"],"Preflight lineage differs")
    metadata_snapshot={str(p):digest(p) for p in (summary_path,ROOT/"tmp_analysis/diagnose_full_mean_quality.py",
                      ROOT/"tmp_analysis/full_mean_pair_gate_protocol_20261023.json",ROOT/"tmp_analysis/full_mean_independent_preflight_20261023.json",
                      support/"independent_support_audit.json",Path(independent.__file__))}
    rows,x,dump_count=read_eligible_features(support,inputs)
    with np.load(out/"features_predictions.npz",allow_pickle=False) as saved:
        check(saved["feature_names"].tolist()==independent.FEATURES and saved["model_names"].tolist()==NAMES,"Saved model/feature schema differs")
        check(saved["observation_id"].tolist()==[r["observation_id"] for r in rows],"Saved observation order differs")
        check(np.allclose(saved["x"],x,atol=1e-12,rtol=1e-12),"Saved features differ from independent raw extraction")
        y=saved["error"].copy();split=saved["split"].copy();groups=saved["source_group"].copy()
        trajectories=saved["trajectory"].copy();fold=saved["outer_fold"].copy();prediction=saved["prediction"].copy()
    check(y.tolist()==[r["error"] for r in rows] and split.tolist()==[r["split"] for r in rows],"Saved outcomes/splits differ")
    check(groups.tolist()==[r["source_group"] for r in rows] and trajectories.tolist()==["%d:%d"%(r["video_id"],r["gt_track_id"]) for r in rows],"Saved hierarchy IDs differ")
    check(np.isfinite(prediction).all() and np.all((prediction>=EPS)&(prediction<=1-EPS)),"Nonfinite/out-of-range predictions")
    fit=np.flatnonzero(split=="fit");tune=np.flatnonzero(split=="tune")
    check(not set(groups[fit]).intersection(groups[tune]),"Fit/tune source groups overlap")
    expected_fold,mapping=independent_folds(groups[fit],5,"full_mean_outer_v1")
    check(summary["outer_fold_mapping"]==mapping and np.array_equal(fold[fit],expected_fold) and np.all(fold[tune]==-1),"Saved outer folds differ")
    nested=joblib.load(out/"nested_audit.joblib");final=joblib.load(out/"final_fit_pipeline.joblib")
    check(np.array_equal(nested["fit_rows_in_full"],fit) and np.array_equal(nested["tune_rows_in_full"],tune),"Persisted fit/tune index scope differs")
    check(len(nested["outer"])==5 and len(summary["outer_training_receipts"])==5,"Missing outer receipts/models")
    pipeline_checks=[];prediction_delta=0.;model_count=0
    for bundle,info in zip(nested["outer"],summary["outer_training_receipts"]):
        f=bundle["outer_fold"];train_in_fit=np.flatnonzero(expected_fold!=f);valid_in_fit=np.flatnonzero(expected_fold==f)
        check(np.array_equal(bundle["training_rows_in_fit"],train_in_fit) and np.array_equal(bundle["validation_rows_in_fit"],valid_in_fit),"Persisted outer indices differ")
        training,validation=fit[train_in_fit],fit[valid_in_fit]
        check(info["outer_fold"]==f and info["validation_groups"]==sorted(set(groups[validation])) and info["validation_rows"]==len(validation),"Outer validation receipt differs")
        check(not set(groups[training]).intersection(groups[validation]),"Outer source overlap")
        train_rows=[rows[i] for i in training]
        selection=check_inner(bundle["inner"],info,x[training],y[training],groups[training],train_rows,grid)
        check(bundle["calibrator"]==info["calibrator"],"Saved outer Platt differs from receipt")
        models=[]
        for name,candidate in (("baseline",dict(name="mean",kind="mean")),("fixed",dict(name="ridge_1.0",kind="ridge",ridge=1.)),("selected",info["selected_candidate"])):
            models.append(model_contract(bundle[name],x[training],y[training],train_rows,candidate))
        prediction_delta=max(prediction_delta,check_pipeline_predictions(bundle,bundle["calibrator"],x[validation],prediction[validation]))
        model_count+=selection["inner_candidate_models_checked"]+3
        pipeline_checks.append(dict(outer_fold=f,selection=selection,outer_training_model_checks=models,
                                    train_sources=len(set(groups[training])),validation_sources=len(set(groups[validation])),source_disjoint=True))
    final_info=summary["final_all_fit_pipeline"];train_rows=[rows[i] for i in fit]
    selection=check_inner(nested["final_inner"],final_info,x[fit],y[fit],groups[fit],train_rows,grid)
    check(final["feature_names"]==independent.FEATURES and final["calibrator"]==final_info["calibrator"],"Saved final schema/calibrator differs")
    final_checks=[]
    for name,candidate in (("baseline",dict(name="mean",kind="mean")),("fixed",dict(name="ridge_1.0",kind="ridge",ridge=1.)),("selected",final_info["selected_candidate"])):
        final_checks.append(model_contract(final[name],x[fit],y[fit],train_rows,candidate))
    prediction_delta=max(prediction_delta,check_pipeline_predictions(final,final["calibrator"],x[tune],prediction[tune]))
    model_count+=selection["inner_candidate_models_checked"]+3
    metric_checks={}
    for label,indices in (("fit_nested_source_group_crossfit",fit),("tune_evaluation_only",tune)):
        subset=[rows[i] for i in indices]
        metric_checks[label]=check_summary(y[indices],prediction[indices],groups[indices],subset,summary["results"][label])
        reported_counts=summary["support_counts"]["fit" if label.startswith("fit") else "tune"]
        check(reported_counts==dict(rows=len(indices),errors=int(y[indices].sum()),source_groups=len(set(groups[indices])),
                                   GT_trajectories=len({(r["source_group"],r["video_id"],r["gt_track_id"]) for r in subset})),"Support counts differ")
    for path,value in inputs.items():check(digest(path)==value,"Mean input changed during audit")
    for path,value in support_receipt["inputs"].items():check(digest(path)==value,"Full support source changed during mean audit")
    for path,value in metadata_snapshot.items():check(digest(path)==value,"Mean audit metadata/source changed during audit")
    for filename,value in summary["output_hashes"].items():check(digest(out/filename)==value,"Mean output changed during audit")
    check(digest(summary_path)==summary_sha_before,"Diagnostic summary changed during audit")
    result=dict(status="PASS_INDEPENDENT_FULL_MEAN_QUALITY_DIAGNOSTIC_AUDIT",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                diagnostic_summary_sha256=digest(summary_path),producer_script_sha256=summary["script_sha256"],audit_script_sha256=digest(__file__),
                protocol_sha256=summary["protocol_sha256"],input_hash_count=len(inputs),eligible_raw_dump_files=dump_count,
                all_input_and_output_hashes_verified_before_after=True,all_eligible_features_rebuilt_from_raw=True,
                actual_emitted_error_targets_reverified=True,test_dumps_opened=False,test_or_class_holdout_correctness_opened=False,
                full_source_folds_disjoint=True,inner_and_outer_predictions_reconstructed=True,fit_only_weights_and_ridge_standardization_verified=True,
                nested_candidate_selection_and_Platt_recomputed=True,all_persisted_model_objects_checked=model_count,
                prediction_max_absolute_difference=prediction_delta,pipeline_checks=pipeline_checks,
                final_selection=selection,final_training_model_checks=final_checks,metric_checks=metric_checks,
                baseline_or_HGB_models_refitted=False,bootstrap_loss_intervals_independently_regenerated=True,
                limitations=["Persisted HGB model parameters and initial weighted prevalence checked; historical sample weights are established by reviewed producer code and independent synthetic tests, not by full model refitting.",
                             "Bootstrap intervals are conditional on fitted predictions; no training uncertainty or independent-observation claim.",
                             "Hard mean-error control only; no wrong-evidence-vector calibration, pair dependence or novelty gate."],elapsed_seconds=time.time()-started)
    write(audit_path,result)
    print(json.dumps(dict(status=result["status"],audit=str(audit_path),audit_sha256=digest(audit_path),
                          model_objects_checked=model_count,elapsed_seconds=result["elapsed_seconds"]),indent=2),flush=True)


if __name__=="__main__":main()
