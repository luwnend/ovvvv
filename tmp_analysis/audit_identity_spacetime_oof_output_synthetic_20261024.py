#!/usr/bin/env python3
"""Temporary full 15-worker source-OOF output audit, no real data is opened."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
from collections import Counter,defaultdict,deque
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import verify_identity_spacetime_oof_output_20261024 as v

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path("/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python")


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def run():
    started=time.monotonic()
    trainer_path=ROOT/"tmp_analysis/train_identity_spacetime_oof_model_20261024.py";data_path=ROOT/"tmp_analysis/identity_spacetime_oof_data_20261024.py";coordinator_path=ROOT/"tmp_analysis/run_identity_spacetime_oof_20261024.py";evaluator_path=ROOT/"tmp_analysis/evaluate_identity_spacetime_oof_20261024.py"
    t=module(trainer_path,"oof_synthetic_actual_trainer");d=module(data_path,"oof_synthetic_actual_data");c=module(coordinator_path,"oof_synthetic_actual_coordinator");utility=v.audited_utilities()
    with tempfile.TemporaryDirectory(prefix="independent_oof_full15_output_") as tmp:
        base=Path(tmp);cache=base/"cache";cache.mkdir();out=base/"out";(out/"models").mkdir(parents=True);parent=base/"parent";(parent/"shards").mkdir(parents=True)
        execution=base/"execution";logs=execution/"model_processes";logs.mkdir(parents=True)
        roles={"D/source%d"%i:dict(split="fit",role="mean_role",block="D") for i in range(5)};manifest=v.independent_folds(roles)
        videos={str(i+1):s for i,s in enumerate(sorted(roles))};records=[];manual=[];index=[];anchors={};rng=np.random.RandomState(70724)
        for video_text,source in videos.items():
            video=int(video_text)
            for step,time_index in enumerate((0,5,9,31,50,80)):
                n=0 if video==2 and step==2 else 3;image=video*100+step;path=cache/("v%d_f%d.npz"%(video,step));raw=rng.randn(n,256).astype(np.float16)
                raw[0:1]=0 if video==3 and step==0 else raw[0:1]
                boxes=np.asarray([[time_index+row*20,0,time_index+row*20+10,10,1] for row in range(n)],float).reshape(n,5)
                np.savez(path,video_id=video,image_id=image,frame_id=step,frame_index=time_index,det_bboxes=boxes,track_feats=raw,track_ids=np.arange(n),det_labels=np.zeros(n,np.int32))
                records.append(dict(video_id=video,image_id=image,frame_id=step,frame_index=time_index,observations=n,npz=str(path),npz_sha256=v.sha(path)))
                for row in range(n):
                    if row==1 and step==0:continue
                    oid=len(manual);item=dict(split="fit",observation_id=oid,source_group=source,video_id=video,image_id=image,sampled_step=step,original_frame_index=time_index,local_track_id=row,gt_track_id=7 if row<2 else 8,manual_anchor_id=oid,eligible_for_parameter_fit=True,pred_category_id=1)
                    manual.append(dict(item,gt_category_id="POISON_FORBIDDEN_CLASS",error="POISON_FORBIDDEN_ERROR"));index.append({**{k:item[k] for k in d.INDEX_FIELDS if k!="dump_row"},"dump_row":row});anchors[image,row]=(video,item["gt_track_id"])
        roles_path,plan_path,frames_path,inventory_path,fold_path,manual_path,index_path=(base/n for n in ("roles.json","plan.json","frames.jsonl","inventory.json","folds.json","manual.jsonl","index.jsonl"))
        write(roles_path,dict(groups=roles));write(plan_path,dict(video_records=[dict(video_id=int(vid),source_group=s,split="fit") for vid,s in videos.items()]));write(inventory_path,{Path(r["npz"]).name:r["npz_sha256"] for r in records});write(fold_path,manifest)
        frames_path.write_text("".join(json.dumps(r)+"\n" for r in records));manual_path.write_text("".join(json.dumps(r)+"\n" for r in manual)+json.dumps(dict(split="tune",gt_track_id="POISON_TUNE_ID"))+"\n");index_path.write_text("".join(json.dumps(r)+"\n" for r in index))
        grouped=defaultdict(list)
        for r in records:grouped[r["video_id"]].append(r)
        parentdb=v.open_db(base/"parent.sqlite");cells={(t.SEEDS[0],s,l,scope,pop):v.fresh() for s in roles for l in (1,4) for scope in v.SCOPES for pop in v.POPS};candidate_totals=Counter()
        for video,vr in sorted(grouped.items()):
            history={};ring=deque(maxlen=5);blocks=defaultdict(list);source=videos[str(video)]
            for step,record in enumerate(vr):
                q=utility.read_raw(record,history);ring.append(q);qi=np.array([anchors.get((q["image_id"],i),(video,-1))[1] for i in range(len(q["raw"]))],np.int64)
                for lag in (1,4):
                    if step<lag:continue
                    r=ring[-lag-1];nq,nm=len(q["raw"]),len(r["raw"]);raw=v.unit(q["raw"])@v.unit(r["raw"]).T;blocks[lag].append(raw.reshape(-1));candidate_totals[str(lag)]+=raw.size
                    score=np.repeat(raw[:,:,None],10,axis=2);ri=np.array([anchors.get((r["image_id"],i),(video,-1))[1] for i in range(nm)],np.int64);same=q["predicted_categories"][:,None]==r["predicted_categories"][None,:];cv=np.isfinite(r["previous_times"]);zeros=np.zeros((nq,nm),bool)
                    for scope,selected in ((v.SCOPES[0],np.ones((nq,nm),bool)),(v.SCOPES[1],same)):
                        for pop,mask in ((v.POPS[0],selected),(v.POPS[1],selected&cv[None,:])):v.add_block(cells[t.SEEDS[0],source,lag,scope,pop],score,mask,qi,ri,zeros,zeros,(t.SEEDS[0],source,lag,scope,pop,video,q["image_id"]),parentdb)
            for lag in (1,4):
                raw=np.concatenate(blocks[lag]);scores=np.zeros((len(raw),6));scores[:,0]=raw;np.save(parent/"shards"/("v%d_lag%d.scores.npy"%(video,lag)),scores)
        parent_report=v.reports(parentdb,cells,roles,[t.SEEDS[0]])["per_seed"][str(t.SEEDS[0])];parentdb.close();write(parent/"summary.json",dict(results=parent_report));shared=parent_report["cross_lag"][v.PRIMARY[0]][v.PRIMARY[1]]["shared_hard_queries"]
        protected={}
        for i in range(7):p=base/("protected_%d"%i);p.write_text("frozen %d"%i);protected[str(p)]=dict(exists=True,sha256=v.sha(p))
        protected_path=base/"protected.json";write(protected_path,dict(protected_files=protected))
        protocol=base/"protocol.json";gate=base/"preflight.json"
        spec=dict(status="FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL",schema="IDENTITY_SPACETIME_SOURCE_OOF_V1",score_dtype="float64",score_columns=list(t.COLUMNS),feature_names=list(t.COLUMNS),numeric_training_config=dict(t.CONFIG),
            training=dict(device="cpu",dtype="float64",num_threads=1,optimizer="AdamW",adapter_epochs=50,head_epochs=20,batch_size=64,learning_rate=.001,weight_decay=.0001,grad_norm_clip=5.,adapter_pair_temperature=.1,head_loss_temperature=1.,normalizer_std_floor=1e-6,validation_or_early_stopping=False,checkpoint_selection="fixed_final_epoch",seed_selection=False),
            seeds=list(t.SEEDS),folds=list(range(5)),lags=[1,4],source_roles_path=str(roles_path),collection_plan_path=str(plan_path),frame_execution_path=str(frames_path),dump_inventory_path=str(inventory_path),fold_manifest_path=str(fold_path),
            identity_authority=dict(manual_path=str(manual_path),manual_sha256=v.sha(manual_path),index_path=str(index_path),index_sha256=v.sha(index_path)),
            input_pins={str(p):v.sha(p) for p in (roles_path,plan_path,frames_path,inventory_path,fold_path,*d.PINS)},
            implementation_preflight=dict(path=str(gate),status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_IMPLEMENTATION_PREFLIGHT"),parent_score_directory=str(parent),protected_authority_path=str(protected_path),execution_directory=str(execution),output_directory=str(out),
            population=dict(sealed_fit_source_groups=5,allocated_fit_videos=5,selected_fit_videos=5,selected_frames=len(records),retained_observations=sum(r["observations"] for r in records)),
            resource_bounds=dict(candidates_by_lag=dict(candidate_totals)),interpretation="SYNTHETIC ONLY temporary all-15 worker output audit")
        for key,path in dict(modules=d.MODULES,data_module=data_path,trainer=trainer_path,coordinator=coordinator_path,evaluator=evaluator_path,verifier=Path(v.__file__),fold_manifest=fold_path,finalizer=ROOT/"tmp_analysis/finalize_identity_spacetime_oof_stage_20261024.py").items():spec[key+"_path"]=str(path);spec[key+"_sha256"]=v.sha(path)
        write(protocol,spec);write(gate,dict(coordinator_sha256=v.sha(coordinator_path),evaluator_sha256=v.sha(evaluator_path),verifier_sha256=v.sha(v.__file__),status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_IMPLEMENTATION_PREFLIGHT",trainer_sha256=v.sha(trainer_path),modules_sha256=v.sha(d.MODULES),data_sha256=v.sha(data_path),protocol_sha256=v.sha(protocol)))
        actual=[]
        for seed in t.SEEDS:
            for fold in range(5):
                name="fold%d_seed%d"%(fold,seed);command=[str(PYTHON),"-u",str(trainer_path),"--execute","--fold",str(fold),"--seed",str(seed),"--protocol",str(protocol),"--out",str(out/"models"/name)]
                actual.append(c.child(command,logs,name));print("synthetic actual worker finished "+name,flush=True)
        c.seal_models(out,spec);c.verify_global_seal(out,spec)
        # Only the fixed real-data 1331 anchor is replaced for temporary five-video
        # data. Actual evaluator evaluate() / seal / all score / SQLite / reports run.
        code="import sys,functools;sys.path.insert(0,sys.argv[1]);import evaluate_identity_spacetime_oof_20261024 as e;e.baseline_parity=functools.partial(e.baseline_parity,expected_shared="+str(shared)+");e.evaluate(sys.argv[2],sys.argv[3])"
        evaluator_exit=c.child([str(PYTHON),"-u","-c",code,str(ROOT/"tmp_analysis"),str(out),str(protocol),str(evaluator_path)],logs,"unified_evaluation")
        c.write_new(out/"coordinator_receipt.json",dict(status="COMPLETE_ALL_MODELS_GLOBAL_SEAL_AND_UNIFIED_EVALUATION_PENDING_INDEPENDENT_AUDIT",actual_model_processes=actual,evaluation_process=evaluator_exit,protocol_sha256=v.sha(protocol),global_seal_sha256=v.sha(out/"GT_free_OOF_predictions_sealed.json"),GPU_used=False))
        original=v.verify_parent_report;v.verify_parent_report=functools.partial(original,expected_shared=shared)
        try:
            result=v.audit(out,protocol)
        finally:v.verify_parent_report=original
        v.check(result["status"]=="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT","Actual complete audit failed")
        # Independent reconstruction is challenged after fresh hash seals update;
        # semantic score corruption cannot hide behind a self-consistent digest.
        model=out/"models"/"fold0_seed20261024";shard=next((model/"shards").glob("*.scores.npy"));score=np.load(shard);score[0,7]+=.02;np.save(shard,score)
        rejection=False
        try:v.global_seal_gate(out,spec,manifest,videos,grouped)
        except AssertionError:rejection=True
        v.check(rejection,"Tampered model shard passed independent seal gate")
        return dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_COMPLETE_OUTPUT_SYNTHETIC",script_sha256=v.sha(__file__),verifier_sha256=v.sha(v.__file__),trainer_sha256=v.sha(trainer_path),data_sha256=v.sha(data_path),coordinator_sha256=v.sha(coordinator_path),evaluator_sha256=v.sha(evaluator_path),actual_worker_processes=15,actual_worker_exit_codes=[r["actual_exit_code"] for r in actual],actual_evaluator_exit_code=evaluator_exit["actual_exit_code"],complete_audit_status=result["status"],all_ten_scores_full_stream_and_independent_SQLite_all_reports_verified=True,training_lineage=result["independent_checks"]["training_lineage"],maximum_absolute_score_error=result["independent_checks"]["maximum_absolute_score_error"],tampered_score_independent_global_gate_rejected=True,temporary_shared_query_population=shared,synthetic_authority_overrides=["all cache/labels/roles/folds/protocol/protected/parent files temporary","temporary preflight gate for actual default 50/20 workers","real-data parent shared count1331 replaced by temporary independently derived count in evaluator baseline check and verifier parent check only"],real_cache_or_manual_labels_opened=False,elapsed_seconds=time.monotonic()-started)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);args=p.parse_args();result=run()
    if args.out:
        with args.out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2),flush=True)
