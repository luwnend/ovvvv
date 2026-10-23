#!/usr/bin/env python3
"""Independent temporary 15-model structural seal and evaluator poison audit.

Only generated metadata/scores are used. This exercises the real coordinator
verification functions; it is not a numerical model or real-data output audit.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COORDINATOR=ROOT/"tmp_analysis/run_identity_spacetime_oof_20261024.py"
EVALUATOR=ROOT/"tmp_analysis/evaluate_identity_spacetime_oof_20261024.py"
SEEDS=(20261024,20261025,20261026)
ARMS=tuple("A%d%s"%(a,g) for a in (0,1) for g in ("G0","G1","Gmask","Gperm"))
COLUMNS=("raw_appearance_cosine","adapter_appearance_cosine")+ARMS
CONFIG=dict(adapter_epochs=50,head_epochs=20,batch_size=64,learning_rate=.001,weight_decay=.0001,gradient_clip=5.)


def check(value,message):
    if not value:raise AssertionError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result;spec.loader.exec_module(result);return result


def fixture(directory,coordinator):
    out=directory/"base";out.mkdir();(out/"models").mkdir();pins=directory/"pins";pins.mkdir()
    sources=["S%d"%fold for fold in range(5)];folds={str(f):dict(train_sources=[s for s in sources if s!=sources[f]],held_out_sources=[sources[f]]) for f in range(5)}
    prepared=dict(manifest=dict(groups={s:{} for s in sources},folds=folds),frame_records=[],video_sources={})
    sizes=(2,0,1,2,1,0);times=(0,5,9,31,50,80)
    for fold in range(5):
        video=fold+1;prepared["video_sources"][str(video)]=sources[fold]
        for step,count in enumerate(sizes):prepared["frame_records"].append(dict(video_id=video,frame_id=step,frame_index=times[step],image_id=100*video+step,observations=count))
    spec=dict(folds=list(range(5)),seeds=list(SEEDS),lags=[1,4],feature_names=list(COLUMNS),score_columns=list(COLUMNS),numeric_training_config=dict(CONFIG),
        resource_bounds=dict(candidates_by_lag={"1":20,"4":10}),population=dict(selected_frames=30),input_pins={},identity_authority={},interpretation="Generated structural fixture only")
    for name in ("modules","data_module","trainer","coordinator","evaluator","verifier","fold_manifest","finalizer"):
        p=pins/(name+".txt");p.write_text(name+" synthetic pin\n");spec[name+"_path"]=str(p);spec[name+"_sha256"]=sha(p)
    # Actual evaluator imports are intercepted at the GT-free coordinator/data boundary.
    spec["coordinator_path"]=str(COORDINATOR);spec["coordinator_sha256"]=sha(COORDINATOR)
    for name in ("manual","index"):
        p=pins/(name+".json");write(p,dict(synthetic=True));spec["identity_authority"][name+"_path"]=str(p);spec["identity_authority"][name+"_sha256"]=sha(p)
    p=pins/"input.txt";p.write_text("synthetic input\n");spec["input_pins"][str(p)]=sha(p)
    for seed in SEEDS:
        for fold in range(5):
            video=fold+1;dest=out/"models"/coordinator.model_name(fold,seed);dest.mkdir();(dest/"shards").mkdir()
            meta=dict(fold=fold,seed=seed,**folds[str(fold)],columns=list(COLUMNS),schedule=dict(CONFIG),final_adapter_epoch=50,final_head_epoch=20)
            files={};pairs=[]
            for lag in (1,4):
                n=sum(sizes[step]*sizes[step-lag] for step in range(lag,len(sizes)))
                path=dest/"shards"/("v%d_lag%d.scores.npy"%(video,lag));np.save(path,np.zeros((n,10),np.float64));files[str(path.relative_to(dest))]=sha(path)
            rows=dest/"shards"/("v%d.rows.json"%video)
            frames=[dict(video_id=video,image_id=100*video+s,sampled_step=s,original_frame_index=times[s],boxes=[[0.,0.,1.,1.]]*n,
                released_ids=list(range(n)),predicted_categories=[1]*n,previous_boxes=[[None]*4]*n,previous_times=[None]*n,previous_locators=[None]*n,history_reason=["missing"]*n) for s,n in enumerate(sizes)]
            write(rows,dict(video_id=video,frames=frames));files[str(rows.relative_to(dest))]=sha(rows)
            offsets={1:0,4:0}
            for step,nq in enumerate(sizes):
                for lag in (1,4):
                    ref=step-lag if step>=lag else None;nm=0 if ref is None else sizes[ref];n=nq*nm
                    pairs.append(dict(source=sources[fold],video_id=video,lag=lag,query_step=step,reference_step=ref,query_image_id=100*video+step,reference_image_id=None if ref is None else 100*video+ref,
                        query_rows=nq,reference_rows=nm,candidates=n,offset=offsets[lag],score_file="shards/v%d_lag%d.scores.npy"%(video,lag),original_frame_gap=None if ref is None else times[step]-times[ref]))
                    offsets[lag]+=n
            (dest/"frame_pairs.jsonl").write_text("".join(json.dumps(pair)+"\n" for pair in pairs));files["frame_pairs.jsonl"]=sha(dest/"frame_pairs.jsonl")
            for name in ("checkpoint.json","normalizer.json","training_trace.json"):
                # Structural member fixtures: independent numerical audits own tensor validation.
                value=dict(metadata=meta,synthetic_structural_member=True)
                if name=="training_trace.json":value.update(training_query_lags=65,adapter_optimizer_steps=100,head_optimizer_steps={arm:40 for arm in ARMS},
                    adapter=[dict(epoch=i+1,batch_sizes=[64,1]) for i in range(50)],heads=[dict(epoch=i+1,batch_sizes=[64,1]) for i in range(20)])
                write(dest/name,value);files[name]=sha(dest/name)
            generated=dict(files_sha256=files,candidates_by_lag={"1":4,"4":2},frame_pair_receipts=12,parent_raw_scores_exactly_verified=True)
            write(dest/"GT_free_predictions_sealed.json",dict(status="SEALED_SINGLE_MODEL_FULL_HELDOUT_BEFORE_HELDOUT_LABELS",metadata=meta,all_retained_candidates_scored=True,unknown_candidates_filtered=False,**generated))
            write(dest/"single_model_receipt.json",dict(status="COMPLETE_SINGLE_FOLD_SEED_TRAINING_AND_FULL_HELDOUT_PREDICTION",fold=fold,seed=seed,script_sha256=spec["trainer_sha256"],held_out_identity_values_decoded=False,GPU_used=False,training_sources_only=True,
                global_model_prediction_seal_sha256=sha(dest/"GT_free_predictions_sealed.json"),generated=generated))
    return out,spec,prepared


def revise_model(directory,mutator=None,rehash_member=None):
    seal=read(directory/"GT_free_predictions_sealed.json");receipt=read(directory/"single_model_receipt.json")
    if rehash_member is not None:seal["files_sha256"][rehash_member]=sha(directory/rehash_member)
    if mutator is not None:mutator(seal,receipt)
    write(directory/"GT_free_predictions_sealed.json",seal);receipt["global_model_prediction_seal_sha256"]=sha(directory/"GT_free_predictions_sealed.json")
    receipt["generated"]["files_sha256"]=copy.deepcopy(seal["files_sha256"])
    write(directory/"single_model_receipt.json",receipt)


def run():
    before={str(p):sha(p) for p in (COORDINATOR,EVALUATOR,Path(__file__))}
    c=module(COORDINATOR,"independent_structural_coordinator");e=module(EVALUATOR,"independent_poison_evaluator")
    checks=[];matrix=[]
    with tempfile.TemporaryDirectory(prefix="identity_oof_15model_seal_") as td:
        directory=Path(td);base,spec,prepared=fixture(directory,c);c.prepared_inputs=lambda unused:prepared
        seal=c.seal_models(base,spec);verified=c.verify_global_seal(base,spec)
        check(seal==verified and len(verified["models"])==15,"Complete synthetic 15-model seal rejected")
        check(all(v["frame_pair_receipts"]==12 for v in verified["models"].values()),"Synthetic empty/prefix receipts lost")
        checks.append("actual_seal_models_verify_models_verify_model_and_global_seal_complete_15model_positive_control")
        reached=[]
        def poison_module(path,name,expected=None):
            if Path(path)==COORDINATOR:return c
            reached.append(name);raise AssertionError("SYNTHETIC_IDENTITY_IMPORT_REACHED")
        e.module=poison_module
        # A valid seal reaches the intended external-data boundary.
        try:e.evaluate(base,spec)
        except AssertionError as exc:check(str(exc)=="SYNTHETIC_IDENTITY_IMPORT_REACHED","Unexpected valid-gate failure")
        else:raise AssertionError("Valid seal did not reach poison boundary")
        check(reached==["oof_unified_evaluation_data"],"Valid gate boundary differs");reached.clear()
        checks.append("actual_evaluator_positive_gate_reaches_only_the_explicit_postseal_data_import_boundary")

        def one_model(out):return out/"models"/c.model_name(0,SEEDS[0])
        def global_mutation(out,**updates):
            value=read(out/c.GLOBAL_SEAL);value.update(updates);write(out/c.GLOBAL_SEAL,value)
        def score_shape(out,columns=9,dtype=np.float64):
            model=one_model(out);relative="shards/v1_lag1.scores.npy";np.save(model/relative,np.zeros((4,columns),dtype));revise_model(model,rehash_member=relative)
        def change_meta(out,key,value):revise_model(one_model(out),lambda seal,receipt:seal["metadata"].__setitem__(key,value))
        def change_flag(out,key,value):revise_model(one_model(out),lambda seal,receipt:seal.__setitem__(key,value))
        def change_trace(out,mutate):
            model=one_model(out);path=model/"training_trace.json";value=read(path);mutate(value);write(path,value);revise_model(model,rehash_member=path.name)
        def change_pairs(out,mutate):
            model=one_model(out);path=model/"frame_pairs.jsonl";value=[json.loads(line) for line in path.read_text().splitlines()];mutate(value)
            path.write_text("".join(json.dumps(pair)+"\n" for pair in value));revise_model(model,rehash_member=path.name)
        cases=[
            ("missing_global_seal",lambda out:(out/c.GLOBAL_SEAL).unlink()),
            ("global_14models",lambda out:global_mutation(out,model_count=14)),
            ("global_missing_seed",lambda out:global_mutation(out,seeds=list(SEEDS[:2]))),
            ("missing_fold_directory",lambda out:shutil.rmtree(one_model(out))),
            ("missing_entire_seed",lambda out:[shutil.rmtree(out/"models"/c.model_name(f,SEEDS[1])) for f in range(5)]),
            ("unexpected_model_directory",lambda out:(out/"models"/"fold5_seed20261024").mkdir()),
            ("missing_score_member",lambda out:(one_model(out)/"shards/v1_lag1.scores.npy").unlink()),
            ("missing_checkpoint",lambda out:(one_model(out)/"checkpoint.json").unlink()),
            ("unexpected_model_file",lambda out:(one_model(out)/"unexpected.txt").write_text("unexpected")),
            ("unrepinned_score_tamper",lambda out:(one_model(out)/"shards/v1_lag1.scores.npy").write_bytes(b"tamper")),
            ("nine_score_columns_rehashed_model",lambda out:score_shape(out,9)),
            ("float32_scores_rehashed_model",lambda out:score_shape(out,10,np.float32)),
            ("missing_arm_metadata",lambda out:change_meta(out,"columns",list(COLUMNS[:-1]))),
            ("wrong_fold_metadata",lambda out:change_meta(out,"fold",1)),
            ("incomplete_head_epoch",lambda out:change_meta(out,"final_head_epoch",19)),
            ("missing_heldout_source",lambda out:change_meta(out,"held_out_sources",[])),
            ("incomplete_frame_receipts",lambda out:change_flag(out,"frame_pair_receipts",11)),
            ("unknown_candidates_filtered",lambda out:change_flag(out,"unknown_candidates_filtered",True)),
            ("raw_parent_parity_absent",lambda out:change_flag(out,"parent_raw_scores_exactly_verified",False)),
            ("wrong_candidate_totals",lambda out:change_flag(out,"candidates_by_lag",{"1":3,"4":2})),
            ("single_model_pending_status",lambda out:revise_model(one_model(out),lambda seal,receipt:receipt.__setitem__("status","PENDING"))),
            ("global_columns_tamper",lambda out:global_mutation(out,columns=list(COLUMNS[:-1]))),
            ("missing_receipt_member_hash_selfconsistent",lambda out:change_pairs(out,lambda pairs:pairs.pop())),
            ("pair_offset_member_hash_selfconsistent",lambda out:change_pairs(out,lambda pairs:pairs[0].__setitem__("offset",1))),
            ("adapter_trace_49epochs_member_hash_selfconsistent",lambda out:change_trace(out,lambda trace:trace["adapter"].pop())),
            ("head_trace_19epochs_member_hash_selfconsistent",lambda out:change_trace(out,lambda trace:trace["heads"].pop())),
            ("head_actual_steps_incomplete",lambda out:change_trace(out,lambda trace:trace["head_optimizer_steps"].__setitem__(ARMS[0],39))),
            ("final_short_batch_coverage_wrong",lambda out:change_trace(out,lambda trace:trace["adapter"][-1].__setitem__("batch_sizes",[64,64]))),
        ]
        for name,mutate in cases:
            output=directory/name;shutil.copytree(base,output);mutate(output);reached.clear()
            try:e.evaluate(output,spec)
            except (RuntimeError,ValueError,FileNotFoundError) as exc:
                check(not reached,"Failed seal reached identity/data import: "+name)
                matrix.append(dict(case=name,status="REJECTED_BEFORE_IDENTITY_IMPORT",exception=type(exc).__name__,reason=str(exc).replace(str(directory),"<temporary>")))
            else:raise AssertionError("Invalid seal accepted: "+name)
        # Pin mutation is exercised last because every copied seal references it.
        pinned=Path(next(iter(spec["input_pins"])));pinned.write_text("changed pinned bytes")
        try:e.evaluate(base,spec)
        except RuntimeError as exc:
            check(not reached,"Changed pin reached identity import");matrix.append(dict(case="changed_input_pin",status="REJECTED_BEFORE_IDENTITY_IMPORT",exception=type(exc).__name__,reason=str(exc).replace(str(directory),"<temporary>")))
        else:raise AssertionError("Changed input pin accepted")
        checks.append("actual_evaluator_entry_rejection_matrix_before_data_or_identity_import")
        # Evidence publication is non-replacing even with a complete old seal.
        try:c.seal_models(base,spec)
        except FileExistsError:checks.append("existing_global_seal_preserved_by_nonreplacing_publication")
        else:raise AssertionError("Existing global seal replaced")
    check(before=={p:sha(p) for p in before},"Audited implementation changed during synthetic audit")
    return dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_COORDINATOR_SEAL_SYNTHETIC",checks=checks,rejection_matrix=matrix,rejected_cases=len(matrix),
        coordinator_sha256=before[str(COORDINATOR)],evaluator_sha256=before[str(EVALUATOR)],script_sha256=before[str(Path(__file__))],
        real_cache_or_identity_opened=False,real_training_executed=False,generated_models=15,score_fixture_only=True,
        limits="Structural complete-model publication and pre-identity gate audit only; independent numerical output audit validates tensor state and learned values.")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path);args=parser.parse_args();result=run()
    if args.out:
        with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
