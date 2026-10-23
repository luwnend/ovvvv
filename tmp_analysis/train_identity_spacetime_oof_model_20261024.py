#!/usr/bin/env python3
"""One source-OOF fold/seed: ordinary CPU adapter and geometry controls.

Real IO is gated by a final independent preflight. Synthetic entry points use
generated arrays only. No held-out identity loader exists in this module.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
MODULES=ROOT/"tmp_analysis/identity_spacetime_modules_20261024.py"
MODULES_SHA="2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3"
CONTRACT=ROOT/"tmp_analysis/identity_spacetime_followup_recommendation_20261024.md"
CONTRACT_SHA="23a1361129af172e201e10c0fefa82cf3e76c78ce640d824329652785f40f173"
SEEDS=(20261024,20261025,20261026)
MODES=("G0","G1","Gmask","Gperm")
ARMS=tuple("A%d%s"%(a,g) for a in (0,1) for g in MODES)
COLUMNS=("raw_appearance_cosine","adapter_appearance_cosine")+ARMS
CONFIG=dict(adapter_epochs=50,head_epochs=20,batch_size=64,learning_rate=1e-3,weight_decay=1e-4,gradient_clip=5.)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def canonical(value):
    def numpy_scalar(item):
        if isinstance(item,np.generic):return item.item()
        raise TypeError("Unsupported deterministic JSON value: "+type(item).__name__)
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False,default=numpy_scalar)


def object_sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path,value):
    with Path(path).open("x") as f:f.write(canonical(value)+"\n")


def module(path,name,expected=None):
    path=Path(path)
    require(expected is None or sha(path)==expected,"Module SHA mismatch: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    # dataclasses resolve their defining module through sys.modules.
    import sys
    sys.modules[name]=result;spec.loader.exec_module(result);return result


def normalized(raw):
    raw=np.asarray(raw,np.float64)
    require(raw.ndim==2 and raw.shape[1]==256 and np.isfinite(raw).all(),"Invalid frozen raw features")
    return raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)


def raw_cosine(query,reference):
    # Exactly the parent float64 NumPy normalization and GEMM expression.
    return normalized(query)@normalized(reference).T


def tensor(raw):return torch.as_tensor(np.asarray(raw,np.float64),dtype=torch.float64,device="cpu")


def tensor_state(model):
    return {name:dict(shape=list(value.shape),values=value.detach().cpu().numpy().tolist())
            for name,value in sorted(model.state_dict().items())}


def restore_state(model,state):
    expected=model.state_dict();require(set(state)==set(expected),"Checkpoint state names differ")
    rebuilt={}
    for name,target in expected.items():
        info=state[name];value=tensor(info["values"])
        require(list(target.shape)==info["shape"] and list(value.shape)==info["shape"] and torch.isfinite(value).all(),"Checkpoint shape/finite error")
        rebuilt[name]=value
    model.load_state_dict(rebuilt,strict=True);return model


def compact_examples(stream,train_sources,modules):
    """Full GT-free donor construction precedes the known-only loss gather.

    Unknown raw rows are not forwarded through the trainable adapter. Their
    GT-free geometry may still supply a fixed donor in the Gperm control.
    """
    allowed=set(train_sources);result=[]
    for example in stream:
        record=dict(example["record"]);require(record["source"] in allowed,"Held-out source in training")
        p=np.asarray(record["positive_rows"],np.int64);n=np.asarray(record["negative_rows"],np.int64)
        require(record["qgt"]>=0 and len(p) and len(n),"Only known P+N query-lags train")
        full_raw=np.asarray(example["reference_raw"],np.float64);query=np.asarray(example["query_raw"],np.float64)
        require(query.shape==(1,256) and full_raw.ndim==2 and full_raw.shape[1]==256 and np.isfinite(query).all() and np.isfinite(full_raw).all(),"Bad raw training arrays")
        categories=np.asarray(example["reference_predicted_categories"])
        history=np.asarray(example["reference_history_valid"],bool)
        dump_rows=np.asarray(example["reference_dump_rows"],np.int64)
        require(len(categories)==len(history)==len(dump_rows)==len(full_raw),"Incomplete retained reference metadata")
        plan=modules.gperm_donors(categories,history,dump_rows)
        supplied=example.get("gperm_plan")
        if supplied is not None:require(np.array_equal(plan.donor_rows,supplied.donor_rows),"Data Gperm plan differs from complete GT-free pool")
        known=np.sort(np.concatenate((p,n)))
        require(len(np.unique(known))==len(known) and len(known)>0 and known[0]>=0 and known[-1]<len(full_raw),"Invalid known P/N projection")
        values=np.asarray(example["values"],np.float64);masks=np.asarray(example["masks"],bool)
        require(values.shape==masks.shape==(len(full_raw),9) and np.isfinite(values[masks]).all(),"Invalid original geometry")
        local={int(old):new for new,old in enumerate(known)}
        record["positive_rows"]=np.asarray([local[int(i)] for i in p],np.int64)
        record["negative_rows"]=np.asarray([local[int(i)] for i in n],np.int64)
        result.append(dict(record=record,query_raw=query.copy(),reference_raw=full_raw[known].copy(),
            values=values[known].copy(),masks=masks[known].copy(),
            donor_values=values[plan.donor_rows[known]].copy(),donor_masks=masks[plan.donor_rows[known]].copy(),
            full_reference_count=len(full_raw),full_donor_rows=plan.donor_rows.copy(),
            selected_reference_dump_rows=dump_rows[known].copy(),full_reference_dump_rows=dump_rows.copy()))
    return result


def adapter_batch_losses(adapter,examples,indices,modules):
    selected=[examples[int(i)] for i in indices]
    query=tensor(np.concatenate([e["query_raw"] for e in selected]))
    references=tensor(np.concatenate([e["reference_raw"] for e in selected]))
    query_features=adapter(query);reference_features=adapter(references)
    sizes=[len(e["reference_raw"]) for e in selected]
    owners=torch.repeat_interleave(torch.arange(len(selected)),torch.as_tensor(sizes))
    score=(query_features[owners]*reference_features).sum(-1)
    losses=[];offset=0
    for e,size in zip(selected,sizes):
        record=e["record"]
        losses.append(modules.pairwise_loss(score[offset:offset+size],record["positive_rows"],record["negative_rows"],.1));offset+=size
    return torch.stack(losses)


def prepare_head_examples(examples,adapter,normalizer,modules):
    """Cache only known-edge frozen cosine and nine-value control inputs."""
    result=[]
    for example in examples:
        q=example["query_raw"];r=example["reference_raw"]
        raw=raw_cosine(q,r)[0]
        adapted=modules.frozen_appearance_cosine(tensor(q),tensor(r),adapter).numpy()[0]
        values=normalizer.transform(example["values"],example["masks"])
        donor=normalizer.transform(example["donor_values"],example["donor_masks"])
        modes={}
        for mode in MODES:
            vv=values.copy();mm=example["masks"].copy()
            if mode=="G0":vv[:]=0;mm[:]=False
            elif mode=="Gmask":vv[:,modules.POSITION_SLOTS]=0
            elif mode=="Gperm":vv[:,modules.POSITION_SLOTS]=donor[:,modules.POSITION_SLOTS]
            modes[mode]=np.concatenate((vv,mm.astype(np.float64)),axis=1)
        result.append(dict(record=example["record"],cosine=(raw,adapted),geometry_inputs=modes))
    return result


def head_batch_losses(head,examples,indices,appearance,mode,modules):
    selected=[examples[int(i)] for i in indices]
    blocks=[np.concatenate((e["cosine"][appearance][:,None],e["geometry_inputs"][mode]),axis=1) for e in selected]
    output=head(tensor(np.concatenate(blocks)))
    losses=[];offset=0
    for e,block in zip(selected,blocks):
        record=e["record"];size=len(block)
        losses.append(modules.pairwise_loss(output[offset:offset+size],record["positive_rows"],record["negative_rows"],1.));offset+=size
    return torch.stack(losses)


def epoch_batches(n,batch_size,random):
    order=random.permutation(n)
    return order,[order[i:i+batch_size] for i in range(0,n,batch_size)]


def train_model(examples,all_sources,train_sources,seed,modules,config=None):
    """Pure in-memory training; caller supplies only allowed training examples."""
    config=dict(CONFIG if config is None else config)
    require(config["adapter_epochs"]>=0 and config["head_epochs"]>=0 and config["batch_size"]>0,"Invalid schedule")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    weights,coverage=modules.hierarchical_weights([e["record"] for e in examples],all_sources,train_sources)
    require(len(examples)>0 and np.all(weights>0),"No supported training population or nonhard row included")
    n=len(examples)
    def normalizer_stream():
        for example,weight in zip(examples,weights):
            yield dict(source=example["record"]["source"],values=example["values"],masks=example["masks"],
                positive_rows=example["record"]["positive_rows"],negative_rows=example["record"]["negative_rows"],query_lag_weight=float(weight))
    normalizer=modules.fit_geometry_normalizer(normalizer_stream,train_sources)
    adapter=modules.IdentityAdapter256(seed)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    trace=dict(adapter=[],heads=[],coverage=coverage,global_weight_sum=float(weights.sum()),
        training_query_lags=n,adapter_optimizer_steps=0,head_optimizer_steps={a:0 for a in ARMS})
    rng=np.random.RandomState(seed)
    for epoch in range(config["adapter_epochs"]):
        order,batches=epoch_batches(n,config["batch_size"],rng);step_losses=[]
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            losses=adapter_batch_losses(adapter,examples,indices,modules)
            loss=modules.weighted_batch_loss(losses,weights[indices],n)
            require(torch.isfinite(loss),"Nonfinite adapter objective")
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),config["gradient_clip"],error_if_nonfinite=True);optimizer.step()
            trace["adapter_optimizer_steps"]+=1;step_losses.append(float(loss.detach()))
        trace["adapter"].append(dict(epoch=epoch+1,permutation_sha256=hashlib.sha256(np.asarray(order,np.int64).tobytes()).hexdigest(),
            batch_sizes=[len(b) for b in batches],surrogate_step_objective_mean=float(np.mean(step_losses))))
    modules.freeze_adapter(adapter);adapter_before=object_sha(tensor_state(adapter))
    frozen_examples=prepare_head_examples(examples,adapter,normalizer,modules)
    heads={arm:modules.ResidualAssociationHead19(seed) for arm in ARMS}
    optimizers={arm:torch.optim.AdamW(heads[arm].parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]) for arm in ARMS}
    rng=np.random.RandomState(seed)  # The same per-epoch permutation for all eight heads.
    for epoch in range(config["head_epochs"]):
        order,batches=epoch_batches(n,config["batch_size"],rng);step_losses={arm:[] for arm in ARMS}
        for indices in batches:
            for arm in ARMS:
                optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                appearance=int(arm[1]);mode=arm[2:]
                losses=head_batch_losses(heads[arm],frozen_examples,indices,appearance,mode,modules)
                loss=modules.weighted_batch_loss(losses,weights[indices],n)
                require(torch.isfinite(loss),"Nonfinite head objective")
                loss.backward();torch.nn.utils.clip_grad_norm_(heads[arm].parameters(),config["gradient_clip"],error_if_nonfinite=True);optimizer.step()
                trace["head_optimizer_steps"][arm]+=1;step_losses[arm].append(float(loss.detach()))
        trace["heads"].append(dict(epoch=epoch+1,permutation_sha256=hashlib.sha256(np.asarray(order,np.int64).tobytes()).hexdigest(),
            batch_sizes=[len(b) for b in batches],shared_permutation_all_arms=True,
            surrogate_step_objective_mean={arm:float(np.mean(values)) for arm,values in step_losses.items()}))
    require(object_sha(tensor_state(adapter))==adapter_before and all(not p.requires_grad and p.grad is None for p in adapter.parameters()),"Geometry head modified shared frozen adapter")
    for head in heads.values():
        head.eval()
        for p in head.parameters():p.requires_grad_(False);p.grad=None
    trace["adapter_frozen_parameter_sha256"]=adapter_before
    trace["schedule"]=config
    return dict(adapter=adapter,heads=heads,normalizer=normalizer,training_trace=trace)


def score_frame_pair(pair,model,modules):
    query=pair["query"];reference=pair["reference"]
    nq=len(query["raw"]);nm=0 if reference is None else len(reference["raw"])
    if not nq or not nm:
        moved=0
        if nm:
            plan=modules.gperm_donors(reference["predicted_categories"],np.isfinite(reference["previous_times"]),np.arange(nm,dtype=np.int64))
            moved=int(np.count_nonzero(plan.donor_rows!=np.arange(nm)))
        return np.empty((nq,nm,len(COLUMNS)),np.float64),dict(mapping_nonidentity_references=moved,mapping_nonidentity_candidates=moved*nq,geometry_values_changed_candidates=0)
    raw=raw_cosine(query["raw"],reference["raw"])
    adapted=modules.frozen_appearance_cosine(tensor(query["raw"]),tensor(reference["raw"]),model["adapter"])
    plan=modules.gperm_donors(reference["predicted_categories"],np.isfinite(reference["previous_times"]),np.arange(nm,dtype=np.int64))
    if pair.get("gperm_plan") is not None:require(np.array_equal(plan.donor_rows,pair["gperm_plan"].donor_rows),"Prediction donor map differs from full retained pool")
    output=np.empty((nq,nm,len(COLUMNS)),np.float64);output[:,:,0]=raw;output[:,:,1]=adapted.numpy()
    coverage=None
    with torch.no_grad():
        for column,arm in enumerate(ARMS,2):
            cosine=torch.from_numpy(raw) if arm[1]=="0" else adapted
            inputs,counts=modules.head_inputs(cosine,pair["geometry_values"],pair["geometry_masks"],model["normalizer"],arm[2:],plan if arm[2:]=="Gperm" else None)
            output[:,:,column]=model["heads"][arm](inputs).numpy()
            if arm=="A0Gperm":coverage=counts
    require(np.isfinite(output).all(),"Nonfinite complete candidate scores")
    return output,coverage


def frame_metadata(frame):
    def safe(value):
        if isinstance(value,np.ndarray):return safe(value.tolist())
        if isinstance(value,(list,tuple)):return [safe(v) for v in value]
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
        if isinstance(value,np.generic):return value.item()
        return value
    fields=("video_id","image_id","sampled_step","original_frame_index","boxes","released_ids","predicted_categories",
        "previous_boxes","previous_times","previous_locators","history_reason")
    return {name:safe(frame[name]) for name in fields if name in frame}


def write_predictions(pair_stream,frame_records,model,modules,out,parent_directory=None):
    """All retained QxM, including unknown and exact empty/prefix receipts."""
    out=Path(out);(out/"shards").mkdir()
    grouped=defaultdict(list)
    for record in frame_records:grouped[int(record["video_id"])].append(record)
    sizes={}
    for video,records in grouped.items():
        records.sort(key=lambda r:r["frame_id"])
        require([r["frame_id"] for r in records]==list(range(len(records))),"Missing sampled frame")
        for lag in (1,4):sizes[video,lag]=sum(records[i]["observations"]*records[i-lag]["observations"] for i in range(lag,len(records)))
    files={};maps={};offsets=Counter();metadata={};counts=Counter();seen_pairs=set();current=None;maximum=0;parent={};coverage=Counter()
    def close_video(video):
        if video is None:return
        for lag in (1,4):
            key=(video,lag);require(offsets[key]==sizes[key],"Incomplete held-out score rows")
            maps[lag].flush();del maps[lag]
            path=out/"shards"/("v%d_lag%d.scores.npy"%(video,lag));files[str(path.relative_to(out))]=sha(path)
        path=out/"shards"/("v%d.rows.json"%video);write(path,dict(video_id=video,frames=[metadata[k] for k in sorted(metadata)]));files[str(path.relative_to(out))]=sha(path)
        parent.clear();metadata.clear()
    receipt_path=out/"frame_pairs.jsonl"
    with receipt_path.open("x") as stream:
        for pair in pair_stream:
            query=pair["query"];reference=pair["reference"];video=int(pair["video_id"]);lag=int(pair["lag"])
            require(video in grouped and lag in (1,4),"Unexpected held-out frame pair")
            if current!=video:
                close_video(current);require(not any(v==video for v,l,s in seen_pairs),"Noncontiguous held-out video stream")
                current=video
                for ll in (1,4):
                    path=out/"shards"/("v%d_lag%d.scores.npy"%(video,ll));maps[ll]=np.lib.format.open_memmap(path,mode="w+",dtype=np.float64,shape=(sizes[video,ll],len(COLUMNS)))
                    if parent_directory is not None:
                        parent[ll]=np.load(Path(parent_directory)/"shards"/("v%d_lag%d.scores.npy"%(video,ll)),mmap_mode="r",allow_pickle=False)
            qstep=int(query["sampled_step"]);key=(video,lag,qstep);require(key not in seen_pairs,"Duplicate held-out pair")
            seen_pairs.add(key);metadata[qstep]=frame_metadata(query)
            expected_ref=qstep-lag
            require((reference is None)==(expected_ref<0),"Exact prefix/lag mismatch")
            if reference is not None:require(reference["sampled_step"]==expected_ref,"Skipped empty reference")
            block,permutation_coverage=score_frame_pair(pair,model,modules)
            nq,nm=block.shape[:2];count=nq*nm;maximum=max(maximum,count)
            require(count<=5929 and block.shape==(nq,nm,len(COLUMNS)),"Held-out block bound exceeded")
            offset=offsets[video,lag]
            if reference is not None:
                if parent_directory is not None:require(np.array_equal(block[:,:,0],parent[lag][offset:offset+count,0].reshape(nq,nm)),"Full raw column differs from sealed parent")
                maps[lag][offset:offset+count]=block.reshape(count,len(COLUMNS))
            offsets[video,lag]+=count;counts[str(lag)]+=count;coverage.update(permutation_coverage)
            receipt=dict(source=pair["source"],video_id=video,lag=lag,query_step=qstep,reference_step=None if reference is None else expected_ref,
                query_image_id=int(query["image_id"]),reference_image_id=None if reference is None else int(reference["image_id"]),
                query_rows=nq,reference_rows=nm,candidates=count,offset=offset,score_file="shards/v%d_lag%d.scores.npy"%(video,lag),
                original_frame_gap=None if reference is None else int(query["original_frame_index"]-reference["original_frame_index"]),gperm_coverage=permutation_coverage)
            stream.write(canonical(receipt)+"\n")
    close_video(current)
    require(len(seen_pairs)==2*len(frame_records) and {v for v,l,s in seen_pairs}==set(grouped),"Dropped held-out frames or zero-support videos")
    files[receipt_path.name]=sha(receipt_path)
    return dict(files_sha256=files,candidates_by_lag=dict(counts),frame_pair_receipts=len(seen_pairs),maximum_candidate_block=maximum,
        videos=len(grouped),gperm_coverage=dict(coverage),score_dtype="float64",columns=list(COLUMNS),parent_raw_scores_exactly_verified=parent_directory is not None)


def final_preflight(protocol,protocol_path):
    gate=protocol["implementation_preflight"];path=Path(gate["path"]);record=json.loads(path.read_text())
    require(record["status"]==gate["status"],"Independent trainer preflight absent")
    require(record["trainer_sha256"]==sha(__file__) and record["modules_sha256"]==MODULES_SHA and record["data_sha256"]==protocol["data_module_sha256"] and record["protocol_sha256"]==sha(protocol_path),"Final preflight pin mismatch")
    return {str(path):sha(path)}


def validate_real_config(spec):
    require(spec.get("status")=="FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL" and spec.get("schema")=="IDENTITY_SPACETIME_SOURCE_OOF_V1","Protocol is not frozen/launchable")
    require(spec.get("numeric_training_config")==CONFIG,"Frozen numeric training configuration differs")
    require(spec.get("score_dtype")=="float64" and tuple(spec.get("feature_names",spec.get("score_columns",())))==COLUMNS,"Frozen score contract differs")
    require(tuple(spec.get("score_columns",()))==COLUMNS and spec["score_columns"]==spec["feature_names"],"Feature names and score columns differ")
    training=spec["training"]
    for key in ("adapter_epochs","head_epochs","batch_size","learning_rate","weight_decay"):
        require(training[key]==CONFIG[key],"Training schedule differs: "+key)
    require(training.get("grad_norm_clip",training.get("gradient_clip"))==CONFIG["gradient_clip"],"Gradient clip differs")
    for key,value in (("device","cpu"),("dtype","float64"),("num_threads",1),("optimizer","AdamW"),
        ("adapter_pair_temperature",.1),("head_loss_temperature",1.),("normalizer_std_floor",1e-6),
        ("validation_or_early_stopping",False),("checkpoint_selection","fixed_final_epoch"),("seed_selection",False)):
        require(training.get(key)==value,"Frozen numerical/training contract differs: "+key)
    require(tuple(spec.get("seeds",()))==SEEDS and tuple(spec.get("folds",()))==tuple(range(5)) and tuple(spec.get("lags",()))==(1,4),"Fixed seeds/folds/lags differ")


def train_and_predict(fold,seed,protocol,out):
    """Root's gated single-model entry; never imports held-out label decoder."""
    started=time.monotonic();protocol_path=Path(protocol);spec=json.loads(protocol_path.read_text())
    require(fold in range(5) and seed in SEEDS,"Unregistered fold/seed")
    validate_real_config(spec)
    require(sha(MODULES)==MODULES_SHA and sha(CONTRACT)==CONTRACT_SHA,"Frozen modules/recommendation changed")
    gate_pins=final_preflight(spec,protocol_path)
    out=Path(out);require(not out.exists(),"Preserve previous model output");out.mkdir(parents=False)
    modules=module(MODULES,"identity_spacetime_trainer_modules",MODULES_SHA)
    data=module(spec["data_module_path"],"identity_spacetime_trainer_data",spec["data_module_sha256"])
    prepared=data.prepare_inputs(spec);manifest=prepared["manifest"];split=manifest["folds"][str(fold)]
    train_sources=split["train_sources"];heldout_sources=split["held_out_sources"];all_sources=sorted(manifest["groups"])
    require(set(train_sources).isdisjoint(heldout_sources) and set(train_sources)|set(heldout_sources)==set(all_sources),"Source OOF partition mismatch")
    print("[fold=%d seed=%d] GT-free preparation complete; elapsed_seconds=%.3f"%(fold,seed,time.monotonic()-started),flush=True)
    anchors,label_receipt=data.load_train_anchors(train_sources,prepared["roles"],prepared["video_sources"],prepared["authority"])
    train_videos=sorted(int(v) for v,s in prepared["video_sources"].items() if s in set(train_sources))
    heldout_videos=sorted(int(v) for v,s in prepared["video_sources"].items() if s in set(heldout_sources))
    def stream(videos):return data.iter_gtfree_frame_pairs(videos,frame_records=prepared["frame_records"],video_sources=prepared["video_sources"])
    train_coverage=Counter()
    examples=compact_examples(data.build_train_examples(stream(train_videos),anchors,train_sources,train_coverage),train_sources,modules)
    print("[fold=%d seed=%d] allowed training query-lags=%d; elapsed_seconds=%.3f"%(fold,seed,len(examples),time.monotonic()-started),flush=True)
    model=train_model(examples,all_sources,train_sources,seed,modules)
    print("[fold=%d seed=%d] training complete; elapsed_seconds=%.3f"%(fold,seed,time.monotonic()-started),flush=True)
    model["training_trace"]["loss_projection_coverage"]=dict(train_coverage)
    del examples,anchors
    lineage=dict(fold_manifest_sha256=object_sha(manifest),cache_sha256=object_sha(sorted((int(r["image_id"]),r["npz_sha256"]) for r in prepared["frame_records"])),
        modules_sha256=MODULES_SHA,data_module_sha256=spec["data_module_sha256"],recommendation_sha256=CONTRACT_SHA)
    metadata=dict(fold=fold,seed=seed,train_sources=sorted(train_sources),held_out_sources=sorted(heldout_sources),
        train_label_projection_sha256=label_receipt["train_label_projection_sha256"],GT_free_lineage=lineage,architecture="ordinary_256_64_256_adapter_19_32_1_heads",
        schedule=dict(CONFIG),final_adapter_epoch=50,final_head_epoch=20,columns=list(COLUMNS))
    checkpoint=dict(schema="IDENTITY_SPACETIME_DETERMINISTIC_CHECKPOINT_V1",metadata=metadata,adapter_state=tensor_state(model["adapter"]),head_states={arm:tensor_state(model["heads"][arm]) for arm in ARMS})
    write(out/"checkpoint.json",checkpoint)
    norm=model["normalizer"];write(out/"normalizer.json",dict(metadata=metadata,mean=norm.mean.tolist(),std=norm.std.tolist(),valid_marginal_weight=norm.valid_marginal_weight.tolist()))
    write(out/"training_trace.json",dict(metadata=metadata,**model["training_trace"]))
    heldout_records=[r for r in prepared["frame_records"] if int(r["video_id"]) in set(heldout_videos)]
    generated=write_predictions(stream(heldout_videos),heldout_records,model,modules,out,spec.get("parent_score_directory",ROOT/"results/fit_lag_identity_support_20261024_v1"))
    print("[fold=%d seed=%d] full held-out generation complete; candidates=%d; elapsed_seconds=%.3f"%(fold,seed,sum(generated["candidates_by_lag"].values()),time.monotonic()-started),flush=True)
    for name in ("checkpoint.json","normalizer.json","training_trace.json"):generated["files_sha256"][name]=sha(out/name)
    write(out/"GT_free_predictions_sealed.json",dict(status="SEALED_SINGLE_MODEL_FULL_HELDOUT_BEFORE_HELDOUT_LABELS",metadata=metadata,**generated,
        held_out_identity_values_decoded=False,all_retained_candidates_scored=True,unknown_candidates_filtered=False))
    # Full authority/protocol hashes intentionally live only outside model bytes.
    pins=dict(prepared["input_sha256"]);pins.update(gate_pins);pins.update({str(protocol_path):sha(protocol_path),str(Path(__file__).resolve()):sha(__file__)})
    for path,digest in pins.items():require(sha(path)==digest,"Input changed during single-model execution")
    for relative,digest in generated["files_sha256"].items():require(sha(out/relative)==digest,"Prediction member changed")
    result=dict(status="COMPLETE_SINGLE_FOLD_SEED_TRAINING_AND_FULL_HELDOUT_PREDICTION",fold=fold,seed=seed,
        script_sha256=sha(__file__),input_sha256=pins,train_projection_receipt=label_receipt,training_trace_sha256=sha(out/"training_trace.json"),
        global_model_prediction_seal_sha256=sha(out/"GT_free_predictions_sealed.json"),generated=generated,
        elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        held_out_identity_values_decoded=False,GPU_used=False,training_sources_only=True,overall_experiment_complete=False)
    write(out/"single_model_receipt.json",result);return result


def synthetic_examples(modules):
    rng=np.random.RandomState(31);stream=[]
    for i in range(5):
        q=rng.randn(1,256);r=rng.randn(4,256);r[0]=q[0]+.2*rng.randn(256);r[2]=0
        values=rng.randn(4,9);mask=np.ones((4,9),bool)
        stream.append(dict(record=dict(source="S" if i<4 else "T",video_id=1 if i<4 else 2,qgt=i//2,
            image_id=i//2,qrow=0,lag=1 if i%2==0 else 4,positive_rows=np.array([0]),negative_rows=np.array([1,3])),
            query_raw=q,reference_raw=r,values=values,masks=mask,reference_predicted_categories=np.ones(4),
            reference_history_valid=np.ones(4,bool),reference_dump_rows=np.arange(4),gperm_plan=modules.gperm_donors(np.ones(4),np.ones(4,bool),np.arange(4))))
    return compact_examples(stream,["S","T"],modules)


def synthetic(number):
    require(sha(MODULES)==MODULES_SHA,"Frozen modules changed");modules=module(MODULES,"identity_spacetime_trainer_synthetic_modules",MODULES_SHA)
    torch.set_num_threads(1);checks=[];examples=synthetic_examples(modules);config=dict(CONFIG,adapter_epochs=2,head_epochs=2,batch_size=3)
    if number==1:
        require(examples[0]["full_reference_count"]==4 and len(examples[0]["reference_raw"])==3 and examples[0]["full_donor_rows"].tolist()==[3,0,1,2],"Full-pool donors were projected early")
        require(np.array_equal(examples[0]["donor_values"][1],examples[0]["values"][0]),"Unknown donor gather changed")
        checks.append("known_only_adapter_rows_full_pool_Gperm_donors_before_supervision_projection")
        model=train_model(examples,["S","T","ZERO","HELD"],["S","T","ZERO"],20261024,modules,config)
        trace=model["training_trace"]
        require(trace["adapter_optimizer_steps"]==4 and all(n==4 for n in trace["head_optimizer_steps"].values()),"Schedule or final partial batch changed")
        require(all(e["batch_sizes"]==[3,2] for e in trace["adapter"]+trace["heads"]) and trace["coverage"]["zero_support_train_sources"]==["ZERO"],"Short batch/zero support changed")
        require(all(p.grad is None and not p.requires_grad for p in model["adapter"].parameters()),"Frozen adapter has geometry gradients")
        checks.append("actual_toy_optimizer_schedule_N_over_actualB_all_arms_same_permutation_adapter_frozen_zero_sources")
    elif number==2:
        weights,_=modules.hierarchical_weights([e["record"] for e in examples],["S","T"],["S","T"])
        adapter=modules.IdentityAdapter256(20261024);losses=adapter_batch_losses(adapter,examples,np.array([0,1]),modules)
        manual=[]
        for e in examples[:2]:manual.append(modules.adapter_pairwise_loss(adapter,tensor(e["query_raw"]),tensor(e["reference_raw"]),dict(hard=True,**e["record"])))
        require(torch.allclose(losses,torch.stack(manual),atol=1e-14,rtol=1e-14),"Vectorized gathered adapter differs from full PxN oracle")
        require(abs(modules.weighted_batch_loss(torch.tensor([2.,3.],dtype=torch.float64),[.1,.2],5).item()-2.)<1e-14,"N/B scaling changed")
        checks.append("batched_adapter_full_PxN_oracle_and_global_weight_short_batch_not_renormalized")
        model=train_model(examples,["S","T"],["S","T"],20261024,modules,config);normalizer=model["normalizer"]
        encoded=canonical(dict(adapter=tensor_state(model["adapter"]),heads={k:tensor_state(v) for k,v in model["heads"].items()}))
        rebuilt=restore_state(modules.IdentityAdapter256(1),json.loads(encoded)["adapter"])
        require(torch.equal(model["adapter"](tensor(examples[0]["query_raw"])),rebuilt(tensor(examples[0]["query_raw"]))),"Deterministic checkpoint reload differs")
        again=train_model(examples,["S","T"],["S","T"],20261024,modules,config)
        require(object_sha(tensor_state(again["adapter"]))==object_sha(tensor_state(model["adapter"])) and all(object_sha(tensor_state(again["heads"][k]))==object_sha(tensor_state(model["heads"][k])) for k in ARMS),"Identical train projection was not deterministic")
        restored_trace=json.loads(canonical(model["training_trace"]))
        require(all(type(value) is int for value in restored_trace["coverage"]["all_source_coverage"].values()),"NumPy integer coverage must serialize as JSON integers")
        checks.append("deterministic_JSON_checkpoint_roundtrip_and_repeat_train_projection_model_bytes")
    elif number==3:
        model=train_model(examples,["S","T"],["S","T"],20261024,modules,config)
        rng=np.random.RandomState(8);frames=[];records=[]
        for step,count in enumerate((2,0,3,2,1,2)):
            frame=dict(video_id=1,image_id=100+step,sampled_step=step,original_frame_index=(0,5,9,31,50,80)[step],
                raw=rng.randn(count,256),boxes=np.zeros((count,4)),released_ids=np.arange(count),predicted_categories=np.ones(count,dtype=int),
                previous_times=np.full(count,np.nan),previous_locators=[None]*count,history_reason=["missing"]*count)
            frames.append(frame);records.append(dict(video_id=1,frame_id=step,observations=count))
        pairs=[]
        for q in frames:
            for lag in (1,4):
                rr=frames[q["sampled_step"]-lag] if q["sampled_step"]>=lag else None
                shape=(len(q["raw"]),0 if rr is None else len(rr["raw"]),9)
                pairs.append(dict(source="H",video_id=1,lag=lag,query=q,reference=rr,geometry_values=np.zeros(shape),geometry_masks=np.zeros(shape,bool)))
        with tempfile.TemporaryDirectory(prefix="identity_trainer_complete_scores_") as td:
            out=Path(td);generated=write_predictions(iter(pairs),records,model,modules,out)
            require(generated["frame_pair_receipts"]==12 and generated["columns"]==list(COLUMNS),"Empty/prefix/columns lost")
            receipts=[json.loads(x) for x in (out/"frame_pairs.jsonl").read_text().splitlines()]
            for receipt,pair in zip(receipts,pairs):
                if pair["reference"] is None:require(receipt["candidates"]==0,"Prefix received scores");continue
                scores=np.load(out/receipt["score_file"],mmap_mode="r")[receipt["offset"]:receipt["offset"]+receipt["candidates"]]
                raw=raw_cosine(pair["query"]["raw"],pair["reference"]["raw"]).reshape(-1)
                require(np.array_equal(scores[:,0],raw),"Pure raw candidate column changed")
            require(sum(generated["candidates_by_lag"].values())==sum(len(p["query"]["raw"])*(0 if p["reference"] is None else len(p["reference"]["raw"])) for p in pairs),"Full candidate population changed")
        checks.append("temporary_full_retained_QxM_ten_columns_raw_exact_empty_frames_prefix_receipts_no_known_filter")
        with tempfile.TemporaryDirectory(prefix="identity_trainer_gate_") as td:
            training=dict(CONFIG,device="cpu",dtype="float64",num_threads=1,optimizer="AdamW",grad_norm_clip=5.,adapter_pair_temperature=.1,head_loss_temperature=1.,normalizer_std_floor=1e-6,validation_or_early_stopping=False,checkpoint_selection="fixed_final_epoch",seed_selection=False)
            protocol=Path(td)/"protocol.json";write(protocol,dict(status="FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL",schema="IDENTITY_SPACETIME_SOURCE_OOF_V1",numeric_training_config=dict(CONFIG),score_dtype="float64",feature_names=list(COLUMNS),score_columns=list(COLUMNS),training=training,seeds=list(SEEDS),folds=list(range(5)),lags=[1,4],implementation_preflight=dict(path=str(Path(td)/"missing"),status="PASS")))
            try:train_and_predict(0,20261024,protocol,Path(td)/"out")
            except FileNotFoundError:pass
            else:raise AssertionError("Missing preflight allowed real IO")
        checks.append("real_entry_missing_preflight_blocks_before_data_or_identity_import")
    else:raise ValueError("Three rounds only")
    return dict(status="PASS_IDENTITY_SPACETIME_SINGLE_MODEL_TRAINER_SYNTHETIC",round=number,checks=checks,script_sha256=sha(__file__),
        modules_sha256=MODULES_SHA,contract_sha256=CONTRACT_SHA,real_cache_or_identity_opened=False,GPU_used=False,
        toy_adapter_epochs=2,toy_head_epochs=2,real_training_executed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--protocol",type=Path);parser.add_argument("--fold",type=int);parser.add_argument("--seed",type=int);parser.add_argument("--output-directory","--out",dest="output_directory",type=Path)
    parser.add_argument("--synthetic-round",type=int,choices=(1,2,3));parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    require(bool(args.execute)!=bool(args.synthetic_round),"Choose exactly one real/synthetic mode")
    if args.synthetic_round:
        result=synthetic(args.synthetic_round)
        if args.synthetic_out:write(args.synthetic_out,result)
    else:
        require(args.protocol and args.fold is not None and args.seed is not None and args.output_directory,"Explicit gated protocol/fold/seed/output required")
        result=train_and_predict(args.fold,args.seed,args.protocol,args.output_directory)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
