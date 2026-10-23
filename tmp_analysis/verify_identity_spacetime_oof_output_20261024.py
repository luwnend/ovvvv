#!/usr/bin/env python3
"""Independent numerical reconstruction for the pending source-OOF output audit.

Reconstruct every retained candidate and every report from pinned raw caches,
checkpoint tensors and safe eligible identities after an independent all-model
seal check. The real path imports no producer or evaluator forward function.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = ("raw_appearance_cosine", "adapter_appearance_cosine", "A0G0", "A0G1", "A0Gmask", "A0Gperm", "A1G0", "A1G1", "A1Gmask", "A1Gperm")
SCORE_ATOL = 1e-10
SCORE_RTOL = 1e-10
POSITION_SLOTS = (0, 1, 2, 3, 5, 6, 7)


def check(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(values):
    values = np.asarray(values, np.float64)
    check(values.ndim == 2 and np.isfinite(values).all(), "Expected finite feature rows")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def gelu_erf(values):
    """Independent formula, without nn.GELU/F.gelu or subject imports."""
    tensor = torch.from_numpy(np.asarray(values, np.float64))
    return (.5 * tensor * (1 + torch.erf(tensor / np.sqrt(2.)))).numpy()


def arrays(state, first_width, hidden, output):
    shapes = {"first.weight": (hidden, first_width), "first.bias": (hidden,),
              "last.weight": (output, hidden), "last.bias": (output,)}
    check(set(state) == set(shapes), "Checkpoint contains undeclared tensor names")
    result = {}
    for key, shape in shapes.items():
        saved = state[key]
        if isinstance(saved, dict):
            check(set(saved) == {"shape", "values"} and saved["shape"] == list(shape), "Declared checkpoint tensor shape differs")
            saved = saved["values"]
        result[key] = np.asarray(saved, np.float64)
    for key, shape in shapes.items():
        check(result[key].shape == shape and np.isfinite(result[key]).all(), "Invalid checkpoint tensor: " + key)
    return result


def adapter_forward(raw, state):
    weights = arrays(state, 256, 64, 256)
    source = np.asarray(raw, np.float64)
    initial = unit(source)
    hidden = gelu_erf(initial @ weights["first.weight"].T + weights["first.bias"])
    residual = hidden @ weights["last.weight"].T + weights["last.bias"]
    return unit(initial + residual) * (np.linalg.norm(source, axis=1) > 0)[:, None]


def head_forward(inputs, state):
    weights = arrays(state, 19, 32, 1)
    features = np.asarray(inputs, np.float64)
    check(features.shape[-1] == 19 and np.isfinite(features).all(), "Invalid independent head input")
    hidden = gelu_erf(features @ weights["first.weight"].T + weights["first.bias"])
    return features[..., 0] / .1 + (hidden @ weights["last.weight"].T + weights["last.bias"])[..., 0]


def independent_geometry(query, reference):
    """Nine geometry slots, derived directly from immutable original boxes."""
    qb, rb = np.asarray(query["boxes"], float), np.asarray(reference["boxes"], float)
    nq, nr = len(qb), len(rb)
    qsize, rsize = qb[:, 2:] - qb[:, :2], rb[:, 2:] - rb[:, :2]
    qcenter, rcenter = (qb[:, :2] + qb[:, 2:]) / 2, (rb[:, :2] + rb[:, 2:]) / 2
    scale = np.sqrt(qsize[:, None, :] * rsize[None, :, :])
    gap = query["original_frame_index"] - reference["original_frame_index"]
    check(gap > 0 and np.all(qsize > 0) and np.all(rsize > 0), "Invalid original geometry/time")
    values = np.full((nq, nr, 9), np.nan)
    masks = np.zeros((nq, nr, 9), bool)
    values[:, :, :2] = (qcenter[:, None, :] - rcenter[None, :, :]) / scale
    values[:, :, 2:4] = np.log(qsize[:, None, :] / rsize[None, :, :])
    values[:, :, 4] = np.log1p(gap)
    masks[:, :, :5] = True
    for row, past_time in enumerate(reference["previous_times"]):
        if not np.isfinite(past_time):
            continue
        past_box = np.asarray(reference["previous_boxes"][row], float)
        past_center = (past_box[:2] + past_box[2:]) / 2
        history_gap = reference["original_frame_index"] - past_time
        check(history_gap > 0 and np.isfinite(past_box).all(), "Invalid independent past locator")
        velocity = (rcenter[row] - past_center) / history_gap
        predicted_center = rcenter[row] + velocity * gap
        predicted_box = np.r_[predicted_center-rsize[row]/2, predicted_center+rsize[row]/2]
        values[:, row, 5:7] = (qcenter - predicted_center) / scale[:, row]
        intersection = np.prod(np.maximum(np.minimum(qb[:, 2:], predicted_box[2:]) - np.maximum(qb[:, :2], predicted_box[:2]), 0), axis=1)
        values[:, row, 7] = intersection / (np.prod(qsize, axis=1) + np.prod(rsize[row]) - intersection)
        values[:, row, 8] = np.log1p(history_gap)
        masks[:, row, 5:] = True
    check(np.array_equal(np.isfinite(values), masks), "Independent geometry value/mask mismatch")
    return values, masks


def independent_donors(categories, history_valid, immutable_dump_rows=None):
    rows = np.arange(len(categories)) if immutable_dump_rows is None else np.asarray(immutable_dump_rows)
    groups = defaultdict(list)
    for index, category in enumerate(categories):
        groups[int(category), bool(history_valid[index])].append(index)
    donors = np.arange(len(categories))
    for group in groups.values():
        ordered = sorted(group, key=lambda index: int(rows[index]))
        for offset, index in enumerate(ordered):
            donors[index] = ordered[(offset-1) % len(ordered)]
    return donors


def independent_inputs(cosine, values, masks, mean, std, mode, donors):
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    check(mean.shape == std.shape == (9,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std >= 1e-6), "Invalid independent train normalizer")
    transformed = np.zeros_like(values)
    transformed[masks] = np.broadcast_to((values-mean)/std, values.shape)[masks]
    valid = masks.copy()
    if mode == "G0":
        transformed[:] = 0
        valid[:] = False
    elif mode == "Gmask":
        transformed[:, :, POSITION_SLOTS] = 0
    elif mode == "Gperm":
        original = transformed.copy()
        transformed[:, :, POSITION_SLOTS] = original[:, donors, :][:, :, POSITION_SLOTS]
    else:
        check(mode == "G1", "Unknown independent intervention")
    return np.concatenate((cosine[:, :, None], transformed, valid.astype(float)), axis=2)


def independent_model_scores(query, reference, normalizer, adapter_state, heads):
    rawcos = unit(query["raw"]) @ unit(reference["raw"]).T
    adapted = adapter_forward(query["raw"], adapter_state) @ adapter_forward(reference["raw"], adapter_state).T
    values, masks = independent_geometry(query, reference)
    donors = independent_donors(reference["predicted_categories"], np.isfinite(reference["previous_times"]))
    result = [rawcos, adapted]
    for appearance, cosine in (("A0", rawcos), ("A1", adapted)):
        for mode in ("G0", "G1", "Gmask", "Gperm"):
            inputs = independent_inputs(cosine, values, masks, normalizer["mean"], normalizer["std"], mode, donors)
            result.append(head_forward(inputs, heads[appearance+mode]))
    return np.stack(result, axis=2)


def math_self_test():
    """Nonzero synthetic weights test numerical math; no real model is read."""
    path = ROOT / "tmp_analysis/identity_spacetime_modules_20261024.py"
    check(sha(path) == "2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3", "Frozen pure module changed")
    spec = importlib.util.spec_from_file_location("only_synthetic_oof_math_subject", path)
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1774)
    adapter = subject.IdentityAdapter256(20261024)
    heads = {appearance+mode: subject.ResidualAssociationHead19(20261024)
             for appearance in ("A0", "A1") for mode in ("G0", "G1", "Gmask", "Gperm")}
    for model in [adapter] + list(heads.values()):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, dtype=torch.float64, generator=generator)*.1)
    subject.freeze_adapter(adapter)
    rawq = torch.randn((3, 256), dtype=torch.float64, generator=generator)
    rawr = torch.randn((4, 256), dtype=torch.float64, generator=generator)
    rawq[2] = 0
    rawr[3] = 0
    query = dict(raw=rawq.numpy(), boxes=np.array([[4.,0.,6.,2.], [1.,1.,5.,5.], [10.,10.,20.,20.]]), original_frame_index=20)
    reference = dict(raw=rawr.numpy(), boxes=np.array([[0.,0.,2.,2.], [1.,1.,5.,5.], [9.,9.,19.,19.], [15.,15.,25.,25.]]),
        original_frame_index=10, previous_times=np.array([0.,np.nan,5.,np.nan]),
        previous_boxes=np.array([[-2.,0.,0.,2.], [np.nan]*4, [8.,8.,18.,18.], [np.nan]*4]),
        predicted_categories=np.array([1,1,1,2]))
    values, masks = independent_geometry(query, reference)
    normalizer = subject.GeometryNormalizer(np.arange(9)/10, np.ones(9)*.7, np.ones(9))
    adapter_state = {name: tensor.detach().numpy().tolist() for name,tensor in adapter.state_dict().items()}
    head_states = {name: {key: tensor.detach().numpy().tolist() for key,tensor in head.state_dict().items()} for name,head in heads.items()}
    reconstructed = independent_model_scores(query, reference, dict(mean=normalizer.mean, std=normalizer.std), adapter_state, head_states)
    pure = [subject.frozen_appearance_cosine(rawq,rawr), subject.frozen_appearance_cosine(rawq,rawr,adapter)]
    plan = subject.gperm_donors(reference["predicted_categories"], np.isfinite(reference["previous_times"]), np.arange(4))
    expected = [value.numpy() for value in pure]
    for appearance, cosine in zip(("A0", "A1"), pure):
        for mode in ("G0", "G1", "Gmask", "Gperm"):
            inputs,_ = subject.head_inputs(cosine, values, masks, normalizer, mode, plan if mode=="Gperm" else None)
            expected.append(heads[appearance+mode](inputs).detach().numpy())
    expected = np.stack(expected,axis=2)
    np.testing.assert_allclose(reconstructed, expected, atol=SCORE_ATOL, rtol=SCORE_RTOL)
    # Compare the independently derived geometry against reviewed module math.
    geometry_path=ROOT/"tmp_analysis/relative_track_geometry.py"
    check(sha(geometry_path)=="c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a","Reviewed geometry changed")
    geometry_spec=importlib.util.spec_from_file_location("only_synthetic_reviewed_oof_geometry",geometry_path)
    geometry=importlib.util.module_from_spec(geometry_spec);geometry_spec.loader.exec_module(geometry)
    reviewed=geometry.pair_geometry(query["boxes"],reference["boxes"],query["original_frame_index"],np.full(4,10.),reference["previous_boxes"],reference["previous_times"])
    gv,gm=subject.geometry_slots(reviewed,np.full(4,10.),reference["previous_times"])
    np.testing.assert_allclose(values,gv,atol=1e-13,rtol=1e-13,equal_nan=True)
    np.testing.assert_array_equal(masks,gm)
    return dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_NUMERICAL_PRIMITIVES_SYNTHETIC_ONLY",
        verifier_sha256=sha(__file__),score_columns=list(SCORE_COLUMNS),score_atol=SCORE_ATOL,score_rtol=SCORE_RTOL,
        nonzero_synthetic_checkpoint_weights=True,maximum_absolute_score_error=float(np.max(np.abs(reconstructed-expected))),
        independent_linear_GELU_erf_adapter_zero_mask_head_interventions_geometry=True,
        real_cache_or_labels_or_checkpoint_opened=False,complete_runner_or_output_audit=False)

# Complete-output audit helpers. These never import the trainer, evaluator,
# coordinator, data module, or their numerical forward functions.
from collections import Counter, deque
import copy
import sqlite3
import tempfile
import time
import resource

FEATURES=("raw_appearance_cosine","adapter_appearance_cosine","A0G0","A0G1","A0Gmask","A0Gperm","A1G0","A1G1","A1Gmask","A1Gperm")
SCOPES=("all_candidates","same_predicted_category")
POPS=("all_known_population","common_CV_covered_population")
SUBSETS=("all_nonempty_known_queries","positive_and_negative_available_queries")
COMPARISONS=(("adapter_cosine_minus_raw",0,1),("A0G1_minus_A0G0",2,3),("A1G1_minus_A1G0",6,7),("A0G1_minus_A0Gmask",4,3),("A0G1_minus_A0Gperm",5,3),("A1G1_minus_A1Gmask",8,7),("A1G1_minus_A1Gperm",9,7))
PRIMARY=SCOPES[1],POPS[1]
BASE_AUDITOR_SHA="74a5c59c957439411acdb7139822c8d83c583584ab07e9e21957e731e648d3ce"
HISTORY_AUDITOR_SHA="c12bcd2ad7024a559cf0fa5c22ea7af9d3f6d2a18df8641cd379820fd139babd"
SCANNER_SHA="862d31f63331a288804d808fb303e6704d6f4fdfa80f97daf13971f44c0f795d"


def load(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def compare(actual,expected,where="root"):
    if isinstance(expected,dict):
        check(isinstance(actual,dict) and set(actual)==set(expected),"Keys differ: "+where)
        for key,value in expected.items():compare(actual[key],value,where+"/"+str(key))
    elif isinstance(expected,(list,tuple)):
        check(isinstance(actual,(list,tuple)) and len(actual)==len(expected),"List differs: "+where)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,where+"/"+str(i))
    elif isinstance(expected,(float,np.floating)):
        check(isinstance(actual,(float,int,np.number)) and np.isfinite(actual) and np.isclose(actual,expected,atol=1e-12,rtol=1e-12),"Float differs: "+where)
    else:check(actual==expected,"Value differs: "+where)


def independent_folds(roles):
    strata=defaultdict(list)
    for source,row in roles.items():
        if row["split"]=="fit":strata[row["block"],row["role"]].append(source)
    groups={};orders={}
    for (block,role),sources in sorted(strata.items()):
        ordered=sorted(sources,key=lambda s:(hashlib.sha256(b"identity_spacetime_source_oof_v1\0"+s.encode()).hexdigest(),s))
        orders[block+"/"+role]=ordered
        for i,source in enumerate(ordered):groups[source]=dict(fold=i%5,role=role,dataset_block=block,stratum=[block,role])
    sources=sorted(groups)
    return dict(seed="identity_spacetime_source_oof_v1",number_of_folds=5,groups={s:groups[s] for s in sources},
        folds={str(f):dict(held_out_sources=[s for s in sources if groups[s]["fold"]==f],train_sources=[s for s in sources if groups[s]["fold"]!=f]) for f in range(5)},
        strata_order=orders,source_count=len(sources),roles_are_metadata_only=True,labels_used=False)


def metadata(spec):
    roles=load(spec["source_roles_path"]);roles=roles.get("groups",roles)
    roles={s:r for s,r in roles.items() if r["split"]=="fit"}
    manifest=independent_folds(roles);compare(load(spec["fold_manifest_path"]),manifest,"Independent source folds")
    videos={str(r["video_id"]):r["source_group"] for r in load(spec["collection_plan_path"])["video_records"] if r["split"]=="fit"}
    inventory=load(spec["dump_inventory_path"]);grouped=defaultdict(list)
    for line in Path(spec["frame_execution_path"]).open():
        row=json.loads(line)
        if str(row["video_id"]) not in videos:continue
        check(videos[str(row["video_id"])] in roles,"Cached source outside original roles")
        check(row["npz_sha256"]==inventory[Path(row["npz"]).name],"Inventory differs")
        grouped[int(row["video_id"])].append(row)
    counts=Counter();frames=[]
    for video,records in sorted(grouped.items()):
        records.sort(key=lambda r:r["frame_id"]);check([r["frame_id"] for r in records]==list(range(len(records))),"Nonexact sampled timeline")
        frames.extend(records)
        for lag in (1,4):counts[str(lag)]+=sum(records[i]["observations"]*records[i-lag]["observations"] for i in range(lag,len(records)))
    population=dict(sealed_fit_source_groups=len(roles),allocated_fit_videos=len(videos),selected_fit_videos=len(grouped),selected_frames=len(frames),retained_observations=sum(r["observations"] for r in frames))
    for key,value in population.items():check(spec["population"][key]==value,"Independent metadata population: "+key)
    compare(dict(counts),spec["resource_bounds"]["candidates_by_lag"],"Independent QxM totals")
    return roles,manifest,videos,frames,grouped,population


def all_pins(spec):
    pins=dict(spec["input_pins"])
    for key in ("modules","data_module","trainer","coordinator","evaluator","verifier","fold_manifest","finalizer"):
        pins[spec[key+"_path"]]=spec[key+"_sha256"]
    for key in ("manual","index"):pins[spec["identity_authority"][key+"_path"]]=spec["identity_authority"][key+"_sha256"]
    return pins


def pinned_bytes(pins):
    for path,digest in pins.items():check(Path(path).is_file() and sha(path)==digest,"Pinned bytes differ: "+str(path))


def protected_snapshot(authority):
    return {p:dict(exists=Path(p).exists(),sha256=sha(p) if Path(p).exists() else None) for p in authority}


def global_seal_gate(out,spec,manifest,videos,grouped):
    """No identity scanner or identity-valued JSON decoding before this returns."""
    seal=load(out/"GT_free_OOF_predictions_sealed.json")
    check(seal["status"]=="SEALED_ALL_15_SOURCE_OOF_MODELS_BEFORE_UNIFIED_EVALUATION" and seal["model_count"]==15,"Incomplete global source-OOF seal")
    compare([seal[k] for k in ("folds","seeds","columns")],[spec["folds"],spec["seeds"],list(FEATURES)],"Global dimensions")
    compare(seal["input_sha256"],all_pins(spec),"Global authority pins");pinned_bytes(seal["input_sha256"])
    names={"fold%d_seed%d"%(f,s) for f in range(5) for s in spec["seeds"]}
    check({p.name for p in (out/"models").iterdir()}==names and set(seal["models"])==names,"Exact fifteen model set missing")
    for seed in spec["seeds"]:
        totals=Counter();receipts=0
        for fold in range(5):
            name="fold%d_seed%d"%(fold,seed);directory=out/"models"/name;entry=seal["models"][name]
            held=set(manifest["folds"][str(fold)]["held_out_sources"])
            selected={v:r for v,r in grouped.items() if videos[str(v)] in held}
            shapes={"shards/v%d_lag%d.scores.npy"%(v,l):(sum(rows[i]["observations"]*rows[i-l]["observations"] for i in range(l,len(rows))),10) for v,rows in selected.items() for l in (1,4)}
            files=set(shapes)|{"shards/v%d.rows.json"%v for v in selected}|{"checkpoint.json","normalizer.json","training_trace.json","frame_pairs.jsonl"}
            expected=files|{"GT_free_predictions_sealed.json","single_model_receipt.json"}
            check(set(entry["files_sha256"])==expected,"Global sealed member coverage differs")
            check({str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}==expected,"Unexpected/missing model file")
            for relative,digest in entry["files_sha256"].items():
                path=directory/relative
                check(not Path(relative).is_absolute() and ".." not in Path(relative).parts and not any(p.is_symlink() for p in (path,*path.parents)),"Unsafe sealed member")
                check(sha(path)==digest,"Sealed model bytes changed: "+str(path))
            single=load(directory/"GT_free_predictions_sealed.json");receipt=load(directory/"single_model_receipt.json");meta=single["metadata"]
            check(single["status"]=="SEALED_SINGLE_MODEL_FULL_HELDOUT_BEFORE_HELDOUT_LABELS" and single["all_retained_candidates_scored"] is True and single["unknown_candidates_filtered"] is False,"Population filtered")
            check(single["held_out_identity_values_decoded"] is False and receipt["held_out_identity_values_decoded"] is False and receipt["training_sources_only"] is True and receipt["GPU_used"] is False,"Model isolation claims differ")
            check(receipt["status"]=="COMPLETE_SINGLE_FOLD_SEED_TRAINING_AND_FULL_HELDOUT_PREDICTION" and receipt["script_sha256"]==spec["trainer_sha256"] and receipt["fold"]==fold and receipt["seed"]==seed,"Worker receipt differs")
            check(receipt["global_model_prediction_seal_sha256"]==sha(directory/"GT_free_predictions_sealed.json"),"Single seal hash differs")
            compare(single["files_sha256"],receipt["generated"]["files_sha256"],"Receipt file hashes");check(set(single["files_sha256"])==files,"Single seal missing files")
            for relative,digest in single["files_sha256"].items():check(entry["files_sha256"][relative]==digest,"Single/global seal differs")
            for key,value in dict(fold=fold,seed=seed,train_sources=manifest["folds"][str(fold)]["train_sources"],held_out_sources=sorted(held),columns=list(FEATURES),schedule=spec["numeric_training_config"],final_adapter_epoch=50,final_head_epoch=20).items():compare(meta[key],value,"Model metadata "+key)
            for filename in ("checkpoint.json","normalizer.json","training_trace.json"):compare(load(directory/filename)["metadata"],meta,"Shared metadata")
            for relative,shape in shapes.items():
                values=np.load(directory/relative,mmap_mode="r",allow_pickle=False);check(values.shape==shape and values.dtype==np.float64,"Sealed shape/columns/dtype differs");del values
            nframes=sum(len(r) for r in selected.values());check(single["frame_pair_receipts"]==entry["frame_pair_receipts"]==2*nframes,"Missing frame-pair receipts")
            with (directory/"frame_pairs.jsonl").open() as pair_stream:
                for video,records in sorted(selected.items()):
                    rowmap=load(directory/"shards"/("v%d.rows.json"%video))
                    check(rowmap["video_id"]==video and len(rowmap["frames"])==len(records),"GTfree row map coverage incomplete before labels")
                    offsets=Counter()
                    for step,record in enumerate(records):
                        frame=rowmap["frames"][step]
                        check(frame["image_id"]==record["image_id"] and frame["sampled_step"]==step and frame["original_frame_index"]==record["frame_index"] and len(frame["boxes"])==record["observations"],"GTfree row map locator incomplete before labels")
                        for lag in (1,4):
                            line=next(pair_stream,None);check(line is not None,"GTfree frame-pair receipt missing before labels");pair=json.loads(line)
                            reference=None if step<lag else records[step-lag];nm=0 if reference is None else reference["observations"];count=record["observations"]*nm
                            expected_pair=dict(source=videos[str(video)],video_id=video,lag=lag,query_step=step,reference_step=None if reference is None else step-lag,query_image_id=record["image_id"],reference_image_id=None if reference is None else reference["image_id"],query_rows=record["observations"],reference_rows=nm,candidates=count,offset=offsets[lag],score_file="shards/v%d_lag%d.scores.npy"%(video,lag),original_frame_gap=None if reference is None else record["frame_index"]-reference["frame_index"])
                            for key,value in expected_pair.items():compare(pair[key],value,"Complete GTfree receipt before labels")
                            offsets[lag]+=count
                    for lag in (1,4):check(offsets[lag]==shapes["shards/v%d_lag%d.scores.npy"%(video,lag)][0],"GTfree shard offsets incomplete before labels")
                check(next(pair_stream,None) is None,"Extra GTfree receipt before labels")
            check(single["parent_raw_scores_exactly_verified"] is True,"Raw parent parity missing")
            compare(entry["candidates_by_lag"],single["candidates_by_lag"],"Global candidate receipt")
            totals.update(single["candidates_by_lag"]);receipts+=entry["frame_pair_receipts"]
        compare(dict(totals),spec["resource_bounds"]["candidates_by_lag"],"Global candidate totals")
        check(receipts==2*spec["population"]["selected_frames"],"Global receipt totals differ")
    return seal


def independent_rowmap(frame,utility):
    row=utility.independent_row_metadata(frame)
    # The new deterministic JSON writer represents a missing four-vector as
    # four null slots. Derive that representation from independently read NaNs.
    row["previous_boxes"]=[[None]*4 if box is None else box for box in row["previous_boxes"]]
    return row


def audited_utilities():
    for filename,digest in (("verify_fit_lag_identity_support_output_20261024.py",BASE_AUDITOR_SHA),("verify_angular_appearance_control_output_20261024.py",HISTORY_AUDITOR_SHA),("verify_fit_identity_edge_support_output.py",SCANNER_SHA)):
        check(sha(ROOT/"tmp_analysis"/filename)==digest,"Previous independent audit utility changed")
    import verify_angular_appearance_control_output_20261024 as independent
    return independent


def independent_anchors(spec,videos,allowed,project):
    """Split, source, eligibility, then exact allowlisted identity fields."""
    authority=spec["identity_authority"];allowed=set(allowed);identities={};counts=Counter();source_counts=Counter()
    fields={"split","observation_id","source_group","video_id","image_id","sampled_step","original_frame_index","local_track_id","gt_track_id","manual_anchor_id","eligible_for_parameter_fit","pred_category_id"}
    index_fields=fields-{"gt_track_id","manual_anchor_id","eligible_for_parameter_fit"}|{"dump_row"}
    for line in Path(authority["manual_path"]).open():
        split=project(line,{"split"}).get("split");check(split in ("fit","tune"),"Identity split differs")
        if split!="fit":counts["skipped_nonfit_rows"]+=1;continue
        source=project(line,{"source_group"}).get("source_group")
        if source not in allowed:counts["skipped_unallowed_source_rows"]+=1;continue
        eligible=project(line,{"eligible_for_parameter_fit"}).get("eligible_for_parameter_fit");check(isinstance(eligible,bool),"Eligibility type differs")
        if not eligible:counts["skipped_ineligible_train_rows"]+=1;continue
        row=project(line,fields);check(set(row)==fields,"Identity projection incomplete")
        for key in fields-{"split","source_group","eligible_for_parameter_fit"}:check(isinstance(row[key],int) and not isinstance(row[key],bool) and row[key]>=0,"Identity integer differs")
        check(row["pred_category_id"]>0 and videos[str(row["video_id"])]==source and row["observation_id"] not in identities,"Identity authority/source duplicate")
        identities[row["observation_id"]]=row;source_counts[source]+=1
    anchors={};joined=set()
    for line in Path(authority["index_path"]).open():
        if project(line,{"split"}).get("split")!="fit":continue
        if project(line,{"source_group"}).get("source_group") not in allowed:continue
        row=project(line,index_fields);identity=identities.get(row["observation_id"])
        if identity is None:continue
        for key in index_fields-{"dump_row"}:check(row[key]==identity[key],"Immutable ID locator differs")
        key=row["image_id"],row["dump_row"];check(key not in anchors and row["observation_id"] not in joined,"Duplicate immutable ID")
        anchors[key]=(row["video_id"],identity["gt_track_id"]);joined.add(row["observation_id"])
    check(joined==set(identities),"Unjoined identity")
    receipt=dict(allowed_sources=sorted(allowed),eligible_train_anchors=len(anchors),eligible_identity_counts_by_source={s:source_counts[s] for s in sorted(allowed)},
        train_label_projection_sha256=canonical([identities[k] for k in sorted(identities)]),immutable_anchor_sha256=canonical([dict(image_id=k[0],dump_row=k[1],video_id=v[0],gt_track_id=v[1]) for k,v in sorted(anchors.items())]),counts=dict(counts))
    return anchors,receipt


def open_db(path):
    db=sqlite3.connect(path);db.execute("PRAGMA temp_store=FILE");db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE query_support(seed INTEGER,source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,image INTEGER,qrow INTEGER,qgt INTEGER,p INTEGER,n INTEGER,raw_correct INTEGER,correct_bits INTEGER,gperm_changed INTEGER,PRIMARY KEY(seed,lag,scope,population,video,image,qrow))")
    db.execute("CREATE INDEX query_source ON query_support(seed,lag,scope,population,source,video,qgt)")
    return db


def fresh():
    result={key:0 for key in ("candidate_pairs","known_pairs","positive_pairs","negative_pairs","unknown_pairs","nonempty_known_queries","positive_available_queries","negative_available_queries","hard_queries","gperm_mapping_nonidentity_candidates","gperm_geometry_changed_candidates")}
    result["top1"]={subset:{f:{k:0 for k in ("raw_wrong_feature_wrong","raw_wrong_feature_correct","raw_correct_feature_wrong","raw_correct_feature_correct")} for f in FEATURES} for subset in SUBSETS}
    result["orders"]={f:np.zeros((3,3),np.int64) for f in FEATURES}
    return result


def add_block(cell,score,mask,qi,ri,changed,moved,key,db):
    seed,source,lag,scope,pop,video,image=key
    known=mask&(qi[:,None]>=0)&(ri[None,:]>=0);positive=qi[:,None]==ri[None,:]
    for name,value in (("candidate_pairs",mask.sum()),("known_pairs",known.sum()),("positive_pairs",(known&positive).sum()),("negative_pairs",(known&~positive).sum()),("unknown_pairs",(mask&~known).sum()),("gperm_mapping_nonidentity_candidates",(mask&moved).sum()),("gperm_geometry_changed_candidates",(mask&changed).sum())):cell[name]+=int(value)
    for qr in range(len(qi)):
        rows=np.flatnonzero(known[qr])
        if not len(rows):continue
        truth=positive[qr,rows];p=int(sum(truth));n=len(rows)-p;values=score[qr,rows,:]
        flags=[bool(truth[np.argmax(values[:,c])]) for c in range(10)] # ascending rows provide fixed lowest-row ties
        bits=sum(int(x)<<c for c,x in enumerate(flags));actual_change=int(np.any(mask[qr]&changed[qr]))
        db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(seed,source,lag,scope,pop,video,image,qr,int(qi[qr]),p,n,int(flags[0]),bits,actual_change))
        for name,value in (("nonempty_known_queries",1),("positive_available_queries",p>0),("negative_available_queries",n>0),("hard_queries",p>0 and n>0)):cell[name]+=int(value)
        for subset in SUBSETS if p and n else SUBSETS[:1]:
            for feature,flag in zip(FEATURES,flags):cell["top1"][subset][feature]["raw_"+("correct" if flags[0] else "wrong")+"_feature_"+("correct" if flag else "wrong")]+=1
        if p and n:
            for positive_index in np.flatnonzero(truth):
                difference=values[positive_index][None,:]-values[~truth]
                ranks=np.sign(difference).astype(np.int64)+1
                for c,f in enumerate(FEATURES):cell["orders"][f]+=np.bincount(3*ranks[:,0]+ranks[:,c],minlength=9).reshape(3,3)


def combine(values):
    result=fresh()
    for value in values:
        for key in result:
            if key=="top1":
                for subset in SUBSETS:
                    for f in FEATURES:
                        for name,count in value[key][subset][f].items():result[key][subset][f][name]+=count
            elif key=="orders":
                for f in FEATURES:result[key][f]+=value[key][f]
            else:result[key]+=value[key]
    return result


def filtered_rows(db,seed,lag,scope,pop,sources=None,hard=False,changed=False):
    clause="seed=? AND lag=? AND scope=? AND population=?";args=[seed,lag,scope,pop]
    if hard:clause+=" AND p>0 AND n>0"
    if changed:clause+=" AND gperm_changed=1"
    if sources is not None:
        if not sources:clause+=" AND 0"
        else:clause+=" AND source IN ("+",".join("?" for _ in sources)+")";args+=list(sources)
    return list(db.execute("SELECT source,video,qgt,correct_bits FROM query_support WHERE "+clause+" ORDER BY source,video,image,qrow",args))


def statistics(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for source,video,gt,bits in rows:groups[source][video,gt].append(np.array([(bits>>c)&1 for c in range(10)],np.int64))
    n=len(rows);trajectories=[np.asarray(v) for group in groups.values() for v in group.values()]
    total=np.sum([x.sum(axis=0) for x in trajectories],axis=0) if trajectories else np.zeros(10,np.int64)
    result=dict(queries=n,sources=len(groups),query_GT_trajectories=len(trajectories),features={})
    for c,f in enumerate(FEATURES):
        correct=int(total[c]);errors=n-correct;source_error=[sum(len(v)-int(np.sum(np.asarray(v)[:,c])) for v in group.values()) for group in groups.values()]
        traj_error=[len(v)-int(v[:,c].sum()) for v in trajectories]
        macro=[np.mean([np.mean(np.asarray(v)[:,c]) for v in group.values()]) for group in groups.values()]
        source_query=[sum(int(np.sum(np.asarray(v)[:,c])) for v in group.values())/sum(map(len,group.values())) for group in groups.values()]
        result["features"][f]=dict(correct=correct,errors=errors,micro_correctness=correct/n if n else None,source_GTtrajectory_query_macro_correctness=float(np.mean(macro)) if macro else None,
            source_query_macro_correctness=float(np.mean(source_query)) if source_query else None,GTtrajectory_macro_correctness=float(np.mean([v[:,c].mean() for v in trajectories])) if trajectories else None,
            error_sources=sum(e>0 for e in source_error),error_GT_trajectories=sum(e>0 for e in traj_error),largest_source_error_share=max(source_error,default=0)/errors if errors else None,largest_GTtrajectory_error_share=max(traj_error,default=0)/errors if errors else None)
    return result


def paired(rows,stats):
    result={};n=len(rows)
    for name,a,b in COMPARISONS:
        transitions=Counter(((bits>>a)&1,(bits>>b)&1) for _,_,_,bits in rows);fa,fb=stats["features"][FEATURES[a]],stats["features"][FEATURES[b]]
        result[name]=dict(queries=n,baseline_wrong_alternative_correct=transitions[0,1],baseline_correct_alternative_wrong=transitions[1,0],both_correct=transitions[1,1],both_wrong=transitions[0,0],net_correct=fb["correct"]-fa["correct"],micro_correctness_delta=fb["micro_correctness"]-fa["micro_correctness"] if n else None,source_GTtrajectory_query_macro_delta=fb["source_GTtrajectory_query_macro_correctness"]-fa["source_GTtrajectory_query_macro_correctness"] if n else None)
    hist=Counter();group=defaultdict(lambda:defaultdict(list))
    for source,video,gt,bits in rows:
        value=((bits>>7)&1)-((bits>>6)&1)-((bits>>3)&1)+((bits>>2)&1);hist[value]+=1;group[source][video,gt].append(value)
    means=[np.mean([np.mean(v) for v in trajectories.values()]) for trajectories in group.values()]
    result["appearance_geometry_2x2_interaction"]=dict(queries=n,query_value_histogram={str(i):hist[i] for i in range(-2,3)},micro_mean=sum(k*v for k,v in hist.items())/n if n else None,source_GTtrajectory_query_macro_mean=float(np.mean(means)) if means else None,formula="(A1G1-A1G0)-(A0G1-A0G0) in paired query correctness")
    return result


def decorated(db,cell,seed,lag,scope,pop,sources):
    result={k:v for k,v in cell.items() if k not in ("top1","orders")};result["top1"]=copy.deepcopy(cell["top1"])
    for subset in SUBSETS:
        for f in FEATURES:
            value=result["top1"][subset][f]
            value.update(repair_raw_error=value["raw_wrong_feature_correct"],break_raw_correct=value["raw_correct_feature_wrong"],raw_correct=value["raw_correct_feature_wrong"]+value["raw_correct_feature_correct"],feature_correct=value["raw_wrong_feature_correct"]+value["raw_correct_feature_correct"])
    result["positive_negative_pair_order"]={f:dict(comparisons=int(cell["orders"][f].sum()),raw_to_feature_loss_tie_win_matrix=cell["orders"][f].tolist()) for f in FEATURES}
    result["query_statistics"]={};result["comparisons"]={};result["Gperm_actual_changed_GTfree_queries"]={}
    for subset,hard in zip(SUBSETS,(False,True)):
        rows=filtered_rows(db,seed,lag,scope,pop,sources,hard);stats=statistics(rows)
        result["query_statistics"][subset]=stats;result["comparisons"][subset]=paired(rows,stats)
        changed=filtered_rows(db,seed,lag,scope,pop,sources,hard,True);stats=statistics(changed)
        result["Gperm_actual_changed_GTfree_queries"][subset]=dict(query_statistics=stats,comparisons=paired(changed,stats))
    return result


def shared_lags(db,seed,scope,pop,sources):
    a=filtered_rows_with_key(db,seed,1,scope,pop,sources);b=filtered_rows_with_key(db,seed,4,scope,pop,sources)
    keys=sorted(a.keys()&b.keys());result=dict(shared_hard_queries=len(keys),sources=len({a[k][0] for k in keys}),features={})
    for c,f in enumerate(FEATURES):
        counts=Counter();group=defaultdict(lambda:defaultdict(list))
        for key in keys:
            source,video,gt,bits1=a[key];bits4=b[key][3];x,y=(bits1>>c)&1,(bits4>>c)&1
            counts[x,y]+=1;group[source][video,gt].append((x,y))
        macros=[np.mean([np.mean(v,axis=0) for v in trajectories.values()],axis=0) for trajectories in group.values()]
        macro=np.mean(macros,axis=0).tolist() if macros else [None,None]
        result["features"][f]=dict(lag1_wrong_lag4_wrong=counts[0,0],lag1_wrong_lag4_correct=counts[0,1],lag1_correct_lag4_wrong=counts[1,0],lag1_correct_lag4_correct=counts[1,1],error_lag1=counts[0,0]+counts[0,1],error_lag4=counts[0,0]+counts[1,0],source_GTtrajectory_query_macro_lag1=macro[0],source_GTtrajectory_query_macro_lag4=macro[1])
    return result


def filtered_rows_with_key(db,seed,lag,scope,pop,sources):
    clause="seed=? AND lag=? AND scope=? AND population=? AND p>0 AND n>0";args=[seed,lag,scope,pop]
    if sources is not None:
        if not sources:clause+=" AND 0"
        else:clause+=" AND source IN ("+",".join("?" for _ in sources)+")";args+=list(sources)
    return {(v,image,qr):(source,v,gt,bits) for source,v,image,qr,gt,bits in db.execute("SELECT source,video,image,qrow,qgt,correct_bits FROM query_support WHERE "+clause,args)}


def reports(db,cells,roles,seeds):
    sources=sorted(roles);perseed={}
    for seed in seeds:
        def population(selected):return {str(lag):{scope:{pop:decorated(db,combine(cells[seed,s,lag,scope,pop] for s in selected),seed,lag,scope,pop,selected) for pop in POPS} for scope in SCOPES} for lag in (1,4)}
        source_reports={s:dict(role=roles[s]["role"],dataset_block=roles[s]["block"],lags=population([s]),cross_lag={scope:{pop:shared_lags(db,seed,scope,pop,[s]) for pop in POPS} for scope in SCOPES}) for s in sources}
        deletion={}
        for lag in (1,4):
            deletion[str(lag)]={}
            for deleted in sources:
                stats=[source_reports[s]["lags"][str(lag)][PRIMARY[0]][PRIMARY[1]]["query_statistics"][SUBSETS[1]] for s in sources if s!=deleted]
                valid=[r for r in stats if r["queries"]];n=sum(r["queries"] for r in valid)
                values={f:(sum(r["features"][f]["correct"] for r in valid)/n if n else None,float(np.mean([r["features"][f]["source_GTtrajectory_query_macro_correctness"] for r in valid])) if valid else None) for f in FEATURES}
                comparisons={name:dict(micro_delta=values[FEATURES[b]][0]-values[FEATURES[a]][0] if n else None,source_GTtrajectory_query_macro_delta=values[FEATURES[b]][1]-values[FEATURES[a]][1] if n else None) for name,a,b in COMPARISONS}
                comparisons["appearance_geometry_2x2_interaction"]={key:(values["A1G1"][i]-values["A1G0"][i]-values["A0G1"][i]+values["A0G0"][i] if n else None) for i,key in enumerate(("micro_delta","source_GTtrajectory_query_macro_delta"))}
                deletion[str(lag)][deleted]=dict(remaining_queries=n,comparisons=comparisons)
        perseed[str(seed)]=dict(micro_union=population(sources),per_source=source_reports,
            per_role={r:population([s for s in sources if roles[s]["role"]==r]) for r in sorted({r["role"] for r in roles.values()})},
            per_dataset_block={r:population([s for s in sources if roles[s]["block"]==r]) for r in sorted({r["block"] for r in roles.values()})},
            cross_lag={scope:{pop:shared_lags(db,seed,scope,pop,None) for pop in POPS} for scope in SCOPES},source_deletion_sensitivity=deletion)
    robustness={}
    for lag in (1,4):
        robustness[str(lag)]={}
        for name,_,_ in COMPARISONS:
            values=[perseed[str(s)]["micro_union"][str(lag)][PRIMARY[0]][PRIMARY[1]]["comparisons"][SUBSETS[1]][name]["source_GTtrajectory_query_macro_delta"] for s in seeds]
            deletions=[row["comparisons"][name]["source_GTtrajectory_query_macro_delta"] for s in seeds for row in perseed[str(s)]["source_deletion_sensitivity"][str(lag)].values()]
            valid=[v for v in values if v is not None];valid_deletions=[v for v in deletions if v is not None]
            robustness[str(lag)][name]=dict(seeds=list(seeds),source_macro_deltas=values,seed_mean_descriptive_only=float(np.mean(valid)) if valid else None,all_fixed_seeds_strictly_positive=len(valid)==len(seeds) and bool(valid) and all(v>0 for v in valid),minimum_source_deletion_macro_delta=min(valid_deletions) if valid_deletions else None,positive_direction_survives_all_source_deletions=bool(valid_deletions) and all(v>0 for v in valid_deletions))
    return dict(per_seed=perseed,macro_definition="Equal supported source -> equal official query GTtrajectory -> equal query observation; zero-support sources remain in reports but contribute no numerical macro.",Gperm_changed_definition="Any actually changed geometric value in the complete retained GT-free scope/population for this query; defined before known-label projection, never by GT or gain.",seed_robustness=robustness)


def independent_train_design(spec,manifest,videos,frames,grouped,anchors,utility):
    """Rebuild legal hard query-lags, hierarchy and original geometry moments."""
    examples=[]
    for video,records in sorted(grouped.items()):
        history={};ring=deque(maxlen=5);source=videos[str(video)]
        for record in records:
            q=utility.read_raw(record,history);ring.append(q)
            qi=np.array([anchors.get((q["image_id"],r),(video,-1))[1] for r in range(len(q["raw"]))],np.int64)
            for lag in (1,4):
                if len(ring)<=lag:continue
                r=ring[-lag-1];ri=np.array([anchors.get((r["image_id"],j),(video,-1))[1] for j in range(len(r["raw"]))],np.int64)
                values,masks=independent_geometry(q,r)
                for qr,gt in enumerate(qi):
                    if gt<0:continue
                    known=(ri>=0)&(r["predicted_categories"]==q["predicted_categories"][qr]);p=np.flatnonzero(known&(ri==gt));n=np.flatnonzero(known&(ri!=gt))
                    if not len(p) or not len(n):continue
                    selected=np.sort(np.r_[p,n]);ispositive=ri[selected]==gt
                    examples.append(dict(source=source,video=video,gt=int(gt),image=q["image_id"],row=qr,lag=lag,values=values[qr,selected].copy(),masks=masks[qr,selected].copy(),positive=ispositive))
    results={}
    for fold in range(5):
        allowed=set(manifest["folds"][str(fold)]["train_sources"]);selected=[e for e in examples if e["source"] in allowed]
        tree=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,e in enumerate(selected):tree[e["source"]][e["video"],e["gt"]][e["image"],e["row"]].append(i)
        weights=np.zeros(len(selected),np.float64)
        for source,trajectories in tree.items():
            for trajectory,observations in trajectories.items():
                for observation,indices in observations.items():
                    for index in indices:weights[index]=1/len(tree)/len(trajectories)/len(observations)/len(indices)
        check(len(selected)>0 and np.all(weights>0) and abs(weights.sum()-1)<1e-12,"Independent hierarchical weights invalid")
        sums=np.zeros(9);mass=np.zeros(9)
        def marginal(e,w):
            p=e["positive"];count=int(p.sum());return np.where(p,w/(2*count),w/(2*(len(p)-count)))
        for e,w in zip(selected,weights):
            ew=marginal(e,w)[:,None]*e["masks"];mass+=ew.sum(axis=0);sums+=(np.where(e["masks"],e["values"],0)*ew).sum(axis=0)
        mean=np.divide(sums,mass,out=np.zeros(9),where=mass>0);variance=np.zeros(9)
        for e,w in zip(selected,weights):
            ew=marginal(e,w)[:,None]*e["masks"];variance+=(np.where(e["masks"],e["values"]-mean,0)**2*ew).sum(axis=0)
        std=np.sqrt(np.divide(variance,mass,out=np.ones(9),where=mass>0));std=np.where(mass>0,np.maximum(std,1e-6),1.)
        results[fold]=dict(mean=mean,std=std,valid_marginal_weight=mass,n=len(selected),supported_sources=sorted(tree),source_querylag_counts={s:sum(e["source"]==s for e in selected) for s in sorted(allowed)},global_weight_sum=float(weights.sum()))
    return results


def verify_training_lineage(out,spec,manifest,videos,frames,grouped,utility,anchors):
    designs=independent_train_design(spec,manifest,videos,frames,grouped,anchors,utility);projections={}
    for fold in range(5):
        allowed=manifest["folds"][str(fold)]["train_sources"]
        _,projections[fold]=independent_anchors(spec,videos,allowed,utility.base.project)
    expected_lineage=dict(fold_manifest_sha256=canonical(manifest),cache_sha256=canonical(sorted((int(r["image_id"]),r["npz_sha256"]) for r in frames)),modules_sha256=spec["modules_sha256"],data_module_sha256=spec["data_module_sha256"],recommendation_sha256=spec.get("recommendation_sha256"))
    for seed in spec["seeds"]:
        for fold in range(5):
            model=out/"models"/("fold%d_seed%d"%(fold,seed));checkpoint=load(model/"checkpoint.json");norm=load(model/"normalizer.json");trace=load(model/"training_trace.json");receipt=load(model/"single_model_receipt.json");design=designs[fold];meta=checkpoint["metadata"]
            check(meta["train_label_projection_sha256"]==projections[fold]["train_label_projection_sha256"],"Allowed train label projection differs")
            for key,value in projections[fold].items():compare(receipt["train_projection_receipt"][key],value,"Train projection receipt")
            for key,value in expected_lineage.items():
                if value is not None:compare(meta["GT_free_lineage"][key],value,"GTfree checkpoint lineage")
            check(checkpoint["schema"]=="IDENTITY_SPACETIME_DETERMINISTIC_CHECKPOINT_V1" and set(checkpoint["head_states"])==set(FEATURES[2:]),"Missing checkpoint arms")
            arrays(checkpoint["adapter_state"],256,64,256)
            for state in checkpoint["head_states"].values():arrays(state,19,32,1)
            for field in ("mean","std","valid_marginal_weight"):np.testing.assert_allclose(norm[field],design[field],atol=1e-11,rtol=1e-11)
            n=design["n"];batch=spec["numeric_training_config"]["batch_size"];steps=(n+batch-1)//batch
            check(trace["training_query_lags"]==n and trace["adapter_optimizer_steps"]==50*steps and all(x==20*steps for x in trace["head_optimizer_steps"].values()),"Independent train population/step count differs")
            check(set(trace["head_optimizer_steps"])==set(FEATURES[2:]) and trace["adapter_frozen_parameter_sha256"]==canonical(checkpoint["adapter_state"]),"Frozen adapter SHA or head set differs")
            compare(trace["coverage"]["all_source_coverage"],{s:design["source_querylag_counts"].get(s,0) for s in sorted(manifest["groups"])},"Independent source support")
            check(abs(trace["global_weight_sum"]-design["global_weight_sum"])<1e-12,"Global train weight differs")
            compare(trace["schedule"],spec["numeric_training_config"],"Trace schedule")
            for phase,epochs in (("adapter",50),("heads",20)):
                check(len(trace[phase])==epochs,"Missing training epochs");rng=np.random.RandomState(seed)
                for epoch,entry in enumerate(trace[phase],1):
                    order=rng.permutation(n);sizes=[len(order[i:i+batch]) for i in range(0,n,batch)]
                    check(entry["epoch"]==epoch and entry["permutation_sha256"]==hashlib.sha256(np.asarray(order,np.int64).tobytes()).hexdigest() and entry["batch_sizes"]==sizes,"Seeded epoch permutation/short B differs")
                    losses=entry["surrogate_step_objective_mean"]
                    check(all(np.isfinite(v) for v in losses.values()) if isinstance(losses,dict) else np.isfinite(losses),"Nonfinite saved train objectives")
    return dict(per_fold_training_query_lags={str(f):d["n"] for f,d in designs.items()},all_15_allowed_label_projections_original_geometry_normalizers_global_hierarchy_and_fixed_epoch_traces=True,optimizer_replay_performed=False)


def inspect_output_scores(out,spec,manifest,videos,grouped,anchors,db,utility):
    cells={(s,source,l,scope,pop):fresh() for s in spec["seeds"] for source in manifest["groups"] for l in (1,4) for scope in SCOPES for pop in POPS}
    counts=Counter();maximum=0;error=0.;blocks=0;absolute=np.zeros(10);relative=np.zeros(10)
    parent_dir=Path(spec["parent_score_directory"])
    for seed in spec["seeds"]:
        for fold in range(5):
            model=out/"models"/("fold%d_seed%d"%(fold,seed));checkpoint=load(model/"checkpoint.json");norm=load(model/"normalizer.json");held=set(manifest["folds"][str(fold)]["held_out_sources"])
            with (model/"frame_pairs.jsonl").open() as stream:
                for video,records in sorted(grouped.items()):
                    source=videos[str(video)]
                    if source not in held:continue
                    rows=load(model/"shards"/("v%d.rows.json"%video));check(rows["video_id"]==video and len(rows["frames"])==len(records),"Full row map frame coverage")
                    maps={l:np.load(model/"shards"/("v%d_lag%d.scores.npy"%(video,l)),mmap_mode="r",allow_pickle=False) for l in (1,4)}
                    parent={l:np.load(parent_dir/"shards"/("v%d_lag%d.scores.npy"%(video,l)),mmap_mode="r",allow_pickle=False) for l in (1,4)}
                    history={};ring=deque(maxlen=5);offsets=Counter()
                    for step,record in enumerate(records):
                        check(sha(record["npz"])==record["npz_sha256"],"Raw cache bytes changed")
                        q=utility.read_raw(record,history);ring.append(q)
                        compare(rows["frames"][step],independent_rowmap(q,utility),"Independent immutable row map/history")
                        qi=np.array([anchors.get((q["image_id"],i),(video,-1))[1] for i in range(len(q["raw"]))],np.int64)
                        for lag in (1,4):
                            text=next(stream,None);check(text is not None,"Missing frame receipt");receipt=json.loads(text)
                            r=None if step<lag else ring[-lag-1];nq=len(q["raw"]);nm=0 if r is None else len(r["raw"]);count=nq*nm;offset=offsets[lag]
                            for key,value in dict(source=source,video_id=video,lag=lag,query_step=step,reference_step=None if r is None else step-lag,query_image_id=q["image_id"],reference_image_id=None if r is None else r["image_id"],query_rows=nq,reference_rows=nm,candidates=count,offset=offset,score_file="shards/v%d_lag%d.scores.npy"%(video,lag),original_frame_gap=None if r is None else q["original_frame_index"]-r["original_frame_index"]).items():compare(receipt[key],value,"Frame receipt "+key)
                            counts[seed,"receipts"]+=1;counts[seed,lag]+=count;offsets[lag]+=count;maximum=max(maximum,count)
                            if r is None:
                                compare(receipt["gperm_coverage"],dict(mapping_nonidentity_references=0,mapping_nonidentity_candidates=0,geometry_values_changed_candidates=0),"Prefix coverage");continue
                            stored=maps[lag][offset:offset+count].copy().reshape(nq,nm,10)
                            check(np.isfinite(stored).all(),"Nonfinite stored scores")
                            expected=independent_model_scores(q,r,norm,checkpoint["adapter_state"],checkpoint["head_states"])
                            np.testing.assert_allclose(stored[:,:,1:],expected[:,:,1:],atol=SCORE_ATOL,rtol=SCORE_RTOL)
                            check(np.array_equal(stored[:,:,0],expected[:,:,0]) and np.array_equal(stored[:,:,0],parent[lag][offset:offset+count,0].reshape(nq,nm)),"Raw original parent score not bitwise exact")
                            if count:
                                differences=np.abs(stored-expected);absolute=np.maximum(absolute,np.max(differences,axis=(0,1)))
                                relative=np.maximum(relative,np.max(differences/np.maximum(np.abs(expected),SCORE_ATOL),axis=(0,1)))
                                error=max(error,float(np.max(differences)))
                            blocks+=1
                            ri=np.array([anchors.get((r["image_id"],i),(video,-1))[1] for i in range(nm)],np.int64);values,masks=independent_geometry(q,r);donors=independent_donors(r["predicted_categories"],np.isfinite(r["previous_times"]))
                            g1=independent_inputs(np.zeros((nq,nm)),values,masks,norm["mean"],norm["std"],"G1",donors);gp=independent_inputs(np.zeros((nq,nm)),values,masks,norm["mean"],norm["std"],"Gperm",donors)
                            changed=np.any(g1[:,:,np.array(POSITION_SLOTS)+1]!=gp[:,:,np.array(POSITION_SLOTS)+1],axis=2);moved=np.broadcast_to(donors!=np.arange(nm),(nq,nm));moved_refs=int(sum(donors!=np.arange(nm)))
                            compare(receipt["gperm_coverage"],dict(mapping_nonidentity_references=moved_refs,mapping_nonidentity_candidates=moved_refs*nq,geometry_values_changed_candidates=int(changed.sum())),"Full GTfree Gperm coverage")
                            same=q["predicted_categories"][:,None]==r["predicted_categories"][None,:];cv=np.isfinite(r["previous_times"])
                            for scope,selection in ((SCOPES[0],np.ones((nq,nm),bool)),(SCOPES[1],same)):
                                for pop,mask in ((POPS[0],selection),(POPS[1],selection&cv[None,:])):add_block(cells[seed,source,lag,scope,pop],stored,mask,qi,ri,changed,moved,(seed,source,lag,scope,pop,video,q["image_id"]),db)
                    for lag in (1,4):check(offsets[lag]==len(maps[lag]),"Unconsumed score rows")
                    db.commit();maps.clear();parent.clear()
                check(next(stream,None) is None,"Unexpected frame receipt")
            print("independent scores complete: seed=%d fold=%d"%(seed,fold),flush=True)
    stream=dict(maximum_score_rows_read=maximum,exact_parent_raw_blocks=blocks,candidates_by_seed={str(s):{str(l):counts[s,l] for l in (1,4)} for s in spec["seeds"]},frame_pair_receipts_by_seed={str(s):counts[s,"receipts"] for s in spec["seeds"]})
    return cells,stream,dict(maximum_absolute_score_error=error,maximum_absolute_error_by_column={f:float(absolute[c]) for c,f in enumerate(FEATURES)},maximum_relative_error_by_column={f:float(relative[c]) for c,f in enumerate(FEATURES)},relative_error_denominator="max(abs(independently_recomputed_score),1e-10); raw column additionally bitwise exact")


def compare_databases(independent,actual):
    other=sqlite3.connect("file:"+str(actual)+"?mode=ro",uri=True)
    try:
        check([r[1:3] for r in independent.execute("PRAGMA table_info(query_support)")]==[r[1:3] for r in other.execute("PRAGMA table_info(query_support)")],"Query SQLite schema differs")
        order="seed,lag,scope,population,video,image,qrow"
        left=independent.execute("SELECT * FROM query_support ORDER BY "+order);right=other.execute("SELECT * FROM query_support ORDER BY "+order);count=0
        while True:
            a,b=left.fetchmany(4096),right.fetchmany(4096)
            check(a==b,"Independent query SQLite rows differ")
            if not a:break
            count+=len(a)
        return count
    finally:other.close()


def verify_parent_report(results,parent,seeds,expected_shared=1331):
    count=0;keys=("candidate_pairs","known_pairs","positive_pairs","negative_pairs","unknown_pairs","nonempty_known_queries","positive_available_queries","negative_available_queries","hard_queries")
    for seed in seeds:
        report=results["per_seed"][str(seed)]
        for source in [None]+sorted(report["per_source"]):
            a=report["micro_union"] if source is None else report["per_source"][source]["lags"];b=parent["results"]["micro_union"] if source is None else parent["results"]["per_source"][source]["lags"]
            for lag in (1,4):
                for scope in SCOPES:
                    for pop in POPS:
                        x,y=a[str(lag)][scope][pop],b[str(lag)][scope][pop]
                        compare({k:x[k] for k in keys},{k:y[k] for k in keys},"Parent counts")
                        for subset in SUBSETS:compare(x["top1"][subset][FEATURES[0]],y["top1"][subset][FEATURES[0]],"Parent raw top1")
                        compare(x["positive_negative_pair_order"][FEATURES[0]],y["positive_negative_pair_order"][FEATURES[0]],"Parent raw pair order");count+=1
        check(report["cross_lag"][PRIMARY[0]][PRIMARY[1]]["shared_hard_queries"]==expected_shared,"Fixed common-CV shared query set differs")
    return dict(status="PASS_EXACT_PARENT_RAW_SOURCE_UNION_COUNTS_TOP1_PAIRORDER",cells=count,shared_hard_queries=expected_shared)


def actual_chain(out,spec,protocol):
    receipt=load(out/"coordinator_receipt.json");check(receipt["status"]=="COMPLETE_ALL_MODELS_GLOBAL_SEAL_AND_UNIFIED_EVALUATION_PENDING_INDEPENDENT_AUDIT" and receipt["GPU_used"] is False,"Coordinator incomplete")
    check(receipt["protocol_sha256"]==sha(protocol) and receipt["global_seal_sha256"]==sha(out/"GT_free_OOF_predictions_sealed.json"),"Coordinator binding differs")
    entries=receipt["actual_model_processes"];check(len(entries)==15,"Missing actual model wait records")
    expected={(f,s) for f in range(5) for s in spec["seeds"]};seen=set()
    logs=Path(spec["execution_directory"])/"model_processes"
    for entry in entries:
        cmd=entry["command"];fold=int(cmd[cmd.index("--fold")+1]);seed=int(cmd[cmd.index("--seed")+1]);name="fold%d_seed%d"%(fold,seed)
        check((fold,seed) not in seen and (fold,seed) in expected,"Duplicate actual model command");seen.add((fold,seed))
        check(entry["actual_exit_code"]==0 and entry["waited_by_parent_pid"]>0 and "--execute" in cmd and spec["trainer_path"] in cmd and str(protocol) in cmd,"Actual model process did not exit0")
        compare(load(logs/(name+".exit.json")),entry,"Actual model exit receipt")
    check(seen==expected,"Actual fifteen process population differs")
    evaluation=receipt["evaluation_process"];check(evaluation["actual_exit_code"]==0 and spec["evaluator_path"] in evaluation["command"],"Actual unified evaluator exit absent")
    compare(load(logs/"unified_evaluation.exit.json"),evaluation,"Evaluation exit receipt")
    return dict(actual_model_exit_codes=[0]*15,actual_evaluator_exit_code=0,actual_waited_child_records_and_commands_checked=True)


def audit(out,protocol):
    started=time.monotonic();out=Path(out);protocol=Path(protocol);spec=load(protocol)
    check(spec["status"]=="FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL" and spec["schema"]=="IDENTITY_SPACETIME_SOURCE_OOF_V1","Protocol not frozen")
    check(spec["folds"]==list(range(5)) and spec["seeds"]==[20261024,20261025,20261026] and spec["score_columns"]==spec["feature_names"]==list(FEATURES),"Fixed experiment dimensions changed")
    check(spec["verifier_sha256"]==sha(__file__) and Path(spec["verifier_path"]).resolve()==Path(__file__).resolve(),"Independent auditor not pinned")
    pins=all_pins(spec);pinned_bytes(pins);protected=load(spec["protected_authority_path"])["protected_files"];check(len(protected)==7,"Protected authority count differs");compare(protected_snapshot(protected),protected,"Protected before")
    roles,manifest,videos,frames,grouped,population=metadata(spec)
    seal=global_seal_gate(out,spec,manifest,videos,grouped) # NO identity decoder before this line.
    gate=load(spec["implementation_preflight"]["path"])
    check(gate["status"]==spec["implementation_preflight"]["status"]=="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_IMPLEMENTATION_PREFLIGHT" and gate["protocol_sha256"]==sha(protocol),"Implementation preflight absent or stale")
    for key in ("trainer","coordinator","evaluator","verifier"):check(gate[key+"_sha256"]==spec[key+"_sha256"],"Stale preflight component")
    check(gate["data_sha256"]==spec["data_module_sha256"] and gate["modules_sha256"]==spec["modules_sha256"],"Stale preflight data/modules")
    worker_pins=dict(spec["input_pins"])
    for key in ("manual","index"):worker_pins[spec["identity_authority"][key+"_path"]]=spec["identity_authority"][key+"_sha256"]
    worker_pins.update({str(protocol):sha(protocol),spec["trainer_path"]:spec["trainer_sha256"],spec["implementation_preflight"]["path"]:sha(spec["implementation_preflight"]["path"])})
    for directory in (out/"models").iterdir():
        recorded=load(directory/"single_model_receipt.json")["input_sha256"]
        compare(recorded,worker_pins,"Worker complete external authority pins")
    utility=audited_utilities();anchors,label_projection=independent_anchors(spec,videos,sorted(roles),utility.base.project)
    summary_path=out/"evaluation/summary.json";summary=load(summary_path);database_path=out/"evaluation/query_support.sqlite"
    check(summary["status"]=="COMPLETE_UNIFIED_ALL_SEALED_SOURCE_OOF_EVALUATION_PENDING_INDEPENDENT_AUDIT" and summary["protocol_sha256"]==sha(protocol) and summary["script_sha256"]==spec["evaluator_sha256"] and summary["query_database_sha256"]==sha(database_path),"Evaluation summary bindings differ")
    for key,value in dict(global_seal_verified_before_identity_import_and_after=True,all_fifteen_models_sealed_before_evaluation=True,unknown_is_negative=False,no_positive_is_unmatched=False,training_modified=False,GPU_used=False,eligible_all_FIT_manual_anchors=len(anchors)).items():compare(summary[key],value,"Summary "+key)
    for key,value in label_projection.items():compare(summary["label_projection_receipt"][key],value,"Unified label projection")
    print("independent global seal and safe eligible projection complete",flush=True)
    lineage=verify_training_lineage(out,spec,manifest,videos,frames,grouped,utility,anchors)
    print("independent allowed train projections, geometry moments and epoch traces complete",flush=True)
    with tempfile.TemporaryDirectory(prefix="identity_oof_independent_sql_") as temporary:
        db=open_db(Path(temporary)/"independent.sqlite")
        cells,stream,error=inspect_output_scores(out,spec,manifest,videos,grouped,anchors,db,utility)
        compare(summary["stream_receipt"],stream,"Full independent stream receipt")
        query_rows=compare_databases(db,database_path)
        print("independent full candidate numerical reconstruction and SQLite rows complete",flush=True)
        independent=reports(db,cells,roles,spec["seeds"]);compare(summary["results"],independent,"All independent report statistics")
        parity=verify_parent_report(independent,load(Path(spec["parent_score_directory"])/"summary.json"),spec["seeds"]);compare(summary["baseline_parity"],parity,"Parent report parity")
        db.close()
    chain=actual_chain(out,spec,protocol);global_seal_gate(out,spec,manifest,videos,grouped);pinned_bytes(pins);compare(protected_snapshot(protected),protected,"Protected after")
    value=dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT",protocol_sha256=sha(protocol),global_seal_sha256=sha(out/"GT_free_OOF_predictions_sealed.json"),evaluation_summary_sha256=sha(summary_path),query_database_sha256=sha(database_path),coordinator_receipt_sha256=sha(out/"coordinator_receipt.json"),trainer_sha256=spec["trainer_sha256"],evaluator_sha256=spec["evaluator_sha256"],verifier_sha256=sha(__file__),watched_pins_sha256=pins,
        independent_checks=dict(global_all15_seal_complete_hash_pass_before_identity_decoder=True,population=population,source_fold_manifest_independently_rebuilt=True,all_ten_columns_every_retained_candidate_independently_recomputed=True,raw_parent_bitwise_exact_every_candidate=True,learned_score_atol=SCORE_ATOL,learned_score_rtol=SCORE_RTOL,numerical_reconstruction=error,maximum_absolute_score_error=error["maximum_absolute_score_error"],full_GTfree_Gperm_donors_and_actual_value_change_verified=True,immutable_maps_original_times_exact_lags_prefix_empty_unknown_coverage_verified=True,training_lineage=lineage,query_SQLite_rows_independently_rebuilt=query_rows,all_source_role_dataset_union_macro_paired_interaction_crosslag_seed_deletion_statistics_independently_rebuilt=True,baseline_parity=parity,actual_child_chain=chain,protected_files_before_after=protected),
        elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,GPU_used=False,interpretation=summary["interpretation"],coverage_note="All-nonempty-known includes queries without an available positive; their conditional top1 is false by construction and is coverage, not a deployed no-match error. Mechanism comparisons use fixed hard P+N queries; no-positive rows never train and never define a null target.")
    with (out/"independent_output_audit.json").open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-self-test",action="store_true")
    parser.add_argument("--out",type=Path);parser.add_argument("--protocol",type=Path)
    args=parser.parse_args()
    if args.math_self_test:result=math_self_test()
    else:
        check(args.out is not None and args.protocol is not None,"Complete output audit requires --out and --protocol")
        result=audit(args.out,args.protocol)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
