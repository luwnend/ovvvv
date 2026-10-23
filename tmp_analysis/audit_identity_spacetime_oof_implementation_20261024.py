#!/usr/bin/env python3
"""Independent complete synthetic boundary/trainer/runner/output audit gate.

Data-only checks never confer the complete implementation preflight status.
No real cached feature or manual identity payload is opened by this checker.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import verify_identity_spacetime_oof_output_20261024 as numerical
import verify_angular_appearance_control_output_20261024 as history_oracle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tmp_analysis/identity_spacetime_oof_data_20261024.py"
TRAINER = ROOT / "tmp_analysis/train_identity_spacetime_oof_model_20261024.py"
DATA_SHA = "54c57aee5cba1153ae18205a451b44eecc2fd6ae196d435b8e5f9c3fb8cdcbee"
TRAINER_SHA = "46be289030b99ef450d38cdc313f2a2d7ca3db97e0d0b5ac0a1101ff84e3d4c7"
MODULES_SHA = "2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3"


def check(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def oracle_folds(roles):
    strata = defaultdict(list)
    for source, row in roles.items():
        if row["split"] == "fit":
            strata[row["block"], row["role"]].append(source)
    groups, orders = {}, {}
    for (block, role), sources in sorted(strata.items()):
        ordered = sorted(sources, key=lambda source: (hashlib.sha256(b"identity_spacetime_source_oof_v1\0" + source.encode()).hexdigest(), source))
        orders[block+"/"+role] = ordered
        for index, source in enumerate(ordered):
            groups[source] = dict(fold=index % 5, role=role, dataset_block=block, stratum=[block, role])
    sources = sorted(groups)
    return dict(seed="identity_spacetime_source_oof_v1", number_of_folds=5, groups={s:groups[s] for s in sources},
        folds={str(fold):dict(held_out_sources=[s for s in sources if groups[s]["fold"]==fold],
            train_sources=[s for s in sources if groups[s]["fold"]!=fold]) for fold in range(5)},
        strata_order=orders, source_count=len(sources), roles_are_metadata_only=True, labels_used=False)


def identity_fixture(folder):
    train, held = "D/train", "D/held"
    roles = {s:dict(split="fit", role="mean_role", block="D") for s in (train, held, "D/zero")}
    roles["D/tune"] = dict(split="tune", role="tune_evaluation", block="D")
    good = dict(observation_id=1, split="fit", source_group=train, video_id=1, image_id=100, sampled_step=0,
        original_frame_index=0, local_track_id=0, gt_track_id=7, pred_category_id=1, manual_anchor_id=1, eligible_for_parameter_fit=True)
    identities = [dict(good, gt_category_id="POISON_CLASS", error={"POISON_ERROR":1}, soft_risk="POISON_RISK", posterior_argmax="POISON_POSTERIOR"),
        dict(good, observation_id="POISON_HELD_OBSERVATION", source_group=held, video_id="POISON_HELD_VIDEO",
             gt_track_id={"POISON_HELD_ID":[10**100]}, eligible_for_parameter_fit="POISON_HELD_ELIGIBILITY"),
        dict(split="tune", source_group={"POISON_TUNE_SOURCE":False}, gt_track_id="POISON_TUNE_ID"),
        dict(good, observation_id="POISON_INELIGIBLE", gt_track_id="POISON_INELIGIBLE_ID", eligible_for_parameter_fit=False)]
    index = [{key:value for key,value in good.items() if key not in ("gt_track_id", "manual_anchor_id", "eligible_for_parameter_fit")},
        dict(split="fit", source_group=held, observation_id="POISON_HELD_INDEX", video_id={"POISON_HELD_INDEX_VIDEO":1}),
        dict(split="tune", source_group="POISON_TUNE_INDEX", observation_id="POISON_TUNE_INDEX_ID")]
    index[0]["dump_row"] = 0
    manual_path, index_path = folder / "manual.jsonl", folder / "index.jsonl"
    def save():
        manual_path.write_text("".join(json.dumps(row)+"\n" for row in identities))
        index_path.write_text("".join(json.dumps(row)+"\n" for row in index))
        return dict(manual_path=str(manual_path), manual_sha256=sha(manual_path), index_path=str(index_path), index_sha256=sha(index_path))
    return roles, identities, index, save


def audit_identity_boundary(data):
    with tempfile.TemporaryDirectory(prefix="independent_oof_identity_boundary_") as temporary:
        roles, identities, index, save = identity_fixture(Path(temporary))
        authority = save()
        scanner = data.component(data.HELPER)
        original_loads, original_whole_loader = json.loads, scanner.load_fit_identities
        decoder_observations = []
        def guarded(text, *args, **kwargs):
            check("POISON_" not in text, "Forbidden held-out/tune/ineligible/correctness value reached JSON decoder")
            decoder_observations.append(text)
            return original_loads(text, *args, **kwargs)
        def forbidden_whole_loader(*args, **kwargs):
            raise AssertionError("Whole-FIT identity loader was imported into model training path")
        json.loads, scanner.load_fit_identities = guarded, forbidden_whole_loader
        try:
            anchors, receipt = data.load_train_anchors(["D/train", "D/zero"], roles, {"1":"D/train"}, authority)
        finally:
            json.loads, scanner.load_fit_identities = original_loads, original_whole_loader
        check(anchors == {(100, 0):(1, 7)}, "Allowed projection not reconstructed")
        allowed = {key:identities[0][key] for key in data.IDENTITY_FIELDS}
        check(receipt["train_label_projection_sha256"] == canonical([allowed]), "Canonical train-only projection digest differs")
        check(receipt["eligible_identity_counts_by_source"] == {"D/train":1, "D/zero":0}, "Zero-support source omitted from train label receipt")
        original_authority = authority.copy()
        identities[1]["gt_track_id"] = {"POISON_MUTATED_HELD_ID": ["different", False]}
        index[1]["observation_id"] = "POISON_MUTATED_HELD_INDEX"
        authority = save()
        check(authority["manual_sha256"] != original_authority["manual_sha256"] and authority["index_sha256"] != original_authority["index_sha256"], "Mutation failed to change actual authority bytes")
        json.loads = guarded
        try:
            mutated, changed_receipt = data.load_train_anchors(["D/train", "D/zero"], roles, {"1":"D/train"}, authority)
        finally:
            json.loads = original_loads
        check(mutated == anchors and changed_receipt["train_label_projection_sha256"] == receipt["train_label_projection_sha256"] and
              changed_receipt["immutable_anchor_sha256"] == receipt["immutable_anchor_sha256"], "Held-out value mutation affected train projection")
        check(changed_receipt["input_sha256"] != receipt["input_sha256"], "External authority receipt failed to observe payload mutation")
        identities[0]["gt_track_id"] = 8
        authority = save()
        changed, train_changed = data.load_train_anchors(["D/train"], roles, {"1":"D/train"}, authority)
        check(changed[(100,0)] == (1,8) and train_changed["train_label_projection_sha256"] != receipt["train_label_projection_sha256"], "Allowed identity mutation had no actual effect")
        stale = dict(authority, manual_sha256=original_authority["manual_sha256"])
        try:
            data.load_train_anchors(["D/train"], roles, {"1":"D/train"}, stale)
        except ValueError:
            pass
        else:
            raise AssertionError("Authority SHA mismatch accepted")
        return dict(split_source_eligibility_before_identity=True, actual_decoder_poison_guard=True,
            whole_FIT_loader_forbidden=True, heldout_mutation_updates_authority_but_preserves_train_projection=True,
            allowed_train_identity_mutation_changes_projection=True, decoded_value_fragments=len(decoder_observations))


def frame_fixture():
    random = np.random.RandomState(4417)
    frames, records, anchors = {}, [], {}
    for video, counts in ((1,[3,3,0,3,3,3]), (2,[3,3,3,3,3,3]), (3,[0,0])):
        for step, n in enumerate(counts):
            time = [0,7,15,42,90,103][step]
            ids = [0,0,-1] if video == 1 and step == 1 else [0,-1,2]
            raw = random.randn(n,256).astype(np.float16).astype(float)
            if step==0:
                raw[:]=0
            frame = dict(video_id=video, image_id=video*100+step, sampled_step=step, original_frame_index=time,
                boxes=np.asarray([[time + row*20,0,time+row*20+10,10] for row in range(n)],float).reshape(n,4),
                raw=raw, released_ids=np.asarray(ids if n else [],np.int64), predicted_categories=np.ones(n,np.int32))
            frames[video,step]=frame
            records.append(dict(video_id=video,frame_id=step,observations=n))
            for row in range(n):
                # Deliberately omit row1 as an unknown donor in first frame.
                if step==0 and row==1:
                    continue
                anchors[frame["image_id"],row]=(video,7 if row in (0,1) else 8)
    return frames, records, anchors


def audit_frame_and_projection(data):
    frames, records, anchors = frame_fixture()
    sources={"1":"D/train","2":"D/other","3":"D/zero"}
    bounds=dict(max_frame_rows=3,max_video_frames=6,max_candidates_per_frame_pair=9,generation_ring_frames=5)
    actual=list(data.iter_gtfree_frame_pairs([1,2,3],frame_records=records,video_sources=sources,
        loader=lambda row:copy.deepcopy(frames[row["video_id"],row["frame_id"]]),bounds=bounds))
    expected_pairs, counts = [], Counter()
    for video in (1,2,3):
        history, ring, offsets = {}, deque(maxlen=5), Counter()
        for record in [r for r in records if r["video_id"]==video]:
            q=history_oracle.independent_history(copy.deepcopy(frames[video,record["frame_id"]]),history)
            ring.append(q)
            for lag in (1,4):
                r=ring[-lag-1] if record["frame_id"]>=lag else None
                nq,nr=len(q["boxes"]),0 if r is None else len(r["boxes"])
                receipt=dict(video_id=video,source_group=sources[str(video)],lag=lag,query_step=q["sampled_step"],query_image_id=q["image_id"],
                    reference_step=None if r is None else r["sampled_step"],reference_image_id=None if r is None else r["image_id"],
                    query_rows=nq,reference_rows=nr,candidates=nq*nr,offset=offsets[lag],
                    original_frame_gap=None if r is None else q["original_frame_index"]-r["original_frame_index"],
                    feature_ring_frames=len(ring),row_order="query_major_then_reference_dump_row")
                expected_pairs.append((q,r,receipt))
                offsets[lag]+=nq*nr
                counts[lag]+=nq*nr
    check(len(actual)==len(expected_pairs)==len(records)*2,"Prefix/empty frame-pair receipts missing")
    expected_examples=[]
    for pair,(q,r,receipt) in zip(actual,expected_pairs):
        check(pair["receipt"]==receipt,"Exact sampled lag/empty prefix/fullQxM offset differs")
        for key in ("previous_times","previous_boxes"):
            np.testing.assert_allclose(pair["query"][key],q[key],equal_nan=True,rtol=0,atol=0)
        check(pair["query"]["previous_locators"]==q["previous_locators"] and pair["query"]["history_reason"]==q["history_reason"],"Past locator/duplicate invalidation differs")
        check(not any("gt" in key.lower() for key in pair["query"]),"GT leaked into GT-free frame iterator")
        if r is None:
            check(pair["geometry_values"].shape==(len(q["boxes"]),0,9),"Prefix geometry shape changed")
            continue
        values,masks=numerical.independent_geometry(q,r)
        np.testing.assert_allclose(pair["geometry_values"],values,rtol=1e-13,atol=1e-13,equal_nan=True)
        np.testing.assert_array_equal(pair["geometry_masks"],masks)
        donors=numerical.independent_donors(r["predicted_categories"],np.isfinite(r["previous_times"]))
        np.testing.assert_array_equal(pair["gperm_plan"].donor_rows,donors)
        qids=[anchors.get((q["image_id"],row),(video,-1))[1] for row in range(len(q["boxes"]))]
        rids=[anchors.get((r["image_id"],row),(video,-1))[1] for row in range(len(r["boxes"]))]
        for row,gt in enumerate(qids):
            p=[j for j,value in enumerate(rids) if gt>=0 and value==gt and q["predicted_categories"][row]==r["predicted_categories"][j]]
            n=[j for j,value in enumerate(rids) if gt>=0 and value>=0 and value!=gt and q["predicted_categories"][row]==r["predicted_categories"][j]]
            if p and n:
                expected_examples.append((q["image_id"],row,receipt["lag"],p,n))
    coverage=Counter()
    examples=list(data.build_train_examples(actual,anchors,set(sources.values()),coverage))
    projected=[(e["record"]["image_id"],e["record"]["qrow"],e["record"]["lag"],e["record"]["positive_rows"].tolist(),e["record"]["negative_rows"].tolist()) for e in examples]
    check(projected==expected_examples,"Training projection changed samepred known p+n population")
    example=next(e for e in examples if e["record"]["image_id"]==101 and e["record"]["qrow"]==0 and e["record"]["lag"]==1)
    check(example["reference_raw"].shape==(3,256) and example["gperm_plan"].donor_rows.tolist()==[2,0,1],"Complete retained reference/donor population changed")
    check(example["record"]["negative_rows"].tolist()==[2] and not example["masks"][:,5:].any(),"Unknown donor became negative or CV availability filtered train query")
    check(data.build_train_examples(iter([]),{},["D/train"]) is not None,"Empty train stream failed")
    check(list(data.build_train_examples(actual,{},set(sources.values())))==[],"No eligible identity generated no-match training labels")
    return dict(frames=len(records),frame_pair_receipts=len(actual),candidates_by_lag={str(k):v for k,v in counts.items()},
        independent_geometry_history_receipts_all_blocks=True,train_query_lags=len(examples),
        complete_unknown_refs_and_GTfree_donors_preserved=True,training_no_CV_or_score_filter=True,coverage=dict(coverage))


def audit_data_only():
    check(sha(DATA)==DATA_SHA,"Frozen data implementation changed")
    data=module(DATA,"independent_oof_data_subject")
    receipt_pins={}
    for number in (1,2,3):
        path=ROOT/("tmp_analysis/identity_spacetime_oof_data_synthetic_round%d_20261024.json"%number)
        receipt=json.loads(path.read_text())
        check(receipt["status"]=="PASS_IDENTITY_SPACETIME_OOF_DATA_SYNTHETIC" and receipt["round"]==number and receipt["script_sha256"]==DATA_SHA,"Data synthetic round mismatch")
        receipt_pins[str(path)]=sha(path)
    roles={"D/%02d"%i:dict(split="fit",role="mean_role" if i%2 else "pair_role",block="D") for i in range(23)}
    roles.update({"E/%02d"%i:dict(split="fit",role="mean_role",block="E") for i in range(8)})
    roles["D/tune"]=dict(split="tune",role="tune_evaluation",block="D")
    check(data.build_fold_manifest(roles)==oracle_folds(roles),"Independent SHA/NUL stratified source fold rule differs")
    check(data.build_fold_manifest(dict(reversed(list(roles.items()))))==oracle_folds(roles),"Folds depend on incoming metadata order")
    result=dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_DATA_ONLY_SYNTHETIC",checker_sha256=sha(__file__),
        data_module_sha256=DATA_SHA,reviewed_synthetic_round_files_sha256=receipt_pins,
        independent_checks=dict(source_fold_manifest_metadata_only=True,identity_boundary=audit_identity_boundary(data),
            frame_and_training_projection=audit_frame_and_projection(data),numerical_primitives=numerical.math_self_test()),
        real_cache_or_identity_opened=False,trainer_audited=False,runner_or_global_seal_audited=False,complete_OOF_preflight=False)
    check(sha(DATA)==DATA_SHA,"Data implementation changed during audit")
    return result


def audit_pure_trainer(data, trainer, modules):
    frames, records, anchors = frame_fixture()
    bounds=dict(max_frame_rows=3,max_video_frames=6,max_candidates_per_frame_pair=9,generation_ring_frames=5)
    pairs=list(data.iter_gtfree_frame_pairs([1,2],frame_records=records,video_sources={"1":"D/train","2":"D/other","3":"D/zero"},
        loader=lambda row:copy.deepcopy(frames[row["video_id"],row["frame_id"]]),bounds=bounds))
    full=list(data.build_train_examples(pairs,anchors,["D/train","D/other"]))
    compact=trainer.compact_examples(iter(full),["D/train","D/other"],modules)
    check(len(compact)==len(full)>3,"Synthetic trainer did not exercise supported examples")
    for original, gathered in zip(full,compact):
        known=np.sort(np.r_[original["record"]["positive_rows"],original["record"]["negative_rows"]])
        expected_donors=numerical.independent_donors(original["reference_predicted_categories"],original["reference_history_valid"],original["reference_dump_rows"])
        np.testing.assert_array_equal(gathered["full_donor_rows"],expected_donors)
        np.testing.assert_array_equal(gathered["reference_raw"],original["reference_raw"][known])
        np.testing.assert_allclose(gathered["donor_values"],original["values"][expected_donors[known]],equal_nan=True)
        np.testing.assert_array_equal(gathered["donor_masks"],original["masks"][expected_donors[known]])
    special=next((a,b) for a,b in zip(full,compact) if a["record"]["image_id"]==101 and a["record"]["qrow"]==0 and a["record"]["lag"]==1)
    check(special[1]["reference_raw"].shape[0]==2 and special[1]["full_reference_count"]==3,"Unknown raw rows entered trainable adapter")
    np.testing.assert_allclose(special[1]["donor_values"][1],special[0]["values"][1],equal_nan=True)
    # Independently enumerate per-example P x N values and gradients against
    # actual vectorized gathered trainer losses, including global N/B scaling.
    a,b=modules.IdentityAdapter256(20261024),modules.IdentityAdapter256(20261024)
    indices=np.array([0,1,3],np.int64)
    weights,_=modules.hierarchical_weights([e["record"] for e in compact],["D/train","D/other","D/zero","D/held"],["D/train","D/other","D/zero"])
    actual=trainer.adapter_batch_losses(a,compact,indices,modules)
    manual=[]
    for index in indices:
        example=compact[index]
        q=b(torch.tensor(example["query_raw"],dtype=torch.float64))
        refs=b(torch.tensor(example["reference_raw"],dtype=torch.float64))
        score=(q@refs.T)[0]
        p,n=example["record"]["positive_rows"],example["record"]["negative_rows"]
        terms=[torch.nn.functional.softplus((score[nr]-score[pr])/.1) for pr in p for nr in n]
        manual.append(torch.stack(terms).mean())
    manual=torch.stack(manual)
    torch.testing.assert_close(actual,manual,atol=1e-13,rtol=1e-13)
    objective=modules.weighted_batch_loss(actual,weights[indices],len(compact))
    expected=len(compact)/len(indices)*sum(float(weights[index])*loss for index,loss in zip(indices,manual))
    objective.backward();expected.backward()
    for left,right in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(left.grad,right.grad,atol=1e-11,rtol=1e-11)
    check(torch.count_nonzero(a.last.weight.grad)>0,"Synthetic wiring ignored legal identity objective")
    config=dict(trainer.CONFIG,adapter_epochs=2,head_epochs=2,batch_size=3)
    model=trainer.train_model(compact,["D/train","D/other","D/zero","D/held"],["D/train","D/other","D/zero"],20261024,modules,config)
    trace=model["training_trace"]
    steps=(len(compact)+2)//3
    check(trace["adapter_optimizer_steps"]==2*steps and all(value==2*steps for value in trace["head_optimizer_steps"].values()),"Actual optimizer step schedule changed")
    for phase in ("adapter","heads"):
        rng=np.random.RandomState(20261024)
        for epoch,entry in enumerate(trace[phase],1):
            order=rng.permutation(len(compact))
            sizes=[len(order[start:start+3]) for start in range(0,len(order),3)]
            digest=hashlib.sha256(np.asarray(order,np.int64).tobytes()).hexdigest()
            check(entry["epoch"]==epoch and entry["permutation_sha256"]==digest and entry["batch_sizes"]==sizes,"Seeded exact-once permutation or actual final B changed")
    initial_adapter=trainer.tensor_state(modules.IdentityAdapter256(20261024))
    check(trainer.object_sha(trainer.tensor_state(model["adapter"]))!=trainer.object_sha(initial_adapter),"Adapter parameters did not move under synthetic supervision")
    initial_head=trainer.object_sha(trainer.tensor_state(modules.ResidualAssociationHead19(20261024)))
    check(all(trainer.object_sha(trainer.tensor_state(head))!=initial_head for head in model["heads"].values()),"Some head ignored supervised gradients")
    check(all(not parameter.requires_grad and parameter.grad is None for parameter in model["adapter"].parameters()),"Head training leaked gradients into shared adapter")
    frozen_examples=trainer.prepare_head_examples(compact,model["adapter"],model["normalizer"],modules)
    for original,gathered,prepared in zip(full,compact,frozen_examples):
        known=np.sort(np.r_[original["record"]["positive_rows"],original["record"]["negative_rows"]])
        donor=numerical.independent_donors(original["reference_predicted_categories"],original["reference_history_valid"],original["reference_dump_rows"])
        for mode in ("G0","G1","Gmask","Gperm"):
            inputs=numerical.independent_inputs(np.zeros((1,len(original["reference_raw"]))),original["values"][None],original["masks"][None],
                model["normalizer"].mean,model["normalizer"].std,mode,donor)
            np.testing.assert_array_equal(prepared["geometry_inputs"][mode],inputs[0,known,1:])
    maximum=0.
    for pair in pairs:
        if pair["reference"] is None:
            continue
        actual,_=trainer.score_frame_pair(pair,model,modules)
        expected=numerical.independent_model_scores(pair["query"],pair["reference"],
            dict(mean=model["normalizer"].mean,std=model["normalizer"].std),trainer.tensor_state(model["adapter"]),
            {arm:trainer.tensor_state(head) for arm,head in model["heads"].items()})
        np.testing.assert_allclose(actual,expected,atol=numerical.SCORE_ATOL,rtol=numerical.SCORE_RTOL)
        check(np.array_equal(actual[:,:,0],expected[:,:,0]),"Raw parent expression is not exact")
        if actual.size:
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
    return model,compact,dict(training_examples=len(compact),independent_PxN_batched_loss_and_parameter_gradients=True,
        compact_unknown_raw_excluded_fullpool_unknown_geometry_donor_retained=True,
        actual_optimizer_steps_per_stage=2*steps,all_seeded_epoch_permutations_and_short_batches_verified=True,
        all_models_parameter_movement_nonzero=True,shared_adapter_frozen_after_head_training=True,
        compact_head_inputs_match_fullpool_normalize_then_intervene=True,
        all_synthetic_candidate_columns_independent_recomputation_maximum_absolute_error=maximum)


def full_worker_mutation_test(data, trainer):
    """Actual gated worker entry with temporary authorities and fixed 50/20."""
    with tempfile.TemporaryDirectory(prefix="independent_oof_full_worker_mutation_") as temporary:
        base=Path(temporary);cache=base/"cache";cache.mkdir()
        roles={"D/source%d"%i:dict(split="fit",role="mean_role",block="D") for i in range(5)}
        manifest=oracle_folds(roles);fold=0;seed=20261024
        held=set(manifest["folds"]["0"]["held_out_sources"])
        allowed=set(manifest["folds"]["0"]["train_sources"])
        video_sources={str(i+1):source for i,source in enumerate(sorted(roles))}
        records,identities,indices=[],[],[]
        rng=np.random.RandomState(1124)
        for v,source in video_sources.items():
            video=int(v)
            for step,time in enumerate((0,5,9,31,50,80)):
                n=0 if video==2 and step==2 else 3
                raw=rng.randn(n,256).astype(np.float16)
                boxes=np.asarray([[time+row*20,0,time+row*20+10,10,1] for row in range(n)],float).reshape(n,5)
                image=video*100+step;path=cache/("v%d_f%d.npz"%(video,step))
                np.savez(path,video_id=video,image_id=image,frame_id=step,frame_index=time,det_bboxes=boxes,track_feats=raw,
                    cls_feats=np.zeros((n,512),np.float16),track_ids=np.arange(n),det_labels=np.zeros(n,np.int32))
                records.append(dict(video_id=video,image_id=image,frame_id=step,frame_index=time,observations=n,npz=str(path),npz_sha256=sha(path)))
                for row in range(n):
                    if step==0 and row==1:
                        continue
                    observation=len(identities)
                    item=dict(observation_id=observation,split="fit",source_group=source,video_id=video,image_id=image,
                        sampled_step=step,original_frame_index=time,local_track_id=row,gt_track_id=7 if row<2 else 8,
                        pred_category_id=1,manual_anchor_id=observation,eligible_for_parameter_fit=True)
                    identities.append(dict(item,gt_category_id="POISON_CLASS",error="POISON_ERROR",soft_risk="POISON_RISK"))
                    indices.append({**{k:item[k] for k in data.INDEX_FIELDS if k!="dump_row"},"dump_row":row})
        held_rows=[row for row in identities if row["source_group"] in held]
        for row in held_rows:
            row["gt_track_id"]={"POISON_HELDOUT_ID":["unreadable",False]}
        manual,index=base/"manual.jsonl",base/"index.jsonl"
        def save_authority():
            manual.write_text("".join(json.dumps(row)+"\n" for row in identities)+json.dumps(dict(split="tune",gt_track_id="POISON_TUNE"))+"\n")
            index.write_text("".join(json.dumps(row)+"\n" for row in indices))
            return dict(manual_path=str(manual),manual_sha256=sha(manual),index_path=str(index),index_sha256=sha(index))
        def save(path,value):
            path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
        roles_path,plan_path,frames_path,inventory_path,fold_path=(base/name for name in ("roles.json","plan.json","frames.jsonl","inventory.json","folds.json"))
        save(roles_path,dict(groups=roles));save(plan_path,dict(video_records=[dict(video_id=int(v),source_group=s,split="fit") for v,s in video_sources.items()]))
        frames_path.write_text("".join(json.dumps(row)+"\n" for row in records))
        save(inventory_path,{Path(row["npz"]).name:row["npz_sha256"] for row in records});save(fold_path,manifest)
        parent=base/"parent";(parent/"shards").mkdir(parents=True)
        for video in sorted(map(int,video_sources)):
            vr=sorted((r for r in records if r["video_id"]==video),key=lambda r:r["frame_id"])
            raw=[]
            for r in vr:
                with np.load(r["npz"],allow_pickle=False) as saved:raw.append(saved["track_feats"].astype(np.float64))
            for lag in (1,4):
                parts=[(numerical.unit(raw[i])@numerical.unit(raw[i-lag]).T).reshape(-1) for i in range(lag,len(raw))]
                score=np.zeros((sum(len(x) for x in parts),6),np.float64)
                if len(score):score[:,0]=np.concatenate(parts)
                np.save(parent/"shards"/("v%d_lag%d.scores.npy"%(video,lag)),score)
        protocol_path,gate_path=base/"protocol.json",base/"temporary_gate.json"
        training=dict(device="cpu",dtype="float64",num_threads=1,optimizer="AdamW",adapter_epochs=50,head_epochs=20,batch_size=64,
            learning_rate=.001,weight_decay=.0001,grad_norm_clip=5.,adapter_pair_temperature=.1,head_loss_temperature=1.,normalizer_std_floor=1e-6,
            validation_or_early_stopping=False,checkpoint_selection="fixed_final_epoch",seed_selection=False)
        spec=dict(status="FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL",schema="IDENTITY_SPACETIME_SOURCE_OOF_V1",
            score_dtype="float64",score_columns=list(trainer.COLUMNS),feature_names=list(trainer.COLUMNS),numeric_training_config=dict(trainer.CONFIG),
            parent_score_directory=str(parent),training=training,seeds=list(trainer.SEEDS),folds=list(range(5)),lags=[1,4],data_module_path=str(DATA),data_module_sha256=sha(DATA),
            implementation_preflight=dict(path=str(gate_path),status="SYNTHETIC_ONLY_TEMPORARY_GATE"),
            source_roles_path=str(roles_path),collection_plan_path=str(plan_path),frame_execution_path=str(frames_path),dump_inventory_path=str(inventory_path),
            fold_manifest_path=str(fold_path),input_pins={str(path):sha(path) for path in (roles_path,plan_path,frames_path,inventory_path,fold_path,*data.PINS)})
        def save_protocol():
            spec["identity_authority"]=save_authority()
            save(protocol_path,spec)
            save(gate_path,dict(status="SYNTHETIC_ONLY_TEMPORARY_GATE",trainer_sha256=sha(TRAINER),modules_sha256=MODULES_SHA,
                data_sha256=sha(DATA),protocol_sha256=sha(protocol_path)))
        original_loads=json.loads
        def guarded(text,*args,**kwargs):
            check("POISON_" not in text,"Actual worker decoded heldout or forbidden payload before prediction seal")
            return original_loads(text,*args,**kwargs)
        outputs=[];authority_hashes=[]
        for variant in (0,1,2):
            if variant==1:
                for row in held_rows:
                    row["gt_track_id"]={"POISON_CHANGED_HELDOUT_ID":[variant,True]}
            elif variant==2:
                first=next(row for row in identities if row["source_group"] in allowed and row["sampled_step"]==0 and row["local_track_id"]==0)
                first["gt_track_id"]=99
            save_protocol();authority_hashes.append(spec["identity_authority"]["manual_sha256"])
            output=base/("model_variant%d"%variant)
            json.loads=guarded
            try:
                receipt=trainer.train_and_predict(fold,seed,protocol_path,output)
            finally:
                json.loads=original_loads
            check(receipt["held_out_identity_values_decoded"] is False,"Worker did not maintain train-only source boundary")
            outputs.append(output)
        check(len(set(authority_hashes))==3,"Mutation did not update exact shared authority each time")
        members=[str(path.relative_to(outputs[0])) for path in outputs[0].rglob("*") if path.is_file() and path.name!="single_model_receipt.json"]
        check(all(sha(outputs[0]/name)==sha(outputs[1]/name) for name in members),"Model f functional checkpoint/norm/trace/pred bytes changed under heldout-only label mutation")
        check(sha(outputs[0]/"checkpoint.json")!=sha(outputs[2]/"checkpoint.json"),"Allowed train-label change failed to affect meaningful worker model")
        check(sha(outputs[0]/"single_model_receipt.json")!=sha(outputs[1]/"single_model_receipt.json"),"External worker receipt did not retain changed authority provenance")
        trace=json.loads((outputs[0]/"training_trace.json").read_text())
        check(len(trace["adapter"])==50 and len(trace["heads"])==20 and trace["metadata"]["final_adapter_epoch"]==50 and trace["metadata"]["final_head_epoch"]==20,"Actual entry did not run fixed final epochs")
        return dict(actual_gated_worker_entry=True,temporary_float16_cache_and_label_authorities_only=True,
            model_fold=fold,model_seed=seed,held_out_source_count=len(held),train_source_count=len(allowed),
            heldout_label_mutation_changes_authority_preserves_all_functional_model_files=True,
            permitted_train_label_mutation_changes_checkpoint=True,actual_decoder_poison_guard=True,
            default_adapter_epochs=50,default_head_epochs=20,all_eight_heads_trained=True,
            checked_functional_file_count=len(members),synthetic_authority_overrides=["temporary source roles/folds/cache/authorities/protocol",
                "temporary independent gate status only; actual train_and_predict and default 50/20 config unmodified"])


COMPONENT_PINS={
    "modules":"2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3",
    "data_module":"54c57aee5cba1153ae18205a451b44eecc2fd6ae196d435b8e5f9c3fb8cdcbee",
    "trainer":"46be289030b99ef450d38cdc313f2a2d7ca3db97e0d0b5ac0a1101ff84e3d4c7",
    "coordinator":"b291c36aeb3ff92ae375954fb6033b35ecdba46c7a88728cc4b91c8048558133",
    "evaluator":"49c3393c44c5f8eb002fa61f28174293552af6eec859175740f3e4d75b858ae6",
    "verifier":"c81ab9fdb510ed50d81082a806da33e70a4212f49f50dc10a53539a3f65c351d",
    "finalizer":"3f424e3eb80657d481af26dabcc9cfff526589da1f09a2148a9c33cdde9e6766"}
INDEPENDENT_SCRIPT_PINS={
    "audit_identity_spacetime_oof_coordinator_20261024.py":"be8017542893aefbfa47f84e33b726ec848bbe495326046d90dfa6de0c5cc3eb",
    "audit_identity_spacetime_oof_background_20261024.py":"5ea2aad4cb861ad4fa0246afe4cbf629afd8cf395094698f219f5870c7def878",
    "qa_identity_spacetime_output_verifier_20261024.py":"da328ec01df64464b691a264848ebeba084b887ea470ebe2e654e690e823fd7c",
    "audit_identity_spacetime_oof_output_synthetic_20261024.py":"33464d0c2c7f0da9e6836f28fda5ae9513cfbceee3c858fb1701ab3e439a5aa0"}


def reviewed_receipts(spec):
    pins={};groups=(
        ("data_module","identity_spacetime_oof_data_synthetic_round%d_20261024.json","PASS_IDENTITY_SPACETIME_OOF_DATA_SYNTHETIC"),
        ("trainer","identity_spacetime_trainer_synthetic_round%d_20261024_v4.json","PASS_IDENTITY_SPACETIME_SINGLE_MODEL_TRAINER_SYNTHETIC"),
        ("coordinator","identity_spacetime_coordinator_synthetic_round%d_20261024_v2.json","PASS_IDENTITY_SPACETIME_OOF_COORDINATOR_SYNTHETIC"),
        ("evaluator","identity_spacetime_oof_evaluator_synthetic_round%d_20261024_v3.json","PASS_IDENTITY_SPACETIME_OOF_EVALUATOR_SYNTHETIC"),
        ("finalizer","identity_spacetime_oof_finalizer_synthetic_round%d_20261024_v2.json","PASS_IDENTITY_SPACETIME_OOF_FINALIZER_SYNTHETIC"))
    for component,pattern,status in groups:
        for number in (1,2,3):
            path=ROOT/"tmp_analysis"/(pattern%number);value=json.loads(path.read_text())
            check(value["status"]==status and value["round"]==number and value["script_sha256"]==COMPONENT_PINS[component],"Stale component synthetic receipt: "+str(path))
            pins[str(path)]=sha(path)
    path=ROOT/"tmp_analysis/identity_spacetime_oof_complete_output_synthetic_20261024_v2.json";value=json.loads(path.read_text())
    check(value["status"]=="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_COMPLETE_OUTPUT_SYNTHETIC" and value["actual_worker_processes"]==15 and value["actual_worker_exit_codes"]==[0]*15 and value["actual_evaluator_exit_code"]==0 and value["complete_audit_status"]=="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT","Complete temporary 15-worker audit absent")
    for key in ("trainer","coordinator","evaluator","verifier"):check(value[key+"_sha256"]==COMPONENT_PINS[key],"Full output synthetic receipt stale")
    check(value["data_sha256"]==DATA_SHA and value["script_sha256"]==INDEPENDENT_SCRIPT_PINS["audit_identity_spacetime_oof_output_synthetic_20261024.py"],"Full output synthetic data/checker stale")
    check(value["all_ten_scores_full_stream_and_independent_SQLite_all_reports_verified"] is True and value["tampered_score_independent_global_gate_rejected"] is True,"Full mathematical/statistical synthetic audit absent")
    pins[str(path)]=sha(path)
    benchmark=ROOT/"tmp_analysis/identity_spacetime_trainer_cpu_benchmark_20261024_v4.json";receipt=json.loads(benchmark.read_text());check(receipt["script_sha256"]==TRAINER_SHA and receipt["real_label_or_cache_opened"] is False,"CPU timing is stale or real-data-derived")
    pins[str(benchmark)]=sha(benchmark)
    return pins,value


def audit_implementation(protocol):
    import time
    started=time.monotonic();protocol=Path(protocol).resolve();spec=json.loads(protocol.read_text());original_protocol_sha=sha(protocol)
    check(spec["status"]=="FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL" and spec["schema"]=="IDENTITY_SPACETIME_SOURCE_OOF_V1","Formal protocol is not frozen")
    check(spec["folds"]==list(range(5)) and spec["seeds"]==[20261024,20261025,20261026] and spec["lags"]==[1,4],"Formal OOF dimensions differ")
    check(spec["feature_names"]==spec["score_columns"]==list(numerical.FEATURES),"Formal ten-column contract differs")
    for key,digest in COMPONENT_PINS.items():check(spec[key+"_sha256"]==digest and sha(spec[key+"_path"])==digest,"Formal component changed: "+key)
    for filename,digest in INDEPENDENT_SCRIPT_PINS.items():check(sha(ROOT/"tmp_analysis"/filename)==digest,"Independent checker changed: "+filename)
    receipt_pins,full_output_receipt=reviewed_receipts(spec)
    # All inputs in the following checks are generated temporary synthetic
    # authorities. The formal real-data protocol is read only as metadata.
    data_result=audit_data_only();data=module(DATA,"oof_final_independent_data");trainer=module(TRAINER,"oof_final_independent_trainer")
    check(trainer.CONFIG==spec["numeric_training_config"],"Formal numerical config differs")
    trainer.validate_real_config(spec)
    modules=trainer.module(data.MODULES,"oof_final_independent_modules",MODULES_SHA)
    _,_,trainer_result=audit_pure_trainer(data,trainer,modules)
    mutation=full_worker_mutation_test(data,trainer)
    seal_checker=module(ROOT/"tmp_analysis/audit_identity_spacetime_oof_coordinator_20261024.py","oof_final_independent_seal_challenges");seal_result=seal_checker.run()
    check(seal_result["status"]=="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_COORDINATOR_SEAL_SYNTHETIC" and seal_result["rejected_cases"]>=29,"Complete fifteen-model seal rejection matrix did not pass")
    background_checker=module(ROOT/"tmp_analysis/audit_identity_spacetime_oof_background_20261024.py","oof_final_independent_background");background_result=background_checker.audit()
    check(background_result["status"]=="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_BACKGROUND_PROCESS_AUDIT" and len(background_result["checks"])==8,"Actual detached supervisor process chain did not pass")
    qa_checker=module(ROOT/"tmp_analysis/qa_identity_spacetime_output_verifier_20261024.py","oof_final_independent_output_QA");qa_result=qa_checker.run()
    check(qa_result["status"]=="PASS_INDEPENDENT_OUTPUT_VERIFIER_GENERATED_DATA_QA" and qa_result["audited_code_sha256"]["verify_identity_spacetime_oof_output_20261024"]==COMPONENT_PINS["verifier"],"Independent full verifier QA bound stale bytes")
    finalizer=module(spec["finalizer_path"],"oof_final_finalizer_subject")
    finalizer_rounds=[finalizer.synthetic(number) for number in (1,2,3)]
    check(all(r["status"]=="PASS_IDENTITY_SPACETIME_OOF_FINALIZER_SYNTHETIC" and r["script_sha256"]==COMPONENT_PINS["finalizer"] for r in finalizer_rounds),"Finalizer binding/primary brief checks failed")
    for key,digest in COMPONENT_PINS.items():check(sha(spec[key+"_path"])==digest,"Formal component changed during preflight")
    for path,digest in receipt_pins.items():check(sha(path)==digest,"Reviewed receipt changed during preflight")
    check(sha(protocol)==original_protocol_sha,"Formal protocol changed during preflight")
    return dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_IMPLEMENTATION_PREFLIGHT",protocol_sha256=original_protocol_sha,
        trainer_sha256=COMPONENT_PINS["trainer"],data_sha256=DATA_SHA,modules_sha256=MODULES_SHA,coordinator_sha256=COMPONENT_PINS["coordinator"],evaluator_sha256=COMPONENT_PINS["evaluator"],verifier_sha256=COMPONENT_PINS["verifier"],finalizer_sha256=COMPONENT_PINS["finalizer"],checker_sha256=sha(__file__),
        reviewed_synthetic_receipt_files_sha256=receipt_pins,independent_checker_files_sha256={str(ROOT/"tmp_analysis"/k):v for k,v in INDEPENDENT_SCRIPT_PINS.items()},
        independent_checks=dict(data_boundary_and_projection=data_result,pure_trainer_losses_gradients_and_numerics=trainer_result,actual_default_worker_heldout_mutation_invariance=mutation,
            complete_fifteen_model_global_seal_negative_challenges=seal_result,actual_detached_process_success_and_failure_chain=background_result,
            independent_verifier_numerical_normalizer_all_statistical_QA=qa_result,actual_15_worker_complete_output_audit=full_output_receipt,finalizer_all_three_rounds=finalizer_rounds),
        CPU_only=True,GPU_used=False,real_training_or_prediction_launched=False,real_cache_or_identity_values_decoded=False,
        complete_OOF_preflight=True,elapsed_seconds=time.monotonic()-started,
        limitations="Implementation preflight only. Formal FIT execution and complete real-output independent audit remain required. Runtime planning is synthetic and descriptive; exploratory source OOF is not independent generalization or official tracking evidence.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only",action="store_true");parser.add_argument("--protocol",type=Path);parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.data_only:result=audit_data_only()
    else:
        check(args.protocol is not None,"Full independent preflight requires the frozen --protocol")
        result=audit_implementation(args.protocol)
    if args.out:
        with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
