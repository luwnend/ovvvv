#!/usr/bin/env python3
"""Independent generated-data QA of the final output auditor contracts."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def check(value,message):
    if not value:raise AssertionError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name):
    path=ROOT/"tmp_analysis"/(name+".py");spec=importlib.util.spec_from_file_location("qa_"+name,path);result=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=result;spec.loader.exec_module(result);return result


def frame(rng,video,step,n=5):
    size=rng.uniform(1,40,(n,2));corner=rng.uniform(-40,100,(n,2));raw=rng.randn(n,256)
    if n:raw[-1]=0
    return dict(video_id=video,image_id=video*100+step,sampled_step=step,original_frame_index=(0,5,9,31,50,80)[step],
        boxes=np.c_[corner,corner+size],raw=raw,released_ids=np.arange(n),predicted_categories=np.ones(n,int))


def numerical(v,t,m,geometry):
    rng=np.random.RandomState(9073);adapter=m.IdentityAdapter256(20261024);heads={arm:m.ResidualAssociationHead19(20261024) for arm in t.ARMS}
    generator=torch.Generator().manual_seed(17744)
    for model in [adapter]+list(heads.values()):
        with torch.no_grad():
            for parameter in model.parameters():parameter.copy_(.1*torch.randn(parameter.shape,dtype=torch.float64,generator=generator))
    m.freeze_adapter(adapter);norm=m.GeometryNormalizer(rng.randn(9),rng.uniform(.1,2,9),np.ones(9));model=dict(adapter=adapter,heads=heads,normalizer=norm)
    states={arm:t.tensor_state(head) for arm,head in heads.items()};maximum=0.;cases=0
    for nq,nr in ((0,0),(0,5),(4,0),(1,1),(3,4),(7,9)):
        for repeat in range(3):
            q=frame(rng,1,5,nq);r=frame(rng,1,2,nr);history=(np.arange(nr)%2==0)
            r.update(previous_times=np.where(history,0.,np.nan),previous_boxes=np.where(history[:,None],r["boxes"]-2,np.nan),predicted_categories=np.arange(nr)%3+1)
            reviewed=geometry.pair_geometry(q["boxes"],r["boxes"],q["original_frame_index"],np.full(nr,r["original_frame_index"]),r["previous_boxes"],r["previous_times"])
            values,masks=m.geometry_slots(reviewed,np.full(nr,r["original_frame_index"]),r["previous_times"])
            actual,coverage=t.score_frame_pair(dict(query=q,reference=r,geometry_values=values,geometry_masks=masks),model,m)
            expected=v.independent_model_scores(q,r,dict(mean=norm.mean,std=norm.std),t.tensor_state(adapter),states)
            check(actual.shape==expected.shape==(nq,nr,10),"Empty/nonempty numerical shape differs")
            np.testing.assert_array_equal(actual[:,:,0],expected[:,:,0]);np.testing.assert_allclose(actual[:,:,1:],expected[:,:,1:],atol=1e-10,rtol=1e-10)
            if nq and nr:maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            if nq:check(np.array_equal(v.adapter_forward(q["raw"],t.tensor_state(adapter))[-1],np.zeros(256)),"Zero raw polluted by learned nonzero biases")
            donor=v.independent_donors(r["predicted_categories"],history);check(np.array_equal(r["predicted_categories"][donor],r["predicted_categories"]) and np.array_equal(history[donor],history),"Donor crosses metadata group")
            check(coverage["mapping_nonidentity_candidates"]==nq*int(np.count_nonzero(donor!=np.arange(nr))),"Zero-query donor coverage changed")
            original=v.independent_inputs(np.zeros((nq,nr)),values,masks,norm.mean,norm.std,"G1",donor)
            permuted=v.independent_inputs(np.zeros((nq,nr)),values,masks,norm.mean,norm.std,"Gperm",donor)
            np.testing.assert_array_equal(original[:,:,[5,9]],permuted[:,:,[5,9]])
            np.testing.assert_array_equal(original[:,:,10:],permuted[:,:,10:]);cases+=1
    return dict(randomized_forward_blocks=cases,maximum_absolute_error=maximum,raw_bitwise_exact=True,zero_query_zero_reference_zero_raw_nonzero_checkpoint=True,Gperm_category_history_time_and_mask_preservation=True)


def normalization(v,t,m,data,history):
    rng=np.random.RandomState(123);sources=["S%d"%i for i in range(5)];grouped={};records=[];videos={};anchors={}
    for video,source in enumerate(sources,1):
        videos[str(video)]=source;rows=[]
        for step in range(6):
            f=frame(rng,video,step,0 if step==1 else 5);n=len(f["raw"])
            f["predicted_categories"]=np.array([1,1,1,1,2][:n]);f["released_ids"]=np.arange(n)
            row=dict(video_id=video,image_id=f["image_id"],frame_id=step,frame_index=f["original_frame_index"],observations=n,fixture=f);rows.append(row);records.append(row)
            if source!="S4":
                for qr,gt in enumerate([0,0,1,-1,2][:n]):
                    if gt>=0:anchors[f["image_id"],qr]=(video,gt)
        grouped[video]=rows
    manifest=dict(folds={str(f):dict(train_sources=[s for i,s in enumerate(sources) if i!=f],held_out_sources=[sources[f]]) for f in range(5)})
    utility=SimpleNamespace(read_raw=lambda row,state:history.independent_history(copy.deepcopy(row["fixture"]),state))
    designs=v.independent_train_design({},manifest,videos,records,grouped,anchors,utility)
    for fold in range(5):
        allowed=manifest["folds"][str(fold)]["train_sources"];selected=[video for video in grouped if videos[str(video)] in allowed]
        pairs=data.iter_gtfree_frame_pairs(selected,frame_records=records,video_sources=videos,loader=lambda row:copy.deepcopy(row["fixture"]))
        compact=t.compact_examples(data.build_train_examples(pairs,anchors,allowed),allowed,m);weights,_=m.hierarchical_weights([e["record"] for e in compact],sources,allowed)
        blocks=[dict(source=e["record"]["source"],values=e["values"],masks=e["masks"],positive_rows=e["record"]["positive_rows"],negative_rows=e["record"]["negative_rows"],query_lag_weight=float(w)) for e,w in zip(compact,weights)]
        actual=m.fit_geometry_normalizer(blocks,allowed);expected=designs[fold]
        check(len(compact)==expected["n"],"Independent hard training population differs")
        for key in ("mean","std","valid_marginal_weight"):np.testing.assert_allclose(getattr(actual,key),expected[key],atol=1e-11,rtol=1e-11)
    return dict(folds=5,hard_population_and_P2_N1_edge_marginal_geometry_moments_match=True,unknown_reference_excluded_from_normalizer=True,zero_support_source_and_empty_sampled_frame_retained=True)


def statistics(v,e):
    rng=np.random.RandomState(1042);sources=["A","B","ZERO"];seeds=(20261024,20261025,20261026);roles={"A":dict(role="mean_role",block="X"),"B":dict(role="pair_role",block="Y"),"ZERO":dict(role="pair_role",block="Y")}
    with tempfile.TemporaryDirectory(prefix="output_verifier_stat_QA_") as td:
        edb=e.open_database(Path(td)/"subject.sqlite");vdb=v.open_db(Path(td)/"oracle.sqlite")
        ecells={(s,source,lag,scope,pop):e.fresh_cell() for s in seeds for source in sources for lag in (1,4) for scope in e.SCOPES for pop in e.POPS};vcells={key:v.fresh() for key in ecells}
        for seed in seeds:
            for video,source in enumerate(sources[:2],1):
                for step in range(7):
                    nq=0 if step==0 else 5;qi=np.array([-1,0,0,1,2][:nq]);image=video*100+step
                    for lag in (1,4):
                        nr=0 if step==1 else 6;ri=np.array([0,0,1,-1,2,1][:nr]);score=rng.randint(-2,3,(nq,nr,10)).astype(float);same=rng.rand(nq,nr)>.2;cv=rng.rand(nr)>.4;changed=rng.rand(nq,nr)>.6;moved=rng.rand(nq,nr)>.6
                        for scope,selected in ((e.SCOPES[0],np.ones((nq,nr),bool)),(e.SCOPES[1],same)):
                            for pop,mask in ((e.POPS[0],selected),(e.POPS[1],selected&cv[None,:])):
                                key=(seed,source,lag,scope,pop);e.update_cell(ecells[key],score,mask,qi,ri,source,seed,lag,scope,pop,video,image,edb,changed,moved)
                                v.add_block(vcells[key],score,mask,qi,ri,changed,moved,(seed,source,lag,scope,pop,video,image),vdb)
        actual=e.summarize(edb,ecells,roles,seeds);actual["seed_robustness"]=e.seed_robustness(actual,seeds);expected=v.reports(vdb,vcells,roles,seeds)
        v.compare(actual,expected,"Randomized complete statistic report");edb.commit();vdb.commit();n=v.compare_databases(vdb,Path(td)/"subject.sqlite");edb.close();vdb.close()
    return dict(independently_rebuilt_query_SQLite_rows=n,tied_scores_multi_positive_unknown_and_empty_population=True,full_source_role_block_union_macro_paired_2x2_crosslag_source_deletion_and_seed_statistics_match=True)


def run():
    names=("verify_identity_spacetime_oof_output_20261024","evaluate_identity_spacetime_oof_20261024","train_identity_spacetime_oof_model_20261024","identity_spacetime_modules_20261024","identity_spacetime_oof_data_20261024","relative_track_geometry","verify_angular_appearance_control_output_20261024")
    paths={name:ROOT/"tmp_analysis"/(name+".py") for name in names};pins={name:sha(path) for name,path in paths.items()}
    modules={name:module(name) for name in names};v,e,t,m,d,g,h=[modules[name] for name in names];torch.set_num_threads(1)
    result=dict(status="PASS_INDEPENDENT_OUTPUT_VERIFIER_GENERATED_DATA_QA",audited_code_sha256=pins,script_sha256=sha(__file__),numerical=numerical(v,t,m,g),normalization=normalization(v,t,m,d,h),statistics=statistics(v,e),
        real_cache_or_identity_opened=False,real_training_executed=False,limits="Generated-data formula/statistic QA; full actual-worker/output-chain synthetic audit and real-output audit remain separate evidence.")
    check(pins=={name:sha(path) for name,path in paths.items()},"Audited implementation changed during QA")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path);args=parser.parse_args();result=run()
    if args.out:
        with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
