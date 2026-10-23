#!/usr/bin/env python3
"""Ordinary identity adapter/head controls, pure modules and synthetic checks.

No data loader, runner, real labels/cache, optimizer, training loop or GPU use.
Caller must enforce source-OOF supervision boundaries and global output seals.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GEOMETRY_NAMES=("dx","dy","log_w_ratio","log_h_ratio","log1p_original_gap","ex","ey","CV_IoU","log1p_history_original_gap")
POSITION_SLOTS=(0,1,2,3,5,6,7)
TEMPERATURE=.1
CONTRACT_SHA="23a1361129af172e201e10c0fefa82cf3e76c78ce640d824329652785f40f173"


def require(condition,message):
    if not condition:raise ValueError(message)


def cpu_finite(value,last_dimension=None):
    require(isinstance(value,torch.Tensor) and value.device.type=="cpu" and value.dtype==torch.float64 and torch.isfinite(value).all(),"Expected finite CPU float64 tensor")
    if last_dimension is not None:require(value.shape[-1]==last_dimension,"Wrong feature dimension")


class IdentityAdapter256(nn.Module):
    """256->64->256 identity-initialized residual; zero raw stays exact zero."""
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed));self.first=nn.Linear(256,64,dtype=torch.float64);self.last=nn.Linear(64,256,dtype=torch.float64)
            nn.init.xavier_uniform_(self.first.weight);nn.init.zeros_(self.first.bias)
            nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,raw):
        cpu_finite(raw,256);require(raw.ndim==2,"Expected [observations,256]")
        unit=F.normalize(raw,dim=-1,eps=1e-12)
        output=F.normalize(unit+self.last(F.gelu(self.first(unit))),dim=-1,eps=1e-12)
        return output*(torch.linalg.vector_norm(raw,dim=-1,keepdim=True)>0).to(raw.dtype)


def freeze_adapter(adapter):
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False);p.grad=None
    return adapter


def frozen_appearance_cosine(query_raw,reference_raw,adapter=None):
    """Same frozen adapter object is reused across every G condition."""
    cpu_finite(query_raw,256);cpu_finite(reference_raw,256)
    if adapter is not None:require(not any(p.requires_grad for p in adapter.parameters()),"Freeze the single identity-trained adapter before every geometry arm")
    with torch.no_grad():
        query=F.normalize(query_raw,dim=-1,eps=1e-12) if adapter is None else adapter(query_raw)
        reference=F.normalize(reference_raw,dim=-1,eps=1e-12) if adapter is None else adapter(reference_raw)
        return query@reference.T


class ResidualAssociationHead19(nn.Module):
    """Fixed ordinary 19->32->1 residual head with zero initial residual."""
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed));self.first=nn.Linear(19,32,dtype=torch.float64);self.last=nn.Linear(32,1,dtype=torch.float64)
            nn.init.xavier_uniform_(self.first.weight);nn.init.zeros_(self.first.bias)
            nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,inputs):
        cpu_finite(inputs,19)
        return inputs[...,0]/TEMPERATURE+self.last(F.gelu(self.first(inputs))).squeeze(-1)


def geometry_slots(geometry,reference_times,previous_times):
    """Assemble nine raw values from reviewed geometry output; no camera warp."""
    require(geometry["time_unit"]=="original_frames" and not geometry["camera_warp_supplied"],"Use original frames with no ego compensation")
    displacement=np.asarray(geometry["normalized_displacement"],np.float64);nq,nm,_=displacement.shape
    size=np.asarray(geometry["log_size_change"],np.float64);innovation=np.asarray(geometry["normalized_CV_innovation"],np.float64)
    iou=np.asarray(geometry["predicted_box_iou"],np.float64);history=np.asarray(geometry["velocity_history_available"],bool)
    gap=np.asarray(geometry["frame_gap"],np.float64);now=np.asarray(reference_times,np.float64);before=np.asarray(previous_times,np.float64)
    require(displacement.shape==size.shape==innovation.shape==(nq,nm,2) and iou.shape==(nq,nm) and history.shape==gap.shape==now.shape==before.shape==(nm,),"Geometry shape mismatch")
    require(np.isfinite(displacement).all() and np.isfinite(size).all() and np.isfinite(gap).all() and np.all(gap>0),"Invalid base geometry")
    require(np.array_equal(history,np.isfinite(before)) and np.all(before[history]<now[history]),"History geometry and timestamps disagree")
    require(np.isfinite(innovation[:,history]).all() and np.isnan(innovation[:,~history]).all() and np.isfinite(iou).all(),"Invalid CV geometry mask")
    values=np.full((nq,nm,9),np.nan);masks=np.zeros((nq,nm,9),bool)
    values[:,:,:2]=displacement;values[:,:,2:4]=size;values[:,:,4]=np.log1p(gap)[None,:]
    masks[:,:,:5]=True
    values[:,:,5:7]=innovation;values[:,history,7]=iou[:,history]
    values[:,history,8]=np.log1p(now[history]-before[history])[None,:];masks[:,:,5:]=history[None,:,None]
    require(np.array_equal(np.isfinite(values),masks),"Value/mask mismatch")
    return values,masks


def training_projection(query_gt,reference_gt,query_predicted_category,reference_predicted_categories):
    """Project legal identities for loss only; retained candidates stay complete."""
    gt=np.asarray(reference_gt,np.int64);categories=np.asarray(reference_predicted_categories)
    require(gt.ndim==1 and categories.shape==gt.shape,"Reference identity/category shape mismatch")
    allowed=(gt>=0)&(categories==query_predicted_category)&(query_gt>=0)
    positive=np.flatnonzero(allowed&(gt==query_gt));negative=np.flatnonzero(allowed&(gt!=query_gt))
    return dict(positive_rows=positive,negative_rows=negative,hard=bool(len(positive) and len(negative)),
        unknown_rows=np.flatnonzero(gt<0),no_positive_is_only_missing_coverage=not len(positive))


def pairwise_loss(scores,positive_rows,negative_rows,temperature=1.):
    """All legal P x N comparisons; no hard mining or unknown/dustbin labels."""
    cpu_finite(scores);require(scores.ndim==1 and temperature>0,"Expected one retained query score vector")
    p=np.asarray(positive_rows,np.int64);n=np.asarray(negative_rows,np.int64)
    require(len(p)>0 and len(n)>0 and len(set(p.tolist()+n.tolist()))==len(p)+len(n) and min(p.min(),n.min())>=0 and max(p.max(),n.max())<len(scores),"Need disjoint legal positive and negative rows")
    return F.softplus((scores[torch.as_tensor(n)][:,None]-scores[torch.as_tensor(p)][None,:])/temperature).mean()


def adapter_pairwise_loss(adapter,query_raw,reference_raw,projection):
    require(projection["hard"],"Nonhard query has coverage only; never a null training target")
    query=adapter(query_raw);reference=adapter(reference_raw)
    require(len(query)==1,"One query observation per query-lag loss")
    return pairwise_loss((query@reference.T)[0],projection["positive_rows"],projection["negative_rows"],TEMPERATURE)


def hierarchical_weights(records,all_sources,train_sources):
    """Equal source -> query GT trajectory -> observation -> available lag."""
    all_sources=tuple(all_sources);train=set(train_sources);require(train<=set(all_sources),"Unknown train source")
    tree=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)));weights=np.zeros(len(records),np.float64)
    for i,row in enumerate(records):
        require(row["source"] in train and row["lag"] in (1,4),"Held-out source or undeclared lag entered training")
        if row["qgt"]<0 or not len(row["positive_rows"]) or not len(row["negative_rows"]):continue
        trajectory=(row["video_id"],row["qgt"]);observation=(row["image_id"],row["qrow"])
        group=tree[row["source"]][trajectory][observation]
        require(row["lag"] not in group,"Duplicate query-lag must not multiply support")
        group[row["lag"]]=i
    for trajectories in tree.values():
        for observations in trajectories.values():
            for lags in observations.values():
                weight=1/(len(tree)*len(trajectories)*len(observations)*len(lags))
                for i in lags.values():weights[i]=weight
    require(not len(tree) or abs(float(weights.sum())-1)<1e-12,"Global training weights do not sum to one")
    return weights,dict(supported_sources=sorted(tree),zero_support_train_sources=sorted(train-set(tree)),
        all_source_coverage={s:sum(weights[i]>0 for i,r in enumerate(records) if r["source"]==s) for s in all_sources},
        training_query_lags=int(np.count_nonzero(weights)))


def weighted_batch_loss(losses,weights,total_query_lags):
    """N/B global weighted sum, including final short batch; never renormalize."""
    cpu_finite(losses);require(losses.ndim==1 and len(losses)>0 and total_query_lags>=len(losses),"Invalid batch population")
    weight=torch.as_tensor(weights,dtype=losses.dtype);cpu_finite(weight)
    require(weight.shape==losses.shape and torch.all(weight>=0),"Invalid fixed global weights")
    return total_query_lags/len(losses)*torch.sum(weight*losses)


def edge_marginal_weights(rows,positive_rows,negative_rows,query_lag_weight):
    p=np.asarray(positive_rows,np.int64);n=np.asarray(negative_rows,np.int64)
    require(len(p)>0 and len(n)>0 and query_lag_weight>0 and len(set(p.tolist()+n.tolist()))==len(p)+len(n),"Need hard known subset")
    require(min(p.min(),n.min())>=0 and max(p.max(),n.max())<rows,"Marginal row out of range")
    result=np.zeros(rows,np.float64);result[p]=query_lag_weight/(2*len(p));result[n]=query_lag_weight/(2*len(n))
    return result


@dataclass(frozen=True)
class GeometryNormalizer:
    mean: np.ndarray
    std: np.ndarray
    valid_marginal_weight: np.ndarray
    def transform(self,values,masks):
        values=np.asarray(values,np.float64);masks=np.asarray(masks,bool)
        require(values.shape==masks.shape and values.shape[-1]==9 and np.isfinite(values[masks]).all(),"Invalid numerical geometry")
        output=np.zeros_like(values);np.subtract(values,self.mean,out=output,where=masks);np.divide(output,self.std,out=output,where=masks)
        return output


def fit_geometry_normalizer(blocks,train_sources):
    """Two streaming passes over ORIGINAL un-intervened geometry only.

    blocks is a re-iterable collection or a callable returning a fresh iterator.
    Each block is one hard query-lag's complete [M,9] values/masks, source,
    positive_rows, negative_rows and global hierarchical query_lag_weight.
    """
    require(callable(blocks) or iter(blocks) is not blocks,"Supply a repeatable stream for two passes")
    factory=blocks if callable(blocks) else lambda:iter(blocks);train=set(train_sources)
    sums=np.zeros(9);mass=np.zeros(9);total_weight=0.;first_digest=hashlib.sha256();second_digest=hashlib.sha256()
    def bind_block(digest,block,values,mask):
        p=np.asarray(block["positive_rows"],np.int64);n=np.asarray(block["negative_rows"],np.int64)
        header=dict(source=block["source"],shape=list(values.shape),positive_rows=len(p),negative_rows=len(n))
        digest.update(json.dumps(header,sort_keys=True,separators=(",",":")).encode())
        for array in (values,mask,p,n,np.asarray([block["query_lag_weight"]],np.float64)):
            digest.update(np.ascontiguousarray(array).tobytes())
    def unpack(block):
        require(block["source"] in train,"Held-out source entered normalizer")
        values=np.asarray(block["values"],np.float64);mask=np.asarray(block["masks"],bool)
        require(values.shape==mask.shape and values.ndim==2 and values.shape[1]==9 and np.isfinite(values[mask]).all(),"Invalid original geometry block")
        weight=edge_marginal_weights(len(values),block["positive_rows"],block["negative_rows"],block["query_lag_weight"])
        return values,mask,weight
    for block in factory():
        values,mask,weight=unpack(block);weighted=weight[:,None]*mask
        bind_block(first_digest,block,values,mask)
        sums+=(np.where(mask,values,0)*weighted).sum(0);mass+=weighted.sum(0);total_weight+=block["query_lag_weight"]
    require(total_weight==0 or abs(total_weight-1)<1e-12,"Normalizer requires complete globally weighted training population")
    mean=np.divide(sums,mass,out=np.zeros(9),where=mass>0);variance=np.zeros(9)
    second_mass=np.zeros(9);second_weight=0.
    for block in factory():
        values,mask,weight=unpack(block);weighted=weight[:,None]*mask
        bind_block(second_digest,block,values,mask)
        variance+=(np.where(mask,values-mean,0)**2*weighted).sum(0);second_mass+=weighted.sum(0);second_weight+=block["query_lag_weight"]
    require(first_digest.digest()==second_digest.digest() and np.array_equal(mass,second_mass) and total_weight==second_weight,"Normalizer stream changed between passes")
    std=np.sqrt(np.divide(variance,mass,out=np.ones(9),where=mass>0));std=np.where(mass>0,np.maximum(std,1e-6),1.)
    for array in (mean,std,mass):array.setflags(write=False)
    return GeometryNormalizer(mean,std,mass)


@dataclass(frozen=True)
class GPermPlan:
    donor_rows: np.ndarray
    dump_rows: np.ndarray
    categories: np.ndarray
    history_valid: np.ndarray


def gperm_donors(predicted_categories,history_valid,immutable_dump_rows):
    """Complete retained reference set only; no labels/known mask accepted."""
    categories=np.array(predicted_categories,copy=True);history=np.array(history_valid,bool,copy=True);rows=np.array(immutable_dump_rows,np.int64,copy=True)
    require(categories.ndim==1 and categories.shape==history.shape==rows.shape and len(set(rows.tolist()))==len(rows),"Invalid full reference metadata")
    donors=np.arange(len(rows));groups=defaultdict(list)
    for i in range(len(rows)):groups[(int(categories[i]),bool(history[i]))].append(i)
    for indices in groups.values():
        indices=np.array(sorted(indices,key=lambda i:int(rows[i])),np.int64)
        if len(indices)>1:donors[indices]=np.roll(indices,1)
    for array in (donors,rows,categories,history):array.setflags(write=False)
    return GPermPlan(donors,rows,categories,history)


def intervene_geometry(normalized_values,masks,mode,plan=None):
    """Normalize/fill first; then fixed intervention; own time and masks stay."""
    values=np.asarray(normalized_values,np.float64);mask=np.asarray(masks,bool)
    require(values.shape==mask.shape and values.ndim==3 and values.shape[-1]==9 and np.isfinite(values).all(),"Expected normalized complete QxMx9")
    out=values.copy();valid=mask.copy();coverage=dict(mapping_nonidentity_references=0,mapping_nonidentity_candidates=0,geometry_values_changed_candidates=0)
    if mode=="G0":out[:]=0;valid[:]=False
    elif mode=="Gmask":out[:,:,POSITION_SLOTS]=0
    elif mode=="Gperm":
        require(isinstance(plan,GPermPlan) and len(plan.donor_rows)==values.shape[1],"Build donors on the full retained reference set before any supervision projection")
        donors=plan.donor_rows
        require(np.array_equal(np.sort(donors),np.arange(len(donors))) and np.array_equal(plan.categories[donors],plan.categories) and np.array_equal(plan.history_valid[donors],plan.history_valid),"Gperm crosses GT-free metadata group")
        out[:,:,POSITION_SLOTS]=values[:,donors,:][:,:,POSITION_SLOTS]
        moved=int(np.count_nonzero(donors!=np.arange(len(donors))))
        changed=int(np.count_nonzero(np.any(out[:,:,POSITION_SLOTS]!=values[:,:,POSITION_SLOTS],axis=2)))
        coverage=dict(mapping_nonidentity_references=moved,mapping_nonidentity_candidates=moved*len(values),geometry_values_changed_candidates=changed)
    else:require(mode=="G1","Only fixed G0/G1/Gmask/Gperm controls")
    return out,valid,coverage


def head_inputs(cosine,geometry_values,masks,normalizer,mode,plan=None):
    cpu_finite(cosine);require(not cosine.requires_grad,"Appearance must be frozen before geometry head training")
    normalized=normalizer.transform(geometry_values,masks)
    values,valid,coverage=intervene_geometry(normalized,masks,mode,plan)
    require(tuple(cosine.shape)==values.shape[:2],"Appearance and retained QxM geometry differ")
    inputs=torch.cat((cosine[:,:,None],torch.from_numpy(values),torch.from_numpy(valid.astype(np.float64))),dim=2)
    return inputs,coverage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_close(actual,expected,message):
    require(np.allclose(np.asarray(actual),np.asarray(expected),atol=1e-12,rtol=1e-12),message)


def synthetic(number):
    torch.set_num_threads(1);checks=[]
    if number==1:
        adapter=IdentityAdapter256(20261024);raw=torch.zeros((3,256),dtype=torch.float64);raw[0,0]=2;raw[1,1]=3
        check_close(adapter(raw).detach().numpy(),F.normalize(raw,dim=-1).numpy(),"Adapter identity initialization differs")
        with torch.no_grad():adapter.last.bias.fill_(3.)
        require(torch.equal(adapter(raw)[2],torch.zeros(256,dtype=torch.float64)),"Learnable bias polluted zero input")
        checks.append("identity_initialization_and_explicit_zero_mask_despite_learned_bias")
        adapter(raw).sum().backward();require(any(p.grad is not None for p in adapter.parameters()),"Synthetic did not create stale identity gradients")
        freeze_adapter(adapter);before={k:v.clone() for k,v in adapter.state_dict().items()};s=frozen_appearance_cosine(raw,raw,adapter)
        require(all(p.grad is None for p in adapter.parameters()),"Freeze retained stale adapter gradients")
        norm=GeometryNormalizer(np.zeros(9),np.ones(9),np.ones(9));geometry=np.zeros((3,3,9));mask=np.ones_like(geometry,bool)
        for mode in ("G0","G1","Gmask","Gperm"):
            plan=gperm_donors([1,1,1],[True]*3,[0,1,2]) if mode=="Gperm" else None
            inputs,_=head_inputs(s,geometry,mask,norm,mode,plan);head=ResidualAssociationHead19(20261024)
            check_close(head(inputs).detach().numpy(),s.numpy()/.1,"Head zero residual initialization differs")
            head(inputs).sum().backward()
        require(all(p.grad is None and not p.requires_grad for p in adapter.parameters()) and all(torch.equal(v,before[k]) for k,v in adapter.state_dict().items()),"Geometry arm changed frozen shared adapter")
        checks.append("nineteen_slot_head_identity_initialization_same_frozen_adapter_all_G_no_gradient_leak")
        projection=training_projection(7,[7,8,-1,9],1,[1,1,1,2]);scores=torch.tensor([.8,.3,999.,999.],dtype=torch.float64,requires_grad=True)
        loss=pairwise_loss(scores,projection["positive_rows"],projection["negative_rows"],.1);loss.backward()
        require(scores.grad[2]==scores.grad[3]==0 and projection["hard"] and projection["unknown_rows"].tolist()==[2],"Unknown/other-category became a training negative")
        require(not training_projection(7,[-1,8],1,[1,1])["hard"],"Missing positive made a null target")
        checks.append("complete_PxN_only_known_same_cached_category_unknown_has_no_gradient_or_null_label")
    elif number==2:
        def record(source,gt,image,lag,p=1,n=1):return dict(source=source,video_id=1 if source=="S" else 2,qgt=gt,image_id=image,qrow=0,lag=lag,positive_rows=np.arange(p),negative_rows=np.arange(p,p+n))
        records=[record("S",1,1,1),record("S",1,1,4),record("S",1,2,1),record("S",2,3,1),record("T",3,4,1),record("ZERO",-1,5,1,p=0)]
        weights,coverage=hierarchical_weights(records,["S","T","ZERO","HELD"],["S","T","ZERO"])
        check_close(weights,[1/16,1/16,1/8,1/4,1/2,0],"Source trajectory observation lag weights differ")
        require(coverage["zero_support_train_sources"]==["ZERO"] and coverage["all_source_coverage"]["HELD"]==0,"Zero-support source disappeared")
        check_close(edge_marginal_weights(6,[0],[1,2,3],.5),[.25,1/12,1/12,1/12,0,0],"P/N marginal weights differ")
        check_close(weighted_batch_loss(torch.tensor([2.,3.],dtype=torch.float64),[.1,.2],5).item(),2.,"N/B global batch scaling differs")
        checks.append("exact_source_GTtrajectory_observation_availablelag_weights_PN_marginals_and_short_batch_scaling")
        values=np.tile(np.array([0.,10.,20.,30.,999.])[:,None],(1,9));mask=np.ones((5,9),bool);mask[:,8]=False;values[:,8]=np.nan
        mask[1:4,7]=False;values[1:4,7]=np.nan
        blocks=[dict(source="S",values=values,masks=mask,positive_rows=np.array([0]),negative_rows=np.array([1,2,3]),query_lag_weight=1.)]
        norm=fit_geometry_normalizer(blocks,["S"])
        check_close(norm.mean[:7],np.full(7,10.),"Normalizer used equal known-edge weights instead of P/N marginals")
        check_close(norm.std[:7],np.full(7,np.sqrt(400/3)),"Weighted population std differs")
        require(norm.mean[7]==0 and norm.std[7]==1e-6 and norm.mean[8]==0 and norm.std[8]==1 and norm.valid_marginal_weight[7]==.5 and norm.valid_marginal_weight[8]==0,"Per-dimension valid renormalization/zero-valid constants differ")
        transformed=norm.transform(values,mask);require(np.equal(transformed[~mask],0).all(),"Invalid geometry not filled zero")
        checks.append("train_original_geometry_exact_PN_valid_marginals_population_std_floor_and_zero_valid")
        try:fit_geometry_normalizer(blocks,["T"])
        except ValueError:pass
        else:raise AssertionError("Held-out source entered normalizer")
        checks.append("held_out_source_rejected_by_normalizer")
        calls=[0]
        def drifting_stream():
            calls[0]+=1;modified=dict(blocks[0]);modified["values"]=values.copy();modified["values"][0,0]+=calls[0]
            return iter([modified])
        try:fit_geometry_normalizer(drifting_stream,["S"])
        except ValueError:pass
        else:raise AssertionError("Changed second-pass values silently altered std")
        checks.append("normalizer_two_pass_values_masks_source_PN_weights_digest_blocks_silent_stream_drift")
    elif number==3:
        # Unknown identities never enter donor API; row1 would be unknown in loss.
        plan=gperm_donors([1,1,1,2,2],[True,True,True,False,True],[20,10,30,40,50])
        require(plan.donor_rows.tolist()==[1,2,0,3,4],"Full retained immutable-row cyclic donors differ")
        values=np.arange(2*5*9,dtype=float).reshape(2,5,9);mask=np.ones_like(values,bool);values[:,:,4]=8;values[:,:,8]=9
        perm,valid,coverage=intervene_geometry(values,mask,"Gperm",plan)
        require(np.array_equal(valid,mask) and np.array_equal(perm[:,:,[4,8]],values[:,:,[4,8]]) and coverage["mapping_nonidentity_references"]==3 and coverage["geometry_values_changed_candidates"]==6,"Gperm changed own time/masks or wrong coverage")
        require(perm[0,0,0]==values[0,1,0],"Donor was projected to known rows before permutation")
        static=np.ones_like(values);_,_,same=intervene_geometry(static,mask,"Gperm",plan)
        require(same["mapping_nonidentity_candidates"]==6 and same["geometry_values_changed_candidates"]==0,"Mapping movement falsely implies geometry value change")
        checks.append("GTfree_fullretained_donors_before_knownprojection_own_time_masks_singletons_and_distinct_coverage")
        norm=GeometryNormalizer(np.ones(9),np.full(9,2.),np.ones(9));s=torch.zeros((2,5),dtype=torch.float64)
        gmask,_=head_inputs(s,values,mask,norm,"Gmask");gzero,_=head_inputs(s,values,mask,norm,"G0")
        require(torch.equal(gzero[:,:,1:],torch.zeros_like(gzero[:,:,1:])) and torch.equal(gmask[:,:,1+np.array(POSITION_SLOTS)],torch.zeros_like(gmask[:,:,1+np.array(POSITION_SLOTS)])),"G0/Gmask slots not fixed zero")
        check_close(gmask[:,:,5].numpy(),(values[:,:,4]-1)/2,"Gmask failed normalize-first time preservation")
        require(torch.equal(gmask[:,:,10:],torch.ones_like(gmask[:,:,10:])),"Gmask discarded masks")
        checks.append("shared_normalizer_before_intervention_G0_allzero_Gmask_time_and_masks_only")
        geometry=dict(time_unit="original_frames",camera_warp_supplied=False,normalized_displacement=np.zeros((1,2,2)),log_size_change=np.zeros((1,2,2)),normalized_CV_innovation=np.array([[[0.,0.],[np.nan,np.nan]]]),predicted_box_iou=np.array([[1.,.8]]),velocity_history_available=np.array([True,False]),frame_gap=np.array([25.,25.]))
        value,valid=geometry_slots(geometry,[35,35],[10,np.nan])
        require(valid[0,0].all() and valid[0,1,:5].all() and not valid[0,1,5:].any() and np.isnan(value[0,1,7]),"Static IoU falsely became CV IoU or missing history became valid")
        check_close(value[0,0,[4,8]],np.log1p([25,25]),"Original-frame history gap differs")
        checks.append("geometry_ninevalues_ninemasks_original_time_static_overlap_never_CV_IoU")
    else:raise ValueError("Three synthetic rounds only")
    return dict(status="PASS_IDENTITY_SPACETIME_MODULES_SYNTHETIC",round=number,checks=checks,script_sha256=sha(__file__),contract_sha256=CONTRACT_SHA,
        real_cache_or_identity_opened=False,training_runner_implemented=False,optimizer_steps=0,GPU_used=False,ordinary_controls_not_novel_method=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--synthetic-round",type=int,choices=(1,2,3),required=True);parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    result=synthetic(args.synthetic_round)
    if args.synthetic_out:
        with args.synthetic_out.open("x") as f:f.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
