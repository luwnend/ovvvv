#!/usr/bin/env python3
"""Isolated inference prototypes for COVTrack feature-memory consistency.

Primary mode raw_confidence preserves the released fused scoring/EMA bank and
changes only appearance cycle-confidence inputs to a separate raw EMA bank.
All five fixed modes must be reported; this is an engineering hypothesis, not
an efficacy or novelty claim. No checkpoint, real cache or GT is opened here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
WORK_COPY=ROOT/"work/covtrack_base_error_20261023"
SOURCE_PINS={
    "ovtrack/models/trackers/ovtracker.py":"0eaa4c96a09058f6f1b6ef73b1f860fb16aa0b6542ef43d0dc238de95a8f6db5",
    "ovtrack/models/losses/cyc_loss.py":"0074cfee8db66dddb45fe59f80cb7e0e6ee22f5bf51f70e38eb052c9a8aeee8e",
    "ovtrack/models/roi_heads/ovtrack_roi_head.py":"955d03011368fba48efde974b73bc4224e63a0d3add1028c17b97da54a2940d7",
    "ovtrack/core/track/similarity.py":"c22848edcc2511f3b608ff8888f776e5d3bd2aef7d6ee6571c8739d7136a00a7",
}
MODES=("released","raw_confidence","raw_only","raw_store_one_sided","raw_store_symmetric")
PRIMARY_MODE="raw_confidence"


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


for _relative,_digest in SOURCE_PINS.items():
    require(sha(WORK_COPY/_relative)==_digest,"Frozen scientific source changed: "+_relative)
if "ovtrack" not in sys.modules:sys.path.insert(0,str(WORK_COPY))
import ovtrack
require(Path(ovtrack.__file__).resolve()==WORK_COPY/"ovtrack/__init__.py","Import the independently frozen scientific work copy")
import torch
from ovtrack.models.builder import TRACKERS
from ovtrack.models.trackers.ovtracker import OVTrackerUncertainty
from ovtrack.core import cal_similarity


@TRACKERS.register_module()
class OVTrackerContextFeatureMemory(OVTrackerUncertainty):
    """Unmodified detector/category readout; explicit ovtrack-teta association.

    raw_appearance is a separate ID-keyed EMA of filtered RAW 256-D inputs.
    It always follows the original positive IDs, momentum and expiry. Context
    fusion never writes it. The inherited tracklet embeds remain fused in the
    primary mode, and raw in the three raw-store/raw-only ablations.
    """
    def __init__(self,feature_memory_mode=PRIMARY_MODE,vis=False,record_context=False,memory_mode=None,**kwargs):
        if memory_mode is not None:
            require(feature_memory_mode==PRIMARY_MODE or feature_memory_mode==memory_mode,"Conflicting mode arguments")
            feature_memory_mode=memory_mode
        require(feature_memory_mode in MODES,"Unknown fixed feature-memory mode")
        require(vis is False,"This prototype disables GT-dependent visualization")
        self.feature_memory_mode=feature_memory_mode
        self.record_context=bool(record_context)
        super().__init__(vis=False,**kwargs)

    def reset(self):
        super().reset()
        self.raw_appearance={}
        self.last_context=None

    def _update_raw_appearance(self,ids,raw,frame_id):
        """Called after original memo update/expiry, using its exact kept IDs."""
        require(len(ids)==len(raw),"Raw bank/update row mismatch")
        for identity,vector in zip(ids,raw):
            identity=int(identity)
            if identity<0:continue
            vector=vector.detach().clone()
            if identity in self.raw_appearance:
                old=self.raw_appearance[identity]["embeds"]
                vector=(1-self.momentum_embed)*old+self.momentum_embed*vector
            self.raw_appearance[identity]=dict(embeds=vector,frame_id=frame_id)
        for identity in list(self.raw_appearance):
            if identity not in self.tracklets:del self.raw_appearance[identity]
        require(set(self.raw_appearance)==set(self.tracklets),"Raw and scoring bank IDs diverged")

    def _raw_memo(self,memo_ids):
        require(set(self.raw_appearance)==set(self.tracklets),"Incomplete independent raw bank")
        return torch.stack([self.raw_appearance[int(identity)]["embeds"] for identity in memo_ids],dim=0)

    def _confidences(self,key_cues,reference_cues,bidirectional):
        require(hasattr(self,"loss_cyc"),"Set the original pretrained fusion head and CycleLoss first")
        forward=[];reverse=[]
        for key,reference in zip(key_cues,reference_cues):
            # Both directions use the same pre-update snapshots. Pass copies
            # because an injected helper must not mutate the persistent banks.
            _,diag=self.loss_cyc.get_pair_consistency([key.clone(),reference.clone()])
            require(diag.shape==(len(key),) and torch.isfinite(diag).all(),"Invalid forward cycle confidence")
            forward.append(diag.reshape(-1,1).detach().clone())
            if bidirectional:
                _,diag_reverse=self.loss_cyc.get_pair_consistency([reference.clone(),key.clone()])
                require(diag_reverse.shape==(len(reference),) and torch.isfinite(diag_reverse).all(),"Invalid reverse cycle confidence")
                reverse.append(diag_reverse.reshape(-1,1).detach().clone())
        return forward,reverse

    def _fuse(self,raw,bboxes,cls,confidences):
        require(hasattr(self,"fusion_head"),"Set the original pretrained fusion head first")
        # Released FeatureFusionModule has ablation branches that mutate its
        # arguments in place. Supplying copies prevents cross-side effects.
        output=self.fusion_head(raw.clone(),bboxes.clone(),cls.clone(),*[c.clone() for c in confidences])
        require(torch.is_tensor(output) and output.shape==raw.shape and torch.isfinite(output).all(),"Invalid temporary fused feature")
        return output.detach()

    def _association_scores(self,key,reference,key_cls,reference_cls,temperature):
        if self.match_metric=="bisoftmax":
            sims=cal_similarity(key,reference,method="dot_product",temperature=temperature)
            exps=torch.exp(sims)
            d2t=exps/(exps.sum(dim=1).view(-1,1)+1e-6)
            t2d=exps/(exps.sum(dim=0).view(1,-1)+1e-6)
            scores=(d2t+t2d)/2
            if self.match_with_cosine:
                scores=(scores+cal_similarity(key,reference,method="cosine"))/2
            return scores
        if self.match_metric=="cosine":
            cosine=cal_similarity(key,reference,method="cosine")
            cls_sims=cal_similarity(key_cls,reference_cls,method="dot_product",temperature=temperature)
            return cosine*(cls_sims>self.contrastive_thr).float().to(cosine.device)
        raise NotImplementedError("Only original bisoftmax/cosine scoring is supported")

    @torch.no_grad()
    def match(self,bboxes,labels,embeds,cls_embeds,frame_id,temperature=-1,method="ovtrack-teta",pre_filtered=False,**kwargs):
        if method!="ovtrack-teta":raise NotImplementedError("This isolated prototype implements ovtrack-teta only")
        self.last_context=None
        # Legacy missing-feature behavior is preserved. A real empty frame uses
        # (0,256) tensors and DOES execute normal memo expiry below.
        if embeds is None:
            return super().match(bboxes.clone(),labels.clone(),None,cls_embeds,frame_id,temperature=temperature,method=method,**kwargs)
        require(bboxes.ndim==2 and bboxes.shape[1]==5 and labels.shape==(len(bboxes),) and
                embeds.shape==(len(bboxes),256) and cls_embeds.shape==(len(bboxes),512),"Expected aligned Nx5/Nx256/Nx512 inference rows")
        raw_input=embeds.detach().clone()
        if pre_filtered:require(bool(torch.all(labels>=0)),"Prefiltered rows must retain valid category labels, including unassigned track rows")
        if self.feature_memory_mode=="released":
            # The superclass owns its exact filtering, fusion, scores, greedy
            # assignment, ID init and fused memo update. The raw sidecar is
            # recorded only AFTER it finishes and cannot affect its decisions.
            output=super().match(bboxes.clone(),labels.clone(),raw_input.clone(),cls_embeds.clone(),frame_id,
                                 temperature=temperature,method=method,**kwargs)
            if pre_filtered:
                require(bool(torch.all(self.keep_mask)) and len(output[0])==len(bboxes),
                        "Released re-filter was not identity on claimed prefiltered input")
            self._update_raw_appearance(output[2],raw_input[self.keep_mask],frame_id)
            return output
        if pre_filtered:
            bboxes,labels,raw,cls=bboxes.clone(),labels.clone(),raw_input,cls_embeds.clone()
            self.keep_mask=torch.ones(len(bboxes),dtype=torch.bool,device=labels.device)
        else:
            bboxes,labels,raw,cls,_=self.remove_distractor(bboxes.clone(),labels.clone(),raw_input,cls_embeds.clone(),nms="inter")
        write_embeds=raw
        key_conf=[];reference_conf=[]
        if len(bboxes)>0 and not self.empty:
            memo_bboxes,memo_labels,memo_embeds,memo_cls,memo_ids=self.memo
            raw_memo=self._raw_memo(memo_ids)
            key_score=raw
            # raw_confidence alone retains released fused scoring/history.
            reference_score=memo_embeds if self.feature_memory_mode=="raw_confidence" else raw_memo
            if self.confused_features and self.feature_memory_mode!="raw_only":
                key_cues=(raw,bboxes[:,:-1],cls)
                reference_cues=(raw_memo,memo_bboxes[:,:-1],memo_cls)
                symmetric=self.feature_memory_mode=="raw_store_symmetric"
                key_conf,reference_conf=self._confidences(key_cues,reference_cues,symmetric)
                key_score=self._fuse(raw,bboxes[:,:-1],cls,key_conf)
                if symmetric:
                    reference_score=self._fuse(raw_memo,memo_bboxes[:,:-1],memo_cls,reference_conf)
                if self.feature_memory_mode=="raw_confidence":write_embeds=key_score
            scores=self._association_scores(key_score,reference_score,cls,memo_cls,temperature)
            require(torch.isfinite(scores).all(),"Nonfinite original association-score formula")
            if self.record_context:
                self.last_context=dict(mode=self.feature_memory_mode,memo_ids=memo_ids.detach().clone(),
                    raw_key=raw.detach().clone(),raw_reference=raw_memo.detach().clone(),
                    scoring_key=key_score.detach().clone(),scoring_reference=reference_score.detach().clone(),
                    key_confidences=[c.clone() for c in key_conf],reference_confidences=[c.clone() for c in reference_conf],
                    scores=scores.detach().clone())
            ids=torch.full((len(bboxes),),-1,dtype=torch.long)
            # Exact released input-order greedy one-to-one assignment, including
            # strict thresholds and zeroing a claimed column for other rows.
            for index in range(len(bboxes)):
                if bboxes[index,-1]<self.obj_score_thr:continue
                confidence,memo_index=torch.max(scores[index,:],dim=0)
                if confidence>self.match_score_thr:
                    ids[index]=memo_ids[memo_index]
                    scores[:index,memo_index]=0
                    scores[index+1:,memo_index]=0
        else:
            ids=torch.full((len(bboxes),),-1,dtype=torch.long)
        ids=self.init_tracklets(ids,bboxes[:,-1])
        # Keep original cls last-observation update, bbox history, momentum,
        # initialization and AFTER-update expiry. No premature memo pruning.
        self.update_memo(ids,bboxes,labels,write_embeds,cls,frame_id)
        self._update_raw_appearance(ids,raw,frame_id)
        return bboxes,labels,ids


# Both names register the same isolated implementation; the short name is the
# public cache-runner API. Defaults preserve all original tracker parameters.
OVTrackerContextMemory=OVTrackerContextFeatureMemory
TRACKERS.register_module(name="OVTrackerContextMemory",module=OVTrackerContextMemory)


def _same_tensor(first,second,message):require(torch.equal(first,second),message)


def _fixture(frame=0,n=3):
    gen=torch.Generator().manual_seed(20261024+frame)
    raw=torch.zeros((n,256))
    for row in range(n):raw[row,row%256]=1
    if n:raw+=torch.randn((n,256),generator=gen)*.015
    cls=torch.randn((n,512),generator=gen)*.05
    boxes=torch.tensor([[10.+40*k,12.,30.+40*k,40.,.9-.02*k] for k in range(n)],dtype=torch.float32).reshape(n,5)
    labels=torch.arange(n,dtype=torch.long)%3
    return boxes,labels,raw,cls


class _SyntheticFusion(torch.nn.Module):
    """Deterministic contextual helper; records real confidence arguments."""
    def __init__(self,mutate=False):
        super().__init__();self.calls=[];self.mutate=mutate
    def forward(self,raw,boxes,cls,app,bbox,semantic):
        self.calls.append(tuple(x.detach().clone() for x in (raw,boxes,cls,app,bbox,semantic)))
        result=raw.clone()+.025*(1-app)+.01*bbox+.005*semantic+.0001*boxes.sum(1,keepdim=True)+.001*cls[:,:256]
        if self.mutate:
            raw.add_(17);boxes.zero_();cls.zero_();app.fill_(0);bbox.fill_(0);semantic.fill_(0)
        return result


def _tracker(mode="raw_confidence",**kwargs):
    from ovtrack.models.losses.cyc_loss import CycleLoss
    cfg=dict(feature_memory_mode=mode,confused_features=True,vis=False,record_context=True,
        match_score_thr=.37,memo_frames=3,momentum_embed=.4)
    cfg.update(kwargs);tracker=OVTrackerContextFeatureMemory(**cfg)
    tracker.set_fusion_head(_SyntheticFusion().eval(),CycleLoss().eval())
    return tracker


def synthetic(round_number):
    torch.set_num_threads(1);torch.manual_seed(20261024)
    before={str(WORK_COPY/name):sha(WORK_COPY/name) for name in SOURCE_PINS}
    checks=[]
    if round_number==1:
        # Baseline is the imported scientific class, not a reproduced formula.
        base=OVTrackerUncertainty(confused_features=True,vis=False,match_score_thr=.37,memo_frames=3,momentum_embed=.4)
        from ovtrack.models.losses.cyc_loss import CycleLoss
        base.set_fusion_head(_SyntheticFusion().eval(),CycleLoss().eval())
        released=_tracker("released")
        for frame,n in ((0,3),(1,3),(2,2),(3,0),(7,0),(8,2)):
            args=_fixture(frame,n)
            a=base.match(*[x.clone() for x in args],frame_id=frame)
            b=released.match(*[x.clone() for x in args],frame_id=frame)
            for x,y in zip(a,b):_same_tensor(x,y,"Released output parity failed")
            require(base.num_tracklets==released.num_tracklets and set(base.tracklets)==set(released.tracklets),"Released ID/expiry parity failed")
            for identity in base.tracklets:
                for key in ("embeds","cls_embeds"):_same_tensor(base.tracklets[identity][key],released.tracklets[identity][key],"Released memo parity failed")
                require(base.tracklets[identity]["frame_ids"]==released.tracklets[identity]["frame_ids"],"Released timestamp parity failed")
        checks.append("released_calls_super_exact_output_memo_ID_parity_over_multiple_frames")
        for mode in MODES:
            tracker=_tracker(mode);tracker.match(*_fixture(0),frame_id=0)
            tracker.match(*_fixture(4,0),frame_id=4)
            require(tracker.empty and not tracker.raw_appearance,"Empty frame failed original expiry")
            tracker.reset();require(tracker.num_tracklets==0 and tracker.empty and not tracker.raw_appearance,"Reset failed")
        checks.append("all_five_modes_empty_frame_expiry_and_reset")
    elif round_number==2:
        for mode in MODES:
            tracker=_tracker(mode);first=_fixture(0);tracker.match(*first,frame_id=0)
            initial={identity:entry["embeds"].clone() for identity,entry in tracker.raw_appearance.items()}
            second=_fixture(1);result=tracker.match(*second,frame_id=1)
            kept=second[2][tracker.keep_mask]
            for identity,vector in zip(result[2],kept):
                identity=int(identity)
                if identity<0:continue
                expected=.6*initial[identity]+.4*vector if identity in initial else vector
                _same_tensor(tracker.raw_appearance[identity]["embeds"],expected,"Raw EMA was contaminated by fusion")
                if mode in ("raw_only","raw_store_one_sided","raw_store_symmetric"):
                    _same_tensor(tracker.tracklets[identity]["embeds"],expected,"Raw-store scoring memo not raw")
            if mode=="raw_confidence":
                require(any(not torch.equal(tracker.tracklets[k]["embeds"],tracker.raw_appearance[k]["embeds"]) for k in tracker.tracklets),"Primary failed to preserve distinct fused scoring EMA")
                for identity,row in zip(tracker.last_context["memo_ids"],tracker.last_context["scoring_reference"]):
                    _same_tensor(row,first[2][int(identity)],"Primary scoring reference did not retain original fused bank snapshot")
                old_fused={k:v["embeds"].clone() for k,v in tracker.tracklets.items()}
                old_raw={k:v["embeds"].clone() for k,v in tracker.raw_appearance.items()}
                tracker.match(*_fixture(2),frame_id=2)
                context=tracker.last_context
                for identity,scoring,confidence_input in zip(context["memo_ids"],context["scoring_reference"],context["raw_reference"]):
                    _same_tensor(scoring,old_fused[int(identity)],"Primary did not score against original fused EMA")
                    _same_tensor(confidence_input,old_raw[int(identity)],"Primary appearance confidence did not read raw EMA")
                _,expected_conf=tracker.loss_cyc.get_pair_consistency([context["raw_key"],context["raw_reference"]])
                require(torch.allclose(context["key_confidences"][0].flatten(),expected_conf),"Primary raw appearance confidence mismatch")
        checks.append("all_modes_raw_EMA_write_invariance_primary_preserves_fused_scoring_bank")
        symmetric=_tracker("raw_store_symmetric")
        symmetric.fusion_head=_SyntheticFusion(mutate=True).eval()
        symmetric.match(*_fixture(0),frame_id=0)
        second=_fixture(1,2);copies=[t.clone() for t in second]
        old_raw={k:v["embeds"].clone() for k,v in symmetric.raw_appearance.items()}
        symmetric.match(*second,frame_id=1)
        context=symmetric.last_context
        require(len(context["key_confidences"])==len(context["reference_confidences"])==3,"Symmetric directional confidence missing")
        require(all(c.shape==(2,1) for c in context["key_confidences"]) and
                all(c.shape==(3,1) for c in context["reference_confidences"]),"Reverse confidence uses wrong side shape")
        require(len(symmetric.fusion_head.calls)==2,"Symmetric mode must call the unchanged head once on each side")
        for a,b in zip(second,copies):_same_tensor(a,b,"Fusion in-place side effect escaped to caller")
        for row,identity in zip(context["raw_reference"],context["memo_ids"]):_same_tensor(row,old_raw[int(identity)],"Reverse fusion saw a modified bank snapshot")
        for direction,cues in (("key_confidences",(context["raw_key"],copies[0][:,:4],copies[3])),):
            references=(context["raw_reference"],_fixture(0)[0][:,:4],_fixture(0)[3])
            for k,(key,ref) in enumerate(zip(cues,references)):
                _,diag=symmetric.loss_cyc.get_pair_consistency([key,ref])
                _,reverse=symmetric.loss_cyc.get_pair_consistency([ref,key])
                require(torch.allclose(context[direction][k].flatten(),diag) and torch.allclose(context["reference_confidences"][k].flatten(),reverse),"Directional confidence differs from original CycleLoss")
        checks.append("same_raw_snapshot_bidirectional_confidences_with_inplace_head_isolation")
    elif round_number==3:
        permutation=torch.tensor([2,0,1]);inverse=torch.argsort(permutation)
        for mode in MODES[1:]:
            tracker=_tracker(mode);tracker.match(*_fixture(0),frame_id=0)
            other=copy.deepcopy(tracker);second=_fixture(1)
            original=[t.clone() for t in second]
            first=tracker.match(*second,frame_id=1)
            permuted=other.match(*[t[permutation] for t in second],frame_id=1)
            _same_tensor(first[2],permuted[2][inverse],"Unambiguous assignment permutation equivariance failed")
            require(torch.allclose(tracker.last_context["scores"],other.last_context["scores"][inverse],atol=1e-6),"Score row permutation equivariance failed")
            require(len(set(first[2].tolist()))==len(first[2]),"One-to-one IDs violated on positive test thresholds")
            for a,b in zip(second,original):_same_tensor(a,b,"Caller tensor modified")
            require(torch.isfinite(tracker.last_context["scores"]).all(),"Nonfinite inference score")
        checks.append("four_explicit_modes_unambiguous_permutation_one_to_one_inputs_immutable_finite")
        # Verify distractor filtering keeps raw-bank rows aligned, including
        # low-score retained id=-1 objects that are part of current context.
        tracker=_tracker("raw_confidence",init_score_thr=.5,obj_score_thr=.5)
        args=list(_fixture(0));args[0][1]=args[0][0];args[0][1,-1]=.1;args[0][2,-1]=.2
        result=tracker.match(*args,frame_id=0)
        require(tracker.keep_mask.tolist()==[True,False,True] and result[2].tolist()==[0,-1] and set(tracker.raw_appearance)=={0},"Filtered/id=-1 row alignment failed")
        for method in ("ovtrack-tao","appearance"):
            try:tracker.match(*_fixture(1),frame_id=1,method=method)
            except NotImplementedError:pass
            else:raise AssertionError("Unsupported method accepted")
        checks.append("original_filter_alignment_unassigned_rows_and_unsupported_method_rejection")
        for mode in MODES:
            regular=_tracker(mode);prefiltered=_tracker(mode)
            regular.match(*_fixture(0),frame_id=0);prefiltered.match(*_fixture(0),frame_id=0)
            args=list(_fixture(1));args[0][1]=args[0][0];args[0][1,-1]=.1
            filtered=regular.remove_distractor(*[x.clone() for x in args],nms="inter")
            a=regular.match(*args,frame_id=1)
            b=prefiltered.match(*filtered[:4],frame_id=1,pre_filtered=True)
            for x,y in zip(a,b):_same_tensor(x,y,"Prefiltered explicit cache-entry parity failed")
            require(bool(torch.all(prefiltered.keep_mask)),"Prefiltered rows changed")
        checks.append("five_modes_explicit_prefiltered_cache_entry_parity")
        from ovtrack.models.roi_heads.ovtrack_roi_head import FeatureFusionModule
        actual=_tracker("raw_store_symmetric")
        actual.fusion_head=FeatureFusionModule(feature_dim=256,max_fusion_ratio=2.).eval()
        actual.match(*_fixture(0),frame_id=0);actual.match(*_fixture(1),frame_id=1)
        require(torch.isfinite(actual.last_context["scores"]).all() and all(p.grad is None for p in actual.fusion_head.parameters()),"Actual frozen-head inference interface failed")
        checks.append("actual_FeatureFusionModule_256D_CPU_inference_no_parameter_gradients")
    else:raise ValueError("Use exactly three declared synthetic rounds")
    require(len(checks)>=2 and before=={path:sha(path) for path in before},"Synthetic check count or protected-source integrity failed")
    return dict(status="PASS_CONTEXT_FEATURE_MEMORY_SYNTHETIC_ROUND",round=round_number,checks=checks,
        primary_mode=PRIMARY_MODE,reported_modes=list(MODES),module_sha256=sha(__file__),protected_source_sha256=before,
        real_cache_or_GT_opened=False,real_checkpoint_loaded=False,training_started=False,GPU_used=False,
        interpretation="Engineering train/inference feature-state consistency hypothesis only; no novelty or efficacy established.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-round",type=int,choices=(1,2,3),required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args();receipt=synthetic(args.synthetic_round)
    if args.out:
        with args.out.open("x") as stream:stream.write(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
