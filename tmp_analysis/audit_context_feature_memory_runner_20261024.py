"""Independent authenticated-flow fixture; no real cache/GT/checkpoint."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp_analysis"))
import replay_context_feature_memory as subject
import context_feature_memory as module
from ovtrack.models.roi_heads.ovtrack_roi_head import FeatureFusionModule


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj) + "\n")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def independent_b3(rows):
    sources = sorted({r[0] for r in rows})
    result = []
    for source in sources:
        values = [r for r in rows if r[0] == source]
        trajectories = sorted({(r[1],r[2]) for r in values})
        ps, rs = [], []
        for trajectory in trajectories:
            observations = [r for r in values if (r[1],r[2]) == trajectory]
            p, rec = [], []
            for row in observations:
                key = (row[1],row[4]) if row[4] >= 0 else ("singleton",row[3])
                cluster = [r for r in values if ((r[1],r[4]) if r[4] >= 0 else ("singleton",r[3])) == key]
                overlap = sum((r[1],r[2]) == trajectory for r in cluster)
                p.append(overlap/len(cluster))
                rec.append(overlap/len(observations))
            ps.append(np.mean(p)); rs.append(np.mean(rec))
        result.append((np.mean(ps),np.mean(rs)))
    return np.mean(result,axis=0)


def main():
    torch.set_num_threads(1)
    runner_sha, module_sha = digest(subject.__file__), digest(module.__file__)
    helper_sha = digest(ROOT / "tmp_analysis/build_fit_identity_edge_support.py")
    real_protocol = ROOT / "tmp_analysis/context_feature_memory_replay_protocol_20261024.json"
    real_spec = ROOT / "tmp_analysis/context_feature_memory_protocol_20261024.json"
    protected = {str(module.WORK_COPY/k):digest(module.WORK_COPY/k) for k in module.SOURCE_PINS}
    # Brute-force grouping uses no Counter or producer metric helper.
    rows = [("a",1,1,0,0),("a",1,1,1,0),("a",1,2,2,0),
            ("b",2,1,3,-1),("b",2,1,4,-1),("b",3,1,5,0)]
    actual = subject.bcubed([dict(source_group=s,video_id=v,gt_track_id=y,observation_id=o,new_id=z)
                            for s,v,y,o,z in rows])
    expected = independent_b3(rows)
    np.testing.assert_allclose([actual["precision"],actual["recall"]],expected,rtol=0,atol=1e-15)
    assert subject.canonical_partition([8,8,-1,-1,3]) == [0,0,1,2,3]
    assert subject.canonical_partition([2,2,3]) == subject.canonical_partition([9,9,6])

    with tempfile.TemporaryDirectory(prefix="independent_feature_memory_") as temp:
        root = Path(temp); stage = root/"stage"; run = stage/"full_collect_v1"
        support = stage/"support_v1"; pair = stage/"pair_context_v1"
        for p in (run/"dump",support,pair):p.mkdir(parents=True)
        torch.manual_seed(771)
        fusion = FeatureFusionModule(feature_dim=256,max_fusion_ratio=2.).eval()
        checkpoint = root/"synthetic_checkpoint.pth"
        torch.save({"state_dict":{"roi_head.fusion_head."+k:v for k,v in fusion.state_dict().items()}},checkpoint)
        manifest = run/"run_manifest.json"
        dump(manifest,dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint)))
        frames=[]; inventory={}; index=[]; supervised=[]; oid=0
        for vid,source,n in ((1,"alpha",2),(2,"beta",1),(3,"empty",1)):
            for step in range(3):
                iid = vid*10+step
                count = n if step != 2 else 0
                raw = np.zeros((count,256),np.float16)
                cls = np.zeros((count,512),np.float16)
                boxes=np.asarray([[10+40*k+step,10,25+40*k+step,25,.9] for k in range(count)],np.float32).reshape(count,5)
                for k in range(count):raw[k,k]=1;cls[k,k]=1
                path = run/"dump"/("v%d_i%d_f%d.npz"%(vid,iid,step))
                np.savez(path,image_id=iid,video_id=vid,det_bboxes=boxes,det_labels=np.arange(count,dtype=np.int64),
                         track_feats=raw,cls_feats=cls,track_ids=np.arange(count,dtype=np.int64))
                inventory[path.name]=digest(path)
                frames.append(dict(video_id=vid,image_id=iid,frame_id=step,frame_index=step*17,
                                   npz=str(path),npz_sha256=digest(path),observations=count))
                for k in range(count):
                    row=dict(observation_id=oid,split="fit",source_group=source,video_id=vid,image_id=iid,
                             local_track_id=k,dump_row=k,sampled_step=step,original_frame_index=step*17)
                    index.append(row)
                    if source != "empty":
                        identity={k:v for k,v in row.items() if k != "dump_row"}
                        identity.update(gt_track_id=k+1,pred_category_id=k+1,manual_anchor_id=100+oid,
                                        eligible_for_parameter_fit=True,error={"forbidden":"not numeric"},
                                        soft_risk={"forbidden":"not numeric"},gt_category_id={"forbidden":"not numeric"},
                                        posterior_argmax_category_id={"forbidden":"not numeric"})
                        supervised.append(identity)
                    oid+=1
        supervised.append(dict(split="tune",gt_track_id={"forbidden":"not an identity integer"},
                               gt_category_id={"forbidden":"not a category integer"},error={"forbidden":"not numeric"}))
        jsonl(run/"frame_execution.jsonl",frames);dump(run/"dump_inventory.json",inventory)
        jsonl(support/"gt_free_observations.jsonl",index)
        jsonl(support/"supervised_observations.jsonl",supervised)
        dump(support/"support_receipt.json",dict(output_files={"supervised_observations.jsonl":digest(support/"supervised_observations.jsonl")}))
        dump(pair/"source_roles.json",dict(groups={s:dict(split="fit",role="mean_role") for s in ("alpha","beta","empty")}))
        dump(pair/"independent_pair_context_audit.json",dict(output_files_sha256={"source_roles.json":digest(pair/"source_roles.json")}))
        dump(support/"gt_free_index_manifest.json",dict(files={"gt_free_observations.jsonl":digest(support/"gt_free_observations.jsonl")}))
        preflight = root/"synthetic_preflight.json"
        dump(preflight,dict(status="PASS_INDEPENDENT_CONTEXT_FEATURE_MEMORY_IMPLEMENTATION_PREFLIGHT",
                            runner_sha256=runner_sha,module_sha256=module_sha))
        protocol=dict(input_pins={},video_ids=[1,2,3],source_groups=["alpha","beta","empty"],
                      modes=list(module.MODES),primary="raw_confidence",output_directory=str(root/"out"),
                      tracker=dict(type="OVTrackerUncertainty",init_score_thr=.0001,obj_score_thr=.0001,
                                   match_score_thr=.37,memo_frames=50,momentum_embed=.4,momentum_obj_score=.5,
                                   match_metric="bisoftmax",match_with_cosine=True,contrastive_thr=.5,confused_features=True,vis=False),
                      fusion=dict(type="FeatureFusionModule",feature_dim=256,max_fusion_ratio=2.))
        protocol_path=root/"synthetic_protocol.json";dump(protocol_path,protocol)
        fixed_hashes={
            str(manifest):"8a99676a2a45a61d3c76d423480f59381cfe4ab80e0a54df671417a5599198ab",
            str(pair/"independent_pair_context_audit.json"):"df7c386a8f01d3dbfce928e58758f9c455f685e8485dbfa8b19a0cd4ac936e81",
            str(support/"gt_free_index_manifest.json"):"7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667"}
        original_sha=subject.sha
        def fixture_sha(path):
            # A fixture-only accepted-authority map; actual fixture bytes are
            # independently snapshotted before/after. No real accepted hashes
            # are bypassed or files opened for this synthetic run.
            return fixed_hashes.get(str(path),original_sha(path))
        snapshot={str(p):digest(p) for p in root.rglob("*") if p.is_file()}
        subject.STAGE,subject.RUN=stage,run
        subject.PROTOCOL,subject.PROTOCOL_SHA=protocol_path,digest(protocol_path)
        subject.PREFLIGHT=preflight
        subject.sha=fixture_sha
        original_score=subject.score_sealed_predictions
        boundary=[]
        def score_after_seal(out,configuration,selected):
            seal=json.loads((out/"predictions_sealed.json").read_text())
            assert list(seal["output_files_sha256"]) == [m+".jsonl" for m in module.MODES]
            for mode in module.MODES:
                p=out/(mode+".jsonl")
                assert digest(p)==seal["output_files_sha256"][p.name]
            boundary.append("all_five_prediction_files_sealed_before_identity_join")
            return original_score(out,configuration,selected)
        subject.score_sealed_predictions=score_after_seal
        subject.execute()
        result=json.loads((root/"out/summary.json").read_text())
        assert boundary == ["all_five_prediction_files_sealed_before_identity_join"]
        assert result["scoring"]["eligible_anchor_observations"] == 6
        assert result["scoring"]["selected_sources_without_eligible_anchors"] == ["empty"]
        assert result["selected_frames"] == 9
        for mode in module.MODES:
            metric=result["scoring"]["metrics"][mode]
            assert metric["sources"]==2 and metric["observations"]==6
            assert abs(metric["precision"]-1.)<1e-14 and abs(metric["recall"]-1.)<1e-14
            assert len(list((root/"out"/(mode+".jsonl")).open()))==9
        assert snapshot=={p:digest(p) for p in snapshot}
        rejected=False
        try:subject.execute()
        except ValueError as error:rejected="Preserve previous output" in str(error)
        assert rejected
    assert runner_sha==digest(subject.__file__) and module_sha==digest(module.__file__)
    assert helper_sha==digest(ROOT/"tmp_analysis/build_fit_identity_edge_support.py")
    assert protected=={p:digest(p) for p in protected}
    assert not torch.cuda.is_initialized()
    receipt=dict(status="PASS_INDEPENDENT_CONTEXT_FEATURE_MEMORY_IMPLEMENTATION_PREFLIGHT",
                 runner_sha256=runner_sha,module_sha256=module_sha,
                 replay_protocol_path=str(real_protocol),replay_protocol_sha256=digest(real_protocol),
                 module_protocol_path=str(real_spec),module_protocol_sha256=digest(real_spec),
                 identity_decoder_sha256=helper_sha,
                 checks=["independent_B3_bruteforce_source_GT_observation_weighting_crossvideo_IDs_singletons",
                         "full_9_frame_3_video_5_mode_synthetic_runner_fusion_only_synthetic_checkpoint",
                         "all_five_predictions_verified_sealed_before_identity_join",
                         "tune_identity_and_correctness_poison_not_decoded_fixed_anchor_denominator",
                         "unknown_empty_source_reported_and_output_overwrite_rejected",
                         "reviewed_code_and_scientific_sources_unchanged_CPU_only"],
                 module_independent_preflight=str(ROOT/"tmp_analysis/context_feature_memory_20261024_independent_preflight.json"),
                 protected_source_sha256=protected,
                 synthetic_fixture_authentication="Fixture-only accepted-authority digest map for three temp files; all physical fixture bytes independently snapshotted before/after. No real authority hashes changed.",
                 real_cache_opened=False,real_supervision_opened=False,real_checkpoint_loaded=False,GPU_initialized=False,
                 limitations=["Implementation preflight only; no real-data efficacy", "FP16-cache CPU comparisons cannot guarantee original float32 GPU baseline parity", "Final real outputs require independent output audit"])
    output=ROOT/"tmp_analysis/context_feature_memory_independent_preflight_20261024.json"
    output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
