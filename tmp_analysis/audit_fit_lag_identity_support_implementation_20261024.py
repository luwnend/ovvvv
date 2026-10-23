"""Independent synthetic/GT-free-metadata implementation preflight only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

import verify_fit_lag_identity_support_output_20261024 as independent

ROOT=Path(__file__).resolve().parents[1]
SUBJECT=ROOT/"tmp_analysis/audit_fit_lag_identity_support_20261024.py"
PROTOCOL=ROOT/"tmp_analysis/fit_lag_identity_support_protocol_20261024.json"
EXPECTED_SCRIPT="9beb8faf932d13f417d491ddd8f853de001d17a414eac0d59d016b0ffcf3af3b"
EXPECTED_PROTOCOL="dd84da8fd028af36c223f42a30d19f9f23725eb790e60aaa668700c80a0806ad"


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def fixtures():
    random=np.random.RandomState(20261024);frames={};records=[];anchors={}
    for video,counts in ((1,[2,3,0,2,2,2,3,2,2,2]),(2,[2,2,1,3,2,2,2,2,2,2]),(3,[0,0])):
        times=[0,5,9,31,50,52,90,131,145,190]
        for step,n in enumerate(counts):
            boxes=np.asarray([[10+times[step]+k*21,10+video,22+times[step]+k*21,30+video] for k in range(n)],float).reshape(n,4)
            ids=np.asarray(([0,0,-1] if video==1 and step==1 else list(range(n))),np.int64)
            if n>2 and step!=1:ids[-1]=-1
            image=1000*video+step
            frame=dict(video_id=video,image_id=image,sampled_step=step,original_frame_index=times[step],boxes=boxes,
                raw=random.randn(n,256).astype(np.float16).astype(float),semantic=random.randn(n,512).astype(np.float16).astype(float),
                released_ids=ids,predicted_categories=np.ones(n,np.int32))
            # Deliberate score ties, unknown anchors, no-positive queries, and
            # known positive/negative references in both predeclared lags.
            if n>0 and step%3==0:frame["raw"][:]=0
            if n>1 and step%4==0:frame["predicted_categories"][-1]=2
            frames[video,step]=frame;records.append(dict(video_id=video,frame_id=step,observations=n))
            for row in range(n):
                if row==2:continue
                anchors[image,row]=(video,9 if row==0 and step==8 else row+1)
    return frames,records,anchors


def main():
    assert independent.sha(SUBJECT)==EXPECTED_SCRIPT and independent.sha(PROTOCOL)==EXPECTED_PROTOCOL
    subject=module(SUBJECT,"independent_preflight_fit_lag_subject")
    geometry=module(subject.GEOMETRY,"independent_preflight_fit_lag_geometry")
    assert subject.PROTOCOL_SHA==EXPECTED_PROTOCOL
    protocol=json.loads(PROTOCOL.read_text())
    for round_number in (1,2,3):
        receipt=json.loads((ROOT/f"tmp_analysis/fit_lag_identity_support_synthetic_round{round_number}_20261024_v3.json").read_text())
        assert receipt["script_sha256"]==EXPECTED_SCRIPT and receipt["protocol_sha256"]==EXPECTED_PROTOCOL
        assert receipt["status"]=="PASS_FIT_LAG_IDENTITY_SUPPORT_SYNTHETIC_ROUND"
    fit,videos,metadata,groups,population=independent.metadata_population(protocol)
    assert population==dict(sealed_fit_source_groups=341,allocated_fit_videos=405,selected_fit_videos=403,selected_frames=13791,retained_observations=613536)
    omitted=subject.validate_population(fit,videos,metadata,protocol["population"])
    assert [row["video_id"] for row in omitted]==[1720,1974] and all(not row["source_in_sealed_roles"] for row in omitted)
    bad_metadata=copy.deepcopy(metadata);bad_metadata[0]["video_id"]=1720
    try:subject.validate_population(fit,videos,bad_metadata,protocol["population"])
    except ValueError:pass
    else:raise AssertionError("Cached source without a sealed role was admitted")
    snapshots,records,anchors=fixtures();video_source={"1":"S","2":"T","3":"ZERO"}
    roles={s:dict(split="fit",role="mean_role" if s=="S" else "pair_role",block="synthetic") for s in ("S","T","ZERO","UNUSED")}
    bounds=dict(max_frame_rows=3,max_video_frames=10,max_candidates_per_frame_pair=9,generation_ring_frames=5)
    with tempfile.TemporaryDirectory(prefix="independent_fit_lag_complete_") as temporary:
        out=Path(temporary)
        generated=subject.generate(out,records,video_source,sorted(roles),geometry,lambda row:copy.deepcopy(snapshots[row["video_id"],row["frame_id"]]),bounds)
        subject.write(out/"GT_free_candidates_sealed.json",dict(status="SEALED_ALL_FIT_LAG_CANDIDATES_BEFORE_SUPERVISION",**generated))
        subject.verify_seal(out)
        actual,cells,history,gaps,maxread=subject.score_sealed(out,anchors,sorted(roles))
        actual_results=subject.summarize(out,actual,cells,roles,history,gaps)
        expected=independent.database(out/"independent.sqlite")
        grouped={video:[row for row in records if row["video_id"]==video] for video in (1,2,3)}
        def loader(record,past):return independent.independent_history(copy.deepcopy(snapshots[record["video_id"],record["frame_id"]]),past)
        ecells,ehistory,egaps,checks=independent.inspect_scores(out,grouped,video_source,sorted(roles),anchors,expected,bounds,loader)
        expected_results=independent.report_results(expected,ecells,roles,ehistory,egaps)
        independent.same(actual_results,expected_results,"Fresh literal support/query/pair-order/union/fold oracle")
        counts=independent.compare_databases(expected,actual)
        assert maxread==checks["maximum_candidate_block"]==generated["maximum_candidate_block"]
        assert generated["maximum_feature_ring_frames"]==5
        assert generated["frame_pair_receipts"]==2*len(records)==44
        assert expected_results["per_source"]["ZERO"]["lags"]["4"]["all_candidates"]["all_known_population"]["candidate_pairs"]==0
        assert expected_results["necessary_support_gate"]["status"]=="HOLD_PENDING_SEPARATE_TRAINING_PROTOCOL"
        assert expected_results["necessary_support_gate"]["sufficient_training_support_established"] is False
        # A corrupted rowmap, score shard, receipt or removed shard must fail
        # before a supervised helper import; every synthetic member is tested.
        original_module=subject.module
        def forbid_import(*args,**kwargs):raise AssertionError("Supervised helper imported before global seal check")
        subject.module=forbid_import
        tested=[]
        try:
            for relative in ("shards/v1.rows.json","shards/v1_lag4.scores.npy","frame_pairs.jsonl"):
                path=out/relative;original=path.read_bytes();path.write_bytes(original+b"corrupt")
                try:subject.join_identities(out,{}, {}, {})
                except ValueError:pass
                else:raise AssertionError("Corrupt GT-free member accepted before join: "+relative)
                path.write_bytes(original);tested.append(relative)
            path=out/"shards/v2_lag1.scores.npy";renamed=out/"only_synthetic_backup.npy";path.rename(renamed)
            try:subject.join_identities(out,{}, {}, {})
            except FileNotFoundError:pass
            else:raise AssertionError("Missing GT-free member accepted before join")
            renamed.rename(path)
        finally:subject.module=original_module
        actual.close();expected.close()
    # Additional one-query matrix controls independent of frame geometry.
    rng=np.random.RandomState(411);seen=np.zeros((3,3),int)
    with tempfile.TemporaryDirectory(prefix="independent_fit_lag_rank_") as temporary:
        actual=subject.open_database(Path(temporary)/"actual.sqlite");expected=independent.database(Path(temporary)/"expected.sqlite")
        for trial in range(100):
            nq,nr=int(rng.randint(1,5)),int(rng.randint(1,6));score=rng.randint(-2,3,(nq,nr,6)).astype(float)
            q=rng.choice([-1,1,2,3],nq);r=rng.choice([-1,1,2,3],nr);mask=rng.rand(nq,nr)>.2
            pop=independent.POPULATIONS[trial%2];a=subject.fresh_cell();e=independent.fresh_cell()
            if pop==independent.POPULATIONS[0]:score[:,:,4]=np.nan
            subject.update_cell(a,score,mask,q,r,np.arange(nr),"S",1,"all_candidates",pop,1,trial,actual)
            independent.count_block(e,score,mask,q,r,"S",1,"all_candidates",pop,1,trial,expected)
            independent.same(subject.finalized_cell(a,pop),independent.final_cell(e,pop),"Random literal ranking scope")
            seen+=np.asarray(e["orders"][independent.FEATURES[1]])
        assert (seen>0).all()
        independent.compare_databases(expected,actual);actual.close();expected.close()
    assert independent.sha(SUBJECT)==EXPECTED_SCRIPT and independent.sha(PROTOCOL)==EXPECTED_PROTOCOL
    receipt=dict(status="PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_IMPLEMENTATION_PREFLIGHT",script_sha256=EXPECTED_SCRIPT,
        protocol_sha256=EXPECTED_PROTOCOL,geometry_sha256=independent.sha(subject.GEOMETRY),checker_sha256=independent.sha(__file__),
        GT_free_metadata_population=population,uncollected_allocated_FIT_video_ids=sorted(set(map(int,videos))-set(groups)),
        synthetic_integration=dict(videos=3,source_groups=4,frames=len(records),candidate_counts=checks["candidates_by_lag"],
            SQLite_table_rows=counts,maximum_score_rows=checks["maximum_candidate_block"],maximum_feature_ring_frames=5),
        checks=["all_frozen_FIT_roles_and_metadata_lag_QxM_counts_independently_recomputed",
                "independent_all_six_feature_original_frame_CV_and_rowmap_recomputation_per_block",
                "empty_previous_short_video_zero_length_lag4_and_zero_support_sources_and_folds",
                "current_duplicate_history_invalidation_no_jump_and_lag4_past_snapshot",
                "all_query_SQLite_identity_pair_trajectory_rows_and_3x3_orders_reconstructed",
                "100_independent_literal_random_ranking_scopes_all_3x3_cells_and_absent_CV_columns",
                "cross_lag_hard_query_join_source_held_out_BOTH_lag_gate_and_micro_union_parity",
                "three_corrupt_member_types_and_missing_shard_rejected_before_supervised_helper_import",
                "uncollected_allocation_without_role_preserved_but_cached_missing_role_rejected",
                "published_three_v3_synthetic_receipts_bound_to_final_script_and_protocol"],
        real_cached_features_opened=False,real_supervised_payload_opened=False,real_model_opened=False,GPU=False,
        limits=["Implementation and GT-free metadata preflight only; real diagnostic and output audit remain unfinished"])
    destination=ROOT/"tmp_analysis/fit_lag_identity_support_independent_preflight_20261024.json"
    with destination.open("x") as stream:json.dump(receipt,stream,indent=2);stream.write("\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
