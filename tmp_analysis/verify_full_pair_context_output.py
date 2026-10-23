#!/usr/bin/env python3
"""Independent GT-free pair-index audit; no producer imports or NPZ/GT IO.

All real track combinations are rehashed, including unretained candidates.
References are independently rebuilt from complete sealed predicted tracks.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/base_error_full_collection_20261023"
MANIFEST = STAGE / "support_v1/gt_free_index_manifest.json"
MANIFEST_SHA = "7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667"
PLAN = STAGE / "full_collection_plan.json"
PLAN_SHA = "cbb14c6fd41135d887b46ce1b5869f0068b76922bf5c40f87f78ec2c2a715f9b"
PROTOCOL = ROOT / "tmp_analysis/full_mean_pair_gate_protocol_20261023.json"
PROTOCOL_SHA = "fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718"
PRODUCER = ROOT / "tmp_analysis/build_full_pair_context_cache.py"
PRODUCER_SHA = "70ac6b7e844cbff3802ced02db732d04d32f72173e32e684ece0befdf9ddb654"
PREFLIGHT = ROOT / "tmp_analysis/full_pair_context_sealer_independent_preflight_20261023.json"


def check(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for row in stream:
            check(bool(row.strip()), "Blank JSONL row")
            yield json.loads(row)


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=True, allow_nan=False)


def exact(actual, expected, message):
    # JSON serialization also distinguishes bool from integer aliases.
    check(compact(actual) == compact(expected), message)


def array_hash(values):
    return hashlib.sha256(json.dumps(values,separators=(",", ":"),ensure_ascii=False).encode()).hexdigest()


def roles_from_metadata(records, images):
    groups = {}
    for video in {im["video_id"] for im in images.values()}:
        row=records[video];group=row["source_group"];part=row["split"]
        check(part in ("fit","tune") and (group not in groups or groups[group]==part),"Metadata source split crossing/test")
        groups[group]=part
    result={g:dict(split="tune",role="tune_evaluation") for g,p in groups.items() if p=="tune"}
    for block in sorted({g.split("/")[0] for g,p in groups.items() if p=="fit"}):
        members=[g for g,p in groups.items() if p=="fit" and g.split("/")[0]==block]
        members.sort(key=lambda g:hashlib.sha256(b"pair_role_v1\0"+g.encode()).hexdigest())
        for i,g in enumerate(members):
            result[g]=dict(split="fit",role="mean_role" if i%2==0 else "pair_role",block=block)
    return dict(sorted(result.items()))


def minima(video, local, ordered):
    """Independent sorted-list top-four implementation; not a producer helper."""
    counts=[0,0,0,0];best=[[],[],[],[]]
    if len(ordered)<4:
        return counts,best
    prefix="pair_edges_v1|%d|%d|"%(video,local)
    for i in range(len(ordered)):
        a=ordered[i]
        for j in range(i+1,len(ordered)):
            b=ordered[j];lag=b[0]-a[0]
            check(lag>0,"Nonpositive sampled lag")
            k=0 if lag<=2 else 1 if lag<=5 else 2 if lag<=10 else 3
            counts[k]+=1
            low,high=sorted((a[1],b[1]))
            code=hashlib.sha256((prefix+str(low)+"|"+str(high)).encode()).hexdigest()
            item=(code,i,j)
            if len(best[k])<4 or item<best[k][-1]:
                bisect.insort(best[k],item)
                if len(best[k])>4:
                    best[k].pop()
    return counts,best


def expected_edge(video,local,ordered,bin_number,entry,part,source,role):
    code,i,j=entry;a,b=ordered[i],ordered[j]
    refs=[row[1] for index,row in enumerate(ordered) if index not in (i,j)]
    check(len(refs)==len(ordered)-2>=2 and a[1] not in refs and b[1] not in refs,"Independent reference complement failure")
    return dict(edge_id=code,video_id=video,local_track_id=local,observation_i=a[1],observation_j=b[1],
                image_i=a[2],image_j=b[2],sampled_step_i=a[0],sampled_step_j=b[0],sample_step_lag=b[0]-a[0],
                original_frame_lag=b[3]-a[3],lag_bin=bin_number,n_track_observations=len(ordered),
                n_reference_observations=len(refs),reference_observation_ids_sha256=array_hash(refs),
                disjoint_reference_supported=len(ordered)>=6,split=part,source_group=source,role=role)


def next_required(iterator, message):
    row=next(iterator,None);check(row is not None,message);return row


def verify(directory, destination):
    started=time.monotonic();check(not destination.exists(),"Preserve existing independent audit")
    receipt_path=directory/"pair_context_receipt.json";receipt=read(receipt_path)
    check(receipt["status"]=="SEALED_GT_FREE_PAIR_EDGE_REFERENCE_INDEX_PENDING_INDEPENDENT_AUDIT","Unfinished pair-context seal")
    check(sha(PROTOCOL)==PROTOCOL_SHA and sha(PLAN)==PLAN_SHA and sha(MANIFEST)==MANIFEST_SHA and sha(PRODUCER)==PRODUCER_SHA,"Frozen source binding changed")
    preflight=read(PREFLIGHT)
    check(preflight["status"]=="PASS_INDEPENDENT_GT_FREE_PAIR_CONTEXT_SEALER_IMPLEMENTATION_PREFLIGHT" and preflight["script_sha256"]==PRODUCER_SHA and preflight["protocol_sha256"]==PROTOCOL_SHA,"Implementation preflight mismatch")
    exact(receipt["aggregation_key"],["video_id","local_track_id"],"Track key changed")
    for flag in ("supervision_opened","supervised_pair_filtering_performed","correctness_opened","test_opened",
                 "class_holdout_correctness_opened","posterior_or_feature_NPZ_opened","training_started","model_imported","GPU_used"):
        check(receipt[flag] is False,"Scope flag opened: "+flag)
    check(receipt["protocol_sha256"]==PROTOCOL_SHA and receipt["full_collection_plan_sha256"]==PLAN_SHA and
          receipt["accepted_gt_free_manifest_sha256"]==MANIFEST_SHA,"Receipt source pins")
    index_files=[MANIFEST.parent/name for name in ("gt_free_observations.jsonl","gt_free_tracks.jsonl")]
    expected_inputs={str(p):sha(p) for p in (PROTOCOL,MANIFEST,PLAN,*index_files,PRODUCER)}
    exact(receipt["input_sha256"],expected_inputs,"Input manifest differs")
    exact(receipt["input_sha256_after"],expected_inputs,"Producer input drift")
    check(set(receipt["opened_input_paths"])==set(expected_inputs)-{str(PRODUCER)},"Unexpected opened input path declaration")
    output_names={"source_roles.json","track_contexts.jsonl","edges.jsonl"}
    check(set(receipt["files"])==output_names,"Unexpected pair-context output file set")
    output_hashes={name:sha(directory/name) for name in output_names}
    exact(receipt["files"],output_hashes,"Pair-context output changed")
    watched={**expected_inputs,**{str(directory/name):value for name,value in output_hashes.items()},
             str(receipt_path):sha(receipt_path),str(PREFLIGHT):sha(PREFLIGHT),str(Path(__file__).resolve()):sha(__file__)}
    manifest=read(MANIFEST);plan=read(PLAN)
    check(manifest["status"]=="SEALED_GT_FREE_ALL_EMITTED_INDEX_BEFORE_SUPERVISION_OPEN" and
          manifest["gt_filtering"] is False and manifest["supervision_opened"] is False,"Invalid upstream seal")
    for name,digest in manifest["files"].items():
        check(expected_inputs[str(MANIFEST.parent/name)]==digest,"Upstream index file binding")
    records={r["video_id"]:r for r in plan["video_records"]};images={r["image_id"]:r for r in plan["images"]}
    check(len(records)==450 and len(images)==15283,"Wrong full metadata scope")
    roles=roles_from_metadata(records,images)
    actual_roles=read(directory/"source_roles.json")
    exact(actual_roles,dict(status="SEALED_METADATA_ONLY_SOURCE_ROLES",role_seed="pair_role_v1",
          rule="Block by source-group prefix; hash order; alternate mean_role/pair_role; tune fixed evaluation",
          groups=roles,counts=dict(Counter(v["role"] for v in roles.values())),outcomes_used=False),"Independent source role mismatch")
    grouped=defaultdict(list);seen=set();split_counts=Counter()
    allowed={"observation_id","video_id","local_track_id","image_id","sampled_step","original_frame_index",
             "split","source_group","dump_file","dump_sha256","dump_row","pred_category_id","score"}
    required=allowed-{"pred_category_id","score"}
    for row in lines(index_files[0]):
        check(required<=set(row)<=allowed,"Supervision/unknown field in GT-free index")
        oid,video,local,iid=(row[k] for k in ("observation_id","video_id","local_track_id","image_id"))
        check(all(type(v) is int for v in (oid,video,local,iid,row["sampled_step"],row["original_frame_index"])) and oid>=0 and local>=0 and oid not in seen,"Invalid observation identity")
        seen.add(oid);im=images[iid];rec=records[video]
        exact((video,row["sampled_step"],row["original_frame_index"],row["split"],row["source_group"]),
              (im["video_id"],im["frame_id"],im["frame_index"],rec["split"],rec["source_group"]),"Index/metadata identity")
        check(im["sample_step"]==im["frame_id"] and rec["split"] in ("fit","tune"),"Timeline/split mismatch")
        grouped[(video,local)].append((row["sampled_step"],oid,iid,row["original_frame_index"]))
        split_counts[rec["split"]]+=1
    check(len(seen)==manifest["observations"]==681602 and seen==set(range(len(seen))),"Full index count/IDs")
    seen_tracks=set()
    for row in lines(index_files[1]):
        key=(row["video_id"],row["local_track_id"])
        check(key not in seen_tracks and key in grouped,"Duplicate/missing upstream track")
        seen_tracks.add(key);members=grouped[key];rec=records[key[0]]
        exact((row["split"],row["source_group"]),(rec["split"],rec["source_group"]),"Upstream track source")
        check(row["n_observations"]==len(members)==len(row["observation_ids"]) and
              sorted(row["observation_ids"])==sorted(r[1] for r in members),"Upstream complete membership")
    check(seen_tracks==set(grouped) and len(grouped)==manifest["tracks"]==144930,"Full upstream track scope")
    combinations=sum(len(rows)*(len(rows)-1)//2 for rows in grouped.values() if len(rows)>=4)
    print(json.dumps(dict(status="INDEPENDENT_FULL_PAIR_AUDIT_INPUTS_READY",tracks=len(grouped),candidate_combinations_to_rehash=combinations,elapsed_seconds=time.monotonic()-started)),flush=True)
    contexts=iter(lines(directory/"track_contexts.jsonl"));edges=iter(lines(directory/"edges.jsonl"))
    counts=Counter();by_role=defaultdict(Counter);by_source=defaultdict(Counter)
    verified_candidates=0;verified_edges=0;max_track_length=0
    for track_number,((video,local),rows) in enumerate(sorted(grouped.items()),1):
        ordered=sorted(rows);n=len(ordered);max_track_length=max(max_track_length,n)
        check(len({r[0] for r in ordered})==n and len({r[2] for r in ordered})==n and
              all(a[3]<b[3] for a,b in zip(ordered,ordered[1:])),"Invalid complete track timeline")
        counts["all_emitted_tracks"]+=1;counts["all_emitted_observations"]+=n
        if n<4:
            counts["tracks_below_reference_support"]+=1
            continue
        rec=records[video];part=rec["split"];source=rec["source_group"];role=roles[source]["role"]
        ids=[r[1] for r in ordered]
        expected_context=dict(video_id=video,local_track_id=local,split=part,source_group=source,role=role,
                              n_observations=n,observation_ids=ids,sampled_steps=[r[0] for r in ordered],
                              original_frame_indices=[r[3] for r in ordered],observation_ids_sha256=array_hash(ids))
        exact(next_required(contexts,"Missing complete track context"),expected_context,"Track context differs from complete input")
        counts["supported_tracks"]+=1;by_role[role]["supported_tracks"]+=1;by_source[source]["supported_tracks"]+=1
        candidate_counts,best=minima(video,local,ordered)
        check(sum(candidate_counts)==n*(n-1)//2,"Not all track combinations evaluated")
        verified_candidates+=sum(candidate_counts)
        for k,num in enumerate(candidate_counts):
            counts["candidate_edges_lag_bin_%d"%k]+=num
            check(len(best[k])==min(4,num),"Independent per-bin cap")
            for entry in best[k]:
                expected=expected_edge(video,local,ordered,k,entry,part,source,role)
                exact(next_required(edges,"Missing retained edge"),expected,"Edge differs: hash minimum, membership, timing, reference complement or role")
                verified_edges+=1;counts["retained_edges"]+=1;counts["retained_edges_lag_bin_%d"%k]+=1
                counts["edges_with_disjoint_reference_support"]+=int(n>=6)
                by_role[role]["retained_edges"]+=1;by_source[source]["retained_edges"]+=1
        if track_number%10000==0:
            print(json.dumps(dict(status="INDEPENDENT_FULL_PAIR_HASH_PROGRESS",tracks=track_number,total_tracks=len(grouped),candidate_hashes=verified_candidates,edges=verified_edges,elapsed_seconds=time.monotonic()-started)),flush=True)
    check(next(contexts,None) is None and next(edges,None) is None,"Extra/duplicate output track or edge")
    check(verified_candidates==combinations,"Partial exhaustive hash verification")
    exact(receipt["counters"],dict(counts),"Independent aggregate counters")
    exact(receipt["split_observation_counts"],dict(split_counts),"Split observation totals")
    exact(receipt["source_role_counts"],dict(Counter(r["role"] for r in roles.values())),"Source role counts")
    exact(receipt["per_role"],{k:dict(v) for k,v in by_role.items()},"Per-role output totals")
    exact(receipt["per_source"],{k:dict(v) for k,v in by_source.items()},"Per-source output totals")
    check(receipt["source_groups_with_edges"]==len(by_source),"Source edge coverage")
    for path,digest in watched.items():
        check(sha(path)==digest,"Input/output/verifier changed during audit: "+path)
    report=dict(status="PASS_INDEPENDENT_FULL_GT_FREE_PAIR_CONTEXT_AUDIT",pair_context=str(directory),
                pair_context_receipt_sha256=sha(receipt_path),producer_sha256=PRODUCER_SHA,
                independent_verifier_sha256=sha(__file__),implementation_preflight_sha256=sha(PREFLIGHT),
                protocol_sha256=PROTOCOL_SHA,accepted_manifest_sha256=MANIFEST_SHA,full_plan_sha256=PLAN_SHA,
                output_files_sha256=output_hashes,all_emitted_observations_verified=len(seen),all_emitted_tracks_verified=len(grouped),
                source_groups_roles_verified=len(roles),supported_tracks_verified=counts["supported_tracks"],
                candidate_hashes_independently_recomputed=verified_candidates,retained_edges_verified=verified_edges,
                every_real_supported_track_minimum_hash_fully_verified=True,every_edge_complete_reference_complement_hash_verified=True,
                all_endpoint_membership_times_source_roles_and_caps_verified=True,counters_recomputed=dict(counts),
                source_role_counts=receipt["source_role_counts"],max_track_observations=max_track_length,
                elapsed_seconds=time.monotonic()-started,real_NPZ_opened=False,supervision_or_correctness_opened=False,
                model_imported_or_trained=False,GPU_used=False,
                scope="All real upstream index/track rows and all output roles/contexts/edges; every candidate hash recomputed, not sampled.",
                limitations=["This audits the compact GT-free reference index, not posterior pooling, split-reference assignments or feature values.",
                             "Endpoint supervision eligibility and same-official-identity joins remain unopened; no pair learner, efficacy or novelty is established.",
                             "Producer historical IO claims rely on the reviewed exact script hash; this verifier itself opens only listed JSON/JSONL and source/hash files."])
    with destination.open("x") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2),flush=True)


def self_test():
    rng=random.Random(20261023);cases=0
    for n in range(45):
        step=0;ordered=[];ids=rng.sample(range(1000,100000),n)
        for i in range(n):
            step+=rng.randrange(1,15);ordered.append((step,ids[i],i,step*13))
        counts,best=minima(1,9,ordered);brute=[[],[],[],[]]
        if n>=4:
            for i,j in itertools.combinations(range(n),2):
                a,b=ordered[i],ordered[j];lag=b[0]-a[0]
                k=next(k for k,(lo,hi) in enumerate(((1,2),(3,5),(6,10),(11,float("inf")))) if lo<=lag<=hi)
                first,last=sorted((a[1],b[1]));key=hashlib.sha256(("pair_edges_v1|1|9|%d|%d"%(first,last)).encode()).hexdigest()
                brute[k].append((key,i,j))
        exact(counts,[len(b) for b in brute],"Synthetic candidate counts")
        exact(best,[sorted(b)[:4] for b in brute],"Synthetic sorted minima")
        for k,entries in enumerate(best):
            for entry in entries:
                edge=expected_edge(1,9,ordered,k,entry,"fit","A/g","mean_role")
                refs=[r[1] for r in ordered if r[1] not in (edge["observation_i"],edge["observation_j"])]
                check(edge["reference_observation_ids_sha256"]==array_hash(refs),"Synthetic endpoint exclusion")
        cases+=1
    records={1:dict(source_group="A/g1",split="fit"),2:dict(source_group="A/g2",split="fit"),
             3:dict(source_group="B/unsupported",split="fit"),4:dict(source_group="A/tune",split="tune")}
    role=roles_from_metadata(records,{i:dict(video_id=i) for i in records})
    check(len(role)==4 and role["B/unsupported"]["role"]=="mean_role" and role["A/tune"]["role"]=="tune_evaluation","Synthetic metadata-only roles")
    print(json.dumps(dict(status="PASS_INDEPENDENT_PAIR_CONTEXT_VERIFIER_SYNTHETIC",track_cases=cases,
          exhaustive_sorted_hash_minima=True,reference_complement=True,metadata_roles_include_unsupported_groups=True,
          producer_imported=False,real_data_opened=False,verifier_sha256=sha(__file__)),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--pair-context",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        check(args.pair_context is not None,"Provide --pair-context or --self-test")
        directory=args.pair_context.resolve();verify(directory,args.output or directory/"independent_pair_context_audit.json")


if __name__=="__main__":
    main()
