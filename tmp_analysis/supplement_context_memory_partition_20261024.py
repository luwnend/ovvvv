"""Independent ID-renaming invariant partition comparison; no model/GT IO."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from verify_fit_identity_edge_support_output import project

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/context_feature_memory_smoke_20261024_v1"
SUPPORT=ROOT/"results/base_error_full_collection_20261023/support_v1"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose2(n):return n*(n-1)//2


def pair_difference(first,second):
    assert len(first)==len(second)
    def keys(ids):return [("neg",i) if k<0 else ("id",k) for i,k in enumerate(ids)]
    a,b=keys(first),keys(second)
    count_a,count_b,table=Counter(a),Counter(b),Counter(zip(a,b))
    same_a=sum(map(choose2,count_a.values()));same_b=sum(map(choose2,count_b.values()))
    same_both=sum(map(choose2,table.values()))
    return dict(observations=len(a),all_unordered_pairs=choose2(len(a)),
                released_same_pairs=same_a,primary_same_pairs=same_b,
                released_same_primary_different=same_a-same_both,
                released_different_primary_same=same_b-same_both,
                pair_disagreements=same_a+same_b-2*same_both,
                partition_equal=same_a==same_b==same_both)


def main():
    paths=[OUT/"released.jsonl",OUT/"raw_confidence.jsonl",OUT/"independent_output_audit.json",
           SUPPORT/"supervised_observations.jsonl",SUPPORT/"gt_free_observations.jsonl"]
    before={str(p):sha(p) for p in paths}
    fit_ids=set()
    for line in paths[3].open():
        split=project(line,{"split"})["split"]
        if split=="tune":continue
        assert split=="fit"
        row=project(line,{"observation_id","eligible_for_parameter_fit"})
        assert row["eligible_for_parameter_fit"] is True
        fit_ids.add(row["observation_id"])
    known=set()
    for line in paths[4].open():
        row=json.loads(line)
        if row["observation_id"] in fit_ids:known.add((row["image_id"],row["dump_row"]))
    released=[json.loads(line) for line in paths[0].open()]
    primary=[json.loads(line) for line in paths[1].open()]
    all_groups=defaultdict(lambda:[[],[]]);known_groups=defaultdict(lambda:[[],[]]);known_count=0
    for a,b in zip(released,primary):
        assert a.keys()==b.keys() and all(a[k]==b[k] for k in a if k!="ids")
        vid=a["video_id"];assert len(a["ids"])==len(b["ids"])
        for j,(x,y) in enumerate(zip(a["ids"],b["ids"])):
            all_groups[vid][0].append(x);all_groups[vid][1].append(y)
            if (a["image_id"],j) in known:
                known_groups[vid][0].append(x);known_groups[vid][1].append(y);known_count+=1
    details={"all_retained":{str(v):pair_difference(*groups) for v,groups in sorted(all_groups.items())},
             "known_eligible_fit_manual_anchors":{str(v):pair_difference(*groups) for v,groups in sorted(known_groups.items())}}
    aggregate={}
    for scope,rows in details.items():
        aggregate[scope]={k:sum(r[k] for r in rows.values()) for k in next(iter(rows.values())) if k!="partition_equal"}
        aggregate[scope]["videos"]=len(rows)
        aggregate[scope]["videos_with_partition_difference"]=sum(not r["partition_equal"] for r in rows.values())
    assert known_count==924
    assert before=={p:sha(p) for p in before}
    result=dict(status="PASS_INDEPENDENT_CONTEXT_MEMORY_PARTITION_SUPPLEMENT",compared_modes=["released","raw_confidence"],
                aggregates=aggregate,per_video=details,input_sha256=before,
                interpretation="Pair relations are invariant to positive local-ID renaming; known anchors are a sparse fixed subset and unknown identity is not a negative",
                GT_identity_or_category_values_decoded=False,real_model_opened=False,GPU=False,
                script_sha256=sha(__file__))
    target=OUT/"independent_primary_partition_supplement.json"
    with target.open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(status=result["status"],aggregates=aggregate)))


if __name__=="__main__":main()
