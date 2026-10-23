#!/usr/bin/env python3
"""Unified labels-only evaluation AFTER all fifteen OOF predictions are sealed.

All retained candidate scores already exist. Unknown identity is never negative;
no-positive known subsets report coverage only, never unmatched training labels.
"""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COORDINATOR=ROOT/"tmp_analysis/run_identity_spacetime_oof_20261024.py"
PARENT=ROOT/"results/fit_lag_identity_support_20261024_v1"
FEATURES=("raw_appearance_cosine","adapter_appearance_cosine","A0G0","A0G1","A0Gmask","A0Gperm","A1G0","A1G1","A1Gmask","A1Gperm")
SEEDS=(20261024,20261025,20261026)
SCOPES=("all_candidates","same_predicted_category")
POPS=("all_known_population","common_CV_covered_population")
SUBSETS=("all_nonempty_known_queries","positive_and_negative_available_queries")
PRIMARY=(SCOPES[1],POPS[1])
COMPARISONS=(("adapter_cosine_minus_raw",0,1),("A0G1_minus_A0G0",2,3),("A1G1_minus_A1G0",6,7),("A0G1_minus_A0Gmask",4,3),("A0G1_minus_A0Gperm",5,3),("A1G1_minus_A1Gmask",8,7),("A1G1_minus_A1Gperm",9,7))


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open("x") as f:f.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def module(path,name,expected=None):
    if expected is not None:require(sha(path)==expected,"Module pin changed")
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=result;spec.loader.exec_module(result);return result


def open_database(path):
    db=sqlite3.connect(path);db.execute("PRAGMA temp_store=FILE");db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE query_support(seed INTEGER,source TEXT,lag INTEGER,scope TEXT,population TEXT,video INTEGER,image INTEGER,qrow INTEGER,qgt INTEGER,p INTEGER,n INTEGER,raw_correct INTEGER,correct_bits INTEGER,gperm_changed INTEGER,PRIMARY KEY(seed,lag,scope,population,video,image,qrow))")
    db.execute("CREATE INDEX query_source ON query_support(seed,lag,scope,population,source,video,qgt)")
    return db


def fresh_cell():
    return dict(candidate_pairs=0,known_pairs=0,positive_pairs=0,negative_pairs=0,unknown_pairs=0,
        nonempty_known_queries=0,positive_available_queries=0,negative_available_queries=0,hard_queries=0,
        gperm_mapping_nonidentity_candidates=0,gperm_geometry_changed_candidates=0,
        top1={subset:{feature:dict(raw_wrong_feature_wrong=0,raw_wrong_feature_correct=0,raw_correct_feature_wrong=0,raw_correct_feature_correct=0) for feature in FEATURES} for subset in SUBSETS},
        pair_order={feature:np.zeros((3,3),np.int64) for feature in FEATURES})


def update_cell(cell,scores,mask,qids,rids,source,seed,lag,scope,pop,video,image,db,changed=None,moved=None):
    nq,nm=mask.shape;known=(qids[:,None]>=0)&(rids[None,:]>=0);selected=mask&known
    positive=qids[:,None]==rids[None,:]
    cell["candidate_pairs"]+=int(mask.sum());cell["known_pairs"]+=int(selected.sum());cell["positive_pairs"]+=int((selected&positive).sum())
    cell["negative_pairs"]+=int((selected&~positive).sum());cell["unknown_pairs"]+=int((mask&~known).sum())
    changed=np.zeros_like(mask) if changed is None else changed;moved=np.zeros_like(mask) if moved is None else moved
    require(changed.shape==moved.shape==mask.shape,"Gperm coverage mapping differs")
    cell["gperm_mapping_nonidentity_candidates"]+=int((mask&moved).sum());cell["gperm_geometry_changed_candidates"]+=int((mask&changed).sum())
    for qr in range(nq):
        refs=np.flatnonzero(selected[qr])
        if not len(refs):continue
        truth=positive[qr,refs];values=scores[qr,refs];p=int(truth.sum());n=len(refs)-p
        require(values.shape==(len(refs),10) and np.isfinite(values).all(),"Incomplete finite score columns")
        correct=[bool(truth[np.lexsort((refs,-values[:,c]))[0]]) for c in range(10)]
        bits=sum((1<<c)*int(value) for c,value in enumerate(correct));is_changed=bool(np.any(mask[qr]&changed[qr]))
        db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(seed,source,lag,scope,pop,video,image,qr,int(qids[qr]),p,n,int(correct[0]),bits,int(is_changed)))
        cell["nonempty_known_queries"]+=1;cell["positive_available_queries"]+=int(p>0);cell["negative_available_queries"]+=int(n>0);cell["hard_queries"]+=int(p>0 and n>0)
        for subset in SUBSETS[:1]+(SUBSETS[1:] if p and n else ()):
            for c,feature in enumerate(FEATURES):
                key="raw_"+("correct" if correct[0] else "wrong")+"_feature_"+("correct" if correct[c] else "wrong")
                cell["top1"][subset][feature][key]+=1
        if p and n:
            signs=np.sign(values[truth,None,:]-values[None,~truth,:]).reshape(-1,10).astype(np.int8)+1
            for c,feature in enumerate(FEATURES):cell["pair_order"][feature]+=np.bincount(3*signs[:,0]+signs[:,c],minlength=9).reshape(3,3)


def combine_cells(cells):
    result=fresh_cell()
    for cell in cells:
        for key in result:
            if key=="top1":
                for subset in SUBSETS:
                    for feature in FEATURES:
                        for count,value in cell[key][subset][feature].items():result[key][subset][feature][count]+=value
            elif key=="pair_order":
                for feature in FEATURES:result[key][feature]+=cell[key][feature]
            else:result[key]+=cell[key]
    return result


def finalize_cell(cell):
    result={k:v for k,v in cell.items() if k not in ("top1","pair_order")};result["top1"]={}
    for subset in SUBSETS:
        result["top1"][subset]={}
        for feature in FEATURES:
            value=dict(cell["top1"][subset][feature]);value.update(repair_raw_error=value["raw_wrong_feature_correct"],break_raw_correct=value["raw_correct_feature_wrong"],
                raw_correct=value["raw_correct_feature_wrong"]+value["raw_correct_feature_correct"],feature_correct=value["raw_wrong_feature_correct"]+value["raw_correct_feature_correct"])
            result["top1"][subset][feature]=value
    result["positive_negative_pair_order"]={feature:dict(comparisons=int(cell["pair_order"][feature].sum()),raw_to_feature_loss_tie_win_matrix=cell["pair_order"][feature].tolist()) for feature in FEATURES}
    return result


def select_where(seed,lag,scope,pop,sources=None,hard=False,changed=False):
    where="seed=? AND lag=? AND scope=? AND population=?";args=[seed,lag,scope,pop]
    if hard:where+=" AND p>0 AND n>0"
    if changed:where+=" AND gperm_changed=1"
    if sources is not None:
        if not sources:where+=" AND 0"
        else:where+=" AND source IN ("+",".join("?" for _ in sources)+")";args+=list(sources)
    return where,args


def query_statistics(db,seed,lag,scope,pop,sources=None,hard=False,changed=False):
    where,args=select_where(seed,lag,scope,pop,sources,hard,changed)
    sums=",".join("SUM((correct_bits>>%d)&1)"%c for c in range(10))
    rows=list(db.execute("SELECT source,video,qgt,COUNT(*),"+sums+" FROM query_support WHERE "+where+" GROUP BY source,video,qgt ORDER BY source,video,qgt",args))
    grouped=defaultdict(list)
    for row in rows:grouped[row[0]].append(row)
    total=sum(row[3] for row in rows);result=dict(queries=total,sources=len(grouped),query_GT_trajectories=len(rows),features={})
    for c,feature in enumerate(FEATURES):
        correct=sum(row[4+c] for row in rows);errors=total-correct
        source_errors=[sum(row[3]-row[4+c] for row in group) for group in grouped.values()]
        means=[np.mean([row[4+c]/row[3] for row in group]) for group in grouped.values()]
        result["features"][feature]=dict(correct=correct,errors=errors,micro_correctness=correct/total if total else None,
            source_GTtrajectory_query_macro_correctness=float(np.mean(means)) if means else None,
            source_query_macro_correctness=float(np.mean([sum(r[4+c] for r in group)/sum(r[3] for r in group) for group in grouped.values()])) if grouped else None,
            GTtrajectory_macro_correctness=float(np.mean([row[4+c]/row[3] for row in rows])) if rows else None,
            error_sources=sum(x>0 for x in source_errors),error_GT_trajectories=sum(row[3]>row[4+c] for row in rows),
            largest_source_error_share=max(source_errors,default=0)/errors if errors else None,
            largest_GTtrajectory_error_share=max((row[3]-row[4+c] for row in rows),default=0)/errors if errors else None)
    return result


def comparison_statistics(db,seed,lag,scope,pop,sources=None,hard=False,changed=False,query_stats=None):
    where,args=select_where(seed,lag,scope,pop,sources,hard,changed)
    rows=list(db.execute("SELECT source,video,qgt,correct_bits FROM query_support WHERE "+where+" ORDER BY source,video,image,qrow",args))
    result={};stats=query_statistics(db,seed,lag,scope,pop,sources,hard,changed) if query_stats is None else query_stats
    for name,a,b in COMPARISONS:
        transition=Counter((bool(bits&(1<<a)),bool(bits&(1<<b))) for _,_,_,bits in rows)
        fa,fb=stats["features"][FEATURES[a]],stats["features"][FEATURES[b]]
        result[name]=dict(queries=len(rows),baseline_wrong_alternative_correct=transition[False,True],baseline_correct_alternative_wrong=transition[True,False],
            both_correct=transition[True,True],both_wrong=transition[False,False],net_correct=fb["correct"]-fa["correct"],
            micro_correctness_delta=None if not rows else fb["micro_correctness"]-fa["micro_correctness"],
            source_GTtrajectory_query_macro_delta=None if not rows else fb["source_GTtrajectory_query_macro_correctness"]-fa["source_GTtrajectory_query_macro_correctness"])
    histogram=Counter();groups=defaultdict(lambda:defaultdict(list))
    for source,video,gt,bits in rows:
        value=int(bool(bits&(1<<7)))-int(bool(bits&(1<<6)))-int(bool(bits&(1<<3)))+int(bool(bits&(1<<2)))
        histogram[value]+=1;groups[source][(video,gt)].append(value)
    means=[np.mean([np.mean(values) for values in trajectories.values()]) for trajectories in groups.values()]
    result["appearance_geometry_2x2_interaction"]=dict(queries=len(rows),query_value_histogram={str(i):histogram[i] for i in range(-2,3)},
        micro_mean=float(sum(k*v for k,v in histogram.items())/len(rows)) if rows else None,source_GTtrajectory_query_macro_mean=float(np.mean(means)) if means else None,
        formula="(A1G1-A1G0)-(A0G1-A0G0) in paired query correctness")
    return result


def decorate_cell(db,cell,seed,lag,scope,pop,sources=None):
    result=finalize_cell(cell);result["query_statistics"]={};result["comparisons"]={}
    for subset,hard in zip(SUBSETS,(False,True)):
        stats=query_statistics(db,seed,lag,scope,pop,sources,hard);result["query_statistics"][subset]=stats
        result["comparisons"][subset]=comparison_statistics(db,seed,lag,scope,pop,sources,hard,query_stats=stats)
    result["Gperm_actual_changed_GTfree_queries"]={subset:dict(query_statistics=query_statistics(db,seed,lag,scope,pop,sources,hard,True),
        comparisons=comparison_statistics(db,seed,lag,scope,pop,sources,hard,True)) for subset,hard in zip(SUBSETS,(False,True))}
    return result


def cross_lag(db,seed,scope,pop,sources=None):
    where="a.seed=? AND a.scope=? AND a.population=? AND a.lag=1 AND b.lag=4 AND a.p>0 AND a.n>0 AND b.p>0 AND b.n>0";args=[seed,scope,pop]
    if sources is not None:
        if not sources:where+=" AND 0"
        else:where+=" AND a.source IN ("+",".join("?" for _ in sources)+")";args+=list(sources)
    rows=list(db.execute("SELECT a.source,a.video,a.qgt,a.correct_bits,b.correct_bits FROM query_support a JOIN query_support b ON a.seed=b.seed AND a.video=b.video AND a.image=b.image AND a.qrow=b.qrow AND a.scope=b.scope AND a.population=b.population WHERE "+where,args))
    result=dict(shared_hard_queries=len(rows),sources=len({r[0] for r in rows}),features={})
    for c,feature in enumerate(FEATURES):
        counts=Counter((bool(a&(1<<c)),bool(b&(1<<c))) for _,_,_,a,b in rows)
        groups=defaultdict(lambda:defaultdict(list))
        for source,video,gt,a,b in rows:groups[source][(video,gt)].append((int(bool(a&(1<<c))),int(bool(b&(1<<c)))))
        means=[]
        for trajectories in groups.values():means.append(np.mean([np.mean(v,axis=0) for v in trajectories.values()],axis=0))
        macro=np.mean(means,axis=0).tolist() if means else [None,None]
        result["features"][feature]=dict(lag1_wrong_lag4_wrong=counts[False,False],lag1_wrong_lag4_correct=counts[False,True],
            lag1_correct_lag4_wrong=counts[True,False],lag1_correct_lag4_correct=counts[True,True],
            error_lag1=counts[False,False]+counts[False,True],error_lag4=counts[False,False]+counts[True,False],
            source_GTtrajectory_query_macro_lag1=macro[0],source_GTtrajectory_query_macro_lag4=macro[1])
    return result


def gperm_coverage(query,reference,normalizer,geometry,modules):
    nq=len(query["boxes"]);nm=len(reference["boxes"])
    before=np.array([np.nan if x is None else x for x in reference["previous_times"]],float)
    previous=np.array([[np.nan]*4 if x is None else x for x in reference["previous_boxes"]],float).reshape(nm,4)
    current=np.full(nm,reference["original_frame_index"])
    geom=geometry.pair_geometry(np.array(query["boxes"],float).reshape(nq,4),np.array(reference["boxes"],float).reshape(nm,4),query["original_frame_index"],current,previous,before)
    values,masks=modules.geometry_slots(geom,current,before);normalized=normalizer.transform(values,masks)
    plan=modules.gperm_donors(reference["predicted_categories"],np.isfinite(before),np.arange(nm))
    permuted,_,_=modules.intervene_geometry(normalized,masks,"Gperm",plan)
    changed=np.any(permuted[:,:,modules.POSITION_SLOTS]!=normalized[:,:,modules.POSITION_SLOTS],axis=2)
    moved=np.broadcast_to(plan.donor_rows!=np.arange(nm),(nq,nm))
    return changed,moved


def score_models(out,db,anchors,sources,spec,manifest,parent_out=None,gperm_fn=None):
    cells={};counts=Counter();maxread=0;parent_blocks=0;seen=set()
    for seed in spec["seeds"]:
        for fold in spec["folds"]:
            model=out/"models"/f"fold{fold}_seed{seed}";video=None;metadata=None
            held=set(manifest["folds"][str(fold)]["held_out_sources"])
            with (model/"frame_pairs.jsonl").open() as stream:
                for line in stream:
                    receipt=json.loads(line);source=receipt.get("source",receipt.get("source_group"));v=receipt["video_id"];lag=receipt["lag"]
                    require(source in held and source in sources and lag in (1,4),"OOF source/lag membership differs")
                    pair_key=(seed,v,lag,receipt["query_step"]);require(pair_key not in seen,"Duplicate OOF frame pair")
                    seen.add(pair_key)
                    if v!=video:metadata=json.loads((model/"shards"/f"v{v}.rows.json").read_text())["frames"];video=v
                    q=metadata[receipt["query_step"]];nq=receipt["query_rows"];nm=receipt["reference_rows"]
                    require(len(q["boxes"])==nq and q["image_id"]==receipt["query_image_id"],"OOF query rowmap differs")
                    for scope in SCOPES:
                        for pop in POPS:cells.setdefault((seed,source,lag,scope,pop),fresh_cell())
                    counts[(seed,lag)]+=receipt["candidates"];counts[(seed,"receipts")]+=1
                    if receipt["reference_step"] is None:
                        require(receipt["query_step"]<lag and nm==0 and receipt["candidates"]==0,"Malformed prefix receipt")
                        continue
                    require(receipt["query_step"]-receipt["reference_step"]==lag,"Nonexact sampled lag")
                    r=metadata[receipt["reference_step"]]
                    require(len(r["boxes"])==nm and r["image_id"]==receipt["reference_image_id"] and receipt["candidates"]==nq*nm,"OOF reference rowmap differs")
                    stored=np.load(model/receipt["score_file"],mmap_mode="r",allow_pickle=False)
                    block=stored[receipt["offset"]:receipt["offset"]+nq*nm].copy().reshape(nq,nm,10);del stored
                    require(block.dtype==np.float64 and np.isfinite(block).all(),"OOF scores must be ten finite float64 columns")
                    maxread=max(maxread,nq*nm)
                    if parent_out is not None:
                        parent=np.load(parent_out/receipt["score_file"],mmap_mode="r",allow_pickle=False)
                        require(np.array_equal(block[:,:,0],parent[receipt["offset"]:receipt["offset"]+nq*nm,0].reshape(nq,nm)),"Raw OOF scores differ from pinned parent")
                        del parent;parent_blocks+=1
                    def ids(frame):
                        result=[]
                        for row in range(len(frame["boxes"])):
                            value=anchors.get((frame["image_id"],row));require(value is None or value[0]==v,"Identity video crossed")
                            result.append(-1 if value is None else value[1])
                        return np.array(result,np.int64)
                    qi,ri=ids(q),ids(r);cv=np.array([x is not None for x in r["previous_times"]],bool)
                    same=np.array(q["predicted_categories"])[:,None]==np.array(r["predicted_categories"])[None,:]
                    changed,moved=gperm_fn(model,q,r) if gperm_fn is not None else (np.zeros((nq,nm),bool),np.zeros((nq,nm),bool))
                    for scope,selected in ((SCOPES[0],np.ones((nq,nm),bool)),(SCOPES[1],same)):
                        for pop,mask in ((POPS[0],selected),(POPS[1],selected&cv[None,:])):
                            update_cell(cells[(seed,source,lag,scope,pop)],block,mask,qi,ri,source,seed,lag,scope,pop,v,q["image_id"],db,changed,moved)
                    db.commit()
    for seed in spec["seeds"]:
        for source in sources:
            for lag in (1,4):
                for scope in SCOPES:
                    for pop in POPS:cells.setdefault((seed,source,lag,scope,pop),fresh_cell())
    return cells,dict(maximum_score_rows_read=maxread,exact_parent_raw_blocks=parent_blocks,
        candidates_by_seed={str(seed):{str(lag):counts[seed,lag] for lag in (1,4)} for seed in spec["seeds"]},
        frame_pair_receipts_by_seed={str(seed):counts[seed,"receipts"] for seed in spec["seeds"]})


def source_deletion(per_source,lag):
    key=str(lag);reports={};valid={}
    for source,row in per_source.items():
        cell=row["lags"][key][PRIMARY[0]][PRIMARY[1]];valid[source]=cell["query_statistics"][SUBSETS[1]]
    for deleted in per_source:
        remaining=[r for s,r in valid.items() if s!=deleted and r["queries"]]
        n=sum(r["queries"] for r in remaining);features={}
        for feature in FEATURES:
            correct=sum(r["features"][feature]["correct"] for r in remaining)
            means=[r["features"][feature]["source_GTtrajectory_query_macro_correctness"] for r in remaining]
            features[feature]=dict(correct=correct,micro=correct/n if n else None,macro=float(np.mean(means)) if means else None)
        comparisons={name:dict(micro_delta=None if not n else features[FEATURES[b]]["micro"]-features[FEATURES[a]]["micro"],
            source_GTtrajectory_query_macro_delta=None if not n else features[FEATURES[b]]["macro"]-features[FEATURES[a]]["macro"]) for name,a,b in COMPARISONS}
        comparisons["appearance_geometry_2x2_interaction"]=dict(micro_delta=None if not n else features["A1G1"]["micro"]-features["A1G0"]["micro"]-features["A0G1"]["micro"]+features["A0G0"]["micro"],
            source_GTtrajectory_query_macro_delta=None if not n else features["A1G1"]["macro"]-features["A1G0"]["macro"]-features["A0G1"]["macro"]+features["A0G0"]["macro"])
        reports[deleted]=dict(remaining_queries=n,comparisons=comparisons)
    return reports


def summarize(db,cells,roles,seeds):
    sources=sorted(roles);result={}
    for seed in seeds:
        reports={};union={};role_reports={};block_reports={}
        for source in sources:
            report=dict(role=roles[source]["role"],dataset_block=roles[source]["block"],lags={})
            for lag in (1,4):
                report["lags"][str(lag)]={scope:{pop:decorate_cell(db,cells[(seed,source,lag,scope,pop)],seed,lag,scope,pop,[source]) for pop in POPS} for scope in SCOPES}
            report["cross_lag"]={scope:{pop:cross_lag(db,seed,scope,pop,[source]) for pop in POPS} for scope in SCOPES};reports[source]=report
        def population_report(selected):
            return {str(lag):{scope:{pop:decorate_cell(db,combine_cells(cells[(seed,s,lag,scope,pop)] for s in selected),seed,lag,scope,pop,selected) for pop in POPS} for scope in SCOPES} for lag in (1,4)}
        union=population_report(sources)
        for role in sorted({r["role"] for r in roles.values()}):role_reports[role]=population_report([s for s in sources if roles[s]["role"]==role])
        for block in sorted({r["block"] for r in roles.values()}):block_reports[block]=population_report([s for s in sources if roles[s]["block"]==block])
        result[str(seed)]=dict(micro_union=union,per_source=reports,per_role=role_reports,per_dataset_block=block_reports,
            cross_lag={scope:{pop:cross_lag(db,seed,scope,pop) for pop in POPS} for scope in SCOPES},
            source_deletion_sensitivity={str(lag):source_deletion(reports,lag) for lag in (1,4)})
    return dict(per_seed=result,macro_definition="Equal supported source -> equal official query GTtrajectory -> equal query observation; zero-support sources remain in reports but contribute no numerical macro.",
        Gperm_changed_definition="Any actually changed geometric value in the complete retained GT-free scope/population for this query; defined before known-label projection, never by GT or gain.")


def baseline_parity(results,parent,seeds,expected_shared=1331):
    keys=("candidate_pairs","known_pairs","positive_pairs","negative_pairs","unknown_pairs","nonempty_known_queries","positive_available_queries","negative_available_queries","hard_queries")
    checked=0
    for seed in seeds:
        report=results["per_seed"][str(seed)]
        for source in [None]+sorted(report["per_source"]):
            a=report["micro_union"] if source is None else report["per_source"][source]["lags"]
            b=parent["results"]["micro_union"] if source is None else parent["results"]["per_source"][source]["lags"]
            for lag in (1,4):
                for scope in SCOPES:
                    for pop in POPS:
                        x,y=a[str(lag)][scope][pop],b[str(lag)][scope][pop]
                        require(all(x[k]==y[k] for k in keys),"Parent source/union population count changed")
                        require(all(x["top1"][subset][FEATURES[0]]==y["top1"][subset][FEATURES[0]] for subset in SUBSETS),"Parent raw paired top1 changed")
                        require(x["positive_negative_pair_order"][FEATURES[0]]==y["positive_negative_pair_order"][FEATURES[0]],"Parent raw pair-order changed")
                        checked+=1
        require(report["cross_lag"][PRIMARY[0]][PRIMARY[1]]["shared_hard_queries"]==expected_shared,"Fixed primary shared query population changed")
    return dict(status="PASS_EXACT_PARENT_RAW_SOURCE_UNION_COUNTS_TOP1_PAIRORDER",cells=checked,shared_hard_queries=expected_shared)


def seed_robustness(results,seeds):
    result={}
    for lag in (1,4):
        result[str(lag)]={}
        for name,_,_ in COMPARISONS:
            values=[results["per_seed"][str(seed)]["micro_union"][str(lag)][PRIMARY[0]][PRIMARY[1]]["comparisons"][SUBSETS[1]][name]["source_GTtrajectory_query_macro_delta"] for seed in seeds]
            deletions=[row["comparisons"][name]["source_GTtrajectory_query_macro_delta"] for seed in seeds for row in results["per_seed"][str(seed)]["source_deletion_sensitivity"][str(lag)].values()]
            valid=[x for x in values if x is not None];valid_deletions=[x for x in deletions if x is not None]
            result[str(lag)][name]=dict(seeds=list(seeds),source_macro_deltas=values,seed_mean_descriptive_only=float(np.mean(valid)) if valid else None,
                all_fixed_seeds_strictly_positive=bool(valid) and len(valid)==len(seeds) and all(x>0 for x in valid),
                minimum_source_deletion_macro_delta=min(valid_deletions) if valid_deletions else None,
                positive_direction_survives_all_source_deletions=bool(valid_deletions) and all(x>0 for x in valid_deletions))
    return result


def evaluate(out,protocol):
    out=Path(out);spec=json.loads(Path(protocol).read_text()) if isinstance(protocol,(str,Path)) else protocol
    # This is deliberately the first external module imported by the real entry.
    coordinator=module(spec.get("coordinator_path",COORDINATOR),"oof_evaluation_global_seal_gate",spec.get("coordinator_sha256"))
    global_seal=coordinator.verify_global_seal(out,spec)
    require(tuple(spec["feature_names"])==FEATURES and tuple(spec["seeds"])==SEEDS and tuple(spec["folds"])==tuple(range(5)),"Fixed evaluation columns/seeds/folds differ")
    data=module(spec["data_module_path"],"oof_unified_evaluation_data",spec["data_module_sha256"])
    prepared=data.prepare_inputs(spec);sources=sorted(prepared["manifest"]["groups"])
    anchors,label_receipt=data.load_train_anchors(sources,prepared["roles"],prepared["video_sources"],prepared["authority"])
    dest=out/"evaluation";require(not dest.exists(),"Preserve prior unified evaluation output");dest.mkdir()
    modules=data.component(data.MODULES);geometry=data.component(data.GEOMETRY);normalizers={}
    def permutation(model,q,r):
        if model not in normalizers:
            stored=json.loads((model/"normalizer.json").read_text());normalizers[model]=modules.GeometryNormalizer(np.array(stored["mean"]),np.array(stored["std"]),np.array(stored["valid_marginal_weight"]))
        return gperm_coverage(q,r,normalizers[model],geometry,modules)
    db=open_database(dest/"query_support.sqlite")
    cells,stream_receipt=score_models(out,db,anchors,sources,spec,prepared["manifest"],Path(spec.get("parent_score_directory",PARENT)),permutation)
    for seed in SEEDS:
        require(stream_receipt["candidates_by_seed"][str(seed)]==spec["resource_bounds"]["candidates_by_lag"] and stream_receipt["frame_pair_receipts_by_seed"][str(seed)]==2*spec["population"]["selected_frames"],"Incomplete full population OOF predictions")
    roles={s:prepared["roles"][s] for s in sources};results=summarize(db,cells,roles,SEEDS);db.commit();db.close()
    parent=json.loads((Path(spec.get("parent_score_directory",PARENT))/"summary.json").read_text())
    parity=baseline_parity(results,parent,SEEDS);results["seed_robustness"]=seed_robustness(results,SEEDS)
    coordinator.verify_global_seal(out,spec)
    summary=dict(status="COMPLETE_UNIFIED_ALL_SEALED_SOURCE_OOF_EVALUATION_PENDING_INDEPENDENT_AUDIT",results=results,baseline_parity=parity,
        stream_receipt=stream_receipt,eligible_all_FIT_manual_anchors=len(anchors),label_projection_receipt=label_receipt,
        script_sha256=sha(__file__),protocol_sha256=sha(protocol) if isinstance(protocol,(str,Path)) else data.canonical_sha(spec),
        global_seal_verified_before_identity_import_and_after=True,all_fifteen_models_sealed_before_evaluation=True,unknown_is_negative=False,
        no_positive_is_unmatched=False,training_modified=False,GPU_used=False,query_database_sha256=sha(dest/"query_support.sqlite"),
        interpretation="Exploratory source OOF on already examined FIT; sparse-known conditional ranking, ordinary controls, no official tracking or independent generalization claim.")
    write(dest/"summary.json",summary);return summary


def synthetic(number):
    checks=[]
    if number==1:
        with tempfile.TemporaryDirectory(prefix="oof_eval_top1_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell();scores=np.zeros((2,3,10))
            scores[0,:,0]=[.1,.9,999];scores[1,:,0]=[.1,.9,999]
            for c in range(1,10):scores[:,:,c]=scores[:,:,0][:,::-1]
            q=np.array([1,2]);r=np.array([1,2,-1]);mask=np.ones((2,3),bool)
            for seed in SEEDS:
                for lag in (1,4):update_cell(cell if seed==SEEDS[0] and lag==1 else fresh_cell(),scores,mask,q,r,"S",seed,lag,*PRIMARY,1,100,db)
            require(cell["known_pairs"]==4 and cell["unknown_pairs"]==2 and cell["positive_pairs"]==cell["negative_pairs"]==2,"Unknown became negative or known population differs")
            stats=query_statistics(db,SEEDS[0],1,*PRIMARY);require(stats["queries"]==2 and stats["features"][FEATURES[0]]["source_GTtrajectory_query_macro_correctness"]==.5,"Macro unit differs")
            require(cross_lag(db,SEEDS[0],*PRIMARY)["shared_hard_queries"]==2 and db.execute("SELECT COUNT(DISTINCT seed) FROM query_support").fetchone()[0]==3,"Seeds merged or same-query join differs")
            comparison=comparison_statistics(db,SEEDS[0],1,*PRIMARY)
            require(comparison["adapter_cosine_minus_raw"]["queries"]==2 and comparison["appearance_geometry_2x2_interaction"]["query_value_histogram"]["0"]==2,"Paired comparator/interaction oracle differs")
            db.close()
        checks.append("ten_column_bitsets_three_seed_isolation_unknown_notnegative_query_macro_crosslag_and_paired_interaction")
        with tempfile.TemporaryDirectory(prefix="oof_eval_ties_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell()
            update_cell(cell,np.zeros((1,2,10)),np.ones((1,2),bool),np.array([2]),np.array([1,2]),"S",SEEDS[0],1,*PRIMARY,1,101,db)
            require(all(v["raw_wrong_feature_wrong"]==1 for v in cell["top1"][SUBSETS[1]].values()),"Tie selected favourable GT")
            db.close()
        checks.append("lowest_immutable_reference_row_tie_for_all_ten_columns")
    elif number==2:
        with tempfile.TemporaryDirectory(prefix="oof_eval_population_") as temp:
            out=Path(temp);model=out/"models"/f"fold0_seed{SEEDS[0]}";(model/"shards").mkdir(parents=True)
            frames=[dict(video_id=1,image_id=100+k,sampled_step=k,original_frame_index=k*10,boxes=[[0,0,10,10],[20,0,30,10]],predicted_categories=[1,1],previous_times=[None,None] if k==0 else [0,0]) for k in range(6)]
            write(model/"shards/v1.rows.json",dict(frames=frames));receipts=[]
            for lag in (1,4):
                size=(6-lag)*4;np.save(model/"shards"/f"v1_lag{lag}.scores.npy",np.tile(np.array([[1.]*10,[0.]*10,[0.]*10,[1.]*10]),(6-lag,1)))
            parent=out/"parent";(parent/"shards").mkdir(parents=True)
            for lag in (1,4):
                data=np.load(model/"shards"/f"v1_lag{lag}.scores.npy");legacy=np.zeros((len(data),6));legacy[:,0]=data[:,0]
                np.save(parent/"shards"/f"v1_lag{lag}.scores.npy",legacy)
            offsets=Counter()
            for k in range(6):
                for lag in (1,4):
                    ref=k-lag if k>=lag else None;count=0 if ref is None else 4
                    receipts.append(dict(source="D/S",video_id=1,lag=lag,query_step=k,reference_step=ref,query_image_id=100+k,reference_image_id=None if ref is None else 100+ref,query_rows=2,reference_rows=0 if ref is None else 2,candidates=count,offset=offsets[lag],score_file=f"shards/v1_lag{lag}.scores.npy"));offsets[lag]+=count
            (model/"frame_pairs.jsonl").write_text("".join(json.dumps(x)+"\n" for x in receipts))
            anchors={(100+k,row):(1,row+1) for k in range(6) for row in range(2)}
            spec=dict(seeds=[SEEDS[0]],folds=[0]);manifest=dict(folds={"0":dict(held_out_sources=["D/S","D/ZERO"])})
            db=open_database(out/"query.sqlite");cells,receipt=score_models(out,db,anchors,["D/S","D/ZERO"],spec,manifest,parent)
            roles={s:dict(role="mean_role",block="D") for s in ("D/S","D/ZERO")};result=summarize(db,cells,roles,[SEEDS[0]])
            union=result["per_seed"][str(SEEDS[0])]["micro_union"]
            require(union["1"][SCOPES[0]][POPS[0]]["known_pairs"]==20 and union["1"][SCOPES[0]][POPS[1]]["known_pairs"]==16 and result["per_seed"][str(SEEDS[0])]["per_source"]["D/ZERO"]["lags"]["4"][SCOPES[1]][POPS[1]]["hard_queries"]==0,"Scope/population or zero-source dropped")
            require(result["per_seed"][str(SEEDS[0])]["cross_lag"][PRIMARY[0]][PRIMARY[1]]["shared_hard_queries"]==2 and receipt["frame_pair_receipts_by_seed"][str(SEEDS[0])]==12,"Exact commonCV shared query or receipts differ")
            require("D/ZERO" in result["per_seed"][str(SEEDS[0])]["source_deletion_sensitivity"]["1"],"Zero-source deletion report omitted")
            reference=copy.deepcopy(dict(results=dict(micro_union=union,per_source=result["per_seed"][str(SEEDS[0])]["per_source"])))
            parity=baseline_parity(result,reference,[SEEDS[0]],2)
            require(parity["cells"]==24 and receipt["exact_parent_raw_blocks"]==7,"Source/union or every-frame parent raw parity omitted")
            reference["results"]["micro_union"]["1"][SCOPES[0]][POPS[0]]["known_pairs"]+=1
            try:baseline_parity(result,reference,[SEEDS[0]],2)
            except ValueError:pass
            else:raise AssertionError("Accepted mismatched parent raw population")
            db.close()
            parent_data=np.load(parent/"shards/v1_lag1.scores.npy",mmap_mode="r+");parent_data[0,0]+=.1;parent_data.flush();del parent_data
            rejected_db=open_database(out/"rejected.sqlite")
            try:score_models(out,rejected_db,anchors,["D/S","D/ZERO"],spec,manifest,parent)
            except ValueError:pass
            else:raise AssertionError("Accepted changed parent raw candidate score")
            finally:rejected_db.close()
        checks.append("full_temporary_OOF_shards_receipts_two_scopes_two_populations_union_zero_sources_role_block_and_source_deletion")
        checks.append("every_candidate_parent_raw_exact_source_union_counts_top1_and_changed_score_count_rejection")
        with tempfile.TemporaryDirectory(prefix="oof_eval_changed_") as temp:
            db=open_database(Path(temp)/"support.sqlite");cell=fresh_cell();scores=np.zeros((1,3,10));scores[0,0]=1
            # Only unknown candidate changes: change eligibility remains GT-free.
            update_cell(cell,scores,np.ones((1,3),bool),np.array([1]),np.array([1,2,-1]),"S",SEEDS[0],1,*PRIMARY,1,100,db,np.array([[False,False,True]]),np.ones((1,3),bool))
            require(query_statistics(db,SEEDS[0],1,*PRIMARY,hard=True,changed=True)["queries"]==1 and cell["gperm_geometry_changed_candidates"]==1,"Gperm actual-change coverage defined using labels")
            db.close()
        checks.append("Gperm_actual_change_query_membership_uses_complete_GTfree_pool_including_unknown")
    elif number==3:
        with tempfile.TemporaryDirectory(prefix="oof_eval_gate_") as temp:
            out=Path(temp);coordinator=out/"gate.py";coordinator.write_text("def verify_global_seal(out,protocol):\n    raise ValueError('SYNTHETIC_UNSEALED')\n")
            try:evaluate(out,dict(coordinator_path=str(coordinator),data_module_path=str(out/"POISON_decoder_must_not_import.py")))
            except ValueError as error:require(str(error)=="SYNTHETIC_UNSEALED","Identity decoder preceded global gate")
            else:raise AssertionError("Unsealed evaluator entry accepted")
        checks.append("actual_evaluator_entry_global_seal_failure_precedes_data_identity_module_import")
        with tempfile.TemporaryDirectory(prefix="oof_eval_macro_") as temp:
            db=open_database(Path(temp)/"support.sqlite")
            # Source S: two trajectories (one with 3 correct queries; one wrong).
            # Source T: one wrong query. Macro=(.5+0)/2=.25; micro=3/5.
            for source,video,gt,image,correct in (("S",1,1,1,True),("S",1,1,2,True),("S",1,1,3,True),("S",1,2,4,False),("T",2,1,5,False)):
                bits=1023 if correct else 0
                db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(SEEDS[0],source,1,*PRIMARY,video,image,0,gt,1,1,int(correct),bits,0))
            stats=query_statistics(db,SEEDS[0],1,*PRIMARY,hard=True)
            require(stats["features"][FEATURES[0]]["source_GTtrajectory_query_macro_correctness"]==.25 and stats["features"][FEATURES[0]]["micro_correctness"]==.6,"Source/GTtrajectory/query macro collapsed to micro")
            require(query_statistics(db,SEEDS[0],1,*PRIMARY,["S"],True)["features"][FEATURES[0]]["source_GTtrajectory_query_macro_correctness"]==.5,"Source deletion recomputation oracle differs")
            db.execute("INSERT INTO query_support VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(SEEDS[0],"X",1,*PRIMARY,3,6,0,1,1,1,0,1<<7,0))
            interaction=comparison_statistics(db,SEEDS[0],1,*PRIMARY,["X"],True)["appearance_geometry_2x2_interaction"]
            require(interaction["query_value_histogram"]["1"]==1 and interaction["micro_mean"]==interaction["source_GTtrajectory_query_macro_mean"]==1.,"Nonzero paired interaction oracle differs")
            db.close()
        checks.append("unequal_source_trajectory_query_macro_exact_oracle_and_source_deletion_population")
    else:raise ValueError("Exactly three rounds")
    return dict(status="PASS_IDENTITY_SPACETIME_OOF_EVALUATOR_SYNTHETIC",round=number,checks=checks,script_sha256=sha(__file__),real_cache_or_identity_opened=False,training_started=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path);parser.add_argument("--protocol",type=Path);parser.add_argument("--synthetic-round",type=int,choices=(1,2,3));parser.add_argument("--synthetic-out",type=Path);args=parser.parse_args()
    if args.synthetic_round:
        result=synthetic(args.synthetic_round)
        if args.synthetic_out:write(args.synthetic_out,result)
    else:require(args.out and args.protocol,"Explicit out/protocol or synthetic round required");result=evaluate(args.out,args.protocol)
    print(json.dumps(dict(status=result["status"],script_sha256=sha(__file__)),indent=2))


if __name__=="__main__":main()
