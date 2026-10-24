#!/usr/bin/env python3
"""CPU-only history/position/locality decomposition of the ROI gate.

This is a diagnostic, not a trained model.  It uses the same 122 known
identity observations and fixed same-predicted-category cross-step candidate
pool as the frozen three-fold gate.  GT only defines evaluation labels after
candidate construction and never selects a history member or candidate.
"""
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/raid/datasets/chc_data/geometry_roi_full_20261024_v1')
GATE=ROOT/'results/geometry_roi_gate_manifest_20261024_v1/manifest.json'
OUT=ROOT/'results/geometry_roi_history_confound_20261024_v1'
FOLDS={}


def cos(a,b):
    a=np.asarray(a,np.float64).reshape(-1); b=np.asarray(b,np.float64).reshape(-1)
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))

def boxfeat(b,w,h):
    x1,y1,x2,y2=b; return np.array([(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h],np.float64)

def iou(a,b):
    aa=(a[2]-a[0])*(a[3]-a[1]); bb=(b[2]-b[0])*(b[3]-b[1]);
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(aa+bb-inter,1e-12)

def auroc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int)
    p=int(y.sum()); n=len(y)-p
    if not p or not n:return None
    o=np.argsort(s,kind='mergesort'); ss=s[o]; ranks=np.empty(len(s),float); i=0
    while i<len(ss):
        j=i+1
        while j<len(ss) and ss[j]==ss[i]:j+=1
        ranks[o[i:j]]=.5*(i+1+j); i=j
    return float((ranks[y==1].sum()-p*(p+1)/2)/(p*n))

def load():
    g=json.load(open(GATE)); lab={x['observation_id']:x['gt_track_id'] for x in g['evaluation_only_labels']}
    fmap={v['video_id']:f['fold'] for f in g['folds'] for v in f['videos']}
    obs={}
    for r in g['model_input']:
        p=CACHE/r['dump_file']
        with np.load(p,allow_pickle=False) as z:
            j=int(r['dump_row']); pre=json.loads(str(z['preprocessing_json'].item()))
            w=pre['img_shape'][1]; h=pre['img_shape'][0]
            obs[r['observation_id']]=dict(o=r,label=lab[r['observation_id']],fold=fmap[r['video_id']],raw=z['raw_output'][j].astype(float),tok=z['conv_spatial'][j].astype(float),box=z['det_bboxes'][j,:4].astype(float),boxn=boxfeat(z['det_bboxes'][j,:4].astype(float),w,h))
    return g,obs

def build_hist(obs):
    # group by source/video/local predicted track.  Current row never enters
    # its own descriptor.  Prior same-step rows are excluded from history.
    groups={}
    for o in obs.values():groups.setdefault((o['o']['source_group'],o['o']['video_id'],o['o']['local_track_id']),[]).append(o)
    for gg in groups.values():
        gg.sort(key=lambda x:(x['o']['sampled_step'],x['o']['dump_row']))
        for current in gg:
            past=[x for x in gg if x['o']['sampled_step']<current['o']['sampled_step']]
            current['hist']=past
            current['raw_hist_mean']=np.mean([x['raw'] for x in past],axis=0) if past else current['raw']
            current['tok_hist_mean']=np.mean([x['tok'] for x in past],axis=0) if past else current['tok']
            current['box_hist_mean']=np.mean([x['boxn'] for x in past],axis=0) if past else current['boxn']
            current['last_raw']=past[-1]['raw'] if past else current['raw']
            current['last_tok']=past[-1]['tok'] if past else current['tok']

ARMS={
'raw_cur':lambda q,r:cos(q['raw'],r['raw']),
'raw_last':lambda q,r:cos(q['last_raw'],r['last_raw']),
'raw_hist_mean':lambda q,r:cos(q['raw_hist_mean'],r['raw_hist_mean']),
'raw_cur_plus_hist_mean':lambda q,r:(cos(q['raw'],r['raw'])+cos(q['raw_hist_mean'],r['raw_hist_mean']))/2,
'tok_cur_mean':lambda q,r:cos(q['tok'].mean((1,2)),r['tok'].mean((1,2))),
'tok_last_mean':lambda q,r:cos(q['last_tok'].mean((1,2)),r['last_tok'].mean((1,2))),
'tok_hist_mean':lambda q,r:cos(q['tok_hist_mean'].mean((1,2)),r['tok_hist_mean'].mean((1,2))),
'raw_cur_box':lambda q,r:cos(q['raw'],r['raw'])-2*iou(q['box'],r['box']),
'raw_hist_box':lambda q,r:cos(q['raw_hist_mean'],r['raw_hist_mean'])-2*iou(q['box'],r['box']),
'tok_hist_box':lambda q,r:cos(q['tok_hist_mean'].mean((1,2)),r['tok_hist_mean'].mean((1,2)))-2*iou(q['box'],r['box']),
}

def evaluate(obs):
    by={}
    for o in obs.values():by.setdefault((o['o']['source_group'],o['o']['video_id']),[]).append(o)
    pair=[]
    for (s,v),gg in by.items():
        for i,q in enumerate(gg):
            for r in gg[i+1:]:
                if q['o']['sampled_step']==r['o']['sampled_step'] or q['label'] is None or r['label'] is None:continue
                same=bool(q['label']==r['label']); neg=(not same and q['o']['pred_category_id']==r['o']['pred_category_id'])
                if same or neg: pair.append((s,q['fold'],same,neg,q,r))
    out={}
    prim=[x for x in pair if x[2] or x[3]]
    for arm,fn in ARMS.items():
        per={}
        pooled=[]
        for s,fold,same,neg,q,r in prim:
            sc=fn(q,r);pooled.append((sc,same));per.setdefault((s,fold),[]).append((sc,same))
        vals=[auroc([a for a,b in v],[b for a,b in v]) for v in per.values()]
        per_source={}
        for (s,f),v in per.items():
            z=auroc([a for a,b in v],[b for a,b in v])
            if z is not None:per_source[s]=z
        out[arm]={'source_macro':float(np.mean(list(per_source.values()))) if per_source else None,'pairs':len(prim),'per_source':per_source,'pooled':auroc([a for a,b in pooled],[b for a,b in pooled])}
    # top1 same candidate pool
    query=[]
    for (s,v),gg in by.items():
        for q in gg:
            if q['label'] is None:continue
            cand=[r for r in gg if r['o']['observation_id']!=q['o']['observation_id'] and r['o']['sampled_step']!=q['o']['sampled_step'] and r['o']['pred_category_id']==q['o']['pred_category_id']]
            if any(r['label']==q['label'] for r in cand) and any(r['label'] is not None and r['label']!=q['label'] for r in cand):query.append((q,cand))
    for arm,fn in ARMS.items():
        ok=0; rec=[]
        for q,cand in query:
            sel=max(cand,key=lambda r:(fn(q,r),-r['o']['dump_row'],-r['o']['observation_id']))
            c=bool(sel['label']==q['label']);ok+=c;rec.append({'q':q['o']['observation_id'],'s':sel['o']['observation_id'],'correct':c,'source':q['o']['source_group']})
        out[arm]['top1']={'correct':ok,'queries':len(query),'accuracy':ok/len(query) if query else None,'records':rec}
    return out

def main():
    if OUT.exists():raise RuntimeError('preserve existing output; use a new version')
    g,obs=load(); build_hist(obs); result={'status':'DIAGNOSTIC_CPU_HISTORY_CONFOUND','rows':len(obs),'arms':evaluate(obs),'uses_gt_only_for_scoring':True,'training':False,'gpu_used':False}
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'macro':v.get('source_macro'),'top1':v.get('top1',{}).get('accuracy'),'correct':v.get('top1',{}).get('correct')} for k,v in result['arms'].items()},indent=2))
if __name__=='__main__':main()
