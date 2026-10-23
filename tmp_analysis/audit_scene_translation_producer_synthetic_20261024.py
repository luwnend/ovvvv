#!/usr/bin/env python3
"""Independent synthetic producer boundary, per-target and hierarchy audit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from unittest import mock

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def digest(value):
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return np.frombuffer(hashlib.sha256(encoded).digest(),dtype=np.uint8)


def exclusion(frame, row):
    result=[];target=frame['boxes'][row]
    for i,b in enumerate(frame['boxes']):
        same=frame['proposals'][i]==frame['proposals'][row]
        intersects=min(b[2],target[2])>max(b[0],target[0]) and min(b[3],target[3])>max(b[1],target[1])
        if i==row or same or intersects:
            result.append(i)
    return result


def check_loader(producer):
    allowed={'det_bboxes','track_feats','roi_proposal_row','video_id','image_id','frame_id','frame_index'}
    visited=[]
    values=dict(det_bboxes=np.asfortranarray(np.tile([0.,0.,10.,10.,np.nan],(5,1))),
                track_feats=np.asfortranarray(np.eye(256)[:5]),roi_proposal_row=np.arange(5,dtype=np.int64),
                video_id=np.int64(2),image_id=np.int64(3),frame_id=np.int64(1),frame_index=np.int64(30))
    class Poisoned:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            assert key in allowed,'Forbidden member decoded: '+key
            visited.append(key);return values[key]
    record=dict(npz='synthetic-poisoned-file',npz_sha256='fixed',observations=5,
                video_id=2,image_id=3,frame_id=1,frame_index=30)
    with mock.patch.object(producer,'sha',return_value='fixed'),mock.patch.object(producer.np,'load',return_value=Poisoned()):
        frame=producer.load_frame(record)
    assert set(visited)==allowed
    assert frame['boxes'].shape==(5,4) and np.isfinite(frame['boxes']).all()
    assert frame['boxes'].flags.c_contiguous and frame['raw'].flags.c_contiguous
    assert frame['boxes'].dtype==np.float64 and frame['raw'].dtype==np.float64
    # The forbidden detector score is NaN and does not enter prediction input.


def check_predictions(producer,engine,oracle):
    rng=np.random.RandomState(2026102401)
    count=0
    for n in (0,1,3,4,7,10):
        rxy=rng.uniform(0,100,(n,2));rbox=np.column_stack((rxy,rxy+10))
        qbox=rbox+np.tile([3.,-4.,3.,-4.],(n,1))
        qraw=np.eye(max(n,1),256)[:n];rraw=qraw.copy()
        proposals=np.arange(n)
        if n>=7:
            # Same proposal without overlap; direct overlap without same proposal;
            # touching-only row remains; no connected-component expansion.
            rbox[:4]=[[0,0,10,10],[20,0,30,10],[9,0,19,10],[10,10,20,20]]
            qbox[:4]=rbox[:4]+[3,-4,3,-4]
            proposals[1]=proposals[0]
        q=dict(boxes=np.ascontiguousarray(qbox),raw=np.ascontiguousarray(qraw),proposals=proposals.copy())
        r=dict(boxes=np.ascontiguousarray(rbox),raw=np.ascontiguousarray(rraw),proposals=proposals.copy())
        actual,locs,ah,eh=producer.predict_frame_pair(q,r,engine.CachedSceneMotion)
        full=oracle.oracle(q['boxes'],r['boxes'],q['raw'],r['raw'])
        assert actual.shape==(len(full['anchors']),37)
        assert locs.shape==(len(full['anchors']),2) and ah.shape==(len(full['anchors']),3,32)
        assert eh.shape==ah.shape
        for index,h in enumerate(full['anchors']):
            qi,ri=h['query_row'],h['reference_row']
            assert locs[index].tolist()==[qi,ri]
            qscale=np.sqrt(qbox[qi,2:]-qbox[qi,:2]);rscale=np.sqrt(rbox[ri,2:]-rbox[ri,:2])
            delta=(qbox[qi,:2]/2+qbox[qi,2:]/2)-(rbox[ri,:2]/2+rbox[ri,2:]/2)
            static=abs(delta/(qscale*rscale)).sum()
            np.testing.assert_allclose(actual[index,:4],[h['raw_cosine'],np.nan if h['query_margin'] is None else h['query_margin'],np.nan if h['reference_margin'] is None else h['reference_margin'],static],equal_nan=True,rtol=2e-13,atol=2e-13)
            masks=[([],[]),([qi],[ri]),(exclusion(q,qi),exclusion(r,ri))]
            for arm,(qe,re) in enumerate(masks):
                assert producer.strict_exclusions(q,qi)==exclusion(q,qi)
                fit=full if arm==0 else oracle.oracle(qbox,rbox,qraw,rraw,qe,re)
                expected=np.full(11,np.nan);expected[:2]=[fit['available'],fit['anchor_count']]
                if fit['available']:
                    innovation=(delta-fit['translation'])/(qscale*rscale)
                    corrected=abs(innovation).sum()
                    expected[2:]=fit['translation']+fit['residual_mad_xy']+[fit['max_leave_one_anchor_shift']]+innovation.tolist()+[corrected,static-corrected]
                np.testing.assert_allclose(actual[index,4+11*arm:4+11*(arm+1)],expected,equal_nan=True,rtol=2e-13,atol=2e-13)
                assert np.array_equal(ah[index,arm],digest([[a['query_row'],a['reference_row']] for a in fit['anchors']]))
                assert np.array_equal(eh[index,arm],digest(dict(query=qe,reference=re)))
                count+=1
        if n>=7:
            assert 1 in exclusion(q,0) and 2 in exclusion(q,0) and 3 not in exclusion(q,0)
    return count


def block(gains, available=None):
    # gains is per target (full,row,strict). Unavailable leaves all evidence NaN.
    result=np.full((len(gains),37),np.nan)
    for i,triplet in enumerate(gains):
        result[i,:4]=[1,1,1,1]
        for a,gain in enumerate(triplet):
            ok=gain is not None
            result[i,4+11*a:6+11*a]=[ok,3 if ok else 2]
            if ok:
                result[i,6+11*a:15+11*a]=[0,0,0,0,0,0,0,1,gain]
    return result


def check_hierarchy(producer):
    with tempfile.TemporaryDirectory(prefix='donorout-hierarchy-') as tmp:
        root=Path(tmp);outputs={}
        video_data={0:('A',[(1,block([(5,100,8),(5,1,4),(5,3,None),(5,-100,None)])),
                            (1,block([(1,-2,2)])),(4,block([(3,9,-2)]))]),
                    1:('B',[(1,block([(20,10,6)])),(4,block([(2,4,None)]))]),
                    2:('C',[(1,block([])),(4,block([]))])}
        for video,(source,frames) in video_data.items():
            values=np.concatenate([b for lag,b in frames]);offset=0;receipts=[]
            for index,(lag,b) in enumerate(frames):
                receipts.append(dict(lag=lag,offset=offset,holdouts=len(b),query_step=index))
                offset+=len(b)
            npz='v%d.npz'%video;receipt='v%d.json'%video
            np.savez(root/npz,values=values)
            (root/receipt).write_text(json.dumps(dict(frame_pairs=receipts)))
            outputs[video]=dict(npz=npz,receipt=receipt)
        result=producer.summarize(['A','B','C'],{0:'A',1:'B',2:'C'},outputs,root)
        strict='proposal_overlap_excluded';row='row_excluded'
        assert result['readouts'][strict]['1']['source_macro_gain'][strict]==5
        assert result['readouts'][row]['1']['source_macro_gain'][row]==5
        common=result['readouts']['common_row_strict']['1']['source_macro_gain']
        assert common[row]==17.125 and common[strict]==5
        assert result['readouts'][strict]['4']['source_macro_gain'][strict]==-2
        assert result['minimal_gate']=='NOT_PASSED_MINIMAL_GTFREE_DONOR_PREDICTABILITY'
        for name,value in result['readouts'].items():
            for lag in ('1','4'):
                assert set(value[lag]['by_source'])=={'A','B','C'}
                missing=value[lag]['by_source']['C']
                assert missing['supported_frame_pairs']==0
                assert all(v is None for v in missing['source_mean_frame_median_gain'].values())
        cross=result['readouts'][strict]['common_supported_sources_across_lags']
        assert cross['sources']==['A'] and cross['source_macro_gain_by_lag']['1'][strict]==4


def check_overflow(producer,engine):
    # Two individually finite normalized coordinates can have an overflowing L1.
    huge=1e308;width=np.spacing(huge);tiny=1/width
    qb=np.tile([0.,0.,1.,1.],(4,1));rb=qb.copy()
    qb[0]=[huge,huge,huge+width,huge+width];rb[0]=[0.,0.,tiny,tiny]
    frameq=dict(boxes=qb,raw=np.eye(4),proposals=np.arange(4))
    framer=dict(boxes=rb,raw=np.eye(4),proposals=np.arange(4))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        try:
            producer.predict_frame_pair(frameq,framer,engine.CachedSceneMotion)
        except ValueError:
            return
    raise AssertionError('Nonfinite reduced static L1 was not rejected')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer',type=Path,required=True)
    parser.add_argument('--producer-sha256',required=True)
    parser.add_argument('--cached-module',type=Path,required=True)
    parser.add_argument('--cached-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    assert sha(args.producer)==args.producer_sha256 and sha(args.cached_module)==args.cached_sha256
    producer=module(args.producer,'independent_producer_subject')
    engine=module(args.cached_module,'independent_engine_subject')
    oracle=module(Path(__file__).with_name('audit_cached_scene_motion_independent_20261024.py'),'independent_matrix_oracle')
    check_loader(producer);count=check_predictions(producer,engine,oracle)
    check_hierarchy(producer);check_overflow(producer,engine)
    assert sha(args.producer)==args.producer_sha256 and sha(args.cached_module)==args.cached_sha256
    result=dict(status='PASS_INDEPENDENT_DONOROUT_PRODUCER_SYNTHETIC_AUDIT',
        producer_sha256=args.producer_sha256,cached_module_sha256=args.cached_sha256,
        auditor_sha256=sha(__file__),target_arm_schema_comparisons=count,
        checks=['poisoned_loader_exact_allowlist_Cfloat64_score_unused',
        'fullpool_targets_every_h_masks_alias_nonoverlap_direct_overlap_touching_and_hashes',
        'threelevel_hierarchy_common_h_intersections_all_sources_missing_and_crosslag_sources',
        'finite_components_but_overflowing_L1_fail_closed'],
        real_cache_opened=False,new_OOF_outputs_opened=False,GT_or_class_or_track_ids_decoded=False,
        elapsed_seconds=time.monotonic()-started)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
