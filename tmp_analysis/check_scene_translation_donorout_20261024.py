#!/usr/bin/env python3
"""Synthetic checks for the actual shared producer paths, never real data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent


def load(name):
    path=HERE/name
    spec=importlib.util.spec_from_file_location(name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def frame(index,n=6):
    xy=np.column_stack((np.arange(n)*100.,np.arange(n)*20.))
    boxes=np.column_stack((xy,xy+5))+np.array([index*8,index*2,index*8,index*2])
    return dict(record=dict(video_id=7,image_id=index+100,frame_id=index,
        frame_index=index*30,observations=n),boxes=boxes,raw=np.eye(n,256),proposals=np.arange(n))


def run(round_number):
    p=load('scene_translation_donorout_20261024.py')
    c=load('cached_endpoint_scene_motion_20261024.py');checks=[]
    if round_number==1:
        fixtures=[frame(i,0 if i==2 else 6) for i in range(6)]
        records=[f['record'] for f in fixtures]
        arrays,receipts,counts=p.predict_video(records,c.CachedSceneMotion,lambda r:fixtures[r['frame_id']])
        assert len(receipts)==12 and sum(r['reason']=='missing_reference_prefix' for r in receipts)==5
        assert all(r['reference_step'] is None or r['query_step']-r['reference_step']==r['lag'] for r in receipts)
        assert receipts[6]['reference_step']==2 and receipts[6]['reference_rows']==0 and receipts[6]['holdouts']==0
        assert counts=={1:108,4:72}
        checks.append('actual_predict_video_exact_lags_all_prefixes_and_empty_reference_not_skipped')
        assert arrays['values'].shape[1]==37 and arrays['rowloc'].shape[1]==3
        for i,r in enumerate(receipts):
            block=arrays['values'][r['offset']:r['offset']+r['holdouts']]
            loc=arrays['rowloc'][r['offset']:r['offset']+r['holdouts']]
            assert np.all(loc[:,0]==i)
            if len(block):
                assert np.all(block[:,4::11]==1)
                np.testing.assert_allclose(block[:,[13,24,35]],0,atol=1e-12)
        checks.append('full_numeric_hash_rowlocator_schema_and_known_shared_translation')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);np.savez(root/'toy.npz',**arrays)
            p.write_new(root/'toy.json',dict(frame_pairs=receipts))
            report=p.summarize(['A/source','Z/unsupported'],{7:'A/source'},
                {7:dict(npz='toy.npz',receipt='toy.json')},root)
            assert report['minimal_gate']=='PASS_MINIMAL_GTFREE_DONOR_PREDICTABILITY'
            for name in report['readouts']:
                for lag in (1,4):
                    row=report['readouts'][name][str(lag)]
                    assert row['supported_sources']==1
                    assert row['by_source']['Z/unsupported']['supported_frame_pairs']==0
                    assert all(v is None for v in row['by_source']['Z/unsupported']['source_mean_frame_median_gain'].values())
        checks.append('all_sources_and_both_lags_summary_missing_is_not_zero')
    elif round_number==2:
        q,r=frame(1),frame(0)
        q['boxes'][1]=q['boxes'][0];r['boxes'][1]=r['boxes'][0]
        q['boxes'][2]=q['boxes'][0]+[5,0,5,0]
        r['boxes'][2]=r['boxes'][0]+[5,0,5,0]
        q['proposals'][3]=q['proposals'][0];r['proposals'][3]=r['proposals'][0]
        assert p.strict_exclusions(q,0)==[0,1,3]
        assert p.strict_exclusions(r,0)==[0,1,3]
        checks.append('strict_pool_includes_nonoverlap_proposal_alias_but_not_zero_area_touch')
        values,loc,ah,eh=p.predict_frame_pair(q,r,c.CachedSceneMotion)
        row=np.flatnonzero((loc==[0,0]).all(1))[0]
        assert values[row,4+11+1]==5 and values[row,4+22+1]==3
        np.testing.assert_array_equal(eh[row,2],p.digest_array(dict(query=[0,1,3],reference=[0,1,3])))
        checks.append('row_and_strict_arms_recompute_distinct_pools_with_exclusion_hash')
        q,r=frame(1,3),frame(0,3)
        values,loc,ah,eh=p.predict_frame_pair(q,r,c.CachedSceneMotion)
        assert len(values)==3 and np.all(values[:,15]==0) and np.all(values[:,26]==0)
        assert np.isnan(values[:,17:26]).all() and np.isnan(values[:,28:37]).all()
        assert np.all(values[:,16]==2) and np.all(values[:,27]==2)
        checks.append('every_fullpool_holdout_preserved_when_both_excluded_fits_missing')
    elif round_number==3:
        f=frame(0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.npz';rec=dict(f['record']);rec['npz']=str(path)
            np.savez(path,det_bboxes=np.column_stack((f['boxes'],np.ones(6))),
                track_feats=np.asfortranarray(f['raw']),roi_proposal_row=f['proposals'],
                video_id=rec['video_id'],image_id=rec['image_id'],frame_id=rec['frame_id'],
                frame_index=rec['frame_index'],
                track_ids=np.asarray([{'must_not_decode':'identity'}],dtype=object),
                p_fused=np.asarray([{'must_not_decode':'posterior'}],dtype=object),
                det_labels=np.asarray([{'must_not_decode':'category'}],dtype=object))
            rec['npz_sha256']=p.sha(path)
            result=p.load_frame(rec)
            assert result['raw'].flags.c_contiguous and result['boxes'].flags.c_contiguous
            assert set(result)=={'record','boxes','raw','proposals'}
        checks.append('actual_loader_only_allowed_members_and_canonical_C_layout')
        q,r=frame(1),frame(0)
        a=p.predict_frame_pair(q,r,c.CachedSceneMotion)
        q['raw']=np.asfortranarray(q['raw']);r['raw']=np.asfortranarray(r['raw'])
        b=p.predict_frame_pair(q,r,c.CachedSceneMotion)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        checks.append('equal_C_F_features_produce_exactly_equal_membership_hashes_and_values')
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'keep.json';p.write_new(target,dict(a=1))
            try:p.write_new(target,dict(a=2))
            except FileExistsError:pass
            else:raise AssertionError('Evidence overwritten')
            assert json.loads(target.read_text())==dict(a=1)
        checks.append('nonoverwriting_receipt_contract')
    else:raise ValueError(round_number)
    return dict(status='PASS_SCENE_TRANSLATION_DONOROUT_PRODUCER_SYNTHETIC',round=round_number,
        checks=checks,producer_sha256=p.sha(p.__file__),cached_sha256=p.sha(c.__file__),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        real_cache_or_identity_or_new_OOF_evaluation_opened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.round)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))
