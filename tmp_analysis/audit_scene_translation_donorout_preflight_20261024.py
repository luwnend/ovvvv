#!/usr/bin/env python3
"""Final read-only independent donor-out protocol, lineage and population gate.

No real predictor/evaluator is run. Only synthetic receipt JSON, frozen input
hashes, allowed population metadata, protected byte hashes and disk/stat data
are read. No raw NPZ array, current OOF score/metric, SQLite or GT is decoded.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time


ARMS=('fullpool','row_excluded','proposal_overlap_excluded')
SLOTS=('available','anchor_count','translation_x','translation_y','MAD_x','MAD_y',
       'fixed_donor_LOO_max','innovation_x','innovation_y','corrected_normalized_L1','gain')
COLUMNS=('raw_cosine','query_margin','reference_margin','static_normalized_L1')+tuple(
    arm+'/'+slot for arm in ARMS for slot in SLOTS)
ALLOWED=('det_bboxes','track_feats','roi_proposal_row','video_id','image_id','frame_id','frame_index')
POPULATION=dict(source_groups=341,videos=403,frames=13791,retained_rows=613536,
    frame_pair_receipts=27582,candidate_pairs_by_lag={'1':29096237,'4':26255927})


def require(condition,message):
    if not condition:raise AssertionError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def component(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def check_population(spec):
    scanner=component(spec['metadata_projection_helper'],'independent_preflight_metadata_scanner')
    roles=read(spec['roles_path']);roles=roles.get('groups',roles)
    sources=sorted(s for s,r in roles.items() if r['split']=='fit')
    require(len(sources)==341,'Sealed FIT source count')
    require(all(roles[s]['role'] in ('mean_role','pair_role') for s in sources),'FIT role boundary')
    plan=scanner.project_object(Path(spec['plan_path']).read_text(),{'video_records'})
    fit_records=[r for r in plan['video_records'] if r['split']=='fit']
    require(len(fit_records)==405,'Original FIT allocation count')
    selected={int(r['video_id']):r['source_group'] for r in fit_records if r['source_group'] in sources}
    require(len(selected)==403,'Sealed cached video population')
    excluded={int(r['video_id']):r['source_group'] for r in fit_records if r['source_group'] not in sources}
    require(excluded=={1720:'LaSOT/racing-11',1974:'LaSOT/hippo-12'},'Uncollected allocation boundary')
    require(spec['population']['uncollected_allocation_video_ids']==[1720,1974],'Protocol omitted allocation exceptions')
    inventory=read(spec['inventory_path']);videos=defaultdict(list)
    fields={'video_id','image_id','frame_id','frame_index','observations','npz','npz_sha256'}
    with Path(spec['frame_execution_path']).open() as stream:
        for line in stream:
            # Other baseline collection receipt fields are not decoded.
            locator=scanner.project_object(line,{'video_id'})
            if int(locator['video_id']) not in selected:continue
            row=scanner.project_object(line,fields)
            require(set(row)==fields,'Incomplete frozen frame metadata')
            require(row['npz_sha256']==inventory[Path(row['npz']).name],'Metadata inventory mismatch')
            videos[int(row['video_id'])].append(row)
    require(set(videos)==set(selected),'Cached video coverage differs')
    require(set(selected.values())==set(sources),'Sources without allocated cached video')
    rows=0;frames=0;maxrows=0;maxframes=0
    pair_counts={1:0,4:0};prefix={1:0,4:0};nonprefix={1:0,4:0};targets={1:0,4:0}
    locators=set()
    for video,records in videos.items():
        records.sort(key=lambda row:row['frame_id'])
        require([r['frame_id'] for r in records]==list(range(len(records))),'Nonconsecutive cached ordinal')
        require(all(a['frame_index']<b['frame_index'] for a,b in zip(records,records[1:])),'Cached timeline')
        maxframes=max(maxframes,len(records));frames+=len(records)
        for step,row in enumerate(records):
            require(type(row['observations']) is int and 0<=row['observations']<=77,'Row metadata outside bound')
            require((video,row['image_id']) not in locators,'Duplicate immutable frame locator')
            locators.add((video,row['image_id']));rows+=row['observations'];maxrows=max(maxrows,row['observations'])
            for lag in (1,4):
                if step<lag:prefix[lag]+=1
                else:
                    nonprefix[lag]+=1
                    q,m=row['observations'],records[step-lag]['observations']
                    pair_counts[lag]+=q*m;targets[lag]+=min(q,m)
    expected=dict(source_groups=len(sources),videos=len(videos),frames=frames,retained_rows=rows,
        frame_pair_receipts=2*frames,candidate_pairs_by_lag={str(k):v for k,v in pair_counts.items()})
    require(expected==POPULATION,'Independently reconstructed population changed')
    for key,value in expected.items():require(spec['population'][key]==value,'Protocol population field '+key)
    require(prefix=={1:403,4:1606} and nonprefix=={1:13388,4:12185},'Pair-prefix boundary')
    require(targets=={1:564223,4:506219} and maxrows==77 and maxframes==41,'Resource metadata bounds')
    return dict(**expected,all_source_identifiers=sources,prefix_by_lag=prefix,
        nonprefix_by_lag=nonprefix,target_count_upper_by_lag=targets,
        maximum_rows=maxrows,maximum_frames_per_video=maxframes,
        excluded_original_allocation=excluded),sources,selected,dict(videos)


def check_receipts(spec,principals):
    base=Path(spec['producer_path']).parent
    required=[('cached_scene_motion_independent_round%d_20261024_v2.json'%r,
        'PASS_INDEPENDENT_CACHED_SCENE_MOTION_CANONICAL_MATRIX_SYNTHETIC_AUDIT','module_sha256','cached') for r in (1,2,3)]
    required += [
      ('cached_scene_motion_independent_layout_20261024_v2.json','PASS_INDEPENDENT_CACHED_CANONICAL_LAYOUT_AND_NEARTIE_MEMBERSHIP','module_sha256','cached'),
      ('scene_translation_producer_independent_synthetic_20261024_v1.json','PASS_INDEPENDENT_DONOROUT_PRODUCER_SYNTHETIC_AUDIT','producer_sha256','producer'),
      ('scene_translation_runner_independent_synthetic_20261024_v1.json','PASS_INDEPENDENT_DONOROUT_RUNNER_ACTUAL_CHILD_WAIT_AND_FAILSTOP_SYNTHETIC_AUDIT','runner_sha256','runner'),
      ('scene_translation_parallel_storage_precheck_20261024_v1.json','PASS_CURRENT_READONLY_MAIN_OOF_PINS_PROTECTED_AND_PARALLEL_STORAGE_PRECHECK',None,None),
      ('scene_translation_donorout_scalar_fixture_QA_20261024_v2.json','PASS_SCALAR_DONOROUT_SYNTHETIC_FIXTURES','verifier_sha256','auditor'),
      ('scene_translation_donorout_producer_independent_toys_20261024_v2.json','PASS_INDEPENDENT_DONOROUT_PRODUCER_TOYS','verifier_sha256','auditor'),
      ('scene_translation_donorout_output_CLI_independent_QA_20261024_v2.json','PASS_TEMPORARY_DONOROUT_OUTPUT_CLI_QA','verifier_sha256','auditor')]
    receipts=[]
    for filename,status,binding,role in required:
        path=(base/filename).resolve();name=str(path)
        require(name in spec['input_pins'],'Independent evidence not pinned: '+filename)
        value=read(path);require(value['status']==status,'Independent receipt status: '+filename)
        require(sha(path)==spec['input_pins'][name],'Independent receipt SHA changed')
        if binding:require(value[binding]==principals[role],'Stale independent evidence: '+filename)
        if 'producer_sha256' in value:require(value['producer_sha256']==principals['producer'],'Stale producer dependency')
        if 'cached_sha256' in value:require(value['cached_sha256']==principals['cached'],'Stale cached dependency')
        if 'cached_module_sha256' in value:require(value['cached_module_sha256']==principals['cached'],'Stale cached dependency')
        if filename=='scene_translation_donorout_output_CLI_independent_QA_20261024_v2.json':
            require(value['positive_actual_returncode']==0,'Final output-auditor CLI positive trial failed')
            require(len(value['resealed_rejection_cases'])==5 and all(c['actual_returncode']!=0 for c in value['resealed_rejection_cases']),'Final output-auditor mutation CLI failstop')
        if 'auditor_sha256' in value:
            # Bind each independent test receipt to its generating script.
            scripts={
                'cached_scene_motion_independent_layout_20261024_v2.json':'audit_cached_scene_motion_layout_20261024.py',
                'scene_translation_producer_independent_synthetic_20261024_v1.json':'audit_scene_translation_producer_synthetic_20261024.py',
                'scene_translation_runner_independent_synthetic_20261024_v1.json':'audit_scene_translation_runner_independent_20261024.py',
                'scene_translation_parallel_storage_precheck_20261024_v1.json':'audit_scene_translation_parallel_storage_20261024.py'}
            script=base/scripts.get(filename,'audit_cached_scene_motion_independent_20261024.py')
            require(str(script.resolve()) in spec['input_pins'],'Independent script not pinned')
            require(value['auditor_sha256']==sha(script),'Receipt generator changed')
        receipts.append(dict(path=name,sha256=sha(path),status=status))
    # Read only synthetic/QA JSON evidence, never baseline/current summaries.
    synthetic=[]
    for name,digest in spec['input_pins'].items():
        path=Path(name)
        if path.suffix!='.json' or not any(word in path.name for word in ('synthetic_round','fixture_QA','implementation_QA','auditor_QA','verifier_QA','scalar_fixture')):
            continue
        value=read(path)
        require(str(value.get('status','')).startswith('PASS'),'Non-PASS synthetic evidence: '+name)
        for key,role in (('producer_sha256','producer'),('cached_sha256','cached'),('cached_module_sha256','cached'),
                         ('module_sha256','cached'),('verifier_sha256','auditor'),('runner_sha256','runner')):
            if key in value:require(value[key]==principals[role],'Stale synthetic binding: '+name+' '+key)
        for key in ('checker_sha256','auditor_sha256','qa_sha256'):
            if key in value:
                require(value[key] in set(spec['input_pins'].values())|set(principals.values()),'Synthetic generating code unpinned: '+name)
        synthetic.append(dict(path=name,sha256=digest,status=value['status']))
    require(any('scalar_fixture' in r['path'] for r in synthetic),'No pinned final scalar oracle fixture evidence')
    return receipts,synthetic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    require(sha(args.protocol)==args.expected_protocol_sha256,'Protocol external SHA mismatch')
    spec=read(args.protocol)
    require(spec['status']=='FROZEN_GTFREE_SCENE_TRANSLATION_DONOROUT_V1','Protocol not frozen')
    require(spec['schema']=='GTFREE_SCENE_TRANSLATION_DONOROUT_V1','Schema')
    require(spec['lags']==[1,4] and spec['minimum_anchors']==3,'Fixed lags/minimum')
    require(spec['arms']==list(ARMS) and spec['columns']==list(COLUMNS),'37-column arm schema')
    require(spec['raw_NPZ_member_allowlist']==list(ALLOWED),'Raw data boundary')
    require(spec['minimum_free_disk_bytes']==31*(1024**3),'Parallel disk gate')
    require(spec['resource_bounds']['upper_bound_uncompressed_array_bytes']==548066304,'Numeric array storage bound')
    require(spec['resource_bounds']['CPU_only'] is True and spec['resource_bounds']['threads']==1,'Execution resource convention')
    require(not Path(spec['output_directory']).exists() and not Path(spec['execution_directory']).exists(),'Output already exists; no overwrite/retry')
    pin_checks=[]
    for name,expected in spec['input_pins'].items():
        require(sha(name)==expected,'Frozen input pin changed: '+name)
        pin_checks.append(dict(path=name,sha256=expected))
    runner_path=spec.get('runner_path',str(Path(spec['producer_path']).with_name('run_scene_translation_donorout_20261024.py')))
    paths=dict(producer=spec['producer_path'],cached=spec['cached_module_path'],auditor=spec['auditor_path'],runner=runner_path)
    principals={k:sha(v) for k,v in paths.items()}
    for role,path in paths.items():
        require(str(Path(path).resolve()) in spec['input_pins'],'Principal component not pinned: '+role)
        require(spec['input_pins'][str(Path(path).resolve())]==principals[role],'Principal pin changed: '+role)
    require(spec['producer_sha256']==principals['producer'],'Producer direct pin')
    own_path=str(Path(__file__).resolve())
    # This checker is bound by its SHA in the external preflight gate and
    # publication snapshot, avoiding a protocol/gate self-reference.
    protected={};authority=read(spec['protected_authority_path'])
    for name,record in authority['protected_files'].items():
        current=dict(exists=Path(name).exists(),sha256=sha(name) if Path(name).exists() else None)
        require(current==record,'Protected file changed: '+name);protected[name]=current
    require(len(protected)==7,'Protected population changed')
    evidence,synthetic=check_receipts(spec,principals)
    population,sources,video_sources,videos=check_population(spec)
    # Actual producer authentication is a read-only public gate. Compare its
    # metadata projection with the separate reconstruction above, without IO to
    # any raw NPZ file or predictor.
    producer=component(paths['producer'],'independent_preflight_producer_authentication')
    require(producer.authenticate(args.protocol)==spec,'Producer authentication returned altered protocol')
    actual_sources,actual_video_sources,actual_videos=producer.prepare_metadata(spec)
    require(actual_sources==sources and actual_video_sources==video_sources,'Producer source projection differs')
    require(actual_videos==videos,'Producer frame metadata projection differs')
    free=shutil.disk_usage(Path(spec['output_directory']).parent).free
    require(free>=spec['minimum_free_disk_bytes'],'Current parallel disk gate failed')
    require(sha(args.protocol)==args.expected_protocol_sha256,'Protocol changed during preflight')
    for name,expected in spec['input_pins'].items():require(sha(name)==expected,'Input changed during preflight: '+name)
    result=dict(status='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_PREFLIGHT',
        recorded_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=args.expected_protocol_sha256,
        runner_sha256=principals['runner'],producer_sha256=principals['producer'],
        auditor_sha256=principals['auditor'],cached_module_sha256=principals['cached'],
        checker_path=own_path,checker_sha256=sha(__file__),input_pin_count=len(pin_checks),input_pin_checks=pin_checks,
        independent_evidence=evidence,final_synthetic_evidence=synthetic,
        independently_reconstructed_population=population,protected_files=protected,
        current_free_disk_bytes=free,parallel_minimum_free_disk_bytes=spec['minimum_free_disk_bytes'],
        checks=['final_protocol_external_SHA_and_all_input_pins',
            'all_final_synthetic_and_independent_receipt_principal_generator_pins',
            'actual_waited_subprocess_success_and_failstop_receipt_bound_to_final_runner',
            'all341_sealed_FIT_sources_all403_cached_videos_all13791_frames_and_fixed_lag_population',
            'seven_protected_existence_and_byte_hash_states',
            'read_only_producer_authentication_and_metadata_projection_match',
            'no_existing_output_execution_and_current_parallel_31GiB_disk_gate'],
        raw_NPZ_arrays_decoded=False,new_OOF_metrics_or_score_payload_opened=False,
        SQLite_opened=False,GT_class_track_or_released_ID_values_decoded=False,
        real_diagnostic_launched=False,existing_pinned_files_mutated=False,
        elapsed_seconds=time.monotonic()-started)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_pin_checks','protected_files','final_synthetic_evidence','independent_evidence','independently_reconstructed_population')},indent=2))


if __name__=='__main__':main()
