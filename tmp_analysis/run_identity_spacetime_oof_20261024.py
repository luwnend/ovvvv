#!/usr/bin/env python3
"""Pinned CPU source-OOF execution, global seal, detached supervision.

No identity decoder is imported here. Evaluation runs in a fresh subprocess
only after every model's complete held-out output is globally sealed.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
PYTHON=Path('/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python').resolve()
PROTOCOL=ROOT/'tmp_analysis/identity_spacetime_oof_protocol_20261024.json'
GLOBAL_SEAL='GT_free_OOF_predictions_sealed.json'
SEALED='SEALED_ALL_15_SOURCE_OOF_MODELS_BEFORE_UNIFIED_EVALUATION'
PREFLIGHT='PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_IMPLEMENTATION_PREFLIGHT'
SUCCESS='PASS_IDENTITY_SPACETIME_OOF_EXECUTION_AND_INDEPENDENT_AUDIT'
FROZEN='FROZEN_FIXED_EXPLORATORY_SOURCE_OOF_IDENTITY_SPACETIME_PROTOCOL'


def require(value,message):
    if not value:raise RuntimeError(message)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    """Atomic non-replacing JSON publication, including parent fsync."""
    path=Path(path);temporary=path.parent/('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as f:
            f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        os.link(str(temporary),str(path))
        descriptor=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary.exists():temporary.unlink()


def component(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def model_name(fold,seed):return 'fold%d_seed%d'%(fold,seed)


def snapshot(expected):
    return {p:dict(exists=Path(p).exists(),sha256=sha(p) if Path(p).exists() else None) for p in expected}


def all_pins(spec):
    result=dict(spec['input_pins'])
    for key in ('modules','data_module','trainer','coordinator','evaluator','verifier','finalizer','fold_manifest'):
        result[spec[key+'_path']]=spec[key+'_sha256']
    for key in ('manual','index'):result[spec['identity_authority'][key+'_path']]=spec['identity_authority'][key+'_sha256']
    return result


def check_pins(pins):
    for path,digest in pins.items():
        require(Path(path).is_file() and sha(path)==digest,'Pinned bytes changed: '+str(path))


def authenticate(protocol,check_gate=True):
    path=Path(protocol);spec=read(path)
    require(spec['status']==FROZEN and spec['schema']=='IDENTITY_SPACETIME_SOURCE_OOF_V1','Protocol is not frozen')
    require(spec['folds']==list(range(5)) and spec['seeds']==[20261024,20261025,20261026],'OOF dimensions changed')
    require(spec['coordinator_path']==str(SELF) and spec['coordinator_sha256']==sha(SELF),'Coordinator is not pinned')
    require(spec['score_columns']==spec['feature_names'] and len(spec['score_columns'])==10,'Score columns changed')
    check_pins(all_pins(spec))
    authority=read(spec['protected_authority_path'])
    require(len(authority['protected_files'])==7 and snapshot(authority['protected_files'])==authority['protected_files'],'Protected authority changed')
    if check_gate:
        gate=read(spec['implementation_preflight']['path'])
        require(gate['status']==PREFLIGHT,'Independent implementation preflight is not PASS')
        require(gate['protocol_sha256']==sha(path),'Preflight protocol pin stale')
        for key in ('trainer','coordinator','evaluator','verifier'):
            require(gate[key+'_sha256']==spec[key+'_sha256'],'Preflight '+key+' pin stale')
        require(gate['data_sha256']==spec['data_module_sha256'] and gate['modules_sha256']==spec['modules_sha256'],'Preflight data/modules stale')
    return spec


def safe_member(directory,relative):
    part=Path(relative)
    require(not part.is_absolute() and '..' not in part.parts,'Unsafe sealed member')
    path=directory/part
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink in sealed member path')
    return path


def model_expected(spec,prepared,fold):
    sources=set(prepared['manifest']['folds'][str(fold)]['held_out_sources'])
    records=[r for r in prepared['frame_records'] if prepared['video_sources'][str(r['video_id'])] in sources]
    grouped=defaultdict(list)
    for row in records:grouped[int(row['video_id'])].append(row)
    shapes={}
    for video,rows in grouped.items():
        rows.sort(key=lambda r:r['frame_id'])
        require([r['frame_id'] for r in rows]==list(range(len(rows))),'Missing sampled frame metadata')
        for lag in spec['lags']:
            shapes['shards/v%d_lag%d.scores.npy'%(video,lag)]=(sum(rows[i]['observations']*rows[i-lag]['observations'] for i in range(lag,len(rows))),10)
    files=set(shapes)|{'shards/v%d.rows.json'%v for v in grouped}|{'frame_pairs.jsonl','checkpoint.json','normalizer.json','training_trace.json'}
    return records,shapes,files


def verify_row_coverage(directory,records,prepared):
    grouped=defaultdict(list);metadata={}
    for row in records:grouped[int(row['video_id'])].append(row)
    for video,rows in grouped.items():
        rows.sort(key=lambda r:r['frame_id']);stored=read(directory/('shards/v%d.rows.json'%video))
        require(stored['video_id']==video and len(stored['frames'])==len(rows),'Incomplete held-out rowmap')
        for row,frame in zip(rows,stored['frames']):
            require(frame['video_id']==video and frame['image_id']==row['image_id'] and frame['sampled_step']==row['frame_id'] and frame['original_frame_index']==row['frame_index'],'Held-out frame locator differs')
            for key in ('boxes','released_ids','predicted_categories','previous_boxes','previous_times','previous_locators','history_reason'):
                require(len(frame[key])==row['observations'],'Incomplete retained row metadata')
        metadata[video]=stored['frames']
    expected=[];offsets=Counter()
    for video,rows in sorted(grouped.items()):
        for row in rows:
            step=row['frame_id']
            for lag in (1,4):
                reference=None if step<lag else rows[step-lag];nr=0 if reference is None else reference['observations'];count=row['observations']*nr
                expected.append(dict(source=prepared['video_sources'][str(video)],video_id=video,lag=lag,query_step=step,
                    reference_step=None if reference is None else step-lag,query_image_id=row['image_id'],
                    reference_image_id=None if reference is None else reference['image_id'],query_rows=row['observations'],reference_rows=nr,
                    candidates=count,offset=offsets[video,lag],score_file='shards/v%d_lag%d.scores.npy'%(video,lag),
                    original_frame_gap=None if reference is None else row['frame_index']-reference['frame_index']))
                offsets[video,lag]+=count
    index=0
    with (directory/'frame_pairs.jsonl').open() as stream:
        for line in stream:
            receipt=json.loads(line);require(index<len(expected),'Extra held-out pair receipt')
            require(all(receipt.get(k)==v for k,v in expected[index].items()),'Missing/nonexact held-out candidate block or offset')
            index+=1
    require(index==len(expected),'Missing held-out prefix/empty/full pair receipts')


def verify_model(directory,spec,prepared,fold,seed):
    directory=Path(directory);receipt=read(directory/'single_model_receipt.json');seal=read(directory/'GT_free_predictions_sealed.json')
    require(receipt['status']=='COMPLETE_SINGLE_FOLD_SEED_TRAINING_AND_FULL_HELDOUT_PREDICTION','Incomplete single model')
    require((receipt['fold'],receipt['seed'])==(fold,seed),'Single model identity differs')
    require(receipt['script_sha256']==spec['trainer_sha256'],'Model receipt trainer differs')
    require(receipt['held_out_identity_values_decoded'] is False and receipt['GPU_used'] is False and receipt['training_sources_only'] is True,'Single model isolation claim differs')
    require(receipt['global_model_prediction_seal_sha256']==sha(directory/'GT_free_predictions_sealed.json'),'Single model seal hash differs')
    require(seal['status']=='SEALED_SINGLE_MODEL_FULL_HELDOUT_BEFORE_HELDOUT_LABELS' and seal['all_retained_candidates_scored'] is True and seal['unknown_candidates_filtered'] is False,'Full retained population not sealed')
    meta=seal['metadata'];split=prepared['manifest']['folds'][str(fold)]
    require(meta['fold']==fold and meta['seed']==seed and meta['train_sources']==split['train_sources'] and meta['held_out_sources']==split['held_out_sources'],'Fold lineage differs')
    require(meta['columns']==spec['score_columns'] and meta['final_adapter_epoch']==50 and meta['final_head_epoch']==20,'Incomplete arms or epochs')
    require(meta['schedule']==spec['numeric_training_config'],'Model optimizer schedule differs')
    records,shapes,expected=model_expected(spec,prepared,fold)
    require(set(seal['files_sha256'])==expected and seal['files_sha256']==receipt['generated']['files_sha256'],'Model seal file coverage differs')
    require(seal['frame_pair_receipts']==2*len(records) and seal['parent_raw_scores_exactly_verified'] is True,'Model frame coverage/raw parent parity absent')
    for name,digest in seal['files_sha256'].items():
        path=safe_member(directory,name);require(path.is_file() and sha(path)==digest,'Sealed model member changed: '+name)
    for name,shape in shapes.items():
        values=np.load(directory/name,mmap_mode='r',allow_pickle=False)
        require(values.shape==shape and values.dtype==np.float64,'Full score shape/dtype differs')
        del values
    for name in ('checkpoint.json','normalizer.json','training_trace.json'):
        require(read(directory/name)['metadata']==meta,'Model metadata differs: '+name)
    trace=read(directory/'training_trace.json');n=trace['training_query_lags'];steps=(n+63)//64
    require(n>0 and len(trace['adapter'])==50 and len(trace['heads'])==20 and trace['adapter_optimizer_steps']==50*steps,'Actual adapter schedule incomplete')
    require(set(trace['head_optimizer_steps'])==set(spec['score_columns'][2:]) and all(v==20*steps for v in trace['head_optimizer_steps'].values()),'Actual head schedule incomplete')
    batches=[64]*(n//64)+([n%64] if n%64 else [])
    for entries in (trace['adapter'],trace['heads']):
        require(all(item['epoch']==i+1 and item['batch_sizes']==batches for i,item in enumerate(entries)),'Actual epoch/batch coverage incomplete')
    verify_row_coverage(directory,records,prepared)
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    expected.update({'single_model_receipt.json','GT_free_predictions_sealed.json'})
    require(actual==expected,'Unexpected or missing single model members')
    return dict(fold=fold,seed=seed,files_sha256={name:sha(directory/name) for name in sorted(expected)},
                candidates_by_lag=seal['candidates_by_lag'],frame_pair_receipts=seal['frame_pair_receipts'])


def prepared_inputs(spec):
    data=component(spec['data_module_path'],'identity_oof_coordinator_data')
    return data.prepare_inputs(spec)


def verify_models(out,spec):
    out=Path(out);prepared=prepared_inputs(spec);models={};counts={str(s):Counter() for s in spec['seeds']};frames=Counter()
    expected={model_name(f,s) for s in spec['seeds'] for f in spec['folds']}
    actual={p.name for p in (out/'models').iterdir()}
    require(actual==expected,'All fifteen model directories are required before supervision')
    for seed in spec['seeds']:
        for fold in spec['folds']:
            name=model_name(fold,seed);result=verify_model(out/'models'/name,spec,prepared,fold,seed);models[name]=result
            counts[str(seed)].update(result['candidates_by_lag']);frames[seed]+=result['frame_pair_receipts']
    for seed in spec['seeds']:
        require(dict(counts[str(seed)])=={str(k):v for k,v in spec['resource_bounds']['candidates_by_lag'].items()},'OOF candidate totals differ')
        require(frames[seed]==2*spec['population']['selected_frames'],'OOF frame totals differ')
    return models


def seal_models(out,protocol):
    spec=read(protocol) if isinstance(protocol,(str,Path)) else protocol
    models=verify_models(out,spec)
    value=dict(status=SEALED,schema='IDENTITY_SPACETIME_OOF_GLOBAL_SEAL_V1',models=models,
        columns=spec['score_columns'],folds=spec['folds'],seeds=spec['seeds'],model_count=15,
        per_model_held_out_source_identities_decoded_before_prediction=False,unified_evaluation_started=False,
        input_sha256=all_pins(spec),interpretation=spec['interpretation'])
    write_new(Path(out)/GLOBAL_SEAL,value);return value


def verify_global_seal(out,protocol):
    """Mandatory gate; complete hash pass occurs before an identity decoder."""
    spec=read(protocol) if isinstance(protocol,(str,Path)) else protocol
    seal=read(Path(out)/GLOBAL_SEAL)
    require(seal['status']==SEALED and seal['model_count']==15,'Global OOF seal absent')
    require(seal['folds']==spec['folds'] and seal['seeds']==spec['seeds'] and seal['columns']==spec['score_columns'],'Global dimensions differ')
    require(seal['input_sha256']==all_pins(spec),'Global lineage differs')
    check_pins(seal['input_sha256'])
    require(seal['models']==verify_models(out,spec),'Global members changed or incomplete')
    return seal


def child(command,log_directory,label):
    started=utc();begin=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (log_directory/(label+'.stdout.log')).open('xb') as stdout,(log_directory/(label+'.stderr.log')).open('xb') as stderr:
        process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
        identity=process_identity(process.pid);write_new(log_directory/(label+'.started.json'),dict(command=command,identity=identity,started_utc=started))
        code=process.wait()
    record=dict(command=command,pid=process.pid,identity=identity,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-begin,actual_exit_code=code,waited_by_parent_pid=os.getpid())
    write_new(log_directory/(label+'.exit.json'),record)
    require(code==0,'Child failed: '+label+' exit '+str(code));return record


def execute(protocol,execution):
    spec=authenticate(protocol);out=Path(spec['output_directory']);execution=Path(execution)
    require(not out.exists(),'Preserve existing OOF output')
    require(shutil.disk_usage(out.parent).free>=spec['resource_bounds']['minimum_free_disk_bytes'],'Insufficient disk space')
    out.mkdir();(out/'models').mkdir();logs=execution/'model_processes';logs.mkdir()
    actual=[];started=utc()
    for seed in spec['seeds']:
        for fold in spec['folds']:
            name=model_name(fold,seed)
            command=[str(PYTHON),'-u',spec['trainer_path'],'--execute','--protocol',str(protocol),'--fold',str(fold),'--seed',str(seed),'--out',str(out/'models'/name)]
            actual.append(child(command,logs,name))
    seal_models(out,spec);verify_global_seal(out,spec)
    evaluation=child([str(PYTHON),'-u',spec['evaluator_path'],'--out',str(out),'--protocol',str(protocol)],logs,'unified_evaluation')
    verify_global_seal(out,spec);authenticate(protocol)
    write_new(out/'coordinator_receipt.json',dict(status='COMPLETE_ALL_MODELS_GLOBAL_SEAL_AND_UNIFIED_EVALUATION_PENDING_INDEPENDENT_AUDIT',
        started_utc=started,finished_utc=utc(),actual_model_processes=actual,evaluation_process=evaluation,
        protocol_sha256=sha(protocol),global_seal_sha256=sha(out/GLOBAL_SEAL),GPU_used=False))


def process_identity(pid):
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split();ticks=int(fields[19])
        result=dict(pid=pid,start_ticks=ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            cmdline=[x.decode(errors='surrogateescape') for x in (path/'cmdline').read_bytes().split(b'\0') if x],state=fields[0])
        require(int((path/'stat').read_text().split(')')[-1].split()[19])==ticks,'PID changed during identity read')
        return result
    except (FileNotFoundError,ProcessLookupError):return None


def supervise(launch_path):
    launch=read(launch_path);execution=Path(launch['execution_directory']);started=utc()
    try:
        check_pins(launch['file_pins']);spec=authenticate(launch['protocol'])
        write_new(execution/'supervisor_identity.json',dict(identity=process_identity(os.getpid()),started_utc=started,launch_sha256=sha(launch_path)))
        runner=child([str(PYTHON),'-u',str(SELF),'--execute','--protocol',launch['protocol'],'--execution',str(execution)],execution,'coordinator')
        audit=child([str(PYTHON),'-u',spec['verifier_path'],'--out',spec['output_directory'],'--protocol',launch['protocol']],execution,'independent_output_audit')
        output=Path(spec['output_directory']);audit_record=read(output/'independent_output_audit.json')
        require(audit_record['status']=='PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT','Independent output audit did not pass')
        check_pins(launch['file_pins']);authenticate(launch['protocol']);verify_global_seal(output,spec)
        write_new(execution/'execution_receipt.json',dict(status=SUCCESS,started_utc=started,finished_utc=utc(),coordinator=runner,independent_auditor=audit,
            launch_sha256=sha(launch_path),global_seal_sha256=sha(output/GLOBAL_SEAL),independent_audit_sha256=sha(output/'independent_output_audit.json'),GPU_used=False))
        child([str(PYTHON),'-u',spec['finalizer_path'],'--out',str(output),'--execution',str(execution)],execution,'finalization')
    except BaseException:
        write_new(execution/'execution_failure.json',dict(status='FAILED_IDENTITY_SPACETIME_OOF_EXECUTION',finished_utc=utc(),traceback=traceback.format_exc()))
        raise


def launch(protocol,preflight_sha256):
    spec=authenticate(protocol);execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    gate=Path(spec['implementation_preflight']['path']);require(sha(gate)==preflight_sha256,'Launch preflight SHA differs')
    require(not execution.exists() and not out.exists(),'Preserve prior execution/output')
    require(shutil.disk_usage(out.parent).free>=spec['resource_bounds']['minimum_free_disk_bytes'],'Insufficient disk space')
    execution.mkdir();pins=all_pins(spec);pins[str(Path(protocol).resolve())]=sha(protocol);pins[str(gate)]=preflight_sha256
    value=dict(schema='IDENTITY_SPACETIME_OOF_DETACHED_LAUNCH_V1',created_utc=utc(),protocol=str(Path(protocol).resolve()),
        output_directory=str(out),execution_directory=str(execution),file_pins=pins,CPU_only=True,retry_enabled=False)
    path=execution/'launch_spec.json';write_new(path,value)
    with (execution/'supervisor.stdout.log').open('xb') as stdout,(execution/'supervisor.stderr.log').open('xb') as stderr:
        process=subprocess.Popen([str(PYTHON),'-u',str(SELF),'--supervise',str(path)],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        identity=process_identity(process.pid)
    receipt=dict(status='LAUNCHED_NOT_COMPLETE',launcher_pid=os.getpid(),supervisor_identity=identity,launch_spec_sha256=sha(path),execution_directory=str(execution),output_directory=str(out))
    write_new(execution/'launch_receipt.json',receipt);return receipt


def status(protocol):
    spec=read(protocol);execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    result=dict(execution_directory=str(execution),output_directory=str(out),status='NOT_LAUNCHED',completed_models=[],active_workers=[])
    if not execution.exists():return result
    result['status']='RUNNING_OR_INTERRUPTED'
    if (execution/'execution_failure.json').exists():result.update(status='FAILED',failure=read(execution/'execution_failure.json'))
    if (execution/'launch_receipt.json').exists():
        bound=read(execution/'launch_receipt.json')['supervisor_identity'];current=None if bound is None else process_identity(bound['pid'])
        result['supervisor_identity_matches']=bound is not None and current is not None and all(current[k]==bound[k] for k in ('pid','start_ticks','boot_id','cmdline'))
        result['supervisor_running']=result['supervisor_identity_matches'] and current['state'] not in ('Z','X')
    logs=execution/'model_processes'
    if logs.exists():
        for started in sorted(logs.glob('fold*_seed*.started.json')):
            label=started.name[:-len('.started.json')];finished=logs/(label+'.exit.json')
            if finished.exists():
                if read(finished)['actual_exit_code']==0:result['completed_models'].append(label)
                continue
            record=read(started);bound=record['identity'];current=None if bound is None else process_identity(bound['pid'])
            matched=bound is not None and current is not None and all(current[k]==bound[k] for k in ('pid','start_ticks','boot_id','cmdline'))
            stdout=logs/(label+'.stdout.log')
            result['active_workers'].append(dict(model=label,identity_matches=matched,running=matched and current['state'] not in ('Z','X'),
                latest_log_lines=stdout.read_text(errors='replace').splitlines()[-5:] if stdout.exists() else []))
    result['global_seal_exists']=(out/GLOBAL_SEAL).exists();result['evaluation_summary_exists']=(out/'evaluation/summary.json').exists()
    if (execution/'execution_receipt.json').exists():result['execution_receipt_status']=read(execution/'execution_receipt.json')['status']
    if (out/'completion_receipt.json').exists() and (execution/'finalization.exit.json').exists():
        final=read(execution/'finalization.exit.json');result['finalization_actual_exit_code']=final['actual_exit_code']
        if final['actual_exit_code']==0:result['status']='COMPLETE'
    return result


def synthetic(number):
    checks=[]
    if number==1:
        with tempfile.TemporaryDirectory(prefix='oof_atomic_') as td:
            p=Path(td)/'value.json';write_new(p,dict(value=1))
            try:write_new(p,dict(value=2))
            except FileExistsError:pass
            else:raise AssertionError('Atomic publication replaced evidence')
            require(read(p)==dict(value=1),'Publication changed')
            checks.append('atomic_complete_nonreplacing_JSON')
            for bad in ('../escape','/absolute'):
                try:safe_member(Path(td),bad)
                except RuntimeError:pass
                else:raise AssertionError('Unsafe member accepted')
            checks.append('sealed_member_traversal_rejected')
    elif number==2:
        with tempfile.TemporaryDirectory(prefix='oof_children_') as td:
            d=Path(td);r=child([sys.executable,'-c','print(7)'],d,'success');require(r['actual_exit_code']==0,'Actual success exit missing')
            checks.append('actual_child_wait_success_and_identity')
            try:child([sys.executable,'-c','raise SystemExit(3)'],d,'failure')
            except RuntimeError:pass
            else:raise AssertionError('Child failure ignored')
            require(read(d/'failure.exit.json')['actual_exit_code']==3,'Actual failure exit missing')
            checks.append('actual_child_wait_failure_receipt_and_stop')
    elif number==3:
        with tempfile.TemporaryDirectory(prefix='oof_seal_gate_') as td:
            d=Path(td);spec=dict(folds=list(range(5)),seeds=[20261024,20261025,20261026],score_columns=list(range(10)))
            for value in (None,dict(status=SEALED,model_count=14)):
                if value is not None:write_new(d/GLOBAL_SEAL,value)
                try:verify_global_seal(d,spec)
                except (FileNotFoundError,RuntimeError):pass
                else:raise AssertionError('Missing global seal/dimension accepted')
            checks.append('missing_global_seal_and_incomplete_fifteen_models_rejected')
            current=process_identity(os.getpid());require(current is not None and current['pid']==os.getpid() and current['state'] not in ('Z','X'),'Current process binding failed')
            checks.append('process_identity_binds_boot_startticks_cmdline')
    else:raise ValueError('Only three synthetic rounds')
    return dict(status='PASS_IDENTITY_SPACETIME_OOF_COORDINATOR_SYNTHETIC',round=number,checks=checks,script_sha256=sha(SELF),real_data_opened=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch',action='store_true');group.add_argument('--supervise',type=Path);group.add_argument('--execute',action='store_true');group.add_argument('--status',action='store_true');group.add_argument('--synthetic-round',type=int,choices=(1,2,3))
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--execution',type=Path);parser.add_argument('--preflight-sha256');parser.add_argument('--synthetic-out',type=Path);args=parser.parse_args()
    if args.supervise:supervise(args.supervise);return
    if args.execute:require(args.execution is not None,'Execution directory required');execute(args.protocol,args.execution);return
    if args.status:print(json.dumps(status(args.protocol),ensure_ascii=False,indent=2));return
    result=launch(args.protocol,args.preflight_sha256) if args.launch else synthetic(args.synthetic_round)
    if args.synthetic_out:write_new(args.synthetic_out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
