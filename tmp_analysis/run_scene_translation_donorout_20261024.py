#!/usr/bin/env python3
"""Detached CPU producer -> fresh independent audit -> waited finalizer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'tmp_analysis/scene_translation_donorout_protocol_20261024.json'
HELPERS=ROOT/'tmp_analysis/run_identity_spacetime_oof_20261024.py'
HELPER_SHA='b291c36aeb3ff92ae375954fb6033b35ecdba46c7a88728cc4b91c8048558133'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise RuntimeError(message)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def import_path(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m


def helpers():
    require(sha(HELPERS)==HELPER_SHA,'Frozen process helper changed')
    return import_path(HELPERS,'donorout_process_helpers')


def authenticate(protocol):
    spec=read(protocol)
    require(sha(spec['producer_path'])==spec['producer_sha256'],'Producer changed')
    module=import_path(spec['producer_path'],'donorout_producer_authentication')
    return module.authenticate(protocol)


def authenticate_launch(launch):
    helper=helpers()
    for path,digest in launch['file_pins'].items():require(sha(path)==digest,'Launch pin changed: '+path)
    spec=authenticate(launch['protocol'])
    require(spec['execution_directory']==launch['execution_directory'],'Execution destination changed')
    return helper,spec


def finalize(launch_path):
    launch=read(launch_path);helper,spec=authenticate_launch(launch)
    execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    require(read(execution/'execution_receipt.json')['status']=='PASS_DONOROUT_PRODUCER_AND_FRESH_INDEPENDENT_AUDIT','Execution not passed')
    for label in ('producer','independent_output_audit'):
        require(read(execution/(label+'.exit.json'))['actual_exit_code']==0,'Child did not exit successfully')
    audit=read(out/'independent_output_audit.json')
    require(audit['status']=='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_OUTPUT_AUDIT','Output audit not passed')
    require(audit['protocol_sha256']==sha(launch['protocol']),'Audit protocol mismatch')
    require(audit['seal_sha256']==sha(out/'GT_free_predictions_sealed.json'),'Audit seal mismatch')
    require(audit['summary_sha256']==sha(out/'summary.json'),'Audit summary mismatch')
    summary=read(out/'summary.json');checks=read(out/'producer_receipt.json')
    require(checks['summary_sha256']==sha(out/'summary.json'),'Producer summary changed')
    require(checks['seal_sha256']==sha(out/'GT_free_predictions_sealed.json'),'Producer seal changed')
    write(out/'predictability_brief.json',dict(status=summary['minimal_gate'],
        primary_strict={str(l):summary['readouts']['proposal_overlap_excluded'][str(l)] for l in (1,4)},
        summary_sha256=sha(out/'summary.json'),audit_sha256=sha(out/'independent_output_audit.json'),
        is_identity_or_tracking_or_novelty_evidence=False))
    write(out/'completion_receipt.json',dict(status='PASS_DONOROUT_DIAGNOSTIC_AND_INDEPENDENT_OUTPUT_AUDIT',
        launch_sha256=sha(launch_path),execution_receipt_sha256=sha(execution/'execution_receipt.json'),
        protocol_sha256=sha(launch['protocol']),seal_sha256=sha(out/'GT_free_predictions_sealed.json'),
        independent_audit_sha256=sha(out/'independent_output_audit.json'),
        summary_sha256=sha(out/'summary.json'),brief_sha256=sha(out/'predictability_brief.json'),
        protected_files_unchanged=True,all_pinned_files_unchanged=True,
        method_effectiveness_or_novelty_established=False))


def supervise(launch_path):
    launch=read(launch_path);helper,spec=authenticate_launch(launch)
    execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    try:
        producer=helper.child([sys.executable,'-u',spec['producer_path'],'--execute','--protocol',launch['protocol']],execution,'producer')
        helper,spec=authenticate_launch(launch)
        auditor=helper.child([sys.executable,'-u',spec['auditor_path'],'--verify','--protocol',launch['protocol']],execution,'independent_output_audit')
        helper,spec=authenticate_launch(launch)
        audit=read(out/'independent_output_audit.json')
        require(audit['status']=='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_OUTPUT_AUDIT','Audit did not pass')
        require(audit['protocol_sha256']==sha(launch['protocol']) and audit['seal_sha256']==sha(out/'GT_free_predictions_sealed.json'),'Audit lineage mismatch')
        require(audit['summary_sha256']==sha(out/'summary.json'),'Audit summary mismatch')
        write(execution/'execution_receipt.json',dict(status='PASS_DONOROUT_PRODUCER_AND_FRESH_INDEPENDENT_AUDIT',
            producer_actual_exit_code=producer['actual_exit_code'],auditor_actual_exit_code=auditor['actual_exit_code'],
            producer_exit_sha256=sha(execution/'producer.exit.json'),
            auditor_exit_sha256=sha(execution/'independent_output_audit.exit.json'),
            audit_sha256=sha(out/'independent_output_audit.json'),protocol_sha256=sha(launch['protocol'])))
        finalizer=helper.child([sys.executable,'-u',str(Path(__file__).resolve()),'--finalize',str(launch_path)],execution,'finalization')
        write(execution/'supervisor_complete.json',dict(status='COMPLETE_WITH_WAITED_FINALIZER',
            finalization_actual_exit_code=finalizer['actual_exit_code'],
            finalization_exit_sha256=sha(execution/'finalization.exit.json'),
            completion_receipt_sha256=sha(out/'completion_receipt.json')))
    except BaseException as error:
        write(execution/'execution_failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',
            error_type=type(error).__name__,error=str(error)))
        raise


def launch(protocol,preflight,preflight_sha):
    helper=helpers();spec=authenticate(protocol)
    require(sha(preflight)==preflight_sha,'Preflight must have exact external hash')
    gate=read(preflight)
    require(gate['status']=='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_PREFLIGHT','Preflight did not pass')
    require(gate['protocol_sha256']==sha(protocol),'Preflight protocol mismatch')
    require(gate['runner_sha256']==sha(__file__),'Preflight runner mismatch')
    require(gate['producer_sha256']==sha(spec['producer_path']),'Preflight producer mismatch')
    require(gate['auditor_sha256']==sha(spec['auditor_path']),'Preflight auditor mismatch')
    require(gate['cached_module_sha256']==sha(spec['cached_module_path']),'Preflight cached module mismatch')
    execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    require(not execution.exists() and not out.exists(),'Never restart or overwrite existing diagnostic')
    require(shutil.disk_usage(ROOT).free>=spec['minimum_free_disk_bytes'],'Insufficient free disk')
    execution.mkdir()
    pins=dict(spec['input_pins']);pins[str(Path(protocol).resolve())]=sha(protocol)
    pins[str(Path(preflight).resolve())]=preflight_sha
    record=dict(protocol=str(Path(protocol).resolve()),execution_directory=str(execution),
        output_directory=str(out),preflight=str(Path(preflight).resolve()),file_pins=pins,
        CPU_only=True,automatic_retry=False)
    write(execution/'launch_spec.json',record)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--supervise',str(execution/'launch_spec.json')]
    with (execution/'supervisor.stdout.log').open('x') as stdout,(execution/'supervisor.stderr.log').open('x') as stderr:
        process=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
    identity=helper.process_identity(process.pid)
    require(identity is not None,'Supervisor identity could not be read')
    write(execution/'supervisor_identity.json',identity)
    write(execution/'launch_receipt.json',dict(status='LAUNCHED_CPU_DIAGNOSTIC_NOT_COMPLETE',
        supervisor_identity=identity,command=command,launch_sha256=sha(execution/'launch_spec.json')))
    return dict(execution_directory=str(execution),supervisor_identity=identity)


def status(protocol):
    spec=read(protocol);execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    helper=helpers();result=dict(execution_directory=str(execution),status='NOT_LAUNCHED',workers=[])
    identity_path=execution/'supervisor_identity.json'
    if not identity_path.exists():return result
    bound=read(identity_path);live=helper.process_identity(bound['pid'])
    matched=live is not None and all(bound[k]==live[k] for k in ('pid','start_ticks','boot_id','cmdline'))
    result.update(status='RUNNING_OR_INTERRUPTED',supervisor_identity_matches=matched,
        supervisor_running=matched and live['state'] not in ('Z','X'))
    for label in ('producer','independent_output_audit','finalization'):
        finished=execution/(label+'.exit.json');started=execution/(label+'.started.json')
        if finished.exists():result['workers'].append(dict(label=label,actual_exit_code=read(finished)['actual_exit_code']));continue
        if started.exists():
            identity=read(started)['identity'];current=None if identity is None else helper.process_identity(identity['pid'])
            valid=current is not None and identity is not None and all(identity[k]==current[k] for k in ('pid','start_ticks','boot_id','cmdline'))
            result['workers'].append(dict(label=label,identity_matches=valid,running=valid and current['state'] not in ('Z','X'),
                latest_log_lines=(execution/(label+'.stdout.log')).read_text(errors='replace').splitlines()[-3:]))
    result['failure_exists']=(execution/'execution_failure.json').exists()
    if (execution/'supervisor_complete.json').exists() and (execution/'finalization.exit.json').exists():
        if read(execution/'finalization.exit.json')['actual_exit_code']==0 and (out/'completion_receipt.json').exists():result['status']='COMPLETE'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch',action='store_true');group.add_argument('--supervise',type=Path)
    group.add_argument('--finalize',type=Path);group.add_argument('--status',action='store_true')
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--preflight',type=Path);parser.add_argument('--preflight-sha256')
    args=parser.parse_args()
    if args.supervise:supervise(args.supervise)
    elif args.finalize:finalize(args.finalize)
    elif args.status:print(json.dumps(status(args.protocol),indent=2))
    else:
        require(args.preflight is not None and args.preflight_sha256 is not None,'Preflight path/hash required')
        print(json.dumps(launch(args.protocol,args.preflight,args.preflight_sha256),indent=2))
