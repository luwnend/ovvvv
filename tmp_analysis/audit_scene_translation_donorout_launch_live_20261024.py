#!/usr/bin/env python3
"""Read-only startup audit: pins, protected bytes and live CPU process facts."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time


EXPECTED_ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise AssertionError(message)


def helpers(path):
    spec=importlib.util.spec_from_file_location('live_process_identity_helpers',str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def proc_snapshot(helper,bound,expected_executable):
    current=helper.process_identity(bound['pid'])
    require(current is not None and all(current[k]==bound[k] for k in ('pid','start_ticks','boot_id','cmdline')),'Live process identity changed')
    require(current['state'] not in ('Z','X'),'Process is not live')
    process_dir=Path('/proc')/str(bound['pid'])
    # Only explicitly named environment keys are emitted. Other values, which
    # could contain credentials, are never returned or logged.
    raw=(process_dir/'environ').read_bytes().split(b'\0')
    selected={}
    for entry in raw:
        if b'=' not in entry:continue
        key,value=entry.split(b'=',1)
        name=key.decode(errors='replace')
        if name in EXPECTED_ENV:selected[name]=value.decode(errors='replace')
    require(selected==EXPECTED_ENV,'CPU/thread environment changed')
    status={}
    for line in (process_dir/'status').read_text().splitlines():
        if ':' not in line:continue
        key,value=line.split(':',1)
        if key in ('State','PPid','Threads','VmRSS','VmPeak','Cpus_allowed_list'):status[key]=value.strip()
    require(int(status['Threads'])==1,'Expected single-thread process')
    executable=(process_dir/'exe').resolve()
    require(executable==Path(expected_executable).resolve(),'Pinned Python executable changed')
    fds=[]
    for fd in (process_dir/'fd').iterdir():
        try:target=str(fd.resolve())
        except FileNotFoundError:continue
        if target.startswith('/dev/nvidia') or target.startswith('/dev/dri'):
            fds.append(target)
    require(not fds,'Live process has GPU device file open')
    after=helper.process_identity(bound['pid'])
    require(after is not None and all(after[k]==current[k] for k in ('pid','start_ticks','boot_id','cmdline')),'Identity changed during inspection')
    return dict(identity=current,selected_environment=selected,status=status,
        resolved_executable=str(executable),GPU_device_file_descriptors=fds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--expected-launch-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    spec=read(args.protocol);execution=Path(spec['execution_directory'])
    launch_path=execution/'launch_spec.json';launch=read(launch_path)
    require(sha(launch_path)==args.expected_launch_sha256,'Launch external SHA mismatch')
    require(launch['CPU_only'] is True and launch['automatic_retry'] is False,'Launch policy changed')
    require(launch['protocol']==str(args.protocol.resolve()),'Launch protocol destination changed')
    require(launch['execution_directory']==spec['execution_directory'] and launch['output_directory']==spec['output_directory'],'Launch output boundary changed')
    pins=[]
    for name,expected in launch['file_pins'].items():
        require(sha(name)==expected,'Launch pin changed: '+name)
        pins.append(dict(path=name,sha256=expected))
    require(len(spec['input_pins'])==37 and len(launch['file_pins'])==39,'Launch pin population changed')
    require(all(launch['file_pins'].get(p)==d for p,d in spec['input_pins'].items()),'Frozen 37 pins not in launch')
    preflight=read(launch['preflight'])
    require(preflight['status']=='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_PREFLIGHT','Launch gate status changed')
    require(preflight['protocol_sha256']==sha(args.protocol),'Launch gate protocol changed')
    checker=Path(preflight['checker_path'])
    require(sha(checker)==preflight['checker_sha256'],'External preflight checker changed')
    authority=read(spec['protected_authority_path']);protected={}
    for name,expected in authority['protected_files'].items():
        current=dict(exists=Path(name).exists(),sha256=sha(name) if Path(name).exists() else None)
        require(current==expected,'Protected file changed: '+name);protected[name]=current
    require(len(protected)==7,'Protected file population changed')
    helper_path=Path(spec['producer_path']).with_name('run_identity_spacetime_oof_20261024.py')
    require(sha(helper_path)==launch['file_pins'][str(helper_path)],'Frozen process helper changed')
    helper=helpers(helper_path)
    bound=read(execution/'supervisor_identity.json')
    require(bound==read(execution/'launch_receipt.json')['supervisor_identity'],'Supervisor receipt mismatch')
    supervisor=proc_snapshot(helper,bound,'/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python')
    producer_start=read(execution/'producer.started.json')
    producer=proc_snapshot(helper,producer_start['identity'],'/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python')
    require(int(producer['status']['PPid'])==bound['pid'],'Producer is not supervisor child')
    require(producer_start['command']==producer['identity']['cmdline'],'Producer command differs from receipt')
    require('--execute' in producer['identity']['cmdline'] and spec['producer_path'] in producer['identity']['cmdline'],'Unexpected live producer entrypoint')
    require(not (execution/'execution_failure.json').exists(),'Execution failure marker exists')
    require(not (execution/'producer.exit.json').exists(),'Startup audit expected a still-running producer')
    require(not (execution/'supervisor_complete.json').exists(),'Startup audit cannot certify completed output')
    result=dict(status='PASS_INDEPENDENT_DONOROUT_LAUNCH_LIVE_STARTUP_AUDIT_NOT_COMPLETION',
        recorded_utc=datetime.now(timezone.utc).isoformat(),auditor_path=str(Path(__file__).resolve()),auditor_sha256=sha(__file__),
        launch_sha256=args.expected_launch_sha256,protocol_sha256=sha(args.protocol),
        preflight_sha256=sha(launch['preflight']),preflight_checker_sha256=sha(checker),
        frozen_input_pin_count=37,launch_pin_count=39,launch_pin_checks=pins,
        protected_files=protected,supervisor=supervisor,producer=producer,
        live_CPU_evidence='CUDA disabled, all three thread caps one, one live OS thread, pinned Python executable, no GPU device FD',
        execution_failure_marker_exists=False,completion_certified=False,
        diagnostic_or_main_OOF_metrics_read=False,diagnostic_prediction_values_read=False,
        raw_NPZ_arrays_or_SQLite_or_GT_decoded=False,existing_pinned_files_mutated=False,
        elapsed_seconds=time.monotonic()-started)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('launch_pin_checks','protected_files')},indent=2))


if __name__=='__main__':main()
