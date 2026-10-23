#!/usr/bin/env python3
"""Actual subprocess/wait synthetic audit of donor-out detached runner chain."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time


PRODUCER = r'''import argparse,hashlib,json,os,sys
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def authenticate(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--protocol');a=p.parse_args();s=authenticate(a.protocol)
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
 with Path(s['journal']).open('a') as f:f.write(json.dumps(dict(role='producer',pid=os.getpid(),ppid=os.getppid()))+'\n')
 if s['scenario']=='producer_fail':sys.exit(17)
 out=Path(s['output_directory']);out.mkdir()
 write(out/'GT_free_predictions_sealed.json',dict(status='SYNTHETIC_SEAL'))
 write(out/'summary.json',dict(minimal_gate='NOT_PASSED_MINIMAL_GTFREE_DONOR_PREDICTABILITY',readouts={'proposal_overlap_excluded':{'1':{'synthetic':True},'4':{'synthetic':True}}}))
 write(out/'producer_receipt.json',dict(summary_sha256=sha(out/'summary.json'),seal_sha256=sha(out/'GT_free_predictions_sealed.json')))
 if s['scenario']=='finalizer_fail':write(out/'predictability_brief.json',dict(preexisting=True))
'''

AUDITOR = r'''import argparse,hashlib,json,os,sys
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--protocol');a=p.parse_args();s=json.loads(Path(a.protocol).read_text())
with Path(s['journal']).open('a') as f:f.write(json.dumps(dict(role='auditor',pid=os.getpid(),ppid=os.getppid()))+'\n')
if s['scenario']=='auditor_fail':sys.exit(18)
out=Path(s['output_directory'])
result=dict(status='PASS_INDEPENDENT_SCENE_TRANSLATION_DONOROUT_OUTPUT_AUDIT',protocol_sha256=sha(a.protocol),seal_sha256=sha(out/'GT_free_predictions_sealed.json'),summary_sha256=sha(out/'summary.json'))
if s['scenario']=='auditor_bad_status':result['status']='FAIL_SYNTHETIC_AUDIT'
if s['scenario']=='auditor_bad_summary':result['summary_sha256']='0'*64
(out/'independent_output_audit.json').write_text(json.dumps(result))
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path):
    spec=importlib.util.spec_from_file_location('independent_runner_subject',str(path))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def read(path):return json.loads(Path(path).read_text())


def scenario(runner,runner_path,root,name,require_summary_binding):
    trial=root/name;trial.mkdir();execution=trial/'execution';execution.mkdir()
    out=trial/'output';producer=trial/'producer.py';auditor=trial/'auditor.py'
    producer.write_text(PRODUCER);auditor.write_text(AUDITOR)
    protocol=trial/'protocol.json';journal=trial/'journal.jsonl'
    spec=dict(producer_path=str(producer),producer_sha256=sha(producer),auditor_path=str(auditor),
              execution_directory=str(execution),output_directory=str(out),journal=str(journal),scenario=name)
    protocol.write_text(json.dumps(spec))
    pins={str(producer):sha(producer),str(auditor):sha(auditor),str(protocol):sha(protocol),str(runner_path):sha(runner_path),str(runner.HELPERS):runner.HELPER_SHA}
    launch=execution/'launch_spec.json'
    launch.write_text(json.dumps(dict(protocol=str(protocol),execution_directory=str(execution),output_directory=str(out),file_pins=pins)))
    expected_failure=name!='success'
    if name=='auditor_bad_summary' and not require_summary_binding:
        expected_failure=False
    failed=False
    try:runner.supervise(launch)
    except (RuntimeError,FileExistsError):failed=True
    assert failed==expected_failure,(name,failed,expected_failure)
    records=[json.loads(line) for line in journal.read_text().splitlines()]
    producer_exit=read(execution/'producer.exit.json')
    assert producer_exit['actual_exit_code']==(17 if name=='producer_fail' else 0)
    assert producer_exit['waited_by_parent_pid']==os.getpid()
    for label in ('producer','independent_output_audit','finalization'):
        started=execution/(label+'.started.json');finished=execution/(label+'.exit.json')
        if started.exists():
            assert finished.exists(),(name,label,'started without actual wait receipt')
            value=read(finished)
            assert value['waited_by_parent_pid']==os.getpid()
            assert value['pid']!=os.getpid() and value['elapsed_seconds']>=0
            assert value['identity'] is not None
            for key in ('pid','start_ticks','boot_id','cmdline'):
                assert key in value['identity']
            assert read(started)['identity']==value['identity']
            assert value['identity']['pid']==value['pid']
    if name=='producer_fail':
        assert [r['role'] for r in records]==['producer']
        assert not (execution/'independent_output_audit.started.json').exists()
        assert not (execution/'execution_receipt.json').exists()
    else:
        assert [r['role'] for r in records]==['producer','auditor']
        assert records[0]['pid']!=records[1]['pid']
        assert records[0]['ppid']==os.getpid() and records[1]['ppid']==os.getpid()
        audit_exit=read(execution/'independent_output_audit.exit.json')
        assert audit_exit['actual_exit_code']==(18 if name=='auditor_fail' else 0)
    if name in ('producer_fail','auditor_fail','auditor_bad_status') or (name=='auditor_bad_summary' and require_summary_binding):
        assert not (execution/'finalization.started.json').exists()
        assert not (out/'completion_receipt.json').exists()
    elif name=='finalizer_fail':
        assert read(execution/'finalization.exit.json')['actual_exit_code']!=0
        assert not (out/'completion_receipt.json').exists()
    else:
        assert read(execution/'finalization.exit.json')['actual_exit_code']==0
        completed=read(execution/'supervisor_complete.json')
        assert completed['status']=='COMPLETE_WITH_WAITED_FINALIZER'
        assert completed['finalization_actual_exit_code']==0
        assert completed['completion_receipt_sha256']==sha(out/'completion_receipt.json')
        assert read(out/'completion_receipt.json')['brief_sha256']==sha(out/'predictability_brief.json')
    assert (execution/'execution_failure.json').exists()==expected_failure
    assert (execution/'supervisor_complete.json').exists()==(not expected_failure)
    # Exercise PID-bound status without relying on a still-alive child.
    identity=dict(pid=os.getpid(),start_ticks=-1,boot_id='wrong',cmdline=['wrong'],state='R')
    (execution/'supervisor_identity.json').write_text(json.dumps(identity))
    status=runner.status(protocol)
    assert status['supervisor_identity_matches'] is False and status['supervisor_running'] is False
    if expected_failure:
        assert status['status']!='COMPLETE'
    else:
        assert status['status']=='COMPLETE'
    return dict(scenario=name,expected_failure=expected_failure,actual_child_exit_codes={
        label:read(execution/(label+'.exit.json'))['actual_exit_code']
        for label in ('producer','independent_output_audit','finalization') if (execution/(label+'.exit.json')).exists()},
        producer_and_auditor_are_distinct_fresh_processes=(len(records)==2 and records[0]['pid']!=records[1]['pid']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--require-summary-binding',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    assert sha(args.runner)==args.expected_sha256
    runner=module(args.runner)
    with tempfile.TemporaryDirectory(prefix='donorout-independent-runner-') as tmp:
        trials=[scenario(runner,args.runner.resolve(),Path(tmp),name,args.require_summary_binding) for name in
            ('success','producer_fail','auditor_fail','finalizer_fail','auditor_bad_status','auditor_bad_summary')]
    assert sha(args.runner)==args.expected_sha256,'Runner changed during audit'
    result=dict(status='PASS_INDEPENDENT_DONOROUT_RUNNER_ACTUAL_CHILD_WAIT_AND_FAILSTOP_SYNTHETIC_AUDIT',
        runner_sha256=args.expected_sha256,auditor_sha256=sha(__file__),helper_sha256=sha(runner.HELPERS),
        authentication_boundary='temporary protocol and fixture producer.authenticate; real root files unchanged',
        actual_Popen_and_parent_wait_used=True,summary_binding_required=args.require_summary_binding,
        trials=trials,real_cache_opened=False,new_OOF_outputs_opened=False,
        elapsed_seconds=time.monotonic()-started)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
