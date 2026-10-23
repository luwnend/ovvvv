#!/usr/bin/env python3
"""Independent real detached-process framework audit with synthetic workers.

Uses the frozen coordinator's actual launch/supervise/child/status/PID identity.
Numerical authenticate/model/global-seal gates are explicitly synthetic stubs.
This audits process supervision only, never fifteen-model mathematics or data.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"tmp_analysis/run_identity_spacetime_oof_20261024.py"
RUNNER_SHA="b291c36aeb3ff92ae375954fb6033b35ecdba46c7a88728cc4b91c8048558133"
PYTHON="/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python"


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runner():
    require(sha(RUNNER)==RUNNER_SHA,"Frozen coordinator changed")
    spec=importlib.util.spec_from_file_location("independent_background_real_runner",RUNNER);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);return runner


def write(path,value):
    with Path(path).open("x") as f:f.write(json.dumps(value,indent=2)+"\n")


WRAPPER=r'''import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
RUNNER_PATH=__RUNNER_PATH__
RUNNER_SHA=__RUNNER_SHA__
assert hashlib.sha256(Path(RUNNER_PATH).read_bytes()).hexdigest()==RUNNER_SHA
s=importlib.util.spec_from_file_location('real_frozen_oof_supervisor',RUNNER_PATH);runner=importlib.util.module_from_spec(s);s.loader.exec_module(runner)
parser=argparse.ArgumentParser();parser.add_argument('--supervise');parser.add_argument('--execute',action='store_true');parser.add_argument('--protocol');parser.add_argument('--execution');args=parser.parse_args()
runner.SELF=Path(__file__).resolve()
runner.authenticate=lambda protocol,check_gate=True:runner.read(protocol)
runner.all_pins=lambda spec:dict(spec['fixture_pins'])
def synthetic_global(out,spec):
    value=runner.read(Path(out)/runner.GLOBAL_SEAL)
    assert value['status']=='SYNTHETIC_NUMERICAL_GATE_STUB'
    return value
runner.verify_global_seal=synthetic_global
if args.supervise:
    runner.supervise(args.supervise)
else:
    spec=runner.read(args.protocol);out=Path(spec['output_directory']);out.mkdir()
    runner.write_new(out/'synthetic_coordinator_process.json',dict(pid=os.getpid(),parent=os.getppid(),session=os.getsid(0),numerical_gates_stubbed=True))
    time.sleep(.3)
    if spec['fixture_mode']=='coordinator_fail':raise SystemExit(3)
    runner.write_new(out/runner.GLOBAL_SEAL,dict(status='SYNTHETIC_NUMERICAL_GATE_STUB'))
'''
VERIFIER=r'''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--protocol');a=p.parse_args();spec=json.loads(Path(a.protocol).read_text());out=Path(a.out)
(out/'synthetic_auditor_process.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid(),session=os.getsid(0))))
if spec['fixture_mode']=='auditor_fail':raise SystemExit(5)
(out/'independent_output_audit.json').write_text(json.dumps(dict(status='PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT',synthetic_numeric_stub=True)))
'''
FINALIZER=r'''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--execution');a=p.parse_args();out=Path(a.out);execution=Path(a.execution)
r=json.loads((execution/'execution_receipt.json').read_text());assert r['status']=='PASS_IDENTITY_SPACETIME_OOF_EXECUTION_AND_INDEPENDENT_AUDIT' and r['coordinator']['actual_exit_code']==r['independent_auditor']['actual_exit_code']==0
spec=json.loads(Path(json.loads((execution/'launch_spec.json').read_text())['protocol']).read_text())
(out/'synthetic_finalizer_process.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid(),session=os.getsid(0))))
if spec['fixture_mode']=='finalizer_fail':raise SystemExit(7)
(out/'completion_receipt.json').write_text(json.dumps(dict(status='SYNTHETIC_PROCESS_FRAMEWORK_COMPLETE',numerical_result=False)))
'''


def run_case(root,mode):
    root.mkdir();wrapper=root/"wrapper.py";verifier=root/"verifier.py";finalizer=root/"finalizer.py"
    wrapper.write_text(WRAPPER.replace("__RUNNER_PATH__",repr(str(RUNNER))).replace("__RUNNER_SHA__",repr(RUNNER_SHA)))
    verifier.write_text(VERIFIER);finalizer.write_text(FINALIZER)
    gate=root/"preflight.json";write(gate,dict(status="SYNTHETIC_PROCESS_GATE_STUB"))
    out=root/"out";execution=root/"execution";protocol=root/"protocol.json"
    fixture_pins={str(p):sha(p) for p in (RUNNER,wrapper,verifier,finalizer)}
    write(protocol,dict(output_directory=str(out),execution_directory=str(execution),fixture_mode=mode,fixture_pins=fixture_pins,
        verifier_path=str(verifier),finalizer_path=str(finalizer),implementation_preflight=dict(path=str(gate)),resource_bounds=dict(minimum_free_disk_bytes=0)))
    runner=load_runner();runner.SELF=wrapper;runner.authenticate=lambda protocol,check_gate=True:runner.read(protocol);runner.all_pins=lambda spec:dict(spec["fixture_pins"])
    runner.verify_global_seal=lambda out,spec:runner.read(Path(out)/runner.GLOBAL_SEAL)
    launched=runner.launch(protocol,sha(gate));identity=launched["supervisor_identity"]
    require(identity is not None,"Launch lost kernel PID identity")
    pid=identity["pid"];detached_session=os.getsid(pid)
    require(detached_session==pid and detached_session!=os.getsid(0),"start_new_session did not detach supervisor session")
    status=runner.status(protocol);require(status["supervisor_identity_matches"] and status["supervisor_running"],"Live status did not bind kernel PID/start ticks/boot/cmdline")
    # This audit is the launcher's actual parent and really waits for the detached
    # supervisor, separately from its own child wait receipts.
    deadline=time.monotonic()+12;wait_status=None
    while time.monotonic()<deadline:
        got,value=os.waitpid(pid,os.WNOHANG)
        if got==pid:wait_status=value;break
        time.sleep(.02)
    require(wait_status is not None,"Synthetic detached supervisor exceeded bounded timeout")
    actual_exit=os.WEXITSTATUS(wait_status) if os.WIFEXITED(wait_status) else -os.WTERMSIG(wait_status)
    finalstatus=runner.status(protocol)
    expected=0 if mode=="success" else 1
    require(actual_exit==expected,"Detached supervisor actual exit differs")
    kernel=runner.read(execution/"supervisor_identity.json")["identity"]
    require(all(identity[k]==kernel[k] for k in ("pid","start_ticks","boot_id","cmdline")),"Detached supervisor kernel identity changed after launch")
    records={path.name:runner.read(path) for path in sorted(execution.glob("*.exit.json"))}
    for record in records.values():require(record["waited_by_parent_pid"]==pid,"Child exit was not actually waited by bound supervisor")
    if mode=="success":
        require(finalstatus["status"]=="COMPLETE" and all(records[name]["actual_exit_code"]==0 for name in ("coordinator.exit.json","independent_output_audit.exit.json","finalization.exit.json")),"Success did not pass all three child wait stages")
        chain=runner.read(execution/"execution_receipt.json")
        require(chain["coordinator"]==records["coordinator.exit.json"] and chain["independent_auditor"]==records["independent_output_audit.exit.json"],"PASS receipt does not bind actual child exits")
    else:
        require(finalstatus["status"]=="FAILED" and (execution/"execution_failure.json").exists() and not (out/"completion_receipt.json").exists(),"Failed actual stage falsely completed")
        label,code={"coordinator_fail":("coordinator.exit.json",3),"auditor_fail":("independent_output_audit.exit.json",5),"finalizer_fail":("finalization.exit.json",7)}[mode]
        require(records[label]["actual_exit_code"]==code,"Nonzero child exit evidence lost")
        if mode=="coordinator_fail":require("independent_output_audit.exit.json" not in records and "finalization.exit.json" not in records,"Failure did not stop later stages")
        if mode=="auditor_fail":require("finalization.exit.json" not in records and not (execution/"execution_receipt.json").exists(),"Failed auditor falsely published execution PASS")
        if mode=="finalizer_fail":require((execution/"execution_receipt.json").exists(),"Post-audit stage failure erased earlier actual success evidence")
    return dict(mode=mode,supervisor_actual_wait_exit_code=actual_exit,supervisor_identity=identity,supervisor_session_id=detached_session,
        final_status=finalstatus["status"],actual_child_exit_records=records,launch_receipt=launched,
        evidence_files_sha256={str(path.relative_to(root)):sha(path) for path in sorted(execution.glob("*.json"))},numerical_authentication_and_global_model_gates_are_explicit_stubs=True)


def audit():
    started=time.monotonic();cases=[]
    with tempfile.TemporaryDirectory(prefix="identity_oof_detached_audit_") as temp:
        for mode in ("success","coordinator_fail","auditor_fail","finalizer_fail"):cases.append(run_case(Path(temp)/mode,mode))
    require(sha(RUNNER)==RUNNER_SHA,"Real coordinator changed during process audit")
    checks=["actual_frozen_launch_supervise_child_functions_in_new_detached_Python_processes",
        "start_new_session_kernel_supervisor_PID_startticks_boot_cmdline_binding_and_live_status",
        "actual_parent_wait_for_detached_supervisor_and_all_three_child_stages_success",
        "coordinator_nonzero_exit3_rejected_and_auditor_finalizer_not_started",
        "independent_auditor_nonzero_exit5_rejected_and_execution_PASS_not_written",
        "finalizer_nonzero_exit7_preserves_prior_wait_chain_but_never_reports_complete",
        "success_auditor_finalizer_wiring_actual_exit_records_and_completed_status",
        "all_launch_pinned_bytes_and_frozen_real_runner_unchanged"]
    return dict(status="PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_BACKGROUND_PROCESS_AUDIT",checks=checks,cases=cases,
        script_sha256=sha(__file__),coordinator_sha256=RUNNER_SHA,elapsed_seconds=time.monotonic()-started,
        scope="Process supervision only; synthetic authenticate/model/global/numerical gates are stubs, not fifteen-model mathematical or data validation.",
        real_data_or_identity_opened=False,training_started=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    result=audit();write(args.out,result);print(json.dumps(dict(status=result["status"],checks=len(result["checks"]),elapsed_seconds=result["elapsed_seconds"],script_sha256=result["script_sha256"])))


if __name__=="__main__":main()
