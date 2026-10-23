"""Detached, bounded completion audit for an already launched collection.

Waits only for this runner PID/output. Never launches GPU work, training, GT
matching or a retry. An interrupted collection stays partial and is reported.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pid", type=int, required=True)
    args = ap.parse_args()
    report = args.run / "background_completion_audit.json"
    assert not report.exists()
    state = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), supervisor_pid=os.getpid(), collection_pid=args.pid,
                 run=str(args.run.resolve()), launches_gpu_job=False, training_started=False, stages=[])
    deadline = time.monotonic() + 1200
    receipt = args.run / "collection_receipt.json"
    while not receipt.exists() and time.monotonic() < deadline:
        cmdline = Path("/proc/%d/cmdline" % args.pid)
        if not cmdline.exists() or b"collect_base_error_pilot.py" not in cmdline.read_bytes():
            state["status"] = "COLLECTION_PROCESS_ENDED_WITHOUT_COMPLETION_RECEIPT"
            break
        time.sleep(5)
    else:
        state["status"] = "COLLECTION_RECEIPT_PRESENT" if receipt.exists() else "WAIT_TIMEOUT_COLLECTION_NOT_SEALED"
    if receipt.exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        commands = [
            ("dump_verification.json", [PYTHON, str(ROOT / "tmp_analysis/verify_base_error_dump.py"),
              "--dump-dir", str(args.run / "dump"), "--annotation", str(args.stage / "pilot_inference.json"),
              "--expected-foreground-classes", "1203", "--output", str(args.run / "dump_verification.json")]),
            ("independent_collection_audit.json", [PYTHON, str(ROOT / "tmp_analysis/verify_base_error_collection.py"),
              "--stage", str(args.stage), "--run", str(args.run), "--output", str(args.run / "independent_collection_audit.json")])]
        for name, cmd in commands:
            target = args.run / name
            if target.exists():
                state["stages"].append(dict(name=name, execution="preexisting_not_reexecuted", sha256=sha(target)))
                continue
            with (args.run / (name + ".background.log")).open("x") as log:
                child = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            state["stages"].append(dict(name=name, observed_exit_code=child.returncode,
                                        sha256=sha(target) if target.exists() else None))
            if child.returncode:
                state["status"] = "BACKGROUND_COLLECTION_AUDIT_FAILED"
                break
        else:
            state["status"] = "PASS_BACKGROUND_COLLECTION_AUDITS_PENDING_ROOT_REVIEW"
        state["collection_receipt_sha256"] = sha(receipt)
    state["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state["original_collection_terminal_exit_code"] = None
    state["original_exit_limitation"] = "Original runner is not a child of this detached supervisor; do not invent its exit code. Root tool session91862 can provide it."
    report.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
