"""Collect one frozen full fit/tune stream, record exit, then run CPU audits.

Detached execution entry point. No retry, learner, GT matching or automatic
method selection. Uses the audited collector and explicit full metadata.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, obj):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    stage, run = args.stage.resolve(), args.run.resolve()
    assert run.parent == stage and not run.exists()
    annotation = stage / "full_inference.json"
    preflight = stage / "independent_metadata_audit.json"
    assert annotation.is_file() and preflight.is_file()
    verified = json.loads(preflight.read_text())
    assert verified["status"] == "PASS_INDEPENDENT_FULL_COLLECTION_METADATA_AUDIT"
    assert verified["annotation_sha256"] == sha(annotation)
    logs = stage / "execution_logs"
    logs.mkdir(exist_ok=True)
    start = dict(status="STARTING_FROZEN_FULL_FIT_TUNE_COLLECTION", started_utc=utc(), supervisor_pid=os.getpid(),
                 stage=str(stage), run=str(run), annotation_sha256=sha(annotation),
                 independent_metadata_audit_sha256=sha(preflight), collector_sha256=sha(ROOT / "tmp_analysis/collect_base_error_pilot.py"),
                 full_validator_sha256=sha(ROOT / "tmp_analysis/verify_base_error_full_collection.py"),
                 supervisor_sha256=sha(__file__), new_method_training=False, test_images_selected=False)
    write(stage / "background_launch_receipt.json", start)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [PYTHON, "-u", str(ROOT / "tmp_analysis/collect_base_error_pilot.py"), "--annotation", str(annotation), "--out", str(run)]
    with (logs / (run.name + ".log")).open("x") as log:
        child = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    result = dict(status="COLLECTION_FAILED" if child.returncode else "COLLECTION_EXIT_ZERO_PENDING_CPU_AUDIT", completed_utc=utc(),
                  collection_command=command, terminal_exit_code=child.returncode, terminal_exit_observed=True,
                  observed_by="detached supervisor subprocess.run of its own child", collection_log_sha256=sha(logs / (run.name + ".log")),
                  launch_receipt_sha256=sha(stage / "background_launch_receipt.json"),
                  stages=[], new_method_training=False, research_goal_complete=False)
    if child.returncode == 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
        commands = [
            ("dump_verification.json", [PYTHON, str(ROOT / "tmp_analysis/verify_base_error_dump.py"), "--dump-dir", str(run / "dump"),
              "--annotation", str(annotation), "--expected-foreground-classes", "1203", "--output", str(run / "dump_verification.json")]),
            ("independent_collection_audit.json", [PYTHON, str(ROOT / "tmp_analysis/verify_base_error_full_collection.py"),
              "--stage", str(stage), "--run", str(run), "--output", str(run / "independent_collection_audit.json")])]
        for name, command in commands:
            with (logs / (run.name + "." + name + ".log")).open("x") as log:
                audit = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            result["stages"].append(dict(file=name, terminal_exit_code=audit.returncode, terminal_exit_observed=True,
                                          sha256=sha(run / name) if (run / name).exists() else None))
            if audit.returncode:
                result["status"] = "FULL_COLLECTION_CPU_AUDIT_FAILED"
                break
        else:
            result["status"] = "PASS_FULL_COLLECTION_EXECUTION_AND_CPU_AUDITS_PENDING_ROOT_REVIEW"
        result["collection_receipt_sha256"] = sha(run / "collection_receipt.json")
    write(stage / "background_execution_receipt.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] == "FULL_COLLECTION_CPU_AUDIT_FAILED":
        return 1
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
