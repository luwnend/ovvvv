"""Detached, receipt-bound execution of the frozen FIT lag support audit.

The producer alone creates the support output. This supervisor creates a
separate execution directory, records actual child exits, then runs the
independent output verifier. No retry, training or selection is performed.
Linux /proc start ticks, boot ID and cmdline distinguish reused PIDs.
"""
import argparse
import datetime
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PYTHON = Path("/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python").resolve()
OUT = ROOT / "results/fit_lag_identity_support_20261024_v1"
EXECUTION = ROOT / "results/fit_lag_identity_support_execution_20261024_v1"
PRODUCER = ROOT / "tmp_analysis/audit_fit_lag_identity_support_20261024.py"
PROTOCOL = ROOT / "tmp_analysis/fit_lag_identity_support_protocol_20261024.json"
PREFLIGHT = ROOT / "tmp_analysis/fit_lag_identity_support_independent_preflight_20261024.json"
CHECKER = ROOT / "tmp_analysis/audit_fit_lag_identity_support_implementation_20261024.py"
VERIFIER = ROOT / "tmp_analysis/verify_fit_lag_identity_support_output_20261024.py"
AUTHORITY = ROOT / "tmp_analysis/feature_spacetime_protected_before_20261024.json"
VERIFIER_SYNTHETIC = ROOT / "tmp_analysis/fit_lag_identity_support_output_verifier_synthetic_20261024.json"
PREFLIGHT_PASS = "PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_IMPLEMENTATION_PREFLIGHT"
OUTPUT_AUDIT_PASS = "PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_OUTPUT_AUDIT"
SUCCESS = "PASS_FIT_LAG_SUPPORT_EXECUTION_AND_INDEPENDENT_AUDIT"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, obj):
    """Publish complete JSON atomically, without replacing any existing file."""
    path = Path(path)
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(path))
        descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def process_identity(pid):
    """Read one PID only; no process-name search or external dependency."""
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        start_ticks = int(fields[19])
        cmdline = (directory / "cmdline").read_bytes().split(b"\0")
        cmdline = [value.decode(errors="surrogateescape") for value in cmdline if value]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        boot_seconds = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                            if line.startswith("btime "))
        again = (directory / "stat").read_text()
        require(int(again[again.rfind(")") + 2:].split()[19]) == start_ticks,
                "PID changed during process identity read")
        return dict(pid=pid, start_ticks=start_ticks, boot_id=boot_id,
                    create_time_unix=boot_seconds + start_ticks / os.sysconf("SC_CLK_TCK"),
                    cmdline=cmdline, state=fields[0], session_id=int(fields[3]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(bound):
    current = process_identity(bound["pid"])
    matched = current is not None and all(current[key] == bound[key]
                for key in ("pid", "start_ticks", "boot_id", "cmdline"))
    return dict(identity_matches=matched, running=matched and current["state"] not in ("Z", "X"),
                current=current)


def snapshot(expected):
    records = {}
    for name in expected:
        path = Path(name)
        exists = path.exists()
        records[name] = dict(exists=exists, sha256=sha(path) if exists else None)
    return records


def check_pins(spec):
    mismatches = {}
    for name, expected in spec["file_pins"].items():
        actual = sha(name) if Path(name).is_file() else None
        if actual != expected:
            mismatches[name] = dict(expected=expected, actual=actual)
    require(not mismatches, "Pinned files changed: " + json.dumps(mismatches))


def no_symlinks(paths):
    for name in paths:
        path = Path(name)
        require(path.is_absolute(), "Controlled path must be absolute: " + name)
        for part in (path, *path.parents):
            require(not part.is_symlink(), "Controlled path or ancestor is a symlink: " + str(part))


def validate(spec, require_out_absent=True):
    require(spec["schema"] == "FIT_LAG_SUPPORT_DETACHED_EXECUTION_V1", "Unknown launch schema")
    require(Path(spec["execution_directory"]) != Path(spec["output_directory"]), "Output and execution directories overlap")
    require(Path(spec["output_directory"]).parent != Path(spec["execution_directory"]), "Output nested in execution directory")
    require(Path(spec["execution_directory"]).parent != Path(spec["output_directory"]), "Execution nested in output directory")
    no_symlinks([spec["root"], spec["python"], spec["execution_directory"], spec["output_directory"],
                 *spec["file_pins"], *spec["protected_expected"]])
    check_pins(spec)
    require(spec["file_pins"][str(SELF)] == sha(SELF), "Supervisor pin mismatch")
    protocol = read(spec["protocol"])
    require(protocol["output_directory"] == spec["output_directory"], "Protocol output differs from launch output")
    preflight = read(spec["preflight"])
    require(preflight["status"] == PREFLIGHT_PASS, "Independent preflight is not PASS")
    require(preflight["script_sha256"] == spec["file_pins"][spec["producer"]], "Preflight producer pin stale")
    require(preflight["protocol_sha256"] == spec["file_pins"][spec["protocol"]], "Preflight protocol pin stale")
    if not spec["synthetic"]:
        verifier_test = read(spec["verifier_synthetic_receipt"])
        require(verifier_test["status"] == "PASS_INDEPENDENT_FIT_LAG_OUTPUT_AUDITOR_SYNTHETIC", "Verifier selftest not PASS")
        require(verifier_test["verifier_sha256"] == spec["file_pins"][spec["verifier"]], "Verifier selftest script pin stale")
    protected = snapshot(spec["protected_expected"])
    require(protected == spec["protected_expected"], "Seven protected files do not match frozen authority")
    if require_out_absent:
        require(not Path(spec["output_directory"]).exists(), "Preserve existing producer output")
        require(shutil.disk_usage(Path(spec["output_directory"]).parent).free >= spec["minimum_free_bytes"],
                "Insufficient free disk for support audit (requires at least 6 GiB)")
    return protected


def make_real_spec(args):
    requested = {str(PRODUCER): args.producer_sha256, str(PROTOCOL): args.protocol_sha256,
                 str(PREFLIGHT): args.preflight_sha256, str(VERIFIER): args.verifier_sha256}
    for name, digest in requested.items():
        require(len(digest) == 64 and all(value in "0123456789abcdef" for value in digest), "Invalid SHA256 for " + name)
        require(sha(name) == digest, "Requested final pin does not match " + name)
    protocol = read(PROTOCOL)
    authority = read(AUTHORITY)
    require(authority["status"] == "PASS_SEVEN_PROTECTED_FILES_MATCH_PRIOR_SEALED_RECEIPT", "Wrong protected authority")
    require(len(authority["protected_files"]) == 7, "Expected exactly seven protected files")
    pins = dict(protocol["input_pins"])
    pins.update(requested)
    for path in (SELF, CHECKER, AUTHORITY, VERIFIER_SYNTHETIC):
        pins[str(path)] = sha(path)
    spec = dict(schema="FIT_LAG_SUPPORT_DETACHED_EXECUTION_V1", created_utc=utc(), synthetic=False,
                root=str(ROOT), python=str(PYTHON), producer=str(PRODUCER), protocol=str(PROTOCOL),
                preflight=str(PREFLIGHT), verifier=str(VERIFIER), output_directory=str(OUT),
                execution_directory=str(EXECUTION), file_pins=pins,
                verifier_synthetic_receipt=str(VERIFIER_SYNTHETIC), minimum_free_bytes=6 * 1024 ** 3,
                protected_expected=authority["protected_files"],
                producer_command=[str(PYTHON), "-u", str(PRODUCER), "--execute"],
                verifier_command=[str(PYTHON), "-u", str(VERIFIER), "--out", str(OUT)],
                output_audit_file="independent_output_audit.json", output_audit_pass_status=OUTPUT_AUDIT_PASS,
                environment=dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                training_started=False, retry_enabled=False, source_or_lag_selection=False)
    validate(spec)
    return spec


def launch(spec):
    validate(spec)
    directory = Path(spec["execution_directory"])
    directory.mkdir()  # Atomic exclusive claim; never merge or overwrite a prior launch.
    write_new(directory / "launch_spec.json", spec)
    spec_sha = sha(directory / "launch_spec.json")
    command = [spec["python"], "-u", str(SELF), "run", "--execution-dir", str(directory),
               "--launch-spec-sha256", spec_sha]
    try:
        with (directory / "supervisor.log").open("x") as log:
            child = subprocess.Popen(command, cwd=spec["root"], env=dict(os.environ, **spec["environment"]),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        identity = process_identity(child.pid)
        receipt = dict(status="DETACHED_SUPERVISOR_LAUNCHED", launched_utc=utc(), supervisor_pid=child.pid,
                       supervisor_process=identity, supervisor_command=command, launch_spec_sha256=spec_sha,
                       execution_directory=str(directory), output_directory=spec["output_directory"],
                       start_new_session=True, stdin="DEVNULL", stdout_stderr=str(directory / "supervisor.log"))
        write_new(directory / "launch_receipt.json", receipt)
        return receipt
    except BaseException as error:
        write_new(directory / "launch_failure.json", dict(status="SUPERVISOR_LAUNCH_FAILED", at_utc=utc(), error=repr(error)))
        raise


def run_child(spec, name, command, directory):
    log_path = directory / (name + ".log")
    started = utc()
    with log_path.open("x") as log:
        child = subprocess.Popen(command, cwd=spec["root"], env=dict(os.environ, **spec["environment"]),
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
        bound = process_identity(child.pid)
        write_new(directory / (name + "_process.json"), dict(stage=name, started_utc=started,
                  process=bound, pid=child.pid, command=command, start_new_session=True, stdin="DEVNULL"))
        exit_code = child.wait()
    # This receipt precedes log hashing, protected-file hashing and verifier work.
    receipt = dict(status=name.upper() + ("_EXIT_ZERO" if exit_code == 0 else "_EXIT_NONZERO"),
                   actual_tool_execution=dict(exit_code=exit_code, pid=child.pid, subprocess=True,
                       command=command, started_utc=started, completed_utc=utc(),
                       observed_by="detached supervisor wait() for its own " + name + " child"),
                   process=bound, launch_spec_sha256=sha(directory / "launch_spec.json"),
                   producer_sha256=spec["file_pins"][spec["producer"]],
                   protocol_sha256=spec["file_pins"][spec["protocol"]],
                   log_path=str(log_path), file_pins=spec["file_pins"])
    write_new(directory / (name + "_exit.json"), receipt)
    if name == "producer" and Path(spec["output_directory"]).is_dir():
        write_new(Path(spec["output_directory"]) / "producer_execution_receipt.json", receipt)
    return receipt


def run(directory, expected_spec_sha):
    directory = Path(directory).resolve()
    spec_path = directory / "launch_spec.json"
    require(sha(spec_path) == expected_spec_sha, "Launch specification pin mismatch")
    spec = read(spec_path)
    require(str(directory) == spec["execution_directory"], "Wrong execution directory")
    # An interrupted or finished run remains claimed; status is read-only, never a retry.
    with (directory / "run.claim").open("x") as claim:
        claim.write(str(os.getpid()) + "\n")
        claim.flush()
        os.fsync(claim.fileno())
    result = dict(status="SUPERVISOR_FAILED", started_utc=utc(), launch_spec_sha256=expected_spec_sha,
                  execution_directory=str(directory), output_directory=spec["output_directory"], stages=[],
                  training_started=False, source_or_lag_selection=False)
    try:
        write_new(directory / "supervisor_process.json", dict(process=process_identity(os.getpid()), recorded_utc=utc()))
        before = validate(spec)
        write_new(directory / "protected_before.json", dict(status="PASS_SEVEN_PROTECTED_BEFORE", protected_files=before, recorded_utc=utc()))
        write_new(directory / "supervisor_ready.json", dict(status="READY_PINNED_FIT_LAG_SUPPORT_PRODUCER",
                  recorded_utc=utc(), process=process_identity(os.getpid()), launch_spec_sha256=expected_spec_sha,
                  file_pins=spec["file_pins"], protected_before=evidence(directory / "protected_before.json"),
                  free_disk_bytes=shutil.disk_usage(Path(spec["output_directory"]).parent).free,
                  environment=spec["environment"], producer_command=spec["producer_command"],
                  verifier_command=spec["verifier_command"], output_directory=spec["output_directory"],
                  training_started=False))
        producer = run_child(spec, "producer", spec["producer_command"], directory)
        result["stages"].append(producer)
        if producer["actual_tool_execution"]["exit_code"]:
            result["status"] = "PRODUCER_FAILED"
        else:
            require(Path(spec["output_directory"]).is_dir(), "Producer exit0 without output directory")
            check_pins(spec)
            require(snapshot(spec["protected_expected"]) == before, "Protected files changed after producer")
            verifier = run_child(spec, "verifier", spec["verifier_command"], directory)
            result["stages"].append(verifier)
            if verifier["actual_tool_execution"]["exit_code"]:
                result["status"] = "INDEPENDENT_VERIFIER_FAILED"
            else:
                result["status"] = "INDEPENDENT_VERIFIER_RESULT_INVALID"
                audit_path = Path(spec["output_directory"]) / spec["output_audit_file"]
                audit = read(audit_path)
                require(audit["status"] == spec["output_audit_pass_status"], "Verifier exit0 without expected PASS audit")
                result["independent_output_audit"] = evidence(audit_path)
                result["status"] = SUCCESS
    except BaseException as error:
        result["error"] = repr(error)
        traceback.print_exc()
    finally:
        try:
            after = snapshot(spec["protected_expected"])
            unchanged = after == spec["protected_expected"]
            write_new(directory / "protected_after.json", dict(status="PASS_SEVEN_PROTECTED_AFTER" if unchanged else "PROTECTED_FILES_CHANGED",
                       protected_files=after, recorded_utc=utc()))
            result["seven_protected_unchanged"] = unchanged
            if not unchanged:
                result["status"] = "PROTECTED_FILES_CHANGED"
            no_symlinks([spec["execution_directory"], spec["output_directory"],
                         *spec["file_pins"], *spec["protected_expected"]])
            check_pins(spec)
            result["all_bound_file_pins_unchanged"] = True
        except BaseException as error:
            result["final_validation_error"] = repr(error)
            result["status"] = "FINAL_BINDING_OR_PROTECTED_CHECK_FAILED"
        result["completed_utc"] = utc()
        result["logs"] = {name: evidence(directory / name) for name in ("producer.log", "verifier.log")
                          if (directory / name).is_file()}
        write_new(directory / "execution_receipt.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == SUCCESS else 1


def status(directory):
    directory = Path(directory).resolve()
    result = dict(execution_directory=str(directory), observed_utc=utc(), status="NOT_LAUNCHED")
    if not directory.exists():
        return result
    result["status"] = "INCOMPLETE_NO_FINAL_RECEIPT"
    for name in ("launch_receipt", "supervisor_process", "supervisor_ready", "producer_process", "producer_exit", "verifier_process", "verifier_exit", "execution_receipt", "launch_failure"):
        path = directory / (name + ".json")
        if path.exists():
            record = read(path)
            result[name] = record
            bound = record.get("process")
            if bound:
                result[name + "_liveness"] = same_process(bound)
    if "execution_receipt" in result:
        result["status"] = result["execution_receipt"]["status"]
    elif "launch_failure" in result:
        result["status"] = "SUPERVISOR_LAUNCH_FAILED"
    elif result.get("supervisor_process_liveness", {}).get("running"):
        result["status"] = "RUNNING"
    result["logs"] = {name: dict(bytes=(directory / name).stat().st_size,
                         modified_unix=(directory / name).stat().st_mtime)
                      for name in ("supervisor.log", "producer.log", "verifier.log") if (directory / name).exists()}
    return result


def selftest(output):
    """Tiny real subprocesses in temporary directories, never real FIT work."""
    checks = []
    with tempfile.TemporaryDirectory(prefix="fit_lag_supervisor_synthetic_") as temporary:
        base = Path(temporary)

        def fixture(name, producer_exit=0, verifier_exit=0, mutate=False, audit=True):
            root = base / name
            root.mkdir()
            out, execution = root / "support", root / "execution"
            protected = root / "protected.txt"
            protected.write_text("frozen\n")
            producer, verifier = root / "producer.py", root / "verifier.py"
            producer.write_text("from pathlib import Path\nimport json,time\n" +
                "out=Path(" + repr(str(out)) + "); out.mkdir()\n" +
                "time.sleep(0.15)\n(out/'summary.json').write_text('{}')\n" +
                (("Path(" + repr(str(protected)) + ").write_text('changed')\n") if mutate else "") +
                "raise SystemExit(" + repr(producer_exit) + ")\n")
            verifier.write_text("from pathlib import Path\nimport json\n" +
                "out=Path(" + repr(str(out)) + ")\n" +
                "r=json.loads((out/'producer_execution_receipt.json').read_text())\n" +
                "assert r['actual_tool_execution']['exit_code']==0\n" +
                (("(out/'independent_output_audit.json').write_text(json.dumps({'status':" + repr(OUTPUT_AUDIT_PASS) + "}))\n") if audit else "") +
                "raise SystemExit(" + repr(verifier_exit) + ")\n")
            protocol, preflight = root / "protocol.json", root / "preflight.json"
            write_new(protocol, dict(output_directory=str(out)))
            write_new(preflight, dict(status=PREFLIGHT_PASS, script_sha256=sha(producer), protocol_sha256=sha(protocol)))
            return dict(schema="FIT_LAG_SUPPORT_DETACHED_EXECUTION_V1", synthetic=True, root=str(root), python=str(Path(sys.executable).resolve()),
                producer=str(producer), protocol=str(protocol), preflight=str(preflight), verifier=str(verifier),
                output_directory=str(out), execution_directory=str(execution),
                file_pins={str(p): sha(p) for p in (SELF, producer, verifier, protocol, preflight)},
                minimum_free_bytes=0,
                protected_expected=snapshot({str(protected): None, str(root / "must_remain_absent"): None}),
                producer_command=[sys.executable, "-u", str(producer)], verifier_command=[sys.executable, "-u", str(verifier)],
                output_audit_file="independent_output_audit.json", output_audit_pass_status=OUTPUT_AUDIT_PASS,
                environment=dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))

        def await_done(spec):
            directory = Path(spec["execution_directory"])
            deadline = time.monotonic() + 15
            while not (directory / "execution_receipt.json").exists():
                require(time.monotonic() < deadline, "Synthetic subprocess timeout")
                time.sleep(0.03)
            return read(directory / "execution_receipt.json")

        def rejected(operation, label):
            try:
                operation()
            except (RuntimeError, FileExistsError):
                checks.append(label)
            else:
                raise AssertionError("Expected rejection: " + label)

        good = fixture("success")
        started = time.monotonic()
        receipt = launch(good)
        require(time.monotonic() - started < 2, "Launch blocked on work")
        require(receipt["supervisor_process"]["session_id"] == receipt["supervisor_pid"], "Detached session missing")
        require(receipt["supervisor_process"]["cmdline"] == receipt["supervisor_command"], "Command binding mismatch")
        require(await_done(good)["status"] == SUCCESS, "Synthetic success failed")
        checks += ["detached launch returns before work and has its own session", "actual producer exit receipt precedes independent verifier", "success final receipt and protected absence preserved"]
        directory = Path(good["execution_directory"])
        unchanged = sha(directory / "execution_receipt.json")
        rejected(lambda: launch(good), "duplicate launch or existing output rejected")
        rejected(lambda: run(directory, sha(directory / "launch_spec.json")), "duplicate run rejected without overwrite")
        require(sha(directory / "execution_receipt.json") == unchanged, "Duplicate attempt overwrote receipt")
        require(status(directory)["status"] == SUCCESS, "Status omitted final receipt")
        require(read(directory / "supervisor_ready.json")["status"] == "READY_PINNED_FIT_LAG_SUPPORT_PRODUCER", "Missing ready receipt")
        orphan = fixture("launcher_exit")
        launcher_spec = base / "launcher_exit_spec.json"
        write_new(launcher_spec, orphan)
        launcher_code = "import json,runpy,sys; module=runpy.run_path(sys.argv[1]); module['launch'](json.load(open(sys.argv[2])))"
        launched = subprocess.run([str(Path(sys.executable).resolve()), "-c", launcher_code, str(SELF), str(launcher_spec)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  cwd=str(base), timeout=10)
        require(launched.returncode == 0, "Independent launcher failed: " + launched.stderr.decode())
        require(await_done(orphan)["status"] == SUCCESS, "Background task did not survive launcher exit")
        checks.append("detached supervisor completes after independent launcher process exits")
        wrong_pid = dict(process_identity(os.getpid()), start_ticks=-1)
        require(not same_process(wrong_pid)["identity_matches"], "PID reuse binding not checked")
        checks.append("PID start ticks binding rejects identity mismatch")
        for name, kwargs, expected in (
            ("producer_failure", dict(producer_exit=7), "PRODUCER_FAILED"),
            ("verifier_failure", dict(verifier_exit=9), "INDEPENDENT_VERIFIER_FAILED"),
            ("missing_audit", dict(audit=False), "INDEPENDENT_VERIFIER_RESULT_INVALID"),
            ("protected_mutation", dict(mutate=True), "PROTECTED_FILES_CHANGED"),
        ):
            spec = fixture(name, **kwargs)
            launch(spec)
            final = await_done(spec)
            require(final["status"] == expected, "Wrong synthetic status for " + name + ": " + final["status"])
            if name in ("producer_failure", "protected_mutation"):
                require(not (Path(spec["execution_directory"]) / "verifier_process.json").exists(), "Verifier ran after failed producer/protection")
            if name == "producer_failure":
                require(read(Path(spec["output_directory"]) / "producer_execution_receipt.json")["actual_tool_execution"]["exit_code"] == 7, "Actual failure exit lost")
            checks.append(name + " lands distinct final receipt")
        stale = fixture("stale_pin")
        Path(stale["producer"]).write_text("raise SystemExit(0)\n")
        rejected(lambda: launch(stale), "stale producer SHA rejected before execution directory creation")
        require(not Path(stale["execution_directory"]).exists(), "Rejected launch created execution directory")
        existing = fixture("existing_output")
        Path(existing["output_directory"]).mkdir()
        rejected(lambda: launch(existing), "existing producer output rejected before launch")
        existing_execution = fixture("existing_execution")
        Path(existing_execution["execution_directory"]).mkdir()
        rejected(lambda: launch(existing_execution), "existing execution directory rejected before launch")
        symbolic = fixture("symlink_output")
        Path(symbolic["output_directory"]).symlink_to(base / "absent_target")
        rejected(lambda: launch(symbolic), "dangling output symlink rejected before launch")
        disk = fixture("disk_low")
        disk["minimum_free_bytes"] = 10 ** 30
        rejected(lambda: launch(disk), "insufficient free disk rejected before launch")
        spec = fixture("spec_tamper")
        target = Path(spec["execution_directory"])
        target.mkdir()
        write_new(target / "launch_spec.json", spec)
        rejected(lambda: run(target, "0" * 64), "wrong launch spec SHA rejected before run claim")
        require(not (target / "run.claim").exists(), "Tampered spec claimed a run")
    result = dict(status="PASS_FIT_LAG_SUPPORT_BACKGROUND_SYNTHETIC", completed_utc=utc(), checks=checks,
                  checks_passed=len(checks), supervisor=evidence(SELF), real_support_launched=False,
                  protected_real_files_opened=False, training_started=False)
    write_new(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    launch_parser = modes.add_parser("launch")
    for name in ("producer", "protocol", "preflight", "verifier"):
        launch_parser.add_argument("--" + name + "-sha256", required=True)
    run_parser = modes.add_parser("run")
    run_parser.add_argument("--execution-dir", type=Path, required=True)
    run_parser.add_argument("--launch-spec-sha256", required=True)
    status_parser = modes.add_parser("status")
    status_parser.add_argument("--execution-dir", type=Path, default=EXECUTION)
    test_parser = modes.add_parser("selftest")
    test_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "launch":
        print(json.dumps(launch(make_real_spec(args)), ensure_ascii=False), flush=True)
        return 0
    if args.mode == "run":
        return run(args.execution_dir, args.launch_spec_sha256)
    if args.mode == "status":
        print(json.dumps(status(args.execution_dir), ensure_ascii=False), flush=True)
        return 0
    return selftest(args.output)


if __name__ == "__main__":
    sys.exit(main())
