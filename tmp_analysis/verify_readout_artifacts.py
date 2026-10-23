#!/usr/bin/env python3
"""Read-only verification of label-only readouts and protected source artifacts."""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream an ordinary JSON array using the standard JSON decoder."""
    decoder = json.JSONDecoder()
    with open(path, "r") as stream:
        buffer, pos, eof = "", 0, False

        def fill():
            nonlocal buffer, pos, eof
            new = stream.read(chunk_size)
            buffer, pos = buffer[pos:] + new, 0
            eof = not new

        def skip_space():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                fill()

        fill()
        skip_space()
        if pos >= len(buffer) or buffer[pos] != "[":
            raise ValueError("Expected a top-level JSON array: " + str(path))
        pos += 1
        first = True
        while True:
            skip_space()
            if pos >= len(buffer):
                raise ValueError("Unterminated JSON array: " + str(path))
            if buffer[pos] == "]":
                pos += 1
                if (buffer[pos:] + stream.read()).strip():
                    raise ValueError("Trailing content after JSON array: " + str(path))
                return
            if not first:
                if buffer[pos] != ",":
                    raise ValueError("Expected comma in JSON array: " + str(path))
                pos += 1
                skip_space()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            if not isinstance(row, dict):
                raise ValueError("Prediction array item is not an object: " + str(path))
            yield row
            first = False


def exact_json_equal(left, right):
    """Compare JSON values strictly, including scalar types and list ordering."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, default=root / "results/tces_audit_20261023")
    ap.add_argument("--baseline", type=Path, default=root / "results/cov_val_full_official_a100/tao_track.json")
    ap.add_argument("--rules", nargs="+", default=["fused_sumprob", "fused_sumlog", "corrected_mask_sumprob", "corrected_mask_tcc"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    args.audit = args.audit.resolve()
    args.baseline = args.baseline.resolve()
    out = (args.out or args.audit / "readout_artifacts_verification.json").resolve()
    if out.exists():
        raise FileExistsError("Preserve existing verification; choose a new --out: " + str(out))
    started = time.monotonic()
    fingerprints = {}

    def fingerprint(path):
        resolved = str(Path(path).resolve())
        if resolved not in fingerprints:
            fingerprints[resolved] = sha256(resolved)
        return fingerprints[resolved]

    protected_path = args.audit / "protected_before.json"
    checkpoint_path = args.audit / "checkpoint_receipt.json"
    protected = json.load(open(protected_path))
    checkpoint = json.load(open(checkpoint_path))
    readouts = args.audit / "readouts"
    baseline_receipt = json.load(open(readouts / "tcc/receipt.json"))
    require(baseline_receipt["status"] == "COMPLETE", "Baseline receipt is incomplete")
    baseline_hash = fingerprint(args.baseline)
    require(baseline_hash == baseline_receipt["baseline_sha256"], "Baseline source hash changed")
    annotation = Path(baseline_receipt["ann"])
    annotation_hash = fingerprint(annotation)
    require(annotation_hash == baseline_receipt["ann_sha256"], "Annotation hash changed")

    protected_checks = {}
    for path, before in protected.items():
        if not isinstance(before, dict) or "exists" not in before:
            continue
        exists = Path(path).exists()
        require(exists == before["exists"], "Protected file existence changed: " + path)
        check = {"expected_exists": before["exists"], "actual_exists": exists}
        if before["exists"]:
            actual = fingerprint(path)
            require(actual == before["sha256"], "Protected file hash changed: " + path)
            check.update(expected_sha256=before["sha256"], actual_sha256=actual, unchanged=True)
        else:
            check["absence_preserved"] = True
        protected_checks[path] = check
    checkpoint_file = Path(checkpoint["checkpoint"])
    require(checkpoint_file.exists(), "Checkpoint disappeared")
    require(fingerprint(checkpoint_file) == checkpoint["sha256"], "Checkpoint hash changed")
    require(checkpoint_file.stat().st_size == checkpoint["size"], "Checkpoint size changed")
    require(str(checkpoint_file.resolve()) == checkpoint["resolved"], "Checkpoint target changed")

    receipt_checks = {}
    expected_artifacts = [("tao_track.json", "result_sha256"),
                          ("metrics.json", "metrics_sha256"),
                          ("classification_counts.npz", "counts_sha256"),
                          ("aggregate_raw.pkl", "aggregate_raw_sha256"),
                          ("OVTrack/teta_summary_results.pth", "summary_sha256")]
    receipts = {}
    for rule in ["tcc"] + args.rules:
        directory = readouts / rule
        rp = directory / "receipt.json"
        receipt = json.load(open(rp))
        receipts[rule] = receipt
        require(receipt["status"] == "COMPLETE", rule + ": receipt is incomplete")
        require(receipt["rule"] == rule, rule + ": receipt rule mismatch")
        require(receipt["baseline_sha256"] == baseline_hash, rule + ": baseline lineage mismatch")
        require(receipt["ann_sha256"] == annotation_hash, rule + ": annotation lineage mismatch")
        require(receipt["evaluator_files"] == baseline_receipt["evaluator_files"], rule + ": evaluator lineage mismatch")
        artifact_checks = {}
        for relative, key in expected_artifacts:
            actual = fingerprint(directory / relative)
            require(actual == receipt[key], rule + ": artifact hash mismatch: " + relative)
            artifact_checks[relative] = {"sha256": actual, "matches_receipt": True}
        if "cache" in receipt:
            require(fingerprint(receipt["cache"]) == receipt["cache_sha256"], rule + ": cache hash changed")
        for evaluator_file, expected in receipt["evaluator_files"].items():
            require(fingerprint(evaluator_file) == expected, rule + ": evaluator source changed")
        receipt_checks[rule] = {"status_observed": receipt["status"], "receipt_path": str(rp),
                                "receipt_sha256_before_comparison": fingerprint(rp),
                                "artifacts": artifact_checks,
                                "same_baseline_annotation_evaluator": True}
        print("Verified COMPLETE receipt and artifact hashes:", rule, flush=True)

    paths = [args.baseline] + [readouts / rule / "tao_track.json" for rule in args.rules]
    changed_observations = dict.fromkeys(args.rules, 0)
    changed_tracks = {rule: set() for rule in args.rules}
    baseline_tracks = set()
    missing = object()
    n_rows = 0
    for index, rows in enumerate(zip_longest(*(iter_json_array(p) for p in paths), fillvalue=missing)):
        require(all(row is not missing for row in rows), "Prediction row counts differ at row " + str(index))
        base = rows[0]
        require("category_id" in base, "Baseline row lacks category_id")
        key = (base["video_id"], base["track_id"])
        baseline_tracks.add(key)
        for rule, candidate in zip(args.rules, rows[1:]):
            require(candidate.keys() == base.keys(), rule + ": keys differ at row " + str(index))
            for field in base:
                if field != "category_id":
                    require(exact_json_equal(base[field], candidate[field]),
                            rule + ": non-category field differs at row %d: %s" % (index, field))
            if not exact_json_equal(base["category_id"], candidate["category_id"]):
                changed_observations[rule] += 1
                changed_tracks[rule].add(key)
        n_rows = index + 1
        if n_rows % 250000 == 0:
            print("Verified ordered records:", n_rows, flush=True)

    row_checks = {}
    for rule in args.rules:
        receipt = receipts[rule]
        require(n_rows == receipt["observations"], rule + ": observation receipt mismatch")
        require(len(baseline_tracks) == receipt["tracks"], rule + ": track receipt mismatch")
        require(changed_observations[rule] == receipt["changed_observations"], rule + ": changed-observation receipt mismatch")
        require(len(changed_tracks[rule]) == receipt["changed_tracks"], rule + ": changed-track receipt mismatch")
        row_checks[rule] = {"n_rows": n_rows, "row_count_equal": True, "record_order_preserved": True,
                            "all_keys_equal": True, "all_non_category_fields_equal_with_types": True,
                            "n_tracks": len(baseline_tracks), "changed_observations": changed_observations[rule],
                            "changed_tracks": len(changed_tracks[rule]), "counts_match_receipt": True}
    # Hash prediction inputs again after their full content comparison. This also
    # makes an incomplete/concurrently changing artifact fail explicitly.
    post_comparison_hashes = {}
    for path in paths:
        actual = sha256(path)
        require(actual == fingerprint(path), "Prediction changed during verification: " + str(path))
        post_comparison_hashes[str(path)] = actual
    for rule in ["tcc"] + args.rules:
        rp = readouts / rule / "receipt.json"
        require(sha256(rp) == receipt_checks[rule]["receipt_sha256_before_comparison"],
                "Receipt changed during verification: " + rule)

    report = {
        "status": "PASS_READOUT_ARTIFACTS_AND_PROTECTED_SOURCES",
        "verification_script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "scope": "Read-only verification; no evaluation, training or existing-artifact writes. Script hashes from readout receipts and changing summary code are deliberately outside scope.",
        "rules": args.rules,
        "annotation": {"path": str(annotation), "sha256": annotation_hash, "matches_baseline_receipt": True},
        "checkpoint": {"path": str(checkpoint_file), "resolved": str(checkpoint_file.resolve()),
                       "sha256": fingerprint(checkpoint_file), "size": checkpoint_file.stat().st_size,
                       "matches_checkpoint_receipt": True},
        "protected_manifest": {"path": str(protected_path), "sha256": fingerprint(protected_path)},
        "checkpoint_receipt": {"path": str(checkpoint_path), "sha256": fingerprint(checkpoint_path)},
        "protected_checks": protected_checks,
        "complete_receipt_checks": receipt_checks,
        "record_comparison": row_checks,
        "prediction_hashes_stable_before_after_comparison": post_comparison_hashes,
        "source_fingerprints": fingerprints,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    # O_EXCL prevents replacement if another writer creates the output meanwhile.
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS", out, "sha256", sha256(out), flush=True)


if __name__ == "__main__":
    main()
