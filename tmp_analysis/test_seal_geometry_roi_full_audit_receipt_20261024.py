#!/usr/bin/env python3
"""Contract tests for the independent full-ROI audit receipt sealer."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp_analysis"))
import seal_geometry_roi_full_audit_receipt_20261024 as subject  # noqa: E402


class ReceiptContractTests(unittest.TestCase):
    def test_rejects_wrong_audit_status(self) -> None:
        audit = subject.read_json(subject.DEFAULT_AUDIT)
        audit["status"] = "PASS_PAYLOAD"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(subject.SealError):
                subject.seal(
                    audit_path, subject.DEFAULT_GATE,
                    subject.DEFAULT_MANIFEST, subject.DEFAULT_EVALUATION,
                    subject.DEFAULT_CACHE, root / "receipt.json")

    def test_rejects_missing_coverage(self) -> None:
        audit = subject.read_json(subject.DEFAULT_AUDIT)
        audit["conclusion"]["payload_valid_frames"] = 275
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(subject.SealError):
                subject.seal(
                    audit_path, subject.DEFAULT_GATE,
                    subject.DEFAULT_MANIFEST, subject.DEFAULT_EVALUATION,
                    subject.DEFAULT_CACHE, root / "receipt.json")

    def test_rejects_terminal_receipt_claim(self) -> None:
        audit = subject.read_json(subject.DEFAULT_AUDIT)
        audit["rounds"][0]["evidence"]["provenance"][
            "terminal_collection_receipt_present"] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(subject.SealError):
                subject.seal(
                    audit_path, subject.DEFAULT_GATE,
                    subject.DEFAULT_MANIFEST, subject.DEFAULT_EVALUATION,
                    subject.DEFAULT_CACHE, root / "receipt.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
