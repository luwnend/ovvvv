# Geometry ROI full collection independent audit

**Status: `PASS_PAYLOAD_WITH_PROVENANCE_HOLD`**

- Token directory: `/raid/datasets/chc_data/geometry_roi_full_20261024_v1`
- Expected/actual npz: `276/276`
- Per-frame payload PASS: `276/276`
- Retained rows: `15153`
- Parent raw+geometry exact: `True`
- Terminal `collection_receipt.json`: `False`

## Finding

The 276 per-frame token payloads are complete and match the frozen parent cache exactly; this does not turn the missing terminal receipt or post-collection path re-sign into a formal collection PASS. Use as audited data with provenance caveat; do not claim an official completed receipt.

The runtime `started.json` binds the superseded manifest SHA `bc18bcda740d319d126dceb2d8751146bf3da9efbae0c3a13fd79a6d7dd48068`, while the current manifest is `687c88325f05f57c7e7806a629b20549c7f522bc5fa7d0fdf830bcca098b6677`. The sole manifest diff is `output_directory` (`_v1` -> `_v2`); the current SHA256SUMS are internally consistent, and all pins still match. This is consistent with a post-generation atomic rewrite of the path, but it does not supply the missing end-of-run receipt.

## Three rounds

### Round 1: inventory/schema/order/manifest pins and atomic re-sign

- `inventory_and_names`: PASS
- `frame_schema_and_prefixes`: PASS
- `manifest_pins_and_sha256sums`: PASS

### Round 2: per-frame ZIP, tensors, raw output, locators, boxes and hashes against parent

- `all_zip_and_member_schema`: PASS
- `raw_and_row_geometry_exact_parent`: PASS
- `parent_and_image_hashes`: PASS
- `finite_float16_spatial_tokens`: PASS

### Round 3: timeline, diversity, leakage, terminal receipt reconciliation

- `three_round_timeline_contiguous`: PASS
- `token_payload_not_degenerate_or_duplicated`: PASS
- `no_gt_identity_posterior_members_or_metadata`: PASS
- `provenance_gap_correctly_detected`: PASS

## Repro

```bash
/home/chenhc/miniconda3/envs/covtrack/bin/python /home/chenhc/ov/OVMOT_RESEARCH/tmp_analysis/audit_geometry_roi_full_independent_20261024.py
```

Per-frame status and exact problem strings are in the JSON sidecar.
