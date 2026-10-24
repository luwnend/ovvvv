# Geometry ROI pilot independent audit

**Status: PASS_INDEPENDENT_GEOMETRY_ROI_PILOT_AUDIT**

- Manifest: `/home/chenhc/ov/OVMOT_RESEARCH/tmp_analysis/geometry_roi_pilot_manifest_20261024_v13.json`
- Output: `/raid/datasets/chc_data/geometry_roi_pilot_20261024_v10`
- Manifest SHA-256: `d0b8b842d38e4377f619fdca8f253c331748e2d3c1589cc6659fee5312a92dd1`
- 18 frames / 995 retained rows / 18 NPZ pages
- Rounds: 3; 2–4 independent checks per round.

## Round 1: frozen_manifest_and_artifact_integrity

- **PASS** — manifest_sha_matches_started_and_receipt
  - Current manifest digest equals both `started.json` and `collection_receipt.json`.
- **PASS** — runner_sha_matches_all_receipts
  - All subchecks passed; see JSON for exact evidence.
- **PASS** — all_input_pins_rehashed
  - All subchecks passed; see JSON for exact evidence.
- **PASS** — frozen_model_source_tree_rehashed
  - All subchecks passed; see JSON for exact evidence.
- **PASS** — output_inventory_exact
  - All subchecks passed; see JSON for exact evidence.
- **PASS** — per_page_hash_schema_dtype_finite_metadata
  - All 18 pages: receipt/content SHA, exact member schema, `[N,256,7,7]` spatial shape, float16 tokens, finite values, and page metadata match; total rows=995.
- **PASS** — receipt_counts_exact
  - All subchecks passed; see JSON for exact evidence.

## Round 2: parent_row_provenance_and_geometry

- **PASS** — all_rows_and_boxes_reproject_through_parent
  - All 995 retained rows: keep-mask positions, raw256, det boxes, pre boxes, proposal rows verified; statuses all PASS.

## Round 3: sequence_media_scope_and_process_closure

- **PASS** — single_video_contiguous_prefix
  - All subchecks passed; see JSON for exact evidence.
- **PASS** — image_ids_unique
  - All subchecks passed; see JSON for exact evidence.
- **PASS** — all_media_hashes_and_dimensions
  - All 18 original image hashes and decoded `1200x1920x3` orientation match preprocessing metadata.
- **PASS** — collector_pid_terminal
  - PID 3886610 is absent; receipt status is `PASS_GEOMETRY_ROI_TOKEN_COLLECTION_V1`.
- **PASS** — no_gt_identity_posterior_payload
  - Exactly six allowlisted token members plus safe provenance metadata; no GT/identity/posterior/class-label payload found.
- **PASS** — frozen_checkpoint_causal_replay_diagnostic
  - CPU float32 replay max deviations: spatial=0.00173771, raw256=0.000530303; exact parent raw equality is separately PASS.
- **PASS** — runtime_and_scope_flags
  - All subchecks passed; see JSON for exact evidence.

## Verdict

`PASS_INDEPENDENT_GEOMETRY_ROI_PILOT_AUDIT`. The author receipt was not used as proof; it was compared against independently recomputed file hashes and parent-cache contents.

Caveat: device ownership during the completed run is evidenced only by its frozen start/receipt records. The checkpoint causal replay is diagnostic and approximate across CPU float32 versus production CUDA float16; exact raw256 parity against the parent cache passes.
