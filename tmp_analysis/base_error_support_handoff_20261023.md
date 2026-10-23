# Support-audit handoff: paused before real execution

## Resumed implementation update

The source-chain implementation is now complete; current status is **CPU_CHECKS_PASSED_PENDING_FINAL_INDEPENDENT_REVIEW**. The historical pause-state section below is retained for provenance and no longer lists current implementation blockers.

Current script SHA256: `64e702efa4e0956b19e9f9a0991ecfff8022c527e3e8bfb3a50710bbf4090186`. Current pure-CPU receipt: `tmp_analysis/base_error_support_cpu_check_20261023_v2.json` (SHA256 `91c6a768430dddf23451527b6d768d591a6c41287c89457424308fc51de1426b`). Tests cover14 corrupt provenance mutations, exact old-to-new synset mapping, unmapped GT retention, deliberately unusable test labels not joined, video-local IDs, geometry ambiguity and source/GT-trajectory pair concentration.

New checks establish official and C-TAO image-ID/coordinate uniqueness, canonical filename and dimensions, source `is_ori=True`, preserved original `frame_id`, exact source annotation ID and key fields, sealed-plan target IDs, and official bbox/category bridge. The original TAO and C-TAO files enter the before/after input-hash snapshot directly. Whole-source JSONs contain labels from the metadata test partition; test prediction/GT joins and outcomes remain closed.

The all-GT matched-identity purity scope is now **diagnostic oracle only**, because it can use heldout/nonmanual identity labels. Its flag is absent from main `supervised_pairs.jsonl`; it cannot choose training masks, weights, features or hyperparameters. Main pair eligibility uses only the two independently eligible endpoints. Exhaustive pair counts are diagnostics, not independent samples or training weights; per-source event counts and max source/GT-trajectory shares are reported.

No real support run was started by this subagent. Root will execute into a fresh support directory only after independent review GO. Mean fitting and training remain absent.

## Historical pause snapshot

Status: **IMPLEMENTED_CPU_SELF_TESTED_PENDING_INDEPENDENT_REVIEW_AND_SEALED_CACHE**.

`tmp_analysis/build_base_error_support.py` is on disk. Pinned interpreter:
`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`.

Interface after collection is sealed and both audits pass:

```bash
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python tmp_analysis/build_base_error_support.py \
  --stage results/base_error_collection_20261023_v3 \
  --run results/base_error_collection_20261023_v3/pilot_collect_v1 \
  --out results/base_error_collection_20261023_v3/pilot_support_v1
```

Confirm the actual run directory from the root handoff before executing; this note does not authorize using an incomplete run. Required run files: `collection_receipt.json` complete; `dump_verification.json` status `PASS_BASE_ERROR_DUMP`; `independent_collection_audit.json` status `PASS_INDEPENDENT_GT_FREE_COLLECTION_AUDIT`. The script refuses smoke/partial collections, changed dump inventories, changed input hashes and pre-existing output directories.

Implemented order: bind complete collection/audit hashes; seal all-emitted observation and `(video_id,local_track_id)` track indices before opening targets; keep `-1` out of tracks; load full official original TAO GT for fit/tune frames; bridge old1230 categories by unique synset; geometry-only Hungarian with inclusive IoU>=.5; exclude predictions overlapping multiple GT after assignment; then admit only exact manual-anchor fit-category endpoints. Unmatched is unknown. Test frames are indexed structurally and never matched/scored. Mean models and training are absent.

Outputs: `support_receipt.json` and lightweight `support_summary.json`; raw `gt_free_observations.jsonl`, `gt_free_tracks.jsonl`, `gt_free_index_manifest.json`, `frame_matching_receipts.jsonl`, `supervised_observations.jsonl`, `supervised_pairs.jsonl` stay local. Summary includes fit/tune outcomes, soft risk, source/trajectory/category support, same-predicted-track same-GT pairs with agreement and lag; test only closed structural counts. Raw supervision rows/indices must not be Git-synced.

Pure CPU `--self-test` passed: full-GT-before-eligibility matching, no category matching input, multi-GT ambiguity, exact IoU=.5, unmatched unknown, repeated local ID across videos distinct, negative ID exclusion and test split closed. Receipt: `tmp_analysis/base_error_support_cpu_check_20261023.json`. Script SHA at that check: `0fc8a797319f6bbeafd9eb717edb53973804bed3d1c87f709b9461628198c186`.

Outstanding before real support execution:

1. Independent read-only review by evaluator agent is pending; resolve its findings and rerun meaningful CPU checks if code changes.
2. Strengthen/audit official image coordinate uniqueness and canonical filename agreement, and explicit C-TAO `is_ori`/manual source-ID provenance if the independent stage audit is not considered sufficient. Current implementation checks the sealed target/hash chain plus original track, category-synset and bbox1e-6 parity, but does not reopen C-TAO to recheck `is_ori` itself.
3. Wait for pilot collection completion plus both independent receipts. No real support audit has been run and no empirical support results exist.
4. Run once into a new directory, verify summary/receipt and protected/input hashes, and interpret pair counts as descriptive clustered support, not independent samples or training readiness.

At pause instruction the root reported collection at450/1299 images; that is a historical progress point, not final status. This subagent did not stop or alter collection, use a GPU, access validation labels, compute test errors, or start training.
