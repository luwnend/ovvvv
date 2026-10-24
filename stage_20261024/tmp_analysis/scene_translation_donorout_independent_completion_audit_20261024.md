# Independent completed donor-out audit

PASS: actual producer, fresh raw auditor, and finalizer exit codes are 0; supervisor completion and all chain SHA links verified. Protocol37 / launch39 input pins and seven protected states match before and after this audit.

All403 video arrays and403 receipts were rehashed. Independent metadata reconstruction confirms341 sources,13791 frames,613536 retained rows,27582 frame-pair receipts, and657099 fixed fullpool h. All five support schemes were independently reaggregated from sealed arrays and matched the summary within2e-12.

| Readout | lag1 gain | lag4 gain | supported sources lag1/4 |
|---|---:|---:|---:|
| fullpool / fullpool | 0.626520050293 | 0.974262016727 | 341 / 340 |
| row_excluded / row_excluded | 0.530495197139 | 0.750691779022 | 341 / 340 |
| proposal_overlap_excluded / proposal_overlap_excluded | 0.446546749414 | 0.566239510456 | 341 / 340 |
| common_row_strict / row_excluded | 0.539708636585 | 0.772761598846 | 341 / 340 |
| common_row_strict / proposal_overlap_excluded | 0.446546749414 | 0.566459150921 | 341 / 340 |
| common_all_three / fullpool | 0.649978003189 | 0.998650281455 | 341 / 340 |
| common_all_three / row_excluded | 0.542250919798 | 0.772761598846 | 341 / 340 |
| common_all_three / proposal_overlap_excluded | 0.449000650702 | 0.566459150921 | 341 / 340 |

Minimal gate: `PASS_MINIMAL_GTFREE_DONOR_PREDICTABILITY`. Strict primary gains0.446546749414 /0.566239510456 establish only GT-free donor predictability. The lag4 unsupported source remains null. The source-intersection strict lag1 mean is0.447240594308, which differs from the primary lag1 population.

Fresh raw audit separately reconstructed all37 columns and exact h/anchor/exclusion membership; maximum numeric difference0.0, scalar102 frame checks maximum dot difference7.771561172376096e-16. This final audit did not rerun it.

- GT-free endpoint donor predictability on raw strict-MNN pseudo-correspondences only; selected h are not verified identities.
- Fullpool includes the target endpoints and is a self-including control; exclusion comparisons require common-h framewise support.
- Row/proposal/positive-overlap exclusions remove direct endpoint leakage before rematching; no unconditional endpoint independence or physical camera motion is established.
- Source-macro gains average frame medians over supported sources; lag1 uses341 sources, lag4 uses340; null source is preserved and cross-lag means use source intersection only.
- No statistical significance, independent generalization, novel mechanism, identity ranking gain, or downstream tracking gain is established.
- Frozen protocol and other task outputs were not modified; no model/GPU/training/OOF evaluation was run by this audit.

Receipt: `scene_translation_donorout_independent_completion_audit_20261024.json`; SHA256 `6b5fe0862c6d22b0350c58d67aa4f33ee834d1d8917cd02b5750b645606c3a92`.
