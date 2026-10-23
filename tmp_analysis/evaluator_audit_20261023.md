# TETA evaluator and existing-artifact audit — 2026-10-23

Scope: read-only inspection of installed evaluator source, annotation JSON, prediction JSON and pickled evaluator artifacts. No training, detector inference, baseline overwrite or heavy full-dataset evaluation was performed. A tiny synthetic evaluator check used a temporary directory and was deleted automatically. Existing report prose was not treated as evidence.

## 1. Evaluator identity

Both installed environments have byte-identical evaluator files:

| File, under either Python 3.8 site-packages/teta | SHA256 |
|---|---|
| metrics/teta.py | `47a132d92b2e9c92fd47f9e36140278e7ac263418539b606bf58ceead20d97d4` |
| datasets/tao.py | `c174d9cd3df6962474db588eff7cb3e82736d9a5b387f8653cefa603b9a7a6ab` |
| eval.py | `8219b9a2858faf0c166bff2fa1d25eb96a96578329916d2e0196f84e5aadbbcc` |

Environments: `/home/chenhc/.conda/envs/ovtrack_ovmot` and `/home/chenhc/miniconda3/envs/covtrack`. The following source references use the latter.

## 2. What TETA50 and its macro average mean here

`metrics/teta.py:19-20` defines the localization/association IoU grid as `0.00, 0.05, ..., 0.95` (20 points) and the classification IoU grid as `0.50, 0.55, ..., 0.95` (10 points). `eval.py:240` selects the outer threshold `[50]`; this is the local-cluster overlap setting, not a claim that all reported components are computed at only IoU 0.50. `datasets/tao.py:308` sets the local-overlap gate to 0.5.

At each localization IoU point, `LocA = Loc_TP / max(1, Loc_TP + Loc_FN + Loc_FP)`. Association Jaccard is calculated for each GT/predicted track pair and averaged with the number of matched observations as weights (`metrics/teta.py:220-240`). `ClsA` is the mean, over the 10 classification IoU points, of `Cls_TP / max(1, Cls_TP + Cls_FN + Cls_FP)`. `TETA = (LocA + AssocA + ClsA)/3` (`metrics/teta.py:361-385`), followed by the mean over the 20 localization IoU points at summary time.

The aggregation order matters:

1. For each GT class, sum Loc/Cls counts across sequences and combine AssocA/Re/Pr with Loc_TP weights before calculating the class result (`metrics/teta.py:275-284`). It is not the equal-weight mean of video scores.
2. Average the class results equally, ignoring empty classes (`eval.py:125-155`; `metrics/teta.py:309-329`). TAO's default evaluated class set includes only classes actually present in GT (`datasets/tao.py:79-109`). It is not a box- or track-micro average and does not average all 1,203 ontology classes.
3. The base/novel wrapper takes an equal-weight mean of the available `COMBINED_SEQ` class summaries for each group. Full validation therefore has 261 base and 35 novel evaluated classes.

By default `TETA(exhaustive=False)` is used. Localization FP candidates and classification observations follow TAO's local-cluster handling; they are not equivalent to treating every unmatched detection anywhere in an image as a conventional detector FP. Classification FP is assigned to the predicted category only when that category is in the evaluated class mapping. Thus matched-box top-1 accuracy or top-k recall cannot be substituted for ClsA.

## 3. Base/novel source: frequency versus model fusion mask

`work/covtrack_base/ovtrack/datasets/tao_dataset.py:319-330` defines metric base as annotation categories with `frequency != 'r'`, and novel as `frequency == 'r'`. `ovmot_improve/eval_teta.py` uses the same definition. The standalone `tools/eval_teta_from_formated_results.py` has hard-coded sets; a direct AST-to-annotation comparison found the sets identical to the local 866 base / 337 rare ontology split, for both category names and IDs.

This metric grouping is different from the model's `validation_novel_index` in `work/covtrack_tces_v2/ovtrack/models/roi_heads/class_name.py`. Direct AST inspection gives 296 foreground category IDs and a 297-entry mask (including background), with 86 true foreground entries. The 296 IDs contain only 35 annotation-rare categories. There are 93 disagreements: 72 frequency-base categories have model-mask true and 21 frequency-rare categories have model-mask false. For example, armchair has mask true but frequency f; beeper has mask false but frequency r.

To reconstruct the model's fused posterior, preserve its code mask. To report base/novel metrics, use annotation frequency. Replacing one definition with the other changes either the model or the reporting protocol.

## 4. Verified annotation identities

Root: `/raid/datasets/ring_data/data/tao/annotations/`.

| Annotation | Videos | Images | GT annotations | Present classes | Present rare classes | SHA256 |
|---|---:|---:|---:|---:|---:|---|
| validation_ours_v1.json | 988 | 36,375 | 112,798 | 296 | 35 | `559811dd8166922c537e71457fecc0faf9ad0acc5522cafca7954273c0e2a67d` |
| ovmot_subset/val60.json | 63 | 2,305 | 10,891 | 64 | 12 | `13056c40824c92c1f4cfa09ffcbf0cb2982ff0354bb3d52be44cbe528feebef3` |
| ovmot_subset/val3.json | 3 | 120 | 201 | 3 | 0 | `53d584e7db4e29f711f6c224e4fea3629686e45c6f77f3ae0289e579a018b599` |

`val60` is a filename alias; its actual current content contains 63 videos. `val3` has no novel class, so it cannot establish novel-class performance.

## 5. Current evaluator artifacts, independently read

Values below were read directly from `results/<name>/OVTrack/teta_summary_results.pth`, using NumPy only for re-averaging class records. Columns are TETA / LocA / AssocA / ClsA in percentage points. Base/novel values are the wrapper-equivalent mean of the serialized class summaries, shown to six decimals; the serialized summaries themselves are already rounded.

| Artifact | Combined | Base | Novel |
|---|---|---|---|
| cov_val_full_official_a100 | 38.921 / 57.275 / 41.839 / 17.650 | 39.553912 / 57.154935 / 41.960254 / 19.546529 | 34.205429 / 58.172714 / 40.936229 / 3.506849 |
| cov_tces_v3_fullval_eval | 38.921 / 57.275 / 41.839 / 17.650 | 39.553912 / 57.154935 / 41.960254 / 19.546529 | 34.205429 / 58.172714 / 40.936229 / 3.506849 |
| cov_val_full_paper_a100 | 38.942 / 57.421 / 41.850 / 17.556 | 39.608078 / 57.307778 / 42.021526 / 19.494943 | 33.975943 / 58.262714 / 40.567886 / 3.097109 |
| cov_val_full_100bbox_a100 | 39.064 / 57.960 / 41.893 / 17.339 | 39.644929 / 57.814034 / 41.884838 / 19.235920 | 34.734257 / 59.050371 / 41.956857 / 3.195377 |
| cov_val60_base | 36.444 / 51.806 / 35.615 / 21.913 | 37.468219 / 51.232096 / 36.388650 / 24.783923 | 32.008417 / 54.290500 / 32.261583 / 9.472343 |
| cov_tces_v3_val60_eval | 36.444 / 51.806 / 35.615 / 21.913 | 37.468219 / 51.232096 / 36.388650 / 24.783923 | 32.008417 / 54.290500 / 32.261583 / 9.472343 |
| tl_val60_majority | 30.006 / 46.379 / 29.498 / 14.140 | 31.489348 / 46.866860 / 30.221031 / 17.380210 | 23.576917 / 44.264333 / 26.365400 / 0.100750 |
| tl_val60_sumprob | 31.090 / 46.379 / 29.498 / 17.394 | 32.765637 / 46.866860 / 30.221031 / 21.209106 | 23.830917 / 44.264333 / 26.365400 / 0.862667 |
| tl_val60_sumlog | 30.351 / 46.379 / 29.498 / 15.177 | 30.733252 / 46.866860 / 30.221031 / 15.111860 | 28.696500 / 44.264333 / 26.365400 / 15.459500 |

The fullval official and TCES-v3 JSON predictions are byte-identical, not merely score-equivalent:

- Both `tao_track.json`: 297,725,529 bytes; SHA256 `a01eb48eb91c7116330fd31158b219f61fb3d2c68ff805ff44b6a43749370a48`.
- Both `OVTrack/teta_summary_results.pth`: SHA256 `621b769b60f88816be02c9cb79339b493ef60f17717a135cf663e3599a8a1d62`.
- Both prediction JSONs contain 1,529,908 records, 988 videos, 36,371 distinct images, 324,639 distinct `(video_id, track_id)` pairs and 291 predicted category IDs.

The val60 official and TCES-v3 prediction JSONs are likewise byte-identical: SHA256 `0c90e72558c1a35354001667206f1db1cf2ca65955e78507b5a45c9391f02e8d`; both contain 105,447 records, 63 videos, 2,305 images and 22,294 `(video_id, track_id)` pairs.

Consequently the existing v3 fullval/val60 evaluator directories establish cache/replay parity, not a TCES label-change gain. The separate `paper` and `100bbox` directories have different LocA/AssocA and must not be used as interchangeable label-only controls based on their names.

The old `tl_val60_*` artifacts each contain 79,273 records and 23,528 `(video_id, track_id)` pairs; they differ substantially from the 105,447-record official COVTrack baseline. Their shared LocA/AssocA allows a label-rule comparison within that artifact family, but comparison against `cov_val60_base` mixes observation/track sets. Their scripts use `cls_full` replay/cache inputs; this is not proof that they use current COVTrack's dual-branch posteriors.

## 6. Exact label-only invariance in this installed evaluator

Conditional claim: when only `category_id` changes, while record membership/order, `image_id`, `video_id`, `track_id`, `bbox`, `score`, annotations, evaluator code and configuration are fixed, LocA and AssocA are invariant in this implementation.

Source argument:

1. Loader operations `_limit_dets_per_image`, `_fill_video_ids_inplace`, `_make_tk_ids_unique` do not read predicted categories. `_merge_categories` changes only categories. `_compute_vid_mappings` stores per-track category metadata, but the raw evaluation arrays are loaded from per-image annotations without category-based prediction filtering (`datasets/tao.py:139-171`, `195-228`, `492-559`, `581-641`).
2. The all-class preprocessing and global assignment use GT IDs, predicted IDs and bbox similarity only (`eval.py:238-247`; `metrics/teta.py:40-70`, `178-218`). Changing predicted categories cannot change that assignment.
3. Class-specific preprocessing selects GT using its GT category, selects local overlapping predicted IDs using bbox IoU, and excludes predictions assigned to other GT classes using the same category-independent global assignment (`datasets/tao.py:312-338`). The final predicted masks use track-ID membership (`366-390`). `tk_classes` is carried through, but does not decide localization/association observation membership.
4. `compute_matches` uses only global alignment and similarity. Loc counts and match-count matrices do not depend on predicted classes. Predicted classes enter ClsTP/FN and cross-category ClsFP only (`metrics/teta.py:139-170`). Thus every per-class/per-alpha localization and association result is fixed; the class set and macro-average weights are also GT-fixed.

A three-frame, two-GT-class synthetic example was evaluated through the actual installed `teta.eval.eval_sequence`, with overlapping GT boxes, imperfect predictions and an additional unrelated prediction track. Flipping every predicted label changed each class's ClsA from 1 to 0; `np.array_equal` returned true for all alpha entries of Loc_TP, Loc_FP, Loc_FN, LocA, LocRe, LocPr, AssocA, AssocRe and AssocPr for both classes.

Therefore the unrounded label-only change obeys `delta TETA = delta ClsA / 3` for each class and for each fixed group macro average. This is a consequence of the inspected implementation, not a universal property of any evaluator named TETA. It does not license calculating a claimed score without running/validating the actual prediction transformation; observation deletion, changed tracking IDs, changed score filtering, record reordering in a Hungarian tie, or different GT/configuration can invalidate the premise.

## 7. What can and cannot be recomputed from current files

`OUTPUT_TEM_RAW_DATA=True` is misleadingly named: `eval.py:178-224` replaces raw arrays with `_summary_row` before serializing `teta_summary_results.pth`. `_BaseMetric._summary_row` emits percentage strings with five significant digits (`metrics/_base_metric.py:91-96`). The existing pickle can support exact reproduction of the wrapper's mean of these stored values and sequence/class inventory; it does not retain unrounded counts or per-alpha arrays. A mean of stored per-video summaries also cannot reproduce combined-sequence metrics because sequence aggregation is count/LocTP weighted.

The prediction JSON plus hashed GT and evaluator code suffice for a fresh evaluator run. For high precision invariance/attribution, capture the in-memory `eval_sequence`/combined raw arrays before the summary serialization or compare all preserved prediction fields directly. At the normal table precision, delta-TETA and delta-ClsA/3 may differ in the last displayed digit due to independent rounding.

Existing matching diagnostics have further scope limits: `ovmot_improve/cov_tces_label.py` keeps only boxes independently matched to a GT at IoU >=0.5, groups by `int(tid)` without video in the key, and reports observation-micro accuracy. Other analysis scripts use `(video_id, tid)` but some assign an entire track the first matched GT class or restrict to classes with a minimum count. These are useful labeled diagnostics, not the official macro TETA/ClsA result, and excluding unmatched observations from posterior aggregation is unavailable at inference time. Any new paired attribution must aggregate all emitted track observations with a composite key and use GT only in the diagnostic scoring stage.

## 8. Independent diagnostic and real-video control receipts

`tmp_analysis/score_track_cache_diagnostic.py` completed on the audited all-observation posterior cache. Its output is `results/tces_audit_20261023/diagnostic.json`, SHA256 `f0cd932a04c1db7d00ab0e443c12bedf2f2af0c792887b31e96cc56307711949`. It verifies per-track cache observation counts and TCC labels against the formal baseline, then scores a separately cached per-image class-agnostic one-to-one maximum-total-IoU Hungarian assignment, filtered at IoU >=0.5. The cache is `scoreinput.npz`, with source hashes and matching protocol in `scoreinput_manifest.json`.

There are 65,711 matched observations from 10,219 tracks, versus all 1,529,908 emitted observations from 324,639 tracks used for posterior aggregation. All 35 rare GT classes have diagnostic matches (1,720 observations); three base GT classes have no match. Among matched tracks, 8,573 have a single matched GT identity and 1,646 have multiple identities. Mean track purity is 0.946105, and observation-weighted purity is 0.930666.

The following are macro matched-observation classification diagnostics across all 35 rare classes, **not official ClsA**:

| Readout | Rare macro top1 | Rare macro top10 | Rare macro top50 |
|---|---:|---:|---:|
| Formal TCC | 5.9239 | 19.6616 | 27.6187 |
| Fused sumprob | 6.2801 | 50.9481 | 79.6787 |
| Fused sumlog | 11.7146 | 50.5502 | 83.9579 |
| Text sumprob | 4.8416 | 43.2677 | 82.5895 |
| Image sumprob | 15.4660 | 55.8323 | 87.6236 |
| Frequency-mask-corrected sumprob | 8.1695 | 81.1701 | 95.7520 |

TCC top10/50 here ranks per-frame argmax-vote counts, with zero-vote ties filled by lowest category ID; this is not a calibrated TCC posterior. The script also reports all/base macro scores, matched-only versus all-GT-support zero-for-unmatched denominators, per-class detail, single/mixed identity strata and explicitly GT-selected oracle upper bounds. The oracle and corrected-mask outputs are separately labeled interventions/upper bounds, not silently substituted into the baseline.

One discrepancy between posterior-argmax TCC and emitted-det-label TCC affects only global track 87926 in video 768, changing category 692 to 207 across three observations. Their maximum IoU to any GT is 0.2961449, so the trajectory never enters a TETA local cluster under the current non-exhaustive defaults. This source argument was verified using the actual installed evaluator: `tmp_analysis/check_argmax_control_local_teta.py` ran `teta.eval.eval_sequence` twice on video 768, preserving the full 296-class inventory, changing only those three categories. Every one of the 16 raw fields for every class, including all alpha entries of Cls_TP/FN/FP and Loc/Assoc arrays, was bitwise identical. Receipt: `results/tces_audit_20261023/argmax_tcc_control_local_teta.json`, status `PASS_ACTUAL_LOCAL_TETA_RAW_ARRAY_INVARIANCE`; protected baseline/GT source hashes remained unchanged. Consequently this specific posterior-argmax control discrepancy is metric-neutral for the corresponding full-validation evaluation.
