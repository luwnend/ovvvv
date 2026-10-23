# Base-error collection protocol: independent review, 2026-10-23

**Verdict:** the corrected `results/base_error_collection_20261023_v3` preparation and its real `smoke_v2` pass the independent collection audit. This admits a frozen sparse-anchor pilot under the same executable contract. It does not admit a learned method, validate a target matcher, or demonstrate prediction/calibration/tracking gains.

## Executable evidence

`tmp_analysis/verify_base_error_collection.py` accepts `--stage DIR --run DIR` (or the run manifest), and performs a preparation-only audit when `--run` is omitted. It uses CPU NumPy and an independently reviewed shared payload validator; it does not import the model, run inference, match targets, fit parameters, or modify stage/run/source artifacts. The effective mmcv config and canonical novelty ID list are parsed as restricted literal AST, without executing either config.

The independent real-smoke command exited **0**:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/chenhc/miniconda3/envs/covtrack/bin/python \
  tmp_analysis/verify_base_error_collection.py \
  --stage results/base_error_collection_20261023_v3 \
  --run results/base_error_collection_20261023_v3/smoke_v2 \
  --output results/base_error_collection_20261023_v3/smoke_v2/independent_collection_audit.json
```

The report status is `PASS_INDEPENDENT_GT_FREE_COLLECTION_AUDIT`: **4 frames / 2 videos / 150 retained observations / 150 emitted observations**, with **1204 posterior columns**, canonical **337 true novelty-mask entries**, and maximum reconstructed fusion error **1.8621592845491364e-7**. All four NPZ files and four receipts bind by SHA256, size, annotation identity, and actual ordered execution records. The runner compared saved emitted rows with real `model.track_results` exactly. The real tracker object changed on both frame-0 events, did not change at within-video frame 1, and initial IDs restarted from a fresh tracker in both videos. All **71 isolated Python source hashes**, the frozen released checkpoint SHA, and **7 protected-file before/after hashes** passed.

The smoke contained no empty frames or unassigned rows. Their retention contracts are covered by the separate CPU schema tests; real empty/unassigned examples must remain explicitly absent from this smoke's evidence.

Report SHA256: `ab7a8eeb522afaa107c8593d175e6463dee3cdce844964867c3a332803b5db69`.

Earlier static/source-method evidence remains preserved at `tmp_analysis/base_collection_protocol_evidence_20261023.json`, SHA256 `7c10fe12cce90a56bcaa047cc8f130bf12cead9bdb5cbea394c24228f368a05a`. Its preparation snapshot and runtime limitations describe an earlier stage; the live v3/smoke_v2 report above supplies the subsequent executable evidence.

## Preparation corrections and supervision boundary

The original preparation copied image/video dictionaries and thereby retained `neg_category_ids` and `not_exhaustive_category_ids`. The first whitelist repair removed those fields. Independent review then found unnecessary category-level `image_count` and `instance_count` in v2. V3 removes these counts as well. Old preparations and failed runs remain separate provenance; they are not accepted collection outputs.

V3 inference images are restricted to `id, video_id, file_name, width, height, frame_id, frame_index, sample_step, source_frame_id`; videos to `id, name, width, height, fps`; categories to fixed vocabulary metadata `id, name, synonyms, synset, def, frequency`. The `frequency` field is the fixed LVIS vocabulary prior associated with the released rare-class fusion mask, rather than a per-image target. `annotations` and `tracks` are empty. No per-image class/box/negative/exhaustiveness metadata enters the model input. The dump metadata is independently restricted to a reviewed subset of these identity/geometry/timeline fields.

The preparation contains **1299 selected images / 41 videos**, with **3953 separate manual target boxes**, of which **2138** satisfy the parameter-fit eligibility gate. The target file is not an inference input. These frames were selected using a sealed manual-anchor protocol; GT-free prediction here means no target information is used to produce or select output rows within a selected frame. This does not make the sample a dense, label-independent video sample.

`smoke_v1` in v2 exited with the implementation error `KeyError: frame_id` before producing accepted frame outputs. The corrected runner installs a `CollectionFrameStep` pipeline transform so the explicit inference frame ordinal reaches the model. The accepted v3/smoke_v2 keeps that effective pipeline in its hashed config. This repair changes collection plumbing, not detector weights.

## Timeline, reset, and memo semantics

Only selected manual-anchor images are executed. The 1258 within-video source gaps are usually 25 or 30 raw frames; 100 gaps are at least 50 raw frames. All 41 current first source frames happen to be zero. Each video is nevertheless explicitly reindexed to `frame_id = sample_step = 0..N-1`, preserving original `source_frame_id` and `frame_index` separately. Thus tracker `memo_frames=50` means **50 selected anchor steps**, not 50 original frames or a fixed duration in seconds. Results cannot be presented as dense-video association or continuous occlusion evidence.

The active MOT implementation constructs a new tracker at `frame_id==0`. Accepted runtime evidence checks tracker object identity, ordered frame events, and actual first-frame output IDs; small IDs alone are not sufficient reset evidence. Downstream track keys must be `(video_id, local_track_id)`.

The preserved tracker matches against its existing memo, refreshes matched tracks, then removes stale unmatched entries in `update_memo`. CPU execution of the AST-extracted actual method confirms that an unmatched track is removed at raw step 60 but retained at selected step 1; a successfully matched track at raw step 60 is refreshed and survives. Therefore a raw gap of at least 50 does not itself imply that every existing track expired before matching.

## Actual vocabulary, fusion, and row provenance

The active registered class is `OVTrackRoIHeadUncertainty`, with `bbox_head.num_classes=1203`. The prompt is finite with shape `[1203,512]`; `fix_bg=False` supplies the learned background embedding separately. Both validation/test category flags and special-prompt/record-probability modes are off; the ensemble and both static-ratio inference flags are on; `fixed_lambda=None` selects the asymmetric actual-mask fusion.

Full posterior columns are canonical LVIS category IDs `1..1203`, followed by background sentinel `-1`. The full novelty mask is constructed from the released **zero-based** `lvis_novel_label_ids`, contains **337 true** values, has false background, and is asserted equal to the released full-mask construction. The actual active full branch uses `self.novel_index`; it does not use the historical validation mask or need a `validation_index` alias. Text exponents are `1/3` on mask-true columns and `2/3` on other columns. The dump exports the actual mask/exponents and reconstructs normalized fusion from both normalized branch posteriors. Image and fused background entries are zero in this regime.

The old val296 cache has 297 columns and a released validation mask with 86 true entries. Projecting the canonical full mask onto its foreground vocabulary disagrees with that released mask in **93 columns**; **46 of 182** observed C-TAO base training categories are absent from val296. This full1203 execution is a new frozen detector/cache protocol. Its branch softmax denominators, mask-dependent fusion, thresholds and selected outputs need not match val296. Slicing or renormalizing the old cache cannot supply the new collection. A CPU numerical counterexample also shows that conditioning full fused posteriors and conditioning branches before asymmetric fusion differ by 0.0661617473 in probability.

For flattened NMS indices, `C=1203`, `proposal_row=flat_index//C`, and `foreground_col=flat_index%C`; the divisor must never be hardcoded to 296. The dump retains pretracker detector boxes/scores, flattened indices and proposal rows, then uses the actual tracker keep mask to identify post-distractor rows. The tracker may smooth scores; detector scores are preserved separately. Posttracker geometry/labels and row order must agree with the selected pretracker rows.

All post-distractor observations are retained, including `track_id=-1`. `emitted_track_mask = track_id>=0` exactly identifies rows returned by `track2result`; unassigned rows remain valid observation records but must never be pooled as a single track. Fixed detector/tracker suppression is part of the frozen output protocol, not GT-dependent filtering. Every executed frame must have an NPZ and receipt, including zero-output frames; missing/stale caches and incomplete files fail rather than being silently skipped. Tracker visualization is disabled to prevent its optional GT metadata loader from running.

## Holdout and generalization limits

The readout has **146 fit categories / 36 heldout categories** among detector-base classes. These are pseudo-novel **to the proposed readout**, not classes excluded from the released detector or CLIP pretraining. The whole inference sequence and all unlabeled observations remain intact. A supervised observation is fit-eligible only when its separate target is both in a fit video and a fit category. A pair target requires **both endpoints** independently eligible. Unknown/unmatched predictions are unknown, not automatic negative labels.

A mixed predicted track can include heldout-class targets. Track-majority GT class, oracle purity, fit-class aggregate correctness, class routing, support masks, auxiliary/contrastive objectives, normalization choices, and hyperparameter selection must not indirectly use heldout labels. Filtering a whole track by its majority GT class does not satisfy the endpoint rule. Unlabeled heldout appearances in retained contexts are a form of transductive input exposure and must be reported if used. A fit-class cross-entropy also sends gradients to competing heldout coordinates; the legitimate claim is that heldout-object target labels were withheld, not that every heldout coordinate received zero gradients.

The plan's source-group split is verified disjoint across readout fit/tune/test. It controls this readout's label partition; it does not establish detector-unseen classes/videos, content-level duplicate absence, Charades-person separation, or YFCC-uploader separation. No causal covariance, independent-noise mechanism, detector generalization, or learned efficacy follows from collection completion.

## Acceptance of the full pilot

The real pilot must use a new run directory, the sealed v3 `pilot_inference.json`, the same hashed isolated code and frozen checkpoint, and the same effective vocabulary/fusion/timeline flags. Acceptance requires actual completion receipt and launcher exit status, **all 1299 frames / 41 resets**, every NPZ/receipt hash and inventory, real emitted-output equality records, canonical columns/mask, source/protection hashes, and independent verifier exit 0. A successful smoke is permission to execute this collection contract, not evidence that the full pilot has already completed.
