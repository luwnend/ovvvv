# COVTrack feature / memory path: independent source and synthetic review

Scope: read-only review of the released tracker, ROI head, train path and isolated collection copy. No real NPZ, model checkpoint, GT, classification outcomes, detector replay, training or GPU were opened/run. The companion synthetic auditor completed actual exit 0 in tool session 30290. Its JSON records source hashes before/after and all five checks. This establishes tensor routing and replay limits, not a real-data error rate or improvement.

## 1. Concrete state mismatch worth testing

Released `OVTrackerUncertainty.match`, `repos/COVTrack/ovtrack/models/trackers/ovtracker.py`:

- 514–515: remove distractors before any consistency calculation or matching.
- 531–532: current appearance is the raw track-head vector; reference appearance is `memo_embeds`. Geometry and semantics use the latest box and latest class embedding.
- 536–540: calculate three cue confidences from this current/reference pair.
- 637, 640: fuse only current features, then replace local `embeds` with the fused result.
- 644–658: use fused current features against memo features for dot-product bi-softmax and cosine matching.
- 690, and `update_memo` 420–429: write those fused features into the appearance EMA. An empty initial memo instead stores raw appearance, because the fusion branch is skipped. The stored appearance consequently mixes raw initialization and residuals computed under earlier scene contexts.

Training `ovtrack_roi_head.py` differs: 2470–2489 extracts raw appearance and mean class features for both frames; 2505–2516 calculates both directional confidences from both raw sides; 2570–2571 fuses both sides; 2612–2614 supervises fused-to-fused matching. Inference instead compares a transient current fusion with historical context-conditioned EMA. This is a real implementation/design difference. It is not sufficient evidence to call the released tracker broken, nor to claim changing it helps.

The source routing was exercised with the unmodified released class AST and an explicitly synthetic deterministic residual. Frame 0 stored raw vectors; frame 1 stored `0.6*old + 0.4*fused`; frame 2 calculated raw-current confidence against that modified historical state. The stub tests the path only; learned residual magnitude and effect remain unmeasured.

## 2. Small, separately interpretable controls

Use an independent subclass/file, preserve the released source and checkpoint, and fix the runtime parameters to the actual collection config. The collection runner uses match threshold .37, memo horizon 50 and current-feature EMA coefficient .4; the released training config defaults .5/10/.8 do not describe that run.

| Mode | Appearance state retained | Confidence inputs | Score inputs | What it isolates |
|---|---|---|---|---|
| released | legacy fused EMA | raw current vs fused EMA | fused current vs fused EMA | exact code control, subject to cache precision |
| raw confidence | both legacy fused EMA and a separate raw EMA | raw current vs raw EMA for appearance confidence only | fused current vs legacy fused EMA | recurrent contamination of the confidence computation |
| raw only | raw EMA | no fusion | raw current vs raw EMA | contribution of the entire released fusion pathway |
| raw store, one side | raw EMA | raw current vs raw EMA | fused current vs raw EMA | removes stored scene residuals but leaves one-sided matching |
| raw store, symmetric | raw EMA plus latest raw box/class feature | both directions from one pre-update raw snapshot | transient fused current vs transient fused memo | removes recurrent scene residuals and aligns the two-sided fusion form |

For the primary raw-confidence control, keep the fused EMA used by matching; replacing it with raw EMA changes two mechanisms at once. Clone/save raw vectors after filtering and before rebinding `embeds`. Update both memories for the same accepted IDs with the same .4 coefficient. Keep class memory, latest boxes, frame IDs, lifecycle and assignment unchanged.

For symmetric fusion, calculate both directions before either memory is updated. Never write transient memo fusion back to raw identity memory. Symmetric fusion still does not make inference identical to training: a temporal raw EMA is not a raw single-frame proposal, and train/test candidate populations differ.

This direction uses the existing learned extractor and fusion module. It has a concrete mechanism hypothesis and a small intervention surface. Generic memory banks, EMA, attention and multi-cue fusion already exist in related methods; this finding alone does not establish paper novelty.

## 3. Existing cache: sufficient fields and hard limits

The isolated `work/covtrack_base_error_20261023` tracker differs from the frozen released tracker only by exporting `keep_mask` in `remove_distractor`. Its embed head and cycle loss are byte-identical to the released versions. The collector takes raw track features before local tracker rebinding and saves all filtered rows, including `id=-1`.

`ovtrack/base_error_dump.py` 138–149 and 166–167 makes the following exact boundary explicit:

- `track_feats`: raw 256-dimensional output before tracker fusion.
- `cls_feats`: 512-dimensional `(text_head + image_head)/2`.
- Both are cast to float16 at line 142. Their original float32 values are unavailable. CPU/GPU matrix operations can also differ. Thus exact original baseline IDs cannot be promised from this cache; first run a released-mode replay and report every frame's row count, label, box, ID and score parity, plus first divergence. Do not interpret changes caused by quantization as a new method.
- `det_bboxes`, `det_labels`, raw feature rows and pretracker row maps are available for every retained row. Use all of them. Retaining only emitted rows changes column softmax and cue consistency; a synthetic duplicate appearance row changed the first candidate's final score from approximately 1 to .875.
- `pretracker_det_bboxes[tracker_keep_mask]` is the clean cut point. For the actual `ovtrack-teta` branch, scores are not smoothed, so retained posttracker boxes also match input geometry/scores. Other method branches may smooth scores and require the saved pretracker values.
- Reapplying `remove_distractor` to already retained, same-order rows is idempotent when scores, geometry, labels and thresholds match. The audit checked 100 random cases and a non-transitive overlap chain. It is safe to implement a clearly documented after-filter entry point; it is unsafe to rerun a new filter or sort that changes the population.
- Empty-frame calls still advance memo expiry. Reset at every video boundary and retain frame IDs, original row order, dictionary/insertion order for tie behavior, latest box/class state and all unmatched retained detections. No GT or predicted old track ID should select replay inputs or route memory updates.
- Once association changes, released IDs, support joins and classification contexts no longer score the changed system correctly. A later association experiment needs its own protocol and evaluator. Frozen-ID counterfactual checks can probe feature scores, but cannot establish tracking gain.

At this stage the useful next action is independent synthetic preflight of the new five-mode subclass and baseline replay feasibility. No real replay or efficacy result has been produced by this review.

## 4. Spatial and temporal inputs: separate mathematical check

The independent script `audit_covtrack_geometry_time_math_20261024.py` completed actual exit 0 and wrote `covtrack_geometry_time_math_independent_review_20261024.json`. It imports only the released cycle-loss class AST, with all examples synthetic.

The tracker sends absolute `[x1,y1,x2,y2]` vectors to `CycleLoss.get_pair_consistency`, which normalizes each vector and calculates cosine. This is not a relative-motion model. Translating both frames by the same `(1000,1000)` leaves pairwise displacement, widths, heights and relative center/scale features unchanged, but changes the two example geometry cycle confidences from `[.662466,.644507]` to `[.500000,.500000]`. The score depends on the coordinate origin.

Common positive uniform rescaling does preserve this cosine; it would be inaccurate to claim it lacks all scale invariance. However, anisotropic resizing changes cosine while properly normalized relative center/scale features stay invariant. Furthermore, `[10,10,20,20]` versus `[20,20,40,40]` has cosine 1 despite zero intersection: independent radial scaling of an absolute-coordinate vector preserves direction and discards useful size/location differences.

With one candidate on each side, both cycle softmax matrices are `[1]`, so confidence is exactly 1 for disjoint boxes and even orthogonal appearance vectors. This is a limitation of interpreting cycle assignment uniqueness as absolute match quality, not a new bug. Existing learned fusion was trained with these signals; do not silently alter their units or meaning inside a released baseline.

Timeline code is explicit: `prepare_base_error_full_collection.py` 204–207 rewrites `frame_id=sample_step`; 220–221 says `frame_index` remains the original frame index. `collect_base_error_pilot.py` 174 passes sampled step to the tracker. Thus existing memo age uses selected-frame steps, while a motion helper needs actual original-frame gaps. A synthetic constant-velocity path with original indices `[0,10,40]` and centers `[0,10,40]` predicts the third center as 40 when velocity/time use original indices, but as 20 when each selected sample is treated as one time unit. Original frame index is still not seconds unless FPS is verified.

A relative-motion helper should be independent of the pretrained `bbox_transform`, use center displacement normalized by object dimensions and appropriate original-frame gaps, retain uncertainty/invalid-history behavior, and expose a separate control. Keep released lifecycle units unchanged in this first control so timing and memory-horizon changes do not become inseparable. Common motion, Kalman prediction and geometric gating are established tools; neither these examples nor adding such a helper establish novelty or real-data benefit.

## 5. Reusing manual identity anchors after prediction is sealed

Read-only source review of `build_base_error_support.py` 480–508 confirms that each existing supervised observation first receives class-blind per-frame geometry assignment against full official GT. It then requires unambiguous assignment, non-crowd/non-ignore status, exact manual-anchor intersection and the immutable fit-class gate. The old predicted track ID at line 492 names the record and accumulates later diagnostics; it does not filter the individual eligible observation by whole-track purity. Pair construction and observed-single-identity scopes occur later and must not be reused for new predictions.

The collection receipt and independent collection audit both record `retained_observations=emitted_observations=681602` and zero unassigned rows. Thus the original emitted-only assignment population equals the retained population in this particular frozen collection. This is a verified property of this run, not a general tracker assumption.

After all new predictions are sealed, the same legal manual identity can be attached by immutable image plus dump-row/observation ID and scored under each new `(video,new_track_id)` partition. Declare handling of new `id=-1` before scores: retain the fixed known-anchor denominator, using either singleton unassigned observations or an explicit unassigned category as the protocol specifies. Unknown identity observations cannot be treated as negative pairs. B-cubed or pair scores here describe only the eligible fit-class manual-anchor subset; they are not complete-trajectory or official benchmark scores. Category/correctness values, old predicted IDs, old supervised pairs and selected edges are unnecessary for that identity projection.
