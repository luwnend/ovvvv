# Candidate B: identity contamination, future information and concrete code interfaces

**Conclusion: HOLD a new identity-memory method or network. Proceed only with the separately specified FIT observed-label oracle contrast and observability checks.** Different-GT endpoint support alone establishes neither useful purification nor a GT-free separator. Existing methods already cover identity propagation gates, tracklet splitting, multi-state tracking and attention memory. This review found a concrete raw/fused-memory interface worth testing in COVTrack, but no established new research contribution.

This audit first read the current design's §6 and existing source-boundary note. It does not reopen stopped candidate A. Sources were limited to two targeted primary-method readings and one blocked lead; no network was implemented, no real model was fitted, and no new tune/test result file was opened. The surrounding existing design report displayed its previously written mean fit/tune summary during context reading; those values were not used to formulate or select B. Current evidence below contains no new performance claim.

## 1. Closest primary sources and exact collision boundaries

| Source and reading coverage | What is already present | Remaining distinction is not a novelty claim |
|---|---|---|
| [Uncertainty Modeling of Contextual-Connections between Tracklets for Unconstrained Video-based Face Recognition](https://arxiv.org/abs/1905.02756), ICCV2019, UGG. Reused primary PDF/text, targeted reading Sec3.1–3.4, PDFpp3–5, Eq1–12. Not a new entire-paper review. | Tracklet identity distributions, uncertain positive edge gates, fixed cannot-link negative gates, joint mean-field updates of identities and gates; positive gate depends on appearance plus current identity-distribution agreement, Eq11. | Separate semantic categories and instance identity in OV-MOT; avoid explicit semantic feedback in routing. Neither identity uncertainty nor uncertainty gate is new. |
| [Split and Connect: A Universal Tracklet Booster for Multi-Object Tracking](https://arxiv.org/abs/2105.02426), arXiv2021. HF Markdown/API404; primary arXiv HTML/PDF200. Targeted reading SecIII-A/B/C and IV-B/C; not an entire-paper review. | Predict ID-switch positions from appearance and normalized boxes using temporal dilated convolutions and adaptive Gaussian-smoothed supervision; split impure tracklets, then attention-based identity embedding and connect. Training simulates imperfect tracker outputs from jittered GT boxes. | A probabilistic contamination variable or category readout without changing released IDs is a different output contract, but adding a router/slots instead of a hard split does not establish substantive novelty. Sparse-anchor labels cannot provide their dense switch masks. |
| [ERM-Track: Disturbance-Aware and Reliability-Guided Multi-Object Tracking for UGVs Under Dynamic Viewpoints](https://www.mdpi.com/2504-446X/10/9/692), 2026 search discovery. **Lead only.** Publisher fetch returned403; browser fetch429. | Search-indexed text mentions reliable/contaminated observation posterior, causal evidence accumulation and separate motion/identity-memory writes. These claims require primary-method verification. | It is a high-priority possible collision for “contamination state + causal memory writing”; do not claim its exact method has been excluded or its full text read. |

The full source receipt preserves successful arXiv downloads, HF failures, publisher failures, source hashes and reading scope. Raw papers remain local-only.

The existing verified OV-MOT boundary still applies: COVTrack MCF already learns multi-cue fusion; COVTrack++ MGA uses parent/child spatial relations and cross-attention, TCP propagates same-predicted-class scalar confidence and can continue across frames; OVTR CIP propagates learned category-related query features; TRACT TCR has appearance/category banks and confidence-dependent updates, while TFA has sampled-frame attention pooling. This review relies on the existing code/primary-method audit for those claims and does not pretend to have newly excluded all their code paths. Generic mixture tracking, graph identity inference, split/merge and memory attention remain prior art families.

## 2. One falsifiable scientific question

**Do naturally mis-associated references cause semantic readout harm that can be selectively predicted from past raw appearance/geometry, while retaining a difficult but genuinely same-identity new view?**

This is not “is a track mixed?” and not “does semantic consistency improve?” It has two necessary parts: privileged identity information must reveal meaningful readout headroom under a fixed rule; deployment-visible features must distinguish harmful foreign-identity evidence from useful same-identity variation. A classifier that detects only semantic change, or rejects every unusual view, fails this question.

The immediate authorized test is offline, fixed-ID, FIT-only observed-label reference contrast. A future causal memory version would require a new past-only filtration and a new frozen protocol. Its results cannot be inferred from an all-history oracle containing future frames.

## 3. What the existing cache can and cannot identify

Available inputs include all-emitted text/image/fused posteriors, raw track-head appearance and raw classification features, box/score, predicted local track ID and sampled/original frame indices. These are useful frozen features. They do not record physical identity by themselves.

Two latent worlds can produce the same cache: one object undergoes appearance/semantic change across a gap, or a different object takes its place with the same emitted features, boxes and times. Without an additional assumption or observable separator, the latent identity sequence is not point-identifiable. Supervised base-domain prediction can still work statistically; it must demonstrate transfer and uncertainty, not claim that raw appearance uniquely reveals contamination.

| Observable | Legitimate use | Unsupported inference |
|---|---|---|
| Independently eligible manual-anchor GT identities | Score same/different identity at those anchors; scope IDs by `(video_id,gt_track_id)` | Label every reference, infer exact switch time between anchors, or call unmatched observations foreign identity |
| Raw track-head features | Input-class-blind appearance predictor with boxes/time | Statistically semantic-free sensor; backbone and upstream association still share semantic information |
| Raw classification features/p | Semantic readout and explicit semantic-confounding control | Primary identity purity definition or identity-router input while claiming class-blind routing |
| All-emitted released tracks | Preserve complete deployed input and study fixed-ID readout | Recover suppressed/rejected detector alternatives or original complete association-cost matrices |
| Box/time differences | Relative geometry and lag controls | Dense motion, camera compensation, elapsed seconds or independence of new views |
| Posterior change/entropy reduction | A change in the chosen model's belief | Correctness gain, causal information gain or a new physical identity |

The existing cached reference is all other emitted observations, including future ones. That supports offline reference analysis. For online claims define `F_t` from observations available no later than t, select anchors without consulting GT, and construct past-only references separately. Do not silently mutate the sealed all-history reference index.

“Future innovation” should mean an expected improvement in a declared proper score or decision loss conditional on the past, averaged over independent source units. A single held-out frame's class-posterior change is not mutual information. Neither appearance distance nor de-duplication establishes conditional independence.

## 4. Minimal oracle and GT-free controls

Use the already written `base_fit_identity_edge_support_20261023_oracle_design.md`. Keep all sealed edges, endpoint exclusions and source roles. At eligible FIT manual-anchor endpoints compare the same fixed full1203 mean/argmax rule using: original all-emitted reference; references with independently known manual identities; same-GT references within that known subset; and deterministic hash-selected known-reference samples matched to the same-GT count. Actual emitted label and endpoint posterior argmax stay separate baselines.

The primary privileged comparison is **same-GT versus known-GT on exactly the same covered endpoints/contexts**, with declared source→official GT trajectory→unique endpoint→context weighting. Preserve full-denominator outputs and full-weight coverage. Unknown is unknown. Empty/singleton pools are reported. Pooling rule, count-control seed and membership hashes precede classification outcomes.

Count matching controls count, not quality, time, occlusion or viewpoint composition. Therefore call this an **observed-label oracle contrast**, not a mathematical purification upper bound or an identified causal effect of foreign identity. Foreign same-class observations can help; same-identity observations can remain consistently wrong. A small/negative contrast limits this rule and its labeled-reference scope, not every imaginable identity-aware method.

Only if meaningful headroom survives coverage/count accounting, consider a separate class-blind feasibility probe. The minimal deployment-visible controls are raw-track cosine or medoid selection, geometry/time segmentation, single-state quality pooling, and same-budget generic attention. Hold explicit p/raw-classification features out of identity decisions; use them as a diagnostic comparator. Required challenge cells are **same identity with large appearance/semantic change** and **different identities of the same class**. Predeclare cells and source-group evaluation; do not choose thresholds or class subsets from tune. GT identities label/score the eligible FIT subset, never select actual reference membership or replace unknown rows.

Artificial interventions are limited contract tests: changing only p while holding raw appearance/boxes fixed must not change an explicitly class-blind routing output; duplicating a cached row must not create an allegedly new identity or independent acquisition. A same-class foreign-identity transplant versus same-identity view-change test can expose a shortcut, but does not replace natural held-out-source evidence.

## 5. Decision boundary

- **HOLD network/method novelty now:** no oracle classification headroom or GT-free discrimination result has been established by this audit. Identity mixture/slot/gate plus category pooling is insufficient novelty.
- **Continue the separately authorized diagnostic:** different-GT support just makes that diagnostic nonempty; it does not pass the scientific gate.
- **STOP this cache-only point-identification claim:** it cannot uniquely recover complete identity sequences, dense switch times or causal independent information from feature similarity and sparse labels alone.
- If labeled-reference coverage is inadequate, report inconclusive and stop expanding the model. If this fixed oracle has no useful headroom, deprioritize this specific fixed-ID readout route; do not generalize a finite-rule negative to all tracking.

## 6. Concrete COVTrack code candidate 1: keep confidence in its original feature space

The frozen source has a specific feedback path:

1. `ovtrack/models/trackers/ovtracker.py:531–537` computes appearance consistency between **current raw embeds** and `memo_embeds`.
2. At `:637–640`, `fusion_head` injects classification/location residuals and replaces current `embeds`.
3. At `:690`, those fused embeds are passed to `update_memo`; `:420–422` writes them into the embedding EMA.
4. At the next observation, `memo_embeds` therefore contains previous semantic/location fusion while the current input to appearance consistency is raw.

Training uses raw key/ref appearance to compute consistency before fusing both (`ovtrack_roi_head.py:2500–2571`); the recursive inference confidence is consequently evaluated in a different pair of feature spaces. This is a verified dataflow distinction, not proof of a bug or performance loss.

**Concrete engineering candidate:** retain a raw-appearance EMA for the appearance-consistency query; retain the existing fused EMA for final association. Preserve raw features before fusion and apply the same assigned IDs to both stores. Location and semantic confidence keep their own inputs. This needs no new router, attention or GT at inference and directly tests whether past fusion feeds back into a confidence intended to measure appearance. It is ordinary decoupled-memory engineering until supported by a broader result and literature boundary.

Falsifiable mechanism: under fixed raw current/past appearances and boxes, perturb only historical class features. Raw-to-raw appearance confidence must stay invariant; current raw-to-fused confidence generally can change. Verify shadow-cache mode reproduces original assignments exactly before activating the new confidence input. Then separately test frozen-parameter association on source-held-out FIT replay, including same-class distractors, class-conflicting views and lag; compare original, shadow cache, raw-only final matching and the proposed split. A confidence change without improved same-GT versus different-GT matching does not validate the mechanism. Changed association requires its own complete evaluation; fixed-ID classification gains cannot substitute.

**Stronger, separately controlled prototype: reinterpret history in the current comparison context.** Retain raw appearance, classification features and box/time for memory. At the current frame, compute raw-to-raw confidence in both directions between the current candidate set and raw memory; apply the same frozen fusion head to each side using that current comparison context; compare the resulting current-context embeddings. Write only raw appearance to its EMA, so past fused residuals do not recursively become the next “raw” cue. This more closely follows training's fuse-both-sides path than comparing a newly fused current vector with a historical vector fused under an older candidate set. Original inference, raw-confidence-only, and re-fusion-of-both-sides must be separate ablations. Re-fusion is not a new information source and is not a claim of new attention/memory theory.

An exact limitation survives both prototypes: when current and memory sets each have one candidate, `get_pair_consistency` uses `log(1)/.1=0`, both row-softmax values are1 and cycle confidence is1 regardless of feature similarity. Consequently `max_fusion_ratio*(1-confidence)^2=0` and the auxiliary residual is suppressed. A perfectly invertible wrong permutation likewise has high round-trip confidence. The score measures determinacy within the offered candidate sets, not physical correctness. Include a singleton negative pair and a same-class swapped-pair control; retain absolute appearance/geometric compatibility separately rather than treating a closed cycle as verified reliability.

## 7. Concrete COVTrack code candidate 2: replace coordinate-direction similarity with meaningful relative geometry

`ovtracker.py:531–537` sends absolute `[x1,y1,x2,y2]` vectors to the same `CycleLoss.get_pair_consistency` used for embeddings. At `cyc_loss.py:104–107` these vectors are L2-normalized and compared by cosine. `FeatureFusionModule.bbox_transform` (`ovtrack_roi_head.py:1523`) also receives raw coordinates.

Cosine of absolute box-coordinate vectors does not measure motion plausibility or overlap. Applying the same image translation to both frames preserves relative object motions/overlap but changes these cosines; with a sufficiently large common offset, all normalized box vectors tend toward one direction. Thus the location confidence has no coordinate-origin invariance by construction. This algebraic observation does not imply every real image translation harms the trained model.

**Concrete ordinary baseline:** use frame-size-normalized boxes for the node feature and normalized relative center displacement plus log width/height ratios for pairwise geometric compatibility, with lag explicit. This requires a clear node-versus-edge interface: pair compatibility cannot be passed as a fake 4D node vector. Do not claim camera-motion invariance without camera observations. Compare to IoU/standard motion gating and retain exact box/output protocol.

A concrete pair innovation can start from `(dx/sqrt(w_t*w_s), dy/sqrt(h_t*h_s), log(w_t/w_s), log(h_t/h_s))`, keeping elapsed acquisition gap as a separate input. It removes common coordinate-origin and uniform-scale dependence; it does not remove camera motion, acceleration or depth effects. For a true velocity innovation use `original_frame_index` and verified per-video fps/timestamps, not `sampled_step`; the latter counts sparse collected samples. If fps is unavailable report frame-gap units and do not call the result physical velocity. Ego-motion normalization needs a separately verified image transform or camera-motion record; it is not contained in raw box/feature similarity. Warping the past box into the current image before computing innovations is a proposal only if that observable transform is available.

The useful challenge is whether a same-class, similar-appearance distractor has incompatible motion while a same-object changed view remains geometrically consistent. Motion can help in that regime. It cannot separate two similar objects occupying indistinguishable positions during a crossing, or distinguish object motion from unobserved camera motion without assumptions. Evaluate those ambiguous cases explicitly and allow uncertainty. Combining current-context appearance re-fusion with object-scale/time/ego-motion-normalized pair evidence is therefore a falsifiable use of existing features; its scientific claim must be about the specific failure it resolves, not “using time, space and appearance.”

This is an established geometry correction, not new tracking theory. A potential research problem would be to learn **when geometric and semantic cues disagree about a particular candidate identity**, using truly pair-specific evidence and bounded cost while preserving instance discrimination. Generic edge GNNs and learned pair fusion already cover the broad form; a new paper would need a concrete unsolved failure, matched controls and a stronger source audit. I do not currently recommend claiming that contribution.

## 8. Final recommendation

Prioritize the raw/fused-confidence dataflow probe over adding an identity-slot network. It has a precise source location, a controlled intervention and a useful negative outcome. Keep identity-memory novelty on HOLD until the FIT oracle contrast and class-blind challenge show both relevant headroom and available separating information. Neither proposed code change is implemented or claimed novel here.
