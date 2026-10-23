# COVTrack / TCES novelty and mechanism audit, 2026-10-23

> Resolution note: this audit records the pre-repair state. The invalid tables and conclusions have now been removed from formal reports. Four candidate readouts plus the released baseline completed official full-validation TETA; released sumlog achieved Base/Novel 39.774/35.771. Current evidence and limits are in `report/COVTrack_完整val轨迹读出_20261023.md`. The mechanism and novelty cautions below remain applicable.

Scope: read-only source and artifact review. No new training, no shared implementation edits, no baseline changes. This audit does not establish global literature novelty. It identifies mechanism overlap, invalid comparisons, and two remaining hypotheses that current caches can falsify without training.

## 1. Decision

The immediate priority is to repair the evidence chain and evaluate the existing classification readout on complete cached observations. The present reports do not support either declaring TCES effective or declaring it disproved. Several published negative conclusions were computed by scripts with different inputs from the claimed baseline.

The current TCES proposal is largely a design. The implemented `ovmot_improve/cov_tces_label.py` is a track-label aggregation probe; it is not the proposed learned state with classification, association, and lifecycle readers. Merely replacing vote with reliability-weighted pooling is a weak novelty boundary against TRACT, OVTR, and COVTrack++.

## 2. Implementation facts that affect every conclusion

### 2.1 Cross-video identity collision in the main TCES label probe

- `ovmot_improve/cov_tces_label.py:77` groups by `tracks[int(tid)]`.
- `work/covtrack_tces_v2/ovtrack/models/mot/ovtrack.py:177-179` reconstructs the tracker at `frame_id == 0`.
- `work/covtrack_tces_v2/ovtrack/models/trackers/ovtracker.py:387-389` resets `num_tracklets = 0`.

Thus the key must include the video, and negative/unassigned IDs must be excluded. The existing saved `vid` field does not fix grouping. `tmp_analysis/val60_analysis.py` and `macro_novel.py` do use `(video_id, tid)`, so their identity handling differs from the main probe.

### 2.2 The macro report calls the text branch the official TCC

`tmp_analysis/macro_novel.py:30` stores `(p_text, p_image, label)`; lines 32-37 apply `tcc`, `sumprob`, and `sumlog` to the first element, `p_text`. They never use `p_fused`. Therefore the corresponding report table in `report/COVTrack_论文级改进设计.md:388-397` cannot establish that official TCC obtains 2/90 tracks or that fused sumlog fails to beat it. Its one label per track is also the first matched GT category, so impure tracks require a separate definition.

### 2.3 Claimed official branch routing uses GT in one probe

`tmp_analysis/fusion_adaptive.py:48-51` chooses an exponent for the entire probability vector using the GT category's rare status. Actual COVTrack uses a separate mask value for each candidate class column. The printed `official fixed routing` is not the official algorithm. The margin route at lines 59-63 also gives more weight to the image branch when the text margin grows, contrary to its explanation that the sharper branch is upweighted.

### 2.4 Aggregating after GT filtering is a different method

`cov_tces_label.py:75-76`, `val60_analysis.py`, and `macro_novel.py` discard observations that lack a GT match before creating the trajectory evidence. Official TCC votes over every emitted observation of the track. GT filtering may be used to score an output but must not select the evidence that produces that output. The current independent per-detection nearest-GT mapping is also many-to-one; it is not one-to-one localization matching or the complete TETA protocol.

Other bookkeeping: `cov_tces_label.py:131` sums the same observation count across all five rules. This is not a distinct observation total. Observational accuracy, track accuracy, category macro accuracy, and TETA ClsA are different estimands; none can substitute for another.

### 2.5 The released fusion mask has an index offset

The `class_name.py` files in `repos/COVTrack`, `work/covtrack_base`, and `work/covtrack_tces_v2` have identical SHA256:

`6f50d7b3f16eef96e15b6cfbf97a53b195e26dbc34df51e7679645c99e6937d9`

AST inspection found one effective `validation_novel_index` assignment at line 3954, and one `test_novel_index` assignment at line 3955. Lines 3952-3953 contain commented lists; they do not overwrite the active definitions. The effective validation mask has 297 entries and 86 true values. On the first 296 columns:

- `mask == [cid in lvis_novel_label_ids for cid in tao_validation_used_cate_ids]`: zero differences.
- `[cid-1 in lvis_novel_label_ids]`: 35 true values; zero differences from annotation `frequency == 'r'`.
- Active mask versus true rare status: 93 disagreements, comprising 72 base classes marked true and 21 rare classes marked false. Only 14 rare classes are marked true.

In the active ROI-head class, text-feature selection uses `cid-1`, but the precomputed mask is loaded directly (`repos/COVTrack/ovtrack/models/roi_heads/ovtrack_roi_head.py:1819-1822`). This localizes a 1-based category-ID / 0-based LVIS-label-index discrepancy. It is inherited from the released repository; preserve the released A baseline. A corrected-mask counterfactual must be labeled an implementation/protocol correction, not a new method.

Actual formula: mask true gives `p_image^(2/3) * p_text^(1/3)`, false gives `p_text^(2/3) * p_image^(1/3)`. `report/COVTrack_论文级改进设计.md:358` describes these weights in the opposite direction, in addition to treating the mask as a correct base/novel indicator.

### 2.6 Scalar trajectory temperature cannot improve ranking

For positive scalar T, `softmax(E/T)` has exactly the same top-1 and top-k ordering as E. A trajectory temperature can calibrate confidence or change a downstream gate, but a standalone temperature readout cannot repair classification ranking. This should not be presented as a separate classification-improvement mechanism.

## 3. Exact overlap boundaries

### COVTrack

The active model already fuses appearance, location, and semantic features with learned confidence gates. Semantic features influencing association is therefore not new. Its released classification path is per-ROI dual-branch exponent fusion followed by track majority voting; it does not maintain the proposed complete posterior state. The current TCES scripts are readout probes over those released features and posteriors.

### COVTrack++

Primary source: `papers/COVTrackpp_arxiv2603.24016.pdf`, Sections IV-B through IV-D, particularly Eqs. 18-23. A text extraction for this audit was stored at `/tmp/covpp_20261023.txt`.

- MCF already uses appearance/location/semantic features jointly to estimate intra/inter-frame reliability and modulate fusion. Adding bbox/semantic confidence to a residual strength is an MCF variant until independently demonstrated otherwise.
- MGA already aggregates reliable spatially contained child detections through TAQ-gated attention. Spatial context, nested boxes, and reliable parts are occupied directions.
- TCP uses association-feature similarity plus IoU to connect high-confidence sources to low-confidence candidates. Eq. 22 requires the same predicted category; Eq. 23 propagates scalar category confidence. Recovered candidates may become sources in later frames. It is inaccurate to describe TCP as incapable of a multi-frame propagation chain merely because its graph is constructed between adjacent frames.
- An explicit full categorical posterior with a dependence model differs from TCP's same-label scalar confidence propagation. Merely adding confidence memory or propagating labels does not.

No local COVTrack++ implementation was found in the inspected repository tree. These boundaries are from the paper, not an executed reproduction.

### OVTR

Primary code: `repos/OVTR/ovtr/models/updater.py:186-294`; call site `ovtr.py:796-825`.

`Category_Information_Propagator._aggregate_category_info` takes `output_embedding_img`, adds query position for self-attention across active track queries, applies an MLP, and writes an updated `query_tgt`. It also updates detached reference boxes. The propagated category information is represented in learned features/query state, rather than a literal previous-frame class-probability EMA. The earlier research note's description as probability propagation is imprecise.

Thus online category-aware track state is already present in OVTR. The inspected CIP code does not explicitly model covariance between posterior evidence from repeated views, and does not retain a mixture of competing identities with separate class posteriors. Those narrower mechanisms are potential boundaries, not proof that the whole topic is new.

### TRACT

Primary source: official arXiv HTML `https://arxiv.org/html/2503.08145v1`, Sections 3.3-3.4. Downloaded read-only to `/tmp/tract_2503.08145.html` and extracted to `/tmp/tract_2503.08145.txt`.

- TCR keeps an appearance feature bank and combines similarity against the EMA memory with the mean similarity against past features (Eq. 2). A multi-view memory bank is occupied.
- TCR also keeps a category bank. High-confidence current predictions are accepted; middle-confidence predictions join the bank and vote; low-confidence predictions are replaced by historical voting. Confidence-dependent online category memory is occupied.
- TFA selects trajectory frames using detection confidence, applies CLIP visual encoding followed by self-attention and an MLP, and averages the enhanced features. Generic learned temporal aggregation and selecting reliable views are occupied.
- TSE expands text categories with visual attributes. A final selector chooses among category-text similarity, attribute-text similarity, and vote confidence. Multi-source classification selection is occupied.

Local code limitation: `repos/TRACT/TraCLIP/main.py:15` imports `models.track_classifier.TraCLIP`, but the local `TraCLIP/models` directory is absent. The actual MASA tracker implementation is likewise absent from the inspected clone. Only entrypoints, helpers, configs, README, and paper mechanisms could be checked. Do not report a completed source-level exclusion of its TFA/TSE/TCR internals.

## 4. Candidates that remain distinguishable enough to probe

These are hypotheses only. First complete released-TCC parity and the corrected-mask readout control. A large gain from fixing the mask would change which error remains worth researching.

### A. Distinguish new visual evidence from repeated correlated evidence

Scientific question: do pooling and voting become overconfident in a repeated but wrong view, while complementary views are underweighted? Existing TCES reliability uses margin, entropy, agreement, and consistency. None explicitly accounts for dependence between observations. TRACT attention can express some redundancy weighting, but its stated mechanism and supervision are not a probabilistic independence model or a duplicate-invariance constraint.

Use the cached appearance embeddings to estimate view redundancy, while retaining the complete dual-branch posterior. Maintain evidence for appearance/view components rather than treating each frame as an independent vote. A component contributes according to distinct support; replicating an identical observation must not add evidence. Explicitly separate epistemic evidence volume from score temperature. Any prior-correction derivation must account for the fact that multiplying posterior vectors repeats the class prior, but a mere global inverse-frequency correction repeats the already tested `prior_debias.py` direction and is insufficient novelty.

Zero-training falsification, in order:

1. Compare complete-observation TCC and existing pools under duplicate-frame insertion and removal. Check whether natural repeated-view density predicts label errors after controlling for length and class.
2. Compute an analysis-only oracle upper bound that picks complementary appearance-view components, with GT used only to measure attainable gains. If this cannot rescue labels, stop.
3. Use one fixed prediction-only redundancy rule and compare against random frame subsampling with the same evidence budget, confidence-based selection, ordinary pooling, and an unweighted appearance-bank control. Evaluate official Base/Novel ClsA/TETA and per-class paired changes. The rule must gain on natural tracks, not merely satisfy the synthetic invariance test.

Current caches provide `track_feats`, `cls_feats`, `p_text`, `p_image`, `p_fused`, frame index, boxes, and filename. They suffice for this diagnostic. A scalar confidence gate or arbitrary clustering threshold without the dependence test would reduce this proposal to another aggregation trick.

### B. Separate uncertain class from uncertain identity membership

Scientific question: when a track contains conflicting semantic evidence, is it one object observed poorly, or multiple identities mixed by association? The proposed TCES `purity_proxy` is calculated from class agreement, creating a circular tendency to trust whichever class already dominates. TCR regularizes a single trajectory's consistency. CIP carries one query state. TCP assumes equal predicted category. None of the inspected mechanisms explicitly carries separate latent identity membership and class evidence.

The cache supports an analysis-only upper bound: quantify how often GT identity purification changes the official track label, then compare it with selecting the largest appearance-coherent component without GT. If purified evidence does not improve the relevant error population, stop this direction before any learned model.

If the upper bound is substantial, a candidate state could hold competing appearance components with separate class posteriors and infer whether an observation joins an existing component or starts a new one. Category agreement must not define identity membership. First test label readout with frozen IDs; only test identity splitting separately if classification evidence supports it. Compare with ordinary medoid filtering, simple feature-bank selection, and change-point baselines.

Novelty risk is significant: mixture tracking, robust aggregation, and tracklet splitting have broad prior art. The possible OVMOT-specific contribution is a demonstrated distinction between identity contamination and open-vocabulary semantic uncertainty, plus joint inference that retains both alternatives. A renamed feature gate or generic split heuristic would not suffice.

## 5. What is already failed, and how many checks are actually evidenced

| Route | Evidence found | Audit status / rounds |
|---|---|---|
| OVTrack global motion fusion | Small val3 gain, val200 drop; `report/OVTrack_收尾总结.md:92-96` | Negative in that OVTrack setting; at least 2 scale checks documented, not 3 separately evidenced failure audits |
| OVTrack R4 normalization | Sigmoid collapse, min-max failure, explicit third-round rejection; same report:98-102 | 3 rounds explicitly recorded; retain stop |
| OVTrack motion gate / Hungarian | val60 small gain, Hungarian worse; same report:104-108 | Negative/weak; no explicit 3-round log in inspected files |
| OVTrack `rams3` modes | `score_mode` ignored; same report:110-112 | Implementation invalidation; affected comparisons cannot be theoretical disproof |
| Global class-prior / rare boost | `prior_debias.py` sweeps global beta/temp/boost on OVTrack caches | Already attempted; not a new candidate; no 3-round audit found |
| Adding class posterior to pair score | OVTrack pair probe: app 92.697 vs app+cls .3 92.563; design report:326-343 | One pair probe, not end-to-end COVTrack failure; no 3 rounds evidenced |
| Simple reliability routing | OVTrack entropy/margin/novel-mass quantiles below oracle; design report:312-324 | Weak rule-family probe, not proof that learned state works; no 3-round audit evidenced |
| Sparse-val dormant recovery | memo=50 rarely expires on sequences of about 40 sampled steps; design report:269-276 | Protocol invalidation; stop physical-occlusion claims on this split; requires dense validation rather than more threshold sweeps |
| Fusion-strength C2 | COVTrack++ MCF overlap, design report:360-375 | Novelty rejection, not measured mechanism failure; no need to pretend 3 performance rounds |
| COVTrack sumlog / macro negative | Cross-video key bug in main script; macro script uses text instead of fused probabilities | Existing negative conclusion invalid; rerun unified readout before deciding |
| TCES posterior dump | NMS class-count divisor, class-column mapping, tracker keep-mask issues documented in `report/项目进度.md` §2.5 | At least 3 implementation problems fixed; debug version counts are not 3 scientific-failure rounds |

The reports require three failure reviews (implementation, evaluation protocol, theory). This is a requirement, not evidence that every route has completed three. Do not infer review counts from filenames such as `dbg11` or `v3`.

## 6. Required handoff

1. Keep the released A baseline and ongoing training artifacts intact.
2. Use complete prediction observations to form every state; use GT only for scoring/explicit oracle bounds.
3. Bind keys to video and track, preserve chronological order, exclude unassigned IDs, and reproduce official TCC tie-breaking and category IDs.
4. Report released-mask baseline, corrected-mask control, and proposed-method changes separately. Do not attribute the correction's gain to TCES.
5. Only after complete TETA and paired per-class evidence should either new hypothesis receive implementation effort. No training is justified by the current scripts alone.

## 7. A falsifiable mathematical version of candidate A

### 7.1 Hypothesis and its limitations

Fix the released detector, IDs, boxes, complete set of observations, and one explicitly named class-column fusion mask. Let `a(c)` be its fixed image-branch exponent (1/3 or 2/3), and define the evidence vector for observation t:

`e_t(c) = (1-a(c))*log(p_text_t(c)) + a(c)*log(p_image_t(c))`.

Use one declared clipping floor/dtype throughout, and report its sensitivity if cached probabilities contain zeros. Log clipping can itself change a sumlog ranking.

The difference from `log(p_fused)` is a per-observation normalization constant and does not alter ranking under any common temporal weight. Do not select a fusion exponent from the GT class. Run released-mask and corrected-mask controls separately.

A specific statistical hypothesis on a stable-identity track k is:

`e_t(c) = mu_k(c) + epsilon_t(c)`, with `Cov(epsilon_t(c), epsilon_u(c)) = sigma_c^2 * K_k[t,u]`.

Here K represents dependence between observations, estimated from prediction-visible appearance features, not from GT correctness. This is a working covariance model, not an established property of COVTrack. A feature-similarity kernel is not automatically the covariance of classification errors. Moreover, estimating the mean evidence `mu` does not imply the correct class has the largest mean. Systematic branch bias or a consistently wrong view can defeat any such estimator.

Under this hypothesis, a standard generalized least-squares estimate is

`w_GLS = K^+*1 / (1^T*K^+*1)`, `E_k(c) = sum_t w_GLS[t] * e_t(c)`.

`K^+` denotes a suitable generalized inverse. K proportional to identity reduces the rank to ordinary sumlog. The normalization is a common positive scalar and does not create a new classifier. Unconstrained GLS can produce negative weights and unstable estimates; an explicitly labeled nonnegative minimum-variance alternative is

`min_w w^T*K*w`, subject to `w>=0`, `1^T*w=1`.

This constrained estimator and GLS are existing statistical tools. Neither is itself a contribution. Using a kernel to assign weights and calling it a new evidence module would overstate novelty. The only currently defensible new hypothesis is that a measurable dependence failure in OVMOT survives the corrected implementation and that an explicit dependence model addresses it beyond attention/selection baselines.

Exact duplicate observations need perfect covariance with their source; adding an independent diagonal noise term per duplicated row incorrectly invents information. Collapsing exact duplicate groups before estimation is an implementation control. Near-duplicate treatment must be sensitivity-tested. A mathematically valid duplicated-row covariance should preserve the estimate and effective information under exact repetition.

No new ground-truth fitting is authorized. A kernel scale, if used, must be fixed from an unlabeled distance statistic or predeclared, then compared with matched controls. Selecting the best kernel/hyperparameters on novel GT and reporting the same split as independent validation would not support generalization.

### 7.2 What would distinguish it from nearby methods

| Nearby mechanism | Necessary distinction for candidate A |
|---|---|
| TRACT score sampling | Improvement beyond selecting equally many high-score frames and beyond random sampling; dependence should explain errors conditional on score and track length |
| TRACT self-attention / generic attention pooling | Explicit model of correlated evidence and duplicate invariance, with a matched fixed-feature pooling control; a different weight function alone is insufficient |
| Ordinary GLS / covariance-aware pooling | These are the proposed diagnostic estimators, not new contributions; a paper needs a verified OVMOT-specific failure, justified covariance/identity treatment, and evidence beyond applying GLS |
| Existing TCES margin/entropy/agreement gate | Marginal reliability asks whether one frame seems trustworthy; dependence asks whether it adds evidence beyond other frames; ablate both separately |
| COVTrack++ TCP / MCF | Freeze detection/recovery/association. The probe modifies only full-posterior evidence readout, without confidence propagation or feature-fusion changes |
| OVTR CIP | No new learned query-update network or cross-object attention; the probe measures dependence among observations already assigned to a fixed track |

### 7.3 Three review rounds, two checks each

| Round | Check 1 | Check 2 |
|---|---|---|
| 1. Implementation and protocol | Complete observation-inventory equality: same video/track keys, chronological observations, IDs, boxes, scores, candidate-class columns, and TCC tie rule. Hash the input/output identity+box inventory. The ordinary TCC readout must reproduce the released JSON before testing weighting. | GT is loaded only by a separate scoring/oracle process. No GT-dependent frame filtering, branch selection, component choice, or kernel tuning. Score Base/Novel separately with the true frequency split and official evaluator, keeping its clustering/matching protocol unchanged. |
| 2. Controlled mechanism falsification | Duplicate-frame intervention occurs in a diagnostic copy of a track's evidence stream only; do not add rows, frames, IDs, or boxes to the main prediction table. Choose repeated observations without GT, repeat identical/near-identical views, and compare label/rank/confidence changes with matched random repetitions. Exact duplication must not create effective evidence under the dependence estimator. | Test limiting cases and competing explanations: diagonal K gives sumlog, one retained frame gives that frame, exact duplicates collapse, weights are finite and normalized. Compare against equal-budget random/score subsampling, uniform pool, confidence weights, and a simple appearance-component pool. A gain explained entirely by discarding low-score frames rejects the claimed dependence mechanism. |
| 3. Theory and external validity within the fixed split | Separate identity contamination from classification noise using read-only GT diagnostics. Analyze high-purity tracks with diverse views, high-purity tracks with repeated views, and mixed-identity tracks separately; do not purify actual inference evidence with GT. Measure whether dependence predicts errors within pure tracks after conditioning on class, length, and confidence. If effects exist only on mixed tracks, candidate A's class-noise explanation is unsupported. | Evaluate complete-val readout with fixed IDs/boxes/mask and paired per-video/per-class changes, reporting full Base/Novel ClsA/TETA plus macro effects and intervals. Require natural-track gains beyond controls, rather than success only under artificial repetition. If the feature kernel fails to track error covariance, or gains vanish after the mask correction/equal-budget controls, stop and do not train. |

The three rounds are distinct reviews, not repeated threshold sweeps. A failed implementation check restarts the dependent evidence; it is not a theoretical rejection.

### 7.4 Feature provenance

The dumped `track_feats` come from the appearance `_track_forward` output (`work/covtrack_tces_v2/ovtrack/models/roi_heads/ovtrack_roi_head.py:2973`). The tracker reassigns its local `embeds` variable to fused features (`ovtracker.py:640`); the outer `track_feats` used by the dump (`mot/ovtrack.py:261`) remains the pre-fusion appearance tensor. `cls_feats` is the two branch embedding average. These support an appearance-based dependency proxy, but sharing detector/backbone input means appearance and class errors cannot be assumed statistically independent.
