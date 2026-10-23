# Base-only error learning: implementable objectives and evidence gates

2026-10-23. Independent objective review. No training, inference, GPU work, or validation-data access was performed. The only empirical inputs are the already sealed **training metadata plan** and readiness report. Pure finite-support synthetic checks are recorded separately below.

**Decision:** start from the frozen classifier's average error and a class-shared single-observation quality baseline. A pair learner is justified only if it predicts held-out excess error after these means and prediction agreement are controlled. The present pilot is a schema/signal pilot, not sufficient evidence for a trained shared-error mechanism. The following is implementable, but neither its efficacy nor novelty is established.

## 1. Unit, prediction, and supervision

For an emitted observation `i`, retain a frozen, aligned foreground posterior `p_i in R^K`, its text/image branches, box/score, raw appearance summaries, and time. `K` is the actual recorded vocabulary regime; background handling and foreground renormalization are fixed in the collection manifest. Let `h_i=argmax_c p_i,c`. On a **class-blind geometry-matched original manual anchor only**, the training true class is `y_i`.

Scalar targets:

```text
e_i = 1[h_i != y_i]              # hard classifier error
r_i = 1 - p_i,y_i                # soft wrong-class mass
```

Use only fit-video × fit-target-class anchors in optimization. A pair for semantic dependence additionally requires the same annotated GT identity (`y_i=y_j=y`), two valid manual matches, and the declared predicted-track support rule. Matching must not use predicted category. Mixed identities, ambiguous geometry matches, pseudo/propagated boxes, and unmatched observations receive separate provenance; they are not negative examples and cannot be promoted to pure semantic dependence by a track's dominant class. Same-GT support on matched anchors does not prove that the entire predicted track is pure.

Source groups are the independent sampling units. Video, GT trajectory, predicted track, frame, and pair are nested repeated observations. Keep a target inclusion receipt with counts at every level. Prefer equal source-group, then trajectory weighting, with fixed capped temporal pair sampling; exhaustive `T(T-1)/2` pairs must not give a long easy video dominant weight. The lag bins and sampling seed are declared before opening tune/test labels. No augmentation-generated error or synthetic occlusion substitutes for natural error targets.

## 2. First baselines: classifier average error, then quality

**B0, no learnable network:** estimate the frozen classifier's average hard error `b_bar` and average soft risk `a_bar` from fit-only matched anchors, under the same source/trajectory weighting used for the pair task. Report raw/micro and group-weighted versions separately. If studying branches, estimate separate constants for text, image, fused. Do not use true class, frequency, or pseudo-novel membership to select a constant at inference. This is the required “average classifier error” baseline.

**B1, class-shared quality:** use a small scalar model `b_i=f_theta(u_i)` with BCE against `e_i`; optionally a separate bounded scalar `a_i` with squared loss against `r_i`. It estimates mean risk, not correlation. Its inputs can include normalized entropy `H(p)/log K`, top probability/margins, sorted probability summaries, branch disagreement/JSD, object score, normalized box area/aspect, feature norms and GT-free history summaries. All normalization and actual `K` are explicit. No GT label, category ID embedding, base/novel flag, GT IoU, track purity, future annotated state, or source-group identity is an input.

Start with B0 and B1 before increasing network size. Measure held-out Brier/BCE, calibration intercept/slope and error rates in fixed quality/agreement/lag strata, with source-group uncertainty. AUROC alone cannot establish calibration. A pair improvement over B0 that disappears after B1 is quality estimation, not evidence of additional dependence.

For a **strong calibration control**, use a shared candidate calibrator rather than a `K x K` confusion table:

```text
s_i,c = log(p_i,c + eps) + g_theta(u_i, p_text_i,c, p_image_i,c,
                                 p_fused_i,c, normalized_rank_i,c)
pi_i = softmax_c(s_i,c)
mu_i,c = p_i,c * (1 - pi_i,c)
```

`g_theta` has the same weights for every candidate and no learned absolute class index. Fit `pi_i` with CE on fit-only base targets; its implied hard risk is `1-pi_i,h_i`, and implied soft risk is `sum_c mu_i,c = 1-sum_c p_i,c*pi_i,c`. Compare these with B1. Shared-candidate CE uses other vocabulary columns as competing negative candidates for a matched base object; this must be disclosed. “Class holdout” means no positive target label from that class enters optimization/selection; it does not mean that the held-out column has never appeared in the softmax.

Calibration does not require a flexible MLP initially: a shared scalar temperature is the low-capacity control. A flexible class-shared calibrator must beat that control before its residuals are trusted. Raw visual embeddings, if later enabled, may encode semantics even without class IDs; add them as an ablation rather than call them semantically blind.

**This plug-in `mu=p*(1-pi)` is a control, not the primary covariance-centering model.** For a common true label its residual is `R_i,c=p_i,c*(pi_i,c-1[y=c])`. Each coordinate has the same sign across frames, so all same-identity diagonal residual products are nonnegative by construction. A positive product cannot establish a natural correlated-error process. The direct mean regression in section4 is required to avoid this algebraic degeneracy.

## 3. Pair excess is not a product of mean risks

For a common true identity define:

```text
J_ij = e_i * e_j                             # both wrong
S_ij = 1[h_i == h_j] * e_i * e_j             # same wrong top class
```

`b_i*b_j` is an independence reference for **joint both-wrong probability**. It is not a covariance, not a same-wrong-direction prediction, and not an estimate of the full classification residual covariance.

With appropriate means under the same conditioning set `V`, the binary covariance is

```text
Cov(e_i,e_j | V) = E[J_ij | V] - E[e_i | V]*E[e_j | V].
```

A directly usable calibration target is the centered product

```text
t_hard_ij = (e_i - b_i) * (e_j - b_j).
```

Centering must use **out-of-source-group** mean predictions. `J_ij-b_i*b_j` and `t_hard_ij` are not equal for an individual pair: the centered product contains both cross terms. Their expectations agree only under the needed mean-calibration/conditioning conditions. Single-observation calibration does not imply `E[e_i | pair context]=b_i`; quality-mixture composition alone can create a positive residual. Therefore call learned outputs **excess calibration residuals** until this condition is checked, rather than measured conditional covariance.

A bounded probability parameterization for a hard-event calibration control is:

```text
L_ij = max(0, b_i+b_j-1)
U_ij = min(b_i,b_j)
q_J = L_ij + (U_ij-L_ij)*sigmoid(logit(q0_scaled)+delta_ij)
q0_scaled = (b_i*b_j-L_ij)/(U_ij-L_ij)
```

Handle degenerate bounds exactly; numerical clipping is explicit. With `delta=0`, this recovers `b_i*b_j`. Fit with BCE to `J_ij`, and report `q_J-b_i*b_j`. These Fréchet bounds enforce compatibility with the proposed margins only; if pair-conditioned margins differ, a fitted pair probability can expose that misspecification rather than identify dependence.

**Critical hard-label identity constraint:** given the actual argmax labels of two observations with one true label:

```text
h_i == h_j  =>  e_i == e_j
h_i != h_j  =>  e_i+e_j >= 1, and J_ij=e_i+e_j-1
S_ij = agreement_ij * J_ij
```

Consequently a network can predict hard same-wrong through ordinary agreement and quality, without learning a new natural shared-error process. Adding a pair network to those labels is not by itself a substantive mechanism. The hard target is a necessary diagnostic, not the main novelty-bearing target. Condition/match by agreement and endpoint quality before interpreting any appearance-similarity increment.

## 4. Separate wrong-evidence magnitude from direction

Define the soft wrong-class evidence vector on training anchors:

```text
W_i,c = p_i,c * 1[c != y]
sum_c W_i,c = r_i
v_i = W_i / r_i               # only if r_i > 0
G_ij = dot(W_i,W_j) = r_i*r_j*dot(v_i,v_j)
```

This distinguishes endpoint risk magnitude `r_i*r_j` from **which wrong classes receive common evidence**. Two pairs can have identical mean error and both-wrong rates but different same-wrong directions. `dot(v_i,v_j)` is a conditional wrong-direction overlap; it does not establish centered covariance or causal view dependence. When risk is almost zero, do not let the normalization by `r_i` amplify numerical noise: the primary target is unnormalized `W`/`G`, with direction-only summaries restricted by an explicit fixed support threshold.

For the **primary implementable mean model**, define a coarsened observed context `V_ij` and directly regress the mean wrong-evidence vector, rather than using `p_i*(1-pi_i)`. One concrete context is endpoint quality summaries, explicit argmax agreement, lag and observed feature cosine, together with a GT-free reference posterior `q_ref` pooled from other observations of the predicted track, excluding **both endpoints**. For the first offline protocol require at least2 other reference observations (>=4 observed track frames); unsupported pairs fall back to quality-only. This is a declared design choice, not a sufficiency guarantee. Online use instead needs a separately declared strictly-past reference.

The per-candidate features of the primary mean/second-moment heads contain `q_ref,c`, reference branch scores/rank and the same coarsened context. They exclude the individual current endpoint probabilities `p_i,c,p_j,c` and GT-dependent coordinates. This intentional coarsening allows current wrong evidence to vary around its conditional mean; otherwise the plug-in sign constraint above returns. Mean and second-moment heads use the **same conditioning set**; adding richer inputs only to the second-moment head changes the estimand and requires a new conditional mean audit.

```text
mu_i|ij,c = sigmoid(a_theta(V_ij, endpoint=i))
            * softmax_c(g_theta(V_ij, endpoint=i, q_ref,c, ...))
L_mean_direction = sum_c (W_i,c - mu_i|ij,c)^2   # direct supervised regression
```

The output is nonnegative and has total mass<=1, but is not forced below current `p_i,c`; its residual can change sign across natural views. Both `a_theta` and `g_theta` are class-shared. Source-group cross-fitting and endpoint/pair sampling weights prevent repeated copies of one endpoint from dominating this regression. True `y` appears only in the supervision `W`, never the network input or reference construction. The reference can contain identity contamination because it is predicted-track based; record this limitation and analyze it separately. It may also carry semantic difficulty/true-label uncertainty; this construction does not remove that alternative explanation automatically.

Using these out-of-group, pair-context mean predictions `mu_i=mu_i|ij` and `mu_j=mu_j|ij`, form centered residuals:

```text
R_i,c = W_i,c - mu_i,c
t_dir_ij,c = R_i,c * R_j,c
t_trace_ij = sum_c t_dir_ij,c
```

The full centered product is essential:

```text
t_trace_ij = dot(W_i,W_j) - dot(W_i,mu_j)
             - dot(mu_i,W_j) + dot(mu_i,mu_j).
```

`dot(mu_i,mu_j)` is the **factorized mean-direction reference**, not the full covariance. `t_trace` targets a diagonal sum of cross-frame wrong-evidence residual products. It omits all `c != d` terms; wrong-class-1 versus wrong-class-2 dependence can be nonzero while this trace is zero. It is not the full covariance of `p-onehot(y)`, does not identify a covariance matrix, and is not a GLS precision matrix. Even under correct means it can include shared true-label uncertainty and semantic difficulty; it is not automatically an appearance-induced mechanism.

Mean model misspecification is material. A mean `mu_i=E[W_i | single input]` does not guarantee that it is the mean under the richer pair conditioning; the proposed primary direct mean explicitly queries `V_ij`. Evaluate pair-conditioned calibration, residual dependence on single-observation quality, agreement, lag and reference candidate overlap. Where those checks fail, retain the name **centered wrong-evidence surrogate**, and report the bias rather than upgrade the claim. The primary target can be negative, unlike the full-current-posterior plug-in control. It still does not establish appearance-caused error covariance: uncertainty about the common true label and source/identity difficulty remain possible causes.

## 5. Minimal candidate network and downstream readout

Freeze the mean model before constructing pair targets. Obtain its predictions through source-group cross-fitting **inside fit only**, or split fit source groups into separate mean-fit and pair-fit subsets. Do not learn means on the same observations and then interpret an in-sample centered target. With only14 pilot fit groups, either approach can be unstable; first collect the support receipt, then decide whether fitting any pair model is justified. Cross-fit folds, ensemble/refit behavior at inference and their distribution difference are fixed in the training manifest.

The smallest primary candidate has a symmetric shared MLP over the declared `V_ij`: endpoint summaries plus `abs(u_i-u_j)`, `u_i*u_j`, time gap, track/cls-feature cosine, coarsened overlap/JSD, reference state and explicit argmax agreement. A candidate-token head additionally receives the aligned **reference** scores/ranks, using one common function for all `c`. It predicts `d_hat_ij,c` with squared or fixed-Huber loss against `t_dir_ij,c`; its sum predicts `t_trace`. No `K`-specific output weights, GT-dependent gather at inference, or absolute category embedding is allowed. Class-token sampling, if used, includes a documented tail estimator; silently training/evaluating only a top-k union changes the estimand. The first CPU implementation should simply retain all `K` candidates. A richer attention/calibration control can receive current `(p_i,c,p_j,c,pi_i,c,pi_j,c)`; any benefit of that richer model is not assigned to the coarsened conditional covariance objective.

Use the same shared trunk/parameter budget for the required controls below. Fixed-Huber improves robustness but changes the conditional-mean estimand; squared loss estimates the conditional mean. Select one before optimization and name the difference. Pair-edge subsampling uses observed time/track state and a fixed seed, never labels.

A concrete downstream experiment can keep all boxes/IDs fixed and use the pair score as a redundancy penalty:

```text
D_ij = max(0, sum_c d_hat_ij,c)
q_i = 1 - b_i
w_i = q_i / (1 + lambda * sum_{j in E(i)} D_ij)
track_score_c = sum_i w_i*log(max(p_i,c,eps)) / sum_i w_i
track_label = argmax_c track_score_c
```

Set a deterministic fallback when `sum_i w_i=0`; with no supported pair use the quality-only rule. Edges, `lambda`, clipping and causal/offline time availability are declared. This is an operational weighted log-pool, **not** covariance inversion or a claimed Bayes-optimal fusion. Ignoring negative pair scores is a declared conservative design choice, not evidence that negative dependence does not exist. The learned pair target must improve held-out calibration as well as final readout; an improvement only in the readout may simply be a useful weighting heuristic. No guarantee of improvement follows from these equations.

## 6. Necessary controls and attribution

All controls receive the same frozen inference stream, full vocabulary, target split, edge budget, optimization budget, train seeds and tuning budget. Match active trainable parameter count within a documented small tolerance; keep width/depth and feature access equal where possible. Never give a pair model extra raw features unavailable to controls and attribute the gap solely to its objective.

| Control | Purpose |
|---|---|
| B0 average classifier error; unweighted sumlog/meanlog | No-learned reference and confidence/length effects |
| B1 quality-only; shared temperature/candidate calibration | Single-view mean risk/calibration |
| Full-current-posterior plug-in residual `p*(pi-onehot(y))` | Strong label-uncertainty/calibration control; its same-identity diagonal products are always nonnegative |
| Same-capacity quality+agreement pair MLP | Tests whether agreement and quality alone explain hard/soft targets |
| Same-capacity generic attention/log-pooling | Tests whether any learned fusion works equally well |
| Same-capacity correlated-classifier-fusion/CFM-style control | Tests existing learned correlation/redundancy machinery; if an exact implementation is unavailable, mark this comparison missing, not completed |
| Same pair architecture with uncentered `G_ij` target | Tests whether explicit centering helps over raw common error mass |
| Pair target shuffled within source/quality/agreement/lag strata | Tests informative pairing beyond marginal composition; no cross-split shuffling |
| Remove appearance similarity; remove temporal gap; freeze pair score at0 | Attributes observed increments to pair inputs and excess head |

“CFM-style” is a comparison family, not permission to mislabel a new approximation as an official reproduction. Correlation/attention/GLS and redundant-video-frame handling already have prior art. Exceeding a generic quality rule does not establish novelty; exact literature/mechanism collision checks remain open.

Primary predictive readouts are out-of-group hard-risk calibration and held-out centered soft-target error/calibration, under fixed quality/agreement/lag strata. Secondary outcomes are fixed-geometry track classification/readout metrics with the same official protocol. Report paired differences and uncertainty by source group, plus GT trajectory support. Candidate improvement should persist over the same-capacity quality+agreement control and not be carried by one source group. If the soft-target advantage vanishes under this control, stop the shared-error attribution even if aggregate readout improves.

## 7. Data sufficiency and train/tune/test gates

The sealed plan has146 fit-category IDs and36 class-holdout IDs. This is **pseudo-novel to the new learner**, not detector-unseen classes. The pilot has2,138 eligible fit boxes over24 videos/14 source groups, before actual detection matching or error filtering; the19 annotated fit-video categories include3 excluded holdout categories, leaving at most16 positive fit-target classes. The127 GT tracks and92 tracks with>=6 anchors in the pilot fit metadata include holdout targets and are upper bounds. Pair counts are not independent sample counts.

Pilot tune has647 fit-category boxes/7 source groups. Pilot test has903 fit-category boxes and81 class-holdout boxes over only3 class-holdout classes/7 source groups. These counts cannot establish a36-class transfer result. The full metadata fit pool has39,903 fit-category boxes, but natural error rates, pair error support and detector-matched coverage are still unknown. No claim of enough natural-error training data follows from either pool.

Before any learned pair run, collect a **fit/tune-only support receipt** with: emitted/matched/ambiguous/unmatched counts; correct/error counts and soft-risk distributions; valid same-GT pairs; hard both-wrong and same-wrong counts; agreement/lag/quality strata; eligible target classes; source groups and trajectories contributing each event; pair/trajectory weight concentration; identity-contaminated versus observed-clean support. Keep held-out test labels closed. The static metadata file can report test availability, but test natural-error outcomes must not guide model choice or expansion.

Continue only if both mean quality and excess targets have usable variation across multiple independent source groups, tune strata have actual positive/negative support, the source-group uncertainty is informative at a predeclared practical tolerance, and the cross-fit mean calibration is stable. Choose that tolerance from the intended scientific effect before fitting; do not manufacture an arbitrary “enough pairs” threshold. All-positive/all-negative cells, one-source-group errors, negligible centered variation or unstable mean calibration are stop/expand-train-collection outcomes, not reasons to select a more complex model. The current14 fit groups and only2,138 eligible boxes justify a schema/signal check, not a data-sufficiency assertion.

Fit optimization uses only fit videos × fit target classes. Tune selection uses only tune videos × fit target classes. Class-holdout labels and test-video labels never choose loss, hyperparameters, calibration thresholds, edge bins or checkpoint. Once the method is frozen, evaluate the preregistered video test and pseudo-novel class-holdout cells and report overlap/support explicitly. Existing validation diagnostic labels/results are never training examples, moment estimates or tuning targets for this learner.

Train full1203 and existing validation296 remain distinct vocabulary regimes with different candidate competition and detection filtering. Class permutation equivariance does not solve this distribution shift. Keep vocabulary/mask metadata in model/cache receipts, normalize entropy by actual `K`, and require a separate frozen transfer-schema check before any claim across regimes.

**Remaining missing evidence:** a legal train posterior/feature cache; original-anchor class-blind matches; actual natural-error and independent-pair support; out-of-group mean calibration; incremental pair predictive value over quality+agreement; held-out readout benefit over matched attention/correlated-fusion controls; and a defensible mechanism/novelty comparison. This specification resolves what can be trained and falsified, not whether training is currently justified.

## 8. Exact CPU synthetic consistency receipt

`tmp_analysis/check_base_error_objective_synthetic.py` accesses no dataset, checkpoint, validation artifact, GPU or optimizer. It completed with `PASS_EXACT_SYNTHETIC_CONSISTENCY_ONLY`; receipt: `tmp_analysis/base_error_objective_synthetic_20261023.json`.

- Identical marginal risks `b_i=b_j=.5` give the same product `.25`, while independent/positively coupled/negatively coupled errors have covariance `0 / +.25 / -.25`.
- Equal both-wrong probability `.25` can yield same-wrong probability `.25` or0 depending on wrong direction. Off-diagonal wrong-class covariance can be `.25` while diagonal same-wrong excess is0.
- An easy/hard quality mixture creates pooled centered excess `.16` even though each quality group is independent.
- Enumerating27 `(true class, argmax_i, argmax_j)` combinations verifies the hard identity constraints.
- Exact soft enumeration gives centered trace `.0822` equal to its correct expectation formula; on one pair, the correct centered target `.0958` differs from the shortcut `.1162`. Joint class permutation preserves the target.
- Enumerating243 combinations of common true label, endpoint posterior and calibrated label posterior confirms that the full-current-posterior plug-in residual products are always nonnegative. Direct coarsened mean regression on independent/positively coupled/negatively coupled soft wrong masses yields trace `0 / +.04 / -.04`, avoiding that forced-sign degeneracy.

These checks establish algebra and known failure cases only. They provide no empirical natural-error support, training result, performance result or novelty evidence.
