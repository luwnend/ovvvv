# Frozen pair candidate: conditioning and identifiability audit — 2026-10-23

**Decision: do not start real pair fitting as evidence for a new dependence mechanism.** The frozen protocol contains an estimand-level conditioning mismatch. This finding does not alter the frozen specification or invalidate the separately authorized full-support/strong-mean study. It also does not imply that every relation network is useless. No full tune outcomes, test correctness, or class-holdout correctness were consulted for this audit.

Protected specification: `full_mean_pair_gate_protocol_20261023.json`, SHA256 `fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718`. The new exact script and receipt are `check_pair_conditioning_mismatch_20261023.py` and `pair_conditioning_mismatch_synthetic_20261023.json`. They use rational finite distributions and no dataset. The protected SHA is unchanged before/after.

## Round 1 — distinguish the estimand from its fitted approximation

**Declared conditioning.** Let `X=(p_i,p_j,A)` denote current observed endpoint posteriors and the frozen scalar context. For shared true class Y:

```text
W_i,c = p_i,c 1[c != Y]
G = W_i dot W_j = p_i dot p_j - p_i,Y p_j,Y.
mu_i^y(A) = E[W_i | A,Y=y]
d(A,y) = E[(W_i-mu_i^y) dot (W_j-mu_j^y) | A,Y=y]
h_y(A) = mu_i^y dot mu_j^y + d(A,y) = E[G | A,Y=y].
```

Even if all means, moments and label posteriors are exact, the frozen deployment expression is generally:

```text
G_cond(X) = sum_y P(Y=y | X) h_y(A)
         != E[G | X]
          = sum_y P(Y=y | X) [p_i dot p_j - p_i,y p_j,y].
```

The two sigma-fields `(A,Y)` and `X` are not nested. There is no applicable tower-rule identity. Equality needs an additional conditional-mean sufficiency condition `E[G|X,Y]=E[G|A,Y]`; the frozen exclusion of current candidate endpoint probabilities does not establish it. Integrating `h_y(A)` with `P(Y|A)` correctly gives `E[G|A]`, a different prediction task. Integrating with richer `P(Y|X)` changes the question without updating the inner conditional moment.

**Check 1A, strict same-A counterexample.** Two equally probable states have `p_i=p_j=(.6,.3,.1)` or `(.3,.6,.1)`, with Y always coordinate0. All three branches equal p, the reference is uniform, and raw score/box/background/appearance cosines/lags/reference count are fixed. Frozen entropy/top1/margin quality summaries are identical; both endpoints agree; `p_i dot p_j=.46`, cosine1, JS0. Frozen quality risks receive identical quality vectors. Candidate reference probabilities and ranks are identical. Thus the frozen Q/A context and y-query cannot distinguish the states.

| Exact quantity | State 0 | State 1 |
|---|---:|---:|
| G | .10 | .37 |
| `mu_i^0=mu_j^0` | `(0,.45,.1)` | `(0,.45,.1)` |
| `d(A,0)` | .0225 | .0225 |
| `G_cond`, even with true posterior `(1,0,0)` | .235 | .235 |
| Same-conditioning exact posterior formula | .10 | .37 |

The deployment excess MSE is exactly `.018225`; it is not caused by mean fitting, posterior estimation, train leakage, sampling noise, or implementation. The script also embeds this example in K1203 with all 1203 probabilities strictly positive: tail epsilon `1e-8`, rescale the first three coordinates by `.999988`. The excess MSE is `.018225 * .999988^4 > 0`.

This ideal-mean counterexample intentionally grants an oracle mean outside the restricted six-basis family. With a uniform reference, the frozen JSON's six bases (`q_ref_fused/text/image`, normalized `q_ref_fused²`, normalized `sqrt(q_ref_fused)`, uniform) are all identical. That family cannot represent `(0,.45,.1)` after masking y; its approximation error is an additional issue. The ideal counterexample shows the conditioning flaw survives even if that family problem were solved. It does not claim the current runner actually outputs the oracle vector.

**Check 1B, genuinely uncertain common Y.** Keep the two p states and equal state weights; set `P(Y=0|state0)=.8`, `P(Y=0|state1)=.4`, with the remaining mass on Y1. The four atoms have weights `.4,.1,.2,.3` for `(state0,Y0),(state0,Y1),(state1,Y0),(state1,Y1)`.

```text
P(Y0|A)=.6; mu^0=(0,.4,.1); d(A,0)=.02; h0=.19.
P(Y1|A)=.4; mu^1=(.375,0,.1); d(A,1)=.016875; h1=.1675.
```

| Quantity | State 0 | State 1 |
|---|---:|---:|
| Exact `E[G|X]` | .154 | .208 |
| Mixed-conditioning `sum_y pi(y|X) h_y(A)` | .1855 | .1765 |
| Bias | +.0315 | −.0315 |

The excess MSE is exactly `.00099225`, despite exact coarse means, residual moments and full label posterior.

## Round 2 — independent arithmetic and invariance checks

**Check 2A, risk decomposition and full1203.** The script checks the squared-loss projection identity `MSE(G_cond)-MSE(E[G|X])=E[(G_cond-E[G|X])²]` exactly on the finite atoms. The strictly positive1203 construction above preserves the same-A counterexample, with the specified scale-to-the-fourth excess risk. This rules out zero-padding and floating-point artifacts.

**Check 2B, class/endpoint invariance.** Twelve joint class-permutation cases are checked: six for the same-A unknown-Y example and six on a separate three-state example with nonidentical endpoints and all label probabilities positive. Endpoint exchange preserves G, the oracle formula, mixed predictor and risks; conditional endpoint means exchange roles. Thus the endpoint-swap check is substantive, not only the trivial swap of identical p vectors. Every oracle formula is compared with direct atom enumeration.

## Round 3 — specify the mechanism stop boundary

**Check 3A, coarse and rich conditioning are different tasks.** In the uncertain-Y example, coarse integration `.6*.19+.4*.1675=.181` exactly recovers `E[G|A]`. Substituting the richer correct posterior yields `.1855/.1765`, which does not recover `.154/.208=E[G|X]`. The proper tower rule is valid for the coarse task; the failure is the unmatched conditioning in the deployment formula, not a failure of probability integration generally.

**Check 3B, same-conditioning optimum, finite controls and useful coarse relations.** For current observed p and any context X containing them, the conditional minimum-MSE target is exactly:

```text
G_star(X) = p_i dot p_j - sum_y pi_star(y|X) p_i,y p_j,y,
pi_star(y|X) = P(Y=y | X).
```

Conditioning each inner moment on the same X makes `mu_i^y=p_i*1[c!=y]` and `d(X,y)=0`, since W is then deterministic given X,Y. This proves sufficiency of a correct common-label posterior for this particular G target. It does not prove that a finite C129 loglinear calibrator attains that posterior. A relation regressor can outperform C129 because the latter has limited features, interaction form, CE optimization/regularization, or finite data. Same parameter count/search budget does not match function classes, and a win cannot identify an extra stochastic covariance of already observed p.

A scalar prediction for G may also need only the posterior-weighted product moment, rather than the full Y distribution. Lower G-MSE than a CE-trained semantic classifier therefore need not mean a better calibrated full semantic posterior. Report the precise target and separate CE/calibration metrics.

Useful relation prediction remains possible: with Y0 fixed, let both endpoint marginals be equally likely `(.8,.1,.1)` or `(.6,.3,.1)`. Ideal coarse mean is `(0,.2,.1)`. Independent endpoints give centered trace0; perfectly coupled endpoints give trace `.01`. An observed coarse relation indicator that distinguishes these populations can usefully predict the moment. This is a valid coarse forecasting/representation problem. It does not supply extra information beyond the correctly conditioned common-label posterior for G after both full p vectors have been observed. The exact script verifies both values.

The unresolved practical question is whether relational context improves a specified finite predictor under distribution shift, not whether the algebra requires a new intrinsic residual covariance. No finite control establishes a theorem against all attention/relation networks.

## Additional literature checks — not part of the six mechanism checks

The primary full text of *Hierarchical Aggregation of Semantic Uncertainty in 3D Scene Graphs*, arXiv2609.22351v1, has now been read; earlier abstract-only discussion was not a full-text review. Its Eq15 is `max_i q_i`, justified by repeated correlated semantic errors and the fully correlated dependence limit. This covers the broad story “shared model/vocabulary causes repeated errors, so independent aggregation becomes overconfident.” It is not a learned residual covariance, inverse-covariance fusion, conditional-Y pair learner, attention baseline or C129. Exact method collision is not established. See the separate full-text audit and source receipt for verified scope and weaknesses. Positive-association/noisy-OR and constant-Brier arithmetic are additional paper checks; neither is counted toward our three-round mechanism audit.

## Classification of findings

| Finding | Type | Consequence |
|---|---|---|
| `E[G|A,Y]` integrated with `P(Y|p_i,p_j,A)` | Conditioning/estimand flaw | Real pair fit cannot establish the advertised operational moment identity |
| Six reference bases do not span arbitrary conditional means | Model-family approximation limit | Finite projection checks cannot certify a full vector mean |
| Full conditional G determined by p and Y | Algebraic property of this target | Strong semantic posterior is the correct matched optimum |
| Candidate may beat finite C129 | Possible predictive result | Does not identify intrinsic dependence or defeat all semantic calibration |
| Independent repeated-error aggregation already challenged in 3DSGs | Prior-work narrative overlap | Broad motivation alone is not novelty |
| No real pair implementation audited/executed here | Execution boundary | This audit identifies no demonstrated pair-code bug or real-data performance |

## Recommended action and a concrete research boundary

Complete the authorized full-support and strong-mean audit. Preserve this frozen candidate as a rejected or HOLD specification; do not rewrite it after outcomes or start a costly pair fit to rescue its mechanism claim. Correcting only the equation by expanding conditioning collapses this G task to supervised common-label readout. Keeping it coarse requires naming and validating a different forecasting target. Neither is a new covariance mechanism by itself.

My current recommendation is **STOP this current-G residual-covariance novelty route**. A new experiment should start only after choosing a scientific problem that the exact posterior identity does not already solve. One concrete next feasibility audit would target **future observations across an unobserved identity switch/contaminated track memory**, where the unknowns are whether observations belong to the same object and how new evidence should change a semantic belief. Predeclare an intervention that changes duplicate exposure while holding distinct information fixed, and a second intervention that introduces genuinely independent informative views. A candidate must suppress the former without suppressing the latter; pure max does not accumulate independent evidence, and unconstrained attention can encode either behavior. Require an observable/identifiable source of contamination or acquisition structure, a specified deployment filtration, held-out-source/class transfer, and matched attention/log-pool/shared-label controls before claiming a mechanism.

This is a proposal for a separate literature/observability audit, not an approved novel method or an instruction to add features to the frozen candidate. If the available cache cannot distinguish duplication, informative novelty and identity contamination without correctness labels, stop that direction as unidentifiable too. Renaming semantic calibration or adding a relation network does not meet this boundary.

## Completion receipt

- Three rounds, two checks each; exact rational script PASS.
- Fulltext HTML read plus PDF pages5–7 rendered and visually checked; source and coverage audit saved separately.
- No frozen protocol edits; before/after SHA matches.
- No true pair fitting, test/heldout correctness reads, full tune-based design or GPU work.
