# Primary full-text audit: Hierarchical Aggregation of Semantic Uncertainty in 3D Scene Graphs

**Scope correction:** earlier discussion used the abstract only. On 2026-10-23, the primary arXiv HTML was read through all methods, experiments, conclusion and 28 references; the 8-page PDF was downloaded, and pages5–7 were rendered and visually inspected for Eq9–15 and TablesI–III. No appendix was found. This note supports a full-text review now; it does not retroactively turn the earlier abstract check into one.

Paper: Carlos Cueto Zumaya, Iacopo Catalano, Wallace Moreira Bessa, Julio A. Placed, **arXiv2609.22351v1**, 17September2026. Primary links: [abstract](https://arxiv.org/abs/2609.22351), [HTML full text](https://arxiv.org/html/2609.22351), [PDF](https://arxiv.org/pdf/2609.22351). The linked small source receipt contains fetch failures/successes, metadata, exact SHA256s and local-only paths. Raw HTML/PDF/text/rendered images remain local and are not synchronization artifacts.

## 1. What it actually does

The task is to attach reliability to **stored object nodes** and propagate it into **room-class presence assertions** in an open-vocabulary 3D scene graph. It reads an existing HOV-SG graph on HM3DSem. It does not train a MOT pair relation, estimate a correlation matrix, or invert a covariance. It uses detector score, fused visual embedding, class text embeddings and containment edges (SecIII-B, Eq2).

| Signal | Exact construction | Qualification in the primary text |
|---|---|---|
| Detection | Stored detector score `s_i` | Real-object probability estimate; calibration not guaranteed (SecIII-C1) |
| Semantic margin | Sigmoid of CLIP-scaled margin against nearest text-distinct class; exclude near synonyms using text-text cosine threshold | Binary comparison, upper bound on full-vocabulary top probability; not a calibrated full-vocabulary correctness probability (Eq3–4) |
| Coherence | Leave-one-out prototype among other objects with same predicted label; analogous margin | Same labeling error as semantic score; combine by **min**, not product; semantic fallback without peers (Eq5–6) |
| Vocabulary membership | Mean exponentiated similarity over vocabulary vs mined lexical negatives, each normalized by set size | Bayesian reading assumes equal group priors, uniform within-group class priors and exponentiated-similarity likelihood (Eq7) |
| Cross-view consistency | Recover average pair cosine from norm of mean unit embeddings and n; map by `(1+cosine)/2` | Not likelihood-derived; single-view score1 is explicitly optimistic (Eq8) |

The authors count semantic/coherence as one combined signal, hence “four signals.” Detector calibration, binary margin probability and view-agreement heuristic should not be silently relabeled as empirically calibrated event probabilities.

## 2. Object probability: exact chain rule, approximate factors

SecIII-D1, PDFp5, Eq9 factorizes the joint event that detection is real, view fusion represents one object, object belongs to vocabulary and assigned semantic label is correct. Eq11 substitutes `q_i≈s_i P_view P_mem min(P_sem,P_coh)`. The authors explicitly say the first three factors were computed without conditioning on preceding events, so substituting them for the required conditionals is an approximation. The displayed chain rule does not itself validate those substitutions.

Eq12 uses a positive-association intersection inequality for true event marginals; the paper motivates association by shared observation degradation. That qualitative motivation is not a measurement or proof of association on the evaluated population. The text also correctly says estimated factors mean Eq11 is a **conservative estimate rather than a guaranteed bound**. The same distinction applies to any attempt to import this argument into our data: shared backbone/vocabulary does not establish the magnitude or sign of every conditional covariance.

## 3. “Fully correlated” is max-dependence aggregation

SecIII-D2, PDFpp5–6, Eq13 also defines query-conditioned object scores. Room aggregation nevertheless uses only stored objects **hard labeled c**, `O(r,c)`, not every object's soft evidence for c. It sets:

```text
Eq14: max_i q_i <= P(room contains c | map) <= 1-prod_i(1-q_i).
Eq15: b(r,c)=max_{i in O(r,c)} q_i; empty set -> 0.
```

The paragraph introducing Eq14 says “regardless of correlation,” but the next paragraph explicitly limits the noisy-OR upper bound to **positively associated correctness events**. In general the upper bound is `min(1,sum_i q_i)`. For example two mutually exclusive events with marginals.4 have union.8, exceeding noisy-OR.64. Max is always a lower bound for the selected-object union if its q marginals are accurate; nested/comonotone events attain it. With unequal marginals, literal binary Pearson correlation1 need not be attainable, so “max dependence/comonotone limit” is more precise than “full covariance fusion.”

There is a second scope issue: the exact selected-object union omits correctly present objects labeled differently, or objects missing from reconstruction. Its bracket is not an exact bracket for complete physical room existence unless the missing-event contribution is handled. The paper acknowledges this restriction and the relevant omitted terms; it must remain visible in any comparison.

Eq15 is duplicate-idempotent: repeated objects of no greater reliability add no evidence. It does not estimate partial correlation or gain confidence from independent repeated support. This paper directly covers the broad narrative “one shared model/vocabulary repeats the same semantic error, and independent aggregation falsely rewards repetition.”

## 4. Experimental evidence and its limits

SecIV-A uses **six HM3DSem scenes**, one reconstructed HOV-SG graph per scene, spatial-proximity association of predicted objects to annotated instances and room presence from annotated room extents. There are **500 instantiated scene-class queries**, 72–100 classes per scene. Room evaluation covers **object-supported room-class entries**, not the full room×vocabulary space. Pipeline misses without any stored entry have belief0 in every configuration. SecIII-A says construction recall is reported separately, but this audit found no numerical recall table in the text/PDF tables.

PDFp6, TableI object search macro:

| Method | OS@1 | OS@3 | OS@5 | OMRR@5 |
|---|---:|---:|---:|---:|
| HOV-SG similarity | 24.07 | 29.04 | 29.81 | .26 |
| Query-conditioned belief | 26.21 | 30.29 | 30.84 | .28 |

OS@1 improves four scenes and decreases two. The authors identify much of the gain as reordering already retrieved objects. The evidence supports a small retrieval/ranking gain in this protocol, not MOT association or downstream navigation gains.

PDFp7, TableII room results:

| Method | Macro Brier [95% interval] | Macro AUROC | Frequent/Common/Rare Brier |
|---|---:|---:|---|
| As built, constant1 | .80 [.76,.83] | .50 | .17/.74/.91 |
| Noisy-OR | .47 [.44,.50] | .54 | .23/.44/.49 |
| Max belief | .33 [.30,.34] | .61 | .25/.32/.33 |

TableIII shows max belief below noisy-OR in all six scenes. Frequent classes get worse than constant1 (.25 vs.17). The 10,000 bootstrap draws resample only six independent scenes; more draws do not create more independent scenes.

The macro Brier weighting is **entries within room → rooms within scene → scenes**, not simply pooled entries. Under this identical normalized weighting, `Brier(constant1)=1-weighted_mean(Y)`. Thus the published rounded .80 implies weighted prevalence approximately.20. The same-weight **constant0 Brier is approximately.20**; constant.2 gives `(.2)^2+(1-2*.2)*.20≈.16`. These arithmetic diagnostics follow from the reported metric and preserve the exact aggregation; they are not newly measured model results, and oracle empirical prevalence would use evaluation labels. Constant0 needs no such fitting and would have AUROC.50. The paper's .33 therefore supports improvement over certainty/noisy-OR and some discrimination (AUROC.61), but not superiority to simple probability baselines or adequate absolute calibration.

Room baselines are only certainty and noisy-OR on the same graph. No empirical-prevalence/constant0, trained shared calibration, learned log-pool, attention or learned correlation competitor is reported. No per-signal ablation was found in the full text. Threshold tau, negative-mining size/cutoff and exact model logit-scale settings were not established from the text; no code/project URL was established. These are reproducibility/control gaps, not proof the reported numbers are false.

## 5. Collision audit against our frozen proposal

| Topic | Primary paper | Our frozen specification | Judgment |
|---|---|---|---|
| Correlated semantic errors | Shared model/vocabulary/trajectory causes repeated object errors | Shared semantic label/model plus pair residuals | Broad motivation already covered |
| Fusion | Max of per-object correctness estimates; fixed dependence limit | Learn label-conditioned residual trace, then integrate Y | Not the same method |
| Conditional moments | No supervised `E[W|A,Y]` or residual-moment fitting | Explicit conditional means and d(A,y) | Exact collision not established; our conditioning flaw is separate |
| Calibration | Training-free signal products, no correctness fit | Mean-role and pair-role supervised readout/calibration | Different setting; finite supervised gains are not novel by themselves |
| Attention/log-pool | Neither implemented as a control | Still deferred | This paper cannot substitute for those comparisons |
| C129 | No analogous shared label CE model | Finite 129-parameter calibrated semantic competitor | Must separate finite-control outcome from Bayes identifiability |
| Evaluation | Static 3DSG retrieval and room-entry reliability | OV-MOT readout/pair diagnostics | Task transfer is not a novelty proof |

The original broad novelty claim about learning to avoid repeated correlated semantic evidence is unsafe. This single paper does **not** establish total method coverage of the exact candidate, and its own heuristic/probability limitations do not repair our candidate's conditioning mismatch. See `pair_conditioning_identifiability_audit_20261023.md` for the exact proof, not a literature-based dismissal.

## 6. Three rounds, two checks and remaining leads

1. **Definitions:** checked stored interface/signal construction; checked exact chain rule versus approximate probability substitution (SecIII-B/C/D1; PDFp5).
2. **Dependence:** checked Eq14 assumptions and selected-object event; checked Eq15=max and its distinction from learned covariance (PDFp6 plus rational counterexample).
3. **Evidence:** transcribed and visually checked TablesI–III/protocol scope; checked strongest novelty claims against actual method and missing calibrated/attention/C129 controls (PDFpp6–7).

The exact synthetic companion additionally verifies the association counterexample and constant-Brier arithmetic. No new real pair model or dataset analysis was run for this review.

References [2], [8], [16] are **leads cited by this fully read paper; their full texts have not been reviewed here**: Marques etal., *On the overconfidence problem in semantic 3D mapping*, ICRA2024, pp11095–11102; Zhang etal., *Remember with confidence*, arXiv2606.08277; Zhang/Carlone, *FUSE*, arXiv2606.14728. Reference[25] is Esary/Proschan/Walkup1967 on association. Do not count these leads as additional completed primary full-text reviews.
