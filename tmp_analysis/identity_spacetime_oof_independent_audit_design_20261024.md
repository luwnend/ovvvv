# Identity/spacetime OOF independent preflight and output audit

Status: DESIGN_ONLY_PENDING_NEW_DATA_TRAINER_RUNNER_IMPLEMENTATIONS.
This note does not extend the pure-module PASS, authorize training, or inspect
real cached features/manual identities. Old pinned files remain unchanged.

## Independent preflight matrix

1. Construct temporary shared manual JSONL with allowed train sources A/B,
   held-out C/D and tune rows. Use valid JSON containing forbidden objects,
   strings and oversized numbers in held-out identity fields and every
   category/error/risk/posterior field. Monkeypatch actual JSON/numeric decoder
   entry points to raise if a forbidden value reaches them. Check the ordered
   split -> source -> eligible identity projection, including the immutable
   GT-free observation index. The exact authority SHA is updated for each
   mutation, so passing cannot merely mean rejecting a tampered file.
2. For one model f, mutate only held-out identity values and forbidden/tune
   payloads. Require identical canonical allowed-train projection, example
   ordering, hierarchy weights, normalizer arrays, final numerical checkpoint
   bytes/digests and retained score shards. Require global authority SHA to
   change. Keep whole-file authority provenance outside the functional numeric
   checkpoint/prediction payload; otherwise this invariant is impossible even
   without leakage. Other models legitimately training on those sources need
   not remain unchanged. A permitted train-label change must affect the legal
   projection and a nondegenerate synthetic loss/gradient: the mutation check
   must not pass because labels are ignored altogether.
3. Metadata-only folds: independently sort SHA256(seed + NUL + source) in each
   (dataset block, original role) stratum; check all 341 sources exactly once,
   all videos/history/observations of a source remain together, and zero
   support sources are retained. Original roles are metadata only, never model
   numeric input. Synthetic folded-role/source permutations must not change
   fold assignments except as implied by the fixed source metadata.
4. Temporary frames include retained unknown IDs, duplicate released IDs,
   empty frames, a two-frame all-empty video, irregular original times,
   no-positive known subsets, zero features and zero-support sources. Compare
   all exact-lag {1,4} QxM receipts and row maps with an independent ring/history
   oracle. No nearest-nonempty fallback or CV/score/raw-error query selection.
5. Build Gperm donors on the complete GT-free retained reference array before
   known P/N projection. Include a known row whose donor is an unknown row and
   a three-column cyclic group for which projecting first changes the donor.
   Check own time/history-gap/masks, group boundaries, singleton behavior and
   separate mapping-moved/value-changed coverage. Reuse one train-fold original
   geometry normalizer for all A/G arms; held-out geometry cannot enter fitting.
6. Verify exact equal-source/trajectory/observation/available-lag global
   weights, all P x N means, w/(2P),w/(2N) edge marginals and dimensionwise valid
   global renormalization. Perturb held-out geometry to force a different mean
   if it leaks. Verify zero-valid slots and std floor. The existing independent
   pure-module audit supplies hand-math evidence, but runner population and
   wiring must be tested anew.
7. Synthetic adapter/head use the actual fixed schedule, seeded per-epoch
   query-lag permutation with exact once-per-epoch coverage, final short batch
   actual B and N/B scaling, fixed epoch counts, no validation/early stop, and
   final checkpoint selection. Assert parameter tensors change from meaningful
   synthetic supervised gradients; zero input stays zero after changed biases.
   Freeze once and clear stale adapter gradients before all G arms; no geometry
   gradient or optimizer update reaches the shared adapter.
8. A final OOF seal contains exactly 5 folds x 3 fixed seeds, all required arms,
   all expected held-out video/lag score shards and complete row mappings,
   checkpoint/normalizer/training-source provenance and file SHA values. Missing
   fold/seed/arm, duplicate source, omitted unknown candidate, changed score,
   changed map/checkpoint/receipt or missing file must prevent evaluation before
   its eligible-identity decoder is called. Test the evaluation entry point with
   a decoder sentinel, not just a helper returning a Boolean seal check.

## Complete real-output independent verifier design

The verifier first checks protocol, implementations, independent preflight,
actual process-exit receipt and the complete global seal, before opening any
eligible identity payload. It never imports data/trainer/runner/module forward
functions for score reconstruction. It may use the previously independently
audited safe lexical scanner, with its exact SHA pinned.

GT-free population metadata independently establishes every expected fold,
source, video, frame, exact lag, prefix/empty receipt, immutable retained row and
candidate offset. It checks every score shard as bounded per-frame-pair blocks
using a ring of five feature snapshots. Parent raw score column 0 must match
every A0 pure-cosine candidate exactly. If the runner uses Torch for learned
scores, it should preserve the original NumPy raw column or read its sealed
parent value; NumPy/Torch BLAS roundoff cannot be advertised as bitwise parity.

For each model checkpoint the auditor loads only declared tensor arrays with
name, shape, dtype and hash checks. It independently computes linear layers,
GELU via 0.5*x*(1+erf(x/sqrt(2))), unit normalization with the declared floor,
original-zero output mask, adapted cosine, and the residual head s/0.1+MLP19.
It independently rebuilds relative geometry, original-frame history, train-fold
normalization, masks and all fixed interventions including full-retained donors.
Every learned output block in every arm/seed is compared at a predeclared
float64 tolerance; maximum observed errors are reported. Neither random score
sampling nor only known-candidate checking is sufficient.

After this global seal boundary, independently reconstruct all legal eligible
FIT identities by split-first safe projection and immutable row join. Rebuild
each model f training projection from allowed sources only, prove held-out
sources absent, recompute hierarchy weights and the original-geometry
normalizer once per fold (and compare the three seed copies). Bind example
order/digests, exact fixed epoch/batch/schedule receipts, optimizer configuration,
gradient/parameter movement and final-epoch checkpoint lineage. Do not claim
the logs alone prove every numerical optimizer update; full synthetic training
replay and independent score reconstruction provide distinct evidence. A
real 15-model optimizer replay is unnecessary unless those checks expose a
discrepancy.

Finally reconstruct query ranking and all paired reports independently from
the sealed scores: complete/known/positive/negative/unknown counts, lowest
immutable reference-row top1 tie, fixed common-CV population, all nonempty and
p+n subsets, source/trajectory/query macros, error concentration, same-query
cross-lag intersection, pure A1-A0, G1-G0/Gmask/Gperm, 2x2 interaction and
predeclared source-deletion/seed robustness summaries. Preserve all zero
sources/folds. Evaluation may read the full eligible FIT identity set only after
all 15 model predictions are globally sealed; no early fold metric feedback.

The final receipt must distinguish producer completion, independent complete
score/lineage/statistics audit, and ordinary exploratory control interpretation.
It must not imply backbone training, tracker/assignment evaluation, official
test/generalization, sufficient power, dynamics truth or novelty.
