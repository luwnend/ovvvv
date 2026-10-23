# Classification oracle contrast: initial design

The identity-support table describes whether the fixed GT-free edges contain
eligible manual-anchor endpoints from both same and different official GT
identities. It cannot show that removing contamination improves classification.
The following is the initial FIT-only classification diagnostic design within
the existing authorized research. Implementation and execution status follow
the new base_fit_identity_purification_headroom_protocol_20261023 protocol and
its independent review receipts. No code, classification join, model fit,
posterior load or reference modification is implemented by this initial note.

## Scope and frozen denominator

Keep the existing source roles, selected edges, endpoint exclusions and original
all-emitted reference index fixed. Use only FIT eligible exact manual-anchor
endpoints. Keep mean-role and pair-role reports separate. Report the original
eligible endpoint denominator, reference-label coverage and every unsupported
case; an oracle-covered subset must not silently replace the full denominator.
No tune, test or class-holdout correctness is opened.

The current support identities cover only eligible manual-anchor observations.
An unknown reference is not a different-identity reference. A claimed oracle
with identities for every emitted reference is unavailable from this table and
must not be fabricated from interpolation, predicted-track purity or matching
failure.

## Observable oracle reference contrast

For a fixed endpoint and its existing all-emitted, endpoint-excluded reference
pool, compare three explicitly separate diagnostic reference snapshots:

1. The original all-emitted pool (the only deployable/readout reference index).
2. All references with independently eligible manual-anchor identities, retaining
   both same and different identities. This measures the observed-label coverage
   change, which is a confounder of comparing a sparse oracle directly with the
   original complete pool.
3. Only those independently eligible manual-anchor references with the same
   official GT identity as the endpoint. This is an oracle diagnostic using GT
   identity; it is never a candidate readout or a replacement cached index.

Unknown references are absent from snapshots 2 and 3 by observed-label coverage,
not labelled as contamination. Report same-GT, different-GT and unknown reference
counts separately. A reference identity is scoped by (video_id, gt_track_id).
All snapshots exclude the original endpoints. No whole-track purity condition
is allowed. Report singleton and empty oracle pools explicitly.

Use a single, declared deterministic classification rule with no fitted
parameters or label-dependent selection: for example, foreground-normalize each
reference posterior, average it uniformly, then take the complete-vocabulary
argmax. If endpoint/reference fusion is also examined, fix its coefficient and
normalization in advance and report it as another declared diagnostic. Preserve
the actual emitted det_labels+1 baseline separately; posterior argmax is a
different baseline. No choice of rule, coefficient, reference count or favorable
subset may be selected using observed classification outcomes.

For the primary oracle contrast, compare snapshot 3 with snapshot 2 on exactly
the same oracle-covered endpoints, role and source groups. Report snapshot 1 on
that same denominator and separately on the complete eligible denominator.
A deterministic hash-based, count-matched sample from snapshot 2 can be an
additional predeclared reference-count control when it is feasible. It is a
diagnostic snapshot, not a change to the original references. Retain its seed
and membership hashes; never choose samples according to class correctness.

## Reporting and interpretation

For the FIT classification outcomes covered by the new protocol, report paired
correct-to-wrong and wrong-to-correct endpoint counts, source -> official GT
trajectory -> unique endpoint weighted accuracy/error, and source concentration.
Repeated edge contexts must not multiply the endpoint denominator: declare
unique endpoint weighting and distribute its weight equally over its retained
contexts. Source-bootstrap intervals are conditional on these frozen snapshots;
they do not supply training uncertainty or independent-pair evidence.

Call the result an observed-label oracle classification contrast. Sparse labels,
reference-count changes and the particular pooling rule prevent it from being a
mathematical upper bound on all identity-aware classification methods. A gain
would show available improvement under privileged identity information on the
reported FIT subset. It would not establish a GT-free separator, deployable
efficacy, novelty, cross-class generalization or readiness to train. A small or
negative contrast would restrict this specific diagnostic and remain
inconclusive outside its reference-label coverage and fixed classification rule.

Store all oracle memberships, predictions and metrics in a new diagnostic-only
directory under the new protocol. No oracle information may enter
the original reference cache, actual readout, supervision masks, learned
features, hyperparameter choice or reported deployable result.
