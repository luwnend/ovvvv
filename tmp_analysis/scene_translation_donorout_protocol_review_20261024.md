# Independent pre-outcome review: FIT donor-out scene translation

Date: 2026-10-24. Reviewer: evaluator_audit. This review uses the frozen collector source and sealed FIT allocation/frame metadata, plus synthetic module inspection. It does not open the new OOF readout, identity/class arrays, GT, query SQLite, or intermediate experiment metrics. Parent owns the producer, protocol and real launch; this file changes none of the existing OOF pins.

## Decision and estimand

The design is suitable for a small, complete, **conditional GT-free position-predictability diagnostic**, subject to the implementation/output audit below. It cannot establish matching accuracy, true object identity, camera motion, physical independence of donors, or tracking improvement. The full-pool predicted MNN edge is the target, not a known positive or negative. Selection by its own raw appearance makes this a selected population of predicted edges.

The population is all 341 sealed FIT sources, 403 cached videos, both lag 1 and lag 4, and every allocated frame-pair receipt. Lag denotes the frozen cached-frame ordinal difference, not a promise of the same physical elapsed time across videos. Prefix pairs, missing inputs, invalid geometry and insufficient support remain explicit missing cases. No support or margin threshold may be selected after outcomes. The allowed fitted rule is at least three rematched anchors and a componentwise median in original image pixels.

The primary prediction gate is the **strict arm** source-balanced mean gain being strictly greater than zero at each lag. This is only a minimum descriptive GT-free predictability gate. It is not a significance test or a sufficient mechanism, generalization, or causal claim. Row-only and in-sample controls cannot rescue a failed strict gate.

## Frozen target selection and exclusion interventions

For each nonprefix frame pair, normalize every finite nonzero raw track-head row independently with the frozen peak-rescale then L2 rule. Build exactly one full float64 normalized dot-product matrix. On this canonical matrix, define the ordered target set H as exact, unique, strictly positive mutual row/column maxima. Ties use exact equality; a tolerance must not turn distinct top-1 candidates into ties or undo a tie. Zero raw vectors are ineligible. No detector confidence, class, track/released ID, GT, margin cut, residual filter or geometry-based matching is allowed.

Freeze H once. For every h=(qi,rj), separately apply these interventions **before fresh directed maxima and MNN are recomputed** on the remaining pool:

* `row`: remove exactly query row qi and reference row rj.
* `strict`: independently in each endpoint's own frame, remove the endpoint, every row with the same `roi_proposal_row`, and every row whose xyxy box has positive-area intersection with that endpoint box. This is their union, not an overlap-connected-component closure. Edge/corner touching has zero area and alone does not exclude a row.
* `allpool`: fit once to the original full-pool MNN set with at least three anchors. Evaluate each h with that same translation. This is an explicitly self-including control.

After an exclusion, matching starts from **all remaining eligible raw rows**, not from H with rows deleted. Rematching can create new reciprocal pairs; do not reject a target merely because full-pool H has fewer than four members. Only actual rematched anchor count determines fit availability. A full-pool target may have no valid fit; it must still be recorded, with reason.

Collector provenance confirms the retained `roi_proposal_row` is aligned with retained `track_feats` and boxes: `proposal_rows=flat//C`, followed by the same tracker keep mask. C comes from the stored category schema and is not a hardcoded class count. Class-dependent box regression means equal proposal rows are not guaranteed to overlap; positive overlap alone would not establish alias exclusion. The explicit alias union resolves this implementation issue without decoding class arrays. A proposal index is frame-local; never equate it across frames.

The alias union removes an observable duplicate-proposal route, not every physical duplicate or related object. Overlap also removes real nearby objects and may reduce support. Therefore arm-specific mean differences mix support and fitted-rule changes unless the same target intersection is used.

## Geometry, gain and the precise invariance claim

Let c be xyxy midpoint, w/h the finite positive box dimensions, and s_xy=sqrt(q_wh)*sqrt(r_wh), componentwise. Do not multiply widths first. Fit translation t as the componentwise median of donor c_q-c_r in original pixels. For each h, `static=sum(abs((cq-cr)/s))`, `corrected=sum(abs((cq-cr-t)/s))`, and `gain=static-corrected`. These are normalized L1 **sums**, not coordinate means. No clipping, post-hoc epsilon, target weighting, donor normalization before the median, or identity-warp fallback is allowed. Arithmetic exceeding finite representation must fail explicitly rather than report an available NaN/Inf fit.

Endpoint boxes legitimately define the target error and strict overlap mask. Endpoint appearance legitimately selected H. Consequently the estimator's mutation test is conditional: **given fixed target membership and fixed excluded row locators**, modifying excluded endpoint features/boxes must preserve remaining matches, translation and donor diagnostics (to the frozen numerical convention). Recomputing H or the strict mask after mutation is allowed to change the target/exclusion set. Do not claim unconditional endpoint independence.

Median/MAD/leave-one-anchor diagnostics measure agreement of predicted donors. A coherent moving foreground majority can yield low MAD and a stable translation. Under displacement d=camera+object, adding a common vector to camera and subtracting it from all object motions preserves observations. No GT-free geometry test here resolves that ambiguity.

## Aggregation and support alignment

For each arm and lag: compute gain for every available h; take the median across valid h within a frame pair; take the arithmetic mean across supported frame pairs within a source; take the equal mean across supported sources. A frame with zero valid h and a source with zero supported frames are missing, not zero. Sources and frames are never weighted by detection count or number of targets. Emit a row for **all 341 sources** at each lag, including zero-support sources and their reason/count fields.

The primary strict-arm gate uses its prespecified supported population. Also report a paired row/strict comparison: for each frame, intersect h with both fits available and recompute both medians on exactly that h set, then the same source hierarchy. Separately report the three-arm common-h intersection. Independent arm-specific medians on different targets are not a paired intervention comparison. Arm coverage/counts accompany each readout.

For cross-lag source comparisons, use the intersection of supported source identifiers and report it explicitly. Do not invent a same-h longitudinal intersection: lag 1 and lag 4 generally have different endpoint pairs. Source deletion/distribution summaries, frame/target micro averages, margins and diagnostics may be labeled descriptive, but cannot replace the frozen strict-arm two-lag gate.

## Canonical numerical contract and reference limitations

The cached full normalized float64 matrix is the protocol's numerical definition. Every exclusion selects a submatrix of that same matrix; the MNN extrema are recomputed exactly on that submatrix. Computing a newly shaped BLAS product after row deletion can change the last bits and hence exact ties, positivity or top-1 membership. Therefore the older sliced reference is a structural/numeric comparator on synthetic cases, not an assertion of universal bitwise equivalence.

An independent oracle must scan the canonical matrix without using the module's MNN reducer and compare target and donor **row-pair membership exactly**. Numeric fit/residual comparisons can use a stated tolerance after that discrete equality passes. The prior 2e-13 sliced-reference comparison must not hide a distinct-candidate membership disagreement. Pin C-contiguous float64 raw inputs and normalization layout: independent synthetic inspection found identical C/F-layout values can change full-pool top-1 membership for nearly equal candidates. Record NumPy/BLAS/interpreter and single-thread environment in the frozen launch. Repeated runs should use that same implementation environment; no result-dependent tie rule change is allowed.

## Data boundary, storage and auditability

The new cache reader must open only the fixed allowlist: retained `track_feats`, `det_bboxes` xyxy (never score), `roi_proposal_row`, proposal-count integrity if specified, immutable frame/video/image locators, and allowed schema/shape metadata. Reject malformed/missing required fields; do not silently fall back to row-only strict exclusion. Do not import the collector's `validate_payload`: it accesses forbidden class/posterior/track-ID arrays. NPZ byte hashes can be checked without decoding those arrays. Include an allowlist/poisoned-NPZ access test.

Write per-video compact numeric NPZ arrays and a SHA-bound schema. Keep a complete frame-pair JSON receipt including prefix/missing cases, offsets, full-pool MNN target set, per-h arm availability/reasons, target/exclusion/anchor locator hashes, and sufficient values to independently recompute errors and hierarchy. The independent auditor must reopen the allowed raw frame arrays and reconstruct **every h**, all masks, rematched anchors, errors and source summaries. Stored exclusion/anchor hashes alone cannot prove correct construction. Record every missing h; output only available rows would permit silent selection.

Avoid repeating raw vectors or full similarity matrices per target. A five-frame ring, bounded one-pair matrix, streaming per-video outputs and running source summaries suffice. Numeric arrays must distinguish missing values with explicit availability/reason codes and not cast a missing translation to zero. The protocol pins schema, code/source hashes, output layout, thread settings, runtime budget and completion checks before launch. Verify original pins/protected artifacts after completion; the producer's exit alone is not final audit success.

## Metadata-only workload bounds

The following counts were computed only from sealed source allocation, full collection plan and frame execution metadata, without opening raw features, boxes or outcomes:

| Quantity | lag 1 | lag 4 |
|---|---:|---:|
| Frame-pair receipts | 13,791 | 13,791 |
| Nonprefix receipts | 13,388 | 12,185 |
| Prefix receipts | 403 | 1,606 |
| Empty nonprefix by observation metadata | 0 | 0 |
| Sum Q*M | 29,096,237 | 26,255,927 |
| Sum min(Q,M), target-count upper bound | 564,223 | 506,219 |
| Sum min(Q,M)*Q*M, one-arm scan bound | 1,456,001,989 | 1,298,300,789 |
| Largest single-pair such scan bound | 456,533 | 433,125 |

Max Q and M are 77. There are at most 1,070,442 held-out predicted targets across lags, not 55 million edge targets. The two exclusion arms together have a loose matrix-scan bound of 5,508,605,556 entries; this excludes matching reductions, medians/LOO, storage and hashing. Full-pool MNN count is at most min(Q,M), and usually smaller; the bounds do not predict outcome support.

The author's 77x77, 30-pair cached benchmark is synthetic evidence only and includes 2,310 leaveouts; it is not a calibrated full-cache wall-time estimate. The old uncached all-edge 52-hour extrapolation is inapplicable. Freeze a bounded runtime from the actual cached algorithm and retain progress/exit/receipt markers; do not alter statistics when the job runs slowly.

## Prelaunch audit disposition

No conceptual blocker remains after the parent-fixed alias/overlap union and canonical full-matrix contract above. This is **not yet an implementation/output PASS**. Remaining checks are independent three-round cached-module synthetic audits, producer/reader allowlist and exclusion tests, exact per-h reconstruction, complete 341-source coverage, fixed-support paired aggregation, runtime/storage completion, and protected-pin verification. This file is frozen before any new diagnostic readout.
