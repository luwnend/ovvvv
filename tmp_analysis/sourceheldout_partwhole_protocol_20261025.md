# Source/video-held-out protocol for part–whole class routing

Date: 2026-10-25
Status: **PROTOCOL DESIGN ONLY — no large experiment launched, no model trained,
no frozen result modified.**
Scope: fixed COVTrack val detections, identities, scores and track posteriors;
the intervention changes only a track-level `category_id`.
This document is read-only with respect to every result cited below.

## 1. Decision and the central measurement problem

The current part–whole mapping is a useful diagnostic, not yet a method result.
On the 9,466 single-GT-class predicted trajectories in the classification-event
cache it changes 1,789 final labels and repairs 1,671, but those repair counts
are dominated by `person`. These are predicted tracks joined to one GT class,
not the 5,473 GT trajectories in the val annotation. In the TAO TETA protocol,
class accuracy (`ClsA`) is an equal-weight macro average over 296 classes.
Correctly turning clothing predictions into `person` raises `person` recall,
while destroying every correct prediction for the child class.

The already observed global mapping provides a concrete warning:

| quantity | existing full-val diagnostic |
|---|---:|
| single-GT-class predicted trajectories | 9,466 |
| base correct | 3,715 |
| after mapping correct | 5,365 |
| repaired / broken | 1,671 / 21 |
| track accuracy | 0.3925 → 0.5668 |
| existing `corrected_mask_sumlog` ClsA | 19.1475 |
| existing `corrected_mask_sumlog` novel ClsA | 9.8245 |

The track-accuracy gain must therefore be reported separately from official
`ClsA`. The latter is the primary classification endpoint. Repair/break and
child-class recall explain *how* a change happened; they do not replace `ClsA`.
The ClsA and Novel ClsA shown above are full-val references used to expose the
measurement problem. They must be recomputed on the frozen development and
heldout provider subsets; neither number is a heldout result.

The mapping's provenance cannot support a held-out method claim. The existing
artifact says `mapping_frozen_from_names_before_audit=True`, but the selection
chronology cannot be established from the artifacts themselves. It is
conservatively treated as outcome-informed regardless of selection chronology.
No source or video split can undo that uncertainty. The existing mapping can
be a strong control; a new relation set must be frozen from development-only
evidence or from an external vocabulary definition before the held-out domains
are scored.

## 2. Available source and video metadata

TAO `validation_ours_v1.json` (SHA-256
`559811dd8166922c537e71457fecc0faf9ad0acc5522cafca7954273c0e2a67d`)
contains 988 validation videos and seven provider names:

| provider | videos | metadata-only grouping unit |
|---|---:|---|
| HACS | 206 | `youtube_id` |
| LaSOT | 200 | video name |
| Charades | 166 | `subject` |
| AVA | 127 | `movie` |
| BDD | 108 | video name |
| YFCC100M | 107 | `user_id` for leakage control; video name for the fine unit |
| ArgoVerse | 74 | `sequence` |

The seven providers are available directly as `videos[].metadata.dataset`.
They are not inferred from category outcomes. With the leakage-unit definition
used here, there are 718 distinct grouping units at the coarsest level:
HACS 205, LaSOT 200, Charades 42, AVA 25, BDD 108, YFCC100M uploader 100, and
ArgoVerse 38. Provider videos are often segments of one source asset. Any
video split must hash and assign the whole grouping unit. This 718 count groups
HACS by `youtube_id` (205 units) and YFCC100M by `user_id` (100 units).
Alternative choices of grouping field produce a different coarse-unit count
and must not be mixed with this one.

All seven official TAO video lists are present and cover all 988 validation
videos after source-specific normalization:

| official list | matching val field | matches |
|---|---|---:|
| `tao_ava.txt` | `metadata.movie` | 127/127 |
| `tao_hacs.txt` | `metadata.youtube_id` | 206/206 |
| `tao_charades.txt` | last component of video name | 166/166 |
| `tao_argoverse.txt` | `metadata.sequence` | 74/74 |
| `tao_bdd.txt` | last component of video name | 108/108 |
| `tao_lasot.txt` | last component of video name | 200/200 |
| `tao_yfcc.txt` | last component after removing `v_` | 107/107 |

The last row matters: comparing the raw last component yields 0/107. A
preflight must verify all 988 names against the proper field and reject any
unmatched video. The 7 source lists contain validation and test videos and
cannot be used to establish how portions of those lists were split between
TAO train and val. Video-name disjointness is required, but name disjointness
alone is not evidence of content or uploader disjointness.

The existing sealed FIT/TUNE roles
(`results/base_error_full_collection_20261023/pair_context_v1/source_roles.json`,
SHA-256 `afa61df067f392b84ba917e1c536a9b64b869dbda19a2236b0a05b6f5c51cb8b`)
describe train-side source groups only: 341 FIT sources, 42 TUNE sources, and
zero overlap with the 988 val videos. They must not be relabeled as val
development or val heldout. Their valid role is to define what training/readout
data was visible.

## 3. Frozen val domain split

This split is reproducibly specified from provider names only. It is
prospective/confirmatory only if this manifest is sealed before any heldout
class scoring; otherwise it remains an exploratory domain split. It uses the
following rule and seed:

```text
seed = part_whole_val_domain_holdout_20261025_v1
order provider names by sha256(seed + "\0" + provider), tie break provider
development = first 3 providers
heldout     = remaining 4 providers
```

This yields:

| role | providers | videos | single-GT-class predicted trajectories |
|---|---|---|---:|
| development | YFCC100M, LaSOT, ArgoVerse | 381 | 3,552 |
| heldout | AVA, BDD, Charades, HACS | 607 | 5,914 |

Every video, image, GT trajectory and predicted trajectory belongs to exactly
one role. Parent–child edges cannot transfer across roles. The official
video-list hashes are:

```text
tao_argoverse.txt  f397e9ce1acd22b0184611363604ad3af1293898dab113bffbe49beec8989aab
tao_ava.txt        6ec6f77af1062c2234d8c3d91a606b9025fad9e623814f7afd598239e4083784
tao_bdd.txt        030b6a79beb4d59ca5a8b6c0b897a79139f8ab3cb9acf934cf528349221f9979
tao_charades.txt   7ce05068f38dd5f83e8e074f5abc434a2bf7d88fb50d8ab22ab48bb08ffa8d22
tao_hacs.txt       7dec94d9c960fb263a6f332b262d023069f67167aa054f3673f344e15886daaa
tao_lasot.txt      c9653f38012e7dca364ce13a8aa9a0afd47018f48fd5828b72ed3b1907ad3253
tao_yfcc.txt       f378fc7087135aaaf4852e2a185d3a41f87d043c8db490609f39bfb13495c8f6
```

The existing val split files `ovmot_subset/val60.json` and `val200.json` are
not used as new heldout evidence. They contain all seven providers and were
already used in exploratory work. The full 988-video split above is available
without GPU work because the fixed readout and events caches exist.

This provider-level split is a **domain transfer test**. It is not a
leave-one-source-out estimate over hundreds of small sources. Report every
held-out provider separately. For sensitivity, treat each provider as a fold
using `leave-one-provider-out`: fit/freeze on the other six and score the
omitted provider. Do three coarser folds if compute is constrained:

1. YFCC100M + LaSOT + ArgoVerse development, AVA + BDD + Charades + HACS heldout;
2. BDD + Charades + AVA heldout;
3. HACS + YFCC100M heldout.

The primary comparison in fold 1 is evaluated once if sealed first. Folds 2–3
are stability checks and must be fixed before their results are opened. The
provider roles are never changed after seeing class outcomes.

## 4. What must be frozen on development

The following objects must be written to a new, immutable development
manifest before any heldout class score or prediction is opened:

- parent/child class pairs;
- mapping direction and whether a correction replaces or only reranks;
- containment definition (`IoU` or center-in-box threshold, area ratio,
  minimum common frames);
- whether a child must be slightly smaller or strictly smaller;
- evidence ownership score and any text/class table parameters;
- thresholds, priors, temperature and tie-breaking;
- the exact parent/child endpoint keys and the candidate selection rule;
- the bootstrap seed and number of provider-cluster resamples.

The parent/child mapping for a new method must come from one of:

1. a frozen external hierarchy or definitions, hashed before val scoring;
2. development-provider vocabulary structure only, with no val GT correctness;
3. a fit on development-provider *legal training* labels, with heldout GT
   labels not decoded;
4. a predeclared relation supplied by a training-vocabulary source.

The currently observed 11-pair mapping
(`shirt/jacket/jersey/sweater/trousers/jean/coat/skirt -> person`,
`cab_(taxi)/pickup_truck -> car_(automobile)`,
`trailer_truck -> truck`) is **exploratory and conservatively treated as
outcome-informed**. It may be used as a diagnostic arm on development and as a
named strong control on heldout, but it cannot serve as the new method's
independently frozen relation.

The development split exists to choose among mechanisms, not to rescue any
particular mapping. Once the manifest is sealed, heldout outcomes are used
once for the planned primary comparison. If a threshold changes after looking
at heldout, that run is exploratory and cannot be advertised as heldout.

## 5. Endpoint and support construction

The established fixed caches are:

```text
results/dependence_audit_20261023/readout_cache.npz
  3c764166e42740b4d9df00272044129c55dab6cd7909a2e7b0c6e6e00d484149
results/dependence_audit_20261023/teta_events/classification_events.npz
  fc11440c3ce750ee4918e1290e32cc5b6a20c4fde20aca1ae70f19fee6e0e996
results/cov_tces_v3_fullval_eval/tao_track.json
  a01eb48eb91c7116330fd31158b219f61fb3d2c68ff805ff44b6a43749370a48
```

The readout cache has 324,639 predicted tracks and 296 evaluation classes.
The event cache has 383,115 rows and 9,734 predicted tracks with at least one
classification event; 9,466 have a single GT class and 268 (260 two-class and
eight three-class) are multi-class and must stay excluded. The val annotation
has 5,473 GT trajectories, each with one class. The official replay has
1,529,908 observation records. These files are already available on local
storage; no large copy is needed.

Canonical joins:

- cache `key = (video_id, local_track_id)` → **internal `global_track_id`**.
  This ID is a replay offset, not an official TAO `track_id`.
- event `sequence_names` → canonical val video name/provider.
- the event cache does not contain `gt_track_id`; a mapping from its
  `global_track_id` to official GT-trajectory IDs must be rebuilt from the
  annotation matching table before any provider→video→GT-trajectory bootstrap.
- cache/event `global_track_id` and `(video_id, local_track_id)` must be
  checked one-to-one before source grouping.
- the parent/child v2 rows contain only `parent_gid` / `child_gid`; a conversion
  to `(video_id, local_track_id, source, provider)` must be built and hashed.
  Its absence is a concrete missing artifact, not a reason to infer a source.
- official TETA replay keys are `(image_id, video_id, track_id)`; its
  `track_id` is the video-offset ID used by `TaoDataset._track2json`, not the
  cache's internal ID or a globally unique TAO track ID.

Unknown predictions and multi-GT trajectories are never negatives. A GT class
is used only after all candidate/container relationships have been frozen.
Parent selection, child selection, memory content and thresholds must be
deployable from detections and metadata. GT may be used only to score.

## 6. Required arms

All arms use exactly the same track population, observations, boxes, scores,
IDs and evaluation split. Only the final class assignment may differ. The
minimum set is:

| arm | purpose |
|---|---|
| official baseline / TCC | frozen reference |
| `sumlog` corrected mask | strong existing readout |
| global exploratory hierarchy | diagnostic upper-bound-style control |
| frequency prior only | tests whether mapping is merely class prior |
| random mapping of equal cardinality | lower control |
| reversed / child-destroying mapping | checks macro damage |
| size/position-only gate | tests non-semantic containment |
| geometric containment only | tests relation geometry |
| semantic containment routing | tests parent-class compatibility |
| class-conditioned child explain-away (CEL) | proposed narrow mechanism |
| child-mask / area-matched mask | causal visual control |
| MHA / additive child pooling | generic attention control |
| class-agnostic / learned-class-table gate | text-conditioning control |

For CEL, the only defensible new claim is: remove from a parent's class
evidence the amount explained by a contained child, conditional on the parent
candidate class and the child's observed class. Do not claim generic
parent–child geometry, part attention or scene graph reasoning as new; those
have prior art. CEL must beat the class table and area-matched mask on heldout
`ClsA`, not only track accuracy.

## 7. Metrics and statistical units

Primary:

- official TETA at the standard match threshold;
- `Base`, `Novel` and combined `ClsA`, using the evaluator's equal-class
  averaging over the 296 eval classes;
- `LocA`, `AssocA`, `TETA` reported separately.

Classification mechanism diagnostics:

- track-level correct / wrong, repair / break / unchanged;
- per-class ClsA delta and support count;
- child-class recall, before and after; every semantic child class with zero
  support is reported as zero-support, not silently omitted;
- parent-target accuracy and repair/break on parent GT tracks;
- containment coverage, false-positive routing and fallback count.

The primary uncertainty unit is the provider. Bootstrap resample provider
clusters, then videos within each sampled provider, then GT trajectories within
each sampled video. Do not bootstrap individual parent–child pairs as
independent samples. Report:

- provider-macro point estimate;
- paired provider/video-cluster bootstrap interval;
- leave-one-provider-out sensitivity;
- each provider's support and effect, so one provider cannot carry the result.

There are only seven providers in the whole val set, and only four in the
primary heldout set. A provider bootstrap is coarse; the four-provider interval
cannot support precise significance claims. Leave-one-provider-out is a
sensitivity analysis, not an independent replication. With only 146
single-GT-class predicted trajectories and 99 GT trajectories in the 11 mapped
child classes, threshold selection on these child classes would be unreliable.
The earlier 122-track number belongs to a different, broader anchor population
and is not used here.

Acceptance requires all of:

1. heldout `ClsA` increases over `sumlog` and over the size/position and
   frequency controls;
2. child-class recall does not collapse; any class with base support above
   zero must retain a predeclared minimum or the loss is explicitly charged
   against the method;
3. repair is greater than break without relying on one provider or one
   `person` class pair;
4. the official full TETA/ClsA replay agrees in direction with the diagnostic;
5. CEL exceeds child mask and class-table controls on a source-heldout
   counterfactual gate;
6. no GT-derived parent or child selection occurs in the deployed routing.

The existing exploration would fail condition 2 if applied globally: all
correct `cab_(taxi)` (13), `jacket` (3), `shirt` (2), `pickup_truck` (2) and
`skirt` (1) tracks are destroyed. The exact numbers above come from the frozen
diagnostic artifact and are not new heldout evidence.

## 8. Official TETA replay

When the development decision is made, produce a new output directory; never
overwrite `results/cov_tces_v3_fullval_eval/` or the existing
`corrected_mask_sumlog` result.

Required replay invariant:

```text
records: unchanged count, order, image_id, video_id, track_id, bbox, score
only field allowed to change: category_id
```

A pre/post audit must prove:

- equal record count and a positionwise identical sequence of
  `(image_id, video_id, track_id, bbox, score)`, checked both element by element
  and by a canonical sequence hash;
- every changed record differs only in `category_id`;
- the category IDs are TAO `categories[].id`, not 296-column indices;
- track IDs remain the per-video replay offsets and no track merges/splits;
- no second TCC is applied after the track-level intervention;
- the evaluator and annotation files are hashed and kept fixed;
- the replay output has its own manifest and receipt.

The established evaluator command is:

```bash
source /home/chenhc/miniconda3/etc/profile.d/conda.sh
conda activate ovtrack_ovmot
python ovmot_improve/eval_teta.py \
  --ann /raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json \
  --out results/<new_run_directory> \
  --eval_only
```

That path is valid only when `<new_run_directory>/tao_track.json` already exists
and has passed the replay invariant. CPU-only TETA is appropriate after
recipe selection; no GPU is needed for this local classification replay.

## 9. Minimal feasibility checklist

Available now, no GPU and no large copy:

- 988-video metadata and provider labels;
- all seven official video lists and hashes;
- 296-class fixed readout cache (324,639 tracks);
- 9,466 single-GT-class predicted trajectories and 268 excluded multi-class
  predicted tracks; 5,473 GT trajectories in the annotation;
- official 296-class per-class metrics for `sumlog`;
- 2,574 contained parents and 5,109 parent–child edges;
- 146 single-GT-class predicted trajectories and 99 GT trajectories in the 11
  mapped child classes, with provider support for the latter;
- fixed baseline replay with 1,529,908 observations;
- official TETA evaluator in `ovtrack_ovmot`.

Missing or must be created before a sealed heldout run:

- an immutable translation of every parent/child `global_gid` to
  `(provider, video_id, local_track_id)` with a content hash;
- a development-only relation manifest and its provenance hash, independent
  of the existing full-val confusion-inspired mapping;
- a frozen candidate/parent selection rule that is GT-free;
- a real-image child occlusion cache for the counterfactual CEL gate, if the
  mechanism is pursued beyond the fixed-posterior routing; the fixed readout
  cannot test visual evidence ownership;
- per-child-class heldout support: most classes have very small support, so
  several child classes will be descriptive only;
- a predeclared exact class-macro formula after parent/child routing, including
  whether child classes are retained, merged or softly reweighted.

Provider support for the 11 semantic child classes in the fixed
`classification_events` cache (single-GT-class predicted-trajectory counts) is:

| provider | shirt | jacket | jersey | sweater | trousers | jean | coat | skirt | cab | pickup | trailer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AVA | 1 | 2 | 0 | 1 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| ArgoVerse | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 0 |
| BDD | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 12 | 1 |
| Charades | 7 | 6 | 0 | 3 | 2 | 3 | 0 | 0 | 0 | 0 | 0 |
| HACS | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| LaSOT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| YFCC100M | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

This table is a feasibility warning: a provider-level split does not create
enough independent child-class evidence in every provider. Class-specific
claims must be limited to classes with support across multiple providers;
otherwise the claim is about `person` routing, not general part–whole
disambiguation. The official annotation has 99 GT trajectories in these
classes with provider totals AVA 8, ArgoVerse 13, BDD 44, Charades 21, HACS 6,
LaSOT 2, and YFCC100M 5. Predicted-trajectory support and GT support are not
interchangeable.

## 10. Execution order and stopping rules

1. Build and independently audit the provider/video mapping and the
   `global_gid` translation. Stop on any unmatched name, duplicate key or
   non-reversible ID.
2. Reproduce the existing `sumlog` and global-mapping diagnostics on the
   development providers without changing predictions. Verify record counts,
   category IDs, per-class supports and macro arithmetic.
3. Freeze the relation and routing manifest. Only then open heldout scoring.
4. Run a CPU fixed-posterior test for mapping, prior, random/reverse, size and
   geometry controls. Record per-provider and per-class outcomes.
5. If CEL's fixed-posterior routing is not better than all controls, stop it;
   do not proceed to visual occlusion or training.
6. If it passes, collect the minimal real-image parent/child occlusion cache
   from `/raid`, with area-matched and class-agnostic controls. Require
   repair/break ≥2:1 and a positive provider-cluster lower bound.
7. Only after both gates, attach the mechanism to COVTrack and run an isolated
   official evaluation. Never overwrite a frozen baseline.

The present stage is between steps 1 and 2. The next action should be a
small CPU implementation audit, not a GPU training run. There is currently no
evidence sufficient to justify training a new network module.

## 11. Relationship to the existing results

This protocol supersedes only the vague statement that source-heldout might
validate the global mapping. It does not alter:

- the frozen COVTrack baseline,
- the corrected `sumlog` result,
- the geometry ROI STOP decision,
- the containment `PAUSE_CONTAINMENT_EVIDENCE` decision,
- the existing parent/child diagnostic files.

The fixed-posterior experiment can decide whether class routing is worth
pursuing. It cannot establish that a neural evidence-ownership module works.
For a paper claim, the routing result and the visual counterfactual result
must remain separate, and official TETA must be the final endpoint.
