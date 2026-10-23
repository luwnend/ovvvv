# Independent support-builder review, 2026-10-23

**Executable conclusion: GO for a descriptive CPU support audit** using `tmp_analysis/build_base_error_support.py` at SHA256 **`64e702efa4e0956b19e9f9a0991ecfff8022c527e3e8bfb3a50710bbf4090186`**. This is not approval of a learner, training readiness, a covariance mechanism, or efficacy. The reviewer did not modify the support builder or run GPU inference, training, or real prediction/GT scoring.

## Independent execution

The command below actually exited **0**, producing `tmp_analysis/base_support_contract_audit_20261023.json` with status `PASS_INDEPENDENT_SUPPORT_CONTRACT_AUDIT`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/chenhc/miniconda3/envs/covtrack/bin/python \
  tmp_analysis/audit_base_error_support_contracts.py --real-metadata \
  --output tmp_analysis/base_support_contract_audit_20261023.json
```

The independent fixture checks prove that the full geometry pool retains heldout and unmapped GT, an old numeric category collision resolves by synset, planned `image_id` records are accepted, and deliberately invalid test annotation values never enter GT joins or outcomes. Six corrupt provenance cases are rejected: duplicate original frame coordinate, changed original filename, nonoriginal C-TAO frame, changed manual annotation ID, changed C-TAO source box, and changed preserved source frame ID. The geometry counterexample has IoU matrix `[[1,.5],[.5,.2]]`: the implementation prioritizes two valid matches over the larger single-match IoU sum, uses no category/posterior argument, and flags the ambiguous prediction after assignment. Negative local IDs cannot become track keys.

The current pure provenance validator also ran on the actual frozen source metadata, without opening prediction dumps. It accepted **1025 fit/tune frames / 2969 exact manual anchors**, verified unique official frame coordinates, canonical filenames/dimensions, C-TAO `is_ori`, preserved raw frame IDs, sealed manual IDs, the unique synset bridge, and original/manual boxes within `1e-6`. No test GT frame map or test prediction/GT outcome was constructed.

An additional independent implementation of the metadata checks, which does not call the builder's validator, exited 0 and saved `tmp_analysis/base_support_independent_metadata_provenance_20261023.json`. It confirms the same 1025/2969 support and source hashes. Among the old **1230** categories, **1068** uniquely mapped categories change numeric ID and **79** have no unique canonical mapping. Numeric-ID joining would therefore be materially wrong.

The first combined run completed its logic checks but was correctly rejected by the final source-hash guard because the author was still editing. No passing report was written for that moving source. The final result above binds the stable SHA explicitly.

## Reviewed implementation boundaries

1. `verified_inputs` requires both full-collection verifier passes, complete matching frame inventories, sealed annotation/run/receipt bindings, and every dump hash. `seal_gt_free_index` writes every emitted observation and `(video_id,local_track_id)` trajectory before `full_gt_and_manual` parses supervision files. It excludes `id=-1` from trajectories while reporting unassigned counts. Later label subsets cannot replace or mutate the hashed GT-free index.
2. All emitted predictions are matched against **all official original-TAO GT** at each open fit/tune frame. Rare, heldout, unmapped, crowd/ignore, and nonmanual-intersection GT remain in the class-blind geometry pool. Only after assignment are ambiguous, crowd/ignore, nonmanual, and heldout matches excluded from supervised outcomes. Unmatched predictions remain unknown.
3. Exact manual supervision requires the sealed selected annotation ID, original C-TAO image/annotation fields, `is_ori=True`, matching canonical filename and coordinates, original preserved frame ID, unique official track/frame identity, uniquely mapped synset, and matching boxes. Per-observation fit-category eligibility precedes error/soft-risk construction. Pair endpoints come only from these eligible observations, require distinct frames, and share both predicted and official identity. Fit/tune split identity is retained.
4. Test frames participate only in the immutable GT-free observation/track index. They are skipped in official GT mapping, manual class/box joins, and prediction/GT scoring. Complete source JSONs necessarily contain test-partition labels and are parsed in memory; the valid claim is **test outcomes remain closed**, not that no file containing test labels was ever opened.
5. The extra whole-track single-identity view can depend on heldout/nonmanual identities. The final code names it `diagnostic_oracle_observed_single_matched_identity_tracks`, removes its unsafe Boolean from `supervised_pairs.jsonl`, separates oracle identity summaries, and explicitly forbids using them as training/tuning masks, weights or features. The main usable support is `all_same_identity_subpairs` with both endpoints independently eligible.
6. Primary correctness uses the actual emitted category, while posterior-argmax correctness is separately reported. `1-p_fused[GT]`, its endpoint products, both-wrong rates, and same-wrong-class rates are descriptive statistics. Source/GT-trajectory concentration checks do not turn dependent pairs into independent samples or establish centered covariance.

## Findings fixed before GO

The reviewer identified two execution/interpretation blockers in intermediate code. The new provenance validator originally passed plan entries with `image_id` to a helper requiring `id`, which would raise `KeyError`; the final code explicitly normalizes that planned schema. The earlier pair artifact exposed an oracle single-identity flag computed before class eligibility; the final code restricts it to the diagnostic scope and removes it from supervised pair rows. The author also completed direct C-TAO/original-TAO source-chain checks and includes both source hashes in the input snapshot and final recheck.

## Scope remaining after this review

The root may now run the stable script on the sealed full pilot in a new output directory, then verify the actual exit code, support receipt, GT-free index seal, input/output hashes and test-closed counters. This reviewer has not yet reviewed such real support outputs. Existing full-pilot receipts report **1299 frames / 41 resets / 54924 retained and emitted observations**, 1204 columns and canonical 337-mask fusion; the background verifier receipt records both validator exits 0. The original collection terminal's actual exit 0 was supplied by the root's separate tool session, not inferred from that detached receipt.

Training still needs its own frozen specification and admissible feature/target pathway. All-label diagnostic oracle scopes, heldout-object outcomes, and test outcomes must remain excluded. Sparse-anchor time, readout-only category holdout, frozen-detector exposure, manual-match selection and source-group dependence remain limits of any subsequent research claim.
