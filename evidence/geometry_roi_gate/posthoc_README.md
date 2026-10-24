# Geometry ROI 276-frame post-hoc acceptance

The 276 token NPZs in
`/raid/datasets/chc_data/geometry_roi_full_20261024_v1` passed a read-only
content audit. This is **not** an official collector completion receipt. The
collector stopped during postflight with `Manifest changed during collection`
and never wrote `collection_receipt.json`.

## Why the postflight failed

The manifest at the launch path had SHA-256
`bc18bcda740d319d126dceb2d8751146bf3da9efbae0c3a13fd79a6d7dd48068`
when `started.json` was written. Its exact bytes are preserved as
`results/geometry_roi_full_collect_manifest_20261024_v1_superseded_pre_pair_audit/collector_manifest.json`.

That path was later replaced by a manifest whose SHA-256 is
`687c88325f05f57c7e7806a629b20549c7f522bc5fa7d0fdf830bcca098b6677`.
The two files are otherwise identical; the sole changed field is
`output_directory`, from the collected `_v1` directory to the uncollected `_v2`
directory. The runner reauthenticated the replaced bytes at the end and
correctly refused to write a final receipt.

The 276 frame hashes, frame records, input pins, and token contents themselves
were not changed by this incident.

## What passed

The audit checked all 276 files against the launch manifest and the frozen
parent collection:

- exact filename set and continuous 0-based frame runs for 12 videos;
- 15,153 retained rows; every token has nonzero rows;
- frozen runner, trace, model config, checkpoint, and parent manifest pins;
- live parent-cache and image hashes;
- exact NPZ member schema, float16 spatial/raw tensors, and finite values;
- keep-mask row locators, detector boxes, proposal rows, and raw output parity
  with the parent cache;
- source, video, image, frame, image SHA, cache SHA, and preprocessing metadata;
- no GT, identity, class, posterior, or matching arrays in the token NPZs.

The first 18 frames also match the independently collected pilot token files
exactly across every NPZ member. This tests byte-level repeatability; it does
not establish that ROI tokens improve tracking.

## Safe use

Use the NPZs only together with the launch-manifest SHA above and
`TOKEN_SHA256SUMS`. The corrected provenance manifest is
`results/geometry_roi_full_collect_manifest_20261024_v3/`; it is not a receipt
for this already completed run. Before training, still apply source/video holdout,
hard-negative, and leakage gates. Do not cite this directory as a collector
run with a completion receipt, or as evidence of method effectiveness.

For a future collection, keep the manifest path immutable after launch. Do not
regenerate or replace a manifest at an active output path. Publish a separate
new manifest for a new output directory.
