#!/usr/bin/env python3
"""Evaluate a sealed dependence readout by changing only category_id.

The original TETA wrapper performs all official metric operations unchanged.
This script binds its input labels to audited global IDs and verifies every
serialized non-category field and row position against the frozen baseline.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    temp.replace(path)


def bind_sources(args):
    manifest = json.load(args.cache_manifest.open())
    assert manifest["status"] == "PASS_DEPENDENCE_READOUTS_FULL"
    source_manifest = json.load(args.reference_manifest.open())
    assert source_manifest["status"] == "PASS_FULL_TRACKER_POSTERIOR_CACHE_AUDIT"
    hashes = {str(path): sha256(path) for path in
              (args.cache, args.cache_manifest, args.reference, args.reference_manifest, args.baseline, args.ann)}
    assert hashes[str(args.cache)] == manifest["output_sha256"]
    assert hashes[str(args.reference)] == source_manifest["output_sha256"]
    assert hashes[str(args.baseline)] == source_manifest["input_sha256"]["official"]
    assert hashes[str(args.ann)] == source_manifest["input_sha256"]["annotation"]
    # Match the source paths recorded by the builder even if CLI paths are
    # absolute aliases of the same files.
    for source_path, expected in manifest["inputs_sha256"].items():
        assert sha256(ROOT / source_path if not Path(source_path).is_absolute() else source_path) == expected
    return manifest, hashes


def build(args):
    started = time.time()
    manifest, protected = bind_sources(args)
    out = args.out / args.rule
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh rule directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache, allow_pickle=False) as cache:
        names = cache["method_names"].tolist()
        assert names.count(args.rule) == 1
        labels = cache["pred_category_id"][names.index(args.rule)]
        keys, gids, n_obs = cache["key"], cache["global_track_id"], cache["n_obs"]
        assert labels.shape == gids.shape == n_obs.shape == (324639,)
        assert np.isin(labels, cache["category_ids"]).all()
    with np.load(args.reference, allow_pickle=False) as reference:
        assert np.array_equal(keys, reference["key"])
        assert np.array_equal(gids, reference["global_track_id"])
        assert np.array_equal(n_obs, reference["n_obs"])
        tcc = reference["tcc_category_id"]
    assert keys.shape == (len(gids), 2) and len(np.unique(gids)) == len(gids)
    assert len(set(map(tuple, keys))) == len(gids)
    expected_obs = int(n_obs.sum())
    assert expected_obs == 1529908
    track_index = {int(gid): j for j, gid in enumerate(gids)}
    original = json.load(args.baseline.open())
    assert len(original) == expected_obs
    observed = Counter()
    changed_obs = 0
    emitted = []
    for source in original:
        gid = int(source["track_id"])
        row = track_index[gid]
        assert int(source["video_id"]) == int(keys[row, 0])
        assert int(source["category_id"]) == int(tcc[row])
        observed[gid] += 1
        prediction = dict(source)
        prediction["category_id"] = int(labels[row])
        changed_obs += prediction["category_id"] != source["category_id"]
        emitted.append(prediction)
    assert set(observed) == set(track_index)
    assert np.array_equal(np.asarray([observed[int(gid)] for gid in gids]), n_obs)
    result = out / "tao_track.json"
    temp = out / "tao_track.incomplete.json"
    with temp.open("w") as handle:
        json.dump(emitted, handle, allow_nan=False)
    temp.replace(result)
    del emitted
    # Read the actual serialized JSON and compare all fields, all boxes and
    # row positions. This also rules out reordering or dropping a detection.
    persisted = json.load(result.open())
    assert len(persisted) == len(original)
    noncategory = hashlib.sha256()
    verified = 0
    for source, prediction in zip(original, persisted):
        assert source.keys() == prediction.keys()
        for key in source:
            if key != "category_id":
                assert source[key] == prediction[key], (verified, key)
        row = track_index[int(source["track_id"])]
        assert int(prediction["category_id"]) == int(labels[row])
        noncategory.update(json.dumps({k: v for k, v in source.items() if k != "category_id"},
                                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        noncategory.update(b"\n")
        verified += 1
    del original, persisted
    for path, digest in protected.items():
        assert sha256(path) == digest
    receipt = dict(rule=args.rule, status="PREDICTIONS_BUILT", baseline=str(args.baseline),
                   baseline_sha256=protected[str(args.baseline)], cache=str(args.cache),
                   cache_sha256=protected[str(args.cache)], result=str(result), result_sha256=sha256(result),
                   cache_manifest=str(args.cache_manifest), cache_manifest_sha256=protected[str(args.cache_manifest)],
                   source_hashes=protected, wrapper_script_sha256=sha256(__file__),
                   observations=verified, tracks=len(gids), changed_tracks=int(np.count_nonzero(labels != tcc)),
                   changed_observations=int(changed_obs), modified_fields=["category_id"], second_tcc=False,
                   aggregation_uses_instance_gt=False, masks_use_public_vocabulary_metadata=True,
                   parity=dict(all_rows_checked=verified, all_noncategorical_fields_equal=True,
                               row_order_equal=True, boxes_equal=True, scores_equal=True, ids_equal=True,
                               per_track_observation_counts_equal=True,
                               noncategory_ordered_rows_sha256=noncategory.hexdigest()),
                   build_seconds=time.time() - started)
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def evaluate(args):
    # Import only after prediction construction, source checks and the complete
    # serialization parity audit have written their reviewable receipt.
    from evaluate_track_readouts import evaluate as official_evaluate
    official_evaluate(args)
    receipt_path = args.out / args.rule / "receipt.json"
    receipt = json.load(receipt_path.open())
    assert receipt["status"] == "COMPLETE"
    for path, digest in receipt["source_hashes"].items():
        assert sha256(path) == digest
    receipt["protected_sources_unchanged_after_evaluation"] = True
    receipt["wrapper_script_sha256"] = sha256(__file__)
    write_json(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--phase", choices=["build", "evaluate", "both"], default="both")
    parser.add_argument("--cache", type=Path, default=ROOT / "results/dependence_audit_20261023/readout_cache.npz")
    parser.add_argument("--cache-manifest", type=Path, default=ROOT / "results/dependence_audit_20261023/manifest.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/tces_audit_20261023/track_cache.npz")
    parser.add_argument("--reference-manifest", type=Path, default=ROOT / "results/tces_audit_20261023/manifest.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/cov_val_full_official_a100/tao_track.json")
    parser.add_argument("--ann", type=Path, default=ROOT / "data_tao/annotations/validation_ours_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/dependence_audit_20261023/readouts")
    parser.add_argument("--cores", type=int, default=8)
    args = parser.parse_args()
    if args.phase in ("build", "both"):
        build(args)
    if args.phase in ("evaluate", "both"):
        evaluate(args)


if __name__ == "__main__":
    main()
