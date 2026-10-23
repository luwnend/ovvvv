#!/usr/bin/env python3
"""Build fixed-track, GT-free dependence readouts and fixed-budget controls.

Annotation use is restricted to image/video metadata. Vocabulary masks, global
IDs, baseline TCC and reference full-track sums come from the sealed cache.
No detections, IDs, scores, baseline files or training artifacts are changed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np


FAMILIES = ["tcc", "full_sumlog", "score_weighted_log", "entropy_weighted_log",
            "budget_random_seed0", "budget_random_seed1", "budget_random_seed2",
            "budget_topscore", "budget_fps_track", "dep_inverse_density"]
MASKS = ["released", "corrected"]
EPS = 1e-12
BUDGET = 4
KERNEL_POWER = 4


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_weights(values):
    values = np.asarray(values, np.float64)
    assert np.isfinite(values).all() and np.all(values >= 0)
    total = float(values.sum())
    return values / total if total > EPS else np.full(len(values), 1. / len(values))


def first_exact_evidence_rows(features, pt, pi, pf):
    """Exact means equal stored track features and all three posteriors.

    Frame/time/box are deliberately excluded: repeating the same observed
    evidence at another timestamp remains one evidence row for this method.
    Detection score is excluded as a nuisance value, used only by controls.
    """
    kept = []
    seen = set()
    for i in range(len(features)):
        signature = (features[i].tobytes(), pt[i].tobytes(), pi[i].tobytes(), pf[i].tobytes())
        if signature not in seen:
            seen.add(signature)
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def selections_and_dependence(features, pt, pi, pf, scores, video_id, local_id):
    n = len(features)
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    assert np.isfinite(norms).all() and np.all(norms > 0), "Cosine requires nonzero track features"
    z = features.astype(np.float64) / norms[:, None]
    cosine = np.clip(z @ z.T, -1, 1)
    np.fill_diagonal(cosine, 1)
    k = min(n, BUDGET)
    selected = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(video_id), int(local_id)]))
        selected.append(np.sort(rng.choice(n, size=k, replace=False)))
    top = np.argsort(-scores.astype(np.float64), kind="stable")[:k]
    selected.append(np.sort(top))
    # Same fixed budget, highest-score initialization, earliest-row ties.
    fps = [int(top[0])]
    best_similarity = cosine[:, fps[0]].copy()
    while len(fps) < k:
        best_similarity[fps] = np.inf
        nxt = int(np.argmin(best_similarity))
        fps.append(nxt)
        best_similarity = np.maximum(best_similarity, cosine[:, nxt])
    selected.append(np.sort(fps))
    unique = first_exact_evidence_rows(features, pt, pi, pf)
    kernel = ((1 + cosine[np.ix_(unique, unique)]) / 2) ** KERNEL_POWER
    weights = normalize_weights(1. / kernel.sum(axis=1))
    return selected, unique, weights, norms


def fuse_corrected(pt, pi, rare_mask):
    powers = np.where(np.r_[rare_mask, False], 1. / 3, 2. / 3)
    p = pt.astype(np.float64) ** powers * pi.astype(np.float64) ** (1 - powers)
    p /= np.maximum(p.sum(1, keepdims=True), EPS)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=2e-6, rtol=0)
    return p


def one_track(features, pt, pi, pf, scores, det_label_cols, rare_mask, video_id, local_id):
    selected, unique, dependence_weights, norms = selections_and_dependence(
        features, pt, pi, pf, scores, video_id, local_id)
    pc = fuse_corrected(pt, pi, rare_mask)
    result = []
    entropy_ess = []
    full_sums = []
    score_weights = normalize_weights(scores)
    for mask_id, p in enumerate((pf.astype(np.float64), pc)):
        logp = np.log(np.clip(p, EPS, None))
        full_sums.append(logp.sum(0))
        c = p.shape[1] - 1
        label_cols = det_label_cols if mask_id == 0 else p[:, :c].argmax(1)
        counts = np.bincount(label_cols, minlength=c)
        tcc = int(counts.argmax())
        fg = p[:, :c] / np.maximum(p[:, :c].sum(1, keepdims=True), EPS)
        entropy = -(fg * np.log(np.clip(fg, EPS, None))).sum(1)
        entropy_weights = normalize_weights(np.maximum(0, 1 - entropy / np.log(c)))
        entropy_ess.append(float(1. / np.sum(entropy_weights ** 2)))
        # Labels use all foreground columns; background remains in log sums
        # solely for parity with the original sealed aggregation cache.
        result.extend([tcc,
                       int(full_sums[-1][:c].argmax()),
                       int((score_weights @ logp[:, :c]).argmax()),
                       int((entropy_weights @ logp[:, :c]).argmax()),
                       *[int(logp[ix, :c].sum(0).argmax()) for ix in selected],
                       int((dependence_weights @ logp[unique, :c]).argmax())])
    diagnostics = dict(dep_unique_count=len(unique), dep_ess=1. / np.sum(dependence_weights ** 2),
                       dep_maxweight=float(dependence_weights.max()),
                       dep_weight_entropy=float(-(dependence_weights * np.log(np.maximum(dependence_weights, EPS))).sum()),
                       score_weight_ess=1. / np.sum(score_weights ** 2),
                       entropy_weight_ess=np.asarray(entropy_ess), feature_norm_min=float(norms.min()),
                       feature_norm_max=float(norms.max()))
    return np.asarray(result, np.int64), diagnostics, full_sums, selected


def self_test():
    # Boundary 1: a single observation must retain its per-frame class under
    # every rule in both masks, including budget and dependence controls.
    f = np.asarray([[1., 2., 0., -1.]], np.float16)
    pt = np.asarray([[.1, .7, .15, .05]], np.float32)
    pi = np.asarray([[.2, .6, .2, 0]], np.float32)
    rare = np.asarray([False, True, False])
    pf = fuse_corrected(pt, pi, rare).astype(np.float32)
    score = np.asarray([.4], np.float32)
    y, d, _, _ = one_track(f, pt, pi, pf, score, np.asarray([1]), rare, 7, 2)
    assert np.all(y == 1) and d["dep_unique_count"] == 1 and d["dep_ess"] == 1
    # Boundary 2: an exact copy of one evidence row must leave dependence
    # scores unchanged, even when that row is copied many times. This is not
    # claimed for approximately matching observations or the ordinary controls.
    f = np.asarray([[1., 0., .1], [.7, .4, .2], [0., 1., .2]], np.float16)
    pt = np.asarray([[.65,.15,.15,.05],[.2,.55,.2,.05],[.1,.2,.65,.05]], np.float32)
    pi = np.asarray([[.5,.3,.2,0],[.1,.7,.2,0],[.25,.25,.5,0]], np.float32)
    pf = fuse_corrected(pt, pi, rare).astype(np.float32)
    score = np.asarray([.5,.4,.7], np.float32)
    dep_scores = []
    for trial, ix in enumerate((np.arange(3), np.asarray([0, 1, 1, 1, 1, 1, 1, 2]))):
        nuisance_scores = score[ix].copy()
        if trial:
            nuisance_scores[1:-1] = np.linspace(.01, .99, len(ix) - 2)
        _, u, w, _ = selections_and_dependence(f[ix], pt[ix], pi[ix], pf[ix], nuisance_scores, 7, 2)
        dep_scores.append(np.stack([w @ np.log(np.clip(p[u], EPS, None))
                                    for p in (pf[ix].astype(np.float64), fuse_corrected(pt[ix], pi[ix], rare))]))
    assert np.array_equal(dep_scores[0], dep_scores[1])
    return {"single_observation_all_methods": "PASS", "exact_evidence_duplicate_invariance_with_nuisance_score_changes": "PASS",
            "exact_duplicate_score_max_abs_error": float(np.max(np.abs(dep_scores[0] - dep_scores[1])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path("results/cov_tces_v3_fullval"))
    parser.add_argument("--reference", type=Path, default=Path("results/tces_audit_20261023/track_cache.npz"))
    parser.add_argument("--reference-manifest", type=Path, default=Path("results/tces_audit_20261023/manifest.json"))
    parser.add_argument("--annotation", type=Path, default=Path("/raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dependence_audit_20261023"))
    parser.add_argument("--limit-videos", type=int, default=0, help="Smoke test only; full run uses 0")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    tests = self_test()
    print(json.dumps({"self_tests": tests}), flush=True)
    if args.self_test_only:
        return
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "readout_cache.npz"
    manifest_path = args.output_dir / "manifest.json"
    assert not target.exists() and not manifest_path.exists(), "Refuse to overwrite a completed audit"
    protected = {str(p): sha256(p) for p in (args.reference, args.reference_manifest, args.annotation)}
    ref_manifest = json.load(open(args.reference_manifest))
    assert ref_manifest["status"] == "PASS_FULL_TRACKER_POSTERIOR_CACHE_AUDIT"
    assert protected[str(args.reference)] == ref_manifest["output_sha256"]
    with np.load(args.reference, allow_pickle=False) as ref:
        keys = ref["key"]
        gids, n_obs, category_ids = ref["global_track_id"], ref["n_obs"], ref["category_ids"]
        rare, released_mask = ref["frequency_rare_mask"], ref["fusion_validation_novel_index"]
        tcc = ref["tcc_category_id"]
        corrected_tcc = category_ids[ref["p_corrected_mask_argmax_counts"].argmax(1)]
        reference_sums = [ref["p_fused_sumlog"], ref["p_corrected_mask_sumlog"]]
    assert np.all(np.diff(category_ids) > 0)
    cat_to_col = {int(cid): j for j, cid in enumerate(category_ids)}
    key_to_row = {tuple(map(int, key)): j for j, key in enumerate(keys)}
    raw_ann = json.load(open(args.annotation))
    images = defaultdict(list)
    videos = [int(v["id"]) for v in raw_ann["videos"]]
    for im in raw_ann["images"]:
        images[int(im["video_id"])].append({k: im[k] for k in ("file_name", "frame_id", "id")})
    del raw_ann  # No GT annotations/tracks used by any readout or selection.
    if args.limit_videos:
        videos = videos[:args.limit_videos]
    expected_rows = np.flatnonzero(np.isin(keys[:, 0], videos))
    output_rows = []
    labels, diags, selection_positions = [], defaultdict(list), []
    audit = dict(observations=0, dump_files=0, tracks=0, missing_dump_images=0,
                 released_tcc_mismatches=0, corrected_tcc_mismatches=0,
                 released_full_sumlog_label_mismatches=0, corrected_full_sumlog_label_mismatches=0,
                 full_sumlog_vs_reference_max_abs_error=[0., 0.],
                 original_fusion_max_abs_error=0., all_methods_single_observation_mismatches=0,
                 exact_collapsed_observations=0, detection_score_min=float("inf"), detection_score_max=-float("inf"))
    inventory = hashlib.sha256()
    for vi, video_id in enumerate(videos):
        block = defaultdict(list)
        for im in sorted(images[video_id], key=lambda im: int(im["frame_id"])):
            name = ("data/tao/frames/" + im["file_name"]).replace("/", "-").replace(".jpg", "") + ".npz"
            path = args.dump_dir / name
            if not path.exists():
                audit["missing_dump_images"] += 1
                continue
            inventory.update((name + "\t" + str(path.stat().st_size) + "\n").encode())
            with np.load(path, allow_pickle=False) as raw:
                assert int(raw["frame_id"]) == int(im["frame_id"])
                assert str(raw["filename"].item()).split("data/tao/frames/")[-1] == im["file_name"]
                data = {k: raw[k] for k in ("track_ids", "track_feats", "p_text", "p_image", "p_fused", "det_labels", "det_bboxes")}
            n = len(data["track_ids"])
            assert data["track_feats"].shape == (n, 256) and data["track_feats"].dtype == np.float16
            assert len(np.unique(data["track_ids"])) == n and np.all(data["track_ids"] >= 0)
            for branch in ("p_text", "p_image", "p_fused"):
                p = data[branch]
                assert p.shape == (n, len(category_ids) + 1) and p.dtype == np.float32
                assert np.isfinite(p).all() and np.all(p >= 0)
                assert np.allclose(p.sum(1, dtype=np.float64), 1, atol=2e-6, rtol=0)
            assert np.isfinite(data["track_feats"]).all() and np.isfinite(data["det_bboxes"]).all()
            if n:
                audit["detection_score_min"] = min(audit["detection_score_min"], float(data["det_bboxes"][:, -1].min()))
                audit["detection_score_max"] = max(audit["detection_score_max"], float(data["det_bboxes"][:, -1].max()))
            powers = np.where(released_mask, 1. / 3, 2. / 3)
            rebuilt = data["p_text"].astype(np.float64) ** powers * data["p_image"].astype(np.float64) ** (1 - powers)
            rebuilt /= np.maximum(rebuilt.sum(1, keepdims=True), EPS)
            error = float(np.abs(rebuilt - data["p_fused"]).max()) if n else 0.
            audit["original_fusion_max_abs_error"] = max(audit["original_fusion_max_abs_error"], error)
            assert error < 2e-6
            for k, values in data.items():
                block[k].append(values)
            audit["observations"] += n
            audit["dump_files"] += 1
        data = {k: np.concatenate(v, axis=0) for k, v in block.items()}
        # Stable sort preserves temporal observation order within each track.
        order = np.argsort(data["track_ids"], kind="stable")
        ids, starts, counts = np.unique(data["track_ids"][order], return_index=True, return_counts=True)
        for local_id, start, count in zip(ids, starts, counts):
            ix = order[start:start + count]
            row = key_to_row[(video_id, int(local_id))]
            assert int(n_obs[row]) == int(count)
            cols = np.asarray([cat_to_col[int(q) + 1] for q in data["det_labels"][ix]], np.int64)
            pred_cols, diagnostic, sums, selected = one_track(
                data["track_feats"][ix], data["p_text"][ix], data["p_image"][ix], data["p_fused"][ix],
                data["det_bboxes"][ix, -1], cols, rare, video_id, int(local_id))
            pred = category_ids[pred_cols]
            audit["released_tcc_mismatches"] += int(pred[0] != tcc[row])
            audit["corrected_tcc_mismatches"] += int(pred[len(FAMILIES)] != corrected_tcc[row])
            for mi, sums_i in enumerate(sums):
                reference = reference_sums[mi][row]
                error = float(np.abs(sums_i - reference).max())
                audit["full_sumlog_vs_reference_max_abs_error"][mi] = max(audit["full_sumlog_vs_reference_max_abs_error"][mi], error)
                assert np.allclose(sums_i, reference, atol=1e-4, rtol=1e-7)
                field = MASKS[mi] + "_full_sumlog_label_mismatches"
                audit[field] += int(pred_cols[mi * len(FAMILIES) + 1] != int(reference[:-1].argmax()))
            if count == 1:
                # Official emitted labels can differ from posterior argmax in
                # five frames. TCC uses emitted labels; all other rules agree.
                for mi in range(2):
                    y = pred_cols[mi * len(FAMILIES) + 1:(mi + 1) * len(FAMILIES)]
                    audit["all_methods_single_observation_mismatches"] += int(np.count_nonzero(y != y[0]))
            labels.append(pred)
            output_rows.append(row)
            for k, value in diagnostic.items():
                diags[k].append(value)
            positions = np.full((5, BUDGET), -1, dtype=np.int16)
            for j, chosen in enumerate(selected):
                positions[j, :len(chosen)] = chosen
            selection_positions.append(positions)
            audit["tracks"] += 1
            audit["exact_collapsed_observations"] += int(count) - diagnostic["dep_unique_count"]
        if (vi + 1) % 40 == 0 or vi + 1 == len(videos):
            print("videos=%d/%d obs=%d tracks=%d elapsed=%.1fs" %
                  (vi + 1, len(videos), audit["observations"], audit["tracks"], time.time() - started), flush=True)
    rows = np.asarray(output_rows, np.int64)
    assert np.array_equal(rows, expected_rows), "Every reference track must be emitted exactly once in original cache order"
    assert audit["observations"] == int(n_obs[rows].sum())
    for k in ("released_tcc_mismatches", "corrected_tcc_mismatches", "released_full_sumlog_label_mismatches", "corrected_full_sumlog_label_mismatches", "all_methods_single_observation_mismatches"):
        assert audit[k] == 0, (k, audit[k])
    if not args.limit_videos:
        assert audit["tracks"] == 324639 and audit["observations"] == 1529908 and audit["dump_files"] == 36371
        assert inventory.hexdigest() == ref_manifest["dump_filename_size_inventory_sha256"]
    cache = dict(key=keys[rows], global_track_id=gids[rows], n_obs=n_obs[rows], category_ids=category_ids,
                 method_names=np.asarray([m + "." + f for m in MASKS for f in FAMILIES]),
                 mask_names=np.asarray(MASKS), method_families=np.asarray(FAMILIES),
                 pred_category_id=np.asarray(labels, np.int64).T,
                 reference_row=rows,
                 budget_selection_names=np.asarray(FAMILIES[4:9]),
                 budget_selection_positions=np.asarray(selection_positions, np.int16),
                 **{k: np.asarray(v) for k, v in diags.items()})
    assert cache["pred_category_id"].shape == (2 * len(FAMILIES), len(rows))
    assert np.isin(cache["pred_category_id"], category_ids).all()
    assert np.all(cache["dep_unique_count"] >= 1) and np.all(cache["dep_unique_count"] <= cache["n_obs"])
    assert np.all(cache["dep_ess"] >= 1 - 1e-10) and np.all(cache["dep_ess"] <= cache["dep_unique_count"] + 1e-8)
    assert np.all(cache["dep_maxweight"] <= 1 + 1e-12)
    for key, values in cache.items():
        if values.dtype.kind == "f":
            assert np.isfinite(values).all(), key
    temp = args.output_dir / "readout_cache.incomplete.npz"
    np.savez_compressed(temp, **cache)
    temp.replace(target)
    for path, digest in protected.items():
        assert sha256(path) == digest, "Protected input changed: " + path
    manifest = dict(status="PASS_DEPENDENCE_READOUTS_FULL" if not args.limit_videos else "PASS_DEPENDENCE_READOUTS_SMOKE",
                    elapsed_seconds=time.time() - started, inputs_sha256=protected,
                    output_sha256=sha256(target), script_sha256=sha256(__file__),
                    fixed_configuration=dict(log_clip=EPS, budget=BUDGET, seeds=[0, 1, 2],
                                             kernel="((1+L2_normalized_track_feats_cosine)/2)**4", kernel_power=KERNEL_POWER,
                                             dependence_weights="Exact evidence collapse, then w_u=(1/sum_v K_uv)/sum_u(1/sum_v K_uv)",
                                             exact_evidence="Equality of stored float16 track_feats and float32 p_text/p_image/p_fused; detection score, timestamp and box excluded",
                                             fps="Start highest-score observation; repeatedly maximize minimum cosine distance; temporal-order ties",
                                             random="SeedSequence([seed,video_id,local_track_id]), uniform without replacement",
                                             score_weight="Ordinary control only: normalize nonnegative detection scores; uniform if total<=1e-12. Scores are not calibrated reliability probabilities; their observed raw scale is reported in audit.",
                                             entropy_weight="Normalize max(0,1-H(foreground_renormalized_p)/log(296)); uniform if total<=1e-12"),
                    masks="released uses original p_fused; corrected uses fixed public-vocabulary frequency_rare_mask, not per-sample GT",
                    labels="20 rows: mask-major, method-family-minor. tcc released uses det_labels; corrected uses corrected-posterior foreground argmax mode; minimum category ID resolves ties.",
                    frozen_geometry="Only labels and track diagnostics are cached. Original boxes, identities, observations and detection scores are unchanged.",
                    interpretation="track_feats is raw track-head appearance before tracker fusion; shared backbone/training and selection mean it is not independent noise. cls_feats, not used here, is (text_head+img_head)/2. Similarity is an inference-visible dependence proxy, not an estimate of independent covariance. Strict duplicate invariance applies only to exact-evidence rows under the stated signature and dependence method. The three random seeds control sampling variation, not independent data replicates.",
                    audit=audit, self_tests=tests, fields={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in cache.items()},
                    environment={k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    temp_manifest = args.output_dir / "manifest.incomplete.json"
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(manifest_path)
    print(json.dumps({"status": manifest["status"], "audit": audit, "output_sha256": manifest["output_sha256"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
