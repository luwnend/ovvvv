#!/usr/bin/env python3
"""Aggregate every emitted tracker observation without GT matching/filtering.

Only annotation image/category metadata is read. Ground-truth annotations and
tracks are deliberately discarded before aggregation. Official JSON is used to
audit the emitted boxes, scores, global IDs and the exact pandas-mode TCC rule.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_filename(value):
    value = str(value).replace("\\", "/")
    marker = "data/tao/frames/"
    if marker in value:
        value = value.split(marker, 1)[1]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default="results/cov_tces_v3_fullval")
    parser.add_argument("--ann", default="/raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json")
    parser.add_argument("--class-names", default="work/covtrack_tces_v2/ovtrack/models/roi_heads/class_name.py")
    parser.add_argument("--official", default="results/cov_tces_v3_fullval_eval/tao_track.json")
    parser.add_argument("--output-dir", default="results/tces_audit_20261023")
    parser.add_argument("--log-eps", type=float, default=1e-12)
    args = parser.parse_args()
    started = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256(path) for name, path in
                     (("annotation", args.ann), ("class_names", args.class_names),
                      ("official", args.official))}
    classes = runpy.run_path(args.class_names)
    category_ids = np.asarray(classes["tao_validation_used_cate_ids"], dtype=np.int64)
    fusion_mask = np.asarray(classes["validation_novel_index"], dtype=bool)
    c = len(category_ids)
    assert np.all(np.diff(category_ids) > 0), "TCC tie needs category-ID order"
    assert fusion_mask.shape == (c + 1,)
    cat_to_col = {int(cid): col for col, cid in enumerate(category_ids)}
    raw_ann = json.load(open(args.ann))
    ann = {k: raw_ann[k] for k in ("images", "videos", "categories")}
    del raw_ann
    category_by_id = {int(row["id"]): row for row in ann["categories"]}
    frequency = np.asarray([category_by_id[int(cid)]["frequency"] for cid in category_ids])
    category_names = np.asarray([category_by_id[int(cid)]["name"] for cid in category_ids])
    rare_mask = frequency == "r"
    images_by_video = defaultdict(list)
    image_ids = set()
    for im in ann["images"]:
        assert im["id"] not in image_ids
        image_ids.add(im["id"])
        images_by_video[int(im["video_id"])].append(im)
    for rows in images_by_video.values():
        rows.sort(key=lambda row: int(row["frame_id"]))
        assert len({int(row["frame_id"]) for row in rows}) == len(rows)
        assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    videos = [int(row["id"]) for row in ann["videos"]]
    all_dump_paths = {p.name: p for p in Path(args.dump_dir).glob("*.npz")}
    assert len(all_dump_paths) > 0
    official = json.load(open(args.official))
    official_by_image = defaultdict(dict)
    official_counts = defaultdict(int)
    official_categories = {}
    for row in official:
        image_id, track_id = int(row["image_id"]), int(row["track_id"])
        assert image_id in image_ids
        assert track_id not in official_by_image[image_id], "Duplicate official track per frame"
        official_by_image[image_id][track_id] = row
        official_counts[track_id] += 1
        if track_id in official_categories:
            assert official_categories[track_id] == int(row["category_id"])
        official_categories[track_id] = int(row["category_id"])
    official_n = len(official)
    del official
    stat = dict(dump_files=len(all_dump_paths), annotation_images=len(ann["images"]),
                annotation_videos=len(videos), official_observations=official_n,
                processed_files=0, observations=0, zero_observation_dump_files=0,
                argmax_det_label_mismatches=0, fused_formula_max_abs_error=0.,
                fused_formula_mean_abs_error_sum=0., fused_formula_values=0,
                box_max_abs_error=0., score_max_abs_error=0.,
                first_frame_label_disagrees_with_tcc=0, majority_tied_tracks=0,
                tcc_label_mismatches=0, compared_official_observations=0)
    prob_stats = {b: dict(min=float("inf"), max=-float("inf"), row_sum_min=float("inf"),
                          row_sum_max=-float("inf"), row_sum_max_abs_error=0.,
                          background_min=float("inf"), background_max=-float("inf"),
                          background_nonzero_rows=0, zeros=0, nonfinite_values=0)
                  for b in ("p_fused", "p_text", "p_image")}
    fields = defaultdict(list)
    missing_images = []
    seen_dump_names = set()
    seen_global_ids = set()
    dump_inventory = hashlib.sha256()
    offset = 0
    for video_index, video_id in enumerate(videos):
        blocks = defaultdict(list)
        for im in images_by_video[video_id]:
            relative_name = canonical_filename(im["file_name"])
            dump_name = ("data/tao/frames/" + relative_name).replace("/", "-").replace(".jpg", "") + ".npz"
            path = all_dump_paths.get(dump_name)
            if path is None:
                assert not official_by_image.get(int(im["id"])), "Missing dump with official predictions"
                missing_images.append({k: im[k] for k in ("id", "video_id", "frame_id", "file_name")})
                continue
            seen_dump_names.add(dump_name)
            dump_inventory.update((dump_name + "\t" + str(path.stat().st_size) + "\n").encode())
            with np.load(path, allow_pickle=False) as data:
                assert canonical_filename(data["filename"].item()) == relative_name
                assert int(data["frame_id"]) == int(im["frame_id"])
                ids = data["track_ids"]
                labels = data["det_labels"]
                boxes = data["det_bboxes"]
                probs = {b: data[b].astype(np.float64) for b in prob_stats}
                n = len(ids)
                assert ids.dtype.kind in "iu" and ids.shape == (n,)
                assert labels.dtype.kind in "iu" and labels.shape == (n,)
                assert boxes.shape == (n, 5) and np.isfinite(boxes).all()
                assert len(np.unique(ids)) == n and np.all(ids >= 0)
                assert np.isfinite(data["track_feats"]).all() and data["track_feats"].shape[0] == n
                assert np.isfinite(data["cls_feats"]).all() and data["cls_feats"].shape[0] == n
                assert len(official_by_image.get(int(im["id"]), {})) == n
                label_col = np.asarray([cat_to_col[int(label) + 1] for label in labels], dtype=np.int64)
                stat["processed_files"] += 1
                stat["observations"] += n
                stat["zero_observation_dump_files"] += int(n == 0)
                for branch, p in probs.items():
                    assert p.shape == (n, c + 1), (path, branch, p.shape)
                    ps = prob_stats[branch]
                    ps["nonfinite_values"] += int((~np.isfinite(p)).sum())
                    assert np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1 + 1e-6)
                    if n:
                        rs = p.sum(axis=1)
                        ps["min"] = min(ps["min"], float(p.min()))
                        ps["max"] = max(ps["max"], float(p.max()))
                        ps["row_sum_min"] = min(ps["row_sum_min"], float(rs.min()))
                        ps["row_sum_max"] = max(ps["row_sum_max"], float(rs.max()))
                        ps["row_sum_max_abs_error"] = max(ps["row_sum_max_abs_error"], float(np.abs(rs - 1).max()))
                        ps["background_min"] = min(ps["background_min"], float(p[:, -1].min()))
                        ps["background_max"] = max(ps["background_max"], float(p[:, -1].max()))
                        ps["background_nonzero_rows"] += int(np.count_nonzero(p[:, -1]))
                        ps["zeros"] += int((p == 0).sum())
                        assert np.allclose(rs, 1, rtol=0, atol=2e-6)
                    blocks[branch].append(p)
                if not n:
                    continue
                # Official asymmetric geometric fusion uses its own historical
                # mask; this is NOT the GT frequency rare/base split.
                text_power = np.where(fusion_mask, 1. / 3, 2. / 3)
                rebuilt = probs["p_text"] ** text_power * probs["p_image"] ** (1 - text_power)
                rebuilt /= np.maximum(rebuilt.sum(axis=1, keepdims=True), 1e-12)
                delta = np.abs(rebuilt - probs["p_fused"])
                stat["fused_formula_max_abs_error"] = max(stat["fused_formula_max_abs_error"], float(delta.max()))
                stat["fused_formula_mean_abs_error_sum"] += float(delta.sum())
                stat["fused_formula_values"] += delta.size
                assert float(delta.max()) < 2e-6, "Original fusion reconstruction must pass before corrected-mask diagnostic"
                corrected_text_power = np.where(np.concatenate([rare_mask, [False]]), 1. / 3, 2. / 3)
                corrected = probs["p_text"] ** corrected_text_power * probs["p_image"] ** (1 - corrected_text_power)
                corrected /= np.maximum(corrected.sum(axis=1, keepdims=True), 1e-12)
                assert np.isfinite(corrected).all() and np.allclose(corrected.sum(1), 1, atol=2e-6, rtol=0)
                blocks["p_corrected_mask"].append(corrected)
                stat["argmax_det_label_mismatches"] += int(np.count_nonzero(probs["p_fused"][:, :c].argmax(1) != label_col))
                frame_official = official_by_image[int(im["id"])]
                for j, local_id in enumerate(ids):
                    global_id = offset + int(local_id)
                    row = frame_official[global_id]
                    assert int(row["video_id"]) == video_id
                    xywh = np.asarray([boxes[j, 0], boxes[j, 1], boxes[j, 2] - boxes[j, 0], boxes[j, 3] - boxes[j, 1]])
                    box_error = float(np.max(np.abs(xywh - np.asarray(row["bbox"]))))
                    score_error = abs(float(boxes[j, -1]) - float(row["score"]))
                    stat["box_max_abs_error"] = max(stat["box_max_abs_error"], box_error)
                    stat["score_max_abs_error"] = max(stat["score_max_abs_error"], score_error)
                    assert box_error <= 1e-4 and score_error <= 1e-7, (path, local_id, box_error, score_error)
                    stat["compared_official_observations"] += 1
                blocks["ids"].append(ids)
                blocks["label_col"].append(label_col)
                blocks["scores"].append(boxes[:, -1].astype(np.float64))
                blocks["frame_id"].append(np.full(n, int(im["frame_id"]), dtype=np.int64))
                blocks["image_id"].append(np.full(n, int(im["id"]), dtype=np.int64))
        if not blocks["ids"]:
            continue
        data = {k: np.concatenate(v, axis=0) for k, v in blocks.items()}
        local_ids, first, inverse, n_obs = np.unique(data["ids"], return_index=True, return_inverse=True, return_counts=True)
        n_tracks = len(local_ids)
        global_ids = local_ids + offset
        assert not seen_global_ids.intersection(global_ids.tolist())
        seen_global_ids.update(global_ids.tolist())
        fields["key"].append(np.stack([np.full(n_tracks, video_id, dtype=np.int64), local_ids], axis=1))
        fields["global_track_id"].append(global_ids)
        fields["n_obs"].append(n_obs)
        fields["first_label_col"].append(data["label_col"][first])
        fields["first_label"].append(category_ids[data["label_col"][first]])
        for source in ("frame_id", "image_id"):
            lo = np.full(n_tracks, np.iinfo(np.int64).max, dtype=np.int64)
            hi = np.full(n_tracks, -1, dtype=np.int64)
            np.minimum.at(lo, inverse, data[source])
            np.maximum.at(hi, inverse, data[source])
            fields["first_" + source].append(lo)
            fields["last_" + source].append(hi)
        counts = np.zeros((n_tracks, c), dtype=np.uint32)
        np.add.at(counts, (inverse, data["label_col"]), 1)
        fields["majority_counts"].append(counts)
        tcc_col = counts.argmax(axis=1)  # sorted category IDs => pandas .mode()[0]
        tcc_categories = category_ids[tcc_col]
        fields["tcc_category_id"].append(tcc_categories)
        fields["tcc_label_col"].append(tcc_col)
        stat["majority_tied_tracks"] += int(np.count_nonzero((counts == counts.max(axis=1, keepdims=True)).sum(1) > 1))
        stat["first_frame_label_disagrees_with_tcc"] += int(np.count_nonzero(data["label_col"][first] != tcc_col))
        for j, global_id in enumerate(global_ids):
            assert official_counts[int(global_id)] == int(n_obs[j])
            stat["tcc_label_mismatches"] += int(official_categories[int(global_id)] != int(tcc_categories[j]))
        s = np.zeros(n_tracks, np.float64)
        np.add.at(s, inverse, data["scores"])
        fields["score_sum"].append(s)
        fields["score_mean"].append(s / n_obs)
        for kind, init, operation in (("min", np.inf, np.minimum), ("max", -np.inf, np.maximum)):
            s = np.full(n_tracks, init, dtype=np.float64)
            operation.at(s, inverse, data["scores"])
            fields["score_" + kind].append(s)
        for branch in list(prob_stats) + ["p_corrected_mask"]:
            p = data[branch]
            for suffix, values in (("sumprob", p), ("sumlog", np.log(np.clip(p, args.log_eps, None)))):
                sums = np.zeros((n_tracks, c + 1), dtype=np.float64)
                np.add.at(sums, inverse, values)
                fields[branch + "_" + suffix].append(sums.astype(np.float32))
            argmax_counts = np.zeros((n_tracks, c), dtype=np.uint32)
            np.add.at(argmax_counts, (inverse, p[:, :c].argmax(1)), 1)
            fields[branch + "_argmax_counts"].append(argmax_counts)
            fg = p[:, :c] / np.maximum(p[:, :c].sum(axis=1, keepdims=True), args.log_eps)
            entropy = -(fg * np.log(np.clip(fg, args.log_eps, None))).sum(axis=1)
            weights = np.maximum(1 - entropy / np.log(c), 0)
            for suffix, values in (("entropy_sum", entropy), ("entropy_weight_sum", weights)):
                sums = np.zeros(n_tracks, dtype=np.float64)
                np.add.at(sums, inverse, values)
                fields[branch + "_" + suffix].append(sums)
            sums = np.zeros((n_tracks, c + 1), dtype=np.float64)
            np.add.at(sums, inverse, weights[:, None] * p)
            fields[branch + "_entropy_weighted_sumprob"].append(sums.astype(np.float32))
        agreement = (data["p_text"][:, :c].argmax(1) == data["p_image"][:, :c].argmax(1))
        agreement_count = np.zeros(n_tracks, dtype=np.int64)
        np.add.at(agreement_count, inverse, agreement)
        fields["branch_argmax_agreement_count"].append(agreement_count)
        offset += int(local_ids.max()) + 1
        if (video_index + 1) % 40 == 0 or video_index + 1 == len(videos):
            print("videos=%d/%d files=%d obs=%d tracks=%d elapsed=%.1fs" %
                  (video_index + 1, len(videos), stat["processed_files"], stat["observations"],
                   sum(len(x) for x in fields["key"]), time.time() - started), flush=True)
    assert seen_dump_names == set(all_dump_paths), "Unmapped dump files"
    assert stat["observations"] == official_n == stat["compared_official_observations"]
    assert stat["tcc_label_mismatches"] == 0
    assert seen_global_ids == set(official_categories)
    assert stat["fused_formula_max_abs_error"] < 2e-6
    cache = {k: np.concatenate(v, axis=0) for k, v in fields.items()}
    cache.update(category_ids=category_ids, category_names=category_names,
                 category_frequency=frequency, frequency_rare_mask=rare_mask,
                 fusion_validation_novel_index=fusion_mask,
                 posterior_category_ids=np.concatenate([category_ids, np.array([-1], dtype=np.int64)]))
    assert cache["key"].shape[0] == len(set(map(tuple, cache["key"].tolist())))
    assert int(cache["n_obs"].sum()) == official_n
    for branch in list(prob_stats) + ["p_corrected_mask"]:
        assert np.array_equal(cache[branch + "_argmax_counts"].sum(1), cache["n_obs"])
        assert np.allclose(cache[branch + "_sumprob"].sum(1), cache["n_obs"], atol=2e-3, rtol=2e-6)
    assert np.array_equal(cache["majority_counts"].sum(1), cache["n_obs"])
    for name, value in cache.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), name
    cache_path = out / "track_cache.npz"
    temp_cache = out / "track_cache.incomplete.npz"
    np.savez_compressed(temp_cache, **cache)
    temp_cache.replace(cache_path)
    stat["fused_formula_mean_abs_error"] = stat.pop("fused_formula_mean_abs_error_sum") / stat["fused_formula_values"]
    stat.update(tracks=int(len(cache["key"])), missing_images_without_official_predictions=missing_images,
                short_track_counts={str(n): int(np.count_nonzero(cache["n_obs"] == n)) for n in (1, 2, 3, 4, 5)},
                n_obs_min=int(cache["n_obs"].min()), n_obs_max=int(cache["n_obs"].max()),
                argmax_det_label_mismatch_rate=stat["argmax_det_label_mismatches"] / stat["observations"])
    # Protected source artifacts are read-only, and their hashes must remain stable.
    assert source_hashes["annotation"] == sha256(args.ann)
    assert source_hashes["class_names"] == sha256(args.class_names)
    assert source_hashes["official"] == sha256(args.official)
    manifest = dict(status="PASS_FULL_TRACKER_POSTERIOR_CACHE_AUDIT", command=sys.argv,
                    elapsed_seconds=time.time() - started, python=sys.executable,
                    environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    inputs={"dump_dir": str(Path(args.dump_dir).resolve()), "annotation": str(Path(args.ann).resolve()),
                            "class_names": str(Path(args.class_names).resolve()), "official": str(Path(args.official).resolve())},
                    input_sha256=source_hashes, dump_filename_size_inventory_sha256=dump_inventory.hexdigest(),
                    output_sha256=sha256(cache_path), output_bytes=cache_path.stat().st_size,
                    aggregation="All dumped tracker outputs, no GT matching or annotation-based filtering. Per-video observations sorted by frame_id, float64 accumulation; posterior matrices saved float32.",
                    track_key="(video_id, local_track_id)", posterior_columns="tao_validation_used_cate_ids followed by background (-1 sentinel)",
                    majority="Counts use det_labels+1; pandas category_id.mode()[0] is lowest category_id on ties. Argmax counts are stored separately for each posterior branch.",
                    sumlog="sum(log(max(p, %.17g))) over all observations, including background" % args.log_eps,
                    reliability="entropy computed on renormalized 296-column foreground; weight=max(0,1-H/log(296)); entropy_weighted_sumprob accumulates weight times original 297-column posterior",
                    fusion="normalized text**(1/3)*image**(2/3) where actual validation_novel_index is true, text**(2/3)*image**(1/3) otherwise; source branch row normalization cancels during fused row normalization",
                    corrected_mask_diagnostic="p_corrected_mask uses the same fixed geometric fusion formula with frequency=='r' foreground columns and false background; only public category vocabulary metadata is used, with no per-sample GT. Boxes and IDs stay fixed. This is an implementation diagnostic, not a novelty claim.",
                    fusion_mask_true_count=int(fusion_mask.sum()), frequency_rare_count=int(rare_mask.sum()),
                    fusion_mask_frequency_rare_disagreement_columns=int(np.count_nonzero(fusion_mask[:c] != rare_mask)),
                    caution="Fusion's historical validation_novel_index is a separate mask from GT category frequency. Use category_frequency or frequency_rare_mask for rare/base metrics.",
                    audit=stat, probability_schema=prob_stats,
                    fields={k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in cache.items()})
    temp_manifest = out / "manifest.incomplete.json"
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(out / "manifest.json")
    print(json.dumps({"status": manifest["status"], "audit": stat, "cache_bytes": manifest["output_bytes"],
                      "elapsed_seconds": manifest["elapsed_seconds"], "fields": sorted(cache)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
