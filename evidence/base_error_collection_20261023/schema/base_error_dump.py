"""Strict, CPU-testable posterior dump schema for isolated base-error collection.

This module uses annotation *image metadata* only to identify frames. It never
matches GT, reads annotations to select rows, or changes model/tracker outputs.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 2
_METADATA = {}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_numpy(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def canonical_filename(value):
    value = str(value).replace("\\", "/")
    for marker in ("data/tao/frames/", "/frames/"):
        if marker in value:
            return value.split(marker, 1)[1]
    return value.lstrip("./")


def annotation_image_metadata(annotation_path, filename):
    path = str(Path(annotation_path).resolve())
    if path not in _METADATA:
        raw = json.load(open(path))
        images = {}
        for image in raw["images"]:
            name = canonical_filename(image["file_name"])
            if name in images:
                raise ValueError("Ambiguous annotation image filename: " + name)
            # No GT labels, boxes, annotations, tracks or negative categories.
            fields = ("id", "video_id", "frame_id", "frame_index", "file_name", "width", "height",
                      "source_image_id", "source_video_id", "source_frame_id", "source_frame_index",
                      "original_image_id", "original_video_id", "original_frame_id", "original_frame_index")
            images[name] = {key: image[key] for key in fields if key in image}
        _METADATA[path] = images, sha256(path)
    images, digest = _METADATA[path]
    name = canonical_filename(filename)
    if name not in images:
        raise KeyError("Dump frame absent from annotation image metadata: " + name)
    return images[name], digest


def posterior_schema(roi_head, posterior_columns):
    """Export the actual active vocabulary/mask rather than infer a GT split."""
    from .models.roi_heads.class_name import tao_validation_used_cate_ids, tao_test_used_cate_ids
    c = int(posterior_columns) - 1
    if roi_head.only_validation_categories:
        ids = np.asarray(tao_validation_used_cate_ids, np.int64)
        name = "tao_validation_used_cate_ids"
        mask = as_numpy(roi_head.validation_index, bool)
    elif roi_head.only_test_categories:
        ids = np.asarray(tao_test_used_cate_ids, np.int64)
        name = "tao_test_used_cate_ids"
        mask = as_numpy(roi_head.validation_index, bool)
    elif not roi_head.custom_classes and int(roi_head.num_classes) == 1203:
        ids = np.arange(1, 1204, dtype=np.int64)
        name = "LVIS_1203_one_based_category_ids"
        mask = as_numpy(roi_head.novel_index, bool)
    else:
        raise ValueError("Collection schema requires explicit known category IDs; unsupported vocabulary")
    assert ids.shape == (c,) and mask.shape == (c + 1,)
    assert len(np.unique(ids)) == c and np.all(ids > 0)
    if not roi_head.ensemble:
        powers = np.ones(c + 1, np.float64)
        mode = "text_only"
    elif roi_head.fixed_lambda is not None:
        powers = np.full(c + 1, float(roi_head.fixed_lambda), np.float64)
        mode = "fixed_lambda"
    else:
        powers = np.where(mask, 1. / 3, 2. / 3)
        mode = "asymmetric_actual_mask"
    return dict(posterior_category_ids=np.r_[ids, np.asarray([-1], np.int64)],
                actual_fusion_mask=mask, fusion_text_exponents=powers,
                fusion_mode=np.asarray(mode), vocabulary_name=np.asarray(name),
                posterior_background_index=np.int64(c),
                only_validation_categories=np.bool_(roi_head.only_validation_categories),
                only_test_categories=np.bool_(roi_head.only_test_categories),
                ensemble=np.bool_(roi_head.ensemble),
                fusion_regime=np.asarray(getattr(roi_head, 'base_error_fusion_regime', 'released_subset_mask_unchanged')))


def build_payload(cache, frame_meta, annotation_path, pre_boxes, pre_labels,
                  post_boxes, post_labels, track_ids, track_features, class_features,
                  keep_mask, tracker_frame_id):
    """Select proposal posteriors by recorded detector and tracker row maps."""
    assert cache is not None, "Missing/stale ROI posterior cache is an error, never a skipped frame"
    assert canonical_filename(cache["filename"]) == canonical_filename(frame_meta["filename"])
    identity, annotation_sha = annotation_image_metadata(annotation_path, frame_meta["filename"])
    assert int(identity["frame_id"]) == int(tracker_frame_id), "Pipeline frame ordinal differs from annotation"
    schema = {key: as_numpy(value) for key, value in cache["schema"].items()}
    c = len(schema["posterior_category_ids"]) - 1
    pre_boxes, pre_labels = as_numpy(pre_boxes, np.float32), as_numpy(pre_labels, np.int64)
    n_pre = len(pre_labels)
    assert pre_boxes.shape == (n_pre, 5)
    flat = as_numpy(cache["det_inds"][0], np.int64)
    assert flat.shape == (n_pre,)
    proposal_rows = flat // c
    foreground_cols = flat % c
    assert np.all(flat >= 0)
    if n_pre:
        assert np.all(schema["posterior_category_ids"][foreground_cols] == pre_labels + 1)
    if keep_mask is None:
        assert n_pre == 0, "Nonempty detector output requires tracker keep_mask"
        keep = np.zeros(0, dtype=bool)
    else:
        keep = as_numpy(keep_mask, bool)
        assert keep.shape == (n_pre,)
    local_rows = np.flatnonzero(keep)
    n = len(local_rows)
    if post_boxes is None:
        assert n == 0
        post_boxes = np.zeros((0, 5), np.float32)
        post_labels, track_ids = np.zeros(0, np.int64), np.zeros(0, np.int64)
    else:
        post_boxes, post_labels, track_ids = (as_numpy(post_boxes, np.float32),
                                            as_numpy(post_labels, np.int64), as_numpy(track_ids, np.int64))
    assert post_boxes.shape == (n, 5) and post_labels.shape == track_ids.shape == (n,)
    assert np.array_equal(pre_labels[keep], post_labels)
    # Tracker may smooth scores. Geometry and label selection must still agree.
    assert np.array_equal(pre_boxes[keep, :4], post_boxes[:, :4])
    if track_features is None:
        assert n == n_pre == 0
        tf, cf = np.zeros((0, 256), np.float16), np.zeros((0, 512), np.float16)
    else:
        tf, cf = as_numpy(track_features, np.float16)[keep], as_numpy(class_features, np.float16)[keep]
        assert tf.shape == (n, 256) and cf.shape == (n, 512)
    payload = dict(schema_version=np.int64(SCHEMA_VERSION), **schema,
                   det_bboxes=post_boxes, det_labels=post_labels, track_ids=track_ids,
                   pred_category_id=post_labels + 1, emitted_track_mask=track_ids >= 0,
                   track_feats=tf, cls_feats=cf,
                   pretracker_det_bboxes=pre_boxes, pretracker_det_labels=pre_labels,
                   tracker_keep_mask=keep, pretracker_det_row=local_rows.astype(np.int64),
                   roi_flat_class_index=flat[keep], roi_proposal_row=proposal_rows[keep],
                   roi_foreground_col=foreground_cols[keep],
                   pretracker_roi_flat_class_index=flat,
                   pretracker_roi_proposal_row=proposal_rows,
                   frame_id=np.int64(tracker_frame_id), tracker_frame_id=np.int64(tracker_frame_id),
                   image_id=np.int64(identity["id"]), video_id=np.int64(identity["video_id"]),
                   annotation_frame_id=np.int64(identity["frame_id"]),
                   frame_index=np.int64(identity.get("frame_index", -1)),
                   filename=np.asarray(str(frame_meta["filename"])),
                   canonical_filename=np.asarray(canonical_filename(frame_meta["filename"])),
                   ori_shape=as_numpy(frame_meta["ori_shape"], np.int64),
                   image_metadata_json=np.asarray(json.dumps(identity, sort_keys=True)),
                   annotation_path=np.asarray(str(Path(annotation_path).resolve())),
                   annotation_sha256=np.asarray(annotation_sha),
                   observation_count=np.int64(n), detector_observation_count=np.int64(n_pre),
                   emitted_track_count=np.int64(np.count_nonzero(track_ids >= 0)),
                   feature_provenance=np.asarray("track_feats=raw track-head output before tracker fusion; cls_feats=(text_head+image_head)/2; shared backbone/training/selection, not independent noise"),
                   selection_provenance=np.asarray("ROI flattened NMS class index -> proposal row; tracker keep_mask -> pretracker detection row; all retained tracker rows including id=-1 recorded; emitted_track_mask marks track2result rows"))
    proposal_count = int(as_numpy(cache["p_fused"]).shape[0])
    payload["proposal_count"] = np.int64(proposal_count)
    assert not n_pre or int(proposal_rows.max()) < proposal_count
    for branch in ("p_text", "p_image", "p_fused"):
        assert cache[branch] is not None, "This collection requires the ensemble branch"
        p = as_numpy(cache[branch], np.float32)
        assert p.shape == (proposal_count, c + 1)
        payload[branch] = p[proposal_rows[keep]]
    validate_payload(payload)
    return payload


def validate_payload(data):
    """Validate one saved schema without importing torch/model packages."""
    assert int(data["schema_version"]) == SCHEMA_VERSION
    ids = as_numpy(data["posterior_category_ids"], np.int64)
    c = len(ids) - 1
    assert ids.ndim == 1 and ids[-1] == -1 and np.all(ids[:-1] > 0)
    assert len(np.unique(ids)) == len(ids) and int(data["posterior_background_index"]) == c
    mask, power = as_numpy(data["actual_fusion_mask"], bool), as_numpy(data["fusion_text_exponents"], np.float64)
    assert mask.shape == power.shape == (c + 1,) and np.isfinite(power).all()
    assert np.all((power >= 0) & (power <= 1))
    n, n_pre = int(data["observation_count"]), int(data["detector_observation_count"])
    boxes, labels, tids = data["det_bboxes"], data["det_labels"], data["track_ids"]
    assert boxes.shape == (n, 5) and labels.shape == tids.shape == (n,)
    assert boxes.dtype == np.float32 and labels.dtype == tids.dtype == np.int64
    assert np.isfinite(boxes).all() and np.all(boxes[:, 2:4] >= boxes[:, :2])
    assert np.isin(labels + 1, ids[:-1]).all() and np.array_equal(labels + 1, data["pred_category_id"])
    assert np.array_equal(tids >= 0, data["emitted_track_mask"])
    assert int(data["emitted_track_count"]) == np.count_nonzero(tids >= 0)
    assert len(np.unique(tids[tids >= 0])) == np.count_nonzero(tids >= 0)
    keep = data["tracker_keep_mask"]
    assert keep.shape == (n_pre,) and keep.dtype == bool and int(keep.sum()) == n
    assert np.array_equal(data["pretracker_det_row"], np.flatnonzero(keep))
    assert data["pretracker_det_bboxes"].shape == (n_pre, 5)
    assert np.array_equal(data["pretracker_det_labels"][keep], labels)
    assert np.array_equal(data["pretracker_det_bboxes"][keep, :4], boxes[:, :4])
    flat = data["pretracker_roi_flat_class_index"]
    assert flat.shape == (n_pre,) and np.all(flat >= 0)
    assert np.array_equal(flat // c, data["pretracker_roi_proposal_row"])
    assert np.array_equal(flat[keep], data["roi_flat_class_index"])
    assert np.array_equal(flat[keep] // c, data["roi_proposal_row"])
    assert np.array_equal(flat[keep] % c, data["roi_foreground_col"])
    assert np.array_equal(ids[(flat % c).astype(np.int64)], data["pretracker_det_labels"] + 1)
    assert not n_pre or int((flat // c).max()) < int(data["proposal_count"])
    assert data["track_feats"].shape == (n, 256) and data["cls_feats"].shape == (n, 512)
    assert data["track_feats"].dtype == data["cls_feats"].dtype == np.float16
    assert np.isfinite(data["track_feats"]).all() and np.isfinite(data["cls_feats"]).all()
    for branch in ("p_text", "p_image", "p_fused"):
        p = data[branch]
        assert p.shape == (n, c + 1) and p.dtype == np.float32
        assert np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1 + 1e-6)
        assert np.allclose(p.sum(1, dtype=np.float64), 1, atol=2e-6, rtol=0)
    assert int(data["video_id"]) >= 0 and int(data["image_id"]) >= 0
    assert int(data["frame_id"]) == int(data["tracker_frame_id"]) == int(data["annotation_frame_id"])
    assert canonical_filename(data["filename"].item()) == data["canonical_filename"].item()
    rebuilt = data["p_text"].astype(np.float64) ** power * data["p_image"].astype(np.float64) ** (1 - power)
    rebuilt /= np.maximum(rebuilt.sum(1, keepdims=True), 1e-12)
    error = float(np.max(np.abs(rebuilt - data["p_fused"]))) if n else 0.
    assert error < 2e-6
    return dict(observations=n, emitted_observations=int(data["emitted_track_count"]),
                posterior_columns=c + 1, fusion_max_abs_error=error, empty_frame=n == 0)


def write_frame_dump(output_dir, payload):
    """One atomic NPZ plus receipt for every frame, including empty frames."""
    summary = validate_payload(payload)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = "v%d_i%d_f%d" % (int(payload["video_id"]), int(payload["image_id"]), int(payload["frame_id"]))
    npz, receipt = out / (key + ".npz"), out / (key + ".receipt.json")
    if npz.exists() or receipt.exists():
        raise FileExistsError("Refuse to overwrite a collected frame: " + key)
    temp = out / (key + ".incomplete.npz")
    np.savez_compressed(temp, **payload)
    temp.replace(npz)
    info = dict(status="PASS_FRAME_DUMP", schema_version=SCHEMA_VERSION, filename=payload["filename"].item(),
                video_id=int(payload["video_id"]), image_id=int(payload["image_id"]), frame_id=int(payload["frame_id"]),
                file=npz.name, sha256=sha256(npz), bytes=npz.stat().st_size,
                annotation_sha256=payload["annotation_sha256"].item(), **summary)
    temp_receipt = out / (key + ".receipt.incomplete.json")
    temp_receipt.write_text(json.dumps(info, indent=2) + "\n")
    temp_receipt.replace(receipt)
    return npz
