#!/usr/bin/env python3
"""Check one label perturbation through actual per-sequence TETA raw arrays."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import teta


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    pred = root / "results/cov_val_full_official_a100/tao_track.json"
    ann_path = Path("/raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json")
    receipt_path = root / "results/tces_audit_20261023/argmax_tcc_control_local_teta.json"
    source_hashes = {str(p): sha256(p) for p in (pred, ann_path)}
    ann = json.load(open(ann_path))
    valid_category_ids = {a["category_id"] for a in ann["annotations"]}
    all_categories = [c for c in ann["categories"] if c["id"] in valid_category_ids]
    class_list = [c["name"] for c in all_categories]
    assert len(class_list) == 296
    vid, tid, old, new = 768, 87926, 692, 207
    subset = copy.deepcopy(ann)
    for key in ("videos", "images", "annotations", "tracks"):
        subset[key] = [x for x in ann[key] if (x["id"] if key == "videos" else x["video_id"]) == vid]
    del ann
    rows = json.load(open(pred))
    rows = [x for x in rows if x["video_id"] == vid]
    touched = [i for i, x in enumerate(rows) if x["track_id"] == tid]
    assert len(touched) == 3
    assert all(rows[i]["category_id"] == old for i in touched)
    outputs = []
    evaluate_sequence = importlib.import_module("teta.eval").eval_sequence
    with tempfile.TemporaryDirectory(prefix="argmax_teta_sequence_") as temporary:
        tmp = Path(temporary)
        gp, pp = tmp / "gt.json", tmp / "pred.json"
        json.dump(subset, open(gp, "w"))
        for changed in (False, True):
            current = copy.deepcopy(rows)
            if changed:
                for i in touched:
                    current[i]["category_id"] = new
            json.dump(current, open(pp, "w"))
            ds = teta.datasets.TAO({"GT_FOLDER": str(gp), "TRACKER_SUB_FOLDER": str(pp),
                                    "TRACKERS_TO_EVAL": ["CONTROL"], "PRINT_CONFIG": False})
            # Class inventory and both maps stay identical to the full-dataset protocol,
            # even for classes with no GT in this individual video.
            ds.class_list = class_list
            ds.cls_name2clsid = {c["name"]: c["id"] for c in all_categories}
            ds.clsid2cls_name = {c["id"]: c["name"] for c in all_categories}
            assert len(ds.seq_list) == 1
            outputs.append(evaluate_sequence(ds.seq_list[0], ds, "CONTROL", class_list,
                                             [teta.metrics.TETA()], ["TETA"]))
    equal = {}
    unequal = []
    for cls in class_list:
        left, right = outputs[0][cls]["TETA"][50], outputs[1][cls]["TETA"][50]
        assert left.keys() == right.keys()
        for field in left:
            same = bool(np.array_equal(np.asarray(left[field]), np.asarray(right[field])))
            equal[field] = equal.get(field, True) and same
            if not same:
                unequal.append({"class": cls, "field": field})
    assert not unequal, unequal
    for path, before in source_hashes.items():
        assert sha256(path) == before
    teta_root = Path(teta.__file__).resolve().parent
    evaluator_files = [teta_root / p for p in ("metrics/teta.py", "datasets/tao.py", "eval.py")]
    receipt = {
        "status": "PASS_ACTUAL_LOCAL_TETA_RAW_ARRAY_INVARIANCE",
        "source_sha256": source_hashes,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "evaluator_sha256": {str(p): sha256(p) for p in evaluator_files},
        "video_id": vid, "global_track_id": tid,
        "old_category_id": old, "new_category_id": new,
        "modified_observations": len(touched),
        "video_prediction_observations": len(rows),
        "video_GT_observations": len(subset["annotations"]),
        "video_images": len(subset["images"]),
        "full_dataset_class_inventory_preserved": len(class_list),
        "compared_threshold": 50,
        "all_classes_equal_by_raw_field": equal,
        "unequal_class_fields": unequal,
        "source_files_unchanged": True,
        "scope": (
            "Actual installed teta.eval.eval_sequence was run twice on exact video 768 inputs, "
            "retaining the full 296-class inventory and non-exhaustive TETA defaults. Only category_id "
            "of the specified three prediction observations changed. Every per-class raw field, "
            "including Cls_TP/FN/FP and every localization/association alpha entry, is bitwise equal. "
            "Other sequences receive no changed observations, so this control change is metric-neutral "
            "for the corresponding full-dataset evaluation. This is an equality receipt, not a new method score."
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    temp_receipt = receipt_path.with_name(receipt_path.name + ".incomplete")
    temp_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    temp_receipt.replace(receipt_path)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
