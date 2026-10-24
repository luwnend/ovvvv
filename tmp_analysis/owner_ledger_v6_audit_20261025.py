#!/usr/bin/env python3
"""CPU-only independent audit of EvidenceOwnershipLedger snapshot c0ea9872.

The source is executed only from the frozen /raid snapshot.  No CUDA, tracker,
dataset, occlusion collection, or training is used.
"""
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path("/home/chenhc/ov/OVMOT_RESEARCH")
CURRENT = ROOT / "tmp_analysis/evidence_ownership_head.py"
SNAPSHOT = Path(
    "/raid/datasets/chc_data/partwhole_review_20261025_v1/"
    "evidence_ownership_head_c0ea9872.py"
)
EXPECTED = "c0ea9872ea22e50615093394c5923ba7b59e3af662dcb6a5caea075de861bc9d"
OUT_JSON = ROOT / "tmp_analysis/owner_ledger_v6_audit_evidence_20261025.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


main_before = sha(CURRENT)
snapshot_before = sha(SNAPSHOT)
if main_before != EXPECTED or snapshot_before != EXPECTED:
    raise SystemExit(
        "binding failure: main=%s snapshot=%s expected=%s"
        % (main_before, snapshot_before, EXPECTED)
    )

spec = importlib.util.spec_from_file_location("owner_ledger_v6_subject", SNAPSHOT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
Ledger = subject.EvidenceOwnershipLedger
StableSpatialNorm = subject.StableSpatialNorm

torch.set_num_threads(1)
torch.manual_seed(20261025)
results = []


def record(round_no, name, passed, detail, counterexample=False):
    results.append(
        {
            "round": int(round_no),
            "name": name,
            "passed": bool(passed),
            "counterexample": bool(counterexample),
            "detail": detail,
        }
    )
    print(("PASS" if passed else "FAIL"), "r%d" % round_no, name)


def probe(round_no, name, fn, counterexample=True):
    try:
        passed, detail = fn()
    except BaseException as exc:
        passed, detail = False, {
            "exception": "%s: %s" % (type(exc).__name__, exc)
        }
    record(round_no, name, passed, detail, counterexample=counterexample)


def rejection(fn, kind=None, text=None):
    try:
        fn()
    except BaseException as exc:
        return (
            (kind is None or isinstance(exc, kind))
            and (text is None or text in str(exc)),
            "%s: %s" % (type(exc).__name__, exc),
        )
    return False, "accepted without exception"


class ConstMask(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values.to(x).view(-1, 1, 1, 1).expand(
            -1, 1, x.shape[-2], x.shape[-1]
        )


def make_model(
    classes=5,
    dim=8,
    hidden=8,
    max_children_per_parent=8,
    max_edges=64,
):
    return Ledger(
        dim=dim,
        hidden=hidden,
        class_width=classes,
        max_children_per_parent=max_children_per_parent,
        max_edges=max_edges,
    )


def make_args(
    classes=5,
    dim=8,
    height=2,
    width=2,
    n_parent=1,
    n_child=1,
    n_edge=1,
    time_dtype=torch.int64,
):
    if n_child < 1:
        raise ValueError("helper requires at least one child")
    parent = torch.randn(n_parent, dim, height, width)
    child = torch.randn(n_child, dim, height, width)
    edge_parent = (
        torch.arange(n_edge, dtype=torch.long) % n_parent
        if n_edge
        else torch.empty(0, dtype=torch.long)
    )
    edge_child = (
        torch.arange(n_edge, dtype=torch.long) % n_child
        if n_edge
        else torch.empty(0, dtype=torch.long)
    )
    overlap = torch.ones(n_edge, height, width)
    posterior = torch.softmax(torch.randn(n_child, classes), dim=-1)
    embedding = torch.randn(classes, dim)
    geometry = torch.randn(n_edge, 5)
    parent_time = torch.full((n_parent,), 10, dtype=time_dtype)
    child_time = torch.full((n_edge,), 9, dtype=time_dtype)
    return [
        parent,
        child,
        edge_parent,
        edge_child,
        overlap,
        posterior,
        embedding,
        geometry,
        parent_time,
        child_time,
    ]


def base_args(**kwargs):
    return make_args(**kwargs)


def output_ok(out, shape):
    return (
        tuple(out.shape) == tuple(shape)
        and bool(torch.isfinite(out).all())
        and bool((out <= 1e-7).all())
    )


def freeze_relation(model):
    with torch.no_grad():
        for parameter in model.relation.parameters():
            parameter.zero_()


def copy_learned_modules(destination, source):
    destination.child_query.load_state_dict(source.child_query.state_dict())
    destination.relation.load_state_dict(source.relation.state_dict())


def normalize_matrix_case(n_parent, height, width, training):
    model = make_model()
    model.train(training)
    args = base_args(
        n_parent=n_parent,
        n_child=max(n_parent, 1),
        n_edge=max(n_parent, 1),
        height=height,
        width=width,
    )
    with torch.no_grad():
        out = model(*args)
    return output_ok(out, (n_parent, 5)), {
        "mode": "train" if training else "eval",
        "n": n_parent,
        "height": height,
        "width": width,
        "shape": list(out.shape),
        "finite": bool(torch.isfinite(out).all()),
        "nonpositive": bool((out <= 1e-7).all()),
        "min": float(out.min()),
        "max": float(out.max()),
    }


def parity_case(n_parent):
    model = make_model()
    args = base_args(
        n_parent=n_parent, n_child=n_parent, n_edge=n_parent, height=1, width=1
    )
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.train(True)
    with torch.no_grad():
        train_out = model(*args).clone()
    model.train(False)
    with torch.no_grad():
        eval_out = model(*args).clone()
    state_equal = all(
        torch.equal(state_before[name], value)
        for name, value in model.state_dict().items()
    )
    difference = float((train_out - eval_out).abs().max())
    return bool(state_equal and difference == 0.0), {
        "n": n_parent,
        "train_eval_max_abs_diff": difference,
        "state_unchanged": state_equal,
        "train": train_out.tolist(),
        "eval": eval_out.tolist(),
    }


def batch_independence(training):
    model = make_model(max_children_per_parent=1, max_edges=2)
    args = base_args(
        n_parent=2, n_child=2, n_edge=2, height=1, width=1
    )
    args[2] = torch.tensor([0, 1], dtype=torch.long)
    args[3] = torch.tensor([0, 1], dtype=torch.long)
    model.train(training)
    with torch.no_grad():
        batched = model(*args).clone()
        singleton_rows = []
        for index in (0, 1):
            row_args = list(args)
            for position in (0, 1, 5, 7, 8, 9):
                row_args[position] = args[position][index:index + 1].clone()
            row_args[2] = torch.tensor([0], dtype=torch.long)
            row_args[3] = torch.tensor([0], dtype=torch.long)
            row_args[4] = args[4][index:index + 1].clone()
            singleton_rows.append(model(*row_args).clone())
    singleton = torch.cat(singleton_rows, dim=0)
    difference = float((singleton - batched).abs().max())
    return bool(torch.allclose(singleton, batched, atol=1e-6)), {
        "mode": "train" if training else "eval",
        "batch_output": batched.tolist(),
        "singleton_output": singleton.tolist(),
        "max_abs_diff": difference,
        "tolerance": 1e-6,
    }


def constant_features_case(which):
    model = make_model()
    model.train()
    args = base_args(n_parent=1, n_child=1, n_edge=1, height=1, width=1)
    value = 0.0 if which == "zero" else 1.0
    args[0] = torch.full_like(args[0], value)
    args[1] = torch.full_like(args[1], value)
    args[5] = torch.full_like(args[5], 1.0 / 5.0)
    args[6] = torch.full_like(args[6], value)
    out = model(*args)
    if which == "zero":
        expected_zero = bool(torch.equal(out, torch.zeros_like(out)))
    else:
        expected_zero = True
    return output_ok(out, (1, 5)) and expected_zero, {
        "case": which,
        "output": out.tolist(),
        "finite": bool(torch.isfinite(out).all()),
        "nonpositive": bool((out <= 1e-7).all()),
        "expected_zero": expected_zero,
    }


def default_vocab_case(backward=False):
    model = Ledger(
        dim=256,
        hidden=128,
        max_children_per_parent=1,
        max_edges=1,
    )
    model.train()
    parent = torch.randn(1, 256, 1, 1, requires_grad=True)
    child = torch.randn(1, 256, 1, 1, requires_grad=True)
    posterior = torch.softmax(torch.randn(1, 1203), dim=-1)
    embedding = torch.randn(1203, 256)
    args = [
        parent,
        child,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.ones(1, 1, 1),
        posterior,
        embedding,
        torch.randn(1, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ]
    out = model(*args)
    valid = output_ok(out, (1, 1203))
    detail = {
        "shape": list(out.shape),
        "finite": bool(torch.isfinite(out).all()),
        "nonpositive": bool((out <= 1e-7).all()),
        "backward": bool(backward),
    }
    if backward:
        out.sum().backward()
        grad_names = {
            "parent_input": parent.grad,
            "child_input": child.grad,
            "child_mask": model.child_mask[0].weight.grad,
            "stable_spatial_norm_weight": model.child_mask[1].norm.weight.grad,
            "stable_spatial_norm_bias": model.child_mask[1].norm.bias.grad,
            "child_query": model.child_query[0].weight.grad,
            "relation": model.relation[-1].weight.grad,
        }
        grad_detail = {}
        all_finite = True
        all_nonzero = True
        for name, grad in grad_names.items():
            finite = grad is not None and bool(torch.isfinite(grad).all())
            nonzero = grad is not None and float(grad.abs().sum()) > 0.0
            grad_detail[name] = {
                "finite": finite,
                "nonzero": nonzero,
                "abs_sum": None if grad is None else float(grad.abs().sum()),
            }
            all_finite = all_finite and finite
            all_nonzero = all_nonzero and nonzero
        detail["gradients"] = grad_detail
        valid = valid and all_finite and all_nonzero
    return valid, detail


def parameter_gradient_case(n_parent):
    model = make_model()
    model.train()
    args = base_args(
        n_parent=n_parent, n_child=n_parent, n_edge=n_parent, height=1, width=1
    )
    args[0].requires_grad_(True)
    args[1].requires_grad_(True)
    posterior_leaf = args[5].clone().requires_grad_(True)
    args[5] = posterior_leaf
    out = model(*args)
    out.sum().backward()
    targets = {
        "parent_input": args[0].grad,
        "child_input": args[1].grad,
        "child_mask": model.child_mask[0].weight.grad,
        "stable_spatial_norm_weight": model.child_mask[1].norm.weight.grad,
        "stable_spatial_norm_bias": model.child_mask[1].norm.bias.grad,
        "child_query": model.child_query[0].weight.grad,
        "relation": model.relation[-1].weight.grad,
    }
    details = {}
    valid = output_ok(out, (n_parent, 5))
    for name, grad in targets.items():
        finite = grad is not None and bool(torch.isfinite(grad).all())
        nonzero = grad is not None and float(grad.abs().sum()) > 0.0
        details[name] = {
            "finite": finite,
            "nonzero": nonzero,
            "abs_sum": None if grad is None else float(grad.abs().sum()),
        }
        valid = valid and finite and nonzero
    detached = posterior_leaf.grad is None
    details["posterior_grad_is_none"] = detached
    valid = valid and detached
    return valid, {
        "n_parent": n_parent,
        "output": out.tolist(),
        "gradients": details,
    }


def empty_edge_case(training):
    model = make_model()
    model.train(training)
    parent = torch.randn(1, 8, 1, 1)
    child = torch.empty(0, 8, 1, 1)
    posterior = torch.empty(0, 5)
    embedding = torch.randn(5, 8)
    out = model(
        parent,
        child,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, 1, 1),
        posterior,
        embedding,
        torch.empty(0, 5),
        torch.tensor([10], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    return torch.equal(out, torch.zeros_like(out)), {
        "mode": "train" if training else "eval",
        "shape": list(out.shape),
        "exact_zero": bool(torch.equal(out, torch.zeros_like(out))),
    }


def shape_contract():
    details = {}
    cases = [
        ("empty_overlap_wrong_width", (0, 2, 3), "overlap"),
        ("empty_overlap_wrong_height", (0, 3, 2), "overlap"),
        ("nonempty_overlap_wrong", (1, 2, 2), "overlap"),
    ]
    source_template = r"""
import importlib.util, torch
path = %r
shape = %r
spec = importlib.util.spec_from_file_location("subject", path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
model = subject.EvidenceOwnershipLedger(
    dim=8, hidden=8, class_width=5, max_children_per_parent=2, max_edges=2
)
parent = torch.randn(1, 8, 2, 2)
posterior = torch.empty(0, 5)
try:
    model(
        parent,
        torch.empty(0, 8, 2, 2),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(*shape),
        posterior,
        torch.randn(5, 8),
        torch.empty(0, 5),
        torch.tensor([10], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    print("ACCEPT")
except BaseException as exc:
    print("REJECT", type(exc).__name__, str(exc))
"""
    for name, shape, _ in cases:
        proc = subprocess.run(
            [sys.executable, "-O", "-c", source_template % (str(SNAPSHOT), shape)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        stdout = proc.stdout.strip()
        details[name] = {
            "shape": shape,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": proc.stderr.strip(),
        }
    optimized_rejected = all(
        value["returncode"] == 0
        and value["stdout"].startswith("REJECT")
        and "overlap" in value["stdout"]
        for value in details.values()
    )

    def normal_case(label, mutate, expected=None):
        model = make_model()
        args = base_args(n_edge=2, n_child=2, n_parent=1)
        mutate(args)
        ok, msg = rejection(
            lambda: model(*args),
            expected[0] if expected else None,
            expected[1] if expected else None,
        )
        details[label] = {"rejected": ok, "exception": msg}

    normal_case(
        "overlap_nonempty_wrong_shape",
        lambda a: a.__setitem__(4, torch.ones(2, 1, 2)),
        (ValueError, "overlap"),
    )
    normal_case(
        "child_token_dim_mismatch",
        lambda a: a.__setitem__(1, torch.randn(2, 7, 2, 2)),
        (ValueError, "child token"),
    )
    normal_case(
        "child_token_count_mismatch",
        lambda a: a.__setitem__(1, torch.randn(1, 8, 2, 2)),
        (ValueError, "child token rows"),
    )
    normal_case(
        "edge_length_mismatch",
        lambda a: a.__setitem__(3, torch.tensor([0], dtype=torch.long)),
        (ValueError, None),
    )
    normal_case(
        "geometry_shape_mismatch",
        lambda a: a.__setitem__(7, torch.zeros(2, 4)),
        (ValueError, "child geometry"),
    )
    normal_case(
        "posterior_class_width_mismatch",
        lambda a: a.__setitem__(5, torch.ones(2, 4)),
        (ValueError, None),
    )
    normal_case(
        "parent_time_shape_mismatch",
        lambda a: a.__setitem__(8, torch.tensor([10, 10], dtype=torch.long)),
        (ValueError, "time"),
    )
    normal_case(
        "child_time_shape_mismatch",
        lambda a: a.__setitem__(9, torch.tensor([9], dtype=torch.long)),
        (ValueError, "time"),
    )
    normal_case(
        "index_dtype_int32",
        lambda a: a.__setitem__(2, torch.tensor([0, 0], dtype=torch.int32)),
        (TypeError, "int64"),
    )
    normal_case(
        "index_dtype_bool",
        lambda a: a.__setitem__(3, torch.tensor([False, True])),
        (TypeError, "int64"),
    )
    normal_case(
        "negative_parent_index",
        lambda a: a.__setitem__(2, torch.tensor([-1, 0], dtype=torch.long)),
    )
    normal_case(
        "high_parent_index",
        lambda a: a.__setitem__(2, torch.tensor([0, 1], dtype=torch.long)),
    )
    normal_case(
        "high_child_index",
        lambda a: a.__setitem__(3, torch.tensor([0, 2], dtype=torch.long)),
    )
    normal = True
    for name, value in details.items():
        if name in {key for key, _, _ in cases}:
            continue
        normal = normal and bool(value.get("rejected", False))
    return optimized_rejected and normal, details


def limit_and_time_contract():
    details = {}

    legal = make_model(max_children_per_parent=1, max_edges=2)
    legal_args = base_args(n_parent=2, n_child=2, n_edge=2)
    legal_args[2] = torch.tensor([0, 1], dtype=torch.long)
    legal_args[3] = torch.tensor([0, 1], dtype=torch.long)
    legal_rejected, legal_msg = rejection(lambda: legal(*legal_args))
    details["legal_one_per_parent_under_two_edges"] = {
        "accepted": not legal_rejected,
        "exception": legal_msg,
    }

    per_parent = make_model(max_children_per_parent=1, max_edges=4)
    per_args = base_args(n_parent=1, n_child=2, n_edge=2)
    per_args[2] = torch.tensor([0, 0], dtype=torch.long)
    per_ok, per_msg = rejection(
        lambda: per_parent(*per_args), ValueError, "per-parent"
    )
    details["two_children_same_parent_rejected"] = {
        "rejected": per_ok,
        "exception": per_msg,
    }

    global_cap = make_model(max_children_per_parent=2, max_edges=1)
    global_args = base_args(n_parent=2, n_child=2, n_edge=2)
    global_args[2] = torch.tensor([0, 1], dtype=torch.long)
    global_ok, global_msg = rejection(
        lambda: global_cap(*global_args), ValueError, "max_edges"
    )
    details["global_edge_cap_rejected"] = {
        "rejected": global_ok,
        "exception": global_msg,
    }

    for dtype_name, dtype in (("int32", torch.int32), ("int64", torch.int64)):
        model = make_model()
        args = base_args(
            n_parent=2, n_child=2, n_edge=2, time_dtype=dtype
        )
        args[2] = torch.tensor([0, 1], dtype=torch.long)
        args[3] = torch.tensor([0, 1], dtype=torch.long)
        rejected, message = rejection(lambda: model(*args))
        details["past_" + dtype_name] = {
            "accepted": not rejected,
            "exception": message,
        }

    time_cases = [
        ("mixed_int64_float32", torch.tensor([10]), torch.tensor([9.0])),
        ("bool", torch.tensor([True]), torch.tensor([False])),
        ("float32", torch.tensor([10.0]), torch.tensor([9.0])),
        (
            "mixed_int32_int64",
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int64),
        ),
        (
            "future",
            torch.tensor([10], dtype=torch.int64),
            torch.tensor([11], dtype=torch.int64),
        ),
    ]
    for name, parent_time, child_time in time_cases:
        model = make_model()
        args = base_args()
        args[8], args[9] = parent_time, child_time
        expected_kind = ValueError if name == "future" else TypeError
        rejected, message = rejection(lambda: model(*args), expected_kind)
        details[name] = {"rejected": rejected, "exception": message}

    accepted = (
        not details["legal_one_per_parent_under_two_edges"]["exception"]
        .startswith("ValueError")
        and details["past_int32"]["accepted"]
        and details["past_int64"]["accepted"]
    )
    rejected = (
        details["two_children_same_parent_rejected"]["rejected"]
        and details["global_edge_cap_rejected"]["rejected"]
        and all(
            details[name]["rejected"]
            for name in (
                "mixed_int64_float32",
                "bool",
                "float32",
                "mixed_int32_int64",
                "future",
            )
        )
    )
    return accepted and rejected, details


def nonfinite_contract():
    details = {}
    mutations = {
        "parent": (0, (0, 0, 0, 0), float("nan")),
        "child": (1, (0, 0, 0, 0), float("inf")),
        "overlap": (4, (0, 0, 0), float("nan")),
        "posterior": (5, (0, 0), float("inf")),
        "embedding": (6, (0, 0), float("nan")),
        "geometry": (7, (0, 0), float("inf")),
    }
    for name, (index, location, value) in mutations.items():
        model = make_model()
        args = base_args()
        args[index][location] = value
        rejected, message = rejection(lambda: model(*args), ValueError)
        details[name] = {"rejected": rejected, "exception": message}
    return all(value["rejected"] for value in details.values()), details


def documented_repeated_edge_tokens_case():
    """The public docstring says child_tokens are repeated per edge."""
    model = make_model(
        classes=5,
        dim=8,
        hidden=8,
        max_children_per_parent=2,
        max_edges=2,
    )
    parent = torch.randn(2, 8, 2, 2)
    # One unique child appears in two edges.  The documented tensor is E rows,
    # while child_posterior has n_child rows.
    child = torch.randn(2, 8, 2, 2)
    posterior = torch.softmax(torch.randn(1, 5), dim=-1)
    embedding = torch.randn(5, 8)
    args = [
        parent,
        child,
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        torch.ones(2, 2, 2),
        posterior,
        embedding,
        torch.zeros(2, 5),
        torch.tensor([10, 10], dtype=torch.long),
        torch.tensor([9, 9], dtype=torch.long),
    ]
    try:
        out = model(*args)
        accepted = output_ok(out, (2, 5))
        message = "accepted"
    except BaseException as exc:
        accepted = False
        message = "%s: %s" % (type(exc).__name__, exc)
    return accepted, {
        "documented_child_token_rows_E": 2,
        "unique_child_posterior_rows": 1,
        "two_edges_share_child": True,
        "accepted": accepted,
        "exception_or_result": message,
        "source_relevant_lines": [130, 131],
        "documented_input": "child_tokens: [E,D,H,W], repeated per edge",
    }


def owner_tie_case():
    model = make_model(classes=3, dim=3, hidden=4)
    model.child_mask = ConstMask([1.0, 1.0])
    freeze_relation(model)
    embedding = torch.eye(3)
    parent = torch.zeros(1, 3, 1, 1)
    parent[0, 0, 0, 0] = 1.0
    parent[0, 2, 0, 0] = 1.0
    child = torch.randn(2, 3, 1, 1)
    posterior = torch.stack(
        [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])]
    )
    common = [
        parent,
        child,
        torch.tensor([0, 0], dtype=torch.long),
        None,
        torch.ones(2, 1, 1),
        posterior,
        embedding,
        torch.zeros(2, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9, 9], dtype=torch.long),
    ]
    forward_args = list(common)
    forward_args[3] = torch.tensor([0, 1], dtype=torch.long)
    forward = model(*forward_args).clone()
    reverse_args = list(common)
    reverse_args[3] = torch.tensor([1, 0], dtype=torch.long)
    reverse = model(*reverse_args).clone()

    def single(child_index, class_index):
        args = [
            parent,
            child,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([child_index], dtype=torch.long),
            torch.ones(1, 1, 1),
            posterior,
            embedding,
            torch.zeros(1, 5),
            torch.tensor([10], dtype=torch.long),
            torch.tensor([9], dtype=torch.long),
        ]
        return model(*args).clone()

    expected_forward = single(0, 0)
    expected_reverse = single(1, 2)
    forward_ok = torch.allclose(forward, expected_forward, atol=1e-6)
    reverse_ok = torch.allclose(reverse, expected_reverse, atol=1e-6)
    local_ok = (
        forward[0, 0] < -1e-8
        and abs(float(forward[0, 2])) < 1e-8
        and reverse[0, 2] < -1e-8
        and abs(float(reverse[0, 0])) < 1e-8
    )
    return forward_ok and reverse_ok and local_ok, {
        "forward": forward.tolist(),
        "expected_lower_edge_forward": expected_forward.tolist(),
        "reversed_payload": reverse.tolist(),
        "expected_lower_edge_after_reversal": expected_reverse.tolist(),
        "strict_tie_and_no_class_spill": local_ok,
    }


def owner_local_max_case():
    model = make_model(classes=3, dim=3, hidden=4)
    model.child_mask = ConstMask([1.0, 9.0, 5.0])
    freeze_relation(model)
    embedding = torch.eye(3)
    parent = torch.zeros(1, 3, 1, 1)
    parent[0, 0, 0, 0] = 1.0
    child = torch.randn(3, 3, 1, 1)
    posterior = torch.stack(
        [torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), torch.zeros(3)]
    )
    combined = model(
        parent,
        child,
        torch.tensor([0, 0, 0], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.ones(3, 1, 1),
        posterior,
        embedding,
        torch.zeros(3, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9, 9, 9], dtype=torch.long),
    ).clone()
    single_model = make_model(classes=3, dim=3, hidden=4)
    copy_learned_modules(single_model, model)
    single_model.child_mask = ConstMask([9.0])
    only_winner = single_model(
        parent,
        child[1:2],
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.ones(1, 1, 1),
        posterior[1:2],
        embedding,
        torch.zeros(1, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ).clone()
    return torch.allclose(combined, only_winner, atol=1e-6), {
        "three_child_scores": [1.0, 9.0, 5.0],
        "combined": combined.tolist(),
        "only_score9": only_winner.tolist(),
        "max_abs_diff": float((combined - only_winner).abs().max()),
    }


def overlap_partition_case():
    classes, dim, height, width = 3, 4, 2, 3
    embedding = torch.randn(classes, dim)
    parent = torch.randn(1, dim, height, width)
    child = torch.randn(2, dim, height, width)
    posterior = torch.softmax(torch.randn(2, classes), dim=-1)
    top = torch.zeros(1, height, width)
    top[0, 0, :] = 0.9
    bottom = torch.zeros(1, height, width)
    bottom[0, 1, :] = 0.8
    both_overlap = torch.stack(
        [
            top[0],
            torch.zeros(height, width),
        ]
    )
    both_overlap[1, 0, :] = 0.2
    both_overlap[1, 1, :] = 0.8

    def run(children, overlaps, child_indices):
        model = make_model(classes=classes, dim=dim, hidden=4)
        model.child_mask = ConstMask([1.0] * len(children))
        freeze_relation(model)
        return model(
            parent,
            child,
            torch.zeros(len(children), dtype=torch.long),
            torch.tensor(child_indices, dtype=torch.long),
            torch.cat(overlaps, dim=0),
            posterior,
            embedding,
            torch.zeros(len(children), 5),
            torch.tensor([10], dtype=torch.long),
            torch.full((len(children),), 9, dtype=torch.long),
        ).clone()

    reference = make_model(classes=classes, dim=dim, hidden=4)
    freeze_relation(reference)
    top_model = make_model(classes=classes, dim=dim, hidden=4)
    copy_learned_modules(top_model, reference)
    top_model.child_mask = ConstMask([1.0])
    bottom_model = make_model(classes=classes, dim=dim, hidden=4)
    copy_learned_modules(bottom_model, reference)
    bottom_model.child_mask = ConstMask([1.0])
    combined_model = make_model(classes=classes, dim=dim, hidden=4)
    copy_learned_modules(combined_model, reference)
    combined_model.child_mask = ConstMask([1.0, 1.0])
    one_top = top_model(
        parent, child[:1], torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long), top, posterior[:1], embedding,
        torch.zeros(1, 5), torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ).clone()
    one_bottom = bottom_model(
        parent, child[1:], torch.zeros(1, dtype=torch.long),
        torch.tensor([0], dtype=torch.long), bottom, posterior[1:], embedding,
        torch.zeros(1, 5), torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ).clone()
    combined = combined_model(
        parent, child, torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.cat([both_overlap[0:1], both_overlap[1:2]], dim=0),
        posterior, embedding, torch.zeros(2, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9, 9], dtype=torch.long),
    ).clone()
    expected = one_top + one_bottom
    return torch.allclose(combined, expected, atol=1e-6), {
        "top_only": one_top.tolist(),
        "bottom_only": one_bottom.tolist(),
        "combined": combined.tolist(),
        "expected_no_double_subtract": expected.tolist(),
        "max_abs_diff": float((combined - expected).abs().max()),
    }


def exact_duplicate_overlap_case():
    model = make_model(classes=3, dim=4, hidden=4)
    model.child_mask = ConstMask([1.0, 1.0])
    freeze_relation(model)
    child = torch.randn(1, 4, 1, 1).repeat(2, 1, 1, 1)
    posterior = torch.full((2, 3), 1.0 / 3.0)
    parent = torch.randn(1, 4, 1, 1)
    embedding = torch.randn(3, 4)
    common = [
        parent,
        child,
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.ones(2, 1, 1),
        posterior,
        embedding,
        torch.zeros(2, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9, 9], dtype=torch.long),
    ]
    twice = model(*common).clone()
    once_args = list(common)
    once_args[1] = child[:1]
    once_args[2] = torch.tensor([0], dtype=torch.long)
    once_args[3] = torch.tensor([0], dtype=torch.long)
    once_args[4] = torch.ones(1, 1, 1)
    once_args[5] = posterior[:1]
    once_args[7] = torch.zeros(1, 5)
    once_args[9] = torch.tensor([9], dtype=torch.long)
    once = model(*once_args).clone()
    return torch.allclose(once, twice, atol=1e-6), {
        "once": once.tolist(),
        "twice": twice.tolist(),
        "max_abs_diff": float((once - twice).abs().max()),
    }


def class_condition_case():
    model = make_model(classes=3, dim=3, hidden=4)
    model.child_mask = ConstMask([1.0])
    freeze_relation(model)
    embedding = torch.eye(3)
    parent = torch.zeros(1, 3, 1, 1)
    parent[0, 0, 0, 0] = 1.0
    parent[0, 1, 0, 0] = 1.0
    child = torch.randn(1, 3, 1, 1)
    base = [
        parent,
        child,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.ones(1, 1, 1),
        None,
        embedding,
        torch.zeros(1, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ]
    first_args = list(base)
    first_args[5] = torch.tensor([[1.0, 0.0, 0.0]])
    second_args = list(base)
    second_args[5] = torch.tensor([[0.0, 1.0, 0.0]])
    first = model(*first_args).clone()
    second = model(*second_args).clone()
    no_spill = (
        abs(float(first[0, 1])) < 1e-8
        and abs(float(second[0, 0])) < 1e-8
        and first[0, 0] < -1e-8
        and second[0, 1] < -1e-8
    )
    return no_spill, {
        "posterior_first_class": first.tolist(),
        "posterior_second_class": second.tolist(),
        "class_conditioned_no_spill": no_spill,
    }


def class_relation_not_scalar_case():
    model = make_model(classes=4, dim=8, hidden=8)
    model.child_mask = ConstMask([1.0])
    with torch.no_grad():
        # Deterministic per-class relation values: hidden[0]=1 through GELU(1),
        # and four distinct class logits. This avoids mistaking random tiny
        # initialization spread for a scalar relation.
        for parameter in model.relation[0].parameters():
            parameter.zero_()
        model.relation[0].bias.fill_(1.0)
        model.relation[-1].weight.zero_()
        model.relation[-1].weight[:, 0] = torch.tensor([0.0, 0.5, 1.0, 1.5])
        model.relation[-1].bias.fill_(-4.0)
    args = base_args(
        classes=4, dim=8, height=1, width=1,
        n_parent=1, n_child=1, n_edge=1,
    )
    args[5] = torch.full((1, 4), 0.25)
    # Keep several positive parent classes so a shared scalar gate would give
    # proportional, equal per-class coefficients.
    args[0] = torch.randn(1, 8, 1, 1)
    args[6] = torch.randn(4, 8)
    out = model(*args).clone()
    parent = args[0] / args[0].norm(dim=1, keepdim=True).clamp_min(1e-12)
    embedding = args[6] / args[6].norm(dim=1, keepdim=True).clamp_min(1e-12)
    evidence = (parent[0, :, 0, 0] @ embedding.T).clamp_min(0)
    positive = evidence > 1e-6
    coefficients = torch.zeros(4)
    coefficients[positive] = (-out[0, positive]) / evidence[positive]
    finite_coefficients = coefficients[positive]
    spread = (
        float(finite_coefficients.max() - finite_coefficients.min())
        if finite_coefficients.numel() > 1
        else 0.0
    )
    class_logits = torch.tensor([0.0, 0.5, 1.0, 1.5])
    sigmoid_spread = float(torch.sigmoid(class_logits).max()
                           - torch.sigmoid(class_logits).min())
    return bool(positive.sum() >= 2 and spread > 1e-4), {
        "parent_positive_evidence": evidence.tolist(),
        "correction": out.tolist(),
        "negative_correction_over_positive_evidence": coefficients.tolist(),
        "positive_class_count": int(positive.sum()),
        "coefficient_spread": spread,
        "configured_relation_logits": class_logits.tolist(),
        "configured_sigmoid_spread": sigmoid_spread,
    }


def main():
    # Round 1: the exact regression target plus batch/space/mode controls.
    for training in (True, False):
        for n_parent, height, width in (
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
        ):
            probe(
                1,
                "%s_batch%d_%dx%d"
                % ("train" if training else "eval", n_parent, height, width),
                lambda n=n_parent, h=height, w=width, t=training:
                    normalize_matrix_case(n, h, w, t),
            )
    probe(
        1,
        "train_eval_parity_batch1_and_batch2_1x1",
        lambda: (
            all(parity_case(n)[0] for n in (1, 2)),
            {"batch1": parity_case(1)[1], "batch2": parity_case(2)[1]},
        ),
    )
    probe(
        1,
        "batch_independence_train_and_eval",
        lambda: (
            batch_independence(True)[0] and batch_independence(False)[0],
            {
                "train": batch_independence(True)[1],
                "eval": batch_independence(False)[1],
            },
        ),
    )
    probe(1, "all_zero_features", lambda: constant_features_case("zero"))
    probe(1, "all_one_features", lambda: constant_features_case("one"))
    probe(1, "default_1203_vocab_train_1x1", lambda: default_vocab_case(False))

    # Round 2: empty identity, explicit shape checks, limits, temporal access,
    # and finite-input fail-closed behavior.
    probe(2, "empty_edge_exact_zero_train_1x1", lambda: empty_edge_case(True))
    probe(2, "empty_edge_exact_zero_eval_1x1", lambda: empty_edge_case(False))
    probe(2, "shape_contract_under_python_O_and_controls", shape_contract)
    probe(2, "per_parent_and_global_caps_plus_time_contract",
          limit_and_time_contract)
    probe(2, "six_nonfinite_input_kinds_rejected", nonfinite_contract)
    probe(
        2,
        "documented_edge_repeated_child_tokens_with_one_unique_child",
        documented_repeated_edge_tokens_case,
    )

    # Round 3: ledger math, category conditioning, 1203 backward, and ordinary
    # train-mode backward for the legal 1x1 layouts.
    probe(3, "strict_owner_tie_lower_edge_and_local_class_evidence",
          owner_tie_case)
    probe(3, "owner_local_max_three_children", owner_local_max_case)
    probe(3, "partial_overlap_one_owner_no_double_subtract",
          overlap_partition_case)
    probe(3, "equal_full_overlap_charges_once", exact_duplicate_overlap_case)
    probe(3, "child_posterior_is_class_conditioned", class_condition_case)
    probe(3, "per_class_relation_not_shared_scalar",
          class_relation_not_scalar_case)
    probe(3, "train_batch1_1x1_backward", lambda: parameter_gradient_case(1))
    probe(3, "train_batch2_1x1_backward", lambda: parameter_gradient_case(2))
    probe(3, "default_1203_train_1x1_backward",
          lambda: default_vocab_case(True))

    main_after = sha(CURRENT)
    snapshot_after = sha(SNAPSHOT)
    source_unchanged = (
        main_after == EXPECTED and snapshot_after == EXPECTED
    )
    rounds = {}
    for round_no in (1, 2, 3):
        selected = [item for item in results if item["round"] == round_no]
        rounds[str(round_no)] = {
            "checks": len(selected),
            "counterexamples": sum(item["counterexample"] for item in selected),
            "passed": sum(item["passed"] for item in selected),
            "failed": sum(not item["passed"] for item in selected),
        }
    all_passed = all(item["passed"] for item in results) and source_unchanged
    summary = {
        "status": (
            "PASS_ALLOW_READ_ONLY_OCCLUSION_COLLECTION"
            if all_passed
            else "FAIL_DO_NOT_START_OCCLUSION_OR_TRAINING"
        ),
        "source_sha256": EXPECTED,
        "snapshot_path": str(SNAPSHOT),
        "snapshot_sha256": snapshot_after,
        "main_sha256_after": main_after,
        "source_binding_unchanged": source_unchanged,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_used": False,
        "training_run": False,
        "occlusion_collection_run": False,
        "timestamps_used_for_selection": False,
        "rounds": rounds,
        "all_passed": all_passed,
        "results": results,
        "script_sha256": sha(__file__),
    }
    OUT_JSON.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rounds": rounds,
                "source_binding_unchanged": source_unchanged,
                "evidence_json": str(OUT_JSON),
                "evidence_sha256": sha(OUT_JSON),
            },
            indent=2,
        )
    )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
