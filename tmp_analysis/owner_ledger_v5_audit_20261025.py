#!/usr/bin/env python3
"""Independent CPU audit of EvidenceOwnershipLedger snapshot 417f734e.

The mutable main file is read only for before/after hashes.  All execution
uses the frozen copy.  No CUDA, training on data, or file mutation of the
subject is performed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    "evidence_ownership_head_417f734e.py"
)
V4_SNAPSHOT = Path(
    "/raid/datasets/chc_data/partwhole_review_20261025_v1/"
    "evidence_ownership_head_833a4bf.py"
)
EXPECTED = "417f734e63228637a153657c4106801e2f16a641209154b21ec7ea7ef4ce6ae3"
V4_EXPECTED = "833a4bf6dcd7e6af2763b602f975ee9a0e7fea09e68dde9a5b7904f84305af6b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


current_start = digest(CURRENT)
snapshot_start = digest(SNAPSHOT)
if current_start != EXPECTED or snapshot_start != EXPECTED:
    raise SystemExit(
        "snapshot mismatch: current=%s snapshot=%s expected=%s"
        % (current_start, snapshot_start, EXPECTED)
    )
if digest(V4_SNAPSHOT) != V4_EXPECTED:
    raise SystemExit("v4 comparator snapshot mismatch")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = load(SNAPSHOT, "owner_ledger_v5_subject")
v4subject = load(V4_SNAPSHOT, "owner_ledger_v4_comparator")
Ledger = subject.EvidenceOwnershipLedger
V4Ledger = v4subject.EvidenceOwnershipLedger
torch.set_num_threads(1)
results = []


def record(round_no, name, fn, counterexample=False):
    try:
        passed, detail = fn()
    except BaseException as exc:
        passed = False
        detail = {"exception": "%s: %s" % (type(exc).__name__, exc)}
    results.append(
        {
            "round": int(round_no),
            "name": name,
            "passed": bool(passed),
            "counterexample": bool(counterexample),
            "detail": detail,
        }
    )


def rejects(fn, kind=ValueError, text=None):
    try:
        fn()
    except BaseException as exc:
        ok_kind = isinstance(exc, kind)
        ok_text = text is None or text in str(exc)
        return ok_kind and ok_text, "%s: %s" % (type(exc).__name__, exc)
    return False, "accepted without exception"


class ConstMask(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values.to(x).view(-1, 1, 1, 1).expand(
            -1, 1, x.shape[-2], x.shape[-1]
        )


def make_model(classes=3, dim=8, hidden=8, max_children=8, max_edges=64,
               train=False):
    model = Ledger(
        dim=dim,
        hidden=hidden,
        class_width=classes,
        max_children_per_parent=max_children,
        max_edges=max_edges,
    )
    model.train(train)
    return model


def args_for(classes=3, dim=8, h=1, w=1, n_parent=1, n_child=1, n_edge=1,
             constant=False):
    def token(rows):
        if constant:
            return torch.ones(rows, dim, h, w)
        return torch.randn(rows, dim, h, w)

    return [
        token(n_parent),
        token(n_edge),
        torch.arange(n_edge, dtype=torch.long) % max(n_parent, 1),
        torch.arange(n_edge, dtype=torch.long) % max(n_child, 1),
        torch.ones(n_edge, h, w),
        torch.full((n_child, classes), 1.0 / classes),
        torch.randn(classes, dim),
        torch.zeros(n_edge, 5),
        torch.full((n_parent,), 10, dtype=torch.long),
        torch.full((n_edge,), 9, dtype=torch.long),
    ]


def out_case(h, w, n_parent=1, n_edge=1, n_child=1, constant=False):
    model = make_model(train=True)
    out = model(*args_for(h=h, w=w, n_parent=n_parent, n_edge=n_edge,
                          n_child=n_child, constant=constant))
    return (
        tuple(out.shape) == (n_parent, 3)
        and bool(torch.isfinite(out).all())
        and bool((out <= 1e-7).all()),
        {
            "shape": list(out.shape),
            "finite": bool(torch.isfinite(out).all()),
            "nonpositive": bool((out <= 1e-7).all()),
            "max": float(out.max()),
        },
    )


def eval_case():
    model = make_model(train=False)
    assert not model.training
    out = model(*args_for(h=1, w=1))
    return (
        tuple(out.shape) == (1, 3)
        and bool(torch.isfinite(out).all())
        and bool((out <= 1e-7).all()),
        {
            "shape": list(out.shape),
            "finite": bool(torch.isfinite(out).all()),
            "nonpositive": bool((out <= 1e-7).all()),
        },
    )


def compare_normalizers():
    cases = []
    for n_parent in (1, 2):
        for h, w in ((1, 1), (2, 2), (1, 2)):
            row = {"n_parent": n_parent, "h": h, "w": w}
            for label, cls in (("v4", V4Ledger), ("v5", Ledger)):
                model = cls(dim=8, hidden=8, class_width=3,
                            max_children_per_parent=8, max_edges=8)
                model.train()
                args = args_for(h=h, w=w, n_parent=n_parent,
                                n_edge=n_parent, n_child=n_parent)
                try:
                    out = model(*args)
                    row[label] = {
                        "accepted": True,
                        "finite": bool(torch.isfinite(out).all()),
                    }
                except BaseException as exc:
                    row[label] = {
                        "accepted": False,
                        "exception": "%s: %s" % (type(exc).__name__, exc),
                    }
            cases.append(row)
    exact_single = cases[0]
    expected = (
        not exact_single["v4"]["accepted"]
        and not exact_single["v5"]["accepted"]
        and cases[1]["v4"]["accepted"]
        and not cases[1]["v5"]["accepted"]
        and all(c["v5"].get("accepted", False) for c in cases[2:])
    )
    return expected, {"matrix": cases, "expected_v5_fix": False}


# Round 1: the exact regression patch and batch/spatial boundary cases.
record(
    1,
    "counterexample: single parent under train() with a legal 1x1 ROI is accepted",
    lambda: out_case(1, 1),
    True,
)
record(
    1,
    "counterexample: two parents under train() with 1x1 ROIs are accepted",
    lambda: out_case(1, 1, n_parent=2, n_edge=2, n_child=2),
    True,
)
record(
    1,
    "train() single parent with 2x2 ROI is finite, nonpositive, and has correct shape",
    lambda: out_case(2, 2),
    True,
)
record(
    1,
    "train() batch 2 with 2x2 all-ones ROI is finite and nonpositive",
    lambda: out_case(2, 2, n_parent=2, n_edge=2, n_child=2, constant=True),
    True,
)
record(
    1,
    "eval() single parent with 1x1 ROI remains accepted",
    eval_case,
    True,
)
record(
    1,
    "v4/v5 normalizer comparison catches both the exact 1x1 single-row failure and batch-2 regression",
    compare_normalizers,
    True,
)


# Round 2: v4 input and structural contracts must remain intact.
def optimized_shape_checks():
    cases = []
    for shape in ((0, 2, 3), (0, 3, 2), (1, 2, 2)):
        code = r"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util, torch
p = {snapshot!r}
s = importlib.util.spec_from_file_location("m", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
model = m.EvidenceOwnershipLedger(dim=6, hidden=8, class_width=3).eval()
parent = torch.randn(1, 6, 2, 2)
post = torch.empty(0, 3)
emb = torch.eye(3, 6)
try:
    model(parent, torch.empty(0, 6, 2, 2),
          torch.empty(0, dtype=torch.long),
          torch.empty(0, dtype=torch.long),
          torch.empty(*{shape!r}), post, emb,
          torch.empty(0, 5),
          torch.tensor([10], dtype=torch.long),
          torch.empty(0, dtype=torch.long))
    print("ACCEPT")
except BaseException as exc:
    print("REJECT", type(exc).__name__, str(exc))
""".format(snapshot=str(SNAPSHOT), shape=shape)
        proc = subprocess.run(
            [sys.executable, "-O", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        cases.append({
            "shape": list(shape),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        })
    return (
        all(c["returncode"] == 0
            and c["stdout"].startswith("REJECT")
            and "overlap must be" in c["stdout"]
            for c in cases),
        {"cases": cases},
    )


record(
    2,
    "counterexample: invalid empty overlap shape is rejected under python -O",
    optimized_shape_checks,
    True,
)


def empty_identity_train():
    model = make_model(train=True)
    args = args_for(h=1, w=1)
    args[1] = torch.empty(0, 8, 1, 1)
    args[2] = torch.empty(0, dtype=torch.long)
    args[3] = torch.empty(0, dtype=torch.long)
    args[4] = torch.empty(0, 1, 1)
    args[5] = torch.empty(0, 3)
    args[7] = torch.empty(0, 5)
    args[9] = torch.empty(0, dtype=torch.long)
    out = model(*args)
    return (
        torch.equal(out, torch.zeros_like(out)),
        {"shape": list(out.shape), "exact_zero": bool(torch.equal(
            out, torch.zeros_like(out)))},
    )


record(
    2,
    "counterexample: empty edge is exact zero even in train() with 1x1 ROI",
    empty_identity_train,
    True,
)


def class_width_contract():
    details = {}
    for empty in (True, False):
        model = make_model(classes=4, dim=6, train=False)
        args = args_for(classes=3, dim=6, h=2, w=2)
        if empty:
            args[1] = torch.empty(0, 6, 2, 2)
            args[2] = torch.empty(0, dtype=torch.long)
            args[3] = torch.empty(0, dtype=torch.long)
            args[4] = torch.empty(0, 2, 2)
            args[5] = torch.empty(0, 3)
            args[7] = torch.empty(0, 5)
            args[9] = torch.empty(0, dtype=torch.long)
        ok, msg = rejects(lambda a=args: model(*a), ValueError, "width")
        details["empty" if empty else "nonempty"] = {
            "rejected": ok, "exception": msg
        }
    return all(v["rejected"] for v in details.values()), details


record(
    2,
    "counterexample: class width mismatch is rejected on empty and nonempty paths",
    class_width_contract,
    True,
)


def time_contract():
    details = {}
    for label, dtype in (("int32", torch.int32), ("int64", torch.int64)):
        model = make_model(train=False)
        args = args_for(h=2, w=2, n_parent=2, n_edge=2, n_child=2)
        args[2] = torch.tensor([0, 1])
        args[3] = torch.tensor([0, 1])
        args[8] = torch.full((2,), 10, dtype=dtype)
        args[9] = torch.full((2,), 9, dtype=dtype)
        ok, msg = rejects(lambda a=args: model(*a))
        details["accepted_" + label] = {"accepted": not ok, "exception": msg}
    invalid = [
        ("mixed_int32_int64",
         torch.tensor([10], dtype=torch.int32),
         torch.tensor([9], dtype=torch.int64)),
        ("mixed_int64_float32",
         torch.tensor([10], dtype=torch.int64),
         torch.tensor([9.0])),
        ("bool", torch.tensor([True]), torch.tensor([False])),
        ("float32", torch.tensor([10.0]), torch.tensor([9.0])),
        ("future", torch.tensor([10], dtype=torch.long),
         torch.tensor([11], dtype=torch.long)),
        ("parent_nan", torch.tensor([float("nan")]), torch.tensor([9.0])),
        ("child_inf", torch.tensor([10], dtype=torch.float32),
         torch.tensor([float("inf")])),
    ]
    for label, pt, ct in invalid:
        model = make_model(train=False)
        args = args_for(h=2, w=2)
        args[8], args[9] = pt, ct
        kind = ValueError if label in ("future", "parent_nan", "child_inf") else TypeError
        text = {
            "future": "future",
            "parent_nan": "nonfinite",
            "child_inf": "nonfinite",
        }.get(label)
        ok, msg = rejects(lambda a=args: model(*a), kind, text)
        details[label] = {"rejected": ok, "exception": msg}
    valid = (
        details["accepted_int32"]["accepted"]
        and details["accepted_int64"]["accepted"]
    )
    bad = details["mixed_int32_int64"]["rejected"]
    bad = bad and details["mixed_int64_float32"]["rejected"]
    bad = bad and details["bool"]["rejected"] and details["float32"]["rejected"]
    bad = bad and details["future"]["rejected"]
    bad = bad and details["parent_nan"]["rejected"] and details["child_inf"]["rejected"]
    return valid and bad, details


record(
    2,
    "counterexample: time dtype, finiteness, and future contract",
    time_contract,
    True,
)


def nonfinite_inputs():
    fields = {
        "parent_tokens": (0, float("nan")),
        "child_tokens": (1, float("inf")),
        "overlap": (4, float("nan")),
        "posterior": (5, float("inf")),
        "embedding": (6, float("nan")),
        "geometry": (7, float("inf")),
    }
    details = {}
    for field, (index, value) in fields.items():
        model = make_model(train=False)
        args = args_for(h=2, w=2)
        args[index].reshape(-1)[0] = value
        ok, msg = rejects(lambda a=args: model(*a), ValueError, "nonfinite")
        details[field] = {"rejected": ok, "exception": msg}
    return all(v["rejected"] for v in details.values()), details


record(
    2,
    "counterexample: NaN and Inf are rejected for every tensor input",
    nonfinite_inputs,
    True,
)


def edge_limits():
    model = make_model(max_children=1, max_edges=2, train=False)
    args = args_for(h=2, w=2, n_parent=2, n_edge=2, n_child=2)
    args[2] = torch.tensor([0, 1])
    ok_legal, legal_msg = rejects(lambda a=args: model(*a))
    same = list(args)
    same[2] = torch.tensor([0, 0])
    ok_per_parent, pp_msg = rejects(lambda a=same: model(*a), ValueError, "per-parent")
    too_many = make_model(max_children=2, max_edges=1, train=False)
    ok_global, global_msg = rejects(
        lambda a=args: too_many(*a), ValueError, "max_edges"
    )
    return (
        (not ok_legal) and ok_per_parent and ok_global,
        {
            "legal": legal_msg,
            "same_parent": pp_msg,
            "global_max_edges": global_msg,
        },
    )


record(
    2,
    "counterexample: per-parent and global edge limits stay separate",
    edge_limits,
    True,
)


# Round 3: ownership, class conditioning, gradients, and batch independence.
def freeze_relation(model):
    with torch.no_grad():
        for parameter in model.relation.parameters():
            parameter.zero_()


def owner_local_max_and_tie():
    model = make_model(train=False)
    model.child_mask = ConstMask([1, 9, 5])
    freeze_relation(model)
    embedding = torch.eye(3, 8)
    parent = torch.zeros(2, 8, 1, 1)
    parent[0, :, 0, 0] = (embedding[0] + embedding[2]) / math.sqrt(2)
    parent[1, :, 0, 0] = embedding[1]
    child = torch.randn(3, 8, 1, 1)
    args = [
        parent,
        child,
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 2]),
        torch.ones(3, 1, 1),
        torch.eye(3),
        embedding,
        torch.zeros(3, 5),
        torch.tensor([10, 10], dtype=torch.long),
        torch.tensor([9, 9, 9], dtype=torch.long),
    ]
    out = model(*args)
    local = (
        out[0, 2] < -1e-8
        and abs(float(out[0, 0])) < 1e-10
        and out[1, 1] < -1e-8
        and abs(float(out[1, 2])) < 1e-10
    )

    tie_model = make_model(train=False)
    tie_model.child_mask = ConstMask([1, 1])
    freeze_relation(tie_model)
    parent = torch.zeros(1, 8, 1, 1)
    parent[0, :, 0, 0] = (embedding[0] + embedding[2]) / math.sqrt(2)
    child = torch.randn(2, 8, 1, 1)
    posterior = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    common = [
        parent, child, torch.tensor([0, 0]), None, torch.ones(2, 1, 1),
        posterior, embedding, torch.zeros(2, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9, 9], dtype=torch.long),
    ]
    forward_args = list(common)
    forward_args[3] = torch.tensor([0, 1])
    forward = tie_model(*forward_args)
    reverse_args = list(common)
    reverse_args[3] = torch.tensor([1, 0])
    reverse = tie_model(*reverse_args)
    tie = (
        forward[0, 0] < -1e-8
        and abs(float(forward[0, 2])) < 1e-10
        and reverse[0, 2] < -1e-8
        and abs(float(reverse[0, 0])) < 1e-10
    )
    return local and tie, {
        "local_max": out.tolist(),
        "tie": forward.tolist(),
        "tie_payload_reversed": reverse.tolist(),
    }


record(
    3,
    "counterexample: owner uses per-parent local max and strict lower-edge tie",
    owner_local_max_and_tie,
    True,
)


def overlap_partition():
    C, D, H, W = 3, 8, 2, 3
    embedding = torch.eye(C, D)
    parent = torch.zeros(1, D, H, W)
    parent[0, :, :, :] = (
        (embedding[0] + embedding[2]).view(D, 1, 1) / math.sqrt(2)
    )
    a = torch.randn(1, D, H, W)
    b = torch.randn(1, D, H, W)
    posterior = torch.stack([
        torch.tensor([0.8, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.8]),
    ])
    ov_a = torch.zeros(1, H, W)
    ov_a[0, 0, :] = 0.9
    ov_b = torch.zeros(1, H, W)
    ov_b[0, 0, :] = 0.2
    ov_b[0, 1, :] = 0.8
    ov_a_top = ov_a.clone()
    ov_b_bottom = torch.zeros_like(ov_b)
    ov_b_bottom[0, 1, :] = 0.8

    def run(children, overlaps, posts, ids):
        model = make_model(train=False)
        model.child_mask = ConstMask([1.0] * len(children))
        freeze_relation(model)
        return model(
            parent,
            torch.cat(children, 0),
            torch.zeros(len(children), dtype=torch.long),
            torch.tensor(ids, dtype=torch.long),
            torch.cat(overlaps, 0),
            posts,
            embedding,
            torch.zeros(len(children), 5),
            torch.tensor([10], dtype=torch.long),
            torch.full((len(children),), 9, dtype=torch.long),
        )

    a_only = run([a], [ov_a_top], posterior[:1], [0])
    b_only = run([b], [ov_b_bottom], posterior[1:], [0])
    both = run([a, b], [ov_a, ov_b], posterior, [0, 1])
    expected = a_only + b_only
    full_model = make_model(train=False)
    full_model.child_mask = ConstMask([1, 1])
    freeze_relation(full_model)
    full = full_model(
        parent, torch.cat([a, a], 0), torch.tensor([0, 0]),
        torch.tensor([0, 1]), torch.cat([ov_a, ov_a], 0),
        torch.stack([posterior[0], posterior[0]]), embedding,
        torch.zeros(2, 5), torch.tensor([10], dtype=torch.long),
        torch.tensor([9, 9], dtype=torch.long),
    )
    once = run([a], [ov_a], posterior[:1], [0])
    no_double = torch.allclose(full, once, atol=1e-6)
    exact = torch.allclose(both, expected, atol=1e-6)
    return (
        no_double
        and exact
        and bool((both <= 1e-8).all())
        and bool(torch.isfinite(both).all())
        and float(a_only.abs().max()) > 0
        and float(b_only.abs().max()) > 0,
        {
            "full_overlap_no_double": bool(no_double),
            "partial_partition_exact": bool(exact),
            "a_only": a_only.tolist(),
            "b_only": b_only.tolist(),
            "both": both.tolist(),
            "expected": expected.tolist(),
            "max_abs_diff": float((both - expected).abs().max()),
        },
    )


record(
    3,
    "counterexample: full and partial overlaps prevent duplicate subtraction",
    overlap_partition,
    True,
)


def class_conditioning_and_detach():
    C, D, H, W = 3, 8, 2, 2
    model = make_model(classes=C, dim=D, train=False)
    embedding = torch.eye(C, D)
    parent = torch.randn(1, D, H, W)
    child = torch.randn(1, D, H, W)
    base = args_for(classes=C, dim=D, h=H, w=W)
    base[0], base[1], base[6] = parent, child, embedding
    base[5] = torch.tensor([[0.8, 0.2, 0.0]])
    first = model(*base)
    args = list(base)
    args[5] = torch.tensor([[1.0, 0.0, 0.0]])
    changed = model(*args)
    class_effect = not torch.allclose(first, changed)

    zero_model = make_model(classes=C, dim=D, train=False)
    zero_model.child_mask = ConstMask([1.0])
    freeze_relation(zero_model)
    parent = torch.zeros(1, D, 1, 1)
    parent[0, :, 0, 0] = embedding[0] + 0.5 * embedding[1]
    common = [
        parent, torch.randn(1, D, 1, 1), torch.tensor([0]),
        torch.tensor([0]), torch.ones(1, 1, 1),
        torch.tensor([[1.0, 1.0, 0.0]]), embedding,
        torch.zeros(1, 5), torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ]
    before = zero_model(*common)
    changed_embedding = embedding.clone()
    changed_embedding[0] = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    common[6] = changed_embedding
    after = zero_model(*common)
    delta = (after - before)[0].tolist()
    local = abs(delta[0]) > 0 and abs(delta[1]) < 1e-8 and abs(delta[2]) < 1e-8

    posterior = torch.tensor([[0.8, 0.2, 0.0]], requires_grad=True)
    model = make_model(classes=C, dim=D, train=False)
    args = args_for(classes=C, dim=D, h=2, w=2)
    args[5] = posterior
    model(*args).sum().backward()
    detached = posterior.grad is None
    return (
        class_effect and local and detached,
        {
            "posterior_changes_output": bool(class_effect),
            "embedding_delta": delta,
            "posterior_grad_is_none": bool(detached),
        },
    )


record(
    3,
    "counterexample: class-conditioned evidence and detached posterior",
    class_conditioning_and_detach,
    True,
)


def zero_parent_identity():
    model = make_model(train=False)
    args = args_for(h=2, w=2)
    args[0] = torch.zeros_like(args[0])
    out = model(*args)
    return (
        torch.equal(out, torch.zeros_like(out)),
        {"exact_zero": bool(torch.equal(out, torch.zeros_like(out)))},
    )


record(
    3,
    "counterexample: zero parent evidence has exact zero correction",
    zero_parent_identity,
    True,
)


def gradients_2x2():
    C, D = 3, 8
    model = make_model(classes=C, dim=D, train=True)
    args = args_for(classes=C, dim=D, h=2, w=2)
    args[0].requires_grad_(True)
    args[1].requires_grad_(True)
    out = model(*args)
    out.sum().backward()
    named = {name: parameter.grad for name, parameter in model.named_parameters()}
    finite = all(
        grad is not None and bool(torch.isfinite(grad).all())
        for grad in named.values()
    )
    keys = {
        "child_mask_last_weight": "child_mask.3.weight",
        "child_query_last_weight": "child_query.2.weight",
        "relation_last_weight": "relation.2.weight",
    }
    maxima = {
        label: float(named[key].abs().max())
        for label, key in keys.items()
    }
    return (
        finite
        and all(value > 1e-10 for value in maxima.values())
        and bool(torch.isfinite(args[0].grad).all())
        and bool(torch.isfinite(args[1].grad).all()),
        {
            "all_parameter_grads_finite": bool(finite),
            "selected_maxabs": maxima,
            "parent_input_grad_finite": bool(torch.isfinite(args[0].grad).all()),
            "child_input_grad_finite": bool(torch.isfinite(args[1].grad).all()),
        },
    )


record(
    3,
    "counterexample: train-mode 2x2 gradients are finite and mechanism paths are alive",
    gradients_2x2,
    True,
)


def gradient_1x1():
    model = make_model(train=True)
    args = args_for(h=1, w=1)
    args[0].requires_grad_(True)
    args[1].requires_grad_(True)
    out = model(*args)
    out.sum().backward()
    return bool(torch.isfinite(out).all()), {
        "accepted": True,
        "finite": bool(torch.isfinite(out).all()),
    }


record(
    3,
    "counterexample: train-mode 1x1 gradients are available",
    gradient_1x1,
    True,
)


def batch_independence():
    model = make_model(train=True)
    args = args_for(h=2, w=2, n_parent=2, n_edge=2, n_child=2)
    args[2] = torch.tensor([0, 1])
    args[3] = torch.tensor([0, 1])
    both = model(*args)
    separately = []
    for index in (0, 1):
        single = list(args)
        for position in (0, 1, 5, 7, 8, 9):
            single[position] = args[position][index:index + 1].clone()
        single[2] = torch.tensor([0], dtype=torch.long)
        single[3] = torch.tensor([0], dtype=torch.long)
        single[4] = args[4][index:index + 1].clone()
        separately.append(model(*single))
    expected = torch.cat(separately, 0)
    return (
        torch.allclose(both, expected, atol=1e-6),
        {
            "batch_output": both.tolist(),
            "separate_output": expected.tolist(),
            "max_abs_diff": float((both - expected).abs().max()),
        },
    )


record(
    3,
    "counterexample: per-parent outputs are independent within a train-mode batch",
    batch_independence,
    True,
)


def vocabulary_1203():
    model = Ledger()
    model.train()
    C, D = 1203, 256
    args = [
        torch.randn(1, D, 2, 2),
        torch.randn(1, D, 2, 2),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.ones(1, 2, 2),
        torch.softmax(torch.randn(1, C), 1),
        torch.randn(C, D),
        torch.zeros(1, 5),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([9], dtype=torch.long),
    ]
    out = model(*args)
    return (
        tuple(out.shape) == (1, C)
        and bool(torch.isfinite(out).all())
        and bool((out <= 1e-7).all()),
        {
            "shape": list(out.shape),
            "finite": bool(torch.isfinite(out).all()),
            "nonpositive": bool((out <= 1e-7).all()),
            "max": float(out.max()),
        },
    )


record(
    3,
    "counterexample: 1203-class train-mode 2x2 path remains finite",
    vocabulary_1203,
    True,
)


current_end = digest(CURRENT)
snapshot_end = digest(SNAPSHOT)
if current_end != current_start or snapshot_end != snapshot_start:
    raise SystemExit(
        "subject hash changed during audit: %s -> %s (snapshot %s -> %s)"
        % (current_start, current_end, snapshot_start, snapshot_end)
    )

report = {
    "audit": "EvidenceOwnershipLedger v5 independent CPU audit",
    "main_file": str(CURRENT),
    "snapshot": str(SNAPSHOT),
    "source_sha256_start": current_start,
    "source_sha256_end": current_end,
    "snapshot_sha256_start": snapshot_start,
    "snapshot_sha256_end": snapshot_end,
    "v4_comparator": str(V4_SNAPSHOT),
    "v4_sha256": V4_EXPECTED,
    "torch_version": torch.__version__,
    "device": "cpu",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "rounds": {
        str(i): {
            "total": sum(x["round"] == i for x in results),
            "counterexamples": sum(
                x["round"] == i and x["counterexample"] for x in results
            ),
            "failures": [
                x["name"] for x in results
                if x["round"] == i and not x["passed"]
            ],
        }
        for i in (1, 2, 3)
    },
    "tests": results,
    "failures": [x["name"] for x in results if not x["passed"]],
    "verdict": (
        "FAIL: the exact legal 1x1 single-row train-mode case is not fixed; "
        "InstanceNorm2d also regresses a previously working batch-2 1x1 case."
        if [x for x in results if not x["passed"]]
        else "PASS"
    ),
    "gpu_pilot": "NOT_ALLOWED_BY_THIS_AUDIT",
}

out = ROOT / "tmp_analysis/owner_ledger_v5_audit_evidence_20261025.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({
    "out": str(out),
    "source_sha256": current_end,
    "tests": len(results),
    "rounds": report["rounds"],
    "failures": report["failures"],
    "verdict": report["verdict"],
}, indent=2))
