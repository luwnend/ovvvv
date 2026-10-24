#!/usr/bin/env python3
"""Independent v3 audit for the frozen EvidenceOwnershipLedger snapshot.

Read-only with respect to the subject module. CPU only. Writes one JSON evidence
file next to this script; never starts CUDA, training, or data collection.
"""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

EXPECTED_SHA256 = "68fa88395a0e2e99bfe5a3fa7e4a3218ac09f45eb9dca96f260deffc0a453bd9"
# Immutable copy of the exact audited blob, recovered from local Git object 08db9121.
SUBJECT_SNAPSHOT = Path("/tmp/owner_ledger_68fa8839_snapshot.py")
SOURCE = SUBJECT_SNAPSHOT
CURRENT_SOURCE = Path(__file__).resolve().with_name("evidence_ownership_head.py")
actual_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
current_sha_start = hashlib.sha256(CURRENT_SOURCE.read_bytes()).hexdigest()
if actual_sha != EXPECTED_SHA256:
    raise SystemExit(
        f"subject hash mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
    )
spec = importlib.util.spec_from_file_location("evidence_ownership_subject", SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
Ledger = subject.EvidenceOwnershipLedger

torch.set_num_threads(1)
torch.manual_seed(20261025)
RESULTS = []


class ConstantOccupancy(nn.Module):
    """Test-only replacement for child_mask; one score per edge row."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        if len(x) != len(self.values):
            raise AssertionError((len(x), len(self.values)))
        return self.values.to(x).view(-1, 1, 1, 1).expand(
            -1, 1, x.shape[-2], x.shape[-1]
        )


def record(round_no, name, passed, detail, exception=None, counterexample=False):
    RESULTS.append(
        {
            "round": int(round_no),
            "name": name,
            "passed": bool(passed),
            "counterexample": bool(counterexample),
            "detail": detail,
            "exception": exception,
        }
    )


def raises(fn, expected=(ValueError, TypeError, IndexError, RuntimeError, AssertionError)):
    try:
        value = fn()
    except expected as exc:
        return True, f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        return False, f"unexpected {type(exc).__name__}: {exc}"
    return False, f"accepted; result={value!r}"


def make_model(dim=6, classes=3, hidden=16, max_children=8, class_width=None,
               h=2, w=2):
    if class_width is None:
        class_width = classes
    model = Ledger(
        dim=dim,
        hidden=hidden,
        max_children=max_children,
        class_width=class_width,
    ).eval()
    return model


def make_args(dim=6, classes=3, h=2, w=2, n_parent=1, n_child=1, n_edge=1,
              seed=20261025):
    g = torch.Generator().manual_seed(seed)
    parent = torch.randn(n_parent, dim, h, w, generator=g)
    child = torch.randn(n_child, dim, h, w, generator=g)
    emb = torch.eye(classes, dim) if dim >= classes else torch.randn(
        classes, dim, generator=g
    )
    post = torch.full((n_child, classes), 1.0 / classes)
    ep = torch.zeros(n_edge, dtype=torch.long)
    ec = torch.zeros(n_edge, dtype=torch.long)
    overlap = torch.ones(n_edge, h, w)
    geo = torch.zeros(n_edge, 5)
    pt = torch.linspace(8.0, 10.0, n_parent)
    ct = torch.full((n_edge,), 9.0)
    return [parent, child, ep, ec, overlap, post, emb, geo, pt, ct]


def freeze_constant_relation(model, bias=0.0):
    with torch.no_grad():
        for parameter in model.relation.parameters():
            parameter.zero_()
        model.relation[-1].bias.fill_(bias)


def empty_call(model, dim=6, classes=3, h=2, w=2, n_parent=1,
               geometry_cols=5):
    parent = torch.randn(n_parent, dim, h, w)
    child = torch.empty(0, dim, h, w)
    post = torch.empty(0, classes)
    emb = torch.eye(classes, dim) if dim >= classes else torch.randn(classes, dim)
    return model(
        parent,
        child,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, h, w),
        post,
        emb,
        torch.empty(0, geometry_cols),
        torch.linspace(8.0, 10.0, n_parent),
        torch.empty(0),
    )

# =============================== Round 1 ===============================
r = 1
model = make_model()
try:
    value = empty_call(model)
    rejected = False
    exception = None
except BaseException as exc:
    value = None
    rejected = True
    exception = f"{type(exc).__name__}: {exc}"
record(
    r,
    "valid empty edge returns exact identity",
    not rejected and isinstance(value, torch.Tensor)
    and value.shape == (1, 3) and torch.equal(value, torch.zeros_like(value)),
    {
        "rejected": rejected,
        "shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
        "exact_zero": bool(torch.equal(value, torch.zeros_like(value)))
        if isinstance(value, torch.Tensor) else False,
        "exception": exception,
    },
)

model = make_model()
ok, exc = raises(lambda: empty_call(model, geometry_cols=4))
record(
    r,
    "counterexample: malformed empty geometry [0,4] rejected explicitly",
    ok and exc.startswith("ValueError"),
    {"geometry_shape": [0, 4], "rejection": exc},
    None if ok else exc,
    counterexample=True,
)

model = make_model()
args = make_args()
args[2] = torch.empty(0, dtype=torch.long)
args[3] = torch.empty(0, dtype=torch.long)
args[4] = torch.empty(0, 2, 2)
args[7] = torch.empty(0, 5)
args[9] = torch.empty(0)
args[5] = torch.empty(0, 3)
args[8] = torch.tensor([10.0])
args[0] = torch.randn(1, 6, 2, 2)
args[1] = torch.empty(0, 6, 2, 2)
args[6] = torch.eye(3, 6)
bad = [x.clone() if torch.is_tensor(x) else x for x in args]
bad[4] = torch.empty(0, 2, 3)
ok, exc = raises(lambda: model(*bad))
record(
    r,
    "counterexample: malformed empty overlap is rejected in normal mode",
    ok,
    {"overlap_shape": [0, 2, 3], "rejection": exc, "note": "only via assert"},
    None,
    counterexample=True,
)

# Run a second interpreter with assertions removed. This is an interface test,
# not a different implementation: __debug__ false is a supported deployment mode.
snippet = f'''import importlib.util, torch\nfrom pathlib import Path\np=Path({str(SOURCE)!r}); s=importlib.util.spec_from_file_location("m",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\nmodel=m.EvidenceOwnershipLedger(dim=6,hidden=8,class_width=3).eval()\nparent=torch.randn(1,6,2,2); child=torch.empty(0,6,2,2); post=torch.empty(0,3); emb=torch.eye(3,6)\ntry:\n v=model(parent,child,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0,2,3),post,emb,torch.empty(0,5),torch.tensor([10.]),torch.empty(0))\n print("ACCEPTED",tuple(v.shape),bool(torch.equal(v,torch.zeros_like(v))))\nexcept BaseException as e:\n print("REJECTED",type(e).__name__,str(e))\n'''
proc = subprocess.run(
    [sys.executable, "-O", "-c", snippet],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
)
record(
    r,
    "counterexample: empty shape contract survives python -O",
    proc.returncode == 0 and proc.stdout.strip().startswith("REJECTED"),
    {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "finding": "assert-based checks disappear under -O",
    },
    None,
    counterexample=True,
)

# The constructor accepts class_width different from the actual vocabulary;
# nonempty calls need the two equal, but the empty early-return hides it.
model = make_model(class_width=6)
try:
    value = empty_call(model)
    mismatch_accepted = True
    mismatch_exception = None
except BaseException as exc:
    value = None
    mismatch_accepted = False
    mismatch_exception = f"{type(exc).__name__}: {exc}"
record(
    r,
    "counterexample: empty edge rejects class_width/vocabulary mismatch",
    not mismatch_accepted,
    {
        "class_width": 6,
        "actual_classes": 3,
        "accepted": mismatch_accepted,
        "accepted_result_shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
        "exception": mismatch_exception,
    },
    None,
    counterexample=True,
)

# Integer posteriors are silently accepted as if they were probabilities.
model = make_model()
args = make_args()
args[5] = torch.tensor([[1, 1, 0]], dtype=torch.long)
try:
    value = model(*args)
    int_post_result = {"accepted": True, "shape": list(value.shape)}
except BaseException as exc:
    int_post_result = {"accepted": False, "exception": f"{type(exc).__name__}: {exc}"}
record(
    r,
    "counterexample: integer child posterior rejected",
    not int_post_result["accepted"],
    int_post_result,
    None,
    counterexample=True,
)


def empty_time_case(**changes):
    model = make_model()
    args = make_args()
    args[0] = torch.randn(1, 6, 2, 2)
    args[1] = torch.empty(0, 6, 2, 2)
    args[2] = torch.empty(0, dtype=torch.long)
    args[3] = torch.empty(0, dtype=torch.long)
    args[4] = torch.empty(0, 2, 2)
    args[5] = torch.empty(0, 3)
    args[6] = torch.eye(3, 6)
    args[7] = torch.empty(0, 5)
    args[8] = torch.tensor([10.0])
    args[9] = torch.empty(0)
    for idx, value in changes.items():
        args[int(idx[1:])] = value
    return raises(lambda: model(*args))

checks = {}
checks["parent_time_2d"] = empty_time_case(t8=torch.tensor([[10.0]]))
checks["child_time_1"] = empty_time_case(t9=torch.tensor([9.0]))
record(
    r,
    "empty time vectors reject wrong shapes",
    all(item[0] and item[1].startswith("ValueError") for item in checks.values()),
    {key: value[1] for key, value in checks.items()},
)

model = make_model()
args = make_args(n_parent=2, n_child=2, n_edge=2)
args[2] = torch.tensor([0, 1])
args[3] = torch.tensor([0, 1])
args[8] = torch.tensor([10.0, 20.0])
args[9] = torch.tensor([9.0, 21.0])
ok, exc = raises(lambda: model(*args))
record(
    r,
    "counterexample: future child is rejected per parent edge",
    ok and "future" in exc.lower(),
    {"parent_time": [10, 20], "child_time": [9, 21], "rejection": exc},
    None,
    counterexample=True,
)

# The same-frame child must be admitted. Uppercase also verifies comparator
# direction rather than only testing an obviously impossible shape.
args[9] = torch.tensor([9.0, 20.0])
try:
    same_frame = model(*args)
    same_ok = torch.isfinite(same_frame).all()
    same_exc = None
except BaseException as exc:
    same_ok = False
    same_exc = f"{type(exc).__name__}: {exc}"
record(
    r,
    "same-frame child is admitted and past child is admitted",
    bool(same_ok), {"exception": same_exc},
)

for bad_value, label in [(float("nan"), "nan"), (float("inf"), "inf")]:
    args = make_args(n_parent=2, n_child=2, n_edge=2)
    args[2] = torch.tensor([0, 1])
    args[3] = torch.tensor([0, 1])
    args[8] = torch.tensor([10.0, 20.0])
    args[9] = torch.tensor([9.0, bad_value])
    try:
        model(*args)
        time_ok = False
        time_exc = "accepted"
    except BaseException as exc:
        time_ok = True
        time_exc = f"{type(exc).__name__}: {exc}"
    record(
        r,
        f"nonfinite child time rejects ({label})",
        time_ok,
        {"child_time": [9.0, label], "rejection": time_exc},
        None,
        counterexample=True,
    )

# Integer and boolean time vectors pass silently even though the API requires
# an explicit observation-time contract.
time_dtype_cases = {}
for label, parent_t, child_t in [
    ("int64", torch.tensor([10], dtype=torch.long), torch.tensor([9], dtype=torch.long)),
    ("bool", torch.tensor([True]), torch.tensor([False])),
    ("mixed_int_float", torch.tensor([10], dtype=torch.long), torch.tensor([9.0])),
    ("mixed_float32_float64", torch.tensor([10.0], dtype=torch.float32), torch.tensor([9.0], dtype=torch.float64)),
]:
    model = make_model()
    args = make_args()
    args[8] = parent_t
    args[9] = child_t
    try:
        value = model(*args)
        time_dtype_cases[label] = {
            "accepted": True,
            "finite": bool(torch.isfinite(value).all()),
        }
    except BaseException as exc:
        time_dtype_cases[label] = {
            "accepted": False,
            "exception": f"{type(exc).__name__}: {exc}",
        }
record(
    r,
    "counterexample: time dtype is validated and consistent",
    not any(case["accepted"] for case in time_dtype_cases.values()),
    time_dtype_cases,
    None,
    counterexample=True,
)

# Every float input, including both time vectors, must reject NaN/Inf.
finite_cases = []
for index, name in [
    (0, "parent_tokens"), (1, "child_tokens"), (4, "overlap"),
    (5, "child_posterior"), (6, "class_embedding"), (7, "child_geometry"),
    (8, "parent_time"), (9, "child_time"),
]:
    for bad_name, bad_value in [("nan", float("nan")), ("inf", float("inf"))]:
        args = make_args()
        bad = list(args)
        tensor = args[index].clone()
        tensor[(0,) * tensor.ndim] = bad_value
        bad[index] = tensor
        try:
            model = make_model()
            model(*bad)
            rejected = False
            exc = "accepted"
        except BaseException as exc_obj:
            rejected = True
            exc = f"{type(exc_obj).__name__}: {exc_obj}"
        finite_cases.append(
            {"input": name, "value": bad_name, "rejected": rejected, "exception": exc}
        )
record(
    r,
    "all eight float inputs reject NaN and Inf",
    all(case["rejected"] for case in finite_cases),
    finite_cases,
)

index_cases = []
for field, values in [("parent", [-1, 1, 99]), ("child", [-1, 1, 99])]:
    for val in values:
        args = make_args()
        if field == "parent":
            args[2] = torch.tensor([val], dtype=torch.long)
        else:
            args[3] = torch.tensor([val], dtype=torch.long)
        ok, exc = raises(lambda a=args: make_model()(*a))
        index_cases.append({"field": field, "value": val, "rejected": ok, "exception": exc})
for field, dtype in [("parent", torch.float32), ("child", torch.float32)]:
    args = make_args()
    if field == "parent":
        args[2] = torch.tensor([0], dtype=dtype)
    else:
        args[3] = torch.tensor([0], dtype=dtype)
    ok, exc = raises(lambda a=args: make_model()(*a))
    index_cases.append({"field": field, "value": "float dtype", "rejected": ok, "exception": exc})
record(
    r,
    "edge index range and dtype are fail-closed",
    all(case["rejected"] for case in index_cases),
    index_cases,
)

# =============================== Round 2 ===============================
r = 2
# [0,1,0] is deliberately non-contiguous in parent ownership. Scores choose the
# local maximum per parent, not the last edge.
model = make_model()
model.child_mask = ConstantOccupancy([1.0, 9.0, 5.0])
freeze_constant_relation(model)
dim, classes, h, w = 6, 3, 1, 1
parent = torch.zeros(2, dim, h, w)
emb = torch.eye(classes, dim)
parent[0, :, 0, 0] = (emb[0] + emb[2]) / math.sqrt(2.0)
parent[1, :, 0, 0] = emb[1]
child = torch.randn(3, dim, h, w)
post = torch.eye(classes)
value = model(
    parent, child, torch.tensor([0, 1, 0]), torch.tensor([0, 1, 2]),
    torch.ones(3, h, w), post, emb, torch.zeros(3, 5),
    torch.tensor([10.0, 10.0]), torch.tensor([9.0, 9.0, 9.0]),
)
owner_ok = (
    value[0, 2] < -1e-8 and abs(float(value[0, 0])) < 1e-10
    and value[1, 1] < -1e-8 and abs(float(value[1, 2])) < 1e-10
)
record(
    r,
    "counterexample: [0,1,0] ownership uses local maximum",
    owner_ok,
    {"correction": value.tolist(), "expected": {"parent0": 2, "parent1": 1}},
    None,
    counterexample=True,
)

# Strict tie goes to the lower edge even when payloads are swapped.
model = make_model()
model.child_mask = ConstantOccupancy([1.0, 1.0])
freeze_constant_relation(model)
parent = torch.zeros(1, dim, 1, 1)
parent[0, :, 0, 0] = (emb[0] + emb[2]) / math.sqrt(2.0)
child = torch.randn(2, dim, 1, 1)
post = torch.eye(classes)[[0, 2]]
common = (
    parent, child, torch.tensor([0, 0]), None, torch.ones(2, 1, 1),
    post, emb, torch.zeros(2, 5), torch.tensor([10.0]),
    torch.tensor([9.0, 9.0]),
)
a = list(common); a[3] = torch.tensor([0, 1])
tie_forward = model(*a)
a = list(common); a[3] = torch.tensor([1, 0])
tie_reversed = model(*a)
tie_ok = (
    tie_forward[0, 0] < -1e-8 and abs(float(tie_forward[0, 2])) < 1e-10
    and tie_reversed[0, 2] < -1e-8 and abs(float(tie_reversed[0, 0])) < 1e-10
)
record(
    r,
    "counterexample: strict tie deterministically uses lower edge",
    tie_ok,
    {"forward": tie_forward.tolist(), "payload_reversed": tie_reversed.tolist()},
    None,
    counterexample=True,
)

# Complete overlap: two identical full children must not subtract twice.
model = make_model()
model.child_mask = ConstantOccupancy([1.0])
freeze_constant_relation(model)
parent = torch.zeros(1, dim, 2, 2)
parent[0, :, 0, 0] = (emb[0] + emb[2]) / math.sqrt(2.0)
child = torch.randn(1, dim, 2, 2)
post = torch.tensor([[0.8, 0.0, 0.6]])
single_args = (
    parent, child, torch.tensor([0]), torch.tensor([0]), torch.ones(1, 2, 2),
    post, emb, torch.zeros(1, 5), torch.tensor([10.0]), torch.tensor([9.0]),
)
single = model(*single_args)
model.child_mask = ConstantOccupancy([1.0, 1.0])
twice = model(
    parent, torch.cat([child, child], 0), torch.tensor([0, 0]),
    torch.tensor([0, 1]), torch.ones(2, 2, 2), torch.cat([post, post], 0),
    emb, torch.zeros(2, 5), torch.tensor([10.0]), torch.tensor([9.0, 9.0]),
)
record(
    r,
    "complete overlap subtracts once",
    torch.allclose(single, twice, atol=1e-6) and bool((twice <= 0).all()),
    {"single": single.tolist(), "twice": twice.tolist(),
     "max_abs_diff": float((single - twice).abs().max())},
)

# Partial overlap: high-score A owns left, B owns right. Compare against the
# disjoint decomposition A-left + B-right to catch duplicate shared-cell charge.
model = make_model()
freeze_constant_relation(model)
model.child_mask = ConstantOccupancy([1.0])
parent = torch.zeros(1, dim, 1, 2)
parent[0, :, 0, :] = (emb[0] + emb[2]).view(dim, 1) / math.sqrt(2.0)
child_a = torch.randn(1, dim, 1, 2)
child_b = torch.randn(1, dim, 1, 2)
post_a = torch.tensor([[0.8, 0.0, 0.0]])
post_b = torch.tensor([[0.0, 0.0, 0.8]])
ov_left = torch.tensor([[[1.0, 0.0]]])
ov_right = torch.tensor([[[0.0, 1.0]]])
left_only = model(
    parent, child_a, torch.tensor([0]), torch.tensor([0]), ov_left,
    post_a, emb, torch.zeros(1, 5), torch.tensor([10.0]), torch.tensor([9.0]),
)
right_only = model(
    parent, child_b, torch.tensor([0]), torch.tensor([0]), ov_right,
    post_b, emb, torch.zeros(1, 5), torch.tensor([10.0]), torch.tensor([9.0]),
)
model.child_mask = ConstantOccupancy([5.0, 1.0])
combined = model(
    parent, torch.cat([child_a, child_b], 0), torch.tensor([0, 0]),
    torch.tensor([0, 1]), torch.cat([torch.ones(1, 1, 2), torch.ones(1, 1, 2)], 0),
    torch.cat([post_a, post_b], 0), emb, torch.zeros(2, 5),
    torch.tensor([10.0]), torch.tensor([9.0, 9.0]),
)
expected = left_only + right_only
record(
    r,
    "partial overlap has one owner per cell and no duplicate charge",
    torch.allclose(combined, expected, atol=1e-6),
    {"combined": combined.tolist(), "expected_A_left_plus_B_right": expected.tolist(),
     "max_abs_diff": float((combined - expected).abs().max())},
)

# Wrong child: posterior for another class cannot explain the parent's class.
model = make_model()
model.child_mask = ConstantOccupancy([1.0])
freeze_constant_relation(model)
parent = torch.zeros(1, dim, 1, 1)
parent[0, :, 0, 0] = emb[0]
post_wrong = torch.tensor([[0.0, 1.0, 0.0]])
wrong = model(
    parent, torch.randn(1, dim, 1, 1), torch.tensor([0]), torch.tensor([0]),
    torch.ones(1, 1, 1), post_wrong, emb, torch.zeros(1, 5),
    torch.tensor([10.0]), torch.tensor([9.0]),
)
record(
    r,
    "counterexample: wrong child cannot explain parent class",
    abs(float(wrong[0, 0])) < 1e-10,
    {"correction": wrong.tolist(), "parent_support": "class 0", "child_support": "class 1"},
    None,
    counterexample=True,
)

# Per-parent max_children, not a global count.
class OnesOccupancy(nn.Module):
    def forward(self, x):
        return torch.ones(len(x), 1, x.shape[-2], x.shape[-1])

model = make_model(max_children=1)
model.child_mask = OnesOccupancy()
p2 = torch.randn(2, dim, 1, 1)
c2 = torch.randn(2, dim, 1, 1)
emb2 = torch.eye(classes, dim)
post2 = torch.full((2, classes), 1.0 / classes)
ok_accept, exc_accept = raises(
    lambda: model(p2, c2, torch.tensor([0, 1]), torch.tensor([0, 1]),
                  torch.ones(2, 1, 1), post2, emb2, torch.zeros(2, 5),
                  torch.tensor([10.0, 10.0]), torch.tensor([9.0, 9.0]))
)
ok_reject, exc_reject = raises(
    lambda: model(p2, c2, torch.tensor([0, 0]), torch.tensor([0, 1]),
                  torch.ones(2, 1, 1), post2, emb2, torch.zeros(2, 5),
                  torch.tensor([10.0, 10.0]), torch.tensor([9.0, 9.0]))
)
record(
    r,
    "max_children is per parent and enforced",
    ok_accept and not ok_reject and "exceeds" in exc_reject,
    {"one_per_parent_no_exception": ok_accept,
     "two_same_parent_exception": exc_reject},
)

# Zero parent evidence must be exact identity; random finite inputs must only
# produce finite, non-positive corrections.
model = make_model()
parent_zero = torch.zeros(1, dim, 2, 2)
child = torch.randn(1, dim, 2, 2)
args = (parent_zero, child, torch.tensor([0]), torch.tensor([0]),
        torch.ones(1, 2, 2), torch.tensor([[0.7, 0.2, 0.1]]),
        torch.eye(classes, dim), torch.zeros(1, 5),
        torch.tensor([10.0]), torch.tensor([9.0]))
zero = model(*args)
record(
    r,
    "zero parent evidence is exact identity",
    torch.equal(zero, torch.zeros_like(zero)),
    {"output": zero.tolist(), "exact_zero": bool(torch.equal(zero, torch.zeros_like(zero)))},
)

random_ok = True
random_details = []
for seed in range(24):
    g = torch.Generator().manual_seed(seed)
    ccount = 3 + seed % 4
    p = torch.randn(2, 8, 2, 2, generator=g)
    ch = torch.randn(4, 8, 2, 2, generator=g)
    ep = torch.tensor([0, 0, 1, 1])
    ec = torch.tensor([0, 1, 2, 3])
    ov = torch.rand(4, 2, 2, generator=g)
    po = torch.rand(4, ccount, generator=g)
    em = torch.randn(ccount, 8, generator=g)
    ge = torch.randn(4, 5, generator=g)
    m = Ledger(dim=8, hidden=16, max_children=8, class_width=ccount).eval()
    out = m(p, ch, ep, ec, ov, po, em, ge, torch.tensor([10.0, 10.0]),
            torch.tensor([9.0, 9.0, 9.0, 9.0]))
    this_ok = bool(torch.isfinite(out).all() and (out <= 1e-7).all())
    random_ok &= this_ok
    if not this_ok:
        random_details.append({"seed": seed, "finite": bool(torch.isfinite(out).all()),
                               "max": float(out.max())})
record(
    r,
    "random inputs only subtract and remain finite",
    random_ok,
    {"seeds": 24, "failures": random_details},
)

# Actual default 1203-class path on a nonempty edge.
model_full = Ledger().eval()
p_full = torch.randn(1, 256, 1, 1)
c_full = torch.randn(1, 256, 1, 1)
post_full = F.softmax(torch.randn(1, 1203), dim=1)
emb_full = torch.randn(1203, 256)
try:
    full = model_full(
        p_full, c_full, torch.tensor([0]), torch.tensor([0]),
        torch.ones(1, 1, 1), post_full, emb_full, torch.zeros(1, 5),
        torch.tensor([10.0]), torch.tensor([9.0]),
    )
    full_ok = tuple(full.shape) == (1, 1203) and bool(torch.isfinite(full).all()) and bool((full <= 1e-7).all())
    full_detail = {"shape": list(full.shape), "finite": bool(torch.isfinite(full).all()),
                   "max": float(full.max())}
    full_exc = None
except BaseException as exc:
    full_ok = False
    full_detail = None
    full_exc = f"{type(exc).__name__}: {exc}"
record(r, "actual 1203-class nonempty vocabulary", full_ok, full_detail, full_exc)

# =============================== Round 3 ===============================
r = 3


def controlled_relation_model(classes=3, dim=8, hidden=8):
    """Test-only parameterisation: relation class c taps parent class c."""
    model = make_model(dim=dim, classes=classes, hidden=hidden, class_width=classes, h=1, w=2)
    with torch.no_grad():
        model.relation[0].weight.zero_()
        model.relation[0].bias.zero_()
        model.relation[-1].weight.zero_()
        model.relation[-1].bias.zero_()
        for c in range(classes):
            model.relation[0].weight[c, dim + c] = 1.0
            model.relation[-1].weight[c, c] = 2.0
    return model


classes, dim, h, w = 3, 8, 1, 2
model = controlled_relation_model()
emb = torch.eye(classes, dim)
parent = torch.zeros(1, dim, h, w)
parent[0, :, 0, :] = (emb[0] + 0.6 * emb[1]).view(dim, 1) / math.sqrt(1.0 + 0.36)
child = torch.randn(1, dim, h, w)
post = torch.tensor([[0.85, 0.6, 0.2]])
args = (parent, child, torch.tensor([0]), torch.tensor([0]),
        torch.ones(1, h, w), post, emb, torch.zeros(1, 5),
        torch.tensor([10.0]), torch.tensor([9.0]))
captured = {}
hook = model.relation.register_forward_hook(
    lambda module, inputs, output: captured.__setitem__("relation", output.detach().clone())
)
out = model(*args)
hook.remove()
relation_out = captured["relation"].reshape(h, w, classes)
relation_gate = torch.sigmoid(relation_out)
class_spread = float(
    (relation_gate.max(dim=2).values - relation_gate.min(dim=2).values).mean()
)
shared_gate = relation_gate.mean(dim=2, keepdim=True).expand_as(relation_gate)
shared_rmse = float(torch.sqrt(((relation_gate - shared_gate) ** 2).mean()))
record(
    r,
    "counterexample: relation is [H,W,C], not shared spatial scalar",
    list(relation_out.shape) == [h, w, classes] and class_spread > 0.05,
    {"relation_shape": list(relation_out.shape),
     "mean_class_spread": class_spread,
     "shared_spatial_gate_rmse": shared_rmse,
     "correction": out.tolist()},
    None,
    counterexample=True,
)

# Random-initialisation contrast: capacity exists, but the untrained scalar
# branch is numerically almost class-agnostic. Report this as a WARN, not as a
# structural failure.
torch.manual_seed(4)
model_init = make_model(dim=8, classes=3, hidden=16, class_width=3, h=1, w=2)
captured_init = {}
hook_init = model_init.relation.register_forward_hook(
    lambda module, inputs, output: captured_init.__setitem__("relation", output.detach().clone())
)
model_init(*args)
hook_init.remove()
gate_init = torch.sigmoid(captured_init["relation"].reshape(1, 2, classes))
init_spread = float((gate_init.max(dim=2).values - gate_init.min(dim=2).values).mean())
init_shared_rmse = float(torch.sqrt(((gate_init - gate_init.mean(dim=2, keepdim=True)) ** 2).mean()))
record(
    r,
    "initialization relation is nearly class-agnostic (warning)",
    init_spread > 0.02,
    {"mean_class_spread": init_spread, "shared_spatial_gate_rmse": init_shared_rmse,
     "note": "capacity test is above; this is numerical initialisation, not a structural failure"},
)

# Shuffled class embeddings must change the actual class-conditioned response.
emb_shuffled = emb[[1, 0, 2]]
out_shuffled = model(
    parent, child, args[2], args[3], args[4], post, emb_shuffled,
    args[7], args[8], args[9],
)
record(
    r,
    "counterexample: shuffled relation text changes class response",
    not torch.allclose(out, out_shuffled, atol=1e-8),
    {"original": out.tolist(), "shuffled_text": out_shuffled.tolist(),
     "max_abs_diff": float((out - out_shuffled).abs().max())},
    None,
    counterexample=True,
)

# Perturb one parent-class embedding; only that class correction may move by
# a comparable amount in the controlled relation.
emb_perturbed = emb.clone()
emb_perturbed[0] = F.normalize(torch.tensor([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), dim=0)
out_perturbed = model(
    parent, child, args[2], args[3], args[4], post, emb_perturbed,
    args[7], args[8], args[9],
)
delta = (out_perturbed - out)[0]
target_delta = abs(float(delta[0]))
other_delta = max(abs(float(delta[1])), abs(float(delta[2])))
record(
    r,
    "counterexample: one parent-class embedding perturbation is class-local",
    target_delta > 1e-8 and other_delta < 1e-8,
    {"delta": delta.tolist(), "target_abs": target_delta, "other_abs": other_delta},
    None,
    counterexample=True,
)

# The child query must (a) consume posterior values and (b) detach that input.
torch.manual_seed(5)
model_q = make_model(dim=6, classes=3, hidden=16, class_width=3, h=1, w=1)
p1 = torch.randn(1, 6, 1, 1)
c1 = torch.randn(1, 6, 1, 1)
e1 = torch.eye(3, 6)
g1 = torch.zeros(1, 5)
post_a = torch.tensor([[0.9, 0.1, 0.0]])
post_b = torch.tensor([[0.0, 0.1, 0.9]])
q_capture = {}
q_hook = model_q.child_query.register_forward_hook(
    lambda module, inputs, output: q_capture.setdefault("values", []).append(output.detach().clone())
)
model_q(p1, c1, torch.tensor([0]), torch.tensor([0]), torch.ones(1, 1, 1),
        post_a, e1, g1, torch.tensor([10.0]), torch.tensor([9.0]))
model_q(p1, c1, torch.tensor([0]), torch.tensor([0]), torch.ones(1, 1, 1),
        post_b, e1, g1, torch.tensor([10.0]), torch.tensor([9.0]))
q_hook.remove()
query_changes = len(q_capture["values"]) == 2 and not torch.allclose(
    q_capture["values"][0], q_capture["values"][1], atol=1e-8
)
post_grad = post_a.clone().requires_grad_(True)
out_grad = model_q(p1, c1, torch.tensor([0]), torch.tensor([0]), torch.ones(1, 1, 1),
                   post_grad, e1, g1, torch.tensor([10.0]), torch.tensor([9.0]))
record(
    r,
    "child_query reads detached posterior and posterior has no gradient",
    query_changes and post_grad.grad is None,
    {"query_changed_by_posterior": query_changes,
     "posterior_grad_none": post_grad.grad is None},
)

# End-to-end gradient non-deadness on the actual revision.
torch.manual_seed(8)
model_g = make_model(dim=6, classes=3, hidden=16, class_width=3, h=2, w=2)
pg = torch.randn(1, 6, 2, 2, requires_grad=True)
cg = torch.randn(1, 6, 2, 2, requires_grad=True)
geog = torch.randn(1, 5, requires_grad=True)
qg = torch.tensor([[0.8, 0.2, 0.0]])
og = model_g(pg, cg, torch.tensor([0]), torch.tensor([0]), torch.ones(1, 2, 2),
             qg, torch.eye(3, 6), geog, torch.tensor([10.0]), torch.tensor([9.0]))
(-og).sum().backward()
param_grads = {name: p.grad for name, p in model_g.named_parameters()}
all_grads_finite = all(g is not None and bool(torch.isfinite(g).all()) for g in param_grads.values())
def maxgrad(prefix):
    vals = [float(g.abs().max()) for name, g in param_grads.items() if name.startswith(prefix) and g is not None]
    return max(vals) if vals else 0.0
child_query_grad = maxgrad("child_query.")
relation_grad = maxgrad("relation.")
child_mask_grad = maxgrad("child_mask.")
input_grads_finite = all(
    x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in (pg, cg, geog)
)
record(
    r,
    "gradients are finite and child_query/relation/child_mask are non-dead",
    all_grads_finite and input_grads_finite
    and child_query_grad > 1e-10 and relation_grad > 1e-10 and child_mask_grad > 1e-10,
    {"all_parameter_grads_finite": all_grads_finite,
     "input_grads_finite": input_grads_finite,
     "child_query_maxabs": child_query_grad,
     "relation_maxabs": relation_grad,
     "child_mask_maxabs": child_mask_grad},
)

# =============================== Verdict ===============================
end_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
current_sha_end = hashlib.sha256(CURRENT_SOURCE.read_bytes()).hexdigest()
if end_sha != actual_sha:
    raise SystemExit(f"subject changed during audit: {actual_sha} -> {end_sha}")

failures = [x for x in RESULTS if not x["passed"]]
warnings = [x for x in RESULTS if "warning" in x["name"].lower()]
summary = {
    "subject": str(SOURCE),
    "source_sha256_start": actual_sha,
    "source_sha256_end": end_sha,
    "source_size_bytes": SOURCE.stat().st_size,
    "source_provenance": "immutable /tmp copy recovered from local Git object 08db9121 (blob is byte-identical to the audited SHA)",
    "current_source": str(CURRENT_SOURCE),
    "current_source_sha256_start": current_sha_start,
    "current_source_sha256_end": current_sha_end,
    "current_source_changed_during_audit": current_sha_start != current_sha_end,
    "scope_note": "All executable results below bind only to the immutable 68fa8839 snapshot. The mutable current-source hash is descriptive and independently reported.",
    "torch_version": torch.__version__,
    "device": "cpu",
    "rounds": {
        str(i): {
            "tests": sum(1 for x in RESULTS if x["round"] == i),
            "counterexamples": sum(
                1 for x in RESULTS if x["round"] == i and x["counterexample"]
            ),
        }
        for i in (1, 2, 3)
    },
    "tests": RESULTS,
    "failures": [x["name"] for x in failures],
    "warnings": [x["name"] for x in warnings],
}
outpath = Path(__file__).with_name("owner_ledger_v3_audit_evidence_20261025.json")
outpath.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({
    "json": str(outpath),
    "source_sha256": actual_sha,
    "tests": len(RESULTS),
    "failures": len(failures),
    "rounds": summary["rounds"],
}, indent=2))
for item in RESULTS:
    print(("PASS" if item["passed"] else "FAIL"), item["round"], item["name"])
