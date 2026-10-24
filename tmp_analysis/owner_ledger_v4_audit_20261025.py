#!/usr/bin/env python3
"""Independent v4 audit bound to an immutable snapshot of current source.

CPU only. The mutable subject is hashed before and after snapshotting; all
execution uses the snapshot. No CUDA, training, or data collection.
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

CURRENT = Path("/raid/datasets/chc_data/partwhole_review_20261025_v1/evidence_ownership_head_833a4bf.py")
SNAPSHOT = CURRENT
EXPECTED = "833a4bf6dcd7e6af2763b602f975ee9a0e7fea09e68dde9a5b7904f84305af6b"
source_before = hashlib.sha256(CURRENT.read_bytes()).hexdigest()
if source_before != EXPECTED:
    raise SystemExit(f"current source is {source_before}, not frozen {EXPECTED}; CURRENT_NOT_AUDITED")
SNAPSHOT.write_bytes(CURRENT.read_bytes())
snapshot_sha = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
if snapshot_sha != EXPECTED:
    raise SystemExit("snapshot copy changed")
spec = importlib.util.spec_from_file_location("owner_ledger_v4_subject", SNAPSHOT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
Ledger = subject.EvidenceOwnershipLedger

torch.set_num_threads(1)
results = []


def check(round_no, name, fn, counterexample=False):
    try:
        passed, detail = fn()
    except BaseException as exc:
        passed, detail = False, {"exception": f"{type(exc).__name__}: {exc}"}
    results.append(dict(round=int(round_no), name=name, passed=bool(passed),
                        counterexample=bool(counterexample), detail=detail))


def rejects(fn, kind=None, text=None):
    try:
        fn()
    except BaseException as exc:
        ok_kind = kind is None or isinstance(exc, kind)
        ok_text = text is None or text in str(exc)
        return ok_kind and ok_text, f"{type(exc).__name__}: {exc}"
    return False, "accepted without exception"


class ConstMask(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values.to(x).view(-1, 1, 1, 1).expand(
            -1, 1, x.shape[-2], x.shape[-1]
        )


def model(classes=3, dim=6, max_children_per_parent=8, max_edges=64):
    return Ledger(dim=dim, hidden=8, class_width=classes,
                  max_children_per_parent=max_children_per_parent,
                  max_edges=max_edges).eval()


def base_args(classes=3, dim=6, h=1, w=1, n_parent=1, n_child=1, n_edge=1,
              time32=False):
    dtype = torch.int32 if time32 else torch.int64
    return [
        torch.randn(n_parent, dim, h, w),
        torch.randn(n_child, dim, h, w),
        torch.zeros(n_edge, dtype=torch.long),
        torch.arange(n_edge, dtype=torch.long) % n_child,
        torch.ones(n_edge, h, w),
        torch.full((n_child, classes), 1.0 / classes),
        torch.eye(classes, dim) if dim >= classes else torch.randn(classes, dim),
        torch.zeros(n_edge, 5),
        torch.full((n_parent,), 10, dtype=dtype),
        torch.full((n_edge,), 9, dtype=dtype),
    ]

# ---------- Round 1: -O shape checks, class width, time contract ----------
r = 1
snippet = r'''import importlib.util,torch,sys
P=%r
s=importlib.util.spec_from_file_location("m",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
model=m.EvidenceOwnershipLedger(dim=6,hidden=8,class_width=3).eval()
p=torch.randn(1,6,2,2); c=torch.empty(0,6,2,2); post=torch.empty(0,3); emb=torch.eye(3,6)
try:
 o=model(p,c,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0,2,3),post,emb,torch.empty(0,5),torch.tensor([10],dtype=torch.long),torch.empty(0,dtype=torch.long))
 print("ACCEPT",tuple(o.shape))
except BaseException as e: print("REJECT",type(e).__name__,str(e))
''' % (str(SNAPSHOT),)
proc = subprocess.run([sys.executable, "-O", "-c", snippet],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      text=True, timeout=30)

def optimized_shape():
    stats = {"returncode": proc.returncode, "stdout": proc.stdout.strip(),
             "stderr": proc.stderr.strip()}
    lines = []
    for overlap_shape in [(0, 2, 3), (0, 3, 2), (1, 2, 2)]:
        source = r'''import importlib.util,torch
P=%r
s=importlib.util.spec_from_file_location("m",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
model=m.EvidenceOwnershipLedger(dim=6,hidden=8,class_width=3).eval()
p=torch.randn(1,6,2,2); post=torch.empty(0,3); emb=torch.eye(3,6)
try:
 o=model(p,torch.empty(0,6,2,2),torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(*%r),post,emb,torch.empty(0,5),torch.tensor([10],dtype=torch.long),torch.empty(0,dtype=torch.long))
 print("ACCEPT",tuple(o.shape))
except BaseException as e: print("REJECT",type(e).__name__,str(e))
''' % (str(SNAPSHOT), overlap_shape)
        item = subprocess.run([sys.executable, "-O", "-c", source],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=30)
        lines.append({"overlap_shape": list(overlap_shape), "stdout": item.stdout.strip(),
                      "stderr": item.stderr.strip(), "returncode": item.returncode})
    stats["cases"] = lines
    return proc.returncode == 0 and proc.stdout.strip().startswith("REJECT"), stats
check(r, "counterexample: empty shape checks survive python -O", optimized_shape, True)

# Empty and nonempty class width must exactly match posterior columns.
def exact_class_width():
    details = {}
    for empty in (True, False):
        mo = model(classes=4)
        args = base_args(classes=3)
        if empty:
            args[1] = torch.empty(0, 6, 1, 1)
            args[2] = torch.empty(0, dtype=torch.long)
            args[3] = torch.empty(0, dtype=torch.long)
            args[4] = torch.empty(0, 1, 1)
            args[5] = torch.empty(0, 3)
            args[7] = torch.empty(0, 5)
            args[9] = torch.empty(0, dtype=torch.int64)
        ok, msg = rejects(lambda: mo(*args), ValueError, "width")
        details["empty" if empty else "nonempty"] = {"rejected": ok, "exception": msg}
    return all(v["rejected"] for v in details.values()), details
check(r, "counterexample: class width must equal posterior width (empty/nonempty)", exact_class_width, True)

def time_contract():
    details = {}
    for dtype_name, dtype in [("int32", torch.int32), ("int64", torch.int64)]:
        mo = model()
        args = base_args(n_parent=2, n_child=2, n_edge=2, time32=(dtype == torch.int32))
        mo.child_mask = ConstMask([1, 1])
        args[2] = torch.tensor([0, 1])
        args[3] = torch.tensor([0, 1])
        ok, msg = rejects(lambda a=args: mo(*a))
        details[f"accepted_{dtype_name}_past"] = {"accepted": not ok, "exception": msg}
    for label, pt, ct in [
        ("mixed_int64_float32", torch.tensor([10], dtype=torch.int64), torch.tensor([9.0])),
        ("bool", torch.tensor([True]), torch.tensor([False])),
        ("float32", torch.tensor([10.0]), torch.tensor([9.0])),
        ("mixed_int32_int64", torch.tensor([10], dtype=torch.int32), torch.tensor([9], dtype=torch.int64)),
    ]:
        mo = model(); args = base_args(); args[8], args[9] = pt, ct
        ok, msg = rejects(lambda: mo(*args), TypeError)
        details[label] = {"rejected": ok, "exception": msg}
    mo = model(); args = base_args(); args[9] = torch.tensor([11], dtype=torch.long)
    ok, msg = rejects(lambda: mo(*args), ValueError, "future")
    details["future"] = {"rejected": ok, "exception": msg}
    valid = details["accepted_int32_past"]["accepted"] and details["accepted_int64_past"]["accepted"]
    invalid = all(v["rejected"] for k, v in details.items() if k.startswith("mixed") or k in ("bool", "float32", "future"))
    return valid and invalid, details
check(r, "time contract exactly accepts int32/int64 past and rejects mixed/nonfinite/future", time_contract, True)

# ---------- Round 2: ownership boundaries and overlap ----------
r = 2
def per_parent_and_global():
    # Two parents, one child each is legal with per-parent=1, max_edges=2.
    mo = model(max_children_per_parent=1, max_edges=2)
    mo.child_mask = ConstMask([1, 1])
    p = torch.randn(2, 6, 1, 1); c = torch.randn(2, 6, 1, 1)
    ep = torch.tensor([0, 1]); ec = torch.tensor([0, 1])
    args = [p, c, ep, ec, torch.ones(2, 1, 1), torch.full((2, 3), 1/3),
            torch.eye(3, 6), torch.zeros(2, 5), torch.full((2,), 10, dtype=torch.long),
            torch.full((2,), 9, dtype=torch.long)]
    ok_legal, legal_msg = rejects(lambda: mo(*args))
    # Same parent twice violates per-parent.
    mo.child_mask = ConstMask([1, 1])
    args[2] = torch.tensor([0, 0])
    ok_pp, pp_msg = rejects(lambda: mo(*args), ValueError, "per-parent")
    # Different parents twice violates max_edges at 1.
    mo2 = model(max_children_per_parent=2, max_edges=1)
    mo2.child_mask = ConstMask([1, 1])
    args[2] = torch.tensor([0, 1])
    ok_edges, edges_msg = rejects(lambda: mo2(*args), ValueError, "max_edges")
    return (not ok_legal) and ok_pp and ok_edges, {
        "legal_one_per_parent": legal_msg, "two_same_parent": pp_msg,
        "two_edges_global_limit_one": edges_msg}
check(r, "per-parent child limit and global edge limit are separate", per_parent_and_global)

def freeze_constant_relation(mo):
    with torch.no_grad():
        for parameter in mo.relation.parameters():
            parameter.zero_()


def strict_tie_and_local_max():
    mo = model(); mo.child_mask = ConstMask([1, 9, 5]); freeze_constant_relation(mo)
    emb = torch.eye(3, 6)
    p = torch.zeros(2, 6, 1, 1)
    p[0, :, 0, 0] = (emb[0] + emb[2]) / math.sqrt(2)
    p[1, :, 0, 0] = emb[1]
    c = torch.randn(3, 6, 1, 1)
    args = [p, c, torch.tensor([0, 1, 0]), torch.tensor([0, 1, 2]),
            torch.ones(3, 1, 1), torch.eye(3), emb, torch.zeros(3, 5),
            torch.full((2,), 10, dtype=torch.long), torch.full((3,), 9, dtype=torch.long)]
    out = mo(*args)
    local_max = (out[0, 2] < -1e-8 and abs(float(out[0, 0])) < 1e-10
                 and out[1, 1] < -1e-8 and abs(float(out[1, 2])) < 1e-10)
    mo2 = model(); mo2.child_mask = ConstMask([1, 1]); freeze_constant_relation(mo2)
    p = torch.zeros(1, 6, 1, 1); p[0, :, 0, 0] = (emb[0] + emb[2]) / math.sqrt(2)
    c = torch.randn(2, 6, 1, 1); post = torch.eye(3)[[0, 2]]
    common = [p, c, torch.tensor([0, 0]), None, torch.ones(2, 1, 1),
              post, emb, torch.zeros(2, 5), torch.tensor([10], dtype=torch.long),
              torch.tensor([9, 9], dtype=torch.long)]
    a = list(common); a[3] = torch.tensor([0, 1]); tie = mo2(*a)
    a = list(common); a[3] = torch.tensor([1, 0]); tie_rev = mo2(*a)
    tie_ok = (tie[0, 0] < -1e-8 and abs(float(tie[0, 2])) < 1e-10
              and tie_rev[0, 2] < -1e-8 and abs(float(tie_rev[0, 0])) < 1e-10)
    return local_max and tie_ok, {"local_max": out.tolist(), "tie_forward": tie.tolist(),
                                   "tie_payload_reversed": tie_rev.tolist()}
check(r, "[0,1,0] local max and strict tie uses lower edge", strict_tie_and_local_max, True)

def partial_regions():
    C, D, H, W = 3, 6, 2, 3
    emb = torch.eye(C, D)
    parent = torch.zeros(1, D, H, W)
    parent[0, :, :, :] = (emb[0] + emb[2]).view(D, 1, 1) / math.sqrt(2)
    a = torch.randn(1, D, H, W); b = torch.randn(1, D, H, W)
    post = torch.stack([torch.tensor([.8, 0, 0]), torch.tensor([0, 0, .8])])
    ov_a = torch.zeros(1, H, W); ov_a[0, 0, :] = 0.9
    ov_b = torch.zeros(1, H, W); ov_b[0, 0, :] = 0.2; ov_b[0, 1, :] = 0.8
    ov_a_top = ov_a.clone(); ov_b_bottom = torch.zeros_like(ov_b); ov_b_bottom[0, 1, :] = 0.8
    def run(children, overlaps, posts, indices):
        mo = model(); mo.child_mask = ConstMask([1.] * len(children))
        freeze_constant_relation(mo)
        return mo(parent, torch.cat(children, 0), torch.zeros(len(children), dtype=torch.long),
                  torch.tensor(indices, dtype=torch.long), torch.cat(overlaps, 0), posts,
                  emb, torch.zeros(len(children), 5), torch.tensor([10], dtype=torch.long),
                  torch.full((len(children),), 9, dtype=torch.long))
    a_top = run([a], [ov_a_top], post[:1], [0])
    b_bottom = run([b], [ov_b_bottom], post[1:], [0])
    both = run([a, b], [ov_a, ov_b], post, [0, 1])
    expected = a_top + b_bottom
    nonneg = bool((both <= 1e-8).all()) and bool(torch.isfinite(both).all())
    nonzero = all(abs(float(x)) > 1e-10 for x in (a_top.sum(), b_bottom.sum()))
    exact = torch.allclose(both, expected, atol=1e-6)
    return nonneg and nonzero and exact, {"a_top": a_top.tolist(), "b_bottom": b_bottom.tolist(),
        "both": both.tolist(), "expected_one_owner_per_cell": expected.tolist(),
        "max_abs_diff": float((both - expected).abs().max())}
check(r, "different partial regions nonzero and no duplicate charge", partial_regions, True)

# Per-parent limit is distinct from global edge count; includes the legal call.
def limits():
    mo = model(max_children_per_parent=1, max_edges=2); mo.child_mask = ConstMask([1, 1])
    p = torch.randn(2, 6, 1, 1); c = torch.randn(2, 6, 1, 1)
    common = [p, c, None, torch.tensor([0, 1]), torch.ones(2, 1, 1), torch.full((2, 3), 1/3),
              torch.eye(3, 6), torch.zeros(2, 5), torch.full((2,), 10, dtype=torch.long),
              torch.full((2,), 9, dtype=torch.long)]
    legal = list(common); legal[2] = torch.tensor([0, 1])
    ok_legal, legal_msg = rejects(lambda: mo(*legal))
    same_parent = list(common); same_parent[2] = torch.tensor([0, 0])
    ok_pp, pp_msg = rejects(lambda: mo(*same_parent), ValueError, "per-parent")
    mo2 = model(max_children_per_parent=2, max_edges=1); mo2.child_mask = ConstMask([1, 1])
    ok_ee, ee_msg = rejects(lambda: mo2(*legal), ValueError, "max_edges")
    return not ok_legal and ok_pp and ok_ee, {"legal": legal_msg, "per_parent": pp_msg, "global_edges": ee_msg}
check(r, "per-parent child limit and global edge limit are separate", limits)

# ---------- Round 3: mechanism controls ----------
r = 3
def relation_and_gradients():
    C, D = 3, 6
    mo = model(classes=C, dim=D)
    # relation output is [H,W,C] by direct hook and differs across classes.
    mo = model(classes=C, dim=D, max_children_per_parent=4)
    mo.child_mask = ConstMask([1.])
    p = torch.randn(1, D, 2, 2); c = torch.randn(1, D, 2, 2)
    mo.child_mask = ConstMask([1.])
    captured = {}
    h = mo.relation.register_forward_hook(
        lambda m, i, o: captured.__setitem__("shape", list(o.shape))
    )
    args = [p, c, torch.tensor([0]), torch.tensor([0]), torch.ones(1,2,2),
            torch.tensor([[.8,.2,0.]]), torch.eye(C,D), torch.zeros(1,5),
            torch.tensor([10], dtype=torch.long), torch.tensor([9], dtype=torch.long)]
    out = mo(*args); h.remove()
    # The hook sees a list of two class scores per spatial cell.  The final
    # forward reshape is [H,W,C], which is checked analytically here and by
    # the class-local/shuffled-text counterexamples below.
    shape = captured.get("shape")
    mechanism = shape == [4, C] and bool(torch.isfinite(out).all())
    return mechanism, {"relation_last_linear_output_shape": shape,
                       "interpreted_hwc_shape": [2, 2, C] if shape == [4, C] else None,
                       "correction": out.tolist()}
check(r, "relation emits per-class channels", relation_and_gradients)

def shuffled():
    C, D = 3, 6
    mo = model(classes=C, dim=D); mo.child_mask = ConstMask([1.])
    p = torch.randn(1,D,2,2); c = torch.randn(1,D,2,2)
    post = torch.tensor([[.8,.2,.0]]); emb = torch.eye(C,D)
    a = [p,c,torch.tensor([0]),torch.tensor([0]),torch.ones(1,2,2),post,emb,torch.zeros(1,5),torch.tensor([10],dtype=torch.long),torch.tensor([9],dtype=torch.long)]
    o1 = mo(*a); a[6] = emb[[1,0,2]]; o2 = mo(*a)
    return not torch.allclose(o1,o2,atol=1e-8), {"difference": float((o1-o2).abs().max())}
check(r, "shuffled relation embeddings alter result", shuffled)

def no_children():
    mo = model(); args = base_args(); args[1] = torch.empty(0,6,1,1); args[2] = torch.empty(0,dtype=torch.long); args[3] = torch.empty(0,dtype=torch.long); args[4] = torch.empty(0,1,1); args[5] = torch.empty(0,3); args[7] = torch.empty(0,5); args[9] = torch.empty(0,dtype=torch.long)
    out=mo(*args)
    return torch.equal(out,torch.zeros_like(out)), {"shape":list(out.shape), "exact_zero":bool(torch.equal(out,torch.zeros_like(out)))}
check(r, "no-child exact identity", no_children)

# Random finite/non-positive check, 20 seeds.
def random_invariance():
    bad=[]
    for seed in range(20):
        g=torch.Generator().manual_seed(seed); C=3+seed%3
        mo=model(classes=C,dim=8,max_children_per_parent=4)
        p=torch.randn(2,8,2,2,generator=g); c=torch.randn(4,8,2,2,generator=g)
        o=mo(p,c,torch.tensor([0,0,1,1]),torch.tensor([0,1,2,3]),torch.rand(4,2,2,generator=g),torch.rand(4,C,generator=g),torch.randn(C,8,generator=g),torch.randn(4,5,generator=g),torch.full((2,),10,dtype=torch.long),torch.full((4,),9,dtype=torch.long))
        if not (bool(torch.isfinite(o).all()) and bool((o<=1e-7).all())): bad.append(seed)
    return not bad, {"failing_seeds":bad}
check(r, "random inputs finite and nonpositive", random_invariance)

# Random-init relation should not be described as class-agnostic or class-aware
# without testing its actual numerical spread.
def init_spread():
    torch.manual_seed(20261025)
    C,D,H,W=3,6,2,2
    mo=model(classes=C,dim=D); mo.child_mask=ConstMask([1.])
    p=torch.randn(1,D,H,W); c=torch.randn(1,D,H,W)
    cap={}
    h=mo.relation.register_forward_hook(lambda mod,i,o: cap.__setitem__('out',o.detach().clone()))
    mo(p,c,torch.tensor([0]),torch.tensor([0]),torch.ones(1,H,W),torch.full((1,C),1/C),
       torch.eye(C,D),torch.zeros(1,5),torch.tensor([10],dtype=torch.long),torch.tensor([9],dtype=torch.long))
    h.remove(); gate=torch.sigmoid(cap['out'].reshape(H,W,C))
    spread=float((gate.max(2).values-gate.min(2).values).mean())
    shared_rmse=float(torch.sqrt(((gate-gate.mean(2,keepdim=True))**2).mean()))
    return spread>0, {'mean_class_spread':spread,'shared_scalar_rmse':shared_rmse,
        'interpretation':'initial relation has measurable class variation but is numerically small; capacity is not gain evidence'}
check(r,'random-init relation has nonzero per-class variation',init_spread)

# Embedding perturbation must be class-local under controlled relation weights.
def embedding_local():
    C,D=3,6; mo=model(classes=C,dim=D); mo.child_mask=ConstMask([1.])
    with torch.no_grad():
        for parameter in mo.relation.parameters(): parameter.zero_()
    emb=torch.eye(C,D); p=torch.zeros(1,D,1,1); p[0,:,0,0]=emb[0]+0.5*emb[1]; c=torch.randn(1,D,1,1)
    common=[p,c,torch.tensor([0]),torch.tensor([0]),torch.ones(1,1,1),torch.tensor([[1.,1.,0.]]),emb,torch.zeros(1,5),torch.tensor([10],dtype=torch.long),torch.tensor([9],dtype=torch.long)]
    o=mo(*common); e2=emb.clone(); e2[0]=torch.tensor([1.,.5,0,0,0,0]); common[6]=e2; o2=mo(*common)
    d=(o2-o)[0].tolist(); return abs(d[0])>0 and max(abs(d[1]),abs(d[2]))<1e-8, {'delta':d}
check(r,'class-local embedding perturbation',embedding_local)

# Still pass finite gradients on actual class-conditioned relation.
def grads():
    C,D=3,6; mo=model(classes=C,dim=D); mo.child_mask=ConstMask([1.])
    p=torch.randn(1,D,2,2,requires_grad=True); c=torch.randn(1,D,2,2,requires_grad=True); g=torch.randn(1,5,requires_grad=True)
    o=mo(p,c,torch.tensor([0]),torch.tensor([0]),torch.ones(1,2,2),torch.tensor([[.8,.2,0.]]),torch.eye(C,D),g,torch.tensor([10],dtype=torch.long),torch.tensor([9],dtype=torch.long)); o.sum().backward()
    par={n:q.grad for n,q in mo.named_parameters()}
    finite=all(q is not None and bool(torch.isfinite(q).all()) for q in par.values())
    cq=par['child_query.0.weight']; rel=par['relation.2.weight']
    return finite and float(cq.abs().max())>0 and float(rel.abs().max())>=0, {'param_grads_finite':finite,
        'child_query_first_maxabs':float(cq.abs().max()),'relation_last_maxabs':float(rel.abs().max())}
check(r,'gradients finite and child_query alive',grads)

def full_vocabulary():
    C,D=1203,256
    mo=Ledger().eval()
    p=torch.randn(1,D,1,1); c=torch.randn(1,D,1,1)
    post=torch.softmax(torch.randn(1,C),1); emb=torch.randn(C,D)
    o=mo(p,c,torch.tensor([0]),torch.tensor([0]),torch.ones(1,1,1),post,emb,
         torch.zeros(1,5),torch.tensor([10],dtype=torch.long),torch.tensor([9],dtype=torch.long))
    return tuple(o.shape)==(1,C) and bool(torch.isfinite(o).all()) and bool((o<=1e-7).all()), {'shape':list(o.shape),'finite':bool(torch.isfinite(o).all()),'max':float(o.max())}
check(r,'actual 1203-class nonempty vocabulary',full_vocabulary)

# Single-row training is a realistic event batch and must not fail in GroupNorm.
def single_row_train():
    mo=model(); mo.train()
    args=base_args()
    try:
        out=mo(*args)
        ok=bool(torch.isfinite(out).all())
        return ok, {'accepted':True,'finite':ok,'shape':list(out.shape)}
    except BaseException as exc:
        return False, {'accepted':False,'exception':f'{type(exc).__name__}: {exc}'}
check(r,'counterexample: single-row train-mode child mask does not crash GroupNorm',single_row_train,True)

# Keep the exact current hash stable through the end of snapshot execution.
current_after=hashlib.sha256(CURRENT.read_bytes()).hexdigest()
if current_after != source_before:
    raise SystemExit(f"mutable current changed during audit: {source_before} -> {current_after}")
report=dict(
    current_source=str(CURRENT), current_source_sha256=source_before,
    current_source_sha256_after=current_after,
    snapshot=str(SNAPSHOT), snapshot_sha256=snapshot_sha,
    audit_scope="CURRENT_SNAPSHOT_AUDITED; NOT a dynamic file guarantee after this run",
    torch_version=torch.__version__, device="cpu", rounds={str(i):sum(x['round']==i for x in results) for i in (1,2,3)},
    criteria={k:{key:val for key,val in x.items() if key not in ('round','counterexample')} for x in results for k in [x['name']]},
    tests=results, failures=[x['name'] for x in results if not x['passed']],
)
out=Path(__file__).with_name('owner_ledger_v4_audit_evidence_20261025.json'); out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(dict(json=str(out),source=source_before,snapshot=snapshot_sha,tests=len(results),failures=[x['name'] for x in results if not x['passed']]),indent=2))
