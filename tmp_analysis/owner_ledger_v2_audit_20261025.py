#!/usr/bin/env python3
import json, math, sys, traceback
from pathlib import Path
import torch
from torch import nn
import importlib.util
from pathlib import Path
SOURCE=Path(__file__).resolve().with_name('evidence_ownership_head.py')
EXPECTED_SHA256='c4684e660842e00299ac6943e8af1419575a27753b12c1f96b8e4b2ad7c69c69'
import hashlib
actual=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f'audited source hash mismatch: expected {EXPECTED_SHA256}, got {actual}')
spec=importlib.util.spec_from_file_location('evidence_ownership_head',SOURCE)
ledger=importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger)
EvidenceOwnershipLedger=ledger.EvidenceOwnershipLedger

torch.set_num_threads(1)
torch.manual_seed(20261025)
RESULTS=[]

def record(round_no, name, passed, detail, exception=None):
    RESULTS.append(dict(round=round_no, name=name, passed=bool(passed), detail=detail,
                        exception=exception))

def outcome(fn):
    try:
        value=fn()
        return True, value, None
    except BaseException as e:
        return False, None, f"{type(e).__name__}: {e}"

def raises(fn, kinds=(ValueError, IndexError, TypeError, RuntimeError, AssertionError)):
    try:
        fn()
    except kinds as e:
        return True, f"{type(e).__name__}: {e}"
    except BaseException as e:
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, "accepted without exception"

class ConstantOccupancy(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32))
    def forward(self, x):
        assert len(x) == len(self.values)
        return self.values.to(x).view(-1,1,1,1).expand(-1,1,x.shape[-2],x.shape[-1])

def base(dim=6, classes=3, h=2, w=2, n_parent=1, n_child=1, n_edge=1,
         edge_parent=None, edge_child=None, child_vals=None, class_width=None):
    if class_width is None: class_width=classes
    m=EvidenceOwnershipLedger(dim=dim, hidden=16, max_children=8,
                              class_width=class_width).eval()
    if child_vals is not None: m.child_mask=ConstantOccupancy(child_vals)
    parent=torch.randn(n_parent,dim,h,w)
    child=torch.randn(n_edge,dim,h,w)
    if edge_parent is None: edge_parent=torch.zeros(n_edge,dtype=torch.long)
    if edge_child is None: edge_child=torch.zeros(n_edge,dtype=torch.long)
    overlap=torch.ones(n_edge,h,w)
    posterior=torch.full((n_child,classes),1/max(classes,1))
    embedding=torch.eye(classes,dim) if dim>=classes else torch.randn(classes,dim)
    geometry=torch.zeros(n_edge,5)
    return m,parent,child,edge_parent,edge_child,overlap,posterior,embedding,geometry

def call(t):
    m,p,c,ep,ec,ov,post,emb,geo=t
    return m(p,c,ep,ec,ov,post,emb,geo)

# Round 1: boundaries, empty inputs, parent limit, full vocabulary.
r=1
m,p,c,ep,ec,ov,post,emb,geo=base(edge_parent=torch.empty(0,dtype=torch.long),
    edge_child=torch.empty(0,dtype=torch.long))
ov=torch.empty(0,2,2)
geo=torch.empty(0,5)
ok,val,err=outcome(lambda: call((m,p,c,ep,ec,ov,post,emb,geo)))
record(r,'empty edge exact identity',ok and torch.equal(val,torch.zeros_like(val)),
       {'shape':list(val.shape) if ok else None,'exact_zero':bool(torch.equal(val,torch.zeros_like(val))) if ok else False},err)

# A malformed empty geometry passes despite child_query requiring five values.
m,p,c,ep,ec,ov,post,emb,geo=base(edge_parent=torch.empty(0,dtype=torch.long),
    edge_child=torch.empty(0,dtype=torch.long))
ov=torch.empty(0,2,2)
geo=torch.empty(0,4)
ok,val,err=outcome(lambda: call((m,p,c,ep,ec,ov,post,emb,geo)))
record(r,'malformed empty geometry rejected',not ok and err is not None,
       {'accepted':ok,'shape':list(val.shape) if ok else None},None if not ok else 'accepted and returned identity')

# Empty child token is rejected when edge is non-empty.
m,p,c,ep,ec,ov,post,emb,geo=base()
ok,val,err=outcome(lambda: m(p,torch.empty(0,6,2,2),ep,ec,ov,post,emb,geo))
record(r,'nonempty edge with empty child token rejected',not ok,
       {'accepted':ok},err)

# Non-finite inputs, one probe per input and both NaN/Inf.
base_args=base()
finite_cases=[]
for idx,name in [(1,'parent'),(2,'child'),(5,'overlap'),(6,'posterior'),(7,'embedding'),(8,'geometry')]:
    for badname,bad in [('nan',float('nan')),('inf',float('inf'))]:
        args=list(base_args); x=args[idx].clone(); x[(0,)*x.ndim]=bad; args[idx]=x
        ok,val,err=outcome(lambda a=tuple(args): call(a))
        finite_cases.append({'input':name,'value':badname,'rejected':not ok,'exception':err})
record(r,'all six inputs reject NaN and Inf',all(x['rejected'] for x in finite_cases),finite_cases)

# Out-of-range indices and non-int indices.
range_cases=[]
for field,values in [('parent',[-1,1,99]),('child',[-1,1,99])]:
    for value in values:
        m,p,c,ep,ec,ov,post,emb,geo=base()
        if field=='parent': ep=torch.tensor([value],dtype=torch.long)
        else: ec=torch.tensor([value],dtype=torch.long)
        ok,val,err=outcome(lambda a=(m,p,c,ep,ec,ov,post,emb,geo): call(a))
        range_cases.append({'field':field,'index':value,'rejected':not ok,'exception':err})
for field in ['parent','child']:
    m,p,c,ep,ec,ov,post,emb,geo=base()
    ep=torch.tensor([0.0]); ec=torch.tensor([0],dtype=torch.long)
    ok,val,err=outcome(lambda: m(p,c,ep,ec,ov,post,emb,geo))
    if field=='child': ep=torch.tensor([0],dtype=torch.long); ec=torch.tensor([0.0])
    ok,val,err=outcome(lambda a=(m,p,c,ep,ec,ov,post,emb,geo): call(a))
    range_cases.append({'field':field,'index':'float','rejected':not ok,'exception':err})
record(r,'all index bounds and dtype probes fail closed',all(x['rejected'] for x in range_cases),range_cases)

# Per-parent, not global, child limit.
limit_cases=[]
m=EvidenceOwnershipLedger(dim=6,hidden=16,max_children=1,class_width=3).eval(); m.child_mask=ConstantOccupancy([1.,1.])
p=torch.randn(2,6,2,2); c=torch.randn(2,6,2,2); post=torch.full((2,3),1/3); emb=torch.eye(3,6); geo=torch.zeros(2,5); ov=torch.ones(2,2,2)
# Two parents, one edge each: must pass.
ok,val,err=outcome(lambda: m(p,c,torch.tensor([0,1]),torch.tensor([0,1]),ov,post,emb,geo))
limit_cases.append({'case':'one edge per parent','accepted':ok,'exception':err})
# One parent, two edges: must fail.
ok2,val2,err2=outcome(lambda: m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo))
limit_cases.append({'case':'two edges on one parent','rejected':not ok2,'exception':err2})
record(r,'max_children is enforced per parent',ok and ok2 is False,limit_cases)

# Full 1203-class default vocabulary, actual tensors.
m=EvidenceOwnershipLedger().eval()
p=torch.randn(2,256,2,2); c=torch.randn(1,256,2,2); post=torch.zeros(1,1203); post[0,0]=1
emb=torch.randn(1203,256); geo=torch.zeros(1,5); ov=torch.ones(1,2,2)
ok,val,err=outcome(lambda: m(p,c,torch.tensor([0]),torch.tensor([0]),ov,post,emb,geo))
record(r,'default 1203-class full vocabulary',ok and tuple(val.shape)==(2,1203) and bool(torch.isfinite(val).all()),
       {'class_width':m.class_width,'relation_in_features':m.relation[0].in_features,'output_shape':list(val.shape) if ok else None,
        'finite':bool(torch.isfinite(val).all()) if ok else False,'all_nonpositive':bool((val<=0).all()) if ok else False},err)

# Round 2: actual owner selection, overlap, wrong child, future/causality interface.
r=2
# [0,1,0] parent sequence with scores [1,9,5]. Parent0 has scores 1 and 5; parent1 has 9.
dim,classes=6,3
m=EvidenceOwnershipLedger(dim=dim,hidden=16,max_children=8,class_width=classes).eval(); m.child_mask=ConstantOccupancy([1.,9.,5.])
p=torch.zeros(2,dim,1,1); p[0,:,0,0]=(torch.eye(3,6)[0]+torch.eye(3,6)[2])/math.sqrt(2); p[1,:,0,0]=torch.eye(3,6)[1]
c=torch.randn(3,dim,1,1); ep=torch.tensor([0,1,0]); ec=torch.tensor([0,1,2]); ov=torch.ones(3,1,1)
post=torch.eye(3); emb=torch.eye(3,6); geo=torch.zeros(3,5)
out=m(p,c,ep,ec,ov,post,emb,geo)
# Parent0 must be owned by edge 2 (class2), not edge0 (class0); parent1 by edge1.
ok=out[0,2] < -1e-8 and abs(float(out[0,0])) < 1e-10 and out[1,1] < -1e-8 and abs(float(out[1,2])) < 1e-10
record(r,'[0,1,0] scores [1,9,5] owner is local maximum',ok,
       {'values':out.tolist(),'expected_parent0_class':'2','expected_parent1_class':'1'})

# Highest score with two overlapping wrong/right children.
m=EvidenceOwnershipLedger(dim=dim,hidden=16,max_children=8,class_width=classes).eval(); m.child_mask=ConstantOccupancy([1.,5.])
p=torch.zeros(1,dim,1,1); p[0,:,0,0]=(torch.eye(3,6)[0]+torch.eye(3,6)[2])/math.sqrt(2)
c=torch.randn(2,dim,1,1); post=torch.eye(3)[[0,2]]; ov=torch.ones(2,1,1); geo=torch.zeros(2,5)
out_hi=m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo)
ok_hi=(out_hi[0,2] < -1e-8 and abs(float(out_hi[0,0])) < 1e-10)
m.child_mask=ConstantOccupancy([5.,1.])
out_lo=m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo)
ok_lo=(out_lo[0,0] < -1e-8 and abs(float(out_lo[0,2])) < 1e-10)
record(r,'higher-score child owns and wrong child does not explain',ok_hi and ok_lo,
       {'child1_wins':out_hi.tolist(),'child0_wins':out_lo.tolist()})

# Strict tie chooses lower edge, with DIFFERENT child payloads.
m=EvidenceOwnershipLedger(dim=dim,hidden=16,max_children=8,class_width=classes).eval(); m.child_mask=ConstantOccupancy([1.,1.])
p=torch.zeros(1,dim,1,1); p[0,:,0,0]=(torch.eye(3,6)[0]+torch.eye(3,6)[2])/math.sqrt(2)
c=torch.randn(2,dim,1,1); post=torch.eye(3)[[0,2]]; ov=torch.ones(2,1,1); geo=torch.zeros(2,5)
tie=m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo)
# Reverse payloads: edge0 is now class2 and must own.
tie_rev=m(p,c,torch.tensor([0,0]),torch.tensor([1,0]),ov,post,emb,geo)
ok=(tie[0,0] < -1e-8 and abs(float(tie[0,2])) < 1e-10 and tie_rev[0,2] < -1e-8 and abs(float(tie_rev[0,0])) < 1e-10)
record(r,'fully overlapping strict tie gives lower edge',ok,{'forward':tie.tolist(),'payload_reversed':tie_rev.tolist()})

# Partial overlap: two candidates both claim the left cell; winner score controls class.
m=EvidenceOwnershipLedger(dim=dim,hidden=16,max_children=8,class_width=classes).eval(); m.child_mask=ConstantOccupancy([1.,2.])
p=torch.zeros(1,dim,1,2); p[0,:,0,0]=(torch.eye(3,6)[0]+torch.eye(3,6)[2])/math.sqrt(2)
c=torch.randn(2,dim,1,2); post=torch.eye(3)[[0,2]]; geo=torch.zeros(2,5)
ov=torch.zeros(2,1,2); ov[:,:,0]=1
partial=m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo)
m.child_mask=ConstantOccupancy([2.,1.])
partial_swap=m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo)
ok=(partial[0,2] < -1e-8 and abs(float(partial[0,0])) < 1e-10 and partial_swap[0,0] < -1e-8 and abs(float(partial_swap[0,2])) < 1e-10)
record(r,'partial overlap has one owner and respects score',ok,
       {'child1_wins':partial.tolist(),'child0_wins':partial_swap.tolist()})

# Zero overlap/illegal scores yield exact zero.
m=EvidenceOwnershipLedger(dim=dim,hidden=16,max_children=8,class_width=classes).eval(); m.child_mask=ConstantOccupancy([1.,9.])
p=torch.randn(1,dim,1,2); c=torch.randn(2,dim,1,2); post=torch.full((2,3),1/3); geo=torch.zeros(2,5); ov=torch.zeros(2,1,2)
zero=m(p,c,torch.tensor([0,0]),torch.tensor([0,1]),ov,post,emb,geo)
record(r,'zero legal overlap gives exact no-ownership identity',torch.equal(zero,torch.zeros_like(zero)),{'exact_zero':bool(torch.equal(zero,torch.zeros_like(zero)))})

# Future-information surface: no time/state argument, and extra future kwarg must fail.
import inspect
sig=inspect.signature(EvidenceOwnershipLedger.forward)
ok_sig=all(k not in sig.parameters for k in ('time','timestamp','future_tokens','state','memory','cache'))
m,p,c,ep,ec,ov,post,emb,geo=base()
ok_future,msg=raises(lambda: m(p,c,ep,ec,ov,post,emb,geo,future_tokens=torch.randn(1,6,2,2)), kinds=(TypeError,))
# Non-tied edge permutation with edge payloads restored must be equivalent.
m,_,_,_,_,_,post,emb,geo=base(n_child=3,n_edge=3,edge_parent=torch.tensor([0,0,0]),edge_child=torch.tensor([0,1,2]),
                              child_vals=[1.,2.,3.],class_width=3)
p=torch.randn(1,6,2,2); c=torch.randn(3,6,2,2); ov=torch.ones(3,2,2)
o1=m(p,c,torch.tensor([0,0,0]),torch.tensor([0,1,2]),ov,post,emb,geo)
m.child_mask=ConstantOccupancy([3.,1.,2.])
o2=m(p,c[[2,0,1]],torch.tensor([0,0,0]),torch.tensor([2,0,1]),ov[[2,0,1]],post,emb,geo[[2,0,1]])
perm_ok=torch.allclose(o1,o2,atol=1e-6)
record(r,'future inputs are absent from interface',False,
       {'signature':str(sig),'extra_future_kwarg_result':msg,'note':'structural only; caller must still supply past/current observations; no frame/time index exists, so a future child cannot be rejected by this module'})
record(r,'permuted non-tied edges are invariant',perm_ok,{'max_abs_diff':float((o1-o2).abs().max())})

# Round 3: explain-away direction, detach, spatial support, class conditioning, gradients.
r=3
# Direction: increase same-class child support -> magnitude of deduction increases.
m=EvidenceOwnershipLedger(dim=6,hidden=16,max_children=8,class_width=3).eval(); m.child_mask=ConstantOccupancy([1.])
p=torch.zeros(1,6,1,1); p[0,:,0,0]=torch.eye(3,6)[0]
c=torch.randn(1,6,1,1); emb=torch.eye(3,6); geo=torch.zeros(1,5); ov=torch.ones(1,1,1)
vals=[]
for q in [0.,0.25,0.5,1.]:
    post=torch.zeros(1,3); post[0,0]=q
    vals.append(float(m(p,c,torch.tensor([0]),torch.tensor([0]),ov,post,emb,geo)[0,0]))
ok=vals[0]==0 and vals[1] < -1e-12 and vals[2] < vals[1] and vals[3] < vals[2]
record(r,'child support for parent class increases explain-away',ok,{'q0_sweep':[0,.25,.5,1],'correction_class0':vals})

# Posterior detached.
m,_,_,_,_,_,post,emb,geo=base(); post=post.clone().requires_grad_(True)
p=torch.randn(1,6,2,2); c=torch.randn(1,6,2,2); ep=torch.tensor([0]); ec=torch.tensor([0]); ov=torch.ones(1,2,2)
out=m(p,c,ep,ec,ov,post,emb,geo); (-out).sum().backward()
record(r,'child posterior receives no gradient',post.grad is None,{'posterior_grad':None if post.grad is None else 'non-None'})
# Numeric class posterior ignores gradient but values still used.
post2=torch.zeros(1,3); post2[0,0]=1
out_a=m(p,c,ep,ec,ov,post2.detach(),emb,geo)
post2b=torch.zeros(1,3); post2b[0,1]=1
out_b=m(p,c,ep,ec,ov,post2b,emb,geo)
record(r,'posterior values still control class axis',not torch.allclose(out_a,out_b),{'class0_support':out_a.tolist(),'class1_support':out_b.tolist()})

# Parent positive support spatial localization; negative and zero evidence must not be subtracted.
m=EvidenceOwnershipLedger(dim=6,hidden=16,max_children=8,class_width=3).eval(); m.child_mask=ConstantOccupancy([1.])
p=torch.zeros(1,6,1,3); p[0,:,0,1]=torch.eye(3,6)[0]; p[0,:,0,2]=-torch.eye(3,6)[0]
c=torch.randn(1,6,1,3); post=torch.zeros(1,3); post[0,0]=1; emb=torch.eye(3,6); geo=torch.zeros(1,5); ov=torch.ones(1,1,3)
loc=m(p,c,torch.tensor([0]),torch.tensor([0]),ov,post,emb,geo)
pzero=torch.zeros_like(p); zero=m(pzero,c,torch.tensor([0]),torch.tensor([0]),ov,post,emb,geo)
record(r,'parent positive support is localized and zero is exact identity',
       loc[0,0] < -1e-8 and abs(float(loc[0,1])) < 1e-12 and abs(float(loc[0,2])) < 1e-12 and torch.equal(zero,torch.zeros_like(zero)),
       {'one_positive_cell':loc.tolist(),'all_zero_parent_exact':bool(torch.equal(zero,torch.zeros_like(zero)))})

# Class conditioning diagnostics: actual relation is one scalar per spatial cell.
m=EvidenceOwnershipLedger(dim=6,hidden=16,max_children=8,class_width=3).eval(); m.child_mask=ConstantOccupancy([1.])
p=torch.zeros(1,6,2,2); p[0]=(torch.eye(3,6)[0]+torch.eye(3,6)[1]+torch.eye(3,6)[2]).reshape(6,1,1).expand(6,2,2)/math.sqrt(3)
c=torch.randn(1,6,2,2); post=torch.eye(3)[[0]]; emb=torch.eye(3,6); geo=torch.zeros(1,5); ov=torch.ones(1,2,2)
# Directly inspect all relation outputs shape; the forward defines exactly one score/cell.
with torch.no_grad():
    child_relation=m.child_query(torch.cat([c[0].mean((-2,-1)).unsqueeze(0),geo[0].unsqueeze(0)],dim=1))
    parent_relation=torch.einsum('dhw,cd->chw',torch.nn.functional.normalize(p[0],dim=0),torch.nn.functional.normalize(emb,dim=1))
    pg=torch.zeros(3,2,2); pg[:3]=parent_relation
    rel=m.relation(torch.cat([pg.permute(1,2,0).reshape(-1,3),child_relation[0].expand(4,6)],dim=1)).reshape(2,2)
class_conditioned = rel.ndim==2 and list(rel.shape)==[2,2]
# A shared spatial gate multiplied by equal q and equal parent evidence is exactly class-constant.
peq=torch.randn(1,6,2,2); eq_emb=torch.randn(3,6); eq_emb[:]=eq_emb[0]
postu=torch.full((1,3),1/3)
eqout=m(peq,c,torch.tensor([0]),torch.tensor([0]),ov,postu,eq_emb,geo)
# If normalized embeddings are identical, all three class projections are identical; output should be equal.
shared_scalar=bool(torch.allclose(eqout[0,0],eqout[0,1],atol=1e-7) and torch.allclose(eqout[0,0],eqout[0,2],atol=1e-7))
record(r,'class conditioning is not a shared spatial scalar',False,
       {'relation_shape':list(rel.shape),'relation_is_one_scalar_per_cell':class_conditioned,
        'uniform_support_plus_identical_embeddings_yields_shared_scalar':shared_scalar,
        'finding':'active path is q_child[c] * one class-agnostic spatial relation score * parent_positive[c]; child_query does not consume posterior and has no per-class/cross-class parameters'})

# Gradients: finite and non-dead child_query.
m=EvidenceOwnershipLedger(dim=6,hidden=16,max_children=8,class_width=3).eval(); m.child_mask=ConstantOccupancy([1.])
p=torch.randn(1,6,2,2,requires_grad=True); c=torch.randn(1,6,2,2,requires_grad=True)
post=torch.zeros(1,3); post[0,0]=0.8; emb=torch.eye(3,6); geo=torch.zeros(1,5,requires_grad=True); ov=torch.ones(1,2,2)
out=m(p,c,torch.tensor([0]),torch.tensor([0]),ov,post,emb,geo); (-out).sum().backward()
grads={name:param.grad for name,param in m.named_parameters()}
finite=all(g is not None and torch.isfinite(g).all() for g in grads.values())
nonzero={name:float(g.abs().max()) for name,g in grads.items() if g is not None}
cq1=nonzero.get('child_query.0.weight',0.0); cq2=nonzero.get('child_query.2.weight',0.0)
record(r,'gradients finite and child_query not dead',finite and cq1>0 and cq2>0,
       {'all_parameter_grads_finite':finite,'child_query_first_maxabs':cq1,'child_query_last_maxabs':cq2,
        'input_parent_grad_finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),
        'input_child_grad_finite':c.grad is not None and bool(torch.isfinite(c.grad).all()),
        'geometry_grad_finite':geo.grad is not None and bool(torch.isfinite(geo.grad).all())})

# Extra counterexample: overlap values are silently clamped, not rejected.
m,p,c,ep,ec,ov,post,emb,geo=base(); ov=torch.tensor([[[-1.,2.],[float('nan'),0.5]]])
ok,val,err=outcome(lambda: m(p,c,ep,ec,ov,post,emb,geo))
record(r,'numeric overlap outside [0,1] rejected',not ok and err is not None,
       {'accepted_without_nan':ok and err is None,'notes':'NaN is rejected; -1 and 2 are clamped by implementation'},err if not ok else 'clamped and accepted')

# Save.
report={'source_snapshot':str(SOURCE),'source_sha256':actual,
        'torch_version':torch.__version__,
        'rounds':{str(i):sum(1 for x in RESULTS if x['round']==i) for i in (1,2,3)},
        'tests':RESULTS}
outpath=Path(__file__).with_name('owner_ledger_v2_audit_evidence_20261025.json')
outpath.write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps({'json':str(outpath),'count':len(RESULTS),
                  'passes':sum(x['passed'] for x in RESULTS),
                  'fails':sum(not x['passed'] for x in RESULTS)},indent=2))
for x in RESULTS:
    print(('PASS' if x['passed'] else 'FAIL'), x['round'], x['name'])
