#!/usr/bin/env python3
"""Exact, dataset-free audit of the frozen candidate's conditioning contract.

This adds an audit beside the frozen specification; it does not modify it.
All probability/moment assertions use fractions.Fraction, not fitted models.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "full_mean_pair_gate_protocol_20261023.json"
PROTOCOL_SHA256 = "fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718"
OUT = HERE / "pair_conditioning_mismatch_synthetic_20261023.json"


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def wrong(p, y):
    return tuple(F(0) if c == y else v for c, v in enumerate(p))


def mean(rows, key, k):
    mass = sum((r["weight"] for r in rows), F(0))
    return tuple(sum((r["weight"] * r[key][c] for r in rows), F(0)) / mass for c in range(k))


def analyse(pairs, posterior, state_weight):
    k = len(pairs[0][0])
    atoms = []
    for s, ((pi, pj), ys, sw) in enumerate(zip(pairs, posterior, state_weight)):
        assert sum(pi) == sum(pj) == sum(ys) == 1
        for y, yp in enumerate(ys):
            if yp:
                wi, wj = wrong(pi, y), wrong(pj, y)
                atoms.append(dict(state=s, y=y, weight=sw * yp, wi=wi, wj=wj, g=dot(wi, wj)))
    moments = {}
    for y in range(k):
        rr = [r for r in atoms if r["y"] == y]
        if not rr:
            continue
        mass = sum((r["weight"] for r in rr), F(0))
        mui, muj = mean(rr, "wi", k), mean(rr, "wj", k)
        d = sum((r["weight"] * dot(tuple(a-b for a,b in zip(r["wi"], mui)),
                                             tuple(a-b for a,b in zip(r["wj"], muj))) for r in rr), F(0)) / mass
        h = sum((r["weight"] * r["g"] for r in rr), F(0)) / mass
        assert dot(mui, muj) + d == h
        moments[y] = dict(probability=mass, mu_i=mui, mu_j=muj, d=d, h=h)
    coarse = sum((r["weight"] * r["g"] for r in atoms), F(0))
    assert sum((v["probability"] * v["h"] for v in moments.values()), F(0)) == coarse
    truth, mixed, formula, matched_moments = [], [], [], []
    for s, ((pi,pj), ys) in enumerate(zip(pairs, posterior)):
        rr = [r for r in atoms if r["state"] == s]
        mass = sum((r["weight"] for r in rr), F(0))
        truth.append(sum((r["weight"] * r["g"] for r in rr), F(0)) / mass)
        mixed.append(sum((ys[y] * moments[y]["h"] for y in moments), F(0)))
        formula.append(dot(pi,pj) - sum((q*a*b for q,a,b in zip(ys,pi,pj)), F(0)))
        matched_moments.append(sum((ys[y] * dot(wrong(pi,y),wrong(pj,y)) for y in range(k) if ys[y]), F(0)))
    assert truth == formula == matched_moments
    mse_bias = sum((w * (a-b)**2 for w,a,b in zip(state_weight,mixed,truth)), F(0))
    risk_mixed = sum((r["weight"] * (r["g"] - mixed[r["state"]])**2 for r in atoms), F(0))
    risk_bayes = sum((r["weight"] * (r["g"] - truth[r["state"]])**2 for r in atoms), F(0))
    assert risk_mixed - risk_bayes == mse_bias
    return dict(moments=moments, true_conditional=truth, mixed_conditioning=mixed,
                exact_posterior_formula=formula, matched_conditioning_moments=matched_moments,
                coarse_mean=coarse, excess_MSE=mse_bias, mixed_MSE=risk_mixed, bayes_MSE=risk_bayes)


def serialize(x):
    if isinstance(x, F):
        return {"exact": str(x), "float": float(x)}
    if isinstance(x, dict):
        return {str(k): serialize(v) for k,v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serialize(v) for v in x]
    return x


def summary_features(p):
    """Every posterior-dependent scalar in frozen Q/A is equal here.

    All three branches and both endpoints are p; all other raw quality,
    appearance, lag, box, background and reference-count inputs are fixed.
    A uniform reference makes its probability/rank query identical as well.
    Any frozen quality-risk function therefore receives identical 14-vectors.
    """
    f = [float(v) for v in p]
    entropy = -sum(v * math.log(v) for v in f if v) / math.log(len(p))
    top = sorted(p, reverse=True)
    return dict(entropy=entropy, top1=top[0], margin=top[0]-top[1],
                text_image_JS=0, emitted_label_agreement=True,
                pair_dot=dot(p,p), pair_cosine=1, pair_JS=0)


def main():
    protected_before = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert protected_before == PROTOCOL_SHA256
    p0, p1 = (F(6,10),F(3,10),F(1,10)), (F(3,10),F(6,10),F(1,10))
    pairs = [(p0,p0),(p1,p1)]
    sw = [F(1,2),F(1,2)]
    assert summary_features(p0) == summary_features(p1)
    fixed = analyse(pairs, [(F(1),F(0),F(0))]*2, sw)
    assert fixed["true_conditional"] == [F(1,10),F(37,100)]
    assert fixed["mixed_conditioning"] == [F(47,200)]*2
    assert fixed["moments"][0]["mu_i"] == (F(0),F(45,100),F(1,10))
    assert fixed["moments"][0]["d"] == F(225,10000)
    assert fixed["excess_MSE"] == F(18225,1000000)
    unknown = analyse(pairs, [(F(8,10),F(2,10),F(0)),(F(4,10),F(6,10),F(0))], sw)
    assert unknown["true_conditional"] == [F(154,1000),F(208,1000)]
    assert unknown["mixed_conditioning"] == [F(1855,10000),F(1765,10000)]
    assert unknown["moments"][0]["h"] == F(19,100)
    assert unknown["moments"][1]["h"] == F(1675,10000)
    assert unknown["excess_MSE"] == F(99225,100000000)

    # Strictly positive 1203-dimensional posteriors, not a zero-padding shortcut.
    eps, k = F(1,100000000), 1203
    scale = 1 - (k-3)*eps
    pp = [tuple(v*scale for v in p) + (eps,)*(k-3) for p in (p0,p1)]
    assert all(v > 0 for p in pp for v in p)
    assert summary_features(pp[0]) == summary_features(pp[1])
    yy = (F(1),) + (F(0),)*(k-1)
    full = analyse([(p,p) for p in pp], [yy,yy], sw)
    assert full["excess_MSE"] == F(18225,1000000)*scale**4

    # Refine conditioning to the observed posterior state, recovering the exact
    # answer even though no residual coupling remains conditional on state,Y.
    # Swapping endpoints and jointly permuting classes preserve all identities.
    permutations_checked = 0
    for perm in [(0,1,2),(2,0,1),(1,2,0),(2,1,0),(1,0,2),(0,2,1)]:
        ppairs = [(tuple(a[c] for c in perm),tuple(b[c] for c in perm)) for a,b in pairs]
        yprob = [tuple(z[c] for c in perm) for z in [(F(8,10),F(2,10),F(0)),(F(4,10),F(6,10),F(0))]]
        z = analyse(ppairs,yprob,sw)
        assert z["true_conditional"] == unknown["true_conditional"]
        assert z["mixed_conditioning"] == unknown["mixed_conditioning"]
        zswap = analyse([(b,a) for a,b in ppairs],yprob,sw)
        assert zswap == z
        permutations_checked += 1

    # Nonidentical endpoints make exchange symmetry a substantive check.
    asymmetric_pairs = [
        ((F(5,10),F(4,10),F(1,10)),(F(2,10),F(3,10),F(5,10))),
        ((F(1,10),F(7,10),F(2,10)),(F(6,10),F(2,10),F(2,10))),
        ((F(4,10),F(2,10),F(4,10)),(F(3,10),F(5,10),F(2,10))),
    ]
    asymmetric_pi = [(F(2,10),F(3,10),F(5,10)),(F(6,10),F(1,10),F(3,10)),(F(1,10),F(7,10),F(2,10))]
    asym = analyse(asymmetric_pairs,asymmetric_pi,[F(1,3)]*3)
    for perm in [(0,1,2),(2,0,1),(1,2,0),(2,1,0),(1,0,2),(0,2,1)]:
        ap = [(tuple(a[c] for c in perm),tuple(b[c] for c in perm)) for a,b in asymmetric_pairs]
        ay = [tuple(v[c] for c in perm) for v in asymmetric_pi]
        z = analyse(ap,ay,[F(1,3)]*3)
        zswap = analyse([(b,a) for a,b in ap],ay,[F(1,3)]*3)
        for key in ("true_conditional","mixed_conditioning","exact_posterior_formula","matched_conditioning_moments","coarse_mean","excess_MSE","mixed_MSE","bayes_MSE"):
            assert z[key] == zswap[key] == asym[key]
        for y in z["moments"]:
            assert z["moments"][y]["mu_i"] == zswap["moments"][y]["mu_j"]
        permutations_checked += 1

    # Two distinct conditioning checks: coarse posterior recovers E[G|A];
    # full conditioning and oracle posterior recover E[G|p_i,p_j,A].
    coarse_integration = sum((v["probability"]*v["h"] for v in unknown["moments"].values()),F(0))
    assert coarse_integration == unknown["coarse_mean"] == F(181,1000)
    assert unknown["mixed_conditioning"] != unknown["true_conditional"]
    assert unknown["matched_conditioning_moments"] == unknown["exact_posterior_formula"] == unknown["true_conditional"]

    # A relation can usefully predict a COARSE conditional moment: identical
    # endpoint marginals need not imply identical joint moments. This does not
    # contradict exact posterior sufficiency for G after p_i,p_j are observed.
    lo,hi = (F(8,10),F(1,10),F(1,10)), (F(6,10),F(3,10),F(1,10))
    mu = (F(0),F(2,10),F(1,10))
    def centered(a,b):
        return dot(tuple(x-y for x,y in zip(wrong(a,0),mu)),
                   tuple(x-y for x,y in zip(wrong(b,0),mu)))
    independent = sum((centered(a,b) for a in (lo,hi) for b in (lo,hi)),F(0))/4
    coupled = (centered(lo,lo)+centered(hi,hi))/2
    assert independent == 0 and coupled == F(1,100)

    # The paper's noisy-OR bound needs positive association; it is not universal.
    q = F(4,10)
    exclusive_union, noisy_or, max_dependence_union = 2*q,1-(1-q)**2,q
    assert exclusive_union > noisy_or > max_dependence_union
    q1,q2 = F(2,10),F(6,10)
    assert max(q1,q2) == F(6,10)  # nested/comonotone events attain max marginals

    # Under the paper's EXACT SAME normalized room->scene weighting, constant
    # predictions satisfy Brier(c)=c²+(1-2c)*weighted prevalence. Published .80
    # for constant1 implies prevalence approximately .20 and constant0 .20;
    # these are arithmetic diagnostics, not a fitted or deployment baseline.
    prevalence = F(1,5)
    brier_zero = prevalence
    brier_point_two = F(1,25)+(1-F(2,5))*prevalence
    assert brier_zero == F(1,5) and brier_point_two == F(4,25)
    protected_after = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert protected_after == protected_before
    out = dict(
        status="PASS_EXACT_CONDITIONING_AUDIT_SYNTHETIC_ONLY",
        dataset_files_opened=False, test_or_heldout_correctness_opened=False,
        real_pair_training_started=False, GPU_used=False,
        frozen_protocol_sha256_before=protected_before,
        frozen_protocol_sha256_after=protected_after,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rounds=[
            dict(round=1, checks=["strict_same_A_fixed_Y_counterexample_exact_rational", "unknown_Y_counterexample_with_exact_mu_d_pi"]),
            dict(round=2, checks=["risk_projection_identity_and_strict_positive_K1203", "joint_class_permutation_and_nonidentical_endpoint_exchange"]),
            dict(round=3, checks=["coarse_pi_and_h_recover_coarse_mean_but_richer_pi_does_not_recover_full_conditional", "matched_conditioning_oracle_pi_reconstructs_G_and_coarse_relation_can_still_help"]),
        ],
        fixed_Y=dict(pairs=pairs, state_weights=sw, results=fixed, same_A_features=summary_features(p0)),
        unknown_Y=dict(pairs=pairs, posterior=[(F(8,10),F(2,10),F(0)),(F(4,10),F(6,10),F(0))], results=unknown),
        full1203=dict(K=k, tail_epsilon=eps, main_scale=scale, strictly_positive=True,
                      true_conditional=full["true_conditional"], mixed_conditioning=full["mixed_conditioning"], excess_MSE=full["excess_MSE"]),
        symmetries=dict(joint_class_permutations=permutations_checked, endpoint_swap=True, nonidentical_endpoints=True),
        conditioning_scope=dict(coarse_pi_h_integral=coarse_integration, coarse_target=unknown["coarse_mean"], matched_full_target=unknown["matched_conditioning_moments"]),
        relation_usefulness=dict(independent_coarse_trace=independent,coupled_coarse_trace=coupled),
        additional_literature_checks=["noisy_OR_upper_bound_not_universal", "constant_Brier_arithmetic_with_identical_macro_weighting"],
        noisy_or_qualification=dict(equal_marginal=q,mutually_exclusive_union=exclusive_union,noisy_or=noisy_or,max_dependence_union=max_dependence_union),
        paper_constant_Brier_diagnostic=dict(assumed_same_weight_prevalence=prevalence,constant0=brier_zero,constant_point2=brier_point_two,
            caveat="Published values are rounded; arithmetic diagnostic on identical macro weighting, not a measured or deployment-fitted comparator."),
        interpretation="Perfect coarse moments and perfect richer label posterior do not repair mixed conditioning. Finite C129 gains cannot imply universal relation-network impossibility or intrinsic covariance."
    )
    OUT.write_text(json.dumps(serialize(out),ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status=out["status"],receipt=str(OUT),fixed_excess_MSE=float(fixed["excess_MSE"]),unknown_excess_MSE=float(unknown["excess_MSE"]),protocol_sha256=protected_after)))


if __name__ == "__main__":
    main()
