#!/usr/bin/env python3
"""Synthetic exact target-membership test for canonical input memory layout."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert sha(args.module)==args.sha256
    spec=importlib.util.spec_from_file_location('cached_layout_subject',str(args.module))
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
    checked=0
    for seed in range(30):
        rng=np.random.RandomState(seed)
        q=rng.normal(size=(77,256));r=rng.normal(size=(77,256))
        r[0]=q[0];r[1]=q[0]+1e-8*rng.normal(size=256)
        boxes=np.tile([0.,0.,10.,10.],(77,1))
        baseline=subject.CachedSceneMotion(boxes,boxes,q,r)
        expected=baseline.fullpool_mnn()
        for qalternate,ralternate in ((np.asfortranarray(q),np.asfortranarray(r)),
                (q[:,::-1].copy()[:,::-1],r[:,::-1].copy()[:,::-1])):
            alternative=subject.CachedSceneMotion(np.asfortranarray(boxes),boxes,qalternate,ralternate)
            assert alternative.fullpool_mnn()==expected
            assert np.array_equal(alternative._similarity,baseline._similarity)
            checked+=1
    assert sha(args.module)==args.sha256
    result=dict(status='PASS_INDEPENDENT_CACHED_CANONICAL_LAYOUT_AND_NEARTIE_MEMBERSHIP',
        module_sha256=args.sha256,auditor_sha256=sha(__file__),
        near_tie_frame_pairs=30,exact_layout_comparisons=checked,
        discrete_tie_or_top1_tolerance_used=False,real_cache_opened=False,new_OOF_outputs_opened=False)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
