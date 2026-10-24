"""Explicit-protocol, waited CPU benchmark of four metadata-selected frames.

No real benchmark is run unless --execute is explicitly supplied with the exact
reviewed protocol SHA. The worker reads only allowlisted raw/box/proposal and
frame-locator members. It produces complete candidate arrays, no evaluation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def authenticate(protocol, expected_sha):
    require(sha(protocol)==expected_sha, 'Exact externally supplied protocol SHA required')
    spec=read(protocol)
    require(spec['status']=='FROZEN_METADATA_FOUR_FRAME_CPU_ALL_CANDIDATE_BENCHMARK_V1', 'Protocol not frozen')
    require(sys.executable==spec['runtime']['python'], 'Wrong Python runtime')
    require(sha(__file__)==spec['runner_sha256'], 'Benchmark runner changed')
    for path,digest in spec['input_pins'].items():
        require(sha(path)==digest, 'Input changed: '+path)
    protected=read(spec['protected_authority_path'])['protected_files']
    require(len(protected)==7, 'Protected scope')
    for path,record in protected.items():
        require(dict(exists=Path(path).exists(),sha256=sha(path) if Path(path).exists() else None)==record, 'Protected changed: '+path)
    require(spec['runtime']['CPU_only'] is True and spec['runtime']['hard_timeout_seconds']==300, 'CPU/timeout scope')
    require(spec['allowed_cache_members']==['det_bboxes','track_feats','roi_proposal_row','video_id','image_id','frame_id','frame_index'], 'Raw member allowlist')
    require(len(spec['frames'])==4 and len(spec['pairs'])==3, 'Four frames / three adjacent pairs')
    require([r['frame_id'] for r in spec['frames']]==[0,1,2,3], 'Metadata prefix0..3')
    require(len({r['video_id'] for r in spec['frames']})==1, 'Single selected video')
    require(spec['pairs']==[dict(query_frame=i,reference_frame=i-1) for i in (1,2,3)], 'Fixed past-adjacent pairs')
    for trial in spec['synthetic_actual_exit_receipts']:
        # The frozen protocol stores a structured pair of receipt paths, not a
        # path string.  Validate both layers before allowing a real worker.
        require(sha(trial['receipt'])==trial['receipt_sha256'], 'Synthetic QA receipt hash')
        require(sha(trial['actual_exit_receipt'])==trial['actual_exit_receipt_sha256'], 'Synthetic actual-exit receipt hash')
        result=read(trial['actual_exit_receipt'])
        require(result['actual_exit_code']==0 and result['timed_out'] is False
                and result['receipt']==trial['receipt'], 'Synthetic actual waited exit/receipt')
        qa=read(trial['receipt'])
        require(qa['subject_sha256']==sha(spec['module_path']) and len(qa['checks'])>=2, 'Synthetic subject/coverage')
    return spec


def load_frame(record, allowed):
    require(sha(record['npz'])==record['npz_sha256'], 'Raw cache SHA')
    with np.load(record['npz'], allow_pickle=False) as saved:
        # Explicit indexing only: never decode category, posterior, released ID,
        # instance labels, GT, or any nonallowlisted member.
        for field in ('video_id','image_id','frame_id','frame_index'):
            require(field in allowed and int(saved[field])==record[field], 'Cache locator '+field)
        boxes=np.ascontiguousarray(saved['det_bboxes'][:,:4], dtype=np.float64)
        raw=np.ascontiguousarray(saved['track_feats'], dtype=np.float64)
        proposals=saved['roi_proposal_row'].copy()
    n=record['observations']
    require(boxes.shape==(n,4) and raw.shape==(n,256) and proposals.shape==(n,), 'Frozen retained cache shape')
    return boxes,raw,proposals


def worker(protocol, expected_sha):
    start=time.monotonic();spec=authenticate(protocol,expected_sha)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'GPU visibility must be disabled')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        require(os.environ.get(name)=='1', 'Single CPU thread required')
    out=Path(spec['output_directory']);out.mkdir()
    write(out/'started.json',dict(status='RUNNING_CPU_FOUR_FRAME_ALL_CANDIDATE_BENCHMARK',pid=os.getpid(),
        protocol_sha256=expected_sha,CPU_only=True,training=False,identity_or_new_results_opened=False))
    loader=importlib.util.spec_from_file_location('benchmark_all_candidate_module',spec['module_path'])
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    frames=[load_frame(r,spec['allowed_cache_members']) for r in spec['frames']]
    pair_receipts=[]
    for pair in spec['pairs']:
        qi,ri=pair['query_frame'],pair['reference_frame']
        qb,qf,qp=frames[qi];rb,rf,rp=frames[ri]
        begun=time.monotonic()
        result=module.all_candidate_scene_residual(qb,rb,qf,rf,qp,rp,minimum_anchors=3,memoize=True)
        compute_elapsed=time.monotonic()-begun
        require(result['candidate_count']==len(qb)*len(rb) and result['cosine_products_built']==1, 'Allcandidate/single matrix')
        require(result['fullpool_h_used_to_filter_candidates'] is False, 'Candidate filtering forbidden')
        filename='q%d_r%d.npz'%(qi,ri);arraypath=out/filename
        with arraypath.open('xb') as stream:
            np.savez_compressed(stream,**result['arrays'])
        fitpath=out/('q%d_r%d_fits.json'%(qi,ri))
        write(fitpath,dict(fit_records=result['fit_records'],query_exclusions=result['query_exclusions'],reference_exclusions=result['reference_exclusions']))
        value=dict(query_frame=qi,reference_frame=ri,
            query_image_id=spec['frames'][qi]['image_id'],reference_image_id=spec['frames'][ri]['image_id'],
            shape=list(result['candidate_shape']),total_candidates=result['candidate_count'],
            available_candidates=int(result['arrays']['available'].sum()),
            donor_fit_count=result['donor_fits_computed'],cosine_products_built=result['cosine_products_built'],
            compute_elapsed_seconds=compute_elapsed,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            output_array_nbytes={k:int(v.nbytes) for k,v in result['arrays'].items()},
            output_arrays_nbytes_total=sum(int(v.nbytes) for v in result['arrays'].values()),
            npz=filename,npz_sha256=sha(arraypath),npz_storage_bytes=arraypath.stat().st_size,
            fit_records_file=fitpath.name,fit_records_sha256=sha(fitpath),fit_records_storage_bytes=fitpath.stat().st_size,
            raw_cosine_source='copy of canonical full CachedSceneMotion matrix; no second cosine product',
            no_identity_tracking_or_novelty_evaluation=True)
        pair_receipts.append(value)
        print(json.dumps(value),flush=True)
    authenticate(protocol,expected_sha)
    write(out/'benchmark_receipt.json',dict(status='COMPLETE_FIXED_METADATA_FOUR_FRAME_CPU_BENCHMARK',
        protocol_sha256=expected_sha,module_sha256=sha(spec['module_path']),frames=spec['frames'],pairs=pair_receipts,
        elapsed_seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        all_input_pins_and_protected7_unchanged=True,CPU_only=True,
        full_FIT_extrapolation_or_method_effectiveness_or_novelty_established=False))


def execute(protocol, expected_sha):
    spec=authenticate(protocol,expected_sha)
    out,execution=Path(spec['output_directory']),Path(spec['execution_directory'])
    require(not out.exists() and not execution.exists(), 'No overwrite, retry or partial-run reuse')
    execution.mkdir()
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--protocol',str(protocol.resolve()),'--protocol-sha256',expected_sha]
    write(execution/'launch.json',dict(command=command,protocol_sha256=expected_sha,timeout_seconds=300,CPU_only=True,automatic_retry=False))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    begin=time.monotonic();timed_out=False
    with (execution/'stdout.log').open('x') as stdout,(execution/'stderr.log').open('x') as stderr:
        child=subprocess.Popen(command,cwd=str(Path(__file__).resolve().parents[1]),env=env,
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True)
        try:
            actual_exit=child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();actual_exit=child.wait()
    value=dict(actual_exit_code=actual_exit,timed_out=timed_out,elapsed_seconds=time.monotonic()-begin,
               child_pid=child.pid,waited_by_parent_pid=os.getpid(),protocol_sha256=expected_sha,
               launch_sha256=sha(execution/'launch.json'))
    write(execution/'actual_exit.json',value)
    require(actual_exit==0 and not timed_out, 'Benchmark worker failed or timed out; no retry')
    authenticate(protocol,expected_sha)
    receipt=read(out/'benchmark_receipt.json')
    require(receipt['status']=='COMPLETE_FIXED_METADATA_FOUR_FRAME_CPU_BENCHMARK' and receipt['protocol_sha256']==expected_sha, 'Incomplete benchmark')
    write(execution/'completion_receipt.json',dict(status='PASS_WAITED_CPU_FOUR_FRAME_ALL_CANDIDATE_BENCHMARK',
        protocol_sha256=expected_sha,actual_exit_sha256=sha(execution/'actual_exit.json'),
        benchmark_receipt_sha256=sha(out/'benchmark_receipt.json'),all_input_pins_and_protected7_unchanged=True,
        method_effectiveness_or_novelty_established=False))
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--execute',action='store_true');action.add_argument('--worker',action='store_true')
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--protocol-sha256',required=True)
    arguments=parser.parse_args()
    (execute if arguments.execute else worker)(arguments.protocol,arguments.protocol_sha256)
