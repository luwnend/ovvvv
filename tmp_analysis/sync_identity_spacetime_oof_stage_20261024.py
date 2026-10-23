#!/usr/bin/env python3
"""Copy an explicit, lightweight OOF checkpoint to the managed Git repository."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
    'sync_identity_spacetime_oof_stage_20261024.py',
    'identity_spacetime_followup_recommendation_20261024.md',
    'identity_spacetime_modules_20261024.py',
    'identity_spacetime_modules_independent_audit_20261024.json',
    'identity_spacetime_source_folds_20261024.json',
    'identity_spacetime_oof_metadata_preparation_20261024.json',
    'identity_spacetime_oof_protocol_20261024.json',
    'identity_spacetime_oof_data_20261024.py',
    'train_identity_spacetime_oof_model_20261024.py',
    'evaluate_identity_spacetime_oof_20261024.py',
    'run_identity_spacetime_oof_20261024.py',
    'finalize_identity_spacetime_oof_stage_20261024.py',
    'audit_identity_spacetime_oof_implementation_20261024.py',
    'audit_identity_spacetime_oof_output_synthetic_20261024.py',
    'identity_spacetime_oof_complete_output_synthetic_20261024_v2.json',
    'audit_identity_spacetime_oof_coordinator_20261024.py',
    'identity_spacetime_coordinator_independent_seal_audit_20261024_v2.json',
    'audit_identity_spacetime_oof_background_20261024.py',
    'identity_spacetime_oof_background_independent_audit_20261024_v2.json',
    'verify_identity_spacetime_oof_output_20261024.py',
    'qa_identity_spacetime_output_verifier_20261024.py',
    'identity_spacetime_output_verifier_independent_QA_20261024_v2.json',
    'identity_spacetime_oof_independent_audit_design_20261024.md',
    'identity_spacetime_oof_data_independent_preflight_20261024.json',
    'identity_spacetime_oof_independent_preflight_20261024.json',
    'identity_spacetime_trainer_cpu_benchmark_20261024_v4.json',
    'identity_spacetime_oof_handoff_20261024.md',
]
SCRIPTS+=['identity_spacetime_oof_data_synthetic_round%d_20261024.json'%i for i in (1,2,3)]
SCRIPTS+=['identity_spacetime_trainer_synthetic_round%d_20261024_v4.json'%i for i in (1,2,3)]
SCRIPTS+=['identity_spacetime_oof_evaluator_synthetic_round%d_20261024_v3.json'%i for i in (1,2,3)]
SCRIPTS+=['identity_spacetime_coordinator_synthetic_round%d_20261024_v2.json'%i for i in (1,2,3)]
SCRIPTS+=['identity_spacetime_oof_finalizer_synthetic_round%d_20261024_v2.json'%i for i in (1,2,3)]
REPORTS=['项目进度.md','COVTrack_特征与时空改进_20261024.md','COVTrack_身份外观与时空OOF_20261024.md']
RESULT_DIRS={'identity_spacetime_oof_20261024_v1','identity_spacetime_oof_execution_20261024_v1'}
RECEIPTS={'launch_spec.json','launch_receipt.json','supervisor_identity.json','coordinator.started.json',
          'coordinator.exit.json','independent_output_audit.started.json','independent_output_audit.exit.json',
          'execution_receipt.json','execution_failure.json','coordinator_receipt.json','independent_output_audit.json',
          'completion_receipt.json','control_brief.json','progress_receipt.json'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('implementation','running','complete'))
    parser.add_argument('--repo',type=Path,default=ROOT.parent/'ovvvv');parser.add_argument('--receipt',action='append',default=[]);args=parser.parse_args()
    assert (args.repo/'.git').is_dir()
    preflight=json.loads((ROOT/'tmp_analysis/identity_spacetime_oof_independent_preflight_20261024.json').read_text())
    assert preflight['status']=='PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_IMPLEMENTATION_PREFLIGHT'
    if args.phase=='running':
        assert all(any(Path(p).name==name for p in args.receipt) for name in ('launch_receipt.json','supervisor_identity.json','coordinator.started.json'))
    if args.phase=='complete':
        receipt=ROOT/'results/identity_spacetime_oof_execution_20261024_v1/execution_receipt.json'
        assert json.loads(receipt.read_text())['status']=='PASS_IDENTITY_SPACETIME_OOF_EXECUTION_AND_INDEPENDENT_AUDIT'
    mapping=[(ROOT/'tmp_analysis'/p,Path('tmp_analysis')/p) for p in SCRIPTS]
    mapping += [(ROOT/'report'/p,Path('report')/p) for p in REPORTS]
    for value in args.receipt:
        relative=Path(value)
        assert not relative.is_absolute() and '..' not in relative.parts and len(relative.parts)==2
        assert relative.parts[0] in RESULT_DIRS and relative.name in RECEIPTS
        mapping.append((ROOT/'results'/relative,Path('evidence')/relative))
    payloads=[];destinations=set()
    for source,destination in mapping:
        assert destination not in destinations;destinations.add(destination)
        assert source.is_file() and not source.is_symlink(),str(source)
        data=source.read_bytes();assert len(data)<=2_000_000 and destination.suffix in ('.py','.md','.json'),str(source)
        if destination.suffix=='.json':assert isinstance(json.loads(data),dict)
        if destination.suffix=='.py':ast.parse(data)
        payloads.append((source,destination,data))
    assert sum(len(data) for _,_,data in payloads)<=20_000_000
    manifest=dict(stage='identity_spacetime_oof_20261024',phase=args.phase,phase_is_completion=args.phase=='complete',source_root=str(ROOT),files=[],
        excludes=['dataset','NPZ','NPY scores','JSONL','SQLite','full summary','model checkpoints','logs'])
    for source,destination,data in payloads:
        target=args.repo/destination;assert not target.is_symlink();target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.sync-tmp');temporary.write_bytes(data);temporary.replace(target)
        assert target.read_bytes()==data
        manifest['files'].append(dict(source=str(source),destination=str(destination),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    path=args.repo/'evidence'/('snapshot_20261024_identity_spacetime_oof_'+args.phase+'.json')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(path),files=len(payloads),bytes=sum(len(d) for _,_,d in payloads),phase=args.phase)))


if __name__=='__main__':main()
