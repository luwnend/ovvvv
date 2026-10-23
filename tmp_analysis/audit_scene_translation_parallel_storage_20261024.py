#!/usr/bin/env python3
"""Read-only active OOF pins, protected bytes, process metadata and disk budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def module(path):
    spec=importlib.util.spec_from_file_location('storage_process_metadata_helpers',str(path))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def identity_matches(helper,bound):
    if bound is None:return dict(matches=False,running=False)
    current=helper.process_identity(bound['pid'])
    matches=current is not None and all(bound[k]==current[k] for k in ('pid','start_ticks','boot_id','cmdline'))
    return dict(matches=matches,running=matches and current['state'] not in ('Z','X'),bound=bound,current=current)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    spec=read(args.main_protocol);execution=Path(spec['execution_directory']);out=Path(spec['output_directory'])
    launch_path=execution/'launch_spec.json';launch=read(launch_path)
    pins=[]
    for name,expected in launch['file_pins'].items():
        current=sha(name);assert current==expected,'Main launch pin changed: '+name
        pins.append(dict(path=name,sha256=current,size_bytes=Path(name).stat().st_size))
    assert len(pins)==70
    assert sha(args.main_protocol)==launch['file_pins'][str(args.main_protocol.resolve())]
    authority=read(spec['protected_authority_path']);protected={}
    for name,record in authority['protected_files'].items():
        path=Path(name);actual=dict(exists=path.exists(),sha256=sha(path) if path.exists() else None)
        assert actual==record,'Protected file changed: '+name
        protected[name]=actual
    assert len(protected)==7
    helper=module(Path(spec['coordinator_path']))
    supervisor=identity_matches(helper,read(execution/'launch_receipt.json')['supervisor_identity'])
    completed=[];active=[]
    process_dir=execution/'model_processes'
    for initial in sorted(process_dir.glob('fold*_seed*.started.json')):
        label=initial.name[:-len('.started.json')];finished=process_dir/(label+'.exit.json')
        if finished.exists():
            exit_record=read(finished)
            assert exit_record['actual_exit_code']==0,'Main model failed: '+label
            completed.append(dict(model=label,actual_exit_code=exit_record['actual_exit_code']))
        else:
            active.append(dict(model=label,**identity_matches(helper,read(initial)['identity'])))
    score_paths=list(out.glob('models/*/shards/*.scores.npy'))
    logical=sum(p.stat().st_size for p in score_paths)
    allocated=sum(p.stat().st_blocks*512 for p in score_paths)
    # Only stat metadata is used: no NPY header or score payload is opened.
    # Deduct a full filesystem block from each file before crediting existing
    # allocation; separately reserve all possible shard header/padding blocks.
    credited=sum(max(0,p.stat().st_blocks*512-4096) for p in score_paths)
    total_score_payload=int(spec['resource_bounds']['score_storage_bytes'])
    max_shards=spec['population']['selected_fit_videos']*len(spec['lags'])*len(spec['seeds'])
    remaining_score_budget=max(0,total_score_payload-credited)+max_shards*4096
    max_diagnostic_targets=1070442
    numeric_bytes_per_target=(37*8)+(3*8)+(3*32)+(3*32)
    diagnostic_numeric=max_diagnostic_targets*numeric_bytes_per_target
    diagnostic_budget=int(.55*(1024**3))
    assert diagnostic_numeric==548066304 and diagnostic_budget>diagnostic_numeric
    extra_reserve=2*(1024**3)
    free=shutil.disk_usage(out.parent).free
    after=free-remaining_score_budget-diagnostic_budget-extra_reserve
    frozen_minimum=int(spec['resource_bounds']['minimum_free_disk_bytes'])
    possible=after>=frozen_minimum
    assert possible,'Current conservative parallel storage budget did not pass'
    assert sha(args.main_protocol)==launch['file_pins'][str(args.main_protocol.resolve())]
    result=dict(status='PASS_CURRENT_READONLY_MAIN_OOF_PINS_PROTECTED_AND_PARALLEL_STORAGE_PRECHECK',
        recorded_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        main_protocol_sha256=sha(args.main_protocol),main_launch_spec_sha256=sha(launch_path),
        main_launch_pin_count=len(pins),main_launch_pin_checks=pins,
        protected_count=len(protected),protected_files=protected,
        main_model_process_metadata=dict(completed_actual_exit_zero=completed,active=active,supervisor=supervisor),
        storage=dict(free_bytes_at_snapshot=free,score_files_stat_count=len(score_paths),
            existing_score_logical_bytes=logical,existing_score_allocated_bytes=allocated,
            existing_score_allocated_payload_credit=credited,total_frozen_score_payload_bytes=total_score_payload,
            remaining_score_plus_all_shard_padding_budget_bytes=remaining_score_budget,
            diagnostic_max_targets=max_diagnostic_targets,diagnostic_numeric_bytes_per_target=numeric_bytes_per_target,
            diagnostic_numeric_array_upper_bytes=diagnostic_numeric,
            diagnostic_total_planning_budget_bytes=diagnostic_budget,
            diagnostic_metadata_allowance_bytes=diagnostic_budget-diagnostic_numeric,
            miscellaneous_OOF_evaluation_audit_metadata_reserve_bytes=extra_reserve,
            free_after_remaining_scores_diagnostic_and_extra_reserve_bytes=after,
            main_protocol_minimum_free_disk_bytes=frozen_minimum,
            surplus_over_frozen_minimum_bytes=after-frozen_minimum,
            parallel_storage_planning_check_passed=possible,
            estimates_are_not_disk_reservations_or_peak_runtime_measurements=True),
        current_outcome_content_read=False,new_OOF_score_payload_or_metric_read=False,
        SQLite_opened=False,raw_GT_class_or_track_ID_values_decoded=False,
        existing_files_mutated=False,real_diagnostic_launched=False,
        elapsed_seconds=time.monotonic()-started)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('main_launch_pin_checks','protected_files')},indent=2))


if __name__=='__main__':main()
