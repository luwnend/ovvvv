#!/usr/bin/env python3
"""Publish an explicit lightweight reference-control snapshot, never run data."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / 'ovvvv'
FILES = [
    'report/项目进度.md',
    'report/COVTrack_特征与时空改进_20261024.md',
    'report/COVTrack_共同运动与独立几何参照_20261024.md',
    'tmp_analysis/identity_spacetime_oof_handoff_20261024.md',
    'tmp_analysis/identity_spacetime_scene_audit_live_checkpoint_20261024.json',
    'tmp_analysis/sync_scene_motion_stage_20261024.py',
    'tmp_analysis/endpoint_excluded_scene_motion.py',
    'tmp_analysis/check_endpoint_excluded_scene_motion_20261024.py',
    'tmp_analysis/audit_endpoint_excluded_scene_motion_20261024.py',
    'tmp_analysis/endpoint_excluded_scene_motion_runtime_20261024.json',
    'tmp_analysis/scene_motion_observability_audit_20261024.md',
    'tmp_analysis/scene_motion_novelty_audit_20261024.md',
    'tmp_analysis/scene_motion_sources_20261024/manifest.json',
    'tmp_analysis/spacetime_next_mechanism_design_audit_20261024.md',
]
FILES += ['tmp_analysis/endpoint_excluded_scene_motion_synthetic_round%d_20261024_v2.json' % i for i in (1, 2, 3)]
FILES += ['tmp_analysis/endpoint_excluded_scene_motion_independent_round%d_20261024.json' % i for i in (1, 2, 3)]
EXITS = ['identity_spacetime_oof_execution_20261024_v1/model_processes/fold%d_seed20261024.exit.json' % i for i in (0, 1)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert (REPO / '.git').is_dir()
    module_sha = sha((ROOT / 'tmp_analysis/endpoint_excluded_scene_motion.py').read_bytes())
    for i in (1, 2, 3):
        for pattern in ('endpoint_excluded_scene_motion_synthetic_round%d_20261024_v2.json',
                        'endpoint_excluded_scene_motion_independent_round%d_20261024.json'):
            receipt = json.loads((ROOT / 'tmp_analysis' / (pattern % i)).read_text())
            assert receipt['module_sha256'] == module_sha
            assert receipt['status'].startswith('PASS_')
    checkpoint = json.loads((ROOT / FILES[4]).read_text())
    assert checkpoint['pinned_files_unchanged'] and checkpoint['protected_files_checked'] == 7
    mapping = [(ROOT / p, Path(p)) for p in FILES]
    mapping += [(ROOT / 'results' / p, Path('evidence') / p) for p in EXITS]
    payloads = []
    for source, destination in mapping:
        assert source.is_file() and not source.is_symlink()
        assert destination.suffix in ('.md', '.json', '.py') and '..' not in destination.parts
        data = source.read_bytes()
        assert len(data) <= 2_000_000
        if source.suffix == '.py':
            ast.parse(data)
        if source.suffix == '.json':
            json.loads(data)
        if source.name.endswith('.exit.json'):
            assert json.loads(data)['actual_exit_code'] == 0
        payloads.append((source, destination, data))
    assert len({p for _, p, _ in payloads}) == len(payloads)
    assert sum(len(data) for _, _, data in payloads) <= 10_000_000
    manifest = dict(stage='scene_motion_reference_and_audits_20261024',
        OOF_is_complete=False, new_method_effectiveness_established=False,
        scene_motion_is_ordinary_control=True, source_root=str(ROOT), files=[],
        excludes=['dataset', 'images', 'weights', 'score shards', 'SQLite', 'JSONL',
                  'model checkpoints', 'logs', 'full paper PDFs', 'paper renderings'])
    for source, destination, data in payloads:
        target = REPO / destination
        assert not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.sync-tmp')
        temporary.write_bytes(data)
        temporary.replace(target)
        assert target.read_bytes() == data
        manifest['files'].append(dict(source=str(source), destination=str(destination),
                                     bytes=len(data), sha256=sha(data)))
    snapshot = REPO / 'evidence/snapshot_20261024_scene_motion_control.json'
    snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(snapshot), files=len(payloads),
                         bytes=sum(len(data) for _, _, data in payloads))))


if __name__ == '__main__':
    main()
