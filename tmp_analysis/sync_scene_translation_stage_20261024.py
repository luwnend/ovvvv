#!/usr/bin/env python3
"""Publish exact small sources and launch receipts for a running diagnostic."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / 'ovvvv'
EXECUTION = 'scene_translation_donorout_execution_20261024_v1'
FILES = [
    'report/项目进度.md',
    'report/COVTrack_特征与时空改进_20261024.md',
    'report/COVTrack_外部运动预测诊断_20261024.md',
    'tmp_analysis/identity_spacetime_oof_handoff_20261024.md',
    'tmp_analysis/scene_translation_donorout_handoff_20261024.md',
    'tmp_analysis/sync_scene_translation_stage_20261024.py',
    'tmp_analysis/cached_endpoint_scene_motion_20261024.py',
    'tmp_analysis/audit_cached_scene_motion_independent_20261024.py',
    'tmp_analysis/audit_cached_scene_motion_layout_20261024.py',
    'tmp_analysis/cached_scene_motion_independent_layout_20261024_v2.json',
    'tmp_analysis/scene_translation_donorout_20261024.py',
    'tmp_analysis/check_scene_translation_donorout_20261024.py',
    'tmp_analysis/audit_scene_translation_producer_synthetic_20261024.py',
    'tmp_analysis/scene_translation_producer_independent_synthetic_20261024_v1.json',
    'tmp_analysis/run_scene_translation_donorout_20261024.py',
    'tmp_analysis/audit_scene_translation_runner_independent_20261024.py',
    'tmp_analysis/scene_translation_runner_independent_synthetic_20261024_v1.json',
    'tmp_analysis/verify_scene_translation_donorout_20261024.py',
    'tmp_analysis/scene_translation_donorout_design_20261024.md',
    'tmp_analysis/scene_translation_donorout_protocol_review_20261024.md',
    'tmp_analysis/scene_translation_donorout_protocol_20261024.json',
    'tmp_analysis/scene_translation_donorout_metadata_preparation_20261024.json',
    'tmp_analysis/scene_translation_parallel_storage_precheck_20261024_v1.json',
    'tmp_analysis/audit_scene_translation_parallel_storage_20261024.py',
    'tmp_analysis/audit_scene_translation_donorout_preflight_20261024.py',
    'tmp_analysis/scene_translation_donorout_independent_preflight_20261024.json',
    'tmp_analysis/audit_scene_translation_donorout_launch_live_20261024.py',
    'tmp_analysis/scene_translation_donorout_launch_live_audit_20261024.json',
]
FILES += ['tmp_analysis/' + pattern % i for i in (1, 2, 3) for pattern in (
    'cached_endpoint_scene_motion_synthetic_round%d_20261024_v2.json',
    'cached_scene_motion_independent_round%d_20261024_v2.json',
    'scene_translation_donorout_producer_synthetic_round%d_20261024_v2.json')]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert (REPO / '.git').is_dir()
    protocol = json.loads((ROOT / 'tmp_analysis/scene_translation_donorout_protocol_20261024.json').read_text())
    assert protocol['status'] == 'FROZEN_GTFREE_SCENE_TRANSLATION_DONOROUT_V1'
    for path, digest in protocol['input_pins'].items():
        assert sha(Path(path).read_bytes()) == digest, path
    authority = json.loads(Path(protocol['protected_authority_path']).read_text())
    assert len(authority['protected_files']) == 7
    for path, record in authority['protected_files'].items():
        assert Path(path).exists() == record['exists']
        assert (sha(Path(path).read_bytes()) if record['exists'] else None) == record['sha256']
    # Include only the final auditor's explicitly frozen small QA receipts.
    audit_QA = protocol['independent_auditor_QA_paths']
    mapping = [(ROOT / path, Path(path)) for path in FILES]
    mapping += [(Path(path), Path('tmp_analysis') / Path(path).name) for path in audit_QA]
    mapping += [(ROOT / 'results' / EXECUTION / name, Path('evidence') / EXECUTION / name)
                for name in ('launch_spec.json', 'launch_receipt.json', 'supervisor_identity.json')]
    assert len({dst for _, dst in mapping}) == len(mapping)
    payloads = []
    for source, destination in mapping:
        assert source.is_file() and not source.is_symlink()
        assert destination.suffix in ('.md', '.json', '.py') and '..' not in destination.parts
        data = source.read_bytes()
        assert len(data) <= 2_000_000
        if source.suffix == '.py':
            ast.parse(data)
        elif source.suffix == '.json':
            json.loads(data)
        payloads.append((source, destination, data))
    assert sum(len(data) for _, _, data in payloads) <= 10_000_000
    manifest = dict(stage='scene_translation_donorout_launch_20261024',
        source_root=str(ROOT), diagnostic_complete=False, OOF_complete=False,
        identity_tracking_or_novelty_established=False, files=[],
        excludes=['data', 'images', 'weights', 'cache', 'scores', 'checkpoints', 'JSONL', 'logs'])
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
    snapshot = REPO / 'evidence/snapshot_20261024_scene_translation_launch.json'
    snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(snapshot), files=len(payloads),
                         bytes=sum(len(data) for _, _, data in payloads))))


if __name__ == '__main__':
    main()
