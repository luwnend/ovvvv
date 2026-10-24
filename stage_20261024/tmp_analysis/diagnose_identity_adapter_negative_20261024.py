#!/usr/bin/env python3
"""Six targeted checks of the sealed negative adapter control; no retraining."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/identity_spacetime_oof_20261024_v1'
DEST = ROOT / 'tmp_analysis/identity_adapter_negative_diagnosis_20261024.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(not DEST.exists(), 'Preserve an existing diagnosis')
    completion = read(OUT / 'completion_receipt.json')
    brief = read(OUT / 'control_brief.json')
    audit = read(OUT / 'independent_output_audit.json')
    prepath = ROOT / 'tmp_analysis/identity_spacetime_oof_independent_preflight_20261024.json'
    preflight = read(prepath)
    require(completion['status'] == 'COMPLETE_AUDITED_IDENTITY_SPACETIME_SOURCE_OOF_STAGE', 'Incomplete stage')
    require(audit['status'] == 'PASS_INDEPENDENT_IDENTITY_SPACETIME_OOF_OUTPUT_AUDIT', 'Missing fresh full audit')
    pins = {str(prepath): sha(prepath)}
    for f, name in (('control_brief.json', 'control_brief'),
                    ('independent_output_audit.json', 'independent_output_audit')):
        ref = completion['files_sha256'][name]
        require(ref['path'] == str(OUT / f), 'Unexpected completion path')
        expected = ref['sha256']
        require(sha(OUT / f) == expected, 'Completion reference mismatch: ' + f)
        pins[str(OUT / f)] = expected
    pins[str(OUT / 'completion_receipt.json')] = sha(OUT / 'completion_receipt.json')
    models = []
    for fold in range(5):
        for seed in (20261024, 20261025, 20261026):
            directory = OUT / 'models' / ('fold%d_seed%d' % (fold, seed))
            receipt = read(directory / 'single_model_receipt.json')
            trace = read(directory / 'training_trace.json')
            checkpoint = read(directory / 'checkpoint.json')
            seal = read(directory / 'GT_free_predictions_sealed.json')
            require(sha(directory / 'training_trace.json') == receipt['training_trace_sha256'], 'Trace SHA mismatch')
            for f in ('checkpoint.json', 'training_trace.json'):
                require(sha(directory / f) == seal['files_sha256'][f], 'Model seal mismatch')
                pins[str(directory / f)] = sha(directory / f)
            require(len(trace['adapter']) == 50 and len(trace['heads']) == 20, 'Schedule was not completed')
            n = trace['training_query_lags']
            require(trace['global_weight_sum'] == 1., 'Global weights changed')
            expected_steps = 50 * math.ceil(n / 64)
            require(trace['adapter_optimizer_steps'] == expected_steps, 'Missing adapter steps')
            require(all(v == 20 * math.ceil(n / 64) for v in trace['head_optimizer_steps'].values()), 'Missing head steps')
            for key in ('adapter', 'heads'):
                for i, epoch in enumerate(trace[key], 1):
                    require(epoch['epoch'] == i and sum(epoch['batch_sizes']) == n, 'Incomplete epoch population')
                    value = epoch['surrogate_step_objective_mean']
                    values = list(value.values()) if isinstance(value, dict) else [value]
                    require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid training trace')
            state = {}
            for key, value in checkpoint['adapter_state'].items():
                array = np.asarray(value['values'], dtype=np.float64)
                require(list(array.shape) == value['shape'] and np.isfinite(array).all(), 'Invalid adapter tensor')
                state[key] = array
            last_norm = float(np.linalg.norm(state['last.weight']))
            require(last_norm > 0, 'Adapter stayed at zero residual initialization')
            metadata = checkpoint['metadata']
            require(metadata['fold'] == fold and metadata['seed'] == seed, 'Wrong model lineage')
            train = set(metadata['train_sources']); held = set(metadata['held_out_sources'])
            require(train.isdisjoint(held) and len(train | held) == 341, 'Source leakage')
            projection = receipt['train_projection_receipt']
            require(set(projection['allowed_sources']) == train, 'Projection crossed training partition')
            require(not projection['heldout_identity_values_decoded'] and not projection['forbidden_correctness_fields_decoded'], 'Forbidden labels decoded')
            require(receipt['training_sources_only'] and not receipt['held_out_identity_values_decoded'], 'Invalid training boundary')
            models.append(dict(fold=fold, seed=seed, query_lags=n,
                               adapter_optimizer_steps=expected_steps, last_weight_norm=last_norm,
                               surrogate_first=trace['adapter'][0]['surrogate_step_objective_mean'],
                               surrogate_final=trace['adapter'][-1]['surrogate_step_objective_mean']))
    checks = audit['independent_checks']
    require(checks['all_ten_columns_every_retained_candidate_independently_recomputed']
            and checks['raw_parent_bitwise_exact_every_candidate'], 'Scoring contract not audited')
    require(checks['maximum_absolute_score_error'] < 1e-10, 'Recomputed scores disagree')
    require(checks['baseline_parity']['cells'] == 8208 and checks['baseline_parity']['shared_hard_queries'] == 1331, 'Population drift')
    mutation = preflight['independent_checks']['actual_default_worker_heldout_mutation_invariance']
    for f, expected in preflight['reviewed_synthetic_receipt_files_sha256'].items():
        require(sha(f) == expected, 'Preflight evidence changed')
    require(mutation['heldout_label_mutation_changes_authority_preserves_all_functional_model_files']
            and mutation['permitted_train_label_mutation_changes_checkpoint']
            and mutation['actual_decoder_poison_guard'], 'Missing actual worker mutation challenge')
    all_negative = []
    shared_rows = []
    for seed, item in brief['per_seed'].items():
        for lag, cell in item['lags'].items():
            require(cell['hard_queries'] == {'1': 2949, '4': 2142}[lag], 'Hard query denominator changed')
            require(cell['features']['A0G0']['errors'] == cell['features']['raw_appearance_cosine']['errors'], 'G0 altered raw ranking')
            require(cell['features']['A1G0']['errors'] == cell['features']['adapter_appearance_cosine']['errors'], 'G0 altered adapter ranking')
            delta = cell['comparisons']['adapter_cosine_minus_raw']
            require(delta['net_correct'] < 0 and delta['source_GTtrajectory_query_macro_delta'] < 0, 'Negative direction not consistent')
            all_negative.append(dict(seed=int(seed), lag=int(lag), **delta))
        shared = item['cross_lag_shared']
        require(shared['shared_hard_queries'] == 1331 and shared['sources'] == 107, 'Shared query denominator changed')
        for lag in (1, 4):
            features = shared['features']; key = 'source_GTtrajectory_query_macro_lag%d' % lag
            value = {f: v[key] for f, v in features.items()}
            require(value['adapter_appearance_cosine'] < value['raw_appearance_cosine'], 'Shared-set negative direction differs')
            shared_rows.append(dict(seed=int(seed), lag=lag,
                raw_errors=features['raw_appearance_cosine']['error_lag%d' % lag],
                adapter_errors=features['adapter_appearance_cosine']['error_lag%d' % lag],
                A1G1_errors=features['A1G1']['error_lag%d' % lag],
                adapter_minus_raw=value['adapter_appearance_cosine']-value['raw_appearance_cosine'],
                A1G1_minus_raw=value['A1G1']-value['raw_appearance_cosine'],
                A1G1_minus_A1G0=value['A1G1']-value['A1G0'],
                A1G1_minus_A1Gmask=value['A1G1']-value['A1Gmask'],
                interaction=(value['A1G1']-value['A1G0'])-(value['A0G1']-value['A0G0'])))
    report = dict(status='STOP_THIS_FIXED_IDENTITY_ADAPTER_CONTROL_AFTER_THREE_ROUNDS_TWO_CHECKS',
        rounds=[
            dict(round=1, checks=['All 15 real traces: complete 50/20 epochs, finite surrogate losses, global weights and optimizer-step counts',
                                 'All 15 sealed checkpoints finite with nonzero formerly-zero residual weights; actual worker label-mutation challenge retained'], models=models),
            dict(round=2, checks=['Bind fresh all-candidate ten-score reconstruction and exact parent raw/ranking coverage audit',
                                 'All 15 train-source partitions and allowed identity projections remain disjoint from heldout; preflight QA hashes retained'],
                 maximum_absolute_score_error=checks['maximum_absolute_score_error'], optimizer_replay_performed=False),
            dict(round=3, checks=['All fixed seeds and both lags lose pure appearance retrieval on the prespecified hard query populations',
                                 'Same negative pure-appearance direction on fixed cross-lag 1331 queries; geometry rescue and interaction vary. A1G1 loses raw micro in every shared cell; one shared macro cell exceeds raw by 0.028644 pp, so it does not consistently recover raw'],
                 full_hard=all_negative, shared_1331=shared_rows)],
        interpretation='The executed fixed adapter is ineffective on this examined FIT source-OOF readout. No discovered execution, projection or scoring defect explains the negative result. This is not a proof that identity learning or temporal networks fail.',
        training_trace_limit='Surrogate minibatch objectives decrease descriptively; different step states/orders prevent interpreting them as exact full-data fixed-state loss or convergence proof. No optimizer replay, early-epoch rescue, seed selection, tune/test read or new training.',
        likely_hypotheses_not_identified=['Sparse supervised trajectories may allow source-specific metric distortion', 'Fixed pairwise training objective and hierarchy need not preserve unseen-source top1 geometry'],
        negative_control_scope='Stop this 256-64-256/50-epoch/three-seed control as an improvement; do not rename it a new temporal mechanism.',
        input_sha256=pins, GPU_used=False)
    report['script_sha256'] = sha(__file__)
    DEST.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(DEST), models=len(models), rounds=3, checks_per_round=2)))


if __name__ == '__main__':
    main()
