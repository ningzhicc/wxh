#!/usr/bin/env python3
"""Closed-loop v4 evaluation: historical baselines, guarded policy and offline teacher.

CPU by default; --checkpoint explicitly loads a local model on GPU. Teacher PSNR
privilege is declared separately and must not be reported as learned-model effect.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import qwen_roi as q
from evaluate_semantic_rollout import model_policy, summarize
from prepare_semantic_kev import read_trace
from prepare_semantic_rollout import seal
from semantic_control import guarded_behavior, model_with_fallback, plan_labels, run_control_episode, validate_config
from semantic_rollout import run_episode, safe_baseline_policy, seed_for


def verify_control_dataset(data):
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest.get('format') != 'semantic_guarded_planning_v4':
        raise ValueError('This evaluator requires guarded v4 data')
    for name, expected in manifest['files_sha256'].items():
        if q.sha((data / name).read_bytes()) != expected:
            raise ValueError('Dataset changed: ' + name)
    for name in ('semantic_control.py', 'semantic_rollout.py', 'prepare_semantic_kev.py'):
        if (data / 'code' / name).read_bytes() != Path(__file__).with_name(name).read_bytes():
            raise ValueError('Control/physics implementation differs from dataset: ' + name)
    validate_config(manifest['config'])
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/semantic_kev_dev_v4')
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', help='Explicitly enables GPU model inference')
    p.add_argument('--policies', nargs='+', choices=['cheapest', 'bandwidth', 'guarded_bandwidth', 'offline_quality_teacher'],
                   default=['cheapest', 'bandwidth', 'guarded_bandwidth', 'offline_quality_teacher'])
    a = p.parse_args()
    data, out = q.local_path(a.dataset), q.local_path(a.out)
    if out.exists():
        raise ValueError('Output exists; use a fresh directory')
    manifest = verify_control_dataset(data)
    cfg = manifest['config']
    actions = json.loads((data / 'actions.json').read_text())
    questions = json.loads((data / 'questions.json').read_text())
    specs = [r for r in map(json.loads, (data / 'episodes.jsonl').read_text().splitlines()) if r['split'] == 'valid']
    traces = {}
    for receipt in json.loads((data / 'traces.lock.json').read_text()):
        if receipt['split'] != 'valid':
            continue
        path = Path(receipt['path'])
        if q.sha(path.read_bytes()) != receipt['sha256']:
            raise ValueError('Trace changed')
        traces[receipt['id']] = read_trace(path)
    policies = list(dict.fromkeys(a.policies))
    checkpoint, learned, environment = None, None, None
    if a.checkpoint:
        checkpoint = q.local_path(a.checkpoint)
        for name in ('head.pt', 'adapter_model.safetensors', 'adapter_config.json'):
            if not (checkpoint / name).is_file():
                raise ValueError('Checkpoint is incomplete')
        choose, environment = model_policy(checkpoint, actions)
        learned = model_with_fallback(choose, actions)
        policies.append('guarded_model')
    out.mkdir(parents=True)
    episodes, guard_counts = [], {}
    with (out / 'transitions.jsonl').open('w') as log:
        for policy in policies:
            policy_counts = Counter()
            for spec in specs:
                position = [0]
                if policy == 'offline_quality_teacher':
                    def choose(row):
                        catalogs = [actions[cid] for cid in spec['playlist'][position[0]:position[0] + cfg['planning_horizon_chunks']]]
                        plan = plan_labels(row['state'], catalogs, cfg)
                        return plan['label'], {'offline_quality_teacher': True, 'forecast_return': plan['candidate_planned_return'][plan['label']]}
                elif policy == 'guarded_model':
                    choose = learned
                elif policy == 'guarded_bandwidth':
                    choose = guarded_behavior('bandwidth', actions, cfg, seed_for(cfg['seed'], spec['id'], policy))
                else:
                    choose = safe_baseline_policy(policy, actions, cfg, seed_for(cfg['seed'], spec['id'], policy))
                engine = run_episode if policy in ('cheapest', 'bandwidth') else run_control_episode
                total = {'policy': policy, 'scenario_id': spec['id'], 'chunks': 0, 'reward_sum': 0.0,
                    'rebuffer_s': 0.0, 'startup_delay_s': 0.0, 'psnr_sum': 0.0,
                    'size_bytes': 0, 'playback_duration_s': 0.0, 'regret_sum': 0.0}
                for step, record in enumerate(engine(traces[spec['trace_id']], spec, actions, questions, cfg, choose)):
                    cid = record['request']['state']['case_id']
                    action = next(a for a in actions[cid] if a['action_id'] == record['selected_action'])
                    transition = record['transition']
                    total['chunks'] += 1
                    total['reward_sum'] += transition['reward']
                    total['rebuffer_s'] += transition['rebuffer_s']
                    total['startup_delay_s'] += transition['startup_delay_s']
                    total['psnr_sum'] += action['evidence_rgb_psnr_db']
                    total['size_bytes'] += action['size_bytes']
                    total['playback_duration_s'] += action['duration_s']
                    total['regret_sum'] += max(o['reward'] for o in record['outcomes'].values()) - transition['reward']
                    if 'guard' in record:
                        guard = record['guard']
                        if record['selected_action'] not in guard['allowed_action_ids']:
                            raise ValueError('Selected action escaped the guard')
                        policy_counts['guarded_chunks'] += 1
                        policy_counts['forced_chunks'] += guard['forced']
                        policy_counts['no_candidate_within_budget'] += guard['no_candidate_within_budget']
                        policy_counts['invalid_action_fallback'] += bool(record['decision_info'].get('invalid_action_fallback'))
                        policy_counts['model_prediction_fallback'] += bool(record['decision_info'].get('model_prediction_fallback'))
                    log.write(json.dumps({'policy': policy, 'scenario_id': spec['id'], 'step': step, **record}, ensure_ascii=False) + '\n')
                    position[0] += 1
                episodes.append(total)
            guard_counts[policy] = dict(policy_counts)
            print('Evaluated ' + policy, flush=True)
    q.write_json(out / 'episodes.json', episodes)
    summary = {'created_at': q.now(), 'dataset': a.dataset, 'scenario_count': len(specs),
        'metrics': {name: summarize([e for e in episodes if e['policy'] == name]) for name in policies},
        'guard_counts': guard_counts, 'gpu_used': bool(a.checkpoint), 'training_started': False, 'api_calls': 0,
        'model_environment': environment, 'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': {f: q.sha((checkpoint / f).read_bytes()) for f in ('head.pt', 'adapter_model.safetensors', 'adapter_config.json')} if checkpoint else None,
        'dataset_manifest_sha256': q.sha((data / 'manifest.json').read_bytes()),
        'information_policy': {'cheapest': 'public size only', 'bandwidth': 'past transfer observations and public catalog',
            'guarded_bandwidth': 'public online guard; no measured PSNR',
            'offline_quality_teacher': 'PRIVILEGED measured PSNR for all planning chunks; historical network forecast only; not a learned model',
            'guarded_model': 'filtered candidates, public upcoming size arrays and past transfers; no PSNR or future network'},
        'note': 'Same v3 physical reward, trace scenarios and playlists. Guarded policies evolve independent states. Forecast planning objective is supervision-only; reported reward excludes terminal planning penalty. No semantic QA or measured network playback claim.'}
    q.write_json(out / 'summary.json', summary)
    (out / 'code').mkdir()
    for name in ('evaluate_semantic_control.py', 'semantic_control.py', 'semantic_rollout.py', 'evaluate_semantic_rollout.py',
                 'prepare_semantic_kev.py', 'prepare_semantic_rollout.py', 'qwen_roi.py'):
        (out / 'code' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    q.write_json(out / 'manifest.json', {'dataset_manifest_sha256': summary['dataset_manifest_sha256'], 'files_sha256': seal(out)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
