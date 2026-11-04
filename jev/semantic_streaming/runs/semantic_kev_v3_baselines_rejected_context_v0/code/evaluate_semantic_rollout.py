#!/usr/bin/env python3
"""Evaluate independent, closed-loop policies on frozen v3 replay scenarios.

The default command is entirely CPU-only. --checkpoint explicitly enables local
GPU model inference. Each policy gets its own cursor and evolving buffer.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import qwen_roi as q
from prepare_semantic_kev import read_trace
from prepare_semantic_rollout import seal
from semantic_rollout import run_episode, safe_baseline_policy, seed_for


def verify_dataset(data):
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest.get('format') != 'semantic_continuous_replay_v3':
        raise ValueError('Continuous evaluation requires a v3 dataset')
    for filename, expected in manifest['files_sha256'].items():
        if q.sha((data / filename).read_bytes()) != expected:
            raise ValueError('Dataset changed: ' + filename)
    # Replaying with a modified physics helper would invalidate data provenance.
    for filename in ('semantic_rollout.py', 'prepare_semantic_kev.py'):
        if (data / 'code' / filename).read_bytes() != Path(__file__).with_name(filename).read_bytes():
            raise ValueError('Replay implementation differs from dataset snapshot: ' + filename)
    return manifest


def model_policy(checkpoint, actions):
    os.environ.update(HF_HOME=str(q.ROOT.parent / '.cache/huggingface'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(q.ROOT.parent / 'kev'))
    import torch
    from kev.checkpoint import LoadOptions
    from kev.model import training_context
    from kev.predictors import LocalPredictor
    predictor = LocalPredictor(str(checkpoint), 'cuda', LoadOptions(dtype=torch.bfloat16, merge=False), context=training_context(512))
    cheapest = {cid: min(rows, key=lambda a: (a['size_bytes'], a['action_id']))['action_id'] for cid, rows in actions.items()}

    def choose(row):
        if not row['state']['playback_started']:
            return cheapest[row['state']['case_id']], {'startup_rule': 'cheapest'}
        result = predictor(row)
        probabilities = result['probabilities']['encoding']
        if set(probabilities) != set(row['questions']['encoding']['criteria']):
            raise ValueError('Model returned different candidate keys')
        if (not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > .005):
            raise ValueError('Invalid model probabilities')
        return max(probabilities, key=probabilities.get), {
            'probabilities': probabilities, 'latency_ms': result['latency_ms']}

    return choose, predictor.environment


def summarize(rows):
    n = len(rows)
    return {'episodes': n, 'chunks': sum(r['chunks'] for r in rows),
        'mean_episode_reward': sum(r['reward_sum'] for r in rows) / n,
        'mean_reward_per_chunk': sum(r['reward_sum'] for r in rows) / sum(r['chunks'] for r in rows),
        'mean_episode_rebuffer_s': sum(r['rebuffer_s'] for r in rows) / n,
        'mean_startup_delay_s': sum(r['startup_delay_s'] for r in rows) / n,
        'mean_evidence_psnr_db': sum(r['psnr_sum'] for r in rows) / sum(r['chunks'] for r in rows),
        'rebuffer_ratio': sum(r['rebuffer_s'] for r in rows) / sum(r['playback_duration_s'] + r['rebuffer_s'] for r in rows),
        'mean_episode_bytes': sum(r['size_bytes'] for r in rows) / n,
        'mean_one_step_oracle_regret': sum(r['regret_sum'] for r in rows) / sum(r['chunks'] for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/semantic_kev_dev_v3')
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', help='Explicitly enables GPU inference; root-relative local checkpoint')
    p.add_argument('--policies', nargs='+', choices=['cheapest', 'bandwidth'], default=['cheapest', 'bandwidth'])
    a = p.parse_args()
    data, out = q.local_path(a.dataset), q.local_path(a.out)
    if out.exists():
        raise ValueError('Evaluation output exists; choose a fresh path')
    manifest = verify_dataset(data)
    cfg = manifest['config']
    actions = json.loads((data / 'actions.json').read_text())
    questions = json.loads((data / 'questions.json').read_text())
    specs = [json.loads(l) for l in (data / 'episodes.jsonl').read_text().splitlines() if json.loads(l)['split'] == 'valid']
    traces = {}
    for receipt in json.loads((data / 'traces.lock.json').read_text()):
        if receipt['split'] != 'valid':
            continue
        path = Path(receipt['path'])
        if q.sha(path.read_bytes()) != receipt['sha256']:
            raise ValueError('Trace changed')
        traces[receipt['id']] = read_trace(path)
    policies = list(dict.fromkeys(a.policies))
    model_choose, environment, checkpoint = None, None, None
    if a.checkpoint:
        checkpoint = q.local_path(a.checkpoint)
        for filename in ('head.pt', 'adapter_model.safetensors', 'adapter_config.json'):
            if not (checkpoint / filename).is_file():
                raise ValueError('Incomplete model checkpoint')
        model_choose, environment = model_policy(checkpoint, actions)
        policies.append('model')
    out.mkdir(parents=True)
    episodes, choices = [], {name: Counter() for name in policies}
    with (out / 'transitions.jsonl').open('w') as output:
        for name in policies:
            for spec in specs:
                choose = model_choose if name == 'model' else safe_baseline_policy(name, actions, cfg, seed_for(cfg['seed'], spec['id'], name))
                totals = {'policy': name, 'scenario_id': spec['id'], 'chunks': 0,
                    'reward_sum': 0.0, 'rebuffer_s': 0.0, 'startup_delay_s': 0.0,
                    'psnr_sum': 0.0, 'size_bytes': 0, 'playback_duration_s': 0.0, 'regret_sum': 0.0}
                for step, record in enumerate(run_episode(traces[spec['trace_id']], spec, actions, questions, cfg, choose)):
                    cid = record['request']['state']['case_id']
                    action = next(a for a in actions[cid] if a['action_id'] == record['selected_action'])
                    transition = record['transition']
                    totals['chunks'] += 1
                    totals['reward_sum'] += transition['reward']
                    totals['rebuffer_s'] += transition['rebuffer_s']
                    totals['startup_delay_s'] += transition['startup_delay_s']
                    totals['psnr_sum'] += action['evidence_rgb_psnr_db']
                    totals['size_bytes'] += action['size_bytes']
                    totals['playback_duration_s'] += action['duration_s']
                    totals['regret_sum'] += max(o['reward'] for o in record['outcomes'].values()) - transition['reward']
                    choices[name][record['selected_action']] += 1
                    output.write(json.dumps({'policy': name, 'scenario_id': spec['id'], 'step': step, **record}, ensure_ascii=False) + '\n')
                episodes.append(totals)
    q.write_json(out / 'episodes.json', episodes)
    summary = {'created_at': q.now(), 'dataset': a.dataset, 'scenario_count': len(specs),
        'metrics': {name: summarize([e for e in episodes if e['policy'] == name]) for name in policies},
        'action_counts': {k: dict(v) for k, v in choices.items()}, 'gpu_used': bool(a.checkpoint),
        'training_started': False, 'api_calls': 0, 'model_environment': environment,
        'dataset_manifest_sha256': q.sha((data / 'manifest.json').read_bytes()),
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': {f: q.sha((checkpoint / f).read_bytes()) for f in ('head.pt', 'adapter_model.safetensors', 'adapter_config.json')} if checkpoint else None,
        'note': 'Closed-loop synthetic-playlist trace replay, not measured network playback or semantic QA accuracy. Startup is separate from rebuffer. Each policy evolves its own state; inference latency excluded. One-step regret is diagnostic, not a long-horizon optimum.'}
    q.write_json(out / 'summary.json', summary)
    (out / 'code').mkdir()
    for filename in ('evaluate_semantic_rollout.py', 'semantic_rollout.py', 'prepare_semantic_kev.py', 'prepare_semantic_rollout.py', 'qwen_roi.py'):
        (out / 'code' / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    q.write_json(out / 'manifest.json', {'files_sha256': seal(out), 'dataset_manifest_sha256': summary['dataset_manifest_sha256']})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
