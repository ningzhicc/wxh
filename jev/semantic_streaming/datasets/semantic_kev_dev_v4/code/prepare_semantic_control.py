#!/usr/bin/env python3
"""Build v4 guarded, three-step planning data while preserving frozen v3 assets."""
import argparse
from collections import Counter
import json
from pathlib import Path

import qwen_roi as q
from evaluate_semantic_rollout import verify_dataset
from prepare_semantic_kev import read_trace
from prepare_semantic_rollout import seal
from semantic_control import plan_labels, guarded_behavior, run_control_episode, validate_config
from semantic_rollout import seed_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/semantic_kev_v4.json')
    a = p.parse_args()
    cfgpath = q.local_path(a.config)
    override = json.loads(cfgpath.read_text())
    source = q.local_path(override['source_dataset'])
    parent = verify_dataset(source)
    cfg = {**parent['config'], **override}
    validate_config(cfg)
    dest = q.local_path(cfg['output_dataset'])
    if dest.exists():
        raise ValueError('Dataset exists; choose a new version')
    actions = json.loads((source / 'actions.json').read_text())
    questions = json.loads((source / 'questions.json').read_text())
    grid = [(a['global_crf'], a['roi_qoffset']) for a in next(iter(actions.values()))]
    if any([(a['global_crf'], a['roi_qoffset']) for a in rows] != grid for rows in actions.values()):
        raise ValueError('Future preview requires the same grid order across videos')
    specs = [json.loads(l) for l in (source / 'episodes.jsonl').read_text().splitlines()]
    receipts = json.loads((source / 'traces.lock.json').read_text())
    traces, groups = {}, {}
    for r in receipts:
        path = Path(r['path'])
        if q.sha(path.read_bytes()) != r['sha256']:
            raise ValueError('Network trace changed')
        traces[r['id']] = read_trace(path)
        groups[r['id']] = r['source_parent']
    dest.mkdir()
    for filename in ('actions.json', 'questions.json', 'episodes.jsonl', 'traces.lock.json', 'encoding_sources.lock.json'):
        (dest / filename).write_bytes((source / filename).read_bytes())
    q.write_json(dest / 'config.json', cfg)
    counts, labels, guard_stats = {}, Counter(), {}
    for split in ('train', 'valid'):
        count, stats = 0, Counter()
        with (dest / f'{split}.jsonl').open('w') as output, (dest / f'{split}_rollouts.jsonl').open('w') as log:
            for spec in (s for s in specs if s['split'] == split):
                for policy in cfg['behavior_policies']:
                    choose = guarded_behavior(policy, actions, cfg, seed_for(cfg['seed'], spec['id'], policy))
                    for step, record in enumerate(run_control_episode(traces[spec['trace_id']], spec, actions, questions, cfg, choose)):
                        row = record['request']
                        catalogs = [actions[cid] for cid in spec['playlist'][step:step + cfg['planning_horizon_chunks']]]
                        plan = plan_labels(row['state'], catalogs, cfg)
                        criteria = row['questions']['encoding']['criteria']
                        if set(plan['candidate_planned_return']) != set(criteria):
                            raise ValueError('Planning labels and online guard differ')
                        row['questions']['encoding']['label'] = plan['label']
                        row['_supervision'] = {'candidate_planned_return': plan['candidate_planned_return'],
                            'planning': plan, 'candidate_reward': {k: o['reward'] for k, o in record['outcomes'].items()},
                            'candidate_actual_download_s': record['downloads_s'],
                            'quality_proxy': {a['action_id']: a['offline_quality_proxy'] for a in actions[row['state']['case_id']]},
                            'real_future_bandwidth_used_for_label': False}
                        row['_meta'] = {'id': f"{spec['id']}/{policy}/step{step:02d}", 'split': split,
                            'source': 'semantic_guarded_planning_v4', 'scenario_id': spec['id'],
                            'trace_group': groups[spec['trace_id']], 'behavior_policy': policy, 'step': step,
                            'behavior_selected_action': record['selected_action'], 'transition': record['transition'],
                            'guard': record['guard'], 'decision_info': record['decision_info']}
                        encoded = json.dumps(row, ensure_ascii=False) + '\n'
                        log.write(encoded)
                        stats['all_chunks'] += 1
                        stats['no_candidate_within_budget'] += record['guard']['no_candidate_within_budget']
                        stats['forced_chunks'] += record['guard']['forced']
                        if (step == 0 and not cfg['collect_startup_records']) or (record['guard']['forced'] and not cfg['collect_forced_records']):
                            continue
                        output.write(encoded)
                        labels[f"{split}/{row['state']['case_id']}/{plan['label']}"] += 1
                        count += 1
                print(f"Collected {split}: {spec['id']}, {count} model-decision records", flush=True)
        if not count:
            raise ValueError('No trainable model choices')
        counts[split], guard_stats[split] = count, dict(stats)
    (dest / 'code').mkdir()
    for filename in ('prepare_semantic_control.py', 'semantic_control.py', 'semantic_rollout.py',
                     'prepare_semantic_kev.py', 'prepare_semantic_rollout.py', 'evaluate_semantic_rollout.py', 'qwen_roi.py'):
        (dest / 'code' / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    q.write_json(dest / 'source.lock.json', {'parent_dataset': override['source_dataset'],
        'parent_manifest_sha256': q.sha((source / 'manifest.json').read_bytes()),
        'override_config_sha256': q.sha(cfgpath.read_bytes()), 'preview_grid_order': grid})
    q.write_json(dest / 'manifest.json', {'created_at': q.now(), 'format': 'semantic_guarded_planning_v4',
        'config': cfg, 'counts': counts, 'action_counts': {cid: len(rows) for cid, rows in actions.items()},
        'label_counts': dict(labels), 'guard_collection_counts': guard_stats, 'training_started': False,
        'files_sha256': seal(dest), 'limitations': cfg['dataset_role'],
        'teacher_note': 'Exact depth-3 search with updated forecast buffer and previous rate. Offline PSNR is teacher-only; labels have no dependency on actual future trace or counterfactual replay delay.',
        'guard_note': 'Same guard in collection, planning and model inference. No feasible estimate means explicit cheapest fallback, never a guarantee against real stalls.'})
    print(json.dumps({'dataset': str(dest), 'counts': counts, 'guard_counts': guard_stats, 'training_started': False}))


if __name__ == '__main__':
    main()
