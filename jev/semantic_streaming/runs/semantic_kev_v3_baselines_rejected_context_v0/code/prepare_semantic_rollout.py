#!/usr/bin/env python3
"""Build independently versioned Kev data using continuous transmission rollouts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import qwen_roi as q
from prepare_semantic_kev import read_trace
from semantic_rollout import quality_utility, run_episode, safe_baseline_policy, seed_for


def load_actions(cfg):
    actions = {cid: [] for cid in cfg['case_ids']}
    questions, receipts = {}, []
    used_ids = set()
    for source_index, name in enumerate(cfg['encoding_runs']):
        source = q.local_path('runs/' + name)
        if not json.loads((source / 'summary.json').read_text())['complete']:
            raise ValueError('Incomplete encoding run: ' + name)
        source_ids = {r['candidate_id']: r['action_id'] for rows in json.loads((source / 'actions.json').read_text()).values() for r in rows}
        candidates = json.loads((source / 'candidates.json').read_text())
        for c in sorted(candidates, key=lambda x: x['candidate_id']):
            cid = c['case_id']
            if cid not in actions:
                continue
            video = q.local_path(c['video_path'])
            if video.stat().st_size != c['size_bytes'] or q.sha(video.read_bytes()) != c['video_sha256']:
                raise ValueError('Encoding asset changed: ' + c['candidate_id'])
            if questions.setdefault(cid, c['question']) != c['question']:
                raise ValueError('Question differs between encoding runs')
            key = source_ids[c['candidate_id']] if source_index == 0 else f'a{len(used_ids):02d}'
            if key in used_ids:
                raise ValueError('Duplicate action ID')
            used_ids.add(key)
            psnr = c['metrics']['sampled_api_visible_evidence_rgb_psnr_db']
            actions[cid].append({'action_id': key, 'source_action_id': source_ids[c['candidate_id']],
                'source_encoding_run': name, **{k: c[k] for k in (
                    'candidate_id', 'mode', 'size_bytes', 'duration_s', 'roi_qoffset', 'global_crf',
                    'video_path', 'video_sha256', 'actual_bitrate_kbps', 'roi_review_flag')},
                'evidence_rgb_psnr_db': psnr, 'offline_quality_proxy': quality_utility(psnr, cfg)})
        receipts.append({'run': name, 'files_sha256': {filename: q.sha((source / filename).read_bytes())
            for filename in ('manifest.json', 'config.json', 'summary.json', 'candidates.json', 'actions.json', 'rate_size_quality.csv')}})
    for cid, rows in actions.items():
        if len(rows) != cfg['expected_actions_per_case'] or len({r['candidate_id'] for r in rows}) != len(rows):
            raise ValueError('Incomplete or duplicate candidate set: ' + cid)
        if len({r['duration_s'] for r in rows}) != 1:
            raise ValueError('Candidate durations differ')
    return actions, questions, receipts


def seal(directory):
    return {str(p.relative_to(directory)): q.sha(p.read_bytes())
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/semantic_kev_v3.json')
    a = p.parse_args()
    cfgpath = q.local_path(a.config)
    cfg = json.loads(cfgpath.read_text())
    dest = q.local_path(cfg['output_dataset'])
    if dest.exists():
        raise ValueError('Dataset exists; use a new version')
    actions, questions, sources = load_actions(cfg)
    bundle = (q.ROOT / cfg['trace_bundle']).resolve()
    chosen = {s: json.loads((bundle / f'{s}.json').read_text())[:n] for s, n in cfg['trace_limits'].items()}
    if {r['source_parent'] for r in chosen['train']} & {r['source_parent'] for r in chosen['valid']}:
        raise ValueError('Network acquisition group overlap')
    dest.mkdir()
    q.write_json(dest / 'actions.json', actions)
    q.write_json(dest / 'questions.json', questions)
    q.write_json(dest / 'encoding_sources.lock.json', sources)
    counts, labels, trace_receipts, episodes = {}, Counter(), [], []
    for split, traces in chosen.items():
        count = 0
        with (dest / f'{split}.jsonl').open('w') as output, (dest / f'{split}_rollouts.jsonl').open('w') as rollout_output:
            for tr in traces:
                path = bundle / tr['snapshot']
                if q.sha(path.read_bytes()) != tr['snapshot_sha256']:
                    raise ValueError('Network trace changed')
                trace = read_trace(path)
                trace_receipts.append({'split': split, 'id': tr['id'], 'source_parent': tr['source_parent'],
                    'path': str(path), 'sha256': tr['snapshot_sha256']})
                for position_index, position in enumerate(cfg['trace_positions']):
                    scenario_id = f"{tr['id']}/position{position_index}"
                    rng = random.Random(seed_for(cfg['seed'], scenario_id, 'playlist'))
                    playlist = []
                    for _ in range(cfg['playlist_repeats']):
                        cycle = list(cfg['case_ids'])
                        rng.shuffle(cycle)
                        playlist.extend(cycle)
                    # All policies start from the same cursor and playlist, with zero startup buffer.
                    spec = {'id': scenario_id, 'split': split, 'trace_id': tr['id'],
                        'start_segment': min(len(trace) - 1, max(1, round(position * (len(trace) - 1)))),
                        'start_offset_s': 0.0, 'playlist': playlist,
                        'request_seed': seed_for(cfg['seed'], scenario_id, 'requests')}
                    episodes.append(spec)
                    for policy in cfg['behavior_policies']:
                        choose = safe_baseline_policy(policy, actions, cfg, seed_for(cfg['seed'], scenario_id, policy))
                        for step, record in enumerate(run_episode(trace, spec, actions, questions, cfg, choose)):
                            row = record['request']
                            rewards = {k: o['reward'] for k, o in record['outcomes'].items()}
                            cid = row['state']['case_id']
                            lookup = {a['action_id']: a for a in actions[cid]}
                            best = max(rewards, key=lambda k: (rewards[k], -lookup[k]['size_bytes'], k))
                            row['questions']['encoding']['label'] = best
                            row['_supervision'] = {'quality_and_future_delay_not_online_inputs': True,
                                'candidate_reward': rewards, 'candidate_actual_download_s': record['downloads_s'],
                                'quality_proxy': {k: a['offline_quality_proxy'] for k, a in lookup.items()}}
                            row['_meta'] = {'id': f'{scenario_id}/{policy}/step{step:02d}', 'split': split,
                                'trace_group': tr['source_parent'], 'source': 'semantic_continuous_replay_v3',
                                'scenario_id': scenario_id, 'behavior_policy': policy, 'step': step,
                                'buffer_origin': 'continuous behavior-policy transmission rollout',
                                'behavior_selected_action': record['selected_action'],
                                'transition': record['transition']}
                            encoded = json.dumps(row, ensure_ascii=False) + '\n'
                            rollout_output.write(encoded)
                            if step == 0 and not cfg['collect_startup_records']:
                                continue
                            output.write(encoded)
                            labels[f'{split}/{cid}/{best}'] += 1
                            count += 1
                print(f'Collected {split}: {tr["id"]}, {count} cumulative records', flush=True)
        counts[split] = count
    (dest / 'episodes.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in episodes))
    q.write_json(dest / 'traces.lock.json', trace_receipts)
    (dest / 'config.json').write_bytes(cfgpath.read_bytes())
    (dest / 'code').mkdir()
    for path in (Path(__file__), Path(__file__).with_name('semantic_rollout.py'),
                 Path(__file__).with_name('prepare_semantic_kev.py'), Path(q.__file__)):
        (dest / 'code' / path.name).write_bytes(path.read_bytes())
    q.write_json(dest / 'manifest.json', {'created_at': q.now(), 'format': 'semantic_continuous_replay_v3',
        'config': cfg, 'counts': counts, 'label_counts': dict(labels), 'training_started': False,
        'action_counts': {cid: len(rows) for cid, rows in actions.items()},
        'files_sha256': seal(dest), 'limitations': cfg['dataset_role'],
        'reward_note': 'PSNR/10 without clipping; startup and rebuffer separated; unchanged stall weight 4.3; engineering utility, not QA correctness.',
        'physics_note': 'RTT, payload transfer and buffer-cap waits advance the same fractional trace cursor. Inference latency excluded.',
        'collection_note': 'A common cheapest-chunk startup seeds completed-transfer history. Startup is logged but excluded from training/valid records. Behavior policies only see past transfer observations and public candidate metadata. Labels use privileged one-step counterfactual replay.'})
    print(json.dumps({'dataset': str(dest), 'counts': counts, 'scenario_count': len(episodes), 'training_started': False}))


if __name__ == '__main__':
    main()
