#!/usr/bin/env python3
"""Audit v3 state continuity, candidate feasibility, quality scale and split provenance."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import qwen_roi as q
from audit_semantic_data import online_signature
from evaluate_semantic_rollout import verify_dataset
from prepare_semantic_rollout import seal
from semantic_rollout import outcome


def require_close(a, b, message):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(message)


def audit_continuity(rows, actions, cfg):
    previous = {}
    checked = 0
    for row in rows:
        meta, state = row['_meta'], row['state']
        key = (meta['scenario_id'], meta['behavior_policy'])
        transition = meta['transition']
        selected = actions[state['case_id']][meta['behavior_selected_action']]
        supervision = row['_supervision']
        candidates = actions[state['case_id']]
        if set(supervision['candidate_reward']) != set(candidates):
            raise ValueError('Counterfactual candidate set differs')
        for action_id, candidate in candidates.items():
            expected = outcome(candidate, supervision['candidate_actual_download_s'][action_id], state, cfg)
            require_close(supervision['candidate_reward'][action_id], expected['reward'], 'Incorrect counterfactual reward')
        label = row['questions']['encoding']['label']
        require_close(supervision['candidate_reward'][label], max(supervision['candidate_reward'].values()), 'Incorrect best-action label')
        if key not in previous:
            if meta['step'] != 0 or state['playback_started'] or state['buffer_seconds'] != 0:
                raise ValueError('Invalid startup state')
        else:
            prev = previous[key]
            if meta['step'] != prev['_meta']['step'] + 1 or not state['playback_started']:
                raise ValueError('Non-contiguous rollout steps')
            pt = prev['_meta']['transition']
            require_close(state['buffer_seconds'], pt['buffer_after_s'], 'Buffer transition broken')
            pa = actions[prev['state']['case_id']][prev['_meta']['behavior_selected_action']]
            require_close(state['previous_bitrate_kbps'], pa['actual_bitrate_kbps'], 'Previous action broken')
            require_close(state['recent_link_bandwidth_mbps'][-1], pt['observed_link_mbps'], 'Future/history boundary broken')
            if meta['step'] > 1:
                expected = (prev['state']['recent_link_bandwidth_mbps'] + [pt['observed_link_mbps']])[-cfg['history_length']:]
                if state['recent_link_bandwidth_mbps'] != expected:
                    raise ValueError('Transfer history is not consecutive')
            if transition['trace_before'] != pt['trace_after']:
                raise ValueError('Fractional trace cursor reset between chunks')
            require_close(transition['elapsed_before_s'], pt['elapsed_after_s'], 'Episode clock reset')
            checked += 1
        raw = max(0, state['buffer_seconds'] - transition['download_s']) + selected['duration_s']
        require_close(transition['buffer_after_s'], min(cfg['max_buffer_s'], raw), 'Wrong buffer dynamics')
        require_close(transition['buffer_cap_wait_s'], max(0, raw - cfg['max_buffer_s']), 'Wrong buffer-cap wait')
        previous[key] = row
    return {'checked_transitions': checked, 'episodes': len(previous), 'passed': True}


def split_statistics(rows, actions, original_run):
    labels, modes, signatures, by_case = Counter(), Counter(), defaultdict(set), {}
    no_feasible = old_no_feasible = no_sustainable = old_no_sustainable = 0
    for row in rows:
        cid = row['state']['case_id']
        key = row['questions']['encoding']['label']
        a = actions[cid][key]
        labels[a['global_crf']] += 1
        modes[a['mode']] += 1
        signature = online_signature(row)
        signatures[signature].add(key)
        delays = row['_supervision']['candidate_actual_download_s']
        old_delays = [delay for k, delay in delays.items() if actions[cid][k]['source_encoding_run'] == original_run]
        no_feasible += min(delays.values()) > row['state']['buffer_seconds']
        old_no_feasible += min(old_delays) > row['state']['buffer_seconds']
        no_sustainable += min(delays.values()) > row['state']['chunk_duration_seconds']
        old_no_sustainable += min(old_delays) > row['state']['chunk_duration_seconds']
    for cid in actions:
        subset = [r for r in rows if r['state']['case_id'] == cid]
        counts = Counter(r['questions']['encoding']['label'] for r in subset)
        by_case[cid] = {'records': len(subset), 'selected_action_count': len(counts),
            'largest_label_fraction': max(counts.values()) / len(subset),
            'previous_bitrate_value_count': len({r['state']['previous_bitrate_kbps'] for r in subset}),
            'min_candidate_bitrate_kbps': min(a['actual_bitrate_kbps'] for a in actions[cid].values())}
    n = len(rows)
    buffers = sorted(r['state']['buffer_seconds'] for r in rows)
    return {'records': n, 'global_crf_labels': dict(labels), 'mode_labels': dict(modes),
        'no_nonstall_action_fraction': no_feasible / n,
        'old_candidates_no_nonstall_action_fraction_same_states': old_no_feasible / n,
        'no_sustainable_action_fraction': no_sustainable / n,
        'old_candidates_no_sustainable_action_fraction_same_states': old_no_sustainable / n,
        'buffer_s_p10_p50_p90': [buffers[round((n - 1) * p)] for p in (.1, .5, .9)],
        'unique_inputs': len(signatures), 'duplicate_rows': n - len(signatures),
        'conflicting_input_groups': sum(len(v) > 1 for v in signatures.values()), 'cases': by_case}, set(signatures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/semantic_kev_dev_v3')
    p.add_argument('--out', default='runs/semantic_kev_v3_data_audit_v2')
    a = p.parse_args()
    data, out = q.local_path(a.dataset), q.local_path(a.out)
    if out.exists():
        raise ValueError('Audit output already exists')
    manifest = verify_dataset(data)
    cfg = manifest['config']
    raw = json.loads((data / 'actions.json').read_text())
    actions = {cid: {a['action_id']: a for a in rows} for cid, rows in raw.items()}
    report = {'created_at': q.now(), 'dataset': a.dataset, 'training_started': False, 'api_calls': 0,
        'dataset_manifest_sha256': q.sha((data / 'manifest.json').read_bytes()), 'splits': {}, 'continuity': {}}
    signatures = {}
    for split in ('train', 'valid'):
        rows = [json.loads(l) for l in (data / f'{split}.jsonl').read_text().splitlines()]
        rollouts = [json.loads(l) for l in (data / f'{split}_rollouts.jsonl').read_text().splitlines()]
        report['continuity'][split] = audit_continuity(rollouts, actions, cfg)
        report['splits'][split], signatures[split] = split_statistics(rows, actions, cfg['encoding_runs'][0])
    traces = json.loads((data / 'traces.lock.json').read_text())
    groups = {s: {r['source_parent'] for r in traces if r['split'] == s} for s in ('train', 'valid')}
    clipped = sum(not math.isclose(a['offline_quality_proxy'], a['evidence_rgb_psnr_db'] / cfg['quality_scale_db'])
                  for rows in actions.values() for a in rows.values())
    if clipped:
        raise ValueError('Quality scores differ from the declared unclipped scale')
    report.update(network_source_group_overlap=sorted(groups['train'] & groups['valid']),
                  online_input_overlap_train_valid=len(signatures['train'] & signatures['valid']),
                  quality_clipped_actions=clipped, quality_definition=cfg['quality_proxy'],
                  same_video_cases_in_both_splits=sorted(actions))
    if report['network_source_group_overlap']:
        raise ValueError('Network acquisition group overlap')
    out.mkdir(parents=True)
    q.write_json(out / 'audit.json', report)
    lines = ['# v3 连续传输数据检查', '',
        '只读 CPU 检查；未启动训练或 API 调用。v2 和所有旧试训结果保留。', '',
        '| 划分 | 训练/验证记录 | 无及时传完候选 | 仅旧候选在相同状态下 | 缓冲 p10/中位/p90（秒） |',
        '|---|---:|---:|---:|---|']
    for split, s in report['splits'].items():
        lines.append(f"| {split} | {s['records']} | {s['no_nonstall_action_fraction']:.2%} | {s['old_candidates_no_nonstall_action_fraction_same_states']:.2%} | {s['buffer_s_p10_p50_p90']} |")
    lines += ['', '- 状态由行为策略连续下载得到；每步缓冲、上一档码率、过去已完成传输的带宽估计和小数时间游标均检查连续性。',
        '- 起播先用最小视频块建立观测；起播完整保留在 rollout 日志和闭环评估中，不作为模型训练/静态验证记录。',
        '- 质量为实测证据 RGB PSNR / 10，不封顶；卡顿每秒仍扣4.3，切档系数仍为0.05。权重是显式工程设定，不是语义问答得分。',
        '- 不强行平衡最优动作标签。不同策略共享起点的前几步可能产生相同在线状态，重复数和冲突组在 audit.json 中明确报告。',
        '- 新增低码率候选并不保证所有网络都能无卡顿；可持续传输覆盖率另行记录，不能靠放大缓冲证明网络适配已解决。',
        '- 这是假想播放列表中十个开发片段的连续传输，不是同一原视频的连续切块。网络来源分组分离；视频相同，不能声称新视频泛化。',
        '- 监督目标仍是一步动作的真实回放最优项，训练仍是监督模仿。闭环回放总收益用于验收，不把标签匹配率当问答准确率。',
        '', '[完整诊断](audit.json)。']
    (out / 'REVIEW.md').write_text('\n'.join(lines) + '\n')
    (out / 'code').mkdir()
    for filename in ('audit_semantic_rollout.py', 'evaluate_semantic_rollout.py', 'prepare_semantic_rollout.py', 'semantic_rollout.py', 'audit_semantic_data.py', 'qwen_roi.py', 'prepare_semantic_kev.py'):
        (out / 'code' / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    q.write_json(out / 'manifest.json', {'files_sha256': seal(out), 'dataset_manifest_sha256': report['dataset_manifest_sha256']})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
