#!/usr/bin/env python3
"""CPU audit of v4 labels, guard parity, full rollout continuity and provenance."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import qwen_roi as q
from audit_semantic_data import online_signature
from evaluate_semantic_control import verify_control_dataset
from prepare_semantic_rollout import seal
from semantic_control import candidate_guard, plan_labels
from semantic_rollout import outcome


def close(a, b, message):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/semantic_kev_dev_v4')
    p.add_argument('--out', default='runs/semantic_kev_v4_audit_v1')
    a = p.parse_args()
    data, out = q.local_path(a.dataset), q.local_path(a.out)
    if out.exists():
        raise ValueError('Audit output exists')
    manifest = verify_control_dataset(data)
    cfg = manifest['config']
    actions = json.loads((data / 'actions.json').read_text())
    lookup = {cid: {r['action_id']: r for r in rows} for cid, rows in actions.items()}
    specs = {r['id']: r for r in map(json.loads, (data / 'episodes.jsonl').read_text().splitlines())}
    report = {'created_at': q.now(), 'dataset': a.dataset, 'splits': {}, 'training_started': False,
              'gpu_used': False, 'api_calls': 0, 'teacher_future_network_used': False,
              'dataset_manifest_sha256': q.sha((data / 'manifest.json').read_bytes())}
    signatures = {}
    for split in ('train', 'valid'):
        rows = [json.loads(l) for l in (data / f'{split}.jsonl').read_text().splitlines()]
        model_ids = {r['_meta']['id'] for r in rows}
        if len(model_ids) != len(rows):
            raise ValueError('Repeated record identifiers')
        rollouts = [json.loads(l) for l in (data / f'{split}_rollouts.jsonl').read_text().splitlines()]
        previous, eligible_ids, counters, option_counts, crfs, by_signature = {}, set(), Counter(), Counter(), Counter(), {}
        for row in rollouts:
            state, meta, supervision = row['state'], row['_meta'], row['_supervision']
            cid, step = state['case_id'], meta['step']
            spec = specs[meta['scenario_id']]
            allowed, guard = candidate_guard(state, actions[cid], cfg)
            if guard != meta['guard'] or set(guard['allowed_action_ids']) != set(row['questions']['encoding']['criteria']):
                raise ValueError('Online/collection guard mismatch')
            selected = meta['behavior_selected_action']
            if selected not in guard['allowed_action_ids']:
                raise ValueError('Behavior escaped guard')
            if guard['reason'] == 'within_budget':
                if any(guard['predicted_download_s'][a['action_id']] > guard['download_budget_s'] for a in allowed):
                    raise ValueError('Inadmissible candidate in request')
            catalogs = [actions[x] for x in spec['playlist'][step:step + cfg['planning_horizon_chunks']]]
            recomputed = plan_labels(state, catalogs, cfg)
            if recomputed != supervision['planning'] or recomputed['label'] != row['questions']['encoding']['label']:
                raise ValueError('Planning labels cannot be independently reproduced')
            if supervision['candidate_planned_return'] != recomputed['candidate_planned_return']:
                raise ValueError('Planned returns differ')
            if supervision['real_future_bandwidth_used_for_label']:
                raise ValueError('Unexpected future-network teacher')
            for action_id, action in lookup[cid].items():
                expected = outcome(action, supervision['candidate_actual_download_s'][action_id], state, cfg)
                close(expected['reward'], supervision['candidate_reward'][action_id], 'Actual physical reward changed')
            key = (meta['scenario_id'], meta['behavior_policy'])
            transition = meta['transition']
            if step == 0:
                if key in previous or state['playback_started'] or state['buffer_seconds'] != 0:
                    raise ValueError('Invalid startup')
            else:
                prev = previous[key]
                pm = prev['_meta']
                if pm['step'] != step - 1 or not state['playback_started']:
                    raise ValueError('Broken rollout ordering')
                close(state['buffer_seconds'], pm['transition']['buffer_after_s'], 'Buffer reset')
                close(state['previous_bitrate_kbps'], lookup[prev['state']['case_id']][pm['behavior_selected_action']]['actual_bitrate_kbps'], 'Previous rate reset')
                history = [pm['transition']['observed_link_mbps']] if step == 1 else (
                    prev['state']['recent_link_bandwidth_mbps'] + [pm['transition']['observed_link_mbps']])[-cfg['history_length']:]
                if history != state['recent_link_bandwidth_mbps']:
                    raise ValueError('Past transfer history is not consecutive')
                if transition['trace_before'] != pm['transition']['trace_after']:
                    raise ValueError('Trace cursor reset')
                close(transition['elapsed_before_s'], pm['transition']['elapsed_after_s'], 'Clock reset')
                counters['checked_state_transitions'] += 1
            expected = outcome(lookup[cid][selected], transition['download_s'], state, cfg)
            for field, value in expected.items():
                close(value, transition[field], 'Incorrect physical transition: ' + field)
            previous[key] = row
            counters['rollout_chunks'] += 1
            counters['forced_chunks'] += guard['forced']
            counters['no_candidate_within_budget'] += guard['no_candidate_within_budget']
            if step > 0 and not guard['forced']:
                eligible_ids.add(meta['id'])
            if meta['id'] in model_ids:
                option_counts[len(allowed)] += 1
                crfs[lookup[cid][recomputed['label']]['global_crf']] += 1
                sig = online_signature(row)
                by_signature.setdefault(sig, set()).add(recomputed['label'])
        if eligible_ids != model_ids or len(rows) != manifest['counts'][split]:
            raise ValueError('Training/validation record admission differs from runtime')
        signatures[split] = set(by_signature)
        report['splits'][split] = {**dict(counters), 'records': len(rows),
            'candidate_counts': dict(option_counts), 'label_crf_counts': dict(crfs),
            'unique_inputs': len(by_signature), 'duplicate_rows': len(rows) - len(by_signature),
            'conflicting_input_groups': sum(len(v) > 1 for v in by_signature.values()),
            'recomputed_all_planning_labels': True}
        print('Audited ' + split, flush=True)
    report['online_input_overlap_train_valid'] = len(signatures['train'] & signatures['valid'])
    traces = json.loads((data / 'traces.lock.json').read_text())
    groups = {s: {r['source_parent'] for r in traces if r['split'] == s} for s in ('train', 'valid')}
    report['network_group_overlap'] = sorted(groups['train'] & groups['valid'])
    if report['network_group_overlap']:
        raise ValueError('Network acquisition group overlap')
    out.mkdir(parents=True)
    q.write_json(out / 'audit.json', report)
    lines = ['# v4 候选约束与三块规划检查', '',
        'CPU 逐条重算全部 rollout 的候选约束、三块规划标签、实际物理奖励及连续状态。没有启动训练或模型推理。', '',
        '| 划分 | 模型决策记录 | 全部回放块 | 强制选档块 | 无候选满足预算块 |', '|---|---:|---:|---:|---:|']
    for split, s in report['splits'].items():
        lines.append(f"| {split} | {s['records']} | {s['rollout_chunks']} | {s['forced_chunks']} | {s['no_candidate_within_budget']} |")
    lines += ['', '- 训练记录仅保留真正需要模型选档的多候选状态；起播、单候选和无满足预算候选的回退均保留在完整日志。',
        '- 候选预算使用历史传输估计，不能保证真实未来网络无卡顿；回退不被记作满足预算。',
        '- 标签由历史带宽预测下的三块累计质量／卡顿／切档回报和末端缓冲不足惩罚产生，原来的实际未来回放奖励仅作独立诊断。',
        '- 教师使用离线实测证据 PSNR，模型输入没有 PSNR、答案或未来网络。未来两块的大小数组来自已知 VOD 编码目录。',
        '- 保留同一组十个开发视频和 v3 的网络来源分组／播放列表，未扩充 EVQA 或重新编码。',
        '', '[完整检查](audit.json)。']
    (out / 'REVIEW.md').write_text('\n'.join(lines) + '\n')
    (out / 'code').mkdir()
    for name in ('audit_semantic_control.py', 'semantic_control.py', 'semantic_rollout.py', 'evaluate_semantic_control.py', 'qwen_roi.py'):
        (out / 'code' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    q.write_json(out / 'manifest.json', {'dataset_manifest_sha256': report['dataset_manifest_sha256'], 'files_sha256': seal(out)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
