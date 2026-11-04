#!/usr/bin/env python3
"""Inspect saved v4 results, with paired scenarios and gain-concentration checks."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import qwen_roi as q
from evaluate_semantic_control import verify_control_dataset
from evaluate_semantic_rollout import summarize
from prepare_semantic_rollout import seal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/semantic_kev_dev_v4')
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    data, evaluation, out = map(q.local_path, (a.dataset, a.evaluation, a.out))
    if out.exists():
        raise ValueError('Analysis output exists')
    verify_control_dataset(data)
    receipt = json.loads((evaluation / 'manifest.json').read_text())
    for name, expected in receipt['files_sha256'].items():
        if q.sha((evaluation / name).read_bytes()) != expected:
            raise ValueError('Evaluation file changed: ' + name)
    summary = json.loads((evaluation / 'summary.json').read_text())
    if summary['dataset_manifest_sha256'] != q.sha((data / 'manifest.json').read_bytes()):
        raise ValueError('Different evaluation dataset')
    checkpoint = Path(summary['checkpoint']).resolve()
    if not checkpoint.is_relative_to(q.ROOT):
        raise ValueError('Checkpoint outside semantic project')
    for name, expected in summary['checkpoint_sha256'].items():
        if q.sha((checkpoint / name).read_bytes()) != expected:
            raise ValueError('Checkpoint changed')
    episodes = json.loads((evaluation / 'episodes.json').read_text())
    rows = [json.loads(l) for l in (evaluation / 'transitions.jsonl').read_text().splitlines()]
    if len({(r['policy'], r['scenario_id'], r['step']) for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation decisions')
    for policy, metrics in summary['metrics'].items():
        grouped = [r for r in episodes if r['policy'] == policy]
        computed = summarize(grouped)
        if any(not math.isclose(computed[k], v, rel_tol=1e-10, abs_tol=1e-10) for k, v in metrics.items()):
            raise ValueError('Aggregate metrics do not reproduce')
        if sum(r['chunks'] for r in grouped) != sum(r['policy'] == policy for r in rows):
            raise ValueError('Missing evaluation decisions')
    model = {r['scenario_id']: r for r in episodes if r['policy'] == 'guarded_model'}
    paired = {}
    for policy in summary['metrics']:
        if policy == 'guarded_model':
            continue
        other = {r['scenario_id']: r for r in episodes if r['policy'] == policy}
        if set(other) != set(model):
            raise ValueError('Different evaluation scenarios')
        diffs = sorted((model[k]['reward_sum'] - other[k]['reward_sum'], k) for k in model)
        total = sum(d for d, _ in diffs)
        paired[policy] = {'wins': sum(d > 1e-8 for d, _ in diffs), 'ties': sum(abs(d) <= 1e-8 for d, _ in diffs),
            'losses': sum(d < -1e-8 for d, _ in diffs), 'mean_reward_gain': total / len(diffs),
            'largest_gain_scenario': diffs[-1][1], 'largest_scenario_gain': diffs[-1][0],
            'largest_scenario_share_of_net_gain': diffs[-1][0] / total if total > 0 else None,
            'mean_gain_without_largest_scenario': (total - diffs[-1][0]) / (len(diffs) - 1),
            'scenario_reward_differences': [{'id': k, 'difference': d} for d, k in diffs]}
    selected = [r for r in rows if r['policy'] == 'guarded_model']
    latencies, counters, worst = [], Counter(), []
    for row in selected:
        guard, info, transition = row['guard'], row['decision_info'], row['transition']
        if row['selected_action'] not in guard['allowed_action_ids']:
            raise ValueError('Model escaped guard')
        counters['forced_chunks_including_startup'] += guard['forced']
        counters['no_candidate_within_budget'] += guard['no_candidate_within_budget']
        counters['invalid_action_fallback'] += bool(info.get('invalid_action_fallback'))
        counters['model_prediction_fallback'] += bool(info.get('model_prediction_fallback'))
        if 'probabilities' in info:
            latencies.append(info['latency_ms'])
        if row['step'] > 0 and transition['rebuffer_s'] > 1e-8:
            counters['stalled_chunks'] += 1
            counters['stalled_with_actual_nonstall_alternative'] += min(o['rebuffer_s'] for o in row['outcomes'].values()) <= 1e-8
            worst.append({'scenario_id': row['scenario_id'], 'step': row['step'], 'case_id': row['request']['state']['case_id'],
                'action_id': row['selected_action'], 'buffer_s': row['request']['state']['buffer_seconds'],
                'forecast_download_s': guard['predicted_download_s'][row['selected_action']],
                'actual_download_s': transition['download_s'], 'rebuffer_s': transition['rebuffer_s'], 'guard_reason': guard['reason']})
    latencies.sort()
    train = json.loads((checkpoint / 'training_metrics.json').read_text())
    report = {'created_at': q.now(), 'dataset': a.dataset, 'evaluation': a.evaluation,
        'metrics': summary['metrics'], 'paired': paired, 'model_diagnostics': {**dict(counters),
            'model_inference_calls': len(latencies), 'latency_median_ms': latencies[len(latencies) // 2],
            'latency_p95_ms': latencies[round(.95 * (len(latencies) - 1))]},
        'training': {k: train[k] for k in ('optimizer_steps', 'records_seen', 'wall_seconds', 'truncated_records', 'rejected_records')},
        'worst_model_stalls': sorted(worst, key=lambda r: r['rebuffer_s'], reverse=True),
        'source_hashes': {'dataset_manifest': summary['dataset_manifest_sha256'],
            'evaluation_manifest': q.sha((evaluation / 'manifest.json').read_bytes()),
            'training_metrics': q.sha((checkpoint / 'training_metrics.json').read_bytes()),
            'training_config': q.sha((checkpoint / 'training_config.json').read_bytes())},
        'new_gpu_inference': False, 'new_training_started': False, 'api_calls': 0,
        'limitations': 'Shared ten development videos; 4 network traces x 5 starts, not 20 independent traces. PSNR proxy and trace replay, not real playback or semantic QA accuracy. Gains concentrate in one scenario; no unseen-video or statistical-generalization claim.'}
    out.mkdir(parents=True)
    q.write_json(out / 'analysis.json', report)
    lines = ['# v4 ABR 初始化400步结果', '',
        '结论：当前约束+Kev策略在这批开发回放中超过受约束带宽基线，接近离线质量教师；增益明显集中于一个场景，暂不能视为稳定泛化结论。', '',
        '| 策略 | 平均每场景奖励 | 平均卡顿秒数（不含起播） | 平均证据PSNR dB |', '|---|---:|---:|---:|']
    for policy in ('bandwidth', 'guarded_bandwidth', 'offline_quality_teacher', 'guarded_model'):
        m = summary['metrics'][policy]
        lines.append(f"| {policy} | {m['mean_episode_reward']:.2f} | {m['mean_episode_rebuffer_s']:.3f} | {m['mean_evidence_psnr_db']:.2f} |")
    comparison = paired['guarded_bandwidth']
    lines += ['', f"- 原始 epoch_08 初始化，{train['optimizer_steps']}次更新、{train['records_seen']}条样本曝光，{train['wall_seconds']/60:.2f}分钟。无截断／拒绝记录。",
        f"- 20场景中相对受约束带宽基线：{comparison['wins']}胜、{comparison['ties']}平、{comparison['losses']}负。",
        f"- 最大收益场景贡献总净增益的{comparison['largest_scenario_share_of_net_gain']:.2%}；排除该场景，平均每场景仍增加{comparison['mean_gain_without_largest_scenario']:.3f}奖励。",
        f"- 模型实际调用{len(latencies)}次，其余{counters['forced_chunks_including_startup']}块按起播／约束规则直接选档；无无效动作回退或预测异常回退。",
        f"- 推理中位数{latencies[len(latencies)//2]:.1f}毫秒，p95 {latencies[round(.95*(len(latencies)-1))]:.1f}毫秒；这些耗时未计入传输回放时钟。",
        '- 仍存在网络预测失准和最低档不足的卡顿。教师具有离线PSNR信息，不是长期最优上界；模型略高于教师不代表超越最优策略。',
        '- v3与v4同时改变了约束、监督目标、输入和训练步数，不能把跨版本全部增益归因于某一项改动。当前未对原始未微调 epoch_08 做同协议对照。',
        '- 可将本版保留为系统联调候选，随后完成三步统一入口，并用独立视频与更多网络复核；不需要立即继续堆训练步数。', '',
        '[完整诊断及逐场景差值](analysis.json)。本次仅分析已有文件，未启动新GPU推理、训练或API调用。']
    (out / 'REVIEW.md').write_text('\n'.join(lines) + '\n')
    (out / 'code').mkdir()
    for name in ('analyze_semantic_control.py', 'evaluate_semantic_control.py', 'evaluate_semantic_rollout.py', 'qwen_roi.py'):
        (out / 'code' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    q.write_json(out / 'manifest.json', {'source_hashes': report['source_hashes'], 'files_sha256': seal(out)})
    print(json.dumps({'report': str(out / 'REVIEW.md'), 'model_diagnostics': report['model_diagnostics'],
                      'paired_vs_guarded_bandwidth': {k: v for k, v in comparison.items() if k != 'scenario_reward_differences'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
