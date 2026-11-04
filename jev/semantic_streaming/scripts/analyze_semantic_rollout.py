#!/usr/bin/env python3
"""Read existing closed-loop outputs and produce a traceable CPU-only diagnosis."""
import argparse
from collections import Counter
import json
from pathlib import Path

import qwen_roi as q


def verified_manifest(directory, filename, field):
    manifest = json.loads((directory / filename).read_text())
    for name, expected in manifest[field].items():
        if q.sha((directory / name).read_bytes()) != expected:
            raise ValueError('Recorded file changed: ' + str(directory / name))
    return manifest


def estimate(action, state, cfg):
    observed = [x for x in state['recent_link_bandwidth_mbps'] if x > 0]
    harmonic = len(observed) / sum(1 / x for x in observed)
    return cfg['rtt_s'] + action['size_bytes'] * 8 / (harmonic * 1e6 * cfg['payload_fraction'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/semantic_kev_dev_v3')
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    data, evaluation, out = map(q.local_path, (a.dataset, a.evaluation, a.out))
    if out.exists():
        raise ValueError('Analysis output exists; use a new path')
    manifest = verified_manifest(data, 'manifest.json', 'files_sha256')
    verified_manifest(evaluation, 'manifest.json', 'files_sha256')
    summary = json.loads((evaluation / 'summary.json').read_text())
    if summary['dataset_manifest_sha256'] != q.sha((data / 'manifest.json').read_bytes()):
        raise ValueError('Evaluation used a different dataset')
    checkpoint = Path(summary['checkpoint']).resolve()
    if not checkpoint.is_relative_to(q.ROOT):
        raise ValueError('Checkpoint is outside semantic project')
    for name, expected in summary['checkpoint_sha256'].items():
        if q.sha((checkpoint / name).read_bytes()) != expected:
            raise ValueError('Evaluation checkpoint changed')
    cfg = manifest['config']
    actions = {cid: {r['action_id']: r for r in rows} for cid, rows in json.loads((data / 'actions.json').read_text()).items()}
    rows = [json.loads(l) for l in (evaluation / 'transitions.jsonl').read_text().splitlines()]
    episodes = json.loads((evaluation / 'episodes.json').read_text())
    diagnostics, worst = {}, []
    for policy in summary['metrics']:
        selected_rows = [r for r in rows if r['policy'] == policy and r['step'] > 0]
        stats = Counter()
        crfs, offsets = Counter(), Counter()
        buffers = []
        for r in selected_rows:
            state = r['request']['state']
            cid = state['case_id']
            action = actions[cid][r['selected_action']]
            transition = r['transition']
            crfs[action['global_crf']] += 1
            offsets[action['roi_qoffset']] += 1
            buffers.append(state['buffer_seconds'])
            best = max(r['outcomes'], key=lambda k: r['outcomes'][k]['reward'])
            gap = r['outcomes'][best]['reward'] - transition['reward']
            stats['one_step_optimal_choices'] += gap < 1e-8
            if transition['rebuffer_s'] > 1e-8:
                stats['stalled_choices'] += 1
                stats['stalled_choices_with_actual_nonstall_alternative'] += min(o['rebuffer_s'] for o in r['outcomes'].values()) < 1e-8
            estimates = {key: estimate(candidate, state, cfg) for key, candidate in actions[cid].items()}
            stats['estimated_unsafe_choices_with_estimated_safe_alternative'] += (
                estimates[action['action_id']] > state['buffer_seconds']
                and min(estimates.values()) <= state['buffer_seconds'])
            stats['actual_download_exceeds_chunk_duration'] += transition['download_s'] > state['chunk_duration_seconds']
            if policy == 'model':
                worst.append({'scenario_id': r['scenario_id'], 'step': r['step'], 'case_id': cid,
                    'buffer_s': state['buffer_seconds'], 'selected_candidate': action['candidate_id'],
                    'estimated_download_s': estimates[action['action_id']], 'actual_download_s': transition['download_s'],
                    'rebuffer_s': transition['rebuffer_s'], 'best_one_step_candidate': actions[cid][best]['candidate_id'],
                    'best_one_step_download_s': r['downloads_s'][best], 'one_step_regret': gap,
                    'probability': r['decision_info']['probabilities'][action['action_id']]})
        buffers.sort()
        diagnostics[policy] = {'decision_count_excluding_common_startup': len(selected_rows), **dict(stats),
            'global_crf_counts': dict(crfs), 'roi_offset_counts': dict(offsets),
            'buffer_p10_p50_p90_s': [buffers[round((len(buffers) - 1) * p)] for p in (.1, .5, .9)]}
    train = [json.loads(l) for l in (data / 'train.jsonl').read_text().splitlines()]
    slower_than_playback = sum(r['_supervision']['candidate_actual_download_s'][r['questions']['encoding']['label']]
                               > r['state']['chunk_duration_seconds'] for r in train)
    training = json.loads((checkpoint / 'training_metrics.json').read_text())
    paired = {}
    model_episodes = {r['scenario_id']: r for r in episodes if r['policy'] == 'model'}
    for baseline in ('cheapest', 'bandwidth'):
        other = {r['scenario_id']: r for r in episodes if r['policy'] == baseline}
        if set(other) != set(model_episodes):
            raise ValueError('Different evaluation scenarios')
        differences = [model_episodes[k]['reward_sum'] - other[k]['reward_sum'] for k in other]
        paired[baseline] = {'model_wins': sum(d > 1e-8 for d in differences),
                            'model_losses': sum(d < -1e-8 for d in differences),
                            'mean_episode_reward_difference': sum(differences) / len(differences)}
    report = {'created_at': q.now(), 'evaluation': a.evaluation, 'dataset': a.dataset,
        'metrics': summary['metrics'], 'diagnostics': diagnostics, 'paired_scenarios': paired,
        'training': {k: training[k] for k in ('optimizer_steps', 'records_seen', 'wall_seconds', 'truncated_records', 'rejected_records', 'grad_norm')},
        'train_record_count': len(train), 'train_labels_download_slower_than_playback': slower_than_playback,
        'train_labels_download_slower_than_playback_fraction': slower_than_playback / len(train),
        'worst_model_choices': sorted(worst, key=lambda r: r['one_step_regret'], reverse=True)[:20],
        'limitations': ['Only 100 optimizer updates / 800 record exposures; no convergence claim.',
            '20 scenarios are 5 starts on each of 4 network traces; no claim of 20 independent traces.',
            'Counterfactual avoidability uses privileged future replay and is diagnostic, not online knowledge.',
            'A single slower-than-playback download can legitimately consume buffer; repeated choices can deplete it.',
            'No v3 pre-training checkpoint evaluation; cannot attribute the aggregate result solely to fine-tuning.',
            'Synthetic playlist of shared development videos; PSNR is not semantic QA accuracy.'],
        'gpu_used_for_analysis': False, 'new_training_started': False, 'api_calls': 0,
        'source_hashes': {'dataset_manifest': q.sha((data / 'manifest.json').read_bytes()),
            'evaluation_manifest': q.sha((evaluation / 'manifest.json').read_bytes()),
            'training_metrics': q.sha((checkpoint / 'training_metrics.json').read_bytes()),
            'training_config': q.sha((checkpoint / 'training_config.json').read_bytes())}}
    out.mkdir(parents=True)
    q.write_json(out / 'analysis.json', report)
    lines = ['# v3 ABR 初始化100步试训分析', '',
        '结论：训练与闭环评估完整结束；当前 Kev 的平均综合收益低于两种简单策略。画质较高，但下载过大、卡顿偏多。', '',
        '| 策略 | 平均每场景奖励（越高越好） | 平均每场景卡顿秒（不含起播） | 平均证据 PSNR dB |',
        '|---|---:|---:|---:|']
    for name, m in summary['metrics'].items():
        lines.append(f"| {name} | {m['mean_episode_reward']:.2f} | {m['mean_episode_rebuffer_s']:.3f} | {m['mean_evidence_psnr_db']:.2f} |")
    model = diagnostics['model']
    lines += ['', f"- 从原始 merina656_baseline_seed666/checkpoints/epoch_08 载入，100次更新、800条记录曝光，耗时{training['wall_seconds']/60:.2f}分钟。无拒绝／截断记录。尚未完成一轮4560条训练数据。",
        '- 验证为4条网络轨迹×5起点，20个场景×20片段；所有策略起播规则相同。新增模型推理380次，起播20次不由模型选档。',
        f"- 模型有{model['stalled_choices']}次卡顿选择，其中{model['stalled_choices_with_actual_nonstall_alternative']}次在当时同一游标下存在实际不会卡顿的候选；该判断使用离线未来回放。",
        f"- 另有{model['estimated_unsafe_choices_with_estimated_safe_alternative']}次选择，按已知历史带宽估计就无法在当前缓冲内完成，而存在估计可以及时完成的其他候选。",
        f"- 模型相对带宽策略在20场景中赢{paired['bandwidth']['model_wins']}场、输{paired['bandwidth']['model_losses']}场，平均回报差{paired['bandwidth']['mean_episode_reward_difference']:.2f}。",
        f"- 模型决策时的缓冲中位数{model['buffer_p10_p50_p90_s'][1]:.2f}秒，带宽策略为{diagnostics['bandwidth']['buffer_p10_p50_p90_s'][1]:.2f}秒。与更激进选择、后续缓冲减少的现象一致，不能仅据此认定单一因果。",
        f"- 训练标签仍是一步真实回放最优：{slower_than_playback}/{len(train)}（{slower_than_playback/len(train):.2%}）所选下载耗时超过该片段播放时长。单次可用缓冲容许，但一步目标没有显式计入反复消耗缓冲的后续代价。", '',
        '建议先加入基于历史带宽和可持续传输预算的候选约束，并改进考虑后续缓冲的离线监督目标；再考虑对严重决策损失加权。随后在统一闭环条件下检查多个训练步数的检查点，而非只看分类 loss 或直接延长当前训练。', '',
        '本次分析仅使用已有文件，未启动新训练／GPU推理／API调用。没有未微调 ABR 初始化模型在 v3 上的对照，不把问题归因于 ABR 初始化本身。新旧 v2/v3 奖励尺度不同，不直接跨版本比较奖励。', '',
        '[完整诊断及最差决策](analysis.json)。']
    (out / 'REVIEW.md').write_text('\n'.join(lines) + '\n')
    (out / 'code').mkdir()
    for path in (Path(__file__), Path(q.__file__)):
        (out / 'code' / path.name).write_bytes(path.read_bytes())
    q.write_json(out / 'manifest.json', {'source_hashes': report['source_hashes'],
        'files_sha256': {str(p.relative_to(out)): q.sha(p.read_bytes()) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'report': str(out / 'REVIEW.md'), 'diagnostics': diagnostics['model'], 'paired': paired}, ensure_ascii=False))


if __name__ == '__main__':
    main()
