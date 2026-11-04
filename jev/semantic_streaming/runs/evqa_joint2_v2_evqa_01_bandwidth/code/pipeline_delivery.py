"""Run the pinned v4 decision policy and verify locally delivered encoded bytes."""
import csv
import json
import subprocess

import qwen_roi as q
from evaluate_semantic_rollout import model_policy
from pipeline_media import assemble, receive
from prepare_semantic_kev import read_trace
from semantic_control import guarded_behavior, model_with_fallback, run_control_episode


class GPUUnavailable(RuntimeError):
    pass


def deliver(ffmpeg, chunks, actions, media, run, cfg, control):
    if (media / 'received').exists() or (run / 'decisions.jsonl').exists():
        raise ValueError('Partial delivery is preserved; use a new run rather than overwrite it')
    trace = read_trace(run / 'trace.txt')
    spec = {'start_segment': min(len(trace) - 1, max(1, cfg['trace_start_segment'])),
        'start_offset_s': 0, 'playlist': [c['case_id'] for c in chunks], 'request_seed': cfg['seed']}
    questions = {c['case_id']: cfg['question'] for c in chunks}
    environment = None
    if cfg['policy'] == 'kev':
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True)
        free, util = map(int, gpu.stdout.splitlines()[0].split(','))
        if free < cfg['inference_min_free_mib'] or util > cfg['inference_max_utilization']:
            raise GPUUnavailable('CPU stages complete; resume delivery when GPU resources are available')
        choose, environment = model_policy(q.local_path(cfg['checkpoint']), actions)
        choose = model_with_fallback(choose, actions)
    else:
        choose = guarded_behavior('bandwidth', actions, control, cfg['seed'])
    original = choose

    def choose(row):
        # Empty-ROI chunks preserve the grid slots but apply ordinary encoding.
        row['questions']['encoding']['instructions'] += ' Empty-ROI chunks use ordinary encoding in every ROI offset slot.'
        return original(row)

    receivers, records, table = [], [], []
    totals = {'selected_bytes': 0, 'rebuffer_s': 0.0, 'startup_delay_s': 0.0, 'reward_sum': 0.0,
              'model_calls': 0, 'forced_chunks': 0, 'prediction_fallbacks': 0}
    with (run / 'decisions.jsonl').open('w') as output:
        for chunk, record in zip(chunks, run_control_episode(trace, spec, actions, questions, control, choose), strict=True):
            selected = next(a for a in actions[chunk['case_id']] if a['action_id'] == record['selected_action'])
            receiver = receive(ffmpeg, selected, chunk, media, run)
            receivers.append(receiver)
            transition = record['transition']
            totals['selected_bytes'] += selected['size_bytes']
            totals['rebuffer_s'] += transition['rebuffer_s']
            totals['startup_delay_s'] += transition['startup_delay_s']
            totals['reward_sum'] += transition['reward']
            totals['model_calls'] += 'probabilities' in record['decision_info']
            totals['forced_chunks'] += record['guard']['forced']
            totals['prediction_fallbacks'] += bool(record['decision_info'].get('model_prediction_fallback'))
            row = {'case_id': chunk['case_id'], 'start_window_s': chunk['start_window_s'],
                'duration_s': chunk['duration_s'], 'roi_box_pixels': selected['roi_box_pixels'],
                'action_id': selected['action_id'], 'global_crf': selected['global_crf'],
                'roi_qoffset': selected['roi_qoffset'], 'size_bytes': selected['size_bytes'],
                'actual_bitrate_kbps': selected['actual_bitrate_kbps'], 'video_path': selected['video_path'],
                'received_path': receiver['received_path'], 'quality_basis': selected['quality_basis'],
                'download_s': transition['download_s'], 'rebuffer_s': transition['rebuffer_s'],
                'buffer_before_s': record['request']['state']['buffer_seconds'],
                'buffer_after_s': transition['buffer_after_s'],
                'guard_reason': record['guard']['reason']}
            table.append(row)
            records.append(record)
            output.write(json.dumps({'selection': row, 'receiver': receiver, **record}, ensure_ascii=False) + '\n')
            output.flush()
            print(f"Delivered {chunk['case_id']}: {selected['size_bytes']} bytes, {selected['actual_bitrate_kbps']:.2f} kbps", flush=True)
    playback = assemble(ffmpeg, chunks, receivers, media, run)
    with (run / 'selected_chunks.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    roi_summary = json.loads((run / 'roi_summary.json').read_text())
    summary = {'complete': True, 'created_at': q.now(), 'question': cfg['question'],
        'policy': cfg['policy'], 'chunk_count': len(chunks), 'frames': sum(c['frames'] for c in chunks),
        'duration_s': sum(c['duration_s'] for c in chunks), 'catalog_slots': sum(len(r) for r in actions.values()),
        'encoded_files': len({a['video_path'] for rows in actions.values() for a in rows}),
        'roi': roi_summary, **totals, 'playback': playback,
        'receiver_all_frames_verified': True, 'continuous_source_partition_verified': True,
        'gpu_inference_used': cfg['policy'] == 'kev', 'model_environment': environment,
        'training_started': False, 'network_mode': 'trace_replay_with_verified_local_file_delivery',
        'inference_latency_in_replay_clock': False,
        'limitations': ['Offline VOD preparation; all candidate sizes and next-two-chunk catalogs are available before selection.',
            'No socket/network throughput measurement; transport time is trace replay and received bytes are real local copies.',
            'Per-chunk ROI is a static union of sampled API boxes, not object tracking.',
            'No QA scoring or semantic correctness claim; empty-ROI quality uses full-frame PSNR.']}
    q.write_json(run / 'summary.json', summary)
    lines = ['# 连续视频三步系统运行报告', '',
        f"问题：{cfg['question']}", '',
        f"策略：{cfg['policy']}；{len(chunks)}个连续块，{summary['frames']}帧，{summary['duration_s']:.3f}秒。",
        f"定位：缓存{roi_summary['cached_frames']}帧、新API调用{roi_summary['new_api_calls']}次；编码{summary['encoded_files']}个真实文件，{summary['catalog_slots']}个目录项。", '',
        '| 块 | 起点秒 | ROI | CRF / 偏移 | 字节 | kbps | 回放下载秒 | 卡顿秒 | 下载后缓冲秒 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['case_id']} | {r['start_window_s']:.3f} | {r['roi_box_pixels']} | {r['global_crf']} / {r['roi_qoffset']} | {r['size_bytes']} | {r['actual_bitrate_kbps']:.2f} | {r['download_s']:.3f} | {r['rebuffer_s']:.3f} | {r['buffer_after_s']:.3f} |")
    relative_playback = '../../' + playback['path']
    lines += ['', f'[接收端拼接视频]({relative_playback})、[逐块选择CSV](selected_chunks.csv)、[完整决策与接收记录](decisions.jsonl)。', '',
        '源窗口与每一块逐帧核验，所选MP4真实复制到接收目录并完整解码，拼接后再次逐帧核验。码率来自完整MP4字节，含容器开销。',
        '未定位到区域时普通编码；为保持模型的16槽位预览格式，相同编码使用显式别名，不虚构ROI或重复计作新编码文件。',
        '这是离线VOD的三步联调：网络耗时来自轨迹回放，接收文件是真实字节复制；不是实际限速网络播放实验。模型推理耗时不进入回放时钟。',
        'API密钥、鉴权头和参考答案均不进入运行记录或模型输入。ROI为逐块静态采样并集，跨帧跟踪和自然语言时间约束推断未实现。']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary
