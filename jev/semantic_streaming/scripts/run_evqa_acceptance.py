#!/usr/bin/env python3
"""Five-video acceptance: same encodings, two policies and blinded receiver QA.

Does not train. Fresh-study caps: at most 90 ROI + 15 QA calls. Completed runs are
verified and reused; unknown paid-request outcomes never trigger automatic retry.
"""
import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys

import qwen_roi as q
from acceptance_qa import run_qa
from pipeline_media import file_sha
from prepare_evqa_acceptance import DATASET, prepare, verify_dataset
from run_semantic_pipeline import CODE_FILES, seal_paths, verify_files
from semantic_rollout import seed_for
import encode_roi_grid as grid


def pipeline_config(template, case):
    # Explicit construction: choices, reference answers and evidence labels are
    # never passed to either localization or the decision model.
    return {**template, 'video': case['video_path'], 'expected_video_sha256': case['video_sha256'],
        'question': case['question'], 'start_s': case['start_s'], 'duration_s': case['duration_s'],
        'chunk_s': 5, 'global_sample_times_s': None, 'roi_mode': 'api', 'roi_cache_run': None,
        'max_api_calls': 18, 'evidence_window_s': None, 'policy': 'bandwidth'}


def run_child(argv, log):
    with log.open('ab') as output:
        result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f'Child stopped with code {result.returncode}; inspect {log}', flush=True)
        raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--plan-only', action='store_true', help='Prepare/verify media and print plan; no API or GPU')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run_id):
        p.error('Invalid run ID')
    cases = prepare()
    dataset = q.local_path(DATASET)
    template = json.loads((q.ROOT / 'configs/pipeline_live_dev01_v1.json').read_text())
    for name, expected in template['checkpoint_sha256'].items():
        if file_sha(q.local_path(template['checkpoint']) / name) != expected:
            raise ValueError('Pinned v4 weights changed; acceptance not started')
    if a.plan_only:
        print(json.dumps({'cases': [{k: c[k] for k in ('case_id', 'question', 'duration_s', 'published_fps')} for c in cases],
            'max_ROI_calls': 90, 'max_QA_calls': 15, 'training_started': False, 'api_calls': 0, 'gpu_used': False}, ensure_ascii=False, indent=2))
        return 0
    run = q.local_path('runs/' + a.run_id)
    names = (*CODE_FILES, 'run_evqa_acceptance.py', 'prepare_evqa_acceptance.py', 'acceptance_qa.py', 'replay_semantic_pipeline.py')
    fingerprint = {'dataset_manifest_sha256': file_sha(dataset / 'manifest.json'), 'template': template,
        'code': {name: file_sha(Path(__file__).with_name(name)) for name in names}}
    if run.exists():
        if json.loads((run / 'manifest.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Acceptance inputs/config/code changed; use a fresh study ID')
        if (run / 'files.sha256.json').exists():
            verify_files(json.loads((run / 'files.sha256.json').read_text()))
            saved = json.loads((run / 'policy_results.json').read_text())
            for group in saved.values():
                for field in ('source_run', 'kev_run'):
                    child = q.local_path('runs/' + group[field])
                    if file_sha(child / 'files.sha256.json') != group[field + '_seal_sha256']:
                        raise ValueError('Child run seal changed')
                    verify_files(json.loads((child / 'files.sha256.json').read_text()))
            print('Verified completed acceptance study: ' + str(run / 'REPORT.md'))
            return 0
    else:
        run.mkdir()
        q.write_json(run / 'manifest.json', {'created_at': q.now(), 'kind': 'evqa_five_video_acceptance',
            'fingerprint': fingerprint, 'ROI_call_cap': 90, 'QA_call_cap': 15, 'training_started': False})
        (run / 'code').mkdir()
        for name in names:
            (run / 'code' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    reports, qa_plan = {}, []
    for case in cases:
        cid = case['case_id']
        prefix = a.run_id + '_' + cid
        base_id, kev_id = prefix + '_bandwidth', prefix + '_kev'
        config_path = run / (cid + '_pipeline.json')
        config = pipeline_config(template, case)
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Generated case config changed')
        else:
            q.write_json(config_path, config)
        print('Running ' + cid + ': ' + case['question'], flush=True)
        run_child([sys.executable, '-u', str(q.ROOT / 'scripts/run_semantic_pipeline.py'),
            '--config', str(config_path.relative_to(q.ROOT)), '--run-id', base_id], run / (cid + '_bandwidth.log'))
        run_child([sys.executable, '-u', str(q.ROOT / 'scripts/replay_semantic_pipeline.py'),
            '--source-run', base_id, '--run-id', kev_id, '--policy', 'kev'], run / (cid + '_kev.log'))
        base, kev = q.local_path('runs/' + base_id), q.local_path('runs/' + kev_id)
        if (base / 'actions.json').read_bytes() != (kev / 'actions.json').read_bytes():
            raise ValueError('Policies did not use the exact same candidate catalog')
        base_summary, kev_summary = (json.loads((path / 'summary.json').read_text()) for path in (base, kev))
        reports[cid] = {'bandwidth': base_summary, 'kev': kev_summary, 'source_run': base_id, 'kev_run': kev_id,
            'source_run_seal_sha256': file_sha(base / 'files.sha256.json'), 'kev_run_seal_sha256': file_sha(kev / 'files.sha256.json')}
        frames = base_summary['frames']
        fps = json.loads((base / 'chunks.json').read_text())[0]['fps']
        from fractions import Fraction
        count = min(12, frames)
        indices = [round(i * (frames - 1) / (count - 1)) for i in range(count)]
        choices = list(case['qa_choices'])
        random.Random(seed_for(666, cid, 'qa_options')).shuffle(choices)
        options = {chr(65 + i): text for i, text in enumerate(choices)}
        videos = {'reference': q.local_path('artifacts/' + base_id + '/reference.mp4'),
                  'bandwidth': q.local_path(base_summary['playback']['path']), 'kev': q.local_path(kev_summary['playback']['path'])}
        for variant, video in videos.items():
            samples = q.local_path('artifacts/' + a.run_id + '/qa_frames/' + cid + '/' + variant)
            if samples.exists():
                info = json.loads((samples / 'sampling.json').read_text())
                for frame in info['frames']:
                    if file_sha(q.local_path(frame['path'])) != frame['sha256']:
                        raise ValueError('Cached QA sample changed')
                candidate_frames = info['frames']
            else:
                samples.parent.mkdir(parents=True, exist_ok=True)
                decoded, images = grid.decode_samples(q.ROOT / '.tools/ffmpeg', video, samples, indices)
                if decoded['frame_count'] != frames or decoded['fps'] != fps:
                    raise ValueError('QA variants have different time grids')
                candidate_frames = [{'path': str(path.relative_to(q.ROOT)), 'sha256': file_sha(path),
                    'timestamp_s': float(index / Fraction(fps))} for index, path in zip(indices, images, strict=True)]
                q.write_json(samples / 'sampling.json', {'frames': candidate_frames, 'video_sha256': file_sha(video)})
            qa_plan.append({'case_id': cid, 'variant': variant, 'question': case['question'],
                'options': options, 'frames': candidate_frames})
    q.write_json(run / 'policy_results.json', reports)
    q.write_json(run / 'qa_plan.json', qa_plan)
    scores = run_qa(run, qa_plan, dataset / 'annotations/answers.jsonl', json.loads((q.ROOT / template['api_config']).read_text()))
    lookup = {(row['case_id'], row['variant']): row for row in scores}
    reference_correct = [c['case_id'] for c in cases if lookup[(c['case_id'], 'reference')]['correct']]
    summary = {'complete': True, 'case_count': 5, 'local_finetuning_video_sha_overlap': False,
        'same_real_candidate_files_for_both_policies': True,
        'reference_QA_correct': len(reference_correct),
        'QA_correct': {v: sum(lookup[(c['case_id'], v)]['correct'] for c in cases) for v in ('reference', 'bandwidth', 'kev')},
        'QA_retained_given_reference_correct': {v: (sum(lookup[(cid, v)]['correct'] for cid in reference_correct) / len(reference_correct) if reference_correct else None) for v in ('bandwidth', 'kev')},
        'policy_totals': {v: {field: sum(reports[c['case_id']][v][field] for c in cases)
            for field in ('selected_bytes', 'rebuffer_s', 'startup_delay_s', 'reward_sum')} for v in ('bandwidth', 'kev')},
        'ROI_calls_completed_in_study': sum(reports[c['case_id']]['bandwidth']['roi']['new_api_calls'] for c in cases),
        'QA_calls_completed_in_study': len(qa_plan), 'training_started': False,
        'limits': 'Five published 6fps clips, 2/4-choice QA, one fixed network trace, same Qwen provider for localization and QA; not full EVQA accuracy or proof of pretraining/source-family disjointness.'}
    q.write_json(run / 'summary.json', summary)
    lines = ['# 五视频端到端验收', '',
        '固定v4模型；两种策略共享同一套真实编码文件和同一网络起点，各自演化缓冲。未进行训练。', '',
        '| 视频 | 原视频QA | 带宽策略QA | Kev QA | 带宽字节 | Kev字节 | 带宽卡顿秒 | Kev卡顿秒 |', '|---|---|---|---|---:|---:|---:|---:|']
    for case in cases:
        cid = case['case_id']
        bw, kev = reports[cid]['bandwidth'], reports[cid]['kev']
        lines.append(f"| {cid} | {lookup[(cid,'reference')]['correct']} | {lookup[(cid,'bandwidth')]['correct']} | {lookup[(cid,'kev')]['correct']} | {bw['selected_bytes']} | {kev['selected_bytes']} | {bw['rebuffer_s']:.3f} | {kev['rebuffer_s']:.3f} |")
    lines += ['', '[总结果](summary.json)、[逐条问答与参考评分](qa_scores.json)、[策略记录](policy_results.json)。', '',
        '参考答案只在全部QA推理完成后用于评分。定位和Kev输入不包含参考答案、答案选项或GT时间段；QA接收问题和全部选项，不接收正确选项标签、编码方式、ROI或参考答案。',
        '发布媒体为6fps，网络为轨迹回放，本地真实文件复制和解码。五个样本不能代表完整基准，原视频答错的案例单独保留，条件保真率仅在原视频答对的子集上计算。']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    q.write_json(run / 'files.sha256.json', seal_paths([run, q.local_path('artifacts/' + a.run_id)]))
    print(json.dumps({'complete': True, 'report': str(run / 'REPORT.md'), 'training_started': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
