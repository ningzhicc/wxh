#!/usr/bin/env python3
"""One command: continuous video -> ROI -> real encodings -> guarded Kev -> receiver.

Stages are sealed independently. Completed stages are verified and reused; an
ambiguous API attempt or partial media operation is never silently overwritten.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import qwen_roi as q
from pipeline_media import file_sha, prepare, encode_chunks
from pipeline_roi import locate
from pipeline_delivery import GPUUnavailable, deliver


CODE_FILES = ('run_semantic_pipeline.py', 'pipeline_media.py', 'pipeline_roi.py', 'pipeline_delivery.py',
    'qwen_roi.py', 'encode_roi_smoke.py', 'encode_roi_grid.py', 'semantic_control.py', 'semantic_rollout.py',
    'evaluate_semantic_rollout.py', 'prepare_semantic_kev.py')


def input_path(value):
    path = Path(value).expanduser()
    return (q.ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def seal_paths(paths):
    files = {}
    for path in paths:
        for p in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if p.is_file():
                files[str(p.relative_to(q.ROOT))] = file_sha(p)
    return files


def verify_files(files):
    for relative, expected in files.items():
        if file_sha(q.local_path(relative)) != expected:
            raise ValueError('Recorded file changed: ' + relative)


def resolve_config(args):
    cfg = json.loads(q.local_path(args.config).read_text())
    for option in ('video', 'question', 'start_s', 'duration_s', 'chunk_s', 'roi_mode', 'policy', 'trace', 'checkpoint', 'max_api_calls'):
        value = getattr(args, option, None)
        if value is not None:
            cfg[option] = value
    if args.video is not None:
        cfg.pop('expected_video_sha256', None)
    if args.evidence_window is not None:
        cfg['evidence_window_s'] = args.evidence_window
    if (not cfg['question'].strip() or not 0 < cfg['duration_s'] <= 1800 or cfg['start_s'] < 0
            or cfg['chunk_s'] <= 0 or not 0 <= cfg['max_api_calls'] <= 10000):
        raise ValueError('Invalid question, window, chunk duration or request cap')
    window = cfg.get('evidence_window_s')
    if window is not None and not 0 <= window[0] < window[1]:
        raise ValueError('Invalid explicit evidence window')
    cfg['video'] = str(input_path(cfg['video']))
    cfg['trace'] = str(input_path(cfg['trace']))
    return cfg


def execute(args):
    cfg = resolve_config(args)
    video, trace = Path(cfg['video']), Path(cfg['trace'])
    ffmpeg = q.ROOT / '.tools/ffmpeg'
    api_path = q.local_path(cfg['api_config'])
    api_config = json.loads(api_path.read_text())
    prompt = q.local_path(api_config['prompt']).read_bytes()
    control_path = q.local_path(cfg['control_config'])
    control = json.loads(control_path.read_text())
    video_sha = file_sha(video)
    if cfg.get('expected_video_sha256') and video_sha != cfg['expected_video_sha256']:
        raise ValueError('Pinned example source changed')
    checkpoint_hashes = {}
    if cfg['policy'] == 'kev':
        checkpoint = q.local_path(cfg['checkpoint'])
        for name, expected in cfg['checkpoint_sha256'].items():
            checkpoint_hashes[name] = file_sha(checkpoint / name)
            if checkpoint_hashes[name] != expected:
                raise ValueError('Pinned v4 model changed; explicitly configure a new model version')
    cache_hashes = {}
    if cfg.get('roi_cache_run') and cfg['roi_mode'] != 'api':
        cache = q.local_path('runs/' + cfg['roi_cache_run'])
        for name in ('manifest.json', 'inputs.json', 'results.jsonl'):
            cache_hashes[name] = file_sha(cache / name)
    code = {name: file_sha(Path(__file__).with_name(name)) for name in CODE_FILES}
    fingerprint = {'config': cfg, 'source_sha256': video_sha, 'trace_sha256': file_sha(trace),
        'ffmpeg_sha256': file_sha(ffmpeg), 'api_config_sha256': file_sha(api_path),
        'prompt_sha256': q.sha(prompt), 'control_config_sha256': file_sha(control_path),
        'checkpoint_sha256': checkpoint_hashes, 'roi_cache_sha256': cache_hashes, 'code_sha256': code}
    run, media = q.local_path('runs/' + args.run_id), q.local_path('artifacts/' + args.run_id)
    if run.exists():
        if json.loads((run / 'manifest.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Inputs/config/code/model changed; use a new run ID')
        if (run / 'files.sha256.json').exists():
            verify_files(json.loads((run / 'files.sha256.json').read_text()))
            print(json.dumps({'complete': True, 'verified_cached_run': str(run), 'new_api_calls': 0, 'new_gpu_inference': False}))
            return 0
    else:
        if media.exists():
            raise ValueError('Unrecorded media directory exists')
        run.mkdir()
        media.mkdir()
        (run / 'code').mkdir()
        for name in CODE_FILES:
            (run / 'code' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        q.write_json(run / 'manifest.json', {'kind': 'continuous_semantic_video_pipeline_v1', 'created_at': q.now(),
            'fingerprint': fingerprint, 'no_credentials_or_answers_recorded': True})
        q.write_json(run / 'config.json', cfg)
        (run / 'api_config.json').write_bytes(api_path.read_bytes())
        (run / 'roi_prompt.txt').write_bytes(prompt)
        (run / 'control_config.json').write_bytes(control_path.read_bytes())
        (run / 'trace.txt').write_bytes(trace.read_bytes())
    stages = ['prepare', 'roi', 'encode', 'deliver']
    limit = len(stages) if args.stage == 'all' else stages.index(args.stage) + 1
    try:
        for stage in stages[:limit]:
            receipt = run / 'stages' / (stage + '.json')
            if receipt.exists():
                verify_files(json.loads(receipt.read_text())['files_sha256'])
                continue
            q.write_json(run / 'status.json', {'state': 'running', 'stage': stage, 'updated_at': q.now()})
            if stage == 'prepare':
                if (run / 'prepare').exists():
                    raise ValueError('Partial preparation retained; use a new run ID')
                prepare(ffmpeg, video, media, run, cfg)
                paths = [run / 'prepare', run / 'chunks.json', run / 'roi_inputs.json', run / 'preparation.json', media / 'reference.mp4', media / 'chunks']
            elif stage == 'roi':
                inputs = json.loads((run / 'roi_inputs.json').read_text())
                locate(inputs, run, cfg, api_config, prompt)
                paths = [run / 'roi', run / 'roi_results.jsonl', run / 'roi_summary.json']
            elif stage == 'encode':
                if (run / 'encoding').exists() or (media / 'candidates').exists():
                    raise ValueError('Partial encoding retained; use a new run ID')
                chunks = json.loads((run / 'chunks.json').read_text())
                rois = [json.loads(l) for l in (run / 'roi_results.jsonl').read_text().splitlines()]
                encode_chunks(ffmpeg, chunks, rois, media, run, {**control, **cfg})
                paths = [run / 'encoding', run / 'actions.json', run / 'candidates.json', media / 'candidates']
            else:
                chunks = json.loads((run / 'chunks.json').read_text())
                actions = json.loads((run / 'actions.json').read_text())
                deliver(ffmpeg, chunks, actions, media, run, cfg, control)
                paths = [run / 'decisions.jsonl', run / 'receiver', run / 'selected_chunks.csv', run / 'summary.json', run / 'REPORT.md', media / 'received']
            q.write_json(receipt, {'complete': True, 'finished_at': q.now(), 'files_sha256': seal_paths(paths)})
        complete = limit == len(stages)
        q.write_json(run / 'status.json', {'state': 'complete' if complete else 'stage_complete', 'through': stages[limit - 1], 'updated_at': q.now()})
        if complete:
            q.write_json(run / 'files.sha256.json', seal_paths([run, media]))
        print(json.dumps({'complete': complete, 'through': stages[limit - 1], 'run': str(run), 'training_started': False}))
        return 0
    except Exception as error:
        waiting = isinstance(error, GPUUnavailable)
        q.write_json(run / 'status.json', {'state': 'waiting_gpu' if waiting else 'failed',
            'stage': stage, 'error_type': type(error).__name__, 'message': str(error), 'updated_at': q.now()})
        print(f'Pipeline stopped at {stage}: {error}', flush=True)
        return 3 if waiting else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pipeline_cached_dev01_v1.json')
    p.add_argument('--run-id', required=True)
    p.add_argument('--stage', choices=['all', 'prepare', 'roi', 'encode', 'deliver'], default='all', help='Run through this stage, reusing verified earlier stages')
    p.add_argument('--video')
    p.add_argument('--question')
    p.add_argument('--trace')
    p.add_argument('--checkpoint')
    p.add_argument('--policy', choices=['kev', 'bandwidth'])
    p.add_argument('--roi-mode', choices=['cache', 'cache_or_api', 'api'])
    p.add_argument('--start-s', type=float)
    p.add_argument('--duration-s', type=float)
    p.add_argument('--chunk-s', type=float)
    p.add_argument('--max-api-calls', type=int)
    p.add_argument('--evidence-window', nargs=2, type=float)
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        p.error('Invalid run ID')
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
