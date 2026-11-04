#!/usr/bin/env python3
"""Replay another policy on the exact encoded files of a completed pipeline."""
import argparse
import json
from pathlib import Path

import qwen_roi as q
from pipeline_delivery import GPUUnavailable, deliver
from pipeline_media import file_sha
from run_semantic_pipeline import CODE_FILES, seal_paths, verify_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--policy', choices=['kev', 'bandwidth'], default='kev')
    a = p.parse_args()
    source, run, media = (q.local_path('runs/' + a.source_run), q.local_path('runs/' + a.run_id), q.local_path('artifacts/' + a.run_id))
    verify_files(json.loads((source / 'files.sha256.json').read_text()))
    parent = json.loads((source / 'manifest.json').read_text())
    for name in CODE_FILES:
        if file_sha(Path(__file__).with_name(name)) != parent['fingerprint']['code_sha256'][name]:
            raise ValueError('Policy/media implementation differs from the source run; freeze one version for comparison')
    cfg = {**json.loads((source / 'config.json').read_text()), 'policy': a.policy}
    if a.policy == 'kev':
        for name, expected in cfg['checkpoint_sha256'].items():
            if file_sha(q.local_path(cfg['checkpoint']) / name) != expected:
                raise ValueError('Pinned model changed')
    fingerprint = {'source_manifest_sha256': file_sha(source / 'manifest.json'),
        'source_files_seal_sha256': file_sha(source / 'files.sha256.json'), 'policy': a.policy,
        'script_sha256': file_sha(Path(__file__)), 'config': cfg}
    if run.exists():
        if json.loads((run / 'manifest.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Replay inputs changed')
        if (run / 'files.sha256.json').exists():
            verify_files(json.loads((run / 'files.sha256.json').read_text()))
            print('Verified completed policy replay ' + a.run_id)
            return 0
    else:
        if media.exists():
            raise ValueError('Unrecorded replay media exists')
        run.mkdir()
        media.mkdir()
        q.write_json(run / 'manifest.json', {'kind': 'shared_catalog_policy_replay', 'fingerprint': fingerprint,
            'source_run': a.source_run, 'created_at': q.now()})
        q.write_json(run / 'config.json', cfg)
        for name in ('trace.txt', 'control_config.json', 'chunks.json', 'actions.json'):
            (run / name).write_bytes((source / name).read_bytes())
        roi = json.loads((source / 'roi_summary.json').read_text())
        q.write_json(run / 'roi_summary.json', {**roi, 'new_api_calls': 0, 'cached_frames': roi['frames'],
            'usage': {}, 'inherited_from': a.source_run})
        (run / 'code').mkdir()
        for name in (*CODE_FILES, 'replay_semantic_pipeline.py'):
            (run / 'code' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    try:
        summary = deliver(q.ROOT / '.tools/ffmpeg', json.loads((run / 'chunks.json').read_text()),
            json.loads((run / 'actions.json').read_text()), media, run, cfg,
            json.loads((run / 'control_config.json').read_text()))
        q.write_json(run / 'summary.json', {**summary, 'reused_encodings_from': a.source_run,
            'new_encoding_files': 0, 'localization_api_calls': 0})
        with (run / 'REPORT.md').open('a') as report:
            report.write('\n本运行复用 ' + a.source_run + ' 的同一套编码文件，未重新定位或编码。\n')
        q.write_json(run / 'status.json', {'state': 'complete'})
        q.write_json(run / 'files.sha256.json', seal_paths([run, media]))
        return 0
    except GPUUnavailable:
        q.write_json(run / 'status.json', {'state': 'waiting_gpu'})
        print('GPU busy; completed source catalog preserved. Resume with the same command.')
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
