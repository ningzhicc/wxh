#!/usr/bin/env python3
"""Verify sealed source/ROI/encoding/receiver lineage and independently replay choices."""
import argparse
import json
import math
from pathlib import Path

import qwen_roi as q
from pipeline_media import file_sha, frame_table
from prepare_semantic_kev import read_trace
from run_semantic_pipeline import verify_files, seal_paths
from semantic_control import candidate_guard
from semantic_rollout import Cursor, advance_time, download, outcome


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError('Replay mismatch: ' + label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run, dest = q.local_path('runs/' + a.run_id), q.local_path(a.out)
    if dest.exists():
        raise ValueError('Audit output exists')
    verify_files(json.loads((run / 'files.sha256.json').read_text()))
    manifest = json.loads((run / 'manifest.json').read_text())
    fp = manifest['fingerprint']
    cfg = json.loads((run / 'config.json').read_text())
    control = json.loads((run / 'control_config.json').read_text())
    if cfg != fp['config'] or file_sha(Path(cfg['video'])) != fp['source_sha256']:
        raise ValueError('Source/config lineage changed')
    for name, expected in fp['code_sha256'].items():
        if file_sha(run / 'code' / name) != expected:
            raise ValueError('Code snapshot changed')
    for name in ('semantic_control.py', 'semantic_rollout.py'):
        if file_sha(Path(__file__).with_name(name)) != fp['code_sha256'][name]:
            raise ValueError('Use the original policy/physics version for this audit')
    if cfg['policy'] == 'kev':
        for name, expected in fp['checkpoint_sha256'].items():
            if file_sha(q.local_path(cfg['checkpoint']) / name) != expected:
                raise ValueError('Model checkpoint changed')
    chunks = json.loads((run / 'chunks.json').read_text())
    inputs = json.loads((run / 'roi_inputs.json').read_text())
    rois = {r['frame_id']: r for r in map(json.loads, (run / 'roi_results.jsonl').read_text().splitlines())}
    actions = json.loads((run / 'actions.json').read_text())
    source = frame_table(run / 'prepare/source_window.framemd5')
    reference = frame_table(run / 'prepare/reference.framemd5')
    if source['pixel_hashes'] != reference['pixel_hashes']:
        raise ValueError('Reference is not pixel-identical to normalized source window')
    if [h for c in chunks for h in c['source_pixel_hashes']] != reference['pixel_hashes']:
        raise ValueError('Source chunks are not a contiguous frame sequence')
    for row in inputs:
        result = rois[row['frame_id']]
        if file_sha(q.local_path(row['path'])) != row['sha256'] or result['image_sha256'] != row['sha256']:
            raise ValueError('ROI image provenance differs')
        for region in result['regions']:
            x1, y1, x2, y2 = region['bbox_pixels']
            if not 0 <= x1 < x2 <= row['width'] or not 0 <= y1 < y2 <= row['height']:
                raise ValueError('ROI outside image')
        if result['origin'] == 'verified_cache':
            old = result['source']
            d = q.local_path('runs/' + old['run']) / old.get('frame_relative_path', 'frames/' + old['frame_id'])
            if file_sha(d / 'response.json') != old['response_sha256']:
                raise ValueError('Original cached API response changed')
    for rows in actions.values():
        if len(rows) != 16:
            raise ValueError('Incomplete model preview grid')
        for action in rows:
            video = q.local_path(action['video_path'])
            if video.stat().st_size != action['size_bytes'] or file_sha(video) != action['video_sha256']:
                raise ValueError('Actual encoded candidate differs')
            close(action['actual_bitrate_kbps'], 8 * action['size_bytes'] / action['duration_s'] / 1000, 'measured kbps')
            if not action['roi_present'] and (action['roi_box_pixels'] is not None or action['roi_qoffset'] != '0' or action['mode'] != 'uniform'):
                raise ValueError('Empty evidence was incorrectly represented as an ROI')
    trace = read_trace(run / 'trace.txt')
    cursor = Cursor(min(len(trace) - 1, max(1, cfg['trace_start_segment'])), 0)
    buffer, previous, history, byte_sum = 0.0, 0.0, [], 0
    decisions = [json.loads(l) for l in (run / 'decisions.jsonl').read_text().splitlines()]
    for step, (chunk, decision) in enumerate(zip(chunks, decisions, strict=True)):
        state, transition = decision['request']['state'], decision['transition']
        expected_history = history or [control['startup_bandwidth_prior_mbps']]
        if state['recent_link_bandwidth_mbps'] != expected_history or state['question'] != cfg['question']:
            raise ValueError('Online observation history differs')
        close(state['buffer_seconds'], buffer, 'buffer before')
        close(state['previous_bitrate_kbps'], previous, 'previous rate')
        if decision['guard'] != candidate_guard(state, actions[chunk['case_id']], control)[1]:
            raise ValueError('Guard differs from saved online inputs')
        action = next(r for r in actions[chunk['case_id']] if r['action_id'] == decision['selected_action'])
        if action['action_id'] not in decision['guard']['allowed_action_ids']:
            raise ValueError('Selection outside admissible candidates')
        transfer = download(trace, cursor, action['size_bytes'], control['payload_fraction'], control['rtt_s'])
        expected = outcome(action, transfer['download_s'], state, control)
        close(transfer['download_s'], transition['download_s'], 'download')
        for name, value in expected.items():
            close(value, transition[name], name)
        cursor = advance_time(trace, transfer['cursor'], expected['buffer_cap_wait_s'])
        if cursor.__dict__ != transition['trace_after']:
            raise ValueError('Fractional trace cursor differs')
        received = decision['receiver']
        if file_sha(q.local_path(received['received_path'])) != action['video_sha256']:
            raise ValueError('Receiver bytes differ from selected file')
        if received['frame_info'] != action['frame_info']:
            raise ValueError('Receiver decoded frames differ')
        buffer = expected['buffer_after_s']
        previous = action['actual_bitrate_kbps']
        history = (history + [transfer['observed_link_mbps']])[-control['history_length']:]
        byte_sum += action['size_bytes']
    summary = json.loads((run / 'summary.json').read_text())
    if byte_sum != summary['selected_bytes'] or len(chunks) != summary['chunk_count']:
        raise ValueError('Summary totals differ')
    dest.mkdir(parents=True)
    report = {'passed': True, 'created_at': q.now(), 'pipeline_run': a.run_id,
        'source_frame_count': reference['frames'], 'chunks': len(chunks), 'roi_frames': len(inputs),
        'catalog_slots': sum(map(len, actions.values())), 'encoded_files': len({r['video_path'] for rows in actions.values() for r in rows}),
        'all_source_frames_contiguous': True, 'all_candidate_bytes_verified': True,
        'selected_receiver_files_verified': True, 'all_decisions_independently_replayed': True,
        'new_api_calls': 0, 'new_gpu_inference': False,
        'input_manifest_sha256': file_sha(run / 'manifest.json'),
        'input_files_seal_sha256': file_sha(run / 'files.sha256.json')}
    q.write_json(dest / 'audit.json', report)
    (dest / 'audit_semantic_pipeline.py').write_bytes(Path(__file__).read_bytes())
    q.write_json(dest / 'files.sha256.json', seal_paths([dest]))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
