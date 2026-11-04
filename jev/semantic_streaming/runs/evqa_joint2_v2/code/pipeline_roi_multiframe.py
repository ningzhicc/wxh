"""Versioned offline-VOD joint localization; labels never enter API payloads."""
import json
import time
import urllib.error
import urllib.request

import qwen_roi as q
from pipeline_media import file_sha

MAX_FRAMES = 24


def make_payload(config, prompt, inputs):
    if not inputs or len(inputs) > MAX_FRAMES:
        raise ValueError('Joint localization requires 1..24 frames; use explicit bounded windows')
    if len({r['frame_id'] for r in inputs}) != len(inputs):
        raise ValueError('Duplicate input frame identity')
    if len({r['question'] for r in inputs}) != 1:
        raise ValueError('Joint frames must share one question')
    times = [r['timestamp_s'] for r in inputs]
    if times != sorted(times):
        raise ValueError('Joint frames must be chronological')
    content = [{'type': 'text', 'text': json.dumps({'question': inputs[0]['question'],
        'frame_count': len(inputs)}, ensure_ascii=False)}]
    for row in inputs:
        # Existing allowlisted constructor verifies the exact original PNG.
        single = q.make_payload(config, prompt, row)['messages'][1]['content']
        content += [single[1], single[0]]
    return {'model': config['model'], **config['request_parameters'], 'messages': [
        {'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}]}


def parse_regions(response, inputs, denominator=1000):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Joint response truncated or incomplete')
    value = json.loads(choice['message']['content'])
    if not isinstance(value, dict) or set(value) != {'frames'} or not isinstance(value['frames'], list):
        raise ValueError('Expected exactly one frames list')
    rows = {r['frame_id']: r for r in inputs}
    if len(rows) != len(inputs) or len(value['frames']) != len(rows):
        raise ValueError('Joint output must cover every input frame exactly once')
    parsed = {}
    for item in value['frames']:
        if not isinstance(item, dict) or not isinstance(item.get('frame_id'), str):
            raise ValueError('Invalid joint frame entry')
        fid = item['frame_id']
        if fid not in rows or fid in parsed:
            raise ValueError('Unknown or duplicate output frame identity')
        one = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(item)}}]}
        parsed[fid] = q.parse_regions(one, rows[fid], denominator)
        parsed[fid]['semantics'] = 'joint full-window context; visible plausible question evidence, no GT target disambiguation'
    return [parsed[r['frame_id']] for r in inputs]


def locate(inputs, run, cfg, api_config, prompt):
    if cfg['roi_mode'] != 'api' or cfg.get('roi_cache_run') or cfg.get('evidence_window_s') is not None:
        raise ValueError('Joint v2 uses API with no single-frame cache or GT time filter')
    q.validate_endpoint(api_config['base_url'])
    payload = make_payload(api_config, prompt.decode().strip(), inputs)
    wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    if cfg['max_api_calls'] < 1:
        raise ValueError('Joint request exceeds API cap')
    directory = run / 'roi'
    batch = directory / 'joint_0000'
    status_path = batch / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status['state'] != 'complete':
            raise ValueError('Ambiguous/failed joint API attempt retained; no automatic retry')
        request = json.loads((batch / 'request.json').read_text())
        if request['payload_sha256'] != q.sha(wire) or request['inputs'] != inputs:
            raise ValueError('Resumed joint context changed')
        if file_sha(batch / 'response.json') != status['response_sha256']:
            raise ValueError('Joint raw response changed')
        response = json.loads((batch / 'response.json').read_text())
        parsed = parse_regions(response, inputs, api_config['coordinate_contract']['scale_denominator'])
        elapsed = status['elapsed_s']
    else:
        batch.mkdir(parents=True)
        key = q.load_secret()
        q.write_json(batch / 'request.json', {'inputs': inputs, 'payload_sha256': q.sha(wire),
            'payload_bytes': len(wire), 'prompt_sha256': q.sha(prompt), 'model': api_config['model'],
            'offline_full_window_context': True, 'note': 'No headers/key, GT masks, answers, options or evidence times in wire payload.'})
        q.write_json(status_path, {'state': 'in_flight', 'started_at': q.now()})
        request = urllib.request.Request(api_config['base_url'].rstrip('/') + '/chat/completions', data=wire,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        try:
            with urllib.request.build_opener(q.NoRedirect()).open(request, timeout=120) as handle:
                raw = handle.read().replace(key.encode(), b'[REDACTED]')
            (batch / 'response.json').write_bytes(raw)
            response = json.loads(raw)
            parsed = parse_regions(response, inputs, api_config['coordinate_contract']['scale_denominator'])
            elapsed = time.monotonic() - started
            q.write_json(status_path, {'state': 'complete', 'response_sha256': q.sha(raw),
                'elapsed_s': elapsed, 'finished_at': q.now()})
        except Exception as error:
            failure = {'state': 'failed_no_retry', 'error_type': type(error).__name__, 'finished_at': q.now()}
            if isinstance(error, urllib.error.HTTPError):
                failure['http_status'] = error.code
                (batch / 'error_response.txt').write_bytes(error.read().replace(key.encode(), b'[REDACTED]'))
            q.write_json(status_path, failure)
            raise RuntimeError('Joint ROI request failed; retained without automatic retry') from None
    results = []
    for row, item in zip(inputs, parsed, strict=True):
        result = {**item, 'origin': 'new_API_multiframe', 'actual_model': response.get('model'),
            'response_sha256': file_sha(batch / 'response.json'), 'batch_relative_path': 'roi/joint_0000',
            'elapsed_s': elapsed, 'joint_context_frames': len(inputs)}
        frame = directory / row['frame_id']
        target = frame / 'result.json'
        if target.exists() and json.loads(target.read_text()) != result:
            raise ValueError('Resumed per-frame joint ROI changed')
        q.write_json(target, result)
        q.render(directory, row, result)
        results.append(result)
    (run / 'roi_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    q.write_json(run / 'roi_summary.json', {'strategy': 'joint_full_window_v2', 'frames': len(results),
        'frames_with_regions': sum(bool(r['regions']) for r in results), 'cached_frames': 0,
        'new_api_calls': 1, 'temporal_filter_frames': 0, 'usage': response.get('usage', {}),
        'offline_full_window_context': True, 'model_alias_is_immutable_snapshot': False, 'automatic_retries': False})
    print(f'Joint-localized {len(results)} frames in one request', flush=True)
    return results
