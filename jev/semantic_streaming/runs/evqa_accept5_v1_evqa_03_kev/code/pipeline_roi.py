"""Strict cached-or-live Qwen localization with bounded requests and no key logs."""
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

import qwen_roi as q
from pipeline_media import file_sha


IDENTITY = ('model', 'base_url', 'request_parameters', 'image_parameters', 'coordinate_contract')


def matching_key(row):
    return row['sha256'], row['question'], round(float(row['timestamp_s']), 8), row['width'], row['height']


def cache_index(cache_run, config, prompt):
    if not cache_run:
        return {}
    aggregate = q.local_path('runs/' + cache_run)
    manifest = json.loads((aggregate / 'manifest.json').read_text())
    if manifest.get('kind') == 'continuous_semantic_video_pipeline_v1':
        return pipeline_cache_index(aggregate, config, prompt)
    sources = manifest.get('sources', [{'run': cache_run}])
    result = {}
    for source in sources:
        directory = q.local_path('runs/' + source['run'])
        if source.get('manifest_sha256') and file_sha(directory / 'manifest.json') != source['manifest_sha256']:
            raise ValueError('ROI cache source manifest changed')
        original = json.loads((directory / 'manifest.json').read_text())
        original_config = original['config']
        if any(original_config[k] != config[k] for k in IDENTITY):
            raise ValueError('ROI cache uses different model/settings')
        original_prompt = (directory / 'prompt.txt').read_bytes()
        if q.sha(original_prompt) != q.sha(prompt):
            raise ValueError('ROI cache prompt differs')
        for row in json.loads((directory / 'inputs.json').read_text()):
            frame = directory / 'frames' / row['frame_id']
            status = json.loads((frame / 'status.json').read_text())
            if status['state'] != 'complete' or status.get('parse_error'):
                continue
            if file_sha(frame / 'response.json') != status['response_sha256']:
                raise ValueError('ROI cache response changed')
            receipt = json.loads((frame / 'request.json').read_text())
            if receipt['input'] != row:
                raise ValueError('ROI cache request input changed')
            payload = q.make_payload(original_config, original_prompt.decode().strip(), row)
            wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            if q.sha(wire) != receipt['payload_sha256']:
                raise ValueError('ROI cache cannot reconstruct original request')
            response = json.loads((frame / 'response.json').read_text())
            parsed = q.parse_regions(response, row, config['coordinate_contract']['scale_denominator'])
            result[matching_key(row)] = (parsed, {'run': source['run'], 'frame_id': row['frame_id'],
                'request_payload_sha256': receipt['payload_sha256'], 'response_sha256': status['response_sha256'],
                'actual_model': status.get('actual_model'), 'timestamp_s': row['timestamp_s']})
    return result


def pipeline_cache_index(directory, config, prompt):
    """Reuse a previous pipeline's API provenance, including inherited batch rows."""
    from run_semantic_pipeline import verify_files
    if (directory / 'files.sha256.json').exists():
        verify_files(json.loads((directory / 'files.sha256.json').read_text()))
    else:
        for stage in ('prepare', 'roi'):
            verify_files(json.loads((directory / 'stages' / (stage + '.json')).read_text())['files_sha256'])
    original_config = json.loads((directory / 'api_config.json').read_text())
    original_prompt = (directory / 'roi_prompt.txt').read_bytes()
    if any(original_config[k] != config[k] for k in IDENTITY) or q.sha(original_prompt) != q.sha(prompt):
        raise ValueError('Pipeline cache uses different API/prompt settings')
    parsed_rows = {r['frame_id']: r for r in map(json.loads, (directory / 'roi_results.jsonl').read_text().splitlines())}
    result = {}
    for row in json.loads((directory / 'roi_inputs.json').read_text()):
        stored = parsed_rows[row['frame_id']]
        if stored['origin'] == 'explicit_evidence_window_filter':
            continue  # A user time filter is not an API prediction of absence.
        if stored['origin'] == 'new_API':
            frame = directory / 'roi' / row['frame_id']
            if file_sha(frame / 'response.json') != stored['response_sha256']:
                raise ValueError('Pipeline cached raw response changed')
            request = json.loads((frame / 'request.json').read_text())
            wire = json.dumps(q.make_payload(original_config, original_prompt.decode().strip(), row), ensure_ascii=False, allow_nan=False).encode()
            if q.sha(wire) != request['payload_sha256']:
                raise ValueError('Pipeline cache request cannot be reconstructed')
            parsed = q.parse_regions(json.loads((frame / 'response.json').read_text()), row, config['coordinate_contract']['scale_denominator'])
            source = {'run': directory.name, 'frame_id': row['frame_id'], 'frame_relative_path': 'roi/' + row['frame_id'],
                'request_payload_sha256': request['payload_sha256'], 'response_sha256': stored['response_sha256'],
                'actual_model': stored.get('actual_model'), 'timestamp_s': row['timestamp_s']}
        else:
            source = dict(stored['source'])
            frame = q.local_path('runs/' + source['run']) / source.get('frame_relative_path', 'frames/' + source['frame_id'])
            if file_sha(frame / 'response.json') != source['response_sha256']:
                raise ValueError('Inherited API response changed')
            request_row = json.loads((frame / 'request.json').read_text())['input']
            parsed = q.parse_regions(json.loads((frame / 'response.json').read_text()), request_row, config['coordinate_contract']['scale_denominator'])
            if matching_key(request_row) != matching_key(row):
                raise ValueError('Inherited localization context differs')
        source['cache_via_pipeline'] = directory.name
        source['cache_pipeline_manifest_sha256'] = file_sha(directory / 'manifest.json')
        result[matching_key(row)] = (parsed, source)
    return result


def outside_window(row, window):
    return window is not None and not window[0] <= row['actual_window_timestamp_s'] < window[1]


def locate(inputs, run, cfg, api_config, prompt):
    q.validate_endpoint(api_config['base_url'])
    cache = cache_index(cfg.get('roi_cache_run'), api_config, prompt) if cfg['roi_mode'] != 'api' else {}
    expected_calls = sum(not outside_window(row, cfg.get('evidence_window_s')) and matching_key(row) not in cache for row in inputs)
    if cfg['roi_mode'] == 'cache' and expected_calls:
        missing = [r['frame_id'] for r in inputs if matching_key(r) not in cache and not outside_window(r, cfg.get('evidence_window_s'))]
        q.write_json(run / 'roi_missing_cache.json', {'frame_ids': missing, 'automatic_api_calls': False})
        raise ValueError('No exact image/question/time cache match; choose API mode explicitly')
    if cfg['roi_mode'] != 'cache' and expected_calls > cfg['max_api_calls']:
        raise ValueError(f'Planned {expected_calls} API calls exceed the configured maximum {cfg["max_api_calls"]}')
    directory = run / 'roi'
    directory.mkdir(exist_ok=True)
    key, opener = None, None
    results, counts, usage = [], {'cached_frames': 0, 'new_api_calls': 0, 'temporal_filter_frames': 0}, {}
    for row in inputs:
        frame = directory / row['frame_id']
        status_path = frame / 'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['state'] != 'complete':
                raise ValueError('Ambiguous/failed API attempt preserved; no automatic retry')
            if file_sha(frame / 'result.json') != status['result_sha256']:
                raise ValueError('Recorded ROI result changed')
            result = json.loads((frame / 'result.json').read_text())
            if result['image_sha256'] != row['sha256']:
                raise ValueError('Resumed ROI image differs')
            if result['origin'] == 'new_API' and file_sha(frame / 'response.json') != result['response_sha256']:
                raise ValueError('Resumed API response changed')
        elif outside_window(row, cfg.get('evidence_window_s')):
            result = {'case_id': row['case_id'], 'frame_id': row['frame_id'], 'timestamp_s': row['timestamp_s'],
                'width': row['width'], 'height': row['height'], 'image_sha256': row['sha256'], 'regions': [],
                'origin': 'explicit_evidence_window_filter', 'not_an_API_absence_prediction': True}
        elif matching_key(row) in cache:
            original, receipt = cache[matching_key(row)]
            result = {**original, 'case_id': row['case_id'], 'frame_id': row['frame_id'],
                'origin': 'verified_cache', 'source': receipt,
                'cache_note': 'Same pixels, question, requested timestamp and API settings; opaque frame identity rebound. Not a new request.'}
        else:
            if key is None:
                key = q.load_secret()
                opener = urllib.request.build_opener(q.NoRedirect())
            frame.mkdir(parents=True, exist_ok=True)
            payload = q.make_payload(api_config, prompt.decode().strip(), row)
            wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            q.write_json(frame / 'request.json', {'input': row, 'payload_sha256': q.sha(wire), 'payload_bytes': len(wire),
                'model': api_config['model'], 'prompt_sha256': q.sha(prompt),
                'note': 'No key or headers recorded; image bytes reconstructed from hashed PNG.'})
            q.write_json(status_path, {'state': 'in_flight', 'started_at': q.now()})
            request = urllib.request.Request(api_config['base_url'].rstrip('/') + '/chat/completions', data=wire,
                headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
            started = time.monotonic()
            try:
                with opener.open(request, timeout=90) as response:
                    raw = response.read().replace(key.encode(), b'[REDACTED]')
                (frame / 'response.json').write_bytes(raw)
                response = json.loads(raw)
                parsed = q.parse_regions(response, row, api_config['coordinate_contract']['scale_denominator'])
                result = {**parsed, 'origin': 'new_API', 'actual_model': response.get('model'),
                    'response_sha256': q.sha(raw), 'elapsed_s': time.monotonic() - started,
                    'usage': response.get('usage', {})}
            except Exception as error:
                failure = {'state': 'failed_no_retry', 'error_type': type(error).__name__, 'finished_at': q.now()}
                if isinstance(error, urllib.error.HTTPError):
                    failure['http_status'] = error.code
                    (frame / 'error_response.txt').write_bytes(error.read().replace(key.encode(), b'[REDACTED]'))
                q.write_json(status_path, failure)
                raise RuntimeError('ROI request failed; request state retained without automatic retry') from None
            print('Localized ' + row['frame_id'], flush=True)
        frame.mkdir(parents=True, exist_ok=True)
        q.write_json(frame / 'result.json', result)
        q.write_json(status_path, {'state': 'complete', 'result_sha256': file_sha(frame / 'result.json'), 'origin': result['origin']})
        q.render(directory, row, result)
        results.append(result)
        if result['origin'] == 'verified_cache':
            counts['cached_frames'] += 1
        elif result['origin'] == 'new_API':
            counts['new_api_calls'] += 1
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                usage[field] = usage.get(field, 0) + result['usage'].get(field, 0)
        else:
            counts['temporal_filter_frames'] += 1
    (run / 'roi_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    q.write_json(run / 'roi_summary.json', {**counts, 'frames': len(results),
        'frames_with_regions': sum(bool(r['regions']) for r in results), 'usage': usage,
        'model_alias_is_immutable_snapshot': False, 'automatic_retries': False})
    return results
