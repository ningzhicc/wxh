"""Blinded multiple-choice receiver QA; references are read only after all calls."""
import base64
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

import qwen_roi as q
from pipeline_media import file_sha


PROMPT = ('Answer the video question using only the chronologically ordered full frames. '
          'Select an option key; do not infer an unreadable detail from common sense. '
          'Return JSON only: {"choice":"A","unanswerable":false,"evidence_frame_indices":[0]}. '
          'When evidence is insufficient use choice="", unanswerable=true and an empty evidence list.')


def qa_payload(config, case, frames):
    content = [{'type': 'text', 'text': json.dumps({'question': case['question'], 'options': case['options']}, ensure_ascii=False)}]
    for i, frame in enumerate(frames):
        data = q.local_path(frame['path']).read_bytes()
        if q.sha(data) != frame['sha256']:
            raise ValueError('QA frame changed')
        content += [{'type': 'text', 'text': json.dumps({'frame_index': i, 'timestamp_s': frame['timestamp_s']})},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}, **config['image_parameters']}]
    return {'model': config['model'], **config['request_parameters'], 'max_tokens': 256,
        'messages': [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': content}]}


def parse_choice(response, options, frame_count):
    choice = response['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Incomplete QA answer')
    answer = json.loads(choice['message']['content'])
    if set(answer) != {'choice', 'unanswerable', 'evidence_frame_indices'} or type(answer['unanswerable']) is not bool:
        raise ValueError('Invalid QA schema')
    indices = answer['evidence_frame_indices']
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < frame_count for i in indices):
        raise ValueError('Invalid QA evidence frame indices')
    if answer['unanswerable']:
        if answer['choice'] != '' or indices:
            raise ValueError('Inconsistent unanswerable result')
    elif answer['choice'] not in options:
        raise ValueError('Unknown QA option')
    return answer


def run_qa(run, plan, annotations_path, config):
    if len(plan) > 15:
        raise ValueError('Acceptance QA is capped at fifteen calls')
    q.validate_endpoint(config['base_url'])
    order = list(plan)
    random.Random(666).shuffle(order)
    key, opener = None, None
    for number, candidate in enumerate(order):
        directory = run / 'qa' / f'sample_{number:02d}'
        body = qa_payload(config, candidate, candidate['frames'])
        wire = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if (directory / 'status.json').exists():
            status = json.loads((directory / 'status.json').read_text())
            if status['state'] != 'complete':
                raise ValueError('Failed/ambiguous QA request preserved; no automatic retry')
            receipt = json.loads((directory / 'request.json').read_text())
            if receipt['payload_sha256'] != q.sha(wire) or file_sha(directory / 'response.json') != status['response_sha256']:
                raise ValueError('Cached QA request/response changed')
            reparsed = parse_choice(json.loads((directory / 'response.json').read_text()), candidate['options'], len(candidate['frames']))
            if reparsed != json.loads((directory / 'answer.json').read_text()):
                raise ValueError('Cached parsed QA answer differs from raw response')
            continue
        if key is None:
            key = q.load_secret()
            opener = urllib.request.build_opener(q.NoRedirect())
        directory.mkdir(parents=True)
        q.write_json(directory / 'request.json', {'payload_sha256': q.sha(wire), 'payload_bytes': len(wire),
            'case_id': candidate['case_id'], 'variant': candidate['variant'], 'inputs': candidate,
            'note': 'Variant/case IDs stored locally only; requests contain full frames, question and all options, never the correct option.'})
        q.write_json(directory / 'status.json', {'state': 'in_flight', 'started_at': q.now()})
        request = urllib.request.Request(config['base_url'].rstrip('/') + '/chat/completions', data=wire,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        try:
            with opener.open(request, timeout=90) as response:
                raw = response.read().replace(key.encode(), b'[REDACTED]')
            (directory / 'response.json').write_bytes(raw)
            response = json.loads(raw)
            answer = parse_choice(response, candidate['options'], len(candidate['frames']))
            q.write_json(directory / 'answer.json', answer)
            q.write_json(directory / 'status.json', {'state': 'complete', 'response_sha256': q.sha(raw),
                'elapsed_s': time.monotonic() - started, 'actual_model': response.get('model'), 'usage': response.get('usage', {})})
        except Exception as error:
            status = {'state': 'failed_no_retry', 'error_type': type(error).__name__}
            if isinstance(error, urllib.error.HTTPError):
                status['http_status'] = error.code
                (directory / 'error_response.txt').write_bytes(error.read().replace(key.encode(), b'[REDACTED]'))
            q.write_json(directory / 'status.json', status)
            raise RuntimeError('QA request failed; no automatic retry') from None
        print(f'QA completed {number + 1}/{len(order)}', flush=True)
    # The first reference-label read occurs after every planned inference completed.
    references = {r['case_id']: r for r in map(json.loads, annotations_path.read_text().splitlines())}
    scores = []
    for number, candidate in enumerate(order):
        answer = json.loads((run / 'qa' / f'sample_{number:02d}' / 'answer.json').read_text())
        reference = references[candidate['case_id']]['reference_answer']
        correct = not answer['unanswerable'] and candidate['options'][answer['choice']] == reference
        scores.append({'case_id': candidate['case_id'], 'variant': candidate['variant'], 'answer': answer,
            'reference_answer': reference, 'correct': correct})
    q.write_json(run / 'qa_scores.json', {'results': scores, 'annotations_sha256': file_sha(annotations_path),
        'limits': 'Published two/four-option QA, same Qwen provider as ROI; 5 cases, no independent evaluator or full benchmark accuracy claim.'})
    return scores
