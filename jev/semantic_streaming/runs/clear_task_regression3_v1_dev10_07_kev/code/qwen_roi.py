#!/usr/bin/env python3
"""Run an auditable single-frame Qwen ROI pilot without persisting credentials."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_BYTES = Path(__file__).read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def local_path(relative):
    result = (ROOT / relative).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Input path escapes project root')
    return result


def load_secret():
    # Do not source shell code, log file contents, or put the secret in arguments.
    value = os.environ.get('DASHSCOPE_API_KEY')
    if not value:
        path = ROOT / '.env.local'
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError('Local credentials require a regular file with mode 0600')
        values = [line.split('=', 1)[1].strip().strip('\"\'')
                  for line in path.read_text().splitlines()
                  if line.startswith('DASHSCOPE_API_KEY=')]
        if len(values) != 1:
            raise ValueError('Expected exactly one DASHSCOPE_API_KEY')
        value = values[0]
    if not value or '*' in value or any(c.isspace() for c in value):
        raise ValueError('Missing or masked API credential')
    return value


def validate_endpoint(base_url):
    p = urllib.parse.urlsplit(base_url)
    shared = {'dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com',
              'dashscope-us.aliyuncs.com', 'cn-hongkong.dashscope.aliyuncs.com'}
    if (p.scheme != 'https' or p.username or p.password or p.query or p.fragment
            or p.port not in (None, 443)
            or not (p.hostname in shared or (p.hostname or '').endswith('.maas.aliyuncs.com'))
            or p.path.rstrip('/') != '/compatible-mode/v1'):
        raise ValueError('Expected an official Aliyun HTTPS endpoint without embedded credentials')


def make_payload(config, prompt, row):
    # Construct every input from an allowlist, never serialize a dataset annotation.
    image = local_path(row['path']).read_bytes()
    if sha(image) != row['sha256']:
        raise ValueError('Frame hash changed')
    content = [
        {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii')},
         **config['image_parameters']},
        {'type': 'text', 'text': json.dumps({
            'frame_id': row['frame_id'], 'timestamp_s': row['timestamp_s'],
            'question': row['question']}, ensure_ascii=False)}]
    return {'model': config['model'], **config['request_parameters'], 'messages': [
        {'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}]}


def parse_regions(response, row, denominator=1000):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Response did not finish normally')
    value = json.loads(choice['message']['content'])
    if set(value) != {'frame_id', 'regions'} or value['frame_id'] != row['frame_id']:
        raise ValueError('Incorrect output keys or frame identity')
    if not isinstance(value['regions'], list) or len(value['regions']) > 4:
        raise ValueError('Invalid region count')
    boxes = []
    for region in value['regions']:
        if not isinstance(region, dict) or set(region) != {'bbox_2d'}:
            raise ValueError('Unexpected region structure')
        box = region['bbox_2d']
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) is not int or not 0 <= v <= 999 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('Invalid normalized xyxy box')
        w, h = row['width'], row['height']
        pixel = [math.floor(box[0]*w/denominator), math.floor(box[1]*h/denominator),
                 math.ceil(box[2]*w/denominator), math.ceil(box[3]*h/denominator)]
        boxes.append({'bbox_normalized': box, 'bbox_pixels': pixel,
                      'area_fraction': (pixel[2]-pixel[0])*(pixel[3]-pixel[1])/(w*h)})
    return {'case_id': row['case_id'], 'frame_id': row['frame_id'],
            'timestamp_s': row['timestamp_s'], 'width': row['width'], 'height': row['height'],
            'image_sha256': row['sha256'], 'regions': boxes,
            'semantics': 'current-frame candidate evidence; temporal question constraints not adjudicated',
            'coordinate_system': 'absolute pixel xyxy; right/bottom exclusive'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect refused', headers, fp)


def render(run_dir, row, result):
    from PIL import Image, ImageDraw
    with Image.open(local_path(row['path'])) as im:
        im = im.convert('RGB')
        draw = ImageDraw.Draw(im)
        for n, box in enumerate(result['regions'], 1):
            x1, y1, x2, y2 = box['bbox_pixels']
            draw.rectangle((x1, y1, x2-1, y2-1), outline='#ff2040', width=3)
            draw.text((x1, max(0, y1-14)), f'ROI {n}', fill='#ff2040')
        draw.rectangle((0, 0, 360, 23), fill='black')
        draw.text((5, 5), f"{row['frame_id']} | t={row['timestamp_s']}s | ROIs={len(result['regions'])}", fill='white')
        path = run_dir / 'overlays' / (row['frame_id'] + '.jpg')
        path.parent.mkdir(exist_ok=True)
        im.save(path, quality=95, subsampling=0)


def summarize(run_dir, plan):
    results, usage, latencies = [], {}, []
    successful = parsed = failed = 0
    for row in plan:
        d = run_dir / 'frames' / row['frame_id']
        if not (d/'status.json').exists():
            continue
        s = json.loads((d/'status.json').read_text())
        if s['state'] == 'complete':
            successful += 1
            response = json.loads((d/'response.json').read_text())
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                usage[k] = usage.get(k, 0) + response.get('usage', {}).get(k, 0)
            latencies.append(s['elapsed_s'])
            if (d/'roi.json').exists():
                result = json.loads((d/'roi.json').read_text())
                results.append(result)
                parsed += 1
        elif s['state'] == 'error':
            failed += 1
    (run_dir/'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))
    summary = {'planned_frames': len(plan), 'successful_requests': successful,
               'parsed_frames': parsed, 'failed_requests': failed,
               'frames_with_regions': sum(bool(r['regions']) for r in results),
               'empty_frames': sum(not r['regions'] for r in results),
               'usage': usage, 'latency_s': {'total': sum(latencies),
                   'mean': sum(latencies)/len(latencies) if latencies else None},
               'updated_at': now(), 'api_errors_billing': 'Unknown; no successful usage returned for errors',
               'localization_accuracy': 'Not measured; requires separate review'}
    write_json(run_dir/'summary.json', summary)
    lines = ['# Qwen ROI pilot', '', '逐帧候选证据定位；无答案输入。时间约束需在后续聚合阶段处理。', '',
             '| Frame | t (s) | ROI count | Pixel boxes | Overlay |', '|---|---:|---:|---|---|']
    for r in results:
        fid = r['frame_id']
        lines.append(f"| {fid} | {r['timestamp_s']} | {len(r['regions'])} | "
                     f"{[b['bbox_pixels'] for b in r['regions']]} | [image](overlays/{fid}.jpg) |")
    (run_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qwen_roi_pilot_v1.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int, default=18, help='Maximum NEW paid requests in this invocation')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) or not 0 <= args.limit <= 60:
        parser.error('Invalid run id or request limit')
    config_bytes = local_path(args.config).read_bytes()
    config = json.loads(config_bytes)
    validate_endpoint(config['base_url'])
    prompt_bytes = local_path(config['prompt']).read_bytes()
    prompt = prompt_bytes.decode().strip()
    input_bytes = local_path(config['inputs']).read_bytes()
    cases = [json.loads(line) for line in input_bytes.decode().splitlines() if line.strip()]
    selected = [r for r in cases if r['case_id'] in config['case_ids']]
    if {r['case_id'] for r in selected} != set(config['case_ids']):
        raise ValueError('Missing pilot cases')
    plan = []
    for case in selected:
        for i, frame in enumerate(case['frames']):
            row = {k: frame[k] for k in ('path', 'timestamp_s', 'width', 'height', 'sha256')}
            row.update(case_id=case['case_id'], frame_id=f"{case['case_id']}_f{i:02d}",
                       question=case['question'], language=case['language'])
            # Verify ALL planned inputs before sending any request.
            from PIL import Image
            p = local_path(row['path'])
            if sha(p.read_bytes()) != row['sha256']:
                raise ValueError('Frame hash changed')
            with Image.open(p) as im:
                if im.size != (row['width'], row['height']):
                    raise ValueError('Frame dimensions changed')
            plan.append(row)
    fingerprint = {'config_sha256': sha(config_bytes), 'prompt_sha256': sha(prompt_bytes),
                   'inputs_sha256': sha(input_bytes), 'script_sha256': sha(SCRIPT_BYTES),
                   'plan_sha256': sha(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode())}
    run_dir = ROOT/'runs'/args.run_id
    manifest_path = run_dir/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Run inputs or code changed; use a new run id')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, {'created_at': now(), 'fingerprint': fingerprint,
                   'config': config, 'label_files_read_for_inference': False,
                   'credential_storage': 'Neither credential nor headers are logged',
                   'retry_policy': 'No automatic retries; ambiguous/interrupted requests require manual review',
                   'input_timestamps': 'Requested relative sampling times, not source presentation timestamps',
                   'python_version': sys.version, 'planned_frames': len(plan)})
        (run_dir/'code').mkdir()
        (run_dir/'code'/'qwen_roi.py').write_bytes(SCRIPT_BYTES)
        (run_dir/'config.json').write_bytes(config_bytes)
        (run_dir/'prompt.txt').write_bytes(prompt_bytes)
        write_json(run_dir/'inputs.json', plan)
    key = load_secret()
    opener = urllib.request.build_opener(NoRedirect())
    sent = 0
    for row in plan:
        d = run_dir/'frames'/row['frame_id']
        if (d/'status.json').exists():
            state = json.loads((d/'status.json').read_text())['state']
            if state != 'complete':
                print(f"Stopped at existing {state} request: {row['frame_id']}", flush=True)
                break
            continue
        if sent >= args.limit:
            break
        payload = make_payload(config, prompt, row)
        wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        write_json(d/'request.json', {'input': row, 'payload_sha256': sha(wire),
                   'payload_bytes': len(wire), 'config_sha256': sha(config_bytes),
                   'prompt_sha256': sha(prompt_bytes),
                   'note': 'Base64 image omitted; reconstruct from hashed PNG, code, prompt and config'})
        write_json(d/'status.json', {'state': 'in_flight', 'started_at': now()})
        request = urllib.request.Request(config['base_url'].rstrip('/')+'/chat/completions', data=wire,
                   headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
        start = time.monotonic()
        sent += 1
        print(f"Calling {row['frame_id']} ({sent}/{args.limit} new requests)", flush=True)
        try:
            with opener.open(request, timeout=90) as response:
                body = response.read()
            # Exact response body unless a provider unexpectedly echoes the credential.
            cleaned = body.replace(key.encode(), b'[REDACTED]')
            (d/'response.json').write_bytes(cleaned)
            response_obj = json.loads(cleaned)
            status = {'state': 'complete', 'elapsed_s': time.monotonic()-start,
                      'finished_at': now(), 'response_sha256': sha(cleaned),
                      'credential_redacted': body != cleaned,
                      'actual_model': response_obj.get('model'), 'request_id': response_obj.get('id')}
            try:
                result = parse_regions(response_obj, row, config['coordinate_contract']['scale_denominator'])
                write_json(d/'roi.json', result)
                render(run_dir, row, result)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                status['parse_error'] = str(exc).replace(key, '[REDACTED]')
            write_json(d/'status.json', status)
            print(f"Finished {row['frame_id']}: {status.get('parse_error', 'valid ROI JSON')}", flush=True)
            if 'parse_error' in status:
                break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            status = {'state': 'error', 'elapsed_s': time.monotonic()-start,
                      'finished_at': now(), 'error_type': type(exc).__name__}
            if isinstance(exc, urllib.error.HTTPError):
                body = exc.read().replace(key.encode(), b'[REDACTED]')
                (d/'error_response.txt').write_bytes(body)
                status['http_status'] = exc.code
                try:
                    e = json.loads(body).get('error', {})
                    status['provider_code'] = e.get('code')
                    status['provider_message'] = e.get('message')
                except (ValueError, AttributeError):
                    pass
            write_json(d/'status.json', status)
            print(f"Request stopped: {status.get('http_status', status['error_type'])}; see status.json", flush=True)
            break
    summary = summarize(run_dir, plan)
    # Do not print provider text, request headers, or any credential-derived fingerprint.
    print(json.dumps({'run': str(run_dir), 'successful': summary['successful_requests'],
                      'parsed': summary['parsed_frames'], 'failed': summary['failed_requests']}), flush=True)
    return 0 if summary['failed_requests'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
