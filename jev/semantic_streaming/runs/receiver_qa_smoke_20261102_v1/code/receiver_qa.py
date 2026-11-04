#!/usr/bin/env python3
"""Blind six-frame receiver QA on matched-byte candidates, with cached responses."""
import argparse
import base64
import json
from pathlib import Path
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request

import qwen_roi as q


def normalize_answer(value):
    value=unicodedata.normalize('NFKC',value).strip().lower()
    if re.fullmatch(r'\d{1,2}:\d{2}',value):
        left,right=value.split(':')
        return f'{int(left)}:{right}'
    return ''.join(c for c in value if c.isalnum())


def payload(config, prompt, candidate):
    content=[{'type':'text','text':json.dumps({'question':candidate['question'],
              'instruction':'以下六帧按时间顺序排列，frame_index 从0开始。'},ensure_ascii=False)}]
    for i,f in enumerate(candidate['frames']):
        image=q.local_path(f['path']).read_bytes()
        if q.sha(image)!=f['sha256']:
            raise ValueError('Decoded frame changed')
        content += [{'type':'text','text':json.dumps({'frame_index':i,'timestamp_s':f['timestamp_s']})},
                    {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()},
                     **config['image_parameters']}]
    # Candidate ID, treatment, bitrate, ROI boxes and reference answers stay out.
    parameters={**config['request_parameters'],'max_tokens':256}
    return {'model':config['model'],**parameters,'messages':[
            {'role':'system','content':prompt},{'role':'user','content':content}]}


def parse_answer(response, frame_count):
    ch=response['choices'][0]
    if ch.get('finish_reason')!='stop':
        raise ValueError('Incomplete QA response')
    obj=json.loads(ch['message']['content'])
    if set(obj)!={'answer','unanswerable','evidence_frame_indices'}:
        raise ValueError('Unexpected QA schema')
    if not isinstance(obj['answer'],str) or type(obj['unanswerable']) is not bool:
        raise ValueError('Invalid answer types')
    indices=obj['evidence_frame_indices']
    if not isinstance(indices,list) or any(type(i) is not int or not 0<=i<frame_count for i in indices):
        raise ValueError('Invalid evidence frame index')
    if obj['unanswerable'] and (obj['answer'] or indices):
        raise ValueError('Inconsistent unanswerable response')
    return obj


def score(run, plan):
    # Reference labels are only read AFTER every planned inference has completed.
    label_path=q.ROOT/'datasets/pilot_dev10_v1/annotations/answers.jsonl'
    labels={d['case_id']:d for d in map(json.loads,label_path.read_text().splitlines())}
    rows=[]
    for c in plan:
        d=run/'calls'/c['blind_id']
        if not (d/'answer.json').exists():
            continue
        answer=json.loads((d/'answer.json').read_text())
        match=normalize_answer(answer['answer']) in {normalize_answer(s) for s in labels[c['case_id']]['accepted_answers']}
        anchor_ok='第一帧' not in c['question'] or 0 in answer['evidence_frame_indices']
        row={'candidate_id':c['candidate_id'],'blind_id':c['blind_id'],'case_id':c['case_id'],
             'treatment':c['treatment'],'target_bytes':c['target_bytes'],'qa':answer,
             'reference_answer':labels[c['case_id']]['reference_answer'],
             'reference_match':match and not answer['unanswerable'] and anchor_ok,
             'first_frame_anchor_ok':anchor_ok,'human_verified':False}
        if c['treatment']!='reference':
            row['actual_bytes']=c['encoding']['selected']['bytes']
        rows.append(row)
    q.write_json(run/'scores.json',{'labels_sha256':q.sha(label_path.read_bytes()),'results':rows,
                 'normalization':'NFKC + case/spacing/punctuation normalization; clock leading zeros ignored; no O/0 equivalence',
                 'limits':'Same Qwen service used for ROI and receiver QA; assistant-reviewed development labels, not human gold or independent test accuracy.'})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoding-run',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--limit',type=int,default=21)
    a=p.parse_args()
    if not 0<=a.limit<=30:
        raise ValueError('Request limit outside pilot range')
    source=q.local_path('runs/'+a.encoding_run)
    source_bytes=(source/'candidates.json').read_bytes()
    cs=json.loads(source_bytes)
    if not (source/'summary.json').exists():
        raise ValueError('Encoder run has not finished')
    pairs={}
    for c in cs:
        if c['treatment']!='reference':
            pairs.setdefault((c['case_id'],c['target_bytes']),{})[c['treatment']]=c['encoding']['within_tolerance']
    qualified={key for key,modes in pairs.items() if modes=={'uniform':True,'roi':True}}
    candidates=[c for c in cs if c['treatment']=='reference' or (c['case_id'],c['target_bytes']) in qualified]
    # Fixed random order; no treatment information appears in requests.
    random.Random(20261102).shuffle(candidates)
    plan=[{**c,'blind_id':f'sample_{i:03d}'} for i,c in enumerate(candidates)]
    config=json.loads((q.ROOT/'configs/qwen_roi_pilot_v1.json').read_text())
    q.validate_endpoint(config['base_url'])
    prompt_bytes=(q.ROOT/'configs/receiver_qa_prompt_v1.txt').read_bytes()
    prompt=prompt_bytes.decode().strip()
    script_bytes=Path(__file__).read_bytes()
    fp={'candidate_list_sha256':q.sha(source_bytes),'script_sha256':q.sha(script_bytes),
        'helper_sha256':q.sha(Path(q.__file__).read_bytes()),
        'prompt_sha256':q.sha(prompt_bytes),'config_sha256':q.sha(json.dumps(config,sort_keys=True).encode())}
    run=q.local_path('runs/'+a.run_id)
    if run.exists():
        if json.loads((run/'manifest.json').read_text())['fingerprint']!=fp:
            raise ValueError('QA inputs/code changed; use a new run id')
    else:
        run.mkdir()
        q.write_json(run/'manifest.json',{'kind':'blind_receiver_qa','created_at':q.now(),'encoding_run':a.encoding_run,
                    'fingerprint':fp,'planned_calls':len(plan),'provider_model':config['model'],
                    'label_files_read_for_requests':False,'automatic_retries':False,
                    'inputs':'Six full decoded frames; no ROI boxes/crops/treatment labels/answers'})
        q.write_json(run/'plan.json',plan)
        q.write_json(run/'config.json',config)
        (run/'prompt.txt').write_bytes(prompt_bytes)
        (run/'code').mkdir()
        (run/'code/receiver_qa.py').write_bytes(script_bytes)
        (run/'code/qwen_roi.py').write_bytes(Path(q.__file__).read_bytes())
    key=q.load_secret()
    opener=urllib.request.build_opener(q.NoRedirect())
    sent=0
    for c in plan:
        d=run/'calls'/c['blind_id']
        if (d/'status.json').exists():
            status=json.loads((d/'status.json').read_text())
            if status['state']!='complete':
                print('Existing failed or ambiguous request; manual review required',flush=True)
                break
            continue
        if sent>=a.limit: break
        body=payload(config,prompt,c)
        wire=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        q.write_json(d/'request.json',{'blind_id':c['blind_id'],'payload_sha256':q.sha(wire),'payload_bytes':len(wire),
                    'frames':c['frames'],'question':c['question'],'note':'No headers logged; no treatment labels or reference answers sent to model'})
        q.write_json(d/'status.json',{'state':'in_flight','started_at':q.now()})
        req=urllib.request.Request(config['base_url']+'/chat/completions',data=wire,
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        start=time.monotonic();sent+=1
        print(f"QA {c['blind_id']} ({sent}/{a.limit})",flush=True)
        try:
            with opener.open(req,timeout=90) as response: raw=response.read()
            clean=raw.replace(key.encode(),b'[REDACTED]')
            (d/'response.json').write_bytes(clean)
            response=json.loads(clean)
            answer=parse_answer(response,len(c['frames']))
            q.write_json(d/'answer.json',answer)
            q.write_json(d/'status.json',{'state':'complete','elapsed_s':time.monotonic()-start,
                        'finished_at':q.now(),'response_sha256':q.sha(clean),'actual_model':response.get('model'),
                        'usage':response.get('usage',{}),'credential_redacted':clean!=raw})
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,OSError,ValueError,KeyError,TypeError,IndexError) as exc:
            status={'state':'error','elapsed_s':time.monotonic()-start,'error_type':type(exc).__name__}
            if isinstance(exc,urllib.error.HTTPError):
                (d/'error_response.txt').write_bytes(exc.read().replace(key.encode(),b'[REDACTED]'))
                status['http_status']=exc.code
            q.write_json(d/'status.json',status)
            print('QA stopped on error; inspect cached status',flush=True)
            break
    completed=sum((run/'calls'/c['blind_id']/'answer.json').exists() for c in plan)
    q.write_json(run/'summary.json',{'planned':len(plan),'completed':completed,'updated_at':q.now()})
    if completed==len(plan):
        rows=score(run,plan)
        print(json.dumps({'completed':completed,'reference_matches':sum(r['reference_match'] for r in rows)}),flush=True)
    else: print(json.dumps({'completed':completed,'planned':len(plan)}),flush=True)


if __name__=='__main__':
    main()
