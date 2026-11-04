"""Bounded, blinded free-text QA with explicit repeated-payload verification."""
from collections import Counter
import json
import random
import time
import urllib.error
import urllib.request
import qwen_roi as q
from pipeline_media import file_sha
from receiver_qa import payload as original_payload,parse_answer,normalize_answer


def make_payload(config,prompt,candidate):
    result=original_payload(config,prompt,candidate)
    result['messages'][1]['content'][0]['text']=json.dumps({'question':candidate['question'],
        'instruction':f"以下{len(candidate['frames'])}张完整帧按时间先后排列，frame_index从0开始。"},ensure_ascii=False)
    return result


def match_answer(answer,label,first_frame_only):
    normalized=normalize_answer(answer['answer'])
    match=not answer['unanswerable'] and normalized in {normalize_answer(v) for v in label['accepted_answers']}
    anchor_ok=not first_frame_only or bool(answer['evidence_frame_indices']) and all(i==0 for i in answer['evidence_frame_indices'])
    return bool(match and anchor_ok),bool(anchor_ok)


def run_qa(run,plan,annotation_path,config,prompt):
    if not 1<=len(plan)<=30:raise ValueError('Open QA call cap is 30')
    if len({(r['case_id'],r['variant'],r['repeat']) for r in plan})!=len(plan):raise ValueError('Duplicate repetition')
    q.validate_endpoint(config['base_url'])
    order=list(plan);random.Random(666).shuffle(order);key=None;opener=None
    for index,row in enumerate(order):
        directory=run/'qa'/f'sample_{index:02d}';body=make_payload(config,prompt,row)
        wire=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        if (directory/'status.json').exists():
            status=json.loads((directory/'status.json').read_text())
            if status['state']!='complete':raise ValueError('Failed/ambiguous QA request retained; no automatic retry')
            request=json.loads((directory/'request.json').read_text())
            if request['payload_sha256']!=q.sha(wire) or file_sha(directory/'response.json')!=status['response_sha256']:raise ValueError('QA resume mismatch')
            parsed=parse_answer(json.loads((directory/'response.json').read_text()),len(row['frames']))
            if parsed!=json.loads((directory/'answer.json').read_text()):raise ValueError('QA parse mismatch')
            continue
        if key is None:key=q.load_secret();opener=urllib.request.build_opener(q.NoRedirect())
        directory.mkdir(parents=True)
        q.write_json(directory/'request.json',{'input':row,'payload_sha256':q.sha(wire),'payload_bytes':len(wire),
            'note':'Case, policy and repeat IDs stay local. API sees full frames and question only; no answers, ROIs, crops, choices or GT times.'})
        q.write_json(directory/'status.json',{'state':'in_flight','started_at':q.now()})
        req=urllib.request.Request(config['base_url'].rstrip('/')+'/chat/completions',data=wire,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST');start=time.monotonic()
        try:
            with opener.open(req,timeout=90) as response:raw=response.read().replace(key.encode(),b'[REDACTED]')
            (directory/'response.json').write_bytes(raw);response=json.loads(raw)
            parsed=parse_answer(response,len(row['frames']));q.write_json(directory/'answer.json',parsed)
            q.write_json(directory/'status.json',{'state':'complete','response_sha256':q.sha(raw),
                'elapsed_s':time.monotonic()-start,'model':response.get('model'),'usage':response.get('usage',{})})
        except Exception as error:
            failure={'state':'failed_no_retry','error_type':type(error).__name__}
            if isinstance(error,urllib.error.HTTPError):
                failure['http_status']=error.code;(directory/'error_response.txt').write_bytes(error.read().replace(key.encode(),b'[REDACTED]'))
            q.write_json(directory/'status.json',failure);raise RuntimeError('Open QA failed; no automatic retry') from None
        print(f'Open QA {index+1}/{len(order)} complete',flush=True)
    # Label scoring happens after all planned inference, never during requests.
    labels={r['case_id']:r for r in map(json.loads,annotation_path.read_text().splitlines())}
    scores=[];groups={}
    for index,row in enumerate(order):
        directory=run/'qa'/f'sample_{index:02d}';answer=json.loads((directory/'answer.json').read_text())
        match,anchor=match_answer(answer,labels[row['case_id']],row.get('first_frame_only',False))
        receipt=json.loads((directory/'request.json').read_text())
        score={'case_id':row['case_id'],'variant':row['variant'],'repeat':row['repeat'],'answer':answer,
            'reference_answer':labels[row['case_id']]['reference_answer'],'reference_match':match,'temporal_anchor_ok':anchor,
            'payload_sha256':receipt['payload_sha256']};scores.append(score);groups.setdefault((row['case_id'],row['variant']),[]).append(score)
    stats=[]
    for (cid,variant),rows in groups.items():
        if len({r['payload_sha256'] for r in rows})!=1:raise ValueError('Repeated QA wire payloads differ')
        stats.append({'case_id':cid,'variant':variant,'repeats':len(rows),'correct':sum(r['reference_match'] for r in rows),
            'answer_histogram':dict(Counter(r['answer']['answer'] if not r['answer']['unanswerable'] else 'UNANSWERABLE' for r in rows)),
            'all_normalized_answers_agree':len({normalize_answer(r['answer']['answer']) for r in rows})==1,
            'payload_sha256':rows[0]['payload_sha256']})
    q.write_json(run/'qa_scores.json',{'results':scores,'groups':stats,'annotations_sha256':file_sha(annotation_path),
        'limits':'Same provider as localization; three repetitions per variant, development data used previously, no independent benchmark claim.'})
    return stats
