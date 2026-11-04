#!/usr/bin/env python3
"""Bridge measured ROI encodings to variable-choice Kev requests and replay labels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import qwen_roi as q


def read_trace(path):
    rows=[list(map(float,l.split()[:2])) for l in path.read_text().splitlines() if l.strip()]
    if len(rows)<2 or any(len(r)!=2 or not all(math.isfinite(v) for v in r) for r in rows):
        raise ValueError('Invalid bandwidth trace')
    if rows[0][0]!=0 or any(b[0]<a[0] or b[1]<0 for a,b in zip(rows,rows[1:])):
        raise ValueError('Expected monotonic, zero-based nonnegative trace')
    if sum((b[0]-a[0])*b[1] for a,b in zip(rows,rows[1:]))<=0:
        raise ValueError('Trace has no transmission capacity')
    return rows


def replay_download_s(trace, pointer, size_bytes, fraction=.95, rtt=.08):
    """Replay each action from the same unmodified segment start; bandwidth unit Mbps."""
    remaining=float(size_bytes)
    elapsed=0.0
    i=pointer
    while remaining>0:
        duration=trace[i][0]-trace[i-1][0]
        rate=trace[i][1]*1e6/8*fraction
        sent=duration*rate
        if rate>0 and remaining<=sent:
            elapsed+=remaining/rate
            break
        remaining-=sent
        elapsed+=duration
        i+=1
        if i==len(trace): i=1
    return elapsed+rtt


def request(state, actions, rtt_s=.08, payload_fraction=.95):
    history=state['recent_link_bandwidth_mbps']
    positive=[v for v in history if v>0]
    estimate=len(positive)/sum(1/v for v in positive) if positive else 1e-6
    criteria={}
    for action in actions:
        expected=rtt_s+action['size_bytes']*8/(payload_fraction*estimate*1e6)
        criteria[action['action_id']]={
            k:action[k] for k in ['mode','size_bytes','actual_bitrate_kbps','roi_qoffset']}
        criteria[action['action_id']].update(est_download_s=round(expected,5),
              est_buffer_margin_s=round(state['buffer_seconds']-expected,5))
    return {'state':state,'questions':{'encoding':{'type':'choice',
            'instructions':'Choose the encoding configuration for the next video chunk. Preserve question-relevant evidence while avoiding stalls and large bitrate changes. Candidate download estimates use only past link-rate observations and may be wrong. ROI qoffset is a relative quality adjustment. Pick one available action.',
            'criteria':criteria,'src':'semantic_video_encoding'}}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/semantic_kev_v1.json')
    a=p.parse_args()
    cfgpath=q.local_path(a.config);cfg=json.loads(cfgpath.read_text())
    dest=q.local_path(cfg['output_dataset']);dest.mkdir(exist_ok=False)
    source=q.local_path('runs/'+cfg['encoding_run'])
    candidates=json.loads((source/'candidates.json').read_text())
    measured=json.loads((source/'comparison_report.json').read_text())
    actions={}
    for cid in cfg['case_ids']:
        rows=[]
        for pair in measured['pairs']:
            if pair['case_id']!=cid or not pair['qualifies']: continue
            for mode in ['uniform','roi']:
                c=next(c for c in candidates if c['case_id']==cid and c['target_bytes']==pair['target_bytes'] and c['treatment']==mode)
                selected=c['encoding']['selected'];video=q.local_path(selected['path'])
                if q.sha(video.read_bytes())!=selected['sha256']: raise ValueError('Encoded video changed')
                fps=c['decoded_info']['fps']
                from fractions import Fraction
                duration=float(Fraction(c['decoded_info']['frames'],1)/Fraction(fps))
                psnr=pair['treatments'][mode]['metrics']['sampled_api_visible_evidence_rgb_psnr_db']
                rows.append({'action_id':f'a{len(rows):02d}','candidate_id':c['candidate_id'],
                     'mode':mode,'size_bytes':selected['bytes'],'actual_bitrate_kbps':round(selected['bytes']*8/duration/1000,4),
                     'duration_s':duration,'roi_qoffset':'-1/5' if mode=='roi' else '0',
                     'video_path':selected['path'],'video_sha256':selected['sha256'],
                     'offline_quality_proxy':min(1,max(0,psnr/cfg['quality_scale_db']))})
        if len(rows)<2: raise ValueError('Each case needs at least two feasible actions')
        actions[cid]=rows
    q.write_json(dest/'actions.json',actions)
    # The existing protocol is read-only; a separate semantic protocol uses variable actions.
    bundle=(q.ROOT/cfg['trace_bundle']).resolve()
    chosen={split:json.loads((bundle/f'{split}.json').read_text())[:limit]
            for split,limit in cfg['trace_limits'].items()}
    if {r['source_parent'] for r in chosen['train']} & {r['source_parent'] for r in chosen['valid']}:
        raise ValueError('Network acquisition group overlap')
    trace_receipts=[];counts={};labels=Counter()
    for split,traces in chosen.items():
        records=[]
        for tr in traces:
            path=bundle/tr['snapshot']
            if q.sha(path.read_bytes())!=tr['snapshot_sha256']: raise ValueError('Trace changed')
            trace=read_trace(path)
            trace_receipts.append({'split':split,'id':tr['id'],'source_parent':tr['source_parent'],
                         'path':str(path),'sha256':tr['snapshot_sha256']})
            for position in cfg['trace_positions']:
                pointer=min(len(trace)-1,max(1,round(position*(len(trace)-1))))
                history=[r[1] for r in trace[max(1,pointer-5):pointer]]
                if not history: continue
                for cid in cfg['case_ids']:
                    available=actions[cid]
                    previous=available[0]['actual_bitrate_kbps']
                    delays={r['action_id']:replay_download_s(trace,pointer,r['size_bytes'],cfg['payload_fraction'],cfg['rtt_s']) for r in available}
                    question=next(c['question'] for c in candidates if c['case_id']==cid)
                    for buffer in cfg['buffer_scenarios_s']:
                        state={'question':question,'case_id':cid,'recent_link_bandwidth_mbps':history,
                               'buffer_seconds':buffer,'previous_bitrate_kbps':previous,
                               'chunk_duration_seconds':available[0]['duration_s']}
                        r=request(state,available,cfg['rtt_s'],cfg['payload_fraction'])
                        rewards={action['action_id']:action['offline_quality_proxy']-
                             cfg['stall_weight']*max(0,delays[action['action_id']]-buffer)-
                             cfg['switch_weight']*abs(action['actual_bitrate_kbps']-previous)/1000 for action in available}
                        best=max(rewards,key=rewards.get)
                        r['questions']['encoding']['label']=best
                        r['_supervision']={'quality_and_future_delay_not_online_inputs':True,
                             'candidate_reward':rewards,'candidate_actual_download_s':delays,
                             'quality_proxy':{x['action_id']:x['offline_quality_proxy'] for x in available}}
                        r['_meta']={'id':f"{split}/{tr['id']}/{pointer}/{cid}/{buffer}",'split':split,
                             'trace_group':tr['source_parent'],'source':'semantic_replay_v1',
                             'buffer_origin':'sampled scenario, not a logged policy rollout'}
                        records.append(r);labels[f'{cid}/{best}']+=1
        (dest/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        counts[split]=len(records)
    q.write_json(dest/'traces.lock.json',trace_receipts)
    (dest/'code').mkdir();(dest/'code/prepare_semantic_kev.py').write_bytes(Path(__file__).read_bytes())
    (dest/'config.json').write_bytes(cfgpath.read_bytes())
    q.write_json(dest/'manifest.json',{'created_at':q.now(),'config':cfg,'counts':counts,
           'action_counts':{k:len(v) for k,v in actions.items()},'label_counts':dict(labels),
           'source_encoding_candidates_sha256':q.sha((source/'candidates.json').read_bytes()),
           'source_comparison_report_sha256':q.sha((source/'comparison_report.json').read_bytes()),
           'files_sha256':{p.name:q.sha(p.read_bytes()) for p in dest.glob('*') if p.is_file()},
           'script_sha256':q.sha(Path(__file__).read_bytes()),
           'training_started':False,'reward_note':'Continuous pixel-quality proxy plus replayed stall/switch penalties; no repeated QA calls needed.',
           'limitations':cfg['dataset_role']})
    print(json.dumps({'dataset':str(dest),'counts':counts,'actions':{k:len(v) for k,v in actions.items()}}))


if __name__=='__main__': main()
