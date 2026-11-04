#!/usr/bin/env python3
"""Matched-byte local encoder pilot. No model calls and no answer labels read."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import qwen_roi as q


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_box(results, width, height, padding=8, block=16):
    boxes=[b['bbox_pixels'] for r in results for b in r['regions']]
    if not boxes:
        raise ValueError('No API boxes to derive an encoder ROI')
    return [max(0, math.floor((min(b[0] for b in boxes)-padding)/block)*block),
            max(0, math.floor((min(b[1] for b in boxes)-padding)/block)*block),
            min(width, math.ceil((max(b[2] for b in boxes)+padding)/block)*block),
            min(height, math.ceil((max(b[3] for b in boxes)+padding)/block)*block)]


def command(cmd, log):
    q.write_json(log.with_suffix('.command.json'),{'argv':cmd,'started_at':q.now()})
    start=time.monotonic()
    with log.open('wb') as out:
        result=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
    q.write_json(log.with_suffix('.receipt.json'),{'returncode':result.returncode,
                 'elapsed_s':time.monotonic()-start,'log_sha256':digest(log)})
    if result.returncode:
        raise RuntimeError('FFmpeg failed; inspect '+str(log))
    if 'skipping ROI' in log.read_text(errors='replace'):
        raise RuntimeError('Encoder ignored ROI; inspect '+str(log))


def frame_info(ffmpeg, path, directory):
    cmd=[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-i',str(path),
         '-map','0:v:0','-an','-threads','1','-f','framemd5','-']
    r=subprocess.run(cmd,capture_output=True,text=True,check=True)
    (directory/'source.framemd5').write_text(r.stdout)
    q.write_json(directory/'source_probe.json',{'argv':cmd,'stderr':r.stderr})
    lines=r.stdout.splitlines()
    tb=Fraction(next(l.split(':',1)[1].strip() for l in lines if l.startswith('#tb 0:')))
    entries=[[int(x.strip()) for x in l.split(',')[:5]] for l in lines if l and not l.startswith('#')]
    pts=[e[2] for e in entries]
    steps={b-a for a,b in zip(pts,pts[1:])}
    if len(steps)!=1 or pts[0]!=0:
        raise ValueError('This smoke test supports zero-based CFR references only')
    step=next(iter(steps))
    fps=1/(tb*step)
    return {'frames':len(entries),'fps':str(fps),'duration_s':float(len(entries)/fps),
            'source_framemd5_sha256':digest(directory/'source.framemd5')}


def encode_pair_member(ffmpeg, source, media, logs, cfg, info, box, mode, budget):
    trials=[]
    bitrate=max(1000,int((budget-2000)*8/info['duration_s']))
    for attempt in range(cfg['max_rate_matching_attempts']):
        trial=media/f'{mode}_{budget}_a{attempt}'
        trial.mkdir(exist_ok=False)
        trial_logs=logs/trial.name
        trial_logs.mkdir()
        out=trial/'video.mp4'
        options=['-map','0:v:0','-an','-sn','-dn']
        if mode=='roi':
            x1,y1,x2,y2=box
            options+=['-vf',f"addroi=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:qoffset={cfg['roi_qoffset']}"]
        options+=['-c:v',cfg['codec'],'-preset',cfg['preset'],'-pix_fmt',cfg['pixel_format'],
                  '-threads',str(cfg['threads']),'-b:v',str(bitrate),'-aq-mode','1','-aq-strength','1',
                  '-x264-params','mbtree=0','-g','60','-keyint_min','30','-fastfirstpass','0',
                  '-frames:v',str(info['frames']),'-r',info['fps'],'-fps_mode','cfr',
                  '-passlogfile',str(trial/'pass')]
        try:
            for phase in [1,2]:
                cmd=[str(ffmpeg),'-hide_banner','-nostdin','-y','-i',str(source),*options,'-pass',str(phase)]
                cmd+=['-f','null','/dev/null'] if phase==1 else ['-movflags','+faststart',str(out)]
                command(cmd,trial_logs/f'pass{phase}.log')
        except RuntimeError:
            # Unreachable rates are experimental outcomes, not comparable video candidates.
            if 'requested bitrate is too low' not in (trial_logs/f'pass{phase}.log').read_text():
                raise
            failure={'attempt':attempt,'requested_bitrate_bps':bitrate,'state':'encoder_minimum_bitrate_rejected'}
            trials.append(failure)
            q.write_json(trial_logs/'result.json',failure)
            break
        size=out.stat().st_size
        trial_result={'attempt':attempt,'requested_bitrate_bps':bitrate,'bytes':size,
                      'relative_size_error':abs(size-budget)/budget,'path':str(out.relative_to(q.ROOT)),
                      'sha256':digest(out)}
        trials.append(trial_result)
        q.write_json(trial_logs/'result.json',trial_result)
        if abs(size-budget)/budget <= cfg['size_tolerance_fraction']:
            break
        bitrate=max(1000,round(bitrate*max(1,budget-2000)/max(1,size-2000)))
    valid=[t for t in trials if 'relative_size_error' in t]
    best=min(valid,key=lambda t:t['relative_size_error']) if valid else None
    return {'mode':mode,'target_bytes':budget,'selected':best,'trials':trials,
            'within_tolerance':best is not None and best['relative_size_error']<=cfg['size_tolerance_fraction']}


def extract_frames(ffmpeg, source, dest, times, logs):
    dest.mkdir(parents=True,exist_ok=True)
    frames=[]
    for i,t in enumerate(times):
        path=dest/f'frame_{i:02d}.png'
        cmd=[str(ffmpeg),'-hide_banner','-nostdin','-n','-ss',str(t),'-i',str(source),
             '-map','0:v:0','-frames:v','1','-threads','1',str(path)]
        command(cmd,logs/f'extract_{i:02d}.log')
        frames.append({'path':str(path.relative_to(q.ROOT)),'timestamp_s':t,'sha256':digest(path)})
    return frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/roi_encoding_smoke_v1.json')
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    cfg_path=q.local_path(a.config)
    cfg=json.loads(cfg_path.read_text())
    run=q.local_path('runs/'+a.run_id)
    run.mkdir(exist_ok=False)
    media=q.local_path('artifacts/'+a.run_id)
    media.mkdir(exist_ok=False)
    ff=q.ROOT/'.tools/ffmpeg'
    roi_run=q.local_path('runs/'+cfg['roi_run'])
    rois=[json.loads(l) for l in (roi_run/'results.jsonl').read_text().splitlines()]
    inputs=json.loads((roi_run/'inputs.json').read_text())
    q.write_json(run/'manifest.json',{'kind':'local_encoder_smoke','created_at':q.now(),'config':cfg,
                 'config_sha256':digest(cfg_path),'script_sha256':digest(Path(__file__)),
                 'ffmpeg_sha256':digest(ff),'roi_results_sha256':digest(roi_run/'results.jsonl'),
                 'roi_inputs_sha256':digest(roi_run/'inputs.json'),'label_files_read':False,
                 'roi_qoffset_note':'8-bit x264 -1/5 requests approximately -10.2 relative QP, not a fixed absolute QP.',
                 'comparison_rule':'Only size-matched pairs within configured tolerance qualify.'})
    (run/'config.json').write_bytes(cfg_path.read_bytes())
    (run/'code').mkdir()
    (run/'code/encode_roi_smoke.py').write_bytes(Path(__file__).read_bytes())
    candidates=[]
    for cid in cfg['case_ids']:
        rows=[row for row in inputs if row['case_id']==cid]
        width,height=rows[0]['width'],rows[0]['height']
        box=union_box([r for r in rois if r['case_id']==cid],width,height,
                      cfg['roi_padding_pixels'],cfg['macroblock_size'])
        source=q.ROOT/'artifacts/pilot_dev10_v1'/cid/'reference_lossless.mp4'
        receipt=json.loads((q.ROOT/'datasets/pilot_dev10_v1/provenance/preparation'/f'{cid}.json').read_text())
        original=next(f for f in receipt['files'] if f['path'].endswith('reference_lossless.mp4'))
        if digest(source)!=original['sha256']:
            raise ValueError('Reference media differs from preparation receipt')
        logs=run/cid
        logs.mkdir()
        cm=media/cid
        cm.mkdir()
        info=frame_info(ff,source,logs)
        q.write_json(logs/'roi_policy.json',{'case_id':cid,'box_pixels':box,'width':width,'height':height,
                     'policy':cfg['roi_policy'],'source':str(source.relative_to(q.ROOT)),
                     'source_sha256':digest(source),'frame_info':info,
                     'area_fraction':(box[2]-box[0])*(box[3]-box[1])/(width*height)})
        (logs/'frame_rois.jsonl').write_text(''.join(json.dumps({'frame_index':i,
                   'timestamp_s':float(Fraction(i,1)/Fraction(info['fps'])),'bbox_pixels':box,
                   'derivation':'static_union_not_model_output'})+'\n' for i in range(info['frames'])))
        candidates.append({'candidate_id':cid+'_reference','case_id':cid,'treatment':'reference',
                  'question':rows[0]['question'],'frames':[{k:row[k] for k in ('path','timestamp_s','sha256')} for row in rows],
                  'source_sha256':digest(source),'target_bytes':None})
        for budget in cfg.get('case_target_container_bytes',{}).get(cid,cfg['target_container_bytes']):
            pair=[]
            for mode in ['uniform','roi']:
                print(f'Encoding {cid} {budget} bytes {mode}',flush=True)
                encoded=encode_pair_member(ff,source,cm,logs,cfg,info,box,mode,budget)
                key=f'{cid}_{budget}_{mode}'
                frames,decoded_info=[],None
                if encoded['selected'] is not None:
                    out=q.local_path(encoded['selected']['path'])
                    decode_logs=logs/f'{budget}_{mode}_decode'
                    decode_logs.mkdir()
                    frames=extract_frames(ff,out,cm/(key+'_frames'),cfg['sample_times_s'],decode_logs)
                    # Verify whole output decodes, with exactly the source number of frames and same CFR.
                    decoded_info=frame_info(ff,out,decode_logs)
                    if decoded_info['frames']!=info['frames'] or decoded_info['fps']!=info['fps']:
                        raise ValueError('Encoded frame count or cadence changed')
                row={'candidate_id':key,'case_id':cid,'treatment':mode,'target_bytes':budget,
                     'question':rows[0]['question'],'frames':frames,'encoding':encoded,
                     'source_sha256':digest(source),'decoded_info':decoded_info}
                candidates.append(row)
                pair.append(row)
                q.write_json(logs/f'{budget}_{mode}.json',row)
                q.write_json(run/'candidates.json',candidates)
            actual=[r['encoding']['selected']['bytes'] if r['encoding']['selected'] else None for r in pair]
            q.write_json(logs/f'{budget}_comparison.json',{'case_id':cid,'target_bytes':budget,
                         'qualifies':all(r['encoding']['within_tolerance'] for r in pair),
                         'actual_uniform_bytes':actual[0], 'actual_roi_bytes':actual[1],
                         'maximum_pair_difference_fraction':abs(actual[0]-actual[1])/budget if all(v is not None for v in actual) else None})
    q.write_json(run/'candidates.json',candidates)
    q.write_json(run/'summary.json',{'cases':len(cfg['case_ids']),'encoded_candidates':sum(r['treatment']!='reference' for r in candidates),
                 'reference_candidates':sum(r['treatment']=='reference' for r in candidates),
                 'matched_candidates':sum(r['treatment']!='reference' and r['encoding']['within_tolerance'] for r in candidates),
                 'completed_at':q.now(),'roi_policy':cfg['roi_policy']})
    print(json.dumps({'run':str(run),'candidates':len(candidates)}),flush=True)


if __name__=='__main__':
    main()
