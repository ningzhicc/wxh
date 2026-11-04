#!/usr/bin/env python3
"""CPU-only, resumable full-devset CRF x ROI-offset encoding and actual-size table."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
import time

from PIL import Image, ImageChops, ImageStat, __version__ as pillow_version

import qwen_roi as q
import encode_roi_smoke as codec

ROOT=q.ROOT
SCRIPT=Path(__file__).read_bytes()


def sample_indices(times, fps, count):
    indices=[math.ceil(Fraction(str(t))*fps) for t in times]
    if len(set(indices))!=len(indices) or not all(0<=i<count for i in indices):
        raise ValueError('Invalid sampling positions for this CFR reference')
    return indices


def decode_samples(ffmpeg, source, directory, indices):
    """One full decode: all-frame hashes plus exactly six native RGB PNG samples."""
    directory.mkdir(exist_ok=False)
    expression='+'.join(f'eq(n\\,{i})' for i in indices)
    graph=f'[0:v]split=2[all][sample];[sample]select={expression}[selected]'
    cmd=[str(ffmpeg),'-hide_banner','-nostdin','-n','-threads','2','-i',str(source),
         '-filter_complex',graph,'-map','[all]','-c:v','rawvideo','-threads','1',
         '-fps_mode','passthrough','-f','framemd5',str(directory/'decoded.framemd5'),
         '-map','[selected]','-c:v','png','-threads','1','-fps_mode','vfr','-start_number','0',
         str(directory/'frame_%02d.png')]
    codec.command(cmd,directory/'decode.log')
    text=(directory/'decoded.framemd5').read_text()
    lines=text.splitlines()
    count=sum(bool(l) and not l.startswith('#') for l in lines)
    tb=Fraction(next(l.split(':',1)[1].strip() for l in lines if l.startswith('#tb 0:')))
    entries=[[int(v.strip()) for v in l.split(',')[:5]] for l in lines if l and not l.startswith('#')]
    pts=[v[2] for v in entries]
    if pts[0]!=0 or len({b-a for a,b in zip(pts,pts[1:])})!=1:
        raise ValueError('Decoded video is not zero-based CFR')
    fps=1/(tb*(pts[1]-pts[0]))
    dims=next(l.split(':',1)[1].strip() for l in lines if l.startswith('#dimensions 0:'))
    frames=sorted(directory.glob('frame_*.png'))
    if len(frames)!=len(indices): raise ValueError('Incorrect selected frame count')
    return {'frame_count':count,'fps':str(fps),'dimensions':dims,
            'framemd5_sha256':codec.digest(directory/'decoded.framemd5')},frames


def error(a,b):
    if a.size!=b.size: raise ValueError('Decoded dimensions differ')
    n=a.width*a.height*3
    return sum(v*v for v in ImageStat.Stat(ImageChops.difference(a,b)).rms)*(n/3),n


def psnr(e,n):
    return 10*math.log10(255**2/(e/n)) if n and e>0 else None


def quality(frames, source_frames, box, roi_rows):
    total_e=total_n=static_e=static_n=evidence_e=evidence_n=0
    for frame,source,roi in zip(frames,source_frames,roi_rows,strict=True):
        with Image.open(frame) as ca, Image.open(source) as re:
            ca,re=ca.convert('RGB'),re.convert('RGB')
            e,n=error(ca,re);total_e+=e;total_n+=n
            e,n=error(ca.crop(box),re.crop(box));static_e+=e;static_n+=n
            for region in roi['regions']:
                b=region['bbox_pixels'];e,n=error(ca.crop(b),re.crop(b));evidence_e+=e;evidence_n+=n
    return {'sampled_full_frame_rgb_psnr_db':psnr(total_e,total_n),
            'sampled_static_roi_rgb_psnr_db':psnr(static_e,static_n),
            'sampled_background_rgb_psnr_db':psnr(max(0,total_e-static_e),total_n-static_n),
            'sampled_api_visible_evidence_rgb_psnr_db':psnr(evidence_e,evidence_n),
            'measurement_note':'Six-frame pixel-weighted RGB MSE; API evidence masks are not human gold. null PSNR denotes zero MSE or empty mask.'}


def encode_one(plan, cfg, run, media, ffmpeg):
    cid=plan['case_id'];key=plan['candidate_id']
    logs=run/'candidates'/key
    dest=media/key
    if logs.exists() or dest.exists(): raise ValueError('Unrecorded/partial candidate exists: '+key)
    logs.mkdir(parents=True);dest.mkdir()
    q.write_json(logs/'status.json',{'state':'encoding','started_at':q.now()})
    x1,y1,x2,y2=plan['roi_box_pixels']
    args=[]
    if Fraction(plan['roi_qoffset'])!=0:
        args=['-vf',f"addroi=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:qoffset={plan['roi_qoffset']}"]
    out=dest/'video.mp4'
    cmd=[str(ffmpeg),'-hide_banner','-nostdin','-n','-threads','2','-i',str(q.local_path(plan['source_path'])),
         '-map','0:v:0','-an','-sn','-dn',*args,'-c:v',cfg['codec'],'-preset',cfg['preset'],
         '-pix_fmt',cfg['pixel_format'],'-threads',str(cfg['encoder_threads']),'-crf',str(plan['global_crf']),
         '-aq-mode',str(cfg['aq_mode']),'-aq-strength','1','-x264-params',f"mbtree={cfg['mbtree']}",
         '-g','60','-keyint_min','30','-frames:v',str(plan['frame_info']['frames']),
         '-r',plan['frame_info']['fps'],'-fps_mode','cfr','-movflags','+faststart',str(out)]
    started=time.monotonic()
    codec.command(cmd,logs/'encode.log')
    encode_s=time.monotonic()-started
    info,frames=decode_samples(ffmpeg,out,dest/'decoded',plan['sample_indices'])
    expected=plan['frame_info']
    if (info['frame_count']!=expected['frames'] or info['fps']!=expected['fps']
            or info['dimensions']!=f"{plan['width']}x{plan['height']}"):
        raise ValueError('Encoded output changed dimensions, frame count or cadence')
    metrics=quality(frames,[q.local_path(f['path']) for f in plan['source_frames']],plan['roi_box_pixels'],plan['sampled_api_rois'])
    duration=expected['duration_s'];size=out.stat().st_size
    result={k:plan[k] for k in ['candidate_id','case_id','question','global_crf','roi_qoffset','mode','roi_box_pixels','roi_area_fraction','roi_review_flag','source_sha256']}
    result.update(size_bytes=size,actual_bitrate_kbps=size*8/duration/1000,duration_s=duration,
         video_path=str(out.relative_to(ROOT)),video_sha256=codec.digest(out),encoding_seconds=encode_s,
         frame_info=info,metrics=metrics,
         frames=[{'path':str(p.relative_to(ROOT)),'timestamp_s':t,'frame_index':i,'sha256':codec.digest(p)}
                 for p,t,i in zip(frames,cfg['sample_times_s'],plan['sample_indices'],strict=True)])
    q.write_json(logs/'result.json',result)
    q.write_json(logs/'status.json',{'state':'complete','finished_at':q.now(),'result_sha256':codec.digest(logs/'result.json')})
    return result


def verify_cached(run, plan):
    logs=run/'candidates'/plan['candidate_id']
    if not logs.exists(): return None
    status=json.loads((logs/'status.json').read_text())
    if status['state']!='complete': raise ValueError('Candidate incomplete; inspect before retry: '+plan['candidate_id'])
    if codec.digest(logs/'result.json')!=status['result_sha256']: raise ValueError('Candidate metadata changed')
    r=json.loads((logs/'result.json').read_text())
    if codec.digest(q.local_path(r['video_path']))!=r['video_sha256']: raise ValueError('Encoded MP4 changed')
    for f in r['frames']:
        if codec.digest(q.local_path(f['path']))!=f['sha256']: raise ValueError('Decoded PNG changed')
    return r


def publish(run, plans, results, cfg):
    rows=sorted(results.values(),key=lambda r:r['candidate_id'])
    q.write_json(run/'candidates.json',rows)
    fields=['candidate_id','case_id','global_crf','roi_qoffset','mode','size_bytes','actual_bitrate_kbps',
            'duration_s','roi_area_fraction','video_path','video_sha256',
            'evidence_rgb_psnr_db','background_rgb_psnr_db','roi_review_flag']
    with (run/'rate_size_quality.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r[k] for k in fields if k in r},
                'evidence_rgb_psnr_db':r['metrics']['sampled_api_visible_evidence_rgb_psnr_db'],
                'background_rgb_psnr_db':r['metrics']['sampled_background_rgb_psnr_db']})
    complete=len(rows)==len(plans)
    summary={'complete':complete,'planned':len(plans),'completed':len(rows),'case_count':len(cfg['case_ids']),
             'candidates_per_case':{cid:sum(r['case_id']==cid for r in rows) for cid in cfg['case_ids']},
             'source_decoded_frames':sum(p['frame_info']['frames'] for p in plans[::len(cfg['global_crf_levels'])*len(cfg['roi_qoffsets'])]),
             'decoded_candidate_frames':sum(r['frame_info']['frame_count'] for r in rows),
             'updated_at':q.now(),'training_started':False,'gpu_used':False,'api_calls':0}
    q.write_json(run/'summary.json',summary)
    if not complete: return
    for cid in cfg['case_ids']:
        group=[r for r in rows if r['case_id']==cid]
        for offset in cfg['roi_qoffsets']:
            sizes=[r['size_bytes'] for r in group if r['roi_qoffset']==offset]
            if len(set(sizes))!=len(cfg['global_crf_levels']): raise ValueError('Quality levels did not produce distinct byte sizes')
    actions={cid:[{**r,'action_id':f'a{i:02d}'} for i,r in enumerate(rows) if r['case_id']==cid]
             for cid in cfg['case_ids']}
    q.write_json(run/'actions.json',actions)
    lines=[f"# {len(cfg['case_ids'])} 视频编码候选表",'',
           f"使用单码流 QP 区域控制路线：每个视频 {len(cfg['global_crf_levels'])} 个全局 CRF 档位 × {len(cfg['roi_qoffsets'])} 个 ROI 偏移。完整 MP4 实测字节和码率，不拆分区域码流。",'',
           '| 视频 | 候选数 | 实测码率范围 kbps | 完整片段大小范围 bytes | ROI 占比 | 来源框标记 |',
           '|---|---:|---:|---:|---:|---|']
    for cid in cfg['case_ids']:
        ss=[r for r in rows if r['case_id']==cid]
        lines.append(f"| {cid} | {len(ss)} | {min(r['actual_bitrate_kbps'] for r in ss):.2f}–{max(r['actual_bitrate_kbps'] for r in ss):.2f} | {min(r['size_bytes'] for r in ss)}–{max(r['size_bytes'] for r in ss)} | {ss[0]['roi_area_fraction']:.2%} | {ss[0]['roi_review_flag'] or '无新增标记'} |")
    lines+=['','[全部实测表 CSV](rate_size_quality.csv)、[候选完整元数据](candidates.json)、[按视频分组的动作表](actions.json)。','',
            '## 参数与范围','',
            f"- 全局 CRF：{cfg['global_crf_levels']}；ROI qoffset：{cfg['roi_qoffsets']}，0 是无显式 ROI 的基线。",
            f"- x264 AQ=1、mbtree=0、相同 preset；本配置的 8-bit ROI 偏移请求约为 {[round(float(Fraction(x))*51,1) for x in cfg['roi_qoffsets']]} QP，实际量化由编码器决定。CRF 不是固定背景 QP。",
            '- 码率是实际完整 MP4 字节×8÷解码帧数对应时长，含容器开销。本表不是固定目标码率或等字节公平比较实验。',
            '- 每个候选均完整解码核对帧数、尺寸和 CFR，并导出六张采样帧。采样帧序号先在原参考片上验证与既有证据 PNG 一致。',
            '- ROI 是已有模型采样框的静态并集，8像素扩边并按16像素宏块向外对齐。没有人工修框，也未将其声称为连续目标跟踪。',
            '- 足球主体关联、显卡后续镜头、远处限高牌等来源问题仍明确标记；编码成功不代表语义定位一定正确。',
            '- 质量数值是采样 RGB PSNR，分别记录证据区域、固定保护区、背景和全画面；不是问答成功率。',
            '- 所有视频和 PNG 放在 artifacts/ 对应运行目录；本目录保留命令、输入哈希、响应框来源和代码快照。',
            '- 本轮只用 CPU 编码与数据准备，未调用多模态 API，也未启动 GPU 训练。']
    (run/'REVIEW.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/roi_encoding_dev10_grid_v1.json')
    p.add_argument('--run-id',default='roi_encoding_dev10_grid_v1')
    p.add_argument('--limit',type=int,default=None)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id) or (a.limit is not None and a.limit<0): p.error('Invalid run id or limit')
    cfgpath=q.local_path(a.config);cfg=json.loads(cfgpath.read_text())
    run=q.local_path('runs/'+a.run_id);media=q.local_path('artifacts/'+a.run_id)
    roirun=q.local_path('runs/'+cfg['roi_run']);ffmpeg=ROOT/'.tools/ffmpeg'
    fingerprint={'config_sha256':codec.digest(cfgpath),'code_sha256':q.sha(SCRIPT),
                 'codec_helper_sha256':codec.digest(Path(codec.__file__)),'metadata_helper_sha256':codec.digest(Path(q.__file__)),
                 'ffmpeg_sha256':codec.digest(ffmpeg),'roi_results_sha256':codec.digest(roirun/'results.jsonl'),
                 'roi_inputs_sha256':codec.digest(roirun/'inputs.json')}
    if run.exists():
        if json.loads((run/'manifest.json').read_text())['fingerprint']!=fingerprint:
            raise ValueError('Source/config/code changed; choose a new run id')
        plans=json.loads((run/'plan.json').read_text())
    else:
        run.mkdir();media.mkdir()
        (run/'code').mkdir()
        for path in [Path(__file__),Path(codec.__file__),Path(q.__file__)]:
            (run/'code'/path.name).write_bytes(path.read_bytes())
        (run/'config.json').write_bytes(cfgpath.read_bytes())
        q.write_json(run/'manifest.json',{'created_at':q.now(),'fingerprint':fingerprint,'kind':'full_dev10_crf_roi_grid',
                   'pillow_version':pillow_version,'label_files_read':False,'gpu_training_requested':False})
        roi_rows=[json.loads(l) for l in (roirun/'results.jsonl').read_text().splitlines()]
        inputs=json.loads((roirun/'inputs.json').read_text());plans=[]
        for cid in cfg['case_ids']:
            case_dir=run/cid;case_dir.mkdir()
            rows=sorted([r for r in inputs if r['case_id']==cid],key=lambda r:r['frame_id'])
            rs=sorted([r for r in roi_rows if r['case_id']==cid],key=lambda r:r['frame_id'])
            width,height=rows[0]['width'],rows[0]['height']
            source=ROOT/'artifacts/pilot_dev10_v1'/cid/'reference_lossless.mp4'
            receipt=json.loads((ROOT/'datasets/pilot_dev10_v1/provenance/preparation'/f'{cid}.json').read_text())
            recorded=next(f for f in receipt['files'] if f['path'].endswith('reference_lossless.mp4'))
            if codec.digest(source)!=recorded['sha256']: raise ValueError('Reference changed')
            info=codec.frame_info(ffmpeg,source,case_dir)
            indices=sample_indices(cfg['sample_times_s'],Fraction(info['fps']),info['frames'])
            original_check,original_frames=decode_samples(ffmpeg,source,media/(cid+'_reference_samples'),indices)
            for path,row in zip(original_frames,rows,strict=True):
                if codec.digest(path)!=row['sha256'] or codec.digest(q.local_path(row['path']))!=row['sha256']:
                    raise ValueError('Sampling indices differ from original evidence: '+cid)
            box=codec.union_box(rs,width,height,cfg['roi_padding_pixels'],cfg['macroblock_size'])
            area=(box[2]-box[0])*(box[3]-box[1])/width/height
            q.write_json(case_dir/'roi_policy.json',{'policy':cfg['roi_policy'],'bbox_pixels':box,'area_fraction':area,
                 'source_frame_info':info,'sample_indices':indices,'api_source_frames':rows,
                 'source_review_flag':cfg['roi_review_flags'].get(cid,'')})
            (case_dir/'frame_rois.jsonl').write_text(''.join(json.dumps({'frame_index':i,'bbox_pixels':box,
                   'derivation':'static_union_of_sampled_api_boxes'})+'\n' for i in range(info['frames'])))
            for crf in cfg['global_crf_levels']:
                for offset_index,offset in enumerate(cfg['roi_qoffsets']):
                    plans.append({'candidate_id':f'{cid}_crf{crf}_roi{offset_index}','case_id':cid,'question':rows[0]['question'],
                         'global_crf':crf,'roi_qoffset':offset,'mode':'uniform' if Fraction(offset)==0 else 'roi',
                         'source_path':str(source.relative_to(ROOT)),'source_sha256':recorded['sha256'],
                         'frame_info':info,'width':width,'height':height,'roi_box_pixels':box,'roi_area_fraction':area,
                         'roi_review_flag':cfg['roi_review_flags'].get(cid,''),'sample_indices':indices,
                         'source_frames':rows,'sampled_api_rois':rs})
            print(f"Planned {cid}: {len(cfg['global_crf_levels'])*len(cfg['roi_qoffsets'])} candidates, source sampling verified",flush=True)
        q.write_json(run/'plan.json',plans)
        q.write_json(run/'plan.sha256.json',{'sha256':codec.digest(run/'plan.json')})
    if codec.digest(run/'plan.json')!=json.loads((run/'plan.sha256.json').read_text())['sha256']:
        raise ValueError('Plan changed')
    for plan in plans:
        if codec.digest(q.local_path(plan['source_path']))!=plan['source_sha256']: raise ValueError('Reference changed')
    results={};todo=[]
    for plan in plans:
        cached=verify_cached(run,plan)
        if cached: results[plan['candidate_id']]=cached
        else: todo.append(plan)
    todo=todo[:a.limit]
    publish(run,plans,results,cfg)
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures={pool.submit(encode_one,plan,cfg,run,media,ffmpeg):plan['candidate_id'] for plan in todo}
        for future in as_completed(futures):
            result=future.result();results[result['candidate_id']]=result
            publish(run,plans,results,cfg)
            print(f"Encoded {len(results)}/{len(plans)}: {result['candidate_id']} {result['size_bytes']} bytes",flush=True)
    print(json.dumps({'completed':len(results),'planned':len(plans),'run':str(run)}),flush=True)


if __name__=='__main__': main()
