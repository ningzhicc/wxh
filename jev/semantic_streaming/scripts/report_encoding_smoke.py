#!/usr/bin/env python3
"""Audit encoded candidates and blind QA; calculate sampled RGB evidence PSNR."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageStat

import qwen_roi as q
import receiver_qa


def squared_error(a,b):
    if a.size!=b.size:
        raise ValueError('Reference and candidate dimensions differ')
    difference=ImageChops.difference(a,b)
    pixels=a.width*a.height
    return sum(v*v for v in ImageStat.Stat(difference).rms)*pixels, pixels*3


def psnr(squared, samples):
    return 10*math.log10(255**2/(squared/samples)) if squared>0 and samples>0 else None


def metrics(candidate, reference, api_rois):
    full_error=full_samples=roi_error=roi_samples=0
    for i,(f,ref) in enumerate(zip(candidate['frames'],reference['frames'])):
        for item in [f,ref]:
            if q.sha(q.local_path(item['path']).read_bytes())!=item['sha256']:
                raise ValueError('Frame hash mismatch')
        with Image.open(q.local_path(f['path'])) as ci, Image.open(q.local_path(ref['path'])) as ri:
            ci,ri=ci.convert('RGB'),ri.convert('RGB')
            e,n=squared_error(ci,ri);full_error+=e;full_samples+=n
            for box in api_rois[f"{candidate['case_id']}_f{i:02d}"]['regions']:
                rect=box['bbox_pixels']
                e,n=squared_error(ci.crop(rect),ri.crop(rect));roi_error+=e;roi_samples+=n
    return {'sampled_full_frame_rgb_psnr_db':psnr(full_error,full_samples),
            'sampled_api_visible_evidence_rgb_psnr_db':psnr(roi_error,roi_samples),
            'evidence_rgb_samples':roi_samples,
            'note':'RGB PNG-space pixel-weighted MSE, then PSNR. API boxes on nonempty sampled frames only; not human evidence masks or a semantic metric.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoding-run',required=True)
    p.add_argument('--qa-run',required=True)
    a=p.parse_args()
    run=q.local_path('runs/'+a.encoding_run)
    qa=q.local_path('runs/'+a.qa_run)
    manifest=json.loads((run/'manifest.json').read_text())
    cfg=json.loads((run/'config.json').read_text())
    assert q.sha((run/'code/encode_roi_smoke.py').read_bytes())==manifest['script_sha256']
    assert q.sha((run/'config.json').read_bytes())==manifest['config_sha256']
    assert q.sha((q.ROOT/'.tools/ffmpeg').read_bytes())==manifest['ffmpeg_sha256']
    cs=json.loads((run/'candidates.json').read_text())
    references={c['case_id']:c for c in cs if c['treatment']=='reference'}
    roirun=q.local_path('runs/'+cfg['roi_run'])
    assert q.sha((roirun/'results.jsonl').read_bytes())==manifest['roi_results_sha256']
    assert q.sha((roirun/'inputs.json').read_bytes())==manifest['roi_inputs_sha256']
    api_rois={r['frame_id']:r for r in map(json.loads,(roirun/'results.jsonl').read_text().splitlines())}
    measured={}
    for c in cs:
        if c['treatment']=='reference' or c['encoding']['selected'] is None:
            continue
        selected=c['encoding']['selected']
        path=q.local_path(selected['path'])
        assert path.stat().st_size==selected['bytes'] and q.sha(path.read_bytes())==selected['sha256']
        assert len(c['frames'])==len(references[c['case_id']]['frames'])
        measured[c['candidate_id']]=metrics(c,references[c['case_id']],api_rois)

    qam=json.loads((qa/'manifest.json').read_text())
    assert qam['encoding_run']==a.encoding_run
    assert q.sha((run/'candidates.json').read_bytes())==qam['fingerprint']['candidate_list_sha256']
    assert q.sha((qa/'code/receiver_qa.py').read_bytes())==qam['fingerprint']['script_sha256']
    assert q.sha((qa/'prompt.txt').read_bytes())==qam['fingerprint']['prompt_sha256']
    assert q.sha((qa/'code/qwen_roi.py').read_bytes())==qam['fingerprint']['helper_sha256']
    plan=json.loads((qa/'plan.json').read_text())
    config=json.loads((qa/'config.json').read_text())
    usages=[]
    qa_latencies=[]
    for c in plan:
        d=qa/'calls'/c['blind_id']
        s=json.loads((d/'status.json').read_text())
        assert s['state']=='complete'
        raw=(d/'response.json').read_bytes()
        assert q.sha(raw)==s['response_sha256']
        payload=receiver_qa.payload(config,(qa/'prompt.txt').read_text().strip(),c)
        wire=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
        assert q.sha(wire)==json.loads((d/'request.json').read_text())['payload_sha256']
        assert receiver_qa.parse_answer(json.loads(raw),len(c['frames']))==json.loads((d/'answer.json').read_text())
        usages.append(s['usage']);qa_latencies.append(s['elapsed_s'])
    scores=json.loads((qa/'scores.json').read_text())
    label_path=q.ROOT/'datasets/pilot_dev10_v1/annotations/answers.jsonl'
    assert q.sha(label_path.read_bytes())==scores['labels_sha256']
    labels={r['case_id']:r for r in map(json.loads,label_path.read_text().splitlines())}
    assert len(scores['results'])==len(plan)
    plan_by_id={c['candidate_id']:c for c in plan}
    for s in scores['results']:
        c=plan_by_id[s['candidate_id']]
        answer=json.loads((qa/'calls'/c['blind_id']/'answer.json').read_text())
        assert s['qa']==answer
        expected=receiver_qa.normalize_answer(answer['answer']) in {
            receiver_qa.normalize_answer(v) for v in labels[c['case_id']]['accepted_answers']}
        anchor_ok='第一帧' not in c['question'] or 0 in answer['evidence_frame_indices']
        assert s['reference_match']==(expected and not answer['unanswerable'] and anchor_ok)
    by_id={s['candidate_id']:s for s in scores['results']}
    reference_scores={cid:by_id[r['candidate_id']] for cid,r in references.items()}
    pairs=[]
    for cid in cfg['case_ids']:
        for budget in cfg.get('case_target_container_bytes',{}).get(cid,cfg['target_container_bytes']):
            modes={c['treatment']:c for c in cs if c['case_id']==cid and c['target_bytes']==budget}
            valid=all(c['encoding']['within_tolerance'] for c in modes.values()) and set(modes)=={'uniform','roi'}
            result={'case_id':cid,'target_bytes':budget,'qualifies':valid,
                    'reference_qa_match':reference_scores[cid]['reference_match'],'treatments':{}}
            for mode,c in modes.items():
                selected=c['encoding']['selected']
                result['treatments'][mode]={'actual_bytes':selected['bytes'] if selected else None,
                         'within_tolerance':c['encoding']['within_tolerance'],
                         'metrics':measured.get(c['candidate_id']),
                         'qa':by_id.get(c['candidate_id'])}
            if valid:
                u,r=result['treatments']['uniform'],result['treatments']['roi']
                result['pair_size_difference_fraction']=abs(u['actual_bytes']-r['actual_bytes'])/budget
                result['evidence_psnr_gain_db']=r['metrics']['sampled_api_visible_evidence_rgb_psnr_db']-u['metrics']['sampled_api_visible_evidence_rgb_psnr_db']
            pairs.append(result)
    valid=[p for p in pairs if p['qualifies']]
    improved=sum(not p['treatments']['uniform']['qa']['reference_match'] and p['treatments']['roi']['qa']['reference_match'] for p in valid)
    regressed=sum(p['treatments']['uniform']['qa']['reference_match'] and not p['treatments']['roi']['qa']['reference_match'] for p in valid)
    usage={k:sum(u.get(k,0) for u in usages) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    qa_cost=sum(u.get('prompt_tokens',0)*(1 if u.get('prompt_tokens',0)<=32768 else 1.5)+
                u.get('completion_tokens',0)*(10 if u.get('prompt_tokens',0)<=32768 else 15) for u in usages)/1e6
    report={'created_at':q.now(),'qualified_pairs':len(valid),'excluded_pairs':len(pairs)-len(valid),
            'qa_improved_pairs':improved,'qa_regressed_pairs':regressed,
            'uniform_reference_matches':sum(p['treatments']['uniform']['qa']['reference_match'] for p in valid),
            'roi_reference_matches':sum(p['treatments']['roi']['qa']['reference_match'] for p in valid),
            'original_reference_matches':sum(s['reference_match'] for s in reference_scores.values()),
            'pairs':pairs,'qa_usage':usage,'qa_list_price_estimate_cny':qa_cost,
            'qa_mean_request_latency_s':sum(qa_latencies)/len(qa_latencies),
            'reference_scores':reference_scores,
            'checks':['Encoder/config/reference/encoded-image hashes verified','Full selected videos decoded and source CFR counts checked by encoder runner',
                      'QA raw response hashes verified','All QA request hashes reconstructed','Only paired size-matched candidates compared'],
            'limitations':['3 selected development cases, one QA call per candidate, same model service as ROI',
                           'Static union of six sampled API boxes applied to every frame, not tracking or online causal ROI',
                           'Budget fitting uses two-pass offline encoding; cannot claim online low-latency encoder yet',
                           'Evidence PSNR uses API boxes and is not semantic accuracy; reference labels not human gold']}
    q.write_json(run/'comparison_report.json',report)
    outdir=q.local_path('artifacts/'+a.encoding_run)/'comparisons'
    outdir.mkdir(exist_ok=True)
    representative={'dev10_01':4,'dev10_07':0,'dev10_08':0}
    for result in valid:
        cid,budget=result['case_id'],result['target_bytes']
        i=representative[cid]
        box=json.loads((run/cid/'roi_policy.json').read_text())['box_pixels']
        selected=[references[cid]]+[next(c for c in cs if c['case_id']==cid and c['target_bytes']==budget and c['treatment']==m) for m in ['uniform','roi']]
        crops=[]
        for c in selected:
            with Image.open(q.local_path(c['frames'][i]['path'])) as im:
                crop=im.convert('RGB').crop(box)
                crop=crop.resize((crop.width*2,crop.height*2),Image.Resampling.NEAREST)
                crops.append(crop)
        width,height=crops[0].size
        sheet=Image.new('RGB',(width*3,height+40),'#202020');draw=ImageDraw.Draw(sheet)
        for j,(crop,label) in enumerate(zip(crops,['REFERENCE','UNIFORM','ROI'])):
            sheet.paste(crop,(j*width,40));draw.text((j*width+5,5),f'{label} | {cid} | target={budget}',fill='white')
        path=outdir/f'{cid}_{budget}.png';sheet.save(path)
        result['comparison_image']=str(path.relative_to(q.ROOT))
    # Save image paths after plots are generated.
    q.write_json(run/'comparison_report.json',report)
    lines=['# ROI 编码与接收端问答初步对照','','这是三例开发实验，使用本地真实编码大小和解码画面；没有开始训练 Kev/RLCD。','',
           f"有效匹配 {len(valid)} 组；预算不满足的 {len(pairs)-len(valid)} 组排除。原视频参考答案匹配 {report['original_reference_matches']}/3。",'',
           '| 案例 | 预算 KiB | 均匀实际字节 | ROI 实际字节 | 证据 RGB PSNR 增益 dB | 均匀问答匹配 | ROI 问答匹配 |',
           '|---|---:|---:|---:|---:|---|---|']
    for result in pairs:
        u,r=result['treatments']['uniform'],result['treatments']['roi']
        if result['qualifies']:
            lines.append(f"| {result['case_id']} | {result['target_bytes']/1024:g} | {u['actual_bytes']} | {r['actual_bytes']} | {result['evidence_psnr_gain_db']:+.2f} | {'是' if u['qa']['reference_match'] else '否'} | {'是' if r['qa']['reference_match'] else '否'} |")
        else:
            lines.append(f"| {result['case_id']} | {result['target_bytes']/1024:g} | {u['actual_bytes']} | {r['actual_bytes']} | 排除：预算不匹配 | 未比较 | 未比较 |")
    lines += ['',f"有效对照中，参考答案匹配：均匀 {report['uniform_reference_matches']}/{len(valid)}，ROI {report['roi_reference_matches']}/{len(valid)}；ROI 改善 {improved} 组、退步 {regressed} 组。重复码率档位不视为独立视频样本。",
              '',f"接收端 QA 共 {len(plan)} 次调用，输入 {usage['prompt_tokens']} tokens、输出 {usage['completion_tokens']} tokens；按北京公开单价估算约 {qa_cost:.6f} 元，不是实际账单。",'',
              '## 原视频对照与逐候选答案','']
    for cid,s in reference_scores.items():
        lines.append(f"- {cid} 原视频回答：`{s['qa']['answer']}`；参考：`{s['reference_answer']}`；匹配：{s['reference_match']}。")
    lines += ['',f'[全部逐候选答案](../{a.qa_run}/scores.json)；[完整数值与审计](comparison_report.json)。','',
              '## 原尺寸区域对照图','']
    for result in valid:
        rel=Path('../../')/result['comparison_image']
        lines.append(f"- {result['case_id']} / {result['target_bytes']/1024:g} KiB：[参考、均匀、ROI]("+rel.as_posix()+')。')
    lines += ['', '## 实验约定与局限','',
              '- 均匀表示无显式 ROI 的全画面编码；两组均启用 x264 AQ=1，mbtree=0，其余参数相同。',
              '- ROI 为六帧 API 框的静态并集，加 8 像素边距并按 16 像素宏块向外对齐。每个编码帧使用同一保护区；没有人工改框、没有声称完成跟踪。',
              '- ROI qoffset=-1/5 是相对量化偏移请求；8-bit x264 中约为 -10.2 QP，不是固定绝对 QP。',
              '- 目标是整个 MP4 文件字节，包括容器开销；每个候选误差须在 1% 内，成对大小差可能接近 2%，具体差值保存在 JSON。',
              '- 两遍编码会使用整段信息，属于预录视频离线基线。低预算触及编码下限或无法匹配时排除，不能用超预算结果宣称收益。',
              '- QA 只接收六张完整解码帧和问题，顺序打乱、隐藏编码方式、无 ROI 和答案提示；原图使用相同流程。',
              '- QA 使用同一个 Qwen 服务，开发标签未经用户人工确认，每个候选只调用一次。这些数值不是独立测试集上的系统性能结论。',
              '- 证据 PSNR 来自 API 框覆盖的采样像素，数值提高不自动意味着语义正确。',
              '- 未开始网络仿真或决策训练；需先扩大独立来源、固定评测模型并验证重复性。']
    (run/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    (run/'code/report_encoding_smoke.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ['qualified_pairs','qa_improved_pairs','qa_regressed_pairs','uniform_reference_matches','roi_reference_matches','original_reference_matches']}))


if __name__=='__main__':
    main()
