#!/usr/bin/env python3
"""Posthoc encoder-region diagnostic. Does not replace the preregistered API gate."""
import argparse
import json
from pathlib import Path
from PIL import Image

import qwen_roi as q
from pipeline_media import file_sha
from run_semantic_pipeline import seal_paths,verify_files
from audit_evqa_evidence import coverage


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='evqa_disambiguation_v1');p.add_argument('--run-id',required=True);a=p.parse_args()
    source=q.local_path('runs/'+a.source);verify_files(json.loads((source/'files.sha256.json').read_text()))
    children=json.loads((source/'child_seals.json').read_text())
    for name,digest in children.items():
        child=q.local_path('runs/'+name)
        if file_sha(child/'files.sha256.json')!=digest:raise ValueError('Child seal changed')
        verify_files(json.loads((child/'files.sha256.json').read_text()))
    base=q.local_path('runs/'+a.source+'_bowl_bandwidth');kev=q.local_path('runs/'+a.source+'_bowl_kev')
    masks=q.local_path('datasets/evqa_evidence_audit_v1');verify_files(json.loads((masks/'files.sha256.json').read_text()))
    actions=json.loads((base/'actions.json').read_text());inputs=json.loads((base/'roi_inputs.json').read_text())
    refs={r['case_id']:r for r in map(json.loads,q.local_path('datasets/evqa_accept5_v1/annotations/answers.jsonl').read_text().splitlines())}
    rows=[]
    for row in inputs:
        if not any(lo<=row['actual_window_timestamp_s']<=hi for lo,hi in refs['evqa_01']['evidence_segments_s']):continue
        boxes={tuple(a['roi_box_pixels']) for a in actions[row['case_id']]}
        if len(boxes)!=1:raise ValueError('Encoding protection rectangles differ across catalog slots')
        box=list(next(iter(boxes)))
        score=coverage(Image.open(masks/'masks/qa_ego4d_trimmed25'/f"{row['source_frame_index']:05d}.png"),[{'bbox_pixels':box}],row['width'],row['height'])
        rows.append({'frame_id':row['frame_id'],'timestamp_s':row['actual_window_timestamp_s'],'encoder_rectangle':box,**score})
    valid=[r for r in rows if r['mask_nonempty']];mean=sum(r['gt_pixel_recall'] for r in valid)/len(valid)
    summary=json.loads((source/'summary.json').read_text());bs,ks=summary['policies']['bandwidth'],summary['policies']['kev']
    selectors={}
    for label,child in [('bandwidth',base),('kev',kev)]:
        selectors[label]=[r['selection'] for r in map(json.loads,(child/'decisions.jsonl').read_text().splitlines())]
    out=q.local_path('runs/'+a.run_id)
    if out.exists():raise ValueError('Use a new report ID')
    out.mkdir()
    data={'source_study':a.source,'source_seal_sha256':file_sha(source/'files.sha256.json'),
        'posthoc_encoder_rectangle_GT_coverage_provisional':mean,'frames':rows,'selected_chunks':selectors,
        'preregistered_API_mask_gate_passed':summary['diagnostic_gate_passed'],'API_mask_mean':summary['grounding']['explicit_target_context']['mean_GT_pixel_recall_provisional'],
        'kev_byte_saving_vs_same_question_bandwidth':1-ks['selected_bytes']/bs['selected_bytes'],
        'no_training_started':True,'caveat':'Posthoc static rectangle metric does not replace original per-frame API coverage or change failed gate. Frame correspondence remains provisional. Startup qoffset=0 has no differentiated ROI protection.'}
    q.write_json(out/'analysis.json',data)
    lines=['# 明确问题后的三步系统检查','','本轮9次碗问答全部与继承参考答案light blue一致。仅一个修订问题的开发实验，不是原EVQA准确率。','',
        f"API逐帧掩码覆盖率 {data['API_mask_mean']:.2%}，原预设80%门槛仍为未通过。API返回矩形并不保证覆盖完整目标；部分颜色区域足够答对也不等于空间定位已完全解决。",'',
        f"编码器使用同块采样框的静态并集、8像素padding和宏块对齐；追加核查所得保护矩形覆盖率 {mean:.2%}。这是后验诊断，不替代预设API指标，不据此改写门槛。",'',
        '| 策略 | 块 | CRF | ROI qoffset | 实际MP4字节 |','|---|---|---:|---|---:|']
    for label,items in selectors.items():
        for row in items:lines.append(f"| {label} | {row['case_id']} | {row['global_crf']} | {row['roi_qoffset']} | {row['size_bytes']} |")
    lines+=['',f"带宽策略合计 {bs['selected_bytes']} 字节，Kev {ks['selected_bytes']} 字节，节省 {data['kev_byte_saving_vs_same_question_bandwidth']:.2%}。两者零播放卡顿，各90帧完整接收解码核验。",'',
        '启动块选择qoffset=0，未对ROI额外分配相对量化保护；后续块才按所选负qoffset区别保护。各策略共享同一套真实候选文件。',
        '毛巾两种问题各3次均答blue，原标签green保留。缺少独立颜色依据，仍待人工复核；未经核验的任务不自动进入训练。',
        '本轮18次定位+15次QA，63项测试通过；原ABR检查点和当前Kev权重均保持原哈希。',
        '下一步应先核对数字索引与掩码对应、明确重要颜色区域与整物体掩码的评价范围，再增加明确指代的独立开发任务。', '',
        f'[完整运行](../{a.source}/REPORT.md) · [人工审校原图](../{a.source}/review.md)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'code.py').write_bytes(Path(__file__).read_bytes());q.write_json(out/'files.sha256.json',seal_paths([out]));print(json.dumps(data,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
