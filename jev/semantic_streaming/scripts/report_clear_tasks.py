#!/usr/bin/env python3
"""Independent sealed-study review: task answers, policy calls, bytes and startup."""
import argparse
import json
from pathlib import Path
import qwen_roi as q
from pipeline_media import file_sha
from run_semantic_pipeline import seal_paths,verify_files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='clear_task_regression3_v1');p.add_argument('--run-id',required=True);a=p.parse_args()
    source=q.local_path('runs/'+a.source);verify_files(json.loads((source/'files.sha256.json').read_text()))
    for name,digest in json.loads((source/'child_seals.json').read_text()).items():
        child=q.local_path('runs/'+name)
        if file_sha(child/'files.sha256.json')!=digest:raise ValueError('Child seal changed')
        verify_files(json.loads((child/'files.sha256.json').read_text()))
    results=json.loads((source/'policy_results.json').read_text());summary=json.loads((source/'summary.json').read_text())
    scores=json.loads((source/'qa_scores.json').read_text());stats={(r['case_id'],r['variant']):r for r in scores['groups']}
    frames={(r['case_id'],r['variant']):r['frames'] for r in json.loads((source/'qa_plan.json').read_text())}
    cases=[]
    for cid,group in results.items():
        policy={}
        for name in ('bandwidth','kev'):
            directory=q.local_path('runs/'+group['source_run' if name=='bandwidth' else 'kev_run'])
            choices=[r['selection'] for r in map(json.loads,(directory/'decisions.jsonl').read_text().splitlines())]
            policy[name]={k:group[name][k] for k in ('selected_bytes','rebuffer_s','startup_delay_s','frames','model_calls','forced_chunks','prediction_fallbacks','encoded_files','catalog_slots')}
            policy[name]['selected_chunks']=choices
        same_wire=stats[(cid,'bandwidth')]['payload_sha256']==stats[(cid,'kev')]['payload_sha256']
        same_answers=stats[(cid,'bandwidth')]['answer_histogram']==stats[(cid,'kev')]['answer_histogram']
        changed_indices=[i for i,(b,k) in enumerate(zip(frames[(cid,'bandwidth')],frames[(cid,'kev')],strict=True)) if b['sha256']!=k['sha256']]
        cited={v:sorted({i for r in scores['results'] if r['case_id']==cid and r['variant']==v for i in r['answer']['evidence_frame_indices']}) for v in ('bandwidth','kev')}
        cases.append({'BW_Kev_changed_QA_frame_indices':changed_indices,'API_reported_evidence_indices':cited,'case_id':cid,'QA':{v:stats[(cid,v)] for v in ('reference','bandwidth','kev')},'policies':policy,
            'kev_byte_saving_vs_bandwidth':1-policy['kev']['selected_bytes']/policy['bandwidth']['selected_bytes'],
            'BW_Kev_first_frame_PNG_identical':frames[(cid,'bandwidth')][0]['sha256']==frames[(cid,'kev')][0]['sha256'],
            'BW_Kev_full_QA_payload_identical':same_wire,'different_QA_histograms_on_identical_payload':same_wire and not same_answers,
            'startup_selected_file_identical':policy['bandwidth']['selected_chunks'][0]['video_path']==policy['kev']['selected_chunks'][0]['video_path'],
            'first_frame_temporal_violations':sum(not r['temporal_anchor_ok'] for r in scores['results'] if r['case_id']==cid)})
    totals=summary['policy_totals'];saving=1-totals['kev']['selected_bytes']/totals['bandwidth']['selected_bytes']
    out=q.local_path('runs/'+a.run_id)
    if out.exists():raise ValueError('Use a new review ID')
    out.mkdir();result={'source_run':a.source,'source_seal_sha256':file_sha(source/'files.sha256.json'),
        'cases':cases,'totals':totals,'overall_Kev_byte_saving':saving,'ROI_calls':summary['ROI_calls'],'QA_calls':summary['QA_calls'],
        'spatial_accuracy_not_quantified':True,'previously_used_development_data':True,'training_started':False}
    q.write_json(out/'analysis.json',result)
    lines=['# 三案例回归：结果与边界','','固定Kev、编码与网络的三步流程已完成。开放问答每个问题/视频重复3次，以下次数按原参考答案和首帧条件评分。数据都是已用开发视频。','',
        '| 案例 | 原视频 | 带宽 | Kev | 带宽字节 | Kev字节 | Kev省字节 | Kev模型调用 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in cases:
        cid=row['case_id'];bw,k=row['policies']['bandwidth'],row['policies']['kev']
        counts=[f"{row['QA'][v]['correct']}/3" for v in ('reference','bandwidth','kev')]
        lines.append(f"| {cid} | {' | '.join(counts)} | {bw['selected_bytes']} | {k['selected_bytes']} | {row['kev_byte_saving_vs_bandwidth']:.2%} | {k['model_calls']} |")
    lines+=['',f"合计带宽 {totals['bandwidth']['selected_bytes']} 字节，Kev {totals['kev']['selected_bytes']} 字节，省 {saving:.2%}；卡顿分别 {totals['bandwidth']['rebuffer_s']} / {totals['kev']['rebuffer_s']} 秒。",'',
        f"实际新定位 {summary['ROI_calls']} 次、开放QA {summary['QA_calls']} 次；没有训练。",'',
        '全部案例的起播文件在两策略间相同，后续块才实际选档。计时器问第一帧，若首帧被低质量起播编码损伤，后续Kev选档不能重写已传的首帧；本次不为提高分数改变该策略。',
        '问答可从未降码的其他帧找回答案，答对本身不能证明所有降码帧保持了可读性。结构化报告保留两策略图片不同的索引和QA自报证据索引。',
        '每组原始回答、模型回退、强制选档和接收核验均保留。没有空间密集GT的这三例，不宣称得到ROI空间准确率；其区域框可按源码运行中overlays逐帧复查。',
        '结论仅适用于当前3个开发视频和1条轨迹。重复3次不是3个独立视频；同一Qwen服务作为定位器和评测器，仍需要独立评测。','']
    for row in cases:
        lines += [f"## {row['case_id']}",'']
        for variant,item in row['QA'].items():lines.append(f"{variant}：{item['answer_histogram']}；3次正文SHA {item['payload_sha256']}。")
        lines+=['',f"两策略首帧PNG相同：{row['BW_Kev_first_frame_PNG_identical']}；首帧条件违规次数：{row['first_frame_temporal_violations']}。",'']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'code.py').write_bytes(Path(__file__).read_bytes());q.write_json(out/'files.sha256.json',seal_paths([out]));print(json.dumps({'report':str(out/'REPORT.md'),'summary':result},ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
