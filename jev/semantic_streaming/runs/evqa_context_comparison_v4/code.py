#!/usr/bin/env python3
"""Sealed old/new comparison, including identical-payload QA disagreement flags."""
import argparse
import json
from pathlib import Path

import qwen_roi as q
from pipeline_media import file_sha
from run_semantic_pipeline import seal_paths, verify_files


def read_study(name):
    run=q.local_path('runs/'+name)
    verify_files(json.loads((run/'files.sha256.json').read_text()))
    reports=json.loads((run/'policy_results.json').read_text())
    for group in reports.values():
        for field in ('source_run','kev_run'):
            child=q.local_path('runs/'+group[field])
            if file_sha(child/'files.sha256.json') != group[field+'_seal_sha256']:
                raise ValueError('Child seal differs from study')
            verify_files(json.loads((child/'files.sha256.json').read_text()))
    scores={(r['case_id'],r['variant']):r for r in json.loads((run/'qa_scores.json').read_text())['results']}
    receipts={}
    for path in (run/'qa').glob('*/request.json'):
        receipt=json.loads(path.read_text());receipts[(receipt['case_id'],receipt['variant'])]=receipt
    for key, score in scores.items():
        score['predicted_text']=receipts[key]['inputs']['options'].get(score['answer']['choice'],'')
    return run,reports,scores,receipts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',default='evqa_accept5_v1');p.add_argument('--new',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    old,old_reports,old_scores,old_receipts=read_study(a.old)
    new,new_reports,new_scores,new_receipts=read_study(a.new)
    out=q.local_path('runs/'+a.run_id)
    if out.exists():raise ValueError('Use a fresh comparison ID')
    out.mkdir();cases=[];unchanged_qa_disagreements=[]
    for cid,group in new_reports.items():
        before=old_reports[cid]
        br,ar=[q.local_path('runs/'+g['source_run']) for g in (before,group)]
        bm,am=[json.loads((r/'manifest.json').read_text())['fingerprint'] for r in (br,ar)]
        if any(bm[k]!=am[k] for k in ('source_sha256','trace_sha256','ffmpeg_sha256','control_config_sha256')):
            raise ValueError('Media/network/control input changed')
        bc,ac=[json.loads((r/'config.json').read_text()) for r in (br,ar)]
        for key in bc:
            if key not in ('api_config','max_api_calls') and bc[key]!=ac[key]:
                raise ValueError('Unplanned config change: '+key)
        for name,digest in bm['code_sha256'].items():
            if am['code_sha256'][name]!=digest:
                raise ValueError('Frozen pipeline implementation changed: '+name)
        binputs,ainputs=[json.loads((r/'roi_inputs.json').read_text()) for r in (br,ar)]
        fields=('frame_id','question','sha256','timestamp_s','width','height')
        if [[r[k] for k in fields] for r in binputs]!=[[r[k] for k in fields] for r in ainputs]:
            raise ValueError('Localization sampling/question changed')
        qa={}
        for variant in ('reference','bandwidth','kev'):
            key=(cid,variant);b=old_scores[key];v=new_scores[key]
            same_wire=old_receipts[key]['payload_sha256']==new_receipts[key]['payload_sha256']
            qa[variant]={'old':b,'new':v,'identical_wire_payload':same_wire,
                'different_response_on_identical_wire':same_wire and b['predicted_text']!=v['predicted_text']}
            if qa[variant]['different_response_on_identical_wire']:unchanged_qa_disagreements.append({'case_id':cid,'variant':variant})
        policies={}
        for policy in ('bandwidth','kev'):
            b,v=before[policy],group[policy]
            policies[policy]={'old_bytes':b['selected_bytes'],'new_bytes':v['selected_bytes'],
                'byte_change':v['selected_bytes']-b['selected_bytes'],'old_rebuffer_s':b['rebuffer_s'],'new_rebuffer_s':v['rebuffer_s'],
                'playback_byte_identical_to_old':b['playback']['sha256']==v['playback']['sha256']}
        intra_same_wire=new_receipts[(cid,'bandwidth')]['payload_sha256']==new_receipts[(cid,'kev')]['payload_sha256']
        intra_disagreement=intra_same_wire and qa['bandwidth']['new']['predicted_text']!=qa['kev']['new']['predicted_text']
        cases.append({'same_new_policy_QA_payload':intra_same_wire,'different_QA_answer_on_same_new_policy_payload':intra_disagreement,'case_id':cid,'same_source_sampling_encoding_control_model':True,'QA':qa,'policies':policies,
            'new_policies_have_identical_playback':group['bandwidth']['playback']['sha256']==group['kev']['playback']['sha256']})
    summary=json.loads((new/'summary.json').read_text());totals=summary['policy_totals'];bw,kev=totals['bandwidth'],totals['kev']
    failures=[]
    for name in ('evqa_joint2_v2_evqa_01_bandwidth','evqa_joint2_v3_evqa_01_bandwidth'):
        path=q.local_path('runs/'+name+'/roi/joint_0000');failures.append({'run':name,'status':json.loads((path/'status.json').read_text()),
            'response_sha256':file_sha(path/'response.json'),'paid_calls':1})
    report={'old_study':a.old,'new_study':a.new,'old_seal_sha256':file_sha(old/'files.sha256.json'),
        'new_seal_sha256':file_sha(new/'files.sha256.json'),'cases':cases,'same_payload_QA_disagreements':unchanged_qa_disagreements,
        'new_summary':summary,'same_new_policy_payload_disagreements':[c['case_id'] for c in cases if c['different_QA_answer_on_same_new_policy_payload']],'kev_byte_saving_vs_same_ROI_bandwidth':1-kev['selected_bytes']/bw['selected_bytes'],
        'preserved_rejected_explorations':failures,'training_started':False,
        'limits':['Development sample, not independent final test','Same provider for localization and QA',
            'Full-frame whole-window original QA and labels unchanged','Quality rewards use changed ROI definitions and cannot be directly compared across localization versions',
            'GT masks have provisional frame correspondence; offline audit only']}
    q.write_json(out/'comparison.json',report)
    lines=['# 多帧上下文定位：固定系统对比','','固定原视频、采样、问题、选项、编码实现、控制器、网络轨迹和Kev权重。只替换定位上下文及提示词。', '',
        '| 视频 | 原视频QA（旧→新） | 带宽QA（旧→新） | Kev QA（旧→新） | 旧Kev字节 | 新Kev字节 |',
        '|---|---|---|---|---:|---:|']
    for c in cases:
        qa=c['QA'];vals=[f"{qa[v]['old']['correct']}→{qa[v]['new']['correct']}" for v in ('reference','bandwidth','kev')]
        lines.append(f"| {c['case_id']} | {' | '.join(vals)} | {c['policies']['kev']['old_bytes']} | {c['policies']['kev']['new_bytes']} |")
    lines+=['',f"新实验QA：{summary['QA_correct']}；Kev相对同一套ROI的带宽策略节省 {report['kev_byte_saving_vs_same_ROI_bandwidth']:.2%} 字节；播放卡顿 {kev['rebuffer_s']:.3f} 秒。",'',
        f"完全相同请求字节却回答不同的QA条目：{unchanged_qa_disagreements}。这些变化是问答器波动，不能算成定位或编码改善。",'',
        '原问题与参考答案未改动；GT时间窗和掩码仅用于单独离线核查。相同视频不保证API给出相同回答。',
        'ROI变化后质量代理的取样定义也变化，因此不直接用旧/新reward差值证明进步。',
        '原五段视频已经用于开发，后续需另冻结未参与调试的测试集。本次没有训练。', '',
        '结构化结果见 [comparison.json](comparison.json)，多帧请求上下文见各源运行的 roi/context_plan.json；所有原始响应、失败探索和旧运行均保留。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'code.py').write_bytes(Path(__file__).read_bytes());q.write_json(out/'files.sha256.json',seal_paths([out]));print(json.dumps({'report':str(out/'REPORT.md'),'summary':summary},ensure_ascii=False))

if __name__=='__main__':main()
