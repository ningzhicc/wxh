#!/usr/bin/env python3
"""Three frozen existing development videos: real pipeline + repeated open QA.

Five-second source windows, two-second chunks, up to 54 ROI + 27 QA requests.
No training or automatic paid-call retries. No references/regions in model inputs.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import qwen_roi as q
from pipeline_media import file_sha
from run_semantic_pipeline import seal_paths,verify_files
from run_semantic_pipeline_context_v4 import CODE_FILES
from run_evqa_disambiguation import sample_video
from run_evqa_context_acceptance_v4 import run_child
from repeated_open_qa import run_qa

DATASET='datasets/clear_task_regression3_v1'
SELECTED=('dev10_01','dev10_07','dev10_08')


def task_config(template,case):
    return {**template,'video':case['video_path'],'expected_video_sha256':case['video_sha256'],'question':case['question'],
        'start_s':case['start_s'],'duration_s':5,'chunk_s':2,'global_sample_times_s':None,
        'sample_times_s':[0,0.4,0.8,1.2,1.6,1.8],'roi_mode':'api','roi_cache_run':None,
        'evidence_window_s':None,'max_api_calls':18,'policy':'bandwidth','api_config':'configs/qwen_roi_context_v4.json'}


def prepare_dataset():
    source=q.local_path('datasets/pilot_dev10_v1');dataset=q.local_path(DATASET)
    if (dataset/'files.sha256.json').exists():
        verify_files(json.loads((dataset/'files.sha256.json').read_text()))
        cases=[json.loads(l) for l in (dataset/'cases.jsonl').read_text().splitlines()]
        for row in cases:
            if file_sha(q.local_path(row['video_path']))!=row['video_sha256']:raise ValueError('Raw media changed')
        return dataset,cases
    if dataset.exists():raise ValueError('Partial task dataset retained')
    questions={r['case_id']:r for r in map(json.loads,(source/'task_questions.jsonl').read_text().splitlines())}
    parents={r['case_id']:r for r in map(json.loads,(source/'cases.jsonl').read_text().splitlines())}
    answers={r['case_id']:r for r in map(json.loads,(source/'annotations/answers.jsonl').read_text().splitlines())}
    cases=[];labels=[]
    for cid in SELECTED:
        parent=parents[cid];preparation=source/'provenance/preparation'/(cid+'.json');record=json.loads(preparation.read_text())
        path=f"media/raw/streamingbench/sample_{parent['sample_id']}/video.mp4"
        if file_sha(q.local_path(path))!=record['raw_sha256']:raise ValueError('Pinned source changed')
        cases.append({'case_id':cid,'question':questions[cid]['zh'],'video_path':path,'video_sha256':record['raw_sha256'],
            'start_s':parent['window']['start_s'],'duration_s':5,'first_frame_only':cid=='dev10_07',
            'previously_used_development_media':True,'parent_case_sha256':q.sha(json.dumps(parent,sort_keys=True,ensure_ascii=False).encode()),
            'parent_preparation_sha256':file_sha(preparation),'question_revision':questions[cid]['revision']})
        labels.append({'case_id':cid,'reference_answer':answers[cid]['reference_answer'],'accepted_answers':answers[cid]['accepted_answers'],
            'human_verified':False,'review_status':answers[cid]['review_status'],'training_eligible':False})
    dataset.mkdir();(dataset/'annotations').mkdir()
    for filename,rows in [('cases.jsonl',cases),('annotations/answers.jsonl',labels)]:
        (dataset/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    q.write_json(dataset/'manifest.json',{'created_at':q.now(),'kind':'existing_clear_development_task_regression','case_count':3,
        'selected_before_new_API_calls':list(SELECTED),'not_independent_test_set':True,'training_eligible':False,
        'parent_files_sha256':{name:file_sha(source/name) for name in ('manifest.json','cases.jsonl','task_questions.jsonl','annotations/answers.jsonl')}})
    (dataset/'preparation_code.py').write_bytes(Path(__file__).read_bytes());q.write_json(dataset/'files.sha256.json',seal_paths([dataset]));return dataset,cases


def qa_plan(cases,frames):
    return [{'case_id':case['case_id'],'variant':variant,'repeat':repeat,'question':case['question'],
        'first_frame_only':case['first_frame_only'],'frames':frames[(case['case_id'],variant)]}
        for case in cases for variant in ('reference','bandwidth','kev') for repeat in (1,2,3)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):p.error('Invalid run ID')
    dataset,cases=prepare_dataset();template=json.loads(q.local_path('configs/pipeline_live_dev01_v1.json').read_text())
    for name,expected in template['checkpoint_sha256'].items():
        if file_sha(q.local_path(template['checkpoint'])/name)!=expected:raise ValueError('Frozen v4 weights changed')
    if a.plan_only:
        print(json.dumps({'cases':cases,'ROI_cap':54,'QA_cap':27,'training_started':False,'API_calls':0,'GPU_used':False},ensure_ascii=False,indent=2));return 0
    names=(*CODE_FILES,'run_clear_tasks.py','repeated_open_qa.py','receiver_qa.py','run_evqa_disambiguation.py','run_evqa_context_acceptance_v4.py','replay_semantic_pipeline.py')
    promptpath=q.local_path('configs/clear_tasks_qa_prompt_v1.txt');prompt=promptpath.read_text().strip()
    fingerprint={'dataset_seal_sha256':file_sha(dataset/'files.sha256.json'),'template':template,
        'code':{name:file_sha(Path(__file__).with_name(name)) for name in names},'QA_prompt_sha256':file_sha(promptpath),
        'ROI_api_config_sha256':file_sha(q.local_path('configs/qwen_roi_context_v4.json')),
        'ROI_prompt_sha256':file_sha(q.local_path('configs/qwen_roi_context_prompt_v4.txt'))}
    run=q.local_path('runs/'+a.run_id)
    if run.exists():
        if json.loads((run/'manifest.json').read_text())['fingerprint']!=fingerprint:raise ValueError('Inputs/code changed; use fresh study ID')
        if (run/'files.sha256.json').exists():
            verify_files(json.loads((run/'files.sha256.json').read_text()))
            for name,digest in json.loads((run/'child_seals.json').read_text()).items():
                child=q.local_path('runs/'+name)
                if file_sha(child/'files.sha256.json')!=digest:raise ValueError('Child seal changed')
                verify_files(json.loads((child/'files.sha256.json').read_text()))
            print('Verified completed study; no API/GPU: '+str(run/'REPORT.md'));return 0
    else:
        run.mkdir();(run/'code').mkdir();q.write_json(run/'manifest.json',{'created_at':q.now(),'kind':'three_clear_task_full_pipeline_regression',
            'fingerprint':fingerprint,'ROI_call_cap':54,'QA_call_cap':27,'training_started':False,'not_independent_test':True})
        for name in names:(run/'code'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        (run/'qa_prompt.txt').write_bytes(promptpath.read_bytes())
    reports={};frames={};seals={};media=q.local_path('artifacts/'+a.run_id)
    for case in cases:
        cid=case['case_id'];baseid=a.run_id+'_'+cid+'_bandwidth';kevid=a.run_id+'_'+cid+'_kev'
        config=task_config(template,case);configpath=run/(cid+'_pipeline.json');q.write_json(configpath,config)
        print('Pipeline '+cid+': '+case['question'],flush=True)
        run_child([sys.executable,'-u',str(q.ROOT/'scripts/run_semantic_pipeline_context_v4.py'),'--config',str(configpath.relative_to(q.ROOT)),'--run-id',baseid],run/(cid+'_bandwidth.log'))
        run_child([sys.executable,'-u',str(q.ROOT/'scripts/replay_semantic_pipeline.py'),'--source-run',baseid,'--run-id',kevid,'--policy','kev'],run/(cid+'_kev.log'))
        base,kev=q.local_path('runs/'+baseid),q.local_path('runs/'+kevid)
        if (base/'actions.json').read_bytes()!=(kev/'actions.json').read_bytes():raise ValueError('Policies used different catalogs')
        bs,ks=[json.loads((r/'summary.json').read_text()) for r in (base,kev)]
        reports[cid]={'bandwidth':bs,'kev':ks,'source_run':baseid,'kev_run':kevid}
        for name in (baseid,kevid):seals[name]=file_sha(q.local_path('runs/'+name+'/files.sha256.json'))
        fps=json.loads((base/'chunks.json').read_text())[0]['fps']
        videos={'reference':q.local_path('artifacts/'+baseid+'/reference.mp4'),'bandwidth':q.local_path(bs['playback']['path']),'kev':q.local_path(ks['playback']['path'])}
        for variant,path in videos.items():frames[(cid,variant)]=sample_video(path,bs['frames'],fps,media/'qa_frames'/cid/variant)
    q.write_json(run/'policy_results.json',reports);q.write_json(run/'child_seals.json',seals)
    plan=qa_plan(cases,frames);q.write_json(run/'qa_plan.json',plan)
    stats=run_qa(run,plan,dataset/'annotations/answers.jsonl',json.loads(q.local_path('configs/qwen_roi_context_v4.json').read_text()),prompt)
    totals={v:{field:sum(reports[c['case_id']][v][field] for c in cases) for field in ('selected_bytes','rebuffer_s','frames')} for v in ('bandwidth','kev')}
    summary={'complete':True,'case_count':3,'not_independent_test':True,'QA':stats,'policy_totals':totals,
        'ROI_calls':sum(reports[c['case_id']]['bandwidth']['roi']['new_api_calls'] for c in cases),'QA_calls':len(plan),'training_started':False}
    q.write_json(run/'summary.json',summary)
    lookup={(r['case_id'],r['variant']):r for r in stats}
    lines=['# 三个明确问题的完整流程回归','','衣服文字、首帧计时器、中央黄色出租车车牌；全部使用已有开发视频，不作为独立测试。原分辨率原帧率5秒窗口切成约2秒连续块，冻结定位v4、编码、网络与Kev权重。','',
        '| 案例 | 原图正确次数 | 带宽正确次数 | Kev正确次数 | 带宽字节 | Kev字节 | 卡顿BW/Kev秒 |','|---|---:|---:|---:|---:|---:|---|']
    for case in cases:
        cid=case['case_id'];bw,k=reports[cid]['bandwidth'],reports[cid]['kev']
        counts=[str(lookup[(cid,v)]['correct'])+'/3' for v in ('reference','bandwidth','kev')]
        lines.append(f"| {cid} | {' | '.join(counts)} | {bw['selected_bytes']} | {k['selected_bytes']} | {bw['rebuffer_s']:.3f}/{k['rebuffer_s']:.3f} |")
    lines+=['',f"本轮成功定位 {summary['ROI_calls']} 次、开放问答 {len(plan)} 次，无训练。",'',
        '每个策略/问题的3次请求正文相同，问题使用开放回答，API不接收参考答案或多选项。首帧计时器只允许frame_index=0作依据；之后倒计时读数不能替代。',
        '全程保留原图答错、无法回答、策略退化和无ROI的片段，不删难例。每个问题只有一个源视频，3次重复不是3个独立样本；同一Qwen服务兼任定位与QA。',
        '首块固定最小文件起播，可能损伤首帧细节；后续块才由Kev/带宽策略实际选档。网络为轨迹回放加本地真实文件交付，非实际限速网络。',
        '原图/接收图SHA、真实候选大小、选档和完整解码记录见子运行；[开放问答原始评分](qa_scores.json)、[完整策略记录](policy_results.json)。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n');q.write_json(run/'files.sha256.json',seal_paths([run,media]));print(json.dumps({'complete':True,'report':str(run/'REPORT.md'),'ROI_calls':summary['ROI_calls'],'QA_calls':len(plan),'training_started':False}));return 0

if __name__=='__main__':raise SystemExit(main())
