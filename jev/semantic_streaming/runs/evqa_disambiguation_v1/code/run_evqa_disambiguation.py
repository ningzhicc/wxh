#!/usr/bin/env python3
"""Separate revised-question development task; 18 ROI + 15 repeated QA calls.

Frozen v4 context/encoder/Kev. Original EVQA labels and studies are read-only.
No training, no automatic paid-request retry, no GT regions used for inference.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import random
import re
import sys

import qwen_roi as q
from acceptance_qa import run_qa
from pipeline_media import file_sha
from prepare_evqa_acceptance import verify_dataset
from run_evqa_context_acceptance_v4 import pipeline_config, run_child
from run_semantic_pipeline import seal_paths, verify_files
from run_semantic_pipeline_context_v4 import CODE_FILES
from semantic_rollout import seed_for
import encode_roi_grid as grid

DATASET='datasets/evqa_disambiguation_dev_v1'
PARENT='datasets/evqa_accept5_v1'
PRIOR='evqa_context5_v4'
BOWL='bowl_rack_v1'
TOWEL='towel_cabinet_v1'
CONTROL='towel_original_control_v1'
QUESTIONS={BOWL:'What color is the bowl placed upside down in the metal drying rack to the right of the kitchen sink?',
 TOWEL:'What color is the towel hanging on the cabinet door beneath the bathroom sink?'}


def revised_tasks(parents,references):
    tasks=[];annotations=[]
    lookup={r['case_id']:r for r in parents}
    for cid,parent_id in ((BOWL,'evqa_01'),(TOWEL,'evqa_02'),(CONTROL,'evqa_02')):
        parent=lookup[parent_id]
        tasks.append({**parent,'case_id':cid,'parent_case_id':parent_id,'parent_question':parent['question'],
            'question':QUESTIONS.get(cid,parent['question']),'question_revised':cid!=CONTROL,
            'role':'disambiguated_development' if cid!=CONTROL else 'original_question_repeated_control'})
        reference=references[parent_id]
        annotations.append({**reference,'case_id':cid,'parent_case_id':parent_id,'reference_answer_inherited':True,
            'local_human_verified':False,'training_eligible':False,
            'review_status':'label_disputed_pending_human' if cid!=BOWL else 'revised_task_pending_independent_review'})
    return tasks,annotations


def prepare_dataset():
    parent=q.local_path(PARENT);parents=verify_dataset(parent)
    dataset=q.local_path(DATASET)
    if (dataset/'files.sha256.json').exists():
        verify_files(json.loads((dataset/'files.sha256.json').read_text()))
        if json.loads((dataset/'manifest.json').read_text())['parent_manifest_sha256']!=file_sha(parent/'manifest.json'):
            raise ValueError('Parent annotations changed')
        return dataset,[json.loads(l) for l in (dataset/'cases.jsonl').read_text().splitlines()]
    if dataset.exists():raise ValueError('Partial/unrecorded task dataset retained')
    refs={r['case_id']:r for r in map(json.loads,(parent/'annotations/answers.jsonl').read_text().splitlines())}
    tasks,annotations=revised_tasks(parents,refs)
    dataset.mkdir();(dataset/'annotations').mkdir()
    (dataset/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks))
    (dataset/'annotations/answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in annotations))
    q.write_json(dataset/'manifest.json',{'created_at':q.now(),'kind':'revised_question_development_not_original_EVQA',
        'parent_manifest_sha256':file_sha(parent/'manifest.json'),'parent_answers_sha256':file_sha(parent/'annotations/answers.jsonl'),
        'task_count':3,'unique_source_videos':2,'training_eligible':False,'local_human_verified':False,
        'note':'New questions identify location, never the answer color. Towel original-question control shares the same video. No original labels overwritten.'})
    q.write_json(dataset/'sources.lock.json',{'parent_dataset':PARENT,'parent_manifest_sha256':file_sha(parent/'manifest.json'),
        'previous_context_study':PRIOR,'previous_study_seal_sha256':file_sha(q.local_path('runs/'+PRIOR+'/files.sha256.json')),
        'mask_dataset_seal_sha256':file_sha(q.local_path('datasets/evqa_evidence_audit_v1/files.sha256.json')),
        'question_revision_reason':'Assistant reviewed original frames and offline target masks. This is not a human-verification record.'})
    (dataset/'preparation_code.py').write_bytes(Path(__file__).read_bytes())
    q.write_json(dataset/'files.sha256.json',seal_paths([dataset]))
    return dataset,tasks


def assert_only_question_changed(config, source):
    old=json.loads((source/'config.json').read_text())
    new=dict(config)
    from run_semantic_pipeline import input_path
    for field in ('video','trace'):new[field]=str(input_path(new[field]))
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    if changed!={'question'}:raise ValueError('Unexpected pipeline changes: '+repr(sorted(changed)))
    prior=json.loads((source/'manifest.json').read_text())['fingerprint']['code_sha256']
    for name in CODE_FILES:
        if file_sha(Path(__file__).with_name(name))!=prior[name]:raise ValueError('Frozen code changed: '+name)
    return sorted(changed)


def repeated_plan(tasks,bowl_frames,towel_frames):
    plan=[]
    for task in tasks:
        options=list(task['qa_choices']);random.Random(seed_for(666,task['parent_case_id'],'qa_options')).shuffle(options)
        options={chr(65+i):value for i,value in enumerate(options)}
        variants=bowl_frames if task['case_id']==BOWL else {'reference':towel_frames}
        for variant,frames in variants.items():
            for repeat in range(1,4):
                plan.append({'case_id':task['case_id'],'variant':f'{variant}_r{repeat}',
                    'question':task['question'],'options':options,'frames':frames})
    if len(plan)!=15:raise ValueError('Expected exactly 15 QA calls')
    return plan


def consensus(rows,plan):
    lookup={(r['case_id'],r['variant']):r for r in plan};groups={}
    for row in rows:
        variant=row['variant'].rsplit('_r',1)[0];key=(row['case_id'],variant)
        choice=row['answer']['choice'];candidate=lookup[(row['case_id'],row['variant'])]
        text=candidate['options'].get(choice,'UNANSWERABLE')
        groups.setdefault(key,[]).append((text,row['correct']))
    return [{'case_id':cid,'variant':variant,'repeats':len(values),'answer_histogram':dict(Counter(v[0] for v in values)),
        'all_repeats_agree':len({v[0] for v in values})==1,'reference_matches':sum(v[1] for v in values)} for (cid,variant),values in groups.items()]


def sample_video(video,frames,fps,directory):
    count=min(12,frames);indices=[round(i*(frames-1)/(count-1)) for i in range(count)]
    if directory.exists():
        info=json.loads((directory/'sampling.json').read_text())
        if info['video_sha256']!=file_sha(video):raise ValueError('QA video changed')
        for row in info['frames']:
            if file_sha(q.local_path(row['path']))!=row['sha256']:raise ValueError('QA image changed')
        return info['frames']
    directory.parent.mkdir(parents=True,exist_ok=True)
    decoded,images=grid.decode_samples(q.ROOT/'.tools/ffmpeg',video,directory,indices)
    if decoded['frame_count']!=frames or decoded['fps']!=fps:raise ValueError('QA variants differ in time grid')
    rows=[{'path':str(path.relative_to(q.ROOT)),'sha256':file_sha(path),'timestamp_s':float(index/Fraction(fps))} for index,path in zip(indices,images,strict=True)]
    q.write_json(directory/'sampling.json',{'frames':rows,'video_sha256':file_sha(video)})
    return rows


def offline_grounding(run,source,old_source):
    # Called only after all localization, encoding, policy and QA calls complete.
    from PIL import Image
    from audit_evqa_evidence import coverage
    root=q.local_path('datasets/evqa_evidence_audit_v1');verify_files(json.loads((root/'files.sha256.json').read_text()))
    inputs=json.loads((source/'roi_inputs.json').read_text())
    variants={'generic_context':old_source,'explicit_target_context':source}
    refs={r['case_id']:r for r in map(json.loads,q.local_path(PARENT+'/annotations/answers.jsonl').read_text().splitlines())}
    rows=[]
    for name,directory in variants.items():
        previous=json.loads((directory/'roi_inputs.json').read_text())
        if [(r['sha256'],r['timestamp_s']) for r in previous]!=[(r['sha256'],r['timestamp_s']) for r in inputs]:raise ValueError('Offline comparison sampling changed')
        roi={r['frame_id']:r for r in map(json.loads,(directory/'roi_results.jsonl').read_text().splitlines())}
        for row in inputs:
            if not any(lo<=row['actual_window_timestamp_s']<=hi for lo,hi in refs['evqa_01']['evidence_segments_s']):continue
            mp=root/'masks/qa_ego4d_trimmed25'/f"{row['source_frame_index']:05d}.png"
            metric=coverage(Image.open(mp),roi[row['frame_id']]['regions'],row['width'],row['height'])
            shifts={}
            for shift in (-3,-1,0,1,3):
                p=root/'masks/qa_ego4d_trimmed25'/f"{row['source_frame_index']+shift:05d}.png"
                if p.exists():shifts[str(shift)]=coverage(Image.open(p),roi[row['frame_id']]['regions'],row['width'],row['height'])['gt_pixel_recall']
            rows.append({'variant':name,'frame_id':row['frame_id'],'timestamp_s':row['actual_window_timestamp_s'],**metric,'index_shift_recall':shifts})
    totals={}
    for name in variants:
        valid=[r for r in rows if r['variant']==name and r['mask_nonempty']]
        totals[name]={'samples':len(valid),'empty_API_frames':sum(r['api_regions']==0 for r in valid),
            'mean_GT_pixel_recall_provisional':sum(r['gt_pixel_recall'] for r in valid)/len(valid) if valid else None}
    result={'summary':totals,'frames':rows,'offline_only':True,
        'warning':'Mask/video frame counts differ; numeric-index correspondence provisional, +/-1/3 sensitivity retained. Not official EVQA J&F.'}
    q.write_json(run/'grounding_audit.json',result);return result


def write_report(run,stats,grounding,base_summary,kev_summary):
    lines=['# 明确目标的新问题：开发试验','',
        '新任务单独命名，原EVQA题目与评分保留。冻结多帧上下文方法、编码、网络和Kev权重，仅改问题指代。', '',
        '| 任务 | 视频 | 回答分布（3次） | 与继承参考答案匹配 | 三次一致 |','|---|---|---|---:|---|']
    for row in stats:lines.append(f"| {row['case_id']} | {row['variant']} | {row['answer_histogram']} | {row['reference_matches']}/3 | {row['all_repeats_agree']} |")
    lines+=['','| 碗的定位版本 | GT时间内非空掩码采样 | API空框 | 暂定像素覆盖率 |','|---|---:|---:|---:|']
    for name,row in grounding['summary'].items():lines.append(f"| {name} | {row['samples']} | {row['empty_API_frames']} | {row['mean_GT_pixel_recall_provisional']:.2%} |")
    lines+=['',f"碗：带宽策略 {base_summary['selected_bytes']} 字节，Kev {kev_summary['selected_bytes']} 字节；卡顿分别 {base_summary['rebuffer_s']} / {kev_summary['rebuffer_s']} 秒。",'',
        '重复问答使用完全相同的问题、选项和全帧图片，仅本地重复编号不同。三次一致不等于独立评测，也不是完整基准准确率。',
        '毛巾的green标签继承自原数据，原图多次输出blue也不自动改标签；保持待人工复核且禁止作为已核验训练标签。新碗问题的标签也尚无独立人工确认。',
        '掩码只用于所有推理完成后的离线核查，绝不输入定位、Kev或主QA；索引对应存在不确定性，覆盖率仅作诊断。',
        '原标注与实际画面审核材料见 [review.md](review.md)。本轮没有训练。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')


def write_review(run,prior_plan,tasks):
    lines=['# 待人工核查的目标与颜色','',
        '这是助手整理的证据，不是人工已确认记录。原标签保留；请记录实际审核者、依据、日期后另存审核记录。','']
    for parent,taskid,label in [('evqa_01',BOWL,'light blue'),('evqa_02',TOWEL,'green')]:
        task=next(t for t in tasks if t['case_id']==taskid)
        frames=next(r['frames'] for r in prior_plan if r['case_id']==parent and r['variant']=='reference')
        lines += [f"## {taskid}",'',f"原问题：{task['parent_question']}",f"新问题：{task['question']}",f'原标签：{label}；人工状态：未核验。','']
        for frame in sorted(frames,key=lambda r:abs(r['timestamp_s']-8))[:3]:
            path=q.local_path(frame['path']);lines += [f"原始完整帧 t={frame['timestamp_s']:.3f}s，SHA256={frame['sha256']}",'',f'![原图]({path})','']
    lines += ['毛巾存在明显暖色照明。仅凭像素或模型输出无法确认物体在标准光源下的真实颜色；需要独立来源或人工依据。保持原green标注并记录争议。']
    (run/'review.md').write_text('\n'.join(lines)+'\n')
    q.write_json(run/'human_review_template.json',{'reviewer':None,'reviewed_at':None,'status':'pending',
        'training_eligible':False,'items':[{'case_id':BOWL,'original_label':'light blue','verified_label':None,'evidence':None},
            {'case_id':TOWEL,'original_label':'green','verified_label':None,'evidence':None}],
        'note':'Fill a separate human_review_<version>.json; do not edit this sealed template or original annotations.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):parser.error('Invalid run ID')
    dataset,tasks=prepare_dataset();prior=q.local_path('runs/'+PRIOR)
    verify_files(json.loads((prior/'files.sha256.json').read_text()))
    reports=json.loads((prior/'policy_results.json').read_text());old_source=q.local_path('runs/'+reports['evqa_01']['source_run'])
    verify_files(json.loads((old_source/'files.sha256.json').read_text()))
    template=json.loads(q.local_path('configs/pipeline_live_dev01_v1.json').read_text());template['api_config']='configs/qwen_roi_context_v4.json'
    for name,expected in template['checkpoint_sha256'].items():
        if file_sha(q.local_path(template['checkpoint'])/name)!=expected:raise ValueError('Frozen Kev weights changed')
    if args.plan_only:
        print(json.dumps({'tasks':[{k:t[k] for k in ('case_id','parent_case_id','question','role')} for t in tasks],
            'ROI_call_cap':18,'QA_call_cap':15,'training_started':False,'gpu_used':False},ensure_ascii=False,indent=2));return 0
    names=(*CODE_FILES,'run_evqa_disambiguation.py','acceptance_qa.py','replay_semantic_pipeline.py','prepare_evqa_acceptance.py','run_evqa_context_acceptance_v4.py','audit_evqa_evidence.py')
    fingerprint={'dataset_seal_sha256':file_sha(dataset/'files.sha256.json'),'prior_study_seal_sha256':file_sha(prior/'files.sha256.json'),
        'template':template,'code':{name:file_sha(Path(__file__).with_name(name)) for name in names},
        'api_config_sha256':file_sha(q.local_path(template['api_config'])),'prompt_sha256':file_sha(q.local_path('configs/qwen_roi_context_prompt_v4.txt')),
        'predeclared_diagnostic_thresholds':{'mask_pixel_recall':0.8,'reference_QA_matches':3,'repeats':3}}
    run=q.local_path('runs/'+args.run_id)
    if run.exists():
        if json.loads((run/'manifest.json').read_text())['fingerprint']!=fingerprint:raise ValueError('Study inputs/code changed; use new ID')
        if (run/'files.sha256.json').exists():
            verify_files(json.loads((run/'files.sha256.json').read_text()))
            for name,digest in json.loads((run/'child_seals.json').read_text()).items():
                child=q.local_path('runs/'+name)
                if file_sha(child/'files.sha256.json')!=digest:raise ValueError('Child seal changed')
                verify_files(json.loads((child/'files.sha256.json').read_text()))
            print('Verified completed study; no API/GPU: '+str(run/'REPORT.md'));return 0
    else:
        run.mkdir();(run/'code').mkdir()
        q.write_json(run/'manifest.json',{'created_at':q.now(),'kind':'explicit_target_repeated_QA_development','fingerprint':fingerprint,
            'ROI_call_cap':18,'QA_call_cap':15,'training_started':False})
        for name in names:(run/'code'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    prior_plan=json.loads((prior/'qa_plan.json').read_text());write_review(run,prior_plan,tasks)
    bowl=next(t for t in tasks if t['case_id']==BOWL);baseid=args.run_id+'_bowl_bandwidth';kevid=args.run_id+'_bowl_kev'
    cfg=pipeline_config(template,bowl);differences=assert_only_question_changed(cfg,old_source)
    q.write_json(run/'controlled_input_receipt.json',{'source_run':old_source.name,'source_seal_sha256':file_sha(old_source/'files.sha256.json'),'changed_pipeline_fields':differences})
    cfgpath=run/'bowl_pipeline.json';q.write_json(cfgpath,cfg)
    print('Running explicit-target bowl pipeline',flush=True)
    run_child([sys.executable,'-u',str(q.ROOT/'scripts/run_semantic_pipeline_context_v4.py'),'--config',str(cfgpath.relative_to(q.ROOT)),'--run-id',baseid],run/'bandwidth.log')
    run_child([sys.executable,'-u',str(q.ROOT/'scripts/replay_semantic_pipeline.py'),'--source-run',baseid,'--run-id',kevid,'--policy','kev'],run/'kev.log')
    base,kev=q.local_path('runs/'+baseid),q.local_path('runs/'+kevid)
    if (base/'actions.json').read_bytes()!=(kev/'actions.json').read_bytes():raise ValueError('Candidate catalogs differ')
    q.write_json(run/'child_seals.json',{name:file_sha(q.local_path('runs/'+name+'/files.sha256.json')) for name in (baseid,kevid)})
    bs,ks=[json.loads((r/'summary.json').read_text()) for r in (base,kev)]
    fps=json.loads((base/'chunks.json').read_text())[0]['fps'];media=q.local_path('artifacts/'+args.run_id)
    videos={'reference':q.local_path('artifacts/'+baseid+'/reference.mp4'),'bandwidth':q.local_path(bs['playback']['path']),'kev':q.local_path(ks['playback']['path'])}
    frames={v:sample_video(path,bs['frames'],fps,media/'qa_frames'/v) for v,path in videos.items()}
    towel_frames=next(r['frames'] for r in prior_plan if r['case_id']=='evqa_02' and r['variant']=='reference')
    plan=repeated_plan(tasks,frames,towel_frames);q.write_json(run/'qa_plan.json',plan)
    rows=run_qa(run,plan,dataset/'annotations/answers.jsonl',json.loads(q.local_path(template['api_config']).read_text()))
    stats=consensus(rows,plan);grounding=offline_grounding(run,base,old_source)
    reference=next(r for r in stats if r['case_id']==BOWL and r['variant']=='reference')
    gate=grounding['summary']['explicit_target_context']['mean_GT_pixel_recall_provisional']>=0.8 and reference['reference_matches']==3
    summary={'complete':True,'question_task_count':3,'unique_videos':2,'ROI_API_calls':18,'QA_API_calls':15,'repeated_QA':stats,
        'grounding':grounding['summary'],'policies':{'bandwidth':bs,'kev':ks},'diagnostic_gate_passed':gate,
        'formal_training_eligible':False,'towel_label_status':'disputed_pending_human','new_questions_are_original_EVQA':False,'training_started':False}
    q.write_json(run/'summary.json',summary)
    # Verify equal requests within each repeat group; variant/repeat never enters the API wire.
    requests={}
    for receipt in (run/'qa').glob('*/request.json'):
        r=json.loads(receipt.read_text());key=(r['case_id'],r['variant'].rsplit('_r',1)[0]);requests.setdefault(key,set()).add(r['payload_sha256'])
    if any(len(v)!=1 for v in requests.values()):raise ValueError('Repeated QA payloads differ')
    q.write_json(run/'repeated_payload_receipt.json',{'groups':[{'case_id':k[0],'variant':k[1],'payload_sha256':next(iter(v)),'repeats':3} for k,v in requests.items()]})
    write_report(run,stats,grounding,bs,ks)
    q.write_json(run/'files.sha256.json',seal_paths([run,media]));print(json.dumps({'complete':True,'report':str(run/'REPORT.md'),'diagnostic_gate_passed':gate,'training_started':False}));return 0

if __name__=='__main__':raise SystemExit(main())
