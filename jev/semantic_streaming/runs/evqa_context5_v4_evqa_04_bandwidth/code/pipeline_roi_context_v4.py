"""Joint chunk context with the proven single-target-frame ROI response contract.

Each paid call sees up to six original frames, outputs one target frame, and is
reconstructible from the sealed context plan. No failed-request auto-retries.
"""
import json

import qwen_roi as q
import pipeline_roi as original

_SINGLE_PAYLOAD = q.make_payload


def context_payload(config, prompt, row, context):
    if not 1 <= len(context) <= 6 or len({r['frame_id'] for r in context}) != len(context):
        raise ValueError('Expected 1..6 distinct contextual frames')
    if sum(r == row for r in context) != 1:
        raise ValueError('Target must be an exact member of the context')
    if any(r['case_id'] != row['case_id'] or r['question'] != row['question'] for r in context):
        raise ValueError('Context must be from the same chunk and question')
    if [r['timestamp_s'] for r in context] != sorted(r['timestamp_s'] for r in context):
        raise ValueError('Context must be chronological')
    content = [{'type':'text','text':'以下为按时间排序的同一视频块的上下文帧。先结合上下文辨认相关对象，最后只为TARGET_IMAGE输出坐标，不要输出上下文帧的坐标。'}]
    for other in context:
        if other['frame_id'] == row['frame_id']:
            continue
        one = _SINGLE_PAYLOAD(config, prompt, other)['messages'][1]['content']
        content += [{'type':'text','text':json.dumps({'context_timestamp_s':other['timestamp_s']})},one[0]]
    target = _SINGLE_PAYLOAD(config, prompt, row)
    content += [{'type':'text','text':'TARGET_IMAGE：只为下面这一张图返回frame_id和regions。'}, *target['messages'][1]['content']]
    return {**target,'messages':[{'role':'system','content':prompt},{'role':'user','content':content}]}


def locate(inputs, run, cfg, api_config, prompt):
    if cfg['roi_mode'] != 'api' or cfg.get('roi_cache_run') or cfg.get('evidence_window_s') is not None:
        raise ValueError('Context v4 requires API mode, no single-frame cache or GT time filter')
    if len({r['frame_id'] for r in inputs}) != len(inputs):
        raise ValueError('Duplicate global frame identity')
    groups = {}
    for row in inputs:
        groups.setdefault(row['case_id'],[]).append(row)
    if not inputs or any(not 1 <= len(rows) <= 6 for rows in groups.values()):
        raise ValueError('Expected nonempty chunks with at most six samples')
    plan={'strategy':'chunk_context_single_target_v4','inputs':inputs,
        'context_frame_ids':{r['frame_id']:[o['frame_id'] for o in groups[r['case_id']]] for r in inputs},
        'target_image_is_last':True,'offline_chunk_context':True,'answers_masks_or_GT_times_used':False}
    path=run/'roi/context_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Resumed contextual inputs changed')
    q.write_json(path,plan)
    def construct(config, text, row):
        return context_payload(config,text,row,groups[row['case_id']])
    saved=q.make_payload
    try:
        q.make_payload=construct
        result=original.locate(inputs,run,cfg,api_config,prompt)
    finally:
        q.make_payload=saved
    summary_path=run/'roi_summary.json'
    summary=json.loads(summary_path.read_text())
    q.write_json(summary_path,{**summary,'strategy':'chunk_context_single_target_v4',
        'maximum_context_frames':6,'offline_chunk_context':True})
    return result
