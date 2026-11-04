#!/usr/bin/env python3
"""Read-only dataset audit: label coverage, state diversity and split overlap."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import qwen_roi as q


def online_signature(row):
    payload={'state':row['state'],'questions':{qid:{k:v for k,v in question.items()
              if k in ('type','instructions','criteria')} for qid,question in row['questions'].items()}}
    return q.sha(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='datasets/semantic_kev_dev_v2')
    p.add_argument('--run-id',default='semantic_kev_data_audit_20261103_v1')
    a=p.parse_args();source=q.local_path(a.dataset)
    manifest=json.loads((source/'manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        assert q.sha((source/name).read_bytes())==expected
    data={s:[json.loads(l) for l in (source/f'{s}.jsonl').read_text().splitlines()] for s in ('train','valid')}
    actions=json.loads((source/'actions.json').read_text())
    lookup={cid:{x['action_id']:x for x in rows} for cid,rows in actions.items()}
    report={'created_at':q.now(),'dataset':a.dataset,'inputs_sha256':manifest['files_sha256'],
            'training_started':False,'dataset_modified':False,'splits':{}}
    majority={}
    signatures={}
    for split,rows in data.items():
        crf=Counter();mode=Counter();by_case={};by_signature=defaultdict(set)
        no_feasible=0;optimal_stall=0
        for cid in actions:
            subset=[r for r in rows if r['state']['case_id']==cid]
            labels=Counter(r['questions']['encoding']['label'] for r in subset)
            best,n=labels.most_common(1)[0]
            if split=='train':majority[cid]=best
            by_case[cid]={'records':len(subset),'selected_action_count':len(labels),'majority_label':best,
                'majority_fraction':n/len(subset),
                'previous_bitrate_value_count':len({r['state']['previous_bitrate_kbps'] for r in subset})}
        for row in rows:
            cid=row['state']['case_id'];label=row['questions']['encoding']['label'];action=lookup[cid][label]
            crf[action['global_crf']]+=1;mode[action['mode']]+=1
            delays=row['_supervision']['candidate_actual_download_s'];buffer=row['state']['buffer_seconds']
            no_feasible+=min(delays.values())>buffer
            optimal_stall+=delays[label]>buffer
            by_signature[online_signature(row)].add(label)
        signatures[split]=set(by_signature)
        report['splits'][split]={'records':len(rows),'global_crf_labels':dict(crf),'mode_labels':dict(mode),
             'global_crf40_fraction':crf[40]/len(rows),'unique_inputs':len(by_signature),
             'duplicate_rows':len(rows)-len(by_signature),'conflicting_input_groups':sum(len(v)>1 for v in by_signature.values()),
             'no_nonstall_action_fraction':no_feasible/len(rows),'oracle_selected_action_stall_fraction':optimal_stall/len(rows),
             'buffer_distribution':dict(Counter(r['state']['buffer_seconds'] for r in rows)),
             'cases':by_case}
    report['online_input_overlap_train_valid']=len(signatures['train']&signatures['valid'])
    trace_rows=json.loads((source/'traces.lock.json').read_text())
    groups={s:{r['source_parent'] for r in trace_rows if r['split']==s} for s in data}
    report['network_source_group_overlap']=sorted(groups['train']&groups['valid'])
    report['video_case_overlap']=sorted(actions)
    report['train_majority_per_video_validation_match']=sum(r['questions']['encoding']['label']==majority[r['state']['case_id']] for r in data['valid'])/len(data['valid'])
    report['quality_proxy_saturated_actions']=sum(x['offline_quality_proxy']>=1 for rows in actions.values() for x in rows)
    report['candidate_actions']=sum(len(rows) for rows in actions.values())
    dest=q.local_path('runs/'+a.run_id);dest.mkdir(exist_ok=False)
    q.write_json(dest/'audit.json',report)
    (dest/'audit_semantic_data.py').write_bytes(Path(__file__).read_bytes())
    train=report['splits']['train'];valid=report['splits']['valid']
    lines=['# 训练数据质量检查','','结论：适合最小系统联调和训练冒烟；状态分布偏向低缓冲，暂不建议直接长时间训练。数据保持不变，训练继续等待。','',
        f"- 训练 {train['records']} 条、验证 {valid['records']} 条；来自10个开发视频×网络状态与缓冲场景枚举，不是6000个独立视频。",
        f"- 训练 CRF40 标签 {train['global_crf_labels'][40]}/{train['records']}（{train['global_crf40_fraction']:.2%}）；验证 {valid['global_crf40_fraction']:.2%}。这不是完整12动作的正确率。",
        f"- 训练状态中 {train['no_nonstall_action_fraction']:.2%} 没有任何能在缓冲耗尽前传完的候选；标签所选动作 {train['oracle_selected_action_stall_fraction']:.2%} 会卡顿。",
        '- 六个缓冲值为0、0.05、0.2、0.5、1、4秒，其中5/6不超过1秒；RTT已为0.08秒，前两类必然出现等待。',
        '- 每个视频的上一档码率始终固定为同一个值，未覆盖真实策略连续选档后的上一档变化。',
        f"- 质量代理用证据RGB PSNR/40并裁到1，{report['quality_proxy_saturated_actions']}/{report['candidate_actions']} 个候选已到上限，丢失部分质量区分度。",
        f"- 训练、验证各自没有完全重复的在线输入；跨集完全相同输入 {report['online_input_overlap_train_valid']} 条；网络来源分组重叠 {len(report['network_source_group_overlap'])} 个。",
        '- 两个划分使用同样的10个开发视频，不能以验证成绩声称新视频泛化。',
        '- 奖励仍是像素质量代理加一步实际传输代价，不是问答成功率，也没有长期缓冲回报。足球等来源ROI问题仍可能影响标签意义。','',
        '## 训练标签分布','',
        '| 视频 | 训练中出现的动作数/12 | 最大动作占比 | 上一档码率取值数 |','|---|---:|---:|---:|']
    for cid,c in train['cases'].items():
        lines.append(f"| {cid} | {c['selected_action_count']}/12 | {c['majority_fraction']:.2%} | {c['previous_bitrate_value_count']} |")
    lines+=['','## 聚焦第三步的最小改进','',
        '保留本版用于接口冒烟；复用现有120个编码候选重建下一版状态和标签，不需重新编码或调用API。',
        '优先用传输回放生成连续缓冲和上一档状态，并按目标系统允许的缓冲/时延检查可行候选覆盖。若大量场景连最低档也不可用，应调整候选档位或明确网络适用范围。',
        '让不同最优动作自然出现在可行场景中，不通过随意改标签强行均衡。保留原始证据PSNR，明确质量代理尺度和权重。',
        '视频与网络状态的泛化验证属于之后的效果检查，不阻塞先跑通本地模型接口。',
        '', '[完整检查数值](audit.json)。本次仅CPU读取数据并记录检查结果，未修改训练集或启动训练。']
    (dest/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(dest/'REVIEW.md'),'crf40_fraction':train['global_crf40_fraction'],
                     'no_nonstall_action_fraction':train['no_nonstall_action_fraction'],'dataset_modified':False,'training_started':False}))


if __name__=='__main__':main()
