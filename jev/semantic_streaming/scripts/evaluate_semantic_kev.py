#!/usr/bin/env python3
"""Evaluate a trained local Kev checkpoint and compare replay rewards to simple baselines."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import qwen_roi as q


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',default='datasets/semantic_kev_dev_v2')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    ck=q.local_path(a.checkpoint);data=q.local_path(a.dataset);out=q.local_path(a.out)
    if not (ck/'head.pt').is_file() or not (ck/'adapter_model.safetensors').is_file():
        raise ValueError('Training checkpoint is incomplete; wait for training to finish')
    if out.exists(): raise ValueError('Evaluation output exists; choose a new directory')
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest.get('format')=='semantic_guarded_planning_v4':
        raise ValueError('v4 has guarded multi-step labels; use evaluate_semantic_control.py for closed-loop evaluation')
    for name,expected in manifest['files_sha256'].items():
        if q.sha((data/name).read_bytes())!=expected: raise ValueError('Dataset file changed: '+name)
    env=dict(os.environ)
    env.update(HF_HOME=str(q.ROOT.parent/'.cache/huggingface'),HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(q.ROOT.parent/'kev'),KEV_DTYPE='bf16',KEV_MERGE='0')
    cmd=[sys.executable,'-u','-m','kev.benchmark','--run',str(ck),'--data',str(data/'valid.jsonl'),
         '--out',str(out),'--device','cuda']
    result=subprocess.run(cmd,env=env,cwd=str(q.ROOT))
    if result.returncode: return result.returncode
    valid=[json.loads(l) for l in (data/'valid.jsonl').read_text().splitlines()]
    train=[json.loads(l) for l in (data/'train.jsonl').read_text().splitlines()]
    actions=json.loads((data/'actions.json').read_text())
    originals={r['_meta']['id']:r for r in valid}
    predictions=json.loads((out/'rows.json').read_text())
    if len(predictions)!=len(valid) or len({r['id'] for r in predictions})!=len(valid):
        raise ValueError('Evaluation skipped or duplicated records')
    majority={cid:Counter(r['questions']['encoding']['label'] for r in train if r['state']['case_id']==cid).most_common(1)[0][0] for cid in actions}
    cheapest={cid:min(rows,key=lambda r:r['size_bytes'])['action_id'] for cid,rows in actions.items()}
    stats={name:{'matches':0,'reward_sum':0.0,'regret_sum':0.0,'stall_sum_s':0.0}
           for name in ['model','cheapest','train_majority_per_video']}
    for row in predictions:
        original=originals[row['id']];cid=original['state']['case_id']
        selected=row['keys'][max(range(len(row['p'])),key=lambda i:row['p'][i])]
        rewards=original['_supervision']['candidate_reward'];best=max(rewards.values())
        label=original['questions']['encoding']['label']
        for name,key in [('model',selected),('cheapest',cheapest[cid]),('train_majority_per_video',majority[cid])]:
            s=stats[name];s['matches']+=key==label;s['reward_sum']+=rewards[key];s['regret_sum']+=best-rewards[key]
            s['stall_sum_s']+=max(0,original['_supervision']['candidate_actual_download_s'][key]-original['state']['buffer_seconds'])
    n=len(predictions)
    summary={'validation_records':n,'checkpoint':str(ck),'metrics':{name:{
        'label_match_fraction':s['matches']/n,'mean_one_step_reward':s['reward_sum']/n,
        'mean_oracle_regret':s['regret_sum']/n,'mean_stall_s':s['stall_sum_s']/n} for name,s in stats.items()},
        'note':('Fixed behavior-policy-state evaluation from continuous rollouts; this does not evolve model-selected states. Use evaluate_semantic_rollout.py for closed-loop replay. Not QA accuracy.'
                if manifest.get('format')=='semantic_continuous_replay_v3' else
                'Development network-state evaluation on shared videos; one-step replay rewards with PSNR proxy, not online playback or QA accuracy.'),
        'dataset_manifest_sha256':q.sha((data/'manifest.json').read_bytes()),
        'checkpoint_head_sha256':q.sha((ck/'head.pt').read_bytes()),
        'checkpoint_adapter_sha256':q.sha((ck/'adapter_model.safetensors').read_bytes())}
    q.write_json(out/'semantic_summary.json',summary)
    q.write_json(out/'evaluation_command.json',{'argv':cmd,'script_sha256':q.sha(Path(__file__).read_bytes()),
                  'model_dtype':'bf16','adapter_merged':False,'api_calls':0})
    (out/'evaluate_semantic_kev.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':sys.exit(main())
