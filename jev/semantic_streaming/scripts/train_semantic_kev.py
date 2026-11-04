#!/usr/bin/env python3
"""Preflight and launch the existing native Kev trainer on semantic encoding choices."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import qwen_roi as q
from kev_initialization import resolve_initialization


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='datasets/semantic_kev_dev_v4')
    p.add_argument('--out',default='runs/semantic_kev_v4_abr_baseline_train_v1')
    p.add_argument('--init-source',choices=['abr_baseline','released'],default=None,
                   help='Default is pinned Merina656 baseline epoch_08; released explicitly selects Kev-4B v1.0')
    p.add_argument('--preflight-out',default=None,help='Optional separate CPU audit path; preserves earlier dataset preflight records')
    p.add_argument('--execute',action='store_true',help='Actually train after CPU preflight and GPU resource check')
    p.add_argument('--max-steps',type=int,default=0,help='Optional smoke-run optimizer step cap; 0 trains both epochs')
    a=p.parse_args()
    if a.max_steps<0: p.error('Invalid max steps')
    control_path=q.ROOT/'configs/training_control.json'
    if a.execute and control_path.exists() and not json.loads(control_path.read_text())['training_enabled']:
        print('Training remains on hold at user request; CPU preflight is available without --execute.',flush=True)
        return 4
    dest=q.local_path(a.dataset);out=q.local_path(a.out)
    m=json.loads((dest/'manifest.json').read_text())
    for name,expected in m['files_sha256'].items():
        if q.sha((dest/name).read_bytes())!=expected: raise ValueError('Training asset changed: '+name)
    if m.get('format')=='semantic_guarded_planning_v4':
        from evaluate_semantic_control import verify_control_dataset
        verify_control_dataset(dest)
    deployment=json.loads((q.ROOT.parent/'deployment.json').read_text())
    init_config_path=q.ROOT/'configs/kev_initialization_v1.json'
    init_config=json.loads(init_config_path.read_text())
    initialization=resolve_initialization(init_config,q.ROOT.parent,deployment,a.init_source)
    init=Path(initialization['checkpoint_path'])
    os.environ.setdefault('HF_HOME',str(q.ROOT.parent/'.cache/huggingface'))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    sys.path.insert(0,str(q.ROOT.parent/'kev'))
    from transformers import AutoTokenizer
    from kev.data import materialize,api_request,load_records
    from kev.model import encode,training_context
    tok=AutoTokenizer.from_pretrained(deployment['base_path'],local_files_only=True)
    limits=training_context(512)
    maxima={'state':0,'packed':0};counts={}
    for split in ['train','valid']:
        rows=load_records(dest/f'{split}.jsonl')
        for row in rows:
            online=api_request(row)
            if set(online)!={'state','questions'}: raise ValueError('Unexpected model payload')
            # Check literal forbidden fields are absent from all actual input keys.
            def check_keys(obj):
                if isinstance(obj,dict):
                    if set(obj)&{'_supervision','candidate_reward','candidate_planned_return','candidate_actual_download_s','offline_quality_proxy','planning','label','reference_answer'}:
                        raise ValueError('Offline supervision leaked into model input')
                    for v in obj.values(): check_keys(v)
                elif isinstance(obj,list):
                    for v in obj: check_keys(v)
            check_keys(online)
            encoded=encode(tok,materialize(row),max_state=limits['max_state'],max_branch=limits['max_branch'],strict=True)
            if len(encoded['ids'])>limits['max_packed']: raise ValueError('Input exceeds Kev packed limit')
            maxima['state']=max(maxima['state'],encoded['state_tokens'])
            maxima['packed']=max(maxima['packed'],len(encoded['ids']))
        counts[split]=len(rows)
    if counts!=m['counts']: raise ValueError('Population changed')
    cmd=[sys.executable,'-m','kev.train','--data',str(dest/'train.jsonl'),
         '--init_from',str(init),'--base',deployment['base'],'--base_revision',deployment['base_revision'],
         '--epochs','2','--lr','2e-5','--batch','1','--accum','8','--lora','16','--head_dim','256',
         '--lora_targets','all','--dtype','bf16','--weights_dtype','fp32','--checkpointing','1',
         '--max_state','512','--p_none','0','--p_none_distract','0','--p_distract','0','--p_none_pair','0',
         '--seed','666','--device','cuda','--out',str(out)]
    if a.max_steps: cmd+=['--max_steps',str(a.max_steps)]
    audit={'complete':True,'counts':counts,'max_tokens':maxima,'limits':limits,
           'model_weights_loaded':False,'training_started':False,'training_argv':cmd,
           'initialization':initialization,'base_deployment':deployment,'objective':m['config']['learning_method'],
           'initialization_config_sha256':q.sha(init_config_path.read_bytes()),
           'initialization_helper_sha256':q.sha((q.ROOT/'scripts/kev_initialization.py').read_bytes()),
           'data_files_sha256':m['files_sha256'],'launcher_script_sha256':q.sha(Path(__file__).read_bytes())}
    q.write_json(q.local_path(a.preflight_out) if a.preflight_out else dest/'kev_preflight.json',audit)
    if not a.execute:
        print(json.dumps({'preflight':'passed','initialization_source':initialization['source_name'],
                         'checkpoint':str(init),'counts':counts,'max_tokens':maxima,'training_started':False}))
        return
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free_mb,util=[int(x.strip()) for x in gpu.stdout.splitlines()[0].split(',')]
    if free_mb<28000 or util>10:
        q.write_json(dest/'training_admission.json',{'checked_at':q.now(),'stage':'deferred_gpu_busy',
                     'free_mib':free_mb,'utilization_percent':util,'training_started':False,
                     'existing_processes_stopped':False})
        print('Training not started: GPU busy; existing processes left running.',flush=True)
        return 3
    if out.exists(): raise ValueError('Training output exists; choose a fresh output directory')
    # Reuse the existing package and weights cache; no changes inside Kev or traditional ABR.
    env=dict(os.environ);env['PYTHONPATH']=str(q.ROOT.parent/'kev')
    out.parent.mkdir(parents=True,exist_ok=True)
    q.write_json(dest/'training_admission.json',{'checked_at':q.now(),'stage':'admitted',
                 'free_mib':free_mb,'training_started':True,'training_argv':cmd})
    return subprocess.run(cmd,cwd=str(q.ROOT),env=env).returncode


if __name__=='__main__': sys.exit(main())
