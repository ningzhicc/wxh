#!/usr/bin/env python3
"""Combine disjoint ROI runs without making calls or modifying source runs."""
import argparse
import json
from pathlib import Path
import statistics

import qwen_roi as q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--source-run', action='append', required=True)
    a = p.parse_args()
    dest = q.local_path('runs/'+a.run_id)
    dest.mkdir(exist_ok=False)
    combined, inputs, sources, latencies = [], [], [], []
    seen, common, cases = set(), None, {}
    totals = {k:0 for k in ('prompt_tokens','completion_tokens','total_tokens')}
    for name in a.source_run:
        run = q.local_path('runs/'+name)
        m = json.loads((run/'manifest.json').read_text())
        c = json.loads((run/'config.json').read_text())
        comparable = {k:c[k] for k in ('model','base_url','request_parameters','image_parameters','coordinate_contract')}
        comparable['prompt_sha256'] = m['fingerprint']['prompt_sha256']
        if common is not None and common != comparable:
            raise ValueError('Source inference settings differ')
        common = comparable
        assert q.sha((run/'code/qwen_roi.py').read_bytes()) == m['fingerprint']['script_sha256']
        assert q.sha((run/'config.json').read_bytes()) == m['fingerprint']['config_sha256']
        assert q.sha((run/'prompt.txt').read_bytes()) == m['fingerprint']['prompt_sha256']
        rows = json.loads((run/'inputs.json').read_text())
        plan_sha=q.sha(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode())
        assert plan_sha==m['fingerprint']['plan_sha256']
        for row in rows:
            fid = row['frame_id']
            assert fid not in seen
            seen.add(fid)
            d = run/'frames'/fid
            s = json.loads((d/'status.json').read_text())
            assert s['state']=='complete' and 'parse_error' not in s
            assert q.sha((d/'response.json').read_bytes()) == s['response_sha256']
            response = json.loads((d/'response.json').read_text())
            r = q.parse_regions(response,row,c['coordinate_contract']['scale_denominator'])
            assert r==json.loads((d/'roi.json').read_text())
            payload = q.make_payload(c,(run/'prompt.txt').read_text().strip(),row)
            wire = json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
            assert q.sha(wire)==json.loads((d/'request.json').read_text())['payload_sha256']
            combined.append({**r,'source_run':name})
            inputs.append({**row,'source_run':name})
            latencies.append(s['elapsed_s'])
            for k in totals: totals[k]+=response.get('usage',{}).get(k,0)
            case = cases.setdefault(row['case_id'],{'question':row['question'],'frames':0,'nonempty':0,'source_run':name})
            case['frames']+=1
            case['nonempty']+=bool(r['regions'])
        sources.append({'run':name,'manifest_sha256':q.sha((run/'manifest.json').read_bytes()),
                        'results_sha256':q.sha((run/'results.jsonl').read_bytes())})
    combined.sort(key=lambda r:r['frame_id'])
    inputs.sort(key=lambda r:r['frame_id'])
    q.write_json(dest/'manifest.json',{'kind':'aggregate_no_new_inference','created_at':q.now(),
                 'sources':sources,'common_inference_settings':common,'frames':len(combined),'cases':len(cases)})
    q.write_json(dest/'inputs.json',inputs)
    (dest/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined))
    q.write_json(dest/'summary.json',{'cases':cases,'frames':len(combined),'nonempty_frames':sum(bool(r['regions']) for r in combined),
                 'usage':totals,'mean_request_latency_s':statistics.mean(latencies),
                 'successful_request_list_price_estimate_cny':(totals['prompt_tokens']+10*totals['completion_tokens'])/1e6,
                 'quality_note':'Nonempty frame count is not accuracy; no human gold labels.'})
    (dest/'code').mkdir()
    (dest/'code/aggregate_roi.py').write_bytes(Path(__file__).read_bytes())
    lines=['# 10 视频开发集 ROI 定位汇总','','两次运行保持相同模型别名、提示词与请求参数；汇总没有重复调用 API。','',
           '| 案例 | 问题 | 有框/总帧数 | 原始运行与标框图 |','|---|---|---:|---|']
    for cid,c in sorted(cases.items()):
        lines.append(f"| {cid} | {c['question']} | {c['nonempty']}/{c['frames']} | [标框图](../{c['source_run']}/overlays/{cid}_contact.jpg) |")
    lines+=['',f"合计 {len(combined)} 帧；有框 {sum(bool(r['regions']) for r in combined)} 帧。这是结果数量，不是正确率。",
            '', '原始响应保留在各来源运行的 `frames/`，汇总结果逐帧保留 `source_run`。',
            '', '模型是可更新服务别名；本次没有获得不可变快照版本。全部是开发集，不作独立测试集报告。']
    (dest/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'aggregate':str(dest),'cases':len(cases),'frames':len(combined)}))


if __name__=='__main__':
    main()
