#!/usr/bin/env python3
"""Audit cached ROI results and produce an offline report; never calls an API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, __version__ as pillow_version

import qwen_roi


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    root = qwen_roi.ROOT
    run = qwen_roi.local_path('runs/' + args.run_id)
    manifest = json.loads((run/'manifest.json').read_text())
    config = json.loads((run/'config.json').read_text())
    rows = json.loads((run/'inputs.json').read_text())
    fp = manifest['fingerprint']
    assert digest(run/'config.json') == fp['config_sha256']
    assert digest(run/'prompt.txt') == fp['prompt_sha256']
    assert digest(run/'code/qwen_roi.py') == fp['script_sha256']
    assert digest(qwen_roi.local_path(config['inputs'])) == fp['inputs_sha256']
    plan_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert plan_sha == fp['plan_sha256']
    results, statuses, usages = {}, {}, []
    for row in rows:
        d = run/'frames'/row['frame_id']
        s = json.loads((d/'status.json').read_text())
        assert s['state'] == 'complete' and 'parse_error' not in s
        assert digest(d/'response.json') == s['response_sha256']
        response = json.loads((d/'response.json').read_text())
        payload = qwen_roi.make_payload(config, (run/'prompt.txt').read_text().strip(), row)
        wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        request = json.loads((d/'request.json').read_text())
        assert hashlib.sha256(wire).hexdigest() == request['payload_sha256']
        assert request['input'] == row
        result = qwen_roi.parse_regions(response, row, config['coordinate_contract']['scale_denominator'])
        assert result == json.loads((d/'roi.json').read_text())
        assert (run/'overlays'/(row['frame_id']+'.jpg')).is_file()
        results[row['frame_id']] = result
        statuses[row['frame_id']] = s
        usages.append(response.get('usage', {}))
    cached = [json.loads(l) for l in (run/'results.jsonl').read_text().splitlines() if l]
    assert cached == [results[row['frame_id']] for row in rows]

    # Labels are opened only here, AFTER inference has completed, for seed diagnostics.
    annotations_path = root/'datasets/pilot_dev10_v1/annotations/answers.jsonl'
    annotations = {d['case_id']: d for d in map(json.loads, annotations_path.read_text().splitlines())}
    cases = []
    for cid in config['case_ids']:
        case_rows = [row for row in rows if row['case_id'] == cid]
        case_results = [results[row['frame_id']] for row in case_rows]
        fractions = [sum(b['area_fraction'] for b in r['regions']) for r in case_results if r['regions']]
        diagnostics = []
        for seed in annotations[cid]['visual_review']['evidence']:
            matches = [r for r in case_rows if r['sha256'] == seed['sha256'] and r['path'] == seed['path']]
            if not matches:
                continue
            row = matches[0]
            sx1, sy1, sx2, sy2 = seed['roi_seed_xyxy']
            seed_area = (sx2-sx1)*(sy2-sy1)
            comparisons = []
            for b in results[row['frame_id']]['regions']:
                x1, y1, x2, y2 = b['bbox_pixels']
                intersection = max(0, min(x2, sx2)-max(x1, sx1))*max(0, min(y2, sy2)-max(y1, sy1))
                union = seed_area+(x2-x1)*(y2-y1)-intersection
                comparisons.append({'model_box': b['bbox_pixels'], 'seed_iou': intersection/union,
                                    'seed_area_covered_fraction': intersection/seed_area})
            diagnostics.append({'frame_id': row['frame_id'], 'seed_box': seed['roi_seed_xyxy'],
                                'comparison': comparisons, 'not_ground_truth': True})
        cases.append({'case_id': cid, 'question': case_rows[0]['question'],
                      'frames': len(case_rows),
                      'frames_with_regions': sum(bool(r['regions']) for r in case_results),
                      'empty_frames': sum(not r['regions'] for r in case_results),
                      'mean_box_area_fraction_when_nonempty': statistics.mean(fractions) if fractions else None,
                      'seed_diagnostics': diagnostics})

        sheet = Image.new('RGB', (1920, 760), '#202020')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(case_rows):
            with Image.open(run/'overlays'/(row['frame_id']+'.jpg')) as im:
                im.thumbnail((640, 360))
                sheet.paste(im, ((i%3)*640, (i//3)*380))
            draw.text(((i%3)*640+6, (i//3)*380+362), row['frame_id'], fill='white')
        sheet.save(run/'overlays'/(cid+'_contact.jpg'), quality=95)

    usage = {k: sum(u.get(k, 0) for u in usages) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    applicable = all(0 < u.get('prompt_tokens', 0) <= 32768 for u in usages)
    price = config['price_reference']
    estimated_cost = (usage['prompt_tokens']*price['input_per_million_tokens']+
                      usage['completion_tokens']*price['output_per_million_tokens'])/1_000_000 if applicable else None
    latency = [s['elapsed_s'] for s in statuses.values()]
    report_source = Path(__file__).read_bytes()
    report_source_sha = hashlib.sha256(report_source).hexdigest()
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'case_results': cases,
              'report_script_sha256': report_source_sha,
              'successful_frames': len(results), 'usage': usage,
              'latency_s': {'mean': statistics.mean(latency), 'min': min(latency), 'max': max(latency),
                            'sum_request_durations': sum(latency)},
              'successful_request_list_price_estimate_cny': estimated_cost,
              'cost_note': 'Not invoice; excludes failed pilot v1 request and account discounts/free quota.',
              'model_version': config['model_version_note'],
              'seed_diagnostics_note': 'Seeds are assistant visual annotations, not human gold. IoU is NOT accuracy.',
              'annotation_sha256': digest(annotations_path), 'pillow_version': pillow_version,
              'pixel_coordinate_note': config['coordinate_contract'],
              'checks': [f'{len(results)} cached responses parse without schema errors', 'response hashes match receipts',
                         'all request payload hashes reconstruct from recorded inputs/code/config',
                         'all frame hashes and dimensions checked before inference',
                         'pixel boxes recompute exactly from raw normalized coordinates',
                         'flattened results match per-frame results', 'all overlays exist']}
    qwen_roi.write_json(run/'offline_audit.json', report)
    (run/'code'/f'report_qwen_roi_{report_source_sha}.py').write_bytes(report_source)
    lines = ['# 三案例首次真实 API 定位实验', '',
             f'{len(results)} 帧全部完成调用且通过 JSON/坐标校验。此报告只描述定位，不代表完成连续视频跟踪、压缩收益或 QA 准确率评测。', '',
             '| 案例 | 问题 | 有框 / 总帧数 | 有框帧平均框面积占比 | 标框接触表 |',
             '|---|---|---:|---:|---|']
    for c in cases:
        cid = c['case_id']
        lines.append(f"| {cid} | {c['question']} | {c['frames_with_regions']} / {c['frames']} | "
                     f"{c['mean_box_area_fraction_when_nonempty']:.3%} | [查看](overlays/{cid}_contact.jpg) |")
    lines += ['', f"成功请求用量：输入 {usage['prompt_tokens']} tokens，输出 {usage['completion_tokens']} tokens。",
              f"单帧平均端到端请求耗时 {statistics.mean(latency):.3f} 秒；{len(latency)} 次请求耗时之和 {sum(latency):.3f} 秒，不含两次执行之间的人工检查时间。",
              f"按配置中北京地区公开单价估算，成功请求约 {estimated_cost:.6f} 元；不是实际账单，也不包含 v1 的失败请求。", '',
              '## 解释边界', '',
              '- 使用 `qwen3-vl-plus` 服务别名，响应同样返回该别名；没有取得不可变模型权重或快照版本，未来复跑不保证逐字一致。',
              '- 原图原尺寸上传。服务端内部缩放不可见；已记录 min_pixels=65536、max_pixels=2621440。',
              '- 文档标注坐标范围 0..999，但官方绘图代码以 1000 为分母。本实验沿用该比例，并对左上向下取整、右下向上取整，输出半开像素框；没有增加安全边距或对齐编码块。',
              '- 时间戳为抽帧请求的片内相对时刻；此处没有将其声称为源视频准确 PTS。',
              '- 电子秤问题以首帧为回答锚点，后续帧框只是相同目标的候选区域，不能把后续读数当作首帧答案。',
              '- 单帧调用没有跨帧记忆，空数组表示当前帧未定位到证据，不能解释为整段视频没有答案。',
              '- 离线审计才读取原有种子框；种子框不是人工金标准，IoU 只作参考，不据此计算定位准确率。',
              '- 只采样 6 帧/片段，不能直接把这些框视作每个编码帧的完整 ROI 轨迹。',
              f'- 单帧平均耗时 {statistics.mean(latency):.2f} 秒；串行 6 次调用约 {6*statistics.mean(latency):.1f} 秒/片段，尚未验证实时吞吐。', '',
              '## 可追溯文件', '',
              '- [逐帧框与原图 SHA256](results.jsonl)、[逐帧结果表](REPORT.md)、[离线完整性审计](offline_audit.json)。',
              '- `frames/<frame_id>/request.json`：输入白名单、请求体哈希；图片 Base64 不重复存储。',
              '- `frames/<frame_id>/response.json`：服务端原始响应；鉴权头从不写入记录。',
              '- `frames/<frame_id>/roi.json`：归一化坐标、绝对像素框与面积。',
              '- `config.json`、`prompt.txt`、`inputs.json`、`code/`：本次运行配置与代码快照。',
              '- 首次 min_pixels=4096 被参数校验拒绝的记录独立保留在相邻 `qwen3vl_roi_pilot_20261102_v1` 目录，没有覆盖。',
              '', '下一步：审阅这些标框，扩展到剩余开发案例，再准备逐编码帧 ROI 和均匀/ROI 编码对照。']
    if (run/'assistant_review.json').is_file():
        review = json.loads((run/'assistant_review.json').read_text())
        lines += ['', '## 助手标框初查', '']
        for cid, note in review['findings'].items():
            lines.append(f'- **{cid}**：{note}')
        lines += ['', '这是助手初查，未记作人工核验或正式准确率。详见 [审阅记录](assistant_review.json)。']
    (run/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    # Rebuilt reports are derived artifacts; refresh their checksums while preserving raw API records.
    checksums = {str(f.relative_to(run)): digest(f) for f in sorted(run.rglob('*'))
                 if f.is_file() and f.name != 'files.sha256.json'}
    qwen_roi.write_json(run/'files.sha256.json', checksums)
    print(json.dumps({'verified_frames': len(results), 'estimated_successful_cost_cny': estimated_cost,
                      'report': str(run/'REVIEW.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
