#!/usr/bin/env python3
"""Build the human review index from case, question and annotation registries."""
import argparse
import os
from pathlib import Path

from devset import DEFAULT_DATASET, ROOT, load_dataset, read_jsonl


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest, lock, cases = load_dataset(dataset)
    questions = {q["case_id"]: q for q in read_jsonl(dataset / manifest["task_questions_file"])}
    groups = {g["case_id"]: g for g in read_jsonl(dataset / manifest["source_groups_file"])}
    answers = {a["case_id"]: a for a in read_jsonl(dataset / manifest["answers_file"])}
    dest = dataset / "reports/CATALOG.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    def link(label, path):
        return f"[{label}]({os.path.relpath(path, dest.parent)})"
    lines = ["# 10个开发视频与证据目录", "",
        "用途：pilot_dev，仅开发。答案不是独立人工金标准；助手视觉审阅状态与人工核验分开。", "",
        "时间窗为原视频绝对秒数，六帧接触表按行对应片内 0、1、2、3、4、4.8 秒。原尺寸证据可点击查看。", "",
        "| ID / 上游视频 | 实际使用的问题 | 当前参考答案 | 原视频窗口 | 审阅状态 | 预览 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for case in cases:
        cid = case["case_id"]
        answer = answers[cid]
        out = ROOT / "artifacts" / manifest["dataset_id"] / cid
        status = {"assistant_visual_supported": "助手画面支持，未人工核验",
                  "assistant_visual_corrected": "助手修正，未人工核验",
                  "assistant_visual_task_revised": "题目时间锚点及答案已修订，未人工核验",
                  "upstream_model_generated_unverified": "上游答案，待画面核对"}.get(answer["review_status"], answer["review_status"])
        preview = link("接触表", out / "contact_sheet.jpg") if (out / "contact_sheet.jpg").exists() else "待准备"
        lines.append("| " + " | ".join(map(cell, [f"{cid} / {case['sample_id']}", questions[cid]["zh"],
            answer["reference_answer"], f"{case['window']['start_s']:g}–{case['window']['end_s']:g}s", status, preview])) + " |")
    lines += ["", "## 逐项取证与使用建议", ""]
    for case in cases:
        cid, window = case["case_id"], case["window"]
        answer, question = answers[cid], questions[cid]
        out = ROOT / "artifacts" / manifest["dataset_id"] / cid
        lines += [f"### {cid}：{case['category']}", "",
            f"- 中文：{question['zh']}", f"- English: {question['en']}",
            f"- 原视频 `sample_{case['sample_id']}`，窗口 `{window['start_s']:g}–{window['end_s']:g}s`；片内时间从零开始。",
            f"- 当前参考答案：`{answer['reference_answer']}`；上游原答案：`{answer['source_answer']}`。",
            f"- 应定位：{case['evidence_target']}。",
            f"- 来源分组：`{groups[cid]['group_id']}`；{groups[cid]['reason']}",
            f"- CSV 数据行：`{case['source_annotation']['data_row_1based']}`（不含表头），原行哈希见 cases.jsonl。"]
        if (out / "reference_lossless.mp4").exists():
            lines.append(f"- {link('5秒无损参考片', out / 'reference_lossless.mp4')}；{link('六帧接触表', out / 'contact_sheet.jpg')}。")
        for change in question["changes"]:
            lines.append(f"- 问题修订：{change}")
        review = answer.get("visual_review")
        if review:
            lines.append(f"- 助手审阅：{review['notes']}")
            for evidence in review["evidence"]:
                lines.append(f"- {link('原尺寸证据帧', ROOT / evidence['path'])}：片内请求时刻 `{evidence['requested_relative_s']:g}s`，"
                    f"原视频名义时刻 `{evidence['nominal_source_s']:g}s`；尺寸 `{evidence['native_size']}`；"
                    f"该帧 ROI 种子 `{evidence['roi_seed_xyxy']}`。")
            lines.append("- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。")
        else:
            lines.append("- 当前尚无完成的画面审阅记录，暂不把上游答案当作金标准。")
        lines.append("")
    lines += ["## 溯源", "",
        f"- DeViBench commit：`{lock['devibench']['revision']}`。",
        f"- StreamingBench revision：`{lock['streamingbench']['revision']}`。",
        "- 原题、原答案、来源 README 和许可文件保留在 provenance/source_snapshots；完整源行另存 annotations/upstream_rows.jsonl。",
        "- 下载和衍生资产的 SHA256、ZIP CRC、命令与工具版本见 provenance/。",
        "- 这些候选尚未接受实际低码率问答测试；是否对压缩敏感需后续实验。", ""]
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
