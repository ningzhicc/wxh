# 明确问题指代与重复问答开发实验

入口：`scripts/run_evqa_disambiguation.py`。本轮只修改碗的问题指代，使用冻结的上下文定位v4、区域编码、网络轨迹和Kev检查点；管线配置核对只允许`question`变化，其他参数或实现变化会中止。原EVQA任务与评分不修改。

## 新任务与角色

数据集版本为`datasets/evqa_disambiguation_dev_v1`，3个问题、2个源视频：

- `bowl_rack_v1`：What color is the bowl placed upside down in the metal drying rack to the right of the kitchen sink?
- `towel_cabinet_v1`：What color is the towel hanging on the cabinet door beneath the bathroom sink?
- `towel_original_control_v1`：保留原始 What color is the towel?，作为相同源视频的新旧问题对照。

修订后的问题不包含答案颜色。参考答案单独放在`annotations/answers.jsonl`，继承原标签并保留未独立人工核验状态；毛巾明确标记争议。所有任务为开发用途，`training_eligible=false`；重复次数不算独立视频或独立测试样本。

## 执行范围与复核

碗重新执行18帧上下文定位、3块×16档真实编码，以及共享同一候选表的带宽策略与Kev选档，接收视频完整解码。原图、带宽接收图、Kev接收图各用12张均匀采样全帧图片重复问答3次，共9次。毛巾复用已验证的原始完整帧，新旧问题各重复3次，共6次。总上限18次定位+15次QA，不自动重试失败或结果不明的请求，无训练。

重复问答的网络请求正文SHA必须完全一致；重复编号和策略名称仅在本地记录。输出保留3次答案分布、与原标签匹配次数和一致性，不把多数票当成独立真值。参考答案只在全部QA完成后用于评分。独立评测器尚未加入。

全部推理结束后，官方掩码仅用于比较通用问题与明确指代问题的定位覆盖。数字帧号对应为暂定对齐，报告保留±1/±3帧敏感性，不择优偏移。掩码、GT时间段和答案均不输入定位或Kev。

预先设置的开发诊断门槛是暂定掩码像素覆盖率至少80%，且原图重复3次均与继承参考答案一致；通过也不等于独立测试或人工核验。毛巾真实颜色缺少独立依据时继续保留争议，不依据模型输出改原标签。

## 命令

在工作区`/home/amora/wxh`：

```bash
# 仅准备/校验版本化任务并展示调用计划，无API/GPU。
python jev/semantic_streaming/scripts/run_evqa_disambiguation.py --run-id evqa_disambiguation_v1 --plan-only

# 完整执行；已完成的同名运行只做哈希核验。
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_evqa_disambiguation.py --run-id evqa_disambiguation_v1
```

结果位于`runs/evqa_disambiguation_v1/`：`REPORT.md`、`summary.json`、`grounding_audit.json`、`qa_scores.json`、`repeated_payload_receipt.json`、`controlled_input_receipt.json`以及源码快照。`child_seals.json`关联真实编码和接收运行。`review.md`展示原始全帧，`human_review_template.json`用于另存人工复核记录；不要编辑封存的模板或原标注。
