# 明确目标的新问题：开发试验

新任务单独命名，原EVQA题目与评分保留。冻结多帧上下文方法、编码、网络和Kev权重，仅改问题指代。

| 任务 | 视频 | 回答分布（3次） | 与继承参考答案匹配 | 三次一致 |
|---|---|---|---:|---|
| towel_original_control_v1 | reference | {'blue': 3} | 0/3 | True |
| towel_cabinet_v1 | reference | {'blue': 3} | 0/3 | True |
| bowl_rack_v1 | reference | {'light blue': 3} | 3/3 | True |
| bowl_rack_v1 | bandwidth | {'light blue': 3} | 3/3 | True |
| bowl_rack_v1 | kev | {'light blue': 3} | 3/3 | True |

| 碗的定位版本 | GT时间内非空掩码采样 | API空框 | 暂定像素覆盖率 |
|---|---:|---:|---:|
| generic_context | 6 | 4 | 28.98% |
| explicit_target_context | 6 | 0 | 58.70% |

碗：带宽策略 630636 字节，Kev 485337 字节；卡顿分别 0.0 / 0.0 秒。

重复问答使用完全相同的问题、选项和全帧图片，仅本地重复编号不同。三次一致不等于独立评测，也不是完整基准准确率。
毛巾的green标签继承自原数据，原图多次输出blue也不自动改标签；保持待人工复核且禁止作为已核验训练标签。新碗问题的标签也尚无独立人工确认。
掩码只用于所有推理完成后的离线核查，绝不输入定位、Kev或主QA；索引对应存在不确定性，覆盖率仅作诊断。
原标注与实际画面审核材料见 [review.md](review.md)。本轮没有训练。
