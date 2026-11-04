# 下一轮定位任务的审校草案

本文件保留明确指代问题的审校依据。下列新问题现已建立独立数据集`evqa_disambiguation_dev_v1`并完成API开发实验，不改变原EVQA问题、标准答案或已封存评分。修订任务仍不进入正式训练，结果不冒充原EVQA得分。

| 原任务 | 候选新问题（不包含答案颜色） | 当前问题 | 审核要求 |
|---|---|---|---|
| evqa_01：What color was the bowl I saw? | What color is the bowl placed upside down in the metal drying rack to the right of the kitchen sink? | 画面中有多个碗，官方掩码对应右侧沥水架的碗，泛指问题无法唯一确定对象；多帧上下文未明显修复漏框。 | 核对上下文指代与框是否对应同一物体；把此问题作为单独任务，保留原问题的失败结果。 |
| evqa_02：What color is the towel? | What color is the towel hanging on the cabinet door beneath the bathroom sink? | 定位覆盖已较好，原图QA仍与green标注不一致；有明显暖色光照。 | 先人工复核目标、光照、色彩与标准答案。保留原green标签，未复核前不拿这个案例作为颜色保真的主要证据。 |

审校时查看原始完整帧、官方掩码和标注时间窗，记录审核者、日期、依据图像SHA、是否改写问题、原/新标签及理由。GT掩码只用于此离线核对，不进入主流程定位或Kev输入。不能用模型自己的答案反过来自动改标准答案。

当前多帧上下文接口与批量输出探索的实现、失败记录及费用调用上限见 [evqa_joint_localization.md](evqa_joint_localization.md)。正式训练前，应先在明确指代的开发任务上验证定位，再冻结另外一批未参与调试的视频与网络轨迹。

本轮进展：碗的明确指代问题在原图/带宽接收图/Kev接收图上各重复3次，9次均答light blue；GT时间窗内6个采样点空框由4降为0。暂定API掩码覆盖率58.7%，未达到预设80%门槛，仍需核对空间覆盖与索引对应。毛巾新旧问题各3次均答blue，green原标注保持待人工核验。详见 [修订任务报告](../runs/evqa_disambiguation_v1/REPORT.md)、[三步系统检查](../runs/evqa_disambiguation_analysis_v1/REPORT.md) 与 [原图审核材料](../runs/evqa_disambiguation_v1/review.md)。
