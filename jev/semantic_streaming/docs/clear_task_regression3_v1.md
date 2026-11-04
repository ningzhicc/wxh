# 衣服文字、计时器和车牌的完整流程回归

入口：`scripts/run_clear_tasks.py`。固定现有Kev检查点、ROI上下文v4、区域编码网格和同一网络轨迹，不进行训练。数据集版本`datasets/clear_task_regression3_v1`在新API调用前选定原开发集的`dev10_01`、`dev10_07`、`dev10_08`；原问题和已有答案保留。已用开发样本不作独立测试集。

每个视频取原定5秒窗口，按原生帧率切成约2秒的连续块，保证每例有起播块和后续选档。每块最多6帧用于API上下文，长2秒块取0/0.4/0.8/1.2/1.6/1.8秒，短尾块仅取实际存在的时刻。API请求上限每例18次，共54次；预计实际每例15次。候选为每块16个CRF×ROI偏移槽位，无ROI时记录普通编码文件别名。

带宽与Kev共享相同真实MP4候选，各自演化缓冲，所选文件真实复制并完整解码、拼接核验；传输时间用轨迹回放，不声称实际限速网络。首块固定最小文件起播，后续满足条件才实际调用Kev，模型调用与强制选档计数保留。

每例的原视频、带宽接收视频、Kev接收视频各抽12张相同时间网格的完整帧，开放问答重复3次，共27次QA；不提供答案选项、正确答案、ROI裁剪、策略名称或编码参数。相同问题/视频的3次请求SHA必须一致。参考答案只在全部QA推理完成后用于评分；使用同一Qwen服务作为QA，尚无独立评测器。

计时器问题保持“第一帧”锚点：必须依据frame_index=0，4:26等之后的倒计时不替换首帧04:30。若起播低质量导致首帧不可读，会保留为失败，不通过改问后续帧来提高分数。车牌不做O/0等字符混同。原图失败、空ROI、无法回答和策略退化均保留。

从工作区`/home/amora/wxh`运行：

```bash
# 准备版本化任务并打印计划，无API/GPU调用。
python jev/semantic_streaming/scripts/run_clear_tasks.py --run-id clear_task_regression3_v1 --plan-only

# 新运行会调用API、编码和GPU推理；完成的原ID再次运行只验证。
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_clear_tasks.py --run-id clear_task_regression3_v1
```

实际运行已放入tmux会话`clear-task-regression3-v1`。GPU忙时保留CPU结果并退出码3，空闲后执行相同命令继续，不停止其他GPU进程；付费请求结果不明时不自动重试。

溯源文件：`runs/clear_task_regression3_v1/{manifest,policy_results,qa_plan,qa_scores,summary,child_seals}.json`、各子运行的原始响应、真实候选字节与完整解码记录，以及源码快照。大媒体在`artifacts/`，元数据、来源与答案在独立`datasets/`目录。几何验收采用 [区域验收口径](region_validation_protocol_v1.md)。

## 已完成结果

27次开放问答全部与参考答案匹配；衣服math Antics、首帧计时器04:30、车牌2K30H，在原图/带宽/Kev三种视频下各3/3。每种策略完整接收解码418帧，均零播放卡顿。Kev实际调用模型5次，无预测异常回退；共45次定位+27次QA。带宽587372字节，Kev560189字节，合计省4.63%；只有衣服一例节省19.07%，另外两例相同。67项测试通过，没有训练。

这是已用开发案例的流程回归；不把27次重复当成27个独立样本。定位仍按每帧产生候选区域，不自动将“第一帧”问题转成后续块的时间预算；本轮验证保真，没有宣称最优任务码率分配。

[原始完整结果](../runs/clear_task_regression3_v1/REPORT.md)、[独立复核](../runs/clear_task_regression3_analysis_v1/REPORT.md)、[代表帧检查](../runs/clear_task_qualitative_review_v1/REPORT.md)。
