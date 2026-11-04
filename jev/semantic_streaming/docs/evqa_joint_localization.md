# 多帧上下文定位 v4 开发实验

固定已有 v4 Kev 权重、编码网格、网络轨迹、原始问题、选项排列及标准答案，只替换定位方式。保留旧五视频验收目录和原入口脚本；新入口复用旧阶段引擎，新增模块与入口写入运行指纹和源码快照。

## 目录与用途

- `datasets/evqa_accept5_v1/`：原始五视频冻结元数据，不修改。
- `datasets/evqa_evidence_audit_v1/`：固定 Hugging Face revision 的两个案例官方掩码、ZIP成员CRC/SHA、HTTP范围下载收据；仅用于离线检查。
- `runs/evqa_evidence_audit_v1/`：旧定位的离线核查。
- `configs/qwen_roi_context_v4.json` 与对应提示词：版本化API设置。
- `scripts/pipeline_roi_context_v4.py`：有上限的联合请求、完整逐帧输出验证、绝对像素换算和可复核收据。
- `scripts/run_semantic_pipeline_context_v4.py`：新定位入口，编码、网络控制和接收端沿用原实现。
- `scripts/run_evqa_context_acceptance_v4.py`：先两例、再五例的开发对比；完整运行可显式复用已完成的两例子运行。

## 实验协议

每个5秒视频块最多6张原始采样帧作为上下文，每次只为最后标为TARGET_IMAGE的目标帧返回一个原始frame_id及regions对象，沿用已验证的单帧输出契约。一个15秒案例18次调用，五段视频共82次定位调用。上下文参与辨认，不是一次批量输出所有帧。输出每帧0至4个矩形，转换为绝对像素坐标。原图尺寸、采样时刻和问题保持原样，API不接收GT掩码、答案、选项或标注时间窗。对不能唯一指代的对象保留合理候选，不能凭预设答案选目标。

这是离线VOD预处理，可使用窗口内后续帧，不是在线因果定位。单块超过6张采样图时明确拒绝，不自动截断；较长视频按块增加请求并受显式调用上限约束，跨块目标跟踪尚未实现。单帧缓存不能冒充联合上下文缓存；付费请求结果不明时不自动重试。密钥仅在内存中读取，凭证和请求头不写入快照。

每例重新实际编码16档；带宽与Kev策略共享这些文件。主QA沿用12张均匀全帧图片、同一问题和同一选项，不使用GT裁剪或时间筛选。参考答案在全部QA完成后读取评分。两个案例先验收，五案例全量验收复用两例的已封存编码/策略结果；QA可独立重测，因此同像素的回答波动须单独核对，不能据此宣称编码收益。

原始结果：原视频3/5，带宽4/5，Kev3/5；Kev比带宽少8.7%字节，均零播放卡顿。v4保留所有案例，不删除难例，不改标准答案。五例已用于开发，不能作为独立最终测试集。

## 运行

在工作区 `/home/amora/wxh`：

```bash
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_evqa_context_acceptance_v4.py --run-id evqa_context2_v4 --cases evqa_01 evqa_02
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_evqa_context_acceptance_v4.py --run-id evqa_context5_v4 --reuse-study evqa_context2_v4
python jev/semantic_streaming/scripts/audit_evqa_evidence.py --run-id evqa_evidence_comparison_v4 --comparison-study evqa_context5_v4
```

首次两例最多36次定位+6次QA，后续五例复用两例定位，最多46次定位+15次QA，v4总上限103次付费请求；加上被拦截的v2、v3探索请求各1次，整个改进过程上限105次；无新训练。遇到GPU占用退出码3时保留编码结果，用相同命令恢复。已完成结果再次运行只核验，不再次调用API。

## 掩码对齐限制

官方掩码导出记录为183/185帧，发布视频各180帧；数字索引匹配属于暂定对齐。6、7、8秒目视检查对应目标吻合，报告另给±1/±3帧敏感性，不选择最优偏移抬高指标。补黑掩码不能当成确认目标不存在。覆盖率仅作诊断，不是官方J&F。

## 保留的失败探索

`runs/evqa_joint2_v2_evqa_01_bandwidth/roi/joint_0000/` 保存18帧一次请求的原始响应。该响应输出19个条目、包含不存在的帧ID和多帧重复坐标，被严格解析器拒绝；未进入编码/训练，也未自动重试。`v2`模块和快照为溯源保留；后续由`v4`的多帧上下文、单目标输出协议替代。

`runs/evqa_joint2_v3_evqa_01_bandwidth/roi/joint_0000/` 保存6帧联合输出的探索响应；虽finish_reason=stop但JSON本身损坏，被解析器拒绝。未自动重试、未进入编码。`v4`改为多帧上下文、单目标输出，复用原单帧解析器和付费请求状态机；上下文清单在`roi/context_plan.json`与源码一起封存。
