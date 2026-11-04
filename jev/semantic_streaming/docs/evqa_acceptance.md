# 五视频验收入口

本轮验证固定的 v4 400步模型，不进行训练。入口为 `scripts/run_evqa_acceptance.py`。

五个样本来自固定版本 `Salesforce/ST-Evidence-Bench@8194befd7e4f52071664a46033ae9415e0c7a939`：碗的颜色、毛巾的颜色、取出的袜子颜色、接住的物体、坐到床上前放下的物体。保留原始英文问题和公开答案选项，不翻译、不改答案。

## 开始

先进入 tmux：

```bash
tmux new -s evqa-accept5
```

在该会话内执行：

```bash
cd /home/amora/wxh
set -o pipefail
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_evqa_acceptance.py \
  --run-id evqa_accept5_v1 \
  2>&1 | tee -a jev/semantic_streaming/runs/evqa_accept5_v1.log
```

按 `Ctrl+B`，松开，再按 `D` 离开；重新进入用 `tmux attach -t evqa-accept5`。完成结果在 `runs/evqa_accept5_v1/REPORT.md`。

上述命令会执行真实 GPU 推理和付费 API 调用。新验收上限为90次ROI定位＋15次接收端QA，共105次；实际视频较短或运行已完成时会更少。密钥继续从既有本地配置读取，不出现在命令和运行记录中。不会启动训练，也不会改原模型。

只核对素材和计划，不调用API或GPU：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/run_evqa_acceptance.py \
  --run-id evqa_accept5_v1 --plan-only
```

## 命令会完成什么

1. 校验五条视频、原始问答行和来源版本。素材只下载ZIP中所需成员，约19MB；不下载整个8.9GB视频包或空间掩码包。发布媒体为6fps，保留该帧率。
2. 每个视频取包含完整标注证据的0～15秒窗口（不足15秒则保留实际长度），调用既有三步入口，用受约束带宽策略建立真实编码目录。
3. `replay_semantic_pipeline.py` 让Kev直接使用同一套候选MP4文件；不重复定位和编码。两种策略使用同一网络起点，各自推进缓冲、带宽观测和上一档码率，真实复制／解码所选文件。
4. 在原参考视频、带宽策略接收视频和Kev接收视频上，以相同帧序号均匀采样最多12张完整画面。随机打乱15项QA调用顺序；同一案例使用相同答案选项排列。
5. QA请求只包含问题、所有候选答案和完整画面，不包含正确选项、编码方式、ROI或参考答案标签。全部推理完成后才解析离线参考答案进行评分。
6. 报告原视频能否答对、两种接收视频的答对情况、字节数、卡顿及决策记录。原视频答错的案例单列；条件保真率仅在原视频答对的子集计算。

## 结构与溯源

```text
datasets/evqa_accept5_v1/
  cases.jsonl                  视频、问题、公开QA选项；不是直接推理输入
  annotations/answers.jsonl    参考答案和取证时间，仅评分使用
  provenance/                 原始CSV、README、源行、下载CRC/SHA与媒体探测
  sources.lock.json            固定版本、ZIP元数据和本地微调源视频SHA
media/raw/evqa_accept5_v1/     五个原始发布视频，Git忽略
runs/<study-id>/              五条任务配置、子运行日志、QA原始响应和验收报告
runs/<study-id>_evqa_XX_bandwidth/  定位、编码及带宽策略输出
runs/<study-id>_evqa_XX_kev/        复用同一候选的Kev输出
artifacts/<study-id>/qa_frames/    原视频／接收视频相同时间网格的QA图片
```

`cases.jsonl` 不能整条传给ROI或Kev；批处理脚本只取视频、问题和窗口生成白名单配置，测试已检查答案选项及参考答案不会进入这两个阶段。QA中的所有选项属于原数据集题目，不指定哪个选项正确。

数据集准备阶段检查新视频字节SHA与当前本地微调使用的原始视频SHA没有重复，不声称所有基础模型预训练没见过它们，也不能仅凭SHA证明不同剪辑的来源家族不重叠。来源是公开研究基准，作者声明CC-BY-NC 4.0／研究使用，原始README随来源记录保留。

## 恢复及边界

同一代码、配置和输入下重跑原命令，会检查已完成子运行和请求缓存。GPU忙时保留已完成定位／编码，等待用户在GPU空闲后重新运行；不会停止其他进程。付费请求若出现失败或结果不明，会停止并保留现场，不自动重试。

这五条只是小规模接线和质量验收：问题为二／四选一，同一个Qwen服务兼任定位与QA，不是独立评测模型，也不是完整EVQA准确率。标注答案未在本地进行人工认证。网络仍为固定轨迹回放和本地文件交付，没有实际网络吞吐测量。本轮不评估空间掩码指标，不扩大训练集或微调模型。

来源：[EVQA项目](https://github.com/SalesforceAIResearch/EVQA)、[ST-Evidence-Bench](https://huggingface.co/datasets/Salesforce/ST-Evidence-Bench)。
