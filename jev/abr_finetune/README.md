# Kev-4B 的 ABR 微调

后续延长训练运行 `sft_extend_v1` 的进度、早停规则和恢复方式见 [CONTINUATION_README.md](CONTINUATION_README.md)。该阶段会在训练时暂停 API、验证时加载候选，结束后回切验证最佳轮；以下表格是先前完整两轮实验的结果。

本目录从本地发布版 Kev-4B v1.0 继续训练 LoRA 与 pointer head，冻结 Qwen3.5-4B-Base。原发布模型与零样本评测结果保留在原目录。

**2026-10-14：两轮训练、验证选模和 275 条全量测试均已完成。** 选中模型为 `checkpoints/sft_v3/epoch_02`，当前服务 `http://127.0.0.1:8009` 已加载这份检查点。训练耗时 38 分钟，完整动作回放校验了 13,200 个视频块与 12,925 份决策概率。

平均 QoE/块（越高越好；不计推理等待）：

| 数据集 | 原发布版直接接入 | ABR 微调后 | RobustMPC-6 |
|---|---:|---:|---:|
| FCC | -33.363 | 1.008 | 1.009 |
| HSR | -0.645 | 2.807 | 2.862 |
| Oboe | -13.180 | 2.151 | 2.081 |
| Norway3G | -9.885 | 1.359 | 1.344 |

四组相对发布版均明显改善。与 MPC 的成对 bootstrap 差值区间在 FCC、HSR、Norway3G 上跨 0；Oboe 的差值为 +0.0697，区间 [+0.0385, +0.1049]。完整 BBA/Throughput 对照、卡顿、延迟与解释范围见 [评测报告](results/sft_v3/REPORT.md)。

本轮运行名为 `sft_v3`。训练和后续验证、测试均使用独立后台进程，客户端断开不会主动结束任务。只有 `results/sft_v3/completion.json` 存在且 `complete=true` 才表示整个流程结束；训练损失下降不等于闭环 QoE 已改善。

## 数据与方法

- 从 NetLLM 官方 `fcc-train` 目录原 235 条轨迹中筛选 55 条：16 条 FCC 采集记录、39 条 Norway 路线日志。不是纯 FCC 来源。
- 排除与验证/测试相同的原始 FCC 采集 ID、完整 Norway 日志、完全重复数值轨迹，以及训练/验证重复状态。还搜索了连续 16 点的非恒定带宽重复片段，没有发现训练/验证与测试之间的匹配。
- 得到 7,224 个唯一训练状态，六档码率标签均由 RobustMPC-6 规划生成。状态来自 MPC、BBA 和探索行为三种 rollout。
- 清洁 FCC-valid 为 30 条窗口、两个原始采集 ID；两轮检查点按其闭环平均 QoE 选择。验证来源少，不能据此声称广泛泛化。
- 训练两轮，LoRA rank=16，lr=2e-5，batch=2，accum=8，904 个优化步骤。冻结基座/主参数 FP32，BF16 autocast，gradient checkpointing。
- 四组测试仍为 FCC 100、HSR 34、Oboe 100、Norway3G 41，共 275 条。与原零样本使用相同轨迹、起点、视频、提示词、候选和仿真器，复用原 BBA/Throughput/MPC 结果作成对比较。
- 这是监督模仿，没有强化学习；没有 ABR 温度校准。主 QoE 排除固定起始块，不计推理等待。

## 查看进度

```bash
cat /home/amora/wxh/jev/abr_finetune/results/sft_v3/status.json
tail -n 5 /home/amora/wxh/jev/abr_finetune/logs/sft_v3.log
tail -n 10 /home/amora/wxh/jev/abr_finetune/logs/sft_v3.pipeline.log
```

训练日志为 `logs/sft_v3.log`，训练指标为 `runs/sft_v3/training_metrics.json`。每轮检查点位于 `checkpoints/sft_v3/epoch_01` 和 `epoch_02`，最终选择写入 `results/sft_v3/selected.json`。默认发布模型文件不被覆盖。

完整报告生成在 [results/sft_v3/REPORT.md](results/sft_v3/REPORT.md)，含均值、配对 bootstrap 区间、码率、卡顿、与三种基线比较、PNG/PDF 图表。原始逐步数据在 `results/sft_v3/test/episodes/`，八列 NetLLM 格式日志在 `test/logs/`。

## 运行与恢复

两轮 LoRA 训练没有保存优化器的逐步恢复状态；若训练进程本身异常退出，只保留已经完成的每轮模型检查点，不能声称从中断步骤精确恢复。`sft_v1` 是客户端中断前的未完成运行；`sft_v2` 在发现同源窗口重叠后主动停用，旧数据保留于 `data_v1_numeric_only/`。这两个运行不进入最终模型选择或测试比较。

已完成的数据、检查点和逐轨迹评测都可复用。后续评测管线中断时可重启相同命令，它会跳过完整结果、恢复尚未完成的轨迹，并检查指纹；不会重复启动训练：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_finetune/pipeline.py --run-name sft_v3 --detach
```

如果训练需要新的尝试，保留已有目录，使用新运行名。训练期间不要同时启动 GPU 推理服务：

```bash
/home/amora/wxh/jev/abr_finetune/.venv/bin/python /home/amora/wxh/jev/abr_finetune/train_job.py --run-name NEW_RUN --detach
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_finetune/pipeline.py --run-name NEW_RUN --detach
```

训练环境单独安装 FLA/Triton；闭环推理使用原 Kev `.venv`，与零样本相同。没有改动上游 Kev 训练算法，`train_wrapper.py` 只记录构造对象并在每轮末保存模型。

## 使用选中的模型

评测流程结束后会保持选中模型运行于 `http://127.0.0.1:8009`，ABR 调用格式与之前相同。查看 `selected.json` 中的 `checkpoint`，可用此路径重新启动：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/kev_service.py stop
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/kev_service.py start --checkpoint CHECKPOINT_PATH
```

不传 `--checkpoint` 会启动原发布模型。`kev-latest` 是请求别名，实际加载哪份权重以 `/v1/models` 中的 `run` 路径为准。

训练数据与隔离结果：[data/manifest.json](data/manifest.json)、[data/source_overlap.json](data/source_overlap.json)、[data_audit.json](data_audit.json)。完整设置见 [PROTOCOL.md](PROTOCOL.md)。源标识由文件名推断；不同重采样、比例变化或无法识别的名称不能仅靠本次检查证明来源独立。
