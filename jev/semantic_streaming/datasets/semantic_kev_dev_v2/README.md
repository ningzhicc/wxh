# 10 视频、每视频12动作的本地决策数据

上游：[完整编码候选表](../../runs/roi_encoding_dev10_grid_v1/REVIEW.md)。每个视频有3个全局CRF档位×4个ROI偏移，共12个实际编码动作，10个视频合计120个MP4。

`actions.json` 使用与上游编码动作表一致的动作键，通过 `candidate_id`、文件路径与SHA256追溯到实际视频块。`train.jsonl`有4800条，`valid.jsonl`有1200条，保留既有16条训练、4条验证网络轨迹的独立来源分组。视频为同一批开发视频，不能据此报告视频泛化性能。

## 当前状态：发布版试训完成，下一轮使用ABR基线初始化

已完成全部记录的真实Kev tokenizer/编码接口预检查，无模型权重加载、无GPU训练。`kev_preflight.json`记录最大状态143 tokens、最大完整输入868 tokens及实际上下文限制，未截断或删减候选。

用户已手动完成发布版Kev初始化的100步试训，模型见 `runs/semantic_kev_v2_trial100_20261103`，验证集1200条预测均与最小文件基线相同。用户现选择 `merina656_baseline_seed666/checkpoints/epoch_08` 作为下一轮初始化；配置已切换并通过全部6000条CPU预检查，尚未启动新的GPU训练。原始模型和已完成试训保持原记录。

只做CPU预检查：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py \
  --dataset datasets/semantic_kev_dev_v2 --out runs/semantic_kev_dev_train_v2
```

## 输入与监督

在线输入只有视频问题、过去链路带宽、缓冲、上一档码率以及各候选的CRF、ROI偏移、实际字节、实际码率、预计下载时间。使用紧凑候选文本以容纳全部12个动作；推理接线须同样调用 `prepare_semantic_kev.request(..., compact=True)`。

标签在带宽轨迹相同起点分别回放各候选实际字节后生成。奖励为 `clip(采样证据RGB PSNR/40,0,1) - 4.3*卡顿秒数 - 0.05*码率变化Mbps`。真实未来下载时间、质量代理和奖励只在 `_supervision`，不送入模型。缓冲来自场景枚举，不冒充实际在线策略轨迹。

本版使用现有Kev的监督微调入口学习选档，是一步回放模仿训练；不称为在线强化学习。已完成的100步checkpoint从发布版初始化，不能误记为从ABR微调版继续训练。

## 下一轮初始化

`configs/kev_initialization_v1.json` 默认选择 `abr_baseline`，指向 `merina656_baseline_seed666/checkpoints/epoch_08`。基座为固定revision的Qwen3.5-4B-Base，加载已有ABR LoRA（rank16）和256维决策头。模型文件逐项验SHA256，来源缺失、文件变化或结构不匹配时退出，不自动换回发布版。

训练入口支持 `--init-source abr_baseline`（当前默认）和 `--init-source released`（明确选择发布版对照）。新训练必须使用新输出目录。`--preflight-out` 可将不同初始化的CPU审计单独保存；本次ABR初始化审计见 [preflight.json](../../runs/semantic_kev_abr_baseline_preflight_v1/preflight.json)。该审计记录的 `training_argv` 已确认指向ABR的epoch_08，`training_started=false`。

epoch_08是这轮8个epoch中最后保存的完整模型，有传统ABR测试记录；它尚未被统一验证协议确认为所有本地ABR版本中的最佳。当前训练数据的偏斜仍按原样保留，切换初始化不会自动修正这些标签。

## 来源与边界

- `manifest.json`：源编码表哈希、配置、数量与质量代理定义。
- `traces.lock.json`：网络轨迹路径、原快照SHA256和来源分组。
- `actions.json`：每视频12个可选动作，含源ROI问题标记。
- `code/prepare_semantic_kev.py`、`config.json`：本次生成代码与参数快照。
- `kev_preflight.json`：CPU兼容检查结果，`training_started=false`。
- 相邻 `semantic_kev_dev_v2_rejected_context_v0`：首次详细候选文本超过上下文限制的失败版本及原因；没有覆盖失败记录。

三个有问题的来源框仍带有标记；质量代理不是语义准确率，也不是人工金标准。此处先完成最小系统的数据接口，不额外调用API或重复问答。
