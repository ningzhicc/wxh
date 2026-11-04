# 延长训练与最佳轮选择

运行名：`sft_extend_v1`。从原 `sft_v3/epoch_02` 的权重继续训练。初步估计累计六轮左右可能进入平台，最少累计六轮、最多十轮；实际按逐轮验证决定何时停止，原模型及报告保留。

每个轮末检查点在固定清洁 FCC-valid 30 条轨迹上评估两次，以平均 QoE 选模。连续三轮没有达到每块 QoE +0.005 的实质改善，且累计至少六轮时停止；保留绝对最高验证均值的模型，停止后回切到该轮。第一、二轮也重新重复验证，作为候选。

训练期间本机 GPU 用于优化，8009 服务暂停；每轮验证期间启动对应候选模型，随后继续训练。结束时才将服务固定到选中检查点。每轮约 19 分钟，另有验证时间；后台运行不依赖客户端保持连接。

## 查看进度

```bash
cat /home/amora/wxh/jev/abr_finetune/results/sft_extend_v1/status.json
tail -n 5 /home/amora/wxh/jev/abr_finetune/logs/sft_extend_v1.log
```

`training_progress.json` 记录当前累计轮数、优化步骤和损失；`controller.json` 记录逐轮验证均值、最佳轮和早停计数。探针 `extend_probe_v1` 验证了两次小样本更新、模型 CPU/GPU 转移、优化器状态完整性，其模型不进入正式候选。

每轮模型位于 `checkpoints/sft_extend_v1/epoch_XX`，其中 `training_state.pt` 保存 AdamW、学习率调度、随机数状态及样本顺序。训练进程异常退出时，从最近完整轮末检查点恢复：

```bash
/home/amora/wxh/jev/abr_finetune/.venv/bin/python /home/amora/wxh/jev/abr_finetune/continue_training.py --run-name sft_extend_v1 --resume --detach
```

轮内未保存的更新需要重做；恢复到的是最近完整轮末状态。进程运行期间不要重复启动。若训练已结束但最后评测中断，单独恢复评测即可：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_finetune/finish_continuation.py --run-name sft_extend_v1
```

只有 `results/sft_extend_v1/completion.json` 中 `complete=true` 才表示训练、选模及四组全量测试均完成。最终模型路径与累计最佳轮数记录在 `selected.json`，完整结果见 `REPORT.md`，验证/损失曲线见 `convergence.png`。

本轮沿用原 55 条源隔离轨迹、7,224 个状态，不改成 84 条。上一阶段没有保存优化器，本轮开始重建一次优化器，学习率峰值 1e-5；随后各轮状态连续。固定测试集不参与选模；后续测试比较不称作新未见测试集。完整规则见 [CONTINUATION_PROTOCOL.md](CONTINUATION_PROTOCOL.md)。
