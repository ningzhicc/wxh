# 完整实验 llama_abr_v1

2026-10-16：完整流程已完成，`results/llama_abr_v1/completion.json` 中 `complete=true`。未微调和选中模型均完成275条测试，回放、训练状态和比较口径校验全部通过。累计训练7轮，连续三轮无实质验证提升后早停，选中第4轮（验证均值1.007064）；纯训练耗时40.9分钟。

结果见 [REPORT.md](results/llama_abr_v1/REPORT.md) 和 [收敛曲线](results/llama_abr_v1/convergence.png)。最佳检查点为 `/home/amora/wxh/jev/abr_llama/checkpoints/llama_abr_v1/epoch_04`。测试服务已按流程退出；再次运行完整工作流会识别已完成结果，不能把重新启动误认为继续训练。

```bash
cat /home/amora/wxh/jev/abr_llama/results/llama_abr_v1/status.json
tail -n 8 /home/amora/wxh/jev/abr_llama/logs/llama_abr_v1.log
```

顺序为：冻结配置 → 保存未微调 epoch_00 → 四组275条未微调测试 → 初始验证两次 → 每轮训练/保存/验证两次 → 至少4轮、最多12轮、patience3/min_delta.005早停 → 恢复验证均值最佳轮进行四组测试 → 校验和报告。训练与推理共用原 Kev 环境，独立使用8010端口，评测后服务退出，不占用其他服务端口。

未微调模型带随机 pointer head；六档候选来自ABR协议，不能称为训练好的通用Kev。7,224条数据、55轨迹、30验证窗口及原测试均保持原样。LoRA lr2e-5、head lr1e-4，有效batch16。完整规则在 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，运行实际配置、代码/权重/数据哈希在 `runs/llama_abr_v1/experiment.json`；它优先于早先 `prepared/llama_1b_v1.json` 的两轮训练草案。

`checkpoints/llama_abr_v1/epoch_00` 是同一seed的未微调起点，之后每轮包含可加载适配器、pointer head、分隔符约定，以及Adam、调度器、随机状态等轮次边界恢复数据。`training_progress.json` 每十步更新；`status.json` 的步骤只在切换阶段时更新，查看实时进度用前者。每轮452步。

完成后主要文件：

- `results/llama_abr_v1/untrained_test/summary.json`：未微调四组结果。
- `results/llama_abr_v1/controller.json`：各轮两次验证、最高分与早停计数。
- `results/llama_abr_v1/selected.json`：选中的轮次和权重位置。
- `results/llama_abr_v1/test/summary.json`：选中模型四组结果。
- `results/llama_abr_v1/REPORT.md`：完整比较、成对区间及限制。
- `results/llama_abr_v1/convergence.png`、`qoe_comparison.png`：训练验证及测试图。

任务若实际退出，可以恢复；客户端中断本身不等于后台退出，不能直接再启动一份：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_llama/workflow.py --run-name llama_abr_v1 --resume --detach
```

恢复要求冻结配置及代码哈希一致，从最后完整轮次恢复；轮内工作需要重做，未完成`.pending`目录保留并改名，不当作已完成检查点。测试轨迹只在指纹相同时复用。

离线极小Llama检查已通过。真实1B模型探针 `llama_probe_v2` 完成两次真实更新、两轮保存/加载、3次小验证，并精确核对CPU/GPU转换前后参数及Adam状态。首次探针 `llama_probe_v1` 在修正端口检查前失败，目录保留，两个探针均不进入正式选模或测试比较。
