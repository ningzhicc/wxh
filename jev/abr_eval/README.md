# Kev 零样本 ABR 评测

使用已有 Kev-4B 服务，在 NetLLM 的 FCC 和 ABRBench 的 HSR、Oboe、Norway3G 测试集上闭环评测，对照 BBA、吞吐率策略和 RobustMPC-6。

实验定义在 [protocol.md](protocol.md)，提示词固定在 `benchmark.py`。不进行模型训练或温度拟合。

## 运行

服务启动方法：

```bash
cd /home/amora/wxh/jev
python3 kev_service.py start
```

全量运行或续跑：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python \
  /home/amora/wxh/jev/abr_eval/run_eval.py \
  --out /home/amora/wxh/jev/abr_eval/results/zero_shot_v1 \
  --methods bba throughput robustmpc kev --workers 4
```

已经完成的会话会复用，设置、数据、代码或模型发生变化时拒绝混合结果，须使用新输出目录。通过 `results/zero_shot_v1/progress.json` 查看已完成数量，`completion.json` 的 `complete=true` 表示所有会话和主报告均已生成。

端到端检查只使用每个集合一条轨迹，不能代替全量结果：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python \
  /home/amora/wxh/jev/abr_eval/run_eval.py \
  --out /home/amora/wxh/jev/abr_eval/results/smoke_v1 \
  --limit 1 --workers 4 --no-report
```

验证仿真器、单位、奖励和 MPC 规划：

```bash
cd /home/amora/wxh/jev/abr_eval
../kev/.venv/bin/python -m pytest test_benchmark.py -q
```

全量结果完成后，生成 QoE 差值分解和动作分布：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python \
  /home/amora/wxh/jev/abr_eval/analyze_results.py
```

回放所有已保存动作并校验结果：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python \
  /home/amora/wxh/jev/abr_eval/audit_results.py
```

## 结果文件

`results/zero_shot_v1/`：

- `REPORT.md`：全量 QoE、码率、卡顿、平滑性、成对差值及延迟。
- `FINDINGS.md`：Kev 的对照结果解释及 QoE 差值分解。
- `summary.csv`、`episode_metrics.csv`、`paired_comparisons.csv`：可直接用于制表或进一步分析。
- `qoe_cdf.png/pdf`、`qoe_comparison.png/pdf`：QoE 分布与汇总比较。
- `action_fractions.png/pdf`、`action_fractions.csv`：各策略六档动作占比。
- `qoe_gap_decomposition.png/pdf`、`qoe_gap_decomposition.csv`：相对每个数据集最强传统基线的码率、卡顿、切换贡献。
- `episodes/`：完整会话 JSON；`logs/`：NetLLM 八列日志格式。
- `manifest.json`：源代码、数据、模型、提示词和起点版本。
- `model_integrity.json`：确认磁盘上的 adapter、head 与配置文件在评测前后未变。
- `audit.json`：全部动作回放、QoE 分解、下一块尺寸映射、概率分布和日志一致性检查。

当前评测使用统一六档 video1/envivio 视频。Norway3G 虽然在 ABRBench 仓库的 4G+ 分组中，本实验将它作为网络轨迹集合接入同一视频环境。主 QoE 不扣除模型推理等待时间，延迟单独记录。上述设置及 RobustMPC-6 与原仓库 MPC 的差异均在协议中明确说明。

延迟定义：Kev 接口 `latency_ms` 是该请求所在批次的模型总耗时，不包含排队；不能视为独享 GPU 的单请求计算时间。客户端延迟包含排队和 HTTP 传输。本轮使用 4 个并发会话。
