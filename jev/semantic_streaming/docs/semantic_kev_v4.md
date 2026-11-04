# v4：候选约束、回退与三块规划监督

后续状态：本文的 `trial400_v1` 已完成400步训练和闭环评估，见 [结果分析](../runs/semantic_kev_v4_abr_baseline_trial400_v1_analysis/REVIEW.md)。模型奖励61.78、每场景卡顿1.197秒，优于受约束带宽基线58.94／1.793秒；增益主要集中于一个场景，尚无独立视频泛化结论。下文保留准备阶段的设计、检查及当时的训练命令；复跑必须同时更换输出目录、日志文件及 tmux 会话名，避免覆盖历史日志。

2026-11-03。已完成用户指定的两项修改：候选约束／回退和考虑后续缓冲的训练目标。本轮仅进行 CPU 数据生成、回放及输入检查，没有启动 GPU 训练、GPU 模型推理或 API 调用。

## 当前入口与历史保留

- 默认训练数据：`datasets/semantic_kev_dev_v4`，3639条训练、999条验证记录。
- 复用 v3 的160个编码文件、网络来源分组、100个场景起点及20片段播放列表。每视频完整目录仍有16候选；每次向模型展示2～16个符合预算的候选。
- 训练、推理、教师规划使用同一个 `candidate_guard`。只有一个候选时不调用模型；无候选满足预算时明确回退到最小文件，不能把这次回退声称为无卡顿选择。
- 训练默认仍从原始 `merina656_baseline_seed666/checkpoints/epoch_08` 初始化；新权重保存至独立目录。v1～v3 数据、旧训练结果和原始 ABR 权重保持不变。
- [数据清单](../datasets/semantic_kev_dev_v4/manifest.json)、[来源锁定](../datasets/semantic_kev_dev_v4/source.lock.json)、[完整检查](../runs/semantic_kev_v4_audit_v1/REVIEW.md)、[实际输入检查](../runs/semantic_kev_v4_preflight_v1/preflight.json)。

起播、单候选和无满足预算候选的情况仍保留在 `train_rollouts.jsonl`／`valid_rollouts.jsonl`：训练4800个回放块中有1161个强制选档块，验证1200个回放块中有201个。这些块不占模型训练记录，但完整进入闭环评估，不能只评估容易选档的部分。

## 1. 候选约束与回退

历史带宽来自已经完成的传输，不读取即将到来的网络区间。保守预测为：

```text
预测带宽 = 0.85 × min(历史带宽调和均值, 最近一次观测)
预测下载时间 = RTT + 8 × 完整文件字节 / (0.95 × 预测带宽 × 1e6)
缓冲预算 = max(0, 当前缓冲 - 1秒)
可持续预算 = 当前片段时长 + 0.25 × max(0, 当前缓冲 - 10秒)
允许下载时间 = min(缓冲预算, 可持续预算)
```

预测下载时间不超过预算的候选全部保留。高缓冲允许有限度消耗储备，低缓冲优先维持传输能力；没有按真实未来带宽筛选候选。

起播统一选择最小文件建立观测。如果没有候选满足预算，仍传最小文件，并记录 `no_candidate_within_budget`。模型返回不在允许集合的动作，或概率验证失败时，回退到允许集合中的最小文件并记录原因。硬件运行错误不会被静默当成正常回退。

## 2. 三块累计目标

教师规划当前块及后续两块，用同一保守历史带宽预测下载时间；每个计划动作都会更新预测缓冲、上一档码率及下一步候选约束。精确搜索当前允许动作对应的最优后续序列，目标为：

```text
累计预测收益 = r0 + 0.95*r1 + 0.95²*r2
              - 0.95³*0.25*max(0, 目标末端缓冲 - 预测末端缓冲)
r = 证据PSNR/10 - 4.3*卡顿秒数 - 0.05*切档幅度(Mbps)
```

目标末端缓冲不超过10秒，也不超过规划窗口之后的剩余播放时长；临近播放结束时缩短规划窗口，不要求结束后仍保留10秒缓冲。起播等待仍按原系数单独计入奖励。

未来片段大小来自 VOD 中已经生成的编码目录。完整字节数组以紧凑 CSV 写在问题说明中，顺序由固定 CRF／ROI 网格说明，未删除任何目录项；kbps 可由字节数和当前片段时长计算。这样避免把长数组塞入512-token状态。

**教师使用离线测得的证据 PSNR 作为监督信息，Kev 输入不包含这些质量值。** 未来带宽、真实反事实下载时间、参考答案和规划回报也不进入输入。真实网络回放仅推进物理状态、产生下一次已完成传输的观测和用于独立评估，不决定当前教师标签。

训练仍调用现有 Kev 监督分类训练器，模仿规划教师的首个动作；本轮没有改成在线 RL，也没有增加按 regret 加权的损失。

## 验证结果

40项单元测试通过，包括：禁止动作拦截、无可行候选回退、无质量信息的在线筛选、硬件错误不被吞掉，以及一个三块规划会主动牺牲当前质量以保留后续缓冲的解析案例。

完整数据逐条重算候选约束、三块标签、物理奖励和连续状态；网络来源组无重叠。全部4638条模型决策记录通过实际 Kev tokenizer／encoder，最大状态236 tokens、完整输入1019 tokens，没有截断或放大限制。

首次 JSON 预览使状态达到561 tokens，失败版本保存在 `datasets/semantic_kev_dev_v4_rejected_context_v0` 及对应 rejected 日志中；最终只调整输入表述，保留完整大小信息和原始算法参数。

同一20个验证场景，每场景20片段、约100秒，回放使用与 v3 相同的质量和卡顿奖励：

| 策略 | 平均每场景奖励 | 平均卡顿秒数 | 平均证据PSNR dB |
|---|---:|---:|---:|
| 原历史带宽策略 | 50.84 | 3.354 | 37.01 |
| 加入候选约束的带宽策略 | 58.94 | 1.793 | 37.80 |
| 同约束一步质量教师 | 61.69 | 1.197 | 37.88 |
| 三块质量教师 | 61.64 | 1.197 | 37.85 |

[CPU基线原始结果](../runs/semantic_kev_v4_cpu_baselines_v1/summary.json)、[同质量信息的一步／三块对照](../runs/semantic_kev_v4_planning_ablation_v1/summary.json)。

目前有收益证据的是候选约束。**三块目标已实现，但这组回放尚未显示优于同约束的一步质量教师。** 两个教师都能访问离线 PSNR，不能把教师结果当作新 Kev 的效果，也不能当作无需质量信息的线上基线。保留所有结果，不根据这批验证结果暗改算法参数。v4 模型是否改善仍需下一次训练和闭环评估。

## 手动试训与评估

先进入 tmux：

```bash
tmux new -s semantic-v4-400
```

在 tmux 内执行下面的400步试训及闭环评估。此命令会使用 GPU，当前没有执行；输出目录必须是新目录。400步约有3200条样本曝光，仍不表示已收敛。

```bash
cd /home/amora/wxh
set -o pipefail
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/train_semantic_kev.py \
  --dataset datasets/semantic_kev_dev_v4 \
  --init-source abr_baseline \
  --execute --max-steps 400 \
  --out runs/semantic_kev_v4_abr_baseline_trial400_v1 \
  --preflight-out runs/semantic_kev_v4_abr_baseline_trial400_v1_preflight/preflight.json \
  2>&1 | tee jev/semantic_streaming/runs/semantic_kev_v4_abr_baseline_trial400_v1.log &&
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/evaluate_semantic_control.py \
  --dataset datasets/semantic_kev_dev_v4 \
  --checkpoint runs/semantic_kev_v4_abr_baseline_trial400_v1 \
  --out runs/semantic_kev_v4_abr_baseline_trial400_v1_closed_loop \
  2>&1 | tee jev/semantic_streaming/runs/semantic_kev_v4_abr_baseline_trial400_v1_closed_loop.log
```

按 `Ctrl+B`，松开，再按 `D` 离开；返回用 `tmux attach -t semantic-v4-400`。评估默认同时比较原基线、受约束基线、两种离线质量教师和受约束模型，各自演化状态。

只做 CPU 检查时不加 `--execute`：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py \
  --dataset datasets/semantic_kev_dev_v4 \
  --preflight-out runs/YOUR_NEW_PREFLIGHT/preflight.json
```

重建数据、审计或评估时使用新版本／新输出目录。配置在 `configs/semantic_kev_v4.json`；数据脚本 `prepare_semantic_control.py`，物理引擎仍是冻结的 `semantic_rollout.py`，策略逻辑在 `semantic_control.py`。旧 `evaluate_semantic_kev.py` 会拒绝把 v4 当作原先一步标签数据来解释。

保留原有限制：同十个开发视频的假想播放列表、四条验证网络的五个起点、PSNR质量代理、部分 ROI 定位问题和未计入模型推理延迟。没有新增 EVQA、真实网络播放或未见视频泛化结果。
