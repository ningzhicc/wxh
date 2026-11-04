# 第三步 v3：连续状态、低码率补充与闭环回放

后续状态：v3 的 ABR 初始化100步训练及闭环评估已完成，见 [分析报告](../runs/semantic_kev_v3_abr_baseline_trial100_v1_analysis/REVIEW.md)。当前默认入口已升级为 [v4 候选约束与三块规划](semantic_kev_v4.md)，下文保留首次 v3 数据准备时的记录及历史操作方法。

2026-11-03。本次完成 CPU 编码、训练数据准备和评估接线，未启动 GPU 训练或模型推理，未调用 API。

v1、v2 数据、两次100步试训结果和原始 ABR `epoch_08` 均保留。当前训练默认读取 `datasets/semantic_kev_dev_v3`，继续以 `merina656_baseline_seed666/checkpoints/epoch_08` 初始化，新模型写入独立 run 目录。

## 具体修改

1. 原120个编码候选保持不变，新增 CRF 46／50 × ROI qoffset 0／-1/5 的40个低码率候选，完整解码5508帧。总共160个候选、每视频16动作；所有大小和码率均来自实际完整 MP4。新动作从 `a120` 起，原 `a00`–`a119` 映射保留；来源 run 和来源 action ID 单独记录。
2. 用同一网络轨迹连续回放20个开发片段。每次下载后更新缓冲、已完成传输的带宽估计、上一档码率和小数网络游标。RTT、传输耗时和达到30秒缓冲上限的等待都会推进网络时间。线上输入不包含未来轨迹、参考答案或离线质量。
3. 起播统一先传最小视频块，建立第一条带宽观测。起播与后续卡顿分别记录；起播日志保留，但不纳入模型训练／静态验证记录，避免让相同冷启动输入学习依赖未知未来的不同标签。
4. 质量代理从封顶的 PSNR/40 改为不封顶的证据 RGB PSNR/10。卡顿系数4.3和切档系数0.05保留；起播等待系数4.3单列。它们是可配置的工程效用，不是问答准确率，也未经正式性能调参。
5. 新增闭环评估：每个策略从相同网络位置、零缓冲和同一播放列表开始，各自选择动作、推进自己的缓冲和网络位置。比较实际回放总收益、卡顿、质量和字节数。旧的固定状态标签匹配评估继续保留，但不能代替闭环回放。

网络使用原先16条训练／4条验证轨迹，各5个起点。行为策略为最小文件、按历史带宽选档、带少量安全候选探索的带宽策略。得到4560条训练、1140条验证记录。原始完整轨迹保存在 `train_rollouts.jsonl`／`valid_rollouts.jsonl`；训练文件不含起播步骤。不同策略共用起点，前几步会有相同状态，审计明确记录重复数，不能当作独立视频样本数量。

## 检查与当前结果

- [v3 数据清单](../datasets/semantic_kev_dev_v3/manifest.json)、[动作表](../datasets/semantic_kev_dev_v3/actions.json)、[编码来源锁定](../datasets/semantic_kev_dev_v3/encoding_sources.lock.json)：配置、源码快照、输入和文件哈希。
- [数据连续性、覆盖与标签检查](../runs/semantic_kev_v3_data_audit_v2/REVIEW.md)：4560个训练状态转换和1140个验证状态转换；同时校验反事实奖励和最优动作标签。
- [实际 Kev 输入预检查](../runs/semantic_kev_v3_preflight_v1/preflight.json)：全部5700条记录通过，最大 state 227 tokens，最大 packed 1038；保留全部16候选，没有截断。
- [CPU 闭环基线结果](../runs/semantic_kev_v3_baselines_v1/summary.json)：20个验证场景，每场景20片段。

在**相同的 v3 已收集状态**上，仅使用原候选时，训练／验证没有及时传完候选的比例为21.43%／9.74%；加入低码率候选后为10.70%／1.58%。最低档仍无法持续跟上网络的比例为16.64%／9.56%，低码率覆盖没有彻底解决。

训练最优标签的全局 CRF 分布为24：51.12%、32：16.16%、40：10.61%、46：7.13%、50：14.98%。这是自然回放结果，没有手动改标签或强行均衡；ROI 标签仍占90.29%，不能视为动作分布已完全均衡。

| v3 闭环策略 | 平均每场景奖励 | 平均每场景卡顿（秒） | 平均证据 PSNR（dB） |
|---|---:|---:|---:|
| 总选最小视频块 | 36.24 | 0.455 | 23.35 |
| 按历史带宽选档 | 50.84 | 3.354 | 37.01 |

两种策略起播相同，平均起播等待1.914秒；卡顿列不含起播。带宽策略以更多传输和部分卡顿换取更高证据质量，按当前效用获得更高总收益。**尚无 v3 Kev 模型结果**。新旧数据的状态、候选和效用都不同，不能直接拿 v2 的奖励数值与本表比较。

首次 v3 的16候选文字描述略超1152行长度限制，保存在 `datasets/semantic_kev_dev_v3_rejected_context_v0` 和对应 rejected 日志中；最终版缩短字段表述后重新生成并检查。没有丢弃候选或悄悄放大上下文限制。

## 手动试训：先进入 tmux

以下指令会启动 GPU 训练，当前并未执行。所有命令从 `/home/amora/wxh` 运行。输出目录必须是尚不存在的新目录。

先执行：

```bash
tmux new -s semantic-v3-100
```

在 tmux 内执行：

```bash
cd /home/amora/wxh
set -o pipefail
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/train_semantic_kev.py \
  --dataset datasets/semantic_kev_dev_v3 \
  --init-source abr_baseline \
  --execute --max-steps 100 \
  --out runs/semantic_kev_v3_abr_baseline_trial100_v1 \
  --preflight-out runs/semantic_kev_v3_abr_baseline_trial100_v1_preflight/preflight.json \
  2>&1 | tee jev/semantic_streaming/runs/semantic_kev_v3_abr_baseline_trial100_v1.log &&
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/evaluate_semantic_rollout.py \
  --dataset datasets/semantic_kev_dev_v3 \
  --checkpoint runs/semantic_kev_v3_abr_baseline_trial100_v1 \
  --out runs/semantic_kev_v3_abr_baseline_trial100_v1_closed_loop \
  2>&1 | tee jev/semantic_streaming/runs/semantic_kev_v3_abr_baseline_trial100_v1_closed_loop.log
```

按 `Ctrl+B`，松开，再按 `D` 离开 tmux；返回用 `tmux attach -t semantic-v3-100`。评估会同时重放两种简单基线和新模型，使用相同场景与效用；每个策略演化自己的状态。

只做 CPU 输入预检查时，不加 `--execute`：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py \
  --dataset datasets/semantic_kev_dev_v3 --init-source abr_baseline \
  --preflight-out runs/YOUR_FRESH_PREFLIGHT/preflight.json
```

## 复现与限制

配置入口是 `configs/roi_encoding_dev10_lowrate_v1.json` 和 `configs/semantic_kev_v3.json`。重新编码、生成数据或评估时先复制配置并设置新的版本／run ID；脚本拒绝覆盖已有数据和评估输出。旧 `common_v1` 只读，原 ABR 模型及共享部署配置没有修改。

当前仍是**离线一步最优动作的监督模仿**，并非在线 RL 或长期最优控制。先用这套新环境检验数据改动是否有效，再考虑加入按决策损失加权的训练。新增的低码率候选可能丢失文字细节，PSNR 不能保证问答成功。

播放列表由同一组十个短片段构造，不是同一原视频连续切出的真实分块；训练、验证使用相同视频和分离的网络来源组。30秒缓冲适用于本地 VOD 开发假设，不代表低延时直播。部分原 ROI 仍有已知定位问题。回放不模拟模型推理延迟，未做真实网络播放或新视频泛化测试。
