# Merina656：两个独立模块与手动训练

数据和代码已准备好。**本次没有启动真实模型训练；训练由你手动执行。** 本目录独立于旧训练和冻结评测目录，旧权重、原NetLLM仓库与common_v1保持原样。

## 固定的数据与初始化

- 输入：`/home/amora/NetLLM/adaptive_bitrate_streaming/artifacts/exp_pools/merina_merged_logits.pkl`。
- SHA-256：`e5d1caef67d4f338d8e49b0b9e07b3326d12b4603a4b04cb171e9846fe302ca1`。
- 656局、每局47条存储决策，共30,832条；原始动作标签全部保留，未增加轨迹，未使用池中teacher_logits。
- `data/merina656_v1/base.jsonl`与`cost.jsonl`有相同状态、标签、顺序、ID与监督数据，只改变候选描述和其解释。
- 初始化统一使用已缓存的发布版Kev-4B v1.0 LoRA与指针头，而不是旧ABR微调模型；基座为固定revision的Qwen3.5-4B-Base。

该池的6×6状态保存了最近历史；吞吐从MB/s转换为Mbps，下载时间与缓冲恢复成秒。片段大小根据原状态匹配已知video1资产并恢复精确字节数。实际存储窗口对应视频大小表第2至48行；首条观测已有两次750kbps历史。转换保留这些事实，不强行改成首片300的另一次仿真经历。

原池未保存逐局带宽文件和教师身份，**不能据此证明训练／验证／测试来源独立，也仍未认证为论文600条原始池**。后续性能结论应注明本地656池设置；严格同池比较还需要让DyLas在相同基础经验和选模协议下训练／重评。

## 模块一：候选代价表示

开关：`--candidate-cost 0|1`。实现复用 `../abr_candidate/features.py`，训练与在线评测使用同一个转换函数。

每档候选包含下一片实际大小、依据最近最多五次吞吐调和均值计算的预计下载时间、当前缓冲减去预计下载时间、与上一档的切换代价。只读取当前在线观测；预计时间和缓冲余量不是未来实测值。这个模块不改变原动作标签或损失函数。

## 模块二：收益差监督

开关：`--regret-loss 0|1`。标签构造在 `regret.py`，损失在 `losses.py`，不依赖模块一是否开启。

相邻存储状态能恢复已执行动作的实测下载时间。第一版采用明确的**一步后果近似**：

```text
候选下载时间 = RTT + (已执行动作实测下载时间 - RTT) × 候选片段大小 / 已执行动作片段大小
候选QoE = 候选码率Mbps - 4.3 × max(候选下载时间 - 当前缓冲, 0) - 切换代价Mbps
候选regret = 六档中最高的候选QoE - 该档候选QoE
训练损失 = 原动作交叉熵 + λ × Σ[模型选档概率 × min(regret, 20)] / 4.3
```

默认λ=0.2，RTT假设为0.08秒。20的裁剪只用于额外训练项，**评测仍用原始QoE奖励**。收益差按码率键与当前候选顺序对齐，避免顺序变化后监督错档。

这假设各候选传输期间的平均有效链路速率相同，忽略动作改变下载完成时刻后的带宽差异和长期缓冲收益。**不是六档真实分支回放，也不是长期最优Q值。**原池缺少网络身份和完整模拟状态，当前不能准确执行那种真实回放。后续拿到可重放记录后，可替换本模块的标签生成器，保持其它模块和消融开关不变。

30,176条非末步样本具有这种监督；656条末步样本缺少下一观测，额外项被屏蔽，原动作交叉熵照常训练。已执行档位的重算QoE与原记录的最大误差约0.000209。未来观测、奖励与收益差只存放在模型输入外的`_supervision`／元信息中，通过实际Kev输入路径确认不会被编码到状态或候选文本。

## 四组消融

| 名称 | 模块一 | 模块二 | 输入文件 | 在线候选模式 |
| --- | ---: | ---: | --- | --- |
| baseline | 0 | 0 | base.jsonl | base |
| cost | 1 | 0 | cost.jsonl | cost |
| regret | 0 | 1 | base.jsonl | base |
| both | 1 | 1 | cost.jsonl | cost |

四组默认统一：8个完整epoch、学习率2e-5、batch2×累积8=有效batch16、LoRA16、head_dim256、fp32基础权重＋bf16前向、梯度检查点、seed666。每轮1,927次优化更新，共15,416次更新、246,656条样本呈现。模型与LoRA结构相同，模块二不增加在线预测分支。

8轮是这轮完整数据消融的起始预算，不能据此宣布已收敛；训练后应看相同FCC验证环境的QoE曲线。需要更长计划时，四组从一开始统一设置例如`--epochs 16`。已经开始的运行不能改epochs后直接resume，因为学习率计划和比较预算都会改变。

## 直接运行训练

从工作区运行，**同一时间只运行一组**。训练器会锁住本目录的其它训练任务，不停止任何已有模型服务或其它进程。

```bash
cd /home/amora/wxh
bash jev/abr_modular/train.sh baseline
```

其余三组分别为：

```bash
bash jev/abr_modular/train.sh cost
bash jev/abr_modular/train.sh regret
bash jev/abr_modular/train.sh both
```

长任务建议先打开`tmux new -s abr-train`，在该会话里执行训练命令。按Ctrl+B再按D离开会话；用`tmux attach -t abr-train`重新进入。训练进度还会写入各run目录的`steps.jsonl`、`progress.json`和`status.json`。

正式长跑前，可由你手动运行20个优化步骤检查显存与速度；它使用完整8轮学习率计划，随后继续同一实验：

```bash
bash jev/abr_modular/train.sh both --stop-after-steps 20
bash jev/abr_modular/train.sh both --resume
```

这里第二条要在第一条结束后运行。本次仅做配置dry-run与CPU检查，没有替你执行这个GPU试跑，也没有验证4B实际训练的显存峰值或吞吐。

## 中断恢复与输出

例如both组中断后：

```bash
bash jev/abr_modular/train.sh both --resume
```

若原命令使用自定义`--epochs`、`--seed`、`--batch`、`--accum`或`--out`，恢复时保留相同参数。数据、模块开关、初始化、环境版本、训练代码和计划不一致时拒绝恢复。重跑相同数据预检查的耗时变化不改变训练声明。

- 每200次优化更新和每轮结束保存优化器、调度器、模型、当前轮次和位置、CPU／CUDA随机数状态。保留最近两个完整恢复点，较旧的本次恢复点会清理；每轮模型快照全部保留。
- Ctrl+C／SIGTERM请求在当前优化组结束后保存退出；进程被强制杀死时从最近完整点继续，可能重做最多199次更新。若在首次模型加载完成前中断、尚无恢复点，`--resume`会从相同发布初始化重新开始。
- 恢复文件先写入暂存目录，完成并计算哈希后才发布。未完成的目录不会被当作可用恢复点；文件被修改会拒绝加载。
- 原始组输出：`runs/merina656_baseline_seed666/`；其余依次为`merina656_cost_seed666`、`merina656_regret_seed666`、`merina656_both_seed666`。
- 每轮可加载模型：`checkpoints/epoch_01/`至`epoch_08/`，含adapter、head和tokenizer。`last_checkpoint.json`指向最后一轮，**不代表验证最优**。
- 换一套参数重做实验时使用新的`--out`，例如`--out /home/amora/wxh/jev/abr_modular/runs/both_seed42 --seed 42`，对应对照组也使用相同种子。

如果batch2显存不足，可从新运行开始统一采用`--batch 1 --accum 16`。同样保留有效batch16；四组应统一设置并另取输出目录。训练器不会自动修改预算或静默跳过样本。

## 训练后的比较

先在同一FCC验证协议中选择checkpoint，再固定模型测试四个数据集；不按测试集挑轮次。在线评测复用`../abr_candidate/run_compare.py`，上表的候选模式必须与训练组一致；regret组在线不需要收益差标签。

本地common_v1物理环境和评分入口保持冻结。若对照论文截图，另外报告去首片保留末片的均值与ddof=1的跨轨迹标准差；common_v1主指标去首尾且报告ddof=0，不能混作同一列。

## 四个测试集的一键评测

`evaluate_test.sh`固定评测一个checkpoint，默认使用本次baseline的`epoch_08`。四个测试集为FCC100、Norway3G41、Oboe100、HSR34，每条轨迹使用common_v1的全部3个固定起点，共825局，采用argmax选档。

```bash
cd /home/amora/wxh
bash jev/abr_modular/evaluate_test.sh --dry-run
bash jev/abr_modular/evaluate_test.sh
```

也可以指定已经确定的checkpoint和一个尚不存在的输出目录：

```bash
bash jev/abr_modular/evaluate_test.sh \
  --checkpoint /absolute/path/to/checkpoint \
  --out /absolute/path/to/new_test_results
```

候选输入模式从该run的`experiment.json`自动读取；外部checkpoint默认使用`base`，需要时显式传入`--candidate-mode cost`。脚本先验证数据与模型文件，再切换本项目8009端口的Kev服务，确认服务实际加载指定checkpoint后开始评测，结束时停止服务以释放显存。它与本目录训练器共用任务锁。

默认结果保存在`results/test_<run>_<checkpoint>_<时间>/`：`report.md`和`summary.csv`汇总QoE、码率、卡顿、切换幅度和决策耗时；`run/`保存原始逐局结果与实时`progress.json`；`evaluation.log`保存评测日志。主指标去首尾，辅助QoE只去首片。已有输出目录拒绝覆盖；失败后保留日志和已完成会话，下次用新的输出目录重新运行。

这个入口不根据测试成绩选择轮次；第8轮只是固定的默认模型，不代表验证最优。Merina656池与测试轨迹的来源独立性仍未认证，报告沿用本地common_v1诊断评测的边界。

## 检查与复现

已对两种输入全部30,832条样本用实际Kev tokenizer和编码器检查：无跳过、无截断；最大状态323tokens，base最大497tokens，cost最大811tokens，均在既定限制内。记录见`data/merina656_v1/preflight.json`。

```bash
python jev/abr_modular/prepare.py
jev/kev/.venv/bin/python jev/abr_modular/preflight.py
jev/kev/.venv/bin/python -m pytest jev/abr_modular/test_modular.py -q
bash jev/abr_modular/train.sh both --dry-run
```

`prepare.py`发现已有数据时只验证，拒绝悄悄覆盖。标签构造或候选属性变化应使用新的数据版本。数据、版本、初始化哈希和训练声明会写入`experiment.json`与模型head元信息。

`verification/prepare_attempt_v0`保留一次被预检查拒绝的初始清单（整数键JSON往返顺序问题），不属于当前训练数据；当前数据已单独重新生成并通过完整检查。
