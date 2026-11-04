# DyLas 与 Jev 公共实验配置

本目录新增一套独立实验入口。旧 `abr_eval`、`abr_finetune`、模型和 `/home/amora/NetLLM` 保持原样。当前目标是固定数据与评测条件；没有进行新训练。

## 已固定的条件

| 项目 | common_v1 |
| --- | --- |
| 训练 | 16条可识别为FCC来源、与验证／测试无已识别同源交集的记录 |
| 验证 | 30个FCC窗口，来自**2个**原始采集来源 |
| 测试 | FCC100、Norway3G41、Oboe100、HSR34，共275条 |
| 视频 | video1，六档大小文件；实际下载前48片 |
| 分片／缓冲 | 4秒／60秒 |
| 码率 | 300/750/1200/1850/2850/4300 kbps |
| 首片 | 固定300 kbps |
| 网络 | RTT80ms，payload0.95，关闭随机下载噪声，轨迹按原模拟器循环 |
| 主QoE | 每局 `mean(reward[1:-1])`，排除首尾，保留46片 |
| 辅助QoE | `mean(reward[1:])`，便于对照旧Jev结果；另存全部48片均值 |
| 奖励 | 码率Mbps − 4.3×卡顿秒 − 相邻码率变化Mbps |
| 测试／验证起点 | 每条3个固定起点，基础seed666，保存实际指针 |
| 训练起点 | 每条1个固定起点 |
| 汇总 | 先片内均值，再同trace各起点平均，再等权平均trace |
| 决策耗时 | 独立记录，不推进主播放／网络时钟 |

这是**本地公平比较协议**，不宣称复现正文49片、100次会话的论文表格。多个起点不是多个独立训练种子。

FCC-only 指按可识别采集ID筛选，原始 `fcc-train` 中的39个Norway日志不进入此训练集。来源身份仍依赖文件名规则；不宣称已经证明所有未知来源的独立性。

验证集只有2个原始来源，这是当前公开划分清理后的限制。30个窗口和重复起点不能替代新增独立验证来源。后续扩充数据应创建新版本，不悄悄修改common_v1。

## 文件

- `protocol.json`：可读协议模板。
- `bundles/common_v1/manifest.json`：冻结的完整清单、来源、哈希和起点。
- `bundles/common_v1/train.json`、`valid.json`、`test.json`：给后续训练和两类策略使用的同一划分。
- `bundles/common_v1/traces/`、`video1_sizes/`：数值快照，不依赖外部仓库继续存在。
- `bundles/common_v1/split_audit.json`：每条排除原因及来源数。
- `simulator/`：从DyLas本地环境复制的物理模拟器，仅把constants导入改为相对导入；来源记录在`source.json`。
- `compare.py`：两类模型和传统策略共用的仿真、评分、日志与报告入口。
- `policies.py`、`dylas_worker.py`：策略接口。DyLas在独立Python进程内加载，依赖不会污染公共模拟器。
- `verification/`：测试记录、接入检查和说明。

训练清单保留了来源路径供审计；**实际输入应读取 `bundle / record["snapshot"]`**。不要把旧经验池直接标成使用了新划分，也不要再从原始235条目录无筛选生成正式训练集。

## 使用

以下命令从 `/home/amora/wxh` 执行。普通数据、基线和检查只需已有的NumPy环境。

```bash
python jev/abr_compare/compare.py verify
python -m pytest jev/abr_compare/test_compare.py -q
```

首次建立新版本时运行`prepare`。已经存在的bundle只验证，不覆盖。协议、代码或数据改动后，应使用新的bundle路径；不允许沿用旧指纹混合结果。

```bash
python jev/abr_compare/compare.py prepare
```

先跑少量**验证集**会话检查接入；不以测试集挑参数：

```bash
python jev/abr_compare/compare.py run \
  --policy bba --split valid --rounds 0 --limit 2 \
  --out jev/abr_compare/results/bba_valid_pilot
```

完整验证不加`--limit`和`--rounds`，将使用该划分全部3个起点。完整测试需显式写`--split test`，四组共825局。任何输出目录已经存在时均拒绝覆盖。

### Jev / Kev

使用已启动的本地服务；评测器不会自动重启、切换或停止现有服务。`/v1/models`中的模型信息会记录到结果，能找到本地checkpoint时还会保存权重哈希。

```bash
python jev/abr_compare/compare.py run \
  --policy jev --base-url http://127.0.0.1:8009 \
  --split valid --rounds 0 --limit 2 \
  --out jev/abr_compare/results/kev_valid_pilot
```

默认保留原Jev的argmax，`--action-mode sample`可另做采样变体。接口必须返回六档合法概率；错误／截断会中止，不静默回退为传统策略。

### DyLas

必须显式指定checkpoint，避免自动挑选测试成绩最好的权重。当前桥接器复用原仓库的legacy-v4加载器，仅支持`sfd512/sahd2048/ld16`这一已核对结构；其他v4_rebuttal等变体需另写明确加载配置，不能部分加载后假装成功。

```bash
python jev/abr_compare/compare.py run \
  --policy dylas --split valid --rounds 0 --limit 2 \
  --checkpoint /home/amora/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/artifacts_exp_pools_ss_15/abrllm_v4_rank_128_w_20_gamma_1.0_sfd_512_sattn_True_sahd_2048_fusion_weighted_sum_loss_ce_bwvae_ld16_beta_0.4_klw_0.01_bwf_residual_lr_5e-05_wd_0.0001_warm_300_epochs_70_seed_42/best_model \
  --out jev/abr_compare/results/dylas_valid_pilot
```

默认用 `/home/amora/anaconda3/envs/netllm/bin/python`，可通过`--dylas-python`指定。使用本地模型并禁止HuggingFace联网下载。桥接器从指定旧经验池读取旧模型所需的奖励归一化／目标回报信息，**不是拿该经验池生成新训练数据**。

默认保留原DyLas的动作采样；`--action-mode argmax`作为单独变体。每次会话重置历史和随机种子；模拟器及主QoE计算都由公共进程完成。

### 同条件报告

```bash
python jev/abr_compare/compare.py report \
  --runs jev/abr_compare/results/kev_valid_pilot jev/abr_compare/results/dylas_valid_pilot \
  --out jev/abr_compare/results/valid_pilot_comparison.md
```

数据指纹、轨迹选择、起点轮次不同会拒绝比较。该报告是描述性均值／跨trace标准差，不自动证明统计显著性。每会话原始记录另行保存，供后续配对统计。

## 旧模型与正式公平训练的边界

现有checkpoint默认标为`diagnostic_existing_checkpoint`。同环境重评不代表旧模型也使用了共同训练集：旧Jev训练含Norway，部分DyLas本地复跑报告注明曾用Norway3G-test选模。它们可用于接入检查和定位差距，不能自动成为新的独立OOD测试证据。

下一次正式训练应让数据生成器读取本bundle的`train.json`，选模只读`valid.json`，固定后再跑`test.json`。教师版本、状态采样和预算需另行冻结；本次没有擅自替换现有专家池、重新训练或重写旧报告。
