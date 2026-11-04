# 第一项改进：每档码率的代价表示

本目录是可独立选择的候选输入改进。没有修改 Kev 源码、旧训练数据、已有权重、DyLas 仓库或 `abr_compare` 的冻结文件。本次没有训练模型、调用真实模型评测或计算新的四测试集成绩。

## 已实现

- `features.py`：只利用当前接口已有的历史吞吐、缓冲、上一档码率、下一片六档大小，在候选文本中加入四项属性。
- `prepare.py`：同一份已标注 JSONL 生成 `base.jsonl` 和 `cost.jsonl`。两个版本的状态、动作标签、顺序和来源元信息一致；记录哈希，拒绝覆盖，不删样本。原始划分继续有效，验证样本不会因为转换而成为训练样本。
- `check_context.py`：使用本地真实 tokenizer 和 Kev 编码入口检查全部输入，不允许截断或跳过。对正式数据应重新做完整检查。
- `run_compare.py`：显式选择 `base` 或 `cost`，复用公共模拟器、起点、动作校验和评分；结果记录实际输入、改进版本与本目录代码哈希。

四项候选属性：

| 属性 | 定义 |
| --- | --- |
| size_bytes | 下一片该档的已知字节数 |
| est_download_s | 片段兆比特数 ÷ 最近最多5次实测有效吞吐的调和均值 |
| est_buffer_margin_s | 当前缓冲秒数 − 预计下载秒数；负值表示预计出现卡顿 |
| switch_cost_qoe | 本协议下与上一档的码率差，单位Mbps，切换惩罚系数为1 |

下载时间和缓冲余量是预测，不是真实未来结果。有效吞吐由数据量与完整实测下载时间计算，不再重复添加 RTT 或 payload 系数。第一版没有另外拟合预测网络，也没有增加新的样本或教师。它只是让已有信息更容易被候选评分器使用，效果尚待验证。

## 已完成的检查

`python -m pytest jev/abr_candidate/test_features.py -q`：9项通过。包括单位换算、只读取历史、不依赖教师动作、标签与划分保留、拒绝非法输入，以及模拟HTTP服务下两个输入版本各完成一局48片播放。动作相同时，两版本的下载、缓冲、卡顿、奖励和播放时钟完全一致。这是接口与物理一致性检查，不是模型能力实验。

`verification/valid_preview_v1` 取旧验证集前64个状态，仅用于接口检查，不能训练或证明已对齐DyLas。使用正式 Kev tokenizer、`max_state=512` 时：

| 版本 | 样本数 | 最大状态tokens | 最大完整输入tokens | 跳过／截断 |
| --- | ---: | ---: | ---: | ---: |
| base | 64 | 321 | 495 | 0／0 |
| cost | 64 | 321 | 805 | 0／0 |

完整输入长度限制为1152（单问题行）与2176（packed）；当前预览均通过。实测记录在 `verification/context_preview_v1.json`。不能据64个预览断定其他完整训练数据都能通过。

冻结公共配置验证通过，指纹仍为 `7a21822ffc3be5b6618e58175d17813b31534a03f149492bced57a7d7cb06eec`。

## 接下来按什么顺序做

1. 找到或确认论文对应的DyLas训练池及对应权重。当前尚未把424／640／656条池中的任何一份指定为论文600条池，详见 `../abr_alignment/README.md`。
2. 把确认的经验转换成原有Kev状态接口，再用本目录生成配对输入；检查每条标签、历史和样本数量，并对完整数据运行长度检查。这里没有实现从无来源信息的pickle中恢复隐藏模拟状态。
3. 从同一个Kev初始化分别训练 base 与 cost，保持同一基础经验、样本呈现预算和验证选模规则，先比较验证集闭环QoE及其码率、卡顿、切换分项。不要拿旧训练结果与新数据训练结果归因到此改进。
4. 第一项有效后，再做候选收益差监督。第二项需要能恢复同一训练状态的模拟器快照或可准确重放的来源和动作历史；本目录没有生成反事实价值标签。
5. 方案和权重在验证阶段固定后，与DyLas通过同一评测入口测试四组数据。对照过程中保存方法差别；统一本地环境不自动代表复现原论文表格。

## 使用入口

确认来源后，转换指定输入；输出目录必须是新目录：

```bash
python jev/abr_candidate/prepare.py --input /absolute/path/to/confirmed.jsonl --out /absolute/path/to/new_pair
```

长度检查使用已有Kev虚拟环境，不加载模型权重：

```bash
jev/kev/.venv/bin/python jev/abr_candidate/check_context.py \
  --pair-dir /absolute/path/to/new_pair \
  --tokenizer jev/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b \
  --max-state 512 --out /absolute/path/to/new_context_audit.json
```

评测时，使用已加载对应权重的Kev服务，明确选择候选输入版本。下面是少量验证示例，不会自动训练或切换服务：

```bash
python jev/abr_candidate/run_compare.py --candidate-mode cost \
  run --policy jev --split valid --rounds 0 --limit 1 \
  --base-url http://127.0.0.1:8009 --out /absolute/path/to/new_valid_run
```

原始组使用 `--candidate-mode base`；其它评测参数保持一致。已有权重在新输入上的测试属于诊断，正式实验需要输入版本与对应训练版本一致。候选描述长度增加的推理成本由公共入口继续记录。
