# 开源 Jev 风格项目推荐目录

核对日期：2026-09-30。

本目录收录 8 个 Jev 风格的社区实现，包括开放权重模型、现有 LLM 的决策适配器和训练框架。它们不等同于 TypeSafe 官方 Jev，也不能仅凭兼容接口就认定能力相同。

排序面向你的用途：**让 Codex 在本地搭建项目，先尝试免训练的六选一 ABR 决策，再研究推理系统或任务适配**。依据是可运行程度、推理路径透明度、接入成本与研究价值，不是实测 ABR 性能排行榜。本次核对了仓库文档及关键推理代码，没有加载模型或复测作者的性能数字。

## 推荐顺序

| 排名 | 项目 | 推荐用途 | 使用前是否必须自己训练 | 简要理由 |
|---:|---|---|---|---|
| 1 | [Kev](https://github.com/jaredpalmer/kev) | 首个本地 Jev 风格服务 | 否，可加载已发布模型组件与基座 | 权重、训练、推理和兼容服务较完整，最适合先把项目跑起来 |
| 2 | [AnyJev](https://github.com/nokia-applied-research/AnyJev) | 免训练基线、基座能力与输出机制研究 | raw/L0 不需要；L1/L2 需要任务标签 | 清楚区分直接概率读取、去偏、校准与任务头，适合你的 ABR 对照实验 |
| 3 | [Open-Jev — Zefan-Cai](https://github.com/Zefan-Cai/Open-Jev) | 专用决策头、候选并行与缓存研究 | 否，已有 adapter、决策头和温度参数 | 有模型组件、数据及推理代码，适合深入检查候选评分如何实现 |
| 4 | [Laya](https://github.com/NandhaKishorM/laya) | 小模型低延迟决策、领域微调 | 可直接推理；新任务可能需要微调 | 编码器路线更轻量，适合与 LLM 路线比较；通用免训练迁移需谨慎验证 |
| 5 | [Jeff](https://github.com/Gestalt-Lab/jeff) | 本地类型化决策、标签概率读出 | 否，已有 LoRA adapter | 本地客户端和 HTTP 服务清楚，但主要验证集中于事实核查 |
| 6 | [open-jev — Dasein Labs](https://github.com/daseinlabs/open-jev) | 共享前缀 KV cache、候选批量评分 | 基础评分不需要；任务头需要 | 推理机制直接，适合学习缓存优化；默认仍是候选文本似然评分 |
| 7 | [Tev1](https://github.com/togethercomputer/tev1) | 常规 LLM 微调的决策基线 | 已发布权重；复现训练则需要 | 数据与训练流程有参考价值，但保留 LM head，本地环境及权重许可仍需核对 |
| 8 | [Jevlike](https://github.com/vinnylarouge/jevlike) | 从头研究小型候选评分网络 | 针对 ABR 应准备数据并训练 | 架构简洁、方便修改，适合研究原型，不适合作为现成通用 ABR 决策器 |

## 1. Kev：优先搭建本地服务

- **推理路径**：Qwen 基座 + LoRA + pointer head，对请求中的候选输出概率。当前 Qwen3.5/3.8 混合架构采用独立问题行，并复用 state 缓存；不能只用“一张注意力掩码”概括所有基座的实现。
- **推荐理由**：提供多种规模的发布模型、本地服务及 `/v1/systemone` 接口，较适合接到已经设计好的 ABR provider 层。可以先试作者建议的 4B，再依据设备选择更小或更大版本。
- **需要注意**：发布包包括 adapter 和决策头，仍需相应基座；现有校准参数不保证在 ABR 状态上仍校准良好。当前发布家族为 0.8B、4B、9B、27B，注意旧介绍可能使用不同规模名称。
- **入口**：[权重集合](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd) · [4B 模型](https://huggingface.co/jaredpalmer/kev-4b) · [`kev/model.py`](https://github.com/jaredpalmer/kev/blob/main/kev/model.py) · [`kev/serve.py`](https://github.com/jaredpalmer/kev/blob/main/kev/serve.py)。代码许可：Apache-2.0。

## 2. AnyJev：优先做免训练与输出机制对照

- **推理路径**：raw 从候选标签 token 的 logits 读取分布；L0 通过选项轮换及先验处理减轻偏差；L1 用标签校准温度；L2 用标签拟合隐藏状态上的闭式决策头，不更新基座权重。
- **推荐理由**：很适合回答“同一个基座，仅改变概率读出方式，ABR 表现会怎样”。可以先做 raw/L0，之后再用 ABR 训练数据研究 L2，与替换 action head 的方法对照。
- **需要注意**：完整六选一 L0 通常涉及六个轮换 prompt，不是一次前向就免费完成去偏。自适应轮换可以减少计算，但要记录实际轮换数。L2 虽然不用梯度训练基座，仍然需要当前任务的标签，不能称为零样本。
- **入口**：[`anyjev/readout.py`](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/readout.py) · [`backends/hf.py`](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/backends/hf.py) · [`backends/vllm.py`](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/backends/vllm.py)。HF 路径可直接读 logits；vLLM raw/L0/L1 路径使用受限的单 token 请求，L2 使用 pooling 隐藏状态。代码许可：Apache-2.0。

## 3. Open-Jev（Zefan-Cai）：优先研究专用决策头

- **推理路径**：Qwen + LoRA + scalar decision head。每个候选作为独立序列评分，再跨候选归一化；提供候选批处理和可选的请求内前缀缓存。
- **推荐理由**：已发布 2B、9B、27B v1.1 的模型组件，训练数据、校准与评测记录较完整。适合研究“共享上下文、候选独立计算、输出概率”的具体实现。
- **需要注意**：下载的是模型组件，需要匹配版本的上游权重及专用加载器；前缀缓存仍标为实验性，默认关闭。六候选批处理不能直接理解为只承担一个候选的计算成本。
- **入口**：[2B 模型](https://huggingface.co/ZefanCai/Open-Jev-2B) · [模型集合](https://huggingface.co/collections/ZefanCai/open-jev) · [`jev/model.py`](https://github.com/Zefan-Cai/Open-Jev/blob/main/jev/model.py) · [`jev/server.py`](https://github.com/Zefan-Cai/Open-Jev/blob/main/jev/server.py) · [前缀缓存说明](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/prefix-caching.md)。代码许可：MIT。

## 4. Laya：优先做小模型与低延迟路线

- **推理路径**：ModernBERT/mmBERT 编码器配合决策模块，直接返回 Choice/Score/Noul；支持问题批处理和不同 checkpoint 路由，不生成答案文本。
- **推荐理由**：英文版本约 421M、多语言版本约 322M，适合研究较小模型能否通过 ABR 领域适配获得实用的决策质量与延迟。
- **需要注意**：作者明确报告了基础 checkpoint 在新型 typed-decisions 任务上的迁移弱点，不能把任务微调后的结果当成通用免训练能力。默认输入预算随 checkpoint 不同，英文版为 512、多语言版通常为 1024 tokens；接入 ABR 时先核对并压缩上下文。
- **入口**：[英文模型](https://huggingface.co/convaiinnovations/laya) · [多语言模型](https://huggingface.co/convaiinnovations/laya-multilingual) · [`laya/agent.py`](https://github.com/NandhaKishorM/laya/blob/main/laya/agent.py) · [`laya/serve.py`](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)。代码许可：Apache-2.0。

## 5. Jeff：已训练的本地标签评分模型

- **推理路径**：Qwen3-4B-Instruct-2507 + LoRA，使用原语言模型的标签 token 概率。候选首 token 不同则采用首 token 读出，否则进行完整候选序列评分。
- **推荐理由**：开放 adapter，提供本地 Python 客户端和 `/v1/systemone` 服务，适合对比“专用决策头”与“保留 LM head”的差异。
- **需要注意**：主要评测是事实核查，不能由此推定擅长 ABR。不同候选标签会触发不同评分路径，比较延迟时应固定标签构造方式。
- **入口**：[Jeff-1 adapter](https://huggingface.co/GestaltLabs/Jeff-1) · [`jev_clf/readout.py`](https://github.com/Gestalt-Lab/jeff/blob/main/jev_clf/readout.py) · [`scripts/jev_clf_server.py`](https://github.com/Gestalt-Lab/jeff/blob/main/scripts/jev_clf_server.py)。代码与 adapter 许可：Apache-2.0。

## 6. open-jev（Dasein Labs）：推理缓存实现参考

- **推理路径**：Gemma 3 4B 先 prefill 上下文，扩展 KV cache 后批量计算候选后缀的 token log-probability，再归一化；另有冻结特征上的任务头训练路径。
- **推荐理由**：非常适合阅读“前缀只算一次，六个候选如何复用”的代码。当前已有 Apple Silicon MLX 和 Windows/Linux PyTorch 评分路径，不再仅限 Mac。
- **需要注意**：候选文本似然不是动作长期价值，归一化概率也不自动具备校准意义。特征提取、任务头训练等部分仍依赖 MLX；Gemma 权重需遵守其独立许可。
- **入口**：[`openjev/scorer.py`](https://github.com/daseinlabs/open-jev/blob/main/openjev/scorer.py) · [`openjev/torch_backend.py`](https://github.com/daseinlabs/open-jev/blob/main/openjev/torch_backend.py) · [`openjev/server.py`](https://github.com/daseinlabs/open-jev/blob/main/openjev/server.py)。代码许可：MIT。

## 7. Tev1：普通 LLM 微调路线的参照

- **推理路径**：对 Qwen3.5-4B 做监督微调，保留标准 LM head，约束模型输出选项字母；作者模型卡明确区分它与非自回归 Jev runtime。
- **推荐理由**：适合复用数据构建与监督训练流程，作为“把决策问题当成短文本输出”的对照。
- **需要注意**：已有公开权重页，但模型卡仍提示本地加载环境需要验证，微调权重的发布许可尚在完善；不能把仓库 MIT 直接等同于权重许可。仓库请求示例只取 top-5 logprobs，不能直接作为六档完整概率分布。
- **入口**：[模型卡与权重](https://huggingface.co/togethercomputer/Tev1-4B-experimental) · [`examples/decide.py`](https://github.com/togethercomputer/tev1/blob/main/examples/decide.py) · [训练说明](https://github.com/togethercomputer/tev1/blob/main/docs/TRAINING.md)。代码许可：MIT。

## 8. Jevlike：适合自己训练的最小研究原型

- **推理路径**：候选编码作为 query，关注上下文表示，再由共享评分函数输出候选分数。支持从零学习的 byte encoder，以及冻结预训练编码器、只训练评分模块的路径。
- **推荐理由**：结构简单、容易修改，适合把 ABR 状态编码、候选码率描述与小型决策头结合起来做方法实验。
- **需要注意**：不是现成的通用 Jev 权重；提供的合成任务或游戏 checkpoint 不代表已经学会 ABR。默认 byte encoder 的上下文截断很短，需要按实验调整。
- **入口**：[`jevlike/model.py`](https://github.com/vinnylarouge/jevlike/blob/main/jevlike/model.py) · [`jevlike/predict.py`](https://github.com/vinnylarouge/jevlike/blob/main/jevlike/predict.py)。代码许可：MIT。

## 针对你的 ABR 实验如何选

- **先完成本地闭环**：首选 Kev；模型下载后先核对六档概率、动作映射及实际推理耗时。
- **先研究“是否无需训练就能做 ABR”**：用 AnyJev raw/L0 做清楚的零标签对照，保留同一基座、相同状态与相同提示。
- **研究专用输出头和缓存**：重点看 Zefan-Cai/Open-Jev 与 Kev；补充阅读 Dasein Labs 的候选后缀评分代码。
- **已有 ABR 专家轨迹、准备任务适配**：考虑 Laya、AnyJev L2 或 Jevlike。此时应单独设监督训练实验组，避免和免训练组混报。

同一模型在文本分类上的公开成绩，不足以判断 ABR 的 session QoE。复用同一网络轨迹、状态、奖励和延迟统计口径，报告实际闭环结果；目录排名只表示优先尝试顺序。

## 补充导航

[awesome-jev](https://github.com/yibie/awesome-jev) 可用于继续发现项目，但其中也包含调用官方 API 的应用，并非每个条目都提供本地模型。两个名称接近的项目必须以 owner 区分：`Zefan-Cai/Open-Jev` 与 `daseinlabs/open-jev` 是不同实现。

下面记录本次核对的源码提交。项目及模型链接指向各作者的第一方资料；正式运行时还应固定模型 revision、依赖与配置。

| 项目 | 核对提交 | 提交日期（仓库时区） |
|---|---|---|
| `jaredpalmer/kev` | [`0fe8fc9`](https://github.com/jaredpalmer/kev/tree/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91) | 2026-09-29 |
| `nokia-applied-research/AnyJev` | [`10d5db9`](https://github.com/nokia-applied-research/AnyJev/tree/10d5db91dda38dbde74c6abc1c075ce6463723d1) | 2026-09-28 |
| `Zefan-Cai/Open-Jev` | [`3308a15`](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16) | 2026-09-23 |
| `NandhaKishorM/laya` | [`6d942c9`](https://github.com/NandhaKishorM/laya/tree/6d942c92081fbc139e736bbd9ac0023223c29b7f) | 2026-09-29 |
| `Gestalt-Lab/jeff` | [`14ee67e`](https://github.com/Gestalt-Lab/jeff/tree/14ee67e2814a19ebd0a67f508ac1b0f424ba1ebc) | 2026-09-19 |
| `daseinlabs/open-jev` | [`4627a04`](https://github.com/daseinlabs/open-jev/tree/4627a04a075bf714f4b60b69de19fa2de0a8a431) | 2026-09-29 |
| `togethercomputer/tev1` | [`1dde778`](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0) | 2026-09-23 |
| `vinnylarouge/jevlike` | [`94f5fd1`](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452) | 2026-09-16 |
