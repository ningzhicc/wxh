# Llama-3.2-1B 的 ABR 适配

2026-10-16：已通过 ModelScope 下载完整 Llama-3.2-1B 到 `/home/amora/wxh/jev/models/Llama-3.2-1B`。全部 14 个文件（共 4,954,701,035 字节）的大小和 SHA-256 均与仓库清单一致；包括 Transformers 权重和原始格式权重。

完整实验 `llama_abr_v1` 已完成：未微调275条测试、累计7轮训练、验证平台早停、选中第4轮、最终275条测试与校验均完成。纯训练40.9分钟，最佳检查点为 `checkpoints/llama_abr_v1/epoch_04`；完整结果见 [实验报告](results/llama_abr_v1/REPORT.md)，使用和恢复方式见 [RUN_README.md](RUN_README.md)。下面的两轮准备命令为早先草案，正式实验以 `runs/llama_abr_v1/experiment.json` 为准。

来源与完整性记录：[model_download.json](model_download.json)。已生成本地训练配置：[prepared/llama_1b_v1.json](prepared/llama_1b_v1.json)。真实分词器对 7,224 条训练样本的检查全部通过，最长状态 240 tokens，最长完整输入 390 tokens，无截断或超长。

原 Kev/Qwen 代码、检查点、冻结数据和已完成报告保持原样。本目录复用官方 Kev 训练器和 pointer head 结构，换用 `meta-llama/Llama-3.2-1B` 原始基座（非 Instruct）；冻结基座，重新初始化 LoRA rank 16 和 pointer head，不加载 Qwen/Kev 的权重。

编码使用 Llama 词表中前五个现有 `reserved_special_token`，不扩展词表；检查点保存五个 token 的字符串与 ID。训练、推理使用同一份契约，加载时拒绝缺少或不匹配的契约。每个入口运行在独立进程中，不在同一进程混用 Qwen 和 Llama。

## 下载和准备

模型已按用户指定的 [ModelScope 仓库](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B) 下载，不需要再登录 Hugging Face。下载工具安装在独立环境 `/home/amora/wxh/jev/.venvs/modelscope`，没有向训练或推理环境安装 ModelScope。

本次使用 ModelScope 1.40.2；其新 CLI 使用仓库位置参数和 `--local-dir`，对应旧版的 `--model`、`--local_dir`。复现下载和校验：

```bash
/home/amora/wxh/jev/.venvs/modelscope/bin/python -m pip install --index-url https://pypi.org/simple modelscope==1.40.2
NO_PROXY='*' no_proxy='*' /home/amora/wxh/jev/.venvs/modelscope/bin/modelscope download LLM-Research/Llama-3.2-1B --local-dir /home/amora/wxh/jev/models/Llama-3.2-1B --cache-dir /home/amora/wxh/jev/.cache/modelscope --max-workers 4
NO_PROXY='*' no_proxy='*' /home/amora/wxh/jev/.venvs/modelscope/bin/python /home/amora/wxh/jev/abr_llama/verify_modelscope.py
```

Transformers 使用根目录下的 `model.safetensors`，原始格式保存在 `original/consolidated.00.pth`。检查点中的 `base` 指向上述本地模型目录，后续训练和加载不需要重新联网下载。

如准备另一个运行名：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_llama/prepare.py --base /home/amora/wxh/jev/models/Llama-3.2-1B --run-name NEW_RUN
```

准备脚本检查基座类型、权重存在、原训练数据 SHA-256、所有 7,224 条状态在 Llama tokenizer 下均不截断，并生成 `prepared/llama_1b_v1.json` 和完整训练命令；不会自动启动训练。模型分词不同，如果 512 token 状态预算不够，脚本拒绝继续，需要记录新的共同上下文预算后再比较，不能悄悄删除训练样本。

初始方案为两轮、batch 2、accum 8、LoRA lr 2e-5、独立 head lr 1e-4、BF16 autocast、FP32 冻结权重与主参数、seed 666。决策头从零初始化，因此初始学习率与原 Kev 热启动不同；这个学习率仅为待验证的起点。不能从旧 Kev 的“两轮已经够用”推断 Llama 也只需两轮。

训练命令保存每轮模型到 `checkpoints/<run>/epoch_NN`，最终模型到 `runs/<run>`。先确认小批量真实模型试运行通过，再执行完整训练。不要在 GPU 被其他训练占用时启动。这里的薄封装不实现逐步优化器恢复；训练中断时保留已完成轮次，不能声称精确恢复优化器状态。

## 推理入口

```bash
KEV_DTYPE=bf16 /home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_llama/serve.py --run /path/to/llama/checkpoint --port 8010
```

请求协议与 Kev 相同，默认地址 `127.0.0.1:8010`。Llama 检查点必须使用本目录入口，以恢复分隔符约定；不能直接交给原来的 `kev_service.py`。原评测脚本固定使用 8009 和原服务管理器，不能直接套用：完整自动选模、服务管理和四组测试驱动尚待真实权重可用后接通验证。

## 验证与比较约束

- 沿用原 55 条轨迹、7,224 个 MPC 标签状态，不使用已有四组测试集作训练或超参数选择。
- 原验证集 30 窗口只有两个采集来源，仍存在选择不稳定的问题；更换基座不会自动解决它。
- 比较时固定视频、状态文本、动作、QoE 和起点；另报延迟及显存，不假定 1B 一定更快或 QoE 更高。
- Kev 已有通用决策预训练，Llama 原始基座没有同样的训练历史。因此这是两条完整训练路线的比较，不是控制其他因素后的基座大小消融。

无需真实权重的离线集成检查：

```bash
/home/amora/wxh/jev/kev/.venv/bin/python /home/amora/wxh/jev/abr_llama/test_runtime.py
```

该检查随机生成极小 Llama，在 CPU 上执行官方训练器、两轮更新、逐轮保存和重新加载，并检验编码契约及概率有效性。它不代表真实 1B 模型已经训练或 ABR 效果得到验证。

2026-10-16 离线验证：两个集成检查通过；同时确认无本地模型时准备脚本明确退出，不启动训练、不生成伪造的准备成功记录。
