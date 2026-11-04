# 本地 Kev 语义视频码率决策

这一版直接服务三步系统的第三步：输入当前视频问题、历史带宽、缓冲和可选编码配置，输出一个配置键。`actions.json` 将配置键映射到实际编码 MP4、实际字节和码率。

## 已完成

- 输入来自真实 QP 区域编码结果，包含普通编码和 ROI 编码；三个视频分别有 6、2、6 个动作，不再写死传统 ABR 六档码率。
- 1440 条训练、360 条验证样本。使用现有 common_v1 快照中 16 条训练与 4 条验证带宽轨迹；网络来源分组不交叉，原 ABR 数据和训练器保持只读。
- 每条样本从相同轨迹位置分别回放各候选的真实 MP4 字节数，带宽单位 Mbps，payload fraction=0.95，RTT=0.08 秒。
- 标签奖励：`clip(证据 RGB PSNR / 40,0,1) - 4.3*卡顿秒数 - 0.05*码率变化Mbps`。选最高奖励动作作 Kev CE 微调标签；这是一步回放模仿训练，不是在线强化学习或长期最优策略。
- 各候选真实质量、未来下载时间和奖励只放 `_supervision`，不放实际模型输入；带宽路径与身份仅作溯源元数据。
- 已用发布版 Kev-4B 对应的真实 tokenizer 和编码接口检查所有样本，不加载权重、不占 GPU。

## 当前状态

`kev_preflight.json` 保存输入检查结果与实际训练参数。`training_admission.json` 保存最近一次试跑资源检查；GPU 忙导致未启动。没有训练出的 checkpoint，不能将数据准备或预检查记为训练完成。

## 运行

从 `/home/amora/wxh` 执行。只做 CPU 输入检查：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py
```

GPU 空闲后先独立跑 4 步，验证训练路径：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py \
  --execute --max-steps 4 --out runs/semantic_kev_smoke_v1
```

正式开发训练使用新的输出目录，默认 2 epoch、batch1×accum8、学习率2e-5，复用本机缓存的发布版 Kev-4B LoRA 与指针头初始化：

```bash
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py \
  --execute --out runs/semantic_kev_dev_train_v1
```

训练入口不会停止其它进程；显存不足或 GPU 忙时在加载权重前退出。新目录拒绝覆盖，当前直接使用原生 `kev.train`；不支持从这一 LoRA 运行的中间优化步骤恢复，不将 smoke checkpoint 当作正式训练结果。

## 数据边界与后续接线

当前用三个相同开发视频验证不同网络状态下的选档，不能声称视频泛化；缓冲来自明确的场景枚举，不冒充已记录的在线策略经历。质量代理是像素指标，不冒充问答成功概率。后续效果验证不阻塞先完成这一最小本地决策模型。

训练后推理使用同样的状态和候选构造函数 `prepare_semantic_kev.request`，通过 Kev 的 `choice` 输出选项键，再查 `actions.json[case_id]` 得到实际 MP4。验证集是准备好的后续检查输入；训练器当前只读取 `train.jsonl`，没有自动根据验证集选模或测试集选模。

来源见 `manifest.json`、`config.json`、`traces.lock.json` 和 `code/`；旧 API、编码与问答运行均保持原记录。
