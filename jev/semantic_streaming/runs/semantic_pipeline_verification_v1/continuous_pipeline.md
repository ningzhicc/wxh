# 连续视频三步系统：统一入口

入口为 `scripts/run_semantic_pipeline.py`。输入连续视频、问题、网络轨迹和已训练的 v4 模型，一次运行完成：

```text
原视频窗口 → 无损参考与连续切块 → Qwen API/已验证缓存定位
→ 各块真实CRF×ROI编码目录 → v4候选约束与Kev选择
→ 网络耗时回放 → 复制所选MP4到接收端 → 完整解码与拼接核验 → 报告
```

这是离线 VOD 系统：候选和后续两块的大小预先可用。传输时间由带宽轨迹回放，所选编码文件真实复制、解码和拼接；没有运行实际限速网络或模拟真实网络包。模型推理耗时记录在决策中，但不进入回放时钟。

## 已完成的实测

1. [缓存CPU联调](../runs/semantic_pipeline_cached_cpu_v1/REPORT.md)：同一段5.005秒视频切成三个相邻块，150帧，6帧API缓存；第一个块没有重要区域，因此普通编码。48个目录槽位对应37个真实编码文件，无新API／GPU调用。
2. [真实API＋Kev实测](../runs/semantic_pipeline_live_kev_v1/REPORT.md)：从原视频110秒附近取连续15秒，450帧、3块；18张原生帧中6张命中历史缓存，新增12次Qwen API调用。生成48个真实编码文件，当前v4模型实际选择后两块，起播按规则选最小文件。
3. [最终版本复用定位的Kev运行](../runs/semantic_pipeline_live_cached_kev_v2/REPORT.md)：从上一项的已验证原始API响应重建18帧定位，零新增API；重新切块、编码、模型决策和接收核验，验证跨运行复用而不重复付费。

实测选择如下；这些是完整MP4实测码率，不是固定码率目标：

| 块 | 原参考窗口起点秒 | CRF | 实际ROI偏移 | 文件字节 | 实际码率kbps |
|---|---:|---:|---|---:|---:|
| 0 | 0 | 50 | 0（起播最小文件） | 53006 | 84.72 |
| 1 | 5.005 | 40 | -3/10 | 187565 | 299.80 |
| 2 | 10.010 | 32 | -3/10 | 342759 | 547.87 |

总计583330字节；该条网络轨迹回放起播等待约0.700秒，后续卡顿0秒。接收端450帧均完整解码，拼接结果与逐块所选文件的帧序列一致。此结果用于证明流程接通，不能视作新视频泛化、定位准确率或问答正确率。

## 一条命令运行已有示例

从工作区 `/home/amora/wxh` 执行。每个新运行使用新的 `--run-id`，下列名字是示例名。

```bash
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_semantic_pipeline.py \
  --run-id MY_NEW_PIPELINE_RUN
```

默认使用已固定的5秒视频窗口、6帧历史定位缓存、2秒左右连续块和已训练的 `semantic_kev_v4_abr_baseline_trial400_v1` 模型。源视频、模型三份权重文件、配置、提示词、FFmpeg、轨迹和本地脚本均进入指纹。

只用 CPU 联调时：

```bash
python -u jev/semantic_streaming/scripts/run_semantic_pipeline.py \
  --policy bandwidth --run-id MY_NEW_CPU_RUN
```

复用已完成的15秒定位并使用 Kev：

```bash
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_semantic_pipeline.py \
  --config configs/pipeline_live_dev01_v1.json \
  --roi-cache-run semantic_pipeline_live_kev_v1 --roi-mode cache \
  --run-id MY_NEW_15SEC_RUN
```

这一命令不会新增定位API调用，但会重新编码和运行GPU推理。仅重查同一个已完成运行时，使用原命令和原run ID；完整哈希校验通过后直接返回，不重复调用API、编码或推理。代码、问题、源文件等发生变化时必须使用新run ID。

## 运行自己的视频

下面的 `api` 模式会产生真实定位调用。15秒、每块5秒、每块最多6帧，最多18次请求；未知结果的失败不会自动重试。

```bash
jev/kev/.venv/bin/python -u jev/semantic_streaming/scripts/run_semantic_pipeline.py \
  --config configs/pipeline_live_dev01_v1.json \
  --video /absolute/path/to/video.mp4 \
  --question '这个人衣服上的品牌标识是什么？' \
  --start-s 0 --duration-s 15 --chunk-s 5 \
  --roi-mode api --max-api-calls 18 \
  --run-id MY_NEW_VIDEO_RUN
```

凭据继续由现有 `.env.local` 或进程环境读取，不作为参数传入，不写到源码、鉴权头日志、运行配置或指纹。模型返回归一化框后转成原采样图的绝对像素坐标；原始响应和转换结果都留存，缓存不冒充新调用。

`cache_or_api` 模式先查缓存再调用；`cache` 模式缺少匹配结果时会停止，绝不自动转成付费调用。跨运行缓存同时匹配像素哈希、问题、请求采样时间、尺寸、模型、参数和提示词，保留原始请求／响应来源。新运行可用 `--roi-cache-run PREVIOUS_PIPELINE_RUN` 指向过去的定位结果。原始归一化坐标重新解析，帧ID的重新绑定明确记录；用户时间过滤产生的空框不会被当作API预测缓存。

针对“第一帧”这样的时间条件，请显式给出相对于准备窗口的取证范围，例如 `--evidence-window 0 0.05`。范围外不调用API，不宣称API判断区域不存在。系统没有自动从问题解析全局时间条件，也不做跨帧目标跟踪。

## 分阶段与中断恢复

```bash
# 完成准备、定位和CPU编码，暂不加载GPU模型。
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/run_semantic_pipeline.py \
  --run-id MY_STAGED_RUN --stage encode

# 同一代码、配置和输入下，检查已完成阶段后继续选档与接收。
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/run_semantic_pipeline.py \
  --run-id MY_STAGED_RUN --stage deliver
```

GPU被占用时保留全部CPU结果，以 `waiting_gpu` 状态退出；不会停止其他进程。同一个run ID可在GPU空闲后继续。已完成阶段先核对文件哈希；API模糊失败和没有完成记录的部分媒体操作会保留现场并停止，不覆盖历史产物。

长运行建议先执行 `tmux new -s semantic-pipeline`，然后在该会话中运行命令。按 `Ctrl+B`，松开，再按 `D` 离开；用 `tmux attach -t semantic-pipeline` 返回。

## 目录与溯源

```text
configs/pipeline_cached_dev01_v1.json    固定的缓存示例与v4权重哈希
configs/pipeline_live_dev01_v1.json      连续15秒的API示例
scripts/run_semantic_pipeline.py        统一入口和阶段校验
scripts/pipeline_media.py               连续切块、真实编码和接收核验
scripts/pipeline_roi.py                 有限额的API与严格缓存
scripts/pipeline_delivery.py            v4决策、回放与接收报告
scripts/audit_semantic_pipeline.py      独立逐块重放和溯源核验
runs/<run-id>/                         配置、代码、命令、API响应、选择、报告与哈希
artifacts/<run-id>/                    参考片、采样PNG、候选与接收端MP4
```

主要运行文件是 `chunks.json`、`roi_inputs.json`、`roi_results.jsonl`、`actions.json`、`selected_chunks.csv`、`decisions.jsonl`、`summary.json` 和 `REPORT.md`。`stages/` 独立封存各阶段，`files.sha256.json` 封存整个完成运行；大媒体仍受 Git 忽略规则管理。

独立审计示例：

```bash
python jev/semantic_streaming/scripts/audit_semantic_pipeline.py \
  --run-id semantic_pipeline_live_cached_kev_v2 \
  --out runs/MY_NEW_PIPELINE_AUDIT
```

审计会检查原视频窗口与连续块的逐帧来源、API图像与响应来源、真实编码字节及码率、所选与接收文件一致性，并用同一轨迹从零缓冲重新计算下载和缓冲变化。它不调用API或GPU。

## 当前边界

- 支持至少两帧、偶数尺寸、固定解码帧率的视频。参考窗口按相同尺寸／原生CFR转换为8位yuv420p，再逐帧与同样归一化的源解码结果核验。不是原始压缩码流，也不声称保留10位色深；VFR会明确停止。
- 按帧划分块，不丢帧或补帧；最后只有一帧时合并到前一块。30000/1001 fps下，名义5秒块会是5.005秒，全部预算使用实测帧数对应时长。
- 每块ROI为采样框的静态并集、8像素外扩和16像素宏块对齐。没有ROI时实际普通编码，相同CRF的重复槽位使用文件别名，实际偏移记录为0。
- PSNR是离线质量代理，不进入Kev输入。无ROI时用全帧PSNR作回放质量代理；零MSE的无限PSNR只在有限效用中按60dB表示，测量字段仍为null，不把它误记为无ROI。
- 首块按最小文件起播，其余块由v4约束和模型选择；模型异常回退会单独计数。没有任何候选满足预算时仍可能卡顿。
- 本次使用已见过的公开视频来源验证系统接线，尚未进行独立视频效果测试、接收端问答评分或真实网络播放实验。旧模型、旧数据和训练接口未改写。
