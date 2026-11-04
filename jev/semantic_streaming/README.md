# 问题相关区域视频传输实验

这里存放新的视频、问题、证据区域与编码配置实验，与现有传统 ABR 的 `common_v1` 协议分开。

当前入口是 **pilot_dev10_v1：10 个公开视频细节问答开发样本**。问题和上游参考答案选自与 Artic 配套的 DeViBench；视频来自其指定的 StreamingBench。数据用于定位、编码与问答链路开发，不能再把这些样本作为未接触过的正式测试集。

已完成真实 Qwen3-VL-Plus API 定位：10 个案例共 60 帧，见 [开发集定位汇总与失败记录](runs/qwen3vl_roi_dev10_summary_20261102_v1/REVIEW.md)。原始前三例报告保留在 [首次实验报告](runs/qwen3vl_roi_pilot_20261102_v2/REVIEW.md)。

也已完成衣服文字、计时器、车牌三个案例的本地真实编码与接收端问答初步对照：7 组预算匹配有效，2 组预算不满足而排除；均匀编码参考答案匹配 6/7、ROI 编码 7/7。详见 [编码与问答报告](runs/roi_encoding_smoke_20261102_v3/REVIEW.md)。这是静态保护区、两遍编码、单次同模型问答的开发实验，不是跟踪系统或正式测试集结果。

当前主线是三个交付：**问题相关 ROI → QP 区域编码与真实字节表 → 本地 Kev 选择编码档位**。

三步现已提供[连续视频统一入口与操作手册](docs/continuous_pipeline.md)：`scripts/run_semantic_pipeline.py` 一次完成连续切块、API／缓存定位、真实编码、v4选档、耗时回放以及接收文件完整解码／拼接核验。[15秒API＋Kev运行报告](runs/semantic_pipeline_live_kev_v1/REPORT.md)保留了12次新API调用、6帧历史缓存、48个编码文件和450帧接收核验；[最终版本复用定位的运行](runs/semantic_pipeline_live_cached_kev_v2/REPORT.md)验证了跨运行复用而不重复调用API。这是离线VOD加轨迹回放和本地真实文件交付，不是实际限速网络播放。

新增[五视频验收入口](docs/evqa_acceptance.md)：固定v4模型，在本地微调没有使用过相同源文件SHA的五条EVQA发布视频上，共用真实编码候选比较带宽策略／Kev，再进行原视频及接收视频的盲化多选问答。`run_evqa_acceptance.py --plan-only` 只检查计划，默认执行会使用GPU与有上限的API调用，不进行训练。

**2026-11-04 已完成多帧上下文定位核查与五例复测。** Kev问答仍为3/5，字节比旧实验增加6.54%，且相同请求存在问答波动；本轮没有证明定位或传输质量改善，继续保留现有默认入口。详见 [本轮结论与溯源](docs/evqa_context_results_v4.md)、[多帧上下文实验入口](docs/evqa_joint_localization.md) 和 [下一轮问题审校草案](docs/evqa_grounding_review.md)。全程固定编码器、网络与v4权重，未追加训练。

**追加明确目标的独立开发任务已完成。** 把碗指定为“水槽右侧沥水架内倒扣的碗”后，原图/带宽/Kev各3次问答均正确，Kev比同轮带宽策略少23.04%字节且零卡顿；掩码检查空框4→0，但暂定像素覆盖率58.7%未达到预设80%门槛。毛巾6次均答blue，原green标注待人工复核。新任务不计入原EVQA得分，也不自动加入训练；见 [修订任务与操作](docs/evqa_disambiguation.md)、[运行结果](runs/evqa_disambiguation_v1/REPORT.md) 和 [三步系统检查](runs/evqa_disambiguation_analysis_v1/REPORT.md)。

**定位几何审查与三个明确任务回归已完成。** 碗的低覆盖主要包含API框过小、9秒框错到手臂、12秒偏左；已检查的6个官方掩码基本贴合可见目标，原80%门槛仍未通过。衣服文字/首帧计时器/出租车车牌三例完整执行定位、真实编码、Kev与带宽选档及重复开放问答，27/27与参考答案匹配，418帧/策略完整核验，零卡顿；Kev合计少4.63%字节，节省仅来自衣服一例。数据均为已用开发案例，无新训练。见 [逐帧几何审查](runs/bowl_geometry_review_v1/REPORT.md)、[三例结果](runs/clear_task_regression3_analysis_v1/REPORT.md) 和 [验收口径](docs/region_validation_protocol_v1.md)。

第二步保留 **全部 10 个视频 × 12 个原始候选，共 120 个真实编码候选**，见 [原始完整编码表](runs/roi_encoding_dev10_grid_v1/REVIEW.md)。另补 CRF 46／50 × ROI 偏移 0／-1/5，共40个低码率候选，见 [补充实测表](runs/roi_encoding_dev10_lowrate_v1/rate_size_quality.csv)。合计每视频16动作。这是按质量参数编码后测量实际码率，不是固定目标码率表。每个候选均完整解码，尺寸、帧数和帧率已核验。

**2026-11-03 当前训练入口为 `datasets/semantic_kev_dev_v4`：候选约束与三块规划标签。** 复用原160个编码文件、v3 的网络分组和播放列表，不扩充视频或调用 API。v3 的 ABR 初始化100步训练及闭环评估已完成，平均每场景奖励9.99、卡顿13.91秒，低于带宽基线50.84／3.35秒，见 [v3 分析](runs/semantic_kev_v3_abr_baseline_trial100_v1_analysis/REVIEW.md)。v4 用历史带宽和缓冲筛选候选，并以三块累计预测收益及末端缓冲不足惩罚生成标签。

**v4 的 ABR 初始化400步试训和闭环评估已完成。** 模型平均每场景奖励61.78、卡顿1.197秒、证据PSNR 37.93dB；受约束带宽基线为58.94／1.793秒／37.80dB。20场景中14胜3平3负，但90.5%的总净增益集中于一个场景，仍是当前开发数据上的候选模型。详见 [v4 结果分析](runs/semantic_kev_v4_abr_baseline_trial400_v1_analysis/REVIEW.md)。

默认初始化仍为 `merina656_baseline_seed666/checkpoints/epoch_08`，原始权重只读校验，新权重写入独立目录。当前操作见 [v4 候选约束、规划及 tmux 操作说明](docs/semantic_kev_v4.md)；[v3 说明](docs/semantic_kev_v3.md)保留为历史版本。重复 QA 和扩大视频集不作为这轮修改的前置条件。

从工作区 `/home/amora/wxh` 运行。工具只需要现有 Python 标准库；FFmpeg 已放在本目录 `.tools/`，不会改变旧模型的 Python 环境。

```bash
python jev/semantic_streaming/scripts/devset.py validate
python jev/semantic_streaming/scripts/devset.py validate --require-media
```

## 目录约定

```text
semantic_streaming/
  configs/tools.lock.json             FFmpeg 包、二进制和校验值
  scripts/                           下载、提取、输入导出和核验工具
  configs/qwen_roi_pilot_v1.json      首批 API 模型、地域、参数与坐标约定
  configs/qwen_roi_prompt_v1.txt      不含答案的定位提示词
  configs/roi_encoding_smoke_v2.json  各案例字节预算、保护区派生和编码参数
  configs/roi_encoding_dev10_grid_v1.json 全部10视频的质量参数网格
  configs/semantic_kev_v2.json        对应120候选的决策数据配置
  configs/training_control.json      当前用户要求的训练等待状态
  configs/kev_initialization_v1.json  固定ABR基线初始化及发布版对照，含模型文件哈希
  configs/receiver_qa_prompt_v1.txt   无答案、无编码方式提示的接收端问答指令
  runs/<run_id>/                     实验输入、请求哈希、响应、坐标、代码快照与报告
  tests/                             下载完整性与输入隔离检查
  third_party/                       所使用工具的许可与来源说明
  datasets/pilot_dev10_v1/
    manifest.json                    数据角色、数量和使用边界
    cases.jsonl                      问题、视频 ID、时间窗、来源分组
    task_questions.jsonl             实际使用的中英文问题与修订原因，无答案
    source_groups.jsonl              用于避免潜在同源跨集的当前保守分组
    annotations/answers.jsonl        参考答案与独立核验状态
    annotations/upstream_rows.jsonl  原始行，保留上游答案
    sources.lock.json                Git/HF 固定版本、ZIP 成员及校验信息
    provenance/source_snapshots/     原始 CSV、README 和许可快照
    provenance/archive_indices/      指定版本的 ZIP 中央目录
    provenance/code_snapshots/       产生资产/下载记录的脚本快照
    provenance/downloads/            原视频下载记录、CRC、SHA256
    provenance/preparation/          视频截取/抽帧命令与逐文件 SHA256
    reports/                        样本目录、审阅记录与核验结果
  media/raw/                         本地原视频，Git 忽略
  artifacts/pilot_dev10_v1/           5秒无损参考片、原尺寸帧、接触表；Git 忽略
  .tools/                            本地工具二进制；Git 忽略
```

来源、案例与标注归数据集版本管理；原视频与衍生资产放在独立目录，通过清单和哈希关联。下载记录不代表取得了原视频的再分发权。`DeViBench` 的 MIT 文件是上游仓库声明，不能直接套用到 StreamingBench 中的第三方视频。

## 看样本

打开 [开发样本目录](datasets/pilot_dev10_v1/reports/CATALOG.md)，从每个案例进入接触表或原尺寸证据帧。六帧按行排列，对应片内请求时刻 **0、1、2、3、4、4.8 秒**；实际帧由 FFmpeg 按源帧率选择。

上游 `start_time` 定义一个五秒窗口，本项目保留原视频绝对时间与片内相对时间。`reference_lossless.mp4` 是对原视频已解码画面的无损 H.264 截取，不是原始压缩码流，也不是最终低码率候选。

截取先固定原时间窗内的原生帧数，再无损编码这些帧。部分源视频的首帧落在窗口起点后的一个帧周期内，参考片重置时间戳后的播放时长可能略少于五秒；不能通过补帧凑时长。`audit_pixels.py` 逐帧比较原窗口与参考片的解码像素 SHA256。早期边界问题的失败检查和旧处理记录保留在 `provenance/rejected_derivations/boundary_v0/`。

参考答案与核验记录见 `annotations/answers.jsonl`。`human_verified=false` 表示没有用户/人工审阅记录；助手视觉核对不会被记成人工核验。上游答案保留不覆盖，修正单独记录原因和证据。实际使用的问题在 `task_questions.jsonl`：对计时读数补充首帧锚点，对显卡文字区分产品名称和品牌；上游原题仍保留在 `cases.jsonl` 和 CSV 快照。

## 恢复与复跑

```bash
# 原视频约1.4 GB；只读取所选 ZIP 成员，不下载全数据集。
python jev/semantic_streaming/scripts/devset.py fetch --workers 2

# 在其他机器恢复同一个 FFmpeg；锁定 Linux x86_64 wheel，不安装到共享环境。
python jev/semantic_streaming/scripts/bootstrap_tools.py

# 原尺寸、原帧率、无音轨的5秒无损参考片与六帧证据。
python jev/semantic_streaming/scripts/devset.py prepare

# 换机器后，若保留了处理记录但衍生媒体全部缺失：保留旧记录后重新物化。
python jev/semantic_streaming/scripts/devset.py prepare --rebuild-missing

# 仅导出定位/问答需要的白名单字段，不读取答案文件。
python jev/semantic_streaming/scripts/export_inputs.py --language en
python jev/semantic_streaming/scripts/export_inputs.py --language zh

# 核对来源快照、原视频、衍生资产与审阅证据。
python jev/semantic_streaming/scripts/devset.py validate --require-media
python jev/semantic_streaming/scripts/audit_pixels.py
python -m unittest discover -s jev/semantic_streaming/tests -v
```

下载或准备部分样本时，可重复添加 `--case dev10_01`。脚本拒绝修改过的缓存、错误字节区间和不匹配来源的记录，原视频会核对 ZIP 成员大小及 CRC，再记录 SHA256。整个 ZIP 的 SHA256 由上游元数据提供，本地没有下载整个 ZIP，因此不声称完整 ZIP 已本地验哈希。

已完成且有记录的资产会先核验再跳过；未完成目录或无记录已有文件会拒绝覆盖，保留失败现场供检查。恢复时使用同一版本和脚本；修改窗口、选择或处理流程应创建新数据版本。

`--rebuild-missing` 仅适用于某个案例的衍生文件全部缺失，旧记录保存在 `provenance/preparation_history/`。重建采用当前脚本并生成新资产哈希，不承诺与历史 MP4 的码流字节完全相同；随后必须运行逐帧像素审计与完整性校验。已有但被修改的媒体不会被该选项覆盖。

## 给后续模型的接口

整个 `cases.jsonl` 是数据管理记录，**不要原样传给模型**，其中的来源系列名可能间接提示答案。使用 `export_inputs.py` 生成的 `reports/inference_inputs_en.jsonl` 或中文版，字段仅包含问题、case ID、图像、尺寸与相对时间。

禁止把 `annotations/`、来源原始行、实际候选 QA 得分或未来带宽放入模型在线输入。API 调用保存模型版本、提示词、请求输入哈希、原始返回、耗时与用量，放在独立 run 目录；原始样本准备与后续 API 实验分别记录。

## Qwen API 定位实验

密钥仅从当前进程 `DASHSCOPE_API_KEY` 或本目录 `.env.local` 读取；本地文件权限必须为 `600`，已加入 Git 忽略规则，不会写入实验记录或请求快照。首次默认使用百炼北京地域，地域不匹配时停止，不自动切换境外地域。

以下命令会产生真实 API 调用费用。新实验使用新的 run ID；同一 run ID 只跳过已完成请求，代码、输入或配置哈希变化时拒绝混跑。HTTP 错误或结果不明的中断不自动重试，避免重复计费。首次 `--limit 1`，检查通过后同一 run ID 运行 `--limit 17`，即可完成计划的 18 帧。

```bash
python jev/semantic_streaming/scripts/qwen_roi.py --run-id YOUR_NEW_RUN_ID --limit 1
python jev/semantic_streaming/scripts/qwen_roi.py --run-id YOUR_NEW_RUN_ID --limit 17

# 离线重建报告与请求哈希校验，不发 API 请求；此阶段才读取种子标注。
python jev/semantic_streaming/scripts/report_qwen_roi.py --run-id qwen3vl_roi_pilot_20261102_v2
```

调用使用 Python 标准库 HTTP 接口，绘图使用当前环境已有的 Pillow，不改旧训练环境。原尺寸 PNG 直接上传，服务端视觉像素预算写入配置；模型归一化坐标转为原图半开像素框，当前没有扩边或对齐编码块。`qwen3-vl-plus` 是服务别名，本次响应没有提供不可变快照版本，已在运行清单中明确记录这一复现限制。

原始数据准备阶段未调用 API；上述首批实验才是实际模型定位结果。此前的 `annotations/` 种子框仍保留助手标注身份，不会被改记为 API 输出或人工金标准。

## 完整开发集与编码对照

定位结果分两批保存，汇总逐帧保留来源运行，不复制或改写原始 API 返回。足球案例存在主体关联错误和过大框，原始框未被人工替换；运动问题需进一步做跨帧主体确认。所有失败与待核验案例均在定位汇总报告中保留。

编码阶段只读模型框，不读答案标注。首轮选择三个不同来源分组、目标位置相对稳定的案例。保护区为六帧框的并集，加 8 像素边距后对齐 16 像素宏块，每个编码帧应用同一区域；`frame_rois.jsonl` 明确标记为本地派生区域，不能当作模型逐帧跟踪结果。

均匀/ROI 两组均为同分辨率 `libx264`、相同 preset、AQ=1、mbtree=0。两遍编码迭代匹配完整 MP4 实际字节，含容器开销；每个候选相对目标误差在 1% 内才可比较，成对差值另记。ROI 使用 `addroi` 的相对 qoffset=-1/5。预算过低被编码器拒绝，或迭代后仍无法匹配的结果排除，不能用超预算候选声称收益。

接收端 QA 对有效成对候选和原视频使用相同六帧完整画面，隐藏编码方式和预算，打乱调用顺序，不提供 ROI 或参考答案；全部推理完成后才读取答案评分。使用同一个 Qwen 服务，每个候选只调用一次，尚未验证独立评测模型及重复性。

关键运行：

- `qwen3vl_roi_remaining_20261102_v1`：新增 7 例、42 帧；提示词和调用参数与首批一致。
- `qwen3vl_roi_dev10_summary_20261102_v1`：10 例、60 帧汇总及助手审阅，保留 `source_run`。
- `roi_encoding_smoke_20261102_v3`：18 个编码候选，14 个满足预算匹配，即 7 组有效对照；编码日志、命令、保护区、数值与对照图。
- `receiver_qa_smoke_20261102_v1`：14 个编码候选加 3 个原图对照，共 17 次盲化问答及评分。
- `roi_encoding_smoke_20261102_v1`、`v2`：码率下限导致的中断记录，有 `INCOMPLETE.json`；不混入最终结果。

候选 MP4、两遍统计文件及解码 PNG 在 `artifacts/<encoding_run_id>/`，Git 忽略；运行清单和命令记录通过 SHA256 关联这些资产。

```bash
# 下列新运行会调用 API 或重新编码；使用新的唯一 run ID。
python jev/semantic_streaming/scripts/qwen_roi.py --config configs/qwen_roi_dev10_remaining_v1.json --run-id YOUR_NEW_ROI_RUN --limit 42
python jev/semantic_streaming/scripts/encode_roi_smoke.py --config configs/roi_encoding_smoke_v2.json --run-id YOUR_NEW_ENCODING_RUN
python jev/semantic_streaming/scripts/receiver_qa.py --encoding-run YOUR_NEW_ENCODING_RUN --run-id YOUR_NEW_QA_RUN --limit 21

# 仅从缓存响应、编码资产和评分重建审计报告，不调用 API。
python jev/semantic_streaming/scripts/report_encoding_smoke.py --encoding-run roi_encoding_smoke_20261102_v3 --qa-run receiver_qa_smoke_20261102_v1
```

当前优先完成本地决策训练与推理接线。运动目标关联、独立评测与更大数据作为后续效果改进，不继续扩展为训练前的必做任务。第一版训练采用离线证据 PSNR 质量代理与真实带宽回放卡顿惩罚，输出可用编码配置；数据仍仅用于最小系统开发，不能据此报告正式性能。

## v2 历史编码候选表与输入检查

完整表采用原始分辨率 `libx264`，全局 CRF 为 24／32／40，ROI qoffset 为 0／-1/10／-1/5／-3/10。qoffset=0 是无显式区域偏移的基线；其余三个偏移给重要区域更好的相对量化。CRF 不是固定背景 QP，最终码率以实测完整 MP4 字节计算。

每个候选的 `video_path`、`video_sha256`、`size_bytes`、`actual_bitrate_kbps`、`duration_s`、原生尺寸、帧数、保护区及六帧解码图均保存在运行记录中。120 个候选合计完整解码 16524 帧；六帧抽样位置首先在原参考片上逐项核对原 PNG 哈希。背景与重要区域质量分别记录为 RGB PSNR，区域码率不被虚构成可分离的两路码流。

源 ROI 仍为 API 采样框的静态并集，足球等已知定位问题保留在 `roi_review_flag`，不由编码阶段人工改框。旧的三视频等字节对照表继续保留；新表覆盖范围更全，用于后续实际候选选档，不与旧对照的匹配预算列混用。

```bash
# 新运行编码全部候选；同一代码/配置的部分运行可用原 run ID 继续，已完成候选先校验后跳过。
python jev/semantic_streaming/scripts/encode_roi_grid.py --run-id YOUR_NEW_GRID_RUN

# 只做当前决策数据的 CPU 输入预检查，未带 --execute，不启动训练。
jev/kev/.venv/bin/python jev/semantic_streaming/scripts/train_semantic_kev.py \
  --dataset datasets/semantic_kev_dev_v2 --out runs/semantic_kev_dev_train_v2
```

`semantic_kev_dev_v1` 保留三视频旧版，`v2` 保留十视频静态场景版，`v3` 保留连续传输但一步标签版，当前默认使用 `v4`。v2 最初12候选的详细结构描述略超 Kev 行长度限制，失败版本保存在 `semantic_kev_dev_v2_rejected_context_v0`；v2 最终版全部6000条记录通过实际 tokenizer/encoder 检查，未截断、未删除候选。上面显式指定 v2 的命令仅用于复查旧数据；当前操作见本文顶部入口。

## 数据使用边界

- 这十项已有初步均匀/ROI 编码和接收端 QA，不构成正式压缩敏感性基准。
- 10 个不同 `sample_id` 不等于 10 个独立拍摄/上传来源。当前分组以 `source_groups.jsonl` 为准，合并了数学教学和可能同源的厨房片段；`cases.jsonl` 的分组字段保留初筛记录。上游所查 CSV 未提供原始上传链接。
- 价格案例来自数学教学画面，不能描述成真实超市拍摄；若需要场景泛化应增加实拍数据。
- `question.en` 为上游原题，中文版为助手翻译；对照实验应固定语言，不把翻译差异混成编码效果。
- 文件夹采用 Git 友好的结构，但本工作区当前没有顶层 `.git`，本次没有初始化或提交 Git。

来源：[DeViBench](https://github.com/pku-netvideo/DeViBench)、[StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench)、[Artic](https://github.com/pku-netvideo/Artic)。具体不可变版本、文件快照和各案例 CSV 行号以 `sources.lock.json` 和 `cases.jsonl` 为准。

