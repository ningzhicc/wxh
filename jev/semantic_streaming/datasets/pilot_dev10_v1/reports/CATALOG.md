# 10个开发视频与证据目录

用途：pilot_dev，仅开发。答案不是独立人工金标准；助手视觉审阅状态与人工核验分开。

时间窗为原视频绝对秒数，六帧接触表按行对应片内 0、1、2、3、4、4.8 秒。原尺寸证据可点击查看。

| ID / 上游视频 | 实际使用的问题 | 当前参考答案 | 原视频窗口 | 审阅状态 | 预览 |
| --- | --- | --- | --- | --- | --- |
| dev10_01 / 223 | 衣服上的文字是什么？ | math Antics | 110–115s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_01/contact_sheet.jpg) |
| dev10_02 / 63 | 镊子上印着哪个品牌？ | Cricut | 100–105s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_02/contact_sheet.jpg) |
| dev10_03 / 354 | 第一帧中，电子秤右侧的重量读数是多少？ | 16.5 | 110–115s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_03/contact_sheet.jpg) |
| dev10_04 / 119 | 第一帧中，显卡外壳上压印的名称是什么？ | SUPRIM | 60–65s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_04/contact_sheet.jpg) |
| dev10_05 / 217 | 画面中橙子堆前面的价签显示多少钱？ | 1.49 | 40–45s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_05/contact_sheet.jpg) |
| dev10_06 / 4 | 正在踢球的巴塞罗那球员穿几号？ | 7 | 110–115s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_06/contact_sheet.jpg) |
| dev10_07 / 23 | 这段视频第一帧中，计时器显示什么时间？ | 04:30 | 160–165s | 题目时间锚点及答案已修订，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_07/contact_sheet.jpg) |
| dev10_08 / 388 | 画面中央黄色出租车的车牌号码是什么？ | 2K30H | 150–155s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_08/contact_sheet.jpg) |
| dev10_09 / 451 | 天桥限高标志标注的高度是多少？ | 4.6m | 110–115s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_09/contact_sheet.jpg) |
| dev10_10 / 21 | 案板上的小标签写着什么？ | BUTTER | 30–35s | 助手画面支持，未人工核验 | [接触表](../../../artifacts/pilot_dev10_v1/dev10_10/contact_sheet.jpg) |

## 逐项取证与使用建议

### dev10_01：教育演示 / 衣物文字

- 中文：衣服上的文字是什么？
- English: What is written on the person's t-shirt?
- 原视频 `sample_223`，窗口 `110–115s`；片内时间从零开始。
- 当前参考答案：`math Antics`；上游原答案：`math Antics`。
- 应定位：衣服胸前文字。
- 来源分组：`math_antics`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`137`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_01/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_01/contact_sheet.jpg)。
- 助手审阅：片内前2秒仅见公式，3秒以后讲解者出现；已在4秒原尺寸帧读到 math Antics。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_01/frame_04.png)：片内请求时刻 `4s`，原视频名义时刻 `114s`；尺寸 `[1280, 720]`；该帧 ROI 种子 `[860, 390, 1060, 540]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_02：工具品牌

- 中文：镊子上印着哪个品牌？
- English: What brand name is printed on the tweezers?
- 原视频 `sample_63`，窗口 `100–105s`；片内时间从零开始。
- 当前参考答案：`Cricut`；上游原答案：`Cricut`。
- 应定位：镊子手柄上的品牌文字。
- 来源分组：`craft_tutorials`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`38`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_02/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_02/contact_sheet.jpg)。
- 助手审阅：首帧白色镊子手柄清晰可见Cricut；后续工具移动并转向。右下角DIY CRAFT TUTORIALS是视频水印，不是所问工具品牌。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_02/frame_00.png)：片内请求时刻 `0s`，原视频名义时刻 `100s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[1530, 382, 1665, 469]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_03：仪表读数

- 中文：第一帧中，电子秤右侧的重量读数是多少？
- English: What value is shown on the right-hand weight readout of the digital scale in the first frame?
- 原视频 `sample_354`，窗口 `110–115s`；片内时间从零开始。
- 当前参考答案：`16.5`；上游原答案：`16.5`。
- 应定位：电子秤数字显示屏。
- 来源分组：`kitchen_coffee`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`237`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_03/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_03/contact_sheet.jpg)。
- 问题修订：秤面同时有计时字段与重量字段，明确所问为右侧重量读数。
- 助手审阅：首帧电子秤右侧重量读数清晰显示16.5，左侧另有0:00计时字段；实际题目明确问右侧重量读数，未臆测画面未显示的单位。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_03/frame_00.png)：片内请求时刻 `0s`，原视频名义时刻 `110s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[1125, 640, 1230, 714]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_04：硬件标识

- 中文：第一帧中，显卡外壳上压印的名称是什么？
- English: What name is embossed on the graphics-card shroud in the first frame?
- 原视频 `sample_119`，窗口 `60–65s`；片内时间从零开始。
- 当前参考答案：`SUPRIM`；上游原答案：`SUPRIM`。
- 应定位：显卡外壳上的名称。
- 来源分组：`pc_hardware`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`1172`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_04/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_04/contact_sheet.jpg)。
- 问题修订：上游称 brand name，但可见词可能是产品系列；改问压印名称，保留同一证据目标。
- 助手审阅：首帧显卡外壳清晰压印 SUPRIM；它是可见名称，不将其误标为制造商品牌。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_04/frame_00.png)：片内请求时刻 `0s`，原视频名义时刻 `60s`；尺寸 `[1920, 960]`；该帧 ROI 种子 `[450, 235, 810, 320]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_05：教学画面 / 价格数字

- 中文：画面中橙子堆前面的价签显示多少钱？
- English: What is the price displayed on the tag in front of the orange pyramid in the grocery store?
- 原视频 `sample_217`，窗口 `40–45s`；片内时间从零开始。
- 当前参考答案：`1.49`；上游原答案：`1.49`。
- 应定位：橙子堆前面的价签。
- 来源分组：`math_antics`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`1214`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_05/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_05/contact_sheet.jpg)。
- 助手审阅：前3秒为讲解者，4秒时出现超市照片合成背景。橙子堆前标牌写 1,49，小数逗号对应上游 1.49；未推断货币。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_05/frame_04.png)：片内请求时刻 `4s`，原视频名义时刻 `44s`；尺寸 `[1280, 720]`；该帧 ROI 种子 `[195, 258, 311, 408]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_06：运动目标 / 球衣号码

- 中文：正在踢球的巴塞罗那球员穿几号？
- English: What number does the Barcelona player taking the kick wear?
- 原视频 `sample_4`，窗口 `110–115s`；片内时间从零开始。
- 当前参考答案：`7`；上游原答案：`7`。
- 应定位：踢球球员球衣背后的号码。
- 来源分组：`football_broadcast`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`16`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_06/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_06/contact_sheet.jpg)。
- 问题修订：近景号码在短裤上清晰可见；改问球员号码，不强制认定证据只来自球衣。
- 助手审阅：片内3秒踢球队员短裤上的7清晰可见；修订题目为球员穿几号，避免把仅在短裤可见的号码强说成球衣背部证据。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_06/frame_03.png)：片内请求时刻 `3s`，原视频名义时刻 `113s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[515, 570, 620, 680]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_07：烹饪 / 时间读数

- 中文：这段视频第一帧中，计时器显示什么时间？
- English: What time is displayed on the timer in the first frame of the clip?
- 原视频 `sample_23`，窗口 `160–165s`；片内时间从零开始。
- 当前参考答案：`04:30`；上游原答案：`4:26`。
- 应定位：画面中的计时器。
- 来源分组：`cooking_kitchen_conservative`；相同风格的厨房布景与剪辑，保守合并以避免潜在同源跨集；不声称已取得原始上传身份。
- CSV 数据行：`667`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_07/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_07/contact_sheet.jpg)。
- 问题修订：计时器会变化；补充首帧时间锚点，避免把整个五秒窗口的不同读数混为错误。
- 助手审阅：首帧计时器清晰显示04:30。原题磨胡椒动作对应窗口后段，约4秒显示04:26；本项目改问首帧，因此参考答案相应改为04:30，未认定原题答案错误。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_07/frame_00.png)：片内请求时刻 `0s`，原视频名义时刻 `160s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[1530, 850, 1860, 1020]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_08：街景 / 运动文字

- 中文：画面中央黄色出租车的车牌号码是什么？
- English: What is the license plate number of the central yellow taxi?
- 原视频 `sample_388`，窗口 `150–155s`；片内时间从零开始。
- 当前参考答案：`2K30H`；上游原答案：`2K30H`。
- 应定位：中央黄色出租车的车牌。
- 来源分组：`city_walk`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`580`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_08/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_08/contact_sheet.jpg)。
- 助手审阅：中央出租车的后牌照读作2K30H。车顶牌只显示2K30，不能把它当作完整车牌证据；画面有相邻出租车等干扰目标。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_08/frame_00.png)：片内请求时刻 `0s`，原视频名义时刻 `150s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[905, 923, 1048, 1002]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_09：道路 / 限高读数

- 中文：天桥限高标志标注的高度是多少？
- English: What is the clearance height indicated on the overpass sign?
- 原视频 `sample_451`，窗口 `110–115s`；片内时间从零开始。
- 当前参考答案：`4.6m`；上游原答案：`4.6m`。
- 应定位：天桥上的限高标志。
- 来源分组：`road_driving`；保留初筛内容分组，未认证原始上传身份。
- CSV 数据行：`297`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_09/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_09/contact_sheet.jpg)。
- 助手审阅：片内1秒的天桥标牌可读CLEARANCE 4.6m；约2秒后车辆驶过，目标离开画面，不能给整段套一个静态框。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_09/frame_01.png)：片内请求时刻 `1s`，原视频名义时刻 `111s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[1010, 328, 1142, 394]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

### dev10_10：烹饪 / 食材标签

- 中文：案板上的小标签写着什么？
- English: What is written on the small label on the cutting board?
- 原视频 `sample_21`，窗口 `30–35s`；片内时间从零开始。
- 当前参考答案：`BUTTER`；上游原答案：`BUTTER`。
- 应定位：案板上的小标签。
- 来源分组：`cooking_kitchen_conservative`；相同风格的厨房布景与剪辑，保守合并以避免潜在同源跨集；不声称已取得原始上传身份。
- CSV 数据行：`968`（不含表头），原行哈希见 cases.jsonl。
- [5秒无损参考片](../../../artifacts/pilot_dev10_v1/dev10_10/reference_lossless.mp4)；[六帧接触表](../../../artifacts/pilot_dev10_v1/dev10_10/contact_sheet.jpg)。
- 助手审阅：首帧案板上的包装标签可读BUTTER，文字朝向倒置。后续转为人物与食材近景，标签不再持续可见。
- [原尺寸证据帧](../../../artifacts/pilot_dev10_v1/dev10_10/frame_00.png)：片内请求时刻 `0s`，原视频名义时刻 `30s`；尺寸 `[1920, 1080]`；该帧 ROI 种子 `[1080, 768, 1216, 845]`。
- ROI 是助手对该证据帧的初步像素框，不是多帧标注、模型输出或经过人工认证的真值。

## 溯源

- DeViBench commit：`cab75d3e96ebf35fb3b833498c20787d4e75f93f`。
- StreamingBench revision：`48872fa707124474ce4c5172ddc58efb8bc88058`。
- 原题、原答案、来源 README 和许可文件保留在 provenance/source_snapshots；完整源行另存 annotations/upstream_rows.jsonl。
- 下载和衍生资产的 SHA256、ZIP CRC、命令与工具版本见 provenance/。
- 这些候选尚未接受实际低码率问答测试；是否对压缩敏感需后续实验。
