# 多帧上下文定位：固定系统对比

固定原视频、采样、问题、选项、编码实现、控制器、网络轨迹和Kev权重。只替换定位上下文及提示词。

| 视频 | 原视频QA（旧→新） | 带宽QA（旧→新） | Kev QA（旧→新） | 旧Kev字节 | 新Kev字节 |
|---|---|---|---|---:|---:|
| evqa_01 | False→True | True→False | False→True | 518792 | 572676 |
| evqa_02 | False→False | False→False | False→False | 572973 | 588440 |

新实验QA：{'reference': 1, 'bandwidth': 0, 'kev': 1}；Kev相对同一套ROI的带宽策略节省 10.06% 字节；播放卡顿 0.000 秒。

完全相同请求字节却回答不同的QA条目：[{'case_id': 'evqa_01', 'variant': 'reference'}]。这些变化是问答器波动，不能算成定位或编码改善。

原问题与参考答案未改动；GT时间窗和掩码仅用于单独离线核查。相同视频不保证API给出相同回答。
ROI变化后质量代理的取样定义也变化，因此不直接用旧/新reward差值证明进步。
原五段视频已经用于开发，后续需另冻结未参与调试的测试集。本次没有训练。

结构化结果见 [comparison.json](comparison.json)，多帧请求上下文见各源运行的 roi/context_plan.json；所有原始响应、失败探索和旧运行均保留。
