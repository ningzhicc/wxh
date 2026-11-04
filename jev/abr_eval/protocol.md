# Kev 零样本 ABR 评测协议 v1

本轮固定使用已经部署的 Kev-4B v1.0，仅通过提示词描述任务，不训练、不拟合温度、不使用专家动作标签或 few-shot 样例。不根据这四个测试集的 QoE 修改提示词。

数据来源：NetLLM `fcc-test` 全部 100 条；ABRBench `HSR` 全部 34 条 OOD 轨迹、`Oboe/test` 全部 100 条、`Norway3G/test` 全部 41 条。仓库提交及逐文件 SHA256、实际起点记录在输出 manifest 中。FCC 保留仓库的 `mahimahi_ptrs.pkl` 起点；其他集合通过固定 seed=666 和文件名派生起点，所有算法逐轨迹共享同一起点。

统一使用 NetLLM 原版 `baseline_special.env.Environment` 和 video1（与 ABRBench envivio_3g 字节一致）：48 个 4 秒视频块、[300,750,1200,1850,2850,4300] kbps，RTT 80 ms、payload 0.95、60 秒缓冲上限。fixed=True，关闭随机下载噪声，循环读取轨迹的行为与原环境一致。

每条轨迹第一个块固定为 300 kbps，策略决定后面 47 个块。主指标遵循 NetLLM：排除第一个块的 reward，逐块 reward = bitrate_Mbps − 4.3×rebuffer_seconds − |bitrate_Mbps−previous_bitrate_Mbps|，先取每个会话的均值，再对会话等权平均。另报告总 QoE、平均码率、启动时间、启动后卡顿秒数、码率切换幅度与次数。卡顿比例定义为启动后卡顿秒数/(192 秒视频+启动后卡顿秒数)。

Kev 仅观察最近 6 个已下载块的实测吞吐率、下载耗时、码率，以及当前缓冲、上一个动作、下一块六档大小、剩余块数。不会看到未来网络带宽、数据集名称、轨迹名称或基线动作。动作是六档概率的 argmax，HTTP/输出异常即评测失败，不静默退化到另一策略。逐步请求状态、概率、动作和延迟全部保存。

传统对照：BBA（NetLLM reservoir=5s、cushion=10s）；吞吐率策略（最近 5 个观测的调和均值×0.85，选不超过该预算的最高码率）；RobustMPC（调和均值/(1+最近 5 次最大预测相对误差)，规划未来最多 5 个块，遍历所有六档动作序列）。RobustMPC 使用视频 manifest 的未来块大小，没有未来带宽；其动作空间为六档，不使用 NetLLM 原 MPC 的 ±1 档限制；并修正原 MPC 下一块下标偏移和跨会话预测历史问题。因此这里标为 RobustMPC-6，不直接等同于仓库历史 mpc 数字。

推理耗时独立记录，默认不扣减仿真缓冲、不推进网络轨迹，遵循仓库离线模拟口径。因此 QoE 不是含模型响应时间的真实播放器结果。并行会话用于运行效率，会报告并发数及客户端/服务端延迟；并发下客户端延迟包含排队。

全量结果不与未核实起点、数据和仿真设置的旧日志混合。主要比较在每个集合内部，同一 trace/start-point 的策略成对比较；汇总差值使用固定种子的 95% 会话 bootstrap 区间，仅反映本次轨迹样本的不确定性。

输出包括可续跑的每会话 JSON、NetLLM 格式的八列日志、episode_metrics.csv、summary.csv、paired_comparisons.csv、分集合 QoE CDF PNG/PDF、汇总图 PNG/PDF 和中文结果报告。未完成全部会话时不能生成标为完成的全量结果。
