# common_v1 验证记录

已固定bundle：`../bundles/common_v1`。

指纹：`7a21822ffc3be5b6618e58175d17813b31534a03f149492bced57a7d7cb06eec`。

## 已完成

1. `python -m pytest jev/abr_compare/test_compare.py -q`：**10 passed**。
2. 来源／数值隔离检查：训练16条纯FCC记录，验证30窗口／2个来源，测试275条；三个划分之间无当前规则可识别的同源或数值重复交集。
3. 对四组数据各3个起点回放48片，共576次分片转移，与 `/home/amora/NetLLM` 的原环境逐步对比，下载、缓冲、卡顿、下一片大小、终止状态及网络指针一致。
4. 首尾排除测试：48片保留46片；辅助指标只排除首片。
5. 真实Kev ABR两轮checkpoint经临时本地HTTP服务完成一局；真实DyLas本地rank128/70轮checkpoint经原加载器完成同一局。两份结果具有相同bundle和会话选择指纹，均为48片／47次选档。它们是接入验证，不是性能优劣结论。
6. BBA、Throughput、RobustMPC各完成30个验证窗口×3个起点，共270局CPU参照回合。
7. 最终`compare.py verify`通过；重新核对660条源目录记录及原模拟器哈希，源内容未变化。
8. 仅为接入验证启动的8012端口临时Jev服务已停止。没有停止或切换用户原有服务，没有训练或改写权重。

## 结果位置

- [真实模型接入比较](./common_v1_model_adapters.md)
- [Jev接入摘要](./final_jev_smoke/summary.json)
- [DyLas接入摘要](./final_dylas_smoke/summary.json)
- [完整验证集传统参照](../results/valid_references_v1.md)
- [来源排除审计](../bundles/common_v1/split_audit.json)

较早的`*_adapter_smoke*`、临时bundle路径记录等目录是开发接入过程的证据，包含一次Python3.10哈希API兼容性失败，已修复；不与final/common_v1结果混合。最终比较只使用上面列出的final文件。

## 尚未完成／不能由这些检查推出

- 没有重新训练DyLas或Jev，也没有生成新的共同专家经验池。
- 没有运行两模型的完整四组测试，不能宣称超过论文。
- 现有旧checkpoint默认只作诊断；它们的训练来源与选模独立性需另行核对。
- 验证只有2个原始来源；重复窗口、起点不能增加独立来源数。
- 当前桥接器仅支持README声明的legacy-v4结构，不包括所有DyLas消融／rebuttal变体。
- 正式训练必须接着使用冻结的train/valid清单；教师、监督格式、优化预算还需在训练阶段单独固定。
