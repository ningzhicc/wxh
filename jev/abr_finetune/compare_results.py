"""Compare the selected ABR checkpoint with the frozen zero-shot experiment."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import atomic_json, digest
from report import interval, write_csv

LABELS = {"kev_sft": "Kev ABR SFT", "kev": "Kev zero-shot", "bba": "BBA",
          "throughput": "Throughput-0.85", "robustmpc": "RobustMPC-6"}
COLORS = {"kev_sft": "#dc7625", "kev": "#c84031", "bba": "#697780",
          "throughput": "#319a88", "robustmpc": "#3574b4"}


def generate(run_name):
    out = ROOT / "results" / run_name
    old = ROOT.parent / "abr_eval/results/zero_shot_v1"
    new = out / "test"
    selected = json.loads((out / "selected.json").read_text())
    before = json.loads((old / "manifest.json").read_text())
    after = json.loads((new / "manifest.json").read_text())
    assert json.loads((old / "completion.json").read_text())["complete"]
    assert json.loads((new / "audit.json").read_text())["passed"]
    assert before["traces"] == after["traces"], "Test trace contents or starts changed"
    assert before["video_sha256"] == after["video_sha256"]
    for key in ["base", "base_revision"]:
        assert before["model_deployment"][key] == after["model_deployment"][key]
    for key in ["device", "backend", "dtype"]:
        assert before["service"][key] == after["service"][key]
    initial_path = Path(before["model_deployment"]["checkpoint_path"])
    assert all(digest(initial_path / filename) == sha for filename, sha in before["model_files_sha256"].items())
    for key in ["prompt", "criteria", "skip_first_reward", "charge_decision_latency"]:
        assert before["config"][key] == after["config"][key], key
    assert before["source_sha256"]["benchmark.py"] == after["source_sha256"]["benchmark.py"]
    original = [json.loads(p.read_text()) for p in sorted((old / "episodes").glob("*/*/*.json"))]
    adapted = [json.loads(p.read_text()) for p in sorted((new / "episodes").glob("*/*/*.json"))]
    assert len(original) == 1100 and len(adapted) == 275
    assert all(ep["fingerprint"] == before["fingerprint"] for ep in original)
    assert all(ep["fingerprint"] == after["fingerprint"] for ep in adapted)
    results = original + [{**ep, "method": "kev_sft"} for ep in adapted]
    datasets = ["fcc", "hsr", "oboe", "norway3g"]
    methods = ["kev", "kev_sft", "bba", "throughput", "robustmpc"]
    groups = {(d, m): [r for r in results if r["dataset"] == d and r["method"] == m]
              for d in datasets for m in methods}
    summary, pairs = [], []
    fields = ["qoe_mean", "mean_bitrate_mbps", "rebuffer_s", "startup_s", "smoothness_mbps", "switches"]
    for dataset in datasets:
        for method in methods:
            group = groups[dataset, method]
            low, high = interval([r["metrics"]["qoe_mean"] for r in group])
            latencies = [s["decision"]["client_ms"] for r in group for s in r["steps"][1:]]
            summary.append({"dataset": dataset, "method": method, "episodes": len(group),
                            **{f: float(np.mean([r["metrics"][f] for r in group])) for f in fields},
                            "qoe_ci_low": low, "qoe_ci_high": high,
                            "no_stall_fraction": float(np.mean([r["metrics"]["no_stall"] for r in group])),
                            "client_latency_mean_ms": float(np.mean(latencies)),
                            "client_latency_p95_ms": float(np.quantile(latencies, .95))})
        adapted_by_key = {r["key"]: r for r in groups[dataset, "kev_sft"]}
        for method in ["kev", "bba", "throughput", "robustmpc"]:
            other = {r["key"]: r for r in groups[dataset, method]}
            assert adapted_by_key.keys() == other.keys()
            gaps = np.array([adapted_by_key[k]["metrics"]["qoe_mean"] - other[k]["metrics"]["qoe_mean"]
                             for k in sorted(other)])
            low, high = interval(gaps)
            pairs.append({"dataset": dataset, "baseline": method, "episodes": len(gaps),
                          "sft_minus_baseline_qoe": float(gaps.mean()), "ci_low": low, "ci_high": high,
                          "sft_win_fraction": float(np.mean(gaps > 1e-9)),
                          "tie_fraction": float(np.mean(np.abs(gaps) <= 1e-9))})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_comparisons.csv", pairs)
    write_csv(out / "episode_metrics.csv", [{"dataset": r["dataset"], "method": r["method"],
                                             "trace": r["trace"], **r["metrics"]} for r in results])
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "paired_comparisons.json", pairs)
    training = json.loads((ROOT / "runs" / run_name / "training_metrics.json").read_text())
    data = json.loads((ROOT / "data/manifest.json").read_text())
    source = json.loads((ROOT / "data/source_overlap.json").read_text())
    data_audit = json.loads((ROOT / "data_audit.json").read_text())
    assert data_audit["passed"]
    atomic_json(out / "source_overlap.json", source)
    atomic_json(out / "data_audit.json", data_audit)
    validation = json.loads((out / "valid/summary.json").read_text())[0]
    repeat = json.loads((out / "validation_repeat_audit.json").read_text()) if (out / "validation_repeat_audit.json").exists() else None
    lines = ["# Kev-4B ABR 监督微调与闭环评测", "",
             "从已发布 Kev-4B 的 LoRA 和 pointer head 继续训练。基座 Qwen3.5-4B-Base 冻结；不是全参数训练，也不是直接训练裸 Qwen。",
             "训练标签由 RobustMPC-6 提供，属于监督行为模仿，本轮没有强化学习或 ABR 概率校准。", "",
             f"训练使用 NetLLM 官方 fcc-train 目录隔离后 {data['counts']['train_traces']} 条轨迹、{data['counts']['selected_train_records']:,} 个状态，"
             f"训练两轮，共 {training['optimizer_steps']} 个优化步骤；有效 batch=16、lr=2e-5。耗时 {training['wall_seconds']/60:.1f} 分钟。",
             "原始训练/验证集与测试集存在数值重复及同源记录的不同窗口；已按完整原始采集记录隔离后再训练。"
             f"有效验证集 {data['counts']['valid_traces']} 条，使用 {data['counts']['valid_selection_traces']} 条选择检查点，"
             f"仅来自 {source['retained_valid_unique_parents']} 个 FCC 采集 ID，验证来源较少。",
             "目录名 fcc-train 不表示纯 FCC 来源：保留训练轨迹含 16 条 trace_* 与 39 条 Norway 路线日志。"
             "因此 Norway3G 的结果不能解读为完全没有接触该采集项目的跨域泛化；同一原始日志及完全相同数值轨迹仍已剔除。详情见 source_overlap.json。", "",
             f"选中检查点：`{selected['checkpoint']}`。选择依据仅为清洁 FCC-valid 的闭环平均 QoE，未使用测试结果选参。", "",
             "| 检查点 | 清洁 FCC-valid QoE/块 |", "|---|---:|"]
    for row in selected["candidates"]:
        lines.append(f"| {row['name']} | {row['qoe_mean']:.4f} |")
    lines += [f"| 发布版参照 | {selected['original_validation_qoe']:.4f} |", "",
              f"选中模型完整清洁 FCC-valid {data['counts']['valid_traces']} 条：QoE/块 {validation['qoe_mean']:.4f}，"
              f"平均码率 {validation['mean_bitrate_mbps']:.3f} Mbps，平均卡顿 {validation['rebuffer_s']:.3f} 秒/会话。", "",
              "四组测试采用与原零样本实验完全相同的轨迹、起点、视频、仿真器、提示词和候选。BBA/Throughput/MPC 引用原固定实验结果。",
              "QoE 越高越好，卡顿秒越低越好。每条会话播放 48 块，每块 4 秒；主指标排除固定起始块，"
              "QoE/块 = 平均码率 Mbps − 4.3 × 卡顿秒/47 − 平均切换幅度 Mbps。推理等待未计入 QoE。", "",
              "| 数据集 | 方法 | 会话 | QoE/块 [95% CI] | 平均码率 Mbps | 平均卡顿秒 |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {LABELS[row['method']]} | {row['episodes']} | "
                     f"{row['qoe_mean']:.4f} [{row['qoe_ci_low']:.4f}, {row['qoe_ci_high']:.4f}] | "
                     f"{row['mean_bitrate_mbps']:.3f} | {row['rebuffer_s']:.3f} |")
    if repeat:
        lines += ["", f"同一检查点复跑验证时，{repeat['episodes']} 条轨迹的 {repeat['chunks']-repeat['episodes']} 个决策中 "
                  f"{repeat['changed_actions_on_repeated_validation']} 个动作发生变化，单会话 QoE 最大差异 "
                  f"{repeat['maximum_qoe_difference']:.4f}。可能与 BF16/动态批处理的数值差异有关；"
                  "这是基于复跑现象的推测，未定位到具体算子。两个检查点的验证均值差距很小，"
                  "本轮按预定规则选中第二轮，不能据此宣称第二轮显著优于第一轮。"]
    lines += ["", "成对差值使用同一轨迹的微调模型减去对照方法；区间为 5,000 次会话 bootstrap 的 95% 区间。", "",
              "| 数据集 | 对照 | SFT − 对照 QoE/块 [95% CI] | 胜出会话比例 |", "|---|---|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['dataset']} | {LABELS[row['baseline']]} | {row['sft_minus_baseline_qoe']:+.4f} "
                     f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['sft_win_fraction']:.1%} |")
    lines += ["", "监督模仿不能保证超过规划专家；MPC 可读取未来五块的视频大小，Kev 只读取下一块大小。"
              "单种子、单起点结果与会话 bootstrap 不代表重复训练或真实部署的全部不确定性；"
              "同一来源的不同窗口可能相关。", "",
              "测试轨迹仅用于来源与数值去重隔离，未读取测试 QoE 或动作作为训练/选参标签。概率温度为 1，"
              "未拟合 ABR 校准。零样本发布版使用其自带温度；温度不改变 argmax，但两个模型的概率不能直接作校准比较。", "",
              "- [QoE 比较图](qoe_comparison.png)", "- [QoE 分布](qoe_cdf.png)",
              "- [卡顿与码率](streaming_metrics.png)", "- [训练损失](training_loss.png)",
              "- [汇总 CSV](summary.csv)", "- [成对比较 CSV](paired_comparisons.csv)",
              "- [每会话指标](episode_metrics.csv)", "- [检查点选择](selected.json)",
              "- [测试数值回放校验](test/audit.json)", "- [模型训练来源与完整性](training_audit.json)",
              "- [数据来源与重复片段检查](data_audit.json)", "",
              "原始逐步状态、动作、概率、延迟在 test/episodes/，兼容 NetLLM 的八列日志在 test/logs/。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(out, datasets, methods, groups, summary, ROOT / "logs" / f"{run_name}.log", training["optimizer_steps"])
    atomic_json(out / "comparison_audit.json", {"passed": True, "same_test_traces_and_starts": True,
                "same_prompt_and_candidates": True, "same_benchmark_code": True,
                "same_video_files": True, "same_serving_backend_and_dtype": True,
                "same_base_revision": True, "original_checkpoint_unchanged": True,
                "original_fingerprint": before["fingerprint"], "adapted_fingerprint": after["fingerprint"],
                "source_sha256": digest(__file__)})
    return summary


def figures(out, datasets, methods, groups, summary, log, optimizer_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    def save(fig, name):
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for axis, dataset in zip(axes, datasets):
        for method in methods:
            values = np.sort([r["metrics"]["qoe_mean"] for r in groups[dataset, method]])
            axis.step(values, np.arange(1, len(values)+1)/len(values), where="post", color=COLORS[method],
                      label=LABELS[method], linewidth=2 if method == "kev_sft" else 1.3)
        axis.set(title=dataset.upper(), xlabel="Mean QoE / chunk", ylim=(0, 1))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Session CDF")
    axes[-1].legend(fontsize=8, loc="lower right")
    save(fig, "qoe_cdf")
    # Two panels show both the full zero-shot range and the adapted/baseline detail without hiding values.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for axis, shown in zip(axes, [methods, [m for m in methods if m != "kev"]]):
        width = .8 / len(shown)
        for index, method in enumerate(shown):
            rows = [next(r for r in summary if r["dataset"] == d and r["method"] == method) for d in datasets]
            means = np.array([r["qoe_mean"] for r in rows])
            error = [[max(0, r["qoe_mean"]-r["qoe_ci_low"]) for r in rows],
                     [max(0, r["qoe_ci_high"]-r["qoe_mean"]) for r in rows]]
            axis.bar(np.arange(4)-.4+width/2+index*width, means, width, yerr=error, capsize=2,
                     color=COLORS[method], label=LABELS[method])
        axis.set(xticks=np.arange(4), xticklabels=[d.upper() for d in datasets], ylabel="Mean QoE / chunk")
        axis.axhline(0, color="black", lw=.6)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8, ncol=2)
    axes[0].set_title("All policies")
    axes[1].set_title("Adapted policy and ABR baselines")
    save(fig, "qoe_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for axis, field, label in zip(axes, ["rebuffer_s", "mean_bitrate_mbps"], ["Mean rebuffer seconds / session", "Mean bitrate (Mbps)"]):
        width = .8 / len(methods)
        for index, method in enumerate(methods):
            values = [next(r for r in summary if r["dataset"] == d and r["method"] == method)[field] for d in datasets]
            axis.bar(np.arange(4)-.4+width/2+index*width, values, width, color=COLORS[method], label=LABELS[method])
        axis.set(xticks=np.arange(4), xticklabels=[d.upper() for d in datasets], ylabel=label)
        axis.grid(axis="y", alpha=.2)
    axes[-1].legend(fontsize=8, ncol=2)
    save(fig, "streaming_metrics")
    entries = [(int(s), float(v)) for s, v in re.findall(r"step (\d+)/\d+ loss ([\d.]+)", log.read_text())]
    assert len(entries) == optimizer_steps // 10 and optimizer_steps - entries[-1][0] < 10
    write_csv(out / "training_loss.csv", [{"step": s, "training_cross_entropy": v} for s, v in entries])
    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(*zip(*entries), color=COLORS["kev_sft"])
    axis.axvline(optimizer_steps / 2, color="gray", linestyle="--", lw=.8)
    axis.set(xlabel="Optimizer step", ylabel="Training cross entropy (last 10 steps)")
    axis.grid(alpha=.2)
    save(fig, "training_loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sft_v3")
    print(json.dumps(generate(parser.parse_args().run_name), indent=2))
