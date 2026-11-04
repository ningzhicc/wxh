"""Aggregate complete paired episodes and export figures and a Chinese report."""
import csv
import json
from pathlib import Path

import numpy as np

LABELS = {"kev": "Kev-4B zero-shot", "bba": "BBA", "throughput": "Throughput-0.85", "robustmpc": "RobustMPC-6"}
COLORS = {"kev": "#c84031", "bba": "#697780", "throughput": "#319a88", "robustmpc": "#3574b4"}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, seed=20261014):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(boot, [.025, .975]))


def generate(out):
    from benchmark import atomic_json
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("*/*/*.json"))]
    expected = {(method, trace["key"]) for method in manifest["config"]["methods"] for trace in manifest["traces"]}
    assert {(r["method"], r["key"]) for r in results} == expected, "Cannot report an incomplete run"
    assert all(r["fingerprint"] == manifest["fingerprint"] and r["metrics"]["chunks"] == 48 for r in results)
    metrics = [{"dataset": r["dataset"], "method": r["method"], "trace": r["trace"],
                "start_pointer": r["start_pointer"], **r["metrics"]} for r in results]
    write_csv(out / "episode_metrics.csv", metrics)
    datasets, methods = manifest["config"]["datasets"], manifest["config"]["methods"]
    summary, pairs = [], []
    fields = ["qoe_mean", "mean_bitrate_mbps", "rebuffer_s", "startup_s", "rebuffer_ratio", "smoothness_mbps", "switches"]
    groups = {}
    for dataset in datasets:
        for method in methods:
            group = [r for r in results if r["dataset"] == dataset and r["method"] == method]
            groups[dataset, method] = group
            mean = {field: float(np.mean([r["metrics"][field] for r in group])) for field in fields}
            lower, upper = interval([r["metrics"]["qoe_mean"] for r in group])
            latencies = [step["decision"]["client_ms"] for r in group for step in r["steps"][1:]]
            server = [step["decision"]["server_ms"] for r in group for step in r["steps"][1:] if step["decision"]["server_ms"] is not None]
            summary.append({"dataset": dataset, "method": method, "episodes": len(group), **mean,
                            "qoe_ci_low": lower, "qoe_ci_high": upper,
                            "no_stall_fraction": float(np.mean([r["metrics"]["no_stall"] for r in group])),
                            "client_latency_mean_ms": float(np.mean(latencies)),
                            "client_latency_p50_ms": float(np.quantile(latencies, .5)),
                            "client_latency_p95_ms": float(np.quantile(latencies, .95)),
                            "server_latency_mean_ms": float(np.mean(server)) if server else None})
        if "kev" in methods:
            kev = {r["key"]: r["metrics"]["qoe_mean"] for r in groups[dataset, "kev"]}
            for method in methods:
                if method == "kev":
                    continue
                other = {r["key"]: r["metrics"]["qoe_mean"] for r in groups[dataset, method]}
                assert kev.keys() == other.keys()
                gaps = np.array([kev[key] - other[key] for key in sorted(kev)])
                lower, upper = interval(gaps)
                pairs.append({"dataset": dataset, "baseline": method, "episodes": len(gaps),
                              "kev_minus_baseline_qoe": float(gaps.mean()), "ci_low": lower, "ci_high": upper,
                              "kev_win_fraction": float(np.mean(gaps > 1e-9)),
                              "tie_fraction": float(np.mean(np.abs(gaps) <= 1e-9))})
    write_csv(out / "summary.csv", summary)
    if pairs:
        write_csv(out / "paired_comparisons.csv", pairs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.5), squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        for method in methods:
            values = np.sort([r["metrics"]["qoe_mean"] for r in groups[dataset, method]])
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=LABELS[method],
                      color=COLORS[method], linewidth=2 if method == "kev" else 1.4)
        axis.set(title=f"{dataset.upper()} (n={len(groups[dataset, methods[0]])})", xlabel="Mean linear QoE / chunk", ylim=(0, 1))
        axis.grid(alpha=.18)
    axes[0, 0].set_ylabel("Session CDF")
    axes[0, -1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"qoe_cdf.{extension}", bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 4))
    width = .8 / len(methods)
    for i, method in enumerate(methods):
        rows = [next(r for r in summary if r["dataset"] == d and r["method"] == method) for d in datasets]
        means = np.array([r["qoe_mean"] for r in rows])
        errors = np.array([[max(0, r["qoe_mean"] - r["qoe_ci_low"]) for r in rows],
                           [max(0, r["qoe_ci_high"] - r["qoe_mean"]) for r in rows]])
        axis.bar(np.arange(len(datasets)) - .4 + width / 2 + i * width, means, width=width,
                 yerr=errors, capsize=2, color=COLORS[method], label=LABELS[method])
    axis.set(xticks=np.arange(len(datasets)), xticklabels=[d.upper() for d in datasets], ylabel="Mean linear QoE / chunk")
    axis.axhline(0, color="black", lw=.6)
    axis.legend(ncol=2, fontsize=9)
    axis.grid(axis="y", alpha=.18)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"qoe_comparison.{extension}", bbox_inches="tight")
    plt.close(fig)
    lines = ["# Kev-4B 零样本 ABR 全量评测", "",
             "本轮使用已发布的 Kev-4B v1.0，未做 ABR 微调、温度拟合或 few-shot 训练。四组测试集统一视频和仿真器，所有方法逐轨迹共享起点。", "",
             "## 结果", "",
             "QoE 越高越好；码率、卡顿与切换均为会话均值，QoE 排除固定起始块。区间是 95% 会话 bootstrap 区间。", "",
             "| 数据集 | 方法 | 会话数 | QoE/块 [95% CI] | 平均码率 Mbps | 卡顿秒/会话 | 切换幅度 Mbps/块 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {LABELS[row['method']]} | {row['episodes']} | {row['qoe_mean']:.4f} [{row['qoe_ci_low']:.4f}, {row['qoe_ci_high']:.4f}] | {row['mean_bitrate_mbps']:.3f} | {row['rebuffer_s']:.3f} | {row['smoothness_mbps']:.3f} |")
    if pairs:
        lines += ["", "## 同轨迹成对比较", "", "正差值表示 Kev 较好，负差值表示基线较好。", "",
                  "| 数据集 | 对照 | Kev − 基线 QoE/块 [95% CI] | Kev 胜出会话比例 |", "|---|---|---:|---:|"]
        for row in pairs:
            lines.append(f"| {row['dataset']} | {LABELS[row['baseline']]} | {row['kev_minus_baseline_qoe']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['kev_win_fraction']:.1%} |")
    kev_rows = [r for r in summary if r["method"] == "kev"]
    if kev_rows:
        lines += ["", "## 推理延迟", "", f"评测会话并发数：{manifest['config']['workers']}。客户端延迟包含服务排队；QoE 仿真未计入决策等待时间。", "",
                  "| 数据集 | 服务端平均 ms | 客户端平均 ms | 客户端 P50 ms | 客户端 P95 ms |", "|---|---:|---:|---:|---:|"]
        for row in kev_rows:
            lines.append(f"| {row['dataset']} | {row['server_latency_mean_ms']:.2f} | {row['client_latency_mean_ms']:.2f} | {row['client_latency_p50_ms']:.2f} | {row['client_latency_p95_ms']:.2f} |")
    lines += ["", "## 解释范围与复现", "",
              "跨数据集的绝对 QoE 同时受网络难度影响，应优先读每个数据集内的成对差值。每条轨迹只有一个固定起点，结果不代表所有起点。轨迹重叠可能降低 bootstrap 对真实环境不确定性的覆盖。", "",
              "Kev 输出概率未在 ABR 标签上校准；本轮不能据此宣称动作概率等于长期最优动作的可信概率。主结果不含推理时间对缓冲的影响。RobustMPC-6 使用已知视频未来五块大小，而 Kev 只读取下一块大小；完整信息预算见评测协议。", "",
              "数据源：[NetLLM](https://github.com/duowuyms/NetLLM)、[ABRBench](https://github.com/luopeng69131/ABRBench)。代码和数据提交、逐轨迹哈希、起点、提示词、模型信息以及 adapter/head 文件校验值见 `manifest.json`。", "",
              "- [QoE CDF](qoe_cdf.png)", "- [QoE 对照图](qoe_comparison.png)",
              "- [每会话指标](episode_metrics.csv)", "- [汇总指标](summary.csv)",
              "- [成对比较](paired_comparisons.csv)", "- [完整运行信息](manifest.json)", "",
              "`episodes/` 保留每一步的观察、动作、六档概率和延迟；`logs/` 保留 NetLLM 八列格式日志。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(out / "completion.json", {"complete": True, "episodes": len(results), "fingerprint": manifest["fingerprint"]})
    return summary
