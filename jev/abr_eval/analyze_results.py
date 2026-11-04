"""Explain completed QoE differences through quality, stalls and switching."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark import BITRATES, ROOT
from report import LABELS, write_csv


def analyze(out):
    out = Path(out)
    completion = json.loads((out / "completion.json").read_text())
    assert completion["complete"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert completion["fingerprint"] == manifest["fingerprint"]
    with (out / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    with (out / "paired_comparisons.csv").open(newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    datasets, methods = manifest["config"]["datasets"], manifest["config"]["methods"]
    episodes = [json.loads(p.read_text()) for p in (out / "episodes").glob("*/*/*.json")]
    fractions, gaps = [], []
    for dataset in datasets:
        for method in methods:
            group = [r for r in episodes if r["method"] == method and r["dataset"] == dataset]
            actions = [r["action"] for episode in group for r in episode["steps"][1:]]
            counts = np.bincount(actions, minlength=6)
            assert len(actions) == len(group) * 47
            fractions.append({"dataset": dataset, "method": method, "decisions": len(actions),
                              **{f"fraction_{int(bitrate)}": float(count / len(actions)) for bitrate, count in zip(BITRATES, counts)}})
        group = [r for r in summary if r["dataset"] == dataset]
        kev = next(r for r in group if r["method"] == "kev")
        best = max((r for r in group if r["method"] != "kev"), key=lambda r: float(r["qoe_mean"]))
        quality = float(kev["mean_bitrate_mbps"]) - float(best["mean_bitrate_mbps"])
        stall = -4.3 * (float(kev["rebuffer_s"]) - float(best["rebuffer_s"])) / 47
        switch = -(float(kev["smoothness_mbps"]) - float(best["smoothness_mbps"]))
        total = float(kev["qoe_mean"]) - float(best["qoe_mean"])
        assert np.isclose(quality + stall + switch, total, atol=1e-10)
        gaps.append({"dataset": dataset, "baseline": best["method"], "quality_contribution": quality,
                     "stall_contribution": stall, "switch_contribution": switch, "total_qoe_gap": total})
    write_csv(out / "action_fractions.csv", fractions)
    write_csv(out / "qoe_gap_decomposition.csv", gaps)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        rows = [next(r for r in fractions if r["dataset"] == dataset and r["method"] == method) for method in methods]
        bottom = np.zeros(len(methods))
        for bitrate, color in zip(BITRATES, plt.cm.viridis(np.linspace(.12, .9, 6))):
            values = np.array([r[f"fraction_{int(bitrate)}"] for r in rows])
            axis.bar(np.arange(len(methods)), values, bottom=bottom, color=color, label=f"{int(bitrate)} kbps")
            bottom += values
        axis.set(title=dataset.upper(), xticks=np.arange(len(methods)),
                 xticklabels=["BBA" if m == "bba" else "Throughput" if m == "throughput" else "MPC-6" if m == "robustmpc" else "Kev" for m in methods],
                 ylim=(0, 1), ylabel="Fraction of decisions")
        axis.tick_params(axis="x", rotation=25)
    axes[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"action_fractions.{extension}", bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(gaps))
    for offset, field, label, color in [(-.24, "quality_contribution", "Bitrate contribution", "#3574b4"),
                                        (0, "stall_contribution", "Rebuffer contribution", "#c84031"),
                                        (.24, "switch_contribution", "Switching contribution", "#319a88")]:
        axis.bar(x + offset, [r[field] for r in gaps], width=.24, label=label, color=color)
    axis.axhline(0, color="black", lw=.7)
    axis.set(xticks=x, xticklabels=[r["dataset"].upper() for r in gaps], ylabel="Kev - best baseline: QoE / chunk")
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=.18)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"qoe_gap_decomposition.{extension}", bbox_inches="tight")
    plt.close(fig)
    lines = ["# 零样本 Kev 的比较与解释", "", "本页解释已经完成的全量结果，不用于调整这轮提示词或模型。", ""]
    all_worse = all(
        float(next(r for r in summary if r["dataset"] == dataset and r["method"] == "kev")["qoe_mean"])
        < min(float(r["qoe_mean"]) for r in summary if r["dataset"] == dataset and r["method"] != "kev")
        for dataset in datasets
    )
    if all_worse:
        lines += ["**本轮固定提示和数值状态表示下，Kev 在所有测试集的平均 QoE 均低于全部传统对照。**"
                  "相对最强传统基线的主要负贡献来自卡顿，见下方精确分解。", ""]
    kev_actions = [r for r in fractions if r["method"] == "kev"]
    if all(r["fraction_300"] == 0 and r["fraction_750"] == 0 for r in kev_actions):
        lines += ["Kev 的所有已记录决策中，没有一次选择 300 或 750 kbps。这个动作分布说明"
                  "当前直接接入策略没有有效使用最低两档来控制卡顿风险。", "",
                  "| 数据集 | 2850 kbps 占比 | 4300 kbps 占比 |", "|---|---:|---:|"]
        for row in kev_actions:
            lines.append(f"| {row['dataset']} | {row['fraction_2850']:.1%} | {row['fraction_4300']:.1%} |")
        lines.append("")
    for dataset in datasets:
        group = sorted((r for r in summary if r["dataset"] == dataset), key=lambda r: float(r["qoe_mean"]), reverse=True)
        kev = next(r for r in group if r["method"] == "kev")
        rank = next(i for i, r in enumerate(group, 1) if r["method"] == "kev")
        best = next(r for r in group if r["method"] != "kev")
        pair = next(r for r in comparisons if r["dataset"] == dataset and r["baseline"] == best["method"])
        lines += [f"## {dataset.upper()}", "",
                  f"Kev 的 QoE/块为 **{float(kev['qoe_mean']):.4f}**，在本轮 {len(group)} 个方法中排名 {rank}。"
                  f"最强传统基线为 {LABELS[best['method']]}（{float(best['qoe_mean']):.4f}）；"
                  f"Kev 的成对差值为 {float(pair['kev_minus_baseline_qoe']):+.4f}，95% 区间 "
                  f"[{float(pair['ci_low']):+.4f}, {float(pair['ci_high']):+.4f}]，胜出会话比例 {float(pair['kev_win_fraction']):.1%}。", "",
                  f"Kev 平均码率 {float(kev['mean_bitrate_mbps']):.3f} Mbps，启动后卡顿 {float(kev['rebuffer_s']):.3f} 秒/会话，"
                  f"平均切换幅度 {float(kev['smoothness_mbps']):.3f} Mbps/块。", ""]
    lines += ["## QoE 差值来自哪里", "",
              "下面将 Kev 相对每组最强传统基线的 QoE 差值精确分解。正贡献帮助 Kev，负贡献削弱 Kev。"
              "卡顿贡献 = −4.3×会话卡顿差/47，切换贡献 = −平均切换幅度差；三项之和等于总差值。", "",
              "| 数据集 | 最强传统基线 | 码率贡献 | 卡顿贡献 | 切换贡献 | 总差值 |", "|---|---|---:|---:|---:|---:|"]
    for row in gaps:
        lines.append(f"| {row['dataset']} | {LABELS[row['baseline']]} | {row['quality_contribution']:+.4f} | {row['stall_contribution']:+.4f} | {row['switch_contribution']:+.4f} | {row['total_qoe_gap']:+.4f} |")
    lines += ["", "- [差值分解图](qoe_gap_decomposition.png)", "- [动作占比图](action_fractions.png)",
              "- [完整指标与置信区间](REPORT.md)", "",
              "这是固定提示词、固定起点下的零样本 ABR 表现，不能由单次评测断言微调后的上限。模型概率没有 ABR 标签校准，模型等待时间没有计入仿真 QoE。"]
    lines += ["", "延迟口径：服务端 `latency_ms` 是请求所在批次的模型总耗时（排队时间除外），客户端延迟包含排队和 HTTP 传输；本轮为四会话并发。"]
    (out / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gaps, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/zero_shot_v1")
    analyze(parser.parse_args().out)
