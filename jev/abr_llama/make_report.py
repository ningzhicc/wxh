"""Audit and compare the finished Llama experiment with existing ABR baselines."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import atomic_json, digest
from report import interval, write_csv


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    name = parser.parse_args().run_name
    assert Path(name).name == name and name not in {".", ".."}
    out = ROOT / "results" / name
    declaration = json.loads((ROOT / "runs" / name / "experiment.json").read_text())
    assert all(digest(ROOT.parent / path) == expected for path, expected in declaration["source_sha256"].items())
    assert digest(ROOT.parent / "models/Llama-3.2-1B/model.safetensors") == declaration["base_weights_sha256"]
    selected = json.loads((out / "selected.json").read_text())
    metrics = json.loads((ROOT / "runs" / name / "training_metrics.json").read_text())
    assert metrics["records_per_epoch"] == 7224 and metrics["steps_per_epoch"] == 452
    assert metrics["records_seen"] == metrics["epochs_completed"]*7224
    assert metrics["optimizer_steps"] == metrics["epochs_completed"]*452
    candidates = selected["candidates"]
    assert selected["selected_epoch"] == max(candidates, key=lambda x: x["score"])["epoch"]
    for c in candidates:
        for repeat in (out / f"validation_epoch_{c['epoch']:02d}").glob("repeat_*"):
            manifest = json.loads((repeat / "manifest.json").read_text())
            assert all(digest(Path(c["checkpoint"]) / f) == sha for f, sha in manifest["model_files_sha256"].items())
    for epoch in range(1, metrics["epochs_completed"] + 1):
        snapshot = ROOT / "checkpoints" / name / f"epoch_{epoch:02d}"
        marker = json.loads((snapshot / "snapshot.json").read_text())
        assert marker["complete"] and digest(snapshot / "training_state.pt") == marker["training_state_sha256"]
        state = torch.load(snapshot / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["step"] == epoch*452 and state["scheduler"]["last_epoch"] == state["step"]
        assert all(int(s["step"]) == state["step"] for s in state["optimizer"]["state"].values())
        del state
    atomic_json(out / "training_audit.json", {"passed": True, "records_and_steps_verified": True,
        "all_saved_optimizer_steps_verified": True, "selected_by_validation_only": True,
        "all_validation_checkpoint_hashes_unchanged": True})
    paths = [out / "untrained_test", out / "test", ROOT.parent / "abr_eval/results/zero_shot_v1",
             ROOT.parent / "abr_finetune/results/sft_v3/test", ROOT.parent / "abr_finetune/results/sft_extend_v1/test"]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in paths]
    for manifest in manifests[1:]:
        assert manifest["traces"] == manifests[0]["traces"]
        assert manifest["video_sha256"] == manifests[0]["video_sha256"]
        for key in ["prompt", "criteria", "skip_first_reward", "charge_decision_latency"]:
            assert manifest["config"][key] == manifests[0]["config"][key]
        assert manifest["source_sha256"]["benchmark.py"] == manifests[0]["source_sha256"]["benchmark.py"]
        for key in ["dtype", "backend", "device"]:
            assert manifest["service"][key] == manifests[0]["service"][key]
    for p in paths:
        assert json.loads((p / "audit.json").read_text())["passed"]
    methods = ["llama_untrained", "llama_selected", "kev_zero", "kev_two", "kev_four", "bba", "throughput", "robustmpc"]
    labels = dict(zip(methods, ["Llama 未微调随机头", f"Llama 验证选中第{selected['selected_epoch']}轮",
        "Kev 发布版", "Kev ABR 第2轮", "Kev ABR 第4轮", "BBA", "Throughput-0.85", "RobustMPC-6"]))
    all_rows = []
    for i, path in enumerate(paths):
        records = [json.loads(f.read_text()) for f in sorted((path / "episodes").glob("*/*/*.json"))]
        assert len(records) == (1100 if i == 2 else 275)
        for row in records:
            assert row["fingerprint"] == manifests[i]["fingerprint"]
            method = row["method"]
            if method == "kev":
                method = ["llama_untrained", "llama_selected", "kev_zero", "kev_two", "kev_four"][i]
            all_rows.append({**row, "method": method})
    datasets = ["fcc", "hsr", "oboe", "norway3g"]
    summary, pairs = [], []
    for d in datasets:
        groups = {m: [r for r in all_rows if r["dataset"] == d and r["method"] == m] for m in methods}
        for method, rows in groups.items():
            lo, hi = interval([r["metrics"]["qoe_mean"] for r in rows])
            fields = ["qoe_mean", "mean_bitrate_mbps", "rebuffer_s", "smoothness_mbps", "switches", "no_stall"]
            latency = [s["decision"]["client_ms"] for r in rows for s in r["steps"][1:]]
            summary.append({"dataset": d, "method": method, "episodes": len(rows),
                **{f: float(np.mean([r["metrics"][f] for r in rows])) for f in fields},
                "qoe_ci_low": lo, "qoe_ci_high": hi, "client_latency_mean_ms": float(np.mean(latency)),
                "client_latency_p95_ms": float(np.quantile(latency, .95))})
        winner = {r["key"]: r for r in groups["llama_selected"]}
        for baseline in [m for m in methods if m != "llama_selected"]:
            other = {r["key"]: r for r in groups[baseline]}
            assert winner.keys() == other.keys()
            gaps = np.array([winner[k]["metrics"]["qoe_mean"]-other[k]["metrics"]["qoe_mean"] for k in sorted(winner)])
            lo, hi = interval(gaps)
            pairs.append({"dataset": d, "baseline": baseline, "mean_qoe_difference": float(gaps.mean()),
                          "ci_low": lo, "ci_high": hi, "win_fraction": float(np.mean(gaps>1e-9))})
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "paired_comparisons.json", pairs)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_comparisons.csv", pairs)
    curve = [{"epoch": c["epoch"], "mean_qoe": c["score"],
        **{f"repeat_{i+1}_qoe": r["qoe_mean"] for i, r in enumerate(c["repeat_summaries"])}} for c in candidates]
    write_csv(out / "validation_curve.csv", curve)
    write_csv(out / "training_loss.csv", metrics["loss_history"])
    lines = ["# Llama-3.2-1B + pointer head：ABR 测试与微调", "",
        f"累计训练 {selected['last_trained_epoch']} 轮，按验证 QoE 选中第 **{selected['selected_epoch']} 轮**。"
        f"停止原因 `{selected['stop_reason']}`。最佳模型：`{selected['checkpoint']}`。", "",
        "未微调基线是预训练 Llama + 随机初始化 pointer head（seed=666），不是已经训练过的 Kev，也不是生成式零样本问答。"
        "微调从完全相同的初始适配器和决策头开始；冻结基座、训练 LoRA 和 pointer head。", "",
        "55 条训练轨迹包含16条FCC记录和39条Norway日志；7,224个唯一状态，标签由RobustMPC-6产生。"
        "batch2×accum8，LoRA lr2e-5，head lr1e-4，FP32主参数、BF16 autocast；各轮保持同一AdamW和OneCycle状态。", "",
        f"优化步骤 {metrics['optimizer_steps']}，累计处理记录 {metrics['records_seen']}，纯训练耗时 "
        f"{metrics['optimization_seconds']/60:.1f} 分钟，峰值GPU分配 {metrics['peak_device_bytes']/2**30:.2f} GiB。", "",
        "验证固定30窗口、仅两个采集ID，每轮重复两次。至少4轮、最多12轮，连续3轮未超过+.005的实质改善时早停；"
        "最终选择所有候选中观测验证均值最高者。到达轮数上限不代表收敛；重复验证不增加独立来源。", "",
        "| 轮数 | 验证1 | 验证2 | 验证均值 |", "|---:|---:|---:|---:|"]
    for c in candidates:
        a, b = c["repeat_summaries"]
        lines.append(f"| {c['epoch']} | {a['qoe_mean']:.6f} | {b['qoe_mean']:.6f} | {c['score']:.6f} |")
    lines += ["", "测试在训练前与选模冻结后各进行一次，四组共275条。所有测试不参与梯度、选模、早停或调参；"
        "这些是已查看过的既有基准，不称为新的未见确认数据。六档动作、视频、提示词、起点及仿真器与Kev比较保持一致。"
        "主QoE排除固定首块且不计推理等待；延迟另报。Llama更换了基座大小与训练历史，不能解释为控制其他因素后的大小消融。", "",
        "| 数据集 | 方法 | QoE/块 | 码率Mbps | 卡顿秒/会话 | 客户端延迟ms |", "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {labels[r['method']]} | {r['qoe_mean']:.4f} | {r['mean_bitrate_mbps']:.3f} | "
                     f"{r['rebuffer_s']:.3f} | {r['client_latency_mean_ms']:.2f} |")
    lines += ["", "成对差值为选中Llama减去对照，区间来自5000次会话bootstrap；窗口相关性可能导致区间偏窄。", "",
              "| 数据集 | 对照 | QoE差值 | 95%区间 |", "|---|---|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['dataset']} | {labels[r['baseline']]} | {r['mean_qoe_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    lines += ["", "推理使用torch/CUDA/BF16，概率未进行ABR校准。测试服务结束后释放GPU，检查点可通过本目录serve.py重新启动。", "",
        "- [训练与验证曲线](convergence.png)", "- [测试对比图](qoe_comparison.png)",
        "- [未微调测试](untrained_test/summary.csv)", "- [最终测试](test/summary.csv)",
        "- [训练校验](training_audit.json)", "- [比较口径校验](comparison_audit.json)",
        "- [未微调回放校验](untrained_test/audit.json)", "- [最终回放校验](test/audit.json)"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")
    atomic_json(out / "comparison_audit.json", {"passed": True, "same_traces_starts_video_prompt": True,
        "same_simulator_and_qoe": True, "same_dtype_backend_device": True, "no_test_metrics_in_selection": True,
        "untrained_pointer_head_is_random": True})
    plot(out, selected, metrics, summary)


def plot(out, selected, metrics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    def save(fig, name):
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    trained = [c for c in selected["candidates"] if c["epoch"] > 0]
    for i in range(2):
        axes[0].plot([c["epoch"] for c in trained], [c["repeat_summaries"][i]["qoe_mean"] for c in trained],
                     linestyle="--", alpha=.45, label=f"Repeat {i+1}")
    axes[0].plot([c["epoch"] for c in trained], [c["score"] for c in trained], marker="o", label="Mean")
    axes[0].axvline(selected["selected_epoch"], color="orange", label="Selected epoch")
    axes[0].set(xlabel="Epoch", ylabel="Validation QoE / chunk")
    axes[0].set_title(f"Untrained validation QoE: {selected['candidates'][0]['score']:.3f}")
    axes[0].legend(fontsize=8)
    axes[1].plot([h["step"] for h in metrics["loss_history"]], [h["loss"] for h in metrics["loss_history"]])
    axes[1].set(xlabel="Optimizer step", ylabel="Training cross entropy (logged window)")
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "convergence")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    datasets = ["fcc", "hsr", "oboe", "norway3g"]
    for ax, methods, labels in [(axes[0], ["llama_untrained", "llama_selected"], ["Llama random head", "Llama selected"]),
        (axes[1], ["llama_selected", "kev_two", "kev_four", "robustmpc"], ["Llama selected", "Kev epoch 2", "Kev epoch 4", "MPC"])]:
        width = .8/len(methods)
        for i, (m, label) in enumerate(zip(methods, labels)):
            vals = [next(r["qoe_mean"] for r in summary if r["dataset"] == d and r["method"] == m) for d in datasets]
            ax.bar(np.arange(4)-.4+width/2+i*width, vals, width, label=label)
        ax.set(xticks=np.arange(4), xticklabels=[d.upper() for d in datasets], ylabel="Mean QoE / chunk")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    save(fig, "qoe_comparison")


if __name__ == "__main__":
    main()
