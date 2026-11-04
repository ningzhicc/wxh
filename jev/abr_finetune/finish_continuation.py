"""Freeze the validation winner, evaluate original tests and compare with two-epoch SFT."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import atomic_json, digest
from report import interval, write_csv


def generate(name):
    out = ROOT / "results" / name
    selected = json.loads((out / "selected.json").read_text())
    metrics = json.loads((ROOT / "runs" / name / "training_metrics.json").read_text())
    initial = ROOT.parent / "abr_eval/results/zero_shot_v1"
    previous = ROOT / "results/sft_v3/test"
    current = out / "test"
    paths = [initial, previous, current]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in paths]
    assert manifests[0]["traces"] == manifests[1]["traces"] == manifests[2]["traces"]
    for key in ["prompt", "criteria", "skip_first_reward", "charge_decision_latency"]:
        assert all(m["config"][key] == manifests[0]["config"][key] for m in manifests)
    assert all(m["video_sha256"] == manifests[0]["video_sha256"] for m in manifests)
    assert all(m["source_sha256"]["benchmark.py"] == manifests[0]["source_sha256"]["benchmark.py"] for m in manifests)
    for key in ["backend", "dtype", "device"]:
        assert all(m["service"][key] == manifests[0]["service"][key] for m in manifests)
    assert json.loads((current / "audit.json").read_text())["passed"]
    results = []
    for index, path in enumerate(paths):
        group = [json.loads(p.read_text()) for p in sorted((path / "episodes").glob("*/*/*.json"))]
        assert len(group) == (1100 if index == 0 else 275)
        assert all(r["fingerprint"] == manifests[index]["fingerprint"] for r in group)
        for row in group:
            method = row["method"]
            if method == "kev":
                method = ["kev_zero", "kev_two_epochs", "kev_selected"][index]
            results.append({**row, "method": method})
    methods = ["kev_zero", "kev_two_epochs", "kev_selected", "bba", "throughput", "robustmpc"]
    datasets = ["fcc", "hsr", "oboe", "norway3g"]
    labels = {"kev_zero": "Kev 发布版", "kev_two_epochs": "Kev 原两轮", "kev_selected": f"Kev 选中累计第 {selected['selected_epoch']} 轮",
              "bba": "BBA", "throughput": "Throughput-0.85", "robustmpc": "RobustMPC-6"}
    summary, pairs = [], []
    groups = {(d, m): [r for r in results if r["dataset"] == d and r["method"] == m] for d in datasets for m in methods}
    for dataset in datasets:
        for method in methods:
            group = groups[dataset, method]
            low, high = interval([r["metrics"]["qoe_mean"] for r in group])
            fields = ["qoe_mean", "mean_bitrate_mbps", "rebuffer_s", "smoothness_mbps", "switches"]
            summary.append({"dataset": dataset, "method": method, "episodes": len(group),
                **{f: float(np.mean([r["metrics"][f] for r in group])) for f in fields},
                "qoe_ci_low": low, "qoe_ci_high": high})
        winner = {r["key"]: r for r in groups[dataset, "kev_selected"]}
        for method in ["kev_two_epochs", "kev_zero", "bba", "throughput", "robustmpc"]:
            baseline = {r["key"]: r for r in groups[dataset, method]}
            assert winner.keys() == baseline.keys()
            gap = np.array([winner[k]["metrics"]["qoe_mean"]-baseline[k]["metrics"]["qoe_mean"] for k in sorted(winner)])
            low, high = interval(gap)
            pairs.append({"dataset": dataset, "baseline": method, "mean_qoe_difference": float(gap.mean()),
                          "ci_low": low, "ci_high": high, "win_fraction": float(np.mean(gap>1e-9))})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_comparisons.csv", pairs)
    write_csv(out / "validation_curve.csv", [{"epoch": c["epoch"], "mean_qoe": c["score"],
               "repeat_1_qoe": c["repeat_summaries"][0]["qoe_mean"],
               "repeat_2_qoe": c["repeat_summaries"][1]["qoe_mean"]} for c in selected["candidates"]])
    write_csv(out / "training_loss.csv", metrics["loss_history"])
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "paired_comparisons.json", pairs)
    lines = ["# Kev ABR 延长训练、逐轮选模与早停", "",
             f"从原第二轮权重继续训练 {metrics['epochs_completed']} 轮，累计训练至第 {selected['last_trained_epoch']} 轮。"
             f"按重复验证的平均 QoE 选中了累计第 **{selected['selected_epoch']} 轮**，服务回切到这份检查点。",
             f"停止原因：`{selected['stop_reason']}`。最佳检查点：`{selected['checkpoint']}`。", "",
             "初步预估累计六轮左右可能进入平台，最多十轮；阈值 +0.005，连续三轮无实质改善且累计至少六轮时早停。"
             "严格保留观测到的验证最高分，最后训练轮可以晚于最佳轮。", "",
             "延长阶段使用相同 55 条轨迹、7,224 个状态，六档动作、提示词、标签和视频均保持原样。"
             "由于上一阶段未保存优化器，新阶段开始重建一次 AdamW/OneCycle，峰值学习率 1e-5；随后每轮之间保持优化器、"
             "调度器、随机状态和数据顺序，检查点附带训练状态。这是两阶段权重继续训练，不是对旧优化器的精确恢复。",
             f"本阶段 {metrics['optimizer_steps']} 个优化步骤，处理 {metrics['records_seen']} 条记录，"
             f"实际优化耗时 {metrics['wall_seconds']/60:.1f} 分钟（不含逐轮验证）。", "",
             "每个候选在固定 30 条验证窗口上评估两次，以两次平均 QoE 的均值选模。验证窗口仅来自两个采集 ID，"
             "重复评估用于减弱数值波动，不增加独立数据来源。选择分数是有限候选中的观测最好值，不证明全局最优或广泛泛化。", "",
             "| 累计轮数 | 第一次验证 QoE | 第二次验证 QoE | 平均 QoE |", "|---:|---:|---:|---:|"]
    for candidate in selected["candidates"]:
        repeats = candidate["repeat_summaries"]
        lines.append(f"| {candidate['epoch']} | {repeats[0]['qoe_mean']:.6f} | {repeats[1]['qoe_mean']:.6f} | {candidate['score']:.6f} |")
    lines += ["", "四组测试不用于梯度、轮数、学习率或早停。新检查点在选模冻结后评测；若选中原第二轮，"
              "则引用其既有已校验测试结果。它们是已查看过的同一批测试集，"
              "本报告属于后续比较，不将它们称为新的未见确认数据。QoE 不计推理等待，排除固定起始视频块。", "",
              "| 数据集 | 方法 | 平均 QoE/块 | 平均码率 Mbps | 卡顿秒/会话 |", "|---|---|---:|---:|---:|"]
    if (out / "test_reuse.json").exists():
        lines[-3:-3] = ["最终选中的仍是原第二轮，且模型文件校验值完全一致，因此四组测试直接引用其既有已校验结果，"
                         "没有重复推理；两轮模型与本轮选中模型的测试指标相同。", ""]
    for row in summary:
        lines.append(f"| {row['dataset']} | {labels[row['method']]} | {row['qoe_mean']:.4f} | {row['mean_bitrate_mbps']:.3f} | {row['rebuffer_s']:.3f} |")
    lines += ["", "同轨迹成对差值：选中模型减去对照。区间为 5,000 次会话 bootstrap，相关窗口可能使区间过窄。", "",
              "| 数据集 | 对照 | 平均 QoE 差值 [95% CI] | 胜出比例 |", "|---|---|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['dataset']} | {labels[row['baseline']]} | {row['mean_qoe_difference']:+.4f} "
                     f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['win_fraction']:.1%} |")
    lines += ["", "训练目录包含 16 条 FCC 风格原始记录和 39 条 Norway 日志；Norway 测试不能解释为从未接触该采集项目。"
              "源标识由文件名推断，数值与非恒定片段另作去重检查。输出概率没有进行 ABR 校准。", "",
              "- [验证与训练曲线](convergence.png)", "- [QoE 对比图](qoe_comparison.png)",
              "- [每轮验证](validation_curve.csv)", "- [最终选择](selected.json)", "- [成对比较](paired_comparisons.csv)",
              "- [测试回放校验](test/audit.json)", "- [训练来源与数据校验](training_audit.json)", "",
              "逐步动作、状态、概率及原始八列日志保存在 test/episodes/ 与 test/logs/。旧两轮模型及报告完整保留。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot(out, selected, metrics, summary)
    atomic_json(out / "comparison_audit.json", {"passed": True, "same_traces_starts_video_prompt": True,
        "same_benchmark_and_serving_environment": True, "no_test_metrics_in_selection": True,
        "test_results_reused": (out / "test_reuse.json").exists(),
        "fingerprints": [m["fingerprint"] for m in manifests], "source_sha256": digest(__file__)})


def plot(out, selected, metrics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    def save(fig, name):
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    candidates = selected["candidates"]
    for repeat in range(2):
        axes[0].plot([c["epoch"] for c in candidates], [c["repeat_summaries"][repeat]["qoe_mean"] for c in candidates],
                     alpha=.4, linestyle="--", label=f"Repeat {repeat+1}")
    axes[0].plot([c["epoch"] for c in candidates], [c["score"] for c in candidates], marker="o", label="Mean validation QoE")
    axes[0].axvline(selected["selected_epoch"], color="orange", lw=1, label="Selected epoch")
    axes[0].set(xlabel="Cumulative training epoch", ylabel="Validation mean QoE / chunk")
    axes[0].legend(fontsize=8)
    history = metrics["loss_history"]
    axes[1].plot([h["phase_step"] for h in history], [h["loss"] for h in history])
    axes[1].set(xlabel="Optimizer step in continuation phase", ylabel="Training cross entropy (logged window)")
    for axis in axes:
        axis.grid(alpha=.2)
    save(fig, "convergence")
    datasets = ["fcc", "hsr", "oboe", "norway3g"]
    methods = ["kev_two_epochs", "kev_selected", "bba", "throughput", "robustmpc"]
    labels = ["Kev two epochs", f"Kev selected epoch {selected['selected_epoch']}", "BBA", "Throughput", "RobustMPC-6"]
    fig, axis = plt.subplots(figsize=(10, 4))
    width = .8/len(methods)
    for index, (method, label) in enumerate(zip(methods, labels)):
        values = [next(r for r in summary if r["dataset"] == d and r["method"] == method)["qoe_mean"] for d in datasets]
        axis.bar(np.arange(4)-.4+width/2+index*width, values, width, label=label)
    axis.set(xticks=np.arange(4), xticklabels=[d.upper() for d in datasets], ylabel="Mean QoE / chunk")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=.2)
    save(fig, "qoe_comparison")


def main():
    import subprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sft_extend_v1")
    args = parser.parse_args()
    name = args.run_name
    assert Path(name).name == name and name not in {".", ".."}
    out = ROOT / "results" / name
    if (out / "completion.json").exists():
        print("Completed continuation evaluation:", out)
        return
    selected = json.loads((out / "selected.json").read_text())
    metrics = json.loads((ROOT / "runs" / name / "training_metrics.json").read_text())
    assert not selected["uses_test_for_selection"]
    assert metrics["records_seen"] == metrics["epochs_completed"]*7224
    assert metrics["optimizer_steps"] == metrics["epochs_completed"]*452
    candidate = next(c for c in selected["candidates"] if c["epoch"] == selected["selected_epoch"])
    assert selected["selected_validation_qoe"] == max(c["score"] for c in selected["candidates"])
    for row in selected["candidates"]:
        checkpoint = Path(row["checkpoint"])
        for repeat in [1, 2]:
            manifest = json.loads((out / f"validation_epoch_{row['epoch']:02d}" / f"repeat_{repeat:02d}/manifest.json").read_text())
            assert all(digest(checkpoint / f) == sha for f, sha in manifest["model_files_sha256"].items())
    subprocess.run([sys.executable, str(ROOT / "audit_data.py")], check=True)
    previous_test = ROOT / "results/sft_v3/test"
    reuse_test = Path(selected["checkpoint"]).resolve() == (ROOT / "checkpoints/sft_v3/epoch_02").resolve()
    for scope in ["valid", "test"]:
        atomic_json(out / "status.json", {"stage": scope, "selected_epoch": selected["selected_epoch"]})
        if scope == "test" and reuse_test:
            old_manifest = json.loads((previous_test / "manifest.json").read_text())
            assert json.loads((previous_test / "audit.json").read_text())["passed"]
            assert all(digest(Path(selected["checkpoint"]) / f) == sha for f, sha in old_manifest["model_files_sha256"].items())
            if not (out / "test").exists():
                (out / "test").symlink_to(previous_test.resolve(), target_is_directory=True)
            assert (out / "test").resolve() == previous_test.resolve()
            atomic_json(out / "test_reuse.json", {"reused": True, "reason": "Identical selected checkpoint to the original two-epoch model",
                "source": str(previous_test), "fingerprint": old_manifest["fingerprint"], "model_hashes_unchanged": True})
            continue
        subprocess.run([sys.executable, str(ROOT / "evaluate_model.py"), "--checkpoint", selected["checkpoint"],
                        "--scope", scope, "--out", str(out / scope)], check=True)
    if not reuse_test:
        subprocess.run([sys.executable, str(ROOT.parent / "abr_eval/audit_results.py"), "--out", str(out / "test")], check=True)
    atomic_json(out / "training_audit.json", {"passed": True, "epochs_completed": metrics["epochs_completed"],
        "cumulative_epochs_completed": metrics["cumulative_epochs_completed"], "record_count_verified": True,
        "optimizer_steps_verified": True, "all_validation_checkpoint_hashes_unchanged": True,
        "selected_by_validation_only": True, "selected_epoch": selected["selected_epoch"],
        "data_audit": json.loads((ROOT / "data_audit.json").read_text()),
        "continuation_config": json.loads((ROOT / "runs" / name / "continuation_config.json").read_text())})
    generate(name)
    import httpx
    manager = ROOT.parent / "kev_service.py"
    checkpoint_path = Path(selected["checkpoint"]).resolve()
    with httpx.Client(trust_env=False, timeout=5) as client:
        try:
            response = client.get("http://127.0.0.1:8009/v1/models")
            response.raise_for_status()
            matches = Path(response.json()["models"][0]["run"]).resolve() == checkpoint_path
        except httpx.HTTPError:
            matches = False
        if not matches:
            subprocess.run([sys.executable, str(manager), "stop"], check=True)
            subprocess.run([sys.executable, str(manager), "start", "--checkpoint", str(checkpoint_path)], check=True)
        deadline = time.monotonic() + 180
        while True:
            try:
                response = client.get("http://127.0.0.1:8009/v1/models")
                response.raise_for_status()
                card = response.json()["models"][0]
                assert Path(card["run"]).resolve() == checkpoint_path
                atomic_json(out / "service_verification.json", {"passed": True, "model": card})
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise SystemExit("Selected service failed to become ready")
                time.sleep(5)
    atomic_json(out / "completion.json", {"complete": True, "selected_checkpoint": selected["checkpoint"],
        "selected_epoch": selected["selected_epoch"], "last_trained_epoch": selected["last_trained_epoch"],
        "stop_reason": selected["stop_reason"], "test_episodes": 275, "test_results_reused": reuse_test})
    atomic_json(out / "status.json", {"stage": "complete", "selected_epoch": selected["selected_epoch"],
                                     "last_trained_epoch": selected["last_trained_epoch"]})
    print("Continuation, early stopping and full test evaluation complete:", out, flush=True)


if __name__ == "__main__":
    main()
