"""Wait for training, select on validation, evaluate frozen tests and export the comparison."""
import argparse
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sft_v3")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    name = args.run_name
    assert Path(name).name == name and name not in {".", ".."}
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        print("Completed pipeline:", out, flush=True)
        return
    if args.detach:
        with (ROOT / "logs" / f"{name}.pipeline.log").open("a") as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--run-name", name],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / "logs" / f"{name}.pipeline.pid").write_text(str(process.pid) + "\n")
        print("Detached evaluation pipeline PID", process.pid, flush=True)
        return
    import fcntl
    with (out / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This evaluation pipeline is already running")
        execute(name, out)


def execute(name, out):
    deployment = json.loads((ROOT.parent / "deployment.json").read_text())
    original = Path(deployment["checkpoint_path"])
    metrics_path = ROOT / "runs" / name / "training_metrics.json"
    atomic_json(out / "status.json", {"stage": "waiting_for_training", "run": name})
    print("Waiting for training:", metrics_path, flush=True)
    while not metrics_path.exists():
        pidfile = ROOT / "logs" / f"{name}.pid"
        if pidfile.exists():
            pid = int(pidfile.read_text())
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            except FileNotFoundError:
                cmdline = b""
            if b"train_wrapper.py" not in cmdline:
                atomic_json(out / "status.json", {"stage": "training_failed", "run": name})
                raise SystemExit("Training exited without final metrics; preserve its log and inspect")
        time.sleep(20)
    # Training writes its metrics last, then the process releases GPU memory.
    pid = int((ROOT / "logs" / f"{name}.pid").read_text())
    for _ in range(60):
        try:
            if b"train_wrapper.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                break
        except FileNotFoundError:
            break
        time.sleep(1)
    else:
        raise SystemExit("Training did not release its process after final metrics")
    training = json.loads(metrics_path.read_text())
    data = json.loads((ROOT / "data/manifest.json").read_text())
    records = data["counts"]["selected_train_records"]
    assert training["records_seen"] == training["requested_records"] == 2 * records
    assert training["optimizer_steps"] == 2 * math.ceil(records / 16) and training["rejected_records"] == 0
    assert digest(original / "adapter_model.safetensors") == deployment["adapter_sha256"]
    from kev.checkpoint import Checkpoint
    import torch
    initial = Checkpoint(str(original))
    initial_source = json.loads((ROOT / "runs" / name / "training_config.json").read_text())["init_source"]
    assert digest(original / "head.pt") == initial_source["head_sha256"]
    candidates = []
    model_audit = []
    for epoch in [1, 2]:
        checkpoint = ROOT / "checkpoints" / name / f"epoch_{epoch:02d}"
        meta = Checkpoint(str(checkpoint)).meta
        assert meta.base == initial.meta.base and meta.base_revision == deployment["base_revision"]
        assert meta.extra["data_sha256"] == digest(ROOT / "data/train.jsonl")
        changed = [key for key, value in meta.head.items() if not torch.equal(value, initial.meta.head[key])]
        assert changed and digest(checkpoint / "adapter_model.safetensors") != deployment["adapter_sha256"]
        model_audit.append({"checkpoint": str(checkpoint), "head_tensors_changed": changed,
                            "adapter_sha256": digest(checkpoint / "adapter_model.safetensors"),
                            "head_sha256": digest(checkpoint / "head.pt")})
        candidates.append({"name": f"epoch_{epoch:02d}", "checkpoint": str(checkpoint)})
    atomic_json(out / "training_audit.json", {"passed": True, "original_model_unchanged": True,
                "base_revision": deployment["base_revision"], "training_metrics": training,
                "candidates": model_audit, "init_source": initial_source,
                "data_manifest_sha256": digest(ROOT / "data/manifest.json")})
    def evaluate(checkpoint, scope, destination):
        atomic_json(out / "status.json", {"stage": scope, "checkpoint": str(checkpoint), "output": str(destination)})
        subprocess.run([sys.executable, str(ROOT / "evaluate_model.py"), "--checkpoint", str(checkpoint),
                        "--scope", scope, "--out", str(destination)], check=True)
        return json.loads((destination / "summary.json").read_text())[0]
    if (out / "selected.json").exists():
        selection = json.loads((out / "selected.json").read_text())
        assert [c["checkpoint"] for c in selection["candidates"]] == [c["checkpoint"] for c in candidates]
    else:
        original_score = evaluate(original, "selection", out / "selection_original")["qoe_mean"]
        for candidate in candidates:
            candidate["qoe_mean"] = evaluate(candidate["checkpoint"], "selection", out / ("selection_" + candidate["name"]))["qoe_mean"]
        winner = max(candidates, key=lambda row: row["qoe_mean"])
        selection = {"checkpoint": winner["checkpoint"], "name": winner["name"],
                     "rule": "Highest mean closed-loop QoE on fixed clean FCC-valid selection traces; ties prefer earlier epoch",
                     "candidates": candidates, "original_validation_qoe": original_score,
                     "uses_test_for_selection": False}
        atomic_json(out / "selected.json", selection)
    print("Selected checkpoint:", json.dumps(selection), flush=True)
    evaluate(selection["checkpoint"], "valid", out / "valid")
    evaluate(selection["checkpoint"], "test", out / "test")
    atomic_json(out / "status.json", {"stage": "audit_and_report"})
    subprocess.run([sys.executable, str(ROOT.parent / "abr_eval/audit_results.py"), "--out", str(out / "test")], check=True)
    subprocess.run([sys.executable, str(ROOT / "compare_results.py"), "--run-name", name], check=True)
    atomic_json(out / "completion.json", {"complete": True, "selected_checkpoint": selection["checkpoint"],
                "test_episodes": 275, "validation_episodes": data["counts"]["valid_traces"], "comparison_report": str(out / "REPORT.md")})
    atomic_json(out / "status.json", {"stage": "complete", "selected_checkpoint": selection["checkpoint"]})
    print("Training, validation, frozen test comparison complete:", out, flush=True)


if __name__ == "__main__":
    main()
