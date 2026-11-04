"""Run a local training probe or the two-epoch ABR adaptation job."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEV_ROOT = ROOT.parent / "kev"
STOCK_PYTHON = KEV_ROOT / ".venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--run-name")
    parser.add_argument("--detach", action="store_true", help="Keep training alive if the client disconnects")
    parser.add_argument("--stop-run", action="store_true", help="Stop only this directory's matching training process")
    args = parser.parse_args()
    assert args.batch in [1, 2, 4] and 16 % args.batch == 0
    deployment = json.loads((ROOT.parent / "deployment.json").read_text())
    name = args.run_name or ("probe_v1" if args.probe else "sft_v1")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise SystemExit("Run name must be a single directory name")
    out = ROOT / "runs" / name
    if args.stop_run:
        pid = int((ROOT / "logs" / f"{name}.pid").read_text())
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            print("Training already stopped")
            return
        if str(ROOT / "train_wrapper.py").encode() not in argv or str(out).encode() not in argv:
            raise SystemExit("PID does not belong to this training run; refusing to stop")
        os.kill(pid, signal.SIGTERM)
        print("Stopped training run", name, "PID", pid, flush=True)
        return
    if (out / "training_metrics.json").exists():
        print("Completed training already exists:", out)
        return
    if out.exists():
        raise SystemExit(f"Incomplete run already exists at {out}; preserve it and choose a new attempt name before restarting.")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if args.detach:
        argv = [sys.executable, str(Path(__file__).resolve()), "--batch", str(args.batch), "--run-name", name]
        if args.probe:
            argv.append("--probe")
        with (logs / f"{name}.driver.log").open("x") as stream:
            driver = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        (logs / f"{name}.driver.pid").write_text(str(driver.pid) + "\n")
        print(f"Detached training driver PID {driver.pid}; log {logs / (name + '.driver.log')}", flush=True)
        return
    versions = {package: importlib.metadata.version(package) for package in
                ["torch", "transformers", "peft", "triton", "flash-linear-attention", "fla-core", "einops"]}
    print("Training environment:", json.dumps(versions), flush=True)
    subprocess.run([str(STOCK_PYTHON), str(ROOT.parent / "kev_service.py"), "stop"], check=True)
    env = os.environ.copy()
    env.update(HF_HOME=str(ROOT.parent / ".cache/huggingface"), HF_HUB_OFFLINE="1",
               TRITON_CACHE_DIR=str(ROOT / ".cache/triton"), XDG_CACHE_HOME=str(ROOT / ".cache"),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, str(ROOT / "train_wrapper.py"),
               "--epoch-checkpoints", str(ROOT / "checkpoints" / name),
               "--base", deployment["base"], "--base_revision", deployment["base_revision"],
               "--init_from", deployment["checkpoint_path"],
               "--data", str(ROOT / "data" / ("probe.jsonl" if args.probe else "train.jsonl")),
               "--epochs", "1" if args.probe else "2", "--lr", "2e-5",
               "--lora", "16", "--head_dim", "256", "--batch", str(args.batch),
               "--accum", str(16 // args.batch), "--device", "cuda", "--dtype", "bf16",
               "--weights_dtype", "fp32", "--checkpointing", "1", "--max_state", "512",
               "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0",
               "--p_none_pair", "0", "--seed", "666", "--out", str(out)]
    if args.probe:
        command += ["--max_steps", "4"]
    (logs / f"{name}.command.json").write_text(json.dumps({"command": command, "versions": versions}, indent=2) + "\n")
    log_path = logs / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=KEV_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (logs / f"{name}.pid").write_text(str(process.pid) + "\n")
        print(f"Training PID {process.pid}; log {log_path}", flush=True)
        while process.poll() is None:
            time.sleep(20)
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print(" | ".join(lines[-2:])[-900:], flush=True)
    if process.returncode:
        print(log_path.read_text(encoding="utf-8", errors="replace")[-16000:], flush=True)
        raise SystemExit(process.returncode)
    metrics = json.loads((out / "training_metrics.json").read_text())
    assert metrics["records_seen"] > 0 and metrics["optimizer_steps"] > 0
    assert metrics["rejected_records"] == 0
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
