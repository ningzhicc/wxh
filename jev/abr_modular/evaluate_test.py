"""Evaluate one fixed Kev checkpoint on all four common_v1 test datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
JEV = ROOT.parent
PYTHON = JEV / "kev/.venv/bin/python"
DATASETS = ("fcc", "norway3g", "oboe", "hsr")
DEFAULT_CHECKPOINT = ROOT / "runs/merina656_baseline_seed666/checkpoints/epoch_08"
sys.path.insert(0, str(JEV / "abr_compare"))
from common import DEFAULT_BUNDLE, verify


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--candidate-mode", choices=("base", "cost"), help="Default: infer from the run's experiment.json, otherwise base")
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the command without starting a model or creating results")
    args = parser.parse_args()
    if args.startup_timeout < 1:
        parser.error("--startup-timeout must be positive")
    args.checkpoint = args.checkpoint.resolve()
    args.out = (args.out or ROOT / "results" / (
        f"test_{args.checkpoint.parent.parent.name}_{args.checkpoint.name}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}")).resolve()
    return args


def inspect_inputs(args):
    if not PYTHON.is_file():
        raise ValueError(f"Kev Python environment missing: {PYTHON}")
    for name in ("adapter_model.safetensors", "adapter_config.json", "head.pt"):
        if not (args.checkpoint / name).is_file():
            raise ValueError(f"Checkpoint file missing: {args.checkpoint / name}")
    declaration = args.checkpoint.parent.parent / "experiment.json"
    declared_mode = None
    if declaration.is_file():
        config = json.loads(declaration.read_text())
        declared_mode = "cost" if config["settings"]["candidate_cost"] else "base"
        from checkpoints import validate_checkpoint
        validate_checkpoint(args.checkpoint, config["fingerprint"])
    if args.candidate_mode and declared_mode and args.candidate_mode != declared_mode:
        raise ValueError("Candidate mode differs from this checkpoint's training inputs")
    mode = args.candidate_mode or declared_mode or "base"
    if args.out.exists():
        raise ValueError(f"Output already exists; choose a new --out: {args.out}")
    manifest = verify(DEFAULT_BUNDLE)
    records = [r for r in manifest["records"] if r["split"] == "test"]
    counts = Counter(r["dataset"] for r in records)
    sessions = Counter(r["dataset"] for r in records for _ in r["sessions"])
    if set(counts) != set(DATASETS):
        raise ValueError("The frozen test split must contain all four datasets")
    return mode, manifest, counts, sessions


def wait_for_model(checkpoint, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_error = "API not ready"
    while time.monotonic() < deadline:
        try:
            with opener.open("http://127.0.0.1:8009/v1/models", timeout=5) as response:
                cards = json.load(response)["models"]
            card = next(c for c in cards if c["name"] == "kev-latest")
            if Path(card["run"]).resolve() != checkpoint:
                raise ValueError(f"Service loaded a different checkpoint: {card['run']}")
            if not str(card.get("device", "")).startswith("cuda"):
                raise ValueError(f"Service is not using CUDA: {card.get('device')}")
            print(f"Model ready: {card['run']} ({card.get('dtype')}, {card['device']})", flush=True)
            return
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
            time.sleep(2)
    raise TimeoutError(f"Model startup timed out: {last_error}; inspect {JEV / 'runtime/kev.log'}")


def write_report(run_dir, output, expected_counts, expected_sessions):
    summary = json.loads((run_dir / "summary.json").read_text())
    if not summary["complete"] or not summary["full_split"]:
        raise ValueError("Evaluation is incomplete or did not cover the full test split")
    fields = ("mean_bitrate_mbps", "rebuffer_s_primary", "rebuffer_s_after_startup",
              "smoothness_mbps_primary", "decision_latency_mean_ms")
    # Match the frozen runner: average start points within each trace, then traces.
    traces = defaultdict(lambda: defaultdict(list))
    for path in (run_dir / "episodes").rglob("round_*.json"):
        ep = json.loads(path.read_text())
        if ep["split"] != "test" or ep["run_fingerprint"] != summary["run_fingerprint"]:
            raise ValueError(f"Episode does not belong to this test run: {path}")
        traces[ep["dataset"]][ep["record_id"]].append(ep["metrics"])
    rows = []
    for dataset in DATASETS:
        value = summary["results"][dataset]
        group = traces[dataset]
        if (len(group) != expected_counts[dataset] or
                sum(map(len, group.values())) != expected_sessions[dataset]):
            raise ValueError(f"Incomplete episode coverage for {dataset}")
        rows.append({"dataset": dataset, "traces": value["traces"], "sessions": value["sessions"],
                     "mean_qoe": value["mean_qoe"], "trace_sd_ddof0": value["trace_sd_ddof0"],
                     "mean_qoe_skip_first": value["mean_qoe_skip_first"],
                     **{field: statistics.mean(statistics.mean(m[field] for m in metrics)
                                              for metrics in group.values()) for field in fields}})
    with (output / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Four-dataset test evaluation", "",
             "One fixed checkpoint; all frozen common_v1 test traces and all three start points.",
             "Primary metrics exclude the first and last chunks. Startup stall is excluded.",
             "Average start points within each trace, then average traces. SD uses ddof=0.", "",
             "| Dataset | Traces | Sessions | Mean QoE | QoE SD | Bitrate (Mbps) | Stall (s/session) | Switch magnitude (Mbps) | Decision (ms) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['traces']} | {row['sessions']} | "
                     f"{row['mean_qoe']:.6f} | {row['trace_sd_ddof0']:.6f} | "
                     f"{row['mean_bitrate_mbps']:.6f} | {row['rebuffer_s_primary']:.6f} | "
                     f"{row['smoothness_mbps_primary']:.6f} | {row['decision_latency_mean_ms']:.3f} |")
    lines += ["", "This report does not certify independence of the Merina656 training pool from the test traces."]
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report)
    print(report, flush=True)


def main():
    args = parse_args()
    mode, manifest, counts, sessions = inspect_inputs(args)
    run_dir = args.out / "run"
    command = [str(PYTHON), str(JEV / "abr_candidate/run_compare.py"), "--candidate-mode", mode,
               "run", "--policy", "jev", "--bundle", str(DEFAULT_BUNDLE), "--split", "test",
               "--datasets", *DATASETS, "--action-mode", "argmax", "--base-url",
               "http://127.0.0.1:8009", "--out", str(run_dir)]
    print(f"Checkpoint: {args.checkpoint}\nCandidate mode: {mode}\nOutput: {args.out}", flush=True)
    for name in DATASETS:
        print(f"  {name}: {counts[name]} traces, {sessions[name]} sessions", flush=True)
    print(f"Total: {sum(sessions.values())} sessions\n{shlex.join(command)}", flush=True)
    if args.dry_run:
        print("Dry run complete: no service changes, model loading or evaluation.", flush=True)
        return
    # Share the trainer's lock so this package cannot start training during evaluation.
    with (ROOT / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another abr_modular training/evaluation job is running") from None
        args.out.mkdir(parents=True, exist_ok=False)
        plan = {"checkpoint": str(args.checkpoint), "candidate_mode": mode, "split": "test",
                "bundle_fingerprint": manifest["fingerprint"], "trace_counts": dict(counts),
                "session_counts": dict(sessions), "action_mode": "argmax", "command": command}
        (args.out / "evaluation.json").write_text(json.dumps(plan, indent=2) + "\n")
        manager = [str(PYTHON), str(JEV / "kev_service.py")]
        print("Switching this project's managed Kev service on port 8009.", flush=True)
        subprocess.run([*manager, "stop"], check=True)
        subprocess.run([*manager, "start", "--checkpoint", str(args.checkpoint)], check=True)
        try:
            wait_for_model(args.checkpoint, args.startup_timeout)
            with (args.out / "evaluation.log").open("x") as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait():
                        raise RuntimeError(f"Evaluation failed; see {args.out / 'evaluation.log'}")
            write_report(run_dir, args.out, counts, sessions)
            print(f"Evaluation complete. Results: {args.out}", flush=True)
        finally:
            # Release GPU memory, including when evaluation or service startup fails.
            subprocess.run([*manager, "stop"], check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
