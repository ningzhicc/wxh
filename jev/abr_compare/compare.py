#!/usr/bin/env python3
"""Freeze/verify a common ABR bundle and evaluate policies on that exact bundle."""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from common import (ROOT, DEFAULT_BUNDLE, DEFAULT_DYLAS, DEFAULT_JEV, canonical, fingerprint,
                    prepare, verify, environment, episode_metrics, write_json)
from policies import Traditional, JevHTTP, DyLasProcess


def run_episode(bundle, record, session, policy, settings):
    sim = environment(bundle, record, session)
    policy.reset(session["action_seed"])
    rows, action, previous, decision_ms, detail = [], settings["initial_action"], settings["initial_action"], 0.0, {}
    elapsed = 0.0
    for index in range(settings["chunks"]):
        before = float(sim.buffer_size)
        delay, returned_sleep, buffer, rebuffer, size, next_sizes, end, remaining = sim.get_video_chunk(action)
        switch = abs(settings["bitrates_kbps"][action] - settings["bitrates_kbps"][previous]) / 1000
        reward = settings["bitrates_kbps"][action] / 1000 - settings["rebuffer_penalty"] * float(rebuffer)
        reward -= settings["switch_penalty"] * switch
        # Upstream mutates sleep_time while advancing the trace; preserve full physical idle time in our log.
        projected = max(before - float(delay), 0) + settings["chunk_duration_seconds"] * 1000
        sleep_ms = math.ceil(max(projected - settings["buffer_limit_seconds"] * 1000, 0) / 500) * 500
        elapsed += (float(delay) + sleep_ms) / 1000
        rows.append({"chunk": index, "action": action, "bitrate_kbps": settings["bitrates_kbps"][action],
                     "buffer_s": float(buffer), "download_ms": float(delay), "rebuffer_s": float(rebuffer),
                     "size_bytes": int(size), "throughput_mbps": float(size * 8 / delay / 1000),
                     "switch_mbps": switch, "reward": reward, "sleep_ms": sleep_ms,
                     "upstream_returned_sleep_ms": float(returned_sleep), "elapsed_s": elapsed,
                     "decision_ms": decision_ms, "decision": detail})
        if bool(end) != (index == settings["chunks"] - 1):
            raise ValueError("Unexpected simulator episode boundary")
        previous = action
        if end:
            break
        # Only causally available observations cross the policy interface.
        context = {"chunk_index": index, "next_index": index + 1, "buffer_s": float(buffer),
                   "previous_action": action, "next_sizes": list(map(int, next_sizes)),
                   "remaining": int(remaining), "bitrates": settings["bitrates_kbps"],
                   "history": [{k: r[k] for k in ("chunk", "bitrate_kbps", "buffer_s", "download_ms", "rebuffer_s",
                                                   "size_bytes", "throughput_mbps", "switch_mbps", "reward")} for r in rows]}
        started = time.perf_counter()
        action, detail = policy.choose(context)
        decision_ms = (time.perf_counter() - started) * 1000
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 6:
            raise ValueError(f"Invalid policy action: {action}")
    return {"record_id": record["id"], "split": record["split"], "dataset": record["dataset"],
            "trace": record["name"], "source_parent": record["source_parent"], "session": session,
            "rows": rows, "metrics": episode_metrics(rows, settings)}


def summarize(episodes):
    result = {}
    for dataset in sorted({e["dataset"] for e in episodes}):
        group = [e for e in episodes if e["dataset"] == dataset]
        traces = sorted({e["record_id"] for e in group})
        per_trace = {key: float(np.mean([e["metrics"]["qoe_primary"] for e in group if e["record_id"] == key]))
                     for key in traces}
        result[dataset] = {"traces": len(traces), "sessions": len(group),
                           "mean_qoe": float(np.mean(list(per_trace.values()))),
                           "trace_sd_ddof0": float(np.std(list(per_trace.values()))),
                           "mean_qoe_skip_first": float(np.mean([e["metrics"]["qoe_skip_first"] for e in group])),
                           "per_trace_qoe": per_trace}
    return result


def evaluate(args):
    manifest = verify(args.bundle)
    settings = manifest["settings"]
    available = [r for r in manifest["records"] if r["split"] == args.split]
    datasets = args.datasets or sorted({r["dataset"] for r in available})
    if len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate dataset names would duplicate and overwrite sessions")
    if set(datasets) - {r["dataset"] for r in available}:
        raise ValueError("Dataset unavailable in selected split")
    records = []
    for dataset in datasets:
        group = [r for r in available if r["dataset"] == dataset]
        records.extend(group if args.limit is None else group[:args.limit])
    jobs = [(r, s) for r in records for s in r["sessions"] if args.rounds is None or s["round"] in args.rounds]
    if not jobs:
        raise ValueError("No selected sessions")
    if args.rounds is not None and any(index not in {s["round"] for r in records for s in r["sessions"]} for index in args.rounds):
        raise ValueError("Requested round was not frozen in the bundle")
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Output directory already exists; choose a new directory (no implicit overwrite)")
    out.mkdir(parents=True)
    sizes = np.asarray([np.loadtxt(args.bundle / f"video1_sizes/video_size_{i}", dtype=np.int64) for i in range(6)])
    policy = None
    try:
        if args.policy == "jev":
            policy = JevHTTP(args.base_url, args.model, args.action_mode)
        elif args.policy == "dylas":
            if args.checkpoint is None:
                raise ValueError("DyLas requires an explicit --checkpoint; never automatically select a best test result")
            policy = DyLasProcess(args.dylas_python, args.dylas_repo, args.checkpoint, args.dylas_pool,
                                  args.action_mode, out / "dylas_worker.log", args.device)
        else:
            policy = Traditional(args.policy, settings, sizes)
        selection = [{"id": r["id"], **s} for r, s in jobs]
        run = {"bundle_fingerprint": manifest["fingerprint"], "policy": policy.metadata(),
               "selection": selection, "selection_fingerprint": fingerprint(selection), "split": args.split,
               "evaluation_scope": "diagnostic_existing_checkpoint" if args.policy in ("jev", "dylas") else "traditional_reference",
               "full_split": len(jobs) == sum(len(r["sessions"]) for r in available),
               "scoring": {"primary": "mean(rewards[1:-1])", "auxiliary": "mean(rewards[1:])"},
               "python": platform.python_version(), "numpy": np.__version__}
        run["fingerprint"] = fingerprint(run)
        write_json(out / "run.json", run)
        episodes = []
        for index, (record, session) in enumerate(jobs, 1):
            episode = run_episode(args.bundle, record, session, policy, settings)
            episode["run_fingerprint"] = run["fingerprint"]
            path = out / "episodes" / record["id"] / f"round_{session['round']:03}.json"
            write_json(path, episode)
            episodes.append(episode)
            write_json(out / "progress.json", {"completed": index, "expected": len(jobs), "complete": False})
            if index % 10 == 0 or index == len(jobs):
                print(f"{args.policy}: {index}/{len(jobs)} sessions", flush=True)
        policy.close()
        policy = None
        # Refuse to mark completed if frozen inputs/runtime were modified during the run.
        verify(args.bundle)
        result = {"run_fingerprint": run["fingerprint"], "bundle_fingerprint": manifest["fingerprint"],
                  "selection_fingerprint": run["selection_fingerprint"], "complete": True,
                  "full_split": run["full_split"], "evaluation_scope": run["evaluation_scope"], "results": summarize(episodes)}
        write_json(out / "summary.json", result)
        write_json(out / "progress.json", {"completed": len(jobs), "expected": len(jobs), "complete": True})
        print(json.dumps({k: {a: b for a, b in v.items() if a != "per_trace_qoe"} for k, v in result["results"].items()}, indent=2))
    except BaseException as error:
        write_json(out / "failure.json", {"complete": False, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if policy is not None:
            policy.close()


def report(args):
    runs = []
    for directory in args.runs:
        summary = json.loads((directory / "summary.json").read_text())
        run = json.loads((directory / "run.json").read_text())
        if fingerprint({k: v for k, v in run.items() if k != "fingerprint"}) != run["fingerprint"]:
            raise ValueError("Run metadata changed")
        if summary["run_fingerprint"] != run["fingerprint"]:
            raise ValueError("Run and summary do not match")
        if fingerprint(run["selection"]) != summary["selection_fingerprint"]:
            raise ValueError("Session selection changed")
        if summary["bundle_fingerprint"] != run["bundle_fingerprint"]:
            raise ValueError("Bundle references do not match")
        if not summary["complete"]:
            raise ValueError("Cannot compare an incomplete run")
        runs.append((directory, summary, run))
    if len({(s["bundle_fingerprint"], s["selection_fingerprint"]) for _, s, _ in runs}) != 1:
        raise ValueError("Runs differ in frozen environment/data/session selection; cannot compare")
    lines = ["# 共同协议对比", "", "旧模型结果默认是诊断性重评，不证明训练／选模来源已独立。", "",
             "| Run | Dataset | Sessions | Mean QoE | Trace SD | Skip-first QoE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for directory, summary, run in runs:
        for dataset, value in summary["results"].items():
            lines.append(f"| {directory.name} | {dataset} | {value['sessions']} | {value['mean_qoe']:.6f} | {value['trace_sd_ddof0']:.6f} | {value['mean_qoe_skip_first']:.6f} |")
    if args.out.exists():
        raise ValueError("Report already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        child = subs.add_parser(name)
        child.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
        if name == "prepare":
            child.add_argument("--dylas-root", type=Path, default=DEFAULT_DYLAS)
            child.add_argument("--jev-root", type=Path, default=DEFAULT_JEV)
    child = subs.add_parser("run")
    child.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    child.add_argument("--policy", required=True, choices=["bba", "throughput", "robustmpc", "lowest", "uniform_random", "jev", "dylas"])
    child.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    child.add_argument("--datasets", nargs="+", choices=["fcc", "norway3g", "oboe", "hsr"])
    child.add_argument("--rounds", nargs="+", type=int)
    child.add_argument("--limit", type=int, help="Maximum traces per dataset; run is marked partial")
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--base-url", default="http://127.0.0.1:8009")
    child.add_argument("--model", default="kev-latest")
    child.add_argument("--action-mode", choices=["native", "sample", "argmax"], default="native")
    child.add_argument("--dylas-python", type=Path, default=Path("/home/amora/anaconda3/envs/netllm/bin/python"))
    child.add_argument("--dylas-repo", type=Path, default=DEFAULT_DYLAS)
    child.add_argument("--checkpoint", type=Path)
    child.add_argument("--dylas-pool", type=Path, default=DEFAULT_DYLAS / "artifacts/exp_pools/merina_merged_logits.pkl")
    child.add_argument("--device", default="cuda:0")
    child = subs.add_parser("report")
    child.add_argument("--runs", type=Path, nargs="+", required=True)
    child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        m = prepare(args.bundle, args.dylas_root, args.jev_root)
        print(json.dumps({"bundle": str(args.bundle), "fingerprint": m["fingerprint"], "counts": m["audit"]["counts"],
                          "source_counts": m["audit"]["source_counts"]}, indent=2))
    elif args.command == "verify":
        m = verify(args.bundle)
        print(f"Verified {m['fingerprint']}; {m['audit']['counts']}")
    elif args.command == "run":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        evaluate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
