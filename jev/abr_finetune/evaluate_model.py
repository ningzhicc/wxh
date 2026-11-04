"""Evaluate checkpoints on clean FCC validation or the frozen original tests."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import (BITRATES, CRITERIA, PROMPT, VIDEO_DIR, ROOT as EVAL_ROOT, RobustMPC, atomic_json,
                       canonical_hash, digest, load_traces, run_episode, video_sizes)
from report import write_csv
from prepare_data import fcc_traces
from kev.checkpoint import Checkpoint


def candidates_traces(scope):
    metadata = json.loads((ROOT / "data/traces.json").read_text())
    if scope == "test":
        return load_traces()
    keys = {row["key"] for row in metadata["valid_selection" if scope == "selection" else "valid"]}
    return [trace for trace in fcc_traces("valid") if trace.key in keys]


def summarize(results, out):
    rows = [{"dataset": r["dataset"], "method": "kev", "trace": r["trace"],
             "start_pointer": r["start_pointer"], **r["metrics"]} for r in results]
    write_csv(out / "episode_metrics.csv", rows)
    summary = []
    sizes = video_sizes()
    for dataset in sorted({row["dataset"] for row in results}):
        group = [r for r in results if r["dataset"] == dataset]
        client_latency = [r["decision"]["client_ms"] for ep in group for r in ep["steps"][1:]]
        server_latency = [r["decision"]["server_ms"] for ep in group for r in ep["steps"][1:]]
        matches = 0
        for ep in group:
            expert = RobustMPC()
            for index, step in enumerate(ep["steps"][1:], 1):
                prior = ep["steps"][index - 1]
                label = expert.choose(ep["steps"][:index], prior["buffer_s"], prior["action"], index, sizes)
                matches += label == step["action"]
        fields = ["qoe_mean", "mean_bitrate_mbps", "rebuffer_s", "smoothness_mbps", "switches", "no_stall"]
        summary.append({"dataset": dataset, "episodes": len(group),
                        **{field: float(np.mean([r["metrics"][field] for r in group])) for field in fields},
                        "client_latency_mean_ms": float(np.mean(client_latency)),
                        "client_latency_p95_ms": float(np.quantile(client_latency, .95)),
                        "server_latency_mean_ms": float(np.mean(server_latency)),
                        "mpc_action_agreement_on_own_states": matches / (47 * len(group))})
    write_csv(out / "summary.csv", summary)
    atomic_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scope", choices=["selection", "valid", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trace-limit", type=int, help="small validation probe only; never truncate the final tests")
    args = parser.parse_args()
    if args.trace_limit is not None:
        assert args.trace_limit > 0 and args.scope != "test"
    checkpoint = args.checkpoint.resolve()
    out = args.out.resolve()
    if (out / "completion.json").exists():
        saved = json.loads((out / "manifest.json").read_text())
        assert Path(saved["model_deployment"]["checkpoint_path"]) == checkpoint
        print("Completed evaluation already exists:", out, flush=True)
        print((out / "summary.json").read_text(), flush=True)
        return
    stock_python = ROOT.parent / "kev/.venv/bin/python"
    manager = ROOT.parent / "kev_service.py"
    subprocess.run([str(stock_python), str(manager), "stop"], check=True)
    subprocess.run([str(stock_python), str(manager), "start", "--checkpoint", str(checkpoint)], check=True)
    with httpx.Client(trust_env=False, timeout=5) as client:
        deadline = time.monotonic() + 180
        while True:
            try:
                response = client.get("http://127.0.0.1:8009/v1/models")
                response.raise_for_status()
                card = response.json()["models"][0]
                assert Path(card["run"]).resolve() == checkpoint
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise SystemExit("Serving failed to become ready; inspect jev/runtime/kev.log")
                time.sleep(5)
    ck = Checkpoint(str(checkpoint))
    trace_list = candidates_traces(args.scope)
    if args.trace_limit is not None:
        trace_list = trace_list[:args.trace_limit]
    sizes = video_sizes()
    files = {name: digest(checkpoint / name) for name in ["adapter_model.safetensors", "head.pt", "adapter_config.json"]}
    frozen = {"config": {"datasets": list(dict.fromkeys(t.dataset for t in trace_list)), "methods": ["kev"],
                          "workers": args.workers, "seed": 666, "limit": args.trace_limit, "scope": args.scope,
                          "prompt": PROMPT, "criteria": CRITERIA, "charge_decision_latency": False,
                          "skip_first_reward": True},
              "traces": [trace.metadata() for trace in trace_list],
              "video_sha256": {f"video_size_{i}": digest(VIDEO_DIR / f"video_size_{i}") for i in range(6)},
              "model_deployment": {"checkpoint_path": str(checkpoint), "base": ck.meta.base,
                                   "base_revision": ck.meta.base_revision, "temperature": ck.meta.temperature},
              "model_files_sha256": files, "data_manifest_sha256": digest(ROOT / "data/manifest.json"),
              "source_sha256": {"evaluate_model.py": digest(__file__), "benchmark.py": digest(EVAL_ROOT / "benchmark.py")},
              "service": {key: card[key] for key in ["device", "backend", "dtype", "temperature", "run"]}}
    fingerprint = canonical_hash(frozen)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text())["fingerprint"] == fingerprint
    else:
        atomic_json(out / "manifest.json", {**frozen, "fingerprint": fingerprint})
    results, failures = [], []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_episode, trace, "kev", sizes, "http://127.0.0.1:8009", fingerprint, out): trace for trace in trace_list}
        for future in concurrent.futures.as_completed(futures):
            try:
                result, _ = future.result()
                results.append(result)
            except Exception as error:
                failures.append({"trace": futures[future].name, "error": repr(error)})
                atomic_json(out / "failures.json", failures)
            if len(results) % 10 == 0 or len(results) == len(trace_list):
                progress = {"complete": len(results), "total": len(trace_list), "elapsed_s": time.monotonic() - started}
                atomic_json(out / "progress.json", progress)
                print(progress, flush=True)
    if failures:
        raise SystemExit(f"Evaluation has {len(failures)} failed episodes")
    assert len(results) == len(trace_list)
    current = {name: digest(checkpoint / name) for name in files}
    assert current == files
    atomic_json(out / "model_integrity.json", {"unchanged": True, "before": files, "after": current})
    summary = summarize(results, out)
    atomic_json(out / "completion.json", {"complete": True, "episodes": len(results), "fingerprint": fingerprint})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
