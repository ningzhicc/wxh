"""Run or resume the frozen zero-shot ABR experiment."""
import argparse
import concurrent.futures
import datetime
import json
import subprocess
import time
from pathlib import Path

import httpx

from benchmark import (ROOT, NETLLM, ABRBENCH, DATASETS, METHODS, PROMPT, CRITERIA,
                       atomic_json, canonical_hash, digest, load_traces, run_episode, video_sizes)
from report import generate


def commit(repository):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/zero_shot_v1")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    assert args.workers > 0 and (args.limit is None or args.limit > 0)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    traces, sizes = load_traces(args.seed, args.datasets, args.limit), video_sizes()
    deploy = json.loads((ROOT.parent / "deployment.json").read_text())
    model_files = {name: digest(Path(deploy["checkpoint_path"]) / name)
                   for name in ["adapter_model.safetensors", "head.pt", "adapter_config.json"]}
    service = None
    if "kev" in args.methods:
        with httpx.Client(trust_env=False, timeout=30) as client:
            response = client.get(args.base_url + "/v1/models")
            response.raise_for_status()
            card = next(card for card in response.json()["models"] if card["name"] == "kev-latest")
        assert card["base"] == deploy["base"] and Path(card["run"]).resolve() == Path(deploy["checkpoint_path"]).resolve()
        assert abs(card["temperature"] - deploy["temperature"]) < 1e-10
        assert card["device"] == "cuda", card
        service = {key: card[key] for key in ["run", "base", "device", "backend", "dtype", "temperature"]}
    config = {"datasets": args.datasets, "methods": args.methods, "workers": args.workers, "seed": args.seed,
              "limit": args.limit, "base_url": args.base_url, "video": "video1/envivio_3g", "chunks": 48,
              "initial_bitrate_kbps": 300, "skip_first_reward": True, "charge_decision_latency": False,
              "fixed_simulation": True, "rebuffer_penalty": 4.3, "smoothness_penalty": 1,
              "prompt": PROMPT, "criteria": CRITERIA,
              "baselines": {"bba": {"reservoir_s": 5, "cushion_s": 10},
                            "throughput": {"window": 5, "safety_factor": .85},
                            "robustmpc": {"horizon": 5, "full_six_actions": True, "correct_next_chunk_index": True}}}
    frozen = {"config": config, "traces": [trace.metadata() for trace in traces],
              "repositories": {"NetLLM": commit(NETLLM.parent), "ABRBench": commit(ABRBENCH)},
              "video_sha256": {f"video_size_{i}": digest(NETLLM / f"data/videos/video1_sizes/video_size_{i}") for i in range(6)},
              "source_sha256": {name: digest(ROOT / name) for name in ["benchmark.py", "run_eval.py", "report.py", "protocol.md"]},
              "model_deployment": deploy, "model_files_sha256": model_files, "service": service}
    fingerprint = canonical_hash(frozen)
    manifest = {**frozen, "fingerprint": fingerprint,
                "started_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["fingerprint"] == fingerprint, "Run settings changed; use a new output directory"
    else:
        atomic_json(manifest_path, manifest)
    total = len(traces) * len(args.methods)
    completed, cached, failures = 0, 0, []
    started = time.monotonic()
    last_progress = started

    def progress(method, result, reused):
        nonlocal completed, cached, last_progress
        completed += 1
        cached += reused
        if completed % 10 == 0 or time.monotonic() - last_progress > 20 or completed == total:
            print(f"{completed}/{total} episodes complete; method={method}; dataset={result['dataset']}; "
                  f"cached={cached}; elapsed={time.monotonic()-started:.1f}s", flush=True)
            atomic_json(out / "progress.json", {"completed": completed, "total": total, "cached": cached,
                                                "method": method, "elapsed_s": time.monotonic() - started})
            last_progress = time.monotonic()

    for method in args.methods:
        print(f"Starting {method}: {len(traces)} episodes", flush=True)
        if method != "kev":
            for trace in traces:
                result, reused = run_episode(trace, method, sizes, args.base_url, fingerprint, out)
                progress(method, result, reused)
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_episode, trace, method, sizes, args.base_url, fingerprint, out): trace for trace in traces}
            for future in concurrent.futures.as_completed(futures):
                trace = futures[future]
                try:
                    result, reused = future.result()
                    progress(method, result, reused)
                except Exception as error:
                    failures.append({"method": method, "trace": trace.name, "dataset": trace.dataset,
                                     "error": repr(error)})
                    print(f"FAILED {method} {trace.dataset}/{trace.name}: {error!r}", flush=True)
                    atomic_json(out / "failures.json", failures)
    after_hashes = {name: digest(Path(deploy["checkpoint_path"]) / name) for name in model_files}
    assert after_hashes == model_files, "Model checkpoint files changed during the experiment"
    atomic_json(out / "model_integrity.json", {"unchanged": True, "before": model_files, "after": after_hashes})
    if failures:
        raise SystemExit(f"{len(failures)} failed episodes; resume the same command after resolving the failure")
    assert completed == total
    if not args.no_report:
        summary = generate(out)
        print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE: {out}; {total} episodes; {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
