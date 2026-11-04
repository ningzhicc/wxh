"""Evaluate a Llama pointer checkpoint using the frozen ABR simulator and protocol."""
import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT.parent / "kev/.venv/bin/python"
sys.path.insert(0, str(ROOT.parent / "abr_finetune"))
from evaluate_model import candidates_traces, summarize
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import CRITERIA, PROMPT, VIDEO_DIR, atomic_json, canonical_hash, digest, run_episode, video_sizes
from audit_results import audit
from runtime import checkpoint_runtime
from kev.checkpoint import Checkpoint

PORT = 8010
URL = f"http://127.0.0.1:{PORT}"


@contextlib.contextmanager
def service(checkpoint, logfile, expected):
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    with (runtime / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pidfile = runtime / "service.pid"
        if pidfile.exists():
            pid = int(pidfile.read_text())
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if str(ROOT / "serve.py").encode() in argv and str(PORT).encode() in argv:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1)
            except FileNotFoundError:
                pass
            pidfile.unlink(missing_ok=True)
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", PORT))
            except OSError as error:
                raise RuntimeError("Port 8010 cannot be bound; refusing to replace another instance") from error
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "KEV_DTYPE": "bf16", "KEV_BACKEND": "torch",
               "KEV_FUSED": "0", "KEV_CUDA_GRAPHS": "0", "PYTHONUNBUFFERED": "1"}
        with logfile.open("a") as stream:
            child = subprocess.Popen([str(PYTHON), str(ROOT / "serve.py"), "--run", str(checkpoint),
                "--device", "cuda", "--port", str(PORT)], env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
        pidfile.write_text(str(child.pid))
        try:
            with httpx.Client(trust_env=False, timeout=3) as client:
                deadline = time.monotonic() + 180
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f"Llama serving exited; inspect {logfile}")
                    try:
                        response = client.get(URL + "/v1/models")
                        response.raise_for_status()
                        card = response.json()["models"][0]
                        assert {k: card[k] for k in expected} == expected
                        break
                    except httpx.HTTPError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"Llama serving timeout; inspect {logfile}")
                        time.sleep(1)
            yield card
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            pidfile.unlink(missing_ok=True)


def evaluate(checkpoint, scope, out, trace_limit=None):
    checkpoint, out = Path(checkpoint).resolve(), Path(out).resolve()
    if trace_limit is not None:
        assert scope != "test" and trace_limit > 0
    checkpoint_runtime(checkpoint)
    ck = Checkpoint(checkpoint)
    traces = candidates_traces(scope)
    if trace_limit:
        traces = traces[:trace_limit]
    files = {name: digest(checkpoint / name) for name in ["adapter_model.safetensors", "head.pt", "adapter_config.json"]}
    expected_service = {"device": "cuda", "backend": "torch", "dtype": "bfloat16",
                        "temperature": ck.meta.temperature, "run": str(checkpoint)}
    frozen = {"config": {"datasets": list(dict.fromkeys(t.dataset for t in traces)), "methods": ["kev"],
        "workers": 4, "seed": 666, "limit": trace_limit, "scope": scope, "prompt": PROMPT, "criteria": CRITERIA,
        "charge_decision_latency": False, "skip_first_reward": True},
        "traces": [t.metadata() for t in traces],
        "video_sha256": {f"video_size_{i}": digest(VIDEO_DIR / f"video_size_{i}") for i in range(6)},
        "model_deployment": {"checkpoint_path": str(checkpoint), "base": ck.meta.base,
            "base_revision": ck.meta.base_revision, "temperature": ck.meta.temperature},
        "model_files_sha256": files, "data_manifest_sha256": digest(ROOT.parent / "abr_finetune/data/manifest.json"),
        "source_sha256": {"evaluate.py": digest(__file__), "runtime.py": digest(ROOT / "runtime.py"),
            "summarize.py": digest(ROOT.parent / "abr_finetune/evaluate_model.py"),
            "benchmark.py": digest(ROOT.parent / "abr_eval/benchmark.py")}, "service": expected_service}
    fingerprint = canonical_hash(frozen)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text())["fingerprint"] == fingerprint
    else:
        atomic_json(out / "manifest.json", {**frozen, "fingerprint": fingerprint})
    if (out / "completion.json").exists():
        done = json.loads((out / "completion.json").read_text())
        assert done["complete"] and done["fingerprint"] == fingerprint and done["episodes"] == len(traces)
        if scope == "test" and not (out / "audit.json").exists():
            audit(out)
        return json.loads((out / "summary.json").read_text())
    started = time.monotonic()
    results = []
    sizes = video_sizes()
    with service(checkpoint, out / "server.log", expected_service):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_episode, t, "kev", sizes, URL, fingerprint, out) for t in traces]
            for future in concurrent.futures.as_completed(futures):
                result, _ = future.result()
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(traces):
                    progress = {"complete": len(results), "total": len(traces), "elapsed_s": time.monotonic()-started}
                    atomic_json(out / "progress.json", progress)
                    print(f"Evaluation {out.name}: {progress}", flush=True)
    after = {name: digest(checkpoint / name) for name in files}
    assert after == files
    atomic_json(out / "model_integrity.json", {"unchanged": True, "before": files, "after": after})
    summary = summarize(results, out)
    atomic_json(out / "completion.json", {"complete": True, "episodes": len(results), "fingerprint": fingerprint})
    if scope == "test":
        audit(out)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scope", choices=["selection", "valid", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-limit", type=int)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.scope, args.out, args.trace_limit)
