"""Frozen zero-shot ABR benchmark using the unmodified NetLLM simulator."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent
NETLLM = ROOT / "vendor/NetLLM/adaptive_bitrate_streaming"
ABRBENCH = ROOT / "vendor/ABRBench"
sys.path.insert(0, str(NETLLM))
from baseline_special.env import Environment

BITRATES = np.array([300, 750, 1200, 1850, 2850, 4300])
METHODS = ["kev", "bba", "throughput", "robustmpc"]
DATASETS = {
    "fcc": NETLLM / "data/traces/test/fcc-test",
    "hsr": ABRBENCH / "trace/ABRBench-3G/HSR",
    "oboe": ABRBENCH / "trace/ABRBench-3G/Oboe/test",
    "norway3g": ABRBENCH / "trace/ABRBench-4G+/Norway3G/test",
}
VIDEO_DIR = NETLLM / "data/videos/video1_sizes"
EXPECTED_COUNTS = {"fcc": 100, "hsr": 34, "oboe": 100, "norway3g": 41}
PROMPT = (
    "Choose the next video chunk bitrate to maximize long-run streaming quality of experience. "
    "Each chunk reward is bitrate_Mbps minus 4.3 times rebuffer_seconds minus the absolute "
    "bitrate change in Mbps. Chunks last 4 seconds. Use the measured throughput history, "
    "buffer and candidate chunk sizes to balance quality, rebuffering and quality changes. "
    "Future network bandwidth is unknown. Select exactly one available bitrate."
)
CRITERIA = {str(int(b)): f"{int(b)} kbps video bitrate" for b in BITRATES}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class Trace:
    dataset: str
    name: str
    path: Path
    times: np.ndarray
    bandwidth: np.ndarray
    start_pointer: int
    sha256: str

    @property
    def key(self):
        return self.dataset + "_" + hashlib.sha256(self.name.encode()).hexdigest()[:16]

    def metadata(self):
        return {"dataset": self.dataset, "trace": self.name, "path": str(self.path),
                "start_pointer": self.start_pointer, "sha256": self.sha256,
                "samples": len(self.times), "key": self.key}


def load_traces(seed=666, datasets=None, limit=None):
    traces = []
    for dataset in datasets or list(DATASETS):
        directory = DATASETS[dataset]
        files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix != ".pkl")
        assert len(files) == EXPECTED_COUNTS[dataset], (dataset, len(files))
        pointers_file = directory / "mahimahi_ptrs.pkl"
        pointers = pickle.loads(pointers_file.read_bytes()) if pointers_file.exists() else None
        if pointers is not None:
            assert len(pointers) == len(files)
        for index, path in enumerate(files[:limit] if limit else files):
            values = np.loadtxt(path, ndmin=2)
            assert values.shape[0] >= 2 and values.shape[1] == 2, path
            assert np.isfinite(values).all() and (values[:, 1] >= 0).all(), path
            assert (np.diff(values[:, 0]) >= 0).all(), path
            assert np.any((np.diff(values[:, 0]) > 0) & (values[1:, 1] > 0)), path
            pointer = int(pointers[index]) if pointers is not None else 1 + int.from_bytes(
                hashlib.sha256(f"{seed}:{path.name}".encode()).digest()[:8], "big") % (len(values) - 1)
            assert 1 <= pointer < len(values), (path, pointer)
            traces.append(Trace(dataset, path.name, path, values[:, 0], values[:, 1], pointer, digest(path)))
    return traces


def video_sizes():
    sizes = np.array([np.loadtxt(VIDEO_DIR / f"video_size_{i}", dtype=np.int64) for i in range(6)])
    assert sizes.shape[0] == 6 and sizes.shape[1] >= 48 and (sizes > 0).all()
    for i in range(6):
        assert (VIDEO_DIR / f"video_size_{i}").read_bytes() == (ABRBENCH / f"video/envivio_3g/video_size_{i}").read_bytes()
    return sizes


def environment(trace):
    return Environment(all_cooked_time=[trace.times], all_cooked_bw=[trace.bandwidth],
                       all_file_names=[trace.name], all_mahimahi_ptrs=[trace.start_pointer],
                       video_size_dir=str(VIDEO_DIR) + "/", fixed=True, trace_num=1)


def reward(bitrate, previous, rebuffer):
    return float(BITRATES[bitrate] / 1000 - 4.3 * rebuffer - abs(BITRATES[bitrate] - BITRATES[previous]) / 1000)


def observation(history, buffer, previous, sizes, remaining):
    recent = history[-6:]
    return {
        "recent_throughput_mbps": [round(r["throughput_mbps"], 4) for r in recent],
        "recent_download_seconds": [round(r["delay_ms"] / 1000, 4) for r in recent],
        "recent_bitrates_kbps": [int(r["bitrate_kbps"]) for r in recent],
        "buffer_seconds": round(float(buffer), 4),
        "previous_bitrate_kbps": int(BITRATES[previous]),
        "next_chunk_sizes_bytes": {str(int(b)): int(s) for b, s in zip(BITRATES, sizes)},
        "chunk_duration_seconds": 4,
        "chunks_remaining": int(remaining),
    }


def make_request(state):
    return {"model": "kev-latest", "state": state,
            "questions": {"bitrate": {"type": "choice", "instructions": PROMPT, "criteria": CRITERIA}}}


def validate_answer(response):
    answer = response["answers"]["bitrate"]
    probabilities = answer["probabilities"]
    assert set(probabilities) == set(CRITERIA), probabilities
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()), probabilities
    assert math.isclose(sum(probabilities.values()), 1, abs_tol=0.002), probabilities
    choice = answer["choice"]
    assert choice in CRITERIA and probabilities[choice] >= max(probabilities.values()) - 0.00011
    return list(CRITERIA).index(choice), probabilities


def harmonic(values):
    selected = np.asarray(values[-5:], dtype=float)
    assert len(selected) and np.isfinite(selected).all() and (selected > 0).all()
    return float(len(selected) / np.sum(1 / selected))


class BBA:
    def choose(self, history, buffer, previous, next_index, sizes):
        return 0 if buffer < 5 else 5 if buffer >= 15 else int(5 * (buffer - 5) / 10)


class Throughput:
    def choose(self, history, buffer, previous, next_index, sizes):
        budget = 0.85 * harmonic([r["throughput_mbps"] for r in history]) * 1000
        return int(max(0, np.searchsorted(BITRATES, budget, side="right") - 1))


COMBOS = {h: np.array(list(itertools.product(range(6), repeat=h)), dtype=np.int64) for h in range(1, 6)}


class RobustMPC:
    def __init__(self):
        self.estimates, self.errors = [], []

    def choose(self, history, buffer, previous, next_index, sizes):
        recent = [r["throughput_mbps"] for r in history]
        self.errors.append(abs(self.estimates[-1] - recent[-1]) / recent[-1] if self.estimates else 0)
        estimate = harmonic(recent)
        future_bw = estimate / (1 + max(self.errors[-5:]))
        self.estimates.append(estimate)
        horizon = min(5, 48 - next_index)
        combos = COMBOS[horizon]
        buffers = np.full(len(combos), float(buffer))
        rewards = np.zeros(len(combos))
        last = np.full(len(combos), previous)
        for offset in range(horizon):
            actions = combos[:, offset]
            download = sizes[actions, next_index + offset] * 8 / 1e6 / future_bw
            stall = np.maximum(download - buffers, 0)
            buffers = np.maximum(buffers - download, 0) + 4
            rewards += BITRATES[actions] / 1000 - 4.3 * stall - np.abs(BITRATES[actions] - BITRATES[last]) / 1000
            last = actions
        # Ascending sequences and argmax make ties deterministic, preferring the lower first bitrate.
        return int(combos[int(np.argmax(rewards)), 0])


def summarize_episode(rows):
    scored = rows[1:]
    inference = [r["decision"]["client_ms"] for r in scored]
    server = [r["decision"]["server_ms"] for r in scored if r["decision"]["server_ms"] is not None]
    stalls = sum(r["rebuffer_s"] for r in scored)
    return {
        "qoe_mean": float(np.mean([r["reward"] for r in scored])),
        "qoe_total": float(sum(r["reward"] for r in scored)),
        "qoe_total_including_startup": float(sum(r["reward"] for r in rows)),
        "mean_bitrate_mbps": float(np.mean([r["bitrate_kbps"] / 1000 for r in scored])),
        "rebuffer_s": float(stalls), "startup_s": float(rows[0]["delay_ms"] / 1000),
        "rebuffer_ratio": float(stalls / (192 + stalls)),
        "smoothness_mbps": float(np.mean([r["smoothness_mbps"] for r in scored])),
        "switches": int(sum(r["smoothness_mbps"] > 0 for r in scored)),
        "stall_chunks": int(sum(r["rebuffer_s"] > 0 for r in scored)),
        "no_stall": bool(stalls < 1e-9), "chunks": len(rows), "decisions": len(scored),
        "client_latency_mean_ms": float(np.mean(inference)),
        "client_latency_p95_ms": float(np.quantile(inference, .95)),
        "server_latency_mean_ms": float(np.mean(server)) if server else None,
    }


def run_episode(trace, method, sizes, base_url, fingerprint, out):
    path = out / "episodes" / method / trace.dataset / f"{trace.key}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved["fingerprint"] == fingerprint and saved["metrics"]["chunks"] == 48
        return saved, True
    sim = environment(trace)
    policy = {"bba": BBA, "throughput": Throughput, "robustmpc": RobustMPC}.get(method)
    policy = policy() if policy else None
    rows, previous, action, elapsed_s = [], 0, 0, 0.0
    pending = {"client_ms": 0.0, "server_ms": None, "state": None, "probabilities": None}
    client = httpx.Client(base_url=base_url, timeout=180, trust_env=False) if method == "kev" else None
    try:
        for chunk in range(48):
            before_buffer_ms = float(sim.buffer_size)
            delay, _, buffer, rebuf, size, next_sizes, end, remaining = sim.get_video_chunk(action)
            predicted_buffer = max(before_buffer_ms - delay, 0) + 4000
            actual_sleep_ms = math.ceil(max(predicted_buffer - 60000, 0) / 500) * 500
            elapsed_s += (delay + actual_sleep_ms) / 1000
            rows.append({"chunk": chunk, "action": int(action), "bitrate_kbps": int(BITRATES[action]),
                         "buffer_s": float(buffer), "rebuffer_s": float(rebuf),
                         "size_bytes": int(size), "delay_ms": float(delay), "sleep_ms": actual_sleep_ms,
                         "time_s": elapsed_s, "throughput_mbps": float(size * 8 / delay / 1000),
                         "smoothness_mbps": float(abs(BITRATES[action] - BITRATES[previous]) / 1000),
                         "reward": reward(action, previous, rebuf), "decision": pending})
            previous = action
            if end:
                assert chunk == 47
                break
            state = observation(rows, buffer, previous, next_sizes, remaining)
            started = time.perf_counter()
            if method == "kev":
                response = client.post("/v1/systemone", json=make_request(state))
                response.raise_for_status()
                data = response.json()
                action, probabilities = validate_answer(data)
                server_ms = float(data["latency_ms"])
            else:
                action = policy.choose(rows, buffer, previous, chunk + 1, sizes)
                probabilities, server_ms = None, None
            pending = {"client_ms": (time.perf_counter() - started) * 1000,
                       "server_ms": server_ms, "state": state, "probabilities": probabilities}
    finally:
        if client:
            client.close()
    assert len(rows) == 48 and all(0 <= r["action"] < 6 for r in rows)
    result = {"fingerprint": fingerprint, "method": method, **trace.metadata(),
              "metrics": summarize_episode(rows), "steps": rows}
    log = out / "logs" / method / trace.dataset / ("result_sim_abr_" + trace.name)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join("\t".join(str(v) for v in [r["time_s"], r["bitrate_kbps"], r["buffer_s"],
                   r["rebuffer_s"], r["size_bytes"], r["delay_ms"], r["smoothness_mbps"], r["reward"]]) + "\n"
                   for r in rows), encoding="utf-8")
    atomic_json(path, result)
    return result, False
