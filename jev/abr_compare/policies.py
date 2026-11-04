"""Policy adapters. All policies use the common runner's simulator and rewards."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
import selectors
import subprocess
import os
import sys
import urllib.request

import numpy as np

from common import ROOT, file_hash


PROMPT = (
    "Choose the next video chunk bitrate to maximize long-run streaming quality of experience. "
    "Each chunk reward is bitrate_Mbps minus 4.3 times rebuffer_seconds minus the absolute "
    "bitrate change in Mbps. Chunks last 4 seconds. Use the measured throughput history, "
    "buffer and candidate chunk sizes to balance quality, rebuffering and quality changes. "
    "Future network bandwidth is unknown. Select exactly one available bitrate."
)


def jev_state(context):
    recent = context["history"][-6:]
    return {
        "recent_throughput_mbps": [round(r["throughput_mbps"], 4) for r in recent],
        "recent_download_seconds": [round(r["download_ms"] / 1000, 4) for r in recent],
        "recent_bitrates_kbps": [r["bitrate_kbps"] for r in recent],
        "buffer_seconds": round(context["buffer_s"], 4),
        "previous_bitrate_kbps": context["bitrates"][context["previous_action"]],
        "next_chunk_sizes_bytes": {str(b): int(s) for b, s in zip(context["bitrates"], context["next_sizes"])},
        "chunk_duration_seconds": 4, "chunks_remaining": context["remaining"],
    }


def request_for(context, model):
    return {"model": model, "state": jev_state(context), "questions": {"bitrate": {
        "type": "choice", "instructions": PROMPT,
        "criteria": {str(b): f"{b} kbps video bitrate" for b in context["bitrates"]}}}}


class Traditional:
    def __init__(self, name, settings, sizes):
        self.name, self.settings, self.sizes = name, settings, sizes
        self.combos = {h: np.array(list(itertools.product(range(6), repeat=h)), dtype=np.int64)
                       for h in range(1, settings["baselines"]["mpc_horizon"] + 1)} if name == "robustmpc" else {}

    def metadata(self):
        return {"type": self.name, "parameters": self.settings["baselines"],
                "video_horizon": self.settings["baselines"]["mpc_horizon"] if self.name == "robustmpc" else 1,
                "future_bandwidth_visible": False}

    def reset(self, seed):
        self.estimates, self.errors = [], []
        self.rng = np.random.RandomState(seed)

    def choose(self, context):
        cfg, buffer = self.settings["baselines"], context["buffer_s"]
        if self.name == "lowest":
            return 0, {}
        if self.name == "uniform_random":
            return int(self.rng.randint(6)), {}
        if self.name == "bba":
            ratio = (buffer - cfg["bba_reservoir_s"]) / cfg["bba_cushion_s"]
            return int(max(0, min(5, 5 * ratio))), {}
        throughputs = np.asarray([r["throughput_mbps"] for r in context["history"][-5:]])
        estimate = float(len(throughputs) / np.sum(1 / throughputs))
        bitrates = np.asarray(context["bitrates"])
        if self.name == "throughput":
            action = max(0, np.searchsorted(bitrates, estimate * cfg["throughput_factor"] * 1000, side="right") - 1)
            return int(action), {}
        actual = throughputs[-1]
        self.errors.append(abs(self.estimates[-1] - actual) / actual if self.estimates else 0)
        self.estimates.append(estimate)
        bandwidth = estimate / (1 + max(self.errors[-5:]))
        horizon = min(cfg["mpc_horizon"], context["remaining"])
        combos = self.combos[horizon]
        buffers, rewards = np.full(len(combos), buffer), np.zeros(len(combos))
        previous = np.full(len(combos), context["previous_action"])
        for offset in range(horizon):
            actions = combos[:, offset]
            download = self.sizes[actions, context["next_index"] + offset] * 8 / 1e6 / bandwidth
            stall = np.maximum(download - buffers, 0)
            buffers = np.maximum(buffers - download, 0) + self.settings["chunk_duration_seconds"]
            rewards += bitrates[actions] / 1000 - self.settings["rebuffer_penalty"] * stall
            rewards -= self.settings["switch_penalty"] * np.abs(bitrates[actions] - bitrates[previous]) / 1000
            previous = actions
        return int(combos[int(np.argmax(rewards)), 0]), {}

    def close(self):
        pass


class JevHTTP:
    def __init__(self, base_url, model="kev-latest", action_mode="native"):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.mode = "argmax" if action_mode == "native" else action_mode
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.card = self._card()
        path = Path(self.card.get("run", ""))
        self.model_hashes = {p.name: file_hash(p) for p in sorted(path.glob("*"))
                             if p.is_file() and p.suffix in (".pt", ".json", ".safetensors")} if path.is_dir() else {}

    def _get(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=180) as response:
            return json.load(response)

    def _card(self):
        cards = self._get("/v1/models")["models"]
        card = next(c for c in cards if c["name"] == self.model)
        return {k: card[k] for k in ("name", "run", "base", "temperature", "dtype", "backend", "device", "truncate_states") if k in card}

    def metadata(self):
        return {"type": "jev_http", "model": self.model, "action_mode": self.mode,
                "card": self.card, "checkpoint_sha256": self.model_hashes,
                "prompt": PROMPT, "state_builder": "legacy_six_observations"}

    def reset(self, seed):
        if self._card() != self.card:
            raise ValueError("Jev service model changed")
        self.rng = np.random.RandomState(seed)

    def choose(self, context):
        request = request_for(context, self.model)
        response = self._get("/v1/systemone", request)
        if response.get("truncated"):
            raise ValueError("Jev truncated input; comparison invalid")
        answer = response["answers"]["bitrate"]
        labels = [str(b) for b in context["bitrates"]]
        probabilities = answer["probabilities"]
        if set(probabilities) != set(labels):
            raise ValueError("Wrong Jev candidate set")
        p = np.asarray([probabilities[key] for key in labels], dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or abs(p.sum() - 1) > .002:
            raise ValueError("Invalid Jev probability vector")
        if answer["choice"] not in labels or p[labels.index(answer["choice"])] < p.max() - .00011:
            raise ValueError("Jev choice disagrees with probabilities")
        action = int(self.rng.choice(6, p=p / p.sum())) if self.mode == "sample" else labels.index(answer["choice"])
        return action, {"probabilities": probabilities, "state": request["state"],
                        "server_latency_ms": response.get("latency_ms")}

    def close(self):
        if self._card() != self.card:
            raise ValueError("Jev service model changed during evaluation")


class DyLasProcess:
    """Keep DyLas dependencies and global import names outside the scoring process."""
    def __init__(self, python, repo, checkpoint, pool, mode, log_path, device="cuda:0"):
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        self.log = Path(log_path).open("a")
        command = [str(python), str(ROOT / "dylas_worker.py"), "--repo", str(repo), "--checkpoint", str(checkpoint),
                   "--pool", str(pool), "--action-mode", mode, "--device", device]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self._receive()
            self.info = ready["metadata"]
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.selector.select(timeout=240):
            raise TimeoutError("DyLas worker timed out; inspect its log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("DyLas worker exited; inspect its log")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _call(self, message):
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def metadata(self):
        return self.info

    def reset(self, seed):
        self._call({"operation": "reset", "seed": seed})

    def choose(self, context):
        # No trace identity, start pointer, or future network values enter the worker.
        response = self._call({"operation": "choose", "context": context})
        return int(response["action"]), {}

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.log.close()
