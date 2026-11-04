"""Candidate costs derived solely from the existing online ABR observation."""
from __future__ import annotations

import copy
import math

VERSION = "candidate_cost_v1"
BITRATES = (300, 750, 1200, 1850, 2850, 4300)
EXPLANATION = (
    " Candidate est_download_s uses the harmonic mean of the last five measured effective "
    "throughputs, not future bandwidth. est_buffer_margin_s is current buffer minus estimated "
    "download time; a negative value indicates estimated stall. switch_cost_qoe is the "
    "absolute bitrate change in Mbps. Estimates may be wrong; consider network history."
)


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {name}: {value}")
    return float(value)


def candidate_costs(state):
    """No trace identity, future samples, teacher action or return is consulted."""
    history = state["recent_throughput_mbps"][-5:]
    if not history:
        raise ValueError("Empty throughput history")
    values = [number(v, "throughput", positive=True) for v in history]
    bandwidth = len(values) / sum(1 / v for v in values)
    buffer = number(state["buffer_seconds"], "buffer")
    previous = number(state["previous_bitrate_kbps"], "previous bitrate", positive=True)
    if previous not in BITRATES:
        raise ValueError("Unsupported previous bitrate")
    sizes = state["next_chunk_sizes_bytes"]
    if set(sizes) != {str(b) for b in BITRATES}:
        raise ValueError("Expected exactly six candidate sizes")
    result = {}
    for bitrate in BITRATES:
        size = number(sizes[str(bitrate)], "chunk size", positive=True)
        if size != int(size):
            raise ValueError("Chunk size must be an integer byte count")
        # Measured throughput is payload / total observed delay: do not add RTT again.
        duration = size * 8 / 1e6 / bandwidth
        result[str(bitrate)] = {
            "size_bytes": int(size), "est_download_s": round(duration, 4),
            "est_buffer_margin_s": round(buffer - duration, 4),
            "switch_cost_qoe": round(abs(bitrate - previous) / 1000, 4),
        }
    return result


def transform_request(request, mode="cost"):
    """Retain all state, labels and IDs; change only option text and its explanation."""
    result = copy.deepcopy(request)
    if mode == "base":
        return result
    if mode != "cost":
        raise ValueError(f"Unknown candidate mode: {mode}")
    question = result["questions"]["bitrate"]
    if set(question["criteria"]) != {str(b) for b in BITRATES}:
        raise ValueError("Expected exactly six bitrate choices")
    if "est_download_s=" in " ".join(question["criteria"].values()):
        raise ValueError("Request already has candidate costs")
    costs = candidate_costs(result["state"])
    question["criteria"] = {
        key: f"{key} kbps; " + "; ".join(f"{name}={value}" for name, value in costs[key].items())
        for key in question["criteria"]
    }
    question["instructions"] += EXPLANATION
    return result
