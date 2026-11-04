"""Module 2 labels: explicit one-step, observed-delay-scaled counterfactual proxy."""
from __future__ import annotations

import math

METHOD = "observed_delay_size_ratio_one_step_v1"
BITRATES = (300, 750, 1200, 1850, 2850, 4300)


def proxy_targets(observation, action, observed_delay_s, observed_reward, rtt_s=.08):
    """Use the next recorded observation only for training supervision, never model input.

    T(a) = RTT + (observed T(chosen) - RTT) * size(a)/size(chosen).
    This assumes equal average payload rate across alternatives and ignores future effects.
    """
    if not math.isfinite(observed_delay_s) or observed_delay_s <= rtt_s or rtt_s < 0:
        raise ValueError("Invalid measured delay / assumed RTT")
    if int(action) != action or not 0 <= int(action) < len(BITRATES):
        raise ValueError("Invalid recorded action")
    if not math.isfinite(observed_reward):
        raise ValueError("Invalid recorded reward")
    sizes = observation["next_chunk_sizes_bytes"]
    if set(sizes) != {str(b) for b in BITRATES} or any(v <= 0 for v in sizes.values()):
        raise ValueError("Invalid candidate sizes")
    buffer = observation["buffer_seconds"]
    previous = observation["previous_bitrate_kbps"]
    values, delays = {}, {}
    chosen_size = sizes[str(BITRATES[int(action)])]
    for bitrate in BITRATES:
        delay = rtt_s + (observed_delay_s - rtt_s) * sizes[str(bitrate)] / chosen_size
        delays[str(bitrate)] = delay
        values[str(bitrate)] = bitrate / 1000 - 4.3 * max(delay - buffer, 0) - abs(bitrate - previous) / 1000
    error = abs(values[str(BITRATES[int(action)])] - observed_reward)
    if error > .001:
        raise ValueError(f"Recorded reward inconsistent with decoded chosen outcome: {error}")
    best = max(values.values())
    return {"valid": True, "method": METHOD, "horizon_chunks": 1,
            "estimated_qoe": values, "regret_qoe": {k: best - v for k, v in values.items()},
            "chosen_reward_reconstruction_error": error,
            "observed_chosen_download_s": observed_delay_s,
            "assumed_rtt_s": rtt_s,
            "is_exact_counterfactual_replay": False}


def terminal_target():
    return {"valid": False, "method": METHOD, "reason": "terminal_transition_has_no_next_observation",
            "regret_qoe": None, "is_exact_counterfactual_replay": False}
