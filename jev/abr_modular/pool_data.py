"""Decode existing 6x6 observations without relabelling expert actions."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jev.abr_alignment.audit_provenance import RestrictedPoolReader
from jev.abr_candidate.features import BITRATES

PROMPT = (
    "Choose the next video chunk bitrate to maximize long-run streaming quality of experience. "
    "Each chunk reward is bitrate_Mbps minus 4.3 times rebuffer_seconds minus the absolute "
    "bitrate change in Mbps. Chunks last 4 seconds. Use the measured throughput history, "
    "buffer and candidate chunk sizes to balance quality, rebuffering and quality changes. "
    "Future network bandwidth is unknown. Select exactly one available bitrate."
)


def load_pool(path):
    with Path(path).open("rb") as stream:
        pool = RestrictedPoolReader(stream).load()
    n = len(pool.states)
    if not n or any(len(getattr(pool, k)) != n for k in ("actions", "rewards", "dones")):
        raise ValueError("Pool fields have different lengths")
    if sum(bool(d) for d in pool.dones) != 656 or n != 30832 or not pool.dones[-1]:
        raise ValueError("Expected 656 complete episodes / 30832 transitions")
    return pool


def bitrate_index(normalized):
    differences = np.abs(np.asarray(BITRATES) / 4300 - normalized)
    index = int(np.argmin(differences))
    if differences[index] > 2e-6:
        raise ValueError("Invalid normalized bitrate")
    return index


def decode_state(raw, sizes):
    state = np.asarray(raw, dtype=float)
    if state.shape != (6, 6) or not np.isfinite(state).all():
        raise ValueError("Invalid 6x6 observation")
    positions = np.flatnonzero(state[0] > 0)
    if len(positions) == 0 or positions[-1] != 5:
        raise ValueError("Missing current history")
    if any(state[2, j] <= 0 or state[3, j] <= 0 for j in positions):
        raise ValueError("Invalid measured throughput/delay")
    history_bitrates = [BITRATES[bitrate_index(state[0, j])] for j in positions]
    matches = np.flatnonzero(np.all(np.isclose(sizes, state[4][None, :] * 1e6, rtol=2e-6, atol=.1), axis=1))
    if len(matches) != 1:
        raise ValueError("Candidate sizes do not uniquely match known video1 assets")
    next_index = int(matches[0])
    remaining = int(round(state[5, -1] * 48))
    if abs(remaining - state[5, -1] * 48) > 1e-4 or remaining < 1:
        raise ValueError("Invalid chunks remaining")
    if state[1, -1] < 0:
        raise ValueError("Negative buffer")
    observation = {
        "recent_throughput_mbps": [round(float(state[2, j] * 8), 4) for j in positions],
        "recent_download_seconds": [round(float(state[3, j] * 10), 4) for j in positions],
        "recent_bitrates_kbps": history_bitrates,
        "buffer_seconds": round(float(state[1, -1] * 10), 4),
        "previous_bitrate_kbps": history_bitrates[-1],
        "next_chunk_sizes_bytes": {str(b): int(v) for b, v in zip(BITRATES, sizes[next_index])},
        "chunk_duration_seconds": 4,
        "chunks_remaining": remaining,
    }
    return observation, next_index


def request_from_state(observation, action, episode, step, index):
    if isinstance(action, bool) or int(action) != action or not 0 <= int(action) < 6:
        raise ValueError("Invalid recorded action")
    return {"model": "kev-latest", "state": observation,
            "questions": {"bitrate": {"type": "choice", "instructions": PROMPT,
                "criteria": {str(b): f"{b} kbps video bitrate" for b in BITRATES},
                "label": str(BITRATES[int(action)]), "src": "abr_merina656"}},
            "_meta": {"source": "abr_merina656", "split": "train", "id": f"merina656/{episode}/{step}",
                      "group_id": f"merina656/{episode}", "episode": episode, "step": step,
                      "pool_index": index, "trace_identity": "not_stored_in_original_pool"}}
