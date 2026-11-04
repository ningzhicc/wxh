"""Freeze all selected merina656 experiences into paired Kev training inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from io_utils import DATA, POOL, POOL_SHA256, VIDEO, atomic_json, canonical, data_code_hashes, digest, fingerprint, verify_data
from pool_data import BITRATES, decode_state, load_pool, request_from_state
from regret import METHOD, proxy_targets, terminal_target
from jev.abr_candidate.features import transform_request


def prepare(pool_path=POOL, destination=DATA, video=VIDEO):
    pool_path, destination, video = map(lambda p: Path(p).resolve(), (pool_path, destination, video))
    if destination.exists():
        return verify_data(destination)
    if digest(pool_path) != POOL_SHA256:
        raise ValueError("Selected merina pool SHA-256 differs")
    pool = load_pool(pool_path)
    sizes = np.asarray([np.loadtxt(video / f"video_size_{i}", dtype=np.int64) for i in range(6)]).T
    if sizes.shape != (49, 6):
        raise ValueError("Expected known video1 assets (49 size rows)")
    video_hashes = {f"video_size_{i}": digest(video / f"video_size_{i}") for i in range(6)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".merina-data-", dir=destination.parent))
    counts, action_counts, errors = Counter(), Counter(), []
    indices = Counter()
    episode, step = 0, 0
    try:
        with (staging / "base.jsonl").open("x", encoding="utf-8") as base, (staging / "cost.jsonl").open("x", encoding="utf-8") as cost:
            for i, raw in enumerate(pool.states):
                observation, video_index = decode_state(raw, sizes)
                action = int(pool.actions[i])
                request = request_from_state(observation, pool.actions[i], episode, step, i)
                if pool.dones[i]:
                    if step != 46:
                        raise ValueError("Unexpected episode length")
                    target = terminal_target()
                else:
                    following, next_index = decode_state(pool.states[i + 1], sizes)
                    if following["previous_bitrate_kbps"] != BITRATES[action] or next_index != video_index + 1:
                        raise ValueError("Broken transition chain / action alignment")
                    target = proxy_targets(observation, action, float(np.asarray(pool.states[i + 1])[3, -1]) * 10,
                                           float(pool.rewards[i]))
                    errors.append(target["chosen_reward_reconstruction_error"])
                request["_supervision"] = target
                request["_meta"]["original_reward"] = float(pool.rewards[i])
                request["_meta"]["video_size_row"] = video_index
                base.write(json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                cost.write(json.dumps(transform_request(request, "cost"), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                counts["records"] += 1
                counts["regret_valid" if target["valid"] else "regret_masked"] += 1
                action_counts[str(BITRATES[action])] += 1
                indices[video_index] += 1
                if pool.dones[i]:
                    episode += 1
                    step = 0
                else:
                    step += 1
                if counts["records"] % 5000 == 0:
                    print(f"Prepared {counts['records']}/30832 records", flush=True)
        if episode != 656 or counts["regret_valid"] != 30176 or counts["records"] != 30832:
            raise ValueError("Unexpected final counts")
        if digest(pool_path) != POOL_SHA256 or video_hashes != {n: digest(video / n) for n in video_hashes}:
            raise ValueError("Source changed during preparation")
        manifest = {"version": "merina656_modular_v1", "pool": str(pool_path), "pool_sha256": POOL_SHA256,
            "episodes": episode, "counts": dict(counts), "label_counts": dict(action_counts),
            "video_size_rows": {str(k): v for k, v in sorted(indices.items())}, "video_sha256": video_hashes,
            "regret_method": METHOD, "max_chosen_reward_reconstruction_error": max(errors),
            "teacher_logits_used": False, "original_actions_preserved": True,
            "trace_identity_and_split_independence_verified": False,
            "data_notes": ["All original 656 episodes remain training experiences; no test labels or extra trajectories added.",
                "Observed pool windows start at video size row 2 and end at row 48; do not infer identical 48-chunk simulation settings from episode count.",
                "Reward-gap labels are one-step constant-effective-rate proxies, not exact six-action replay or long-horizon Q values.",
                "Terminal rows retain hard action supervision and mask only the missing outcome supervision."],
            "files_sha256": {n: digest(staging / n) for n in ("base.jsonl", "cost.jsonl")},
            "data_code_sha256": data_code_hashes()}
        manifest["fingerprint"] = fingerprint(manifest)
        atomic_json(staging / "manifest.json", manifest)
        if destination.exists():
            raise ValueError("Destination appeared during preparation")
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--out", type=Path, default=DATA)
    parser.add_argument("--video", type=Path, default=VIDEO)
    args = parser.parse_args()
    result = prepare(args.pool, args.out, args.video)
    print(json.dumps({k: result[k] for k in ("version", "fingerprint", "episodes", "counts", "max_chosen_reward_reconstruction_error")}, indent=2))
