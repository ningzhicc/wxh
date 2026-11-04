"""JSONL adapter for the existing legacy-v4 DyLas checkpoint family; no environment."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import traceback
import types

from common import file_hash


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--action-mode", choices=["native", "sample", "argmax"], default="native")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo, checkpoint = args.repo.resolve(), args.checkpoint.resolve()
    # The upstream helper hardcodes this architecture. Refuse other variants instead of partially loading weights.
    name = checkpoint.parent.name
    if not (name.startswith("abrllm_v4_rank_") and "_sfd_512_" in name and "_sahd_2048_" in name and "_ld16_" in name):
        raise ValueError("Adapter supports legacy v4/sfd512/sahd2048/ld16 checkpoints only; other variants need an explicit loader")
    sys.path.insert(0, str(repo))
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from plm_special.data.dataset import ExperienceDataset
        helper_path = repo / "run/benchmark_dylas_inference_memory.py"
        spec = importlib.util.spec_from_file_location("dylas_existing_loader", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with args.pool.open("rb") as stream:
            pool = pickle.load(stream)
        info = ExperienceDataset(pool, gamma=1.0, scale=1000, max_length=20, sample_step=15).exp_dataset_info
        target_initial = float(info["max_return"])
        low, high = float(info["min_reward"]), float(info["max_reward"])
        if not high > low:
            raise ValueError("Invalid reward normalization")
        model, adapter_config = helper.load_model(str(checkpoint), history_length=20, device=torch.device(args.device))
        if args.action_mode == "argmax":
            def select_max(self, logits):
                p = torch.softmax(logits, 0)
                index = int(torch.argmax(p).item())
                return index, float(torch.log(p[index] + 1e-8).item())
            model._sample = types.MethodType(select_max, model)
    source_names = ["ABRLLM_v4.py", "config.py", "run/benchmark_dylas_inference_memory.py",
                    "plm_special/data/dataset.py", "plm_special/data/exp_pool.py"]
    metadata = {"type": "dylas", "implementation": "existing_legacy_v4_sample_API",
                "action_mode": "argmax" if args.action_mode == "argmax" else "sample",
                "repo": str(repo), "checkpoint": str(checkpoint), "pool": str(args.pool.resolve()),
                "pool_sha256": file_hash(args.pool), "rank": adapter_config["r"], "history": 20,
                "source_sha256": {n: file_hash(repo / n) for n in source_names},
                "checkpoint_sha256": {n: file_hash(checkpoint / n) for n in
                                       ["adapter_config.json", "adapter_model.safetensors", "modules_except_plm.bin"]},
                "conditioning": {"target_return": target_initial, "min_reward": low, "max_reward": high,
                                 "update": "clipped_minmax_reward_div1000_after_first_chunk"},
                "torch": torch.__version__, "device": args.device,
                "selection_status": "historical_checkpoint_provenance_not_certified_by_this_adapter"}
    emit({"metadata": metadata})
    state, target, expected_step = None, None, None
    for line in sys.stdin:
        message = json.loads(line)
        try:
            if message["operation"] == "reset":
                seed = message["seed"]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model.clear_dq()
                state, target, expected_step = np.zeros((6, 6), dtype=np.float32), target_initial, 0
                emit({"ok": True})
                continue
            context = message["context"]
            if state is None or context["chunk_index"] != expected_step:
                raise ValueError("Missing reset or nonsequential observation")
            row = context["history"][-1]
            state = np.roll(state, -1, axis=-1)
            state[0, -1] = row["bitrate_kbps"] / 4300
            state[1, -1] = context["buffer_s"] / 10
            state[2, -1] = row["size_bytes"] / row["download_ms"] / 1000
            state[3, -1] = row["download_ms"] / 1000 / 10
            state[4, :] = np.asarray(context["next_sizes"], dtype=np.float32) / 1e6
            state[5, -1] = min(context["remaining"], 48) / 48
            if expected_step > 0:
                target -= (min(high, max(low, row["reward"])) - low) / (high - low) / 1000
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                tensor = torch.tensor(state, dtype=torch.float32, device=args.device).reshape(1, 1, 6, 6)
                action = model.sample(tensor, target, expected_step)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
            expected_step += 1
            emit({"action": int(action)})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"error": f"{type(error).__name__}: {error}"})
        raise SystemExit(1)
