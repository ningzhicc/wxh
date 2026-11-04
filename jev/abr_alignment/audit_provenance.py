"""Read-only forensic inventory of current and Git-retained ABR experience pools."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np


class Pool:
    pass


class RestrictedPoolReader(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ExperiencePool" and module in ("plm_special.data.exp_pool", "exp_pool", "__main__"):
            return Pool
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(importlib.import_module("numpy._core.multiarray"), name)
        raise pickle.UnpicklingError(f"Unexpected pickle class: {module}.{name}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def inspect_pool(data):
    pool = RestrictedPoolReader(io.BytesIO(data)).load()
    fields = vars(pool) if not isinstance(pool, dict) else pool
    lengths = {key: len(fields.get(key, [])) for key in ("states", "actions", "rewards", "dones", "teacher_logits")}
    ends = [i + 1 for i, done in enumerate(fields.get("dones", [])) if bool(done)]
    episode_lengths = np.diff([0] + ends).tolist()
    hashes = []
    start = 0
    for end in ends:
        h = hashlib.sha256()
        for key in ("states", "actions", "rewards", "dones"):
            arr = np.asarray(fields[key][start:end], dtype="<f8")
            h.update(key.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        hashes.append(h.hexdigest())
        start = end
    return {"sha256": sha(data), "bytes": len(data), "stored_fields": sorted(fields),
            "field_lengths": lengths, "complete_episodes": len(ends),
            "episode_length_counts": dict(sorted(Counter(episode_lengths).items())),
            "trailing_steps": lengths["states"] - (ends[-1] if ends else 0),
            "first_state_shape": list(np.asarray(fields["states"][0]).shape) if lengths["states"] else None,
            "legacy_windows_w20_stride15": len(range(0, lengths["states"] - 20 + 1, 15)),
            "episode_numeric_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/amora/NetLLM"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = "adaptive_bitrate_streaming/artifacts/exp_pools"
    rows, sources = [], {}
    # Reachable historical blobs, including pools absent from today's working tree.
    objects = git(args.repo, "rev-list", "--objects", "--all", "--", prefix).decode().splitlines()
    for line in objects:
        pieces = line.split(" ", 1)
        if len(pieces) != 2 or not pieces[1].endswith((".pkl", ".pickle")):
            continue
        oid, path = pieces
        if git(args.repo, "cat-file", "-t", oid).strip() != b"blob":
            continue
        data = git(args.repo, "cat-file", "blob", oid)
        try:
            info = inspect_pool(data)
        except Exception as error:
            info = {"sha256": sha(data), "bytes": len(data), "read_error": f"{type(error).__name__}: {error}"}
        row = {"kind": "git_blob", "git_oid": oid, "historical_path": path, **info}
        rows.append(row)
        if info.get("complete_episodes") == 600:
            target = args.out / f"candidate600_{oid}.pkl"
            target.write_bytes(data)
            row["exported_candidate"] = str(target.resolve())
        print(json.dumps({k: row[k] for k in ("historical_path", "complete_episodes", "legacy_windows_w20_stride15", "read_error") if k in row}), flush=True)
    current_paths = list((args.repo / "adaptive_bitrate_streaming").rglob("*.pkl"))
    current_paths += list((args.repo / "AAAI rebuttal").rglob("*.pkl"))
    vendor = Path("/home/amora/wxh/jev/abr_eval/vendor/NetLLM/adaptive_bitrate_streaming/artifacts/exp_pools/exp_pool.pkl")
    if vendor.exists():
        current_paths.append(vendor)
    for path in sorted(current_paths):
        if "mahimahi" in path.name or not ("pool" in path.name.lower() or "exp_pools" in str(path.parent)):
            continue
        data = path.read_bytes()
        try:
            info = inspect_pool(data)
        except Exception as error:
            info = {"sha256": sha(data), "bytes": len(data), "read_error": f"{type(error).__name__}: {error}"}
        rows.append({"kind": "working_file", "path": str(path), **info})
    # Preserve evidence snapshots: exact scripts, dataset definition, and result provenance.
    paths = ["adaptive_bitrate_streaming/bash/run_abr_v4_train.sh",
             "adaptive_bitrate_streaming/plm_special/data/dataset.py",
             "adaptive_bitrate_streaming/run/run_abr.py",
             "adaptive_bitrate_streaming/artifacts/qoe_cdf_columns/video1/abrllm_v4_all_traces_manifest.json"]
    for path in paths:
        data = (args.repo / path).read_bytes()
        target = args.out / "source_snapshots" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        sources[path] = {"sha256": sha(data), "snapshot": str(target.relative_to(args.out))}
    logs = []
    for path in sorted((args.repo / "adaptive_bitrate_streaming/data/ft_plms").rglob("console.log")):
        data = path.read_bytes()
        lines = data.decode(errors="replace").splitlines()
        hits = [{"line": i, "text": line} for i, line in enumerate(lines, 1)
                if re.search(r"Dataset size:|Steps per epoch:|Total training steps:", line)]
        target = args.out / "source_snapshots" / path.relative_to(args.repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        logs.append({"path": str(path), "sha256": sha(data), "snapshot": str(target.relative_to(args.out)), "hits": hits})
    unique = {}
    for row in rows:
        unique.setdefault(row["sha256"], row)
    report = {"repo": str(args.repo), "git_head": git(args.repo, "rev-parse", "HEAD").decode().strip(),
              "git_reachable_pool_blob_count": sum(row["kind"] == "git_blob" for row in rows),
              "working_pool_file_count": sum(row["kind"] == "working_file" for row in rows),
              "unique_pool_file_hashes": len(unique), "pools": rows, "training_logs": logs, "source_snapshots": sources,
              "numeric_episode_hash_method": "SHA256 over field name, array shape and little-endian float64 bytes for states/actions/rewards/dones; teacher_logits excluded",
              "identity_rule": "Matching counts or numeric content alone do not link a pool to the manuscript's training run"}
    (args.out / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"git_blobs": report["git_reachable_pool_blob_count"], "working_files": report["working_pool_file_count"],
                      "unique_pool_hashes": len(unique), "600_episode_candidates": [r.get("historical_path", r.get("path")) for r in rows if r.get("complete_episodes") == 600]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
