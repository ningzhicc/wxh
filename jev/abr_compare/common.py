"""Immutable data bundles and one simulation/scoring contract for both policies."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "bundles/common_v1"
DEFAULT_DYLAS = Path("/home/amora/NetLLM/adaptive_bitrate_streaming")
DEFAULT_JEV = ROOT.parent / "abr_eval"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trace_files(directory):
    return sorted(p for p in Path(directory).iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix != ".pkl")


def load_numeric(path):
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid trace: {path}")
    if values[0, 0] < 0 or (np.diff(values[:, 0]) < 0).any() or (values[:, 1] < 0).any():
        raise ValueError(f"Invalid time/bandwidth values: {path}")
    if not np.any((np.diff(values[:, 0]) > 0) & (values[1:, 1] > 0)):
        raise ValueError(f"No positive transmission opportunity: {path}")
    return values


def numeric_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def source_parent(name, dataset):
    """Source inference is explicit and conservative; unknown names are not FCC."""
    match = re.search(r"(?:fcc_)?trace_(\d+)_", name)
    if match:
        return "fcc-acquisition:" + match.group(1)
    normalized = name.removeprefix("norway_")
    if "-report." in normalized and ".log" in normalized:
        return "norway-log:" + normalized.split(".log", 1)[0] + ".log"
    return f"unresolved:{dataset}:{name}"


def round_seed(seed, round_index):
    return int(seed) if round_index == 0 else int(np.random.SeedSequence([seed, round_index]).generate_state(1)[0])


def action_seed(seed, dataset, name, round_index):
    data = f"{seed}:{dataset}:{name}:{round_index}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def code_hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(ROOT.glob("*.py"))
            if not p.name.startswith("test_")}


def validate_settings(settings):
    """This version deliberately supports one explicit contract, not silent overrides."""
    required = {"chunks": 48, "chunk_duration_seconds": 4.0, "bitrates_kbps": [300, 750, 1200, 1850, 2850, 4300],
                "initial_action": 0, "buffer_limit_seconds": 60.0, "link_rtt_ms": 80,
                "packet_payload_fraction": .95, "fixed_simulation": True, "charge_decision_latency": False,
                "primary_reward_slice": "1:-1", "auxiliary_reward_slice": "1:", "observation_history": 6,
                "trace_wrap": True, "training_source": "fcc_acquisition_only"}
    for key, value in required.items():
        if settings.get(key) != value:
            raise ValueError(f"Unsupported protocol override {key}; explicitly version the runner first")
    for key in ("evaluation_rounds", "training_rounds"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if settings["baselines"]["mpc_horizon"] not in range(1, 6):
        raise ValueError("Unsupported MPC horizon")


def prepare(bundle=DEFAULT_BUNDLE, dylas=DEFAULT_DYLAS, jev=DEFAULT_JEV):
    bundle, dylas, jev = map(lambda p: Path(p).resolve(), (bundle, dylas, jev))
    if bundle.exists():
        verify(bundle)
        return json.loads((bundle / "manifest.json").read_text())
    settings = json.loads((ROOT / "protocol.json").read_text())
    validate_settings(settings)
    source_dirs = {
        "test": {"fcc": dylas / "data/traces/test/fcc-test",
                 "norway3g": dylas / "data/traces/test/Norway3G-test",
                 "oboe": dylas / "data/traces/test/Oboe-test",
                 "hsr": dylas / "data/traces/test/hsr-test"},
        "train": {"fcc": dylas / "data/traces/train/fcc-train"},
        "valid": {"fcc": dylas / "data/traces/valid/fcc-valid"},
    }
    mirror_dirs = {"fcc": jev / "vendor/NetLLM/adaptive_bitrate_streaming/data/traces/test/fcc-test",
                   "norway3g": jev / "vendor/ABRBench/trace/ABRBench-4G+/Norway3G/test",
                   "oboe": jev / "vendor/ABRBench/trace/ABRBench-3G/Oboe/test",
                   "hsr": jev / "vendor/ABRBench/trace/ABRBench-3G/HSR"}
    candidates = []
    mirror_audit = {}
    for split, datasets in source_dirs.items():
        for dataset, directory in datasets.items():
            paths = trace_files(directory)
            if split == "test":
                if len(paths) != settings["expected_test_counts"][dataset]:
                    raise ValueError(f"Unexpected test count: {dataset}")
                mirrors = {p.name: p for p in trace_files(mirror_dirs[dataset])}
                if set(mirrors) != {p.name for p in paths}:
                    raise ValueError(f"Different test filenames: {dataset}")
                mirror_audit[dataset] = {"count": len(paths), "numeric_matches": 0}
            for path in paths:
                values = load_numeric(path)
                record = {"split": split, "dataset": dataset, "name": path.name,
                          "source_path": str(path), "source_sha256": file_hash(path),
                          "numeric_sha256": numeric_hash(values), "samples": len(values),
                          "source_parent": source_parent(path.name, dataset)}
                if split == "test":
                    mirror = mirrors[path.name]
                    if numeric_hash(load_numeric(mirror)) != record["numeric_sha256"]:
                        raise ValueError(f"Test mirror differs: {path.name}")
                    record["jev_mirror_path"] = str(mirror)
                    record["jev_mirror_sha256"] = file_hash(mirror)
                    mirror_audit[dataset]["numeric_matches"] += 1
                candidates.append(record)
    test = [r for r in candidates if r["split"] == "test"]
    test_parents = {r["source_parent"] for r in test}
    test_hashes = {r["numeric_sha256"] for r in test}
    # Preserve official validation assignment; never move a test record into training.
    all_valid = [r for r in candidates if r["split"] == "valid"]
    valid_parents_all = {r["source_parent"] for r in all_valid}
    valid_hashes_all = {r["numeric_sha256"] for r in all_valid}
    retained, excluded, seen_train, seen_valid = [], [], set(), set()
    for record in candidates:
        reason = None
        if record["split"] != "test":
            if not record["source_parent"].startswith("fcc-acquisition:"):
                reason = "not_verified_FCC_acquisition"
            elif record["source_parent"] in test_parents:
                reason = "source_parent_shared_with_test"
            elif record["numeric_sha256"] in test_hashes:
                reason = "numeric_duplicate_of_test"
            elif record["split"] == "train" and record["source_parent"] in valid_parents_all:
                reason = "source_parent_reserved_for_validation"
            elif record["split"] == "train" and record["numeric_sha256"] in valid_hashes_all:
                reason = "numeric_duplicate_of_validation"
            else:
                seen = seen_train if record["split"] == "train" else seen_valid
                if record["numeric_sha256"] in seen:
                    reason = "duplicate_within_split"
                seen.add(record["numeric_sha256"])
        if reason:
            excluded.append({**record, "reason": reason})
        else:
            retained.append(record)
    if not any(r["split"] == "train" for r in retained) or not any(r["split"] == "valid" for r in retained):
        raise ValueError("No clean training or validation data")
    audit = {"test_mirror_audit": mirror_audit, "excluded": excluded, "counts": {}, "source_counts": {}}
    for split in ("train", "valid", "test"):
        subset = [r for r in retained if r["split"] == split]
        audit["counts"][split] = len(subset)
        audit["source_counts"][split] = len({r["source_parent"] for r in subset})
    audit["limitations"] = [
        "Source identity is inferred from acquisition IDs/log names, not independently proven.",
        "Historical checkpoints were not necessarily trained on this split; evaluation is diagnostic by default.",
        "Only two independent original FCC acquisitions remain in official validation; repeated windows do not increase that count.",
        "This local 48-chunk/three-start protocol does not reproduce the paper's 49-chunk/100-session description.",
    ]
    staging = bundle.with_name(bundle.name + ".preparing")
    if staging.exists():
        raise ValueError(f"Partial preparation exists; inspect before retrying: {staging}")
    staging.mkdir(parents=True)
    shutil.copyfile(ROOT / "protocol.json", staging / "protocol.json")
    # Canonical numerical snapshots remain usable if either external checkout moves.
    for record in retained:
        relative = Path("traces") / record["split"] / record["dataset"] / (record["numeric_sha256"] + ".txt")
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(target, load_numeric(record["source_path"]), fmt="%.17g", delimiter="\t")
        record["snapshot"] = str(relative)
        record["snapshot_sha256"] = file_hash(target)
    for split in ("train", "valid", "test"):
        for dataset in sorted({r["dataset"] for r in retained if r["split"] == split}):
            group = sorted((r for r in retained if r["split"] == split and r["dataset"] == dataset), key=lambda r: r["name"])
            rounds = settings["training_rounds"] if split == "train" else settings["evaluation_rounds"]
            for record in group:
                record["sessions"] = []
                record["id"] = f"{split}/{dataset}/{hashlib.sha256(record['name'].encode()).hexdigest()[:16]}"
            for index in range(rounds):
                rng = np.random.RandomState(round_seed(settings["evaluation_seed"], index))
                for record in group:
                    pointer = int(rng.randint(1, record["samples"]))
                    record["sessions"].append({"round": index, "start_pointer": pointer,
                                               "action_seed": action_seed(settings["evaluation_seed"], dataset, record["name"], index)})
    videos = []
    video_dir = staging / "video1_sizes"
    video_dir.mkdir()
    for level in range(6):
        source = dylas / f"data/videos/video1_sizes/video_size_{level}"
        mirror = jev / f"vendor/NetLLM/adaptive_bitrate_streaming/data/videos/video1_sizes/video_size_{level}"
        if file_hash(source) != file_hash(mirror):
            raise ValueError(f"Different video sizes at level {level}")
        sizes = np.loadtxt(source, dtype=np.int64)
        if len(sizes) <= settings["chunks"] or (sizes <= 0).any():
            raise ValueError("Video must cover requested chunks and terminal next-size lookup")
        target = video_dir / source.name
        shutil.copyfile(source, target)
        videos.append({"snapshot": str(target.relative_to(staging)), "sha256": file_hash(target),
                       "source_path": str(source), "rows": len(sizes)})
    simulator = {}
    for name in ("env.py", "constants.py", "__init__.py", "source.json"):
        simulator[name] = file_hash(ROOT / "simulator" / name)
    manifest = {"schema_version": 1, "settings": settings, "protocol_sha256": file_hash(staging / "protocol.json"),
                "source_root": str(dylas), "jev_root": str(jev), "records": retained, "videos": videos,
                "simulator_sha256": simulator, "runtime_code_sha256": code_hashes(),
                "source_environment_sha256": file_hash(dylas / "baseline_special/env.py"), "audit": audit}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "split_audit.json", audit)
    for split in ("train", "valid", "test"):
        write_json(staging / f"{split}.json", [r for r in retained if r["split"] == split])
    staging.rename(bundle)
    verify(bundle)
    return manifest


def verify(bundle=DEFAULT_BUNDLE):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    validate_settings(manifest["settings"])
    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if fingerprint(content) != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch")
    if file_hash(bundle / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    if json.loads((bundle / "protocol.json").read_text()) != manifest["settings"]:
        raise ValueError("Protocol and manifest disagree")
    if code_hashes() != manifest["runtime_code_sha256"]:
        raise ValueError("Runtime code changed; create a newly versioned bundle")
    for name, digest in manifest["simulator_sha256"].items():
        if file_hash(ROOT / "simulator" / name) != digest:
            raise ValueError(f"Frozen simulator changed: {name}")
    parents, hashes = {}, {}
    for split in ("train", "valid", "test"):
        subset = [r for r in manifest["records"] if r["split"] == split]
        if json.loads((bundle / f"{split}.json").read_text()) != subset:
            raise ValueError(f"Split export changed: {split}")
        parents[split] = {r["source_parent"] for r in subset}
        hashes[split] = {r["numeric_sha256"] for r in subset}
        if split != "test" and any(not p.startswith("fcc-acquisition:") for p in parents[split]):
            raise ValueError("Non-FCC source in train/validation")
        for r in subset:
            path = bundle / r["snapshot"]
            if file_hash(path) != r["snapshot_sha256"] or numeric_hash(load_numeric(path)) != r["numeric_sha256"]:
                raise ValueError(f"Trace snapshot changed: {r['id']}")
            for session in r["sessions"]:
                if not 1 <= session["start_pointer"] < r["samples"]:
                    raise ValueError("Invalid frozen start pointer")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if parents[left] & parents[right] or hashes[left] & hashes[right]:
            raise ValueError(f"Source/numeric overlap: {left}/{right}")
    for video in manifest["videos"]:
        if file_hash(bundle / video["snapshot"]) != video["sha256"]:
            raise ValueError("Video snapshot changed")
    return manifest


def environment(bundle, record, session):
    from simulator.env import Environment
    values = load_numeric(Path(bundle) / record["snapshot"])
    return Environment(all_cooked_time=[values[:, 0].tolist()], all_cooked_bw=[values[:, 1].tolist()],
                       all_file_names=[record["name"]], all_mahimahi_ptrs=[session["start_pointer"]],
                       video_size_dir=str(Path(bundle) / "video1_sizes") + os.sep, fixed=True, trace_num=1,
                       chunk_duration_s=4.0, total_video_chunks=48)


def episode_metrics(rows, settings):
    if len(rows) != settings["chunks"]:
        raise ValueError("Incomplete episode")
    rewards = [r["reward"] for r in rows]
    return {"qoe_primary": float(np.mean(rewards[1:-1])),
            "qoe_skip_first": float(np.mean(rewards[1:])), "qoe_all": float(np.mean(rewards)),
            "scored_chunks_primary": len(rows) - 2,
            "mean_bitrate_mbps": float(np.mean([r["bitrate_kbps"] / 1000 for r in rows[1:-1]])),
            "rebuffer_s_primary": sum(r["rebuffer_s"] for r in rows[1:-1]),
            "rebuffer_s_after_startup": sum(r["rebuffer_s"] for r in rows[1:]),
            "smoothness_mbps_primary": float(np.mean([r["switch_mbps"] for r in rows[1:-1]])),
            "decision_latency_mean_ms": float(np.mean([r["decision_ms"] for r in rows[1:]])),
            "decision_latency_p95_ms": float(np.quantile([r["decision_ms"] for r in rows[1:]], .95))}
