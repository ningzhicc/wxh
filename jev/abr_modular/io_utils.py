"""Small file helpers shared by data preparation and the resumable trainer."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
JEV = ROOT.parent
DATA = ROOT / "data/merina656_v1"
POOL = Path("/home/amora/NetLLM/adaptive_bitrate_streaming/artifacts/exp_pools/merina_merged_logits.pkl")
POOL_SHA256 = "e5d1caef67d4f338d8e49b0b9e07b3326d12b4603a4b04cb171e9846fe302ca1"
VIDEO = Path("/home/amora/NetLLM/adaptive_bitrate_streaming/data/videos/video1_sizes")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def data_code_hashes():
    paths = [ROOT / n for n in ("io_utils.py", "pool_data.py", "regret.py", "prepare.py")]
    paths += [JEV / "abr_candidate/features.py", JEV / "abr_alignment/audit_provenance.py"]
    return {str(p.relative_to(JEV)): digest(p) for p in paths}


def verify_data(directory=DATA):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if fingerprint(unsigned) != manifest["fingerprint"]:
        raise ValueError("Data manifest changed")
    if manifest["pool_sha256"] != POOL_SHA256:
        raise ValueError("This experiment requires the selected merina656 pool")
    for name, expected in manifest["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Prepared data changed: {name}")
    if manifest["data_code_sha256"] != data_code_hashes():
        raise ValueError("Data recipe changed; prepare a new data version")
    return manifest
