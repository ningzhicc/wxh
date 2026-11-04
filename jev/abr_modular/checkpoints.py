"""Atomic LoRA snapshots and two rotating, complete optimizer/RNG resume points."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
import shutil
import tempfile

from io_utils import atomic_json, digest


def publish(parent, name, model, tokenizer, meta, info, state=None):
    import torch
    from kev.checkpoint import write_meta
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / name
    if destination.exists():
        marker = validate_checkpoint(destination, info["fingerprint"])
        if marker["step"] != info["step"]:
            raise ValueError("Existing checkpoint belongs to a different position")
        return destination
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.pending-", dir=parent))
    # Interrupted staging is deliberately preserved for inspection.
    model.lm.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    saved = dataclasses.replace(meta, head={k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
                                extra={**meta.extra, "snapshot": info})
    write_meta(staging, saved)
    if state is not None:
        torch.save(state, staging / "training_state.pt")
    hashes = {p.name: digest(p) for p in staging.iterdir() if p.is_file()}
    atomic_json(staging / "complete.json", {**info, "files_sha256": hashes})
    staging.rename(destination)
    return destination


def validate_checkpoint(path, expected_fingerprint):
    path = Path(path)
    marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if marker["fingerprint"] != expected_fingerprint:
        raise ValueError("Checkpoint declaration differs from this run")
    for name, expected in marker["files_sha256"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Incomplete or modified checkpoint file: {path / name}")
    return marker


def latest_resume(run, expected_fingerprint):
    run = Path(run)
    candidates = sorted(p for p in (run / "resume").glob("step_*")
                        if p.is_dir() and re.fullmatch(r"step_\d{8}", p.name) and (p / "complete.json").is_file())
    if not candidates:
        raise ValueError("No complete resume point; use a new run name to restart")
    selected = candidates[-1]
    validate_checkpoint(selected, expected_fingerprint)
    if not (selected / "training_state.pt").is_file():
        raise ValueError("Optimizer/RNG state missing")
    return selected


def prune_resume(run, expected_fingerprint, keep=2):
    """Delete only this run's verified, explicitly rotating resume points; epoch models stay."""
    root = Path(run) / "resume"
    points = sorted(p for p in root.glob("step_*") if p.is_dir() and re.fullmatch(r"step_\d{8}", p.name)
                    and (p / "complete.json").is_file())
    if keep < 2:
        raise ValueError("Keep at least two resume points")
    for path in points[:-keep]:
        validate_checkpoint(path, expected_fingerprint)
        shutil.rmtree(path)
