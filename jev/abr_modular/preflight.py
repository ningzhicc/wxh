"""Audit paired data with the real Kev tokenizer/encoder, without loading weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from io_utils import DATA, JEV, atomic_json, digest, verify_data


def check(directory=DATA, max_state=512, output=None):
    started = time.monotonic()
    manifest = verify_data(directory)
    deployment = json.loads((JEV / "deployment.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(JEV / "kev"))
    from transformers import AutoTokenizer
    from kev.data import materialize, api_request
    from kev.model import encode, training_context
    token = AutoTokenizer.from_pretrained(deployment["base_path"], local_files_only=True)
    limits = training_context(max_state)
    counts = {"records": 0, "regret_valid": 0, "regret_masked": 0}
    maxima = {name: {"state": 0, "packed": 0} for name in ("base", "cost")}
    with (Path(directory) / "base.jsonl").open(encoding="utf-8") as a, (Path(directory) / "cost.jsonl").open(encoding="utf-8") as b:
        for first, second in zip(a, b, strict=True):
            base, cost = json.loads(first), json.loads(second)
            if base["state"] != cost["state"] or base["_meta"] != cost["_meta"] or base["_supervision"] != cost["_supervision"]:
                raise ValueError("Candidate module changed underlying experience")
            if base["questions"]["bitrate"]["label"] != cost["questions"]["bitrate"]["label"]:
                raise ValueError("Candidate module changed action label")
            if list(base["questions"]["bitrate"]["criteria"]) != list(cost["questions"]["bitrate"]["criteria"]):
                raise ValueError("Candidate order differs between variants")
            for name, request in (("base", base), ("cost", cost)):
                payload = api_request(request)
                if set(payload) != {"state", "questions"} or set(payload["questions"]["bitrate"]) != {"type", "instructions", "criteria"}:
                    raise ValueError("Training supervision leaked into model input")
                encoded = encode(token, materialize(request), max_state=limits["max_state"], max_branch=limits["max_branch"], strict=True)
                if len(encoded["ids"]) > limits["max_packed"]:
                    raise ValueError("Packed input too long; no silent sample filtering allowed")
                maxima[name]["state"] = max(maxima[name]["state"], encoded["state_tokens"])
                maxima[name]["packed"] = max(maxima[name]["packed"], len(encoded["ids"]))
            counts["records"] += 1
            counts["regret_valid" if base["_supervision"]["valid"] else "regret_masked"] += 1
            if counts["records"] % 2048 == 0:
                print(f"Context audit {counts['records']}/30832 paired records", flush=True)
    if counts != manifest["counts"]:
        raise ValueError("Counts differ from manifest")
    versions = {}
    for package in ("torch", "transformers", "peft", "einops", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if any(versions[n] is None for n in ("torch", "transformers", "peft")):
        raise ValueError("Training dependencies are missing")
    result = {"complete": True, "data_fingerprint": manifest["fingerprint"], "counts": counts,
              "tokenizer": deployment["base_path"], "tokenizer_sha256": digest(Path(deployment["base_path"]) / "tokenizer.json"),
              "limits": limits, "max_tokens": maxima, "dropped": 0, "truncated": 0,
              "versions": versions, "seconds": time.monotonic() - started,
              "model_weights_loaded": False, "training_started": False}
    verify_data(directory)
    atomic_json(output or Path(directory) / "preflight.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--max-state", type=int, default=512)
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()
    print(json.dumps(check(options.data_dir, options.max_state, options.out), indent=2))
