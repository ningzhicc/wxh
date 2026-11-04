"""Create paired baseline/candidate-cost JSONL without altering labels or source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import shutil

from features import VERSION, transform_request


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination, limit=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Output exists; choose a new directory")
    if limit is not None and limit < 1:
        raise ValueError("Preview limit must be positive")
    initial_hash = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=destination.parent))
    try:
        count = 0
        with source.open() as stream, (temporary / "base.jsonl").open("x") as base, (temporary / "cost.jsonl").open("x") as cost:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "label" not in row["questions"]["bitrate"]:
                    raise ValueError("Expected existing supervision; no new labels are generated")
                if row["questions"]["bitrate"]["label"] not in row["questions"]["bitrate"]["criteria"]:
                    raise ValueError("Label is not one of the candidates")
                augmented = transform_request(row)
                base.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                cost.write(json.dumps(augmented, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
        if not count or digest(source) != initial_hash:
            raise ValueError("Source empty or changed during conversion")
        manifest = {"version": VERSION, "source": str(source), "source_sha256": initial_hash,
                    "records": count, "limited_preview": limit is not None,
                    "source_training_alignment_verified": False,
                    "note": "Conversion only; inherits original split. No new teachers or states. Not certification of DyLas training alignment.",
                    "files_sha256": {n: digest(temporary / n) for n in ("base.jsonl", "cost.jsonl")},
                    "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if destination.exists():
            raise ValueError("Output appeared during conversion")
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.out, args.preview_limit), ensure_ascii=False, indent=2))
