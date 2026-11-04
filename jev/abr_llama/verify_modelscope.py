"""Verify the downloaded ModelScope snapshot and record its provenance."""
import hashlib
import importlib.metadata
import json
import math
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "models/Llama-3.2-1B"
REPO = "LLM-Research/Llama-3.2-1B"
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT.parent / ".cache/modelscope"))


def main():
    from modelscope_hub import HubApi
    entries = HubApi().list_repo_files(REPO, "model", revision="master", recursive=True)
    rows = []
    for entry in entries:
        if entry.is_dir:
            continue
        local = (MODEL / entry.path).resolve()
        if not local.is_relative_to(MODEL.resolve()) or not local.is_file():
            raise SystemExit(f"Missing or invalid file: {entry.path}")
        with local.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if not entry.sha256 or actual != entry.sha256 or local.stat().st_size != entry.size:
            raise SystemExit(f"Size/checksum mismatch or unavailable checksum: {entry.path}")
        rows.append({"path": entry.path, "bytes": entry.size, "sha256": actual})
    weight = MODEL / "model.safetensors"
    with weight.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_bytes < 16 * 1024 * 1024:
            raise SystemExit("Invalid safetensors header length")
        header = json.loads(stream.read(header_bytes))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if max(v["data_offsets"][1] for v in tensors.values()) != weight.stat().st_size - 8 - header_bytes:
        raise SystemExit("Safetensors payload size mismatch")
    cfg = json.loads((MODEL / "config.json").read_text())
    if cfg["model_type"] != "llama" or cfg["hidden_size"] != 2048 or cfg["num_hidden_layers"] != 16:
        raise SystemExit("Unexpected Llama-3.2-1B architecture")
    if tensors["model.embed_tokens.weight"]["shape"] != [cfg["vocab_size"], cfg["hidden_size"]]:
        raise SystemExit("Embedding shape does not match config")
    receipt = {"passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPO, "source_url": f"https://modelscope.cn/models/{REPO}",
        "revision": "master", "local_directory": str(MODEL),
        "modelscope_version": importlib.metadata.version("modelscope"),
        "file_count": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
        "all_files_sha256_match_registry": True, "files": rows,
        "safetensors": {"tensor_count": len(tensors),
            "stored_parameter_elements": sum(math.prod(v["shape"]) for v in tensors.values()),
            "dtypes": sorted({v["dtype"] for v in tensors.values()})},
        "training_started": False}
    target = ROOT / "model_download.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
