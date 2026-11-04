"""Download the released Kev checkpoint and its exact base revision."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import snapshot_download
from kev.checkpoint import Checkpoint

checkpoint = Checkpoint("jaredpalmer/kev-4b@v1.0")
print(f"Checkpoint: {checkpoint.path}", flush=True)
print(f"Base: {checkpoint.meta.base}@{checkpoint.meta.base_revision}", flush=True)
base_path = snapshot_download(
    checkpoint.meta.base,
    revision=checkpoint.meta.base_revision,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    max_workers=2,
    local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
)
manifest = {
    "source": "https://github.com/jaredpalmer/kev",
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT / "kev", text=True
    ).strip(),
    "checkpoint": checkpoint.requested,
    "checkpoint_revision": Path(checkpoint.path).name,
    "checkpoint_path": checkpoint.path,
    "base": checkpoint.meta.base,
    "base_revision": checkpoint.meta.base_revision,
    "base_path": base_path,
    "temperature": checkpoint.meta.temperature,
    "adapter_sha256": checkpoint.weights_sha256(),
}
(ROOT / "deployment.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
