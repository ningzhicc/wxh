"""Copy exact locked CUDA dependencies from existing local Python 3.12 installs."""
import base64
import csv
import email
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent
venv = root / "kev/.venv"
target = venv / "lib/python3.12/site-packages"
target.mkdir(parents=True, exist_ok=True)
lock = tomllib.loads((root / "kev/uv.lock").read_text())
wanted = {p["name"]: p["version"] for p in lock["package"]
          if p["name"].startswith("nvidia-") or p["name"] == "triton"}
copied = {}
for environment in sorted(Path("/home/amora/anaconda3/envs").iterdir()):
    for metadata in environment.glob("lib/python3.12/site-packages/*.dist-info/METADATA"):
        if not (metadata.parent.name.startswith("nvidia_") or metadata.parent.name.startswith("triton-")):
            continue
        fields = email.message_from_string(metadata.read_text())
        name, version = fields["Name"].lower().replace("_", "-"), fields["Version"]
        if name in copied or wanted.get(name) != version:
            continue
        source_root = metadata.parent.parent
        record = metadata.parent / "RECORD"
        if not record.exists():
            continue
        with record.open(newline="") as handle:
            rows = list(csv.reader(handle))
        for relative, expected_hash, _ in rows:
            source = source_root / relative
            destination = (target / relative).resolve()
            if not destination.is_relative_to(venv.resolve()):
                raise ValueError(f"Path outside the virtual environment: {relative}")
            if not source.is_file():
                raise ValueError(f"Missing installed dependency file: {source}")
            if expected_hash:
                algorithm, encoded = expected_hash.split("=", 1)
                with source.open("rb") as handle:
                    actual = base64.urlsafe_b64encode(hashlib.file_digest(handle, algorithm).digest()).decode().rstrip("=")
                if actual != encoded:
                    raise ValueError(f"Installed file differs from its wheel RECORD: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied[name] = {"version": version, "source": str(source_root)}
        print(f"Copied {name}=={version} from {environment.name}", flush=True)
(root / "reused_dependencies.json").write_text(json.dumps(copied, indent=2) + "\n")
print("Still to download:", sorted(set(wanted) - set(copied)))
