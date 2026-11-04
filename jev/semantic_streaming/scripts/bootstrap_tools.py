#!/usr/bin/env python3
"""Restore the exact local FFmpeg binary; never install into a shared Python env."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    config = json.loads((ROOT / "configs/tools.lock.json").read_text())["imageio_ffmpeg"]
    target = ROOT / config["binary_path"]
    if target.exists():
        if sha(target.read_bytes()) != config["binary_sha256"]:
            raise ValueError("Existing FFmpeg differs from the pinned binary; refusing replacement")
        print("Pinned FFmpeg already present")
        return
    cache = ROOT / ".tools/wheels"
    cache.mkdir(parents=True, exist_ok=True)
    wheel = cache / config["wheel_filename"]
    if not wheel.exists():
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                        f"imageio-ffmpeg=={config['version']}", "-d", str(cache)], check=True)
    if sha(wheel.read_bytes()) != config["wheel_sha256"]:
        raise ValueError("Wheel checksum differs; check OS/architecture and package source")
    with zipfile.ZipFile(wheel) as archive:
        data = archive.read(config["wheel_member"])
    if sha(data) != config["binary_sha256"]:
        raise ValueError("FFmpeg binary checksum differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".part")
    temp.write_bytes(data)
    temp.chmod(0o755)
    temp.replace(target)
    print(target)


if __name__ == "__main__":
    main()

