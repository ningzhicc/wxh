"""Manage this directory's local Kev service without changing system services."""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
PID_FILE = RUNTIME / "kev.pid"
LOG_FILE = RUNTIME / "kev.log"
BASE_URL = "http://127.0.0.1:8009"


def managed_pid():
    if not PID_FILE.exists():
        return None
    pid = int(PID_FILE.read_text().strip())
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None
    if str(ROOT / "kev/.venv/bin/python").encode() in command and b"kev.serve" in command:
        return pid
    return None


def models():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE_URL + "/v1/models", timeout=5) as response:
        return json.load(response)


def start(checkpoint=None):
    pid = managed_pid()
    if pid:
        print(f"Kev already running: PID {pid}; {BASE_URL}; log {LOG_FILE}")
        return
    try:
        models()
    except OSError:
        pass
    else:
        raise SystemExit("Port 8009 already serves another instance; refusing to start.")
    manifest = json.loads((ROOT / "deployment.json").read_text(encoding="utf-8"))
    checkpoint_path = str(Path(checkpoint).resolve()) if checkpoint else manifest["checkpoint_path"]
    if not (Path(checkpoint_path) / "head.pt").is_file():
        raise SystemExit(f"Checkpoint is incomplete: {checkpoint_path}")
    RUNTIME.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRITON_CACHE_DIR": str(ROOT / ".cache/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache/torchinductor"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"),
        "PYTHONUNBUFFERED": "1",
        "KEV_DTYPE": "bf16",
    })
    with LOG_FILE.open("ab") as log:
        process = subprocess.Popen(
            [str(ROOT / "kev/.venv/bin/python"), "-m", "kev.serve", "--run",
             checkpoint_path,
             "--device", "cuda", "--host", "127.0.0.1", "--port", "8009"],
            cwd=ROOT / "kev", env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    PID_FILE.write_text(str(process.pid) + "\n")
    print(f"Kev starting: PID {process.pid}; {BASE_URL}; log {LOG_FILE}")


def stop():
    pid = managed_pid()
    if pid is None:
        print("Kev is stopped.")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if managed_pid() is None:
            PID_FILE.unlink(missing_ok=True)
            print("Kev stopped.")
            return
        time.sleep(0.1)
    raise SystemExit(f"PID {pid} has not stopped yet; inspect {LOG_FILE}.")


def status():
    pid = managed_pid()
    print(f"Managed PID: {pid or 'stopped'}")
    try:
        print(json.dumps(models(), ensure_ascii=False, indent=2))
    except OSError as error:
        print(f"API not ready: {error}; log {LOG_FILE}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "stop", "status"])
    parser.add_argument("--checkpoint", help="optional local checkpoint to serve instead of released Kev-4B")
    args = parser.parse_args()
    if args.command == "start":
        start(args.checkpoint)
    else:
        {"stop": stop, "status": status}[args.command]()
