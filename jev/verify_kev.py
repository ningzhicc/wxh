"""Start the service and verify GPU serving, ABR output and the official API."""
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from kev_service import LOG_FILE, ROOT, managed_pid, models

assert torch.cuda.is_available(), "CUDA is unavailable in the deployment environment"
report = {
    "gpu": torch.cuda.get_device_name(0),
    "torch_cuda": torch.version.cuda,
    "versions": {name: importlib.metadata.version(name)
                 for name in ["torch", "transformers", "peft", "fastapi", "typesafe-sdk", "triton"]},
}
print(json.dumps(report, indent=2), flush=True)
subprocess.run([sys.executable, str(ROOT / "kev_service.py"), "start"], check=True)
deadline = time.monotonic() + 300
while True:
    try:
        report["models"] = models()
        break
    except OSError:
        if managed_pid() is None:
            print(LOG_FILE.read_text(encoding="utf-8", errors="replace")[-10000:])
            raise SystemExit("Kev exited during startup")
        if time.monotonic() > deadline:
            raise SystemExit(f"Startup timed out; inspect {LOG_FILE}")
        print("Waiting for model loading...", flush=True)
        time.sleep(10)
print("API ready. Checking six bitrate options...", flush=True)
report["abr"] = []
for _ in range(2):
    subprocess.run([sys.executable, str(ROOT / "check_abr.py")], check=True)
    report["abr"].append(json.loads((ROOT / "runtime/abr_response.json").read_text()))
env = os.environ.copy()
env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", KEV_BASE_URL="http://127.0.0.1:8009")
print("Running official API tests...", flush=True)
tests = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_api.py", "-q"],
    cwd=ROOT / "kev", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
(ROOT / "runtime/api_tests.log").write_text(tests.stdout, encoding="utf-8")
print(tests.stdout, flush=True)
report["api_tests"] = {"returncode": tests.returncode, "output": tests.stdout}
(ROOT / "runtime/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
raise SystemExit(tests.returncode)
