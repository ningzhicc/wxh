"""Check that the local service returns a complete six-bitrate distribution."""
import json
import math
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
body = (root / "abr_request.json").read_bytes()
request = urllib.request.Request(
    "http://127.0.0.1:8009/v1/systemone", data=body,
    headers={"Content-Type": "application/json"}, method="POST",
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
started = time.perf_counter()
with opener.open(request, timeout=180) as response:
    result = json.load(response)
elapsed_ms = (time.perf_counter() - started) * 1000
answer = result["answers"]["bitrate"]
probabilities = answer["probabilities"]
expected = set(json.loads(body)["questions"]["bitrate"]["criteria"])
assert set(probabilities) == expected, probabilities
assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
assert math.isclose(sum(probabilities.values()), 1, abs_tol=0.03)
assert answer["choice"] == max(probabilities, key=probabilities.get)
result["client_elapsed_ms"] = round(elapsed_ms, 2)
(root / "runtime/abr_response.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
