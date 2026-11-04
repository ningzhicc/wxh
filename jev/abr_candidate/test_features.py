import copy
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import BITRATES, candidate_costs, transform_request
from prepare import prepare
from run_compare import CandidateHTTP, compare
from common import DEFAULT_BUNDLE, verify


def request():
    return {"model": "kev-latest", "state": {
        "recent_throughput_mbps": [2, 2], "buffer_seconds": 3,
        "previous_bitrate_kbps": 750,
        "next_chunk_sizes_bytes": {str(b): 1_000_000 for b in BITRATES}},
        "questions": {"bitrate": {"type": "choice", "instructions": "Select.",
                                  "criteria": {str(b): f"{b} kbps" for b in BITRATES}, "label": "750"}},
        "_meta": {"id": "same-row", "split": "valid"}}


def test_units_and_switch_cost():
    costs = candidate_costs(request()["state"])
    assert costs["750"] == {"size_bytes": 1_000_000, "est_download_s": 4,
                            "est_buffer_margin_s": -1, "switch_cost_qoe": 0}
    assert costs["4300"]["switch_cost_qoe"] == 3.55


def test_history_only_and_no_label_dependency():
    row = request()
    row["state"]["recent_throughput_mbps"] = [1000, 1, 2, 4, 8, 16]
    a = transform_request(row)
    row["state"]["recent_throughput_mbps"][0] = 0.0001
    row["questions"]["bitrate"]["label"] = "4300"
    row["_meta"]["future_bandwidth"] = [999]
    b = transform_request(row)
    assert a["questions"]["bitrate"]["criteria"] == b["questions"]["bitrate"]["criteria"]
    assert candidate_costs(row["state"])["300"]["est_download_s"] == 3.1


def test_input_and_labels_preserved_and_no_duplicate_augmentation():
    row = request()
    original = copy.deepcopy(row)
    changed = transform_request(row)
    assert row == original and transform_request(row, "base") == original
    assert changed["state"] == original["state"] and changed["_meta"] == original["_meta"]
    assert changed["questions"]["bitrate"]["label"] == "750"
    with pytest.raises(ValueError, match="already"):
        transform_request(changed)


@pytest.mark.parametrize("history", [[], [0], [-1], [float("nan")]])
def test_invalid_network_input_rejected(history):
    state = request()["state"]
    state["recent_throughput_mbps"] = history
    with pytest.raises(ValueError):
        candidate_costs(state)


def test_paired_conversion_and_no_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(request()) + "\n")
    before = source.read_bytes()
    target = tmp_path / "pair"
    manifest = prepare(source, target)
    assert manifest["records"] == 1 and source.read_bytes() == before
    assert json.loads((target / "base.jsonl").read_text())["_meta"]["split"] == "valid"
    with pytest.raises(ValueError, match="exists"):
        prepare(source, target)


def test_mock_http_full_episode_keeps_frozen_physics():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_json(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.send_json({"models": [{"name": "kev-latest", "run": "/no-weights", "base": "mock"}]})
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            self.send_json({"answers": {"bitrate": {"choice": "300", "probabilities": {
                str(b): float(b == 300) for b in BITRATES}}}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = verify(DEFAULT_BUNDLE)
        record = next(r for r in manifest["records"] if r["split"] == "valid")
        episodes = []
        for mode in ("base", "cost"):
            policy = CandidateHTTP(f"http://127.0.0.1:{server.server_port}")
            policy.candidate_mode = mode
            episodes.append(compare.run_episode(DEFAULT_BUNDLE, record, record["sessions"][0], policy, manifest["settings"]))
            assert policy.metadata()["candidate_mode"] == mode
            policy.close()
        assert len(calls) == 94
        assert "est_download_s=" not in calls[0]["questions"]["bitrate"]["criteria"]["300"]
        assert "est_download_s=" in calls[47]["questions"]["bitrate"]["criteria"]["300"]
        assert calls[0]["state"] == calls[47]["state"]
        for a, b in zip(episodes[0]["rows"], episodes[1]["rows"]):
            for key in ("action", "download_ms", "buffer_s", "rebuffer_s", "reward", "elapsed_s"):
                assert a[key] == b[key]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
