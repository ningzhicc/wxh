"""Scientific validity checks for the shared data/simulation/scoring boundary."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from compare import run_episode, summarize, report
from policies import Traditional, JevHTTP, request_for


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    path = tmp_path_factory.mktemp("common_bundle") / "v1"
    return path, common.prepare(path)


def test_real_split_source_and_numeric_isolation(prepared):
    path, manifest = prepared
    assert manifest["audit"]["counts"] == {"train": 16, "valid": 30, "test": 275}
    assert manifest["audit"]["source_counts"]["valid"] == 2
    for split in ("train", "valid"):
        assert all(r["source_parent"].startswith("fcc-acquisition:") for r in manifest["records"] if r["split"] == split)
    assert all(item["count"] == item["numeric_matches"] for item in manifest["audit"]["test_mirror_audit"].values())
    common.verify(path)


def test_prepare_is_idempotent_and_pointers_are_materialized(prepared):
    path, first = prepared
    assert common.prepare(path)["fingerprint"] == first["fingerprint"]
    for split, dataset in (("test", "fcc"), ("test", "hsr"), ("valid", "fcc")):
        group = sorted((r for r in first["records"] if r["split"] == split and r["dataset"] == dataset), key=lambda r: r["name"])
        for index in range(3):
            rng = np.random.RandomState(common.round_seed(666, index))
            for r in group:
                assert r["sessions"][index]["start_pointer"] == int(rng.randint(1, r["samples"]))


def test_mutated_data_and_protocol_are_rejected(prepared, tmp_path):
    bundle, manifest = prepared
    target = tmp_path / "copy"
    shutil.copytree(bundle, target)
    trace = target / manifest["records"][0]["snapshot"]
    trace.write_text(trace.read_text() + "0 1\n")
    with pytest.raises(ValueError, match="Trace snapshot changed"):
        common.verify(target)
    shutil.copyfile(bundle / manifest["records"][0]["snapshot"], trace)
    protocol = json.loads((target / "protocol.json").read_text())
    protocol["chunks"] = 49
    common.write_json(target / "protocol.json", protocol)
    with pytest.raises(ValueError, match="Frozen protocol changed"):
        common.verify(target)


def test_snapshot_physics_matches_external_dylas_for_action_replays(prepared):
    bundle, manifest = prepared
    repo = common.DEFAULT_DYLAS
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("native_dylas_env_test", repo / "baseline_special/env.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    for dataset in ("fcc", "norway3g", "oboe", "hsr"):
        record = next(r for r in manifest["records"] if r["split"] == "test" and r["dataset"] == dataset)
        values = common.load_numeric(bundle / record["snapshot"])
        for session in record["sessions"]:
            left = common.environment(bundle, record, session)
            right = native.Environment(all_cooked_time=[values[:, 0].tolist()], all_cooked_bw=[values[:, 1].tolist()],
                                       all_file_names=[record["name"]], all_mahimahi_ptrs=[session["start_pointer"]],
                                       fixed=True, trace_num=1, video_size_dir=str(bundle / "video1_sizes") + "/",
                                       chunk_duration_s=4, total_video_chunks=48)
            # Exercise large buffers/sleep as well as bitrate changes and stalls.
            actions = [0] * 20 + [5, 0, 4, 1, 3, 2] * 4 + [5, 0, 5, 0]
            for action in actions:
                a, b = left.get_video_chunk(action), right.get_video_chunk(action)
                np.testing.assert_allclose(np.asarray(a[:5], dtype=float), np.asarray(b[:5], dtype=float), rtol=0, atol=0)
                assert a[5:] == b[5:]
                assert left.mahimahi_ptr == right.mahimahi_ptr
                assert left.last_mahimahi_time == right.last_mahimahi_time


def test_primary_metric_excludes_exactly_first_and_last(prepared):
    _, manifest = prepared
    rows = [{"reward": float(i), "bitrate_kbps": 750, "rebuffer_s": 0, "switch_mbps": 0, "decision_ms": 1} for i in range(48)]
    rows[0]["reward"], rows[-1]["reward"] = -10000, 10000
    metrics = common.episode_metrics(rows, manifest["settings"])
    assert metrics["qoe_primary"] == np.mean(np.arange(1, 47))
    assert metrics["scored_chunks_primary"] == 46
    assert metrics["qoe_skip_first"] != metrics["qoe_primary"]


def test_two_policy_interfaces_get_same_dynamics_and_no_future_network(prepared):
    bundle, manifest = prepared
    settings = manifest["settings"]
    record = next(r for r in manifest["records"] if r["split"] == "valid")
    session = record["sessions"][0]
    sizes = np.asarray([np.loadtxt(bundle / f"video1_sizes/video_size_{i}", dtype=np.int64) for i in range(6)])

    class RecordingPolicy:
        def reset(self, seed):
            self.calls = []
        def choose(self, context):
            self.calls.append(context)
            payload = request_for(context, "kev-latest")
            assert "source_parent" not in str(payload) and record["name"] not in str(payload)
            assert "future_bandwidth" not in payload["state"]
            assert len(payload["state"]["next_chunk_sizes_bytes"]) == 6
            assert len(context["history"]) == context["chunk_index"] + 1
            return 0, {}

    policy = RecordingPolicy()
    first = run_episode(bundle, record, session, policy, settings)
    second = run_episode(bundle, record, session, Traditional("lowest", settings, sizes), settings)
    assert len(policy.calls) == 47
    for a, b in zip(first["rows"], second["rows"]):
        for key in ("action", "buffer_s", "download_ms", "rebuffer_s", "reward", "elapsed_s"):
            assert a[key] == b[key]
    assert first["metrics"]["qoe_primary"] == second["metrics"]["qoe_primary"]


def test_random_policy_repeats_without_order_dependence(prepared):
    bundle, manifest = prepared
    settings = manifest["settings"]
    record = next(r for r in manifest["records"] if r["split"] == "valid")
    sizes = np.asarray([np.loadtxt(bundle / f"video1_sizes/video_size_{i}", dtype=np.int64) for i in range(6)])
    policy = Traditional("uniform_random", settings, sizes)
    a = run_episode(bundle, record, record["sessions"][0], policy, settings)
    run_episode(bundle, record, record["sessions"][1], policy, settings)
    b = run_episode(bundle, record, record["sessions"][0], policy, settings)
    assert [r["action"] for r in a["rows"]] == [r["action"] for r in b["rows"]]
    assert a["metrics"]["qoe_primary"] == b["metrics"]["qoe_primary"]


def test_summary_weights_rounds_within_trace_then_traces():
    episodes = [{"dataset": "fcc", "record_id": name, "metrics": {"qoe_primary": value, "qoe_skip_first": value}}
                for name, value in [("a", 0), ("a", 2), ("a", 4), ("b", 10), ("b", 10), ("b", 10)]]
    result = summarize(episodes)["fcc"]
    assert result["mean_qoe"] == 6
    assert result["trace_sd_ddof0"] == 4


def test_report_rejects_different_environment_or_selection(tmp_path):
    from types import SimpleNamespace
    directories = []
    for i in range(2):
        directory = tmp_path / str(i)
        directory.mkdir()
        selection = [{"id": "valid/fcc/trace", "round": i, "start_pointer": i + 1}]
        run = {"bundle_fingerprint": "one-environment", "selection": selection}
        run["fingerprint"] = common.fingerprint(run)
        summary = {"complete": True, "run_fingerprint": run["fingerprint"],
                   "bundle_fingerprint": run["bundle_fingerprint"],
                   "selection_fingerprint": common.fingerprint(selection), "results": {}}
        common.write_json(directory / "run.json", run)
        common.write_json(directory / "summary.json", summary)
        directories.append(directory)
    with pytest.raises(ValueError, match="session selection"):
        report(SimpleNamespace(runs=directories, out=tmp_path / "report.md"))


def test_http_adapter_uses_same_runner_and_fails_invalid_output(prepared):
    bundle, manifest = prepared
    invalid = [False]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_json(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.send_json({"models": [{"name": "kev-latest", "run": "/not-a-checkpoint", "base": "test-only"}]})
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            p = {k: 0.0 for k in payload["questions"]["bitrate"]["criteria"]}
            p["300"] = 1.0 if not invalid[0] else .4
            self.send_json({"answers": {"bitrate": {"choice": "300", "probabilities": p}}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        policy = JevHTTP(f"http://127.0.0.1:{server.server_port}")
        record = next(r for r in manifest["records"] if r["split"] == "valid")
        result = run_episode(bundle, record, record["sessions"][0], policy, manifest["settings"])
        assert len(calls) == 47 and len(result["rows"]) == 48
        invalid[0] = True
        with pytest.raises(ValueError, match="Invalid Jev probability"):
            run_episode(bundle, record, record["sessions"][0], policy, manifest["settings"])
        policy.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
