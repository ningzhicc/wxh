"""Replay stored actions and validate the finished experiment's numeric records."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from benchmark import BITRATES, ROOT, atomic_json, digest, environment, load_traces, reward, validate_answer


def audit(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    complete = json.loads((out / "completion.json").read_text())
    assert complete["complete"] and complete["fingerprint"] == manifest["fingerprint"]
    traces = {trace.key: trace for trace in load_traces(manifest["config"]["seed"], manifest["config"]["datasets"], manifest["config"]["limit"])}
    episodes = [json.loads(p.read_text()) for p in (out / "episodes").glob("*/*/*.json")]
    assert len(episodes) == len(traces) * len(manifest["config"]["methods"]) == complete["episodes"]
    checked_steps, checked_probabilities = 0, 0
    for episode in episodes:
        trace = traces[episode["key"]]
        assert episode["fingerprint"] == manifest["fingerprint"]
        assert episode["sha256"] == trace.sha256 and episode["start_pointer"] == trace.start_pointer
        sim = environment(trace)
        previous = 0
        for index, step in enumerate(episode["steps"]):
            actual = sim.get_video_chunk(step["action"])
            delay, _, buffer, stall, size, _, ended, _ = actual
            assert index == step["chunk"] and step["bitrate_kbps"] == int(BITRATES[step["action"]])
            assert np.allclose([delay, buffer, stall, size],
                               [step["delay_ms"], step["buffer_s"], step["rebuffer_s"], step["size_bytes"]],
                               rtol=0, atol=1e-8), (episode["key"], index)
            assert ended == (index == 47)
            assert abs(step["reward"] - reward(step["action"], previous, stall)) < 1e-10
            if index:
                state = step["decision"]["state"]
                assert state["next_chunk_sizes_bytes"][str(step["bitrate_kbps"])] == step["size_bytes"]
                assert state["previous_bitrate_kbps"] == int(BITRATES[previous])
                assert abs(state["buffer_seconds"] - episode["steps"][index - 1]["buffer_s"]) < .000051
                if episode["method"] == "kev":
                    action, _ = validate_answer({"answers": {"bitrate": {
                        "choice": str(step["bitrate_kbps"]), "probabilities": step["decision"]["probabilities"]}}})
                    assert action == step["action"]
                    checked_probabilities += 1
            previous = step["action"]
            checked_steps += 1
        metrics = episode["metrics"]
        identity = metrics["mean_bitrate_mbps"] - 4.3 * metrics["rebuffer_s"] / 47 - metrics["smoothness_mbps"]
        assert abs(identity - metrics["qoe_mean"]) < 1e-10
        log = out / "logs" / episode["method"] / trace.dataset / ("result_sim_abr_" + trace.name)
        logged = np.loadtxt(log)
        assert logged.shape == (48, 8)
        assert np.allclose(logged[:, 7], [row["reward"] for row in episode["steps"]], rtol=0, atol=1e-10)
    integrity = json.loads((out / "model_integrity.json").read_text())
    assert integrity["unchanged"] and integrity["before"] == integrity["after"]
    current = {name: digest(Path(manifest["model_deployment"]["checkpoint_path"]) / name) for name in integrity["before"]}
    assert current == integrity["after"]
    result = {"passed": True, "episodes": len(episodes), "replayed_chunks": checked_steps,
              "validated_kev_probability_distributions": checked_probabilities,
              "model_files_unchanged": True, "fingerprint": manifest["fingerprint"],
              "client_environment": {name: importlib.metadata.version(name) for name in ["numpy", "httpx", "matplotlib"]},
              "latency_definition": "Kev latency_ms is model time for the request's batch, excluding queue wait; client_ms includes queue and HTTP time."}
    atomic_json(out / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/zero_shot_v1")
    audit(parser.parse_args().out)
