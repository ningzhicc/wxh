"""Checks for ABR simulation, reward accounting and the MPC implementation."""
import ast
import itertools

import numpy as np
import pytest

from benchmark import (NETLLM, BITRATES, BBA, RobustMPC, Trace, environment, load_traces,
                       observation, reward, validate_answer, video_sizes)


def test_constant_link_matches_analytic_download_and_buffer():
    trace = Trace("synthetic", "constant", None, np.array([0., 100., 200.]), np.array([2., 2., 2.]), 1, "test")
    sim = environment(trace)
    sizes = video_sizes()
    first = sim.get_video_chunk(0)
    expected = sizes[0, 0] * 8 / 1e6 / (2 * .95) + .08
    assert first[0] / 1000 == pytest.approx(expected)
    assert first[3] == pytest.approx(expected)
    assert first[2] == pytest.approx(4)
    second = sim.get_video_chunk(3)
    download = sizes[3, 1] * 8 / 1e6 / (2 * .95) + .08
    assert second[0] / 1000 == pytest.approx(download)
    assert second[3] == pytest.approx(max(download - 4, 0))
    assert second[2] == pytest.approx(max(4 - download, 0) + 4)


def test_reward_units_and_switch_penalty():
    assert reward(3, 2, .5) == pytest.approx(1.85 - 4.3 * .5 - .65)
    assert reward(5, 5, 0) == pytest.approx(4.3)


def test_bba_matches_repository_boundaries():
    policy = BBA()
    for buffer, expected in [(0, 0), (5, 0), (7, 1), (10, 2), (14.9, 4), (15, 5), (60, 5)]:
        assert policy.choose([], buffer, 0, 1, None) == expected


def test_full_dataset_inventory_and_shared_start_points():
    traces = load_traces()
    assert len(traces) == 275
    assert len({t.key for t in traces}) == 275
    assert traces[0].start_pointer == 28
    assert [t.metadata() for t in traces] == [t.metadata() for t in load_traces()]


def test_observation_has_no_future_network_or_dataset_identity():
    history = [{"throughput_mbps": 2.0 + i, "delay_ms": 200., "bitrate_kbps": 750} for i in range(10)]
    state = observation(history, 8, 1, [100, 200, 300, 400, 500, 600], 38)
    assert state["recent_throughput_mbps"] == [6, 7, 8, 9, 10, 11]
    assert set(state) == {"recent_throughput_mbps", "recent_download_seconds", "recent_bitrates_kbps",
                          "buffer_seconds", "previous_bitrate_kbps", "next_chunk_sizes_bytes",
                          "chunk_duration_seconds", "chunks_remaining"}


def test_reject_incomplete_or_invalid_distribution():
    body = {"answers": {"bitrate": {"choice": "300", "probabilities": {"300": 1.0}}}}
    with pytest.raises(AssertionError):
        validate_answer(body)


def test_mpc_agrees_with_upstream_scalar_planning_with_corrected_index():
    source = ast.parse((NETLLM / "run_baseline.py").read_text())
    selected = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in {"get_chunk_size", "calculate_rebuffer"}]
    for node in selected:
        node.decorator_list = []
    namespace = {"np": np, "TOTAL_VIDEO_CHUNK": 48, "REBUF_PENALTY": 4.3}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "upstream_mpc", "exec"), namespace)
    sizes = np.random.default_rng(10).integers(20000, 1000000, size=(6, 49))
    policy = RobustMPC()
    history = [{"throughput_mbps": 2.8}]
    expected = namespace["calculate_rebuffer"](sizes, 2, 2.3, 2, 45, 2.8 / 8,
                                                  np.array(list(itertools.product(range(6), repeat=2))), BITRATES)
    assert policy.choose(history, 2.3, 2, 46, sizes) == expected
    # A new session has no prediction-error history inherited from the preceding trace.
    assert RobustMPC().estimates == [] and RobustMPC().errors == []
