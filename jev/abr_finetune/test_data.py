"""Verify leakage filtering and ABR label handling before the training run."""
import json
import random

import numpy as np

from prepare_data import ROOT, fcc_traces, numeric_trace_hash, source_parent
from benchmark import CRITERIA, load_traces
from kev.data import api_request, augment, load_records, materialize


def test_training_states_disjoint_from_validation_and_labels_cover_six_actions():
    train = load_records(ROOT / "data/train.jsonl")
    valid = load_records(ROOT / "data/valid.jsonl")
    assert 5000 <= len(train) <= 8000
    assert set(r["questions"]["bitrate"]["label"] for r in train) == set(CRITERIA)
    assert not ({r["_meta"]["state_sha256"] for r in train} & {r["_meta"]["state_sha256"] for r in valid})


def test_actual_selected_network_traces_disjoint_from_validation_and_test():
    metadata = json.loads((ROOT / "data/traces.json").read_text())
    selected = {r["key"] for r in metadata["train"]}
    fingerprints = lambda traces: {numeric_trace_hash(np.column_stack([r.times, r.bandwidth])) for r in traces}
    train = [r for r in fcc_traces("train") if r.key in selected]
    assert len(train) == 55
    assert not fingerprints(train) & fingerprints(fcc_traces("valid"))
    assert not fingerprints(train) & fingerprints(load_traces())
    parents = lambda traces: {source_parent(t.name) for t in traces}
    assert not parents(train) & (parents(fcc_traces("valid")) | parents(load_traces()))
    valid_keys = {r["key"] for r in metadata["valid"]}
    valid = [r for r in fcc_traces("valid") if r.key in valid_keys]
    assert len(valid) == 30
    assert not parents(valid) & parents(load_traces())


def test_metadata_and_teacher_labels_are_not_model_input():
    request = load_records(ROOT / "data/train.jsonl")[0]
    body = api_request(request)
    assert set(body) == {"state", "questions"}
    assert "label" not in body["questions"]["bitrate"]
    assert "trace" not in body["state"]
    assert "behavior" not in body["state"]


def test_permuted_choice_order_preserves_label_identity_and_six_actions():
    request = load_records(ROOT / "data/train.jsonl")[0]
    label = request["questions"]["bitrate"]["label"]
    variants = [augment(request, random.Random(seed), 0, 0, 0) for seed in range(20)]
    assert len({tuple(r["questions"]["bitrate"]["criteria"]) for r in variants}) > 10
    for variant in variants:
        variant["questions"]["bitrate"]["src"] = "custom_choice"
        question = materialize(variant)["questions"][0]
        assert set(question["keys"]) == set(CRITERIA)
        assert question["keys"][question["label"]] == label
