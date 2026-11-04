"""Prepare FCC-only MPC supervision, with split and state overlap audits."""
import collections
import hashlib
import json
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "abr_eval"))
from benchmark import (NETLLM, BITRATES, BBA, RobustMPC, Trace, atomic_json, canonical_hash,
                       digest, environment, load_traces, make_request, observation, video_sizes)


def numeric_trace_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def source_parent(name):
    """Recognize full FCC acquisitions and Norway log windows independently of split names."""
    match = re.search(r"(?:fcc_)?trace_(\d+)_", name)
    if match:
        return "fcc-acquisition:" + match.group(1)
    stripped = name.removeprefix("norway_")
    if "-report." in stripped and ".log" in stripped:
        return "norway-log:" + stripped.split(".log", 1)[0] + ".log"
    return "unrecognized-name:" + name


def fcc_traces(split):
    directory = NETLLM / "data/traces" / split / ("fcc-" + split)
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix != ".pkl" and not p.name.startswith("."))
    expected = {"train": 235, "valid": 150, "test": 100}[split]
    assert len(files) == expected
    pointers_file = directory / "mahimahi_ptrs.pkl"
    pointers = pickle.loads(pointers_file.read_bytes()) if pointers_file.exists() else None
    if pointers is not None:
        assert len(pointers) == len(files)
    traces = []
    for index, path in enumerate(files):
        values = np.loadtxt(path, ndmin=2)
        assert values.shape[0] >= 2 and values.shape[1] == 2 and np.isfinite(values).all()
        assert (np.diff(values[:, 0]) >= 0).all() and (values[:, 1] >= 0).all()
        assert np.any((np.diff(values[:, 0]) > 0) & (values[1:, 1] > 0))
        pointer = int(pointers[index]) if pointers is not None else 1 + int.from_bytes(hashlib.sha256(f"666:{path.name}".encode()).digest()[:8], "big") % (len(values) - 1)
        traces.append(Trace("fcc_" + split, path.name, path, values[:, 0], values[:, 1], pointer, digest(path)))
    return traces


def teacher_records(trace, behavior, sizes):
    sim, expert, bba = environment(trace), RobustMPC(), BBA()
    rng = random.Random(f"666:{trace.name}:{behavior}")
    rows, action, records = [], 0, []
    for chunk in range(48):
        delay, _, buffer, rebuffer, size, next_sizes, end, remaining = sim.get_video_chunk(action)
        rows.append({"throughput_mbps": float(size * 8 / delay / 1000), "delay_ms": float(delay), "bitrate_kbps": int(BITRATES[action])})
        if end:
            assert chunk == 47
            break
        state = observation(rows, buffer, action, next_sizes, remaining)
        label = expert.choose(rows, buffer, action, chunk + 1, sizes)
        request = make_request(state)
        request["questions"]["bitrate"]["label"] = str(int(BITRATES[label]))
        request["_meta"] = {"source": "abr_fcc_mpc", "split": trace.dataset,
                            "id": f"{trace.key}/{behavior}/{chunk + 1}", "group_id": trace.key,
                            "trace": trace.name, "trace_sha256": trace.sha256, "start_pointer": trace.start_pointer,
                            "behavior": behavior, "chunk": chunk + 1, "state_sha256": canonical_hash(state)}
        records.append(request)
        action = label if behavior == "mpc" else bba.choose(rows, buffer, action, chunk + 1, sizes) if behavior == "bba" else rng.randrange(6) if rng.random() < .3 else label
    assert len(records) == 47
    return records


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare():
    out = ROOT / "data"
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["prepare_source_sha256"] == digest(__file__)
        assert all(digest(out / filename) == value for filename, value in manifest["files_sha256"].items())
        print("Prepared dataset already present:", manifest["counts"], flush=True)
        return
    out.mkdir(exist_ok=True)
    sizes = video_sizes()
    train, valid, test = fcc_traces("train"), fcc_traces("valid"), load_traces()
    original_train, original_valid = train, valid
    parents = lambda traces: {source_parent(trace.name) for trace in traces}
    test_parents, valid_parents = parents(test), parents(valid)
    hashes = lambda traces: {numeric_trace_hash(np.column_stack([trace.times, trace.bandwidth])) for trace in traces}
    test_hashes, valid_hashes = hashes(test), hashes(valid)
    original_counts = {"train_traces": len(train), "valid_traces": len(valid), "test_traces": len(test)}
    train = [trace for trace in train if source_parent(trace.name) not in test_parents | valid_parents
             and numeric_trace_hash(np.column_stack([trace.times, trace.bandwidth])) not in test_hashes | valid_hashes]
    valid = [trace for trace in valid if source_parent(trace.name) not in test_parents
             and numeric_trace_hash(np.column_stack([trace.times, trace.bandwidth])) not in test_hashes]
    assert train and len(valid) >= 20
    assert not parents(train) & (valid_parents | test_parents)
    assert not parents(valid) & test_parents
    source_audit = {"isolation": "FCC acquisition ID / complete Norway log identity plus exact numeric hash",
                    "filename_identity_is_inferred_from_names": True,
                    "original_train_parents_shared_with_test": sorted(parents(original_train) & test_parents),
                    "original_valid_parents_shared_with_test": sorted(valid_parents & test_parents),
                    "retained_train_parents_shared_with_valid_or_test": [],
                    "retained_valid_parents_shared_with_test": [],
                    "retained_train_source_counts": dict(collections.Counter(source_parent(t.name).split(":", 1)[0] for t in train)),
                    "retained_train_unique_parents": len(parents(train)),
                    "retained_valid_unique_parents": len(parents(valid)),
                    "valid_parent_counts": dict(collections.Counter(source_parent(t.name) for t in valid)),
                    "same_region_or_collection_is_not_excluded": True}
    atomic_json(out / "source_overlap.json", source_audit)
    valid_records = [record for trace in valid for record in teacher_records(trace, "mpc", sizes)]
    valid_states = {r["_meta"]["state_sha256"] for r in valid_records}
    pool = []
    for index, trace in enumerate(train, 1):
        for behavior in ["mpc", "bba", "explore"]:
            pool.extend(teacher_records(trace, behavior, sizes))
        if index % 25 == 0:
            print(f"Prepared {index}/{len(train)} FCC training traces", flush=True)
    unique, conflicting, repeated = {}, set(), 0
    validation_overlap = 0
    for row in pool:
        key = row["_meta"]["state_sha256"]
        if key in valid_states:
            validation_overlap += 1
            continue
        if key in unique:
            repeated += 1
            if unique[key]["questions"]["bitrate"]["label"] != row["questions"]["bitrate"]["label"]:
                conflicting.add(key)
        else:
            unique[key] = row
    clean = [row for key, row in unique.items() if key not in conflicting]
    rng = random.Random(666)
    groups = collections.defaultdict(list)
    for row in clean:
        groups[row["questions"]["bitrate"]["label"]].append(row)
    selected = []
    for label in map(str, BITRATES):
        group = groups[label]
        selected.extend(rng.sample(group, min(200, len(group))))
    chosen = {r["_meta"]["state_sha256"] for r in selected}
    remainder = [r for r in clean if r["_meta"]["state_sha256"] not in chosen]
    selected.extend(rng.sample(remainder, min(8000 - len(selected), len(remainder))))
    rng.shuffle(selected)
    assert len({r["_meta"]["state_sha256"] for r in selected}) == len(selected)
    assert not ({r["_meta"]["state_sha256"] for r in selected} & valid_states)
    chosen_valid = sorted(valid, key=lambda trace: hashlib.sha256(f"selection666:{trace.name}".encode()).hexdigest())[:min(40, len(valid))]
    validation_keys = {trace.key for trace in chosen_valid}
    selection_records = [r for r in valid_records if r["_meta"]["group_id"] in validation_keys]
    classification = random.Random(666).sample(selection_records, min(1200, len(selection_records)))
    datasets = {"train.jsonl": selected, "train_pool.jsonl": clean, "valid.jsonl": valid_records,
                "valid_selection_labels.jsonl": classification, "probe.jsonl": selected[:64]}
    for filename, rows in datasets.items():
        write_jsonl(out / filename, rows)
    atomic_json(out / "traces.json", {"train": [t.metadata() for t in train], "valid": [t.metadata() for t in valid],
                                      "valid_selection": [t.metadata() for t in chosen_valid]})
    manifest = {"prepare_source_sha256": digest(__file__), "protocol_sha256": digest(ROOT / "PROTOCOL.md"),
                "original_counts": original_counts,
                "counts": {"train_traces": len(train), "valid_traces": len(valid), "valid_selection_traces": len(chosen_valid),
                           "raw_train_records": len(pool), "deduplicated_train_pool": len(clean), "selected_train_records": len(selected),
                           "valid_records": len(valid_records), "conflicting_states_removed": len(conflicting),
                           "repeated_states_removed": repeated, "train_valid_state_overlap_removed": validation_overlap},
                "train_label_counts": dict(collections.Counter(r["questions"]["bitrate"]["label"] for r in selected)),
                "pool_label_counts": dict(collections.Counter(r["questions"]["bitrate"]["label"] for r in clean)),
                "files_sha256": {name: digest(out / name) for name in [*datasets, "traces.json", "source_overlap.json"]},
                "seed": 666, "teacher": "RobustMPC-6", "behavior_modes": ["mpc", "bba", "explore"],
                "same_prompt_and_observation_as_zero_shot": True, "uses_test_supervision": False}
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
