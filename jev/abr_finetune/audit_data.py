"""Check frozen files and recognizable sources; search for shared bandwidth subsequences."""
import json
import hashlib

import numpy as np

from prepare_data import ROOT, fcc_traces, source_parent, numeric_trace_hash
from benchmark import atomic_json, digest, load_traces


def windows(trace, width=16):
    values = np.round(np.asarray(trace.bandwidth, dtype="<f8"), 6)
    for offset in range(len(values) - width + 1):
        window = values[offset:offset + width]
        if len(set(window)) >= 8:
            yield offset, hashlib.sha256(window.tobytes()).digest()


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    assert manifest["prepare_source_sha256"] == digest(ROOT / "prepare_data.py")
    assert manifest["protocol_sha256"] == digest(ROOT / "PROTOCOL.md")
    assert all(digest(ROOT / "data" / name) == value for name, value in manifest["files_sha256"].items())
    saved = json.loads((ROOT / "data/traces.json").read_text())
    keys = lambda role: {r["key"] for r in saved[role]}
    train = [t for t in fcc_traces("train") if t.key in keys("train")]
    valid = [t for t in fcc_traces("valid") if t.key in keys("valid")]
    test = load_traces()
    parents = lambda group: {source_parent(t.name) for t in group}
    assert not parents(train) & (parents(valid) | parents(test))
    assert not parents(valid) & parents(test)
    numeric = lambda group: {numeric_trace_hash(np.column_stack([t.times, t.bandwidth])) for t in group}
    assert not numeric(train) & (numeric(valid) | numeric(test))
    assert not numeric(valid) & numeric(test)
    forbidden = {}
    for trace in test:
        for offset, fingerprint in windows(trace):
            forbidden.setdefault(fingerprint, (trace.name, offset))
    overlaps = []
    for trace in train + valid:
        for offset, fingerprint in windows(trace):
            if fingerprint in forbidden:
                other, other_offset = forbidden[fingerprint]
                overlaps.append({"candidate": trace.name, "candidate_offset": offset,
                                 "test": other, "test_offset": other_offset})
                break
    result = {"passed": not overlaps, "frozen_file_hashes_verified": True,
              "recognized_original_acquisitions_disjoint": True, "exact_numeric_traces_disjoint": True,
              "search": "All contiguous 16-sample bandwidth windows rounded to 6 decimals, at least 8 distinct values",
              "shared_nonconstant_bandwidth_subsequences": overlaps,
              "test_unique_windows_searched": len(forbidden),
              "limits": "Different resampling or scale factors can conceal a shared source; unrecognized names are not proven independent acquisitions.",
              "source_sha256": digest(__file__)}
    atomic_json(ROOT / "data_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], "Shared test bandwidth sequences found; inspect before using this training run"


if __name__ == "__main__":
    main()
