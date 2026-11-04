"""Recalculate saved DyLas scores; never load weights or run a policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_provenance import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/amora/NetLLM"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    root = args.repo / "adaptive_bitrate_streaming"
    source = root / "artifacts/qoe_cdf_columns/video1/abrllm_v4_all_traces_manifest.json"
    data = source.read_bytes()
    manifest = json.loads(data)
    screenshot = {"fcc-test": (1.11, .65), "norway3G-test": (1.33, .71),
                  "oboe-test": (2.10, 1.10), "hsr-test": (3.01, 1.61)}
    result = {"manifest": str(source), "manifest_sha256": sha(data),
              "collect_params": manifest["collect_params"], "datasets": [],
              "interpretation": "Rounded score agreement links saved results to screenshot numerically. It does not cryptographically identify the training pool or checkpoint used to generate those results."}
    for entry in manifest["entries"]:
        records, arrays = [], []
        for path in sorted(Path(entry["log_root"]).glob("result*")):
            data = path.read_bytes()
            values = np.loadtxt(path)
            if values.shape != (48, 8) or not np.isfinite(values).all():
                raise ValueError(f"Unexpected playback log: {path}")
            arrays.append(values)
            records.append({"path": str(path), "sha256": sha(data)})
        if len(arrays) != entry["num_select"]:
            raise ValueError("Log count differs from manifest")
        variants = {}
        target = screenshot[entry["trace"]]
        for name, section in (("all", slice(None)), ("skip_first", slice(1, None)), ("skip_first_last", slice(1, -1))):
            means = np.array([a[section, -1].mean() for a in arrays])
            mean, sd = float(means.mean()), float(means.std(ddof=1))
            variants[name] = {"mean": mean, "trace_sd_ddof0": float(means.std()), "trace_sd_ddof1": sd,
                              "rounded_mean_sd1": [round(mean, 2), round(sd, 2)],
                              "matches_screenshot_mean_and_sd": (round(mean, 2), round(sd, 2)) == target,
                              "mean_matches_saved_manifest": bool(np.isclose(mean, entry["mean_of_means"], rtol=0, atol=1e-12))}
        result["datasets"].append({"dataset": entry["trace"], "trace_count": len(arrays),
                                   "screenshot_mean_sd": list(target), "variants": variants, "raw_logs": records})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({d["dataset"]: d["variants"]["skip_first"] for d in result["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
