#!/usr/bin/env python3
"""Export an explicit inference allowlist; annotations and source identity stay out."""
import argparse
import json
from pathlib import Path
import struct

from devset import DEFAULT_DATASET, ROOT, digest, load_dataset, read_json, under, write_json


def make_input(case, receipt, language):
    images = [f for f in receipt["files"] if f["path"].endswith(".png")]
    images.sort(key=lambda f: f["path"])
    times = receipt["frame_requested_relative_times_s"]
    if len(images) != len(times):
        raise ValueError("Evidence frame/time count mismatch")
    frames = []
    for item, timestamp in zip(images, times):
        path = under(ROOT, item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError("Evidence image changed")
        with path.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Expected native-size PNG evidence")
        width, height = struct.unpack(">II", header[16:24])
        frames.append({"path": item["path"], "timestamp_s": timestamp,
                       "width": width, "height": height, "sha256": item["sha256"]})
    return {"case_id": case["case_id"], "question": case["question"][language],
            "language": language, "frames": frames,
            "coordinate_system": "absolute pixels, top-left origin, xyxy; width/height per frame"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    manifest, _, cases = load_dataset(args.dataset)
    from devset import read_jsonl
    questions = {q["case_id"]: q for q in read_jsonl(args.dataset / manifest["task_questions_file"])}
    records = []
    for case in cases:
        receipt = read_json(args.dataset / f"provenance/preparation/{case['case_id']}.json")
        active_case = {**case, "question": questions[case["case_id"]]}
        records.append(make_input(active_case, receipt, args.language))
    dest = args.dataset / f"reports/inference_inputs_{args.language}.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    write_json(dest.with_suffix(".receipt.json"), {"path": str(dest.relative_to(args.dataset)),
               "sha256": digest(dest), "cases_sha256": digest(args.dataset / "cases.jsonl"),
               "task_questions_sha256": digest(args.dataset / manifest["task_questions_file"]),
               "script_sha256": digest(__file__), "label_files_read": False})
    print(dest)


if __name__ == "__main__":
    main()

