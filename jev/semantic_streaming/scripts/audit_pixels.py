#!/usr/bin/env python3
"""Check every decoded reference frame against the corresponding source window."""
import argparse
from pathlib import Path
import subprocess

from devset import DEFAULT_DATASET, ROOT, digest, ffmpeg_binary, load_dataset, utcnow, write_json


def frame_hashes(binary, source, start=None):
    command = [str(binary), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start is not None:
        command += ["-ss", str(start)]
    command += ["-i", str(source)]
    if start is not None:
        command += ["-t", "5"]
    command += ["-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-threads", "2",
                "-f", "framemd5", "-hash", "sha256", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    hashes = [line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines()
              if line.strip() and not line.startswith("#")]
    return hashes, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    manifest, lock, cases = load_dataset(args.dataset)
    videos = {v["sample_id"]: v for v in lock["videos"]}
    binary = ffmpeg_binary(args)
    rows = []
    for case in cases:
        source = ROOT / videos[case["sample_id"]]["raw_path"]
        clip = ROOT / "artifacts" / manifest["dataset_id"] / case["case_id"] / "reference_lossless.mp4"
        original, original_command = frame_hashes(binary, source, case["window"]["start_s"])
        extracted, extracted_command = frame_hashes(binary, clip)
        match = bool(original) and original == extracted
        rows.append({"case_id": case["case_id"], "source_frame_count": len(original),
                     "clip_frame_count": len(extracted), "decoded_frames_equal": match,
                     "raw_sha256": digest(source), "clip_sha256": digest(clip),
                     "source_frame_pixel_sha256": original, "clip_frame_pixel_sha256": extracted,
                     "commands": [original_command, extracted_command]})
        print(f"{case['case_id']}: source={len(original)} reference={len(extracted)} equal={match}", flush=True)
    report = {"checked_at_utc": utcnow(), "ffmpeg_sha256": digest(binary),
              "audit_script_sha256": digest(__file__), "cases": rows,
              "all_decoded_frames_equal": all(r["decoded_frames_equal"] for r in rows)}
    write_json(args.dataset / "reports/pixel_integrity.json", report)
    if not report["all_decoded_frames_equal"]:
        raise ValueError("Reference derivation changed decoded frames; inspect pixel_integrity.json")


if __name__ == "__main__":
    main()
