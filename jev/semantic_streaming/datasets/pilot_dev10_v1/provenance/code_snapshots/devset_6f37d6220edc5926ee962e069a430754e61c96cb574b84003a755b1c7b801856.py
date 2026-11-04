#!/usr/bin/env python3
"""Acquire, prepare and audit the pinned pilot set, using only the standard library."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets/pilot_dev10_v1"
FRAME_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 4.8)
RUN_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def under(base, relative):
    result = (base / relative).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("Manifest path escapes its root")
    return result


def load_dataset(dataset):
    manifest = read_json(dataset / "manifest.json")
    lock = read_json(dataset / manifest["sources_lock"])
    cases = read_jsonl(dataset / manifest["cases_file"])
    ids = [c["case_id"] for c in cases]
    if len(ids) != manifest["case_count"] or len(set(ids)) != len(ids):
        raise ValueError("Case count or uniqueness mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", lock["streamingbench"]["revision"]):
        raise ValueError("An immutable upstream revision is required")
    for snap in lock["snapshots"]:
        path = under(dataset, snap["path"])
        if not path.is_file() or digest(path) != snap["sha256"]:
            raise ValueError(f"Source snapshot changed: {snap['path']}")
    for case in cases:
        if case["split"] != "pilot_dev":
            raise ValueError("This set is development-only")
        if any(k in case for k in ("answer", "reference_answer", "accepted_answers", "source_answer")):
            raise ValueError("Reference answers must remain outside inference inputs")
        window = case["window"]
        if window["start_s"] < 0 or abs(window["end_s"] - window["start_s"] - 5) > 1e-9:
            raise ValueError("Expected a five-second source window")
    return manifest, lock, cases


def range_response(url, start, end, total):
    # The range is in the URL as well: intermediary caches must not reuse a different range.
    query = urllib.parse.urlencode({"download": "true", "wxh_range": f"{start}-{end}"})
    request = urllib.request.Request(url + "?" + query, headers={
        "Range": f"bytes={start}-{end}", "User-Agent": "wxh-semantic-pilot/1.0",
    })
    response = urllib.request.urlopen(request, timeout=45)
    expected = f"bytes {start}-{end}/{total}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        response.close()
        raise ValueError("Server did not return the exact requested byte range; full archive download refused")
    return response


def range_bytes(url, start, end, total):
    for attempt in range(4):
        try:
            with range_response(url, start, end, total) as response:
                value = response.read(end - start + 2)
            if len(value) != end - start + 1:
                raise ValueError("Incomplete byte range")
            return value
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def compressed_blocks(url, start, end, total, cache, workers=4):
    """Cache bounded exact ranges; completed blocks survive a process interruption."""
    cache.mkdir(parents=True, exist_ok=True)
    block_size = 8 * 1024 * 1024
    ranges = [(a, min(a + block_size - 1, end)) for a in range(start, end + 1, block_size)]

    def acquire(pair):
        a, b = pair
        path = cache / f"{a}-{b}.bin"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            saved = read_json(receipt)
            if path.stat().st_size == b - a + 1 and digest(path) == saved["sha256"]:
                return path
            raise ValueError("Compressed download cache changed; inspect before retrying")
        data = range_bytes(url, a, b, total)
        temp = path.with_suffix(".part")
        temp.write_bytes(data)
        temp.replace(path)
        write_json(receipt, {"sha256": hashlib.sha256(data).hexdigest(), "range": [a, b]})
        return path

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(acquire, ranges))
    return [cache / f"{a}-{b}.bin" for a, b in ranges]


def fetch_one(dataset, lock, video):
    sid, member = video["sample_id"], video["member"]
    dest = under(ROOT, video["raw_path"])
    receipt_path = dataset / f"provenance/downloads/sample_{sid}.json"
    if dest.exists():
        if not receipt_path.exists():
            raise ValueError(f"Untracked existing source file: sample_{sid}")
        receipt = read_json(receipt_path)
        if (receipt["source_lock_sha256"] != digest(dataset / "sources.lock.json")
                or dest.stat().st_size != receipt["size_bytes"] or digest(dest) != receipt["sha256"]):
            raise ValueError(f"Existing source/receipt changed: sample_{sid}")
        print(f"sample_{sid}: existing bytes verified", flush=True)
        return receipt
    archive = urllib.parse.quote(video["archive_path"])
    revision = lock["streamingbench"]["revision"]
    url = f"https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/{revision}/{archive}"
    total, offset = video["archive_size_bytes"], member["offset"]
    header = range_bytes(url, offset, offset + 29, total)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header")
    fields = struct.unpack("<4s5H3I2H", header)
    flags, method, name_len, extra_len = fields[2], fields[3], fields[-2], fields[-1]
    if flags & 1 or method != member["method"] or method not in (0, 8):
        raise ValueError("Encrypted or unsupported ZIP member")
    name = range_bytes(url, offset + 30, offset + 29 + name_len, total).decode("utf-8")
    if name != member["name"]:
        raise ValueError("ZIP member identity mismatch")
    start = offset + 30 + name_len + extra_len
    end = start + member["compressed"] - 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".mp4.part")
    inflater = zlib.decompressobj(-15) if method == 8 else None
    h, crc, size = hashlib.sha256(), 0, 0
    started = time.monotonic()
    print(f"sample_{sid}: downloading one ZIP member ({member['compressed'] / 1e6:.1f} MB)", flush=True)
    cache = ROOT / f"media/download_cache/{revision}/sample_{sid}"
    blocks = compressed_blocks(url, start, end, total, cache)
    with temp.open("wb") as output:
        for path in blocks:
            with path.open("rb") as compressed:
                for block in iter(lambda: compressed.read(1024 * 1024), b""):
                    decoded = inflater.decompress(block) if inflater else block
                    if size + len(decoded) > member["size"]:
                        raise ValueError("Uncompressed member exceeds pinned size")
                    output.write(decoded)
                    h.update(decoded)
                    crc, size = zlib.crc32(decoded, crc), size + len(decoded)
        if inflater:
            tail = inflater.flush()
            output.write(tail)
            h.update(tail)
            crc, size = zlib.crc32(tail, crc), size + len(tail)
            if not inflater.eof:
                raise ValueError("Incomplete compressed ZIP member")
    if size != member["size"] or crc & 0xffffffff != member["crc32"]:
        raise ValueError(f"ZIP size/CRC mismatch for sample_{sid}")
    receipt = {"schema_version": 1, "sample_id": sid, "retrieved_at_utc": utcnow(),
        "source_lock_sha256": digest(dataset / "sources.lock.json"), "source_url": url,
        "archive_revision": revision, "archive_path": video["archive_path"],
        "archive_upstream_sha256": video.get("archive_lfs_sha256"),
        "archive_full_sha256_verified_locally": False, "member_name": name,
        "member_compressed_range": [start, end], "zip_crc32": f"{crc & 0xffffffff:08x}",
        "raw_path": video["raw_path"], "size_bytes": size, "sha256": h.hexdigest(),
        "download_seconds": round(time.monotonic() - started, 3),
        "download_method": "bounded parallel HTTP ranges, resumable compressed blocks",
        "fetch_script_sha256": RUN_CODE_SHA256}
    temp.replace(dest)
    write_json(receipt_path, receipt)
    print(f"sample_{sid}: CRC and SHA256 recorded", flush=True)
    return receipt


def fetch(args):
    _, lock, cases = load_dataset(args.dataset)
    selected = {c["sample_id"] for c in cases if not args.case or c["case_id"] in args.case}
    if args.case and not selected:
        raise ValueError("No matching case IDs")
    videos = [v for v in lock["videos"] if v["sample_id"] in selected]
    if sum(v["member"]["compressed"] for v in videos) > args.max_total_mib * 1024**2:
        raise ValueError("Selected media exceed the explicit download budget")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(fetch_one, args.dataset, lock, v): v["sample_id"] for v in videos}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as error:
                failures.append({"sample_id": jobs[job], "error_type": type(error).__name__, "reason": str(error)})
                print(f"sample_{jobs[job]} failed: {type(error).__name__}", flush=True)
    if failures:
        write_json(args.dataset / "reports/fetch_failures.json", failures)
        raise RuntimeError(f"{len(failures)} source downloads failed; rerun fetch after checking reports/fetch_failures.json")
    failure_path = args.dataset / "reports/fetch_failures.json"
    if failure_path.exists():
        write_json(failure_path, {"resolved": True, "resolved_at_utc": utcnow(), "remaining_failures": []})


def ffmpeg_binary(args):
    path = Path(args.ffmpeg) if getattr(args, "ffmpeg", None) else ROOT / ".tools/ffmpeg"
    if not path.is_file():
        found = shutil.which("ffmpeg")
        if not found:
            raise ValueError("FFmpeg missing; run scripts/bootstrap_tools.py")
        path = Path(found)
    return path.resolve()


def run_media(binary, options, log):
    command = [str(binary), "-hide_banner", "-nostdin", "-n", *map(str, options)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log).write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed; see {log}")
    return command


def prepare(args):
    manifest, lock, cases = load_dataset(args.dataset)
    binary = ffmpeg_binary(args)
    binary_sha = digest(binary)
    version = subprocess.run([str(binary), "-version"], capture_output=True, text=True, check=True).stdout
    write_json(args.dataset / "provenance/tool_environment.json", {
        "ffmpeg_sha256": binary_sha, "ffmpeg_version": version,
        "python": sys.version, "prepared_at_utc": utcnow()})
    by_sample = {v["sample_id"]: v for v in lock["videos"]}
    for case in cases:
        if args.case and case["case_id"] not in args.case:
            continue
        cid = case["case_id"]
        receipt_path = args.dataset / f"provenance/preparation/{cid}.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["cases_sha256"] != digest(args.dataset / "cases.jsonl"):
                raise ValueError("Case definitions changed after extraction")
            for file in receipt["files"]:
                if digest(under(ROOT, file["path"])) != file["sha256"]:
                    raise ValueError("Prepared media changed")
            print(f"{cid}: existing artifacts verified", flush=True)
            continue
        raw = under(ROOT, by_sample[case["sample_id"]]["raw_path"])
        download = read_json(args.dataset / f"provenance/downloads/sample_{case['sample_id']}.json")
        if not raw.is_file() or digest(raw) != download["sha256"]:
            raise ValueError("Raw video missing or modified; run fetch")
        out = ROOT / "artifacts" / manifest["dataset_id"] / cid
        if out.exists():
            raise ValueError(f"Uncommitted output directory exists: {out}; inspect before retrying")
        out.mkdir(parents=True)
        clip = out / "reference_lossless.mp4"
        commands = []
        commands.append(run_media(binary, ["-ss", case["window"]["start_s"], "-i", raw,
            "-t", 5, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-qp", 0, "-preset", "veryfast", "-threads", 2,
            "-fps_mode", "passthrough", "-movflags", "+faststart", clip], out / "clip.log"))
        for i, relative in enumerate(FRAME_TIMES):
            commands.append(run_media(binary, ["-ss", relative, "-i", clip,
                "-frames:v", 1, "-threads", 1, out / f"frame_{i:02d}.png"], out / f"frame_{i:02d}.log"))
        commands.append(run_media(binary, ["-framerate", 1, "-i", out / "frame_%02d.png",
            "-vf", "scale=640:-2,tile=3x2", "-frames:v", 1, "-q:v", 2, "-threads", 1,
            out / "contact_sheet.jpg"], out / "contact_sheet.log"))
        files = [{"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(out.iterdir()) if p.suffix in (".mp4", ".png", ".jpg", ".log")]
        receipt = {"case_id": cid, "prepared_at_utc": utcnow(),
            "cases_sha256": digest(args.dataset / "cases.jsonl"), "raw_sha256": download["sha256"],
            "ffmpeg_sha256": binary_sha, "script_sha256": RUN_CODE_SHA256,
            "derivation": "accurate seek, lossless H.264 re-encode, no audio, original spatial resolution and frame cadence",
            "clip_is_original_bitstream": False, "source_window": case["window"],
            "frame_requested_relative_times_s": list(FRAME_TIMES),
            "frame_time_note": "FFmpeg selects the first decoded frame at/after each requested clip timestamp; inspect source/clip logs for native fps",
            "commands": commands, "files": files}
        write_json(receipt_path, receipt)
        print(f"{cid}: lossless reference and six evidence frames prepared", flush=True)


def validate(args):
    manifest, lock, cases = load_dataset(args.dataset)
    answers = read_jsonl(args.dataset / manifest["answers_file"])
    if {a["case_id"] for a in answers} != {c["case_id"] for c in cases} or len(answers) != len(cases):
        raise ValueError("Answer annotations do not match cases")
    questions = read_jsonl(args.dataset / manifest["task_questions_file"])
    if len(questions) != len(cases) or {q["case_id"] for q in questions} != {c["case_id"] for c in cases}:
        raise ValueError("Operational questions do not match cases")
    for question in questions:
        if not all(isinstance(question.get(lang), str) and question[lang].strip() for lang in ("en", "zh")):
            raise ValueError("Operational question is missing a language")
    case_map = {c["case_id"]: c for c in cases}
    def check_code_snapshot(sha):
        path = args.dataset / f"provenance/code_snapshots/devset_{sha}.py"
        if not path.is_file() or digest(path) != sha:
            raise ValueError("The code version named by an asset receipt is unavailable or changed")
    downloaded, prepared = 0, 0
    for video in lock["videos"]:
        path = args.dataset / f"provenance/downloads/sample_{video['sample_id']}.json"
        if not path.exists():
            continue
        receipt = read_json(path)
        raw = under(ROOT, receipt["raw_path"])
        if receipt["source_lock_sha256"] != digest(args.dataset / "sources.lock.json"):
            raise ValueError("Download belongs to a different source lock")
        if raw.stat().st_size != video["member"]["size"] or digest(raw) != receipt["sha256"]:
            raise ValueError("Raw video integrity mismatch")
        check_code_snapshot(receipt["fetch_script_sha256"])
        downloaded += 1
    for case in cases:
        path = args.dataset / f"provenance/preparation/{case['case_id']}.json"
        if not path.exists():
            continue
        receipt = read_json(path)
        if receipt["cases_sha256"] != digest(args.dataset / "cases.jsonl"):
            raise ValueError("Preparation belongs to different cases")
        check_code_snapshot(receipt["script_sha256"])
        download = read_json(args.dataset / f"provenance/downloads/sample_{case['sample_id']}.json")
        if receipt["raw_sha256"] != download["sha256"] or receipt["source_window"] != case["window"]:
            raise ValueError("Preparation does not match its raw source/window")
        for file in receipt["files"]:
            if digest(under(ROOT, file["path"])) != file["sha256"]:
                raise ValueError("Artifact integrity mismatch")
        prepared += 1
    reviews = {}
    for answer in answers:
        status = answer["review_status"]
        reviews[status] = reviews.get(status, 0) + 1
        if answer["human_verified"] and not answer.get("human_review"):
            raise ValueError("Human verification claimed without a human review record")
        visual = answer.get("visual_review")
        if visual:
            for evidence in visual["evidence"]:
                image = under(ROOT, evidence["path"])
                if digest(image) != evidence["sha256"]:
                    raise ValueError("Visual review evidence changed")
                with image.open("rb") as stream:
                    header = stream.read(24)
                width, height = struct.unpack(">II", header[16:24])
                if evidence["native_size"] != [width, height]:
                    raise ValueError("Evidence dimensions do not match its PNG")
                x1, y1, x2, y2 = evidence["roi_seed_xyxy"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("Evidence ROI seed is out of bounds")
                start = case_map[answer["case_id"]]["window"]["start_s"]
                if not 0 <= evidence["requested_relative_s"] < 5 or abs(start + evidence["requested_relative_s"] - evidence["nominal_source_s"]) > 1e-9:
                    raise ValueError("Evidence timestamps do not match the source window")
    if args.require_media and (downloaded != len(cases) or prepared != len(cases)):
        raise ValueError("Media assets are incomplete")
    report = {"dataset_id": manifest["dataset_id"], "checked_at_utc": utcnow(),
        "cases": len(cases), "distinct_upstream_sample_ids": len({c['sample_id'] for c in cases}),
        "conservative_content_groups": len({c['source_family_group'] for c in cases}),
        "downloaded": downloaded, "prepared": prepared, "answer_review_status": reviews,
        "human_verified": sum(a["human_verified"] for a in answers),
        "cases_sha256": digest(args.dataset / "cases.jsonl"),
        "task_questions_sha256": digest(args.dataset / manifest["task_questions_file"]),
        "answers_sha256": digest(args.dataset / manifest["answers_file"]),
        "sources_lock_sha256": digest(args.dataset / "sources.lock.json"),
        "development_only": True, "model_compression_sensitivity_measured": False}
    write_json(args.dataset / "reports/validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "prepare", "validate"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case", action="append", help="Repeat to select case IDs")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-total-mib", type=int, default=2048)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-media", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    {"fetch": fetch, "prepare": prepare, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
