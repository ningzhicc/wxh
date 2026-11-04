"""Exercise integrity failures and inference isolation, without external network calls."""
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import devset
import export_inputs


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            self.send_error(400)
            return
        start, end = map(int, match.groups())
        body = self.server.blob[start:end + 1]
        status = 200 if self.server.mode == "ignore" else 206
        self.send_response(status)
        reported_start = start + 1 if self.server.mode == "wrong" else start
        self.send_header("Content-Range", f"bytes {reported_start}-{end}/{len(self.server.blob)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        devset.write_json(self.dataset / "sources.lock.json", {"fixture": "immutable"})
        self.payload = b"raw-video-byte-fixture" * 50
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("sample_1/video.mp4", self.payload)
        self.blob = archive.getvalue()
        with zipfile.ZipFile(io.BytesIO(self.blob)) as source:
            info = source.getinfo("sample_1/video.mp4")
        self.video = {"sample_id": 1, "archive_path": "fixture.zip", "archive_size_bytes": len(self.blob),
            "raw_path": "media/raw/fixture/video.mp4", "member": {"offset": info.header_offset,
            "method": info.compress_type, "name": info.filename, "size": info.file_size,
            "compressed": info.compress_size, "crc32": info.CRC}}
        self.lock = {"streamingbench": {"revision": "a" * 40}}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.server.blob, self.server.mode = self.blob, "valid"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/fixture.zip"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def fetch_local(self):
        original = devset.range_response
        def local_range(_url, start, end, total):
            return original(self.url, start, end, total)
        with patch.object(devset, "ROOT", self.root), patch.object(devset, "range_response", local_range):
            return devset.fetch_one(self.dataset, self.lock, self.video)

    def test_server_ignoring_range_is_rejected(self):
        self.server.mode = "ignore"
        with self.assertRaisesRegex(ValueError, "exact requested byte range"):
            devset.range_response(self.url, 0, 29, len(self.blob))

    def test_wrong_content_range_is_rejected(self):
        self.server.mode = "wrong"
        with self.assertRaisesRegex(ValueError, "exact requested byte range"):
            devset.range_response(self.url, 0, 29, len(self.blob))

    def test_member_extraction_crc_and_sha256(self):
        receipt = self.fetch_local()
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual((self.root / self.video["raw_path"]).read_bytes(), self.payload)
        self.assertFalse(receipt["archive_full_sha256_verified_locally"])

    def test_crc_failure_never_publishes_raw_file(self):
        self.video["member"]["crc32"] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            self.fetch_local()
        self.assertFalse((self.root / self.video["raw_path"]).exists())

    def test_mutated_cache_is_rejected(self):
        self.fetch_local()
        (self.root / self.video["raw_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.fetch_local()

    def test_restore_checks_original_recorded_sha256(self):
        self.fetch_local()
        (self.root / self.video["raw_path"]).unlink()
        receipt_path = self.dataset / "provenance/downloads/sample_1.json"
        receipt = devset.read_json(receipt_path)
        receipt["sha256"] = "0" * 64
        devset.write_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "Restored source differs"):
            self.fetch_local()
        self.assertFalse((self.root / self.video["raw_path"]).exists())


class DatasetTests(unittest.TestCase):
    def test_selected_rows_and_source_answers_match_pinned_snapshot(self):
        dataset = devset.DEFAULT_DATASET
        _, _, cases = devset.load_dataset(dataset)
        with (dataset / "provenance/source_snapshots/devibench_datasets.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        answers = {r["case_id"]: r for r in devset.read_jsonl(dataset / "annotations/answers.jsonl")}
        for case in cases:
            row = rows[case["source_annotation"]["data_row_1based"] - 1]
            canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), case["source_annotation"]["row_sha256"])
            self.assertEqual(int(row["sample_id"]), case["sample_id"])
            self.assertEqual(row["question"], case["question"]["en"])
            self.assertEqual(row["std_answer"], answers[case["case_id"]]["source_answer"])

    def test_inference_allowlist_excludes_annotations_and_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 1920, 1080))
            case = {"case_id": "example", "question": {"en": "Read the label"},
                    "reference_answer": "SECRET_LABEL", "source_family_group": "SECRET_FAMILY"}
            receipt = {"files": [{"path": "frame.png", "sha256": devset.digest(image)}],
                       "frame_requested_relative_times_s": [0.0]}
            with patch.object(export_inputs, "ROOT", root):
                payload = export_inputs.make_input(case, receipt, "en")
            text = json.dumps(payload)
            self.assertNotIn("SECRET_LABEL", text)
            self.assertNotIn("SECRET_FAMILY", text)
            self.assertEqual(set(payload), {"case_id", "question", "language", "frames", "coordinate_system"})
            self.assertEqual(payload["frames"][0]["width"], 1920)


if __name__ == "__main__":
    unittest.main()
