"""Verify API input isolation and pixel-coordinate safety without live calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qwen_roi


def response(regions, frame_id='dev10_01_f00', finish_reason='stop'):
    return {'choices': [{'finish_reason': finish_reason, 'message': {'content': json.dumps({
        'frame_id': frame_id, 'regions': regions})}}]}


class TestQwenROI(unittest.TestCase):
    def setUp(self):
        self.row = {'case_id': 'dev10_01', 'frame_id': 'dev10_01_f00',
                    'timestamp_s': 0.0, 'width': 1920, 'height': 1080,
                    'sha256': 'fixture'}

    def test_normalized_conversion_conservatively_covers_evidence(self):
        r = qwen_roi.parse_regions(response([{'bbox_2d': [101, 201, 899, 999]}]), self.row)
        box = r['regions'][0]['bbox_pixels']
        self.assertEqual(box, [193, 217, 1727, 1079])
        self.assertLessEqual(box[0], 101 * 1920 / 1000)
        self.assertGreaterEqual(box[2], 899 * 1920 / 1000)

    def test_absence_is_empty_not_full_frame(self):
        self.assertEqual(qwen_roi.parse_regions(response([]), self.row)['regions'], [])

    def test_rejects_wrong_frame_truncation_and_invalid_boxes(self):
        bad = [[-1, 0, 100, 100], [0, 0, 1000, 100], [100, 0, 0, 100],
               [0, 1, 100, 1], [True, 0, 100, 100], [0, 0, 1.2, 100]]
        for box in bad:
            with self.subTest(box=box), self.assertRaises(ValueError):
                qwen_roi.parse_regions(response([{'bbox_2d': box}]), self.row)
        with self.assertRaises(ValueError):
            qwen_roi.parse_regions(response([], frame_id='wrong'), self.row)
        with self.assertRaises(ValueError):
            qwen_roi.parse_regions(response([], finish_reason='length'), self.row)

    def test_endpoint_cannot_exfiltrate_key_or_redirect(self):
        qwen_roi.validate_endpoint('https://dashscope.aliyuncs.com/compatible-mode/v1')
        for endpoint in ['http://dashscope.aliyuncs.com/compatible-mode/v1',
                         'https://dashscope.aliyuncs.com.evil.example/compatible-mode/v1',
                         'https://key@dashscope.aliyuncs.com/compatible-mode/v1',
                         'https://dashscope.aliyuncs.com/compatible-mode/v1?key=secret']:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                qwen_roi.validate_endpoint(endpoint)

    def test_api_payload_ignores_injected_annotations_and_source_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'fixture.png'
            Image.new('RGB', (64, 64)).save(image)
            row = {**self.row, 'path': 'fixture.png', 'question': 'What is printed?',
                   'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                   'reference_answer': 'SECRET_REFERENCE', 'seed_box': [0, 0, 50, 50],
                   'source_group': 'ANSWER_LEAK', 'api_key': 'SECRET_CREDENTIAL'}
            config = {'model': 'fixture', 'request_parameters': {}, 'image_parameters': {}}
            with patch.object(qwen_roi, 'ROOT', root):
                payload = qwen_roi.make_payload(config, 'Locate evidence only.', row)
                serialized = json.dumps(payload)
                for secret in ['SECRET_REFERENCE', 'ANSWER_LEAK', 'SECRET_CREDENTIAL', 'seed_box']:
                    self.assertNotIn(secret, serialized)
                row['sha256'] = 'tampered'
                with self.assertRaises(ValueError):
                    qwen_roi.make_payload(config, 'Locate evidence only.', row)


if __name__ == '__main__':
    unittest.main()
