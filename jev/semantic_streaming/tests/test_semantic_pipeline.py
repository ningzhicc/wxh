"""Continuous frame boundaries, strict cache matching and no ambiguous API retry."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qwen_roi as q
from pipeline_media import partitions, sampling, measured_quality_utility
from pipeline_roi import matching_key, locate
from run_semantic_pipeline import seal_paths, verify_files


class PipelineTests(unittest.TestCase):
    def test_native_fractional_fps_partition_is_contiguous_without_padding(self):
        spans = partitions(150, '30000/1001', 2)
        self.assertEqual(spans, [(0, 60), (60, 120), (120, 150)])
        self.assertEqual([i for a, b in spans for i in range(a, b)], list(range(150)))

    def test_single_frame_tail_is_preserved_by_merging_not_dropped(self):
        self.assertEqual(partitions(121, '30000/1001', 2), [(0, 60), (60, 121)])
        with self.assertRaises(ValueError):
            partitions(1, '30', 2)

    def test_requested_and_actual_sample_times_remain_distinct(self):
        spans = partitions(150, '30000/1001', 2)
        cfg = {'global_sample_times_s': [0, 1, 2, 3, 4, 4.8]}
        samples = sampling(spans, '30000/1001', cfg)
        self.assertEqual(list(samples), [0, 30, 60, 90, 120, 144])
        self.assertEqual(samples[30], 1)
        self.assertAlmostEqual(float(30 / Fraction('30000/1001')), 1.001)
        with self.assertRaises(ValueError):
            sampling(spans, '30000/1001', {'global_sample_times_s': [0]})

    def test_cached_localization_requires_same_pixels_question_time_and_dimensions(self):
        row = {'sha256': 'image', 'question': 'Q', 'timestamp_s': 1, 'width': 1280, 'height': 720}
        for field, other in [('sha256', 'other'), ('question', 'different'), ('timestamp_s', 2), ('width', 640)]:
            self.assertNotEqual(matching_key(row), matching_key({**row, field: other}))

    def test_zero_mse_is_not_mislabelled_as_missing_roi(self):
        metrics = {'sampled_api_visible_evidence_rgb_psnr_db': None, 'sampled_full_frame_rgb_psnr_db': 35}
        utility, basis = measured_quality_utility(metrics, True, 10)
        self.assertEqual(utility, 6)
        self.assertIn('API_evidence', basis)
        self.assertIsNone(metrics['sampled_api_visible_evidence_rgb_psnr_db'])
        self.assertEqual(measured_quality_utility(metrics, False, 10)[0], 3.5)

    def test_sealed_stage_rejects_changed_media(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'ROOT', Path(tmp)):
            root = Path(tmp)
            media = root / 'artifacts' / 'sample.mp4'
            media.parent.mkdir()
            media.write_bytes(b'original-video')
            sealed = seal_paths([media])
            verify_files(sealed)
            media.write_bytes(b'modified-video')
            with self.assertRaises(ValueError):
                verify_files(sealed)

    def test_ambiguous_paid_request_is_not_retried_and_key_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'ROOT', Path(tmp)):
            root = Path(tmp)
            run = root / 'run'
            run.mkdir()
            row = {'case_id': 'chunk', 'frame_id': 'frame', 'path': 'image.png', 'sha256': 'image',
                   'question': 'Q', 'timestamp_s': 0, 'actual_window_timestamp_s': 0, 'width': 100, 'height': 100}
            cfg = {'roi_mode': 'api', 'max_api_calls': 1}
            api = {'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'unit',
                   'coordinate_contract': {'scale_denominator': 1000}}
            opener = Mock()
            opener.open.side_effect = TimeoutError()
            secret = 'unit-secret-must-not-be-retained'
            with patch.object(q, 'load_secret', return_value=secret), patch.object(q, 'make_payload', return_value={'model': 'unit'}), patch('pipeline_roi.urllib.request.build_opener', return_value=opener):
                with self.assertRaises(RuntimeError):
                    locate([row], run, cfg, api, b'prompt')
                with self.assertRaises(ValueError):
                    locate([row], run, cfg, api, b'prompt')
            self.assertEqual(opener.open.call_count, 1)
            status = json.loads((run / 'roi/frame/status.json').read_text())
            self.assertEqual(status['state'], 'failed_no_retry')
            for path in run.rglob('*'):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes())


if __name__ == '__main__':
    unittest.main()
