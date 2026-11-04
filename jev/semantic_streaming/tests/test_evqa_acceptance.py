"""Protect answer separation, typed QA and bounded acquisition in acceptance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qwen_roi as q
from acceptance_qa import parse_choice, qa_payload
from prepare_evqa_acceptance import RemoteZipFile
from run_evqa_acceptance import pipeline_config


class AcceptanceTests(unittest.TestCase):
    def test_pipeline_config_excludes_choices_answers_and_annotation_times(self):
        case = {'case_id': 'sample', 'video_path': 'video.mp4', 'video_sha256': 'hash',
                'question': 'What color?', 'start_s': 0, 'duration_s': 15,
                'qa_choices': ['red', 'blue'], 'reference_answer': 'PRIVATE_REFERENCE',
                'evidence_segments_s': [[7, 9]]}
        result = pipeline_config({'checkpoint': 'model'}, case)
        for field in ('qa_choices', 'reference_answer', 'evidence_segments_s'):
            self.assertNotIn(field, result)
        self.assertNotIn('PRIVATE_REFERENCE', repr(result))

    def test_receiver_only_receives_question_all_options_and_full_frames(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'ROOT', Path(tmp)):
            image = Path(tmp) / 'frame.png'
            image.write_bytes(b'unit-image')
            case = {'case_id': 'PRIVATE_CASE', 'variant': 'PRIVATE_VARIANT', 'question': 'What color?',
                    'options': {'A': 'red', 'B': 'blue'}, 'reference_answer': 'PRIVATE_CORRECT_LABEL'}
            config = {'model': 'unit', 'request_parameters': {}, 'image_parameters': {}}
            body = qa_payload(config, case, [{'path': 'frame.png', 'sha256': q.sha(image.read_bytes()), 'timestamp_s': 0}])
            serialized = json.dumps(body)
            for text in ('PRIVATE_CASE', 'PRIVATE_VARIANT', 'PRIVATE_CORRECT_LABEL'):
                self.assertNotIn(text, serialized)
            self.assertIn('red', serialized)
            self.assertIn('blue', serialized)

    def test_choice_parser_rejects_impossible_evidence_and_unanswerable_conflict(self):
        def response(answer):
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
        valid = {'choice': 'A', 'unanswerable': False, 'evidence_frame_indices': [0]}
        self.assertEqual(parse_choice(response(valid), {'A': 'red', 'B': 'blue'}, 12), valid)
        with self.assertRaises(ValueError):
            parse_choice(response({**valid, 'evidence_frame_indices': [12]}), {'A': 'red'}, 12)
        with self.assertRaises(ValueError):
            parse_choice(response({**valid, 'unanswerable': True}), {'A': 'red'}, 12)

    def test_archive_full_read_is_rejected_before_any_network_access(self):
        remote = RemoteZipFile('https://example.invalid/archive.zip', 9_000_000_000)
        with patch('prepare_evqa_acceptance.range_bytes') as request:
            with self.assertRaises(ValueError):
                remote.read()
            request.assert_not_called()
            self.assertEqual(remote.seek(-22, 2), 8_999_999_978)


if __name__ == '__main__':
    unittest.main()
