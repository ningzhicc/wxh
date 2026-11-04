"""Verify answer matching and blinding of compression treatments."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import qwen_roi
import receiver_qa


class ReceiverQATests(unittest.TestCase):
    def test_matching_ignores_format_but_preserves_ambiguous_plate_characters(self):
        self.assertEqual(receiver_qa.normalize_answer('04:30'),receiver_qa.normalize_answer('4:30'))
        self.assertEqual(receiver_qa.normalize_answer('Math Antics'),receiver_qa.normalize_answer('math antics.'))
        self.assertNotEqual(receiver_qa.normalize_answer('2K30H'),receiver_qa.normalize_answer('2K3OH'))

    def test_decimal_price_or_reading_cannot_match_missing_decimal(self):
        self.assertNotEqual(receiver_qa.normalize_answer('1.49'),receiver_qa.normalize_answer('149'))
        self.assertEqual(receiver_qa.normalize_answer('1.49'),receiver_qa.normalize_answer('1,49'))
        self.assertEqual(receiver_qa.normalize_answer('16.50'),receiver_qa.normalize_answer('16.5'))

    def test_full_frame_payload_blinds_treatment_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'roi_encoded.png'
            Image.new('RGB',(64,64)).save(path)
            frame={'path':path.name,'timestamp_s':0,'sha256':qwen_roi.sha(path.read_bytes())}
            c={'candidate_id':'SECRET_CANDIDATE','question':'Read the label.',
               'treatment':'SECRET_TREATMENT','target_bytes':12345,'roi_boxes':'SECRET_ROI',
               'reference_answer':'SECRET_REFERENCE','frames':[frame]}
            cfg={'model':'fixture','request_parameters':{},'image_parameters':{}}
            with patch.object(qwen_roi,'ROOT',root):
                wire=json.dumps(receiver_qa.payload(cfg,'Only answer from images.',c))
            for secret in ['SECRET_CANDIDATE','SECRET_TREATMENT','SECRET_ROI','SECRET_REFERENCE','roi_encoded.png','12345']:
                self.assertNotIn(secret,wire)


if __name__=='__main__':
    unittest.main()
