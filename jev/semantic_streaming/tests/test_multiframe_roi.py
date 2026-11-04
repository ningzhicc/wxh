"""Joint-frame identity, label separation, exact paid-call resume and failure safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qwen_roi as q
import pipeline_roi_multiframe as joint


def response(frames):
    return {'model': 'unit', 'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps({'frames': frames})}}], 'usage': {'total_tokens': 10}}


class JointTests(unittest.TestCase):
    def rows(self, root):
        out = []
        for n in range(2):
            from PIL import Image
            path = root / f'f{n}.png'
            Image.new('RGB', (32, 16), 'white').save(path)
            out.append({'case_id': 'chunk_0000', 'frame_id': f'f{n}', 'path': path.name,
                'sha256': q.sha(path.read_bytes()), 'question': 'What color is the object?',
                'timestamp_s': n, 'width': 32, 'height': 16, 'answer': 'PRIVATE_LABEL',
                'evidence_segments_s': [[1234, 5678]], 'mask': 'PRIVATE_MASK'})
        return out

    def config(self):
        return {'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'unit',
            'request_parameters': {}, 'image_parameters': {}, 'coordinate_contract': {'scale_denominator': 1000}}

    def test_payload_excludes_labels_masks_choices_and_preserves_every_frame(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'ROOT', Path(tmp)):
            rows = self.rows(Path(tmp))
            body = joint.make_payload(self.config(), 'unit', rows)
            wire = json.dumps(body)
            for token in ('PRIVATE_LABEL', 'PRIVATE_MASK', '1234', 'evidence_segments_s'):
                self.assertNotIn(token, wire)
            self.assertEqual(sum(x['type'] == 'image_url' for x in body['messages'][1]['content']), 2)
            with self.assertRaises(ValueError):
                joint.make_payload(self.config(), 'unit', [rows[0], rows[0]])
            with self.assertRaises(ValueError):
                joint.make_payload(self.config(), 'unit', rows[::-1])

    def test_parser_requires_full_frame_coverage_and_maps_out_of_order_by_identity(self):
        rows = [{'case_id': 'c', 'frame_id': f'f{i}', 'timestamp_s': i,
            'width': 32, 'height': 16, 'sha256': 'x'} for i in range(2)]
        boxes = [{'frame_id': 'f1', 'regions': [{'bbox_2d': [100,200,300,400]}]},
                 {'frame_id': 'f0', 'regions': []}]
        parsed = joint.parse_regions(response(boxes), rows)
        self.assertEqual([r['frame_id'] for r in parsed], ['f0', 'f1'])
        self.assertEqual(parsed[1]['regions'][0]['bbox_pixels'], [3,3,10,7])
        for bad in (boxes[:1], [boxes[0], boxes[0]], [boxes[0], {**boxes[1], 'frame_id': 'unknown'}],
                    [boxes[1], {'frame_id': 'f1', 'regions': [{'bbox_2d': [True,0,999,999]}]}]):
            with self.assertRaises(ValueError):
                joint.parse_regions(response(bad), rows)

    def test_paid_call_occurs_once_for_two_frames_and_resume_verifies_raw_response(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'ROOT', Path(tmp)):
            root = Path(tmp); rows = self.rows(root); run = root / 'run'; run.mkdir()
            raw = json.dumps(response([{'frame_id': r['frame_id'], 'regions': []} for r in rows])).encode()
            handle = Mock(); handle.__enter__ = Mock(return_value=handle); handle.__exit__ = Mock(return_value=False)
            handle.read.return_value = raw
            opener = Mock(); opener.open.return_value = handle
            cfg = {'roi_mode': 'api', 'max_api_calls': 1}
            with patch.object(q, 'load_secret', return_value='UNIT_SECRET') as secret, patch.object(joint.urllib.request, 'build_opener', return_value=opener):
                first = joint.locate(rows, run, cfg, self.config(), b'unit')
                second = joint.locate(rows, run, cfg, self.config(), b'unit')
                self.assertEqual(first, second); self.assertEqual(opener.open.call_count, 1); self.assertEqual(secret.call_count, 1)
                self.assertEqual(json.loads((run/'roi_summary.json').read_text())['new_api_calls'], 1)
                (run/'roi/joint_0000/response.json').write_text('{}')
                with self.assertRaises(ValueError):
                    joint.locate(rows, run, cfg, self.config(), b'unit')
            for path in run.rglob('*'):
                if path.is_file():
                    self.assertNotIn(b'UNIT_SECRET', path.read_bytes())

    def test_unknown_outcome_and_zero_cap_never_request_or_load_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'ROOT', Path(tmp)):
            root=Path(tmp); rows=self.rows(root); run=root/'run'; run.mkdir()
            q.write_json(run/'roi/joint_0000/status.json', {'state': 'in_flight'})
            with patch.object(q, 'load_secret') as secret:
                for cfg in ({'roi_mode':'api','max_api_calls':1}, {'roi_mode':'api','max_api_calls':0},
                            {'roi_mode':'cache','max_api_calls':1}, {'roi_mode':'api','max_api_calls':1,'evidence_window_s':[7,9]}):
                    with self.assertRaises(ValueError):
                        joint.locate(rows,run,cfg,self.config(),b'unit')
                secret.assert_not_called()


if __name__ == '__main__':
    unittest.main()
