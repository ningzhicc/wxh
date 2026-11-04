"""Context changes must change the request; labels cannot enter the wire payload."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import qwen_roi as q
from pipeline_roi_context_v4 import context_payload


class ContextTests(unittest.TestCase):
    def test_target_is_last_and_context_is_bound_without_answer_metadata(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(q,'ROOT',Path(tmp)):
            rows=[]
            for i in range(3):
                path=Path(tmp)/f'{i}.png';path.write_bytes(f'image{i}'.encode())
                rows.append({'frame_id':f'f{i}','case_id':'chunk0','question':'What color?',
                    'timestamp_s':i,'path':path.name,'sha256':q.sha(path.read_bytes()),
                    'reference_answer':'PRIVATE_GT','mask':'PRIVATE_MASK','evidence_time':[999,1000]})
            cfg={'model':'unit','request_parameters':{},'image_parameters':{}}
            body=context_payload(cfg,'unit',rows[1],rows);wire=json.dumps(body)
            self.assertNotIn('PRIVATE_',wire);self.assertNotIn('evidence_time',wire)
            content=body['messages'][1]['content']
            self.assertEqual(sum(v['type']=='image_url' for v in content),3)
            self.assertEqual(json.loads(content[-1]['text'])['frame_id'],'f1')
            shorter=context_payload(cfg,'unit',rows[1],rows[1:])
            self.assertNotEqual(wire,json.dumps(shorter))
            for bad in ([rows[0]], [rows[0],rows[1],rows[1]],rows[::-1], [rows[1],{**rows[2],'case_id':'other'}]):
                with self.assertRaises(ValueError):context_payload(cfg,'unit',rows[1],bad)
            (Path(tmp)/'0.png').write_bytes(b'changed-context')
            with self.assertRaises(ValueError):context_payload(cfg,'unit',rows[1],rows)

if __name__=='__main__':unittest.main()
