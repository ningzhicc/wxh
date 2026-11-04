"""Free-text repetition, grounding-label separation and first-frame scoring."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import qwen_roi as q
import repeated_open_qa as qa
from run_clear_tasks import task_config,qa_plan


class ClearTasksTests(unittest.TestCase):
    def test_first_frame_reading_cannot_be_replaced_by_later_identical_answer(self):
        label={'accepted_answers':['04:30','4:30']}
        answer={'answer':'4:30','unanswerable':False,'evidence_frame_indices':[0]}
        self.assertEqual(qa.match_answer(answer,label,True),(True,True))
        for indices in ([1],[0,1],[]):
            self.assertFalse(qa.match_answer({**answer,'evidence_frame_indices':indices},label,True)[0])
        self.assertFalse(qa.match_answer({**answer,'answer':'4:26'},label,True)[0])
        self.assertFalse(qa.match_answer({'answer':'2K3OH','unanswerable':False,'evidence_frame_indices':[0]}, {'accepted_answers':['2K30H']},False)[0])

    def test_payload_excludes_private_metadata_and_repeat_ids(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(q,'ROOT',Path(tmp)):
            image=Path(tmp)/'f.png';image.write_bytes(b'unit-image')
            row={'case_id':'PRIVATE_CASE','variant':'PRIVATE_POLICY','repeat':1,'question':'What is written?',
                'reference_answer':'PRIVATE_ANSWER','roi_box':'PRIVATE_ROI','first_frame_only':True,
                'frames':[{'path':'f.png','sha256':q.sha(image.read_bytes()),'timestamp_s':0}]}
            cfg={'model':'unit','request_parameters':{},'image_parameters':{}}
            a=qa.make_payload(cfg,'unit',row);b=qa.make_payload(cfg,'unit',{**row,'repeat':3,'variant':'another'})
            self.assertEqual(a,b);self.assertNotIn('PRIVATE_',json.dumps(a))

    def test_three_case_plan_has_27_calls_and_config_omits_annotation_fields(self):
        cases=[{'case_id':str(i),'question':'Q','first_frame_only':i==1,'video_path':'v.mp4','video_sha256':'h',
                'start_s':10,'reference_answer':'PRIVATE','evidence_segments_s':[[1,2]]} for i in range(3)]
        frames={(str(i),v):[] for i in range(3) for v in ('reference','bandwidth','kev')}
        plan=qa_plan(cases,frames);self.assertEqual(len(plan),27)
        cfg=task_config({},cases[0]);self.assertNotIn('PRIVATE',repr(cfg));self.assertNotIn('evidence_segments_s',cfg)
        self.assertEqual(cfg['chunk_s'],2);self.assertEqual(cfg['max_api_calls'],18)

    def test_ambiguous_paid_request_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(q,'ROOT',Path(tmp)):
            root=Path(tmp);image=root/'f.png';image.write_bytes(b'unit-image')
            run=root/'run';q.write_json(run/'qa/sample_00/status.json',{'state':'in_flight'})
            row={'case_id':'c','variant':'kev','repeat':1,'question':'Q','frames':[{'path':'f.png','sha256':q.sha(image.read_bytes()),'timestamp_s':0}]}
            cfg={'model':'unit','base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','request_parameters':{},'image_parameters':{}}
            with patch.object(q,'load_secret') as secret:
                with self.assertRaises(ValueError):qa.run_qa(run,[row],root/'labels',cfg,'unit')
                secret.assert_not_called()

if __name__=='__main__':unittest.main()
