"""Revised tasks, repeated identical QA and disputed-label isolation."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_evqa_disambiguation import revised_tasks,repeated_plan,consensus,BOWL,TOWEL,CONTROL
from acceptance_qa import qa_payload
import qwen_roi as q
from unittest.mock import patch
import tempfile


class DisambiguationTests(unittest.TestCase):
    def tasks(self):
        parents=[{'case_id':f'evqa_0{i}','question':question,'qa_choices':options,'video_path':f'v{i}.mp4'}
            for i,question,options in [(1,'What color was the bowl I saw?',['light blue','black','pink','purple']),
                                       (2,'What color is the towel?',['green','purple','red','blue'])]]
        refs={p['case_id']:{'reference_answer':p['qa_choices'][0],'evidence_segments_s':[[7,9]],'local_human_verified':False} for p in parents}
        return revised_tasks(parents,refs)

    def test_new_identity_separate_answers_and_original_control(self):
        tasks,refs=self.tasks();self.assertEqual(len(tasks),3)
        self.assertEqual(len({t['video_path'] for t in tasks}),2)
        self.assertEqual(tasks[2]['question'],'What color is the towel?')
        self.assertNotIn('light blue',tasks[0]['question']);self.assertNotIn('green',tasks[1]['question'])
        for task in tasks:
            for key in ('reference_answer','evidence_segments_s','local_human_verified'):self.assertNotIn(key,task)
        self.assertTrue(all(not r['training_eligible'] and not r['local_human_verified'] for r in refs))
        self.assertEqual(refs[1]['reference_answer'],'green')

    def test_fifteen_calls_repeat_identical_wire_but_question_control_changes_it(self):
        tasks,_=self.tasks()
        with tempfile.TemporaryDirectory() as tmp,patch.object(q,'ROOT',Path(tmp)):
            p=Path(tmp)/'frame.png';p.write_bytes(b'unit-image')
            frames=[{'path':p.name,'sha256':q.sha(p.read_bytes()),'timestamp_s':8}]
            plan=repeated_plan(tasks,{v:frames for v in ('reference','bandwidth','kev')},frames)
            self.assertEqual(len(plan),15)
            cfg={'model':'unit','request_parameters':{},'image_parameters':{}}
            groups={}
            for row in plan:
                key=(row['case_id'],row['variant'].rsplit('_r',1)[0]);body=qa_payload(cfg,row,row['frames'])
                groups.setdefault(key,set()).add(json.dumps(body,sort_keys=True))
            self.assertEqual(len(groups),5);self.assertTrue(all(len(v)==1 for v in groups.values()))
            self.assertNotEqual(groups[(TOWEL,'reference')],groups[(CONTROL,'reference')])

    def test_consensus_keeps_disagreement_and_unanswerable_without_changing_labels(self):
        tasks,_=self.tasks();plan=repeated_plan(tasks,{v:[] for v in ('reference','bandwidth','kev')},[])
        subset=[r for r in plan if r['case_id']==BOWL and r['variant'].startswith('reference_')]
        letters=list(subset[0]['options']);rows=[]
        for index,row in enumerate(subset):
            rows.append({'case_id':row['case_id'],'variant':row['variant'],'answer':{'choice':letters[index] if index<2 else ''},'correct':index==0})
        result=consensus(rows,subset)[0]
        self.assertFalse(result['all_repeats_agree']);self.assertEqual(result['reference_matches'],1)
        self.assertIn('UNANSWERABLE',result['answer_histogram'])

if __name__=='__main__':unittest.main()
