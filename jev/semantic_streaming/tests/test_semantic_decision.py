"""Check replay units, trace wrap and absence of privileged quality features."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_semantic_kev import replay_download_s,request


class SemanticDecisionTests(unittest.TestCase):
    def test_replay_matches_constant_link_and_wraps_without_mutating_trace(self):
        trace=[[0,0],[1,1],[2,1]]
        self.assertAlmostEqual(replay_download_s(trace,1,237500),2.08)
        self.assertAlmostEqual(replay_download_s(trace,2,356250),3.08)
        self.assertEqual(trace,[[0,0],[1,1],[2,1]])

    def test_action_timing_follows_bandwidth_changes(self):
        trace=[[0,0],[1,1],[2,2]]
        # 118750 bytes in first second, then 118750 / 237500 = 0.5 seconds.
        self.assertAlmostEqual(replay_download_s(trace,1,237500),1.58)

    def test_quality_and_paths_never_enter_kev_candidate_text(self):
        state={'recent_link_bandwidth_mbps':[1,2],'buffer_seconds':1}
        actions=[{'action_id':'a0','mode':'roi','size_bytes':10000,'actual_bitrate_kbps':16,
                  'roi_qoffset':'-1/5','offline_quality_proxy':.99,'video_path':'PRIVATE_PATH'}]
        for compact in [False,True]:
            with self.subTest(compact=compact):
                row=request(state,actions,compact=compact)
                fields=row['questions']['encoding']['criteria']['a0']
                self.assertNotIn('offline_quality_proxy',fields)
                self.assertNotIn('video_path',fields)
                self.assertNotIn('PRIVATE_PATH',repr(fields))


if __name__=='__main__': unittest.main()
