"""Analytic checks of physical timing, state feedback and privileged-input separation."""
import copy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from semantic_rollout import (Cursor, advance_time, download, outcome, quality_utility,
                              run_episode, safe_baseline_policy)
from evaluate_semantic_rollout import prediction_record


def config():
    return {'quality_scale_db': 10, 'rtt_s': .08, 'payload_fraction': .95,
            'stall_weight': 4.3, 'startup_weight': 4.3, 'switch_weight': .05,
            'max_buffer_s': 30, 'startup_bandwidth_prior_mbps': .3,
            'bandwidth_safety_factor': .85, 'exploration_probability': .2, 'history_length': 5}


def action(key, size, duration=5):
    return {'action_id': key, 'size_bytes': size, 'actual_bitrate_kbps': size * 8 / duration / 1000,
            'duration_s': duration, 'mode': 'roi', 'roi_qoffset': '-1/5', 'global_crf': 32,
            'offline_quality_proxy': 4.0, 'video_path': 'PRIVATE_FILE', 'evidence_rgb_psnr_db': 40}


class SemanticRolloutTests(unittest.TestCase):
    def test_fractional_cursor_and_rtt_cross_a_bandwidth_boundary(self):
        trace = [[0, 0], [1, 1], [2, 2]]
        result = download(trace, Cursor(1, .5), 118750, rtt_s=.2)
        # RTT leaves .3s of 1 Mbps; the remaining .7 Mbps*s takes .35s at 2 Mbps.
        self.assertAlmostEqual(result['download_s'], .85)
        self.assertEqual(result['cursor'].segment, 2)
        self.assertAlmostEqual(result['cursor'].offset_s, .35)
        self.assertEqual(trace, [[0, 0], [1, 1], [2, 2]])

    def test_wrap_with_zero_capacity_and_duplicate_timestamps(self):
        trace = [[0, 0], [0, 100], [1, 0], [2, 1]]
        result = download(trace, Cursor(), 237500, rtt_s=0)
        self.assertAlmostEqual(result['download_s'], 4)
        self.assertEqual(result['cursor'], Cursor(2, 0))
        self.assertEqual(advance_time(trace, Cursor(3, .5), 4.25), Cursor(3, .75))

    def test_same_cursor_counterfactuals_are_order_independent(self):
        trace = [[0, 0], [1, 1], [2, 3]]
        cursor = Cursor(1, .4)
        results = [download(trace, cursor, size) for size in (10000, 300000, 10000)]
        self.assertEqual(results[0], results[2])
        self.assertEqual(cursor, Cursor(1, .4))

    def test_continuous_buffer_and_previous_action_follow_actual_selection(self):
        cfg = config()
        actions = {'clip': [action('small', 118750), action('large', 237500)]}
        spec = {'start_segment': 1, 'playlist': ['clip'] * 4, 'request_seed': 666}
        selected = iter(['small', 'large', 'small', 'small'])
        def choose(row):
            self.assertNotIn('PRIVATE_FILE', repr(row))
            self.assertNotIn('offline_quality_proxy', repr(row))
            self.assertNotIn('candidate_actual_download_s', repr(row))
            return next(selected), {}
        rows = list(run_episode([[0, 0], [100, 1]], spec, actions, {'clip': 'What text?'}, cfg, choose))
        self.assertAlmostEqual(rows[0]['transition']['startup_delay_s'], 1.08)
        self.assertEqual(rows[0]['transition']['rebuffer_s'], 0)
        self.assertAlmostEqual(rows[1]['request']['state']['buffer_seconds'], 5)
        self.assertAlmostEqual(rows[1]['request']['state']['previous_bitrate_kbps'], 190)
        self.assertAlmostEqual(rows[2]['request']['state']['buffer_seconds'], 7.92)
        self.assertAlmostEqual(rows[2]['request']['state']['previous_bitrate_kbps'], 380)
        self.assertAlmostEqual(rows[2]['transition']['elapsed_before_s'], 3.16)
        self.assertEqual(rows[2]['request']['state']['recent_link_bandwidth_mbps'], [1.0, 1.0])

    def test_buffer_cap_wait_advances_network_clock(self):
        cfg = config()
        cfg['max_buffer_s'] = 5
        actions = {'clip': [action('a', 118750)]}
        spec = {'start_segment': 1, 'playlist': ['clip'] * 3, 'request_seed': 1}
        rows = list(run_episode([[0, 0], [100, 1]], spec, actions, {'clip': 'Q'}, cfg, lambda _: ('a', {})))
        self.assertAlmostEqual(rows[1]['transition']['buffer_cap_wait_s'], 3.92)
        self.assertAlmostEqual(rows[2]['transition']['elapsed_before_s'], 6.08)
        self.assertAlmostEqual(rows[2]['request']['state']['buffer_seconds'], 5)

    def test_future_bandwidth_cannot_change_the_first_online_request(self):
        cfg = config()
        actions = {'clip': [action('a', 118750)]}
        spec = {'start_segment': 1, 'playlist': ['clip'], 'request_seed': 1}
        rows = [list(run_episode(t, spec, actions, {'clip': 'Q'}, cfg, lambda _: ('a', {})))[0]
                for t in ([[0, 0], [10, 1]], [[0, 0], [10, 100]])]
        self.assertEqual(rows[0]['request'], rows[1]['request'])
        self.assertNotEqual(rows[0]['downloads_s'], rows[1]['downloads_s'])

    def test_quality_has_no_upper_clipping_and_policy_ignores_offline_scores(self):
        self.assertEqual(quality_utility(50, config()) - quality_utility(40, config()), 1)
        actions = {'clip': [action('a', 118750), action('b', 237500)]}
        changed = copy.deepcopy(actions)
        changed['clip'][0]['offline_quality_proxy'] = 100000
        state = {'case_id': 'clip', 'recent_link_bandwidth_mbps': [10],
                 'buffer_seconds': 10, 'chunk_duration_seconds': 5, 'playback_started': True}
        row = {'state': state}
        for name in ('cheapest', 'bandwidth', 'epsilon_bandwidth'):
            self.assertEqual(safe_baseline_policy(name, actions, config(), 1)(row),
                             safe_baseline_policy(name, changed, config(), 1)(row))

    def test_closed_loop_policies_accumulate_different_actual_states(self):
        cfg = config()
        actions = {'clip': [action('small', 118750), action('large', 237500)]}
        spec = {'start_segment': 1, 'playlist': ['clip'] * 4, 'request_seed': 1}
        trace = [[0, 0], [100, 1]]
        runs = [list(run_episode(trace, spec, actions, {'clip': 'Q'}, cfg,
                               safe_baseline_policy(name, actions, cfg, 1))) for name in ('cheapest', 'bandwidth')]
        self.assertEqual(runs[0][0]['selected_action'], runs[1][0]['selected_action'])
        self.assertNotEqual(runs[0][2]['request']['state']['buffer_seconds'], runs[1][2]['request']['state']['buffer_seconds'])

    def test_predictor_materializer_receives_only_a_dummy_label(self):
        row = {'state': {'case_id': 'clip'}, 'questions': {'encoding': {
            'type': 'choice', 'criteria': {'a': 'low', 'b': 'high'}, 'src': 'test'}}}
        original = copy.deepcopy(row)
        converted = prediction_record(row)
        self.assertEqual(converted['questions']['encoding']['label'], 'a')
        self.assertEqual(row, original)


if __name__ == '__main__':
    unittest.main()
