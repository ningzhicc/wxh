"""Guard invariants, fallback execution, teacher lookahead and no future leakage."""
import copy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from semantic_control import (candidate_guard, make_request, model_with_fallback,
                              plan_labels, run_control_episode, validate_config)
from semantic_rollout import online_request
from test_semantic_rollout import action, config as rollout_config


def config():
    return {**rollout_config(), 'planning_horizon_chunks': 3, 'planning_discount': .95,
            'terminal_buffer_weight': .25, 'target_buffer_s': 10, 'buffer_margin_s': 1,
            'buffer_spend_fraction': .25, 'forecast_bandwidth_factor': .85,
            'minimum_forecast_mbps': .000001, 'preview_grid_legend': 'fixed test grid'}


def state(buffer=5):
    return {'case_id': 'clip', 'question': 'Q', 'buffer_seconds': buffer,
            'recent_link_bandwidth_mbps': [1, 1], 'previous_bitrate_kbps': 200,
            'chunk_duration_seconds': 5, 'playback_started': True, 'remaining_playback_s': 30}


class SemanticControlTests(unittest.TestCase):
    def test_guard_excludes_oversized_and_preserves_all_budget_feasible_options(self):
        actions = [action('low', 100000), action('medium', 300000), action('large', 1000000)]
        allowed, guard = candidate_guard(state(), actions, config())
        self.assertEqual([a['action_id'] for a in allowed], ['low', 'medium'])
        self.assertEqual(guard['download_budget_s'], 4)
        self.assertTrue(all(guard['predicted_download_s'][a['action_id']] <= 4 for a in allowed))

    def test_guard_does_not_use_measured_quality_or_video_paths(self):
        actions = [action('low', 100000), action('high', 300000)]
        changed = copy.deepcopy(actions)
        changed[0]['offline_quality_proxy'] = 1000000
        changed[1]['video_path'] = 'DO_NOT_READ'
        original = candidate_guard(state(), actions, config())[1]
        self.assertEqual(original, candidate_guard(state(), changed, config())[1])

    def test_no_feasible_and_zero_observation_are_explicit_fallbacks(self):
        actions = [action('low', 100000), action('high', 300000)]
        s = state()
        s['recent_link_bandwidth_mbps'] = [0]
        allowed, guard = candidate_guard(s, actions, config())
        self.assertEqual([a['action_id'] for a in allowed], ['low'])
        self.assertTrue(guard['no_candidate_within_budget'])
        self.assertTrue(guard['predicted_download_s']['low'] > guard['download_budget_s'])

    def test_latest_drop_reduces_budget_eligible_set_without_future_data(self):
        actions = [action('low', 100000), action('high', 300000)]
        s = state()
        s['recent_link_bandwidth_mbps'] = [10, 10, .1]
        allowed, guard = candidate_guard(s, actions, config())
        self.assertAlmostEqual(guard['forecast_bandwidth_mbps'], .085)
        self.assertEqual([a['action_id'] for a in allowed], ['low'])

    def test_three_chunk_teacher_can_trade_immediate_quality_for_future_buffer(self):
        cfg = {**config(), 'rtt_s': 0, 'payload_fraction': 1, 'forecast_bandwidth_factor': 1,
               'planning_discount': 1, 'terminal_buffer_weight': 4, 'buffer_spend_fraction': 1}
        low, high = action('low', 250000), action('high', 687500)
        low['offline_quality_proxy'], high['offline_quality_proxy'] = 3, 4
        future = action('future', 750000)
        future['offline_quality_proxy'] = 3
        s = state(12)
        short = plan_labels(s, [[low, high]], {**cfg, 'planning_horizon_chunks': 1})
        full = plan_labels(s, [[low, high], [future], [future]], cfg)
        self.assertEqual(short['label'], 'high')
        self.assertEqual(full['label'], 'low')
        self.assertGreater(full['plans']['low']['forecast_terminal_buffer_s'], full['plans']['high']['forecast_terminal_buffer_s'])

    def test_teacher_has_no_terminal_reserve_requirement_at_stream_end(self):
        s = state()
        s['remaining_playback_s'] = 5
        result = plan_labels(s, [[action('a', 100000), action('b', 200000)]], config())
        self.assertEqual(result['terminal_buffer_target_s'], 0)

    def test_online_preview_exposes_sizes_without_answers_quality_or_future_link(self):
        actions = [action('a', 100000), action('b', 200000)]
        base = online_request(state(), actions, config(), random.Random(1))
        row = make_request(base, actions, [actions], 10, config())
        self.assertIn('5.0000:100000,200000', row['questions']['encoding']['instructions'])
        self.assertNotIn('next_chunks', row['state'])
        for forbidden in ('offline_quality_proxy', 'PRIVATE_FILE', 'evidence_rgb_psnr_db', 'candidate_reward', 'reference_answer'):
            self.assertNotIn(forbidden, repr(row))

    def test_runtime_guard_bypasses_forced_and_blocks_invalid_model_action(self):
        actions = {'clip': [action('a', 100000), action('b', 200000), action('huge', 5000000)]}
        spec = {'start_segment': 1, 'playlist': ['clip'] * 3, 'request_seed': 1}
        calls = []
        def bad(row):
            calls.append(row)
            self.assertNotIn('huge', row['questions']['encoding']['criteria'])
            return 'huge', {}
        rows = list(run_control_episode([[0, 0], [100, 1]], spec, actions, {'clip': 'Q'}, config(), bad))
        self.assertEqual(len(calls), 2)
        self.assertTrue(rows[0]['decision_info']['model_bypassed'])
        self.assertTrue(rows[1]['decision_info']['invalid_action_fallback'])
        self.assertTrue(all(r['selected_action'] == 'a' for r in rows))
        self.assertAlmostEqual(rows[1]['request']['state']['previous_bitrate_kbps'], 160)

    def test_invalid_prediction_falls_back_but_hardware_failure_is_not_hidden(self):
        actions = {'clip': [action('a', 100000), action('b', 200000)]}
        row = {'state': state(), 'questions': {'encoding': {'criteria': {'a': 'x', 'b': 'y'}}}}
        def invalid(_):
            raise ValueError('invalid probabilities')
        selected, info = model_with_fallback(invalid, actions)(row)
        self.assertEqual(selected, 'a')
        self.assertEqual(info['model_prediction_fallback'], 'ValueError')
        def hardware(_):
            raise RuntimeError('CUDA failure')
        with self.assertRaises(RuntimeError):
            model_with_fallback(hardware, actions)(row)

    def test_invalid_control_configuration_is_rejected(self):
        validate_config(config())
        with self.assertRaises(ValueError):
            validate_config({**config(), 'planning_horizon_chunks': 0})


if __name__ == '__main__':
    unittest.main()
