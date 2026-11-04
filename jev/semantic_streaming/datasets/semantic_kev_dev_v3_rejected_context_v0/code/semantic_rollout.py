"""Continuous, CPU-only transmission physics shared by data collection and evaluation.

Traces use right-end samples in Mbps. Cursors retain the fractional position within
a trace interval, including RTT and buffer-cap waits. No future samples enter a
policy request. The playlist is a synthetic sequence of existing development clips.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import random

from prepare_semantic_kev import request


@dataclass(frozen=True)
class Cursor:
    segment: int = 1
    offset_s: float = 0.0


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).digest()[:8], 'big')


def normalize_cursor(trace, cursor):
    i, offset = cursor.segment, cursor.offset_s
    if not 1 <= i < len(trace) or not math.isfinite(offset) or offset < 0:
        raise ValueError('Invalid trace cursor')
    for _ in range(len(trace)):
        duration = trace[i][0] - trace[i - 1][0]
        if offset < duration:
            return Cursor(i, offset)
        offset -= duration
        i = i + 1 if i + 1 < len(trace) else 1
    raise ValueError('Cursor offset exceeds one trace cycle')


def advance_time(trace, cursor, seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid elapsed time')
    cursor = normalize_cursor(trace, cursor)
    remaining = seconds % trace[-1][0]
    while remaining > 0:
        available = trace[cursor.segment][0] - trace[cursor.segment - 1][0] - cursor.offset_s
        if remaining < available:
            return Cursor(cursor.segment, cursor.offset_s + remaining)
        remaining -= available
        cursor = normalize_cursor(trace, Cursor(cursor.segment, cursor.offset_s + available))
    return cursor


def download(trace, cursor, size_bytes, payload_fraction=.95, rtt_s=.08):
    if size_bytes <= 0 or not 0 < payload_fraction <= 1 or rtt_s < 0:
        raise ValueError('Invalid transfer parameters')
    cursor = advance_time(trace, cursor, rtt_s)
    remaining, elapsed = float(size_bytes), 0.0
    while remaining > 0:
        i = cursor.segment
        available = trace[i][0] - trace[i - 1][0] - cursor.offset_s
        rate = trace[i][1] * 1e6 / 8 * payload_fraction
        if rate > 0 and remaining <= available * rate:
            duration = remaining / rate
            elapsed += duration
            cursor = advance_time(trace, cursor, duration)
            break
        remaining -= available * rate
        elapsed += available
        cursor = normalize_cursor(trace, Cursor(i, cursor.offset_s + available))
    return {'cursor': cursor, 'download_s': elapsed + rtt_s, 'payload_s': elapsed,
            # Estimate link capacity ONLY from the completed transfer, excluding RTT.
            'observed_link_mbps': size_bytes * 8 / (elapsed * payload_fraction * 1e6)}


def quality_utility(psnr_db, cfg):
    if not math.isfinite(psnr_db):
        raise ValueError('A finite measured evidence PSNR is required')
    # No upper clipping; an extra ten dB is one utility unit by default.
    return psnr_db / cfg['quality_scale_db']


def outcome(action, delay, state, cfg):
    startup = not state['playback_started']
    wait = max(0.0, delay - state['buffer_seconds'])
    startup_s, stall_s = (delay, 0.0) if startup else (0.0, wait)
    change = abs(action['actual_bitrate_kbps'] - state['previous_bitrate_kbps']) / 1000 if not startup else 0.0
    reward = (action['offline_quality_proxy'] - cfg['stall_weight'] * stall_s
              - cfg['startup_weight'] * startup_s - cfg['switch_weight'] * change)
    raw_buffer = max(0.0, state['buffer_seconds'] - delay) + action['duration_s']
    idle = max(0.0, raw_buffer - cfg['max_buffer_s'])
    return {'reward': reward, 'rebuffer_s': stall_s, 'startup_delay_s': startup_s,
            'buffer_after_s': raw_buffer - idle, 'buffer_cap_wait_s': idle,
            'bitrate_change_mbps': change, 'quality_utility': action['offline_quality_proxy']}


def estimated_download(action, state, cfg):
    positive = [x for x in state['recent_link_bandwidth_mbps'] if x > 0]
    harmonic = len(positive) / sum(1 / x for x in positive) if positive else cfg['startup_bandwidth_prior_mbps']
    return cfg['rtt_s'] + action['size_bytes'] * 8 / (harmonic * 1e6 * cfg['payload_fraction'])


def choose_baseline(name, state, actions, cfg, rng):
    cheapest = min(actions, key=lambda a: (a['size_bytes'], a['action_id']))
    if name == 'cheapest' or not state['playback_started']:
        return cheapest['action_id']
    if name not in ('bandwidth', 'epsilon_bandwidth'):
        raise ValueError('Unknown behavior policy: ' + name)
    budget = cfg['bandwidth_safety_factor'] * state['chunk_duration_seconds']
    if state['playback_started']:
        budget = min(budget, state['buffer_seconds'])
    safe = [a for a in actions if estimated_download(a, state, cfg) <= budget]
    if not safe:
        return cheapest['action_id']
    if name == 'epsilon_bandwidth' and rng.random() < cfg['exploration_probability']:
        return rng.choice(safe)['action_id']
    # A simple rate baseline has no access to measured quality or future bandwidth.
    return max(safe, key=lambda a: (a['size_bytes'], a['action_id']))['action_id']


def online_request(state, actions, cfg, rng):
    shuffled = list(actions)
    rng.shuffle(shuffled)
    return request(dict(state), shuffled, cfg['rtt_s'], cfg['payload_fraction'], compact=True)


def run_episode(trace, spec, actions, questions, cfg, choose, counterfactual=True):
    """Yield requests BEFORE selection and the resulting actual state transitions.

    choose receives only the online request, never action quality, labels or trace.
    Counterfactual outcomes are confined to the returned offline supervision.
    """
    cursor = normalize_cursor(trace, Cursor(spec['start_segment'], spec.get('start_offset_s', 0)))
    buffer = 0.0
    previous = 0.0
    history = []
    elapsed = 0.0
    rng = random.Random(spec['request_seed'])
    for step, cid in enumerate(spec['playlist']):
        available = actions[cid]
        state = {'question': questions[cid], 'case_id': cid,
                 'recent_link_bandwidth_mbps': list(history) if history else [cfg['startup_bandwidth_prior_mbps']],
                 'buffer_seconds': buffer, 'previous_bitrate_kbps': previous,
                 'chunk_duration_seconds': available[0]['duration_s'], 'playback_started': step > 0}
        row = online_request(state, available, cfg, rng)
        selected, decision_info = choose(row)
        lookup = {a['action_id']: a for a in available}
        if selected not in lookup:
            raise ValueError('Policy returned an unavailable action')
        targets = available if counterfactual else [lookup[selected]]
        transfers = {a['action_id']: download(trace, cursor, a['size_bytes'], cfg['payload_fraction'], cfg['rtt_s']) for a in targets}
        outcomes = {a['action_id']: outcome(a, transfers[a['action_id']]['download_s'], state, cfg) for a in targets}
        result = outcomes[selected]
        transfer = transfers[selected]
        next_cursor = advance_time(trace, transfer['cursor'], result['buffer_cap_wait_s'])
        end_elapsed = elapsed + transfer['download_s'] + result['buffer_cap_wait_s']
        record = {'request': row, 'selected_action': selected, 'decision_info': decision_info,
                  'outcomes': outcomes, 'downloads_s': {k: v['download_s'] for k, v in transfers.items()},
                  'transition': {**result, 'download_s': transfer['download_s'],
                      'observed_link_mbps': transfer['observed_link_mbps'],
                      'trace_before': cursor.__dict__, 'trace_after': next_cursor.__dict__,
                      'elapsed_before_s': elapsed, 'elapsed_after_s': end_elapsed}}
        # Compute next state before yielding; callers cannot mutate it via request.
        history = (history + [transfer['observed_link_mbps']])[-cfg['history_length']:]
        buffer = result['buffer_after_s']
        previous = lookup[selected]['actual_bitrate_kbps']
        cursor, elapsed = next_cursor, end_elapsed
        yield record


def safe_baseline_policy(name, actions, cfg, seed):
    """Return a callback whose decisions depend solely on online candidate fields."""
    rng = random.Random(seed)
    # Strip privileged metadata before creating the policy closure.
    public = {cid: [{k: a[k] for k in ('action_id', 'size_bytes')} for a in rows]
              for cid, rows in actions.items()}
    return lambda row: (choose_baseline(name, row['state'], public[row['state']['case_id']], cfg, rng), {})
