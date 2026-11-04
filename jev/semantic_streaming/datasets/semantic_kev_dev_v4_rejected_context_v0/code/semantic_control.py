"""v4 online candidate guard and three-chunk historical-bandwidth teacher.

Transmission physics remain frozen in semantic_rollout.py for v3 reproducibility.
The guard uses only public size/duration metadata and past completed transfers.
The offline teacher also uses measured PSNR utilities as supervision, never as
Kev input, and has no access to future network samples or real download outcomes.
"""
from functools import lru_cache
import math

from semantic_rollout import outcome, run_episode


PUBLIC_FIELDS = ('action_id', 'size_bytes', 'actual_bitrate_kbps', 'duration_s',
                 'mode', 'global_crf', 'roi_qoffset')


def validate_config(cfg):
    if not 1 <= cfg['planning_horizon_chunks'] <= 3:
        raise ValueError('Planning horizon must be 1..3 chunks')
    for key in ('forecast_bandwidth_factor', 'planning_discount'):
        if not 0 < cfg[key] <= 1:
            raise ValueError('Invalid probability/factor: ' + key)
    for key in ('terminal_buffer_weight', 'buffer_margin_s', 'buffer_spend_fraction', 'target_buffer_s'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('Invalid nonnegative setting: ' + key)
    if not 0 < cfg['minimum_forecast_mbps'] or not 0 < cfg['target_buffer_s'] <= cfg['max_buffer_s']:
        raise ValueError('Invalid forecast floor or buffer target')


def public_catalog(actions):
    return {cid: [{k: a[k] for k in PUBLIC_FIELDS} for a in rows] for cid, rows in actions.items()}


def conservative_bandwidth(state, cfg):
    history = state['recent_link_bandwidth_mbps']
    if not history or any(not math.isfinite(x) or x < 0 for x in history):
        raise ValueError('Invalid past transfer history')
    if any(x == 0 for x in history):
        return cfg['minimum_forecast_mbps']
    harmonic = len(history) / sum(1 / x for x in history)
    return max(cfg['minimum_forecast_mbps'], min(harmonic, history[-1]) * cfg['forecast_bandwidth_factor'])


def forecast_delay(action, bandwidth, cfg):
    return cfg['rtt_s'] + action['size_bytes'] * 8 / (bandwidth * 1e6 * cfg['payload_fraction'])


def candidate_guard(state, actions, cfg):
    if not actions:
        raise ValueError('No encoding candidates')
    cheapest = min(actions, key=lambda a: (a['size_bytes'], a['action_id']))
    capacity = conservative_bandwidth(state, cfg)
    predicted = {a['action_id']: forecast_delay(a, capacity, cfg) for a in actions}
    buffer_budget = max(0.0, state['buffer_seconds'] - cfg['buffer_margin_s'])
    sustainable_budget = (state['chunk_duration_seconds'] + cfg['buffer_spend_fraction']
                          * max(0.0, state['buffer_seconds'] - cfg['target_buffer_s']))
    budget = min(buffer_budget, sustainable_budget)
    within = [a for a in actions if predicted[a['action_id']] <= budget]
    if not state['playback_started']:
        allowed, reason = [cheapest], 'startup_cheapest'
    elif not within:
        allowed, reason = [cheapest], 'no_candidate_within_budget'
    else:
        allowed, reason = within, 'within_budget'
    return allowed, {'forecast_bandwidth_mbps': capacity, 'download_budget_s': budget,
                     'predicted_download_s': predicted,
                     'allowed_action_ids': [a['action_id'] for a in allowed],
                     'forced': len(allowed) == 1, 'reason': reason,
                     'no_candidate_within_budget': bool(state['playback_started'] and not within)}


def make_request(base, allowed, future, remaining_duration_s, cfg):
    state = dict(base['state'])
    state['remaining_playback_s'] = round(remaining_duration_s, 4)
    # Fixed grid order is declared in instructions; no future quality or answers.
    state['next_chunks'] = [[round(rows[0]['duration_s'], 4), [a['size_bytes'] for a in rows]] for rows in future]
    capacity = conservative_bandwidth(state, cfg)
    previous_order = list(base['questions']['encoding']['criteria'])
    rank = {k: i for i, k in enumerate(previous_order)}
    ordered = sorted(allowed, key=lambda a: rank[a['action_id']])
    criteria = {a['action_id']: (
        f"{a['global_crf']},{a['roi_qoffset']},{a['size_bytes']},"
        f"{a['actual_bitrate_kbps']:.1f},{forecast_delay(a, capacity, cfg):.3f}") for a in ordered}
    return {'state': state, 'questions': {'encoding': {'type': 'choice', 'src': 'semantic_video_encoding',
        'instructions': ('Choose for cumulative quality, stall and buffer preservation. '
            'Options: CRF,relative ROI QP offset,bytes,kbps,forecast download seconds. '
            'next_chunks=[duration,bytes array]; arrays follow ' + cfg['preview_grid_legend'] + '.'),
        'criteria': criteria}}}


def plan_labels(state, catalogs, cfg):
    """Exact finite-horizon search; measured quality is OFFLINE teacher information.

    Freeze a conservative estimate from past transfers for the whole horizon.
    Forecast buffers, switching and guards are updated after every planned action.
    Real future bandwidth and actual replay outcomes are not arguments.
    """
    if not catalogs or len(catalogs) > cfg['planning_horizon_chunks']:
        raise ValueError('Invalid planning horizon')
    capacity = conservative_bandwidth(state, cfg)
    horizon = len(catalogs)
    duration_after = max(0.0, state['remaining_playback_s'] - sum(rows[0]['duration_s'] for rows in catalogs))
    target = min(cfg['target_buffer_s'], duration_after)
    delays = [{a['action_id']: forecast_delay(a, capacity, cfg) for a in rows} for rows in catalogs]

    def next_state(depth, buffer, previous):
        return {**state, 'buffer_seconds': buffer, 'previous_bitrate_kbps': previous,
                'chunk_duration_seconds': catalogs[depth][0]['duration_s'],
                'playback_started': state['playback_started'] if depth == 0 else True}

    @lru_cache(maxsize=None)
    def solve(depth, buffer, previous):
        if depth == horizon:
            return -cfg['terminal_buffer_weight'] * max(0.0, target - buffer), (), buffer
        current = next_state(depth, buffer, previous)
        allowed, _ = candidate_guard(current, catalogs[depth], cfg)
        best = None
        for a in allowed:
            step = outcome(a, delays[depth][a['action_id']], current, cfg)
            tail, sequence, terminal = solve(depth + 1, step['buffer_after_s'], a['actual_bitrate_kbps'])
            value = step['reward'] + cfg['planning_discount'] * tail
            candidate = (value, -a['size_bytes'], a['action_id'], (a['action_id'],) + sequence, terminal)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        return best[0], best[3], best[4]

    current = next_state(0, state['buffer_seconds'], state['previous_bitrate_kbps'])
    allowed, guard = candidate_guard(current, catalogs[0], cfg)
    values, plans = {}, {}
    for a in allowed:
        step = outcome(a, delays[0][a['action_id']], current, cfg)
        tail, sequence, terminal = solve(1, step['buffer_after_s'], a['actual_bitrate_kbps'])
        values[a['action_id']] = step['reward'] + cfg['planning_discount'] * tail
        plans[a['action_id']] = {'action_sequence': [a['action_id'], *sequence],
            'forecast_buffer_after_first_s': step['buffer_after_s'], 'forecast_terminal_buffer_s': terminal}
    best = max(allowed, key=lambda a: (values[a['action_id']], -a['size_bytes'], a['action_id']))['action_id']
    return {'label': best, 'candidate_planned_return': values, 'plans': plans,
            'horizon_chunks': horizon, 'terminal_buffer_target_s': target,
            'forecast_bandwidth_mbps': capacity, 'guard': guard,
            'network_forecast_source': 'past_completed_transfers_only',
            'quality_source': 'offline_measured_PSNR_supervision_not_model_input'}


def run_control_episode(trace, spec, actions, questions, cfg, choose):
    """Adapt the frozen physical engine; guard and preview are built before choice.

    A singleton (including impossible-budget fallback) bypasses the model. Invalid
    proposed keys fall back to the smallest admissible file and are logged.
    """
    public = public_catalog(actions)
    context = {}
    step = 0

    def controlled(base):
        cid = spec['playlist'][step]
        allowed, guard = candidate_guard(base['state'], public[cid], cfg)
        future_ids = spec['playlist'][step + 1:step + cfg['planning_horizon_chunks']]
        remaining_s = sum(public[x][0]['duration_s'] for x in spec['playlist'][step:])
        row = make_request(base, allowed, [public[x] for x in future_ids], remaining_s, cfg)
        # Replace the request that the physical engine records, without changing
        # the engine's own physical state or full counterfactual action catalog.
        base.clear()
        base.update(row)
        fallback = min(allowed, key=lambda a: (a['size_bytes'], a['action_id']))['action_id']
        if guard['forced']:
            selected, info = fallback, {'model_bypassed': True}
        else:
            selected, info = choose(row)
        if selected not in guard['allowed_action_ids']:
            info = {**info, 'requested_action': selected, 'invalid_action_fallback': True}
            selected = fallback
        context['guard'] = guard
        return selected, info

    for record in run_episode(trace, spec, actions, questions, cfg, controlled):
        record['guard'] = context['guard']
        yield record
        step += 1


def guarded_behavior(name, actions, cfg, seed):
    import random
    rng = random.Random(seed)
    public = public_catalog(actions)

    def choose(row):
        allowed = set(row['questions']['encoding']['criteria'])
        rows = [a for a in public[row['state']['case_id']] if a['action_id'] in allowed]
        if name == 'cheapest':
            selected = min(rows, key=lambda a: (a['size_bytes'], a['action_id']))
        elif name in ('bandwidth', 'epsilon_bandwidth'):
            selected = (rng.choice(rows) if name == 'epsilon_bandwidth' and rng.random() < cfg['exploration_probability']
                        else max(rows, key=lambda a: (a['size_bytes'], a['action_id'])))
        else:
            raise ValueError('Unknown controlled behavior policy')
        return selected['action_id'], {}

    return choose


def model_with_fallback(choose, actions):
    public = public_catalog(actions)

    def guarded(row):
        try:
            return choose(row)
        except (ValueError, FloatingPointError) as error:
            allowed = row['questions']['encoding']['criteria']
            candidates = [a for a in public[row['state']['case_id']] if a['action_id'] in allowed]
            fallback = min(candidates, key=lambda a: (a['size_bytes'], a['action_id']))
            return fallback['action_id'], {'model_prediction_fallback': type(error).__name__}

    return guarded
