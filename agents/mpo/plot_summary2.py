import os
import re
import sys
import json
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np

# =====================================================
# USER SETTINGS
# =====================================================

REPO_ROOT = "/home/seliu/open-ant"
#REPO_ROOT = "/home/serena-liu/open-ant"

RUN_DIRS = [
    "/home/seliu/open-ant/agents/mpo/runs/40k_start2000_morestats/trial_1_mpo_20260608-000506_seed_0",
    "/home/seliu/open-ant/agents/mpo/runs/40k_start2000_morestats/trial_1_mpo_continual_learning_20260608-013205_seed_0",
    # add or remove paths as needed
]

SMOOTH = 10

OUTPUT_DIR = "/home/seliu/open-ant/agents/mpo/runs/40k_start2000_morestats"

# =====================================================
# HARDCODED Y-AXIS LIMITS  (set any entry to None to let matplotlib auto-scale)
# Same idea as the SAC dashboard: fixed ranges per panel, NO spike-clipping /
# percentile / annotation logic.  Just tweak the numbers below to taste.
# =====================================================
YLIMS = {
    'reward':            None,        # avg reward/s [cm/s]
    'loss_p':            None,        # policy loss
    'loss_q':            None,        # critic loss
    'mean_q':            None,        # mean Q value
    'SPS':               None,        # steps per second
    'qdiag':             None,        # mean_q / TD target / per-critic overlay
    'critic_grad_norm':  (0, 4),
    'actor_grad_norm':   (0, 2),
    'actor_entropy':     None,
    'dual':              None,        # dual variables overlay
    # ── new panels ──────────────────────────────────────────────────────
    'actor_grad_decomp': (0, 2),      # P / M / S vs total actor grad norm overlay
    'bellman_residual':  None,
    'mean_bootstrap_q':  None,
    'delta':             None,        # mean_delta / pos / neg overlay
    'td_target_decomp':  None,        # reward_term / next_q_term / q_k_term overlay
    'is_coef':           None,        # mean_is_coef / min_is_coef overlay
    'utd_ratio':         None,        # utd_ratio + frac_is_clipped overlay
}
# =====================================================

REPO_ROOT = os.path.abspath(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from agents.reward import RewardTracker

TIME_WINDOW = 120.0

DUAL_LINESTYLES = {
    'eta':        '-',
    'kl_mu':      '--',
    'kl_sigma':   '-.',
    'alpha_mu':   ':',
    'alpha_sigma': (0, (3, 1, 1, 1, 1, 1)),
}

LOSS_COLS = ['loss_q', 'loss_p', 'mean_q']
DUAL_COLS = ['eta', 'kl_mu', 'kl_sigma', 'alpha_mu', 'alpha_sigma']

DIAGNOSTIC_COLS = ['critic_grad_norm', 'actor_grad_norm', 'actor_entropy', 'mean_td_target']

# ── NEW metric columns logged by mpo_morestatss.py ───────────────────────────
ACTOR_GRAD_DECOMP_COLS = ['actor_grad_norm_P', 'actor_grad_norm_M', 'actor_grad_norm_S']
DELTA_COLS             = ['mean_delta', 'mean_delta_pos', 'mean_delta_neg']
TD_DECOMP_COLS         = ['mean_reward_term', 'mean_next_q_term', 'mean_q_k_term']
IS_COLS                = ['mean_is_coef', 'min_is_coef', 'frac_is_clipped']
NEW_SCALAR_COLS        = ['mean_bootstrap_q', 'bellman_residual', 'utd_ratio']

NEW_COLS = (ACTOR_GRAD_DECOMP_COLS + DELTA_COLS + TD_DECOMP_COLS
            + IS_COLS + NEW_SCALAR_COLS)

# per-critic Q columns are dynamically detected (mean_q_0, mean_q_1, ...)
PER_CRITIC_Q_PATTERN = re.compile(r'^mean_q_\d+$')


def load_run(run_dir, smooth):
    args_path = os.path.join(run_dir, 'weights_and_args', 'args.json')
    with open(args_path) as f:
        run_args = json.load(f)
    dt = run_args['dt']
    learning_starts = run_args.get('learning_starts', 0)

    perf = pd.read_csv(os.path.join(run_dir, 'performance_variables.csv'))
    perf = perf[pd.to_numeric(perf['step'], errors='coerce').notna()].copy()
    perf['step'] = perf['step'].astype(float)

    info_path = os.path.join(run_dir, 'info_logs.csv')
    info = None
    if os.path.exists(info_path):
        info = pd.read_csv(info_path)
        tracker = RewardTracker(env_dt=dt, env_id='plot', time_window=TIME_WINDOW, log_folder='.')
        avg_rps = []
        for r in info['original_reward']:
            tracker.update(r)
            avg_rps.append(tracker.average_reward_per_second)
        info['avg_reward_per_second'] = avg_rps
        warmup_steps = int(TIME_WINDOW / dt)
        # guard: don't slice away more rows than we have (otherwise reward
        # curve silently becomes empty and nothing plots)
        if len(info) > warmup_steps + smooth:
            info = info.iloc[warmup_steps:].copy()
        info['_reward_smooth'] = (
            info['avg_reward_per_second'].rolling(smooth, min_periods=1).mean() * 100
        )
        if len(info) > smooth:
            info = info.iloc[smooth:].copy()
        info['sim_time_min'] = info['step'] * dt / 60

    per_critic_q_cols = [c for c in perf.columns if PER_CRITIC_Q_PATTERN.match(c)]
    smooth_cols = LOSS_COLS + DUAL_COLS + DIAGNOSTIC_COLS + NEW_COLS + per_critic_q_cols
    for col in [c for c in smooth_cols if c in perf.columns]:
        perf[col] = pd.to_numeric(perf[col], errors='coerce')
        perf[col] = perf[col].rolling(smooth, min_periods=1).mean()
    perf = perf.iloc[smooth:].copy()
    perf['sim_time_min'] = perf['step'] * dt / 60

    return perf, info, dt, learning_starts


def stitch_runs(loaded_runs):
    """Shift each subsequent run's time axis so it starts right after the
    previous run ends, giving one continuous sim1->sim2 timeline regardless of
    whether sim2's step counter restarts at 0 or continues from ~40k.

    loaded_runs : list of (perf, info, dt, learning_starts, run_dir)
    Returns      : list of (perf, info, dt, learning_starts, run_dir, t_offset_min)
    """
    result = []
    t_end = 0.0
    for perf, info, dt, learning_starts, run_dir in loaded_runs:
        t_start = t_end

        step_min = perf['step'].min()
        step_max = perf['step'].max()
        t_span = (step_max - step_min) * dt / 60.0
        denom = max(step_max - step_min, 1e-9)

        perf = perf.copy()
        perf['sim_time_min'] = t_start + (perf['step'] - step_min) / denom * t_span

        # shift the reward (info) axis by the same offset
        if info is not None:
            info = info.copy()
            i_min = info['step'].min()
            i_max = info['step'].max()
            i_denom = max(i_max - i_min, 1e-9)
            info['sim_time_min'] = t_start + (info['step'] - i_min) / i_denom * t_span

        t_end = t_start + t_span
        result.append((perf, info, dt, learning_starts, run_dir, t_start))
    return result


def apply_ylim(ax, key):
    lim = YLIMS.get(key)
    if lim is not None:
        ax.set_ylim(lim)


def add_vline(ax, x, color, legend=False):
    ax.axvline(x, color=color, linestyle='--', linewidth=1.0,
               alpha=0.6, label='learning starts' if legend else '_')


def add_phase_divider(ax, x, label=None):
    """Vertical line marking the sim1->sim2 boundary."""
    ax.axvline(x, color='gray', linestyle=':', linewidth=1.2, alpha=0.8,
               label=label if label else '_')


def main():
    run_dirs = [os.path.abspath(r) for r in RUN_DIRS]
    out_dir  = os.path.abspath(OUTPUT_DIR) if OUTPUT_DIR else os.path.dirname(run_dirs[0])
    os.makedirs(out_dir, exist_ok=True)

    palette = cm.tab10(np.linspace(0, 0.9, max(len(run_dirs), 1)))

    # ── load + stitch all runs onto one continuous timeline ──────────────────
    loaded = []
    for run_dir in run_dirs:
        try:
            perf, info, dt, learning_starts = load_run(run_dir, SMOOTH)
            loaded.append((perf, info, dt, learning_starts, run_dir))
            print(f"Loaded {os.path.basename(run_dir)}: "
                  f"steps {int(perf['step'].min())}-{int(perf['step'].max())}")
        except Exception as e:
            print(f'Skipping {run_dir}: {e}')

    if not loaded:
        print('No runs loaded. Exiting.')
        return

    stitched = stitch_runs(loaded)
    sim2_boundary = stitched[1][5] if len(stitched) > 1 else None

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 36))
    gs  = fig.add_gridspec(10, 2, hspace=0.55, wspace=0.30)
    ax_reward    = fig.add_subplot(gs[0, :])
    ax_policy    = fig.add_subplot(gs[1, 0])
    ax_critic    = fig.add_subplot(gs[1, 1])
    ax_meanq     = fig.add_subplot(gs[2, 0])
    ax_sps       = fig.add_subplot(gs[2, 1])
    ax_qdiag     = fig.add_subplot(gs[3, :])     # mean_q + per-critic + TD target overlay
    ax_gnorm_c   = fig.add_subplot(gs[4, 0])
    ax_gnorm_a   = fig.add_subplot(gs[4, 1])
    ax_entropy   = fig.add_subplot(gs[5, 0])
    ax_dual      = fig.add_subplot(gs[5, 1])
    # ── NEW panels ──
    ax_grad_dec  = fig.add_subplot(gs[6, :])     # actor grad decomp P/M/S vs total
    ax_bellman   = fig.add_subplot(gs[7, 0])     # bellman residual
    ax_bootq     = fig.add_subplot(gs[7, 1])     # mean bootstrap Q
    ax_delta     = fig.add_subplot(gs[8, 0])     # mean_delta / pos / neg
    ax_td_dec    = fig.add_subplot(gs[8, 1])     # TD-target decomposition
    ax_is        = fig.add_subplot(gs[9, 0])     # importance-sampling coefs
    ax_utd       = fig.add_subplot(gs[9, 1])     # UTD ratio + frac IS clipped

    for i, (perf, info, dt, learning_starts, run_dir, t_off) in enumerate(stitched):
        color = palette[i]
        alpha = 0.85 if i == 0 else 0.55
        lw    = 1.5  if i == 0 else 1.0
        label = os.path.basename(run_dir)

        t = perf['sim_time_min']
        learning_starts_min = t_off + learning_starts * dt / 60

        # ── reward ───────────────────────────────────────────────────────────
        if info is not None and '_reward_smooth' in info.columns:
            ax_reward.plot(info['sim_time_min'], info['_reward_smooth'],
                           color=color, alpha=alpha, lw=lw, label=label)
            add_vline(ax_reward, learning_starts_min, color, legend=(i == 0))

        for ax, col in [(ax_policy, 'loss_p'), (ax_critic, 'loss_q')]:
            if col in perf.columns:
                data = perf[['sim_time_min', col]].dropna()
                ax.plot(data['sim_time_min'], data[col], color=color, alpha=alpha, lw=lw, label=label)
            add_vline(ax, learning_starts_min, color)

        if 'mean_q' in perf.columns:
            data = perf[['sim_time_min', 'mean_q']].dropna()
            ax_meanq.plot(data['sim_time_min'], data['mean_q'], color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_meanq, learning_starts_min, color)

        if 'SPS' in perf.columns:
            data = perf[['sim_time_min', 'SPS']].dropna()
            ax_sps.plot(data['sim_time_min'], data['SPS'], color=color, alpha=alpha, lw=lw, label=label)

        # ── Q-value diagnostics overlay: mean_q + per-critic + TD target ──────
        if 'mean_q' in perf.columns:
            data = perf[['sim_time_min', 'mean_q']].dropna()
            ax_qdiag.plot(data['sim_time_min'], data['mean_q'],
                          color=color, alpha=alpha, lw=lw,
                          label=f'mean_q ({label})' if len(run_dirs) > 1 else 'mean_q')
        if 'mean_td_target' in perf.columns:
            data = perf[['sim_time_min', 'mean_td_target']].dropna()
            ax_qdiag.plot(data['sim_time_min'], data['mean_td_target'],
                          color=color, alpha=alpha, lw=lw, linestyle='--',
                          label=f'TD target ({label})' if len(run_dirs) > 1 else 'TD target')
        per_critic_cols = sorted([c for c in perf.columns if PER_CRITIC_Q_PATTERN.match(c)],
                                 key=lambda x: int(x.split('_')[-1]))
        per_critic_linestyles = [':', '-.', (0, (3, 1, 1, 1))]
        for k, col in enumerate(per_critic_cols):
            data = perf[['sim_time_min', col]].dropna()
            ls = per_critic_linestyles[k % len(per_critic_linestyles)]
            ax_qdiag.plot(data['sim_time_min'], data[col],
                          color=color, alpha=alpha * 0.6, lw=lw * 0.8, linestyle=ls,
                          label=f'{col} ({label})' if len(run_dirs) > 1 else col)
        add_vline(ax_qdiag, learning_starts_min, color)

        # ── grad norms ────────────────────────────────────────────────────────
        if 'critic_grad_norm' in perf.columns:
            data = perf[['sim_time_min', 'critic_grad_norm']].dropna()
            ax_gnorm_c.plot(data['sim_time_min'], data['critic_grad_norm'],
                            color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_gnorm_c, learning_starts_min, color)

        if 'actor_grad_norm' in perf.columns:
            data = perf[['sim_time_min', 'actor_grad_norm']].dropna()
            ax_gnorm_a.plot(data['sim_time_min'], data['actor_grad_norm'],
                            color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_gnorm_a, learning_starts_min, color)

        # ── actor entropy ──────────────────────────────────────────────────────
        if 'actor_entropy' in perf.columns:
            data = perf[['sim_time_min', 'actor_entropy']].dropna()
            ax_entropy.plot(data['sim_time_min'], data['actor_entropy'],
                            color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_entropy, learning_starts_min, color)

        # ── dual variables ──────────────────────────────────────────────────────
        for col, ls in DUAL_LINESTYLES.items():
            if col not in perf.columns:
                continue
            lbl = f'{col} ({label})' if len(run_dirs) > 1 else col
            ax_dual.plot(t, perf[col], color=color, alpha=alpha, lw=lw, linestyle=ls, label=lbl)

        # ── NEW: actor grad decomposition  P / M / S  vs total ──────────────────
        decomp_pairs = [
            ('actor_grad_norm_P', '-',  'P'),
            ('actor_grad_norm_M', '--', 'M'),
            ('actor_grad_norm_S', ':',  'S'),
            ('actor_grad_norm',   '-.', 'total'),
        ]
        decomp_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        for (col, ls, comp_lbl), c in zip(decomp_pairs, decomp_colors):
            if col in perf.columns:
                data = perf[['sim_time_min', col]].dropna()
                full_lbl = comp_lbl if i == 0 else f'{comp_lbl} (run {i+1})'
                ax_grad_dec.plot(data['sim_time_min'], data[col],
                                 color=c, alpha=alpha, lw=lw, linestyle=ls, label=full_lbl)
        add_vline(ax_grad_dec, learning_starts_min, color)

        # ── NEW: bellman residual ───────────────────────────────────────────────
        if 'bellman_residual' in perf.columns:
            data = perf[['sim_time_min', 'bellman_residual']].dropna()
            ax_bellman.plot(data['sim_time_min'], data['bellman_residual'],
                            color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_bellman, learning_starts_min, color)

        # ── NEW: mean bootstrap Q ───────────────────────────────────────────────
        if 'mean_bootstrap_q' in perf.columns:
            data = perf[['sim_time_min', 'mean_bootstrap_q']].dropna()
            ax_bootq.plot(data['sim_time_min'], data['mean_bootstrap_q'],
                          color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_bootq, learning_starts_min, color)

        # ── NEW: TD-error delta decomposition (mean / pos / neg) ────────────────
        delta_styles = [('mean_delta', '-', 'delta'), ('mean_delta_pos', '--', 'delta+'),
                        ('mean_delta_neg', ':', 'delta-')]
        for col, ls, comp_lbl in delta_styles:
            if col in perf.columns:
                data = perf[['sim_time_min', col]].dropna()
                lbl = (comp_lbl if len(run_dirs) == 1 else f'{comp_lbl} ({label})')
                ax_delta.plot(data['sim_time_min'], data[col],
                              color=color, alpha=alpha, lw=lw, linestyle=ls, label=lbl)
        add_vline(ax_delta, learning_starts_min, color)

        # ── NEW: TD-target decomposition (reward / next_q / q_k terms) ──────────
        td_styles = [('mean_reward_term', '-', 'reward term'),
                     ('mean_next_q_term', '--', 'next-Q term'),
                     ('mean_q_k_term',    ':', 'q_k term')]
        for col, ls, comp_lbl in td_styles:
            if col in perf.columns:
                data = perf[['sim_time_min', col]].dropna()
                lbl = (comp_lbl if len(run_dirs) == 1 else f'{comp_lbl} ({label})')
                ax_td_dec.plot(data['sim_time_min'], data[col],
                               color=color, alpha=alpha, lw=lw, linestyle=ls, label=lbl)
        add_vline(ax_td_dec, learning_starts_min, color)

        # ── NEW: importance-sampling coefficients (mean / min) ──────────────────
        is_styles = [('mean_is_coef', '-', 'mean IS'), ('min_is_coef', '--', 'min IS')]
        for col, ls, comp_lbl in is_styles:
            if col in perf.columns:
                data = perf[['sim_time_min', col]].dropna()
                lbl = (comp_lbl if len(run_dirs) == 1 else f'{comp_lbl} ({label})')
                ax_is.plot(data['sim_time_min'], data[col],
                           color=color, alpha=alpha, lw=lw, linestyle=ls, label=lbl)
        add_vline(ax_is, learning_starts_min, color)

        # ── NEW: UTD ratio + fraction of IS coefs clipped ───────────────────────
        if 'utd_ratio' in perf.columns:
            data = perf[['sim_time_min', 'utd_ratio']].dropna()
            ax_utd.plot(data['sim_time_min'], data['utd_ratio'],
                        color=color, alpha=alpha, lw=lw,
                        label=f'utd_ratio ({label})' if len(run_dirs) > 1 else 'utd_ratio')
        if 'frac_is_clipped' in perf.columns:
            data = perf[['sim_time_min', 'frac_is_clipped']].dropna()
            ax_utd.plot(data['sim_time_min'], data['frac_is_clipped'],
                        color=color, alpha=alpha, lw=lw, linestyle='--',
                        label=f'frac_is_clipped ({label})' if len(run_dirs) > 1 else 'frac_is_clipped')
        add_vline(ax_utd, learning_starts_min, color)

    # ── sim1 -> sim2 boundary on every panel ──────────────────────────────────
    all_axes = [ax_reward, ax_policy, ax_critic, ax_meanq, ax_sps, ax_qdiag,
                ax_gnorm_c, ax_gnorm_a, ax_entropy, ax_dual,
                ax_grad_dec, ax_bellman, ax_bootq, ax_delta, ax_td_dec, ax_is, ax_utd]
    if sim2_boundary is not None:
        for ax in all_axes:
            add_phase_divider(ax, sim2_boundary,
                              label='sim2 start' if ax is ax_reward else None)

    primary_label = os.path.basename(run_dirs[0])
    suffix = f' (+{len(run_dirs)-1} more)' if len(run_dirs) > 1 else ''

    # ── reward panel ──────────────────────────────────────────────────────────
    ax_reward.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax_reward.set(xlabel='Sim time [minutes]', ylabel='Avg reward/s [cm/s]', xlim=(0, None),
                  title=f'Training reward — {primary_label}{suffix}')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(fontsize='x-small', ncols=2)
    apply_ylim(ax_reward, 'reward')

    # ── simple panels (title, ylim key) ───────────────────────────────────────
    simple_panels = [
        (ax_policy,  'Policy loss',           'loss_p'),
        (ax_critic,  'Critic loss',           'loss_q'),
        (ax_meanq,   'Mean Q value',          'mean_q'),
        (ax_sps,     'Steps per second (SPS)', 'SPS'),
        (ax_dual,    'Dual variables',        'dual'),
        (ax_qdiag,   'Q-value diagnostics (mean_q / TD target / per-critic)', 'qdiag'),
        (ax_gnorm_c, 'Critic gradient norm',  'critic_grad_norm'),
        (ax_gnorm_a, 'Actor gradient norm',   'actor_grad_norm'),
        (ax_entropy, 'Actor entropy  H = 0.5*Sum log(2*pi*e*sigma^2)', 'actor_entropy'),
        # new panels
        (ax_grad_dec, 'Actor grad decomposition: P / M / S vs total', 'actor_grad_decomp'),
        (ax_bellman,  'Bellman residual', 'bellman_residual'),
        (ax_bootq,    'Mean bootstrap Q', 'mean_bootstrap_q'),
        (ax_delta,    'TD error delta  (mean / positive / negative)', 'delta'),
        (ax_td_dec,   'TD-target decomposition (reward / next-Q / q_k)', 'td_target_decomp'),
        (ax_is,       'Importance-sampling coefficient (mean / min)', 'is_coef'),
        (ax_utd,      'UTD ratio  &  fraction IS clipped', 'utd_ratio'),
    ]
    for ax, title, ylim_key in simple_panels:
        ax.set(xlabel='Sim time [minutes]', title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small', ncols=2)
        apply_ylim(ax, ylim_key)

    out_path = os.path.join(out_dir, f'{primary_label}_dashboard.png')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
