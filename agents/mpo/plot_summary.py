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

#REPO_ROOT = "/home/seliu/open-ant"
REPO_ROOT = "/home/serena-liu/open-ant"

RUN_DIRS = [
    "/home/serena-liu/open-ant/agents/mpo/runs/vanilla_morestats/trial_1_mpo_20260606-135202_seed_0",
    "/home/serena-liu/open-ant/agents/mpo/runs/vanilla_morestats/trial_1_mpo_continual_learning_20260606-144723_seed_0",
    # add or remove paths as needed
]

SMOOTH = 10

OUTPUT_DIR = "/home/serena-liu/open-ant/agents/mpo/runs/vanilla_morestats"
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
# per-critic Q columns are dynamically detected (mean_q_0, mean_q_1, ...)

PER_CRITIC_Q_PATTERN = re.compile(r'^mean_q_\d+$')
# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--runs', type=str, nargs='+', required=True,
#                         help='One or more run directories (the timestamped folders with CSVs)')
#     parser.add_argument('--smooth', type=int, default=10)
#     parser.add_argument('--out', type=str, default=None,
#                         help='Output directory for PDFs (defaults to first run dir)')
#     return parser.parse_args()


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
        info = info.iloc[warmup_steps:].copy()
        info['_reward_smooth'] = (
            info['avg_reward_per_second'].rolling(smooth, min_periods=1).mean() * 100
        )
        info = info.iloc[smooth:].copy()
        info['sim_time_min'] = info['step'] * dt / 60

    per_critic_q_cols = [c for c in perf.columns if PER_CRITIC_Q_PATTERN.match(c)]
    smooth_cols = LOSS_COLS + DUAL_COLS + DIAGNOSTIC_COLS + per_critic_q_cols
    for col in [c for c in smooth_cols if c in perf.columns]:
        perf[col] = pd.to_numeric(perf[col], errors='coerce')
        perf[col] = perf[col].rolling(smooth, min_periods=1).mean()
    perf = perf.iloc[smooth:].copy()
    perf['sim_time_min'] = perf['step'] * dt / 60

    return perf, info, dt, learning_starts


def add_vline(ax, x, color, legend=False):
    ax.axvline(x, color=color, linestyle='--', linewidth=1.0,
               alpha=0.6, label='learning starts' if legend else '_')


def main():
  #  args = arg_parser()
    run_dirs = [os.path.abspath(r) for r in RUN_DIRS]
    out_dir  = os.path.abspath(OUTPUT_DIR) if OUTPUT_DIR else os.path.dirname(run_dirs[0])
    os.makedirs(out_dir, exist_ok=True)

    palette = cm.tab10(np.linspace(0, 0.9, max(len(run_dirs), 1)))

    fig = plt.figure(figsize=(12, 22))
    gs  = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.30)
    ax_reward  = fig.add_subplot(gs[0, :])
    ax_policy  = fig.add_subplot(gs[1, 0])
    ax_critic  = fig.add_subplot(gs[1, 1])
    ax_meanq   = fig.add_subplot(gs[2, 0])     # mean_q overlay (existing)
    ax_sps     = fig.add_subplot(gs[2, 1])
    ax_qdiag   = fig.add_subplot(gs[3, :])     # NEW: mean_q + per-critic + TD target overlay (full width)
    ax_gnorm_c = fig.add_subplot(gs[4, 0])     # NEW: critic grad norm
    ax_gnorm_a = fig.add_subplot(gs[4, 1])     # NEW: actor grad norm
    ax_entropy = fig.add_subplot(gs[5, 0])     # NEW: actor entropy
    ax_dual    = fig.add_subplot(gs[5, 1])     # dual variables (moved, now half-width)

    for i, run_dir in enumerate(run_dirs):
        color = palette[i]
        alpha = 0.85 if i == 0 else 0.55
        lw    = 1.5  if i == 0 else 1.0
        label = os.path.basename(run_dir)

        try:
            perf, info, dt, learning_starts = load_run(run_dir, SMOOTH)
        except Exception as e:
            print(f'Skipping {run_dir}: {e}')
            continue

        t = perf['sim_time_min']
        learning_starts_min = learning_starts * dt / 60

        if info is not None:
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

        # ── Q-value diagnostics overlay: mean_q + per-critic + TD target ──
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
        # per-critic Q traces (mean_q_0, mean_q_1, ...)
        per_critic_cols = sorted([c for c in perf.columns if PER_CRITIC_Q_PATTERN.match(c)],
                                 key=lambda x: int(x.split('_')[-1]))
        per_critic_linestyles = [':', '-.', (0, (3, 1, 1, 1))]   # cycles for ensemble > 3
        for k, col in enumerate(per_critic_cols):
            data = perf[['sim_time_min', col]].dropna()
            ls = per_critic_linestyles[k % len(per_critic_linestyles)]
            ax_qdiag.plot(data['sim_time_min'], data[col],
                          color=color, alpha=alpha * 0.6, lw=lw * 0.8, linestyle=ls,
                          label=f'{col} ({label})' if len(run_dirs) > 1 else col)
        add_vline(ax_qdiag, learning_starts_min, color)

        # ── grad norms ────────────────────────────────────────────────────
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

        # ── actor entropy ─────────────────────────────────────────────────
        if 'actor_entropy' in perf.columns:
            data = perf[['sim_time_min', 'actor_entropy']].dropna()
            ax_entropy.plot(data['sim_time_min'], data['actor_entropy'],
                            color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_entropy, learning_starts_min, color)

        for col, ls in DUAL_LINESTYLES.items():
            if col not in perf.columns:
                continue
            lbl = f'{col} ({label})' if len(run_dirs) > 1 else col
            ax_dual.plot(t, perf[col], color=color, alpha=alpha, lw=lw, linestyle=ls, label=lbl)

    primary_label = os.path.basename(run_dirs[0])
    suffix = f' (+{len(run_dirs)-1} more)' if len(run_dirs) > 1 else ''

    ax_reward.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax_reward.set(xlabel='Sim time [minutes]', ylabel='Avg reward/s [cm/s]', xlim=(0, None),
                  title=f'Training reward — {primary_label}{suffix}')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(fontsize='x-small', ncols=2)

    for ax, title in [(ax_policy, 'Policy loss'), (ax_critic, 'Critic loss'), (ax_meanq, 'Mean Q value')]:
        ax.set(xlabel='Sim time [minutes]', title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small')

    ax_sps.set(xlabel='Sim time [minutes]', title='Steps per second (SPS)', xlim=(0, None))
    ax_sps.grid(True, alpha=0.3)
    ax_sps.legend(fontsize='x-small')

    ax_dual.set(xlabel='Sim time [minutes]', title='Dual variables', xlim=(0, None))
    ax_dual.grid(True, alpha=0.3)
    ax_dual.legend(fontsize='x-small', ncols=3)

    for ax, title in [
        (ax_qdiag,   'Q-value diagnostics (mean_q / TD target / per-critic)'),
        (ax_gnorm_c, 'Critic gradient norm'),
        (ax_gnorm_a, 'Actor gradient norm'),
        (ax_entropy, 'Actor entropy  H = 0.5·Σ log(2πe σ²)'),
    ]:
        ax.set(xlabel='Sim time [minutes]', title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small', ncols=2)

    out_path = os.path.join(out_dir, f'{primary_label}_dashboard.png')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
