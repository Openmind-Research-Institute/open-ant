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
REPO_ROOT = "/home/seliu/open-ant"

RUN_DIRS = [
    "/home/seliu/open-ant/agents/mpo/runs_continous_learning/retrace_20260604-063201_seed_0",
    "/home/seliu/open-ant/agents/mpo/runs_continous_learning/retrace_continual_learning_20260604-151042_seed_0",
    # add or remove paths as needed
]

SMOOTH = 10

OUTPUT_DIR = "/home/seliu/open-ant/agents/mpo/runs_continous_learning"
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

    for col in [c for c in LOSS_COLS + DUAL_COLS if c in perf.columns]:
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

    fig = plt.figure(figsize=(12, 14))
    gs  = fig.add_gridspec(4, 2, hspace=0.50, wspace=0.30)
    ax_reward = fig.add_subplot(gs[0, :])
    ax_policy = fig.add_subplot(gs[1, 0])
    ax_critic = fig.add_subplot(gs[1, 1])
    ax_meanq  = fig.add_subplot(gs[2, 0])
    ax_sps    = fig.add_subplot(gs[2, 1])
    ax_dual   = fig.add_subplot(gs[3, :])

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

    out_path = os.path.join(out_dir, f'{primary_label}_dashboard.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
