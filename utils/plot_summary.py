import os
import re
import sys
import json
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.reward import RewardTracker

TIME_WINDOW = 120.0  # seconds, matches plot_rewards.py

DUAL_LINESTYLES = {
    'eta':        '-',
    'kl_mu':      '--',
    'kl_sigma':   '-.',
    'alpha_mu':   ':',
    'alpha_sigma': (0, (3, 1, 1, 1, 1, 1)),
}


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--smooth',   type=int, default=10)
    parser.add_argument('--n_prev',   type=int, default=0,
                        help='Number of previous runs (by time) to overlay for comparison')
    return parser.parse_args()


_DT_PATTERN = re.compile(r'(\d{8}-\d{6})')

def run_datetime(name):
    m = _DT_PATTERN.search(name)
    return datetime.strptime(m.group(1), '%Y%m%d-%H%M%S') if m else None


def sorted_runs(logs_dir):
    """All runs with a performance_variables.csv, sorted newest-first."""
    runs = []
    for d in os.listdir(logs_dir):
        if not os.path.isdir(os.path.join(logs_dir, d)):
            continue
        if not os.path.isfile(os.path.join(logs_dir, d, 'performance_variables.csv')):
            continue
        dt = run_datetime(d)
        if dt:
            runs.append((dt, d))
    return [name for _, name in sorted(runs, reverse=True)]


def select_runs(logs_dir, run_name, n_prev):
    all_runs = sorted_runs(logs_dir)
    if not all_runs:
        return []
    primary = run_name or all_runs[0]
    try:
        idx = all_runs.index(primary)
    except ValueError:
        idx = 0
    return [primary] + all_runs[idx + 1: idx + 1 + n_prev]


def load_run(logs_dir, run_name, smooth):
    run_dir = os.path.join(logs_dir, run_name)

    args_path = os.path.join(run_dir, 'weights_and_args', 'args.json')
    with open(args_path) as f:
        run_args = json.load(f)
    dt = run_args['dt']
    learning_starts = run_args.get('learning_starts', 0)

    # performance_variables.csv — training metrics (one row per update step)
    perf = pd.read_csv(os.path.join(run_dir, 'performance_variables.csv'))

    # info_logs.csv — per-environment step; compute avg reward/s via RewardTracker
    info = pd.read_csv(os.path.join(run_dir, 'info_logs.csv'))
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
    info['sim_time_min'] = info['step'] * dt / 60

    return perf, info, dt, learning_starts


def add_vline(ax, x, color, legend=False):
    ax.axvline(x, color=color, linestyle='--', linewidth=1.0,
               alpha=0.6, label='learning starts' if legend else '_')


def main():
    args     = arg_parser()
    logs_dir = os.path.join(os.getcwd(), 'agents', 'mpo', 'runs')

    runs = select_runs(logs_dir, args.run_name, args.n_prev)
    if not runs:
        print('No runs found.')
        return

    palette = cm.tab10(np.linspace(0, 0.9, max(len(runs), 1)))

    fig = plt.figure(figsize=(12, 14))
    gs  = fig.add_gridspec(4, 2, hspace=0.50, wspace=0.30)
    ax_reward = fig.add_subplot(gs[0, :])
    ax_policy = fig.add_subplot(gs[1, 0])
    ax_critic = fig.add_subplot(gs[1, 1])
    ax_meanq  = fig.add_subplot(gs[2, 0])
    ax_sps    = fig.add_subplot(gs[2, 1])
    ax_dual   = fig.add_subplot(gs[3, :])

    dual_cols = list(DUAL_LINESTYLES.keys())

    for i, run_name in enumerate(runs):
        color = palette[i]
        alpha = 0.85 if i == 0 else 0.55
        lw    = 1.5  if i == 0 else 1.0
        label = run_name

        try:
            perf, info, dt, learning_starts = load_run(logs_dir, run_name, args.smooth)
        except Exception as e:
            print(f'Skipping {run_name}: {e}')
            continue

        perf['sim_time_min'] = perf['step'] * dt / 60
        t = perf['sim_time_min']
        learning_starts_min = learning_starts * dt / 60

        # reward — avg reward/s in cm/s
        ax_reward.plot(info['sim_time_min'], info['_reward_smooth'],
                       color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_reward, learning_starts_min, color, legend=(i == 0))

        # losses
        for ax, col in [(ax_policy, 'loss_p'), (ax_critic, 'loss_q')]:
            if col in perf.columns:
                data = perf[['sim_time_min', col]].dropna()
                ax.plot(data['sim_time_min'], data[col], color=color, alpha=alpha, lw=lw, label=label)
            add_vline(ax, learning_starts_min, color)

        # mean Q
        if 'mean_q' in perf.columns:
            data = perf[['sim_time_min', 'mean_q']].dropna()
            ax_meanq.plot(data['sim_time_min'], data['mean_q'], color=color, alpha=alpha, lw=lw, label=label)
        add_vline(ax_meanq, learning_starts_min, color)

        # SPS
        if 'SPS' in perf.columns:
            data = perf[['sim_time_min', 'SPS']].dropna()
            ax_sps.plot(data['sim_time_min'], data['SPS'], color=color, alpha=alpha, lw=lw, label=label)

        # dual variables — differentiated by linestyle per variable
        for col in dual_cols:
            if col not in perf.columns:
                continue
            ls  = DUAL_LINESTYLES[col]
            lbl = f'{col} ({run_name})' if len(runs) > 1 else col
            ax_dual.plot(t, perf[col], color=color, alpha=alpha, lw=lw,
                         linestyle=ls, label=lbl)

    primary_name = runs[0]
    suffix = f' (+{len(runs)-1} prev)' if len(runs) > 1 else ''

    ax_reward.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax_reward.set(xlabel='Sim time [minutes]', ylabel='Avg reward/s [cm/s]', xlim=(0, None),
                  title=f'Training reward — {primary_name}{suffix}')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(fontsize='x-small', ncols=2)

    for ax, title in [
        (ax_policy, 'Policy loss'),
        (ax_critic, 'Critic loss'),
        (ax_meanq,  'Mean Q value'),
    ]:
        ax.set(xlabel='Sim time [minutes]', title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)

    ax_sps.set(xlabel='Sim time [minutes]', title='Steps per second (SPS)', xlim=(0, None))
    ax_sps.grid(True, alpha=0.3)

    ax_dual.set(xlabel='Sim time [minutes]', title='Dual variables', xlim=(0, None))
    ax_dual.grid(True, alpha=0.3)
    ax_dual.legend(fontsize='x-small', ncols=3)

    out_path = os.path.join(logs_dir, primary_name, 'dashboard.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
