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
    info = info.iloc[smooth:].copy()
    info['sim_time_min'] = info['step'] * dt / 60

    return perf, info, dt, learning_starts


def add_vline(ax, x, color, legend=False):
    ax.axvline(x, color=color, linestyle='--', linewidth=1.0,
               alpha=0.6, label='learning starts' if legend else '_')


AGENTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents')
SIM_RUNS_DIR = os.path.join(AGENTS_DIR, 'mpo', 'runs')
HW_RUNS_DIR  = os.path.join(AGENTS_DIR, 'mpo', 'runs_hw')

_DATE_SEED_RE = re.compile(r'_\d{8}-\d{6}_seed_\d+$')

LOSS_COLS = ['loss_q', 'loss_p', 'mean_q']
DUAL_COLS = ['eta', 'kl_mu', 'kl_sigma', 'alpha_mu', 'alpha_sigma']


def _short_label(run_path):
    return _DATE_SEED_RE.sub('', os.path.basename(run_path))


def _get_weights_path(run_path):
    p = os.path.join(run_path, 'weights_and_args', 'args.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        wp = json.load(f).get('weights_path')
    return os.path.abspath(wp) if wp else None


def _scan_runs(runs_dir):
    if not os.path.isdir(runs_dir):
        return []
    return sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
        and os.path.exists(os.path.join(runs_dir, d, 'performance_variables.csv'))
        and os.path.exists(os.path.join(runs_dir, d, 'weights_and_args', 'args.json'))
    ])


def _build_chains(sim_runs, hw_runs):
    all_runs = sim_runs + hw_runs
    is_hw    = {r: False for r in sim_runs}
    is_hw.update({r: True for r in hw_runs})
    wa_index = {os.path.abspath(os.path.join(r, 'weights_and_args')): r for r in all_runs}

    children = {}
    for r in hw_runs:
        wp = _get_weights_path(r)
        if wp:
            parent = wa_index.get(wp)
            if parent:
                children.setdefault(parent, []).append(r)

    def all_paths(node):
        entry = {'path': node, 'is_hw': is_hw[node]}
        kids  = sorted(children.get(node, []))
        if not kids:
            return [[entry]]
        paths = []
        for kid in kids:
            for sub in all_paths(kid):
                paths.append([entry] + sub)
        return paths

    chains = []
    for sim_run in sim_runs:
        if sim_run in children:
            chains.extend(all_paths(sim_run))
    return chains


def _load_perf(run_path, smooth):
    with open(os.path.join(run_path, 'weights_and_args', 'args.json')) as f:
        dt = json.load(f).get('dt', 0.15)
    perf = pd.read_csv(os.path.join(run_path, 'performance_variables.csv'))
    perf = perf[pd.to_numeric(perf['step'], errors='coerce').notna()].copy()
    perf['step'] = perf['step'].astype(float)
    valid = [c for c in LOSS_COLS + DUAL_COLS if c in perf.columns]
    for col in valid:
        perf[col] = perf[col].rolling(smooth, min_periods=1).mean()
    perf = perf.iloc[smooth:].copy()
    return perf, dt


def plot_sim2real_losses(smooth=10):
    sim_runs = _scan_runs(SIM_RUNS_DIR)
    hw_runs  = _scan_runs(HW_RUNS_DIR)
    chains   = _build_chains(sim_runs, hw_runs)
    if not chains:
        return

    # Each chain gets a pair of adjacent tab10 colors: even index = sim, odd = hw.
    full_palette = cm.tab10(np.linspace(0, 0.9, 10))

    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)
    axes = {
        'loss_q': fig.add_subplot(gs[0, 0]),
        'loss_p': fig.add_subplot(gs[0, 1]),
        'mean_q': fig.add_subplot(gs[1, 0]),
        'eta':    fig.add_subplot(gs[1, 1]),
        'dual':   fig.add_subplot(gs[2, :]),
    }

    legend_handles = []

    for chain_idx, chain in enumerate(chains):
        sim_color = full_palette[(chain_idx * 2)     % 10]
        hw_color  = full_palette[(chain_idx * 2 + 1) % 10]
        label     = _short_label(chain[0]['path'])
        prev_last_step = None
        transfer_steps = []
        first_sim = True
        first_hw  = True

        for i, seg in enumerate(chain):
            try:
                perf, dt = _load_perf(seg['path'], smooth)
            except Exception as e:
                print(f"  Skipping {seg['path']}: {e}")
                continue

            steps = perf['step'].values
            if seg['is_hw'] and prev_last_step is not None and len(steps) > 0:
                steps = steps - steps[0] + prev_last_step

            t     = steps * dt / 60
            color = hw_color if seg['is_hw'] else sim_color

            if seg['is_hw'] and first_hw:
                seg_label = f'{label} hw'
                first_hw  = False
            elif not seg['is_hw'] and first_sim:
                seg_label = f'{label} sim'
                first_sim = False
            else:
                seg_label = None

            for col, ax_key in [('loss_q', 'loss_q'), ('loss_p', 'loss_p'),
                                 ('mean_q', 'mean_q'), ('eta', 'eta')]:
                if col in perf.columns:
                    axes[ax_key].plot(t, perf[col].values, color=color,
                                      linewidth=1.5, label=seg_label)
                    seg_label = None  # only attach label to first subplot

            for col, lstyle in zip(DUAL_COLS, ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]):
                if col not in perf.columns:
                    continue
                # Label each variable once (first chain, sim segment) to show linestyle key.
                dual_label = col if (chain_idx == 0 and not seg['is_hw']) else None
                axes['dual'].plot(t, perf[col].values, color=color,
                                  linestyle=lstyle, linewidth=1.5, label=dual_label)

            if len(steps) > 0:
                prev_last_step = steps[-1]
            if i < len(chain) - 1 and len(steps) > 0:
                transfer_steps.append(t[-1])

        for xv in transfer_steps:
            for ax in axes.values():
                ax.axvline(xv, color='black', linestyle=':', linewidth=1.0, alpha=0.5)

        legend_handles += [
            mlines.Line2D([], [], color=sim_color, linewidth=2, label=f'{label} sim'),
            mlines.Line2D([], [], color=hw_color,  linewidth=2, label=f'{label} hw'),
        ]

    titles = {
        'loss_q': 'Critic loss (loss_q)',
        'loss_p': 'Policy loss (loss_p)',
        'mean_q': 'Mean Q value',
        'eta':    'Dual η (KL constraint)',
        'dual':   'Dual variables (kl / α)',
    }
    for key, ax in axes.items():
        ax.set_title(titles[key])
        ax.set_xlabel('Sim time [min]')
        ax.grid(True, alpha=0.3, linestyle='--')

    axes['loss_q'].legend(handles=legend_handles, fontsize='x-small')
    axes['dual'].legend(fontsize='x-small', ncols=3)

    fig.suptitle('Sim-to-real — loss & dual variable evolution', fontsize=14, y=1.01)

    out_path = os.path.join(AGENTS_DIR, 'sim2real_losses.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'Saved sim2real loss plot to {out_path}')
    return fig


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

    plot_sim2real_losses(smooth=args.smooth)
    plt.show()


if __name__ == '__main__':
    main()
