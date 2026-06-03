"""Plot sim-to-real transfer chains as continuous reward curves.

Each chain is a root sim run followed by one or more HW runs linked via
weights_path. Since global_step is preserved across checkpoints the step
column in info_logs.csv is already globally aligned, so no manual offset
arithmetic is needed.
"""

import re
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from agents.reward import RewardTracker

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams.update({'font.size': 13})

AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents')
SIM_RUNS_DIR = os.path.join(AGENTS_DIR, 'mpo', 'runs')
HW_RUNS_DIR  = os.path.join(AGENTS_DIR, 'mpo', 'runs_hw')

FULL_PALETTE = plt.cm.tab10(np.linspace(0, 0.9, 10))  # same as plot_summary
TIME_WINDOW  = 120.0  # seconds — rolling reward window

_DATE_SEED_RE = re.compile(r'_\d{8}-\d{6}_seed_\d+$')


# ── Helpers ──────────────────────────────────────────────────────────────────

def short_label(run_path):
    return _DATE_SEED_RE.sub('', os.path.basename(run_path))


def get_dt(run_path):
    for p in [os.path.join(run_path, 'weights_and_args', 'args.json'),
              os.path.join(run_path, 'args.json')]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f).get('dt', 0.15)
    return 0.15


def get_weights_path(run_path):
    p = os.path.join(run_path, 'weights_and_args', 'args.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        wp = json.load(f).get('weights_path')
    return os.path.abspath(wp) if wp else None


def scan_runs(runs_dir):
    if not os.path.isdir(runs_dir):
        return []
    return sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
        and os.path.exists(os.path.join(runs_dir, d, 'info_logs.csv'))
        and os.path.exists(os.path.join(runs_dir, d, 'weights_and_args', 'args.json'))
    ])


def load_rewards(run_path):
    """Return (steps, avg_reward_per_second) arrays from info_logs.csv."""
    dt = get_dt(run_path)
    df = pd.read_csv(os.path.join(run_path, 'info_logs.csv'))
    df = df[pd.to_numeric(df['step'], errors='coerce').notna()]
    df['step'] = df['step'].astype(float)

    tracker = RewardTracker(env_dt=dt, env_id='plot', time_window=TIME_WINDOW, log_folder='.')
    avg = []
    for r in df['original_reward']:
        tracker.update(r)
        avg.append(tracker.average_reward_per_second)

    warmup = int(TIME_WINDOW / dt)
    return df['step'].values[warmup:], np.array(avg)[warmup:]


# ── Chain builder ─────────────────────────────────────────────────────────────

def build_chains(sim_runs, hw_runs):
    """
    Follow weights_path links to build root→leaf chains.

    Returns a list of chains; each chain is a list of dicts:
        {'path': str, 'is_hw': bool}
    The first element is always a sim run (root), subsequent elements are HW.
    """
    all_runs = sim_runs + hw_runs
    is_hw    = {r: False for r in sim_runs}
    is_hw.update({r: True for r in hw_runs})

    # Map absolute weights_and_args path → run_path
    wa_index = {os.path.abspath(os.path.join(r, 'weights_and_args')): r for r in all_runs}

    # Build parent → children
    children = {}
    for r in hw_runs:
        wp = get_weights_path(r)
        if wp:
            parent = wa_index.get(wp)
            if parent:
                children.setdefault(parent, []).append(r)

    def all_paths(node):
        """All root→leaf paths starting at node."""
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


# ── Plot ──────────────────────────────────────────────────────────────────────

sim_runs = scan_runs(SIM_RUNS_DIR)
hw_runs  = scan_runs(HW_RUNS_DIR)
chains   = build_chains(sim_runs, hw_runs)

if not chains:
    print("No sim→hw chains found. Make sure HW runs have weights_path set in args.json.")
    sys.exit(0)

fig, ax = plt.subplots(figsize=(14, 6))
legend_handles = []

for chain_idx, chain in enumerate(chains):
    sim_color = FULL_PALETTE[(chain_idx * 2)     % 10]
    hw_color  = FULL_PALETTE[(chain_idx * 2 + 1) % 10]

    root_label = short_label(chain[0]['path'])
    print(f"\nChain: {' → '.join(short_label(s['path']) for s in chain)}")

    transfer_steps = []
    prev_last_step = None
    first_sim = True
    first_hw  = True

    for i, seg in enumerate(chain):
        try:
            steps, avg = load_rewards(seg['path'])
        except Exception as e:
            print(f"  Skipping {seg['path']}: {e}")
            continue

        if seg['is_hw'] and prev_last_step is not None and len(steps) > 0:
            steps = steps - steps[0] + prev_last_step

        color = hw_color if seg['is_hw'] else sim_color

        if seg['is_hw'] and first_hw:
            seg_label = f'{root_label} hw'
            first_hw  = False
        elif not seg['is_hw'] and first_sim:
            seg_label = f'{root_label} sim'
            first_sim = False
        else:
            seg_label = None

        ax.plot(steps, avg * 100, color=color, linewidth=1.5, label=seg_label)

        if len(steps) > 0:
            prev_last_step = steps[-1]

        if i < len(chain) - 1 and len(steps) > 0:
            transfer_steps.append(steps[-1])

        peak = f"{avg.max()*100:.2f} cm/s" if len(avg) else "n/a (too short for warm-up)"
        print(f"  {'hw' if seg['is_hw'] else 'sim'}: {short_label(seg['path'])}"
              f"  steps {int(steps[0]) if len(steps) else '?'}–{int(steps[-1]) if len(steps) else '?'}"
              f"  peak {peak}")

    for xv in transfer_steps:
        ax.axvline(x=xv, color='black', linestyle=':', linewidth=1.2, alpha=0.7)

    legend_handles += [
        mlines.Line2D([], [], color=sim_color, linewidth=2, label=f'{root_label} sim'),
        mlines.Line2D([], [], color=hw_color,  linewidth=2, label=f'{root_label} hw'),
    ]

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
ax.set_xlabel('Training step')
ax.set_ylabel('Avg reward per second [cm/s]')
ax.set_title(f'Sim-to-real transfer — rolling average (window = {TIME_WINDOW:.0f} s)')

ax.legend(handles=legend_handles, loc='lower right', framealpha=0.7, fontsize='x-small')

ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

out_path = os.path.join(AGENTS_DIR, 'sim2real_curves.pdf')
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.show()
