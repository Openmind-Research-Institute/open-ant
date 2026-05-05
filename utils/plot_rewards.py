import re
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from agents.reward import RewardTracker

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams.update({'font.size': 13})

AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents')

RUNS = {
    'SAC': os.path.join(AGENTS_DIR, 'sac', 'runs'),
    # 'PPO': os.path.join(AGENTS_DIR, 'ppo', 'runs'),
    'MPO': os.path.join(AGENTS_DIR, 'mpo', 'runs'),
}

SAC_HW_DIR = os.path.join(AGENTS_DIR, 'sac', 'runs_hw', 'runs_expensive_ant_back_and_forth')
MPO_HW_DIR = os.path.join(AGENTS_DIR, 'mpo', 'runs_hw')
SAC_HW_DT = 0.12

PALETTE = plt.cm.tab10.colors

TIME_WINDOW = 120.0  # seconds

# Strip trailing _YYYYMMDD-HHMMSS_seed_N from a run directory name.
_DATE_SEED_RE = re.compile(r'_\d{8}-\d{6}_seed_\d+$')


def get_dt(run_path):
    for f in [os.path.join(run_path, 'weights_and_args', 'args.json'),
              os.path.join(run_path, 'args.json')]:
        if os.path.exists(f):
            with open(f) as fh:
                return json.load(fh)['dt']
    return 0.15


def load_sac_hw_trials(dir_path):
    """Load all average_rewards_df_trial_N.csv from the SAC hw directory."""
    files = sorted(glob.glob(os.path.join(dir_path, 'average_rewards_df_trial_*.csv')))
    return [(pd.read_csv(f), os.path.splitext(os.path.basename(f))[0]) for f in files]


def load_mpo_hw_runs(hw_dir):
    """Load HwEmbodiedAnt_average_rewards.csv from each MPO hw run directory."""
    runs = []
    for d in sorted(os.listdir(hw_dir)):
        run_path = os.path.join(hw_dir, d)
        csv_path = os.path.join(run_path, 'HwEmbodiedAnt_average_rewards.csv')
        if os.path.isdir(run_path) and os.path.exists(csv_path):
            dt = get_dt(run_path)
            label = _DATE_SEED_RE.sub('', d)
            runs.append((pd.read_csv(csv_path), dt, label))
    return runs


def get_all_runs(runs_path):
    """Return all valid run directories (sorted oldest→newest by name)."""
    if not os.path.isdir(runs_path):
        return []
    return sorted([
        os.path.join(runs_path, d)
        for d in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, d))
        and os.path.exists(os.path.join(runs_path, d, 'info_logs.csv'))
        and os.path.exists(os.path.join(runs_path, d, 'weights_and_args', 'args.json'))
    ])


def load_avg_rewards(run_path):
    """Load info_logs.csv and return smoothed reward series + dt + short label."""
    with open(os.path.join(run_path, 'weights_and_args', 'args.json')) as f:
        config = json.load(f)
    dt = config['dt']

    df = pd.read_csv(os.path.join(run_path, 'info_logs.csv'))

    tracker = RewardTracker(env_dt=dt, env_id="plot", time_window=TIME_WINDOW, log_folder=".")
    avg_rewards = []
    for reward in df['original_reward']:
        tracker.update(reward)
        avg_rewards.append(tracker.average_reward_per_second)

    df['avg_reward_per_second'] = avg_rewards
    df = df.iloc[int(TIME_WINDOW / dt):]   # drop warm-up window

    label = _DATE_SEED_RE.sub('', os.path.basename(run_path))
    return df, dt, label


fig, ax = plt.subplots(figsize=(14, 6))

color_idx = 0
for agent, runs_path in RUNS.items():
    runs = get_all_runs(runs_path)
    if not runs:
        print(f"No runs found for {agent} in {runs_path}")
        continue

    for run_path in runs:
        color = PALETTE[color_idx % len(PALETTE)]
        color_idx += 1

        try:
            df, dt, label = load_avg_rewards(run_path)
        except Exception as e:
            print(f"  Skipping {run_path}: {e}")
            continue

        ax.plot(
            df['step'] * dt / 60,
            df['avg_reward_per_second'] * 100,
            linewidth=1.5,
            color=color,
            label=f'{agent} — {label}',
        )
        print(f"  {agent}: {label}  ({len(df)} steps after warm-up)")

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
ax.set_xlabel('Time [minutes]')
ax.set_ylabel('Average Reward per Second [cm/s]')
ax.set_title(f'Training curves — Average Reward (smoothing window={TIME_WINDOW}s)')
ax.legend(loc='lower right', framealpha=0.7, fontsize='x-small', ncols=2)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_path = os.path.join(AGENTS_DIR, 'training_curves_ensemble.pdf')
plt.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.show()

# ── Hardware runs ────────────────────────────────────────────────────────────
fig_hw, ax_hw = plt.subplots(figsize=(14, 6))
color_idx = 0

# SAC hw: mean ± std across trials, individual trials as faint lines
sac_trials = load_sac_hw_trials(SAC_HW_DIR)
if sac_trials:
    sac_color = PALETTE[color_idx]
    color_idx += 1
    min_len = min(len(df) for df, _ in sac_trials)
    steps_sac = sac_trials[0][0]['step'].values[:min_len] * SAC_HW_DT / 60
    rewards_mat = np.array([df['reward'].values[:min_len] * 100 for df, _ in sac_trials])
    for df, _ in sac_trials:
        ax_hw.plot(df['step'].values[:min_len] * SAC_HW_DT / 60,
                   df['reward'].values[:min_len] * 100,
                   color=sac_color, alpha=0.25, linewidth=0.8)
    mean = rewards_mat.mean(axis=0)
    std = rewards_mat.std(axis=0)
    ax_hw.plot(steps_sac, mean, color=sac_color, linewidth=2.0,
               label=f'SAC hw — mean ± std ({len(sac_trials)} trials)')
    ax_hw.fill_between(steps_sac, mean - std, mean + std, color=sac_color, alpha=0.2)
    print(f"SAC hw: {len(sac_trials)} trials, {min_len} steps each")
else:
    print(f"No SAC hw trials found in {SAC_HW_DIR}")

# MPO hw: one line per run
mpo_hw_runs = load_mpo_hw_runs(MPO_HW_DIR)
for df, dt, label in mpo_hw_runs:
    color = PALETTE[color_idx % len(PALETTE)]
    color_idx += 1
    ax_hw.plot(df['step'] * dt / 60, df['reward'] * 100,
               color=color, linewidth=1.5, label=f'MPO hw — {label}')
    print(f"MPO hw: {label}  ({len(df)} steps)")

ax_hw.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
ax_hw.set_xlabel('Time [minutes]')
ax_hw.set_ylabel('Average Reward per Second [cm/s]')
ax_hw.set_title(f'Hardware Training Curves — Average Reward (smoothing window={TIME_WINDOW}s)')
ax_hw.legend(loc='lower right', framealpha=0.7, fontsize='x-small')
ax_hw.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_path_hw = os.path.join(AGENTS_DIR, 'training_curves_hw.pdf')
fig_hw.savefig(out_path_hw, dpi=150)
print(f"\nSaved hw plot to {out_path_hw}")
plt.show()