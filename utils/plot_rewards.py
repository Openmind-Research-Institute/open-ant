import re
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from agents.reward import RewardTracker

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams.update({'font.size': 13})

AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents')

RUNS = {
    'SAC': os.path.join(AGENTS_DIR, 'sac', 'runs_sim_test'),
    # 'PPO': os.path.join(AGENTS_DIR, 'ppo', 'runs'),
    'MPO': os.path.join(AGENTS_DIR, 'mpo', 'runs'),
}

PALETTE = plt.cm.tab10.colors

TIME_WINDOW = 120.0  # seconds

# Strip trailing _YYYYMMDD-HHMMSS_seed_N from a run directory name.
_DATE_SEED_RE = re.compile(r'_\d{8}-\d{6}_seed_\d+$')


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
out_path = os.path.join(AGENTS_DIR, 'training_curves.pdf')
plt.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.show()