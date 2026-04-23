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
    'MPO': os.path.join(AGENTS_DIR, 'mpo', 'runs_sim_test'),
}

COLORS = {
    'SAC': '#4477AA',
    # 'PPO': '#EE6677',
    'MPO': "#66EEA3",
}

TIME_WINDOW = 120.0  # seconds


def get_latest_run(runs_path):
    """Return the path to the most recent run directory (by name, sorted)."""
    dirs = sorted([
        d for d in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, d))
        and os.path.exists(os.path.join(runs_path, d, 'info_logs.csv'))
    ])
    if not dirs:
        return None
    return os.path.join(runs_path, dirs[-1])


def load_avg_rewards(run_path):
    """Load info_logs.csv and return smoothed reward series + dt + run name."""
    config_path = os.path.join(run_path, 'weights_and_args', 'args.json')
    with open(config_path) as f:
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

    return df, dt, os.path.basename(run_path)


fig, ax = plt.subplots(figsize=(12, 6))

for agent, runs_path in RUNS.items():
    run_path = get_latest_run(runs_path)
    if run_path is None:
        print(f"No runs found for {agent} in {runs_path}")
        continue

    df, dt, run_name = load_avg_rewards(run_path)
    ax.plot(
        df['step'] * dt / 60,
        df['avg_reward_per_second'] * 100,
        linewidth=1.5,
        label=f'{agent}',
        color=COLORS[agent],
    )
    print(f"{agent}: {run_name}  ({len(df)} steps after warm-up)")

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
ax.set_xlabel('Time [seconds]')
ax.set_ylabel('Average Reward per Second [cm/s]')
ax.set_title(f'SAC vs PPO vs MPO — Average Reward (window={TIME_WINDOW}s)')
ax.legend(loc='upper left', framealpha=0.7)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_path = os.path.join(AGENTS_DIR, 'training_curves.pdf')
plt.savefig(out_path, dpi=150)
plt.show()