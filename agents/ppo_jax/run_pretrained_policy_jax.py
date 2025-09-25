import os
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from brax.training.agents.ppo import checkpoint as ppo_checkpoint

import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import json
from etils import epath
import tqdm
import pandas as pd
import seaborn as sns
import imageio


# Local imports.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from reward import RewardTracker
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sim'))
from ant_mujoco import AntEnv
sys.path.append(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env'))
from embodied_ant_env import make_ant_env

# Create the environment
hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is not None:
    env_id = 'ant_hw'
    with open(sys.argv[1], 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                       render_mode='human',
                       dt=0.05,
                       joint_config=None)
else:
    env_id = "ant_sim"
    current_path = os.path.dirname(os.path.abspath(__file__))
    env = AntEnv(xml_file=os.path.join(current_path, "../../sim/assets/ant_position_with_camera.xml"),
                 render_mode="human",
                 dt=0.05)
    default_joint_config = env.model.keyframe("home").qpos[7:]
    print('Default joint config:', default_joint_config)

RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
numeric_folders = [f for f in folders if f[0].isdigit()]
latest_folder = numeric_folders[-1]
folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
latest_weights_folder = folders[-1]
policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)

jit_policy = jax.jit(policy_fn)

# Initialize random key for JAX.
rng = jax.random.PRNGKey(0)

# Run the policy in the environment.
obs, _ = env.reset()
DURATION = 4.0 # seconds
DT = env.dt
num_steps = int(DURATION / DT)
action_list = []
reward_list = []
total_reward = 0

run_ids = [1, 2, 3, 4, 5]
for run_id in run_ids:
    print(f"Running JAX policy for {DURATION} seconds ({num_steps} steps) run {run_id}")
    true_pos_xy_df_list = []
    vis_frame_list = []

    reward_tracker = RewardTracker(env_dt=env.dt,
                                    env_id=f"run_{run_id}_{env_id}",
                                    time_window=10.0,
                                    log_folder=epath.Path(RESULTS_FOLDER_PATH) / latest_folder)

    for i in tqdm.tqdm(range(num_steps)):
        # Get action from JAX policy.
        obs_dict = {
                'privileged_state': jp.zeros(obs.shape[0]),
                'state': jp.array(obs)
            }

        act_rng, rng = jax.random.split(rng)
        action_ppo, _ = jit_policy(obs_dict, act_rng)
        action = np.array(action_ppo)

        # Take step in environment.
        obs, reward, terminated, truncated, info = env.step(action)

        # Update the reward tracker.
        average_reward_per_second = reward_tracker.update(reward)
        reward_tracker.log(i, average_reward_per_second)

        # Save the data.
        true_pos_xy_df_list.append([info['current_x_position'],
                                    info['current_y_position']])
        action_list.append(action)
        reward_list.append(reward)
        if env_id == 'ant_hw':
            vis_frame_list.append(info['vis_frame'])

        if terminated or truncated:
            print(f"Average reward per second: {average_reward_per_second}")
            obs, _ = env.reset(seed=0)

    # Make a gif of the vis_frame_list.
    if env_id == 'ant_hw':
        imageio.mimsave(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"trajectory_run_{run_id}_env_{env_id}.gif"), vis_frame_list, duration=0.1)

    # Make df and save the data.
    true_pos_xy_df = pd.DataFrame(true_pos_xy_df_list, columns=['x', 'y'])
    true_pos_xy_df.to_csv(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"trajectory_run_{run_id}_env_{env_id}.csv"), index=False)

    # Load the data.
    true_pos_xy_df = pd.read_csv(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"trajectory_run_{run_id}_env_{env_id}.csv"))
    true_pos_xy_df['x'] = true_pos_xy_df['x'] - true_pos_xy_df['x'].iloc[0]
    true_pos_xy_df['y'] = true_pos_xy_df['y'] - true_pos_xy_df['y'].iloc[0]

    # Plot the trajectory.
    color = sns.color_palette("tab10")[0]
    sns.lineplot(
        x=true_pos_xy_df['x'],
        y=true_pos_xy_df['y'],
        linewidth=2,
        alpha=0.8,
        color=color,
        marker='o'
    )

    plt.xlabel("X Position", fontsize=13)
    plt.ylabel("Y Position", fontsize=13)
    plt.scatter(true_pos_xy_df['x'].iloc[0], true_pos_xy_df['y'].iloc[0], color="crimson", s=80, edgecolor="black", zorder=5)
    plt.scatter(true_pos_xy_df['x'].iloc[-1], true_pos_xy_df['y'].iloc[-1], color="limegreen", s=80, edgecolor="black", zorder=5)
    print("Start:", true_pos_xy_df['x'].iloc[0], true_pos_xy_df['y'].iloc[0])
    print("End:", true_pos_xy_df['x'].iloc[-1], true_pos_xy_df['y'].iloc[-1])
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"trajectory_run_{run_id}_env_{env_id}.png"), dpi=300, bbox_inches="tight")
    input("Press Enter to continue to next run...")

plt.figure()
# Put all the trajectories on the same plot.
for run_id in run_ids:
    true_pos_xy_df = pd.read_csv(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"trajectory_run_{run_id}_env_{env_id}.csv"))
    true_pos_xy_df['x'] = true_pos_xy_df['x'] - true_pos_xy_df['x'].iloc[0]
    true_pos_xy_df['y'] = true_pos_xy_df['y'] - true_pos_xy_df['y'].iloc[0]

    # Plot the trajectory.
    color = sns.color_palette("tab10")[run_id]
    sns.lineplot(
        x=true_pos_xy_df['x'],
        y=true_pos_xy_df['y'],
        linewidth=2,
        alpha=0.8,
        color=color,
        marker='o',
        label=f"Run {run_id}"
    )
    plt.scatter(true_pos_xy_df['x'].iloc[0], true_pos_xy_df['y'].iloc[0], color="crimson", s=80, edgecolor="black", zorder=5)
    plt.scatter(true_pos_xy_df['x'].iloc[-1], true_pos_xy_df['y'].iloc[-1], color="limegreen", s=80, edgecolor="black", zorder=5)

plt.xlabel("X Position", fontsize=13)
plt.ylabel("Y Position", fontsize=13)
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"trajectory_all_runs.png"), dpi=300, bbox_inches="tight")

# Make a new plot for the rewards.
plt.figure()
# Put all the rewards on the same plot.
colors = sns.color_palette("tab10")
for run_id in run_ids:
    rewards_df = pd.read_csv(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"eval_run_{run_id}_{env_id}_rewards_log.csv"))
    sns.lineplot(
        x=rewards_df['step'] * env.dt,
        y=rewards_df['reward'],
        linewidth=2,
        alpha=0.8,
        color=colors[run_id],
        label=f"Run {run_id}"
    )
plt.xlabel("Time [s]")
plt.ylabel("Reward per Second")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER_PATH, latest_folder, f"rewards_all_runs.png"), dpi=300, bbox_inches="tight")