import os
import sys
import imageio
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pickle
from tqdm import tqdm
import time
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import zipfile
from PIL import Image
import io
import cv2
import swiftsarsa

# Custom imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from reward import RewardTracker
from tilecoding import IHT, tiles
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask

np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)


# Ramp function.
def linear_ramp(start_pos: float, end_pos: float, duration: float):
    num = round(duration / args.dt)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list


# plt.figure()
class OptionEnv:
    def __init__(self, env, options, discount=0.99):
        self.env = env
        self.options = options
        self.discount = discount
        self.joint_action = np.zeros(env.action_space.shape[0])

        # Logs.
        self.obs_list = []
        self.xy_pos_list = []
        self.reward_list = []

        # Reward tracker.
        self.reward_tracker = RewardTracker(env_dt=env.dt,
                                            env_id=f"run_{env_id}",
                                            time_window=120.0,
                                            log_folder=log_dir)
        self.average_rewards_per_second = []
        self.info = None

    def step(self, option_idx: int):
        opt = self.options[option_idx]

        # Populate the joint action trajectory.
        hip_joint_idx = opt['hip_joint_idx']
        knee_joint_idx = opt['knee_joint_idx']

        hip_traj = linear_ramp(self.joint_action[hip_joint_idx], opt['hip_target'], opt['duration'])
        num_steps = len(hip_traj)
        if opt['hip_target'] != self.joint_action[hip_joint_idx]:
            time = np.linspace(0, opt['duration'], num_steps)
            knee_traj = opt['knee_amplitude'] * np.sin(np.pi * time / opt['duration'])
        else:
            # NOTE: This is done to avoid the knee from flopping unnecessarily when the hip is not moving.
            knee_traj = np.full(num_steps, self.joint_action[knee_joint_idx])

        total_reward = 0.0
        gamma_i = 1.0

        for i in range(self.duration_steps(option_idx)):
            self.joint_action[hip_joint_idx] = hip_traj[i]
            self.joint_action[knee_joint_idx] = knee_traj[i]
            obs, reward, terminated, truncated, self.info = self.env.step(self.joint_action)

            # Record data.
            self.obs_list.append(obs)
            self.xy_pos_list.append([self.info["current_x_position"], self.info["current_y_position"]])
            self.reward_list.append(reward)

            # Average reward update.
            self.reward_tracker.update(reward)
            self.average_rewards_per_second.append(self.reward_tracker.average_reward_per_second)

            total_reward += gamma_i * reward
            gamma_i *= self.discount
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, self.info

        return obs, total_reward, terminated, truncated, self.info

    def reset(self, seed=None):
        self.joint_pos = np.zeros(self.env.action_space.shape[0])
        return self.env.reset(seed=SEED)

    def render(self):
        return self.env.render_with_arrow(self.info)
    
    def duration_steps(self, option_idx: int):
        return round(self.options[option_idx]['duration'] / args.dt)


options = []
for i in range(4):  # 4 legs
    options.append({
        "name": "sinusoid_forward",
        "hip_joint_idx": 2 * i,
        "hip_target": np.radians(45),
        "knee_joint_idx": 2 * i + 1,
        "knee_amplitude": np.radians(45),
        "duration": 0.6
    })
    options.append({
        "name": "sinusoid_backward",
        "hip_joint_idx": 2 * i,
        "hip_target": -np.radians(45),
        "knee_joint_idx": 2 * i + 1,
        "knee_amplitude": np.radians(45),
        "duration": 0.6
    })
    options.append({
        "name": "stance_forward",
        "hip_joint_idx": 2 * i,
        "hip_target": np.radians(45),
        "knee_joint_idx": 2 * i + 1,
        "knee_amplitude": np.radians(-20),  # This is so it pushes into the ground for better contact.
        "duration": 0.6
    })
    options.append({
        "name": "stance_backward",
        "hip_joint_idx": 2 * i,
        "hip_target": -np.radians(45),
        "knee_joint_idx": 2 * i + 1,
        "knee_amplitude": np.radians(-20),
        "duration": 0.6
    })

print(len(options), "options defined.")


# Tile coding.
class SuttonTileCoderWrapper:
    def __init__(self, iht: IHT, tiles_per_dim, value_limits, tilings):
        self.iht = iht
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=np.int32)
        self.tilings = int(tilings)
        self.limits = np.asarray(value_limits, dtype=np.float64)
        self.scaling = np.array(tiles_per_dim) / (self.limits[:, 1] - self.limits[:, 0])
        assert self.limits.shape == (self.tiles_per_dim.shape[0], 2)

    def __getitem__(self, x):
        x = np.asarray(x, dtype=np.float64)
        idxs = tiles(self.iht, self.tilings, self.scaling * x)
        return np.asarray(idxs, dtype=np.int64)

    @property
    def n_tiles(self):
        return self.iht.size


def q_of(w, idx, o):
    return w[o, idx].sum()


def select_greedy_option_swift_sarsa(q_vals):
    # Tie-break among maxima, in case of ties.
    maxq = q_vals.max()
    best = np.flatnonzero(q_vals == maxq)
    return int(np.random.choice(best)), q_vals

def select_option_epsilon_greedy_swift_sarsa(q_vals, epsilon):
    # ε-greedy over options using tile-coded T(s).
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_vals))
    O_greedy, _ = select_greedy_option_swift_sarsa(q_vals)
    return O_greedy

def clip_state_to_limits(S, limits):
    S = np.asarray(S, dtype=np.float64)
    return np.clip(S, limits[:, 0], limits[:, 1])


# Parser.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hw_config', type=str, default=None)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--load_previous_weights', type=bool, default=False)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--dt', type=float, default=0.05)

args = parser.parse_args()
SEED = args.seed
np.random.seed(SEED)

# Environment.
joint_config = {
    'hip_zero': 0.0,
    'knee_zero': -np.radians(60),
    'hip_range': np.radians(45),
    'knee_range': np.radians(45),
}

hw_config = args.hw_config if args.hw_config is not None else None
if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    render_mode = "human" if args.render else "rgb_array"
    print(f"Render mode: {render_mode}")
    env = AntEnv(
        dt=args.dt,
        joint_config=joint_config,
        render_mode=render_mode,
        task=ForwardTask(),
    )
else:
    env_id = 'ant_hw'
    render_mode = "human"
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                       render_mode=render_mode,
                       dt=args.dt,
                       joint_config=joint_config)

# Constants.
MAX_OPTIONS_PER_TIMELIMIT_EPISODE = 300
EPSILON = 0.05
EPSILON_START = EPSILON
DISCOUNTING = 0.99
LAMBDA = 0.90

DIM_TILING = 10  # Number of tiles per dimension.
TILINGS = 4 * env.observation_space.shape[0]  # Number of offset tilings.
IHT_SIZE = 2 ** 20

USE_DECAYING_EPSILON = False

# Directories.
log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)

weights_iht_folder = os.path.join(log_dir, "weights_iht")
if not os.path.exists(weights_iht_folder):
    os.makedirs(weights_iht_folder)

frames_folder = os.path.join(log_dir, "frames")
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

folder_trajectory = os.path.join(log_dir, "trajectory")
if not os.path.exists(folder_trajectory):
    os.makedirs(folder_trajectory)

# Environment.
options_env = OptionEnv(env, options)

# Limits from observation space.
state_limits = np.array([env.observation_space.low, env.observation_space.high]).T  # [state_dim, 2]
num_options = len(options)

# Load previous weights.
if args.load_previous_weights == False:
    iht = IHT(IHT_SIZE)
    w = np.zeros((num_options, iht.size), dtype=np.float32)
else:
    log_dir_to_load = 'logs/20250927_162143'
    w = np.load(os.path.join(log_dir_to_load, 'weights_iht/weights_9.npy'))
    # Load iht.
    with open(os.path.join(log_dir_to_load, "weights_iht/iht_9.pkl"), "rb") as f:
        iht = pickle.load(f)
    print('Loaded weights from previous run.')
    if args.train == False:
        EPSILON = 0.0

# IHT table size.
tiles_per_dim = [DIM_TILING] * state_limits.shape[0]
T = SuttonTileCoderWrapper(iht=iht,
                           tiles_per_dim=tiles_per_dim,
                           value_limits=state_limits,
                           tilings=TILINGS)
step_size = 0.1 / TILINGS  # Step-size, see: http://incompleteideas.net/tiles/tiles3.html.

with open(os.path.join(log_dir, "config.json"), "w") as f:
    json.dump({
        "tiles_per_dim": tiles_per_dim,
        "tilings": TILINGS,
        "state_limits": state_limits.tolist(),
        "iht_size": IHT_SIZE,
        "step_size": step_size,
        "discount": DISCOUNTING,
        "epsilon": EPSILON,
        "epsilon_start": EPSILON_START,
        "max_options_per_timelimit_episode": MAX_OPTIONS_PER_TIMELIMIT_EPISODE,
        "use_decaying_epsilon": USE_DECAYING_EPSILON,
        "log_dir": log_dir,
        "env_id": env_id,
        "dt": args.dt,
        "train": args.train,
        "load_previous_weights": args.load_previous_weights,
        "joint_config": joint_config,
    }, f, indent=2)


logging_data = {
    "timelimit_episode": [],
    "return_per_timelimit": [],
    "real_time_seconds": [],
    "reward_per_option": [],
}
logging_data_df = pd.DataFrame(logging_data)

nb_options = 0
return_per_timelimit = 0.0
idx_timelimit_episode = 0
real_time_seconds = 0.0

# Reset environment.
S, _ = options_env.reset(seed=SEED)
O = 0

learner = swiftsarsa.SwiftSarsaBinaryFeatures(1000000, 16, LAMBDA, 1e-2, 1e-3, 0.1, 0.99, 1e-4, 1e-10)
while True:
    if USE_DECAYING_EPSILON:
        EPSILON = max(0.05, EPSILON_START - idx_timelimit_episode * 0.015)
        print(f"Decaying epsilon to {EPSILON}")

    # Step.
    S_prime, R, terminated, truncated, info = options_env.step(O)
    indices = T[S_prime]
    q_vals = np.array(learner.get_action_values(indices))
    # Next option (ε-greedy).
    O = select_option_epsilon_greedy_swift_sarsa(q_vals, EPSILON)
    k = options_env.duration_steps(O)
    if terminated:
        learner.learn(indices, R, 0, O)
    else:
        learner.learn(indices, R, (DISCOUNTING ** k), O)

    return_per_timelimit += R
    real_time_seconds += options_env.duration_steps(O) * args.dt

    nb_options += 1

    if nb_options % 10 == 0:
        if render_mode == "rgb_array":
            cv2.imwrite(os.path.join(frames_folder, f"frame_{nb_options}.png"), options_env.render())

    if terminated or truncated:
        print('Terminated', terminated, 'truncated', truncated)
        S, _ = env.reset(seed=SEED)
        indices = T[S]
        q_vals = np.array(learner.get_action_values(indices))
        O = select_option_epsilon_greedy_swift_sarsa(q_vals, EPSILON)
        
    if nb_options >= MAX_OPTIONS_PER_TIMELIMIT_EPISODE:
        logging_data["timelimit_episode"].append(idx_timelimit_episode)
        logging_data["return_per_timelimit"].append(return_per_timelimit)
        logging_data["reward_per_option"].append(R)
        logging_data["real_time_seconds"].append(real_time_seconds)

        print(f"Episode {idx_timelimit_episode} | reward: {return_per_timelimit:.4f} | time in seconds: {(real_time_seconds):.4f} | time in hours: {(real_time_seconds) / 3600:.4f} | epsilon: {EPSILON:.4f}")

        # Save logging data.
        logging_data_df = pd.concat([logging_data_df, pd.DataFrame(logging_data)], ignore_index=True)
        logging_data_df.to_csv(os.path.join(log_dir, "logging_data.csv"), index=False)

        # Save weights.
        np.save(os.path.join(weights_iht_folder, f"weights.npy"), w)
        pickle.dump(iht, open(os.path.join(weights_iht_folder, f"iht.pkl"), "wb"))

        # Save trajectory.
        trajectory_df = pd.DataFrame(options_env.xy_pos_list, columns=["x", "y"])
        trajectory_df.to_csv(os.path.join(folder_trajectory, f"true_pos_xy_{idx_timelimit_episode}.csv"), index=False)

        idx_timelimit_episode += 1
        nb_options = 0
        return_per_timelimit = 0.0

        with PdfPages(os.path.join(log_dir, f"report.pdf")) as pdf:
            # Average reward plot.
            fig, ax = plt.subplots()
            ax.plot(options_env.average_rewards_per_second[options_env.reward_tracker.window_size:])
            ax.set_xlabel('Steps')
            ax.set_ylabel('Average Reward per Second')
            ax.set_title('Average Reward per Second over Steps')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Reward plot.
            fig, ax1 = plt.subplots()
            ax1.plot(logging_data_df['timelimit_episode'], logging_data_df['return_per_timelimit'], color="blue", label='return')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Return')
            ax1.set_title('Return per Episode')
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax1.legend()
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")
            ax2.spines["bottom"].set_position(("outward", 40))
            time_ticks = np.linspace(0, logging_data_df['real_time_seconds'].max() / 3600, 10)  # 10 evenly spaced time points.
            episode_ticks = np.linspace(0, logging_data_df['timelimit_episode'].max(), 10)  # 10 evenly spaced episode points.
            ax1.set_xticks(episode_ticks)
            ax1.set_xticklabels([f"{int(e)}" for e in episode_ticks])
            ax2.set_xticks(episode_ticks)
            ax2.set_xticklabels([f"{t:.2f}h" for t in time_ticks])
            ax2.set_xlabel("Real Time (hours)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Plot the trajectory.
            plt.figure()
            plt.plot(trajectory_df['x'], trajectory_df['y'], '-.', label=f'traj {idx_timelimit_episode}', alpha=0.5)
            plt.scatter(trajectory_df['x'][0], trajectory_df['y'][0], color='red', label='start')
            plt.scatter(trajectory_df['x'].iloc[-1], trajectory_df['y'].iloc[-1], color='green', label='end')
            plt.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), '--', label='circle')
            plt.plot(0, 0, 'x', markersize=10, color='black')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.title(f'Trajectory {idx_timelimit_episode}')
            plt.legend()
            pdf.savefig()
            plt.close()

            # Debugging.
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
            axs = axs.flatten()
            xy_np = np.array(options_env.xy_pos_list)
            ax_pos = axs[0]
            ax_pos.plot(xy_np[:, 0], label='x', color='tab:blue')
            ax_pos.set_ylabel('X Position [m]', color='tab:blue')
            ax_pos.tick_params(axis='y', labelcolor='tab:blue')

            ax_pos_twin = ax_pos.twinx()
            ax_pos_twin.plot(xy_np[:, 1], label='y', color='tab:orange')
            ax_pos_twin.set_ylabel('Y Position [m]', color='tab:orange')
            ax_pos_twin.tick_params(axis='y', labelcolor='tab:orange')

            ax_pos.set_xlabel('Time')
            ax_pos.set_title('X and Y Position over Time')

            reward_np = np.array(options_env.reward_list)
            axs[1].plot(reward_np, '-o', label='reward')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Reward')
            axs[1].set_title('Reward over time')
            axs[1].legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
