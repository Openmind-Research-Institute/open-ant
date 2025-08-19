import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import cv2

np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)

RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Random seed.
np.random.seed(42)

# Options.
def ramp(start_pos: float, end_pos: float, duration: float):
    num = int(duration / DT)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list

class OptionEnv:
    def __init__(self, env, options, discount=0.99):
        self.env = env
        self.options = options
        self.discount = discount
        self.joint_pos = np.zeros(len(env.q_joints))

    def step(self, option: int):
        opt = self.options[option]
        traj = ramp(self.joint_pos[opt['joint']], opt['target'], opt['duration'])
        total_reward = 0.0
        gamma_i = 1.0
        for i in range(self.duration_steps(option)):
            self.joint_pos[opt['joint']] = traj[i]
            obs, reward, terminated, truncated, info = self.env.step(self.joint_pos)
            total_reward += gamma_i * reward
            gamma_i *= self.discount
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info
        return obs, total_reward, terminated, truncated, info

    def reset(self):
        self.joint_pos = np.zeros(len(self.env.q_joints))
        return self.env.reset()

    def render(self):
        return self.env.render()
    
    def duration_steps(self, option: int):
        return int(self.options[option]['duration'] / DT)

options = []
for i in range(4):
    # hip
    options.append({"joint": 2*i, "target": np.radians(40),  "duration": 0.3})
    options.append({"joint": 2*i, "target": -np.radians(40), "duration": 0.3})
    # knee
    options.append({"joint": 2*i + 1, "target": np.radians(30),  "duration": 0.3})
    options.append({"joint": 2*i + 1, "target": -np.radians(30), "duration": 0.3})

# Constants.
render = "human"
DT = 0.05

# Environment.
joint_config = {
    'hip_zero': 0,
    'knee_zero': -np.radians(60),
    'hip_range': np.radians(45),
    'knee_range': np.radians(30),
}

hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    render_mode = "human" if render else "rgb_array"
    env = AntEnv(xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
                 render_mode="rgb_array",
                 dt=DT,
                 joint_config=joint_config)
else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                       render_mode='human',
                       dt=DT,
                       joint_config=joint_config)

# Tile coding.
from tilecoding import IHT, tiles
class SuttonTileCoderWrapper:
    def __init__(self, iht: IHT, tiles_per_dim, value_limits, tilings):
        self.iht = iht # IHT table.
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=np.int32)
        self.tilings = int(tilings)
        self.limits = np.asarray(value_limits, dtype=np.float64)
        self.norm_dims = np.array(tiles_per_dim) / (self.limits[:, 1] - self.limits[:, 0])
        assert self.limits.shape == (self.tiles_per_dim.shape[0], 2)

    def __getitem__(self, x):
        x = np.asarray(x, dtype=np.float64)
        floats_scaled = (x - self.limits[:, 0]) * self.norm_dims  # in tile units
        # No extra ints; one index per tiling:
        idxs = tiles(self.iht, self.tilings, floats_scaled)
        return np.asarray(idxs, dtype=np.int64)

    @property
    def n_tiles(self):
        return self.iht.size

log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)
df = pd.DataFrame(columns=["episode", "reward", "real_time_seconds"])

def q_of(w, idx, o):
    return w[o, idx].sum()

def greedy_option(w, T, state, num_options):
    idx = T[state]
    q_vals = np.array([w[o, idx].sum() for o in range(num_options)], dtype=np.float64)
    # Tie-break among maxima, in case of ties.
    maxq = q_vals.max()
    best = np.flatnonzero(q_vals == maxq)
    # plt.clf()
    # plt.bar(range(len(q_vals)), q_vals)
    # # Color the highest q-value in red.
    # plt.bar(np.argmax(q_vals), q_vals[np.argmax(q_vals)], color='red')
    # plt.title('Q-values for each option')
    # plt.xlabel('Option')
    # plt.ylabel('Q-value')
    # plt.pause(0.01)
    return int(np.random.choice(best)), q_vals

def select_option_epsilon_greedy(S, epsilon, w, T):
    # ε-greedy over options using tile-coded T(s).
    if np.random.rand() < epsilon:
        return np.random.randint(num_options)
    O_greedy, _ = greedy_option(w, T, S, num_options)
    return O_greedy

def clip_state_to_limits(S, limits):
    S = np.asarray(S, dtype=np.float64)
    return np.clip(S, limits[:, 0], limits[:, 1])

# Constants.
# DURATION_EPISODE = 5 # seconds
MAX_OPTIONS_PER_EPISODE = 300
EPSILON = 0.05
DISCOUNTING = 0.99

DIM_TILING = 10 # Number of tiles per dimension.
TILINGS = 8 # Number of offset tilings.
IHT_SIZE = 2**18

# Environment.
env.reset()
options_env = OptionEnv(env, options)

# Limits from observation space.
state_limits = np.array([env.observation_space.low, env.observation_space.high]).T  # [state_dim, 2]
num_options = len(options)

# IHT table size.
tiles_per_dim = [DIM_TILING] * state_limits.shape[0]
iht = IHT(IHT_SIZE)
T = SuttonTileCoderWrapper(iht=iht,
                           tiles_per_dim=tiles_per_dim,
                           value_limits=state_limits,
                           tilings=TILINGS)

# TODO: integrate Kris's tile coding.
# from tilecoding import KrisTileCoder
# T = KrisTileCoder(tiles_per_dim, state_limits, TILINGS)

# Linear weights: [num_options, iht.size].
# Q is parametrized as w * T(s), with T(s) being the tile-coded state.

load_previous_weights = False
if load_previous_weights == False:
    w = np.zeros((num_options, iht.size), dtype=np.float32)
    print(f"w.shape: {w.shape}")
else:
    log_dir = 'logs/20250817_151556'
    # find the latest weights file.
    # print(f"Latest weights file: {latest_weights_file}")
    w = np.load(os.path.join(log_dir, 'weights_493.npy'))
    print('Loaded weights from previous run.')
    print(f"w.shape: {w.shape}")

# Step-size.
# See: http://incompleteideas.net/tiles/tiles3.html
step_size = 0.1 / TILINGS

with open(os.path.join(log_dir, "tile_config.json"), "w") as f:
    json.dump({
        "tiles_per_dim": tiles_per_dim,
        "tilings": TILINGS,
        "state_limits": state_limits.tolist(),
        "iht_size": IHT_SIZE
    }, f, indent=2)

idx_episode = 0
real_time_seconds = 0.0

while True:
    EPSILON = max(0.05, 0.2 - idx_episode * 0.00015)

    true_pos_xy = []
    reward_per_episode = 0.0

    # Reset environment.
    S, _ = env.reset()
    S = clip_state_to_limits(S, state_limits) # Ensure state is within limits of the tile coder.

    # Select option.
    O = select_option_epsilon_greedy(S, EPSILON, w, T)

    # Run episode.
    for t in range(MAX_OPTIONS_PER_EPISODE):
        # print(f"Episode {idx_episode} | step {t} | option {O}")

        # Run option O.
        S_prime, R, terminated, truncated, info = options_env.step(O)
        S_prime = clip_state_to_limits(S_prime, state_limits) # Ensure state is within limits of the tile coder.
        # if idx_episode % 100 == 0:
        #     frame = env.render()
        #     cv2.imshow("Ant", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(1) # 1 ms delay so the window updates

        # Next option (ε-greedy).
        O_prime = select_option_epsilon_greedy(S_prime, EPSILON, w, T)

        # TD.
        k = options_env.duration_steps(O)
        idx_S  = T[S]
        idx_S_prime = T[S_prime]

        # TODO: add the Delta Ts.
        target = R + (DISCOUNTING ** k) * q_of(w, idx_S_prime, O_prime)
        pred = q_of(w, idx_S,  O)
        delta = target - pred

        # Update weights.
        w[O, idx_S] += step_size * delta

        S = S_prime
        O = O_prime
        reward_per_episode += R
        duration_option = options_env.duration_steps(O)
        real_time_seconds += duration_option * DT

        # true_pos_xy.append([info["current_x_position"], info["current_y_position"]])
        if terminated or truncated:
            break

    print(f"Episode {idx_episode} | reward: {YELLOW}{reward_per_episode:.4f}{RESET} | time in seconds: {(t * idx_episode * DT):.4f} | time in hours: {(t * idx_episode * DT) / 3600:.4f} | epsilon: {EPSILON:.4f}")
    df.loc[idx_episode] = [idx_episode, reward_per_episode, real_time_seconds]
    idx_episode += 1

    # Save weights.
    np.save(os.path.join(log_dir, f"weights_{idx_episode}.npy"), w)

    # Save logs and weights.
    df.to_csv(os.path.join(log_dir, "rewards.csv"), index=False)

    if idx_episode % 100 == 0:
        # Reward plot.
        fig, ax1 = plt.subplots()
        ax1.plot(df['episode'], df['reward'], color="blue", label='rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Rewards over Time')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.legend()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 40))  # shift it down
        max_time = df['real_time_seconds'].max()
        max_episode = df['episode'].max()
        time_ticks = np.linspace(0, max_time, 10)  # 10 evenly spaced time points
        episode_ticks = np.linspace(0, max_episode, 10)  # 10 evenly spaced episode points
        ax1.set_xticks(episode_ticks)
        ax1.set_xticklabels([f"{int(e)}" for e in episode_ticks])
        ax2.set_xticks(episode_ticks)
        ax2.set_xticklabels([f"{t:.0f}s" for t in time_ticks])
        ax2.set_xlabel("Real Time (seconds)")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"rewards.png"))
        plt.close()

        # Save and plot the trajectory.
        # df_true_pos_xy = pd.DataFrame(true_pos_xy, columns=["x", "y"])
        # df_true_pos_xy.to_csv(os.path.join(log_dir, f"true_pos_xy_{idx_episode}.csv"), index=False)
        # # Generate a plot.
        # true_pos_xy_df = pd.read_csv(os.path.join(log_dir, f'true_pos_xy_{idx_episode}.csv'))
        # x0 = true_pos_xy_df['x'][0]
        # y0 = true_pos_xy_df['y'][0]
        # xf = true_pos_xy_df['x'].iloc[-1]
        # yf = true_pos_xy_df['y'].iloc[-1]
        # distance = np.linalg.norm([xf - x0, yf - y0])
        # print(f"distance: {distance/30}")
        # plt.figure()
        # plt.plot(true_pos_xy_df['x'], true_pos_xy_df['y'], label=f'traj {idx_episode}', alpha=0.5)
        # plt.scatter(true_pos_xy_df['x'][0], true_pos_xy_df['y'][0], color='red', label='start')
        # plt.scatter(true_pos_xy_df['x'].iloc[-1], true_pos_xy_df['y'].iloc[-1], color='green', label='end')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Trajectory')
        # plt.ylim(-1, 1)
        # plt.legend()
        # plt.savefig(os.path.join(log_dir, f"trajectory_{idx_episode}.png"))
        # plt.close()