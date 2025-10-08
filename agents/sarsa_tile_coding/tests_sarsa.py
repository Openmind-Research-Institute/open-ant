import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import cv2
import pickle
from tqdm import tqdm
import time

np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)

YELLOW = "\033[93m"
RESET = "\033[0m"

# Random seed.
np.random.seed(42)

# Options.
def ramp(start_pos: float, end_pos: float, duration: float):
    num = int(duration / DT)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list

plt.figure()
class OptionEnv:
    def __init__(self, env, options, discount=0.99):
        self.env = env
        self.options = options
        self.discount = discount
        self.joint_action = np.zeros(env.action_space.shape[0])
        self.obs_list = []
        self.time_list = []
        self.xy_pos_list = []

    def step(self, option_idx: int):
        opt = self.options[option_idx]

        # Populate the joint action trajectory.
        hip_joint = opt['hip_joint']
        hip_traj = ramp(self.joint_action[hip_joint], opt['hip_target'], opt['duration'])

        knee_joint = opt['knee_joint']
        num_steps = len(hip_traj)
        if opt['hip_target'] != self.joint_action[hip_joint]:
            time = np.linspace(0, opt['duration'], num_steps)
            knee_traj = opt['knee_amplitude'] * np.sin(np.pi * time / opt['duration'])
        else:
            # NOTE: This is done to avoid the knee from flopping when the hip is not moving.
            # NOTE: Otherwise, the ant will take advantage of the knee flopping to move forward.
            knee_traj = np.full(num_steps, self.joint_action[knee_joint])

        total_reward = 0.0
        gamma_i = 1.0

        # For plotting.
        joint_pos_true_traj = np.zeros((self.duration_steps(option_idx), self.env.action_space.shape[0]))
        action_traj = np.zeros((self.duration_steps(option_idx), self.env.action_space.shape[0]))

        for i in range(self.duration_steps(option_idx)):
            self.joint_action[hip_joint] = hip_traj[i]
            self.joint_action[knee_joint] = knee_traj[i]

            if option_idx == 3:
                self.joint_action[hip_joint + 2] = hip_traj[i]
                self.joint_action[knee_joint + 2] = knee_traj[i]
            if option_idx == 14:
                self.joint_action[hip_joint - 2] = hip_traj[i]
                self.joint_action[knee_joint - 2] = knee_traj[i]
            obs, reward, terminated, truncated, info = self.env.step(self.joint_action)
            self.obs_list.append(obs)
            self.xy_pos_list.append([info["current_x_position"], info["current_y_position"]])

            # For plotting.
            joint_pos_true_traj[i] = obs[0:self.env.action_space.shape[0]]
            for idx_joint in range(4):
                action_traj[i, 2*idx_joint] = np.clip(self.joint_action[2*idx_joint], -1, 1) * joint_config['hip_range'] + joint_config['hip_zero']
                action_traj[i, 2*idx_joint + 1] = np.clip(self.joint_action[2*idx_joint + 1], -1, 1) * joint_config['knee_range'] + joint_config['knee_zero']
            self.time_list.append(i * DT)

            total_reward += gamma_i * reward
            gamma_i *= self.discount
            if terminated or truncated:
                self.xy_pos_list = []
                return obs, total_reward, terminated, truncated, info

        # plt.clf()
        # if option_idx == 0 or option_idx == 1 or option_idx == 2 or option_idx == 3:
        #     plt.plot(time_, np.rad2deg(joint_pos_true_traj[:, hip_joint+2]), label=f'meas hip nb {hip_joint+2}', color='blue')
        #     plt.plot(time_, np.rad2deg(action_traj[:, hip_joint+2]), label=f'hip action nb {hip_joint+2}', color='blue', linestyle='--')
        #     plt.plot(time_, np.rad2deg(joint_pos_true_traj[:, knee_joint+2]), label=f'meas knee nb {knee_joint+2}', color='red')
        #     plt.plot(time_, np.rad2deg(action_traj[:, knee_joint+2]), label=f'knee action nb {knee_joint+2}', color='red', linestyle='--')
        # if option_idx == 12 or option_idx == 13 or option_idx == 14 or option_idx == 15:
        #     plt.plot(time_, np.rad2deg(joint_pos_true_traj[:, hip_joint-2]), label=f'meas hip nb {hip_joint-2}', color='blue')
        #     plt.plot(time_, np.rad2deg(action_traj[:, hip_joint-2]), label=f'hip action nb {hip_joint-2}', color='blue', linestyle='--')
        #     plt.plot(time_, np.rad2deg(joint_pos_true_traj[:, knee_joint-2]), label=f'meas knee nb {knee_joint-2}', color='red')
        #     plt.plot(time_, np.rad2deg(action_traj[:, knee_joint-2]), label=f'knee action nb {knee_joint-2}', color='red', linestyle='--')
        # plt.plot(time_, np.rad2deg(joint_pos_true_traj[:, hip_joint]), label=f'meas hip nb {hip_joint}', color='blue')
        # plt.plot(time_, np.rad2deg(action_traj[:, hip_joint]), label=f'hip action nb {hip_joint}', color='blue', linestyle='--')
        # plt.plot(time_, np.rad2deg(joint_pos_true_traj[:, knee_joint]), label=f'meas knee nb {knee_joint}', color='red')
        # plt.plot(time_, np.rad2deg(action_traj[:, knee_joint]), label=f'knee action nb {knee_joint}', color='red', linestyle='--')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Joints (deg)')
        # plt.title(f'Option {opt['name']} idx {option_idx}')
        # plt.show(block=False)

        return obs, total_reward, terminated, truncated, info

    def reset(self):
        self.joint_pos = np.zeros(len(self.env.q_joints))
        return self.env.reset()

    def render(self):
        return self.env.render()
    
    def duration_steps(self, option_idx: int):
        return int(self.options[option_idx]['duration'] / DT)


options = []
for i in range(4):  # 4 legs
    options.append({
        "name": "sinusoid_forward",
        "hip_joint": 2*i,
        "hip_target": np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(45),
        "duration": 0.3
    })
    options.append({
        "name": "sinusoid_backward",
        "hip_joint": 2*i,
        "hip_target": -np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(45),
        "duration": 0.3
    })
    options.append({
        "name": "stance_forward",
        "hip_joint": 2*i,
        "hip_target": np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(-20),
        "duration": 0.3
    })
    options.append({
        "name": "stance_backward",
        "hip_joint": 2*i,
        "hip_target": -np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(-20),
        "duration": 0.3
    })

print(len(options), "options defined.")  # should print 8
assert len(options) == 16

# Constants.
render = "human"
DT = 0.05

# Environment.
joint_config = {
    'hip_zero': 0,
    'knee_zero': -np.radians(60),
    'hip_range': np.radians(45),
    'knee_range': np.radians(45),
}

hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    render_mode = "human" if render else "rgb_array"
    env = AntEnv(
                 render_mode="human",
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

options_env = OptionEnv(env, options)

# Go through each option and run the option on the env.
# For debugging.
env.reset()
xy_pos = []
date_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_COMPARISON = 'logs/sim_to_real_comparison_' + date_now
if not os.path.exists(LOGS_COMPARISON):
    os.makedirs(LOGS_COMPARISON)

N = 20
counter = 0
plt.figure()
while True:
    # sinusoid forward, sinusoid backward, stance forward, stance backward
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    # list_options = [0, 4, 3]
    list_options = [0, 4, 3, \
                    9, 13, 14]
    i = np.random.randint(0, len(options))
    option = options[i]
    print('option', option, 'name', option['name'])
    obs, reward, terminated, truncated, info = options_env.step(i)
    xy_pos.append([info["current_x_position"], info["current_y_position"]])
    xy_pos_np = np.array(xy_pos)
    plt.plot(xy_pos_np[-4:, 0], xy_pos_np[-4:, 1], '-o', markersize=5, color='b')
    # last point in red
    plt.plot(xy_pos_np[-1, 0], xy_pos_np[-1, 1], 'o', markersize=10, color='r')
    # Write the position (last one) to the point.
    # plt.text(xy_pos_np[-1, 0], xy_pos_np[-1, 1], f'{xy_pos_np[-1, 0]:.2f}, {xy_pos_np[-1, 1]:.2f}')
    plt.plot(0, 0, 'x')
    # plt.axis('equal')
    plt.pause(0.001)

    print(f"Option {i} | reward: {reward:.4f}")

    counter += 1
    
    if counter % 20 == 0:
        # reset
        env.reset()

# list_obs_names = ['joint_pos_1', 'joint_pos_2', 'joint_pos_3', 'joint_pos_4', 'joint_pos_5', 'joint_pos_6', 'joint_pos_7', 'joint_pos_8',
#                   'joint_vel_1', 'joint_vel_2', 'joint_vel_3', 'joint_vel_4', 'joint_vel_5', 'joint_vel_6', 'joint_vel_7', 'joint_vel_8',
#                   'heading_x', 'heading_y',
#                   'acc_x', 'acc_y', 'acc_z',
#                   'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
# print(len(list_obs_names), "obs names")

# obs_list = np.array(options_env.obs_list)
# time_list = np.array(options_env.time_list).reshape(-1, 1)
# # Save the observations in a csv file
# df_obs = pd.DataFrame(np.concatenate((time_list, obs_list), axis=1), columns=['time'] + list_obs_names, index=None)
# df_obs.to_csv(os.path.join(LOGS_COMPARISON, f"{env_id}_obs.csv"))
# print(df_obs.shape)

# # Make a histogram of all the observations.
# fig, axs = plt.subplots(6, 4, sharex=True)
# axs = axs.flatten()
# for i in range(24):
#     axs[i].hist(obs_list[:, i], bins=100, label=f'obs {list_obs_names[i]} normalized')
#     axs[i].legend()
#     axs[i].set_ylabel('count')
# plt.tight_layout()
# plt.savefig(os.path.join(LOGS_COMPARISON, f"{env_id}_obs_histogram.png"))

# df_xy_pos = pd.DataFrame(xy_pos, columns=["x", "y"])
# df_xy_pos.to_csv(os.path.join(LOGS_COMPARISON, f"{env_id}_xy_pos.csv"), index=False)
# xy_pos_np = np.array(xy_pos)
# x0 = xy_pos_np[0, 0]
# y0 = xy_pos_np[0, 1]
# xf = xy_pos_np[-1, 0]
# yf = xy_pos_np[-1, 1]
# distance = np.linalg.norm([xf - x0, yf - y0])
# print(f"x0: {x0}, y0: {y0}, xf: {xf}, yf: {yf}")
# print('difference in x', xf - x0)
# print(f"Distance: {distance}")
# plt.figure()
# plt.plot(xy_pos_np[:, 0], xy_pos_np[:, 1], label=f'traj', alpha=0.5)
# plt.scatter(xy_pos_np[0, 0], xy_pos_np[0, 1], color='red', label='start')
# plt.scatter(xy_pos_np[-1, 0], xy_pos_np[-1, 1], color='green', label='end')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Trajectory')
# plt.legend()
# plt.savefig(os.path.join(LOGS_COMPARISON, f"{env_id}_trajectory.png"))
# # plt.close()
# plt.show()
