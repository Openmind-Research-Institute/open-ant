import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env
from gymnasium.wrappers import NormalizeObservation

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
            # if opt['name'].startswith('stance'):
            #     knee_traj = opt['knee_amplitude'] * np.ones(num_steps)
        else:
            # NOTE: This is done to avoid the knee from flopping when the hip is not moving.
            # NOTE: Otherwise, the ant will take advantage of the knee flopping to move forward.
            knee_traj = np.full(num_steps, self.joint_action[knee_joint])

        total_reward = 0.0
        gamma_i = 1.0

        # For plotting.
        joint_pos_true_traj = np.zeros((self.duration_steps(option_idx), self.env.action_space.shape[0]))
        action_traj = np.zeros((self.duration_steps(option_idx), self.env.action_space.shape[0]))
        time_ = []

        for i in range(self.duration_steps(option_idx)):
            self.joint_action[hip_joint] = hip_traj[i]
            self.joint_action[knee_joint] = knee_traj[i]
            
            if option_idx == 0 or option_idx == 1 or option_idx == 2 or option_idx == 3:
                self.joint_action[hip_joint+2] = hip_traj[i]
                self.joint_action[knee_joint+2] = knee_traj[i]
            if option_idx == 12 or option_idx == 13 or option_idx == 14 or option_idx == 15:
                self.joint_action[hip_joint-2] = hip_traj[i]
                self.joint_action[knee_joint-2] = knee_traj[i]
            obs, reward, terminated, truncated, info = self.env.step(self.joint_action)

            # For plotting.
            obs_for_plotting = self.unnormalize_obs(obs)
            joint_pos_true_traj[i] = obs_for_plotting[0:self.env.action_space.shape[0]]
            for idx_joint in range(4):
                action_traj[i, 2*idx_joint] = np.clip(self.joint_action[2*idx_joint], -1, 1) * joint_config['hip_range'] + joint_config['hip_zero']
                action_traj[i, 2*idx_joint + 1] = np.clip(self.joint_action[2*idx_joint + 1], -1, 1) * joint_config['knee_range'] + joint_config['knee_zero']
            time_.append(i * DT)

            total_reward += gamma_i * reward
            gamma_i *= self.discount
            if terminated or truncated:
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

    def unnormalize_obs(self, normalized_obs): # Used for plotting.
        return normalized_obs * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean


options = []
for i in range(4):  # 4 legs
    options.append({
        "name": "sinusoid_forward",
        "hip_joint": 2*i,
        "hip_target": np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(80),
        "duration": 0.3
    })
    options.append({
        "name": "sinusoid_backward",
        "hip_joint": 2*i,
        "hip_target": -np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(80),
        "duration": 0.3
    })
    options.append({
        "name": "stance_forward",
        "hip_joint": 2*i,
        "hip_target": np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(-15),
        "duration": 0.5
    })
    options.append({
        "name": "stance_backward",
        "hip_joint": 2*i,
        "hip_target": -np.radians(45),
        "knee_joint": 2*i + 1,
        "knee_amplitude": np.radians(-10),
        "duration": 0.5
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
    env = AntEnv(xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
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

env = NormalizeObservation(env)
options_env = OptionEnv(env, options)

# Go through each option and run the option on the env.
# For debugging.
env.reset()
xy_pos = []
while True:
    # sinusoid forward, sinusoid backward, stance forward, stance backward
    list_options = [0, 3, 13, 14]
    for i in list_options:
        option = options[i]
        print('option', option)
        obs, reward, terminated, truncated, info = options_env.step(i)
        obs_for_plotting = options_env.unnormalize_obs(obs)
        print('angular_velocities', obs_for_plotting[-1])
        print(f"Option {i} | reward: {reward:.4f}")
        # time.sleep(0.2)
        # input("Press Enter to continue...")
import sys
sys.exit()
