import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env

import numpy as np
import json
import matplotlib.pyplot as plt
import sys

hw_config = sys.argv[1] if len(sys.argv) > 1 else None
render = "human"
DT = 0.05

joint_config = {
            'hip_zero': 0,
            'knee_zero': -np.radians(60),
            'hip_range': np.radians(45),
            'knee_range': np.radians(30),
        }

if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    render_mode = "human" if render else "rgb_array"
    env = AntEnv(xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
                 render_mode=render_mode,
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



def ramp(start_pos: float, end_pos: float, duration: float):
    num = int(duration / DT)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list

env.reset()


options = []
for i in range(4):
    # hip
    options.append({
        "joint": 2*i,
        "target": np.radians(30),
        "duration": 0.3
    })
    options.append({
        "joint": 2*i,
        "target": -np.radians(30),
        "duration": 0.3
    })
    # knee
    options.append({
        "joint": 2*i + 1,
        "target": np.radians(20),
        "duration": 0.3
    })
    options.append({
        "joint": 2*i + 1,
        "target": -np.radians(20),
        "duration": 0.3
    })


np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)

class OptionEnv:
    def __init__(self, env, options, discount=0.99):
        self.env = env
        self.options = options
        self.discount = discount
        self.joint_pos = np.zeros(len(env.q_joints))
    
    def step(self, option: int):
        opt = self.options[option]
        traj = ramp(self.joint_pos[opt['joint']], opt['target'], opt['duration'])
        print("from to", self.joint_pos[opt['joint']], opt['target'])
        total_reward = 0
        for i in range(int(opt['duration'] / DT)):
            self.joint_pos[opt['joint']] = traj[i]
            print( traj[i])
            print(self.joint_pos)
            obs, reward, terminated, truncated, info = self.env.step(self.joint_pos)
            print(self.joint_pos)
            total_reward = reward + self.discount * total_reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info
        print("final", self.joint_pos)
        return obs, total_reward, terminated, truncated, info

    def reset(self):
        self.joint_pos = np.zeros(len(self.env.q_joints))
        return self.env.reset()
    
    def render(self):
        return self.env.render()


options_env = OptionEnv(env, options)

while True:
    obs, reward, terminated, truncated, info = options_env.step(np.random.randint(len(options)))
    if hw_config is None:
        options_env.render()
    # if terminated or truncated:
    #     options_env.reset()


import torch.nn as nn
import torch.nn.functional as F


class QFunction(nn.Module):
    def __init__(self, state_dim, num_options):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)   # first hidden layer
        self.fc2 = nn.Linear(128, 128)         # second hidden layer
        self.out = nn.Linear(128, num_options) # output Q-values for each action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x) # [B, num_options]