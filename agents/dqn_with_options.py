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
DT = 0.02

joint_config = {
            'hip_zero': 0,
            'knee_zero': -np.radians(60),
            'hip_range': np.radians(45),
            'knee_range': np.radians(60),
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


def option(start_pos: float, end_pos: float, duration: float):
    input_pos_list = [start_pos]
    for i in range(int(duration / DT)):
        input_pos_list.append(input_pos_list[-1] + (end_pos - start_pos) / (duration / DT))
    return input_pos_list

env.reset()
o_move_hip_positive_dir = option(start_pos=-joint_config['hip_range'], end_pos=joint_config['hip_range'], duration=1)
o_move_hip_negative_dir = option(start_pos=joint_config['hip_range'], end_pos=-joint_config['hip_range'], duration=1)
o_move_knee_positive_dir = option(start_pos=-joint_config['knee_range'], end_pos=joint_config['knee_range'], duration=1)
o_move_knee_negative_dir = option(start_pos=joint_config['knee_range'], end_pos=-joint_config['knee_range'], duration=1)

time_list = np.arange(len(o_move_hip_positive_dir)) * DT
plt.plot(time_list, o_move_hip_positive_dir, label='Option 1: hip positive')
plt.plot(time_list, o_move_hip_negative_dir, label='Option 2: hip negative')
plt.plot(time_list, o_move_knee_positive_dir, label='Option 3: knee positive')
plt.plot(time_list, o_move_knee_negative_dir, label='Option 4: knee negative')
# plt.legend(loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.title('Options')

# List of all options
options_dict = {
    'o_move_hip_positive_dir': {
        'option': o_move_hip_positive_dir,
        'joint_names': ['hip'], 
    },
    'o_move_hip_negative_dir': {
        'option': o_move_hip_negative_dir,
        'joint_names': ['hip'],
    },
    'o_move_knee_positive_dir': {
        'option': o_move_knee_positive_dir,
        'joint_names': ['ankle'],
    },
    'o_move_knee_negative_dir': {
        'option': o_move_knee_negative_dir,
        'joint_names': ['ankle'],
    }
}

expanded_options_dict = {}
for base_name, data in options_dict.items():
    for i in range(1, 5):
        new_name = f"{base_name}_{i}"

        # Update joint_names: hip -> hip_i, ankle -> ankle_i.
        updated_joint_names = [f"{name}_{i}" for name in data['joint_names']]

        expanded_options_dict[new_name] = {
            'option': data['option'],
            'joint_names': updated_joint_names
        }


import torch.nn as nn
import torch.nn.functional as F

import random
options_list = list(expanded_options_dict.keys())
N = 10
time_ = 0
action_list = []
time_list = []
for i in range(N):
    # Pick a random option.
    current_option = random.choice(options_list)
    print('Current option: ', current_option)
    # Execute the current option fully.
    for j in range(len(expanded_options_dict[current_option]['option'])):
        o_value = expanded_options_dict[current_option]['option'][j]
        action = np.zeros(len(env.q_joints))
        for joint_name in expanded_options_dict[current_option]['joint_names']:
            action[env.q_joints[joint_name] - 1] = o_value
        obs, reward, terminated, truncated, info = env.step(action)
        if hw_config is None:
            env.render()
        action_list.append(action)
        time_ += DT
        time_list.append(time_)

action_np = np.array(action_list)
fig, ax = plt.subplots(len(env.name_joints)//2, 2)
for i in range(len(env.name_joints)//2):
    ax[i, 0].plot(time_list, action_np[:, i])
    ax[i, 0].set_title(env.name_joints[i])
    ax[i, 0].set_xlabel('Time (s)')
    ax[i, 0].set_ylabel('Action')
    ax[i, 1].plot(time_list, action_np[:, i + len(env.name_joints)//2])
    ax[i, 1].set_title(env.name_joints[i + len(env.name_joints)//2])
    ax[i, 1].set_xlabel('Time (s)')
    ax[i, 1].set_ylabel('Action')
plt.tight_layout()
plt.show()


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