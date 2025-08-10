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
        total_reward = 0
        for i in range(self.duration_steps(option)):
            self.joint_pos[opt['joint']] = traj[i]
            obs, reward, terminated, truncated, info = self.env.step(self.joint_pos)
            total_reward = reward + self.discount * total_reward
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


import torch
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
        return self.out(x).squeeze(0) # [B, num_options]


N_EPISODES = 100 # Number of episodes.
idx_episode = 0
DURATION_EPISODE = 10 # seconds
MAX_STEPS_PER_EPISODE = int(DURATION_EPISODE / DT)
EPSILON = 0.05 # Epsilon-greedy exploration.
STEP_SIZE_TD = 0.01
DISCOUNTING = 0.99

# LOG 
import datetime
log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)

state_dim = env.observation_space.shape[0]
num_options = len(options)
assert num_options == 16

Q = QFunction(state_dim=state_dim,
              num_options=num_options)
options_env = OptionEnv(env, options)

optimizer = torch.optim.Adam(Q.parameters(), lr=0.001)

# while True:
#     obs, reward, terminated, truncated, info = options_env.step(np.random.randint(len(options)))
#     if hw_config is None:
#         options_env.render()
#     # if terminated or truncated:
#     #     options_env.reset()
import pandas as pd
df = pd.DataFrame(columns=["episode", "reward"])

while True:
    
    # Reward per episode.
    reward_per_episode = 0
    
    # Initialize S.
    S, _ = env.reset()

    # Choose option O from S using policy derived from Q (epsilon-greedy).
    random_number = np.random.rand()
    if random_number < EPSILON:
        O = np.random.randint(len(options))
    else:
        with torch.no_grad():
            S_tensor = torch.FloatTensor(S).unsqueeze(0)
            O = np.argmax(Q(S_tensor))

    for t in range(MAX_STEPS_PER_EPISODE):
        # Take option O, observe R, S'.
        S_prime, R, terminated, truncated, info = options_env.step(O)
        S_prime_tensor = torch.FloatTensor(S_prime).unsqueeze(0)

        # Choose option O' from S' using policy derived from Q (epsilon-greedy).
        random_number = np.random.rand()
        if random_number < EPSILON:
            O_prime = np.random.randint(len(options))
        else:
            with torch.no_grad():
                O_prime = np.argmax(Q(S_prime_tensor))

        k = options_env.duration_steps(O)
        # Update Q(S, O)
        # Q(S, O) = Q(S, O) + STEP_SIZE_TD * (R + DISCOUNTING^k Q(S', O') - Q(S, O))
        with torch.no_grad():
            target = R + (DISCOUNTING * DT) ** k * Q(S_prime_tensor)[O_prime]

        optimizer.zero_grad()
        S_tensor = torch.FloatTensor(S).unsqueeze(0)
        loss = F.mse_loss(Q(S_tensor)[O], target)
        loss.backward()
        optimizer.step()

        S = S_prime.copy()
        O = O_prime

        reward_per_episode += R

        if terminated or truncated:
            break

    print(f"Episode {idx_episode} reward: {reward_per_episode}")
    df.loc[idx_episode] = [idx_episode, reward_per_episode]
    idx_episode += 1

    # Save df.
    df.to_csv(os.path.join(log_dir, "rewards.csv"), index=False)

    # Save Q model.
    torch.save(Q.state_dict(), os.path.join(log_dir, f"Q_model_{idx_episode}.pth"))
