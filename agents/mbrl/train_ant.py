import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import json

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env
from cem import CEM

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

def make_model_step_fn(model: DynamicsModel):
    def step_fn(state: np.ndarray, action: np.ndarray):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            delta = model(s, a)
            # TODO: add the DT correctly.
            next_state = s + delta
            next_state = next_state.squeeze().numpy()
        # Ant reward.
        angular_vel_z = state[-1]
        rotate_reward = angular_vel_z
        reward = rotate_reward
        return next_state, reward
    return step_fn


def train_model(model, data, epochs=40):
    if len(data['states']) < 10:
        print("Not enough data yet.")
        return
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    states = torch.tensor(data['states'], dtype=torch.float32)
    actions = torch.tensor(data['actions'], dtype=torch.float32)
    next_states = torch.tensor(data['next_states'], dtype=torch.float32)
    delta_states = next_states - states
    for _ in range(epochs):
        pred = model(states, actions)
        loss = loss_fn(pred, delta_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")


# Env setup
render = "human"
DT = 0.05
hw_config = sys.argv[1] if len(sys.argv) > 1 else None

if hw_config is None:
    env_id = 'ant_mujoco'
    env = AntEnv(
        # render_mode=render,
        dt=DT,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.0,
        reward_upside_down_weight=0.0
    )
else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode=render, dt=DT)

obs, _ = env.reset(seed=0)

state_dim = 24
action_dim = 8
print(env.action_space.low)
print(env.action_space.high)
print(np.array([env.action_space.low, env.action_space.high]))
action_lims = np.array([env.action_space.low, env.action_space.high]).reshape(8, 2)

model = DynamicsModel(state_dim, action_dim)

HORIZON = 20
NUM_PARTICLES = 5000
NUM_ITERATIONS = 20
NUM_ELITE = 20

cem = CEM(
    action_dim=action_dim,
    action_lims=action_lims,
    horizon=HORIZON,
    num_particles=NUM_PARTICLES,
    num_iterations=NUM_ITERATIONS,
    num_elite=NUM_ELITE,
    mean=np.zeros((HORIZON, action_dim)),
    cov=np.array([np.eye(action_dim)*0.5 for _ in range(HORIZON)]),
)

# Dataset.
data = {
    'states': [],
    'actions': [],
    'next_states': [],
}

obs_list = []
action_list = []

for step in range(200):
    print(f"=== Step {step} ===")

    # Make step function using current model.
    step_fn = make_model_step_fn(model)

    # Plan using CEM and the learned model.
    best_action_seq, _, _ = cem.optimize(
        initial_state=obs,
        step_fn=step_fn,
    )
    action = best_action_seq[0]
    print('action', action)

    # Execute in real env.
    next_obs, reward, terminated, truncated, _ = env.step(action)
    print('next_obs', next_obs)
    print("Reward: ", reward)

    # Add to dataset.
    data['states'].append(obs)
    data['actions'].append(action)
    data['next_states'].append(next_obs)

    # Update one-step model.
    train_model(model, data, epochs=40)

    # Update observation.
    obs = next_obs.copy()
    obs_list.append(obs)
    action_list.append(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# Plot
obs_np = np.array(obs_list)
angles = np.arctan2(obs_np[:, 1], obs_np[:, 0])

fig, axs = plt.subplots(2, 1)
axs[0].plot(angles)
axs[0].set_ylabel("Angle")
axs[1].plot(action_list)
axs[1].set_ylabel("Action")
plt.tight_layout()
plt.show()
