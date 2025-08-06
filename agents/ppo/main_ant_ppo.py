import gymnasium as gym
import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sim'))
from ant_mujoco import AntEnv
sys.path.append(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env'))
from embodied_ant_env import make_ant_env


from ppo import Agent

def plot_rewards(test_rewards: list[float], window_size: int = 50):
    """Plot rewards with moving average and confidence intervals."""
    plt.figure(figsize=(12, 8))

    def compute_moving_average(data, window):
        """Compute moving average with confidence intervals."""
        if len(data) < window:
            return data, None, None
        data_np = np.array(data)
        moving_avg = np.convolve(data_np, np.ones(window)/window, mode='valid')
        moving_std = np.array([np.std(data_np[max(0, i-window+1):i+1]) for i in range(window-1, len(data_np))])
        confidence_interval = 1.96 * moving_std
        x_avg = np.arange(window-1, len(data_np))
        return moving_avg, x_avg, confidence_interval

    if test_rewards is not None and len(test_rewards) > 0:
        test_x = np.arange(0, len(test_rewards)) * 10  # Test rewards are collected every 10 iterations
        plt.plot(test_x, test_rewards, 'r-', linewidth=0.5, label='Test Reward', alpha=0.3)

        if len(test_rewards) >= window_size:
            test_avg, test_x_avg, test_ci = compute_moving_average(test_rewards, window_size)
            test_x_avg_adjusted = test_x_avg * 10
            plt.plot(test_x_avg_adjusted, test_avg, 'r-', linewidth=2, label=f'Test MA ({window_size})')

            if test_ci is not None:
                plt.fill_between(test_x_avg_adjusted, test_avg - test_ci, test_avg + test_ci, 
                               alpha=0.2, color='red', label='Test 95% CI')

    plt.title('Reward Progress with Moving Average')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()


def collect_trajectory(agent: Agent, env: gym.Env, trajectory_length: int):
    """Collect a single long trajectory from the environment.
        Split trajectory into N segments.
    """
    print(f'Collecting trajectory of length {trajectory_length}')

    # Reset environment and get initial observation.
    obs_reset, _ = env.reset()
    obs_tensor = torch.tensor(obs_reset, dtype=torch.float32, device=agent.device).unsqueeze(0)

    observations = []
    logits_list = []
    actions = []
    rewards = []
    dones = []
    truncations = []
    
    cumulative_reward = 0.0
    
    for _ in range(trajectory_length):

        observations.append(obs_tensor)
        logits, action = agent.get_logits_action(obs_tensor)
        action_np = agent.dist_postprocess(action).detach().cpu().numpy()[0]
        obs, reward, terminated, truncated, _ = env.step(action_np)

        cumulative_reward += reward

        # Store action and step data.
        logits_list.append(logits)
        actions.append(action)
        rewards.append(torch.tensor([reward], device=agent.device))
        dones.append(torch.tensor([float(terminated)], device=agent.device))
        truncations.append(torch.tensor([float(truncated)], device=agent.device))

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)

        # # Stop if episode is done
        if terminated or truncated:
            # Add the final observation (needed for bootstrapping)
            observations.append(obs_tensor)
            # break

    # If we didn't break early, add the final observation.
    if len(observations) == len(actions):
        observations.append(obs_tensor)

    # Stack all data into tensors.
    trajectory_data = {
        'observation': torch.cat(observations, dim=0),
        'logits': torch.cat(logits_list, dim=0),
        'action': torch.cat(actions, dim=0),
        'reward': torch.cat(rewards, dim=0),
        'done': torch.cat(dones, dim=0),
        'truncation': torch.cat(truncations, dim=0)
    }

    return trajectory_data, cumulative_reward


current_path = os.path.dirname(os.path.abspath(__file__))
current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(current_path, "results", current_date_time)
os.makedirs(results_dir, exist_ok=True)

# Load the ant environment.
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode='human')
else:
    env = AntEnv(render_mode="human",
                 dt=0.01)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print('Observation dimension:', obs_dim)
print('Action dimension:', act_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy and value networks
policy_layers = [obs_dim, 256, 128, act_dim * 2]
value_layers = [obs_dim, 256, 128, 1]

ENTROPY_COST = 0.005
DISCOUNTING = 0.97
REWARD_SCALING = 1.0
CLIP_EPSILON = 0.3
LEARNING_RATE = 3e-4

agent = Agent(policy_layers,
              value_layers,
              ENTROPY_COST,
              DISCOUNTING,
              REWARD_SCALING,
              CLIP_EPSILON,
              device)

optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

# Simple training loop
TRAJECTORY_LENGTH = 100
NUM_ITERATIONS = 5000
NUM_UPDATE_EPOCHS = 50

total_loss = 0
t = time.time()

# Track rewards for plotting
rewards_history = []
test_rewards_history = []
plot_save_path = os.path.join(results_dir, "reward_progress.png")

for iteration in range(NUM_ITERATIONS):
    print(f'Iteration {iteration + 1}/{NUM_ITERATIONS}')
    
    trajectory, cumulative_reward = collect_trajectory(agent, env, TRAJECTORY_LENGTH)
    
    rewards_history.append(cumulative_reward)

    # Update normalization statistics.
    print(trajectory['observation'].shape)
    agent.update_normalization(trajectory['observation'])

    # Update policy for several epochs.
    for epoch in range(NUM_UPDATE_EPOCHS):
        # Create a fresh copy of trajectory data to avoid backward graph issues.
        trajectory_copy = {}
        for k, v in trajectory.items():
            if isinstance(v, torch.Tensor):
                trajectory_copy[k] = v.clone().detach()
            else:
                trajectory_copy[k] = v

        loss = agent.loss(trajectory_copy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Loss: {loss.item():.4f}')

    if (iteration + 1) % 10 == 0:
        test_trajectory, cumulative_test_reward = collect_trajectory(agent, env, 200)
        test_rewards_history.append(cumulative_test_reward)
        plot_rewards(test_rewards_history)
        print(f'Iteration {iteration + 1} Test Reward: {cumulative_test_reward:.2f}')

duration = time.time() - t
avg_loss = total_loss / (NUM_ITERATIONS * NUM_UPDATE_EPOCHS)
print(f'Training completed in {duration:.2f} seconds')
print(f'Average loss: {avg_loss:.4f}')

# Save final reward plot.
plot_rewards(rewards_history, plot_save_path, test_rewards_history)
print(f'Final reward plot saved to {plot_save_path}')

# Save reward data as numpy array for later analysis.
rewards_array = np.array(rewards_history)
test_rewards_array = np.array(test_rewards_history)

# Save reward data.
np.save(os.path.join(results_dir, "rewards_history.npy"), rewards_array)
np.save(os.path.join(results_dir, "test_rewards_history.npy"), test_rewards_array)
print(f'Reward history saved to {os.path.join(results_dir, "rewards_history.npy")}')
print(f'Test reward history saved to {os.path.join(results_dir, "test_rewards_history.npy")}')

# Save the model.
print('Saving model...')
torch.save(agent.policy.state_dict(), os.path.join(results_dir, "ppo_model_pytorch.pth"))

env.close()
