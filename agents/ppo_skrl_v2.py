from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.envs.wrappers.torch import wrap_env
import torch
import torch.nn as nn

import sys
import os
import json
import pandas as pd
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


def run(agent, env):
    obs, info = env.reset()
    i = 0
    time_window = 10.0  # seconds
    window_size = int(time_window / env.dt)
    moving_average_reward_queue = deque(maxlen=window_size)

    df = pd.DataFrame(columns=['step', 'reward'])
    while True:
        agent.pre_interaction(i, -1)
        with torch.no_grad():
            action = agent.act(obs, i, -1)[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            agent.record_transition(obs, action, reward, next_obs, terminated, truncated, info, i, -1)
        if TRAIN == True:
            agent.post_interaction(i, -1)
        obs = next_obs
        i += 1

        if terminated or truncated:
            obs, info = env.reset()

        reward_per_second = reward.item()/env.dt
        moving_average_reward_queue.append(reward_per_second)
        if len(moving_average_reward_queue) > window_size:
            # Pop the leftmost element.
            moving_average_reward_queue.popleft()

        average_reward_per_second = sum(moving_average_reward_queue) / len(moving_average_reward_queue)
        if i % 5000 == 0:
            print(f"Step {i}, moving average reward {average_reward_per_second:.4f}")
            df = pd.concat([df, pd.DataFrame({'step': [i], 'reward': [average_reward_per_second]})], ignore_index=True)
            df.to_csv(os.path.join(LOG_FOLDER, f'rewards_{DATE_NOW}.csv'), index=False)


if len(sys.argv) > 1:
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode='human', dt=0.05)
else:
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    env = AntEnv(
        xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
        render_mode="human",
        dt=0.05)

env = wrap_env(env, "gymnasium")
device = env.device
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model

DATE_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TRAIN = False
LOG_FOLDER = 'logs_ppo_skrl_v2'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 2048  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 64
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
# cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = LOG_FOLDER

memory = RandomMemory(memory_size=2048, num_envs=env.num_envs, device=device)

agent = PPO(models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device)

agent.init()

if TRAIN == False:
    all_folders = [folder for folder in os.listdir(LOG_FOLDER) if os.path.isdir(os.path.join(LOG_FOLDER, folder))]
    all_folders.sort(key=lambda x: os.path.getctime(os.path.join(LOG_FOLDER, x)))
    print(f"Found {len(all_folders)} folders:")
    # Find the latest folder with non-empty checkpoints.
    latest_folder_with_checkpoints = None
    for folder in reversed(all_folders):  # Start from newest
        checkpoint_path = os.path.join(LOG_FOLDER, folder, 'checkpoints')
        if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
            latest_folder_with_checkpoints = folder
            break

    if latest_folder_with_checkpoints is None:
        print("No folders with checkpoints found!")
        exit()

    print(f"Using folder with checkpoints: {latest_folder_with_checkpoints}")
    latest_folder = latest_folder_with_checkpoints

    agent.load(os.path.join(LOG_FOLDER, latest_folder, 'checkpoints', 'best_agent.pt'))
    agent.set_mode("eval")
    run(agent, env)
else:
    run(agent, env)

