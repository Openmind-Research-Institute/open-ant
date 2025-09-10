import os
import sys
import json
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env

# Set seed for reproducibility
set_seed(42)

# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# Env setup
render = "human"
DT = 0.05
hw_config = sys.argv[1] if len(sys.argv) > 1 else None

if hw_config is None:
    env_id = 'ant_mujoco'
    # env = AntEnv(
    #     render_mode=render,
    #     dt=DT,
    #     forward_reward_weight=1.0,
    #     ctrl_cost_weight=0.0,
    #     reward_upside_down_weight=0.0
    # )
    import gymnasium as gym
    env = gym.make('Ant-v5')

else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode=render, dt=DT)

# Wrap and prepare
env = wrap_env(env)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
LOG_FOLDER = 'logs_sac_skrl'
os.makedirs(LOG_FOLDER, exist_ok=True)

# Models
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Memory
memory = RandomMemory(memory_size=15625, device=device)

# Config
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["experiment"]["directory"] = LOG_FOLDER

# Agent
agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Train or Eval
train = True

if train:
    print("Training...")

    time_in_hours = 2
    total_timesteps = int(time_in_hours * 3600 / DT)
    total_timesteps = 160000
    cfg["experiment"]["checkpoint_interval"] = int(30 * 60 / DT)

    trainer = SequentialTrainer(cfg={"timesteps": total_timesteps, "headless": True}, env=env, agents=agent)
    trainer.train()

else:
    print("Starting evaluation...")

    # Find latest checkpoint
    all_folders = sorted(
        [f for f in os.listdir(LOG_FOLDER) if os.path.isdir(os.path.join(LOG_FOLDER, f))],
        key=lambda x: os.path.getctime(os.path.join(LOG_FOLDER, x)),
        reverse=True
    )

    latest_folder = None
    for folder in all_folders:
        checkpoint_path = os.path.join(LOG_FOLDER, folder, 'checkpoints')
        if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
            latest_folder = folder
            break

    if latest_folder is None:
        print("No valid checkpoint folder found.")
        exit()

    print(f"Using checkpoint from: {latest_folder}")
    agent.load(os.path.join(LOG_FOLDER, latest_folder, "checkpoints", "best_agent.pt"))

    trainer = SequentialTrainer(cfg={"timesteps": 1000, "headless": False}, env=env, agents=agent)
    trainer.eval()
