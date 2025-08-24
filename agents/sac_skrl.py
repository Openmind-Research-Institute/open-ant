from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.envs.wrappers.torch import wrap_env
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env
from collections import deque
import pandas as pd
from datetime import datetime

class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}



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

DATE_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FOLDER = 'logs_sac_skrl'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

if len(sys.argv) > 1:
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode='human', dt=0.05)
else:
    env = AntEnv(
        # render_mode="human",
                 dt=0.05)

env = wrap_env(env, "gymnasium")
device = env.device
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/sac/"

memory = RandomMemory(memory_size=20000, device=device, replacement=False)

agent = SAC(models=models,
            memory=memory,  # only required during training
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.init()

run(agent, env)
