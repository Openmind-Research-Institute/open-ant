from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env
from collections import deque
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

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
    moving_reward_queue = deque(maxlen=window_size)

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
        moving_reward_queue.append(reward_per_second)
        if len(moving_reward_queue) > window_size:
            # Pop the leftmost element.
            moving_reward_queue.popleft()

        average_reward_per_second = sum(moving_reward_queue) / len(moving_reward_queue)
        if i % 1000 == 0:
            print(f"Step {i}, time [s] {i * env.dt:.2f}, time [min] {i * env.dt / 60:.2f}, moving average reward {average_reward_per_second:.4f}")
            df = pd.concat([df, pd.DataFrame({'step': [i], 'reward': [average_reward_per_second]})], ignore_index=True)
            df.to_csv(os.path.join(LOG_FOLDER, f'rewards_{DATE_NOW}.csv'), index=False)
            # Plot the reward curve.
            plt.plot(df['step'], df['reward'])
            plt.savefig(os.path.join(LOG_FOLDER, f'reward_curve.png'))
            plt.close()

DATE_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FOLDER = 'logs_sac_skrl'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

TRAIN = True

DT = 0.05
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode='human', dt=DT)
else:
    env = AntEnv(
        # render_mode="human",
                 dt=DT)

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
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
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
# logging to TensorBoard and write checkpoints (in timesteps)

# Configure and instantiate the RL trainer. 
time_in_hours = 10 # 10 hours
total_timesteps = int(time_in_hours * 3600 / DT)
# Record every 30 minutes.
cfg["experiment"]["checkpoint_interval"] = int(30 * 60 / DT)

memory = RandomMemory(memory_size=20000, device=device, replacement=False)

agent = SAC(models=models,
            memory=memory,  # only required during training
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.init()

if TRAIN == False:
    agent.load('/home/sorina/embodied-mujoco-ant/agents/logs_ppo_skrl_v2/25-08-23_19-01-31-418897_PPO/checkpoints/agent_85000.pt')
    agent.set_mode("eval")
    run(agent, env)
else:
    run(agent, env)
