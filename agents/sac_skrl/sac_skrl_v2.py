import os
import sys
import json
from datetime import datetime

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

import gymnasium as gym

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

# Set seed for reproducibility
set_seed(42)

# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
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

def run(agent, env):
    obs, info = env.reset()
    i = 0
    action_list = []
    while i < total_timesteps:
        agent.pre_interaction(i, -1)
        with torch.no_grad():
            action = agent.act(obs, i, -1)[0]
            action_list.append(action.cpu().numpy().flatten())
            next_obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            agent.record_transition(states=obs,
                                    actions=action,
                                    rewards=reward,
                                    next_states=next_obs,
                                    terminated=terminated,
                                    truncated=truncated,
                                    infos=info,
                                    timestep=i,
                                    timesteps=total_timesteps)

        agent.post_interaction(timestep=i, timesteps=total_timesteps)
        i += 1

        if terminated or truncated:
            with torch.no_grad():
                obs, info = env.reset()
        else:
            obs = next_obs

        average_reward_per_second = reward_tracker.update(reward.item())
        if i % 1000 == 0:
            reward_tracker.log(i, average_reward_per_second)


# Env setup.
render = "human"
DT = 0.05
hw_config = sys.argv[1] if len(sys.argv) > 1 else None

if hw_config is None:
    env_id = 'ant_mujoco'
    # env = AntEnv(
    #     # render_mode=render,
    #     dt=DT,
    #     forward_reward_weight=1.0,
    #     ctrl_cost_weight=0.0,
    #     reward_upside_down_weight=0.0
    # )
    env = gym.make('Ant-v5')

else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode=render, dt=DT)

# Wrap and prepare.
env = wrap_env(env)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging.
LOG_FOLDER = 'logs_sac_skrl'
DATE_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(LOG_FOLDER, exist_ok=True)

# Models.
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Memory.
memory = RandomMemory(memory_size=1_000_000, device=device)

# Config.
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 64
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["directory"] = LOG_FOLDER
cfg["experiment"]["experiment_name"] = f"sac_skrl_{env_id}_{DATE_NOW}"
cfg["experiment"]["checkpoint_interval"] = int(1 * 60 / DT)

# Agent.
agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

reward_tracker = RewardTracker(env_dt=env.dt, env_id=env_id,
                               log_folder=os.path.join(cfg["experiment"]["directory"], cfg["experiment"]["experiment_name"]))

train = True
train_step_by_step = True
total_timesteps = 1_000_000

if train:
    print("Training...")
    if train_step_by_step:
        print("Training step by step...")
        agent.init()
        run(agent, env)

    if train_step_by_step == False:
        trainer = SequentialTrainer(cfg={"timesteps": total_timesteps, "headless": True}, env=env, agents=agent)
        trainer.train()

else:
    print("Evaluating...")
    folder_path = ''
    agent.load(folder_path)
    agent.set_mode("eval")
    run(agent, env)
