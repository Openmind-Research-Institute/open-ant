
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import sys
import argparse
import numpy as np
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.vector import NormalizeObservation as VectorNormalizeObservation


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sim import ant_mujoco  # this will execute the register() if it's in ant_mujoco.py

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
# from ant_mujoco import AntEnv

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

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--terminate_when_upside_down', type=bool, default=False)
parser.add_argument('--upside_down_cost_weight', type=float, default=0.0)
parser.add_argument('--ctrl_cost_weight', type=float, default=0.0)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--nb_envs', type=int, default=1)
args = parser.parse_args()

# env = gym.make_vec("Ant-v5", num_envs=NB_ENVS)
DT = 0.05
if args.nb_envs == 1:
    env = gym.make("CustomAnt-v0",
                   dt=DT,
                #    render_mode=args.render_mode,
                   cost_upside_down_weight=args.upside_down_cost_weight,
                   terminate_on_upside_down=args.terminate_when_upside_down,
                   ctrl_cost_weight=args.ctrl_cost_weight,
                   )
    env = NormalizeObservation(env)
else:
    env = gym.make_vec("CustomAnt-v0",
                    num_envs=args.nb_envs,
                    dt=DT,
                    render_mode=args.render_mode,
                    cost_upside_down_weight=args.upside_down_cost_weight,
                    terminate_on_upside_down=args.terminate_when_upside_down,
                    ctrl_cost_weight=args.ctrl_cost_weight,
                    )
    env = VectorNormalizeObservation(env)

# Logging
LOG_FOLDER = 'logs_sac_skrl'
from datetime import datetime
import json
experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SAC"
os.makedirs(os.path.join(LOG_FOLDER, experiment_name), exist_ok=True)

# Save the config.
with open(os.path.join(LOG_FOLDER, experiment_name, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

set_seed(args.seed)

if args.nb_envs == 1 and args.render_mode == 'rgb_array':
    print(f"Recording video in {os.path.join(LOG_FOLDER, experiment_name)}")
    trigger = lambda t: t % 100 == 0
    env = RecordVideo(env, video_folder=os.path.join(LOG_FOLDER, experiment_name), episode_trigger=trigger, disable_logger=True, video_length=500)

env = wrap_env(env)
device = env.device

# instantiate a memory as experience replay
buffer_size = int(1_000_000/args.nb_envs)
memory = RandomMemory(memory_size=buffer_size, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
if args.nb_envs == 1:
    cfg["batch_size"] = 256
else:
    cfg["batch_size"] = 64 * args.nb_envs
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80*args.nb_envs
cfg["learning_starts"] = 80*args.nb_envs
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 4000
cfg["experiment"]["directory"] = LOG_FOLDER
cfg["experiment"]["experiment_name"] = experiment_name

agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 200_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
train = args.train
if train:
    trainer.train()
else:
    path = '.'
    agent.load(path)
    trainer.eval()
