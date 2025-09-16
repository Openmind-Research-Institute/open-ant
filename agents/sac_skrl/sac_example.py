
import torch
import torch.nn as nn

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.vector import NormalizeObservation as VectorNormalizeObservation
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sim import ant_mujoco  # this will execute the register() if it's in ant_mujoco.py

# Models.
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


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--terminate_when_upside_down', type=bool, default=False)
parser.add_argument('--upside_down_cost_weight', type=float, default=0.0)
parser.add_argument('--ctrl_cost_weight', type=float, default=0.0)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--nb_envs', type=int, default=1)
parser.add_argument('--total_timesteps_train', type=int, default=100_000)
parser.add_argument('--total_timesteps_eval', type=int, default=100_000)
parser.add_argument('--weight_folder', type=str, default='sac_skrl_ant_mujoco_2025-09-15_23-31-02')
args = parser.parse_args()

set_seed(args.seed)

DT = 0.05
if args.nb_envs == 1:
    env_id = 'ant_mujoco_1'
    env = gym.make("CustomAnt-v0",
                   dt=DT,
                   render_mode=args.render_mode,
                   cost_upside_down_weight=args.upside_down_cost_weight,
                   terminate_on_upside_down=args.terminate_when_upside_down,
                   ctrl_cost_weight=args.ctrl_cost_weight,
                   )
    env = NormalizeObservation(env)
else:
    env_id = f'ant_mujoco_nb_envs_{args.nb_envs}'
    env = gym.make_vec("CustomAnt-v0",
                    num_envs=args.nb_envs,
                    dt=DT,
                    render_mode=args.render_mode,
                    cost_upside_down_weight=args.upside_down_cost_weight,
                    terminate_on_upside_down=args.terminate_when_upside_down,
                    ctrl_cost_weight=args.ctrl_cost_weight,
                    )
    env = VectorNormalizeObservation(env)

# Logging.
LOG_FOLDER = 'logs_sac_skrl'
experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SAC"
os.makedirs(os.path.join(LOG_FOLDER, experiment_name), exist_ok=True)

if args.train:
    # Save the config.
    with open(os.path.join(LOG_FOLDER, experiment_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

if args.nb_envs == 1 and args.render_mode == 'rgb_array':
    print(f"Recording video in {os.path.join(LOG_FOLDER, experiment_name)}")
    trigger = lambda t: t % 100 == 0
    env = RecordVideo(env, video_folder=os.path.join(LOG_FOLDER, experiment_name), episode_trigger=trigger, disable_logger=True, video_length=500)

env = wrap_env(env)
device = env.device

buffer_size = int(1_000_000/args.nb_envs)
memory = RandomMemory(memory_size=buffer_size, num_envs=env.num_envs, device=device)

models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

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
if args.train == True:
    cfg["random_timesteps"] = 80
    cfg["learning_starts"] = 80
else:
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = args.total_timesteps_eval
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 4000
if args.train:
    cfg["experiment"]["experiment_name"] = f"train_{env_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    print(f"Training {env_id}...")
else:
    cfg["experiment"]["experiment_name"] = f"eval_{env_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    print(f"Evaluating {env_id}...")
cfg["experiment"]["directory"] = LOG_FOLDER

agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Training or evaluation.
if args.train:
    cfg_trainer = {"timesteps": args.total_timesteps_train, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    cfg_trainer = {"timesteps": args.total_timesteps_eval, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    path = f'logs_sac_skrl/{args.weight_folder}/checkpoints'
    # Sorted by the number in the file name.
    files = sorted(
        [f for f in os.listdir(path) if f != "best_agent.pt"],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    for file in files:
        print(f"Loading checkpoint {file}")
        agent.load(f'{path}/{file}')
        trainer.eval()
    agent.load(f'{path}/best_agent.pt')
    trainer.eval()
