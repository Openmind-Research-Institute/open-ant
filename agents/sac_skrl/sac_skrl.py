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
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker


# Define models.
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


# Main training loop.
def run(agent, env, total_timesteps):
    obs, info = env.reset()
    i = 0
    action_list = []
    for i in tqdm(range(total_timesteps)):
        agent.pre_interaction(i, -1)
        with torch.no_grad():
            action = agent.act(obs, i, -1)[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
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
        action_list.append(action.cpu().numpy().flatten())

        reward_tracker.update(reward.item())
        agent.track_data("average_reward_per_second", reward_tracker.average_reward_per_second)
        reward_tracker.log(every_N_steps=1000, plot=True)

        # if i % 100 == 0:
        #     print('Saving actions histogram...')
        #     action_np = np.array(action_list)
        #     # Make histogram of actions.
        #     fig, ax = plt.subplots(2, 4, figsize=(10, 5))
        #     ax = ax.flatten()
        #     for idx_action in range(action_np.shape[1]):
        #         ax[idx_action].hist(action_np[:, idx_action], color='black', bins=20)
        #         ax[idx_action].set_title(f"Action {idx_action}")
        #     plt.tight_layout()
            # plt.savefig(os.path.join(cfg["experiment"]["directory"],
            #                         cfg["experiment"]["experiment_name"],
            #                         f"actions_histogram_{int(i)}.png"))
            # print('Plot saved.')
            # print(f"actions_histogram_{int(i)}.png")
        if terminated or truncated:
            print(f"Terminated or truncated at timestep {i}")
            with torch.no_grad():
                obs, info = env.reset()
        else:
            obs = next_obs

    return action_list

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dt', type=float, default=0.05)
parser.add_argument('--terminate_when_upside_down', type=bool, default=True)
parser.add_argument('--upside_down_cost_weight', type=float, default=0.0)
parser.add_argument('--ctrl_cost_weight', type=float, default=0.0)
parser.add_argument('--render_mode', type=str, default='rgb_array')
parser.add_argument('--hw_config', type=str, default=None)
parser.add_argument('--total_timesteps_train', type=int, default=150_000)
parser.add_argument('--total_timesteps_eval', type=int, default=150_000)
parser.add_argument('--weight_init', type=str, default='random')
parser.add_argument('--weight_folder', type=str, default=None)

args = parser.parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

set_seed(args.seed)

# Env setup.
render = "human"
DT = args.dt

if args.hw_config is None:
    env_id = 'ant_mujoco'
    env = AntEnv(
        dt=DT,
        render_mode=args.render_mode,
        cost_upside_down_weight=args.upside_down_cost_weight,
        terminate_on_upside_down=args.terminate_when_upside_down,
        ctrl_cost_weight=args.ctrl_cost_weight,
    )
else:
    env_id = 'ant_hw'
    with open(args.hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode=render, dt=DT)

env = wrap_env(env, wrapper="gymnasium")
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

# initialize models' parameters (weights and biases)
if args.weight_init == 'small':
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Memory.
memory = RandomMemory(memory_size=1_000_000, device=device)

# Config.
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
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
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["directory"] = LOG_FOLDER
if args.train:
    cfg["experiment"]["experiment_name"] = f"train_{env_id}_{DATE_NOW}"
    print(f"Training {env_id}...")
else:
    cfg["experiment"]["experiment_name"] = f"eval_{env_id}_{DATE_NOW}"
    print(f"Evaluating {env_id}...")
cfg["experiment"]["checkpoint_interval"] = 4000

# Agent.
agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

reward_tracker = RewardTracker(env_dt=env.dt, env_id=env_id,
                               log_folder=os.path.join(cfg["experiment"]["directory"],
                                                       cfg["experiment"]["experiment_name"]),
                               time_window=120.0)

def safe_json(o):
    if isinstance(o, type):          # classes, e.g. <class 'float'>
        return str(o)
    if hasattr(o, "dtype"):          # torch/numpy dtypes
        return str(o)
    if hasattr(o, "tolist"):         # numpy arrays
        return o.tolist()
    if hasattr(o, "__dict__"):       # custom objects
        return o.__dict__
    return str(o)

# Save config
with open(os.path.join(LOG_FOLDER, cfg["experiment"]["experiment_name"], "cfg.json"), "w") as f:
    json.dump(cfg, f, indent=4, default=safe_json)

# Save args
with open(os.path.join(LOG_FOLDER, cfg["experiment"]["experiment_name"], "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4, default=safe_json)

# Training or evaluation.
if args.train:
    print("Training...")
    agent.init()
    run(agent, env, args.total_timesteps_train)
else:
    print("Evaluating...")
    agent.init()
    path = f'logs_sac_skrl/{args.weight_folder}/checkpoints'
    files = sorted(
        [f for f in os.listdir(path) if f != "best_agent.pt"],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    print('path')
    for file in files:
        print(f"Loading checkpoint {file}")
        path_policy = f'{path}/{file}'
        agent.load(path_policy)
        agent.set_mode("eval")
        run(agent, env, args.total_timesteps_eval)
    agent.load(f'{path}/best_agent.pt')
    agent.set_mode("eval")
    run(agent, env, args.total_timesteps_eval)
