import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env

import json
from datetime import datetime


# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import torch
import torch.nn as nn

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# define shared model (stochastic and deterministic models) using mixins
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



render = "human"
DT = 0.05
hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    render_mode = "human" if render else "rgb_array"
    env = AntEnv(xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
                render_mode="human",
                dt=DT,
                )
else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                    render_mode='human',
                    dt=DT,
                    )

LOG_FOLDER = 'logs_ppo_skrl'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from gymnasium.wrappers import TimeLimit
# env = TimeLimit(env, max_episode_steps=1000)
env = wrap_env(env)

memory = RandomMemory(memory_size=2048, num_envs=env.num_envs, device=device)

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model

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

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


train = False
if train == True:
    # Start training.
    print('Training...')

    # Configure and instantiate the RL trainer. 
    time_in_hours = 10 # 10 hours
    total_timesteps = int(time_in_hours * 3600 / DT)

    # Record every 30 minutes.
    cfg["experiment"]["checkpoint_interval"] = int(30 * 60 / DT)

    print(f"Training for {total_timesteps} timesteps")
    cfg_trainer = {"timesteps": total_timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    print('Starting evaluation...')
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

    path = os.path.join(LOG_FOLDER, latest_folder, 'checkpoints', 'best_agent.pt')
    agent.load(path)
    cfg_trainer = {"timesteps": 1000, "headless": False}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.eval()