
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env


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


def run(agent, env):
    obs, info = env.reset()
    i = 0
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

def run_eval(agent, env):
    obs, info = env.reset()
    i = 0
    episode_return = 0
    while True:
        with torch.no_grad():
            obs_torch = torch.from_numpy(obs).float().to(agent.device)
            action = agent.act(obs_torch, i, -1)[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        obs = next_obs
        i += 1
        if terminated or truncated:
            obs, info = env.reset()
            print(f"Episode return: {episode_return}")
            episode_return = 0


if len(sys.argv) > 1:
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, dt=0.05)
else:
    env = AntEnv(dt=0.05)
    # import gymnasium as gym
    # env = gym.make('Ant-v5', render_mode='human')

env = wrap_env(env, "gymnasium")
device = env.device

memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]


cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 2  # 16 * 4096 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 40
cfg["experiment"]["checkpoint_interval"] = 400
cfg["experiment"]["directory"] = "runs/ppo/Ant"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


agent.init()
TRAIN = False
if TRAIN == True:
    run(agent, env)
else:
    env = AntEnv(dt=0.05, render_mode='human')
    main_folder = 'runs/ppo/Ant/25-08-07_10-47-16-865633_PPO'
    agent.load(os.path.join(main_folder, 'checkpoints', 'best_agent.pt'))
    run_eval(agent, env)