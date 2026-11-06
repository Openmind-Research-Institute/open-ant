# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)
# Training loop structure follows agents/sac/sac_cleanrl.py.

import os

import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Import custom modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
    """Create the vectorized environment outside the PPO class."""
    def make_env(seed, idx, capture_video, run_name):
        def _init():
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(30),
                'knee_range': np.radians(20),
            }
            if args.hw_config is None:
                env = AntEnv(
                    control_dt=args.dt,
                    render_mode=args.render_mode,
                    terminate_on_upside_down=args.terminate_on_upside_down,
                    task=task,
                    joint_config=joint_config,
                    model_path=os.path.join(os.path.dirname(__file__), args.model_path),
                )
            else:
                with open(args.hw_config, 'r') as f:
                    cfg = json.load(f)
                env = make_ant_env(cfg, render_mode=args.render_mode,
                                   dt=args.dt,
                                   joint_config=joint_config,
                                   task=task,
                                   )

            if capture_video and idx == 0:
                print('RecordVideo')
                env = gym.wrappers.RecordVideo(env, os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
                                               step_trigger=lambda x: x % args.save_every_n_steps == 0, video_length=args.save_every_n_steps)
            env.action_space.seed(seed)
            # Reward scaling.
            env = gym.wrappers.TransformReward(env, lambda reward: reward * args.reward_scale)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
    print(f"[√] Created environment with {envs.num_envs} environments.")
    return envs


class PPO:
    def __init__(self,
                 envs: gym.vector.SyncVectorEnv,
                 device: torch.device,
                 seed: int,
                 learning_rate: float,
                 num_steps: int,
                 total_timesteps: int,
                 anneal_lr: bool,
                 gamma: float,
                 gae_lambda: float,
                 num_minibatches: int,
                 update_epochs: int,
                 norm_adv: bool,
                 clip_coef: float,
                 clip_vloss: bool,
                 ent_coef: float,
                 vf_coef: float,
                 max_grad_norm: float,
                 target_kl: float,
                 dt: float,
                 torch_deterministic: bool = True,
                 record_infos_ppo: bool = True,
                 ):

        # Environment.
        self.envs = envs
        self.num_envs = envs.num_envs

        self.device = device
        print(f"[√] Using device: {self.device}")

        # Set seeds for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = torch_deterministic
        torch.backends.cudnn.benchmark = not torch_deterministic

        # Networks.
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.dt = dt

        self.batch_size = int(self.num_envs * self.num_steps)
        if self.batch_size % num_minibatches != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by num_minibatches ({num_minibatches})."
            )
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches

        self.record_infos_ppo = record_infos_ppo

        self.envs.single_observation_space.dtype = np.float32

        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape
        self.obs_buf = torch.zeros((self.num_steps, self.num_envs) + obs_shape, device=self.device)
        self.actions_buf = torch.zeros((self.num_steps, self.num_envs) + act_shape, device=self.device)
        self.logprobs_buf = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.rewards_buf = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.terminations_buf = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.values_buf = torch.zeros((self.num_steps, self.num_envs), device=self.device)

        self.global_step = 0
        self.rollout_step = 0
        self.obs = None
        self._last_logprob = None
        self._last_value = None

    def get_action(self, obs, evaluate=False):
        if self.obs is None:
            self.obs = obs

        obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if evaluate:
                action = self.agent.actor_mean(obs_t)
                self._last_logprob = None
                self._last_value = None
            else:
                action, logprob, _, value = self.agent.get_action_and_value(obs_t)
                self._last_logprob = logprob
                self._last_value = value.reshape(-1)
        return action.detach().cpu().numpy()

    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):
        self.obs_buf[self.rollout_step] = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        self.actions_buf[self.rollout_step] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.logprobs_buf[self.rollout_step] = self._last_logprob
        self.rewards_buf[self.rollout_step] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1)
        # Truncation (out of bounds) still bootstraps. Only true terminations cut the return.
        self.terminations_buf[self.rollout_step] = torch.as_tensor(terminations, dtype=torch.float32, device=self.device).reshape(-1)
        self.values_buf[self.rollout_step] = self._last_value
        del truncations

        self.global_step += 1
        self.obs = next_obs
        self.rollout_step += 1

        metrics = None
        if self.rollout_step == self.num_steps:
            metrics = self._update()
            self.rollout_step = 0

        info_ppo = None
        if self.record_infos_ppo:
            info_ppo = {
                'global_step': self.global_step,
                'policy_loss': None if metrics is None else metrics['policy_loss'],
                'value_loss': None if metrics is None else metrics['value_loss'],
                'entropy': None if metrics is None else metrics['entropy'],
                'approx_kl': None if metrics is None else metrics['approx_kl'],
                'old_approx_kl': None if metrics is None else metrics['old_approx_kl'],
                'clipfrac': None if metrics is None else metrics['clipfrac'],
                'explained_variance': None if metrics is None else metrics['explained_variance'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'rewards': np.asarray(rewards).flatten().tolist(),
                'original_rewards': np.asarray(infos['original_reward']).flatten().tolist(),
            }
        return info_ppo

    def _update(self):
        # Anneal the learning rate across the run.
        if self.anneal_lr:
            frac = 1.0 - (self.global_step - 1.0) / self.total_timesteps
            frac = float(np.clip(frac, 0.0, 1.0))
            self.optimizer.param_groups[0]['lr'] = frac * self.learning_rate

        # Bootstrap value if not done.
        # Discount per control step is gamma**dt, and the reward is scaled by dt.
        # See K. De Asis, R. Sutton, "An Idiosyncrasy of Time-discretization in Reinforcement Learning".
        discount = self.gamma ** self.dt
        with torch.no_grad():
            next_obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
            next_value = self.agent.get_value(next_obs_t).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards_buf)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                nextnonterminal = 1.0 - self.terminations_buf[t]
                nextvalues = next_value if t == self.num_steps - 1 else self.values_buf[t + 1]
                delta = self.rewards_buf[t] * self.dt + discount * nextvalues * nextnonterminal - self.values_buf[t]
                advantages[t] = lastgaelam = delta + discount * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values_buf

        # Flatten the batch.
        b_obs = self.obs_buf.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs_buf.reshape(-1)
        b_actions = self.actions_buf.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values_buf.reshape(-1)

        # Optimize the policy and value network.
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss.
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss.
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'old_approx_kl': old_approx_kl.item(),
            'clipfrac': float(np.mean(clipfracs)),
            'explained_variance': float(explained_var),
        }

    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def get_state(self):
        """Returns the full state of the agent including network weights and the optimizer."""
        return {
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }

    def load_state(self, state_dict):
        """Loads the full state of the agent from a state dictionary."""
        self.agent.load_state_dict(state_dict["agent"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.global_step = state_dict.get("global_step", 0)


def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument("--exp_name", type=str, default="ppo_ant",
                        help="the name of this experiment")
    parser.add_argument("--runs_directory", type=str, default="runs",
                        help="the directory to save the runs in")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--no-torch_deterministic", action="store_false",
                        dest="torch_deterministic",
                        help="disable torch deterministic mode")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture_video", action="store_true",
                        help="capture video of agent performances")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="evaluate the agent")
    parser.add_argument("--save_every_n_steps", type=int, default=4000,
                        help="save every n steps")

    # Algorithm.
    parser.add_argument("--env_id", type=str, default="EAnt",
                        help="environment ID")
    parser.add_argument("--total_timesteps", type=int, default=60_000,
                        help="total training timesteps")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="number of parallel envs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="learning rate of the optimizer")
    parser.add_argument("--num_steps", type=int, default=2048,
                        help="steps to run in each environment per policy rollout")
    parser.add_argument("--no-anneal_lr", action="store_false",
                        dest="anneal_lr",
                        help="disable learning rate annealing")
    parser.add_argument("--gamma", type=float, default=0.92,
                        help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="lambda for general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=32,
                        help="number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=10,
                        help="epochs to update the policy")
    parser.add_argument("--no-norm_adv", action="store_false",
                        dest="norm_adv",
                        help="disable advantage normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="surrogate clipping coefficient")
    parser.add_argument("--no-clip_vloss", action="store_false",
                        dest="clip_vloss",
                        help="disable clipped value loss")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="maximum norm for gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="target KL divergence threshold")

    # Environment.
    parser.add_argument("--dt", type=float, default=0.12,
                        help="environment timestep")
    parser.add_argument("--hw_config", type=str, default=None,
                        help="hardware config file")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="render mode")
    parser.add_argument("--no-terminate_on_upside_down", action="store_false",
                        dest="terminate_on_upside_down",
                        help="do not terminate episode when upside down")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="load previous weights")
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"],
                        help="type of task")
    parser.add_argument("--radius_back_and_forth", type=float, default=0.3,
                        help="radius of the back and forth task")
    parser.add_argument("--origin_back_and_forth", type=float, nargs=2, default=[0.75, -0.3],
                        help="origin of the back and forth task")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                        help="reward scale factor")
    parser.add_argument("--model_path", type=str, default="../../sim/assets/ant_with_camera_after_sys_id.xml",
                        help="XML file to use for the environment")

    parser.set_defaults(
        torch_deterministic=True,
        anneal_lr=True,
        norm_adv=True,
        clip_vloss=True,
        terminate_on_upside_down=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Set up folders for environment creation.
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    disk_folder = ''
    os.makedirs(args.runs_directory, exist_ok=True)
    run_name = f"{args.exp_name}_{date}_seed_{args.seed}"
    os.makedirs(os.path.join(args.runs_directory, run_name), exist_ok=True)

    # Save the args.
    with open(os.path.join(args.runs_directory, run_name, "args.json"), "w") as f:
        json.dump(args.__dict__, f)

    # Create task.
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = args.radius_back_and_forth
        ORIGIN = np.array(args.origin_back_and_forth) # Measured in the environment.
        task = BackAndForthTask(
            radius=RADIUS,
            origin=ORIGIN,
        )
        print(f"BackAndForthTask initialized for radius: {RADIUS}, origin: {ORIGIN}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    # Create environment.
    envs = make_ant_envs(args=args,
                         task=task,
                         disk_folder=disk_folder,
                         run_name=run_name,
                         runs_directory=args.runs_directory)
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create PPO agent.
    agent = PPO(envs=envs,
                device=device,
                seed=args.seed,
                learning_rate=args.learning_rate,
                num_steps=args.num_steps,
                total_timesteps=args.total_timesteps,
                anneal_lr=args.anneal_lr,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                num_minibatches=args.num_minibatches,
                update_epochs=args.update_epochs,
                norm_adv=args.norm_adv,
                clip_coef=args.clip_coef,
                clip_vloss=args.clip_vloss,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                dt=args.dt)

    step = 0
    # Load the model.
    if args.weights_path is not None:
        state = torch.load(os.path.join(args.weights_path, "weights.pth"), weights_only=False)
        agent.load_state(state)
        step = state["global_step"]

    if args.eval:
        step = 0 # Reset the step to 0 when eval.

    agent.set_global_step(step)
    print(f"Step: {step}")

    # Reward tracker.
    env_dt = args.dt
    env_id = args.env_id
    reward_tracker = RewardTracker(env_dt=env_dt,
                                   env_id=env_id,
                                   time_window=120.0,
                                   log_folder=os.path.join(args.runs_directory, run_name),
                                   )

    info_ppo_logs = []
    info_ppo = None

    obs, info = envs.reset(seed=args.seed)

    for step in tqdm(range(step, args.total_timesteps), initial=step):

        # Get action.
        selected_actions = agent.get_action(obs, args.eval)

        # Step env.
        next_obs, rewards, terminations, truncations, infos = envs.step(selected_actions)
        if args.eval == True:
            agent.agent_step_eval(next_obs)
        else:
            # Learn.
            info_ppo = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)

        reward_tracker.update(infos['original_reward'][0])

        if info_ppo is not None:
            info_ppo_logs.append(info_ppo)

        # Gymnasium's vector env is waiting to autoreset. Reset now and hand the new
        # observation back to the agent so the next action is conditioned on it.
        if any(truncations) or any(terminations):
            reset_obs, _ = envs.reset()
            agent.obs = reset_obs

        # Save the model.
        if step % args.save_every_n_steps == 0:
            state = agent.get_state()
            torch.save(state, os.path.join(args.runs_directory, run_name, "weights.pth"))

            # Reward tracker.
            reward_tracker.log()

            # Save to csv.
            df_info_ppo_logs = pd.DataFrame(info_ppo_logs)
            df_info_ppo_logs.to_csv(os.path.join(args.runs_directory, run_name, "info_ppo_logs.csv"), index=False)

    envs.close()
