# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)

import os

import csv
import sys
import json
import time
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
import torch.nn.functional as F

# Import custom modules.
from torchrl.data import ReplayBuffer, LazyTensorStorage, RandomSampler
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

class SoftQNetwork(nn.Module):
    def __init__(self, env, use_layer_norm=False, cond_dim=0):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.cond_dim = cond_dim  # >0 when the critic is conditioned on the discount (meta-gradient gamma).
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + cond_dim,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)

    def forward(self, x, a, c=None):
        if c is not None:
            x = torch.cat([x, a, c], 1)
        else:
            x = torch.cat([x, a], 1)
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)

        # Action rescaling.
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1)).
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
    """Create the vectorized environment outside the SAC class."""
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
                # env.metadata['render_fps'] = 1/args.dt

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


class SAC:
    def __init__(self,
                 envs: gym.vector.SyncVectorEnv,
                 device: torch.device,
                 seed: int,
                 q_lr: float,
                 alpha_lr: float,
                 policy_lr: float,
                 autotune: bool,
                 alpha: float,
                 buffer_size: int,
                 batch_size: int,
                 learning_starts: int,
                 policy_frequency: int,
                 target_network_frequency: int,
                 tau: float,
                 gamma: float,
                 use_layer_norm: bool,
                 dt: float,
                 torch_deterministic: bool = True,
                 record_infos_sac = True,
                 # Meta-gradient discount (Xu, van Hasselt & Silver, 2018), critic-only, greedy (mu = 0).
                 meta_gamma: bool = False,
                 gamma_ref: float = None,
                 gamma_min: float = 0.5,
                 gamma_max: float = 0.99,
                 meta_lr: float = 1e-3,
                 meta_frequency: int = 1,
                 meta_batch_size: int = 256,
                 meta_recent_window: int = 10_000,
                 ):

        # Environment.
        self.envs = envs

        self.device = device
        print(f"[√] Using device: {self.device}")

        # Set seeds for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = torch_deterministic
        torch.backends.cudnn.benchmark = not torch_deterministic

        # Networks. With meta_gamma, the critics take one extra input: a normalized log-horizon feature of gamma.
        self.meta_gamma = meta_gamma
        cond_dim = 1 if meta_gamma else 0
        self.actor = Actor(self.envs, use_layer_norm=use_layer_norm).to(self.device)
        self.qf1 = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm, cond_dim=cond_dim).to(self.device)
        self.qf2 = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm, cond_dim=cond_dim).to(self.device)
        self.qf1_target = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm, cond_dim=cond_dim).to(self.device)
        self.qf2_target = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm, cond_dim=cond_dim).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_params = list(self.qf1.parameters()) + list(self.qf2.parameters())
        self.q_optimizer = optim.Adam(self.q_params, lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        self.learning_starts = learning_starts
        self.autotune = autotune
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.dt = dt

        self.record_infos_sac = record_infos_sac

        # Meta-gradient gamma.
        # Note: in this code gamma is a per-unit-time discount; the per-step discount is gamma ** dt.
        # gamma (self.gamma) is the initial value of the adapted discount; gamma_ref is the fixed
        # reference discount gamma' used in the meta-objective (defaults to the initial gamma).
        self.gamma_ref = gamma if gamma_ref is None else gamma_ref
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.meta_frequency = meta_frequency
        self.meta_batch_size = meta_batch_size
        self.meta_recent_window = meta_recent_window
        self.buffer_size = buffer_size
        self._num_added = 0  # Transitions written to the replay buffer (for recent-window sampling).
        self.last_meta_dot = None
        self._log_h_ref = -np.log(1.0 - self.gamma_ref ** self.dt)  # log per-step horizon at gamma_ref.
        if self.meta_gamma:
            assert gamma_min < gamma < gamma_max, "[!] Need gamma_min < gamma < gamma_max."
            assert gamma_min < self.gamma_ref < gamma_max, "[!] Need gamma_min < gamma_ref < gamma_max."
            # Bounded reparameterization: gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(psi).
            p = (gamma - gamma_min) / (gamma_max - gamma_min)
            psi0 = float(np.log(p / (1.0 - p)))
            self.psi = torch.tensor([psi0], requires_grad=True, device=self.device)
            self.meta_optimizer = optim.Adam([self.psi], lr=meta_lr)
        else:
            self.psi = None
            self.meta_optimizer = None

        # Alpha (entropy coefficient).
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.a_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.a_optimizer = None

        self.envs.single_observation_space.dtype = np.float32

        # Replay buffer.
        self.rb = ReplayBuffer(
                storage=LazyTensorStorage(buffer_size, device=device),
                sampler=RandomSampler(),
                batch_size=batch_size,
            )

        self.global_step = 0
        self.obs = None

    # ------------------------------------------------------------------ #
    # Meta-gradient gamma helpers.
    # ------------------------------------------------------------------ #
    def current_gamma(self) -> float:
        """Current (adapted) per-unit-time discount."""
        if not self.meta_gamma:
            return self.gamma
        sig = torch.sigmoid(self.psi.detach()).item()
        return self.gamma_min + (self.gamma_max - self.gamma_min) * sig

    def _gamma_feature(self, gamma: float, n: int) -> torch.Tensor:
        """Normalized log per-step horizon, log(1 / (1 - gamma**dt)), centred at gamma_ref."""
        log_h = -np.log(1.0 - gamma ** self.dt)
        feat = (log_h - self._log_h_ref) / 2.0
        return torch.full((n, 1), float(feat), dtype=torch.float32, device=self.device)

    def _q(self, net, obs, act, gamma: float) -> torch.Tensor:
        """Evaluate a critic, conditioned on gamma when meta_gamma is on."""
        if self.meta_gamma:
            return net(obs, act, self._gamma_feature(gamma, obs.shape[0]))
        return net(obs, act)

    def _soft_next_value(self, next_obs, next_actions, next_log_pi, gamma: float) -> torch.Tensor:
        """min_i Q_target_i(s', a'; gamma) - alpha * log pi(a'|s'), flattened."""
        q1 = self._q(self.qf1_target, next_obs, next_actions, gamma)
        q2 = self._q(self.qf2_target, next_obs, next_actions, gamma)
        return (torch.min(q1, q2) - self.alpha * next_log_pi).view(-1)

    def _td_target(self, data, soft_next: torch.Tensor, gamma: float) -> torch.Tensor:
        # See K. De Asis, R. Sutton, "An Idiosyncrasy of Time-discretization in Reinforcement Learning"
        not_done = 1 - data["terminations"].flatten()
        return data["rewards"].flatten() * self.dt + not_done * (gamma ** self.dt) * soft_next

    def _sample_meta_batch(self):
        """Fresh batch B' for the meta-objective, drawn from the most recent transitions."""
        n = len(self.rb)
        w = self.meta_recent_window
        if w is None or w <= 0 or n <= w:
            return self.rb.sample(self.meta_batch_size)
        pos = self._num_added % self.buffer_size  # Next write position in the circular storage.
        offsets = torch.randint(1, w + 1, (self.meta_batch_size,))
        idx = (pos - offsets) % self.buffer_size
        return self.rb[idx]

    def get_action(self, obs, evaluate=False):
        if self.obs is None:
            self.obs = obs
            return np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])

        # If we are in evaluation mode or we have reached the learning starts, get the action from the actor.
        if evaluate == True or self.global_step > self.learning_starts:
            actions, _, _ = self.actor.get_action(torch.Tensor(self.obs).to(self.device))
            actions = actions.detach().cpu().numpy()
            return actions

        # Otherwise, pick a random action.
        actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])

        return actions
 
    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):

        tensor_dict = TensorDict({
            "observations": torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            "next_observations": torch.as_tensor(next_obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "terminations": torch.as_tensor(terminations, dtype=torch.float32, device=self.device).unsqueeze(-1),                     # or use "terminated" + "truncated" separately if you prefer
        }, batch_size=[self.envs.num_envs])

        # Use .extend() instead of .add() when adding a batch of transitions.
        self.rb.extend(tensor_dict)
        self._num_added += self.envs.num_envs

        self.global_step += 1
        self.obs = next_obs

        qf1_loss = None
        qf2_loss = None
        alpha_loss = None
        actor_loss = None
        qf1_values_mean = None
        qf2_values_mean = None

        # Learning.
        if self.global_step > self.learning_starts:
            data, info_buffer = self.rb.sample(self.batch_size, return_info=True)
            gamma = self.current_gamma()
            do_meta = self.meta_gamma and (self.global_step % self.meta_frequency == 0)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(data["next_observations"])
                soft_next = self._soft_next_value(data["next_observations"], next_state_actions,
                                                  next_state_log_pi, gamma)
                next_q_value = self._td_target(data, soft_next, gamma)
                if self.meta_gamma:
                    # The critic is also trained at the reference discount, so Q(.; gamma_ref) is an
                    # accurate evaluator for the meta-objective.
                    soft_next_ref = self._soft_next_value(data["next_observations"], next_state_actions,
                                                          next_state_log_pi, self.gamma_ref)
                    next_q_value_ref = self._td_target(data, soft_next_ref, self.gamma_ref)

            # --- Meta step, part 1: h = d(omega')/d(gamma) direction, at the OLD critic omega. ---
            # Semi-gradient: d(target)/d(gamma) = (1 - done) * dt * gamma**(dt - 1) * V_soft(s'),
            # ignoring the gamma input of the (target) critic. Constant factors (2 * q_lr) are absorbed
            # into meta_lr.
            if do_meta:
                not_done = 1 - data["terminations"].flatten()
                dtarget_dgamma = not_done * self.dt * gamma ** (self.dt - 1) * soft_next
                q1_h = self._q(self.qf1, data["observations"], data["actions"], gamma).view(-1)
                q2_h = self._q(self.qf2, data["observations"], data["actions"], gamma).view(-1)
                h = torch.autograd.grad((dtarget_dgamma * (q1_h + q2_h)).mean(), self.q_params)

            qf1_a_values = self._q(self.qf1, data["observations"], data["actions"], gamma).view(-1)
            qf2_a_values = self._q(self.qf2, data["observations"], data["actions"], gamma).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            if self.meta_gamma:
                qf1_ref = self._q(self.qf1, data["observations"], data["actions"], self.gamma_ref).view(-1)
                qf2_ref = self._q(self.qf2, data["observations"], data["actions"], self.gamma_ref).view(-1)
                qf_loss = qf_loss + F.mse_loss(qf1_ref, next_q_value_ref) + F.mse_loss(qf2_ref, next_q_value_ref)
            qf1_values_mean = qf1_a_values.mean().item()
            qf2_values_mean = qf2_a_values.mean().item()

            # Optimize the Action-Value networks. After this step the critic parameters are omega'.
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            # --- Meta step, part 2: critic-only meta-objective J'(omega') at gamma_ref on a fresh batch. ---
            if do_meta:
                mdata = self._sample_meta_batch()
                with torch.no_grad():
                    m_next_a, m_next_log_pi, _ = self.actor.get_action(mdata["next_observations"])
                    m_soft_next = self._soft_next_value(mdata["next_observations"], m_next_a,
                                                        m_next_log_pi, self.gamma_ref)
                    y_ref = self._td_target(mdata, m_soft_next, self.gamma_ref)
                q1_m = self._q(self.qf1, mdata["observations"], mdata["actions"], self.gamma_ref).view(-1)
                q2_m = self._q(self.qf2, mdata["observations"], mdata["actions"], self.gamma_ref).view(-1)
                meta_loss = F.mse_loss(q1_m, y_ref) + F.mse_loss(q2_m, y_ref)
                g_prime = torch.autograd.grad(meta_loss, self.q_params)

                # dJ'/dgamma ~ g'^T h ; chain through the bounded reparameterization to psi.
                meta_dot = sum((gp * hh).sum() for gp, hh in zip(g_prime, h))
                sig = torch.sigmoid(self.psi.detach())
                dgamma_dpsi = (self.gamma_max - self.gamma_min) * sig * (1 - sig)
                self.meta_optimizer.zero_grad()
                self.psi.grad = (meta_dot * dgamma_dpsi).reshape_as(self.psi).detach()
                self.meta_optimizer.step()
                self.last_meta_dot = meta_dot.item()

            if self.global_step % self.policy_frequency == 0:
                for _ in range(
                        self.policy_frequency
                ):  # Compensate for the delay by doing 'actor_update_interval' instead of 1.
                    pi, log_pi, _ = self.actor.get_action(data["observations"])
                    qf1_pi = self._q(self.qf1, data["observations"], pi, gamma)
                    qf2_pi = self._q(self.qf2, data["observations"], pi, gamma)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    # Optimize the Actor network.
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(data["observations"])
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # Update the target networks.
            if self.global_step % self.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

        info_sac = None
        if self.record_infos_sac == True:
            info_sac = {
                'global_step': self.global_step,
                'qf1_loss': qf1_loss.item() if qf1_loss is not None else qf1_loss,
                'qf2_loss': qf2_loss.item() if qf2_loss is not None else qf2_loss,
                'qf1_values_mean': qf1_values_mean if qf1_values_mean is not None else qf1_values_mean,
                'qf2_values_mean': qf2_values_mean if qf2_values_mean is not None else qf2_values_mean,
                'actor_loss': actor_loss.item() if actor_loss is not None else actor_loss,
                'alpha': self.alpha,
                'alpha_loss': alpha_loss.item() if alpha_loss is not None else alpha_loss,
                'gamma': self.current_gamma(),
                'gamma_step': self.current_gamma() ** self.dt,
                'meta_dot': self.last_meta_dot,
                'rewards': rewards.flatten().tolist(),
                'original_rewards': infos['original_reward'].flatten().tolist(),
            }
        return info_sac

    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def get_replay_buffer(self):
        return self.rb

    def load_replay_buffer(self, replay_buffer_path):
        self.rb.loads(replay_buffer_path)
        self._num_added = len(self.rb)  # Assumes the loaded buffer has not wrapped around.

    def get_state(self):
        """Returns the full state of the agent including all network weights and optimizers."""
        state = {
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "global_step": self.global_step
        }
        if self.autotune:
            state["log_alpha"] = self.log_alpha.cpu().detach()
            state["alpha"] = self.alpha
            state["a_optimizer"] = self.a_optimizer.state_dict()
        if self.meta_gamma:
            state["psi"] = self.psi.cpu().detach()
            state["meta_optimizer"] = self.meta_optimizer.state_dict()
        return state

    def load_state(self, state_dict):
        """Loads the full state of the agent from a state dictionary."""
        self.actor.load_state_dict(state_dict["actor"])
        self.qf1.load_state_dict(state_dict["qf1"])
        self.qf2.load_state_dict(state_dict["qf2"])
        self.qf1_target.load_state_dict(state_dict["qf1_target"])
        self.qf2_target.load_state_dict(state_dict["qf2_target"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.global_step = state_dict.get("global_step", 0)
        if self.autotune and "log_alpha" in state_dict:
            log_alpha_tensor = state_dict["log_alpha"].to(self.device)
            if not log_alpha_tensor.requires_grad:
                log_alpha_tensor.requires_grad_(True)
            self.log_alpha.data = log_alpha_tensor
            if "alpha" in state_dict:
                self.alpha = state_dict["alpha"]
            else:
                self.alpha = self.log_alpha.exp().item()
            if "a_optimizer" in state_dict and self.a_optimizer is not None:
                self.a_optimizer.load_state_dict(state_dict["a_optimizer"])
        if self.meta_gamma and "psi" in state_dict:
            self.psi.data.copy_(state_dict["psi"].to(self.device))
            if "meta_optimizer" in state_dict:
                self.meta_optimizer.load_state_dict(state_dict["meta_optimizer"])


# For logging.
def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x

def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument("--exp_name", type=str, default="sac_ant",
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
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="replay buffer size")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--learning_starts", type=int, default=2000,
                        help="timestep to start learning")
    parser.add_argument("--policy_lr", type=float, default=3e-4,
                        help="policy learning rate")
    parser.add_argument("--q_lr", type=float, default=1e-3,
                        help="Q-network learning rate")
    parser.add_argument("--alpha_lr", type=float, default=1e-3,
                        help="alpha learning rate")
    parser.add_argument("--policy_frequency", type=int, default=2,
                        help="policy update frequency")
    parser.add_argument("--target_network_frequency", type=int, default=1,
                        help="target network update frequency")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="entropy regularization coefficient")
    parser.add_argument("--no-autotune", action="store_false",
                        dest="autotune",
                        help="disable automatic entropy tuning")
    parser.add_argument("--gamma", type=float, default=0.92,
                        help="discount factor")
    # Meta-gradient discount.
    parser.add_argument("--meta_gamma", action="store_true", default=False,
                        help="adapt gamma online with a critic-only meta-gradient (--gamma is the initial value)")
    parser.add_argument("--gamma_ref", type=float, default=None,
                        help="fixed reference discount gamma' for the meta-objective (default: --gamma)")
    parser.add_argument("--gamma_min", type=float, default=0.5,
                        help="lower bound on the adapted (per-unit-time) gamma")
    parser.add_argument("--gamma_max", type=float, default=0.99,
                        help="upper bound on the adapted (per-unit-time) gamma")
    parser.add_argument("--meta_lr", type=float, default=1e-3,
                        help="Adam learning rate for the gamma meta-parameter psi")
    parser.add_argument("--meta_frequency", type=int, default=1,
                        help="do a meta-update every n learning steps")
    parser.add_argument("--meta_batch_size", type=int, default=256,
                        help="batch size of the meta (validation) batch B'")
    parser.add_argument("--meta_recent_window", type=int, default=10_000,
                        help="draw B' from the most recent n transitions (<=0: whole buffer)")
    parser.add_argument("--no-use_layer_norm", action="store_false",
                        dest="use_layer_norm",
                        help="disable layer normalization in networks")

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
        autotune=True,
        use_layer_norm=True,
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

    # Create SAC agent.
    agent = SAC(envs=envs,
                device=device,
                q_lr=args.q_lr,
                alpha_lr=args.alpha_lr,
                policy_lr=args.policy_lr,
                autotune=args.autotune,
                alpha=args.alpha,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                policy_frequency=args.policy_frequency,
                target_network_frequency=args.target_network_frequency,
                tau=args.tau,
                gamma=args.gamma,
                use_layer_norm=args.use_layer_norm,
                seed=args.seed,
                dt=args.dt,
                meta_gamma=args.meta_gamma,
                gamma_ref=args.gamma_ref,
                gamma_min=args.gamma_min,
                gamma_max=args.gamma_max,
                meta_lr=args.meta_lr,
                meta_frequency=args.meta_frequency,
                meta_batch_size=args.meta_batch_size,
                meta_recent_window=args.meta_recent_window)

    step = 0
    # Load the model.
    if args.weights_path is not None:
        state = torch.load(os.path.join(args.weights_path, f"weights.pth"))
        agent.load_state(state)
        step = state["global_step"]
        agent.load_replay_buffer(os.path.join(args.weights_path, f"replay_buffer"))

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

    info_sac_logs = []
    info_sac = None

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
            info_sac = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)

        reward_tracker.update(infos['original_reward'][0])

        if info_sac is not None:
            info_sac_logs.append(info_sac)

        if any(truncations) or any(terminations):
            envs.reset()

        # Save the model.
        if step % args.save_every_n_steps == 0:
            state = agent.get_state()
            torch.save(state, os.path.join(args.runs_directory, run_name, f"weights.pth"))

            replay_buffer = agent.get_replay_buffer()
            replay_buffer.dumps(os.path.join(args.runs_directory, run_name, f"replay_buffer"))
            # Reward tracker.
            reward_tracker.log()

            # Save to csv.
            df_info_sac_logs = pd.DataFrame(info_sac_logs)
            df_info_sac_logs.to_csv(os.path.join(args.runs_directory, run_name, "info_sac_logs.csv"), index=False)