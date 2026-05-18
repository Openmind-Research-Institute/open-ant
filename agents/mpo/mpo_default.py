import os
import csv
import sys
import copy
import json
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.buffers import ReplayBuffer, NStepReplayBufferSamples


def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x

class Actor(nn.Module):
    def __init__(self,
                 env,
                 hidden_dims: List[int] = [256, 256],
                 use_layer_norm: bool = False,
                 min_scale: float = 1e-3,
                 init_scale: float = 0.5,):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.min_scale = min_scale

        self.register_buffer("action_low",torch.tensor(env.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high",torch.tensor(env.single_action_space.high, dtype=torch.float32))
        
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_sigma_head = nn.Linear(prev, act_dim)

        self._softplus_bias = float(np.log(np.exp(init_scale - min_scale) - 1.0))

        self.apply(self._init_weights)
        with torch.no_grad():
            self.log_sigma_head.bias.fill_(self._softplus_bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dist.Independent:
        logits = self.net(x)
        mu = self.action_low + (self.action_high - self.action_low) * torch.sigmoid(self.mu_head(logits))
        sigma = F.softplus(self.log_sigma_head(logits)) + self.min_scale
        return dist.Independent(dist.Normal(mu, sigma), 1)

    def get_action(self, obs: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.forward(obs)
        actions = d.rsample((n_samples,))
        log_probs = d.log_prob(actions)

        actions = actions.permute(1, 0, 2)  # (batch, n_samples, act_dim)
        log_probs = log_probs.transpose(0, 1).unsqueeze(-1)
        return actions, log_probs, d.mean

    def get_log_probs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        d = self.forward(obs)
        return d.log_prob(action).unsqueeze(-1)


class Critic(nn.Module):
    def __init__(self, env, hidden_dims: List[int] = [256, 256], use_layer_norm: bool = False):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))

        layers = []
        prev = obs_dim + act_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
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
                                   dt=args.dt, joint_config=joint_config, task=task)
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
                    step_trigger=lambda x: x % args.save_every_n_steps == 0,
                    video_length=args.save_every_n_steps,
                )
            env.action_space.seed(seed)
            env = gym.wrappers.TransformReward(env, lambda r: r * args.reward_scale)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
    print(f"[√] Created environment with {envs.num_envs} environments.")
    return envs


class MPO:
    def __init__(self, args, envs, disk_folder='', run_name=None, runs_directory='runs'):
        self.args = args
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"[√] Using device: {self.device}")
        self.dt = self.args.dt

        self.disk_folder = disk_folder
        self.run_name = run_name
        self.runs_directory = runs_directory
        self.weights_folder = os.path.join(disk_folder, runs_directory, run_name, "weights_and_args")
        os.makedirs(self.weights_folder, exist_ok=True)
        with open(os.path.join(self.weights_folder, "args.json"), 'w') as f:
            json.dump(args.__dict__, f)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        torch.backends.cudnn.benchmark = not args.torch_deterministic

        hidden_dims = [args.hidden_dim] * args.n_hidden_layers

        self.actor = Actor(envs, hidden_dims=hidden_dims, use_layer_norm=args.use_layer_norm).to(self.device)
        self.actor_target = Actor(envs, hidden_dims=hidden_dims, use_layer_norm=args.use_layer_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False
        
        self.critics = nn.ModuleList([Critic(envs,
                                             hidden_dims=hidden_dims,
                                             use_layer_norm=args.use_layer_norm) for _ in range(self.args.ensemble)]).to(self.device)
        self.target_critics = copy.deepcopy(self.critics)
        for p in self.target_critics.parameters():
            p.requires_grad = False
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=args.q_lr) for c in self.critics]

        self.policy_learning_starts = args.policy_learning_starts

        # Dual variable for E-step temperature.
        self.log_eta = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_temp_optimizer = optim.Adam([self.log_eta], lr=args.dual_lr)

        # Scalar Lagrange multipliers for M-step KL constraints.
        self.alpha_mu = 1e-3 #if 0.0 first policy imporvement steps are unconstrained
        self.alpha_sigma = 1e-3

        self.envs.single_observation_space.dtype = np.float32
        self.action_low = envs.single_action_space.low.astype(np.float32)
        self.action_high = envs.single_action_space.high.astype(np.float32)
        assert np.all(np.isfinite(envs.single_action_space.low)) and np.all(np.isfinite(envs.single_action_space.high))
        self._uniform_log_prob = -float(np.sum(np.log(self.action_high - self.action_low)))

        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )

        self.global_step = 0
        self.obs = None
        self.last_actions = None
        self.last_log_probs = None

        # Logging state.
        self.start_time = None
        self.reward_tracker = None
        self.csv_file_info = None
        self.csv_file_agent_vars = None
        self.writer_info = None
        self.writer_agent_vars = None
        self.keys_info = None
        self.keys_agent_vars = [
            'loss_q', 'loss_p', 'mean_q', 'eta',
            'kl_mu', 'kl_sigma', 'alpha_mu', 'alpha_sigma',
            'utd_ratio', 'SPS', 'average_reward_per_second', 'reward',
        ]
        self.info_log_buffer = []
        self.agent_vars_buffer = []

    def get_action(self, obs, evaluate=False):
        act_dim = self.envs.single_action_space.shape[0]

        if self.obs is None:
            self.obs = obs
            rand_actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]) #Gymnasium uses uniform sampling for box --> if other than change logprob formula
            self.last_actions = rand_actions.copy()
            self.last_log_probs = np.full((self.envs.num_envs, 1), self._uniform_log_prob, dtype=np.float32)
            return rand_actions

        if evaluate or self.global_step > self.args.learning_starts:
            with torch.no_grad():
                obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
                actions, log_probs, _ = self.actor.get_action(obs_t)
                actions = actions.squeeze(1)     # (n_envs, act_dim)
                log_probs = log_probs.squeeze(1) # (n_envs, 1)
                self.last_actions = actions.cpu().numpy()
                self.last_log_probs = log_probs.cpu().numpy()
                return actions.cpu().numpy()

        rand_actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        self.last_actions = rand_actions.copy()
        self.last_log_probs = np.full((self.envs.num_envs, 1), self._uniform_log_prob, dtype=np.float32)
        return rand_actions

    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):
        # SyncVectorEnv auto-resets and stores the true final obs; hw env does not.
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx in range(self.envs.num_envs):
                if terminations[idx] or truncations[idx]:
                    real_next_obs[idx] = infos["final_observation"][idx]

        self.rb.add(self.obs, real_next_obs, actions, rewards, terminations,
                    [{}] * self.envs.num_envs,
                    behavior_log_prob=self.last_log_probs)
        self.global_step += 1
        self.obs = next_obs

        metrics = None
        if self.global_step > self.args.learning_starts and self.rb.size() >= self.args.batch_size:
            results = [self._learn() for _ in range(self.args.utd_ratio)]
            keys = results[0].keys()
            metrics = {k: sum(r[k] for r in results) / len(results) for k in keys}
            metrics['utd_ratio'] = self.args.utd_ratio
        return metrics

    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def _learn(self):
        if self.global_step >= self.args.learning_starts + self.policy_learning_starts:
            self.args.decouple_q_learning = False

        data = self.rb.sample_nstep(self.args.batch_size, self.args.td_horizon)

        # t0 = time.time()
        loss_q, mean_q = self._update_critic(data)
        # t_critic = time.time() - t0

        loss_p = kl_mu = kl_sigma = 0.0
        eta = self.log_eta.exp().detach()
        # t_estep = t_mstep = 0.0

        # Always run E-step so eta stays calibrated during Q warmup.
        # t0 = time.time()
        action_samples, weights, eta, b_mu, b_sigma = self._e_step(data)
        # t_estep = time.time() - t0

        if not self.args.decouple_q_learning:
            # t0 = time.time()
            loss_p, kl_mu, kl_sigma = self._m_step(data.observations, action_samples, weights, b_mu, b_sigma)
            # t_mstep = time.time() - t0

        self._update_targets()

        return {
            'loss_q': loss_q,
            'loss_p': loss_p,
            'mean_q': mean_q,
            'eta': eta.item(),
            'kl_mu': kl_mu,
            'kl_sigma': kl_sigma,
            'alpha_mu': self.alpha_mu,
            'alpha_sigma': self.alpha_sigma,
            # 't_critic': t_critic,
            # 't_estep': t_estep,
            # 't_mstep': t_mstep,
        }

    def _update_critic(self, data: NStepReplayBufferSamples) -> Tuple[float, float]:
        n = data.rewards.shape[1]
        gamma_dt = self.args.gamma ** self.dt

        with torch.no_grad():
            # Fixed critic subset for the entire Retrace computation — consistent targets.
            subset_idx = torch.randperm(len(self.critics), device=self.device)[:2]

            # Bootstrap from Q_target(s_0, a_0).
            y = self.aggregation_operator(
                data.observations, data.actions, self.target_critics,
                mode='min_subset', subset_size=2, subset_idx=subset_idx,
            )

            c_prod = torch.ones_like(y)   # running IS product (B, 1)
            alive = torch.ones_like(y)   # episode-still-alive mask (B, 1)
            prev_done = None

            for k in range(n):
                s_k = data.all_observations[:, k, :]       # (B, obs_dim)
                a_k = data.all_actions[:, k, :]            # (B, act_dim)
                r_k = data.rewards[:, k, :]                # (B, 1)
                done_k = data.dones[:, k, :]                  # (B, 1)
                s_kp1 = data.all_next_observations[:, k, :]  # (B, obs_dim)

                a_kp1, _, _ = self.actor_target.get_action(s_kp1)
                a_kp1 = a_kp1.squeeze(1)
                q_kp1 = self.aggregation_operator(
                    s_kp1, a_kp1, self.target_critics,
                    mode='min_subset', subset_size=2, subset_idx=subset_idx,
                )
                # k=0: reuse the initial y to avoid an extra forward pass.
                q_k = y if k == 0 else self.aggregation_operator(
                    s_k, a_k, self.target_critics,
                    mode='min_subset', subset_size=2, subset_idx=subset_idx,
                )

                delta_k = r_k * self.dt + gamma_dt * (1.0 - done_k) * q_kp1 - q_k

                if k > 0:
                    alive  = alive * (1.0 - prev_done)
                    actions_k  = data.all_actions[:, k, :]         # (B, act_dim)
                    log_mu = data.behavior_log_probs[:, k, :]  # (B, 1)
                    log_pi = self.actor_target.get_log_probs(s_k, actions_k)
                    c_k    = self.importance_sampling_coef(log_pi, log_mu)
                    c_prod = c_prod * c_k

                y = y + (gamma_dt ** k) * c_prod * alive * delta_k
                prev_done = done_k

        losses = []
        for critic, opt in zip(self.critics, self.critic_optimizers):
            q_value = critic(data.observations, data.actions)
            loss = F.mse_loss(q_value, y)

            opt.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(critic.parameters(), self.args.max_grad_norm)
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            mean_q = self.aggregation_operator(
                data.observations, data.actions, self.critics, mode='mean'
            ).mean().item()

        return float(np.mean(losses)), mean_q

    def aggregation_operator(self, state, action, critics, mode='mean', beta=1.0, subset_size=2, subset_idx=None):
        q_values = torch.stack([c(state, action) for c in critics], dim=0)
        if mode == 'mean':
            return q_values.mean(dim=0)
        elif mode == 'min_subset':
            if subset_idx is None:
                subset_idx = torch.randperm(q_values.shape[0], device=self.device)[:subset_size]
            return q_values[subset_idx].min(dim=0).values
        elif mode == 'LCB':
            return q_values.mean(dim=0) - beta * (q_values.std(dim=0) + 1e-6)
        elif mode == 'UCB':
            return q_values.mean(dim=0) + beta * (q_values.std(dim=0) + 1e-6)
        elif mode == 'median':
            return q_values.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    def importance_sampling_coef(self, log_pi: torch.Tensor, log_mu: torch.Tensor) -> torch.Tensor:
        return (log_pi - log_mu).exp().clamp(max=1.0)

    def _solve_temp_dual(self, q_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q_values.detach()  # (B, N)
        n_samples = q.shape[-1]

        with torch.enable_grad():
            for _ in range(self.args.dual_steps):
                self.dual_temp_optimizer.zero_grad()
                eta = self.log_eta.exp()
                dual_loss = (eta * self.args.dual_constraint
                             + eta * (torch.logsumexp(q / eta, dim=-1) - np.log(n_samples)).mean())
                dual_loss.backward()
                self.dual_temp_optimizer.step()
                self.log_eta.data.clamp_(-4.0, 4.0)

        eta_star = self.log_eta.exp().detach()
        weights = torch.softmax(q / eta_star, dim=-1)  # (B, N)
        return eta_star, weights

    def _e_step(self, data):
        obs = data.observations
        N = self.args.sample_action_num
        B, ds = obs.shape[0], obs.shape[-1]
        da = self.envs.single_action_space.shape[0]

        with torch.no_grad():
            d = self.actor_target.forward(obs)
            b_mu, b_sigma = d.base_dist.loc, d.base_dist.scale                            # (B, da)
            action_samples = d.rsample((N,)).permute(1, 0, 2)            # (B, N, da)

            obs_exp = obs.unsqueeze(1).expand(-1, N, -1).reshape(-1, ds)
            acts_flat = action_samples.reshape(-1, da)
            q_values = self.aggregation_operator(
                obs_exp, acts_flat, self.target_critics, mode='mean'
            ).reshape(B, N)

        eta, weights = self._solve_temp_dual(q_values)
        return action_samples, weights, eta, b_mu, b_sigma

    def _m_step(self, obs, action_samples, weights, b_mu, b_sigma) -> Tuple[float, float, float]:
        loss_p_val = kl_mu_val = kl_sigma_val = 0.0

        for _ in range(self.args.mstep_iteration_num):
            curr_d = self.actor.forward(obs)

            # Weighted log-likelihood under current policy.
            log_probs = dist.Normal(curr_d.base_dist.loc.unsqueeze(1), curr_d.base_dist.scale.unsqueeze(1)).log_prob(action_samples)
            nll = -(weights.detach() * log_probs.sum(-1)).sum(-1).mean()

            # Decoupled KL: stop-gradient on sigma for mean term, on mu for covariance term.
            kl_mu = dist.kl_divergence(
                dist.Normal(b_mu, b_sigma),
                dist.Normal(curr_d.base_dist.loc, b_sigma),     # σ fixed at old
            ).sum(-1).mean()

            kl_sigma = dist.kl_divergence(
                dist.Normal(b_mu, b_sigma),
                dist.Normal(b_mu, curr_d.base_dist.scale),      # μ fixed at old
            ).sum(-1).mean()

            loss = (nll
                    + self.alpha_mu * (kl_mu - self.args.kl_mean_constraint)
                    + self.alpha_sigma * (kl_sigma - self.args.kl_var_constraint))

            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            loss_p_val = nll.item()
            kl_mu_val = kl_mu.item()
            kl_sigma_val = kl_sigma.item()

        # Dual-ascent update once per _learn() call, not per inner gradient step.
        # With mstep_iteration_num=5 and utd_ratio=3, updating inside the loop
        # applies 15 alpha updates per env step, causing KL constraint oscillation.
        self.alpha_mu += self.args.alpha_mean_scale * (kl_mu_val - self.args.kl_mean_constraint)
        self.alpha_mu = float(np.clip(self.alpha_mu, 0.0, self.args.alpha_mean_max))
        self.alpha_sigma += self.args.alpha_var_scale * (kl_sigma_val - self.args.kl_var_constraint)
        self.alpha_sigma = float(np.clip(self.alpha_sigma, 0.0, self.args.alpha_var_max))

        return loss_p_val, kl_mu_val, kl_sigma_val

    def _update_targets(self):
        if not self.args.decouple_q_learning:
            for p, p_tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_tgt.data.lerp_(p.data, self.args.tau)
        
        for critic, target_critic in zip(self.critics, self.target_critics):
            for p, p_tgt in zip(critic.parameters(), target_critic.parameters()):
                p_tgt.data.lerp_(p.data, self.args.tau)

    def initialize_logging(self, info):
        self.start_time = time.time()

        log_dir = os.path.join(self.disk_folder, self.runs_directory, self.run_name)
        self.csv_file_info = open(os.path.join(log_dir, "info_logs.csv"), "w", newline="")
        self.keys_info = [k for k in info.keys() if not (k.startswith("bodies") or k.startswith("_"))]
        self.writer_info = csv.DictWriter(self.csv_file_info, fieldnames=["step"] + self.keys_info)
        self.writer_info.writeheader()

        self.reward_tracker = RewardTracker(
            env_dt=self.args.dt,
            env_id=self.args.env_id,
            log_folder=log_dir,
            time_window=120.0,
        )

        self.csv_file_agent_vars = open(os.path.join(log_dir, "performance_variables.csv"), "w", newline="")
        self.writer_agent_vars = csv.DictWriter(self.csv_file_agent_vars, fieldnames=["step"] + self.keys_agent_vars)
        self.writer_agent_vars.writeheader()

        self.info_log_buffer = []
        self.agent_vars_buffer = []

    def log_step(self, global_step, infos, rewards, metrics=None):
        if self.writer_info is None:
            return

        self.reward_tracker.update(infos['original_reward'][0])
        self.reward_tracker.log()

        row = {"step": global_step}
        for k in self.keys_info:
            if k in infos:
                row[k] = arr_to_str(infos[k][0])
        self.info_log_buffer.append(row)

        if metrics is not None:
            self.agent_vars_buffer.append({
                "step": global_step,
                "loss_q": metrics.get('loss_q'),
                "loss_p": metrics.get('loss_p'),
                "mean_q": metrics.get('mean_q'),
                "eta": metrics.get('eta'),
                "kl_mu": metrics.get('kl_mu'),
                "kl_sigma": metrics.get('kl_sigma'),
                "alpha_mu": metrics.get('alpha_mu'),
                "alpha_sigma": metrics.get('alpha_sigma'),
                "utd_ratio": metrics.get('utd_ratio'),
                "SPS": int(global_step / (time.time() - self.start_time)) if self.start_time else 0,
                "average_reward_per_second": self.reward_tracker.average_reward_per_second,
                "reward": rewards[0] if hasattr(rewards, '__len__') else float(rewards),
            })

        if global_step % self.args.save_every_n_steps == 0:
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
            self.info_log_buffer = []

            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
            self.agent_vars_buffer = []

    def save_checkpoint(self, global_step):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "target_critics": [c.state_dict() for c in self.target_critics],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "log_eta": self.log_eta.item(),
            "dual_temp_optimizer": self.dual_temp_optimizer.state_dict(),
            "alpha_mu": self.alpha_mu,
            "alpha_sigma": self.alpha_sigma,
            "global_step": global_step,
        }
        torch.save(checkpoint, os.path.join(self.weights_folder, f"checkpoint_{global_step}.pth"))
        self.rb.save(os.path.join(self.weights_folder, "replay_buffer.npz"))

    def load_checkpoint(self, weights_path):
        checkpoint_files = [f for f in os.listdir(weights_path) if f.endswith(".pth")]
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint = torch.load(os.path.join(weights_path, checkpoint_files[-1]), map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        for c, sd in zip(self.critics, checkpoint["critics"]):
            c.load_state_dict(sd)
        for c, sd in zip(self.target_critics, checkpoint["target_critics"]):
            c.load_state_dict(sd)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        for opt, sd in zip(self.critic_optimizers, checkpoint["critic_optimizers"]):
            opt.load_state_dict(sd)
        if "log_eta" in checkpoint:
            self.log_eta.data.fill_(checkpoint["log_eta"])
        if "dual_temp_optimizer" in checkpoint:
            self.dual_temp_optimizer.load_state_dict(checkpoint["dual_temp_optimizer"])
        self.alpha_mu = float(checkpoint.get("alpha_mu", 0.0))
        self.alpha_sigma = float(checkpoint.get("alpha_sigma", 0.0))
        self.global_step = checkpoint.get("global_step", 0)
        print(f"[√] Loaded checkpoint from {weights_path}, global_step={self.global_step}")

        buffer_path = os.path.join(weights_path, "replay_buffer.npz")
        if os.path.exists(buffer_path):
            self.rb.load(buffer_path, self.device)
            print(f"[√] Loaded replay buffer.")

    def cleanup(self):
        if self.writer_info is not None and self.info_log_buffer:
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
        if self.writer_agent_vars is not None and self.agent_vars_buffer:
            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
        if self.csv_file_info:
            self.csv_file_info.close()
        if self.csv_file_agent_vars:
            self.csv_file_agent_vars.close()
        if self.envs:
            self.envs.close()


def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument("--exp_name", type=str, default="mpo_ant")
    parser.add_argument("--runs_directory", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", type=bool, default=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--save_every_n_steps", type=int, default=4000)

    # Algorithm.
    parser.add_argument("--env_id", type=str, default="EAnt")
    parser.add_argument("--total_timesteps", type=int, default=60_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=2000)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--use_layer_norm", type=bool, default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden_layers", type=int, default=2)
    parser.add_argument("--utd_ratio", type=int, default=1)
    parser.add_argument("--td_horizon", type=int, default=1,
                        help="n-step TD horizon for Q learning")
    parser.add_argument("--decouple_q_learning", action='store_true', default=False)
    parser.add_argument("--policy_learning_starts", type=int, default=0,
                        help="steps of Q-only warmup after learning_starts when --decouple_q_learning is set")

    # MPO specific.
    parser.add_argument("--dual_constraint", type=float, default=0.1,
                        help="epsilon for E-step temperature dual")
    parser.add_argument("--kl_mean_constraint", type=float, default=0.01,
                        help="epsilon_mu for M-step mean KL")
    parser.add_argument("--kl_var_constraint", type=float, default=1e-4,
                        help="epsilon_sigma for M-step covariance KL")
    parser.add_argument("--alpha_mean_scale", type=float, default=1.0)
    parser.add_argument("--alpha_var_scale", type=float, default=20.0)
    parser.add_argument("--alpha_mean_max", type=float, default=0.1)
    parser.add_argument("--alpha_var_max", type=float, default=10.0)
    parser.add_argument("--sample_action_num", type=int, default=64,
                        help="actions sampled per state in E-step")
    parser.add_argument("--mstep_iteration_num", type=int, default=5,
                        help="actor gradient steps per learn() call")
    parser.add_argument("--dual_lr", type=float, default=1e-2)
    parser.add_argument("--dual_steps", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ensemble", type=int, default=1)

    # Environment.
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--hw_config", type=str, default=None)
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--terminate_on_upside_down", type=bool, default=True)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"])
    parser.add_argument("--radius_back_and_forth", type=float, default=0.3)
    parser.add_argument("--origin_back_and_forth", type=float, nargs=2, default=[0.75, -0.3])
    parser.add_argument("--reward_scale", type=float, default=100.0)
    parser.add_argument("--model_path", type=str,
                        default="../../sim/assets/ant_with_camera_after_sys_id.xml")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    disk_folder = ''
    os.makedirs(args.runs_directory, exist_ok=True)
    run_name = f"{args.exp_name}_{date}_seed_{args.seed}"
    os.makedirs(os.path.join(args.runs_directory, run_name), exist_ok=True)

    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = args.radius_back_and_forth
        ORIGIN = np.array(args.origin_back_and_forth)
        task = BackAndForthTask(radius=RADIUS, origin=ORIGIN)
        print(f"BackAndForthTask: radius={RADIUS}, origin={ORIGIN}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    envs = make_ant_envs(args, task, disk_folder, run_name, runs_directory=args.runs_directory)

    agent = MPO(args, envs, disk_folder=disk_folder, run_name=run_name, runs_directory=args.runs_directory)

    if args.weights_path is not None:
        agent.load_checkpoint(args.weights_path)
        if args.eval:
            agent.global_step = 0

    obs, info = envs.reset(seed=args.seed)
    agent.initialize_logging(info)

    for step in tqdm(range(agent.global_step, args.total_timesteps)):

        selected_actions = agent.get_action(obs, args.eval)
        next_obs, rewards, terminations, truncations, infos = envs.step(selected_actions)

        if args.eval:
            agent.agent_step_eval(next_obs)
            metrics = None
        else:
            metrics = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)

        agent.log_step(step, infos, rewards, metrics)

        if step % args.save_every_n_steps == 0:
            agent.save_checkpoint(step)

    agent.cleanup()
