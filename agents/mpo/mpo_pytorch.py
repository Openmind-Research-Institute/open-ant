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
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

# Import custom modules.
from utils.buffers import ReplayBuffer, ReplayBufferSamples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from sim.ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

# For logging.
def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x


def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument("--exp_name", type=str, default="mpo_ant",
                        help="the name of this experiment")
    parser.add_argument("--runs_directory", type=str, default="runs",
                        help="the directory to save the runs in")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True,
                        help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture_video", action="store_true",
                        help="capture video of agent performances")

    # Algorithm.
    parser.add_argument("--env_id", type=str, default="EAnt",
                        help="environment ID")
    parser.add_argument("--total_timesteps", type=int, default=100_000,
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
    parser.add_argument("--autotune", type=bool, default=True,
                        help="automatic entropy tuning")
    parser.add_argument("--gamma_discrete", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--use_layer_norm", type=bool, default=True,
                        help="use layer normalization in networks")

     # MPO specific.
    parser.add_argument("--dual_constraint", type=float, default=0.1,
                        help="ε for E-step dual")
    parser.add_argument("--kl_mean_constraint", type=float, default=0.01,
                        help="ε_μ for M-step mean KL")
    parser.add_argument("--kl_var_constraint", type=float, default=0.001,
                        help="ε_Σ for M-step covariance KL")
    parser.add_argument("--alpha_mean_scale", type=float, default=1.0)
    parser.add_argument("--alpha_var_scale", type=float, default=100.0)
    parser.add_argument("--alpha_mean_max", type=float, default=0.1)
    parser.add_argument("--alpha_var_max", type=float, default=10.0)
    parser.add_argument("--sample_action_num", type=int, default=20,
                        help="actions sampled per state in E-step and critic target")
    parser.add_argument("--mstep_iteration_num", type=int, default=5,
                        help="actor gradient steps per learn() call")
    parser.add_argument("--dual_lr", type=float, default=1e-2)
    parser.add_argument("--dual_steps", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Environment.
    parser.add_argument("--dt", type=float, default=0.12,
                        help="environment timestep")
    parser.add_argument("--hw_config", type=str, default=None,
                        help="hardware config file")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="render mode")
    parser.add_argument("--terminate_on_upside_down", type=bool, default=True,
                        help="terminate episode if upside down")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="load previous weights")
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"],
                        help="type of task")
    parser.add_argument("--reward_scale", type=float, default=10.0,
                        help="reward scale factor")
    parser.add_argument("--model_path", type=str, default="../../sim/assets/ant_with_camera_after_sys_id.xml",
                        help="XML file to use for the environment")
    parser.add_argument("--eval", type=bool, default=False,
                        help="evaluate the agent")
    parser.add_argument("--save_every_n_steps", type=int, default=500,
                        help="save every n steps")


    args = parser.parse_args()
    return args

class QNetwork(nn.Module):
    def __init__(self, env, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)

    def forward(self, x, a):
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

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    """
    Decoupled KL between two multivariate Gaussians (MPO paper, eq. 5).
      C_μ = KL(π(·|μi, Σi) || π(·|μ,  Σi))  — mean component only
      C_Σ = KL(π(·|μi, Σi) || π(·|μi, Σ ))  — covariance component only
    Both Ai and A are lower-triangular Cholesky factors, shape (B, da, da).
    Returns scalars C_μ, C_Σ and mean determinants of Σi and Σ for logging.
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)           # (B, da, 1)
    μ  = μ.unsqueeze(-1)
    Σi = Ai @ bt(Ai)                # (B, da, da)
    Σ  = A  @ bt(A)
    Σi_det = Σi.det().clamp_min(1e-6)
    Σ_det  = Σ.det().clamp_min(1e-6)
    Σi_inv = Σi.inverse()
    Σ_inv  = Σ.inverse()
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()
    inner_Σ = torch.log(Σ_det / Σi_det) - n + btr(Σ_inv @ Σi)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ, torch.mean(Σi_det), torch.mean(Σ_det)


class Actor(nn.Module):
    """
    Full-covariance Gaussian policy parameterised by a lower-triangular
    Cholesky factor, as required by the MPO trajectory distribution q(τ).
    forward() returns (mean, cholesky) where cholesky is scale_tril.
    """
    def __init__(self, env, use_layer_norm=False):
        super().__init__()
        ds = int(np.array(env.single_observation_space.shape).prod())
        da = int(np.prod(env.single_action_space.shape))
        self.da = da
        self.fc1 = nn.Linear(ds, 256)
        self.fc2 = nn.Linear(256, 256)
        self.ln1 = nn.LayerNorm(256) if use_layer_norm else None
        self.ln2 = nn.LayerNorm(256) if use_layer_norm else None
        self.fc_mean     = nn.Linear(256, da)
        self.fc_cholesky = nn.Linear(256, (da * (da + 1)) // 2)

        self.register_buffer(
            "action_low",
            torch.tensor(env.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(env.single_action_space.high, dtype=torch.float32),
        )

    def forward(self, x):
        B  = x.size(0)
        da = self.da
        h = F.relu(self.ln1(self.fc1(x)) if self.ln1 else self.fc1(x))
        h = F.relu(self.ln2(self.fc2(h)) if self.ln2 else self.fc2(h))

        # Mean bounded to action range via sigmoid.
        mean = torch.sigmoid(self.fc_mean(h))
        mean = self.action_low + (self.action_high - self.action_low) * mean

        # Cholesky factor: off-diagonals free, diagonal forced positive via softplus.
        chol_vec = self.fc_cholesky(h).clone()          # clone to allow in-place
        diag_idx = torch.arange(da, device=x.device)
        diag_idx = (diag_idx + 1) * (diag_idx + 2) // 2 - 1
        chol_vec[:, diag_idx] = F.softplus(chol_vec[:, diag_idx]) + 1e-4

        tril_idx = torch.tril_indices(da, da, 0, device=x.device)
        cholesky = torch.zeros(B, da, da, device=x.device)
        cholesky[:, tril_idx[0], tril_idx[1]] = chol_vec

        return mean, cholesky


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

            if capture_video and idx == 0:
                print('RecordVideo')
                env = gym.wrappers.RecordVideo(env, os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
                                               step_trigger=lambda x: x % 500 == 0, video_length=500)
            env.action_space.seed(seed)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
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
        # Set up folders.
        self.disk_folder = disk_folder
        self.run_name = run_name
        self.runs_directory = runs_directory
        self.weights_folder = os.path.join(self.disk_folder, self.runs_directory, self.run_name, "weights_and_args")
        os.makedirs(self.weights_folder, exist_ok=True)
        with open(os.path.join(self.weights_folder, "args.json"), 'w') as f:
            json.dump(args.__dict__, f)

        # Set seeds for reproducibility.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        torch.backends.cudnn.benchmark = not args.torch_deterministic

        # Networks.
        self.actor = Actor(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.actor_target = Actor(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.qf1 = QNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf1_target = QNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())

        self.qf2 = QNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf2_target = QNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.learning_starts = args.learning_starts
        self.gamma_discrete = args.gamma_discrete
        self.tau = args.tau

        # Dual variable η for the E-step temperature (parameterised via softplus).
        self._log_eta = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_temp_optimizer = optim.Adam([self._log_eta], lr=args.dual_lr)

        # Lagrange multipliers for M-step KL constraints (scalar, updated by dual ascent).
        self.alpha_mu    = 0.0
        self.alpha_sigma = 0.0

        self.critic_loss_fn = nn.SmoothL1Loss()

        # Load checkpoint if provided.
        checkpoint = None
        self.weights_path = args.weights_path
        if self.weights_path is not None:
            checkpoint_files = [f for f in os.listdir(self.weights_path) if f.endswith(".pth")]
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_file = checkpoint_files[-1]
            checkpoint = torch.load(os.path.join(self.weights_path, checkpoint_file),
                                    map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_target.load_state_dict(checkpoint["actor_target"])
            self.qf1.load_state_dict(checkpoint["qf1"])
            self.qf1_target.load_state_dict(checkpoint["qf1_target"])
            if "qf2" in checkpoint:
                self.qf2.load_state_dict(checkpoint["qf2"])
                self.qf2_target.load_state_dict(checkpoint["qf2_target"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            if "log_eta" in checkpoint:
                self._log_eta.data.fill_(checkpoint["log_eta"])
            if "dual_temp_optimizer" in checkpoint:
                self.dual_temp_optimizer.load_state_dict(checkpoint["dual_temp_optimizer"])
            self.alpha_mu    = checkpoint.get("alpha_mu", 0.0)
            self.alpha_sigma = checkpoint.get("alpha_sigma", 0.0)
            self.learning_starts = 0
            print(f"[√] Loaded checkpoint! {checkpoint_file}. Learning starts set to 0.")

        self.eval = args.eval
        if self.eval == True and self.weights_path is None:
            raise ValueError("[!] Cannot evaluate without weights path.")
        if self.eval:
            self.learning_starts = args.total_timesteps

        self.envs.single_observation_space.dtype = np.float32

        # Replay buffer.
        self.rb = ReplayBuffer(
            args.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
        if self.weights_path is not None:
            buffer_path = os.path.join(self.weights_path, "replay_buffer.npz")
            if os.path.exists(buffer_path):
                self.rb.load(buffer_path, self.device)
                print(f"[√] Loaded replay buffer with {self.rb.size} transitions")
            else:
                print("[!] No replay buffer found, starting empty.")

        # Initialize tracking variables for external control.
        # Load global_step from checkpoint if resuming, otherwise start at 0.
        if checkpoint is not None and "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            print(f"[√] Loaded checkpoint from weights folder {self.weights_path}. Resuming from global_step {self.global_step}.")
        else:
            self.global_step = 0
            print(f"[√] Starting from global_step {self.global_step}.")

        # Logging state.
        self.start_time = None
        self.reward_tracker = None
        self.csv_file_info = None
        self.csv_file_agent_vars = None
        self.writer_info = None
        self.writer_agent_vars = None
        self.keys_info = None
        self.keys_agent_vars = [
            'loss_q', 'loss_p', 'loss_lagrangian', 'mean_q',
            'eta', 'kl_mu', 'kl_sigma', 'sigma_det',
            'alpha_mu', 'alpha_sigma',
            'SPS', 'average_reward_per_second', 'reward',
            't_critic', 't_estep', 't_mstep',
        ]
        self.info_log_buffer = []
        self.agent_vars_buffer = []

    def get_action(self, obs, global_step=None):
        """Get action from observation."""
        if global_step is None:
            global_step = self.global_step

        if global_step < self.learning_starts and self.weights_path is None:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            with torch.no_grad():
                mu, A = self.actor_target(torch.Tensor(obs).to(self.device))
                actions = dist.MultivariateNormal(mu, scale_tril=A).sample().cpu().numpy()

        return actions

    def add_transition(self, obs, next_obs, actions, rewards, terminations, infos):
        """Add transition to replay buffer."""
        self.rb.add(obs, next_obs, actions, rewards, terminations, infos)

    def update_critic(self,
                      batch_data: ReplayBufferSamples) -> Tuple[float, float]:
        s_next = batch_data.next_observations
        B  = s_next.size(0)
        N  = self.args.sample_action_num
        ds = s_next.size(-1)
        da = self.envs.single_action_space.shape[0]

        with torch.no_grad():
            mu_next, A_next = self.actor_target(s_next)
            a_next = dist.MultivariateNormal(mu_next, scale_tril=A_next).sample((N,))  # (N, B, da)
            s_next_exp = s_next.unsqueeze(0).expand(N, -1, -1)                          # (N, B, ds)
            a_next_flat = a_next.reshape(-1, da)
            s_next_flat = s_next_exp.reshape(-1, ds)
            q1_next = self.qf1_target(s_next_flat, a_next_flat).reshape(N, B)
            q2_next = self.qf2_target(s_next_flat, a_next_flat).reshape(N, B)
            q_next = torch.min(q1_next, q2_next).mean(0)                               # (B,) clipped double-Q
            y = (batch_data.rewards.flatten()
                 + (1 - batch_data.dones.flatten()) * self.gamma_discrete * q_next)

        q1_value = self.qf1(batch_data.observations, batch_data.actions).squeeze()
        q2_value = self.qf2(batch_data.observations, batch_data.actions).squeeze()
        loss_q   = self.critic_loss_fn(q1_value, y) + self.critic_loss_fn(q2_value, y)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        return loss_q.item(), q1_value.mean().item()

    def solve_temp_dual(self, q_samples: torch.Tensor, epsilon: float, n_dual_steps: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        q_values = q_samples.detach()                        # (N, B)
        max_q    = q_values.max(dim=0).values                # (B,)
        centered = q_values - max_q.unsqueeze(0)             # (N, B)

        with torch.enable_grad():
            for _ in range(n_dual_steps):
                self.dual_temp_optimizer.zero_grad()
                eta          = F.softplus(self._log_eta) + 1e-8
                log_mean_exp = torch.log(torch.mean(torch.exp(centered / eta), dim=0))  # (B,)
                dual_loss    = eta * epsilon + max_q.mean() + eta * log_mean_exp.mean()
                dual_loss.backward()
                self.dual_temp_optimizer.step()

        with torch.no_grad():
            eta_star = F.softplus(self._log_eta) + 1e-8
            weights  = torch.softmax(q_values / eta_star, dim=0)   # (N, B)

        return eta_star.detach(), weights

    def solve_kl_dual(self, kl_value: torch.Tensor, epsilon: float, n_dual_steps: int = 30) -> torch.Tensor:
        log_alpha = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        dual_optimizer = optim.Adam([log_alpha], lr=1e-2)
        kl = kl_value.detach()

        with torch.enable_grad():
            for _ in range(n_dual_steps):
                dual_optimizer.zero_grad()
                alpha = log_alpha.exp()
                dual_loss = alpha * (epsilon - kl)
                dual_loss.backward()
                dual_optimizer.step()

        return log_alpha.exp().detach()

    def e_step(self,
               batch_data: ReplayBufferSamples):
        obs = batch_data.observations
        N   = self.args.sample_action_num
        B   = obs.size(0)
        ds  = obs.size(-1)
        da  = self.envs.single_action_space.shape[0]

        with torch.no_grad():
            b_mu, b_A       = self.actor_target(obs)                              # (B,da), (B,da,da)
            b_pi            = dist.MultivariateNormal(b_mu, scale_tril=b_A)
            sampled_actions = b_pi.sample((N,))                                    # (N,B,da)
            s_exp           = obs.unsqueeze(0).expand(N, -1, -1)                   # (N,B,ds)
            a_flat          = sampled_actions.reshape(-1, da)
            s_flat          = s_exp.reshape(-1, ds)
            q1_values       = self.qf1_target(s_flat, a_flat).reshape(N, B)
            q2_values       = self.qf2_target(s_flat, a_flat).reshape(N, B)
            q_values        = torch.min(q1_values, q2_values)                     # (N,B) clipped double-Q

        eta, weights = self.solve_temp_dual(q_values, self.args.dual_constraint, self.args.dual_steps)

        return sampled_actions, weights, eta, b_mu, b_A

    def m_step(self,
               obs: torch.Tensor,
               sampled_actions: torch.Tensor,
               weights: torch.Tensor,
               b_mu: torch.Tensor,
               b_A: torch.Tensor) -> Tuple[float, float, float, float]:

        N = self.args.sample_action_num
        B = obs.size(0)

        loss_p_val = kl_mu_val = kl_sigma_val = sigma_det_val = 0.0

        for _ in range(self.args.mstep_iteration_num):
            mu, A = self.actor(obs)                                            # (B,da), (B,da,da)

            # Decoupled log-prob: mean term uses old Σ, covariance term uses old μ.
            pi1    = dist.MultivariateNormal(mu,   scale_tril=b_A)
            pi2    = dist.MultivariateNormal(b_mu, scale_tril=A)
            lp1    = pi1.expand((N, B)).log_prob(sampled_actions)              # (N,B)
            lp2    = pi2.expand((N, B)).log_prob(sampled_actions)              # (N,B)
            loss_p = torch.mean(weights * (lp1 + lp2))

            kl_mu, kl_sigma, _, sigma_det = gaussian_kl(b_mu, mu, b_A, A)

            if torch.isnan(kl_mu) or torch.isnan(kl_sigma):
                raise RuntimeError('KL is nan in M-step')

            # Dual-ascent update of Lagrange multipliers (MPO paper eq. 17).
            self.alpha_mu    -= self.args.alpha_mean_scale * (self.args.kl_mean_constraint - kl_mu.item())
            self.alpha_sigma -= self.args.alpha_var_scale  * (self.args.kl_var_constraint  - kl_sigma.item())
            self.alpha_mu    = float(np.clip(self.alpha_mu,    0.0, self.args.alpha_mean_max))
            self.alpha_sigma = float(np.clip(self.alpha_sigma, 0.0, self.args.alpha_var_max))

            loss_l = -(loss_p
                       + self.alpha_mu    * (self.args.kl_mean_constraint - kl_mu)
                       + self.alpha_sigma * (self.args.kl_var_constraint  - kl_sigma))

            self.actor_optimizer.zero_grad()
            loss_l.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            loss_p_val    = (-loss_p).item()
            kl_mu_val     = kl_mu.item()
            kl_sigma_val  = kl_sigma.item()
            sigma_det_val = sigma_det.item()

        return loss_p_val, kl_mu_val, kl_sigma_val, sigma_det_val

    def _update_targets(self) -> None:
        for p, p_tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)
        for p, p_tgt in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)
        for p, p_tgt in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)

    def learn(self, global_step=None):
        """Perform one learning step."""
        if global_step is None:
            global_step = self.global_step

        if global_step < self.learning_starts:
            return None

        data = self.rb.sample(self.args.batch_size)

        # 1. Policy Evaluation — critic TD update
        t_critic_start = time.time()
        qf1_loss, mean_q = self.update_critic(data)
        t_critic = time.time() - t_critic_start

        # 2. E-step — compute action weights
        t_estep_start = time.time()
        sampled_actions, weights, eta, b_mu, b_A = self.e_step(data)
        t_estep = time.time() - t_estep_start

        # 3. M-step — actor update
        t_mstep_start = time.time()
        loss_p, kl_mu, kl_sigma, sigma_det = self.m_step(data.observations, sampled_actions, weights, b_mu, b_A)
        t_mstep = time.time() - t_mstep_start

        self._update_targets()

        return {
            'loss_q': qf1_loss,
            'loss_p': loss_p,
            'loss_lagrangian': None,
            'mean_q': mean_q,
            'eta': eta.item(),
            'kl_mu': kl_mu,
            'kl_sigma': kl_sigma,
            'sigma_det': sigma_det,
            'alpha_mu': self.alpha_mu,
            'alpha_sigma': self.alpha_sigma,
            't_critic': t_critic,
            't_estep': t_estep,
            't_mstep': t_mstep,
        }

    def initialize_logging(self, info):
        """Initialize logging files and trackers."""
        self.start_time = time.time()

        # Log the information of choice.
        self.csv_file_info = open(os.path.join(self.disk_folder, self.runs_directory, self.run_name, "info_logs.csv"), "w", newline="")
        self.keys_info = list(info.keys())
        self.keys_info = [k for k in self.keys_info if not (k.startswith("bodies") or k.startswith("_"))]

        self.writer_info = csv.DictWriter(self.csv_file_info, fieldnames=["step"] + self.keys_info)
        self.writer_info.writeheader()

        # Reward tracker.
        self.reward_tracker = RewardTracker(env_dt=self.args.dt, env_id=self.args.env_id,
                                    log_folder=os.path.join(self.disk_folder, self.runs_directory, self.run_name),
                                    time_window=120.0)

        # Performance variables.
        self.csv_file_agent_vars = open(os.path.join(self.disk_folder, self.runs_directory, self.run_name, "performance_variables.csv"), "w", newline="")
        self.writer_agent_vars = csv.DictWriter(self.csv_file_agent_vars, fieldnames=["step"] + self.keys_agent_vars)
        self.writer_agent_vars.writeheader()

        # Initialize buffers.
        self.info_log_buffer = []
        self.agent_vars_buffer = []

    def log_step(self, global_step, infos, rewards, metrics=None):
        """Log step information."""
        if self.writer_info is None or self.writer_agent_vars is None:
            return

        # Update the reward tracker.
        if self.args.num_envs == 1:
            self.reward_tracker.update(rewards.item())
            self.reward_tracker.log()
        else:
            raise ValueError("reward_tracker is only supported for single environment")

        # Log the infos - add to buffer instead of writing directly.
        infos_to_log = {}
        for k, v in infos.items():
            if k in self.keys_info:
               infos_to_log[k] = arr_to_str(v[0])
        row = {"step": global_step, **infos_to_log}
        self.info_log_buffer.append(row)

        # Log performance metrics - add to buffer.
        if metrics is not None:
            self.agent_vars_buffer.append({
                "step": global_step,
                "loss_q": metrics.get('loss_q'),
                "loss_p": metrics.get('loss_p'),
                "loss_lagrangian": metrics.get('loss_lagrangian'),
                "mean_q": metrics.get('mean_q'),
                "eta": metrics.get('eta'),
                "kl_mu": metrics.get('kl_mu'),
                "kl_sigma": metrics.get('kl_sigma'),
                "sigma_det": metrics.get('sigma_det'),
                "alpha_mu": metrics.get('alpha_mu'),
                "alpha_sigma": metrics.get('alpha_sigma'),
                "SPS": int(global_step / (time.time() - self.start_time)) if self.start_time else 0,
                "average_reward_per_second": self.reward_tracker.average_reward_per_second,
                "reward": rewards.item(),
                "t_critic": metrics.get('t_critic'),
                "t_estep": metrics.get('t_estep'),
                "t_mstep": metrics.get('t_mstep'),
            })

        # Write to CSV every save_every_n_steps steps.
        if global_step % self.args.save_every_n_steps == 0:
            # Write all buffered info logs.
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
            self.info_log_buffer = []

            # Write all buffered agent vars.
            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
            self.agent_vars_buffer = []

    def save_checkpoint(self, global_step):
        # Save all the networks.
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_eta": self._log_eta.item(),
            "dual_temp_optimizer": self.dual_temp_optimizer.state_dict(),
            "alpha_mu": self.alpha_mu,
            "alpha_sigma": self.alpha_sigma,
            "global_step": global_step,
        }
        torch.save(checkpoint, os.path.join(self.weights_folder, f"checkpoint_{global_step}.pth"))

        # Save the replay buffer.
        self.rb.save(os.path.join(self.weights_folder, "replay_buffer.npz"))

    def cleanup(self):
        """Clean up resources."""
        # Write any remaining buffered data before closing.
        if self.writer_info is not None and self.info_log_buffer:
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
            self.info_log_buffer = []

        if self.writer_agent_vars is not None and self.agent_vars_buffer:
            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
            self.agent_vars_buffer = []

        if self.csv_file_info:
            self.csv_file_info.close()
        if self.csv_file_agent_vars:
            self.csv_file_agent_vars.close()
        if self.envs:
            self.envs.close()

    def run_policy(self):
        """Main training loop - runs the MPO policy."""
        # Reset the environment.
        obs, info = self.envs.reset(seed=self.args.seed)

        # Initialize logging.
        self.initialize_logging(info)

        # Start learning.
        times = {
            'time_get_the_action': [],
            'time_step_the_environment': [],
            'time_add_transition_buffer': [],
            'time_learn': [],
            'time_log_step': [],
        }
        time_start = time.time()
        # Start from the current global_step (0 if new run, loaded value if resuming).
        start_step = self.global_step
        for global_step in tqdm(range(start_step, self.args.total_timesteps)):
            self.global_step = global_step

            # Get the action.
            time_start = time.time()
            actions = self.get_action(obs, global_step)
            times['time_get_the_action'].append(time.time() - time_start)

            # Step the environment.
            time_start = time.time()
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
            rewards = rewards * self.args.reward_scale
            times['time_step_the_environment'].append(time.time() - time_start)

            # Add transition to buffer.
            # gymnasium SyncVectorEnv auto-resets: next_obs is the new episode's
            # first obs when terminated/truncated. Use final_observation instead.
            time_start = time.time()
            real_next_obs = next_obs.copy()
            for idx in range(self.envs.num_envs):
                if terminations[idx] or truncations[idx]:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.add_transition(obs, real_next_obs, actions, rewards, terminations, infos)
            times['time_add_transition_buffer'].append(time.time() - time_start)

            # Update the observation.
            obs = next_obs

            # Learn.
            time_start = time.time()
            metrics = self.learn(global_step)
            times['time_learn'].append(time.time() - time_start)
            # Log step.
            time_start = time.time()
            self.log_step(global_step, infos, rewards, metrics)
            times['time_log_step'].append(time.time() - time_start)

            if global_step % self.args.save_every_n_steps == 0:
                # Save checkpoint.
                self.save_checkpoint(global_step)
                # Save the times to df.
                df = pd.DataFrame(times)
                df.to_csv(os.path.join(self.disk_folder, self.runs_directory, self.run_name, "times.csv"), index=False)

        # Cleanup.
        self.cleanup()

if __name__ == "__main__":
    args = parse_args()

    # Set up folders for environment creation.
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    disk_folder = ''
    os.makedirs(args.runs_directory, exist_ok=True)
    run_name = f"{args.exp_name}_{date}_seed_{args.seed}"

    # Create task.
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = 0.3
        # RADIUS = 1.0
        ORIGIN = np.array([0.75,  -0.35]) # Measured in the environment.
        # ORIGIN = np.array([0, 0])
        task = BackAndForthTask(
            radius=RADIUS,
            origin=ORIGIN,
        )
        print(f"BackAndForthTask initialized for radius: {RADIUS}, origin: {ORIGIN}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    # Create environment.
    envs = make_ant_envs(args, task, disk_folder, run_name, runs_directory=args.runs_directory)

    # Create MPO agent.
    agent = MPO(args, envs, disk_folder=disk_folder, run_name=run_name, runs_directory=args.runs_directory)

    # Run the policy.
    agent.run_policy()
