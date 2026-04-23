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
    parser.add_argument("--kl_var_constraint", type=float, default=0.0001,
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
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

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

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.learning_starts = args.learning_starts
        self.gamma_discrete = args.gamma_discrete
        self.tau = args.tau

        #dual pb init:
        self.log_eta = torch.tensor(1.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_temp_optimizer = optim.Adam([self.log_eta], lr=args.dual_lr)

        self.log_alpha_mu = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_kl_mu_optimizer = optim.Adam([self.log_alpha_mu], lr=args.dual_lr)

        self.log_alpha_sigma = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_kl_sigma_optimizer = optim.Adam([self.log_alpha_sigma], lr=args.dual_lr)

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
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            if "log_eta" in checkpoint:
                self.log_eta.data.fill_(checkpoint["log_eta"])
            if "dual_temp_optimizer" in checkpoint:
                self.dual_temp_optimizer.load_state_dict(checkpoint["dual_temp_optimizer"])
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

        if global_step < self.learning_starts and self.weights_path is None: # if no weights path, start from random actions
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
            actions = actions.detach().cpu().numpy()

        return actions

    def add_transition(self, obs, next_obs, actions, rewards, terminations, infos):
        """Add transition to replay buffer."""
        self.rb.add(obs, next_obs, actions, rewards, terminations, infos)

    def update_critic(self,
                      batch_data: ReplayBufferSamples) -> Tuple[float, float]:
        with torch.no_grad():
            next_action, _, _ = self.actor_target.get_action(batch_data.next_observations)
            q_target = self.qf1_target(batch_data.next_observations, next_action)
            next_q_target = batch_data.rewards.flatten() + (1 - batch_data.dones.flatten()) * self.gamma_discrete * q_target.view(-1)

        q_value = self.qf1(batch_data.observations, batch_data.actions).view(-1)
        qf1_loss = F.mse_loss(q_value, next_q_target)

        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        return qf1_loss.item(), q_value.mean().item()

    def solve_temp_dual(self, q_samples:torch.Tensor, epsilon:float, n_dual_steps:int=200) -> Tuple[torch.Tensor, torch.Tensor]:
        _, n_samples = q_samples.shape
        q_values = q_samples.detach()

        with torch.enable_grad():
            for _ in range(n_dual_steps):
                self.dual_temp_optimizer.zero_grad()
                eta = self.log_eta.exp()
                dual_loss = eta * epsilon + eta * (torch.logsumexp(q_values / eta, dim=-1) - torch.log(torch.tensor(n_samples))).mean()
                dual_loss.backward()
                self.dual_temp_optimizer.step()
                self.log_eta.data.clamp_(-4.0, 4.0)  # keep eta in [~0.02, ~55], prevents q/eta overflow

        eta_star = self.log_eta.exp().detach()

        weights = torch.softmax(q_values/eta_star, dim=-1)
        return eta_star, weights

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
               batch_data: ReplayBufferSamples) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch_data.observations  # (B, obs_dim)
        N = self.args.sample_action_num
        B = obs.size(0)

        obs_exp = obs.unsqueeze(1).expand(-1, N, -1).reshape(-1, obs.shape[-1])  # (B*N, obs_dim)
        with torch.no_grad():
            mean, log_std = self.actor_target(obs_exp)
            std = log_std.exp()
            raw_actions = torch.distributions.Normal(mean, std).rsample()  # (B*N, act_dim)
            bounded_actions = torch.tanh(raw_actions) * self.actor_target.action_scale + self.actor_target.action_bias
            q_values = self.qf1_target(obs_exp, bounded_actions).reshape(B, N)  # (B, N)

        raw_actions = raw_actions.reshape(B, N, -1)  # (B, N, act_dim)

        eta, weights = self.solve_temp_dual(q_values, self.args.dual_constraint, self.args.dual_steps)

        return raw_actions, weights, eta

    def m_step(self,
               obs: torch.Tensor,
               sampled_actions: torch.Tensor,
               weights: torch.Tensor) -> Tuple[float, float, float, float, float]:
        N = self.args.sample_action_num
        B = obs.size(0)

        obs_exp = obs.unsqueeze(1).expand(-1, N, -1).reshape(-1, obs.shape[-1])  # (B*N, obs_dim)
        acts_flat = sampled_actions.reshape(-1, sampled_actions.shape[-1])  # (B*N, act_dim)

        # Weighted NLL (supervised fit to E-step distribution)
        mean, log_std = self.actor(obs_exp)
        std = log_std.exp()
        log_probs = torch.distributions.Normal(mean, std).log_prob(acts_flat).reshape(B, N, -1)
        nll = -(weights.detach() * log_probs.sum(-1)).sum(-1).mean()

        # Decoupled KL constraints
        mean_curr, log_std_curr = self.actor(obs)
        std_curr = log_std_curr.exp()
        with torch.no_grad():
            mean_old, log_std_old = self.actor_target(obs)
            std_old = log_std_old.exp()

        # D_KL^μ: sg on sigma_theta — gradients flow only through mu_theta
        kl_mu = dist.kl_divergence(
            dist.Normal(mean_curr, std_curr.detach()),
            dist.Normal(mean_old, std_old)
        ).sum(-1).mean()

        # D_KL^Σ: sg on mu_theta — gradients flow only through sigma_theta
        kl_sigma = dist.kl_divergence(
            dist.Normal(mean_curr.detach(), std_curr),
            dist.Normal(mean_old, std_old)
        ).sum(-1).mean()

        # No loop needed
        alpha_mu = torch.clamp(kl_mu.detach() - self.args.kl_mean_constraint, min=0.0)
        alpha_sigma = torch.clamp(kl_sigma.detach() - self.args.kl_var_constraint, min=0.0)

        policy_loss = (nll
                       + alpha_mu    * (kl_mu    - self.args.kl_mean_constraint)
                       + alpha_sigma * (kl_sigma - self.args.kl_var_constraint))

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()

        return policy_loss.item(), kl_mu.item(), kl_sigma.item(), alpha_mu.item(), alpha_sigma.item()

    def _update_targets(self) -> None:
        for p, p_tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)
        for p, p_tgt in zip(self.qf1.parameters(), self.qf1_target.parameters()):
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
        sampled_actions, weights, eta = self.e_step(data)
        t_estep = time.time() - t_estep_start

        # 3. M-step — actor update
        t_mstep_start = time.time()
        policy_loss, kl_mu, kl_sigma, alpha_mu, alpha_sigma = self.m_step(data.observations, sampled_actions, weights)
        t_mstep = time.time() - t_mstep_start

        self._update_targets()

        return {
            'loss_q': qf1_loss,
            'loss_p': policy_loss,
            'loss_lagrangian': None,
            'mean_q': mean_q,
            'eta': eta.item(),
            'kl_mu': kl_mu,
            'kl_sigma': kl_sigma,
            'sigma_det': None,
            'alpha_mu': alpha_mu,
            'alpha_sigma': alpha_sigma,
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
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_eta": self.log_eta.item(),
            "dual_temp_optimizer": self.dual_temp_optimizer.state_dict(),
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
            time_start = time.time()
            self.add_transition(obs, next_obs, actions, rewards, terminations, infos)
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
