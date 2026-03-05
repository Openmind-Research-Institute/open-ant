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
import itertools
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
from buffers import ReplayBuffer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
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
    parser.add_argument("--exp_name", type=str, default="sac_ant",
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
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="total training timesteps")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="number of parallel envs")
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="replay buffer size")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--learning_starts", type=int, default=5000,
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

    # Environment.
    parser.add_argument("--dt", type=float, default=0.05,
                        help="environment timestep")
    parser.add_argument("--hw_config", type=str, default=None,
                        help="hardware config file")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="render mode")
    parser.add_argument("--terminate_on_upside_down", type=bool, default=True,
                        help="terminate episode if upside down")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="load previous weights")
    parser.add_argument("--task_type", type=str, default="forward",
                        choices=["forward", "back_and_forth"],
                        help="type of task")
    parser.add_argument("--reward_scale", type=float, default=10.0,
                        help="reward scale factor")
    parser.add_argument("--model_path", type=str, default=None,
                        help="XML file to use for the environment")
    parser.add_argument("--eval", type=bool, default=False,
                        help="evaluate the agent")
    parser.add_argument("--save_every_n_steps", type=int, default=500,
                        help="save every n steps")
    parser.add_argument("--use_previous_action", type=bool, default=False,
                        help="use previous action")
    parser.add_argument("--use_layer_norm", action="store_true", default=False,
                        help="use layer normalization in networks")

    args = parser.parse_args()
    return args

class SoftQNetwork(nn.Module):
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
    def make_env(env_id, seed, idx, capture_video, run_name):
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
    print(f"[√] Created environment with {envs.num_envs} environments.")
    return envs


class SAC:
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
        self.qf1 = SoftQNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf2 = SoftQNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf1_target = SoftQNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf2_target = SoftQNetwork(self.envs, use_layer_norm=args.use_layer_norm).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.learning_starts = args.learning_starts
        self.gamma_discrete = args.gamma_discrete

        # Load checkpoint if provided.
        checkpoint = None
        self.weights_path = args.weights_path
        if self.weights_path is not None:
            # load the checkpoint that has _ the largest number in the name
            checkpoint_files = [f for f in os.listdir(self.weights_path) if f.endswith(".pth")]
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_file = checkpoint_files[-1]
            checkpoint = torch.load(os.path.join(self.weights_path, checkpoint_file), map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.qf1.load_state_dict(checkpoint["qf1"])
            self.qf2.load_state_dict(checkpoint["qf2"])
            self.qf1_target.load_state_dict(checkpoint["qf1_target"])
            self.qf2_target.load_state_dict(checkpoint["qf2_target"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.learning_starts = 0.0
            print(f"[√] Loaded checkpoint! {checkpoint_file}. Learning starts set to 0.")

        self.eval = args.eval
        if self.eval == True and self.weights_path is None:
            raise ValueError("[!] Cannot evaluate without weights path")
        if self.eval:
            self.learning_starts = args.total_timesteps

        # Alpha (entropy coefficient).
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.alpha_lr)

            if checkpoint is not None:
                self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
                self.a_optimizer.load_state_dict(checkpoint["a_optimizer"])
                print(f"[√] Loaded checkpoint! Alpha loaded.")
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = args.alpha
            self.log_alpha = None
            self.a_optimizer = None

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
            print(f"[√] Loaded checkpoint! Resuming from global_step {self.global_step}.")
        else:
            self.global_step = 0
        self.start_time = None
        self.reward_tracker = None
        self.csv_file_info = None
        self.csv_file_agent_vars= None
        self.writer_info = None
        self.writer_agent_vars = None
        self.keys_info = None
        self.keys_agent_vars = [
                                'qf1_values',
                                'qf2_values',
                                'qf1_losses',
                                'qf2_losses',
                                'qf_losses',
                                'actor_losses',
                                'alphas',
                                'alpha_losses',
                                'SPS',
                                'average_reward_per_second',
                                'reward']
        # Buffers for CSV data
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

    def learn(self, global_step=None):
        """Perform one learning step."""
        if global_step is None:
            global_step = self.global_step

        if global_step < self.learning_starts:
            return None, None, None, None, None, None, None

        # Initialize variables for logging.
        actor_loss = None
        alpha_loss = None

        # Sample from the replay buffer.
        data = self.rb.sample(self.args.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # next_q_value = data.rewards.flatten() * self.args.dt + (1 - data.dones.flatten()) * self.gamma_discrete * (min_qf_next_target).view(-1)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma_discrete * (min_qf_next_target).view(-1)
        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Optimize the Action-Value networks.
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.args.policy_frequency == 0:
            for _ in range(self.args.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf1_pi = self.qf1(data.observations, pi)
                qf2_pi = self.qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                # Optimize the Actor network.
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # Update the target networks.
        if global_step % self.args.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        return qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss

    def initialize_logging(self, info):
        """Initialize logging files and trackers."""
        self.start_time = time.time()

        # Log the information of choice.
        self.csv_file_info = open(os.path.join(self.disk_folder, self.runs_directory, self.run_name, "info_logs.csv"), "w", newline="")
        self.keys_info = list(info.keys())
        self.keys_info = [k for k in self.keys_info if not (k.startswith("bodies") or k.startswith("_"))]
        self.keys_info.append("reset")
        self.keys_info.append("qf1_values")
        self.keys_info.append("qf2_values")
        self.keys_info.append("qf_values")

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

    def log_step(self, global_step, infos, rewards, qf1_a_values=None, qf2_a_values=None,
                 qf1_loss=None, qf2_loss=None, qf_loss=None, actor_loss=None, alpha_loss=None):
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
        if all(x is not None for x in [qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss]):
            self.agent_vars_buffer.append({"step": global_step,
                                     "qf1_values": qf1_a_values.mean().item() if qf1_a_values is not None else None,
                                     "qf2_values": qf2_a_values.mean().item() if qf2_a_values is not None else None,
                                     "qf1_losses": qf1_loss.item() if qf1_loss is not None else None,
                                     "qf2_losses": qf2_loss.item() if qf2_loss is not None else None,
                                     "qf_losses": qf_loss.item() / 2.0 if qf_loss is not None else None,
                                     "actor_losses": actor_loss.item() if actor_loss is not None else None,
                                     "alphas": self.alpha,
                                    #  "log_alphas": self.log_alpha.item() if self.log_alpha is not None else None,
                                     "alpha_losses": alpha_loss.item() if alpha_loss is not None else None,
                                     "SPS": int(global_step / (time.time() - self.start_time)) if self.start_time else 0,
                                     "average_reward_per_second": self.reward_tracker.average_reward_per_second,
                                     "reward": rewards.item()})

        # Write to CSV every 1000 steps.
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
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "a_optimizer": self.a_optimizer.state_dict() if self.args.autotune else None,
            "log_alpha": self.log_alpha.detach().cpu() if self.args.autotune else None,
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
        """Main training loop - runs the SAC policy."""
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
            # 'env_time_to_get_obs': [],
            # 'env_time_to_check_errors': [],
            # 'env_time_to_recover_motor_errors': [],
            # 'env_sleep_duration': [],
            # 'env_time_imu_thread': [],
            # 'env_time_tracker_thread': [],
            # 'env_time_get_motor_feedback': [],
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
            infos['reset'] = truncations

            obs_torch = torch.from_numpy(obs).to(self.device).float()
            actions_torch = torch.from_numpy(actions).to(self.device).float()
            qf1_values = self.qf1(obs_torch, actions_torch)
            qf2_values = self.qf2(obs_torch, actions_torch)
            qf_values = torch.min(qf1_values, qf2_values)
            infos['qf1_values'] = np.array([qf1_values.item()])
            infos['qf2_values'] = np.array([qf2_values.item()])
            infos['qf_values'] = np.array([qf_values.item()])

            # for key in [
            #     'env_time_to_get_obs',
            #     'env_time_to_check_errors',
            #     'env_time_to_recover_motor_errors',
            #     'env_sleep_duration',
            #     'env_time_imu_thread',
            #     'env_time_tracker_thread',
            #     'env_time_get_motor_feedback'
            # ]:
            #     if key in infos:
            #         times[key].append(infos[key][0])
            #     else:
            #         times[key].append(None)

            # Add transition to buffer.
            time_start = time.time()
            self.add_transition(obs, next_obs, actions, rewards, terminations, infos)
            times['time_add_transition_buffer'].append(time.time() - time_start)

            # Update the observation.
            obs = next_obs

            # Learn.
            time_start = time.time()
            qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss = self.learn(global_step)
            times['time_learn'].append(time.time() - time_start)
            if times['time_learn'][-1] > 0.06:
                print(f"Time learn: {times['time_learn'][-1]}")
            # Log step.
            time_start = time.time()
            self.log_step(global_step, infos, rewards, qf1_a_values, qf2_a_values,
                         qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss)
            times['time_log_step'].append(time.time() - time_start)

            if global_step % self.args.save_every_n_steps == 0:
                # Save checkpoint.
                self.save_checkpoint(global_step)
                # Save the times to pandas
                df = pd.DataFrame(times)
                df.to_csv(os.path.join(self.disk_folder, self.runs_directory, self.run_name, "times.csv"), index=False)

        # Cleanup.
        self.cleanup()

def main():
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
            use_previous_action=args.use_previous_action,
        )
        print(f"BackAndForthTask initialized for radius: {RADIUS}, origin: {ORIGIN}, use_previous_action: {args.use_previous_action}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    # Create environment.
    envs = make_ant_envs(args, task, disk_folder, run_name, runs_directory=args.runs_directory)

    # Create SAC agent.
    agent = SAC(args, envs, disk_folder=disk_folder, run_name=run_name, runs_directory=args.runs_directory)

    # Run the policy.
    agent.run_policy()

if __name__ == "__main__":
    main()
