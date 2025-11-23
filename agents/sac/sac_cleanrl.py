# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)
# Modified by Sorina Lupu, Openmind Research Institute, 2025

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    # General
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-3],
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True,
                        help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture_video", type=bool, default=False,
                        help="capture video of agent performances")

    # Algorithm
    parser.add_argument("--env_id", type=str, default="EAnt",
                        help="environment ID")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="total training timesteps")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="number of parallel envs")
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="replay buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
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
    parser.add_argument("--policy_frequency", type=int, default=2,
                        help="policy update frequency")
    parser.add_argument("--target_network_frequency", type=int, default=1,
                        help="target network update frequency")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="entropy regularization coefficient")
    parser.add_argument("--autotune", type=bool, default=True,
                        help="automatic entropy tuning")

    # Environment
    parser.add_argument("--dt", type=float, default=0.05,
                        help="environment timestep")
    parser.add_argument("--hw_config", type=str, default=None,
                        help="hardware config file")
    parser.add_argument("--render_mode", type=str, default="human",
                        help="render mode")
    parser.add_argument("--terminate_on_upside_down", type=bool, default=True,
                        help="terminate episode if upside down")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="load previous weights")
    parser.add_argument("--task_type", type=str, default="forward",
                        choices=["forward", "back_and_forth"],
                        help="type of task")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                        help="reward scale factor")

    args = parser.parse_args()
    return args

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


class Experiment:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.params = json.load(f)

        # Separate keys and value lists
        keys = list(self.params.keys())
        values = [v if isinstance(v, list) else [v] for v in self.params.values()]

        # Compute cross product of hyper-parameters
        self.configs = []
        for combo in itertools.product(*values):
            self.configs.append(dict(zip(keys, combo)))

        print("Total experiment configurations:", len(self.configs))

    def get_params(self, idx):
        return self.configs[idx]


if __name__ == "__main__":

    date = datetime.now().strftime("%Y%m%d-%H%M%S")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, help="Path to config file")
    # parser.add_argument("--run", type=int, default=1, help="Run number (int)")

    # args_experiment = parser.parse_args()
    # if not args_experiment.config:
    #     print("--config parameter must be passed")
    #     exit(1)

    # my_exp = Experiment(args_experiment.config)
    # hyper_parameters = my_exp.get_params(args_experiment.run)
    # print("Total experiment configurations:", len(my_exp.configs))
    # print(hyper_parameters)

    # # Retain old-style namespace object for backward compatibility:
    # class ArgsObject:
    #     def __init__(self, d):
    #         self.__dict__.update(d)
    # # Combine loaded hyperparameters and the parser CLI args (expt/run).
    # args_dict = dict(hyper_parameters)
    # args_dict.update(vars(args_experiment))
    # args = ArgsObject(args_dict)
    # print(args)
    
    args = parse_args()
    hyper_parameters = args.__dict__

    # Folders.
    # disk_folder = '/mnt/ramdisk/'
    disk_folder = ''
    # Make RUN_NAME unique by including process PID.
    this_pid = os.getpid()
    RUN_NAME = f"{hyper_parameters['env_id']}__{hyper_parameters['exp_name']}_{date}__pid_{this_pid}"
    WEIGHTS_FOLDER = os.path.join(disk_folder, "runs", RUN_NAME, "weights_and_args")
    os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
    REPORT_FOLDER = os.path.join(disk_folder, "runs", RUN_NAME, "report")
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    with open(os.path.join(WEIGHTS_FOLDER, "args.json"), 'w') as f:
        json.dump(args.__dict__, f)

    # Seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device.
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Task.
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        # Radius and origin for the circular movement boundary.
        RADIUS = 0.55
        ORIGIN = np.array([-1.05668516,  0.00237455])
        task = BackAndForthTask(
            radius=RADIUS,
            origin=ORIGIN,
        )
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    def make_env(env_id, seed, idx, capture_video, RUN_NAME):
        def _init():
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(30),
                'knee_range': np.radians(20),
            }
            if args.hw_config is None:
                env = AntEnv(
                    dt=args.dt,
                    render_mode=args.render_mode,
                    terminate_on_upside_down=args.terminate_on_upside_down,
                    task=task,
                    joint_config=joint_config,
                    xml_file=os.path.join(os.path.dirname(__file__), '../../sim/assets/ant_position_with_camera.xml'),
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
                env = gym.wrappers.RecordVideo(env, os.path.join(disk_folder, "runs", RUN_NAME, "videos", RUN_NAME), episode_trigger=lambda x: x % 50 == 0)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.TransformReward(env, lambda r: r * args.reward_scale)
            env.action_space.seed(seed)
            return env

        return _init

    # Vector environment.
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, RUN_NAME) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Networks.
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    LEARNING_STARTS = args.learning_starts

    # Checkpoints.
    checkpoint = None
    if args.weights_path is not None:
        checkpoint = torch.load(os.path.join(args.weights_path, "checkpoint.pth"), map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        qf1.load_state_dict(checkpoint["qf1"])
        qf2.load_state_dict(checkpoint["qf2"])
        qf1_target.load_state_dict(checkpoint["qf1_target"])
        qf2_target.load_state_dict(checkpoint["qf2_target"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        LEARNING_STARTS = 0.0 # Start learning from the checkpoint.
        print(f"[√] Loaded checkpoint with {LEARNING_STARTS} learning starts")

    if args.autotune:
        # Automatic entropy tuning.
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

        if checkpoint is not None:
            a_optimizer.load_state_dict(checkpoint["a_optimizer"])
            log_alpha = checkpoint["log_alpha"].to(device).requires_grad_()
        alpha = log_alpha.exp().item()
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    # Initialize the replay buffer.
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    # Load replay buffer if there exists one.
    if args.weights_path is not None:
        buffer_path = os.path.join(args.weights_path, "replay_buffer.npz")
        if os.path.exists(buffer_path):
            rb.load(buffer_path, device)
            print(f"[√] Loaded replay buffer with {rb.size} transitions")
        else:
            print("[!] No replay buffer found, starting empty")

    start_time = time.time()

    # Reset the environment.
    obs, info = envs.reset(seed=args.seed)

    # Log the information of choice.
    csv_file = open(os.path.join(disk_folder, "runs", RUN_NAME, "info_logs.csv"), "w", newline="")
    keys_info = list(info.keys())
    # remove the keys start have bodies
    keys_info = [k for k in keys_info if not k.startswith("bodies")]
    writer = csv.DictWriter(csv_file, fieldnames=["step"] + keys_info)
    writer.writeheader()

    # Reward tracker.
    reward_tracker = RewardTracker(env_dt=args.dt, env_id=args.env_id,
                            log_folder=os.path.join(disk_folder, "runs", RUN_NAME),
                            time_window=120.0)

    # Performance variables.
    keys_perf_variables = ['episodic_returns', 'episodic_lengths', 'episodic_step', 'steps', 'qf1_values', 'qf2_values', 'qf1_losses', 'qf2_losses', 'qf_losses', 'actor_losses', 'alphas', 'alpha_losses', 'SPS', 'average_reward_per_second']
    csv_perf_file = open(os.path.join(disk_folder, "runs", RUN_NAME, "performance_variables.csv"), "w", newline="")
    writer_perf = csv.DictWriter(csv_perf_file, fieldnames=["step"] + keys_perf_variables)
    writer_perf.writeheader()

    # Start learning.
    for global_step in tqdm(range(args.total_timesteps)):
        # Get the action.
        if global_step < LEARNING_STARTS:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # Step the environment.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log the episodic return and length.
        if "episode" in infos:
            if infos["episode"] is not None:
                print('infos["episode"]', infos["episode"])
                writer_perf.writerow({"step": global_step, "episodic_returns": infos["episode"]["r"][0], "episodic_lengths": infos["episode"]["l"][0], "episodic_step": global_step})
                csv_perf_file.flush()

        # Add the data to the replay buffer.
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # Update the reward tracker.
        if args.num_envs == 1:
            reward_tracker.update(rewards.item())
            reward_tracker.log()
        else:
            raise ValueError("reward_tracker is only supported for single environment")

        # Update the observation.
        obs = next_obs

        # Log the infos.
        infos_to_log = {}
        for k, v in infos.items():
            if k in keys_info:
               infos_to_log[k] = arr_to_str(v[0])
        row = {"step": global_step, **infos_to_log}
        writer.writerow(row)
        csv_file.flush()

        # Learn.
        if global_step >= LEARNING_STARTS:
            # Sample from the replay buffer.
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() * args.dt + (1 - data.dones.flatten()) * (args.gamma ** args.dt) * (min_qf_next_target).view(-1) # see K. de Asis, R. Sutton, "An Idiosyncrasy of Time-discretization in RL").
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Optimize the Action-Value networks.
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):  # Compensate for the delay by doing 'actor_update_interval' instead of 1.
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    # Optimize the Actor network.
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # Update the target networks.
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # ==== Everything below is for logging and saving. ====
            # Logging.
            if global_step % 1000 == 0:
                # # Save all the networks.
                checkpoint = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "a_optimizer": a_optimizer.state_dict() if args.autotune else None,
                    "log_alpha": log_alpha.detach().cpu() if args.autotune else None,
                    "global_step": global_step,
                    "random_state": random.getstate(),
                    "numpy_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                }
                torch.save(checkpoint, os.path.join(WEIGHTS_FOLDER, "checkpoint.pth"))

                # # Save the replay buffer.
                rb.save(os.path.join(WEIGHTS_FOLDER, "replay_buffer.npz"))

                # # Log performance.
                writer_perf.writerow({"step": global_step,
                                     "qf1_values": qf1_a_values.mean().item(),
                                     "qf2_values": qf2_a_values.mean().item(),
                                     "qf1_losses": qf1_loss.item(),
                                     "qf2_losses": qf2_loss.item(),
                                     "qf_losses": qf_loss.item() / 2.0,
                                     "actor_losses": actor_loss.item(),
                                     "alphas": alpha,
                                     "alpha_losses": alpha_loss.item() if args.autotune else None,
                                     "SPS": int(global_step / (time.time() - start_time)), "average_reward_per_second": reward_tracker.average_reward_per_second})
                csv_perf_file.flush()

                # Plot the performance variables.
                with PdfPages(os.path.join(REPORT_FOLDER, "report.pdf")) as pdf:
                    # Plot the average reward per second.
                    reward_tracker.plot(os.path.join(REPORT_FOLDER, "average_reward.pdf"))

                    # # Plot the writer_perf data.
                    df_perf = pd.read_csv(os.path.join(disk_folder, "runs", RUN_NAME, "performance_variables.csv"))
                    # # Extract all the episodic variables and plot them.
                    df_episodic = df_perf[df_perf['step'].isin(df_perf['episodic_step'])]
                    for key in keys_perf_variables:
                       if key.startswith('episodic_'):
                           fig = plt.figure()
                           plt.plot(df_episodic['episodic_step'], df_episodic[key])
                           plt.xlabel('Episodic Step')
                           plt.ylabel(key)
                           plt.title(key)
                           pdf.savefig()
                           plt.close()

                    for key in keys_perf_variables:
                       if key not in df_perf.columns or key.startswith('episodic_'):
                           continue
                       fig = plt.figure()
                       plt.plot(df_perf['step'], df_perf[key])
                       plt.xlabel('Steps')
                       plt.ylabel(key)
                       plt.title(key)
                       pdf.savefig()
                       plt.close()

                    df_logs = pd.read_csv(os.path.join(disk_folder, "runs", RUN_NAME, "info_logs.csv"))
                    cols_to_plot = [
                       'current_x_position',
                       'current_y_position',
                       'heading_vector_x',
                       'heading_vector_y',
                       'original_reward',
                       'r_b_x',
                       'r_b_y',
                    ]
                    for col in cols_to_plot:
                       # Check if cols exists in df_logs
                       if col not in df_logs.columns:
                           continue
                       fig = plt.figure()
                       plt.plot(df_logs['step'], df_logs[col])
                       plt.xlabel('Steps')
                       plt.ylabel(col)
                       plt.title(col)
                       pdf.savefig()
                       plt.close()

                    # Create an arena plot with the heading vector and the reward direction.
                    fig = plt.figure()
                    duration = 20 # seconds
                    length_plot = int(duration/args.dt)
                    if 'current_x_position' in df_logs.columns and 'current_y_position' in df_logs.columns:
                        plt.plot(df_logs['current_x_position'][-length_plot:], df_logs['current_y_position'][-length_plot:])
                    selected_indices = np.arange(len(df_logs) - length_plot, len(df_logs))[::10]
                    if 'current_x_position' in df_logs.columns and 'current_y_position' in df_logs.columns and 'heading_vector_x' in df_logs.columns and 'heading_vector_y' in df_logs.columns:
                        plt.quiver(df_logs['current_x_position'][selected_indices], df_logs['current_y_position'][selected_indices], df_logs['heading_vector_x'][selected_indices], df_logs['heading_vector_y'][selected_indices], color='black', width=0.01, scale=30, zorder=3)
                    if 'reward_direction_x' in df_logs.columns and 'reward_direction_y' in df_logs.columns:
                        plt.quiver(df_logs['current_x_position'][selected_indices], df_logs['current_y_position'][selected_indices], df_logs['reward_direction_x'][selected_indices], df_logs['reward_direction_y'][selected_indices], color='red', width=0.01, scale=30, zorder=2)

                    # Draw a circle if the task is back and forth
                    if args.task_type == "back_and_forth":
                        plt.plot(ORIGIN[0], ORIGIN[1], 'ro')
                        plt.plot(ORIGIN[0] + RADIUS*np.cos(np.linspace(0, 2*np.pi, 100)),
                                 ORIGIN[1] + RADIUS*np.sin(np.linspace(0, 2*np.pi, 100)))
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.title('Current Position')
                    folder_arena_plots = os.path.join("runs", RUN_NAME, "arena_plots")
                    os.makedirs(folder_arena_plots, exist_ok=True)
                    plt.savefig(os.path.join(folder_arena_plots, f"arena_plot_{global_step}.png"), dpi=300)
                    pdf.savefig()
                    plt.close()

    # Close the environment.
    envs.close()
