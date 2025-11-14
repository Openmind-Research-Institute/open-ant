# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)
# Modified by Sorina Lupu, Openmind Research Institute, 2025

import os
import time
import sys
import tyro
import json
import random
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from buffers import ReplayBuffer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

# Matplotlib font setup.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "EmbodiedAnt"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "EmbodiedAnt"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of learning policy (delayed update)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    dt: float = 0.05
    """the timestep of the environment"""
    hw_config: str = None
    """the hardware configuration file"""
    render_mode: str = "human"
    """the render mode"""
    terminate_on_upside_down: bool = True
    """whether to terminate the episode if the agent is upside down"""
    weights_path: str = None
    """previously learned weights"""
    task_type: str = "forward"
    """the type of task"""
    reward_scale: float = 100.0
    """the reward scale"""

# ALGO LOGIC: initialize agent here:
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_NAME = f"{args.env_id}__{args.exp_name}__{args.seed}__{date}"
    WEIGHTS_FOLDER = os.path.join("runs", RUN_NAME, "weights_and_args")
    os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
    REPORT_FOLDER = os.path.join("runs", RUN_NAME, "report")
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    with open(os.path.join(WEIGHTS_FOLDER, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    if args.track: # TODO: test this.
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=RUN_NAME,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{RUN_NAME}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    render = args.render_mode
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        radius = 0.61
        origin = np.array([0.14194049, -0.82257924])
        task = BackAndForthTask(
            radius=radius,
            origin=origin,
        )
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    def make_env(env_id, seed, idx, capture_video, RUN_NAME):
        def thunk():
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(30),
                'knee_range': np.radians(20),
            }
            if args.hw_config is None:
                env = AntEnv(
                    dt=args.dt,
                    render_mode=render,
                    terminate_on_upside_down=args.terminate_on_upside_down,
                    task=task,
                    joint_config=joint_config,
                )
            else:
                with open(args.hw_config, 'r') as f:
                    cfg = json.load(f)
                env = make_ant_env(cfg, render_mode=render,
                                   dt=args.dt,
                                   joint_config=joint_config,
                                   task=task,
                                   )
            
            if capture_video and idx == 0:
                print('RecordVideo')
                env = gym.wrappers.RecordVideo(env, f"runs/{RUN_NAME}/videos/{RUN_NAME}", episode_trigger=lambda x: x % 10 == 0)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.TransformReward(env, lambda r: r * args.reward_scale)
            env.action_space.seed(seed)
            return env

        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, RUN_NAME) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

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
    if args.weights_path is not None:
        # Load replay buffer
        buffer_path = os.path.join(args.weights_path, "replay_buffer.npz")
        if os.path.exists(buffer_path):
            rb.load(buffer_path, device)
            print(f"[√] Loaded replay buffer with {rb.size} transitions")
        else:
            print("[!] No replay buffer found, starting empty")
    start_time = time.time()

    obs, info = envs.reset(seed=args.seed)

    info_logs = open(os.path.join("runs", RUN_NAME, "info_logs.csv"), 'w')
    keys_to_record = ['step', 'roll_deg', 'pitch_deg', 'yaw_deg', 'timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'joint_positions', 'joint_velocities', 'joint_loads', 'temperatures', 'current_x_position', 'current_y_position', 'heading_vector', 'heading_vector_x', 'heading_vector_y', 'reward_direction', 'reward_direction_x', 'reward_direction_y', 'r_b_x', 'r_b_y', 'original_reward']
    # Record if they exist in info.
    keys_to_record_that_exist = [k for k in keys_to_record if k in info.keys()]
    info_logs.write('step, ' + ','.join(keys_to_record_that_exist) + '\n')
    info_logs.flush()

    reward_tracker = RewardTracker(env_dt=args.dt, env_id=args.env_id,
                            log_folder=os.path.join("runs", RUN_NAME),
                            time_window=120.0)

    # Debugging variables.
    dict_debugging = {}
    dict_debugging['episodic_returns'] = []
    dict_debugging['episodic_lengths'] = []
    dict_debugging['episodic_step'] = []
    dict_debugging['steps'] = []
    dict_debugging['qf1_values'] = []
    dict_debugging['qf2_values'] = []
    dict_debugging['qf1_losses'] = []
    dict_debugging['qf2_losses'] = []
    dict_debugging['qf_losses'] = []
    dict_debugging['actor_losses'] = []
    dict_debugging['alphas'] = []
    dict_debugging['alpha_losses'] = []
    dict_debugging['SPS'] = []
    dict_debugging['average_reward_per_second'] = []

    for global_step in tqdm(range(args.total_timesteps)):
        # Get the action.
        if global_step < args.learning_starts and args.weights_path is None:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # Step the environment.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log the information for a single environment. Ignore those that start with _
        infos_to_log = {}
        for k, v in infos.items():
            # if it doesn't start with _ and is not a dict
            if k in keys_to_record_that_exist:
                # Check if v is not empty and has the expected structure
                v_env_idx_0 = v[0]
                if isinstance(v_env_idx_0, (list, tuple, np.ndarray)):
                    formatted_value = "[" + " ".join(f"{x:.6f}" for x in np.array(v_env_idx_0).flatten()) + "]"
                else:
                    formatted_value = str(v_env_idx_0)
            infos_to_log[k] = formatted_value
        info_logs.write(f"{global_step}, " + ", ".join(infos_to_log.values()) + "\n")
        info_logs.flush()

        original_rewards = infos['original_reward']

        if "episode" in infos:
            if infos["episode"] is not None:
                print('infos["episode"]', infos["episode"])
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
                dict_debugging['episodic_returns'].append(infos["episode"]["r"])
                dict_debugging['episodic_lengths'].append(infos["episode"]["l"])
                dict_debugging['episodic_step'].append(global_step)

        # Add the data to the replay buffer.
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # Update the reward tracker.
        if args.num_envs == 1:
            reward_tracker.update(rewards.item())
            reward_tracker.log()
        else:
            raise ValueError("reward_tracker is only supported for single environment")

        obs = next_obs

        # Learning.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() * args.dt + (1 - data.dones.flatten()) * (args.gamma ** args.dt) * (min_qf_next_target).view(-1)
                # see K. de Asis, R. Sutton, "An Idiosyncrasy of Time-discretization in RL").
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
                for _ in range(
                    args.policy_frequency
                ):  # Compensate for the delay by doing 'actor_update_interval' instead of 1.
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

            if global_step % 100 == 0:

                # Save all the networks.
                checkpoint = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "a_optimizer": a_optimizer.state_dict() if args.autotune else None,
                    "log_alpha": log_alpha.detach().cpu(),
                    "global_step": global_step,
                    "random_state": random.getstate(),
                    "numpy_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                }
                torch.save(checkpoint, os.path.join(WEIGHTS_FOLDER, "checkpoint.pth"))
                rb.save(os.path.join(WEIGHTS_FOLDER, "replay_buffer.npz"))

                dict_debugging['steps'].append(global_step)
                dict_debugging['qf1_values'].append(qf1_a_values.mean().item())
                dict_debugging['qf2_values'].append(qf2_a_values.mean().item())
                dict_debugging['qf1_losses'].append(qf1_loss.item())
                dict_debugging['qf2_losses'].append(qf2_loss.item())
                dict_debugging['qf_losses'].append(qf_loss.item() / 2.0)
                dict_debugging['actor_losses'].append(actor_loss.item())
                dict_debugging['alphas'].append(alpha)
                dict_debugging['alpha_losses'].append(alpha_loss.item())
                dict_debugging['SPS'].append(int(global_step / (time.time() - start_time)))
                dict_debugging['average_reward_per_second'].append(reward_tracker.average_reward_per_second)

                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/average_reward_per_second", reward_tracker.average_reward_per_second, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                with PdfPages(os.path.join(REPORT_FOLDER, "report.pdf")) as pdf:
                    # Plot the average reward per second.
                    fig = plt.figure()
                    plt.plot(dict_debugging['steps'], dict_debugging['average_reward_per_second'], linewidth=2)
                    plt.xlabel('Steps')
                    plt.ylabel('Average Reward per Second')
                    plt.title('Average Reward per Second')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

                    # Plot the episodic returns and lengths.
                    if len(dict_debugging['episodic_returns']) > 0:
                        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                        ax[0].plot(dict_debugging['episodic_step'], dict_debugging['episodic_returns'])
                        ax[0].set_xlabel('Steps')
                        ax[0].set_ylabel('Episodic Returns')
                        ax[0].set_title('Episodic Returns')
                        ax[1].plot(dict_debugging['episodic_step'], dict_debugging['episodic_lengths'])
                        ax[1].set_xlabel('Steps')
                        ax[1].set_ylabel('Episodic Lengths')
                        ax[1].set_title('Episodic Lengths')
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

                    # Plot the other metrics.
                    for key, value in dict_debugging.items():
                        if key.startswith('episodic'):
                            continue
                        fig = plt.figure()
                        plt.plot(dict_debugging['steps'], value)
                        plt.xlabel('Steps')
                        plt.ylabel(key)
                        plt.title(key)
                        pdf.savefig()
                        plt.close()

                    # Open the info and plot all columns.
                    df_logs = pd.read_csv(os.path.join("runs", RUN_NAME, "info_logs.csv"))
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
                    # fig = plt.figure()
                    # duration = 20 # seconds
                    # length_plot = int(duration/args.dt)
                    # if 'current_x_position' in df_logs.columns and 'current_y_position' in df_logs.columns:
                    #     plt.plot(df_logs['current_x_position'][-length_plot:], df_logs['current_y_position'][-length_plot:])
                    # selected_indices = np.arange(len(df_logs) - length_plot, len(df_logs))[::10]
                    # if 'current_x_position' in df_logs.columns and 'current_y_position' in df_logs.columns and 'heading_vector_x' in df_logs.columns and 'heading_vector_y' in df_logs.columns:
                    #     plt.quiver(df_logs['current_x_position'][selected_indices], df_logs['current_y_position'][selected_indices], df_logs['heading_vector_x'][selected_indices], df_logs['heading_vector_y'][selected_indices], color='black', width=0.01, scale=30, zorder=3)
                    # if 'reward_direction_x' in df_logs.columns and 'reward_direction_y' in df_logs.columns:
                    #     plt.quiver(df_logs['current_x_position'][selected_indices], df_logs['current_y_position'][selected_indices], df_logs['reward_direction_x'][selected_indices], df_logs['reward_direction_y'][selected_indices], color='red', width=0.01, scale=30, zorder=2)

                    # # Draw a circle if the task is back and forth
                    # if args.task_type == "back_and_forth":
                    #     plt.plot(origin[0], origin[1], 'ro')
                    #     plt.plot(origin[0] + radius*np.cos(np.linspace(0, 2*np.pi, 100)), origin[1] + radius*np.sin(np.linspace(0, 2*np.pi, 100)))
                    # plt.xlabel('X')
                    # plt.ylabel('Y')
                    # plt.gca().set_aspect('equal', adjustable='box')
                    # plt.title('Current Position')
                    # folder_arena_plots = os.path.join("runs", RUN_NAME, "arena_plots")
                    # os.makedirs(folder_arena_plots, exist_ok=True)
                    # plt.savefig(os.path.join(folder_arena_plots, f"arena_plot_{global_step}.png"), dpi=300)
                    # pdf.savefig()
                    # plt.close()

    # Close the environment and the writer.
    envs.close()
    writer.close()