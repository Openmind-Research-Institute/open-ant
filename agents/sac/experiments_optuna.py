# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)
# Modified by Sorina Lupu, Openmind Research Institute, 2025

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import optuna

# Import SAC implementation from sac_cleanrl
from sac_cleanrl import SAC, make_ant_envs

# Import task classes directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import ForwardTask, BackAndForthTask

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
    parser.add_argument("--total_timesteps", type=int, default=30_000,
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
    parser.add_argument("--gradient_updates", type=int, default=1,
                        help="number of gradient updates")

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

    # Optuna
    parser.add_argument("--optuna_study_name", type=str, default=None,
                        help="Optuna study name for hyperparameter optimization")
    parser.add_argument("--optuna_n_trials", type=int, default=50,
                        help="number of Optuna trials")
    parser.add_argument("--optuna_storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")

    args = parser.parse_args()
    return args


def train_sac(trial, args):
    """Train SAC agent and return average reward for Optuna optimization."""
    
    # Suggest hyperparameters from Appendix D in SAC paper https://arxiv.org/pdf/1801.01290
    if trial is not None:
        args.policy_lr = trial.suggest_float("policy_lr", 1e-5, 1e-2, log=True)
        args.q_lr = trial.suggest_float("q_lr", 1e-5, 1e-2, log=True)
        args.batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
        args.gamma = trial.suggest_float("gamma", 0.9, 0.999)
        args.tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
        args.autotune = trial.suggest_categorical("autotune", [True, False])
        # args.gradient_updates = trial.suggest_int("gradient_updates", 1, 4, step=1)
        if not args.autotune:
            args.alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_id = trial.number if trial is not None else 0
    run_name = f"{args.env_id}__{args.exp_name}_{date}__trial_{trial_id}"
    
    # Folders.
    disk_folder = ''
    
    # Task.
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = 0.55
        ORIGIN = np.array([-1.05668516,  0.00237455])
        task = BackAndForthTask(radius=RADIUS, origin=ORIGIN)
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    # Create environment.
    envs = make_ant_envs(args, task, disk_folder, run_name)

    # Create SAC agent.
    agent = SAC(args, envs, disk_folder=disk_folder, run_name=run_name)
    
    # Update hyperparameters if they were changed by Optuna
    agent._update_hyperparams(args)
    
    # Reset the environment.
    obs, info = envs.reset(seed=args.seed)
    
    # Initialize logging.
    agent.initialize_logging(info)

    # Start learning with manual control for Optuna reporting.
    for global_step in tqdm(range(args.total_timesteps), desc=f"Trial {trial_id}"):
        agent.global_step = global_step

        # Get the action.
        actions = agent.get_action(obs, global_step)

        # Step the environment.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Add transition to buffer.
        agent.add_transition(obs, next_obs, actions, rewards, terminations, infos)

        # Learn - perform multiple gradient updates if specified.
        qf1_a_values = None
        qf2_a_values = None
        qf1_loss = None
        qf2_loss = None
        qf_loss = None
        actor_loss = None
        alpha_loss = None
        
        if global_step >= agent.learning_starts:
            for _ in range(args.gradient_updates):
                qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss = agent.learn(global_step)

        # Log step.
        agent.log_step(global_step, infos, rewards, qf1_a_values, qf2_a_values,
                      qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss)

        # Save checkpoint.
        agent.save_checkpoint(global_step)

        # Update the observation.
        obs = next_obs

        # Report intermediate value to Optuna
        if trial is not None and global_step % 10000 == 0:
            metrics = agent.get_metrics()
            if metrics is not None:
                trial.report(metrics["average_reward_per_second"], global_step)
                if trial.should_prune():
                    agent.cleanup()
                    raise optuna.TrialPruned()

    # Get final average reward
    metrics = agent.get_metrics()
    final_average_reward = metrics["average_reward_per_second"] if metrics is not None else 0.0
    
    # Cleanup.
    agent.cleanup()
    
    return final_average_reward


if __name__ == "__main__":
    args = parse_args()
    
    # Define Optuna objective function
    def objective(trial):
        return train_sac(trial, args)
    
    # Set up Optuna study
    if args.optuna_study_name:
        study = optuna.create_study(
            study_name=args.optuna_study_name,
            direction="maximize",
            storage=args.optuna_storage,
            load_if_exists=True,
        )
        print(f"Created/loaded Optuna study: {args.optuna_study_name}")
        print(f"Running {args.optuna_n_trials} trials...")
        study.optimize(objective, n_trials=args.optuna_n_trials)
        
        print("\n=== Optuna Study Results ===")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial:")
        trial = study.best_trial
        print(f"  Value (average reward): {trial.value}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        # Run single training without Optuna
        print("Running single training without Optuna optimization...")
        average_reward = train_sac(None, args)
        print(f"Final average reward: {average_reward}")
