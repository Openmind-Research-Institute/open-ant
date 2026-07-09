"""
Optuna hyperparameter optimization for SARSA(λ) with Tile Coding.
Searches for hyperparameters that maximize per-episode return.
"""

import os
import sys
import numpy as np
import optuna
from optuna.samplers import TPESampler
import json
import datetime

# Custom imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from tilecoding import IHT, tiles
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import ForwardTask, BackAndForthTask
np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)

# ============================================================================
# Configuration
# ============================================================================
SEED = 0
N_SEEDS = 10  # Number of seeds to average over
DT = 0.05
RENDER_MODE = "rgb_array"
N_TIMELIMIT_EPISODES_PER_TRIAL = 800  # Number of timelimit episodes per trial
N_OPTUNA_TRIALS = 100  # Total number of Optuna trials
MAX_OPTIONS_PER_TIMELIMIT_EPISODE = 30  # Fixed, not tuned
STUDY_NAME = "sarsa_optuna_study"
STORAGE_PATH = "optuna_study.db"

# Task configuration: 'forward' or 'back_and_forth'
TASK_TYPE = "back_and_forth"  # Change to "back_and_forth" to use BackAndForthTask
BACK_AND_FORTH_RADIUS = 1.0  # Radius for BackAndForthTask
BACK_AND_FORTH_ORIGIN = [0.0, 0.0]

# Observation ranges file path (set to None to use default hardcoded limits)
OBSERVATION_RANGES_PATH = 'runs_sarsa_sim/observation_ranges.npy'

# ============================================================================
# Helper functions
# ============================================================================

def linear_ramp(start_pos: float, end_pos: float, duration: float, dt: float):
    num = round(duration / dt)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list


class OptionEnv:
    def __init__(self, env, options, discount, dt, reward_scaling):
        self.env = env
        self.options = options
        self.discount = discount
        self.dt = dt
        self.reward_scaling = reward_scaling
        self.joints_dict = {
            'hip_rr': {'current_pos': 0.0, 'traj': None},
            'knee_rr': {'current_pos': 0.0, 'traj': None},
            'hip_fr': {'current_pos': 0.0, 'traj': None},
            'knee_fr': {'current_pos': 0.0, 'traj': None},
            'hip_fl': {'current_pos': 0.0, 'traj': None},
            'knee_fl': {'current_pos': 0.0, 'traj': None},
            'hip_rl': {'current_pos': 0.0, 'traj': None},
            'knee_rl': {'current_pos': 0.0, 'traj': None}
        }
        self.info = None

    def step(self, option_idx: int):
        opt = self.options['option_' + str(option_idx)]

        for joint_name in opt['joint_names']:
            if joint_name.startswith('hip'):
                self.joints_dict[joint_name]['traj'] = linear_ramp(
                    self.joints_dict[joint_name]['current_pos'],
                    opt['joint_names'][joint_name]['hip_target'],
                    opt['duration'], self.dt
                )
            if joint_name.startswith('knee'):
                num_steps = int(opt['duration'] / self.dt)
                time = np.linspace(0, opt['duration'], num_steps)
                self.joints_dict[joint_name]['traj'] = (
                    opt['joint_names'][joint_name]['knee_amplitude'] *
                    np.sin(np.pi * time / opt['duration'])
                )

        total_reward = 0.0
        gamma_i = 1.0
        action_vector = np.zeros(self.env.action_space.shape[0])
        
        for i in range(int(opt['duration'] / self.dt)):
            for idx, joint_name in enumerate(self.joints_dict):
                if self.joints_dict[joint_name]['traj'] is not None:
                    action_vector[idx] = self.joints_dict[joint_name]['traj'][i]

            obs, reward, terminated, truncated, self.info = self.env.step(action_vector)
            reward *= self.reward_scaling
            total_reward += gamma_i * reward
            gamma_i *= self.discount

            if terminated or truncated:
                return obs, total_reward, terminated, truncated

        for joint_name in self.joints_dict:
            if self.joints_dict[joint_name]['traj'] is not None:
                self.joints_dict[joint_name]['current_pos'] = self.joints_dict[joint_name]['traj'][-1]
            self.joints_dict[joint_name]['traj'] = None

        return obs, total_reward, terminated, truncated

    def reset(self, seed=None):
        for joint_name in self.joints_dict:
            self.joints_dict[joint_name]['current_pos'] = 0.0
            self.joints_dict[joint_name]['traj'] = None
        return self.env.reset(seed=seed)

    def duration_steps(self, option_idx: int):
        opt = self.options['option_' + str(option_idx)]
        return int(opt['duration'] / self.dt)


# Tile coding wrapper
class SuttonTileCoderWrapper:
    def __init__(self, iht: IHT, tiles_per_dim, value_limits, tilings):
        self.iht = iht
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=np.int32)
        self.tilings = int(tilings)
        self.limits = np.asarray(value_limits, dtype=np.float64)
        self.scaling = np.array(tiles_per_dim) / (self.limits[:, 1] - self.limits[:, 0])
        assert self.limits.shape == (self.tiles_per_dim.shape[0], 2)

    def __getitem__(self, x):
        x = np.asarray(x, dtype=np.float64)
        normalized_x = (x - self.limits[:, 0]) * self.scaling
        idxs = tiles(self.iht, self.tilings, normalized_x)
        return np.asarray(idxs, dtype=np.int64)

    @property
    def n_tiles(self):
        return self.iht.size


def q_of(w, idx, o):
    return w[o, idx].sum()


def select_greedy_option(w, T, state, num_options):
    idx = T[state]
    q_vals = np.array([w[o, idx].sum() for o in range(num_options)], dtype=np.float64)
    maxq = q_vals.max()
    best = np.flatnonzero(q_vals == maxq)
    return int(np.random.choice(best)), q_vals


def select_option_epsilon_greedy(S, epsilon, w, T, num_options):
    if np.random.rand() < epsilon:
        return np.random.randint(num_options)
    O_greedy, _ = select_greedy_option(w, T, S, num_options)
    return O_greedy


def create_options(duration_option):
    """Create options dictionary with specified duration."""
    motions = {
        "knee_sinusoid_up": {"knee_amplitude": 1.0},
        "knee_sinusoid_down": {"knee_amplitude": -1.0},
        "hip_forward": {"hip_target": 1.0},
        "hip_backward": {"hip_target": -1.0},
    }
    
    options = {
        'option_0': {
            'duration': duration_option,
            'joint_names': {
                'hip_rr': motions['hip_forward'],
                'knee_rr': motions['knee_sinusoid_down'],
                'hip_fr': motions['hip_backward'],
                'knee_fr': motions['knee_sinusoid_up'],
                'hip_fl': motions['hip_backward'],
                'knee_fl': motions['knee_sinusoid_down'],
                'hip_rl': motions['hip_forward'],
                'knee_rl': motions['knee_sinusoid_up'],
            }
        },
        'option_1': {
            'duration': duration_option,
            'joint_names': {
                'hip_rr': motions['hip_backward'],
                'knee_rr': motions['knee_sinusoid_up'],
                'hip_fr': motions['hip_forward'],
                'knee_fr': motions['knee_sinusoid_down'],
                'hip_fl': motions['hip_forward'],
                'knee_fl': motions['knee_sinusoid_up'],
                'hip_rl': motions['hip_backward'],
                'knee_rl': motions['knee_sinusoid_down'],
            }
        },
        'option_2': {
            'duration': duration_option,
            'joint_names': {
                'hip_rr': motions['hip_backward'],
                'knee_rr': motions['knee_sinusoid_down'],
                'hip_fr': motions['hip_forward'],
                'knee_fr': motions['knee_sinusoid_up'],
                'hip_fl': motions['hip_forward'],
                'knee_fl': motions['knee_sinusoid_down'],
                'hip_rl': motions['hip_backward'],
                'knee_rl': motions['knee_sinusoid_up'],
            },
        },
        'option_3': {
            'duration': duration_option,
            'joint_names': {
                'hip_rr': motions['hip_forward'],
                'knee_rr': motions['knee_sinusoid_up'],
                'hip_fr': motions['hip_backward'],
                'knee_fr': motions['knee_sinusoid_down'],
                'hip_fl': motions['hip_backward'],
                'knee_fl': motions['knee_sinusoid_up'],
                'hip_rl': motions['hip_forward'],
                'knee_rl': motions['knee_sinusoid_down'],
            }
        },
        'option_4': {
            'duration': duration_option,
            'joint_names': {
                'hip_rr': motions['hip_forward'],
                'knee_rr': motions['knee_sinusoid_down'],
                'hip_fl': motions['hip_forward'],
                'knee_fl': motions['knee_sinusoid_down'],
            }
        },
        'option_5': {
            'duration': duration_option,
            'joint_names': {
                'hip_rr': motions['hip_backward'],
                'knee_rr': motions['knee_sinusoid_up'],
                'hip_fl': motions['hip_backward'],
                'knee_fl': motions['knee_sinusoid_up'],
            }
        },
        'option_6': {
            'duration': duration_option,
            'joint_names': {
                'hip_rl': motions['hip_backward'],
                'knee_rl': motions['knee_sinusoid_down'],
                'hip_fr': motions['hip_backward'],
                'knee_fr': motions['knee_sinusoid_down'],
            }
        },
        'option_7': {
            'duration': duration_option,
            'joint_names': {
                'hip_rl': motions['hip_forward'],
                'knee_rl': motions['knee_sinusoid_up'],
                'hip_fr': motions['hip_forward'],
                'knee_fr': motions['knee_sinusoid_up'],
            }
        },
    }
    return options


def create_env():
    """Create the environment."""
    joint_config = {
        'hip_zero': 0.0,
        'knee_zero': -np.radians(60),
        'hip_range': np.radians(30),
        'knee_range': np.radians(45),
    }
    
    # Create task based on configuration.
    if TASK_TYPE == 'back_and_forth':
        task = BackAndForthTask(radius=BACK_AND_FORTH_RADIUS, origin=np.array(BACK_AND_FORTH_ORIGIN))
    else:
        task = ForwardTask()
    
    env = AntEnv(
        control_dt=DT,
        render_mode=RENDER_MODE,
        task=task,
        joint_config=joint_config,
        model_path=os.path.join(os.path.dirname(__file__), '../../sim/assets/ant_with_camera_after_sys_id.xml'),
    )
    return env, joint_config


def run_single_seed(seed: int, epsilon: float, discount: float,
                    lambda_eligibility: float, dim_tiling: int, tilings_multiplier: int,
                    step_size_base: float, duration_option: float, iht_size_power: int,
                    reward_scaling: float, state_limits: np.ndarray) -> float:
    """
    Run training for a single seed.
    Returns mean return per timelimit episode (same metric as return_logging.csv).
    """
    np.random.seed(seed)

    env, _ = create_env()
    options = create_options(duration_option)
    num_options = len(options)
    options_env = OptionEnv(env, options, discount, DT, reward_scaling)

    iht_size = 2 ** iht_size_power
    tilings = tilings_multiplier * env.observation_space.shape[0]
    tiles_per_dim = [dim_tiling] * state_limits.shape[0]

    iht = IHT(iht_size)
    T = SuttonTileCoderWrapper(
        iht=iht,
        tiles_per_dim=tiles_per_dim,
        value_limits=state_limits,
        tilings=tilings,
    )

    w = np.zeros((num_options, iht.size), dtype=np.float32)
    eligibility_traces = {}
    step_size = step_size_base / tilings

    timelimit_returns = []
    idx_options = 0
    return_per_timelimit = 0.0
    idx_timelimit_episode = 0

    S, _ = options_env.reset(seed=seed)
    O = select_option_epsilon_greedy(S, epsilon, w, T, num_options)

    while idx_timelimit_episode < N_TIMELIMIT_EPISODES_PER_TRIAL:
        S_prime, R, terminated, truncated = options_env.step(O)
        O_prime = select_option_epsilon_greedy(S_prime, epsilon, w, T, num_options)

        k = options_env.duration_steps(O)
        idx_S = T[S]
        idx_S_prime = T[S_prime]

        Q = q_of(w, idx_S, O)
        Q_prime = q_of(w, idx_S_prime, O_prime)
        TD_error = R + (discount ** k) * Q_prime - Q

        decay_factor = (lambda_eligibility ** k) * (discount ** k)
        keys_to_remove = []
        for key in eligibility_traces:
            eligibility_traces[key] *= decay_factor
            if abs(eligibility_traces[key]) < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del eligibility_traces[key]

        for tile_idx in idx_S:
            key = (O, tile_idx)
            eligibility_traces[key] = eligibility_traces.get(key, 0.0) + 1.0

        for (opt, tile_idx), trace_value in eligibility_traces.items():
            w[opt, tile_idx] += step_size * TD_error * trace_value

        S = S_prime
        O = O_prime

        return_per_timelimit += R
        idx_options += 1

        if terminated or truncated:
            S, _ = options_env.reset(seed=seed)
            O = select_option_epsilon_greedy(S, epsilon, w, T, num_options)

        if idx_options >= MAX_OPTIONS_PER_TIMELIMIT_EPISODE:
            timelimit_returns.append(return_per_timelimit)
            idx_timelimit_episode += 1
            idx_options = 0
            return_per_timelimit = 0.0

    env.close()

    if timelimit_returns:
        mean_return = float(np.mean(timelimit_returns))
    else:
        mean_return = 0.0

    return mean_return


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    Trains SARSA(λ) agent using continuous while True loop (like original code).
    Runs across multiple seeds and returns mean per-episode return.
    """
    # Sample hyperparameters
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3, log=True)
    discount = trial.suggest_float("discount", 0.9, 0.999)
    lambda_eligibility = trial.suggest_float("lambda_eligibility", 0.0, 0.99)
    dim_tiling = trial.suggest_int("dim_tiling", 4, 16)
    tilings_multiplier = trial.suggest_int("tilings_multiplier", 2, 8)
    step_size_base = trial.suggest_float("step_size_base", 0.001, 0.5, log=True)
    # duration_option = trial.suggest_float("duration_option", 0.5, 1.0)
    duration_option = 0.5
    iht_size_power = 25
    reward_scaling = trial.suggest_float("reward_scaling", 1.0, 10.0, log=True)
    
    # Create environment to get observation space (needed for state_limits)
    env_temp, _ = create_env()
    
    # Load observation ranges from file if available, otherwise use defaults
    if OBSERVATION_RANGES_PATH is not None and os.path.exists(OBSERVATION_RANGES_PATH):
        state_limits = np.load(OBSERVATION_RANGES_PATH)
        print(f'Loaded observation ranges from {OBSERVATION_RANGES_PATH}')
        print('State Limits:', state_limits)
    else:
        # Default hardcoded limits
        state_limits = np.array([env_temp.observation_space.low, env_temp.observation_space.high]).T
    env_temp.close()
    print("state_limits.shape (should be [obs_dim, 2]):", state_limits.shape)
    
    # Run training across multiple seeds
    seed_returns = []
    seeds = [SEED + i for i in range(N_SEEDS)]
    print(f"Running {N_SEEDS} seeds with seeds: {seeds}")
    
    for seed_idx, seed in enumerate(seeds):
        mean_return = run_single_seed(
            seed=seed,
            epsilon=epsilon,
            discount=discount,
            lambda_eligibility=lambda_eligibility,
            dim_tiling=dim_tiling,
            tilings_multiplier=tilings_multiplier,
            step_size_base=step_size_base,
            duration_option=duration_option,
            iht_size_power=iht_size_power,
            reward_scaling=reward_scaling,
            state_limits=state_limits
        )
        seed_returns.append(mean_return)
        
        # Report intermediate score for pruning
        current_return = np.mean(seed_returns)
        trial.report(current_return, seed_idx)
        
        # Pruning: stop if trial is not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    mean_return = float(np.mean(seed_returns))

    trial.set_user_attr("mean_return", mean_return)

    return mean_return


def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "optuna_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Create Optuna study
    storage_path = os.path.join(output_dir, f"optuna_study_{timestamp}.db")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        ),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True
    )
    
    print(f"Starting Optuna optimization with {N_OPTUNA_TRIALS} trials...")
    print(f"Results will be saved to: {output_dir}")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\nBest trial mean return: {study.best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, f"best_params_{timestamp}.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "best_value": study.best_trial.value,
            "best_mean_return": study.best_trial.user_attrs.get("mean_return"),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "timestamp": timestamp
        }, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Save all trials data
    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(output_dir, f"all_trials_{timestamp}.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"All trials saved to: {trials_csv_path}")
    
    # Generate visualization
    try:
        import optuna.visualization as vis
        
        # Parameter importances
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html(os.path.join(output_dir, f"param_importances_{timestamp}.html"))
        
        # Optimization history
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(os.path.join(output_dir, f"optimization_history_{timestamp}.html"))
        
        # Parallel coordinate plot
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.write_html(os.path.join(output_dir, f"parallel_coordinate_{timestamp}.html"))
        
        # Contour plot for top 2 important params
        if len(study.best_params) >= 2:
            fig_contour = vis.plot_contour(study)
            fig_contour.write_html(os.path.join(output_dir, f"contour_{timestamp}.html"))
        
        print(f"\nVisualization plots saved to: {output_dir}")
        
    except ImportError:
        print("\nNote: Install plotly for visualization: pip install plotly")
    
    return study


if __name__ == "__main__":
    main()
