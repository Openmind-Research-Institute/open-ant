import os
import sys
import csv
import json
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym

# Custom imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from tilecoding import IHT, tiles, load_iht_state, save_iht_state
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

np.set_printoptions(precision=4, suppress=True, linewidth=120, threshold=1000)

# For logging.
def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x

# Parser.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='sarsa_ant_forward')
parser.add_argument('--runs_directory', type=str, default='runs',
                    help='the directory to save the runs in')

# Env specific.
parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="render mode")
parser.add_argument('--dt', type=float, default=0.05)
parser.add_argument('--env_id', type=str, default='SimEmbodiedAnt')
parser.add_argument('--hw_config', type=str, default=None)
parser.add_argument('--capture_video', action='store_true')
parser.add_argument('--test_options', action='store_true')
parser.add_argument('--task', type=str, default='back_and_forth', choices=['forward', 'back_and_forth'],
                    help='Task type: forward or back_and_forth')
parser.add_argument('--back_and_forth_radius', type=float, default=1.0,
                    help='Radius for BackAndForthTask (only used if task is back_and_forth)')
parser.add_argument('--back_and_forth_origin', type=float, nargs=2, default=[0.0, 0.0],
                    help='Origin for BackAndForthTask as [x, y] (only used if task is back_and_forth)')

# Algorithm specific.
parser.add_argument('--learn', type=bool, default=True)
parser.add_argument('--reward_scaling', type=float, default=5.0) # Values found with Optuna.
parser.add_argument('--load_weights_from_dir', type=str, default=None)
parser.add_argument('--lambda_eligibility', type=float, default=0.964,
                    help='Eligibility trace decay parameter (lambda)') # Values found with Optuna.
parser.add_argument('--duration_option', type=float, default=0.5,
                    help='Duration of each option in seconds')
parser.add_argument('--epsilon', type=float, default=0.255,
                    help='Epsilon for epsilon-greedy policy') # Values found with Optuna.
parser.add_argument('--discount', type=float, default=0.998,
                    help='Discount factor') # Values found with Optuna.
parser.add_argument('--dim_tiling', type=int, default=4,
                    help='Number of tiles per dimension') # Values found with Optuna.
parser.add_argument('--tilings_multiplier', type=int, default=8,
                    help='Multiplier for number of tilings (tilings = multiplier * obs_dim)')  # Values found with Optuna.
parser.add_argument('--step_size_base', type=float, default=0.008,
                    help='Base step size (step_size = base / tilings)')  # Values found with Optuna.
parser.add_argument('--iht_size_power', type=int, default=25,
                    help='Power of 2 for IHT size (iht_size = 2^power)')
parser.add_argument('--num_timelimit_episodes', type=int, default=1000,
                    help='Number of timelimit episodes')

# Parse arguments.
args = parser.parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# Directories.
RUN_NAME = args.exp_name + '_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + f'_seed_{args.seed}'
os.makedirs(args.runs_directory, exist_ok=True)
LOG_DIR = os.path.join(args.runs_directory, RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)
WEIGHTS_IHT_DIR = os.path.join(LOG_DIR, "weights_iht")
if not os.path.exists(WEIGHTS_IHT_DIR):
    os.makedirs(WEIGHTS_IHT_DIR)

with open(os.path.join(LOG_DIR, "args.json"), "w") as f:
    json.dump(args.__dict__, f, indent=4)

SEED = args.seed
np.random.seed(SEED)

# Ramp function.
def linear_ramp(start_pos: float, end_pos: float, duration: float):
    num = round(duration / args.dt)
    input_pos_list = np.linspace(start_pos, end_pos, num)
    return input_pos_list

class OptionEnv:
    def __init__(self, env, options, discount=0.99):
        self.env = env
        self.options = options
        self.discount = discount
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
        # Select the option.
        opt = self.options['option_' + str(option_idx)]

        # Populate the trajectory for the joints.
        for joint_name in opt['joint_names']:
            if joint_name.startswith('hip'):
                self.joints_dict[joint_name]['traj'] = linear_ramp(self.joints_dict[joint_name]['current_pos'], \
                                                                opt['joint_names'][joint_name]['hip_target'], opt['duration'])
            if joint_name.startswith('knee'):
                num_steps = int(opt['duration'] / args.dt)
                time = np.linspace(0, opt['duration'], num_steps)
                self.joints_dict[joint_name]['traj'] = opt['joint_names'][joint_name]['knee_amplitude'] * np.sin(np.pi * time / opt['duration'])

        total_reward = 0.0
        gamma_i = 1.0
        action_vector = np.zeros(self.env.action_space.shape[0])
        info_list = []
        for i in range(int(opt['duration'] / args.dt)):
            # Build the action vector.
            for idx, joint_name in enumerate(self.joints_dict):
                if self.joints_dict[joint_name]['traj'] is not None:
                    action_vector[idx] = self.joints_dict[joint_name]['traj'][i]

            obs, reward, terminated, truncated, self.info = self.env.step(action_vector)
            reward *= args.reward_scaling

            total_reward += gamma_i * reward
            gamma_i *= self.discount
            # Deep copy to avoid reference issues with numpy arrays.
            info_list.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.info.items()})
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info_list

        for joint_name in self.joints_dict:
            if self.joints_dict[joint_name]['traj'] is not None:
                self.joints_dict[joint_name]['current_pos'] = self.joints_dict[joint_name]['traj'][-1]
            self.joints_dict[joint_name]['traj'] = None

        return obs, total_reward, terminated, truncated, info_list

    def reset(self, seed=None):
        for joint_name in self.joints_dict:
            self.joints_dict[joint_name]['current_pos'] = 0.0
            self.joints_dict[joint_name]['traj'] = None
        return self.env.reset(seed=args.seed if seed is None else seed)

    def render(self):
        return self.env.render_with_arrow(self.info)
    
    def duration_steps(self, option_idx: int):
        opt = self.options['option_' + str(option_idx)]
        return int(opt['duration'] / args.dt)

# Define motions.
motions = {
    "knee_sinusoid_up": {"knee_amplitude": 1.0,},
    "knee_sinusoid_down": {"knee_amplitude": -1.0,},
    "hip_forward": {"hip_target": 1.0,},
    "hip_backward": {"hip_target": -1.0,},
}

# For simplicity, all durations are the same.
options = {
    # Rotate in place left (counterclockwise)
    'option_0': {
        'duration': args.duration_option,
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
        'duration': args.duration_option,
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
        'duration': args.duration_option,
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
        'duration': args.duration_option,
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
        'duration': args.duration_option,
        'joint_names': {
            'hip_rr': motions['hip_forward'],
            'knee_rr': motions['knee_sinusoid_down'],
            'hip_fl': motions['hip_forward'],
            'knee_fl': motions['knee_sinusoid_down'],
        }
    },
    # Rotate in place right (clockwise)
    'option_5': {
        'duration': args.duration_option,
        'joint_names': {
            'hip_rr': motions['hip_backward'],
            'knee_rr': motions['knee_sinusoid_up'],
            'hip_fl': motions['hip_backward'],
            'knee_fl': motions['knee_sinusoid_up'],
        }
    },
    'option_6': {
        'duration': args.duration_option,
        'joint_names': {
            'hip_rl': motions['hip_backward'],
            'knee_rl': motions['knee_sinusoid_down'],
            'hip_fr': motions['hip_backward'],
            'knee_fr': motions['knee_sinusoid_down'],
        }
    },
    'option_7': {
        'duration': args.duration_option,
        'joint_names': {
            'hip_rl': motions['hip_forward'],
            'knee_rl': motions['knee_sinusoid_up'],
            'hip_fr': motions['hip_forward'],
            'knee_fr': motions['knee_sinusoid_up'],
        }
    },
}

len_options = len(options)
print(len(options), "options defined.")

# Tile coding.
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
    # Tie-break among maxima, in case of ties.
    maxq = q_vals.max()
    best = np.flatnonzero(q_vals == maxq)
    return int(np.random.choice(best)), q_vals

def select_option_epsilon_greedy(S, epsilon, w, T):
    # ε-greedy over options using tile-coded T(s).
    if np.random.rand() < epsilon:
        return np.random.randint(num_options)
    O_greedy, _ = select_greedy_option(w, T, S, num_options)
    return O_greedy


# Environment.
joint_config = {
    'hip_zero': 0.0,
    'knee_zero': -np.radians(60),
    'hip_range': np.radians(30),
    'knee_range': np.radians(45),
}

# Create task based on arguments
if args.task == 'back_and_forth':
    task = BackAndForthTask(radius=args.back_and_forth_radius, origin=np.array(args.back_and_forth_origin))
else:
    task = ForwardTask()
hw_config = args.hw_config if args.hw_config is not None else None
if args.hw_config is None:
    env = AntEnv(
        control_dt=args.dt,
        render_mode=args.render_mode,
        task=task,
        joint_config=joint_config,
        model_path=os.path.join(os.path.dirname(__file__), '../../sim/assets/ant_with_camera_after_sys_id.xml'),
    )
    if args.capture_video:
        print('RecordVideo!')
        env = gym.wrappers.RecordVideo(env, os.path.join(LOG_DIR, "videos", args.env_id),
                                        step_trigger=lambda x: x % 20000 == 0)

else:
    with open(args.hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode=args.render_mode,
                        dt=args.dt,
                        joint_config=joint_config,
                        task=task,
                        )

# Constants.
MAX_OPTIONS_PER_TIMELIMIT_EPISODE = 30
EPSILON = args.epsilon
EPSILON_START = EPSILON
DISCOUNTING = args.discount
LAMBDA_ELIGIBILITY = args.lambda_eligibility  # Eligibility trace decay parameter

DIM_TILING = args.dim_tiling  # Number of tiles per dimension.
TILINGS = args.tilings_multiplier * env.observation_space.shape[0]  # Number of offset tilings.
IHT_SIZE = 2**args.iht_size_power

# Environment.
options_env = OptionEnv(env, options, discount=args.discount)

num_options = len(options)

def default_state_limits():
    return np.array([env.observation_space.low, env.observation_space.high]).T  # [state_dim, 2]

def resolve_state_limits():
    # When resuming, tile coordinates must use the same limits as when weights were learned.
    if args.load_weights_from_dir is not None:
        saved_limits_path = os.path.join(args.load_weights_from_dir, "weights_iht/state_limits.npy")
        if os.path.exists(saved_limits_path):
            limits = np.load(saved_limits_path)
            print('Loading state limits from checkpoint:', saved_limits_path)
            return limits

    obs_ranges_path = os.path.join(args.runs_directory, "observation_ranges.npy")
    if os.path.exists(obs_ranges_path):
        limits = np.load(obs_ranges_path)
        print('Loading state limits from', obs_ranges_path)
        return limits

    limits = default_state_limits()
    print('No state limits found, using observation-space defaults:', limits)
    return limits

state_limits = resolve_state_limits()

# Load previous weights.
if args.load_weights_from_dir is None:
    iht = IHT(IHT_SIZE)
    w = np.zeros((num_options, iht.size), dtype=np.float32)
    # Initialize eligibility traces (sparse dictionary for efficiency).
    # Keys are (option, tile_index) tuples, values are eligibility trace values.
    eligibility_traces = {}
else:
    weights_path = os.path.join(args.load_weights_from_dir, 'weights_iht/weights.npy')
    iht_path = os.path.join(args.load_weights_from_dir, "weights_iht/iht.pkl")
    w = np.load(weights_path)
    iht = load_iht_state(iht_path)

    if w.shape[0] != num_options:
        raise ValueError(
            f"Loaded weights have {w.shape[0]} options but this run defines {num_options}."
        )
    if w.shape[1] != iht.size:
        raise ValueError(
            f"weights.npy width {w.shape[1]} does not match IHT size {iht.size}."
        )
    if IHT_SIZE != iht.size:
        raise ValueError(
            f"Current iht_size_power gives IHT size {IHT_SIZE}, "
            f"but checkpoint IHT size is {iht.size}."
        )
    print(
        f'Loaded weights: {int(np.sum(w != 0.0))} nonzero out of {w.size}, '
        f'IHT dictionary has {len(iht.dictionary)} tile coordinates'
    )

    # Load eligibility traces if available.
    elig_path = os.path.join(args.load_weights_from_dir, "weights_iht/eligibility_traces.pkl")
    if os.path.exists(elig_path):
        with open(elig_path, "rb") as f:
            eligibility_traces = pickle.load(f)
        print('Loaded eligibility traces from ', elig_path)
    else:
        eligibility_traces = {}
        print('No eligibility traces found, starting fresh.')
    print('Loaded weights from ', args.load_weights_from_dir)
    if args.learn == False:
        EPSILON = 0.0

# IHT table size.
tiles_per_dim = [DIM_TILING] * state_limits.shape[0]
T = SuttonTileCoderWrapper(iht=iht,
                           tiles_per_dim=tiles_per_dim,
                           value_limits=state_limits,
                           tilings=TILINGS)
step_size = args.step_size_base / TILINGS  # Step-size, see: http://incompleteideas.net/tiles/tiles3.html.

# Initialize info logging.
csv_file_info = None
writer_info = None
keys_info = None
info_log_buffer = []

idx_options = 0
return_per_timelimit = 0.0
idx_timelimit_episode = 0
real_time_seconds = 0.0
global_substep = 0  # Global counter for logging that never resets

return_logging_df = pd.DataFrame(columns=['step', 'return'])

if args.test_options:
    N = 100
    idx = 0
    all_observations = []
    S, _ = options_env.reset(seed=SEED)
    for i in tqdm(range(N)):
        S_prime, R, terminated, truncated, info = options_env.step(np.random.randint(num_options))
        all_observations.append(S_prime.copy())
        if terminated or truncated:
            S, _ = options_env.reset(seed=SEED)
            all_observations.append(S.copy())

    # Convert to numpy array.
    all_observations = np.array(all_observations)
    print('Observation shape: ', all_observations.shape)

    # Compute ranges.
    obs_min = np.min(all_observations, axis=0)
    obs_max = np.max(all_observations, axis=0)
    for i in range(all_observations.shape[1]):
        print(f"Observation {i}: min={obs_min[i]}, max={obs_max[i]}")

    # Save to file.
    np.save(os.path.join(args.runs_directory, "observation_ranges.npy"),
            np.array([obs_min, obs_max]).T)  # [state_dim, 2]
    print(f"\nSaved observation ranges to {os.path.join(args.runs_directory, 'observation_ranges.npy')}")

    sys.exit(0)

# Algorithm implemented based on SARSA(λ).
# http://incompleteideas.net/book/ebook/node77.html

# Reset environment.
S, info = options_env.reset(seed=SEED)
O = select_option_epsilon_greedy(S, EPSILON, w, T)

# Initialize info logging on first step.
csv_file_info = open(os.path.join(LOG_DIR, "info_logs.csv"), "w", newline="")
keys_info = list(info.keys())
keys_info = [k for k in keys_info if not (k.startswith("bodies") or k.startswith("_"))]
writer_info = csv.DictWriter(csv_file_info, fieldnames=["step"] + keys_info)
writer_info.writeheader()

while idx_timelimit_episode < args.num_timelimit_episodes:
    # Step.
    S_prime, R, terminated, truncated, info_list = options_env.step(O)

    # Next option (ε-greedy).
    O_prime = select_option_epsilon_greedy(S_prime, EPSILON, w, T)

    # TD.
    k = options_env.duration_steps(O)
    idx_S = T[S]
    idx_S_prime = T[S_prime]

    Q = q_of(w, idx_S,  O)
    Q_prime = q_of(w, idx_S_prime, O_prime)

    TD_error = R + (DISCOUNTING ** k) * Q_prime - Q

    # Update weights with eligibility traces.
    if args.learn == True:
        # Decay all eligibility traces: e = λ * γ^k * e
        decay_factor = (LAMBDA_ELIGIBILITY ** k) * (DISCOUNTING ** k)
        # Efficiently decay only non-zero traces
        keys_to_remove = []
        for key in eligibility_traces:
            eligibility_traces[key] *= decay_factor
            # Remove traces that have decayed below threshold for efficiency.
            if abs(eligibility_traces[key]) < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del eligibility_traces[key]

        # Accumulate eligibility for current state-action pair.
        for tile_idx in idx_S:
            key = (O, tile_idx)
            eligibility_traces[key] = eligibility_traces.get(key, 0.0) + 1.0

        # Update weights using eligibility traces.
        for (opt, tile_idx), trace_value in eligibility_traces.items():
            w[opt, tile_idx] += step_size * TD_error * trace_value

    # Log info for each substep within the option.
    if writer_info is not None and info_list is not None:
        for substep_info in info_list:
            infos_to_log = {}
            for k, v in substep_info.items():
                if k in keys_info:
                    infos_to_log[k] = arr_to_str(v) if isinstance(v, np.ndarray) else v
            row = {"step": global_substep, **infos_to_log}
            info_log_buffer.append(row)
            global_substep += 1

    S = S_prime
    O = O_prime

    return_per_timelimit += R
    real_time_seconds += options_env.duration_steps(O) * args.dt

    idx_options += 1

    if terminated or truncated:
        print('Terminated', terminated, 'truncated', truncated)
        # Reset eligibility traces on episode termination
        # eligibility_traces.clear()
        S, _ = options_env.reset(seed=SEED)
        O = select_option_epsilon_greedy(S, EPSILON, w, T)

    # Logging.
    if idx_options >= MAX_OPTIONS_PER_TIMELIMIT_EPISODE:
        print(f"Ep. {idx_timelimit_episode} | Return: {return_per_timelimit:.4f} | Time in sec: {(real_time_seconds):.4f} | Time in hours: {(real_time_seconds) / 3600:.4f}")

        return_logging_df = pd.concat(
            [return_logging_df, pd.DataFrame([{'step': global_substep, 'return': return_per_timelimit}])],
            ignore_index=True,
        )
        return_logging_df.to_csv(os.path.join(args.runs_directory, "return_logging.csv"), index=False)
        # Write buffered info logs to CSV.
        if writer_info is not None and info_log_buffer:
            for row in info_log_buffer:
                writer_info.writerow(row)
            csv_file_info.flush()
            info_log_buffer = []

        np.save(os.path.join(WEIGHTS_IHT_DIR, "weights.npy"), w)
        save_iht_state(iht, os.path.join(WEIGHTS_IHT_DIR, "iht.pkl"))
        np.save(os.path.join(WEIGHTS_IHT_DIR, "state_limits.npy"), state_limits)
        with open(os.path.join(WEIGHTS_IHT_DIR, "eligibility_traces.pkl"), "wb") as f:
            pickle.dump(eligibility_traces, f)
        print('Done saving weights and eligibility traces.')

        idx_timelimit_episode += 1
        idx_options = 0
        return_per_timelimit = 0.0