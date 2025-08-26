import os
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from brax.training.agents.ppo import checkpoint as ppo_checkpoint

import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import json
from etils import epath

# Local imports.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from reward import RewardTracker
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sim'))
from ant_mujoco import AntEnv
sys.path.append(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env'))
from embodied_ant_env import make_ant_env

# Create the environment
hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is not None:
    env_id = 'ant_hw'
    with open(sys.argv[1], 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                       render_mode='human',
                       dt=0.05,
                       joint_config=None)
else:
    env_id = "ant_sim"
    current_path = os.path.dirname(os.path.abspath(__file__))
    env = AntEnv(xml_file=os.path.join(current_path, "../../sim/assets/ant_position.xml"),
                 render_mode="human",
                 dt=0.05)
    default_joint_config = env.model.keyframe("home").qpos[7:]
    print('Default joint config:', default_joint_config)

RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
numeric_folders = [f for f in folders if f[0].isdigit()]
latest_folder = numeric_folders[-1]
folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
latest_weights_folder = folders[-1]
policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)

jit_policy = jax.jit(policy_fn)

reward_tracker = RewardTracker(env_dt=env.dt,
                               env_id=env_id,
                               time_window=10.0,
                               log_folder=epath.Path(RESULTS_FOLDER_PATH) / latest_folder)

# Initialize random key for JAX.
rng = jax.random.PRNGKey(0)

# Run the policy in the environment.
obs, _ = env.reset()
DURATION = 100 # seconds
DT = env.dt
num_steps = int(DURATION / DT)
action_list = []
reward_list = []
total_reward = 0

print(f"Running JAX policy for {DURATION} seconds ({num_steps} steps)")
for i in range(num_steps):
    # Get action from JAX policy.
    obs_dict = {
            'privileged_state': jp.zeros(obs.shape[0]),
            'state': jp.array(obs)
        }
            
    act_rng, rng = jax.random.split(rng)
    action_ppo, _ = jit_policy(obs_dict, act_rng)
    action = np.array(action_ppo)
    
    # Take step in environment.
    obs, reward, terminated, truncated, info = env.step(action)

    # Update the reward tracker.
    average_reward_per_second = reward_tracker.update(reward)
    
    action_list.append(action)
    reward_list.append(reward)

    if terminated or truncated:
        print(f"Average reward per second: {average_reward_per_second}")
        reward_tracker.log(i, average_reward_per_second)
        obs, _ = env.reset()
