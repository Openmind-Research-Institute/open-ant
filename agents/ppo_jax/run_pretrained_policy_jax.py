import os

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)


from brax.training.agents.ppo import checkpoint as ppo_checkpoint

import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import mujoco
import numpy as np

import pandas as pd

from etils import epath

RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
print(folders)
numeric_folders = [f for f in folders if f[0].isdigit()]
latest_folder = numeric_folders[-1]
print(f'Latest folder: {latest_folder}')

# In the latest folder, find the latest folder, ignore the files.
folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
print(folders)

ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
print(ABS_FOLDER_RESUlTS)

# Tensorboard.
from torch.utils.tensorboard import SummaryWriter

logdir = f"{RESULTS_FOLDER_PATH}/tensorboard_logs/{latest_folder}"
writer = SummaryWriter(log_dir=logdir)

from robot_learning.src.jax.utils import draw_joystick_command

import time
import robot_learning.src.jax.envs.ant as ant

USE_LATEST_WEIGHTS = True

latest_weights_folder = folders[-1]
print(f'Latest weights folder: {latest_weights_folder}')
policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)
    
jit_policy = jax.jit(policy_fn)


# Create environment and initialize
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sim'))
from ant_mujoco import AntEnv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sim'))
from ant_mujoco import AntEnv
sys.path.append(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env'))
from embodied_ant_env import make_ant_env

import json
# Create the environment
hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is not None:
    env_id = 'ant_hw'
    with open(sys.argv[1], 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                       render_mode='human',
                       dt=0.02,
                       joint_config=None)
else:
    current_path = os.path.dirname(os.path.abspath(__file__))
    env = AntEnv(xml_file=os.path.join(current_path, "../../sim/assets/ant_position.xml"),
                 render_mode="human",
                 dt=0.02)
    default_joint_config = env.model.keyframe("home").qpos[7:]
    print('Default joint config:', default_joint_config)
    
# Initialize random key for JAX
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
    # Get action from JAX policy
    
    obs_dict = {
            'privileged_state': jp.zeros(obs.shape[0]),
            'state': jp.array(obs)
        }
            
    act_rng, rng = jax.random.split(rng)
    action_ppo, _ = jit_policy(obs_dict, act_rng)
    # print(np.array(action_ppo))
    action = np.array(action_ppo)
    
    # Take step in environment
    # action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    action_list.append(action)
    reward_list.append(reward)
    
    # Print progress every 1000 steps
    if i % 1000 == 0:
        print(f"Step {i}/{num_steps}, Total reward: {total_reward:.2f}")
    
    # Handle episode termination
    if terminated or truncated:
        print(f"Episode ended at step {i}, Total reward: {total_reward:.2f}")
        obs, _ = env.reset()
        total_reward = 0

print(f"Final total reward: {total_reward:.2f}")
