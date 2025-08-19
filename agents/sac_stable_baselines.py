
import gymnasium as gym
import sys
import os
from stable_baselines3 import SAC
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env
import numpy as np
import json
from datetime import datetime

render = "human"
DT = 0.05
hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    render_mode = "human" if render else "rgb_array"
    env = AntEnv(xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
                render_mode=render_mode,
                dt=DT,
                )
else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                    render_mode='human',
                    dt=DT,
                    )

LOG_FOLDER = 'logs'
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FOLDER = os.path.join(LOG_FOLDER, current_time)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

train = True
if train == True:
    time_in_sec = 3600 # 1 hour
    total_timesteps = int(time_in_sec / DT)
    model = SAC("MlpPolicy", env, verbose=True, tensorboard_log=LOG_FOLDER)
    model.learn(total_timesteps=total_timesteps, log_interval=4)
    model.save(os.path.join(LOG_FOLDER, "sac_ant_hardware"))
    del model
else:
    model = SAC.load(os.path.join(LOG_FOLDER, "sac_ant_hardware"))
    obs, info = env.reset()
    N = 300
    rewards = []
    for i in range(N):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        rewards.append(reward)
    print(sum(rewards))