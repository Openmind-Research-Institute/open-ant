
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

render = "human"
DT = 0.05
train = False
joint_config = {
    'hip_zero': 0,
    'knee_zero': -np.radians(60),
    'hip_range': np.radians(45),
    'knee_range': np.radians(30),
}

hw_config = sys.argv[1] if len(sys.argv) > 1 else None
if hw_config is None:
    env_id = 'ant_mujoco'
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    render_mode = "human" if render else "rgb_array"
    env = AntEnv(xml_file=os.path.join(current_path, "../sim/assets/ant_position.xml"),
                render_mode=render_mode,
                dt=DT,
                joint_config=joint_config)
else:
    env_id = 'ant_hw'
    with open(hw_config, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg,
                    render_mode='human',
                    dt=DT,
                    joint_config=joint_config)


if train == True:
    model = SAC("MlpPolicy", env, verbose=True)
    model.learn(total_timesteps=100000, log_interval=4, progress_bar=True)
    model.save("sac_ant")
    del model # remove to demonstrate saving and loading
else:
    model = SAC.load("sac_ant")
    obs, info = env.reset()
    N = 300
    rewards = []
    for i in range(N):
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        rewards.append(reward)
    print(sum(rewards))