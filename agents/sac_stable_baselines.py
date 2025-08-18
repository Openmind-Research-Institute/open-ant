
import gymnasium as gym
import sys
import os
from stable_baselines3 import SAC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import make_ant_env

train = True
env = AntEnv(dt=0.05)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_ant")

del model # remove to demonstrate saving and loading

if train == False:
    model = SAC.load("sac_ant")
    env = AntEnv(dt=0.05, render_mode="human")
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()