import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
import os

current_path = os.path.dirname(os.path.abspath(__file__))

env = gym.make("Ant-v5",
               render_mode="human",
               xml_file=os.path.join(current_path, "assets/ant_position.xml"))

env.reset()
env.observation_space = Space(shape=(10,), dtype=np.float32)
print(env.observation_space)

for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(0.01*env.action_space.sample())

env.close()