from gymnasium.envs.registration import register
from sim.ant_mujoco import AntEnv

register(
    id="CustomAnt-v0",
    entry_point="sim.ant_mujoco:AntEnv",
)