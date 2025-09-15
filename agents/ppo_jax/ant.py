# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright 2025 Elena-Sorina Lupu

# pylint:disable=g-multiple-import

"""Trains an ant to run in the +x direction."""

from brax import math
from typing import Optional, Dict, Any, Union
import time
import json

import jax
from jax import numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

import mediapy as media
import numpy as np
import os
import shutil

# Local imports.
import mjx_env as mjx_env

# Constants.
NAME_ROBOT = 'ant'
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
XML_PATH = os.path.join(parent_dir, '../../sim/assets/ant_position.xml')
ROOT_BODY = "torso"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
GRAVITY_SENSOR = "upvector"

WORKSPACE_LENGTH = 10.0 # m
WORKSPACE_WIDTH = 10.0 # m

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.001,
      reward_config=config_dict.create(
        ctrl_cost_weight=0.0,
        reset_noise_scale=0.1,
        upside_down_cost_weight=0.0,
        terminate_when_upside_down=False,
      ),
  )

class Ant(mjx_env.MjxEnv):

  def __init__(
      self,
      xml_path: str = XML_PATH,
      save_config_folder: str = None,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path

    # Initialize the model and the mjx model.
    self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
    self._mjx_model = mjx.put_model(self._mj_model)
    self._torso_body_id = self._mj_model.body(ROOT_BODY).id

    # Set the timesteps.
    self._mj_model.opt.timestep = config.sim_dt
    self.ctrl_dt = config.ctrl_dt
    self._sim_dt = config.sim_dt

    self._joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(45),
                'knee_range': np.radians(20),
            }

    self._ctrl_cost_weight = config.reward_config.ctrl_cost_weight
    self._reset_noise_scale = config.reward_config.reset_noise_scale
    self._terminate_when_upside_down = config.reward_config.terminate_when_upside_down
    self._upside_down_cost_weight = config.reward_config.upside_down_cost_weight

    # Initialize the action space.
    self.nb_joints = self.mj_model.njnt - 1 # First joint is freejoint.
    print(f"Number of joints: {self.nb_joints}")

    # Initialize the initial state.
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_q_joints = jp.array(self._mj_model.keyframe("home").qpos[7:])
    print(f'default_q_joints: {self._default_q_joints}')

    # For debugging and testing.
    if save_config_folder is not None:
      # Copy over the ant.xml file.
      shutil.copy(XML_PATH, os.path.join(save_config_folder, 'ant.xml'))
      path_to_env = 'ant.py'
      shutil.copy(path_to_env, os.path.join(save_config_folder, 'ant.py'))
      # export the config dict to a json file
      with open(os.path.join(save_config_folder, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    
    # Randomize the initial state.
    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
      qpos[7:] * jax.random.uniform(key, (self.mj_model.nu, ), minval=low, maxval=hi))

    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
      jax.random.uniform(key, (6,), minval=-0.1, maxval=0.1))

    # Initialize the data.
    data = mjx.make_data(self.mjx_model)
    if qpos is not None:
      data = data.replace(qpos=qpos)
    if qvel is not None:
      data = data.replace(qvel=qvel)
    if qpos[7:] is not None:
      data = data.replace(ctrl=jp.zeros(self.mj_model.nu))
    data = mjx.forward(self.mjx_model, data)

    # Initialize the observation.
    obs = self._get_obs(data)

    # Initialize the reward and done.
    reward, done = jp.zeros(2)

    # Initialize the metrics.
    metrics = {
        'reward': jp.array(0.0, dtype=jp.float32),
        'reward_forward': jp.array(0.0, dtype=jp.float32),
        'reward_ctrl': jp.array(0.0, dtype=jp.float32),
        'x_position': jp.array(0.0, dtype=jp.float32),
        'y_position': jp.array(0.0, dtype=jp.float32),
        'distance_from_origin': jp.array(0.0, dtype=jp.float32),
    }

    # Initialize the info.
    info = {
        "last_action": jp.zeros(self.action_size),
        "last_last_action": jp.zeros(self.action_size),
        "qpos": data.qpos,
        "qvel": data.qvel,
        "xfrc_applied": data.xfrc_applied,
        "previous_pos_x": jp.array(0.0),
        "truncation": jp.array(0.0, dtype=jp.float32),
    }

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Create a new action array with the joint configuration applied
    action_processed = action.copy()
    for i in range(4):
        action_processed = action_processed.at[2*i].set(
            jp.clip(action[2*i], -1, 1) * self._joint_config['hip_range'] + self._joint_config['hip_zero']
        )
        action_processed = action_processed.at[2*i + 1].set(
            jp.clip(action[2*i + 1], -1, 1) * self._joint_config['knee_range'] + self._joint_config['knee_zero']
        )

    data = mjx_env.step(
      self.mjx_model, state.data, action_processed, self.n_substeps
    )

    # Update the info.
    state.info["last_last_action"] = state.info["last_action"]
    state.info["last_action"] = action
    state.info["qpos"] = data.qpos
    state.info["qvel"] = data.qvel
    state.info["xfrc_applied"] = data.xfrc_applied

    # Get the truncation.
    truncation = self._get_truncation(data)
    state.info["truncation"] = truncation

    # Compute the reward.
    forward_progress = data.qpos[0:2] - state.data.qpos[0:2]
    forward_progress_x = forward_progress[0]

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(state.info["last_last_action"] - state.info["last_action"]))

    up_vector_ant_in_world = math.quat_to_mat(data.qpos[3:7])[:, 2]
    z_world = jp.array([0, 0, 1])
    upside_down = jp.dot(up_vector_ant_in_world, z_world)
    reward_upside_down = jp.where(upside_down < 0, -self._upside_down_cost_weight, 0.0)

    reward = forward_progress_x - ctrl_cost + reward_upside_down

    if self._terminate_when_upside_down:
      done = upside_down < 0
      done = done.astype(reward.dtype)
    else:
      done = jax.numpy.array(False, dtype=reward.dtype)
    
    # Get the observation.
    obs = self._get_obs(data)

    # Update the metrics with scalar values.
    metrics = {
        'reward': reward,
        'reward_forward': forward_progress_x,
        'reward_ctrl': -ctrl_cost,
        'x_position': data.qpos[0],
        'y_position': data.qpos[1],
        'distance_from_origin': math.norm(data.qpos[0:2]),
    }

    state = state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics)
    return state

  def _get_truncation(self, data: mjx.Data) -> jax.Array:
    """Gets the truncation of the environment."""
    x_pos = data.qpos[0]
    y_pos = data.qpos[1]

    truncation = (
        jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() |
        (x_pos < -WORKSPACE_LENGTH / 2.0).astype(jp.bool_) | (x_pos > WORKSPACE_LENGTH / 2.0).astype(jp.bool_) |
        (y_pos < -WORKSPACE_WIDTH / 2.0).astype(jp.bool_) | (y_pos > WORKSPACE_WIDTH / 2.0).astype(jp.bool_)
    )

    return truncation.astype(jp.float32)

  def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = self.mj_model.sensor(sensor_name).id
    sensor_adr = self.mj_model.sensor_adr[sensor_id]
    sensor_dim = self.mj_model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Gets the observation of the environment."""
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    joint_angles = qpos[7:]
    joint_velocities = qvel[6:]
    orientation = qpos[3:7]
    heading_vector = (math.quat_to_mat(orientation) @ jp.array([1, 0, 0]))[0:2]
    heading_vector = heading_vector / jp.linalg.norm(heading_vector)

    imu_data = self._get_sensor_data(data, ACCELEROMETER_SENSOR)
    accelerations = imu_data[:3]
    gyro_data = self._get_sensor_data(data, GYRO_SENSOR)

    obs = jp.hstack([
              joint_angles, # 8
              joint_velocities, # 8
              heading_vector, # 2
              accelerations, # 3
              gyro_data, # 3
            ])

    return {
      "state": obs,
      "privileged_state": obs,
    }

   # Accessors.
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.nb_joints

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


# If running remotely, run 'MUJOCO_GL=egl python3 ant.py'
if __name__ == "__main__":
  eval_env = Ant()
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  print(f'JITing reset and step')
  rng = jax.random.PRNGKey(1)

  rollout = []
  modify_scene_fns = []

  state = jit_reset(rng)

  metrics_list = []
  ctrl_list = []
  state_list = []
  for i in range(1400):
    ctrl = jp.zeros(eval_env.action_size)
    for i in range(8):
      ctrl = ctrl.at[i].set(np.sin(time.time())*0.8)
    ctrl = jp.array(ctrl)
    state = jit_step(state, ctrl)
    state_list.append(state.obs["state"])
    metrics_list.append(state.metrics)

    if state.done:
      break
    rollout.append(state)

  render_every = 1
  fps = 1.0 / eval_env.ctrl_dt / render_every
  traj = rollout[::render_every]

  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[2] = True
  scene_option.geomgroup[3] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

  frames = eval_env.render(
      traj,
      camera="track",
      scene_option=scene_option,
      width=640,
      height=480,
  )

  media.write_video(f'{NAME_ROBOT}_test.mp4', frames, fps=fps)
  print('Video saved')