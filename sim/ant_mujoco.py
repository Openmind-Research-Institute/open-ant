import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
import scipy.spatial.transform as transform
import os
import time
import matplotlib.pyplot as plt
import mujoco
import imageio
from typing import Sequence, Callable
import mediapy as media
import tqdm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import ForwardTask, BackAndForthTask
import gymnasium as gym


WORKSPACE_LENGTH = 10.0 # m

class AntEnv(MujocoEnv, utils.EzPickle):
    metadata = {
          "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            # "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = os.path.join(os.path.dirname(__file__), "assets/ant_position.xml"),
        dt: float = 0.02,
        terminate_on_upside_down: bool = False,
        main_body: int = 1,
        joint_config: dict[str, float] = None,
        task=ForwardTask(),
        **kwargs,
    ):
        sim_dt = 0.001
        frame_skip = int(dt / sim_dt)

        utils.EzPickle.__init__( # Needed for calling gym.register()
            self,
            xml_file,
            frame_skip,
            main_body,
            **kwargs,
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            **kwargs,
        )
        self.model.opt.timestep = sim_dt

        self.task = task

        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = task.observation_space
        self._terminate_on_upside_down = terminate_on_upside_down

        if joint_config is None:
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(45),
                'knee_range': np.radians(20),
            }

        self.joint_config = joint_config
        self.init_qpos = [0] * self.model.nq
        self.init_qpos[2] = 0.2
        self.init_qpos[3] = 1.0
        self.init_qvel = [0] * self.model.nv

    def step(self, action: np.ndarray):
        # Clip action.
        action = action.copy()
        for i in range(4):
            action[2*i] = np.clip(action[2*i], -1, 1) * self.joint_config['hip_range'] + self.joint_config['hip_zero']
            action[2*i + 1] = np.clip(action[2*i + 1], -1, 1) * self.joint_config['knee_range'] + self.joint_config['knee_zero']

        # Do simulation.
        self.do_simulation(action, self.frame_skip)

        # Get observation and reward from task.
        info = self.get_observation()
        observation, reward, terminated, truncated = self.task(info, action)

        # Check if out of bounds or nans or truncated from task.
        truncated = self._get_truncated_out_of_bounds_or_nans() or truncated

        # Terminate on upside down.
        quaternion_wxyz = self.data.qpos[3:7]
        quat_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
        up_vector_ant_in_world = transform.Rotation.from_quat(quat_xyzw).as_matrix()[:, 2]
        z_world = np.array([0, 0, 1])
        upside_down = np.dot(up_vector_ant_in_world, z_world)
        if self._terminate_on_upside_down == True:
            terminated = upside_down < 0
        else:
            terminated = False

        # Render.
        if self.render_mode == "human":
            # Add an arrow to the scene
            self.render_with_arrow(info)

        return observation, reward, terminated, truncated, info

    def render_with_arrow(self, info):
        # Direction of the arrow is the reward_direction
        self.render()
        if info and 'reward_direction' in info:
            reward_direction = info['reward_direction']
            self.mujoco_renderer.viewer._markers = []

            direction = np.array([reward_direction[0], reward_direction[1], 0])
            x = direction
            z = np.array([0, 0, 1])
            y = np.cross(z, x)

            m_z_to_x = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]])
            mat = np.array([x, y, z]).T @ m_z_to_x

            self.add_markers_to_scene(np.array([info['current_x_position'], info['current_y_position'], 0.2]),
                                      mat.flatten(), np.array([1.0, 0.0, 0.0, 1.0]), "Arrow")

        return self.mujoco_renderer.render(self.render_mode)

    def add_markers_to_scene(self, pos, mat, rgba, label):
        marker_params = {
            "type": mujoco.mjtGeom.mjGEOM_ARROW,
            "size": np.array([0.01, 0.01, 1.0]),
            "pos": pos,
            "mat": mat,
            "rgba": rgba,
            "label": label,
        }
        self.mujoco_renderer.viewer.add_marker(**marker_params)

    def _get_truncated_out_of_bounds_or_nans(self):
        truncation_condition = (
            np.isnan(self.data.qpos).any() | np.isnan(self.data.qvel).any() |
            (self.data.qpos[0] < -WORKSPACE_LENGTH / 2.0) | (self.data.qpos[0] > WORKSPACE_LENGTH / 2.0) |
            (self.data.qpos[1] < -WORKSPACE_LENGTH / 2.0) | (self.data.qpos[1] > WORKSPACE_LENGTH / 2.0)
        )

        return bool(truncation_condition)

    def _get_sensor_data(self, sensor_name: str) -> np.ndarray:
        sensor_id = self.model.sensor(sensor_name).id
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def get_observation(self):
        quaternion_wxyz = self.data.qpos[3:7]
        quaternion_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
        heading_vector = (transform.Rotation.from_quat(quaternion_xyzw).as_matrix() @ np.array([1, 0, 0]))[0:2]
        heading_vector = heading_vector / np.linalg.norm(heading_vector)

        imu_data = self._get_sensor_data("accelerometer")
        accelerations = imu_data[:3]
        noisy_accelerations = accelerations + self.np_random.normal(0, 0.01, 3)

        angular_vel = self._get_sensor_data("gyro")
        noisy_angular_vel = angular_vel + self.np_random.normal(0, 0.01, 3)

        info = {}
        info["joint_positions"] = self.data.qpos[7:]
        info["joint_velocities"] = self.data.qvel[6:]
        info["heading_vector"] = heading_vector
        info["ax"] = noisy_accelerations[0]
        info["ay"] = noisy_accelerations[1]
        info["az"] = noisy_accelerations[2]
        info["wx"] = noisy_angular_vel[0]
        info["wy"] = noisy_angular_vel[1]
        info["wz"] = noisy_angular_vel[2]
        info["current_x_position"] = self.data.qpos[0]
        info["current_y_position"] = self.data.qpos[1]
        return info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.step(np.zeros(self.action_space.shape[0]))

        qpos = np.array(self.init_qpos)
        random_yaw = self.np_random.uniform(-np.pi, np.pi)
        qpos[3] = np.cos(random_yaw / 2) # w
        qpos[4] = 0 # x
        qpos[5] = 0 # y
        qpos[6] = np.sin(random_yaw / 2) # z
        # Normalize quaternion.
        qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])

        qvel = np.array(self.init_qvel)
        self.set_state(qpos, qvel)

        info = self.get_observation()
        observation, reward, terminated, truncated = self.task.reset(info)

        return observation, info


    def get_joint_names(self):
        '''Returns the names of the joints.'''
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    env = AntEnv(xml_file=os.path.join(current_path, "assets/ant_position.xml"),
                 render_mode="human",
                 dt=0.01)

    joints_dict = {
        "hip_1":
            {'desired': [], 'actual': []},
        "ankle_1":
            {'desired': [], 'actual': []},
        "hip_2":
            {'desired': [], 'actual': []},
        "ankle_2":
            {'desired': [], 'actual': []},
        "hip_3":
            {'desired': [], 'actual': []},
        "ankle_3":
            {'desired': [], 'actual': []},
        "hip_4":
            {'desired': [], 'actual': []},
        "ankle_4":
            {'desired': [], 'actual': []},
    }

    trajectory = []
    counter = 0
    while counter < 1000:
        delta_actions = [2*np.sin(time.time())*0.8]*8
        env.step(np.array(delta_actions))

        for idx, (joint_name, joint_data) in enumerate(joints_dict.items()):
            joint_data['desired'].append(delta_actions[idx])
            joint_data['actual'].append(env.data.qpos[idx+7])

        time.sleep(0.001)
        print(f"Counter: {counter}")
        counter += 1

    _, axs = plt.subplots(2, 4)
    for idx, (joint_name, joint_data) in enumerate(joints_dict.items()):
        axs[idx//4, idx%4].plot(np.rad2deg(joint_data['desired']))
        axs[idx//4, idx%4].plot(np.rad2deg(joint_data['actual']))
        axs[idx//4, idx%4].set_title(f"Joint {joint_name}")
        axs[idx//4, idx%4].set_xlabel("Time")
        axs[idx//4, idx%4].set_ylabel("Angle")
        axs[idx//4, idx%4].legend(["Desired", "Actual"], loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()