import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import scipy.spatial.transform as transform
import os
import time
import matplotlib.pyplot as plt
import mujoco
import imageio
from typing import Sequence, Callable
import mediapy as media
import tqdm


WORKSPACE_LENGTH = 10.0 # m

class AntEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    def __init__(
        self,
        xml_file: str = os.path.join(os.path.dirname(__file__), "assets/ant_position.xml"),
        dt: float = 0.02,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.0,
        cost_upside_down_weight: float = 0.0,
        terminate_on_upside_down: bool = False,
        main_body: int | str = 1,
        joint_config: dict[str, float] | None = None,
        **kwargs,
    ):
        sim_dt = 0.001
        frame_skip = int(dt / sim_dt)

        utils.EzPickle.__init__( # Needed for calling gym.register()
            self,
            xml_file,
            frame_skip,
            forward_reward_weight,
            cost_upside_down_weight,
            main_body,
            **kwargs,
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            **kwargs,
        )
        self.model.opt.timestep = sim_dt

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._cost_upside_down_weight = cost_upside_down_weight
        self._terminate_on_upside_down = terminate_on_upside_down

        obs_size = 8 + 8 + 2 + 3 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.action_space = Box(
            low=-1, high=1, shape=(8,), dtype=np.float64
        )

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

        self.previous_x_position = 0.0

        self.info = {
            "last_last_action": np.zeros(8),
            "last_action": np.zeros(8),
            "heading_vector": np.zeros(2),
        }

    def step(self, action: np.ndarray):
        action = action.copy()
        for i in range(4):
            action[2*i] = np.clip(action[2*i], -1, 1) * self.joint_config['hip_range'] + self.joint_config['hip_zero']
            action[2*i + 1] = np.clip(action[2*i + 1], -1, 1) * self.joint_config['knee_range'] + self.joint_config['knee_zero']
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        reward, reward_info = self._get_rew()

        if self.render_mode == "human":
            self.render()

        self.info.update({
            "current_x_position": self.data.qpos[0],
            "previous_x_position": self.previous_x_position,
            "last_last_action": self.info["last_action"],
            "last_action": action,
        })
        self.previous_x_position = self.data.qpos[0]

        truncated = self._get_truncated()
        if self._terminate_on_upside_down == True:
            terminated = self.info["upside_down"] < 0
        else:
            terminated = False

        return observation, reward, terminated, truncated, self.info

    def _get_rew(self):
        # Control cost.
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.info["last_last_action"] - self.info["last_action"]))

        # Forward progress reward.
        forward_progress_reward = (self.data.qpos[0] - self.previous_x_position) * self._forward_reward_weight

        # Upside down cost.
        quaternion_wxyz = self.data.qpos[3:7]
        up_vector_ant_in_world = transform.Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix()[:, 2]
        z_world = np.array([0, 0, 1])
        upside_down = np.dot(up_vector_ant_in_world, z_world)
        cost_upside_down = 0.0
        if upside_down < 0:
            cost_upside_down = self._cost_upside_down_weight
            print("Upside down")

        # Total reward.
        reward = forward_progress_reward - cost_upside_down - ctrl_cost
        reward_info = {"reward": reward,
                       "forward_progress_reward": forward_progress_reward,
                       "ctrl_cost": ctrl_cost,
                       "cost_upside_down": cost_upside_down}
        self.info.update({'upside_down': upside_down})

        return reward, reward_info

    def _get_truncated(self):
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

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        joint_angles = qpos[7:]
        joint_velocities = qvel[6:]
        quaternion_wxyz = qpos[3:7]
        heading_vector = (transform.Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix() @ np.array([1, 0, 0]))[0:2]
        heading_vector = heading_vector / np.linalg.norm(heading_vector)

        imu_data = self._get_sensor_data("accelerometer")
        accelerations = imu_data[:3]
        angular_vel = self._get_sensor_data("gyro")

        obs = np.concatenate([
                joint_angles, # 8
                joint_velocities, # 8
                heading_vector, # 2
                accelerations, # 3
                angular_vel, # 3
                ], axis=None)

        self.info["heading_vector"] = heading_vector

        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + 0.1 * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        self.previous_x_position = self.data.qpos[0]

        observation = self._get_obs()

        return observation


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
    while counter < 100:
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