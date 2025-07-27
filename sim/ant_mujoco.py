import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import scipy.spatial.transform as transform


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

WORKSPACE_LENGTH = 10.0 # m
WORKSPACE_WIDTH = 10.0 # m

class AntEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        main_body: int | str = 1,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            main_body,
            reset_noise_scale,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = 8 + 8 + 2 + 3 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.action_space = Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )

        self.init_qpos = [0] * self.model.nq
        self.init_qpos[2] = 0.2
        self.init_qpos[3] = 1.0
        self.init_qvel = [0] * self.model.nv
        
        self.previous_x_position = 0.0

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        reward, reward_info = self._get_rew()

        if self.render_mode == "human":
            self.render()

        info = {
            "current_x_position": self.data.qpos[0],
            "previous_x_position": self.previous_x_position,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
        self.previous_x_position = self.data.qpos[0]

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        terminated = self._get_termination() # TODO: change this to truncated.
        truncated = False
        return observation, reward, terminated, truncated, info

    def _get_rew(self):
        reward = (self.data.qpos[0] - self.previous_x_position) * self._forward_reward_weight
        reward_info = {"reward": reward}

        return reward, reward_info

    def _get_termination(self):

        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]

        termination = (
            np.isnan(self.data.qpos).any() | np.isnan(self.data.qvel).any() |
            (x_pos < -WORKSPACE_LENGTH / 2.0) | (x_pos > WORKSPACE_LENGTH / 2.0) |
            (y_pos < -WORKSPACE_WIDTH / 2.0) | (y_pos > WORKSPACE_WIDTH / 2.0)
        )

        return termination
    
    def _get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = self.model.sensor(sensor_name).id
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[sensor_adr : sensor_adr + sensor_dim]
        
    def _get_obs(self):
        """Observe ant body position and velocities."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        joint_angles = qpos[7:]
        joint_velocities = qvel[6:]
        orientation = qpos[3:7]
        heading_vector = (transform.Rotation.from_quat(orientation).as_matrix() @ np.array([1, 0, 0]))[0:2]
        heading_vector = heading_vector / np.linalg.norm(heading_vector)

        imu_data = self._get_sensor_data("accelerometer")
        accelerations = imu_data[:3]
        gyro_data = self._get_sensor_data("gyro")

        obs = np.concatenate([
                joint_angles, # 8
                joint_velocities, # 8
                heading_vector, # 2
                accelerations, # 3
                gyro_data, # 3
                ])
        
        return obs

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
