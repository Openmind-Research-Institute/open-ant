import os
import sys
import time
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium import spaces
import matplotlib.pyplot as plt
import scipy.spatial.transform as transform
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Import custom modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))
from embodied_ant_env import ForwardTask, BackAndForthTask

# Constants.
WORKSPACE_LENGTH = 10.0 # m


class AntEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    def __init__(
        self,
        model_path: str = os.path.join(os.path.dirname(__file__), "assets/ant_with_camera_after_sys_id.xml"),
        render_mode: str | None = None,
        control_dt: float = 0.02,
        terminate_on_upside_down: bool = False,
        joint_config: dict[str, float] | None = None,
        task: ForwardTask | BackAndForthTask = ForwardTask(),
    ):
        super().__init__()
        # Initialize the environment.
        sim_dt = 0.001
        self.dt = control_dt
        self.nb_sim_per_step = int(control_dt / sim_dt)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = task.observation_space

        # Initialize the renderer.
        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            width=640,
            height=480,
            max_geom=1000,
            visual_options={},
        )

        # Initialize the task.
        self.task = task
        self._terminate_on_upside_down = terminate_on_upside_down

        self._previous_yaw = None

        # Initialize the joint configuration.
        if joint_config is None:
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(45),
                'knee_range': np.radians(20),
            }
        self.joint_config = joint_config

        # Initialize the initial position and velocity based on joint configuration.
        self.init_qpos = [0.0] * self.model.nq
        self.init_qvel = [0.0] * self.model.nv
        self.init_qpos[2] = 0.2
        self.init_qpos[3] = 1.0  # w.

        # Set leg joints based on joint_config.
        for i in range(4):  # 4 legs
            self.init_qpos[7 + i * 2] = self.joint_config['hip_zero']
            self.init_qpos[7 + i * 2 + 1] = self.joint_config['knee_zero']

        self.last_step_time = None


    def step(self, action: np.ndarray):
        # Clip and apply action.
        action = action.copy()
        for i in range(4):
            action[2*i] = np.clip(action[2*i], -1, 1) * self.joint_config['hip_range'] + self.joint_config['hip_zero']
            action[2*i + 1] = np.clip(action[2*i + 1], -1, 1) * self.joint_config['knee_range'] + self.joint_config['knee_zero']
        self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data, nstep=self.nb_sim_per_step)
        mujoco.mj_rnePostConstraint(self.model, self.data) # See https://github.com/openai/gym/issues/1541

        # Get observation and reward from task.
        info = self.get_observation()
        observation, reward, terminated, truncated = self.task(info, action)

        # Check if out of bounds or nans or truncated from task.
        truncated = self._get_truncated_out_of_bounds_or_nans() or truncated

        # Terminate on upside down.
        quaternion_wxyz = self.data.qpos[3:7]
        up_vector_ant_in_world = transform.Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix()[:, 2]
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

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        self.data.ctrl[:] = qpos[7:]
        mujoco.mj_step(self.model, self.data, nstep=self.nb_sim_per_step)
        mujoco.mj_rnePostConstraint(self.model, self.data) # See https://github.com/openai/gym/issues/1541

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def render_with_arrow(self, info):
        # Direction of the arrow is the reward_direction
        self.render()
        if info and 'reward_direction_I' in info:
            reward_direction = info['reward_direction_I']
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
        heading_vector = (transform.Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix() @ np.array([1, 0, 0]))[0:2]
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
        info["position_timestamp"] = time.time()
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.step(np.zeros(self.action_space.shape[0]))

        if self._previous_yaw is None:
            self._previous_yaw = self.np_random.uniform(-np.pi, np.pi)
            current_yaw = self._previous_yaw
        else:
            # Rotate 180 degrees around the z-axis.
            current_yaw = self._previous_yaw + np.pi
            self._previous_yaw = current_yaw

        qpos = np.array(self.init_qpos)
        qpos[3] = np.cos(current_yaw / 2) # w
        qpos[4] = 0.0 # x
        qpos[5] = 0.0 # y
        qpos[6] = np.sin(current_yaw / 2) # z
        # Normalize quaternion.
        qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])

        qvel = np.array(self.init_qvel)
        self.set_state(qpos, qvel)

        info = self.get_observation()
        observation, reward, terminated, truncated = self.task.reset(info, self.data.ctrl)
        self.last_step_time = time.time()

        return observation, info

    def get_joint_names(self):
        '''Returns the names of the joints.'''
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

    def close(self):
        """Close rendering contexts processes."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    env = AntEnv(model_path=os.path.join(current_path, "assets/ant_with_camera_after_sys_id.xml"),
                 render_mode="human",
                 control_dt=0.05)

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
        counter += 1

if __name__ == "__main__":
    main()