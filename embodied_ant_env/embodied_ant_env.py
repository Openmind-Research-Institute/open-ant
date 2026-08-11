import time
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
from collections import defaultdict

from imu_msp import IMU_MSP
from apriltag_tracking import VisionTracker, show_image
from motor_controller import MotorController, MotorControllerError

class IMUError(Exception):
    pass


class CameraError(Exception):
    pass


class ForwardTask:
    def __init__(self, action_cost_weight=0.0):
        self.action_cost_weight = action_cost_weight
        self.last_pos = None
        self.last_action = np.zeros(8)
        self.reward_direction_I = np.array([1, 0])
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(24,), dtype=np.float32)
        self.previous_pos_timestamp = None

    def reset(self, info, action=np.zeros(8)):
        self.last_pos = None
        self.previous_pos_timestamp = None
        return self(info, action)

    def __call__(self, info, action):
        pos = np.array([info['current_x_position'], info['current_y_position']])
        pos_timestamp = info['position_timestamp']
        if self.last_pos is None:
            self.last_pos = pos
            progress = 0.0
        else:
            progress = (pos - self.last_pos)[0]

        if self.previous_pos_timestamp is None:
            self.previous_pos_timestamp = pos_timestamp

        if pos_timestamp == self.previous_pos_timestamp:
            print('Warning!! The frames did not update!')

        cost_action = np.sum(np.square(self.last_action - action)) * self.action_cost_weight
        self.last_pos = pos
        self.previous_pos_timestamp = pos_timestamp
        self.last_action = action.copy()
        terminated = False
        truncated = False

        reward = progress - cost_action
        info['reward_direction_I'] = self.reward_direction_I
        info['original_reward'] = reward
        info['actions'] = action
        observation = np.concatenate([
            info['joint_positions'],
            info['joint_velocities'],
            info['heading_vector'],
            info['ax'],
            info['ay'],
            info['az'],
            info['wx'],
            info['wy'],
            info['wz'],
        ], axis=None)
        return observation, reward, terminated, truncated


class BackAndForthTask:
    def __init__(self, action_cost_weight=0.0, radius=1.0, origin=np.array([0, 0])):
        self.action_cost_weight = action_cost_weight
        self.last_pos = None
        self.reward_direction_I = np.array([1, 0])
        self.last_action = np.zeros(8)
        self.previous_pos_timestamp = None
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(24,), dtype=np.float32)
        self.radius = radius
        self.origin = origin

    def reset(self, info, action=np.zeros(8)):
        self.last_pos = None
        self.previous_pos_timestamp = None
        th = np.random.uniform(-np.pi, np.pi)
        self.reward_direction_I = np.array([np.cos(th), np.sin(th)])
        return self(info, action)

    def __call__(self, info, action):
        pos = np.array([info['current_x_position'], info['current_y_position']])
        pos_timestamp = info['position_timestamp']
        # Bounce back on the circle edge.
        if np.dot(pos - self.origin, self.reward_direction_I) > 0 and np.linalg.norm(pos - self.origin) > self.radius:
            self.reward_direction_I = self.origin - pos
            self.reward_direction_I /= np.linalg.norm(self.reward_direction_I)

        if self.last_pos is None:
            self.last_pos = pos
            progress = 0.0
        else:
            progress = np.dot(pos - self.last_pos, self.reward_direction_I)

        if self.previous_pos_timestamp is None:
            self.previous_pos_timestamp = pos_timestamp

        if pos_timestamp == self.previous_pos_timestamp:
            print("Reward", progress)
            print('Warning!! The frames did not update!')

        cost_action = np.sum(np.square(self.last_action - action)) * self.action_cost_weight
        self.last_pos = pos
        self.previous_pos_timestamp = pos_timestamp
        self.last_action = action.copy()
        terminated = False
        truncated = False

        reward = progress - cost_action
        info['reward_direction_I'] = self.reward_direction_I
        info['reward_direction_I_x'] = self.reward_direction_I[0]
        info['reward_direction_I_y'] = self.reward_direction_I[1]

        heading_perp = np.array([-info['heading_vector'][1], info['heading_vector'][0]])
        reward_direction_B = np.array([np.dot(info['reward_direction_I'], info['heading_vector']),
                             np.dot(info['reward_direction_I'], heading_perp)])

        info['reward_direction_B_x'] = reward_direction_B[0]
        info['reward_direction_B_y'] = reward_direction_B[1]
        info['original_reward'] = reward
        info['actions'] = action
        observation = np.concatenate([
            info['joint_positions'],
            info['joint_velocities'],
            reward_direction_B,
            info['ax'],
            info['ay'],
            info['az'],
            info['wx'],
            info['wy'],
            info['wz'],
        ], axis=None)

        return observation, reward, terminated, truncated


class EmbodiedAnt(gym.Env):
    def __init__(
        self,
        motor_controller,
        imu,
        tracker,
        dt=0.02,
        render_mode=None,
        joint_config=None,
        task=ForwardTask(),
        imu_timeout=0.2,
        camera_timeout=1.0
    ):
        super().__init__()
        self.task = task
        self.motor_controller = motor_controller
        self.motor_controller.enable()
        self.dt = dt
        self.last_step_time = None
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.vis_frame = None
        self.i = 0
        if joint_config is None:
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(45),
                'knee_range': np.radians(20),
            }
        self.joint_config = joint_config

        self._threads_should_exit = False

        self.observation_space = task.observation_space

        self.action_space = Box(
            low=-1, high=1, shape=(8,), dtype=np.float64
        )

        self.imu = imu
        self.imu_timeout = imu_timeout
        self._imu_data = None
        self._imu_data_lock = threading.Lock()
        self._imu_thread = threading.Thread(target=self._poll_imu, daemon=True)
        self._imu_thread.start()

        self.tracker = tracker
        self.camera_timeout = camera_timeout
        self._tracker_data = None
        self._tracker_data_lock = threading.Lock()
        self._tracker_thread = threading.Thread(target=self._poll_tracker, daemon=True)
        self._tracker_thread.start()
        self._wait_for_tracker_frame()

        self.last_pos = None
        self.last_heading_vector = np.array([1.0, 0.0])
        self.last_seen = 0
        self.last_position_timestamp = None

        self.temperature_log = open('temperature_log.csv', 'a')
        # self.temperature_log = open('temperature_log.csv', 'w')
        self.error_log = open('error_log.csv', 'a')

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        self.step(np.zeros(self.action_space.shape[0]))
        print('reset(): please move the ant back to the origin.')
        user_input = input('press enter when ready')
        self.last_step_time = time.time()

        # Recover from any subsystem error and retry until an observation succeeds.
        try:
            info = self.get_observation()
        except (MotorControllerError, IMUError, CameraError) as e:
            info = self._get_observation_with_recovery(e)
        observation, reward, terminated, truncated = self.task.reset(info)
        return observation, info

    def step(self, action, sleep_until_next_step=True):
        if self._threads_should_exit:
            raise RuntimeError("EmbodiedAnt.step() called after close()")

        # Apply action.
        action = action.copy()
        for i in range(4):
            action[2*i] = np.clip(action[2*i], -1, 1) * self.joint_config['hip_range'] + self.joint_config['hip_zero']
            action[2*i + 1] = np.clip(action[2*i + 1], -1, 1) * self.joint_config['knee_range'] + self.joint_config['knee_zero']

        time_since_last_step = 0.0
        hardware_error = False
        try:
            self.motor_controller.set_positions(action)

            # Sleep.
            sleep_duration = self.dt
            if self.last_step_time is not None:
                time_since_last_step = time.time() - self.last_step_time
                sleep_duration = self.dt - time_since_last_step
                if sleep_duration < 0:
                    print(f"Warning: calls to step() exceeded step size (time since last step: {time_since_last_step:.3f}s).")
                    sleep_duration = 0
            if sleep_until_next_step:
                time.sleep(sleep_duration)
            self.last_step_time = time.time()

            # Get observation.
            time_start = time.time()
            info = self.get_observation()
            info['env_time_to_get_obs'] = time.time() - time_start
            info['env_sleep_duration'] = sleep_duration

            # Check for motor controller errors.
            time_start = time.time()
            errors = self.motor_controller.check_errors()
            if len(errors) > 0:
                self.error_log.write(f"{time.time()}, " + ", ".join(map(str, errors)) + "\n")
                self.error_log.flush()
                print('motor controller errors:')
                for error in errors:
                    print(error[2])
                raise MotorControllerError(f"Latched hardware errors: {errors}")
            info['env_time_to_check_errors'] = time.time() - time_start
        except (MotorControllerError, IMUError, CameraError) as e:
            hardware_error = True
            info = self._get_observation_with_recovery(e)

        # Get reward and task termination.
        observation, reward, terminated, truncated = self.task(info, action)

        # self.temperature_log.write(f"{time.time()}, " + ", ".join(map(str, info['temperatures'])) + "\n")
        # self.temperature_log.flush()

        if hardware_error:
            self.last_step_time = time.time()
            truncated = True

        if self.tracker_lost(info):
            truncated = True

        if self.render_mode == 'human' or self.render_mode == 'rgb_array':
            self.i += 1

            if isinstance(self.task, BackAndForthTask):
                # Draw the origin circle.
                origin_3D_O = np.array([self.task.origin[0], self.task.origin[1], 0.0])
                self.tracker.draw_circle(self.vis_frame,
                                            origin_3D_O,
                                            self.task.radius)
                reward_direction_I = np.array([self.task.reward_direction_I[0],
                                                self.task.reward_direction_I[1],
                                                0.0])
                self.tracker.draw_arrow(self.vis_frame,
                                            origin_3D_O,
                                            reward_direction_I)
                r_B = np.array([info['reward_direction_B_x'],
                                info['reward_direction_B_y'],
                                0.0])
                if 'body' in info['bodies']:
                    R_B_I = info['bodies']['body']['orientation']
                    r_I = R_B_I @ r_B
                    self.tracker.draw_arrow(self.vis_frame,
                                            self.last_pos,
                                            r_I)
            if self.render_mode == 'human':
                if self.i % 10 == 0:
                    show_image(self.vis_frame)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human' or self.render_mode == 'rgb_array':
            return self.vis_frame
        return None

    def get_observation(self):
        # IMU.
        time_start = time.time()
        info = defaultdict(lambda: 0.0)
        with self._imu_data_lock:
            imu_data = self._imu_data
        if imu_data is None:
            raise IMUError("No IMU data available")
        imu_age = time.time() - imu_data.get('timestamp', 0.0)
        if imu_age > self.imu_timeout:
            raise IMUError(f"IMU data stale (age {imu_age:.3f}s > {self.imu_timeout}s)")
        info.update(imu_data)

        # Tracker. Raise CameraError if the frame is missing or stale; the tracker
        # thread handles reconnecting in the background.
        time_start = time.time()
        with self._tracker_data_lock:
            tracker_data = self._tracker_data
        if tracker_data is None:
            raise CameraError("No tracker camera frame available")
        (bodies, frame, vis_frame), frame_timestamp = tracker_data
        frame_age = time.time() - frame_timestamp
        if frame_age > self.camera_timeout:
            raise CameraError(
                f"Tracker camera frame stale (age {frame_age:.3f}s > {self.camera_timeout}s)"
            )
        info['bodies'] = bodies
        if 'body' in bodies:
            info['current_x_position'] = bodies['body']['position'][0]
            info['current_y_position'] = bodies['body']['position'][1]
            self.last_pos = bodies['body']['position']
            self.last_seen = time.time()
            heading_vector = (bodies['body']['orientation'] @ np.array([1, 0, 0]))[:2]
            heading_vector /= np.linalg.norm(heading_vector)
            self.last_heading_vector = heading_vector
            info['position_timestamp'] = bodies['body']['timestamp']
            self.last_position_timestamp = info['position_timestamp']
            info['detection_time'] = bodies['body']['detection_time']
        else:
            info['current_x_position'] = self.last_pos[0] if self.last_pos is not None else 0.0
            info['current_y_position'] = self.last_pos[1] if self.last_pos is not None else 0.0
            heading_vector = self.last_heading_vector
            info['position_timestamp'] = self.last_position_timestamp if self.last_position_timestamp is not None else 0.0
            info['detection_time'] = 0.0
        info['heading_vector'] = heading_vector

        self.vis_frame = vis_frame
        info['vis_frame'] = self.vis_frame
        info['env_time_tracker_thread'] = time.time() - time_start

        # Motor outputs.
        joint_positions, joint_velocities, joint_loads = self.motor_controller.get_feedback()
        # temperatures = self.motor_controller.get_temperature()
        info['joint_positions'] = joint_positions
        info['joint_velocities'] = joint_velocities
        info['joint_loads'] = joint_loads
        # info['temperatures'] = temperatures
        info['env_time_get_motor_feedback'] = time.time() - time_start

        return info

    def _recover_from_error(self, error):
        # Dispatch recovery based on which subsystem failed.
        if isinstance(error, MotorControllerError):
            print(f"Motor controller error: {error}; recovering...")
            self.motor_controller.recover_from_error()
        elif isinstance(error, IMUError):
            # The IMU thread reconnects in the background; just pace the retry.
            print(f"IMU error: {error}; waiting for IMU thread to recover...")
            time.sleep(0.1)
        elif isinstance(error, CameraError):
            # The tracker thread reconnects in the background; just pace the retry.
            print(f"Camera error: {error}; waiting for tracker thread to recover...")
            time.sleep(0.1)

    def _get_observation_with_recovery(self, error):
        # Recover from the failed subsystem and re-acquire an observation until it
        # succeeds. `error` is the exception that triggered recovery.
        while True:
            self._recover_from_error(error)
            try:
                return self.get_observation()
            except (MotorControllerError, IMUError, CameraError) as retry_error:
                error = retry_error

    def tracker_lost(self, info):
        if time.time() - self.last_seen > 2:
            print('body tracker not seen for 2 seconds')
            return True
        if 'body' in info['bodies']:
            img_pos = info['bodies']['body']['image_pos']
            if img_pos[0] < 0.1 or img_pos[0] > 0.9 or img_pos[1] < 0.1 or img_pos[1] > 0.9:
                print('body is out of camera frame')
                return True # body is out of frame
        return False

    def close(self):
        self._threads_should_exit = True
        self._imu_thread.join()
        self._tracker_thread.join()
        self.motor_controller.disable()

    def _poll_imu(self):
        while not self._threads_should_exit:
            try:
                imu_data = self.imu.get_data()
                with self._imu_data_lock:
                    self._imu_data = imu_data
            except Exception as e:
                print(f"Error in _poll_imu: {e}; reopening IMU port")
                with self._imu_data_lock:
                    self._imu_data = None
                try:
                    self.imu.reopen()
                except Exception as reopen_error:
                    print(f"Failed to reopen IMU port: {reopen_error}")
                    time.sleep(0.1)

    def _wait_for_tracker_frame(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._tracker_data_lock:
                if self._tracker_data is not None:
                    self.vis_frame = self._tracker_data[0][2]
                    return
            time.sleep(0.01)
        raise RuntimeError('Tracker camera frame not available yet')

    def _poll_tracker(self):
        print("Starting tracker camera thread")
        while not self._threads_should_exit:
            try:
                data = self.tracker.track()
                with self._tracker_data_lock:
                    self._tracker_data = (data, time.time())
            except Exception as e:
                print(f"Error in _poll_tracker: {e}; reopening camera")
                with self._tracker_data_lock:
                    self._tracker_data = None
                try:
                    self.tracker.reopen()
                except Exception as reopen_error:
                    print(f"Failed to reopen tracker camera: {reopen_error}")
                    time.sleep(0.1)

def make_ant_env(cfg, **kwargs):
    motor_controller = MotorController(port=cfg['motor_port'], motor_list=cfg['motor_list'])
    imu = IMU_MSP(port=cfg['imu_port'])
    tracker = VisionTracker(camera_id=cfg['camera_id'],
                            fov_diagonal_deg=cfg['camera_fov_diagonal_deg'],
                            tag_sizes=cfg['camera_tag_sizes'],
                            tag_ids=cfg['camera_tag_ids'])
    return EmbodiedAnt(motor_controller=motor_controller, imu=imu, tracker=tracker, **kwargs)

class DummyMotorController:
    def __init__(self, port=None, motor_list=[0]*8):
        self.nb_motors = len(motor_list)
    def set_positions(self, positions):
        pass
    def get_feedback(self):
        return np.zeros(self.nb_motors), np.zeros(self.nb_motors), np.zeros(self.nb_motors)
    def disable(self):
        pass
    def enable(self):
        pass
    def get_temperature(self):
        return np.zeros(self.nb_motors)
    def check_errors(self):
        return []
    def recover_from_error(self):
        pass

class DummyIMU:
    def __init__(self, port=None):
        pass
    def get_data(self):
        return {'ax': 0, 'ay': 0, 'az': 9.81,
                'wx': 0, 'wy': 0, 'wz': 0,
                'mx': 0, 'my': 0, 'mz': 0,
                'roll_deg': 0, 'pitch_deg': 0, 'yaw_deg': 0,
                'timestamp': time.time()}

class DummyTracker:
    def __init__(self, detector=None, inertial_tag_id=None):
        pass
    def track(self):
        return {}, np.zeros((640, 480, 3)), np.zeros((640, 480, 3))
    def reopen(self):
        pass


if __name__ == "__main__":
    import sys
    import json
    cfg = json.load(open(sys.argv[1]))
    motor_controller = MotorController(port=cfg['motor_port'], motor_list=cfg['motor_list'])
    # motor_controller = DummyMotorController()
    imu = IMU_MSP(port=cfg['imu_port'])
    # imu = DummyIMU()
    print(cfg)
    tracker = VisionTracker(camera_id=cfg['camera_id'],
                            fov_diagonal_deg=cfg['camera_fov_diagonal_deg'],
                            tag_sizes=cfg['camera_tag_sizes'],
                            tag_ids=cfg['camera_tag_ids'])
    env = EmbodiedAnt(motor_controller=motor_controller, imu=imu, tracker=tracker, dt=0.05)
    i = 0

    while True:
        time_now = time.time()
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        # if (i := i + 1) % 10 == 0:
        #     show_image(info['vis_frame'])
