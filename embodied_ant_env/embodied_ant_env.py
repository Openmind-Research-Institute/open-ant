import numpy as np
import threading
import time
from collections import defaultdict

from imu_msp import IMU_MSP
from motor_controller import MotorController
from apriltag_tracking import VisionTracker, show_image

class Space:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

class EmbodiedAnt:
    action_space = Space(shape=(8,), dtype=np.float32)
    observation_space = Space(shape=(24,), dtype=np.float32)

    def __init__(self, motor_controller, imu, tracker, step_size=0.02, render_mode=None):
        self.motor_controller = motor_controller
        self.motor_controller.enable()
        self.step_size = step_size
        self.last_step_time = None
        self.render_mode = render_mode
        self.i = 0

        self._threads_should_exit = False

        self.imu = imu
        self._imu_data = None
        self._imu_data_lock = threading.Lock()
        self._imu_thread = threading.Thread(target=self._poll_imu, daemon=True)
        self._imu_thread.start()

        self.tracker = tracker
        self._tracker_data = None
        self._tracker_data_lock = threading.Lock()
        self._tracker_thread = threading.Thread(target=self._poll_tracker, daemon=True)
        self._tracker_thread.start()

        self.last_pos = np.array([0, 0, 0])
        self.last_heading_vector = np.array([1, 0])
        self.last_seen = 0

    def __del__(self):
        self.close()

    def reset(self):
        print('reset(): please move the ant back to the origin.')
        input('press enter when ready')
        obs, info = self.get_observation()
        self.get_reward(info)
        return obs, info

    def step(self, action, sleep_until_next_step=True):
        if self._threads_should_exit:
            raise RuntimeError("EmbodiedAnt.step() called after close()")

        self.motor_controller.set_positions(action)

        sleep_duration = self.step_size
        if self.last_step_time is not None:
            time_since_last_step = time.time() - self.last_step_time
            sleep_duration = self.step_size - time_since_last_step
            if sleep_duration < 0:
                print(f"Warning: calls to step() exceeded step size (time since last step: {time_since_last_step:.3f}s).")
                sleep_duration = 0
        if sleep_until_next_step:
            time.sleep(sleep_duration)

        observation, info = self.get_observation()
        reward, terminated, truncated = self.get_reward(info)

        if self.render_mode == 'human':
            self.i += 1
            if self.i % 10 == 0:
                show_image(info['vis_frame'])

        self.last_step_time = time.time()
        return observation, reward, terminated, truncated, info

    def get_observation(self):
        with self._imu_data_lock:
            if self._imu_data is not None:
                imu_data = self._imu_data.copy()
            else:
                imu_data = defaultdict(lambda: 0)
        with self._tracker_data_lock:
            if self._tracker_data is not None:
                bodies, frame, vis_frame = self._tracker_data
            else:
                bodies, frame, vis_frame = {}, np.zeros((640, 480, 3)), np.zeros((640, 480, 3))
        joint_positions, joint_velocities, joint_loads = self.motor_controller.get_feedback()
        info = imu_data
        info['joint_positions'] = joint_positions
        info['joint_velocities'] = joint_velocities
        info['joint_loads'] = joint_loads
        info['bodies'] = bodies
        info['frame'] = frame
        info['vis_frame'] = vis_frame
        if 'body' in bodies:
            heading_vector = (bodies['body']['orientation'] @ np.array([1, 0, 0]))[:2]
            heading_vector /= np.linalg.norm(heading_vector)
            self.last_heading_vector = heading_vector
            info['heading_vector'] = heading_vector
        else:
            heading_vector = self.last_heading_vector
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            heading_vector,
            imu_data['ax'],
            imu_data['ay'],
            imu_data['az'],
            imu_data['wx'],
            imu_data['wy'],
            imu_data['wz'],
        ], axis=None)
        return observation, info

    def get_reward(self, info):
        if 'body' in info['bodies']:
            pos = info['bodies']['body']['position']
            self.last_seen = time.time()
        else:
            pos = self.last_pos
        progress = (pos - self.last_pos)[0]
        self.last_pos = pos
        terminated = False
        truncated = False
        if time.time() - self.last_seen > 2:
            truncated = True
        if 'body' in info['bodies']:
            img_pos = info['bodies']['body']['image_pos']
            if img_pos[0] < 0.1 or img_pos[0] > 0.9 or img_pos[1] < 0.1 or img_pos[1] > 0.9:
                truncated = True # body is out of frame
        return progress, terminated, truncated

    def close(self):
        self._threads_should_exit = True
        self._imu_thread.join()
        self.motor_controller.disable()

    def _poll_imu(self):
        while not self._threads_should_exit:
            try:
                imu_data = self.imu.get_data()
                with self._imu_data_lock:
                    self._imu_data = imu_data
            except Exception as e:
                print(f"Error in _poll_imu: {e}")
                self._threads_should_exit = True

    def _poll_tracker(self):
        while not self._threads_should_exit:
            try:
                data = self.tracker.track()
                with self._tracker_data_lock:
                    self._tracker_data = data
            except Exception as e:
                print(f"Error in _poll_tracker: {e}")
                self._threads_should_exit = True

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
    env = EmbodiedAnt(motor_controller=motor_controller, imu=imu, tracker=tracker)
    i = 0
    while True:
        # time.sleep(1)
        obs, rew, term, trunc, info = env.step(np.zeros(8))
        # obs, rew, term, trunc, info = env.step(np.random.uniform(-0.3, 0.3, 8))
        # print(obs)
        print(rew)
        # print(term)
        # print(trunc)
        # print(info)

        if (i := i + 1) % 10 == 0:
            show_image(info['vis_frame'])
