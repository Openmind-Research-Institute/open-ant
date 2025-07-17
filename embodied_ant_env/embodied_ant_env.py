import numpy as np
from imu_msp import IMU_MSP
import threading
import time

class MotorController:
    def __init__(self, port):
        pass


class Space:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class EmbodiedAnt:
    action_space = Space(shape=(8,), dtype=np.float32)
    observation_space = Space(shape=(10,), dtype=np.float32)

    def __init__(self, motor_port, imu_port, step_size=0.02):
        self.motor_port = motor_port
        self.imu_port = imu_port
        self.motor_controller = MotorController(motor_port)
        self.step_size = step_size
        self.last_step_time = None

        self._threads_should_exit = False

        self.imu = IMU_MSP(imu_port)
        self._imu_data = None
        self._imu_data_lock = threading.Lock()
        self._imu_thread = threading.Thread(target=self._poll_imu, daemon=True)
        self._imu_thread.start()

    def __del__(self):
        self.close()

    def reset(self):
        pass

    def step(self, action, sleep_until_next_step=True, sync_observation_with_action=False):
        if sync_observation_with_action:
            observation = self.get_observation()

        # send actuators
        # self.motor_controller.send_actuators(action)

        sleep_duration = self.step_size
        if self.last_step_time is not None:
            time_since_last_step = time.time() - self.last_step_time
            sleep_duration = self.step_size - time_since_last_step
            if sleep_duration < 0:
                print(f"Warning: calls to step() exceeded step size (time since last step: {time_since_last_step:.3f}s).")
                sleep_duration = 0
        if sleep_until_next_step:
            time.sleep(sleep_duration)

        if not sync_observation_with_action:
            observation = self.get_observation()

        self.last_step_time = time.time()
        return observation

    def get_observation(self):
        with self._imu_data_lock:
            imu_data = self._imu_data
        return imu_data

    def render(self, mode='human'):
        if mode == 'human':
            print("render(mode='human') this is the real world, look at your robot!")
        elif mode == 'rgb_array':
            return self.camera.get_image()

    def close(self):
        self._threads_should_exit = True
        self._imu_thread.join()

    def _poll_imu(self):
        while not self._threads_should_exit:
            imu_data = self.imu.get_data()
            with self._imu_data_lock:
                self._imu_data = imu_data


if __name__ == "__main__":
    import sys
    env = EmbodiedAnt(motor_port=sys.argv[1], imu_port=sys.argv[2])
    while True:
        # time.sleep(0.03)
        print(env.step(np.zeros(8)))
