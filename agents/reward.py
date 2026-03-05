import os
import csv
import pandas as pd
from collections import deque


class RewardTracker:
    def __init__(self, env_dt, env_id, time_window=10.0, log_folder=".", logging_freq=1000):
        self.env_dt = env_dt
        self.env_id = env_id

        self.window_size = int(time_window / env_dt)
        self.queue = deque(maxlen=self.window_size)
        self.buffer = []
        self._queue_sum = 0.0 # Running sum for efficient average calculation.

        self.log_folder = log_folder
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        self.step = 0.0
        self._average_reward_per_second = 0.0
        self._logging_freq = logging_freq

        self.csv_path = os.path.join(self.log_folder, f"{self.env_id}_average_rewards.csv")
        self._csv_file_exists = os.path.exists(self.csv_path) # Cache file existence check.

    def update(self, reward):
        reward_per_second = reward / self.env_dt
        if len(self.queue) == self.window_size:
            # Queue is full, remove oldest value
            self._queue_sum -= self.queue[0]
        self.queue.append(reward_per_second)
        self._queue_sum += reward_per_second

        self.step += 1
        self._average_reward_per_second = self._queue_sum / len(self.queue)

        self.buffer.append([self.step, self._average_reward_per_second])

    @property
    def average_reward_per_second(self):
        return self._average_reward_per_second

    def log(self):
        if self.buffer and self.step % self._logging_freq == 0:
            with open(self.csv_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not self._csv_file_exists:
                    writer.writerow(["step", "reward"])
                    self._csv_file_exists = True
                writer.writerows(self.buffer)
            self.buffer.clear()
