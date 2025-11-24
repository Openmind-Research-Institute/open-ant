import os
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt


class RewardTracker:
    def __init__(self, env_dt, env_id, time_window=10.0, log_folder=".", plot=False, frequency_of_logging=100):
        self.env_dt = env_dt
        self.window_size = int(time_window / env_dt)
        self.queue = deque(maxlen=self.window_size)
        self.df = pd.DataFrame(columns=["step", "reward"])
        self.buffer = []
        self.log_folder = log_folder
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        self.env_id = env_id
        self.step = 0.0
        self._average_reward_per_second = 0.0
        self._plot = plot
        self._frequency_of_logging = frequency_of_logging
        self._save_path = log_folder

    def update(self, reward):
        reward_per_second = reward / self.env_dt
        self.queue.append(reward_per_second)
        self.step += 1

        self._average_reward_per_second = sum(self.queue) / len(self.queue)
        self.buffer.append([self.step, self._average_reward_per_second])

    @property
    def average_reward_per_second(self):
        return self._average_reward_per_second

    def log(self):
        # Flush buffer to DataFrame periodically.
        if self.step % self._frequency_of_logging == 0:
            if self.buffer:
                new_df = pd.DataFrame(self.buffer, columns=["step", "reward"])
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.buffer.clear()

            self.df.to_csv(os.path.join(self.log_folder, f"{self.env_id}_average_rewards.csv"), index=False)

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.df["step"][self.window_size:] * self.env_dt,
            self.df["reward"][self.window_size:],
            color="black",
            linewidth=1.0,
        )
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Average Reward per Second", fontsize=14)
        if self._save_path is not None:
            plt.savefig(os.path.join(self._save_path, f"{self.env_id}_average_rewards.png"))
            plt.close()