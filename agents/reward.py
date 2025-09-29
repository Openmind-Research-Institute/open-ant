import os
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


class RewardTracker:
    def __init__(self, env_dt, env_id, time_window=10.0, log_folder="."):
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

    def update(self, reward):
        reward_per_second = reward / self.env_dt
        self.queue.append(reward_per_second)
        self.step += 1

        average_reward_per_second = sum(self.queue) / len(self.queue)
        self.buffer.append([self.step, average_reward_per_second])

    def log(self, plot=False):
        # Flush buffer to DataFrame periodically.
        if self.step % 100 == 0:
            if self.buffer:
                new_df = pd.DataFrame(self.buffer, columns=["step", "reward"])
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.buffer.clear()

            self.df.to_csv(os.path.join(self.log_folder, f"{self.env_id}_average_rewards.csv"), index=False)

            if plot:
                self.plot(os.path.join(self.log_folder, f"{self.env_id}_average_rewards.png"))

    def plot(self, save_path=None):
        plt.plot(
            self.df["step"] * self.env_dt,
            self.df["reward"],
            color="black",
            linewidth=1.0,
        )
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Average Reward per Second", fontsize=14)
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()