import os
import pandas as pd
from collections import deque

class RewardTracker:
    def __init__(self, env_dt, env_id, time_window=10.0, log_folder=".", date_now="log"):
        self.env_dt = env_dt
        self.window_size = int(time_window / env_dt)
        self.queue = deque(maxlen=self.window_size)
        self.df = pd.DataFrame(columns=["step", "reward"])
        self.log_folder = log_folder
        self.date_now = date_now
        self.env_id = env_id

    def update(self, reward):
        reward_per_second = reward / self.env_dt
        self.queue.append(reward_per_second)

        average_reward_per_second = sum(self.queue) / len(self.queue)
        return average_reward_per_second

    def log(self, step, average_reward_per_second, log_interval=5000):
        print(f"Step {step}, time [s] {step * self.env_dt:.2f}, "
                f"time [min] {step * self.env_dt / 60:.2f}, "
                f"moving average reward {average_reward_per_second:.4f}")
        self.df = pd.concat(
            [self.df, pd.DataFrame({"step": [step], "reward": [average_reward_per_second]})],
            ignore_index=True
        )
        self.df.to_csv(os.path.join(self.log_folder, f"eval_{self.env_id}_rewards_{self.date_now}.csv"), index=False)