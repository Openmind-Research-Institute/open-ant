import numpy as np
import matplotlib.pyplot as plt

from skrl.utils import postprocessing

labels = []
rewards = []

tensorboard_iterator = postprocessing.TensorboardFileIterator("sac_skrl/logs_sac_skrl/*/events.out.tfevents.*",
                                                              tags=["Reward / Total reward (mean)"])
for dirname, data in tensorboard_iterator:
    rewards.append(data["Reward / Total reward (mean)"])
    labels.append(dirname)


fig, ax = plt.subplots(1, 1, figsize=(15, 5))

for reward, label in zip(rewards, labels):
    reward_np = np.array(reward)
    ax.plot(reward_np[:,0], reward_np[:,1], label=label)
ax.set_title("Total reward (for each experiment)")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Reward")
ax.grid(True)
ax.legend()
plt.show()
plt.savefig("total_reward.png")