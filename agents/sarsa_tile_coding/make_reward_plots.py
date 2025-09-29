import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PLOTS_DIR = 'plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

DT = 0.05
time_window = 120.0
steps_to_start = int(time_window/DT)

folders_list = ['logs/20250927_140937',
                'logs/20250926_194842',
                'logs/20250927_162143_merged']

rewards_df_list = []
for f in folders_list:
    df = pd.read_csv(os.path.join(f, "rewards.csv"))
    # Cut at 3600 seconds
    df = df[df["real_time_seconds"] < 3600]
    # Check if there exists rewards2.csv and if yes, concatenate it to the df
    if os.path.exists(os.path.join(f, "rewards2.csv")):
        df2 = pd.read_csv(os.path.join(f, "rewards2.csv"))
        df2 = df2[df2["real_time_seconds"] < 3600]
        # Add the last time of the first df to the second df
        df2["real_time_seconds"] += df["real_time_seconds"].iloc[-1]
        df = pd.concat([df, df2], ignore_index=True)
    rewards_df_list.append(df)

folders_for_average_reward = ['logs/20250927_140937']
average_rewards_df_list = []
for f in folders_for_average_reward:
    df = pd.read_csv(os.path.join(f, "eval_run_ant_hw_rewards.csv"))
    average_rewards_df_list.append(df)

# ---- Compute mean & std across runs ----
# Create a common time grid across all runs
max_time = max(df["real_time_seconds"].max() for df in rewards_df_list)
time_grid = np.linspace(0, max_time, 500)

# Interpolate rewards for each run onto the common grid
interpolated_rewards = []
for df in rewards_df_list:
    interp = np.interp(time_grid, df["real_time_seconds"], df["reward"])
    interpolated_rewards.append(interp)

interpolated_rewards = np.vstack(interpolated_rewards)
mean_reward = interpolated_rewards.mean(axis=0)
std_reward = interpolated_rewards.std(axis=0)

# ---- Plot ----
sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl", n_colors=len(rewards_df_list))

fig, ax1 = plt.subplots(dpi=150)
plt.grid(False)

# Plot individual runs
for i, (df, color) in enumerate(zip(rewards_df_list, palette)):
    ax1.plot(
        df["real_time_seconds"]/3600.0,
        df["reward"],
        label=f"Run {i}",
        alpha=0.6,
        marker='o',
        markersize=3,
        linewidth=1.0,
    )

# Plot mean with std shading
ax1.plot(time_grid/3600.0, mean_reward, color="black", linewidth=1, label="Mean")
ax1.fill_between(
    time_grid/3600.0,
    mean_reward - std_reward,
    mean_reward + std_reward,
    alpha=0.2,
    label="±1 Std"
)

ax1.set_xlabel("Real Time (hours)", fontsize=14)
ax1.set_title('Return per Time Limit Episode', fontsize=16, fontweight="bold", pad=15)
ax1.set_ylabel("Return", fontsize=14)
ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
ax1.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "return_over_time.png"), dpi=300, bbox_inches="tight")
plt.show()



DT = 0.05
time_window = 120.0
steps_to_start = int(time_window/DT)

fig, ax1 = plt.subplots(dpi=150)
plt.grid(False)
# Plot all runs
for i, (df, color) in enumerate(zip(average_rewards_df_list, palette)):

    ax1.plot(
        df["step"][steps_to_start:]*0.05,
        df["reward"][steps_to_start:]*1000,
        label=f"Run {i + 1}",
        color=palette[i + 1],
        alpha=1.0,
        linewidth=1.5,
    )

# Primary x-axis: episodes
ax1.set_xlabel("Time [s]", fontsize=14)
ax1.set_title('Average reward per second (120s window)', fontsize=16, fontweight="bold", pad=15)
ax1.set_ylabel("Average reward per second [mm/s]", fontsize=14)
ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
ax1.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "average_reward_over_time.png"), dpi=300, bbox_inches="tight")
plt.show()



# # --- Derive time ticks from episodes ---
# df0 = rewards_df_list[0]
# episodes = df0["episode"].values + 1
# times_h = df0["real_time_seconds"].values / 3600.0  # convert to hours

# # Create a mapping: episode -> hours (linear interpolation)
# def episode_to_hours(ep):
#     return np.interp(ep, episodes, times_h)

# max_episode = episodes.max()
# episode_ticks = np.linspace(episodes.min(), max_episode, int(max_episode/2))
# time_ticks_h = episode_to_hours(episode_ticks)

# # Secondary x-axis
# ax2 = ax1.twiny()
# ax2.set_xlim(ax1.get_xlim())
# ax2.xaxis.set_ticks_position("bottom")
# ax2.xaxis.set_label_position("bottom")
# ax2.spines["bottom"].set_position(("outward", 40))

# ax1.set_xticks(episode_ticks)
# ax1.set_xticklabels([f"{int(e)}" for e in episode_ticks])
# ax2.set_xticks(episode_ticks)
# ax2.set_xticklabels([f"{t:.2f}h" for t in time_ticks_h])
# ax2.set_xlabel("Real Time (hours)", fontsize=12)

# Save and plot the trajectory.
# Generate a plot.
# plt.figure(dpi=150)
# for i, true_pos_xy_df in enumerate(true_pos_xy_df_list):
#     # Compute start, end, and distance
#     x0, y0 = true_pos_xy_df['x'].iloc[0], true_pos_xy_df['y'].iloc[0]
#     xf, yf = true_pos_xy_df['x'].iloc[-1], true_pos_xy_df['y'].iloc[-1]
#     distance = np.linalg.norm([xf - x0, yf - y0])

#     # Plot trajectory with smooth line and soft color
#     color = sns.color_palette("tab10")[i % 10]  # distinct color per trajectory
#     sns.lineplot(
#         x=true_pos_xy_df['x'],
#         y=true_pos_xy_df['y'],
#         linewidth=2,
#         alpha=0.8,
#         color=color,
#         label=f"traj. seed {seeds[i]}"
#     )

#     # Mark start (big red dot) and end (big green dot)
#     plt.scatter(x0, y0, color="crimson", s=80, edgecolor="black", zorder=5)
#     plt.scatter(xf, yf, color="limegreen", s=80, edgecolor="black", zorder=5)

#     offset = 0.1  # adjust as needed (in data units)
#     plt.text(
#         xf + offset, yf + offset,  # offset in both x and y
#         f"{distance:.2f} m",
#         fontsize=10,
#         fontweight="bold",
#         color=color,  # match trajectory color
#         ha="left",
#         va="bottom",
#         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.6)
#     )
#     # Axis labels and title
#     plt.xlabel("X Position", fontsize=13)
#     plt.ylabel("Y Position", fontsize=13)

# # Equal aspect ratio for proper geometry
# plt.axis("equal")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend(frameon=True, fontsize=11)

# plt.tight_layout()
# plt.savefig(os.path.join(PLOTS_DIR, "trajectory_hw.png"), dpi=300, bbox_inches="tight")

# plt.show()
