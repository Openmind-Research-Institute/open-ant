import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/vanilla_sac"

# ← Edit this list to include whichever seeds you want
SEEDS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.figure(figsize=(14, 6))

for seed in SEEDS_TO_PLOT:
    sim1_dirs = sorted(glob.glob(f"{RUNS_DIR}/trial_1_2*_seed_{seed}"), reverse=True)
    sim2_dirs = sorted(glob.glob(f"{RUNS_DIR}/trial_1_continual_learning_*_seed_{seed}"), reverse=True)

    if not sim1_dirs or not sim2_dirs:
        print(f"Seed {seed}: folders not found, skipping.")
        continue

    sim1_csv = os.path.join(sim1_dirs[0], "SimEmbodiedAnt_average_rewards.csv")
    sim2_csv = os.path.join(sim2_dirs[0], "SimEmbodiedAnt_average_rewards.csv")

    if not os.path.exists(sim1_csv) or not os.path.exists(sim2_csv):
        print(f"Seed {seed}: CSV not found, skipping.")
        continue

    df1 = pd.read_csv(sim1_csv)
    df2 = pd.read_csv(sim2_csv)

    last_step_1 = df1["step"].max()
    df2["step"] = df2["step"] + last_step_1
    df_all = pd.concat([df1, df2], ignore_index=True)

    # Plot the reward curve and grab its color
    line, = plt.plot(df_all["step"], df_all["reward"], linewidth=1, label=f"seed {seed}")
    
    # Draw vertical line in same color, only label first one to avoid legend clutter
    plt.axvline(last_step_1, linestyle="--", linewidth=0.8, color=line.get_color(),
                label="_nolegend_")

# Single legend entry for the vertical lines
plt.axvline(-1, linestyle="--", linewidth=0.8, color="gray", label="continual learning starts")

plt.xlabel("Total Step")
plt.ylabel("Reward")
plt.title("Reward vs Step: All Seeds Overlaid")
plt.legend(fontsize=8)
plt.grid(True)
plt.savefig("vanilla_combined_reward_plot_all_seeds.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved")