import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path_1 = "/home/seliu/open-ant/agents/mpo/runs_continous_learning/retrace_20260604-074801_seed_1/SimEmbodiedAnt_average_rewards.csv"
csv_path_2 = "/home/seliu/open-ant/agents/mpo/runs_continous_learning/retrace_continual_learning_20260604-180557_seed_1/SimEmbodiedAnt_average_rewards.csv"

output_dir = "/home/seliu/open-ant/agents/mpo/runs_continous_learning"
figure_name = "seed1_mpo.png"

df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

# Make second CSV continue after first CSV
last_step_1 = df1["step"].max()
df2["step"] = df2["step"] + last_step_1

# Combine them
df_all = pd.concat([df1, df2], ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(df_all["step"], df_all["reward"], linewidth=1)

# Optional vertical line showing where second run starts
plt.axvline(last_step_1, linestyle="--", linewidth=1)
#plt.text(last_step_1, df_all["reward"].max(), " second sim starts", va="top")

plt.xlabel("Total Step")
plt.ylabel("Reward")
plt.title("Reward vs Step: Sim + Continual Learning")
plt.grid(True)

plt.savefig(os.path.join(output_dir, figure_name), dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to {os.path.join(output_dir, figure_name)}")