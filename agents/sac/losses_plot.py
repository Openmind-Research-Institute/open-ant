import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# USER SETTINGS
# =====================================================
csv_path_1 = "/home/seliu/open-ant/agents/sac/runs_sim_less_aggresive/vanilla_sac_mpoparam/trial_1_20260603-182257_seed_0/info_sac_logs.csv"
csv_path_2 = "/home/seliu/open-ant/agents/sac/runs_sim_less_aggresive/vanilla_sac_mpoparam/trial_1_continual_learning_20260603-185340_seed_0/info_sac_logs.csv"
output_dir = "/home/seliu/open-ant/agents/sac/runs_sim_less_aggresive/vanilla_sac_mpoparam"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# LOAD & CONCATENATE WITH CUMULATIVE STEPS
# =====================================================
df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

df1["global_step"] = pd.to_numeric(df1["global_step"], errors="coerce")
df2["global_step"] = pd.to_numeric(df2["global_step"], errors="coerce")

step_offset = df1["global_step"].max()
df2_start = df2["global_step"].min()

# If df2 already continues from where df1 left off, concat directly.
# If df2 resets (starts near 1), offset it.
if df2_start < step_offset * 0.1:  # starts way below df1's max -> reset
    df2 = df2.copy()
    df2["global_step"] = df2["global_step"] + step_offset
    print(f"Detected reset: offsetting df2 steps by {step_offset}")
else:
    print(f"Detected continuation: df2 already starts at {df2_start}, no offset needed")

transition_step = step_offset

# Concatenate in order — NO sort_values, order is already correct
df = pd.concat([df1, df2], ignore_index=True)
# =====================================================
# CONVERT REMAINING COLS TO NUMERIC
# =====================================================
numeric_cols = ["qf1_loss", "qf2_loss", "actor_loss", "alpha_loss", "alpha"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =====================================================
# PLOTS
# =====================================================
fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

# Add a vertical line to mark the transition between runs
transition_step = step_offset

def plot_metric(ax, col, ylabel, title, ylim=None):
    valid = df[["global_step", col]].dropna()
    if len(valid) > 0:
        ax.plot(valid["global_step"], valid[col], linewidth=1)
        ax.axvline(x=transition_step, color="red", linestyle="--", linewidth=1, label="warm start")
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if ylim:
        ax.set_ylim(ylim)

plot_metric(axes[0], "qf1_loss",   "qf1_loss",   "QF1 Loss",                   ylim=(0, 0.3))
plot_metric(axes[1], "qf2_loss",   "qf2_loss",   "QF2 Loss",                   ylim=(0, 0.3))
plot_metric(axes[2], "actor_loss", "actor_loss", "Actor Loss")
plot_metric(axes[3], "alpha_loss", "alpha_loss", "Alpha Loss")
plot_metric(axes[4], "alpha",      "alpha",      "Entropy Coefficient Alpha")

axes[4].set_xlabel("Global Step")

plt.tight_layout()
save_path = os.path.join(output_dir, "seed0_vanilla_sac_metrics.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved plot to: {save_path}")