import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# USER SETTINGS
# =====================================================
csv_path_1 = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/warm_start/trial_1_20260531-151730_seed_0/info_sac_logs.csv"
csv_path_2 = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/warm_start/trial_1_warmstart_continual_learning_20260531-152601_seed_0/info_sac_logs.csv"
output_dir = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/warm_start"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# LOAD & CONCATENATE WITH CUMULATIVE STEPS
# =====================================================
df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

df1["global_step"] = pd.to_numeric(df1["global_step"], errors="coerce")
df2["global_step"] = pd.to_numeric(df2["global_step"], errors="coerce")

# Offset df2 steps so they continue after df1
step_offset = df1["global_step"].max()
df2 = df2.copy()
df2["global_step"] = df2["global_step"] + step_offset

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
save_path = os.path.join(output_dir, "seed0_warmstart_sac_metrics.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved plot to: {save_path}")