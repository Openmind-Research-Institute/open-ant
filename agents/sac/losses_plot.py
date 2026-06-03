import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# USER SETTINGS
# =====================================================

csv_path_1 = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/mixing-buffer/trial_2_20260529-153539_seed_2/info_sac_logs.csv"
csv_path_2 = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/mixing-buffer/trial_2_continual_learning_20260529-154337_seed_2/info_sac_logs.csv"

output_dir = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/mixing-buffer"

os.makedirs(output_dir, exist_ok=True)

# =====================================================
# LOAD CSVS
# =====================================================

df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

# Stack rows from csv2 underneath csv1
df = pd.concat([df1, df2], ignore_index=True)

# =====================================================
# CONVERT TO NUMERIC
# =====================================================

numeric_cols = [
    "global_step",
    "qf1_loss",
    "qf2_loss",
    "actor_loss",
    "alpha_loss",
    "alpha",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort by global step just in case
df = df.sort_values("global_step")

# =====================================================
# PLOTS
# =====================================================

fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

# -----------------------------------------------------
# QF1 LOSS
# -----------------------------------------------------

valid = df[["global_step", "qf1_loss"]].dropna()

if len(valid) > 0:
    axes[0].plot(
        valid["global_step"],
        valid["qf1_loss"],
        linewidth=1,
    )

axes[0].set_ylabel("qf1_loss")
axes[0].set_title("QF1 Loss")
axes[0].grid(True)
axes[0].set_ylim(0, 0.3)

# -----------------------------------------------------
# QF2 LOSS
# -----------------------------------------------------

valid = df[["global_step", "qf2_loss"]].dropna()

if len(valid) > 0:
    axes[1].plot(
        valid["global_step"],
        valid["qf2_loss"],
        linewidth=1,
    )

axes[1].set_ylabel("qf2_loss")
axes[1].set_title("QF2 Loss")
axes[1].grid(True)
axes[1].set_ylim(0, 0.3)

# -----------------------------------------------------
# ACTOR LOSS
# -----------------------------------------------------

valid = df[["global_step", "actor_loss"]].dropna()

if len(valid) > 0:
    axes[2].plot(
        valid["global_step"],
        valid["actor_loss"],
        linewidth=1,
    )

axes[2].set_ylabel("actor_loss")
axes[2].set_title("Actor Loss")
axes[2].grid(True)

# -----------------------------------------------------
# ALPHA LOSS
# -----------------------------------------------------

valid = df[["global_step", "alpha_loss"]].dropna()

if len(valid) > 0:
    axes[3].plot(
        valid["global_step"],
        valid["alpha_loss"],
        linewidth=1,
    )

axes[3].set_ylabel("alpha_loss")
axes[3].set_title("Alpha Loss")
axes[3].grid(True)

# -----------------------------------------------------
# ALPHA
# -----------------------------------------------------

valid = df[["global_step", "alpha"]].dropna()

if len(valid) > 0:
    axes[4].plot(
        valid["global_step"],
        valid["alpha"],
        linewidth=1,
    )

axes[4].set_ylabel("alpha")
axes[4].set_title("Entropy Coefficient Alpha")
axes[4].grid(True)

axes[4].set_xlabel("Global Step")

plt.tight_layout()

save_path = os.path.join(
    output_dir,
    "seed2_combined_sac_metrics.png"
)

plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved plot to: {save_path}")