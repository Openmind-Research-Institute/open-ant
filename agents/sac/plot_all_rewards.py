import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/asymmetric_update"
SEEDS = [0, 1, 2, 3, 4, 5]

for seed in SEEDS:
    # Find sim1 and sim2 folders for this seed
    sim1_dirs = sorted(glob.glob(f"{RUNS_DIR}/trial_4_2*_seed_{seed}"), reverse=True)
    sim2_dirs = sorted(glob.glob(f"{RUNS_DIR}/trial_4_asym_continual_learning_*_seed_{seed}"), reverse=True)

    if not sim1_dirs:
        print(f"Seed {seed}: no sim1 folder found, skipping.")
        continue
    if not sim2_dirs:
        print(f"Seed {seed}: no sim2 folder found, skipping.")
        continue

    sim1_csv = os.path.join(sim1_dirs[0], "SimEmbodiedAnt_average_rewards.csv")
    sim2_csv = os.path.join(sim2_dirs[0], "SimEmbodiedAnt_average_rewards.csv")

    if not os.path.exists(sim1_csv):
        print(f"Seed {seed}: sim1 CSV not found at {sim1_csv}, skipping.")
        continue
    if not os.path.exists(sim2_csv):
        print(f"Seed {seed}: sim2 CSV not found at {sim2_csv}, skipping.")
        continue

    df1 = pd.read_csv(sim1_csv)
    df2 = pd.read_csv(sim2_csv)

    last_step_1 = df1["step"].max()
    df2["step"] = df2["step"] + last_step_1
    df_all = pd.concat([df1, df2], ignore_index=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df_all["step"], df_all["reward"], linewidth=1)
    plt.axvline(last_step_1, linestyle="--", linewidth=1, color="red", label="continual learning starts")
    plt.xlabel("Total Step")
    plt.ylabel("Reward")
    plt.title(f"Reward vs Step: Sim + Continual Learning (seed {seed})")
    plt.legend()
    plt.grid(True)
    out_path = f"asym_combined_reward_plot_seed{seed}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Seed {seed}: saved {out_path}")

print("Done!")
