import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import sys
sys.path.insert(0, "/home/serena-liu/open-ant")  # adjust to your repo root
from agents.reward import RewardTracker
import json
# =====================================================
# USER SETTINGS
# =====================================================

RUN_CONFIGS = [
    {
        "csv":   "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/sac_target_critic/trial_1_20260604-135912_seed_0/info_sac_logs.csv",
        "label": "baseline",
        "color": "#1f77b4",
    },
    {
        "csv":   "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/sac_target_critic/trial_1_continual_learning_20260604-140529_seed_0/info_sac_logs.csv",
        "label": "CL",
        "color": "#ff7f0e",
    },
    # add more runs here if needed
]

OUTPUT_DIR  = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/sac_target_critic"
OUTPUT_NAME = "sac_dashboard.png"
SMOOTH      = 10        # rolling window for smoothing
DT          = 0.12      # env timestep — used to convert steps → sim minutes

# =====================================================

def smooth(series, w):
    return series.rolling(w, min_periods=1).mean()

def parse_reward_col(series):
    """Handles both scalar floats and '[val]' string lists in rewards column."""
    def _parse(v):
        if pd.isna(v):
            return np.nan
        try:
            return float(v)
        except (ValueError, TypeError):
            try:
                parsed = ast.literal_eval(str(v))
                if isinstance(parsed, (list, tuple)):
                    return float(parsed[0])
            except Exception:
                pass
        return np.nan
    return series.apply(_parse)

def load_run(cfg):
    df = pd.read_csv(cfg["csv"])
    
    # read dt from args.json sitting next to the csv
    run_dir = os.path.dirname(cfg["csv"])
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path) as f:
        run_args = json.load(f)
    dt = run_args["dt"]
    cfg["dt"] = dt  # store on cfg so build_runs can use it too
    
    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    df = df[df["global_step"].notna()].copy()
    df["global_step"] = df["global_step"].astype(float)

    # numeric cast for all metric cols
    metric_cols = [
        "qf1_loss", "qf2_loss", "actor_loss", "alpha", "alpha_loss",
        "mean_q1", "mean_q2", "mean_td_target", "mean_q_actor",
        "actor_entropy", "log_std_mean", "log_std_std",
        "actor_grad_norm", "critic_grad_norm",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # reward: parse list-strings too
    if "original_rewards" in df.columns:
        df["_reward"] = parse_reward_col(df["original_rewards"])
    elif "rewards" in df.columns:
        df["_reward"] = parse_reward_col(df["rewards"])

    df["sim_time_min"] = df["global_step"] * dt / 60.0  # use per-run dt
    return df
  


def build_runs(configs):
    runs = []
    for i, cfg in enumerate(configs):
        df = load_run(cfg)         # cfg["dt"] now set inside load_run
        dt = cfg["dt"]
        if i > 0:
            prev_max = runs[-1]["df"]["global_step"].max()
            run_start = df["global_step"].min()
            if run_start < prev_max * 0.1:
                df["global_step"] = df["global_step"] + prev_max
                df["sim_time_min"] = df["global_step"] * dt / 60.0
                cfg["offset"] = prev_max
            else:
                cfg["offset"] = 0.0
        else:
            cfg["offset"] = 0.0
        runs.append({"df": df, "cfg": cfg, "offset": cfg["offset"]})
    return runs


def add_transitions(ax, runs):
    """Vertical dashed lines at each run boundary except the first."""
    for run in runs[1:]:
        x = run["offset"] * DT / 60.0
        ax.axvline(x, color="red", linestyle="--", linewidth=0.9, alpha=0.7)


def plot_col(ax, runs, col, ylabel, title, ylim=None, log_scale=False):
    for run in runs:
        df  = run["df"]
        cfg = run["cfg"]
        if col not in df.columns:
            continue
        sub = df[["sim_time_min", col]].dropna()
        if sub.empty:
            continue
        s = smooth(sub[col], SMOOTH)
        ax.plot(sub["sim_time_min"], s,
                color=cfg["color"], linewidth=1.2,
                label=cfg["label"], alpha=0.85)
        # faint raw trace
        ax.plot(sub["sim_time_min"], sub[col],
                color=cfg["color"], linewidth=0.4, alpha=0.2)
    add_transitions(ax, runs)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if ylim:
        ax.set_ylim(ylim)
    if log_scale:
        ax.set_yscale("log")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs = build_runs(RUN_CONFIGS)

    # ── layout: 5 rows × 2 cols + 1 wide top row for reward ──────────────────
    fig = plt.figure(figsize=(14, 20))
    fig.suptitle(
        f"SAC training dashboard  —  {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=11, y=0.995
    )
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.32)

    ax_reward   = fig.add_subplot(gs[0, :])   # full width

    ax_qf1      = fig.add_subplot(gs[1, 0])
    ax_qf2      = fig.add_subplot(gs[1, 1])

    ax_q_vals   = fig.add_subplot(gs[2, 0])   # mean_q1, mean_q2, mean_td_target overlaid
    ax_q_actor  = fig.add_subplot(gs[2, 1])   # mean_q_actor

    ax_actor    = fig.add_subplot(gs[3, 0])
    ax_alpha    = fig.add_subplot(gs[3, 1])

    ax_entropy  = fig.add_subplot(gs[4, 0])
    ax_logstd   = fig.add_subplot(gs[4, 1])

    ax_gnorm_c  = fig.add_subplot(gs[5, 0])
    ax_gnorm_a  = fig.add_subplot(gs[5, 1])

    # ── reward ────────────────────────────────────────────────────────────────
    TIME_WINDOW = 120.0
    for run in runs:
        df, cfg = run["df"], run["cfg"]
        dt = cfg["dt"]

        if "original_rewards" not in df.columns:
            continue

        raw = parse_reward_col(df["original_rewards"])
        tracker = RewardTracker(env_dt=dt, env_id='plot',
                                time_window=TIME_WINDOW, log_folder='.')
        avg_rps = []
        for r in raw:
            tracker.update(r)
            avg_rps.append(tracker.average_reward_per_second)

        df = df.copy()
        df["_avg_rps"] = avg_rps

        warmup_steps = int(TIME_WINDOW / dt)
        df = df.iloc[warmup_steps:].copy()
        df["_reward_smooth"] = df["_avg_rps"].rolling(SMOOTH, min_periods=1).mean() * 100

        sub = df[["sim_time_min", "_reward_smooth"]].dropna()
        ax_reward.plot(sub["sim_time_min"], sub["_reward_smooth"],
                    color=cfg["color"], linewidth=1.4, label=cfg["label"])

    add_transitions(ax_reward, runs)
    ax_reward.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_reward.set_title("Avg reward/s [cm/s]", fontsize=9, pad=3)
    ax_reward.set_ylabel("reward [cm/s]", fontsize=8)
    ax_reward.grid(True, alpha=0.25, linewidth=0.5)
    ax_reward.legend(fontsize=7, ncols=len(RUN_CONFIGS))

    # ── critic losses ─────────────────────────────────────────────────────────
    plot_col(ax_qf1, runs, "qf1_loss", "loss", "QF1 loss")
    plot_col(ax_qf2, runs, "qf2_loss", "loss", "QF2 loss")

    # ── Q value panel: three traces overlaid per run ──────────────────────────
    linestyles = {"mean_q1": "-", "mean_q2": "--", "mean_td_target": ":"}
    labels_q   = {"mean_q1": "Q1 (online)", "mean_q2": "Q2 (online)", "mean_td_target": "TD target"}
    for run in runs:
        df, cfg = run["df"], run["cfg"]
        for col, ls in linestyles.items():
            if col not in df.columns:
                continue
            sub = df[["sim_time_min", col]].dropna()
            if sub.empty:
                continue
            s = smooth(sub[col], SMOOTH)
            lbl = f"{labels_q[col]} ({cfg['label']})" if len(RUN_CONFIGS) > 1 else labels_q[col]
            ax_q_vals.plot(sub["sim_time_min"], s,
                           color=cfg["color"], linewidth=1.1,
                           linestyle=ls, label=lbl, alpha=0.85)
    add_transitions(ax_q_vals, runs)
    ax_q_vals.set_title("Q values: online vs TD target", fontsize=9, pad=3)
    ax_q_vals.set_ylabel("Q value", fontsize=8)
    ax_q_vals.grid(True, alpha=0.25, linewidth=0.5)
    ax_q_vals.legend(fontsize=6, ncols=1)

    plot_col(ax_q_actor, runs, "mean_q_actor", "Q value",
             "Mean Q seen by actor (min of target critics)")

    # ── actor loss / alpha ────────────────────────────────────────────────────
    plot_col(ax_actor, runs, "actor_loss", "loss",  "Actor loss")
    plot_col(ax_alpha, runs, "alpha",      "alpha", "Entropy coefficient α")

    # ── entropy / log_std ─────────────────────────────────────────────────────
    plot_col(ax_entropy, runs, "actor_entropy", "entropy",
             "Actor entropy  −E[log π]  (should stay > 0)")

    # log_std mean + std band per run
    for run in runs:
        df, cfg = run["df"], run["cfg"]
        if "log_std_mean" not in df.columns:
            continue
        sub = df[["sim_time_min", "log_std_mean", "log_std_std"]].dropna()
        if sub.empty:
            continue
        mu  = smooth(sub["log_std_mean"], SMOOTH)
        sig = smooth(sub["log_std_std"],  SMOOTH)
        ax_logstd.plot(sub["sim_time_min"], mu,
                       color=cfg["color"], linewidth=1.2,
                       label=f"mean ({cfg['label']})")
        ax_logstd.fill_between(sub["sim_time_min"],
                               mu - sig, mu + sig,
                               color=cfg["color"], alpha=0.12)
    add_transitions(ax_logstd, runs)
    ax_logstd.set_title("log σ (mean ± std across action dims)", fontsize=9, pad=3)
    ax_logstd.set_ylabel("log std", fontsize=8)
    ax_logstd.grid(True, alpha=0.25, linewidth=0.5)
    ax_logstd.legend(fontsize=7)

    # ── grad norms ────────────────────────────────────────────────────────────
    plot_col(ax_gnorm_c, runs, "critic_grad_norm", "L2 norm", "Critic gradient norm")
    plot_col(ax_gnorm_a, runs, "actor_grad_norm",  "L2 norm", "Actor gradient norm")

    # ── x-axis labels on bottom row only ─────────────────────────────────────
    for ax in [ax_gnorm_c, ax_gnorm_a]:
        ax.set_xlabel("Sim time [minutes]", fontsize=8)

    # shared legend for transition line
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[Line2D([0], [0], color="red", linestyle="--", linewidth=0.9,
                        label="run boundary")],
        loc="lower center", ncol=1, fontsize=8, framealpha=0.6,
        bbox_to_anchor=(0.5, -0.005)
    )

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
