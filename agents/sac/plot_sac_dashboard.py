import os
import ast
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, "/home/serena-liu/open-ant")  # adjust to your repo root
from agents.reward import RewardTracker

# =====================================================
# USER SETTINGS
# =====================================================

# List run directories in temporal order: sim1 first, then sim2 (continual).
# The script auto-stitches sim2's global_step onto sim1's end so the x-axis
# is one continuous timeline regardless of whether sim2 starts at step 1 or
# step 36002.
RUN_DIRS = [
    "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/warmstart_morestats/trial_1_20260605-151155_seed_0",
    "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/warmstart_morestats/trial_1_warmstart_continual_learning_20260605-151826_seed_0",
    # add or remove paths as needed
]

SMOOTH = 10
OUTPUT_DIR = "/home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/warmstart_morestats"

# Percentile used to clip the y-axis so early spikes don't crush the trend.
# Data OUTSIDE this range still gets drawn (the line goes off the axes) but
# the y-limits are set to show the bulk of the data at full resolution.
# A text annotation is added whenever a plotted point exceeds the y-limit.
SPIKE_CLIP_PCT = 97        # clip y-axis at this upper percentile
SPIKE_ANNOT_SUBSAMPLE = 5  # only annotate every Nth out-of-range point (keeps labels sparse)

# =====================================================

TIME_WINDOW = 120.0

CRITIC_LOSS_COLS = ["qf1_loss", "qf2_loss"]
ACTOR_LOSS_COL   = "actor_loss"
Q_VALUE_COLS = {
    "mean_q1":        ("-",  "Q1 (online)"),
    "mean_q2":        ("--", "Q2 (online)"),
    "mean_td_target": (":",  "TD target"),
}

METRIC_COLS = [
    "qf1_loss", "qf2_loss", "actor_loss", "alpha", "alpha_loss",
    "mean_q1", "mean_q2", "mean_td_target", "mean_q_actor",
    "actor_entropy", "log_std_mean", "log_std_std",
    "actor_grad_norm", "critic_grad_norm",
]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def load_run(run_dir, smooth_w):
    """Load one run directory.  Returns (df, dt, learning_starts).

    df columns include:
      global_step        – original logged step
      sim_time_min       – time in minutes from that step (NOT yet stitched)
      _reward_smooth     – smoothed avg reward/s * 100
      all METRIC_COLS    – smoothed
    """
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path) as f:
        run_args = json.load(f)
    dt = run_args["dt"]
    learning_starts = run_args.get("learning_starts", 0)

    csv_path = os.path.join(run_dir, "info_sac_logs.csv")
    df = pd.read_csv(csv_path)

    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    df = df[df["global_step"].notna()].copy()
    df["global_step"] = df["global_step"].astype(float)

    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── reward ───────────────────────────────────────────────────────────────
    reward_src = (
        "original_rewards" if "original_rewards" in df.columns else
        "rewards"           if "rewards"           in df.columns else None
    )
    if reward_src is not None:
        raw = parse_reward_col(df[reward_src])
        tracker = RewardTracker(env_dt=dt, env_id="plot",
                                time_window=TIME_WINDOW, log_folder=".")
        avg_rps = []
        for r in raw:
            tracker.update(r)
            avg_rps.append(tracker.average_reward_per_second)
        df["avg_reward_per_second"] = avg_rps
        warmup_steps = int(TIME_WINDOW / dt)
        df = df.iloc[warmup_steps:].copy()
        df["_reward_smooth"] = (
            df["avg_reward_per_second"].rolling(smooth_w, min_periods=1).mean() * 100
        )
        df = df.iloc[smooth_w:].copy()

    # ── smooth metrics ────────────────────────────────────────────────────────
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = df[col].rolling(smooth_w, min_periods=1).mean()
    df = df.iloc[smooth_w:].copy()

    # sim_time_min based on the raw step counter (will be corrected by stitch)
    df["sim_time_min"] = df["global_step"] * dt / 60.0

    return df, dt, learning_starts


def stitch_runs(loaded_runs):
    """
    loaded_runs : list of (df, dt, learning_starts, run_dir)

    Each subsequent run's time axis is shifted so that its first data point
    comes right after the last data point of the previous run.  This works
    whether sim2's global_step starts at 1 or at 36002 — we don't care about
    the raw step value; we care only about how much NEW simulated time each
    run contributes.

    Returns list of (df, dt, learning_starts, run_dir, t_offset_min)
    where df["sim_time_min"] has been replaced with the stitched time.
    """
    result = []
    t_end = 0.0   # end time of the previous segment in minutes

    for df, dt, learning_starts, run_dir in loaded_runs:
        # Duration this run actually spans (in minutes), based on its own step count
        t_span = (df["global_step"].max() - df["global_step"].min()) * dt / 60.0
        t_start = t_end

        # Remap: normalize steps within this run to [0, t_span], then shift
        step_min = df["global_step"].min()
        step_max = df["global_step"].max()
        denom = max(step_max - step_min, 1e-9)
        df = df.copy()
        df["sim_time_min"] = t_start + (df["global_step"] - step_min) / denom * t_span

        t_end = t_start + t_span
        result.append((df, dt, learning_starts, run_dir, t_start))

    return result


def robust_ylim(values, lo_pct=1, hi_pct=97, pad=0.1):
    """Y-limits based on percentiles so spikes don't crush the trend."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def annotate_spikes(ax, x_vals, y_vals, ylim, color, subsample=SPIKE_ANNOT_SUBSAMPLE):
    """
    For points outside [ylo, yhi], place a small text label at the axis edge
    showing the actual value.  The line itself is clipped at the boundary
    (clip_on=True on the plot call), so nothing bleeds outside the axes box.
    Only one representative label is shown per contiguous out-of-range run to
    avoid clutter (pick the extremum of each run).
    """
    ylo, yhi = ylim
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    for mask, edge, va in [
        (y_arr > yhi, yhi, "bottom"),   # above upper limit → label at top edge
        (y_arr < ylo, ylo, "top"),       # below lower limit → label at bottom edge
    ]:
        if not mask.any():
            continue

        # Find contiguous runs of out-of-range points
        indices = np.where(mask)[0]
        # Split into contiguous groups
        gaps = np.where(np.diff(indices) > 1)[0] + 1
        groups = np.split(indices, gaps)

        for grp in groups:
            if len(grp) == 0:
                continue
            # Pick the most extreme point in this group as the representative
            if va == "bottom":   # above upper edge → pick max
                rep = grp[np.argmax(y_arr[grp])]
            else:                # below lower edge → pick min
                rep = grp[np.argmin(y_arr[grp])]

            ax.annotate(
                f"{y_arr[rep]:.2g}",
                xy=(x_arr[rep], edge),
                xytext=(0, 3 if va == "bottom" else -3),
                textcoords="offset points",
                fontsize=5,
                color=color,
                alpha=0.75,
                ha="center",
                va=va,
                clip_on=False,      # annotation text can sit just outside the frame
            )


def plot_col(ax, sub_t, sub_y, color, alpha, lw, linestyle="-", label=None,
             ylim=None, subsample=SPIKE_ANNOT_SUBSAMPLE):
    """Plot a column.  Line is hard-clipped at axes boundaries; spikes get
    a text annotation at the edge showing their actual value."""
    ax.plot(sub_t, sub_y, color=color, alpha=alpha, lw=lw,
            linestyle=linestyle, label=label, clip_on=True)   # clip_on=True: no bleed
    if ylim is not None:
        annotate_spikes(ax, sub_t, sub_y, ylim, color, subsample)


def add_vline(ax, x, color, label=None):
    ax.axvline(x, color=color, linestyle="--", linewidth=1.0,
               alpha=0.6, label=label if label else "_")


def add_phase_divider(ax, x, label="sim2 start"):
    """Vertical line marking the sim1→sim2 boundary."""
    ax.axvline(x, color="gray", linestyle=":", linewidth=1.2, alpha=0.8,
               label=label)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    run_dirs = [os.path.abspath(r) for r in RUN_DIRS]
    out_dir  = os.path.abspath(OUTPUT_DIR) if OUTPUT_DIR else os.path.dirname(run_dirs[0])
    os.makedirs(out_dir, exist_ok=True)

    # ── load all runs ─────────────────────────────────────────────────────────
    loaded = []
    for run_dir in run_dirs:
        try:
            df, dt, learning_starts = load_run(run_dir, SMOOTH)
            loaded.append((df, dt, learning_starts, run_dir))
            print(f"Loaded {os.path.basename(run_dir)}: "
                  f"steps {int(df['global_step'].min())}–{int(df['global_step'].max())}")
        except Exception as e:
            print(f"Skipping {run_dir}: {e}")

    if not loaded:
        print("No runs loaded. Exiting.")
        return

    # ── stitch time axes ──────────────────────────────────────────────────────
    stitched = stitch_runs(loaded)
    # t_offsets[i] = wall-clock minute where run i begins on the plot
    t_offsets = [s[4] for s in stitched]
    sim2_boundary = t_offsets[1] if len(stitched) > 1 else None

    # ── collect all values for each column (for ylim computation) ────────────
    col_vals = {col: [] for col in METRIC_COLS + ["_reward_smooth"]}
    for df, dt, learning_starts, run_dir, t_off in stitched:
        for col in col_vals:
            if col in df.columns:
                col_vals[col].extend(df[col].dropna().tolist())

    # pre-compute ylims for every panel so we can annotate spikes consistently
    ylims = {}
    for col in METRIC_COLS + ["_reward_smooth"]:
        ylims[col] = robust_ylim(col_vals[col], lo_pct=1, hi_pct=SPIKE_CLIP_PCT)

    # Alpha special case: the early warmup period holds alpha constant at its
    # initial value (e.g. 0.005) for thousands of steps before learning starts,
    # which dominates the percentile range and crushes the interesting late-run
    # fluctuations in the 0-0.1 range.
    # Fix: compute ylim only over the latter 60% of alpha values so the y-axis
    # zooms into where alpha is actually changing.
    if col_vals["alpha"]:
        alpha_arr = np.asarray(col_vals["alpha"], dtype=float)
        alpha_arr = alpha_arr[np.isfinite(alpha_arr)]
        if alpha_arr.size > 10:
            tail = alpha_arr[int(len(alpha_arr) * 0.40):]   # last 60%
            ylims["alpha"] = robust_ylim(
                tail.tolist(), lo_pct=0, hi_pct=SPIKE_CLIP_PCT, pad=0.15)

    # combined ylim for critic (covers both qf1 and qf2)
    combined_critic_vals = col_vals["qf1_loss"] + col_vals["qf2_loss"]
    ylims["critic_combined"] = robust_ylim(combined_critic_vals, lo_pct=0, hi_pct=SPIKE_CLIP_PCT)

        # ── fixed y-axis ranges requested ─────────────────────────────────────────
    ylims["critic_combined"] = (0, 0.10)
    ylims["alpha_loss"] = (0, 2)
    ylims[ACTOR_LOSS_COL] = (-30, 1)
    ylims["alpha"] = (0, 0.02)
    ylims["actor_entropy"] = (-10, 10)
    ylims["mean_q_actor"] = (0, 190)
    ylims["log_std_mean"] = (-2, 90)
    ylims["actor_grad_norm"] = (0, 2)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 20))
    gs  = fig.add_gridspec(6, 2, hspace=0.50, wspace=0.30)

    ax_reward  = fig.add_subplot(gs[0, :])
    ax_policy  = fig.add_subplot(gs[1, 0])
    ax_critic  = fig.add_subplot(gs[1, 1])
    ax_meanq   = fig.add_subplot(gs[2, 0])
    ax_qactor  = fig.add_subplot(gs[2, 1])
    ax_alpha   = fig.add_subplot(gs[3, 0])
    ax_entropy = fig.add_subplot(gs[3, 1])
    ax_logstd  = fig.add_subplot(gs[4, 0])
    ax_aloss   = fig.add_subplot(gs[4, 1])
    ax_gnorm_c = fig.add_subplot(gs[5, 0])
    ax_gnorm_a = fig.add_subplot(gs[5, 1])

    palette = cm.tab10(np.linspace(0, 0.9, max(len(stitched), 1)))

    # ── plot each run ─────────────────────────────────────────────────────────
    for i, (df, dt, learning_starts, run_dir, t_off) in enumerate(stitched):
        color  = palette[i]
        alpha  = 0.85 if i == 0 else 0.60
        lw     = 1.5  if i == 0 else 1.1
        label  = os.path.basename(run_dir)
        t      = df["sim_time_min"]

        # learning_starts vline is relative to this segment's own t_off
        ls_min = t_off + learning_starts * dt / 60.0

        def _sub(col):
            """Return (t_vals, y_vals) for a column, dropping NaN pairs."""
            sub = df[["sim_time_min", col]].dropna()
            return sub["sim_time_min"].values, sub[col].values

        # ── reward ────────────────────────────────────────────────────────────
        if "_reward_smooth" in df.columns:
            tx, ty = _sub("_reward_smooth")
            plot_col(ax_reward, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["_reward_smooth"])
            add_vline(ax_reward, ls_min, color,
                      label="learning starts" if i == 0 else None)

        # ── actor loss ────────────────────────────────────────────────────────
        if ACTOR_LOSS_COL in df.columns:
            tx, ty = _sub(ACTOR_LOSS_COL)
            plot_col(ax_policy, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims[ACTOR_LOSS_COL])
        add_vline(ax_policy, ls_min, color)

        # ── critic loss (qf1 & qf2) ───────────────────────────────────────────
        for col, ls in zip(CRITIC_LOSS_COLS, ["-", "--"]):
            if col in df.columns:
                tx, ty = _sub(col)
                lbl = f"{col} ({label})" if len(stitched) > 1 else col
                plot_col(ax_critic, tx, ty, color, alpha, lw, linestyle=ls,
                         label=lbl, ylim=ylims["critic_combined"])
        add_vline(ax_critic, ls_min, color)

        # ── mean Q values ─────────────────────────────────────────────────────
        for col, (ls, base_lbl) in Q_VALUE_COLS.items():
            if col in df.columns:
                tx, ty = _sub(col)
                lbl = f"{base_lbl} ({label})" if len(stitched) > 1 else base_lbl
                plot_col(ax_meanq, tx, ty, color, alpha, lw, linestyle=ls,
                         label=lbl, ylim=ylims[col])
        add_vline(ax_meanq, ls_min, color)

        # ── mean Q seen by actor ──────────────────────────────────────────────
        if "mean_q_actor" in df.columns:
            tx, ty = _sub("mean_q_actor")
            plot_col(ax_qactor, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["mean_q_actor"])
        add_vline(ax_qactor, ls_min, color)

        # ── alpha ─────────────────────────────────────────────────────────────
        if "alpha" in df.columns:
            tx, ty = _sub("alpha")
            plot_col(ax_alpha, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["alpha"])
        add_vline(ax_alpha, ls_min, color)

        # ── actor entropy ─────────────────────────────────────────────────────
        if "actor_entropy" in df.columns:
            tx, ty = _sub("actor_entropy")
            plot_col(ax_entropy, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["actor_entropy"])
        add_vline(ax_entropy, ls_min, color)

        # ── log_std mean ± std band ───────────────────────────────────────────
        if "log_std_mean" in df.columns:
            sub = df[["sim_time_min", "log_std_mean", "log_std_std"]].dropna()
            tx  = sub["sim_time_min"].values
            ym  = sub["log_std_mean"].values
            ys  = sub["log_std_std"].values
            plot_col(ax_logstd, tx, ym, color, alpha, lw,
                     label=f"mean ({label})", ylim=ylims["log_std_mean"])
            ax_logstd.fill_between(tx, ym - ys, ym + ys,
                                   color=color, alpha=0.12, clip_on=True)
        add_vline(ax_logstd, ls_min, color)

        # ── alpha loss ────────────────────────────────────────────────────────
        if "alpha_loss" in df.columns:
            tx, ty = _sub("alpha_loss")
            plot_col(ax_aloss, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["alpha_loss"])
        add_vline(ax_aloss, ls_min, color)

        # ── grad norms ────────────────────────────────────────────────────────
        if "critic_grad_norm" in df.columns:
            tx, ty = _sub("critic_grad_norm")
            plot_col(ax_gnorm_c, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["critic_grad_norm"])
        add_vline(ax_gnorm_c, ls_min, color)

        if "actor_grad_norm" in df.columns:
            tx, ty = _sub("actor_grad_norm")
            plot_col(ax_gnorm_a, tx, ty, color, alpha, lw, label=label,
                     ylim=ylims["actor_grad_norm"])
        add_vline(ax_gnorm_a, ls_min, color)

    # ── sim1 → sim2 boundary line on every panel ──────────────────────────────
    all_axes = [ax_reward, ax_policy, ax_critic, ax_meanq, ax_qactor,
                ax_alpha, ax_entropy, ax_logstd, ax_aloss, ax_gnorm_c, ax_gnorm_a]
    if sim2_boundary is not None:
        for ax in all_axes:
            add_phase_divider(ax, sim2_boundary, label="sim2 start" if ax is ax_reward else None)

    # ── apply ylims (after plotting so clip_on=False still draws spikes) ──────
    if ylims["_reward_smooth"]:
        ax_reward.set_ylim(ylims["_reward_smooth"])
    if ylims[ACTOR_LOSS_COL]:
        ax_policy.set_ylim(ylims[ACTOR_LOSS_COL])
    if ylims["critic_combined"]:
        ax_critic.set_ylim(ylims["critic_combined"])
    for col, (_, __) in Q_VALUE_COLS.items():
        pass   # meanq has multiple overlaid cols — let matplotlib auto-scale or set manually
    if ylims["mean_q_actor"]:
        ax_qactor.set_ylim(ylims["mean_q_actor"])
    if ylims.get("alpha"):
        ax_alpha.set_ylim(ylims["alpha"])
    if ylims["actor_entropy"]:
        ax_entropy.set_ylim(ylims["actor_entropy"])
    if ylims["log_std_mean"]:
        ax_logstd.set_ylim(ylims["log_std_mean"])
    if ylims["alpha_loss"]:
        ax_aloss.set_ylim(ylims["alpha_loss"])
    if ylims["critic_grad_norm"]:
        ax_gnorm_c.set_ylim(ylims["critic_grad_norm"])
    if ylims["actor_grad_norm"]:
        ax_gnorm_a.set_ylim(ylims["actor_grad_norm"])

    # ── titles, labels, legends ───────────────────────────────────────────────
    primary_label = os.path.basename(run_dirs[0])
    suffix = f" (+{len(run_dirs)-1} more)" if len(run_dirs) > 1 else ""

    ax_reward.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax_reward.set(xlabel="Sim time [minutes]", ylabel="Avg reward/s [cm/s]",
                  xlim=(0, None),
                  title=f"Training reward — {primary_label}{suffix}")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(fontsize="x-small", ncols=2)

    panel_titles = [
        (ax_policy,  "Actor (policy) loss"),
        (ax_critic,  "Critic loss (qf1 / qf2)"),
        (ax_meanq,   "Mean Q value (Q1 / Q2 / TD target)"),
        (ax_qactor,  "Mean Q seen by actor (min target critics)"),
        (ax_alpha,   "Entropy coefficient alpha"),
        (ax_entropy, "Actor entropy  −E[log π]"),
        (ax_logstd,  "log σ  (mean ± std across action dims)"),
        (ax_aloss,   "Alpha loss"),
        (ax_gnorm_c, "Critic gradient norm"),
        (ax_gnorm_a, "Actor gradient norm"),
    ]
    for ax, title in panel_titles:
        ax.set(xlabel="Sim time [minutes]", title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small")

    out_path = os.path.join(out_dir, f"{primary_label}_dashboard.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")



if __name__ == "__main__":
    main()