"""Plot transfer-vs-scratch learning curves for SAC and PPO on heavy ant.

Four curves are drawn:
  1. SAC transfer  — camera ant (0→100k, blue)  then heavy ant (100k→250k, green)
  2. SAC scratch   — heavy ant from scratch (0→250k, teal)
  3. PPO transfer  — camera ant (0→100k, red)   then heavy ant (100k→250k, gold)
  4. PPO scratch   — heavy ant from scratch (0→250k, orange)

Usage (from agents/):
    python3 plot_transfer.py [--runs_dir runs_transfer] [--output transfer.pdf]
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams.update({'font.size': 13})

COLORS = {
    'sac_env1':     '#4477AA',  # steel blue  — SAC camera ant phase
    'sac_transfer': '#228833',  # forest green — SAC heavy ant (transfer)
    'sac_scratch':  '#66CCEE',  # cyan         — SAC heavy ant (scratch)
    'ppo_env1':     '#EE6677',  # coral red    — PPO camera ant phase
    'ppo_transfer': '#CCBB44',  # gold         — PPO heavy ant (transfer)
    'ppo_scratch':  '#AA3377',  # purple       — PPO heavy ant (scratch)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs_transfer",
                   help="Sub-directory name inside sac/ and ppo/ that holds the runs.")
    p.add_argument("--sac_env1_dir",     type=str, default=None)
    p.add_argument("--sac_transfer_dir", type=str, default=None)
    p.add_argument("--sac_scratch_dir",  type=str, default=None)
    p.add_argument("--ppo_env1_dir",     type=str, default=None)
    p.add_argument("--ppo_transfer_dir", type=str, default=None)
    p.add_argument("--ppo_scratch_dir",  type=str, default=None)
    p.add_argument("--output", type=str, default="transfer_learning_curves.pdf")
    p.add_argument("--dt", type=float, default=0.12,
                   help="Simulation timestep in seconds (default: 0.12).")
    p.add_argument("--sac_smooth", type=int, default=500,
                   help="Rolling-mean window for SAC (rows ≈ 1 per step).")
    p.add_argument("--ppo_smooth", type=int, default=5,
                   help="Rolling-mean window for PPO (rows ≈ 1 per iteration).")
    return p.parse_args()


def find_latest_run(base_dir, prefix):
    """Return the most-recently-created run directory whose name starts with prefix."""
    pattern = os.path.join(base_dir, f"{prefix}*_seed_1")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(
            f"No run directories found matching: {pattern}\n"
            f"Make sure training has completed (./run_transfer.sh)."
        )
    return dirs[-1]


def load_perf(run_dir):
    path = os.path.join(run_dir, "performance_variables.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"performance_variables.csv not found in: {run_dir}")
    return pd.read_csv(path)


def smooth(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def plot_segment(ax, df, col, transfer_step, side, color, label, window):
    """Plot one phase of a continuous curve, split at transfer_step."""
    if side == 'left':
        seg = df[df['step'] <= transfer_step].copy()
    else:
        seg = df[df['step'] > transfer_step].copy()

    if seg.empty:
        print(f"[!] Warning: no data for segment '{label}' (side={side})")
        return

    seg = seg.sort_values('step')
    y = smooth(seg[col], window)
    ax.plot(seg['step'].values, y.values, color=color, label=label, linewidth=1.8)


def plot_scratch(ax, df, col, color, label, window):
    """Plot a scratch run (single continuous curve, x starts at 0)."""
    df = df.sort_values('step')
    y = smooth(df[col], window)
    ax.plot(df['step'].values, y.values,
            color=color, label=label, linewidth=1.8, linestyle='--')


def main():
    args = parse_args()

    sac_base = os.path.join("sac", args.runs_dir)
    ppo_base = os.path.join("ppo", args.runs_dir)

    # ── Discover run directories ──────────────────────────────────────────────
    sac_env1_dir     = args.sac_env1_dir     or find_latest_run(sac_base, "sac_camera")
    sac_transfer_dir = args.sac_transfer_dir or find_latest_run(sac_base, "sac_heavy_transfer")
    sac_scratch_dir  = args.sac_scratch_dir  or find_latest_run(sac_base, "sac_heavy_scratch")
    ppo_env1_dir     = args.ppo_env1_dir     or find_latest_run(ppo_base, "ppo_camera")
    ppo_transfer_dir = args.ppo_transfer_dir or find_latest_run(ppo_base, "ppo_heavy_transfer")
    ppo_scratch_dir  = args.ppo_scratch_dir  or find_latest_run(ppo_base, "ppo_heavy_scratch")

    print(f"SAC env1     : {sac_env1_dir}")
    print(f"SAC transfer : {sac_transfer_dir}")
    print(f"SAC scratch  : {sac_scratch_dir}")
    print(f"PPO env1     : {ppo_env1_dir}")
    print(f"PPO transfer : {ppo_transfer_dir}")
    print(f"PPO scratch  : {ppo_scratch_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    df_sac1  = load_perf(sac_env1_dir)
    df_sac2  = load_perf(sac_transfer_dir)
    df_sacS  = load_perf(sac_scratch_dir)
    df_ppo1  = load_perf(ppo_env1_dir)
    df_ppo2  = load_perf(ppo_transfer_dir)
    df_ppoS  = load_perf(ppo_scratch_dir)

    col = 'average_reward_per_second'

    # Convert step → simulation time in minutes for all DataFrames.
    steps_to_min = args.dt / 60.0
    for df in [df_sac1, df_sac2, df_sacS, df_ppo1, df_ppo2, df_ppoS]:
        df['step'] = df['step'] * steps_to_min

    # Transfer time = last sim-time written by env1 (in minutes).
    sac_transfer = df_sac1['step'].max()
    ppo_transfer = df_ppo1['step'].max()

    # Concatenate env1 + env2 for the continuous transfer curve.
    df_sac_all = pd.concat([df_sac1, df_sac2], ignore_index=True).sort_values('step')
    df_ppo_all = pd.concat([df_ppo1, df_ppo2], ignore_index=True).sort_values('step')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    ax_sac, ax_ppo = axes

    # ── SAC panel ─────────────────────────────────────────────────────────────
    plot_segment(ax_sac, df_sac_all, col,
                 transfer_step=sac_transfer, side='left',
                 color=COLORS['sac_env1'],
                 label='SAC — camera ant (pre-train)',
                 window=args.sac_smooth)

    plot_segment(ax_sac, df_sac_all, col,
                 transfer_step=sac_transfer, side='right',
                 color=COLORS['sac_transfer'],
                 label='SAC — heavy ant (transfer)',
                 window=args.sac_smooth)

    plot_scratch(ax_sac, df_sacS, col,
                 color=COLORS['sac_scratch'],
                 label='SAC — heavy ant (scratch)',
                 window=args.sac_smooth)

    ax_sac.axvline(x=sac_transfer, color='gray',
                   linestyle=':', linewidth=1.0, alpha=0.7,
                   label=f'Transfer point ({sac_transfer:.1f} min)')
    ax_sac.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.4)
    ax_sac.set_xlabel('Simulation Time (minutes)')
    ax_sac.set_ylabel('Avg. Reward / s')
    ax_sac.set_title('SAC: Transfer vs Scratch on Heavy Ant')
    ax_sac.legend(loc='upper left', fontsize=10)

    # ── PPO panel ─────────────────────────────────────────────────────────────
    plot_segment(ax_ppo, df_ppo_all, col,
                 transfer_step=ppo_transfer, side='left',
                 color=COLORS['ppo_env1'],
                 label='PPO — camera ant (pre-train)',
                 window=args.ppo_smooth)

    plot_segment(ax_ppo, df_ppo_all, col,
                 transfer_step=ppo_transfer, side='right',
                 color=COLORS['ppo_transfer'],
                 label='PPO — heavy ant (transfer)',
                 window=args.ppo_smooth)

    plot_scratch(ax_ppo, df_ppoS, col,
                 color=COLORS['ppo_scratch'],
                 label='PPO — heavy ant (scratch)',
                 window=args.ppo_smooth)

    ax_ppo.axvline(x=ppo_transfer, color='gray',
                   linestyle=':', linewidth=1.0, alpha=0.7,
                   label=f'Transfer point ({ppo_transfer:.1f} min)')
    ax_ppo.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.4)
    ax_ppo.set_xlabel('Simulation Time (minutes)')
    ax_ppo.set_title('PPO: Transfer vs Scratch on Heavy Ant')
    ax_ppo.legend(loc='upper left', fontsize=10)

    fig.suptitle('Transfer Learning (Camera Ant → Heavy Ant) vs From Scratch',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"[√] Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()