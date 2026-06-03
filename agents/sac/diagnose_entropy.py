"""
diagnose_entropy.py
====================
Drop this file next to sac_cleanrl.py.

It does two things:

1.  ONLINE DIAGNOSTIC  — monkeypatches SAC.agent_step() so that every
    `log_interval` steps it records the mean and per-dimension std that
    the actor outputs for the current batch of observations.  The extra
    columns are appended to info_sac_logs.csv automatically.

2.  POST-HOC ANALYSIS — if you already have a trained weights.pth and a
    replay buffer, call `post_hoc_entropy_analysis()` to load the actor
    and evaluate it over a fixed batch of states from the buffer.

Usage (online):
---------------
    # In your main script, AFTER creating `agent` and BEFORE the training loop:
    from diagnose_entropy import patch_agent
    patch_agent(agent, log_interval=50)

    # The training loop is unchanged.  info_sac_logs.csv will gain columns:
    #   mean_std_mean, mean_std_min, mean_std_max, mean_log_std_mean,
    #   policy_entropy_nats, alpha

Usage (post-hoc, from a saved run):
-------------------------------------
    python diagnose_entropy.py --weights_path runs_sim/trial_1_<date>_seed_1 \
                               --num_samples 1024

    This produces entropy_diagnosis.png in the run directory.
    If a replay_buffer folder exists in weights_path, the script loads
    num_samples states from it and evaluates the final actor over those
    states to compute a buffer-wide entropy snapshot.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ONLINE PATCH
# ─────────────────────────────────────────────────────────────────────────────

def patch_agent(agent, log_interval: int = 50):
    """
    Monkeypatch agent.agent_step() to append entropy diagnostics to its
    returned info_sac dict.  Call this once after creating your SAC agent.

    Parameters
    ----------
    agent        : SAC instance from sac_cleanrl.py
    log_interval : compute diagnostics every this many global steps
    """
    _original_step = agent.agent_step

    def _patched_step(next_obs, actions, rewards, terminations, truncations, infos):
        info_sac = _original_step(next_obs, actions, rewards, terminations, truncations, infos)

        # Only compute when learning has started and on the right interval.
        if (agent.global_step > agent.learning_starts and
                agent.global_step % log_interval == 0 and
                info_sac is not None):

            with torch.no_grad():
                obs_t = torch.as_tensor(agent.obs, dtype=torch.float32, device=agent.device)
                mean_raw, log_std = agent.actor(obs_t)          # (B, A)
                std = log_std.exp()                              # (B, A)

                # Per-dimension stats averaged over the batch.
                mean_std_per_dim = std.mean(dim=0)               # (A,)
                mean_std_mean    = mean_std_per_dim.mean().item()
                mean_std_min     = mean_std_per_dim.min().item()
                mean_std_max     = mean_std_per_dim.max().item()
                mean_log_std     = log_std.mean().item()

                # Approximate differential entropy of the squashed Gaussian.
                # H(tanh(N(mu,sigma))) ≈ H(N) - correction  (exact only pre-squash)
                # Pre-squash entropy per dim: 0.5 * log(2*pi*e*sigma^2)
                entropy_per_dim  = 0.5 * (1 + torch.log(2 * torch.tensor(torch.pi) * std**2))
                policy_entropy   = entropy_per_dim.mean().item()   # nats

            info_sac.update({
                "mean_std_mean":      mean_std_mean,
                "mean_std_min":       mean_std_min,
                "mean_std_max":       mean_std_max,
                "mean_log_std_mean":  mean_log_std,
                "policy_entropy_nats": policy_entropy,
            })

        return info_sac

    agent.agent_step = _patched_step
    print("[diagnose_entropy] Online patch applied. "
          f"Entropy stats logged every {log_interval} steps.")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  POST-HOC ANALYSIS (standalone)
# ─────────────────────────────────────────────────────────────────────────────

def post_hoc_entropy_analysis(weights_path: str,
                               num_samples: int = 1024,
                               device_str: str = "cpu"):
    """
    Load a saved weights.pth and the info_sac_logs.csv from `weights_path`
    and produce a 4-panel diagnostic figure saved as entropy_diagnosis.png.

    If a replay_buffer/ folder exists alongside weights.pth, this function
    also loads num_samples states from it, runs the final saved actor over
    them, and prints a buffer-wide entropy/std snapshot to stdout.
    """
    import pandas as pd
    from torchrl.data import ReplayBuffer, LazyTensorStorage, RandomSampler

    device = torch.device(device_str)

    weights_file = os.path.join(weights_path, "weights.pth")
    csv_file     = os.path.join(weights_path, "info_sac_logs.csv")
    args_file    = os.path.join(weights_path, "args.json")
    rb_path      = os.path.join(weights_path, "replay_buffer")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"No info_sac_logs.csv found in {weights_path}")

    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["actor_loss"])   # rows before learning_starts have NaN

    # ── check if entropy columns are already present (online patch was active)
    has_entropy_cols = "mean_std_mean" in df.columns

    # ── REPLAY BUFFER: load and evaluate actor over a real state sample ────────
    rb_entropy_stats = None
    if os.path.exists(rb_path) and os.path.exists(weights_file) and os.path.exists(args_file):
        print(f"[diagnose_entropy] Replay buffer found at {rb_path}. Loading...")
        try:
            # 1. Read saved args to get obs_dim, action_dim, use_layer_norm.
            with open(args_file, "r") as f:
                saved_args = json.load(f)
            use_layer_norm = saved_args.get("use_layer_norm", True)

            # 2. Load the replay buffer and sample a batch of observations.
            #    Buffer size in the saved run — use a large cap so loads() works.
            rb = ReplayBuffer(
                storage=LazyTensorStorage(int(1e6), device=device),
                sampler=RandomSampler(),
                batch_size=num_samples,
            )
            rb.loads(rb_path)
            actual_n = min(num_samples, len(rb))
            batch, _ = rb.sample(actual_n, return_info=True)
            obs_batch = batch["observations"].to(device)   # (N, obs_dim)
            obs_dim    = obs_batch.shape[1]

            # 3. Reconstruct the Actor using obs_dim / action_dim inferred from
            #    the stored weights (no env object needed).
            state = torch.load(weights_file, map_location=device)
            actor_sd = state["actor"]
            # Infer action_dim from the mean head weight shape.
            action_dim = actor_sd["fc_mean.weight"].shape[0]

            # Build a minimal namespace so Actor.__init__ can read spaces.
            import types, gymnasium as gym
            dummy_env = types.SimpleNamespace(
                single_observation_space=types.SimpleNamespace(
                    shape=(obs_dim,), dtype=np.float32),
                single_action_space=types.SimpleNamespace(
                    shape=(action_dim,),
                    high=np.ones(action_dim, dtype=np.float32),
                    low=-np.ones(action_dim, dtype=np.float32),
                ),
            )

            # Import Actor from sac_cleanrl — assumes diagnose_entropy.py is
            # in the same directory.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from sac_cleanrl import Actor
            actor = Actor(dummy_env, use_layer_norm=use_layer_norm).to(device)
            actor.load_state_dict(actor_sd)
            actor.eval()

            # 4. Evaluate actor over the sampled observations.
            with torch.no_grad():
                _, log_std = actor(obs_batch)           # (N, A)
                std = log_std.exp()                     # (N, A)
                mean_std_per_dim = std.mean(dim=0)      # (A,)
                entropy_per_dim  = 0.5 * (1 + torch.log(
                    2 * torch.tensor(torch.pi, device=device) * std**2))

            rb_entropy_stats = {
                "n_states":        actual_n,
                "action_dim":      action_dim,
                "mean_std_mean":   mean_std_per_dim.mean().item(),
                "mean_std_min":    mean_std_per_dim.min().item(),
                "mean_std_max":    mean_std_per_dim.max().item(),
                "mean_log_std":    log_std.mean().item(),
                "entropy_mean":    entropy_per_dim.mean().item(),
                "std_per_dim":     mean_std_per_dim.cpu().numpy(),
            }

            print("\n[diagnose_entropy] ── Buffer-wide entropy snapshot ──────────")
            print(f"  States sampled      : {actual_n}")
            print(f"  Action dims         : {action_dim}")
            print(f"  Mean σ (all dims)   : {rb_entropy_stats['mean_std_mean']:.4f}  "
                  f"  (floor = {np.exp(-5):.4f})")
            print(f"  σ range [min, max]  : [{rb_entropy_stats['mean_std_min']:.4f}, "
                  f"{rb_entropy_stats['mean_std_max']:.4f}]")
            print(f"  Mean log σ          : {rb_entropy_stats['mean_log_std']:.4f}  "
                  f"  (LOG_STD_MIN = -5)")
            print(f"  Mean H(π) [nats]    : {rb_entropy_stats['entropy_mean']:.4f}")
            print(f"  σ per action dim    : "
                  + "  ".join(f"a{i}={v:.4f}"
                               for i, v in enumerate(rb_entropy_stats["std_per_dim"])))
            collapsed = rb_entropy_stats["mean_std_mean"] < 2 * np.exp(-5)
            print(f"  Collapse detected?  : {'⚠ YES — policy is near-deterministic' if collapsed else 'No'}")
            print("────────────────────────────────────────────────────────────────\n")

        except Exception as e:
            print(f"[diagnose_entropy] Warning: could not load replay buffer: {e}")
            rb_entropy_stats = None
    else:
        print("[diagnose_entropy] No replay_buffer folder found — skipping buffer-wide eval.")

    # ── FIGURE ────────────────────────────────────────────────────────────────
    n_rows = 3 if rb_entropy_stats is not None else 2
    fig = plt.figure(figsize=(14, 5 * n_rows), facecolor="#0d1117")
    fig.suptitle("SAC Policy Entropy Diagnostic\n" + os.path.basename(weights_path),
                 color="#e6edf3", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    _kw = dict(facecolor="#161b22")
    axes = [fig.add_subplot(gs[r, c], **_kw) for r in range(2) for c in range(2)]

    ACCENT = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]

    def _style(ax, title, xlabel, ylabel):
        ax.set_title(title, color="#e6edf3", fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color="#8b949e", fontsize=8)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.set_facecolor("#161b22")
        ax.grid(alpha=0.15, color="#30363d")

    steps = df["global_step"].values

    # ── Panel 0: alpha (entropy temperature) over time
    ax = axes[0]
    ax.plot(steps, df["alpha"], color=ACCENT[0], lw=1.2)
    ax.axhline(0, color="#30363d", lw=0.5, ls="--")
    _style(ax, "Entropy Temperature α", "Step", "α")

    # ── Panel 1: actor loss
    ax = axes[1]
    ax.plot(steps, df["actor_loss"], color=ACCENT[1], lw=0.8, alpha=0.7)
    # smoothed
    window = max(1, len(df) // 50)
    smoothed = df["actor_loss"].rolling(window, min_periods=1).mean()
    ax.plot(steps, smoothed, color=ACCENT[1], lw=1.8)
    _style(ax, "Actor Loss", "Step", "Loss")

    if has_entropy_cols:
        # ── Panel 2: mean std of the policy across action dims
        ax = axes[2]
        ax.plot(steps, df["mean_std_mean"], color=ACCENT[2], lw=1.2, label="mean")
        ax.fill_between(steps, df["mean_std_min"], df["mean_std_max"],
                        color=ACCENT[2], alpha=0.2, label="[min, max] across dims")
        # Reference lines: LOG_STD bounds
        ax.axhline(np.exp(-5), color="#8b949e", lw=0.8, ls="--",
                   label=f"LOG_STD_MIN floor (σ≈{np.exp(-5):.4f})")
        ax.axhline(np.exp(2),  color="#8b949e", lw=0.8, ls=":",
                   label=f"LOG_STD_MAX ceil (σ≈{np.exp(2):.1f})")
        ax.legend(fontsize=6, facecolor="#0d1117", labelcolor="#e6edf3",
                  edgecolor="#30363d")
        _style(ax, "Policy Std σ (action dims) — collapse → near 0.007",
               "Step", "σ")

        # ── Panel 3: differential entropy
        ax = axes[3]
        ax.plot(steps, df["policy_entropy_nats"], color=ACCENT[3], lw=1.2)
        ax.axhline(0, color="#f78166", lw=0.8, ls="--", label="H=0 (deterministic)")
        target_ent = -df["alpha"].iloc[-1]  # rough proxy
        ax.legend(fontsize=6, facecolor="#0d1117", labelcolor="#e6edf3",
                  edgecolor="#30363d")
        _style(ax, "Pre-squash Gaussian Entropy H(π) [nats]",
               "Step", "Entropy (nats)")

    else:
        # Fallback: just plot Q losses to still be useful
        ax = axes[2]
        ax.plot(steps, df["qf1_loss"], color=ACCENT[2], lw=0.8, alpha=0.7, label="Q1")
        ax.plot(steps, df["qf2_loss"], color=ACCENT[3], lw=0.8, alpha=0.7, label="Q2")
        ax.legend(fontsize=7, facecolor="#0d1117", labelcolor="#e6edf3")
        _style(ax, "Critic Losses", "Step", "MSE Loss")

        axes[3].text(0.5, 0.5,
            "Entropy columns not found.\nRe-run with patch_agent() to get\nmean_std / entropy plots.",
            ha="center", va="center", color="#8b949e", fontsize=9,
            transform=axes[3].transAxes)
        axes[3].set_facecolor("#161b22")
        for spine in axes[3].spines.values():
            spine.set_edgecolor("#30363d")

    # ── Panel 4 & 5 (row 3): buffer-wide per-dim std breakdown ───────────────
    if rb_entropy_stats is not None:
        ax_bar  = fig.add_subplot(gs[2, 0], facecolor="#161b22")
        ax_info = fig.add_subplot(gs[2, 1], facecolor="#161b22")

        # Bar chart: σ per action dimension
        std_per_dim = rb_entropy_stats["std_per_dim"]
        x = np.arange(len(std_per_dim))
        floor = np.exp(-5)
        colors = [("#f78166" if v < 2 * floor else "#3fb950") for v in std_per_dim]
        ax_bar.bar(x, std_per_dim, color=colors, width=0.6)
        ax_bar.axhline(floor, color="#8b949e", lw=0.8, ls="--",
                       label=f"LOG_STD_MIN floor ({floor:.4f})")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"a{i}" for i in x], fontsize=7, color="#8b949e")
        ax_bar.legend(fontsize=6, facecolor="#0d1117", labelcolor="#e6edf3",
                      edgecolor="#30363d")
        _style(ax_bar,
               f"σ per action dim — {rb_entropy_stats['n_states']} buffer states\n"
               f"(red = collapsed, green = healthy)",
               "Action dim", "σ")

        # Text summary panel
        collapsed = rb_entropy_stats["mean_std_mean"] < 2 * floor
        verdict = "⚠  COLLAPSED — near-deterministic" if collapsed else "✓  Healthy — stochastic"
        summary = (
            f"Buffer-wide snapshot\n"
            f"────────────────────\n"
            f"States sampled : {rb_entropy_stats['n_states']}\n"
            f"Action dims    : {rb_entropy_stats['action_dim']}\n\n"
            f"Mean σ         : {rb_entropy_stats['mean_std_mean']:.5f}\n"
            f"  (floor σ     : {floor:.5f})\n"
            f"σ min          : {rb_entropy_stats['mean_std_min']:.5f}\n"
            f"σ max          : {rb_entropy_stats['mean_std_max']:.5f}\n"
            f"Mean log σ     : {rb_entropy_stats['mean_log_std']:.3f}\n"
            f"  (LOG_STD_MIN : -5.000)\n"
            f"Mean H(π) nats : {rb_entropy_stats['entropy_mean']:.4f}\n\n"
            f"Verdict:\n{verdict}"
        )
        ax_info.text(0.05, 0.95, summary, transform=ax_info.transAxes,
                     color="#e6edf3", fontsize=8, verticalalignment="top",
                     fontfamily="monospace",
                     bbox=dict(facecolor="#0d1117", edgecolor="#30363d",
                               boxstyle="round,pad=0.5"))
        ax_info.axis("off")

    out_path = os.path.join(weights_path, "entropy_diagnosis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[diagnose_entropy] Figure saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Post-hoc SAC entropy analysis.")
    p.add_argument("--weights_path", required=True,
                   help="Path to a run directory containing weights.pth and info_sac_logs.csv")
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    post_hoc_entropy_analysis(
        weights_path=args.weights_path,
        num_samples=args.num_samples,
        device_str=args.device,
    )