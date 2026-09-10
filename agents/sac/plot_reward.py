#!/usr/bin/env python3
"""Plot average reward for one or more SAC run directories."""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reward import RewardTracker

TIME_WINDOW = 120.0
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return {}
    with open(args_path) as f:
        return json.load(f)


def _hidden_size_suffix(run_args: dict) -> str:
    if "hidden_size" in run_args:
        return f"hidden={run_args['hidden_size']}"
    actor = run_args.get("actor_hidden_dim")
    critic = run_args.get("critic_hidden_dim")
    if actor is not None or critic is not None:
        if actor is not None and critic is not None and actor == critic:
            return f"hidden={actor}"
        parts = []
        if actor is not None:
            parts.append(f"actor={actor}")
        if critic is not None:
            parts.append(f"critic={critic}")
        return ", ".join(parts)
    # Older MLP SAC runs were saved before --hidden_size existed (default 256).
    return "hidden=256"


def _run_label(run_dir: Path, run_args: dict) -> str:
    base = _algorithm_name(run_dir, run_args)
    suffix = _hidden_size_suffix(run_args)
    return f"{base} ({suffix})" if suffix else base


def _algorithm_name(run_dir: Path, run_args: dict) -> str:
    parent = run_dir.parent.name.lower()
    if "simba" in parent or "actor_hidden_dim" in run_args:
        return "SAC+SimBa"
    return "SAC"


def _hidden_size_value(run_args: dict) -> Optional[int]:
    if "hidden_size" in run_args:
        return int(run_args["hidden_size"])
    actor = run_args.get("actor_hidden_dim")
    critic = run_args.get("critic_hidden_dim")
    if actor is not None and critic is not None:
        if int(actor) == int(critic):
            return int(actor)
        return None
    if actor is not None:
        return int(actor)
    if critic is not None:
        return int(critic)
    return 256


def _is_trial_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "args.json").exists()
        or (path / "info_sac_logs.csv").exists()
        or any(path.glob("*_average_rewards.csv"))
    )


def _expand_run_dirs(paths: list[Path]) -> list[Path]:
    """If a path is a parent runs_* folder, include every trial inside it."""
    resolved = []
    seen = set()
    for path in paths:
        if path.is_dir() and not _is_trial_dir(path):
            children = sorted(p for p in path.iterdir() if _is_trial_dir(p))
            if children:
                candidates = children
            else:
                candidates = [path]
        else:
            candidates = [path]
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(key)
    return resolved


def _find_average_rewards_csv(run_dir: Path, env_id: Optional[str]) -> Optional[Path]:
    candidates = []
    if env_id:
        candidates.append(run_dir / f"{env_id}_average_rewards.csv")
    candidates.extend(sorted(run_dir.glob("*_average_rewards.csv")))
    for path in candidates:
        if path.exists():
            return path
    return None


def load_average_rewards(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_args = _load_args(run_dir)
    env_dt = float(run_args.get("dt", 0.05))
    env_id = run_args.get("env_id", "SimEmbodiedAnt")

    csv_path = _find_average_rewards_csv(run_dir, env_id)
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        if "step" not in df.columns or "reward" not in df.columns:
            raise KeyError(f"{csv_path} must contain columns 'step' and 'reward'")
        print(f"Loaded average rewards from {csv_path} ({len(df)} rows)")
    else:
        info_path = run_dir / "info_sac_logs.csv"
        if not info_path.exists():
            raise FileNotFoundError(
                f"No *_average_rewards.csv or info_sac_logs.csv in {run_dir}"
            )
        info = pd.read_csv(info_path)
        if "original_rewards" not in info.columns:
            raise KeyError("info_sac_logs.csv must contain column 'original_rewards'")

        # original_rewards is logged as a one-element list string, e.g. "[0.01]"
        rewards = (
            info["original_rewards"]
            .astype(str)
            .str.strip("[]")
            .astype(float)
        )
        if "global_step" in info.columns:
            steps = info["global_step"].astype(int)
        else:
            steps = pd.Series(range(1, len(rewards) + 1))

        tracker = RewardTracker(
            env_dt=env_dt,
            env_id=env_id,
            time_window=TIME_WINDOW,
            log_folder=tempfile.mkdtemp(prefix="sac_plot_reward_"),
        )
        rows = []
        for step, reward in zip(steps, rewards):
            tracker.update(reward)
            if tracker.average_reward_per_second is not None:
                rows.append([step, tracker.average_reward_per_second])

        if not rows:
            raise ValueError(
                f"Not enough steps in {run_dir} to fill the "
                f"{TIME_WINDOW:.0f}s averaging window "
                f"(need ~{int(TIME_WINDOW / env_dt)} steps; got {len(rewards)})"
            )
        df = pd.DataFrame(rows, columns=["step", "reward"])
        print(f"Recomputed average rewards from {info_path} ({len(df)} rows)")

    df = df.copy()
    df["reward_cm_s"] = df["reward"] * 100  # m/s -> cm/s
    df["time_minutes"] = df["step"] * env_dt / 60
    return df


def _plot_final_vs_hidden(records: list[dict], output_path: Path) -> None:
    """Plot last-window reward vs hidden size, one series per algorithm."""
    best = {}
    for rec in records:
        key = (rec["algorithm"], rec["hidden"])
        if key not in best or rec["n_steps"] > best[key]["n_steps"]:
            best[key] = rec

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for algorithm, marker in (("SAC", "o"), ("SAC+SimBa", "s")):
        points = sorted(
            (rec["hidden"], rec["reward_cm_s"])
            for rec in best.values()
            if rec["algorithm"] == algorithm
        )
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker=marker, linewidth=1.5, label=algorithm)
        for x, y in points:
            print(f"{algorithm} hidden={x}: final reward={y:.3f} cm/s")

    ax.set_xlabel("Hidden layer size")
    ax.set_ylabel("Final Average Reward [cm/s]")
    ax.set_title("Final Reward vs Hidden Size")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    hidden_ticks = sorted({rec["hidden"] for rec in best.values()})
    if hidden_ticks:
        ax.set_xticks(hidden_ticks)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot SAC average reward curves")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Trial directories or parent runs_* folders (expands to all trials inside)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot (default: <first_run_dir>/average_reward.png)",
    )
    args = parser.parse_args()

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    input_dirs = []
    for run_dir in args.run_dirs:
        path = Path(run_dir)
        if not path.is_absolute():
            candidate = SCRIPT_DIR / path
            path = candidate if candidate.exists() else Path(run_dir)
        input_dirs.append(path)
    resolved_dirs = _expand_run_dirs(input_dirs)
    if not resolved_dirs:
        raise FileNotFoundError(f"No trial directories found in {args.run_dirs}")

    def _dir_sort_key(run_dir: Path):
        run_args = _load_args(run_dir)
        algo = _algorithm_name(run_dir, run_args)
        hidden = _hidden_size_value(run_args)
        algo_order = 0 if algo == "SAC" else 1 if algo == "SAC+SimBa" else 2
        return (algo_order, hidden if hidden is not None else 0, run_dir.name)

    resolved_dirs = sorted(resolved_dirs, key=_dir_sort_key)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    labels = [_run_label(run_dir, _load_args(run_dir)) for run_dir in resolved_dirs]
    if len(labels) != len(set(labels)):
        counts = Counter(labels)
        disambiguated = []
        for label, run_dir in zip(labels, resolved_dirs):
            if counts[label] > 1:
                # trial_1_20260910-115935_seed_1 -> 115935
                stamp = run_dir.name.split("-")[-1].split("_")[0]
                disambiguated.append(f"{label} {stamp}")
            else:
                disambiguated.append(label)
        labels = disambiguated

    plotted = 0
    final_records = []
    for run_dir, label in zip(resolved_dirs, labels):
        try:
            df = load_average_rewards(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {run_dir}: {exc}")
            continue
        ax.plot(df["time_minutes"], df["reward_cm_s"], label=label, linewidth=1.2)
        plotted += 1
        run_args = _load_args(run_dir)
        hidden = _hidden_size_value(run_args)
        if hidden is None:
            print(f"Skipping {run_dir} on hidden-size plot: actor/critic dims differ")
            continue
        final_records.append(
            {
                "algorithm": _algorithm_name(run_dir, run_args),
                "hidden": hidden,
                "reward_cm_s": float(df["reward_cm_s"].iloc[-1]),
                "n_steps": int(df["step"].iloc[-1]),
            }
        )
    if plotted == 0:
        raise ValueError("No runnable trial directories had enough reward data to plot")

    ax.set_xlabel("Time [minutes]")
    ax.set_ylabel("Average Reward [cm/s]")
    ax.set_title("Average Reward over Time")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    if len(resolved_dirs) > 1:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(args.output) if args.output else resolved_dirs[0] / "average_reward.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    if final_records:
        hidden_output = output_path.with_name(f"final_reward_vs_hidden_size{output_path.suffix}")
        _plot_final_vs_hidden(final_records, hidden_output)


if __name__ == "__main__":
    main()
