import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import make_dirs


def load_learning_curve(log_dir: Path):
    try:
        from stable_baselines3.common.monitor import load_results
    except Exception:
        return None

    try:
        data = load_results(str(log_dir))
    except Exception:
        return None

    if data.empty:
        return None

    data = data.sort_values("t")
    data["timesteps"] = data["l"].cumsum()
    data["reward_smooth"] = data["r"].rolling(window=20, min_periods=1).mean()
    return data


def plot_learning_curves(env_id: str, results_dir: Path, summary_df: pd.DataFrame):
    variants = ["baseline", "vectorized", "tuned"]
    plt.figure(figsize=(8, 5))
    have_curve = False

    for variant in variants:
        log_dir = results_dir / "logs" / f"{env_id}_{variant}_seed0"
        curve = load_learning_curve(log_dir)
        if curve is not None:
            have_curve = True
            plt.plot(curve["timesteps"], curve["reward_smooth"], label=variant)

    if not have_curve:
        subset = summary_df[summary_df["env_id"] == env_id]
        means = (
            subset.groupby("variant")["eval_mean_reward"].mean().reindex(variants)
        )
        plt.bar(means.index, means.values)
        plt.ylabel("Final eval mean reward")
        plt.title(f"{env_id} final rewards (no logs found)")
    else:
        plt.xlabel("Timesteps")
        plt.ylabel("Episode reward (smoothed)")
        plt.title(f"{env_id} learning curves")
        plt.legend()

    plot_path = results_dir / "plots" / f"{env_id}_learning_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_stability(env_id: str, results_dir: Path, aggregate_df: pd.DataFrame):
    variants = ["baseline", "vectorized", "tuned"]
    subset = aggregate_df[aggregate_df["env_id"] == env_id]
    subset = subset.set_index("variant").reindex(variants)

    means = subset["eval_mean_reward_mean"].values
    stds = subset["eval_mean_reward_std"].fillna(0.0).values

    plt.figure(figsize=(6, 4))
    plt.bar(variants, means, yerr=stds, capsize=4)
    plt.ylabel("Eval mean reward")
    plt.title(f"{env_id} stability across seeds")

    plot_path = results_dir / "plots" / f"{env_id}_stability.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main() -> int:
    try:
        import matplotlib
    except ImportError as exc:
        print(
            "Missing dependencies. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[1]
    dirs = make_dirs(root)

    summary_path = dirs["tables"] / "summary.csv"
    aggregate_path = dirs["tables"] / "aggregate.csv"
    if not summary_path.exists():
        print(f"Missing summary at {summary_path}. Run evaluate_seeds.py first.")
        return 1

    summary_df = pd.read_csv(summary_path)
    if aggregate_path.exists():
        aggregate_df = pd.read_csv(aggregate_path)
    else:
        aggregate_df = (
            summary_df.groupby(["env_id", "variant"])
            .agg(
                eval_mean_reward_mean=("eval_mean_reward", "mean"),
                eval_mean_reward_std=("eval_mean_reward", "std"),
            )
            .reset_index()
        )

    for env_id in ["CartPole-v1", "Acrobot-v1"]:
        plot_learning_curves(env_id, root / "results", summary_df)
        plot_stability(env_id, root / "results", aggregate_df)

    print(f"Plots saved to {dirs['plots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
