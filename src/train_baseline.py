import argparse
import sys
from pathlib import Path

from utils import evaluate, make_dirs, make_env, set_global_seeds


BASELINE_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.0,
    "clip_range": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline PPO on classic-control envs.")
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_name", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        print(
            "Missing dependencies. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    args = parse_args()
    set_global_seeds(args.seed)

    root = Path(__file__).resolve().parents[1]
    dirs = make_dirs(root)

    log_name = args.log_name or f"{args.env_id}_baseline_seed{args.seed}"
    log_dir = dirs["logs"] / log_name
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, seed=args.seed)
    env = Monitor(env, str(log_dir / "monitor.csv"))

    logger = configure(str(log_dir), ["stdout", "csv"])

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        **BASELINE_PARAMS,
    )
    model.set_logger(logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model_path = dirs["models"] / f"{log_name}.zip"
    model.save(model_path)

    mean_reward, std_reward = evaluate(model, args.env_id, n_eval_episodes=20)
    print(f"Eval over 20 episodes: mean={mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
