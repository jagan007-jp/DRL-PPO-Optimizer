import argparse
import sys
from pathlib import Path

import pandas as pd

from utils import evaluate, load_json, make_dirs, make_env, set_global_seeds, timer


BASELINE_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.0,
    "clip_range": 0.2,
}

VECTORIZED_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 1024,
    "batch_size": 256,
    "ent_coef": 0.0,
    "clip_range": 0.2,
}

DEFAULT_TIMESTEPS = {
    "CartPole-v1": 80000,
    "Acrobot-v1": 120000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for PPO variants.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> int:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import VecMonitor
    except ImportError as exc:
        print(
            "Missing dependencies. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    dirs = make_dirs(root)

    best_params_path = dirs["tables"] / "best_params_cartpole.json"
    tuned_params = None
    if best_params_path.exists():
        tuned_params = load_json(best_params_path)

    results = []
    env_ids = ["CartPole-v1", "Acrobot-v1"]
    variants = ["baseline", "vectorized", "tuned"]

    for env_id in env_ids:
        timesteps = DEFAULT_TIMESTEPS[env_id]
        for variant in variants:
            if variant == "tuned" and tuned_params is None:
                print(
                    f"Missing tuned params at {best_params_path}. Run tune_optuna.py first.",
                    file=sys.stderr,
                )
                return 1
            for seed in args.seeds:
                set_global_seeds(seed)

                log_name = f"{env_id}_{variant}_seed{seed}"
                log_dir = dirs["logs"] / log_name
                log_dir.mkdir(parents=True, exist_ok=True)
                model_path = dirs["models"] / f"{log_name}.zip"

                model = None
                train_time = 0.0

                if model_path.exists():
                    model = PPO.load(model_path)
                else:
                    if variant == "baseline":
                        env = Monitor(make_env(env_id, seed=seed), str(log_dir / "monitor.csv"))
                        logger = configure(str(log_dir), ["stdout", "csv"])
                        model = PPO(
                            "MlpPolicy",
                            env,
                            seed=seed,
                            verbose=0,
                            **BASELINE_PARAMS,
                        )
                        model.set_logger(logger)
                        with timer() as t:
                            model.learn(total_timesteps=timesteps, progress_bar=False)
                        train_time = t["elapsed"]
                        model.save(model_path)
                        env.close()
                    elif variant == "vectorized":
                        n_envs = 8
                        env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
                        env = VecMonitor(env, filename=str(log_dir / "monitor.csv"))
                        logger = configure(str(log_dir), ["stdout", "csv"])
                        model = PPO(
                            "MlpPolicy",
                            env,
                            seed=seed,
                            verbose=0,
                            **VECTORIZED_PARAMS,
                        )
                        model.set_logger(logger)
                        with timer() as t:
                            model.learn(total_timesteps=timesteps, progress_bar=False)
                        train_time = t["elapsed"]
                        model.save(model_path)
                        env.close()
                    else:
                        n_envs = int(tuned_params.get("n_envs", 8))
                        n_steps = int(tuned_params["n_steps"])
                        batch_size = int(tuned_params["batch_size"])
                        total_batch = n_envs * n_steps
                        if batch_size > total_batch:
                            batch_size = total_batch
                        env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
                        env = VecMonitor(env, filename=str(log_dir / "monitor.csv"))
                        logger = configure(str(log_dir), ["stdout", "csv"])
                        model = PPO(
                            "MlpPolicy",
                            env,
                            seed=seed,
                            verbose=0,
                            learning_rate=float(tuned_params["learning_rate"]),
                            gamma=float(tuned_params["gamma"]),
                            n_steps=n_steps,
                            batch_size=batch_size,
                            ent_coef=float(tuned_params["ent_coef"]),
                            clip_range=float(tuned_params["clip_range"]),
                        )
                        model.set_logger(logger)
                        with timer() as t:
                            model.learn(total_timesteps=timesteps, progress_bar=False)
                        train_time = t["elapsed"]
                        model.save(model_path)
                        env.close()

                mean_reward, std_reward = evaluate(model, env_id, n_eval_episodes=20)

                results.append(
                    {
                        "env_id": env_id,
                        "variant": variant,
                        "seed": seed,
                        "timesteps": timesteps,
                        "train_time_sec": train_time,
                        "eval_mean_reward": mean_reward,
                        "eval_std_reward": std_reward,
                    }
                )

    summary_df = pd.DataFrame(results)
    summary_path = dirs["tables"] / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    agg_df = (
        summary_df.groupby(["env_id", "variant"])
        .agg(
            eval_mean_reward_mean=("eval_mean_reward", "mean"),
            eval_mean_reward_std=("eval_mean_reward", "std"),
            train_time_sec_mean=("train_time_sec", "mean"),
        )
        .reset_index()
    )
    aggregate_path = dirs["tables"] / "aggregate.csv"
    agg_df.to_csv(aggregate_path, index=False)

    print(f"Saved summary to {summary_path}")
    print(f"Saved aggregate to {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
