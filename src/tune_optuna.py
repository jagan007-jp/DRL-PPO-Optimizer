import argparse
import sys
from pathlib import Path

from utils import evaluate, make_dirs, save_json, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for PPO on CartPole-v1.")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timesteps_per_trial", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    try:
        import optuna
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
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

    env_id = "CartPole-v1"
    n_envs = 4

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_seed = args.seed + trial.number
        set_global_seeds(trial_seed)

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        total_batch = n_envs * n_steps
        if batch_size > total_batch:
            raise optuna.TrialPruned()
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

        env = make_vec_env(env_id, n_envs=n_envs, seed=trial_seed)

        model = PPO(
            "MlpPolicy",
            env,
            seed=trial_seed,
            verbose=0,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_range,
        )
        model.learn(total_timesteps=args.timesteps_per_trial)

        mean_reward, _ = evaluate(model, env_id, n_eval_episodes=10)
        env.close()
        return float(mean_reward)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_params["n_envs"] = n_envs
    save_json(dirs["tables"] / "best_params_cartpole.json", best_params)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(dirs["tables"] / "optuna_trials.csv", index=False)

    print("Best params:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
