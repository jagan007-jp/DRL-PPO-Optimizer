import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_dirs(base_dir: Path | None = None) -> dict:
    root = Path(base_dir) if base_dir else project_root()
    results = root / "results"
    subdirs = {
        "models": results / "models",
        "logs": results / "logs",
        "plots": results / "plots",
        "tables": results / "tables",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def make_env(env_id: str, seed: int):
    import gymnasium as gym

    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate(model, env_id: str, n_eval_episodes: int = 20, deterministic: bool = True):
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(make_env(env_id, seed=0))
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )
    env.close()
    return mean_reward, std_reward


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer():
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
