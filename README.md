# DRL PPO Optimizer

Cross-platform (macOS + Windows) Deep RL project comparing Baseline PPO vs Optimized PPO (vectorized envs + Optuna tuning + multi-seed stability).

## Project structure

```
drl-ppo-optimizer/
  README.md
  requirements.txt
  src/
    train_baseline.py
    train_vectorized.py
    tune_optuna.py
    evaluate_seeds.py
    plot_results.py
    utils.py
  results/
    models/
    logs/
    plots/
    tables/
```

## Setup

### Conda

```bash
conda create -n drl-ppo python=3.11
conda activate drl-ppo
pip install -r requirements.txt
```

### venv

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run commands

```bash
python src/train_baseline.py --env_id CartPole-v1 --timesteps 80000 --seed 42
python src/train_vectorized.py --env_id CartPole-v1 --timesteps 80000 --n_envs 8 --seed 42
python src/tune_optuna.py --n_trials 30 --timesteps_per_trial 30000 --seed 42
python src/evaluate_seeds.py
python src/plot_results.py
```

## Full run checklist (copy/paste)

```bash
cd "/Users/jaganprasathj/Desktop/CIP/Optimized DRL/drl-ppo-optimizer"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python src/train_baseline.py --env_id CartPole-v1 --timesteps 5000 --seed 0

python src/train_baseline.py --env_id CartPole-v1 --timesteps 80000 --seed 42
python src/train_baseline.py --env_id Acrobot-v1 --timesteps 120000 --seed 42

python src/train_vectorized.py --env_id CartPole-v1 --timesteps 80000 --n_envs 8 --seed 42
python src/train_vectorized.py --env_id Acrobot-v1 --timesteps 120000 --n_envs 8 --seed 42

python src/tune_optuna.py --n_trials 30 --timesteps_per_trial 30000 --seed 42

python src/evaluate_seeds.py
python src/plot_results.py
```

## What "optimized" means here

Optimized PPO = vectorized environments (default 8 envs) + hyperparameters tuned by Optuna on CartPole-v1, then validated across seeds and environments.

## Expected outcomes

- CartPole-v1 should reach ~475+ reward after sufficient training.
- Vectorized + tuned settings should converge faster and show tighter reward variance across seeds compared to baseline.

## Outputs

- Trained models: `results/models/`
- Logs (Monitor + SB3 CSV): `results/logs/`
- Summary tables: `results/tables/summary.csv`, `results/tables/aggregate.csv`
- Best params: `results/tables/best_params_cartpole.json`
- Plots: `results/plots/`

## Troubleshooting

- Torch CPU: This repo uses CPU-only. If torch installs a CUDA build, force CPU by reinstalling with the official CPU wheel.
- Windows PowerShell activation: If activation fails, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and try again.
- Missing dependencies: All scripts exit with a helpful message if imports fail. Reinstall with `pip install -r requirements.txt`.
