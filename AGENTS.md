# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the single entrypoint for training, retraining, and evaluation.
- `agents/` contains policy logic (`dqn_agent.py`, `imitation_agent.py`) and shared abstractions (`base_agent.py`).
- `trainers/` contains training loops (`dqn_trainer.py`, `imitation_trainer.py`, `trainer.py`).
- `envs/` implements the 2048 environment and rendering integration.
- `models/` contains model backbones (`mlp.py`, `resnet.py`, `transformer.py`) and common layers.
- `configs/` stores composable YAML configs (`config.yaml` + `agents/`, `envs/`, `models/`, `trainers/`).
- `data/` stores imitation/offline training data (JSON) and preprocessing (`data_2048.py`).
- `utils/` contains logging, replay buffer, schedulers, normalization, and plotting helpers.

## Build, Test, and Development Commands
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Train from scratch:
  ```bash
  python main.py --config configs/config.yaml --train
  ```
- Resume training:
  ```bash
  python main.py --config configs/config.yaml --retrain
  ```
- Evaluate a checkpoint:
  ```bash
  python main.py --config configs/config.yaml --test --checkpoint outputs/<exp>/<timestamp>/checkpoint.pt
  ```
- Override config values from CLI (OmegaConf dotlist):
  ```bash
  python main.py --train --components.agent=imitation --trainer.exp_name=imitation_run
  ```

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/variables/files, `PascalCase` for classes.
- Keep modules focused by responsibility (`agents`, `trainers`, `envs`, `models`, `utils`).
- Prefer explicit type-friendly interfaces for trainer and agent methods when extending base classes.

## Testing Guidelines
- There is currently no committed automated test suite.
- Every PR should include runnable verification steps (at minimum: one train/retrain/test smoke command and expected outcome).
- For new logic, add targeted tests under a new `tests/` directory and keep fixtures minimal.

## Commit & Pull Request Guidelines
- Recent history favors concise, prefixed commits (for example: `feat: ...`, `fix: ...`, `refactor: ...`, `deploy: ...`).
- Use format: `<type>: <short summary>`; keep one logical change per commit.
- PRs should include:
  - what changed and why,
  - related issue/task link,
  - config changes (`configs/**/*.yaml`),
  - evidence of validation (logs, metrics, or screenshots for game-play evaluation).
