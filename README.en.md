# 2048-RL

[中文](README.md)

A 2048 project combining Reinforcement Learning (RL) and Imitation Learning (IL). It currently supports DQN and imitation pipelines with composable YAML configurations.

## Features
- Two agent types: `dqn`, `imitation`
- Multiple model backbones: `mlp`, `resnet`, `transformer`
- Online/offline DQN training strategy
- Hierarchical configuration powered by `OmegaConf`
- Automatic checkpointing with optimizer state and metadata

## Project Structure
```text
.
├─ main.py                  # Entry: train / retrain / test
├─ agents/
├─ trainers/
├─ envs/
├─ models/
├─ configs/
├─ data/
└─ utils/
```

## Requirements
- Python 3.9+
- CUDA is recommended (CPU also works)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
Train from scratch:
```bash
python main.py --config configs/config.yaml --train
```

Resume training (set `trainer.from_checkpoint` first):
```bash
python main.py --config configs/config.yaml --retrain
```

Evaluate a checkpoint:
```bash
python main.py --config configs/config.yaml --test --checkpoint outputs/<exp_name>/<timestamp>/checkpoint_<episode>
```

## Configuration
Main config: `configs/config.yaml`.
Choose components via:
- `components.agent`: `dqn` / `imitation`
- `components.model`: `mlp` / `resnet` / `transformer`
- `components.trainer`: `dqn` / `imitation`

Override config values with OmegaConf dotlist:
```bash
python main.py --train --components.agent=imitation --components.trainer=imitation --trainer.exp_name=imitation_run
```

## Data
- `data/human_2048.json`: human demonstration data
- `data/agent_2048.json`: agent trajectory data
- preprocessing: `data/data_2048.py`

## Development Notes
- Reuse abstractions in `agents/base_agent.py` and `trainers/trainer.py` when adding new algorithms.
- Run at least one train/test smoke check before submitting changes.
