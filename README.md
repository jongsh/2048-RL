# 2048-RL

[English](README.en.md)

一个基于强化学习（RL）与模仿学习（IL）的 2048 项目。当前支持 DQN 与 Imitation 两条训练路径，并使用可组合 YAML 配置管理实验。

## 项目特性
- 支持两类 Agent：`dqn`、`imitation`
- 支持多模型骨干：`mlp`、`resnet`、`transformer`
- 支持 DQN 在线/离线训练策略
- 基于 `OmegaConf` 的分层配置和命令行覆盖
- 自动保存 checkpoint、优化器状态和训练元数据

## 项目结构
```text
.
├─ main.py                  # 入口：train / retrain / test
├─ agents/                  # 智能体
├─ trainers/                # 训练流程
├─ envs/                    # 2048 环境
├─ models/                  # 网络结构
├─ configs/                 # 全局与组件配置
│  ├─ config.yaml
│  ├─ agents/
│  ├─ envs/
│  ├─ models/
│  └─ trainers/
├─ data/                    # 离线/模仿数据
└─ utils/                   # 日志、缓冲区、可视化等
```

## 环境要求
- Python 3.9+
- 推荐 CUDA 环境（CPU 也可运行）

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始
训练：
```bash
python main.py --config configs/config.yaml --train
```

恢复训练（需先在配置中设置 `trainer.from_checkpoint`）：
```bash
python main.py --config configs/config.yaml --retrain
```

评估：
```bash
python main.py --config configs/config.yaml --test --checkpoint outputs/<exp_name>/<timestamp>/checkpoint_<episode>
```

## 配置说明
主配置：`configs/config.yaml`，通过 `components` 选择组件：
- `components.agent`: `dqn` / `imitation`
- `components.model`: `mlp` / `resnet` / `transformer`
- `components.trainer`: `dqn` / `imitation`

命令行覆盖（OmegaConf dotlist）：
```bash
python main.py --train --components.agent=imitation --components.trainer=imitation --trainer.exp_name=imitation_run
```

## 数据说明
- `data/human_2048.json`：人类演示数据
- `data/agent_2048.json`：智能体轨迹数据
- 预处理代码：`data/data_2048.py`

## 开发建议
- 扩展算法时优先复用 `agents/base_agent.py` 与 `trainers/trainer.py`。
- 提交前至少执行一次 train/test 冒烟验证。
