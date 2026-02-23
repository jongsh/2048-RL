import os
import argparse
import time
import pygame

from configs.config import Configuration
from trainers.trainer import Trainer


# Function to get the agent based on the key
def get_agent(key):
    if key == "dqn":
        from agents.dqn_agent import DQNAgent

        return DQNAgent

    elif key == "imitation":
        from agents.imitation_agent import ImitationAgent

        return ImitationAgent


# Function to get the trainer based on the key
def get_trainer(key) -> Trainer:
    if key == "dqn":
        from trainers.dqn_trainer import DQNTrainer

        return DQNTrainer

    elif key == "imitation":
        from trainers.imitation_trainer import ImitationTrainer

        return ImitationTrainer


# Function to get the environment based on the key
def get_env(key):
    if key == "game2048":
        from envs.game2048_env import Game2048Env

        return Game2048Env


def main():
    parser = argparse.ArgumentParser(description="2048 RL")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode")
    parser.add_argument("--test", action="store_true", help="Flag to indicate testing mode")
    parser.add_argument("--retrain", action="store_true", help="Flag to indicate retraining mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing")

    # Parse arguments
    args, unknown = parser.parse_known_args()
    assert args.train + args.test + args.retrain == 1, "Please specify exactly one mode: --train, --test, or --retrain"
    config_path = args.config
    config = Configuration(config_path=config_path, cli_args=unknown)
    component_config = config["components"]

    # train model from scratch
    if args.train:
        agent = get_agent(component_config["agent"])(config=config)
        env = get_env(component_config["env"])(config=config)
        trainer = get_trainer(component_config["trainer"])(config=config)
        trainer.train(agent, env, is_resume=False)

    # retrain model
    elif args.retrain:
        agent = get_agent(component_config["agent"])(config=config)
        env = get_env(component_config["env"])(config=config)
        trainer = get_trainer(component_config["trainer"])(config=config)
        trainer.train(agent, env, is_resume=True)

    # evaluate model
    elif args.test:
        assert args.checkpoint is not None, "Please provide a checkpoint path using --checkpoint for testing mode."
        assert os.path.exists(args.checkpoint), f"Checkpoint path {args.checkpoint} does not exist!"
        checkpoint_dir = args.checkpoint

        agent = get_agent(component_config["agent"])(config=config)
        agent.load(checkpoint_dir)
        env = get_env(component_config["env"])(config=config, silent_mode=False)

        obs, info = env.reset()
        actions = env.action_names
        running = True
        total_reward = 0
        env.render()
        while running:
            time.sleep(1.5)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = agent.select_action(obs, action_mask=info["action_mask"], method="sample")
            obs, reward, done, _, info = env.step(action)

            total_reward += reward
            print(f"Action: {actions[action]}, Reward: {reward}, Done: {done}")
            env.render()

            if done:
                print(f"Game Over!\n\nTotal Reward: {total_reward:6f}\n\nFinal Info: {info}")
                break


if __name__ == "__main__":
    main()
