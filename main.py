import argparse
import time
import pygame
import random
import os

from configs.config import Configuration
from trainer.trainer import Trainer


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
        from trainer.dqn_trainer import DQNTrainer

        return DQNTrainer

    elif key == "imitation":
        from trainer.imitation_trainer import ImitationTrainer

        return ImitationTrainer


# Function to get the environment based on the key
def get_env(key):
    if key == "game2048":
        from envs.game2048_env import Game2048Env

        return Game2048Env


def main():
    parser = argparse.ArgumentParser(description="2048 RL")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode")
    parser.add_argument("--test", action="store_true", help="Flag to indicate testing mode")
    parser.add_argument("--retrain", action="store_true", help="Flag to indicate retraining mode")

    # Parse arguments
    args, unknown = parser.parse_known_args()
    assert args.train + args.test + args.retrain == 1, "Please specify exactly one mode: --train, --test, or --retrain"
    config_path = args.config
    public_config = Configuration(config_path=config_path, cli_args=unknown).get_config("public")

    # train model from scratch
    if args.train:
        trainer = get_trainer(public_config["trainer"])()
        agent = get_agent(public_config["agent"])()
        env = get_env(public_config["env"])()
        trainer.train(agent, env, is_resume=False)

    # retrain model
    elif args.retrain:
        trainer = get_trainer(public_config["trainer"])()
        agent = get_agent(public_config["agent"])()
        env = get_env(public_config["env"])()
        trainer.train(agent, env, is_resume=True)

    # evaluate model
    elif args.test:
        checkpoint_dir = public_config["from_checkpoint"]
        assert checkpoint_dir and os.path.exists(checkpoint_dir), f"Checkpoint path {checkpoint_dir} does not exist!"

        agent = get_agent(public_config["agent"])()
        agent.load(checkpoint_dir)
        env = get_env(public_config["env"])(silent_mode=False)

        obs, info = env.reset()
        env.render()
        actions = ["left", "right", "up", "down"]
        running = True
        total_reward = 0

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
