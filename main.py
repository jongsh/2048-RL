import argparse

from trainer.trainer import Trainer
from configs.config import Configuration


# Function to get the agent based on the key
def get_agent(key):
    if key == "dqn":
        from agents.dqn_agent import DQNAgent

        return DQNAgent


# Function to get the trainer based on the key
def get_trainer(key):
    if key == "base":
        from trainer.trainer import Trainer

        return Trainer


# Function to get the environment based on the key
def get_env(key):
    if key == "game2048":
        from envs.game2048_env import Game2048Env

        return Game2048Env


def main():
    parser = argparse.ArgumentParser(description="2048 RL")
    parser.add_argument("--mode", type=str, required=True, help="Mode of operation: train | test | retrain")
    parser.add_argument("--config", type=str, default="configs/public_config.yaml", help="Path to config file")

    # Parse arguments
    args = parser.parse_args()
    mode = args.mode.lower()
    config_path = args.config
    public_config = Configuration(config_path=config_path).get_config("public")

    # train model from scratch
    if mode == "train":
        trainer = get_trainer(public_config["trainer"])()
        agent = get_agent(public_config["agent"])()
        env = get_env(public_config["env"])()
        trainer.train(agent, env, is_resume=False)

    # retrain model
    elif mode == "retrain":
        trainer = get_trainer(public_config["trainer"])()
        agent = get_agent(public_config["agent"])()
        env = get_env(public_config["env"])()
        trainer.train(agent, env, is_resume=True)
        pass

    # evaluate model
    elif mode == "test":
        pass


if __name__ == "__main__":
    main()
