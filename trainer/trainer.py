import torch
import math

from datetime import datetime
from tqdm import tqdm

from agents.base_agent import BaseAgent
from utils.logger import Logger
from utils.reply_buffer import ReplayBuffer
from configs.config import load_single_config


class Trainer:
    """Trainer for the RL Agent"""

    def __init__(self, config=load_single_config("trainer", "base"), **kwargs):
        assert config["exp_name"], "Experiment name must be provided"

        self.exp_dir = config["output_dir"] / config["exp_name"] / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(self.exp_dir, config["exp_name"])
        self.replay_buffer = ReplayBuffer(config["replay_buffer_size"], config["replay_buffer_size_min"])

        self.optimizer_cls = self._load_optimizer(config["optimizer"])
        self.optimizer = None
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = config["batch_size"]
        self.episode = config["episode"]
        self.episode_max_step = config["episode_max_step"]
        self.learning_rate = config["learning_rate"]
        self.device = config["device"]

        self.log_interval = config["log_interval"]
        self.save_interval = config["save_interval"]
        self.from_checkpoint = config["from_checkpoint"]

    def _load_optimizer(self, optimizer_name):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD

    def train(self, agent: BaseAgent, env):
        """Train the agent in the environment"""
        self.optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.learning_rate)
        # TODO: Implement the retraining logic
        with tqdm(total=self.episode, desc="Training Progress") as pbar_epoch:

            for ep in range(1, self.episode + 1):
                cur_episode_reward = 0  # episode reward in the current epoch
                cur_episode_step = 0  # steps in the current episode
                cur_train_batch = 0  # total batch in the current episode
                cur_train_loss = 0.0  # current train loss

                state = env.reset()
                done = False

                while not done and cur_episode_step < self.episode_max_step:
                    # Sample action from the agent
                    cur_episode_step += 1
                    action = agent.sample_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # Add experience to the replay buffer
                    self.replay_buffer.add(state, action, reward, next_state, done)

                    # Update the agent with a batch from the replay buffer
                    batch = self.replay_buffer.sample(self.batch_size)
                    if batch is not None:
                        loss = agent.update(*batch, self.optimizer, self.loss_fn)
                        cur_train_batch += 1
                        cur_train_loss += loss

                    state = next_state
                    cur_episode_reward += reward

                avg_loss = cur_train_loss / cur_train_batch if cur_train_batch > 0 else 0

                # Log the current episode results
                if ep % self.log_interval == 0:
                    avg_reward = cur_episode_reward / cur_episode_step if cur_episode_step > 0 else 0
                    self.logger.info(
                        f"Episode {ep}, Total Reward: {cur_episode_reward:.4f} Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}"
                    )

                # Save training progress
                if ep % self.save_interval == 0:
                    self._save_checkpoint(agent, self.optimizer, {"episode": ep})

                pbar_epoch.update(1)
                pbar_epoch.set_postfix(
                    episode=ep,
                    reward=cur_episode_reward,
                    loss=avg_loss if cur_train_batch > 0 else 0,
                )

    def _save_checkpoint(self, agent: BaseAgent, optimizer, metadata):
        """Save the checkpoint"""
        save_dir = self.exp_dir / f"checkpoint_{metadata['episode']}"
        agent.save(save_dir)
        torch.save(optimizer.state_dict(), save_dir / "optimizer.pth")
        torch.save(metadata, save_dir / "metadata.pth")

    def _load_checkpoint(self):
        """Load the checkpoint"""
        pass  # TODO: Implement loading logic if needed
