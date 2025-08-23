import torch
import math
import os

from datetime import datetime
from tqdm import tqdm

from agents.base_agent import BaseAgent
from utils.logger import Logger
from utils.reply_buffer import ReplayBuffer
from configs.config import Configuration


class Trainer:
    """Trainer for the RL Agent"""

    def __init__(self, config: Configuration = Configuration(), **kwargs):
        self.train_config = config.get_config("trainer")
        self.public_config = config.get_config("public")

        assert self.train_config["exp_name"], "Experiment name must be provided"

        self.exp_dir = os.path.join(
            self.train_config["output_dir"], self.train_config["exp_name"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.logger = Logger(self.exp_dir, self.train_config["exp_name"])
        self.replay_buffer = ReplayBuffer(
            self.train_config["replay_buffer_size"], self.train_config["replay_buffer_size_min"]
        )

        self.optimizer_cls = self._load_optimizer(self.train_config["optimizer"])
        self.optimizer = None  # will be initialized during training
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = self.train_config["batch_size"]
        self.episode = self.train_config["episode"]
        self.episode_max_step = self.train_config["episode_max_step"]
        self.learning_rate = self.train_config["learning_rate"]
        self.device = self.public_config["device"]

        self.log_interval = self.train_config["log_interval"]
        self.save_interval = self.train_config["save_interval"]
        self.from_checkpoint = self.public_config["from_checkpoint"]

    def _load_optimizer(self, optimizer_name):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD

    def train(self, agent: BaseAgent, env, is_resume=False):
        """Train the agent in the environment"""

        if is_resume:  # resume training from a checkpoint
            assert self.from_checkpoint, "Checkpoint path must be provided for resuming training"
            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.learning_rate)
            metadata = self._load_checkpoint(agent, optimizer, self.from_checkpoint)
        else:  # start training from scratch
            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.learning_rate)
            metadata = {"episode": 0, "cur_episode_reward": 0, "avg_loss": 0.0}

        # training loop
        with tqdm(total=self.episode, desc="Training Progress") as pbar_epoch:
            # Initialize progress bar
            pbar_epoch.set_postfix(**metadata)
            pbar_epoch.update(metadata["episode"])

            # starting training
            train_loss_list = []
            episode_reward_list = []

            for ep in range(metadata["episode"] + 1, self.episode + 1):
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
                        loss = agent.update(*batch, optimizer, self.loss_fn)
                        cur_train_batch += 1
                        cur_train_loss += loss

                    state = next_state
                    cur_episode_reward += reward

                avg_loss = cur_train_loss / cur_train_batch if cur_train_batch > 0 else 0

                # Log the current episode results
                if ep % self.log_interval == 0:
                    self.logger.info(
                        f"Episode {ep}, Total Reward: {cur_episode_reward:.4f}, Total Steps: {cur_episode_step}, Avg Loss: {avg_loss:.4f}"
                    )

                # Save training progress
                if ep % self.save_interval == 0:
                    self._save_checkpoint(
                        agent,
                        optimizer,
                        {"episode": ep, "cur_episode_reward": cur_episode_reward, "avg_loss": avg_loss},
                    )

                # Store results for analysis
                train_loss_list.append(avg_loss)
                episode_reward_list.append(cur_episode_reward)

                # update pbar
                pbar_epoch.update(1)
                pbar_epoch.set_postfix(episode=ep, reward=cur_episode_reward, loss=avg_loss)

    def _save_checkpoint(self, agent: BaseAgent, optimizer, metadata):
        """Save the checkpoint"""
        save_dir = self.exp_dir / f"checkpoint_{metadata['episode']}"
        agent.save(save_dir)
        torch.save(optimizer.state_dict(), save_dir / "optimizer.pth")
        torch.save(metadata, save_dir / "metadata.pth")
        self.replay_buffer.save(save_dir)
        Configuration().save_config(save_dir / "config.yaml")

    def _load_checkpoint(self, agent: BaseAgent, optimizer, checkpoint_dir):
        """Load the checkpoint"""
        agent.load(checkpoint_dir)
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pth"))
        metadata = torch.load(checkpoint_dir / "metadata.pth")
        self.replay_buffer.load(checkpoint_dir)
        return metadata
