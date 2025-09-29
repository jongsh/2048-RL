import torch
import os

from datetime import datetime
from tqdm import tqdm

from data.data_2048 import read_from_file
from agents.base_agent import BaseAgent
from trainer.trainer import Trainer
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.visualize import plot_training_history
from utils.lr_scheduler import WarmupCosineLR
from configs.config import Configuration


class ImitationDataset(torch.utils.data.Dataset):
    """Imitation Learning Dataset"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = torch.tensor(self.data[idx]["state"], dtype=torch.int32)
        action = torch.tensor(self.data[idx]["action"], dtype=torch.int64)
        action_mask = torch.tensor(self.data[idx]["action_mask"], dtype=torch.float32)
        return state, action, action_mask


class ImitationTrainer(Trainer):
    """Imitation Learning Trainer for the RL Agent"""

    def __init__(self, config: Configuration = Configuration(), **kwargs):
        self.train_config = config.get_config("trainer")
        self.public_config = config.get_config("public")
        assert self.train_config["exp_name"], "Experiment name must be provided"
        assert self.train_config["data_file"], "Data file path must be provided for imitation learning"

        super(ImitationTrainer, self).__init__(**kwargs)

        # logger
        self.exp_dir = os.path.join(
            self.train_config["output_dir"], self.train_config["exp_name"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.logger = Logger(self.exp_dir, self.train_config["exp_name"])
        self.logger.info("\n" + config.to_string() + "\n")

        # checkpoint
        config.config["public"]["from_checkpoint"] = self.exp_dir  # update checkpoint path in config

        # for training
        self.optimizer_cls = self._load_optimizer(self.train_config["optimizer"])
        self.optimizer = None  # will be initialized during training
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.batch_size = self.train_config["batch_size"]
        self.episode = self.train_config["train_episode"]
        self.episode_max_step = self.train_config["episode_max_step"]
        self.lr_config = self.train_config["learning_rate"]
        self.device = self.public_config["device"]

        self.log_interval = self.train_config["log_interval"]
        self.save_interval = self.train_config["save_interval"]
        self.from_checkpoint = self.train_config["from_checkpoint"]

        # load data
        self.dataset = ImitationDataset(read_from_file(self.train_config["data_file"]))
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def _load_optimizer(self, optimizer_name):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD
        elif optimizer_name.lower() == "adamw":
            return torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers: 'adam', 'sgd'")

    def train(self, agent: BaseAgent, env, is_resume=False):
        """Train the agent in the environment"""

        if is_resume:  # resume training from a checkpoint
            assert self.from_checkpoint, "Checkpoint path must be provided for resuming training"
            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.lr_config["eta_max"])
            metadata = self._load_checkpoint(agent, optimizer, self.from_checkpoint)
        else:
            if self.from_checkpoint and os.path.exists(self.from_checkpoint):  # load pre-trained model
                self.logger.info(f"Loading pre-trained model from {self.from_checkpoint}")
                agent.load(self.from_checkpoint)

            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.lr_config["eta_max"])
            metadata = {
                "episode": 0,
                "cur_episode_reward": 0,
                "cur_episode_step": 0,
                "avg_loss": 0.0,
                "loss_list": [],
                "reward_list": [],
                "step_list": [],
            }

        # learning rate scheduler
        warmup_steps = int(self.lr_config["warmup_rate"] * self.lr_config["total_steps"])
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.lr_config["total_steps"],
            eta_min=self.lr_config["eta_min"],
            last_epoch=metadata["episode"] - 1,
        )

        # training loop
        with tqdm(total=self.episode, desc="Training Progress") as pbar_epoch:
            # Initialize progress bar
            pbar_epoch.set_postfix(
                episode=metadata["episode"],
                reward=metadata["cur_episode_reward"],
                steps=metadata["cur_episode_step"],
                loss=metadata["avg_loss"],
            )
            pbar_epoch.update(metadata["episode"])

            # starting training
            train_loss_list = metadata.get("loss_list")
            episode_reward_list = metadata.get("reward_list")
            episode_step_list = metadata.get("step_list")

            for ep in range(metadata["episode"] + 1, self.episode + 1):
                cur_episode_reward = 0  # episode reward in the current epoch
                cur_episode_step = 0  # steps in the current episode
                cur_train_batch = 0  # total batch in the current episode
                cur_train_loss = 0.0  # current train loss

                # train the agent
                for batch in self.data_loader:
                    states, actions, action_mask = batch
                    loss = agent.update(states, actions, action_mask, optimizer, self.loss_fn)
                    cur_train_batch += 1
                    cur_train_loss += loss

                # evaluate the agent
                state, info = env.reset()
                action_mask = info["action_mask"]
                done = False
                while not done and cur_episode_step < self.episode_max_step:
                    # Sample action from the agent
                    cur_episode_step += 1
                    action = agent.select_action(state, action_mask=action_mask, method="greedy")
                    state, reward, done, _, info = env.step(action)
                    action_mask = info["action_mask"]
                    cur_episode_reward += reward

                # Update learning rate
                scheduler.step()

                # Store results for analysis
                avg_loss = cur_train_loss / cur_train_batch if cur_train_batch > 0 else 0
                train_loss_list.append(avg_loss)
                episode_reward_list.append(cur_episode_reward)
                episode_step_list.append(cur_episode_step)

                # Save training progress
                if ep % self.save_interval == 0:
                    self._save_checkpoint(
                        agent,
                        optimizer,
                        {
                            "episode": ep,
                            "cur_episode_reward": cur_episode_reward,
                            "avg_loss": avg_loss,
                            "loss_list": train_loss_list,
                            "reward_list": episode_reward_list,
                            "step_list": episode_step_list,
                        },
                    )

                # update pbar
                pbar_epoch.update(1)
                pbar_epoch.set_postfix(
                    reward=cur_episode_reward,
                    steps=cur_episode_step,
                    loss=avg_loss,
                    lr=scheduler.get_last_lr()[0],
                )

                # Log the current episode results
                if ep % self.log_interval == 0 or ep == self.episode:
                    self.logger.info(
                        f"Episode {ep}, Learning Rate: {optimizer.param_groups[0]['lr']:.10f}, Total Reward: {cur_episode_reward:.4f}, Total Steps: {cur_episode_step}, Avg Loss: {avg_loss:.4f}"
                    )

            # save final model
            self.logger.info(
                f"Training finished, saving final model to {self.exp_dir}, total episodes: {self.episode}, total steps: {sum(episode_step_list)}, average reward: {sum(episode_reward_list)/len(episode_reward_list):.4f}"
            )
            self._save_checkpoint(
                agent,
                optimizer,
                {
                    "episode": ep,
                    "cur_episode_reward": cur_episode_reward,
                    "cur_episode_step": cur_episode_step,
                    "avg_loss": avg_loss,
                    "loss_list": train_loss_list,
                    "reward_list": episode_reward_list,
                    "step_list": episode_step_list,
                },
            )

            # visualize training results
            plot_training_history(
                train_loss_list,
                label="Training Loss",
                xlabel="Episode",
                ylabel="Loss",
                title="Training Loss",
                save_path=os.path.join(self.exp_dir, "training_loss.jpg"),
                smooth_type="ma",
                smooth_param=100,
            )
            plot_training_history(
                episode_reward_list,
                label="Episode Reward",
                xlabel="Episode",
                ylabel="Reward",
                title="Episode Reward",
                save_path=os.path.join(self.exp_dir, "episode_reward.jpg"),
                smooth_type="ma",
                smooth_param=100,
            )
            plot_training_history(
                episode_step_list,
                label="Episode Steps",
                xlabel="Episode",
                ylabel="Steps",
                title="Episode Steps History",
                save_path=os.path.join(self.exp_dir, "episode_steps.jpg"),
                smooth_type="ma",
                smooth_param=100,
            )

            self.logger.info("Training completed.")

    def _save_checkpoint(self, agent: BaseAgent, optimizer, metadata):
        """Save the checkpoint"""
        if metadata["episode"] >= self.episode:
            save_dir = os.path.join(self.exp_dir)
        else:
            save_dir = os.path.join(self.exp_dir, f"checkpoint_{metadata['episode']}")
        os.makedirs(save_dir, exist_ok=True)
        agent.save(save_dir)
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))
        torch.save(metadata, os.path.join(save_dir, "metadata.pth"))
        Configuration().save_config(save_dir)

    def _load_checkpoint(self, agent: BaseAgent, optimizer, checkpoint_dir):
        """Load the checkpoint"""
        agent.load(checkpoint_dir)
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pth")))
        metadata = torch.load(os.path.join(checkpoint_dir, "metadata.pth"))
        return metadata
