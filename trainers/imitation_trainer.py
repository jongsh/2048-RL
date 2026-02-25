import torch
import os
import numpy as np

from datetime import datetime
from tqdm import tqdm

from data.data_2048 import read_preprocess_2048_data
from agents.base_agent import BaseAgent
from trainers.trainer import Trainer
from utils.logger import Logger
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
        action_mask = torch.tensor(self.data[idx]["action_mask"], dtype=torch.int32)
        return state, action, action_mask


class ImitationTrainer(Trainer):
    """Imitation Learning Trainer for the RL Agent"""

    def __init__(self, config: Configuration = None, **kwargs):
        config = config if config else Configuration()
        self.train_config = config["trainer"]
        assert self.train_config["exp_name"], "Experiment name must be provided"
        assert self.train_config["data_files"], "Data file path must be provided for imitation learning"

        super(ImitationTrainer, self).__init__(**kwargs)

        # for training
        self.from_checkpoint = self.train_config["from_checkpoint"]
        self.optimizer_cls = self._load_optimizer(self.train_config["optimizer"])
        self.optimizer = None  # will be initialized during training
        self.loss_fn = torch.nn.CrossEntropyLoss(reduce="none")
        self.batch_size = self.train_config["batch_size"]
        self.epoch = self.train_config["train_epoch"]
        self.lr_config = self.train_config["learning_rate"]
        self.device = config["device"]
        self.log_interval = self.train_config["log_interval"]
        self.save_interval = self.train_config["save_interval"]

        # load data
        data_list = []
        for data_file in self.train_config["data_files"]:
            data_list.extend(
                read_preprocess_2048_data(data_file, threshold_steps=int(self.train_config["data_threshold_steps"]))
            )
        np.random.shuffle(data_list)
        self.dataset = ImitationDataset(data_list)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # logger
        self.exp_dir = os.path.join(
            self.train_config["output_dir"], self.train_config["exp_name"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.logger = Logger(self.exp_dir, self.train_config["exp_name"])
        self.logger.info("\n" + config.to_string() + "\n")
        self.logger.info(f"Training data size: {len(self.dataset)}")

    def train(self, agent: BaseAgent, env, is_resume=False):
        """Train the agent in the environment"""
        agent.to(self.device)

        # resume training from a checkpoint
        if is_resume:
            assert self.from_checkpoint, "Checkpoint path must be provided for resuming training"
            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.lr_config["eta_max"])
            metadata = self._load_checkpoint(agent, optimizer, self.from_checkpoint)

        # start training from scratch
        else:
            if self.from_checkpoint and os.path.exists(self.from_checkpoint):  # load pre-trained model
                self.logger.info(f"Loading pre-trained model from {self.from_checkpoint}")
                agent.load(self.from_checkpoint)

            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.lr_config["eta_max"])
            metadata = {
                "epoch": 0,
                "cur_epoch_reward": 0,
                "cur_epoch_step": 0,
                "cur_epoch_avg_loss": 0.0,
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
            last_epoch=metadata["epoch"] - 1,
        )

        # training loop
        with tqdm(total=self.epoch, desc="Training Progress") as pbar_epoch:
            # Initialize progress bar
            pbar_epoch.set_postfix(
                episode=metadata["epoch"],
                reward=metadata["cur_epoch_reward"],
                steps=metadata["cur_epoch_step"],
                loss=metadata["cur_epoch_avg_loss"],
            )
            pbar_epoch.update(metadata["epoch"])

            # starting training
            train_loss_list = metadata.get("loss_list")
            episode_reward_list = metadata.get("reward_list")
            episode_step_list = metadata.get("step_list")
            for ep in range(metadata["epoch"] + 1, self.epoch + 1):
                cur_epoch_reward = 0  # epoch reward in the current epoch
                cur_epoch_step = 0  # steps in the current epoch
                cur_train_batch = 0  # total batch in the current epoch
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
                while not done:
                    # Sample action from the agent
                    cur_epoch_step += 1
                    action = agent.select_action(state, action_mask=action_mask, method="greedy")
                    state, reward, done, _, info = env.step(action)
                    action_mask = info["action_mask"]
                    cur_epoch_reward += reward

                # Update learning rate
                scheduler.step()

                # Store results for analysis
                cur_epoch_avg_loss = cur_train_loss / cur_train_batch if cur_train_batch > 0 else 0
                train_loss_list.append(cur_epoch_avg_loss)
                episode_reward_list.append(cur_epoch_reward)
                episode_step_list.append(cur_epoch_step)

                # Save training progress
                if ep % self.save_interval == 0 or ep == self.epoch:
                    self._save_checkpoint(
                        agent,
                        optimizer,
                        {
                            "episode": ep,
                            "cur_epoch_reward": cur_epoch_reward,
                            "cur_epoch_step": cur_epoch_step,
                            "cur_epoch_avg_loss": cur_epoch_avg_loss,
                            "loss_list": train_loss_list,
                            "reward_list": episode_reward_list,
                            "step_list": episode_step_list,
                        },
                    )

                # update pbar
                pbar_epoch.update(1)
                pbar_epoch.set_postfix(
                    reward=cur_epoch_reward,
                    steps=cur_epoch_step,
                    loss=cur_epoch_avg_loss,
                    lr=scheduler.get_last_lr()[0],
                )

                # Log the current episode results
                if ep % self.log_interval == 0 or ep == self.epoch:
                    self.logger.info(
                        f"Epoch {ep}: Learning Rate: {optimizer.param_groups[0]['lr']:.10f}, Total Reward: {cur_epoch_reward:.4f}, Total Steps: {cur_epoch_step}, Avg Loss: {cur_epoch_avg_loss:.4f}"
                    )

        # finish training
        self.logger.info(
            f"Training finished, saving final model to {self.exp_dir}, total episodes: {self.epoch}, total steps: {sum(episode_step_list)}, average reward: {sum(episode_reward_list)/len(episode_reward_list):.4f}"
        )

        # visualize training results
        plot_training_history(
            train_loss_list,
            label="Training Loss",
            xlabel="Epoch",
            ylabel="Loss",
            title="Training Loss",
            save_path=os.path.join(self.exp_dir, "training_loss.jpg"),
            smooth_type="ma",
            smooth_param=100,
        )
        plot_training_history(
            episode_reward_list,
            label="Episode Reward",
            xlabel="Epoch",
            ylabel="Reward",
            title="Episode Reward",
            save_path=os.path.join(self.exp_dir, "episode_reward.jpg"),
            smooth_type="ma",
            smooth_param=100,
        )
        plot_training_history(
            episode_step_list,
            label="Episode Steps",
            xlabel="Epoch",
            ylabel="Steps",
            title="Episode Steps History",
            save_path=os.path.join(self.exp_dir, "episode_steps.jpg"),
            smooth_type="ma",
            smooth_param=100,
        )

        self.logger.info("Training completed.")

    def _save_checkpoint(self, agent: BaseAgent, optimizer: torch.optim.Optimizer, metadata):
        """Save the checkpoint"""
        if metadata["episode"] >= self.epoch:
            save_dir = self.exp_dir
        else:
            save_dir = os.path.join(self.exp_dir, f"checkpoint_{metadata['episode']}")
        os.makedirs(save_dir, exist_ok=True)
        agent.save(save_dir)
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))
        torch.save(metadata, os.path.join(save_dir, "metadata.pth"))
        Configuration().save_config(save_dir)

    def _load_checkpoint(self, agent: BaseAgent, optimizer: torch.optim.Optimizer, checkpoint_dir):
        """Load the checkpoint"""
        agent.load(checkpoint_dir, device=self.device)
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pth")))
        metadata = torch.load(os.path.join(checkpoint_dir, "metadata.pth"))
        return metadata
