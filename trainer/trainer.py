import torch
import os

from datetime import datetime
from tqdm import tqdm

from agents.base_agent import BaseAgent
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.visualize import plot_training_history
from utils.lr_scheduler import WarmupCosineLR
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
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.batch_size = self.train_config["batch_size"]
        self.episode = self.train_config["episode"]
        self.episode_max_step = self.train_config["episode_max_step"]
        self.lr_config = self.train_config["learning_rate"]
        self.device = self.public_config["device"]

        self.log_interval = self.train_config["log_interval"]
        self.save_interval = self.train_config["save_interval"]
        self.from_checkpoint = self.public_config["from_checkpoint"]

        self.logger.info("\n" + config.to_string() + "\n")

    def _load_optimizer(self, optimizer_name):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers: 'adam', 'sgd'")

    def train(self, agent: BaseAgent, env, is_resume=False):
        """Train the agent in the environment"""

        if is_resume:  # resume training from a checkpoint
            assert self.from_checkpoint, "Checkpoint path must be provided for resuming training"
            optimizer = self.optimizer_cls(agent.get_model().parameters(), lr=self.lr_config["eta_max"])
            metadata = self._load_checkpoint(agent, optimizer, self.from_checkpoint)
        else:  # start training from scratch
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

                state, _, action_mask = env.reset()
                done = False

                # update replay buffer
                while not done and cur_episode_step < self.episode_max_step:
                    # Sample action from the agent
                    cur_episode_step += 1
                    action = agent.sample_action(state, action_mask)
                    next_state, reward, done, _, action_mask = env.step(action)

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
                    episode=ep,
                    reward=cur_episode_reward,
                    steps=cur_episode_step,
                    loss=avg_loss,
                    lr=scheduler.get_lr()[0],
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
                smooth_param=50,
            )
            plot_training_history(
                episode_reward_list,
                label="Episode Reward",
                xlabel="Episode",
                ylabel="Reward",
                title="Episode Reward",
                save_path=os.path.join(self.exp_dir, "episode_reward.jpg"),
                smooth_type="ma",
                smooth_param=50,
            )
            plot_training_history(
                episode_step_list,
                label="Episode Steps",
                xlabel="Episode",
                ylabel="Steps",
                title="Episode Steps History",
                save_path=os.path.join(self.exp_dir, "episode_steps.jpg"),
                smooth_type="ma",
                smooth_param=50,
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
