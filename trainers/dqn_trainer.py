import torch
import os

from datetime import datetime
from tqdm import tqdm

from agents.base_agent import BaseAgent
from trainers.trainer import Trainer
from utils.logger import Logger
from utils.replay_buffer import PrioritizedReplayBuffer
from utils.visualize import plot_training_history
from utils.lr_scheduler import WarmupCosineLR
from configs.config import Configuration
from data.data_2048 import read_preprocess_2048_data


class DQNTrainer(Trainer):
    """DQN Trainer for the RL Agent"""

    def __init__(self, config: Configuration = None, **kwargs):
        # load training configuration
        config = config if config else Configuration()
        self.train_config = config["trainer"]
        assert self.train_config["exp_name"], "Experiment name must be provided"
        super(DQNTrainer, self).__init__(**kwargs)

        # setup experiment directory and logger
        self.exp_dir = os.path.join(
            self.train_config["output_dir"], self.train_config["exp_name"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.logger = Logger(self.exp_dir, log_name="training_log")

        # prioritized replay buffer
        self.replay_buffer_config = self.train_config["buffer"]
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.replay_buffer_config["size"],
            min_capacity=self.replay_buffer_config["min_size"],
            alpha=self.replay_buffer_config["per_alpha"],
        )

        # learn mode parameters
        self.strategy = self.train_config["strategy"]
        if self.strategy == "online":
            self.epsilon_max = self.train_config["online"]["start_epsilon"]
            self.epsilon_min = self.train_config["online"]["end_epsilon"]
            self.epsilon_decay = self.train_config["online"]["epsilon_decay"]
            self.epsilon = self.epsilon_max

        # load offline data
        elif self.strategy == "offline":
            data_files = self.train_config["offline"]["data_files"]
            data_list = []
            for data_file in data_files:
                data_list.extend(read_preprocess_2048_data(data_file))
            # create replay buffer from offline data
            self.replay_buffer.from_data_list(data_list)

        # invalid strategy
        else:
            raise ValueError(f"Invalid strategy type {self.train_config['strategy']}. Choose 'online' or 'offline'.")

        # training parameters
        self.episode = self.train_config["train_episode"]
        self.episode_max_step = self.train_config["episode_max_step"]
        self.batch_size = self.train_config["batch_size"]
        self.lr_config = self.train_config["learning_rate"]
        self.optimizer_cls = self._load_optimizer(self.train_config["optimizer"])
        self.optimizer = None  # will be initialized during training
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")
        self.network_update_interval = self.train_config["network_update_interval"]
        self.from_checkpoint = self.train_config["from_checkpoint"]
        self.device = config["device"]
        self.log_interval = self.train_config["log_interval"]
        self.save_interval = self.train_config["save_interval"]

        # log the training configuration
        self.logger.info("\n" + config.to_string() + "\n")

    def train(self, agent: BaseAgent, env, is_resume=False):
        """Train the agent in the environment"""
        agent.to(self.device)

        # resume training from a checkpoint if specified
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
                "episode": 0,
                "cur_episode_reward": 0,
                "cur_episode_step": 0,
                "cur_episode_max_tile": 0,
                "cur_episode_loss": 0.0,
                "loss_list": [],
                "reward_list": [],
                "step_list": [],
                "max_tile_list": [],
            }

        warmup_steps = int(self.lr_config["warmup_rate"] * self.lr_config["total_steps"])
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.lr_config["total_steps"],
            eta_min=self.lr_config["eta_min"],
            last_epoch=metadata["episode"] - 1,
        )

        # initialize replay buffer
        self._initialize_replay_buffer(env, agent)

        # training loop
        with tqdm(total=self.episode, desc="Training Progress") as pbar_epoch:
            # Initialize progress bar
            pbar_epoch.set_postfix(
                reward=metadata["cur_episode_reward"],
                steps=metadata["cur_episode_step"],
                # loss=metadata["avg_loss"],
                max_tile=metadata["cur_episode_max_tile"],
            )
            pbar_epoch.update(metadata["episode"])

            # starting training
            episode_train_loss_list = metadata.get("loss_list", [])
            episode_reward_list = metadata.get("reward_list", [])
            episode_step_list = metadata.get("step_list", [])
            episode_max_tile_list = metadata.get("max_tile_list", [])
            for ep in range(metadata["episode"] + 1, self.episode + 1):
                cur_episode_max_tile = 0  # current max tile
                cur_episode_reward = 0  # episode reward in the current epoch
                cur_episode_step = 0  # steps in the current episode
                cur_train_batch = 0  # total batch in the current episode
                cur_train_loss = 0.0  # current train loss

                # update agent from online interaction data
                if self.strategy == "online":
                    self.epsilon = max(
                        self.epsilon_min,
                        self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (ep - 1) / self.epsilon_decay,
                    )

                    # interact with the environment to collect data and update agent
                    state, info = env.reset()
                    action_mask = info["action_mask"]
                    done = False
                    while not done and cur_episode_step < self.episode_max_step:
                        # Sample action from the agent
                        cur_episode_step += 1
                        action = agent.sample_action(state, action_mask, self.epsilon)
                        next_state, reward, done, _, info = env.step(action)
                        next_action_mask = info["action_mask"]

                        # Add experience to the replay buffer
                        if ep % self.replay_buffer_config["update_interval"] == 0:
                            self.replay_buffer.add(
                                state, action, action_mask, reward, next_state, next_action_mask, done
                            )

                        # Update the agent with a batch from the replay buffer
                        if cur_episode_step % self.network_update_interval == 0:
                            beta = min(
                                1.0,
                                self.replay_buffer_config["per_beta_start"]
                                + (1.0 - self.replay_buffer_config["per_beta_start"])
                                * (ep - 1)
                                / self.replay_buffer_config["per_beta_episode"],
                            )
                            batch = self.replay_buffer.sample(
                                self.batch_size, beta=beta, use_per=self.replay_buffer_config["use_per"]
                            )
                            assert batch is not None, "Replay buffer does not have enough samples for training"
                            *transitions, idxs = batch
                            loss, td_errors = agent.update(*transitions, optimizer, self.loss_fn)
                            # Update priorities in the replay buffer based on TD errors
                            if self.replay_buffer_config["use_per"]:
                                self.replay_buffer.update_priorities(idxs, td_errors)
                            cur_train_batch += 1
                            cur_train_loss += loss

                        # Move to the next state
                        state = next_state
                        action_mask = next_action_mask
                        cur_episode_reward += reward

                    cur_episode_max_tile = info["max_tile"]

                # update agent from offline data
                elif self.strategy == "offline":
                    # Update the agent with batches from the replay buffer until all samples are used
                    for _ in range(len(self.replay_buffer) // self.batch_size):
                        batch = self.replay_buffer.sample(self.batch_size, use_per=False)
                        assert batch is not None, "Replay buffer does not have enough samples for training"
                        *transitions, idxs = batch
                        loss, _ = agent.update(*transitions, optimizer, self.loss_fn)
                        cur_train_batch += 1
                        cur_train_loss += loss

                    # Evaluate the agent
                    state, info = env.reset()
                    action_mask = info["action_mask"]
                    done = False
                    while not done and cur_episode_step < self.episode_max_step:
                        cur_episode_step += 1
                        action = agent.select_action(state, action_mask, method="greedy")
                        next_state, reward, done, _, info = env.step(action)
                        state = next_state
                        action_mask = info["action_mask"]
                        cur_episode_reward += reward
                    cur_episode_max_tile = info["max_tile"]

                # Update learning rate
                scheduler.step()

                # Store results for analysis
                cur_episode_loss = cur_train_loss / cur_train_batch if cur_train_batch > 0 else 0
                episode_train_loss_list.append(cur_episode_loss)
                episode_reward_list.append(cur_episode_reward)
                episode_step_list.append(cur_episode_step)
                episode_max_tile_list.append(cur_episode_max_tile)

                # Save training progress
                if ep % self.save_interval == 0:
                    self._save_checkpoint(
                        agent,
                        optimizer,
                        {
                            "episode": ep,
                            "cur_episode_reward": cur_episode_reward,
                            "cur_episode_step": cur_episode_step,
                            "cur_episode_max_tile": cur_episode_max_tile,
                            "cur_episode_loss": cur_episode_loss,
                            "loss_list": episode_train_loss_list,
                            "reward_list": episode_reward_list,
                            "step_list": episode_step_list,
                            "max_tile_list": episode_max_tile_list,
                        },
                    )

                # update pbar
                pbar_epoch.update(1)
                pbar_epoch.set_postfix(
                    reward=cur_episode_reward,
                    steps=cur_episode_step,
                    # loss=avg_loss,
                    lr=scheduler.get_last_lr()[0],
                    max_tile=cur_episode_max_tile,
                )

                # Log the current episode results
                if ep % self.log_interval == 0 or ep == self.episode:
                    self.logger.info(
                        f"Episode {ep}, Learning Rate: {optimizer.param_groups[0]['lr']:.10f}, Reward/Avg Reward: {cur_episode_reward:.4f}/{sum(episode_reward_list[-self.log_interval:])/self.log_interval:.4f}, Steps/Avg Steps: {cur_episode_step}/{sum(episode_step_list[-self.log_interval:])/self.log_interval:.4f}, Loss/Avg Loss: {cur_episode_loss:.4f}/{sum(episode_train_loss_list[-self.log_interval:])/self.log_interval:.4f}, Max Tile/Avg Max Tile: {cur_episode_max_tile}/{sum(episode_max_tile_list[-self.log_interval:])/self.log_interval:.4f}"
                    )

        # save final model
        self.logger.info(
            f"Training finished, saving final model to {self.exp_dir}, total episodes: {self.episode}, total steps: {sum(episode_step_list)}, average reward: {sum(episode_reward_list)/len(episode_reward_list):.4f}"
        )
        self._save_checkpoint(
            agent,
            optimizer,
            {
                "episode": self.episode,
                "cur_episode_reward": cur_episode_reward,
                "cur_episode_step": cur_episode_step,
                "cur_episode_max_tile": cur_episode_max_tile,
                "cur_episode_loss": cur_episode_loss,
                "loss_list": episode_train_loss_list,
                "reward_list": episode_reward_list,
                "step_list": episode_step_list,
                "max_tile_list": episode_max_tile_list,
            },
        )

        # visualize training results
        plot_training_history(
            episode_train_loss_list,
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
        plot_training_history(
            episode_max_tile_list,
            label="Episode Max Tile",
            xlabel="Episode",
            ylabel="Max Tile",
            title="Episode Max Tile History",
            save_path=os.path.join(self.exp_dir, "episode_max_tile.jpg"),
            smooth_type="ma",
            smooth_param=100,
        )
        self.logger.info("Training completed.")

    def _initialize_replay_buffer(self, env, agent: BaseAgent):
        """Initialize the replay buffer"""
        self.logger.info("Initializing replay buffer...")

        while len(self.replay_buffer) < self.replay_buffer.min_capacity:
            state, info = env.reset()
            action_mask = info["action_mask"]
            done = False
            cur_episode_step = 0

            # Interact with the environment
            while not done and cur_episode_step < self.episode_max_step:
                # Sample action from the agent
                cur_episode_step += 1
                action = agent.sample_action(state, action_mask, epsilon=1.0)  # use random policy
                next_state, reward, done, _, info = env.step(action)
                next_action_mask = info["action_mask"]
                # Add experience to the replay buffer
                self.replay_buffer.add(state, action, action_mask, reward, next_state, next_action_mask, done)
                state = next_state
                action_mask = next_action_mask

        self.logger.info(f"Replay buffer initialized with {len(self.replay_buffer)} transitions.")

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
