import torch
import os

from datetime import datetime
from tqdm import tqdm

from agents.base_agent import BaseAgent
from trainer.trainer import Trainer
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.visualize import plot_training_history
from utils.lr_scheduler import WarmupCosineLR
from configs.config import Configuration


class ImitationTrainer(Trainer):
    """Imitation Learning Trainer for the RL Agent"""

    def __init__(self, config: Configuration = Configuration(), **kwargs):
        super(ImitationTrainer, self).__init__(**kwargs)
