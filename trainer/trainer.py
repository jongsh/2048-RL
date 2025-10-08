import torch
from abc import ABC, abstractmethod


class Trainer:
    """Trainer for the RL Agent"""

    def __init__(self):
        pass

    def _load_optimizer(self, optimizer_name):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD
        elif optimizer_name.lower() == "adamw":
            return torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers: 'adam', 'sgd'")

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _save_checkpoint(self):
        """Save the checkpoint"""
        pass

    @abstractmethod
    def _load_checkpoint(self):
        """Load the checkpoint"""
        pass
