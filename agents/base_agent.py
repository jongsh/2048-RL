import torch
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base class for all RL agents.
    Any algorithm-specific agent (PPO, DQN, etc.) should inherit from this.
    """

    def __init__(self):
        """
        Initialize the agent.
        This method can be overridden by subclasses to set up specific parameters or networks.
        """
        pass

    def _torch(self, x, dtype):
        """Convert input to a PyTorch tensor."""
        if isinstance(x, torch.Tensor):
            return x.clone().detach().to(dtype=dtype, device=self.device)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    @abstractmethod
    def to(self):
        """Move the agent's model to the specified device (e.g., CPU or GPU)."""
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def sample_action(self):
        """Sample action given a state. Used to add sample new episodes during training."""
        pass

    @abstractmethod
    def select_action(self):
        """Given a state, select an action. Used during evaluation."""
        pass

    @abstractmethod
    def update(self):
        """Compute loss and update the policy/value network."""
        pass

    @abstractmethod
    def save(self):
        """Save the agent's model to the specified path."""
        pass

    @abstractmethod
    def load(self):
        """Load the agent's model from the specified path."""
        pass
