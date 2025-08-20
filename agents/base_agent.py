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
        """
        Convert input to a PyTorch tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.clone().detach().to(dtype=dtype, device=self.device)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def sample_action(self):
        """
        Sample action given a state.
        """
        pass

    @abstractmethod
    def select_action(self):
        """
        Given a state, select an action.
        Returns the action (int or tensor), and internally may record log-prob or Q-values.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Compute loss and update the policy/value network.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the agent's model to the specified path.
        """
        pass
