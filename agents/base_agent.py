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
        raise NotImplementedError("The 'to' method must be implemented by the agent subclass")

    @abstractmethod
    def get_model(self):
        """Return the agent's underlying model (e.g., policy network or value network)."""
        raise NotImplementedError("The 'get_model' method must be implemented by the agent subclass")

    @abstractmethod
    def sample_action(self):
        """Sample action given a state. Used to add sample new episodes during training."""
        raise NotImplementedError("The 'sample_action' method must be implemented by the agent subclass")

    @abstractmethod
    def select_action(self):
        """Given a state, select an action. Used during evaluation."""
        raise NotImplementedError("The 'select_action' method must be implemented by the agent subclass")

    @abstractmethod
    def update(self):
        """Compute loss and update the policy/value network."""
        raise NotImplementedError("The 'update' method must be implemented by the agent subclass")

    @abstractmethod
    def save(self):
        """Save the agent's model to the specified path."""
        raise NotImplementedError("The 'save' method must be implemented by the agent subclass")

    @abstractmethod
    def load(self):
        """Load the agent's model from the specified path."""
        raise NotImplementedError("The 'load' method must be implemented by the agent subclass")
