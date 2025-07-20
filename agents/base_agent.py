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
