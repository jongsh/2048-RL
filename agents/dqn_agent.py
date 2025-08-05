import torch
from configs.config import load_single_config
from agents.base_agent import BaseAgent
from models import *


class DQNAgent(BaseAgent):
    """
    DQN Agent Uses two neural networks (Q-network and target network) to estimate Q-values.
    The Q-network is updated based on the Bellman equation, while the target network is used to stabilize training
    """

    def __init__(
        self, config=load_single_config("agents", "dqn"), optimizer=None, **kwargs
    ):
        super(DQNAgent, self).__init__()
        self.config = config

        # Initialize the Q-network and target network
        self.q_network = self._build_network(config)
        if config["use_target_network"]:
            self.target_network = self._build_network(config)
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            self.target_network = None

        # Online or offline training
        if config["is_online"]:
            self.epsilon = config["epsilon_start"]
            self.epsilon_decay = config["epsilon_decay"]
            self.epsilon_min = config["epsilon_min"]
            self.batch_size = config["batch_size"]
        else:
            pass

        # Other configurations
        self.gamma = config["gamma"]
        self.optimizer = optimizer

    def _build_network(self, config):
        """
        Build the Q-network based on the configuration
        """
        if config["model"] == "mlp":
            return MLPValue()
        elif config["model"] == "resnet":
            return ResNetValue()
