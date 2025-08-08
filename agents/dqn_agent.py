import torch
import math

from torch import distributions as dist

from configs.config import load_single_config
from agents.base_agent import BaseAgent
from models import *


class DQNAgent(BaseAgent):
    """
    DQN Agent Uses two neural networks (Q-network and target network) to estimate Q-values.
    The Q-network is updated based on the Bellman equation, while the target network is used to stabilize training
    """

    def __init__(self, config=load_single_config("agent", "dqn"), optimizer=None, **kwargs):
        assert (
            len(config["offline"]["action_logit"]) == config["action_space"]
        ), "The length of action_logit must match the action_space"
        assert math.isclose(sum(config["offline"]["action_logit"]), 1.0), "The action_logit must sum to 1.0"

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
        if config["strategy"] == "online":
            self.epsilon_max = config["online"]["start_epsilon"]
            self.epsilon_min = config["online"]["end_epsilon"]
            self.epsilon_decay = config["online"]["epsilon_decay"]
            self.epsilon = self.epsilon_max
            self.steps_done = 0
        elif config["strategy"] == "offline":
            pass
        else:
            raise ValueError(f"Invalid strategy type {config['strategy']}. Choose 'online' or 'offline'.")

        # Other configurations
        self.gamma = config["gamma"]
        self.optimizer = optimizer
        self.device = config["device"]

    def _build_network(self, config):
        if config["model"] == "mlp":
            return MLPValue()
        elif config["model"] == "resnet":
            return ResNetValue()

    def sample_actions(self, states):
        if self.config["strategy"] == "offline":
            action_logit = torch.tensor(self.config["offline"]["action_logit"])
            categorical_dist = dist.Categorical(probs=action_logit)
            actions = categorical_dist.sample((len(states),))
            return actions
        elif self.config["strategy"] == "online":
            if torch.rand(1).item() < self.epsilon:
                actions = torch.randint(0, self.config["action_space"], (len(states),))
            else:
                with torch.no_grad():
                    actions = self.q_network(self._torch(states, dtype=torch.int32)).argmax(dim=1)
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (self.steps_done / self.epsilon_decay),
            )
            self.steps_done += len(states)
            return actions

        else:
            raise ValueError(f"Invalid strategy type {self.config['strategy']}.")

    def select_action(self, state):
        state = self._torch(state, dtype=torch.int32).unsqueeze(0)
        q_values = self.q_network(state)
        action = q_values.argmax(dim=1).item()
        return action

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.target_network is not None:
                next_q_values = self.target_network(next_states)
            else:
                next_q_values = self.q_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_value = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = torch.nn.functional.mse_loss(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = DQNAgent()
    state = [1] * 16  # Example state for a 4x4 grid
    action = agent.select_action(state)
    print(f"Selected action: {action}")
