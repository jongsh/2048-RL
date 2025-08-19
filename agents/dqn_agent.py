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

    def __init__(self, config=load_single_config("agent", "dqn"), **kwargs):
        assert (
            len(config["offline"]["action_logit"]) == config["action_space"]
        ), "The length of action_logit must match the action_space"
        assert math.isclose(sum(config["offline"]["action_logit"]), 1.0), "The action_logit must sum to 1.0"

        super(DQNAgent, self).__init__()

        # Initialize the Q-network and target network
        self.q_network = self._build_network(config).to(device=config["device"])
        if config["target_network"]["use"]:
            self.target_network = self._build_network(config).to(device=config["device"])
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network_update_step = config["target_network"]["update_step"]
            self.target_network_update_count = 0
            self.target_network_update_method = config["target_network"]["update_method"]
            self.target_network_update_method_soft_tau = config["target_network"]["update_soft_tau"]
        else:
            self.target_network = None

        # Online or offline training
        self.strategy = config["strategy"]
        if self.strategy == "online":
            self.epsilon_max = config["online"]["start_epsilon"]
            self.epsilon_min = config["online"]["end_epsilon"]
            self.epsilon_decay = config["online"]["epsilon_decay"]
            self.epsilon = self.epsilon_max
            self.steps_done = 0
        elif self.strategy == "offline":
            self.sample_action_logit = config["offline"]["action_logit"]
        else:
            raise ValueError(f"Invalid strategy type {config['strategy']}. Choose 'online' or 'offline'.")

        # Other configurations
        self.action_space = config["action_space"]
        self.gamma = config["gamma"]
        self.device = config["device"]

    def _build_network(self, config):
        if config["model"] == "mlp":
            return MLPValue()
        elif config["model"] == "resnet":
            return ResNetValue()

    def sample_actions(self, states):
        if self.strategy == "offline":
            categorical_dist = dist.Categorical(probs=self.sample_action_logit)
            actions = categorical_dist.sample((len(states),))
            return actions
        elif self.strategy == "online":
            if torch.rand(1).item() < self.epsilon:
                actions = torch.randint(0, self.action_space, (len(states),))
            else:
                with torch.no_grad():
                    actions = self.q_network(self._torch(states, dtype=torch.int32)).argmax(dim=1)
            self.epsilon = (
                self.epsilon_min
                + (self.epsilon_max - self.epsilon_min)
                * (1 + math.cos(math.pi * self.steps_done / self.epsilon_decay))
                / 2
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

    def _update_target_network(self):
        assert self.target_network is not None, "Target network is not initialized."
        if self.target_network_update_method == "hard":
            self.target_network.load_state_dict(self.q_network.state_dict())
        elif self.target_network_update_method == "soft":
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_network_update_method_soft_tau)
                    + param.data * self.target_network_update_method_soft_tau
                )

    def update(self, states, actions, rewards, next_states, dones, optimizer, loss_fn):
        # Convert inputs to tensors
        states = self._torch(states, dtype=torch.int32)
        actions = self._torch(actions, dtype=torch.int64)
        rewards = self._torch(rewards, dtype=torch.float32)
        next_states = self._torch(next_states, dtype=torch.int32)
        dones = self._torch(dones, dtype=torch.int32)

        # Update Q-network
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.target_network is not None:
                next_q_values = self.target_network(next_states)
            else:
                next_q_values = self.q_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_value = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = loss_fn(q_value, target_q_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        if self.target_network is not None:
            self.target_network_update_count += states.size(0)
            if self.target_network_update_count >= self.target_network_update_step:
                self._update_target_network()
                self.target_network_update_count = 0


if __name__ == "__main__":
    agent = DQNAgent()
    states = torch.randint(0, 4, (5, 16))  # Example states
    actions = torch.randint(0, 4, (5,))  # Example actions
    rewards = torch.randn(5)  # Example rewards
    next_states = torch.randint(0, 4, (5, 16))
    dones = torch.randint(0, 2, (5,))  # Example done flags
    optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    agent.update(states, actions, rewards, next_states, dones, optimizer, loss_fn)
    print("DQN Agent updated successfully.")
