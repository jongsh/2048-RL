import torch
import os
import math
import random
import os

from configs.config import Configuration
from agents.base_agent import BaseAgent
from models.mlp import MLPValue
from models.resnet import ResNetValue


class DQNAgent(BaseAgent):
    """
    DQN Agent Uses two neural networks (Q-network and target network) to estimate Q-values.
    The Q-network is updated based on the Bellman equation, while the target network is used to stabilize training
    """

    def __init__(self, config: Configuration = Configuration(), **kwargs):
        self.agent_config = config.get_config("agent")
        self.public_config = config.get_config("public")
        assert (
            len(self.agent_config["offline"]["action_logit"]) == self.agent_config["action_space"]
        ), "The length of action_logit must match the action_space"
        assert math.isclose(sum(self.agent_config["offline"]["action_logit"]), 1.0), "The action_logit must sum to 1.0"

        super(DQNAgent, self).__init__()

        # Initialize the Q-network and target network
        self.q_network = self._build_network(config).to(device=self.public_config["device"])
        if self.agent_config["target_network"]["use"]:
            self.target_network = self._build_network(config).to(device=self.public_config["device"])
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network_update_step = self.agent_config["target_network"]["update_step"]
            self.target_network_update_count = 0
            self.target_network_update_method = self.agent_config["target_network"]["update_method"]
            self.target_network_update_method_soft_tau = self.agent_config["target_network"]["update_soft_tau"]
        else:
            self.target_network = None

        # Online or offline training
        self.strategy = self.agent_config["strategy"]
        if self.strategy == "online":
            self.epsilon_max = self.agent_config["online"]["start_epsilon"]
            self.epsilon_min = self.agent_config["online"]["end_epsilon"]
            self.epsilon_decay = self.agent_config["online"]["epsilon_decay"]
            self.epsilon = self.epsilon_max
            self.steps_done = 0
        elif self.strategy == "offline":
            self.sample_action_logit = self.agent_config["offline"]["action_logit"]
        else:
            raise ValueError(f"Invalid strategy type {self.strategy}. Choose 'online' or 'offline'.")

        # Other configurations
        self.action_space = self.agent_config["action_space"]
        self.gamma = self.agent_config["gamma"]
        self.device = self.public_config["device"]

    def _build_network(self, config: Configuration):
        public_config = config.get_config("public")
        if public_config["model"] == "mlp":
            return MLPValue(config)
        elif public_config["model"] == "resnet":
            return ResNetValue(config)

    def get_model(self):
        return self.q_network

    def sample_action(self, state):
        if self.strategy == "offline":
            action = random.choices(
                range(self.action_space),
                weights=self.sample_action_logit,
                k=1,
            )[0]
            return action
        elif self.strategy == "online":
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
            else:
                with torch.no_grad():
                    action = self.q_network(self._torch(state, dtype=torch.int32).unsqueeze(0)).argmax(dim=1).item()
            self.epsilon = (
                self.epsilon_min
                + (self.epsilon_max - self.epsilon_min)
                * (1 + math.cos(math.pi * self.steps_done / self.epsilon_decay))
                / 2
            )
            self.steps_done += 1
            return action

        else:
            raise ValueError(f"Invalid strategy type {self.strategy}.")

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

        return loss.item()

    def save(self, dir_path):
        """Save the agent's model to the specified path"""
        os.makedirs(dir_path, exist_ok=True)
        q_network_path = os.path.join(dir_path, "q_network.pth")
        torch.save(self.q_network.state_dict(), q_network_path)
        if self.target_network is not None:
            target_network_path = os.path.join(dir_path, "target_network.pth")
            torch.save(self.target_network.state_dict(), target_network_path)

    def load(self, dir_path):
        """Load the agent's model from the specified path"""
        q_network_path = os.path.join(dir_path, "q_network.pth")
        self.q_network.load_state_dict(torch.load(q_network_path, map_location=self.device))
        if self.target_network is not None:
            target_network_path = os.path.join(dir_path, "target_network.pth")
            self.target_network.load_state_dict(torch.load(target_network_path, map_location=self.device))


if __name__ == "__main__":
    agent = DQNAgent()
    states = torch.randint(0, 4, (5, 16))  # Example states
    actions = torch.randint(0, 4, (5,))  # Example actions
    rewards = torch.randn(5)  # Example rewards
    next_states = torch.randint(0, 4, (5, 16))
    dones = torch.randint(0, 2, (5,))  # Example done flags
    optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    print(agent.sample_action(states[0]))  # Sample an action
    agent.update(states, actions, rewards, next_states, dones, optimizer, loss_fn)
    print("DQN Agent updated successfully.")
