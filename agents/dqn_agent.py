import torch
import os
import random
import numpy as np

from configs.config import Configuration
from agents.base_agent import BaseAgent
from models.mlp import MLPValue
from models.resnet import ResNetValue
from models.transformer import TransformerEncoderValue
from utils.normalize import RunningNormalizer


class DQNAgent(BaseAgent):
    """
    DQN Agent Uses two neural networks (Q-network and target network) to estimate Q-values.
    The Q-network is updated based on the Bellman equation, while the target network is used to stabilize training
    """

    def __init__(self, config: Configuration = None, **kwargs):
        config = config if config else Configuration()
        self.agent_config = config["agent"]

        super(DQNAgent, self).__init__()

        # Initialize the Q-network and target network
        self.q_network = self._build_network(config)
        if self.agent_config["target_network"]["use"]:
            self.target_network = self._build_network(config)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network_update_step = self.agent_config["target_network"]["update_step"]
            self.target_network_update_count = 0
            self.target_network_update_method = self.agent_config["target_network"]["update_method"]
            self.target_network_update_method_soft_tau = self.agent_config["target_network"]["update_soft_tau"]
        else:
            self.target_network = None

        # other configurations
        self.action_space = self.agent_config["action_space"]
        self.gamma = self.agent_config["gamma"]
        self.reward_normalizer = RunningNormalizer()
        self.device = torch.device("cpu")

    def _build_network(self, config: Configuration = None):
        model_name = config["components"]["model"]
        if model_name == "mlp":
            return MLPValue(config)
        elif model_name == "resnet":
            return ResNetValue(config)
        elif model_name == "transformer":
            return TransformerEncoderValue(config)

    def to(self, device):
        self.q_network.to(device)
        if self.target_network is not None:
            self.target_network.to(device)
        self.device = device

    def get_model(self):
        return self.q_network

    def sample_action(self, state, action_mask=None, epsilon=0):
        if random.random() < epsilon:  # select a random valid action
            if action_mask is not None:
                valid_actions = [i for i, valid in enumerate(action_mask) if valid]
                action = random.choice(valid_actions)
            else:
                action = random.randint(0, self.action_space - 1)
        else:  # sample the action according to the Q-values
            with torch.no_grad():
                state = self._torch(state, dtype=torch.int32).unsqueeze(0)
                action_mask = (
                    self._torch(action_mask, dtype=torch.int32).unsqueeze(0) if action_mask is not None else None
                )
                action_prob = torch.softmax(self.q_network(state, action_mask), dim=1)
                action_dist = torch.distributions.Categorical(action_prob)
                action = action_dist.sample().item()
        return action

    def select_action(self, state, action_mask=None, method="greedy"):
        assert method in [
            "random",
            "greedy",
            "sample",
        ], "Invalid action selection method, must be 'random', 'greedy' or 'sample'"

        if method == "random":
            return random.randint(0, self.action_space - 1)
        state = self._torch(state, dtype=torch.int32).unsqueeze(0)
        action_mask = self._torch(action_mask, dtype=torch.int32).unsqueeze(0) if action_mask is not None else None
        q_values = self.q_network(state, action_mask)
        if method == "greedy":
            action = q_values.argmax(dim=1).item()
        elif method == "sample":
            action = random.choices(
                range(self.action_space),
                weights=torch.softmax(q_values, dim=1).squeeze(0).cpu().detach().numpy(),
                k=1,
            )[0]

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

    def update(self, states, actions, rewards, next_states, dones, action_mask, optimizer, loss_fn):
        # Convert inputs to tensors
        self.reward_normalizer.update(rewards)
        rewards = self.reward_normalizer.normalize(rewards)
        states = self._torch(states, dtype=torch.int32)
        actions = self._torch(actions, dtype=torch.int64)
        rewards = self._torch(rewards, dtype=torch.float32)
        next_states = self._torch(next_states, dtype=torch.int32)
        dones = self._torch(dones, dtype=torch.int32)
        action_mask = self._torch(action_mask, dtype=torch.int32)

        # Update Q-network
        q_values = self.q_network(states, action_mask)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.target_network is not None:
                next_q_values = self.target_network(next_states, action_mask)
            else:
                next_q_values = self.q_network(next_states, action_mask)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_value = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = loss_fn(q_value, target_q_value)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        optimizer.step()

        # Update target network
        if self.target_network is not None:
            self.target_network_update_count += states.size(0)
            if self.target_network_update_count >= self.target_network_update_step:
                self._update_target_network()
                self.target_network_update_count = 0

        return loss.item()

    def save(self, dir_path):
        """Save the agent's model and state to the specified path"""
        # save model
        os.makedirs(dir_path, exist_ok=True)
        q_network_path = os.path.join(dir_path, "q_network.pth")
        torch.save(self.q_network.state_dict(), q_network_path)
        if self.target_network is not None:
            target_network_path = os.path.join(dir_path, "target_network.pth")
            torch.save(self.target_network.state_dict(), target_network_path)

    def load(self, dir_path, device=None):
        """Load the agent's model from the specified path"""
        # load model
        q_network_path = os.path.join(dir_path, "q_network.pth")
        self.q_network.load_state_dict(torch.load(q_network_path, map_location=device, weights_only=True))
        if self.target_network is not None:
            target_network_path = os.path.join(dir_path, "target_network.pth")
            self.target_network.load_state_dict(torch.load(target_network_path, map_location=device, weights_only=True))


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
