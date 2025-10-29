import torch
import os
import math
import random
import json

from configs.config import Configuration
from agents.base_agent import BaseAgent
from models.mlp import MLPPolicy
from models.resnet import ResNetPolicy
from models.transformer import TransformerEncoderPolicy


class ImitationAgent(BaseAgent):
    """
    Imitation Learning Agent used for behavior cloning from expert demonstrations.
    Imitation learning is a type of supervised learning where the agent learns to mimic expert behavior by training on a dataset of state-action pairs collected from expert demonstrations. So this agent is just a wrapper around a neural network model that predicts actions given states.
    """

    def __init__(self, config: Configuration = None, **kwargs):
        config = config if config else Configuration()
        super(ImitationAgent, self).__init__(**kwargs)
        self.agent_config = config["agent"]

        # network
        self.network = self._build_network(config)

        # other configurations
        self.action_space = self.agent_config["action_space"]

    def _build_network(self, config: Configuration = None):
        model_name = config["components"]["model"]
        if model_name == "mlp":
            return MLPPolicy(config)
        elif model_name == "resnet":
            return ResNetPolicy(config)
        elif model_name == "transformer":
            return TransformerEncoderPolicy(config)

    def to(self, device):
        self.network.to(device)

    def get_model(self):
        return self.network

    def sample_action(self):
        """Imitation learning does not support action sampling"""
        raise NotImplementedError("Imitation learning does not support action sampling")

    def select_action(self, state, action_mask=None, method="greedy"):
        if method == "random":
            return random.randint(0, self.action_space - 1)

        state = self._torch(state, dtype=torch.int32).unsqueeze(0)
        action_mask = self._torch(action_mask, dtype=torch.int32).unsqueeze(0)
        action_logits = self.network(state, action_mask=action_mask)

        if method == "greedy":
            action = torch.argmax(action_logits, dim=1).item()
        elif method == "sample":
            action = random.choices(
                range(self.action_space),
                weights=action_logits.squeeze(0).tolist(),
                k=1,
            )[0]
        return action

    def update(self, states, actions, action_mask, optimizer, loss_fn):
        # convert inputs to tensors
        states = self._torch(states, dtype=torch.int32)
        actions = self._torch(actions, dtype=torch.int64)
        action_mask = self._torch(action_mask, dtype=torch.int32)

        # update network
        actions = actions.view(-1)  # shape (batch_size,)
        action_logits = self.network(states, action_mask=action_mask)
        loss = loss_fn(action_logits, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10)
        optimizer.step()
        return loss.item()

    def save(self, dir_path):
        """Save the agent's model and state to the specified path"""
        os.makedirs(dir_path, exist_ok=True)
        model_path = os.path.join(dir_path, "model.pth")
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load(self, dir_path):
        model_path = os.path.join(dir_path, "model.pth")
        assert os.path.exists(model_path), f"Model path {model_path} does not exist!"
        self.network.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    agent = ImitationAgent()
    states = torch.randint(0, 4, (5, 16))  # Example states
    actions = torch.randint(0, 4, (5,))  # Example actions
    action_mask = torch.ones((5, 4))  # Example action mask (all actions valid)
    optimizer = torch.optim.Adam(agent.get_model().parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    agent.update(states, actions, action_mask, optimizer, loss_fn)
    print("Imitation Agent updated successfully.")
