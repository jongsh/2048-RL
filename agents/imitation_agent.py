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

    def __init__(self, config=Configuration(), **kwargs):
        super(ImitationAgent, self).__init__(kwargs)
        self.agent_config = config.get_config("agent")
        self.public_config = config.get_config("public")

        self.network = self._build_network(config)

        # other configurations
        self.action_space = self.agent_config["action_space"]
        self.gamma = self.agent_config["gamma"]
        self.device = self.public_config["device"]

    def _build_network(self, config: Configuration):
        public_config = config.get_config("public")
        if public_config["model"] == "mlp":
            return MLPPolicy(config)
        elif public_config["model"] == "resnet":
            return ResNetPolicy(config)
        elif public_config["model"] == "transformer":
            return TransformerEncoderPolicy(config)

    def get_model(self):
        return self.network

    def sample_action(self):
        """Imitation learning does not support action sampling"""
        raise NotImplementedError("Imitation learning does not support action sampling")

    def select_action(self, state, action_mask=None, method="greedy"):
        if method == "random":
            return random.randint(0, self.action_space - 1)
        state = self._torch(state, dtype=torch.int32).unsqueeze(0)

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

    def update(self, states, actions, rewards, next_states, dones, action_mask, optimizer, loss_fn):
        # Convert inputs to tensors
        states = self._torch(states, dtype=torch.int32)
        actions = self._torch(actions, dtype=torch.int64)
        rewards = self._torch(rewards, dtype=torch.float32)
        next_states = self._torch(next_states, dtype=torch.int32)
        dones = self._torch(dones, dtype=torch.int32)
        action_mask = self._torch(action_mask, dtype=torch.int32)

        # TODO: update network
        actions = actions.view(-1)  # shape (batch_size,)
        action_logits = self.network(states, action_mask=action_mask)

        loss = loss_fn(action_logits, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
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
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))
