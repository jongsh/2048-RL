import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

from configs.config import Configuration
from models.layers import FeedForward, AbsolutePositionalEncoding


class MLPBase(torch.nn.Module):
    """Base class for MLP models"""

    def __init__(self, config: Configuration = None):
        config = config if config else Configuration()
        self.model_config = config["model"]

        super(MLPBase, self).__init__()
        # Embedding layer
        self.embed = nn.Embedding(
            num_embeddings=self.model_config["input_embedding"]["num_embeddings"],
            embedding_dim=self.model_config["input_embedding"]["embedding_dim"],
        )
        # Positional Encoding layer
        self.pos_encoder = AbsolutePositionalEncoding(
            max_len=self.model_config["position_embedding"]["max_len"],
            embed_dim=self.model_config["position_embedding"]["embedding_dim"],
            mode=self.model_config["position_embedding"]["mode"],
            method=self.model_config["position_embedding"]["method"],
        )
        # Calculate input dimension
        if self.model_config["position_embedding"]["method"] == "add":
            input_dim = self.model_config["input_len"] * self.model_config["input_embedding"]["embedding_dim"]
        elif self.model_config["position_embedding"]["method"] == "concat":
            input_dim = self.model_config["input_len"] * (
                self.model_config["input_embedding"]["embedding_dim"]
                + self.model_config["position_embedding"]["embedding_dim"]
            )
        else:
            raise ValueError(f"Unknown position embedding method: {self.model_config['position_embedding']['method']}")

        self.network = FeedForward(
            input_dim=input_dim,
            hidden_dim=self.model_config["feed_forward"]["hidden_dim"],
            output_dim=self.model_config["feed_forward"]["output_dim"],
            num_layers=self.model_config["feed_forward"]["num_layers"],
            activation=self.model_config["feed_forward"]["activation"],
            bias=self.model_config["feed_forward"]["bias"],
        )

    def init_weights(m):
        pass

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input to (B, L)
        x = self.embed(x)  # (B, L, embedding_dim)
        x = self.pos_encoder(x)  # (B, L, D)
        B, _, _ = x.size()
        x = x.view(B, -1)  # (B, L * D)
        return self.network(x)  # (B, output_dim)


class MLPPolicy(MLPBase):
    """MLP model for policy"""

    def __init__(self, config: Configuration = None):
        config = config if config else Configuration()
        super(MLPPolicy, self).__init__(config)

    def forward(self, x, action_mask=None):
        action_logits = super(MLPPolicy, self).forward(x)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class MLPValue(MLPBase):
    """MLP model for value function"""

    def __init__(self, config: Configuration = None):
        config = config if config else Configuration()
        super(MLPValue, self).__init__(config)

    def forward(self, x, action_mask=None):
        value = super(MLPValue, self).forward(x)
        if action_mask is not None:
            value = value.masked_fill(action_mask == 0, -1e9)
        return value


if __name__ == "__main__":
    print("=" * 30, "MLP Policy Network", "=" * 30)
    model = MLPPolicy()
    summary(model, input_size=(8, 16), dtypes=[torch.int])

    print()
    print("=" * 30, "MLP Value Network", "=" * 30)
    model = MLPValue()
    summary(model, input_size=(8, 16), dtypes=[torch.int])
