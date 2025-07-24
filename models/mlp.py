import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

from configs.config import load_config
from models.layers import FeedForward


class MLPBase(torch.nn.Module):
    """Base class for MLP models"""

    def __init__(self, config=load_config("mlp")):
        self.config = config
        super(MLPBase, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=config["embedding"]["num_embeddings"],
            embedding_dim=config["embedding"]["embedding_dim"],
        )
        self.network = FeedForward(
            input_dim=config["input"]["input_len"]
            * config["embedding"]["embedding_dim"],
            hidden_dim=config["feed_forward"]["hidden_dim"],
            output_dim=config["feed_forward"]["output_dim"],
            num_layers=config["feed_forward"]["num_layers"],
            activation=config["feed_forward"]["activation"],
            bias=config["feed_forward"]["bias"],
        )

    def forward(self, x):
        # x: (B, L)
        x = self.embed(x)  # (B, L, embedding_dim)
        B, _, _ = x.size()
        x = x.view(B, -1)  # (B, L * embedding_dim)
        return self.network(x)  # (B, output_dim)


class MLPPolicy(MLPBase):
    """MLP model for policy"""

    def __init__(self, config=load_config("mlp")):
        super(MLPPolicy, self).__init__(config)

    def forward(self, x):
        action_logits = super(MLPPolicy, self).forward(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, action_logits


class MLPValue(MLPBase):
    """MLP model for value function"""

    def __init__(self, config=load_config("mlp")):
        super(MLPValue, self).__init__(config)

    def forward(self, x):
        return super(MLPValue, self).forward(x)


if __name__ == "__main__":
    print("=" * 30, "MLP Policy Network", "=" * 30)
    model = MLPPolicy()
    summary(model, input_size=(8, 16), dtypes=[torch.int])

    print()
    print("=" * 30, "MLP Value Network", "=" * 30)
    model = MLPValue()
    summary(model, input_size=(8, 16), dtypes=[torch.int])
