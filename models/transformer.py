import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

from configs.config import Configuration
from models.layers import AbsolutePositionalEncoding, MultiHeadAttention, FeedForward


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(
        self,
        num_heads,
        embed_dim,
        ffn_hidden_dim,
        ffn_num_layers,
        norm_type,
        activation,
        dropout=0.0,
        bias=False,
    ):
        assert norm_type in ["pre", "post"], "norm_type must be either 'pre' or 'post'"
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, embed_dim, dropout, bias)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm_type = norm_type
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim, embed_dim, ffn_num_layers, activation, bias)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if self.norm_type == "pre":
            x = self.norm(x)

        # Multi-head attention
        attn_output, _ = self.attention(x)
        x = x + attn_output
        x = self.norm(x)

        # Feed-forward network
        x = self.ffn(x)
        x = self.ffn_norm(x)

        if self.norm_type == "post":
            x = self.norm(x)

        return x


class TransformerEncoderBase(nn.Module):
    """Base class for Transformer encoder models"""

    def __init__(self, config: Configuration = None):
        config = config if config else Configuration()
        super(TransformerEncoderBase, self).__init__()
        self.model_config = config.get_config("model")
        # token embedding
        self.embed = nn.Embedding(
            num_embeddings=self.model_config["input_embedding"]["num_embeddings"],
            embedding_dim=self.model_config["input_embedding"]["embedding_dim"],
        )
        # positional encoding
        self.pos_encoder = AbsolutePositionalEncoding(
            max_len=self.model_config["position_embedding"]["max_len"],
            embed_dim=self.model_config["position_embedding"]["embedding_dim"],
            mode=self.model_config["position_embedding"]["mode"],
            method=self.model_config["position_embedding"]["method"],
        )
        # transformer blocks
        if self.model_config["position_embedding"]["method"] == "add":
            self.embed_dim = self.model_config["input_embedding"]["embedding_dim"]
        else:
            self.embed_dim = (
                self.model_config["input_embedding"]["embedding_dim"]
                + self.model_config["position_embedding"]["embedding_dim"]
            )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    num_heads=self.model_config["block"]["num_heads"],
                    embed_dim=self.embed_dim,
                    ffn_hidden_dim=self.model_config["block"]["ffn_hidden_dim"],
                    ffn_num_layers=self.model_config["block"]["ffn_num_layers"],
                    norm_type=self.model_config["block"]["norm_type"],
                    activation=self.model_config["block"]["activation"],
                    dropout=self.model_config["block"]["dropout"],
                    bias=False,
                )
                for _ in range(self.model_config["block"]["num_layers"])
            ]
        )
        # output layer
        self.output_layer = nn.Linear(self.embed_dim, self.model_config["output_dim"])

    def forward(self, x):
        # embedding and positional encoding
        x = x.view(x.size(0), -1)  # Flatten the input to (B, L)
        x = self.embed(x)  # (B, L, D)
        x = self.pos_encoder(x)  # (B, L, D)

        # transformer blocks
        for layer in self.layers:
            x = layer(x)  # (B, L, D)

        # global average pooling and output layer
        x = x.mean(dim=1)  # (B, D)
        x = self.output_layer(x)  # (B, output_dim)
        return x


class TransformerEncoderPolicy(TransformerEncoderBase):
    """Transformer encoder model for policy"""

    def __init__(self, config: Configuration = None):
        config = config if config else Configuration()
        super(TransformerEncoderPolicy, self).__init__(config)

    def forward(self, x, action_mask=None):
        action_logits = super(TransformerEncoderPolicy, self).forward(x)  # (B, output_dim)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class TransformerEncoderValue(TransformerEncoderBase):
    """Transformer encoder model for value"""

    def __init__(self, config: Configuration = None):
        config = config if config else Configuration()
        super(TransformerEncoderValue, self).__init__(config)

    def forward(self, x, action_mask=None):
        value = super(TransformerEncoderValue, self).forward(x)  # (B, output_dim)
        if action_mask is not None:
            value = value.masked_fill(action_mask == 0, -1e9)
        return value


if __name__ == "__main__":
    print("=" * 30, "Transformer Policy Network", "=" * 30)
    model = TransformerEncoderPolicy()
    summary(model, input_size=(8, 16), dtypes=[torch.int])

    print()
    print("=" * 30, "Transformer Value Network", "=" * 30)
    model = TransformerEncoderValue()
    summary(model, input_size=(8, 16), dtypes=[torch.int])
