import torch
import math

from torch import nn
from torch.nn import functional as F


class ActivationFunction(nn.Module):
    """Activation function module"""

    def __init__(self, activation):
        super(ActivationFunction, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        return self.activation(x)


class FeedForward(nn.Module):
    """A simple feed-forward neural network"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation, bias=False):
        super(FeedForward, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(ActivationFunction(activation))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.network = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, num_heads, embed_dim, hidden_dim, dropout=0.0, bias=False):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**0.5

        self.q_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()  # B, L, D
        # Reshape x to (B, L, num_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (B, num_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention weights and apply softmax
        attn_weights = (q @ k.transpose(-2, -1)) / self.scale  # B, num_heads, L, L

        if attn_mask is not None:
            if attn_mask.dim() == 2:  # (L, L)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:  # (B, L, L)
                attn_mask = attn_mask.unsqueeze(1)
            expanded_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        dropped_attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = dropped_attn_weights @ v  # B, num_heads, L, head_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # B, L, hidden_dim
        return self.out_proj(attn_output), attn_weights


class AbsolutePositionalEncoding(nn.Module):
    """Absolute Positional Encoding module"""

    def __init__(self, max_len, embed_dim, mode="sinusoidal", method="add"):
        super(AbsolutePositionalEncoding, self).__init__()
        self.method = method
        self.mode = mode

        # sinusoidal absolute positional encoding
        if mode == "sinusoidal":
            self.pe = self._build_sinusoidal(max_len, embed_dim)
        # learnable absolute positional encoding
        elif mode == "learnable":
            self.pe = nn.Embedding(max_len, embed_dim)
        # unsupported mode
        else:
            raise ValueError(f"Unsupported positional encoding mode: {mode}, choose 'sinusoidal' or 'learnable'")

    def _build_sinusoidal(self, max_len: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding matrix"""
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # (dim/2,)
        pe = torch.zeros(max_len, dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, dim)

    def init_weights(self):
        if self.mode == "sinusoidal":
            pass
        elif self.mode == "learnable":
            nn.init.normal_(self.pe.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()  # (B, L, D)
        if self.mode == "sinusoidal":
            pe = self.pe[:seq_len, :].unsqueeze(0).to(x.device)  # (1, seq_len, dim)
        else:
            positions = torch.arange(seq_len, device=x.device)
            pe = self.pe(positions).unsqueeze(0)  # (1, seq_len, dim)

        return x + pe if self.method == "add" else torch.cat([x, pe.expand(batch_size, -1, -1)], dim=-1)
