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

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, activation, bias=False
    ):
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
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )  # B, L, hidden_dim
        return self.out_proj(attn_output), attn_weights
