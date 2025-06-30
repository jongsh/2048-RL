from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class MutiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, hidden_dim, dropout=0.0, bias=False):
        super(MutiHeadAttention, self).__init__()
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
            attn_weights.masked_fill_(attn_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = attn_weights @ v  # B, num_heads, L, head_dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )  # B, L, hidden_dim
        return self.out_proj(attn_output), attn_weights
