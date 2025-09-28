import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

from configs.config import Configuration
from models.layers import ActivationFunction, AbsolutePositionalEncoding


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""

    def __init__(self, channel, stride, padding, activation, kernel_size):
        super(ResidualBlock, self).__init__()

        in_channels = out_channels = channel
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = ActivationFunction(activation)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        out = self.activation(out)
        return out


class ResNetBase(nn.Module):
    """Base class for ResNet models"""

    def __init__(self, config: Configuration = Configuration()):
        self.model_config = config.get_config("model")
        assert (
            self.model_config["input_len"] == self.model_config["input_width"] * self.model_config["input_height"]
        ), f"Input length {self.model_config['input_len']} does not match width {self.model_config['input_width']} * height {self.model_config['input_height']}"

        super(ResNetBase, self).__init__()
        self.input_height = self.model_config["input_height"]
        self.input_width = self.model_config["input_width"]
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
        # Projection layer
        if self.model_config["position_embedding"]["method"] == "add":
            in_channels = self.model_config["input_embedding"]["embedding_dim"]
        else:
            in_channels = (
                self.model_config["input_embedding"]["embedding_dim"]
                + self.model_config["position_embedding"]["embedding_dim"]
            )
        out_channels = self.model_config["residual_block"]["hidden_dim"]
        self.project = nn.Linear(in_channels, out_channels)
        self.activation = ActivationFunction(self.model_config["activation"])
        # Residual blocks
        stride = self.model_config["residual_block"]["stride"]
        padding = self.model_config["residual_block"]["padding"]
        activation = self.model_config["activation"]
        kernel_size = self.model_config["residual_block"]["kernel_size"]

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    channel=out_channels,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    kernel_size=kernel_size,
                )
                for _ in range(self.model_config["residual_block"]["num_blocks"])
            ]
        )
        # Output layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            self.model_config["residual_block"]["hidden_dim"],
            self.model_config["output_dim"],
        )

    def forward(self, x):
        # x: (B, H, W)
        # input embedding
        x = self.embed(x)  # (B, H, W, D)
        B, H, W, D = x.size()
        x = x.view(B, -1, D)  # (B, L, D)
        x = self.pos_encoder(x)  # (B, L, D)
        x = x.view(B, H, W, -1)  # (B, H, W, D)
        x = self.activation(self.project(x))  # (B, H, W, D)

        # residual blocks
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        x = self.residual_blocks(x)  # (B, D, H, W)
        x = self.avg_pool(x)  # (B, D, 1, 1)
        x = x.view(B, -1)  # (B, D)
        x = self.fc(x)  # (B, output_dim)
        return x


class ResNetPolicy(ResNetBase):
    """ResNet model for policy"""

    def __init__(self, config: Configuration = Configuration()):
        super(ResNetPolicy, self).__init__(config)

    def forward(self, x, action_mask=None):
        action_logits = super(ResNetPolicy, self).forward(x)  # (B, output_dim)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class ResNetValue(ResNetBase):
    """ResNet model for value function"""

    def __init__(self, config: Configuration = Configuration()):
        super(ResNetValue, self).__init__(config)

    def forward(self, x, action_mask=None):
        value = super(ResNetValue, self).forward(x)  # (B, output_dim)
        if action_mask is not None:
            value = value.masked_fill(action_mask == 0, -1e9)
        return value


if __name__ == "__main__":
    print("=" * 30, "ResNet Policy Network", "=" * 30)
    model = ResNetPolicy()
    summary(model, input_size=(8, 4, 4), dtypes=[torch.int])

    print()
    print("=" * 30, "ResNet Value Network", "=" * 30)
    model = ResNetValue()
    summary(model, input_size=(8, 4, 4), dtypes=[torch.int])
