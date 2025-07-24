import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

from configs.config import load_config
from models.layers import FeedForward, ActivationFunction


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""

    def __init__(self, config=load_config("resnet")):
        in_channels = config["residual_block"]["in_channels"]
        out_channels = config["residual_block"]["out_channels"]
        stride = config["residual_block"]["stride"]
        padding = config["residual_block"]["padding"]
        activation = config["residual_block"]["activation"]
        kernel_size = config["residual_block"]["kernel_size"]

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = ActivationFunction(activation)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNetBase(nn.Module):
    """Base class for ResNet models"""

    def __init__(self, config=load_config("resnet")):
        super(ResNetBase, self).__init__()
        self.input_height = config["input"]["input_height"]
        self.input_width = config["input"]["input_width"]
        self.embed = nn.Embedding(
            num_embeddings=config["embedding"]["num_embeddings"],
            embedding_dim=config["embedding"]["embedding_dim"],
        )
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(config)
                for _ in range(config["residual_block"]["num_blocks"])
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            config["residual_block"]["out_channels"],
            config["output"]["output_dim"],
        )

    def forward(self, x):
        # x: (B, H, W)
        x = self.embed(x)  # (B, H, W, D)
        B, H, W, D = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        x = self.residual_blocks(x)  # (B, D, H, W)
        x = self.avg_pool(x)  # (B, D, 1, 1)
        x = x.view(B, -1)
        x = self.fc(x)  # (B, output_dim)
        return x


class ResNetPolicy(ResNetBase):
    """ResNet model for policy"""

    def __init__(self, config=load_config("resnet")):
        super(ResNetPolicy, self).__init__(config)

    def forward(self, x):
        action_logits = super(ResNetPolicy, self).forward(x)  # (B, output_dim)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, action_logits


class ResNetValue(ResNetBase):
    """ResNet model for value function"""

    def __init__(self, config=load_config("resnet")):
        super(ResNetValue, self).__init__(config)

    def forward(self, x):
        return super(ResNetValue, self).forward(x)  # (B, output_dim)


if __name__ == "__main__":
    print("=" * 30, "ResNet Policy Network", "=" * 30)
    model = ResNetPolicy()
    summary(model, input_size=(8, 4, 4), dtypes=[torch.int])

    print()
    print("=" * 30, "ResNet Value Network", "=" * 30)
    model = ResNetValue()
    summary(model, input_size=(8, 4, 4), dtypes=[torch.int])
