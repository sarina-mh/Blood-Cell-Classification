"""Convolutional classifier (Part 2)."""
from typing import List

import torch
import torch.nn as nn

from ..config import CNNConfig


class BloodCellCNN(nn.Module):
    """Convolutional classifier. Conv channel list, kernel size, pooling and
    the FC head width are all configurable via CNNConfig. Unlike the MLP,
    the image is never flattened before the convolutional stage, so spatial
    structure is preserved and filters are shared across positions.
    """

    def __init__(self, config: CNNConfig, input_size: int = 64):
        super().__init__()
        self.config = config
        conv_layers: List[nn.Module] = []
        in_ch = config.in_channels
        spatial = input_size
        pad = config.kernel_size // 2
        for out_ch in config.conv_channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=config.kernel_size, padding=pad))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(config.pool_size))
            in_ch = out_ch
            spatial = spatial // config.pool_size
        self.features = nn.Sequential(*conv_layers)
        self.flatten_dim = in_ch * spatial * spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def manual_param_count(self) -> int:
        total = 0
        in_ch = self.config.in_channels
        k = self.config.kernel_size
        for out_ch in self.config.conv_channels:
            total += (in_ch * k * k + 1) * out_ch  # conv weights + bias
            in_ch = out_ch
        total += self.flatten_dim * self.config.fc_hidden + self.config.fc_hidden
        total += self.config.fc_hidden * self.config.num_classes + self.config.num_classes
        return total

    def get_first_conv_layer(self) -> nn.Conv2d:
        """Return the first Conv2d module, used to hook the model for the
        feature-map visualization (Part 4)."""
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No conv layer found in this model")
