"""Reduced-budget CNN with a Global Average Pooling head (Part 7).

BloodCellCNN's flatten-then-linear classifier head accounts for the large
majority of its parameters. Swapping it for Global Average Pooling removes
that term almost entirely, showing how far under the 3.5M parameter budget
a convolutional model can go before accuracy starts to suffer.
"""
from typing import List

import torch
import torch.nn as nn

from ..config import TinyCNNConfig

__all__ = ["TinyCNN", "TinyCNNConfig"]


class TinyCNN(nn.Module):
    def __init__(self, config: TinyCNNConfig):
        super().__init__()
        self.config = config
        conv_layers: List[nn.Module] = []
        in_ch = config.in_channels
        pad = config.kernel_size // 2
        for out_ch in config.conv_channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=config.kernel_size, padding=pad))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(config.pool_size))
            in_ch = out_ch
        self.features = nn.Sequential(*conv_layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_ch, config.num_classes),
        )
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

    def manual_param_count(self) -> int:
        total = 0
        in_ch = self.config.in_channels
        k = self.config.kernel_size
        for out_ch in self.config.conv_channels:
            total += (in_ch * k * k + 1) * out_ch
            in_ch = out_ch
        total += in_ch * self.config.num_classes + self.config.num_classes
        return total
