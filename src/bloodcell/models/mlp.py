"""Fully connected baseline (Part 1)."""
from typing import List

import torch
import torch.nn as nn

from ..config import MLPConfig


class BloodCellMLP(nn.Module):
    """Fully-connected classifier. Depth/width/dropout are fully driven by
    MLPConfig so the same class can be reused across every grid search.
    Every image is flattened to a 1D vector before it reaches the network,
    so the model has no notion of which pixels are spatially close to which.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        layers: List[nn.Module] = []
        prev_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten image into a feature vector
        return self.net(x)

    def manual_param_count(self) -> int:
        """Independent parameter count, used to cross-check nn.Module's own
        count and confirm the architecture was built as intended."""
        total = 0
        prev_dim = self.config.input_dim
        for h in self.config.hidden_dims:
            total += prev_dim * h + h  # weights + biases
            prev_dim = h
        total += prev_dim * self.config.num_classes + self.config.num_classes
        return total
