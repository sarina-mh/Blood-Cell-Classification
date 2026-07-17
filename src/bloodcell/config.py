"""Configuration dataclasses for data loading, training, and each model family.

Centralizing hyperparameters here means the same ModelTrainer and
GridSearchRunner can be reused for the MLP, the CNN, and the reduced-budget
TinyCNN without duplicating any training logic.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    data_root: str = "data/raw"
    image_size: int = 64
    test_split: float = 0.2
    batch_size: int = 64
    num_workers: int = 0  # see README "Notes on this refactor" for why this defaults to 0
    seed: int = 42
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    max_params: int = 3_500_000
    verbose: bool = True


@dataclass
class MLPConfig:
    input_dim: int = 64 * 64 * 3
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 5
    dropout: float = 0.3


@dataclass
class CNNConfig:
    in_channels: int = 3
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 3
    pool_size: int = 2
    fc_hidden: int = 128
    num_classes: int = 5
    dropout: float = 0.3


@dataclass
class TinyCNNConfig:
    in_channels: int = 3
    conv_channels: List[int] = field(default_factory=lambda: [8, 16, 32])
    kernel_size: int = 3
    pool_size: int = 2
    num_classes: int = 5
    dropout: float = 0.2
