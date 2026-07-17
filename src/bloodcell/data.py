"""Dataset loading, the train/test split, and a helper for eyeballing samples."""
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import DataConfig


class BloodCellDataModule:
    """Wraps an ImageFolder dataset, builds a fixed train/test split, and
    hands out DataLoaders. Kept deliberately separate from any model or
    training code so it can be reused as-is by the MLP, CNN, and TinyCNN runs.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.classes: List[str] = []
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []
        self.raw_dataset: Optional[ImageFolder] = None
        self.norm_dataset: Optional[ImageFolder] = None
        self.train_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None

    def _build_transform(self, normalize: bool) -> transforms.Compose:
        ops = [transforms.Resize((self.config.image_size, self.config.image_size)),
               transforms.ToTensor()]
        if normalize:
            ops.append(transforms.Normalize(self.config.normalize_mean, self.config.normalize_std))
        return transforms.Compose(ops)

    def setup(self) -> None:
        self.raw_dataset = ImageFolder(self.config.data_root, transform=self._build_transform(False))
        self.norm_dataset = ImageFolder(self.config.data_root, transform=self._build_transform(True))
        self.classes = self.raw_dataset.classes

        n_total = len(self.raw_dataset)
        n_test = int(n_total * self.config.test_split)
        indices = list(range(n_total))
        random.Random(self.config.seed).shuffle(indices)
        self.test_indices = indices[:n_test]
        self.train_indices = indices[n_test:]

        self.train_dataset = Subset(self.norm_dataset, self.train_indices)
        self.test_dataset = Subset(self.norm_dataset, self.test_indices)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                                   shuffle=True, num_workers=self.config.num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size,
                                  shuffle=False, num_workers=self.config.num_workers)
        return train_loader, test_loader

    def unnormalize(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """Reverse the Normalize transform, used whenever a normalized tensor
        needs to be displayed as a natural-looking image."""
        mean = torch.tensor(self.config.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(3, 1, 1)
        return tensor_img.cpu() * std + mean


class SampleVisualizer:
    """Grid of a few raw examples per class, mostly a sanity check that the
    ImageFolder picked up the right directories."""

    def __init__(self, data_module: BloodCellDataModule):
        self.dm = data_module

    def show_class_samples(self, n_per_class: int = 5, seed: int = 0) -> None:
        rng = random.Random(seed)
        classes = self.dm.classes
        targets = np.array(self.dm.raw_dataset.targets)
        fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2 * n_per_class, 2 * len(classes)))
        for row, cls_name in enumerate(classes):
            cls_idx = np.where(targets == row)[0].tolist()
            chosen = rng.sample(cls_idx, n_per_class)
            for col, idx in enumerate(chosen):
                img, _ = self.dm.raw_dataset[idx]
                ax = axes[row, col]
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.axis("off")
                if col == 0:
                    ax.set_title(cls_name, loc="left", fontsize=9)
        plt.tight_layout()
        plt.show()
