"""Part 7: occlusion sensitivity heatmaps."""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..data import BloodCellDataModule
from ..train import ModelTrainer


class OcclusionSensitivityAnalyzer:
    """Slides an occluding patch across the image and records the drop in
    the true class's softmax probability at every position. A bigger drop
    means that region mattered more to the model's decision."""

    def __init__(self, trainer: ModelTrainer, patch_size: int = 8, stride: Optional[int] = None):
        self.trainer = trainer
        self.patch_size = patch_size
        self.stride = stride or patch_size

    @torch.no_grad()
    def compute_heatmap(self, image: torch.Tensor, true_label: int) -> np.ndarray:
        self.trainer.model.eval()
        device = self.trainer.device
        _, h, w = image.shape
        mean_val = image.mean().item()

        base_out = self.trainer.model(image.unsqueeze(0).to(device))
        base_prob = F.softmax(base_out, dim=1)[0, true_label].item()

        n_rows = (h - self.patch_size) // self.stride + 1
        n_cols = (w - self.patch_size) // self.stride + 1
        heatmap = np.zeros((n_rows, n_cols))

        for r in range(n_rows):
            for c in range(n_cols):
                y0, x0 = r * self.stride, c * self.stride
                occluded = image.clone()
                occluded[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size] = mean_val
                out = self.trainer.model(occluded.unsqueeze(0).to(device))
                prob = F.softmax(out, dim=1)[0, true_label].item()
                heatmap[r, c] = base_prob - prob  # probability drop == region importance

        return heatmap


def plot_occlusion_comparison(image: torch.Tensor, true_label: int, class_names: List[str],
                               dm: BloodCellDataModule,
                               mlp_analyzer: OcclusionSensitivityAnalyzer,
                               cnn_analyzer: OcclusionSensitivityAnalyzer) -> None:
    mlp_heatmap = mlp_analyzer.compute_heatmap(image, true_label)
    cnn_heatmap = cnn_analyzer.compute_heatmap(image, true_label)

    orig_img = dm.unnormalize(image).permute(1, 2, 0).clamp(0, 1).numpy()
    img_size = orig_img.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(orig_img); axes[0].set_title(f"Original: {class_names[true_label]}"); axes[0].axis("off")

    axes[1].imshow(orig_img)
    axes[1].imshow(mlp_heatmap, cmap="jet", alpha=0.5, extent=(0, img_size, img_size, 0))
    axes[1].set_title("MLP Occlusion Heatmap"); axes[1].axis("off")

    axes[2].imshow(orig_img)
    axes[2].imshow(cnn_heatmap, cmap="jet", alpha=0.5, extent=(0, img_size, img_size, 0))
    axes[2].set_title("CNN Occlusion Heatmap"); axes[2].axis("off")

    plt.tight_layout()
    plt.show()
