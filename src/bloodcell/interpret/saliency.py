"""Part 5 (MLP counterpart to feature maps): input-gradient saliency."""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MLPSaliencyVisualizer:
    """The MLP has no internal spatial feature maps, so instead this computes
    the absolute gradient of the predicted class score with respect to every
    input pixel, maximized across color channels."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def compute_saliency(self, image: torch.Tensor) -> np.ndarray:
        self.model.eval()
        img = image.clone().unsqueeze(0).to(self.device)
        img.requires_grad_(True)

        output = self.model(img)
        pred_class = output.argmax(dim=1)
        score = output[0, pred_class]

        self.model.zero_grad()
        score.backward()

        saliency = img.grad.data.abs().squeeze(0)  # (3, H, W)
        saliency, _ = torch.max(saliency, dim=0)    # channel-wise max -> (H, W)
        return saliency.cpu().numpy()

    def visualize(self, image: torch.Tensor, unnorm_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        saliency_map = self.compute_saliency(image)
        orig = unnorm_fn(image).permute(1, 2, 0).clamp(0, 1).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(orig)
        axes[1].imshow(saliency_map, cmap="jet", alpha=0.5)
        axes[1].set_title("MLP Input-Gradient Saliency")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
