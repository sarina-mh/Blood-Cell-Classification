"""Part 4: first-conv-layer feature maps for the CNN."""
from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..models.cnn import BloodCellCNN


class FeatureMapVisualizer:
    """Registers a forward hook on the CNN's first conv layer and plots
    every output channel's activation map next to the input image."""

    def __init__(self, model: BloodCellCNN, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self._activation = None
        self._hook_handle = None

    def _hook_fn(self, module, inp, out):
        self._activation = out.detach().cpu()

    def register_hook(self) -> None:
        layer = self.model.get_first_conv_layer()
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()

    @torch.no_grad()
    def visualize(self, image: torch.Tensor, unnorm_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.register_hook()
        self.model.eval()
        _ = self.model(image.unsqueeze(0).to(self.device))
        self.remove_hook()

        n_filters = self._activation.shape[1]
        cols = 8
        rows = int(np.ceil(n_filters / cols)) + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))

        orig = unnorm_fn(image).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, 0].imshow(orig)
        axes[0, 0].set_title("Input")
        for c in range(cols):
            axes[0, c].axis("off")

        for f in range(n_filters):
            r, c = divmod(f, cols)
            ax = axes[r + 1, c]
            ax.imshow(self._activation[0, f].numpy(), cmap="viridis")
            ax.axis("off")
        for idx in range(n_filters, (rows - 1) * cols):
            r, c = divmod(idx, cols)
            axes[r + 1, c].axis("off")

        plt.tight_layout()
        plt.show()
