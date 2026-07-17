"""Part 3: misclassified examples + their full softmax distribution."""
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..data import BloodCellDataModule
from ..train import ModelTrainer


class ErrorAnalyzer:
    """Finds misclassified test samples for a trained model and plots each
    image next to a bar chart of its full softmax probability distribution.
    """

    def __init__(self, trainer: ModelTrainer, data_module: BloodCellDataModule, class_names: List[str]):
        self.trainer = trainer
        self.dm = data_module
        self.class_names = class_names

    @torch.no_grad()
    def find_misclassified(self, loader: DataLoader, n: int = 5) -> List[Dict[str, Any]]:
        self.trainer.model.eval()
        found = []
        for images, labels in loader:
            outputs = self.trainer.model(images.to(self.trainer.device))
            probs = F.softmax(outputs, dim=1).cpu()
            preds = probs.argmax(dim=1)
            for i in range(images.size(0)):
                if preds[i].item() != labels[i].item():
                    found.append({"image": images[i], "true": labels[i].item(),
                                   "pred": preds[i].item(), "probs": probs[i].numpy()})
                    if len(found) >= n:
                        return found
        return found

    def plot(self, samples: List[Dict[str, Any]], title: str) -> None:
        fig, axes = plt.subplots(len(samples), 2, figsize=(9, 3 * len(samples)))
        for row, s in enumerate(samples):
            img = self.dm.unnormalize(s["image"]).permute(1, 2, 0).clamp(0, 1).numpy()
            ax_img, ax_bar = axes[row, 0], axes[row, 1]
            ax_img.imshow(img)
            ax_img.set_title(f"True: {self.class_names[s['true']]} | Pred: {self.class_names[s['pred']]}")
            ax_img.axis("off")
            ax_bar.bar(self.class_names, s["probs"])
            ax_bar.tick_params(axis="x", rotation=45)
            ax_bar.set_ylim(0, 1)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
