"""Part 6: robustness of each model's predictions to rotation + translation."""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

from ..data import BloodCellDataModule
from ..train import ModelTrainer


class SpatialInvarianceTester:
    """For each class, finds one test sample correctly classified by BOTH
    models, applies a rotation + translation transform, and compares each
    model's softmax output before/after the transform.
    """

    def __init__(self, mlp_trainer: ModelTrainer, cnn_trainer: ModelTrainer,
                 data_module: BloodCellDataModule, class_names: list,
                 degrees: float = 25, translate: Tuple[float, float] = (0.15, 0.15)):
        self.mlp_trainer = mlp_trainer
        self.cnn_trainer = cnn_trainer
        self.dm = data_module
        self.class_names = class_names
        self.geo_transform = transforms.RandomAffine(degrees=degrees, translate=translate)

    @torch.no_grad()
    def _predict_single(self, trainer: ModelTrainer, image: torch.Tensor) -> np.ndarray:
        trainer.model.eval()
        out = trainer.model(image.unsqueeze(0).to(trainer.device))
        return F.softmax(out, dim=1).cpu().numpy()[0]

    def find_agreeing_samples(self) -> Dict[int, Tuple[torch.Tensor, int]]:
        found: Dict[int, Tuple[torch.Tensor, int]] = {}
        for image, label in self.dm.test_dataset:
            if label in found:
                continue
            mlp_pred = int(self._predict_single(self.mlp_trainer, image).argmax())
            cnn_pred = int(self._predict_single(self.cnn_trainer, image).argmax())
            if mlp_pred == label and cnn_pred == label:
                found[label] = (image, label)
            if len(found) == len(self.class_names):
                break
        return found

    def run_and_plot(self) -> None:
        samples = self.find_agreeing_samples()
        for label, (image, _) in samples.items():
            transformed = self.geo_transform(image)

            mlp_probs_o = self._predict_single(self.mlp_trainer, image)
            cnn_probs_o = self._predict_single(self.cnn_trainer, image)
            mlp_probs_t = self._predict_single(self.mlp_trainer, transformed)
            cnn_probs_t = self._predict_single(self.cnn_trainer, transformed)

            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            orig_img = self.dm.unnormalize(image).permute(1, 2, 0).clamp(0, 1).numpy()
            trans_img = self.dm.unnormalize(transformed).permute(1, 2, 0).clamp(0, 1).numpy()

            axes[0, 0].imshow(orig_img); axes[0, 0].set_title(f"Original: {self.class_names[label]}"); axes[0, 0].axis("off")
            axes[0, 1].bar(self.class_names, mlp_probs_o); axes[0, 1].set_title("MLP (original)"); axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 2].bar(self.class_names, cnn_probs_o); axes[0, 2].set_title("CNN (original)"); axes[0, 2].tick_params(axis="x", rotation=45)

            axes[1, 0].imshow(trans_img); axes[1, 0].set_title("Transformed"); axes[1, 0].axis("off")
            axes[1, 1].bar(self.class_names, mlp_probs_t); axes[1, 1].set_title("MLP (transformed)"); axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 2].bar(self.class_names, cnn_probs_t); axes[1, 2].set_title("CNN (transformed)"); axes[1, 2].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()
