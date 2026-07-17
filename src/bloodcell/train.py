"""Generic training / evaluation loop, shared by every model in this project."""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from .config import TrainConfig


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelTrainer:
    """Generic training / evaluation loop that works for any nn.Module +
    DataLoader combination. The MLP, CNN, and TinyCNN in this project are
    all trained through this same class so the comparison between them
    isn't confounded by different training code.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                 train_config: TrainConfig, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = train_config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.history: Dict[str, List[float]] = {"train_loss": [], "test_loss": []}

    def _build_optimizer(self):
        name = self.cfg.optimizer_name.lower()
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer_name}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total_loss, n_samples = 0.0, 0
        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * images.size(0)
                n_samples += images.size(0)
        return total_loss / max(n_samples, 1)

    def fit(self) -> Dict[str, List[float]]:
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            test_loss = self._run_epoch(self.test_loader, train=False)
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            if self.cfg.verbose:
                print(f"Epoch {epoch:03d}/{self.cfg.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            outputs = self.model(images.to(self.device))
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)

    def accuracy(self, loader: DataLoader) -> float:
        preds, labels = self.predict(loader)
        return float((preds == labels).mean())

    def plot_loss_curve(self, title: str = "Loss Curve") -> None:
        plt.figure(figsize=(7, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["test_loss"], label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.show()

    def report(self, loader: DataLoader, class_names: List[str]) -> None:
        preds, labels = self.predict(loader)
        print(classification_report(labels, preds, target_names=class_names))
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix")
        plt.show()
