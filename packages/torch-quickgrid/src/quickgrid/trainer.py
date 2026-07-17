"""A small, dependency-light training/eval loop for classification models.

Framework-agnostic in the sense that it only assumes: an nn.Module, two
DataLoaders yielding (inputs, labels) batches, and cross-entropy loss.
Swap in your own loss/metrics by subclassing and overriding _run_epoch
or predict.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int = 10, lr: float = 1e-3, weight_decay: float = 0.0,
                 optimizer_name: str = "adam", verbose: bool = True,
                 device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.verbose = verbose
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _build_optimizer(self, name: str, lr: float, weight_decay: float):
        name = name.lower()
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer: {name}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total_loss, n_samples = 0.0, 0
        with torch.set_grad_enabled(train):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                n_samples += inputs.size(0)
        return total_loss / max(n_samples, 1)

    def fit(self) -> Dict[str, List[float]]:
        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if self.verbose:
                print(f"Epoch {epoch:03d}/{self.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_labels = [], []
        for inputs, labels in loader:
            outputs = self.model(inputs.to(self.device))
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)

    def accuracy(self, loader: DataLoader) -> float:
        preds, labels = self.predict(loader)
        return float((preds == labels).mean())
