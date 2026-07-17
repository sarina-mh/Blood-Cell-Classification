#!/usr/bin/env python3
"""Runnable end-to-end example on synthetic data, so you can see the API
work without needing to download a real dataset first.

    python examples/synthetic_demo.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quickgrid import Trainer, GridSearch


def make_synthetic_loaders(n_train=800, n_val=200, n_features=20, n_classes=3, batch_size=32):
    torch.manual_seed(0)
    x_train = torch.randn(n_train, n_features)
    y_train = torch.randint(0, n_classes, (n_train,))
    x_val = torch.randn(n_val, n_features)
    y_val = torch.randint(0, n_classes, (n_val,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(params: dict) -> nn.Module:
    hidden = params.get("hidden", 32)
    return nn.Sequential(
        nn.Linear(20, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 3),
    )


def main() -> None:
    train_loader, val_loader = make_synthetic_loaders()

    grid = GridSearch.expand_grid({
        "hidden": [16, 32, 64],
        "lr": [1e-2, 1e-3],
        "optimizer_name": ["adam", "sgd"],
    })
    search = GridSearch(build_model, train_loader, val_loader, param_grid=grid,
                         probe_epochs=3, log_file="synthetic_grid_log.txt")
    best = search.run()
    best_params = {k: v for k, v in best.items() if k not in ("model", "trainer", "n_params", "val_accuracy")}
    print("Best config from probing:", best_params, f"(val_acc={best['val_accuracy']:.3f})")

    final_model = build_model(best_params)
    trainer = Trainer(final_model, train_loader, val_loader, epochs=10, lr=best_params.get("lr", 1e-3),
                       optimizer_name=best_params.get("optimizer_name", "adam"))
    trainer.fit()
    print(f"Final val accuracy after full training: {trainer.accuracy(val_loader):.3f}")


if __name__ == "__main__":
    main()
