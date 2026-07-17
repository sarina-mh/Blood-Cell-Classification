#!/usr/bin/env python3
"""Train one model end to end: grid search over a small config space, then a
full retrain of the winner.

This replaces the notebook's approach of relying on variables left over in
global scope (dm, train_loader, final_mlp, ...). Everything a run needs is
built and passed explicitly here, which also happens to be what fixes the
worker-crash-on-import issue below.

Examples
--------
    python scripts/run_experiment.py --model mlp
    python scripts/run_experiment.py --model cnn --data-root data/raw --epochs 80
    python scripts/run_experiment.py --model tiny_cnn --quick-epochs 5
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bloodcell.config import DataConfig, TrainConfig, MLPConfig, CNNConfig, TinyCNNConfig
from bloodcell.data import BloodCellDataModule
from bloodcell.models import BloodCellMLP, BloodCellCNN, TinyCNN
from bloodcell.train import ModelTrainer, count_trainable_params
from bloodcell.search import GridSearchRunner


def build_mlp_grid(input_dim: int, num_classes: int):
    def builder(params: Dict[str, Any]):
        cfg = MLPConfig(input_dim=input_dim, hidden_dims=params.get("hidden_dims", [512, 256, 128]),
                         num_classes=num_classes, dropout=params.get("dropout", 0.3))
        return BloodCellMLP(cfg)

    grid = GridSearchRunner.expand_grid({
        "hidden_dims": [[512, 256, 128], [1024, 256], [512, 256], [1024, 512, 128]],
        "dropout": [0.2, 0.4],
        "lr": [1e-3, 5e-4],
        "optimizer_name": ["adam", "sgd"],
    })
    return builder, grid


def build_cnn_grid(image_size: int, num_classes: int):
    def builder(params: Dict[str, Any]):
        cfg = CNNConfig(in_channels=3, conv_channels=params.get("conv_channels", [16, 32, 64]),
                         kernel_size=params.get("kernel_size", 3), pool_size=2,
                         fc_hidden=params.get("fc_hidden", 128), num_classes=num_classes,
                         dropout=params.get("dropout", 0.3))
        return BloodCellCNN(cfg, input_size=image_size)

    grid = GridSearchRunner.expand_grid({
        "conv_channels": [[16, 32, 64], [32, 64, 128], [16, 32, 64, 128]],
        "kernel_size": [3, 5],
        "fc_hidden": [128, 256],
        "dropout": [0.2, 0.4],
        "lr": [1e-3],
        "optimizer_name": ["adam"],
    })
    return builder, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "cnn", "tiny_cnn"], required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--epochs", type=int, default=80, help="Full training epochs for the final model")
    parser.add_argument("--quick-epochs", type=int, default=10, help="Probing epochs per grid search candidate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader worker processes. Keep at 0 unless you have "
                              "confirmed multiprocessing works in your environment "
                              "(see README: 'Notes on this refactor').")
    parser.add_argument("--max-params", type=int, default=3_500_000)
    parser.add_argument("--log-dir", default="results")
    parser.add_argument("--save-checkpoint", default=None,
                         help="Optional path to save the final model's state_dict, "
                              "e.g. results/checkpoints/cnn.pt (used by run_interpretability.py)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_config = DataConfig(data_root=args.data_root, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    dm = BloodCellDataModule(data_config)
    dm.setup()
    train_loader, test_loader = dm.get_loaders()
    num_classes = len(dm.classes)
    input_dim = data_config.image_size * data_config.image_size * 3

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if args.model == "mlp":
        builder, grid = build_mlp_grid(input_dim, num_classes)
        log_file = str(log_path / "mlp_grid_search_log.txt")
    elif args.model == "cnn":
        builder, grid = build_cnn_grid(data_config.image_size, num_classes)
        log_file = str(log_path / "cnn_grid_search_log.txt")
    else:
        tiny_cfg = TinyCNNConfig(num_classes=num_classes)
        final_model = TinyCNN(tiny_cfg)
        print(f"TinyCNN parameters: {count_trainable_params(final_model):,}")
        train_cfg = TrainConfig(epochs=args.epochs, max_params=args.max_params)
        trainer = ModelTrainer(final_model, train_loader, test_loader, train_cfg, device=device)
        trainer.fit()
        print(f"Final TinyCNN test accuracy: {trainer.accuracy(test_loader):.4f}")
        trainer.report(test_loader, dm.classes)
        if args.save_checkpoint:
            Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_model.state_dict(), args.save_checkpoint)
            print(f"Saved checkpoint to {args.save_checkpoint}")
        return

    search = GridSearchRunner(model_builder=builder, data_module=dm, param_grid=grid,
                               max_params=args.max_params, quick_epochs=args.quick_epochs,
                               log_file=log_file)
    best = search.run()
    best_params = {k: v for k, v in best.items() if k not in ("model", "trainer", "n_params", "test_accuracy")}

    final_model = builder(best_params)
    train_cfg = TrainConfig(epochs=args.epochs, lr=best_params.get("lr", 1e-3),
                             optimizer_name=best_params.get("optimizer_name", "adam"),
                             max_params=args.max_params)
    trainer = ModelTrainer(final_model, train_loader, test_loader, train_cfg, device=device)
    trainer.fit()
    print(f"Final {args.model.upper()} test accuracy: {trainer.accuracy(test_loader):.4f}")
    trainer.report(test_loader, dm.classes)
    if args.save_checkpoint:
        Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")


if __name__ == "__main__":
    # Guarding the entry point like this matters as soon as num_workers > 0:
    # PyTorch's DataLoader spawns worker processes, and on platforms that use
    # the "spawn" start method (Windows, and macOS by default) that re-imports
    # this module in each worker. Without the __main__ guard, that re-import
    # re-triggers argument parsing and the whole training run inside every
    # worker, which is exactly the kind of hang/crash this project's original
    # log turned up.
    main()
