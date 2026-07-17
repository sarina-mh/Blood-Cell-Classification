"""Small architecture / hyperparameter grid search shared by every model.

Each candidate config is trained for a short probing run and logged to a
plain text file; the best probing config is then retrained for the full
epoch budget by the caller.
"""
import itertools
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from .config import TrainConfig
from .data import BloodCellDataModule
from .train import ModelTrainer, count_trainable_params


class GridSearchRunner:
    """Generic grid search over model architecture + training hyperparameters.
    Works with any model_builder function that takes a params dict and
    returns an nn.Module.
    """

    def __init__(self, model_builder: Callable[[Dict[str, Any]], nn.Module],
                 data_module: BloodCellDataModule,
                 param_grid: List[Dict[str, Any]],
                 max_params: int = 3_500_000,
                 min_accuracy: float = 0.0,
                 quick_epochs: Optional[int] = None,
                 log_file: Optional[str] = None,
                 enforce_param_budget: bool = True):
        self.model_builder = model_builder
        self.dm = data_module
        self.param_grid = param_grid
        self.max_params = max_params
        self.min_accuracy = min_accuracy
        self.quick_epochs = quick_epochs
        self.log_file = log_file
        self.enforce_param_budget = enforce_param_budget
        self.results: List[Dict[str, Any]] = []

    @staticmethod
    def expand_grid(options: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = list(options.keys())
        combos = list(itertools.product(*options.values()))
        return [dict(zip(keys, combo)) for combo in combos]

    def run(self) -> Dict[str, Any]:
        train_loader, test_loader = self.dm.get_loaders()
        best = None
        log_lines: List[str] = []

        for params in self.param_grid:
            model = self.model_builder(params)
            n_params = count_trainable_params(model)

            # NOTE: the original notebook accepted a max_params budget but
            # never actually used it to skip anything (the check existed,
            # commented out). Left as-is, an oversized config could win the
            # search and then fail the assignment's own parameter budget.
            # Re-enabled here; pass enforce_param_budget=False to restore
            # the old (log-everything) behavior.
            if self.enforce_param_budget and n_params > self.max_params:
                self.results.append({**params, "n_params": n_params, "status": "skipped_too_large"})
                log_lines.append(f"Config {params} -> params={n_params} SKIPPED (exceeds {self.max_params})")
                continue

            epochs = self.quick_epochs if self.quick_epochs else params.get("epochs", 80)
            train_cfg = TrainConfig(
                epochs=epochs,
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 0.0),
                optimizer_name=params.get("optimizer_name", "adam"),
                max_params=self.max_params,
                verbose=False,
            )
            trainer = ModelTrainer(model, train_loader, test_loader, train_cfg)
            trainer.fit()
            acc = trainer.accuracy(test_loader)
            record = {**params, "n_params": n_params, "test_accuracy": acc}
            self.results.append(record)
            log_lines.append(f"Config {params} -> params={n_params}, test_acc={acc:.4f}")

            if best is None or acc > best["test_accuracy"]:
                best = {**record, "model": model, "trainer": trainer}

        if best is None and self.results:
            scored = [r for r in self.results if "test_accuracy" in r]
            if scored:
                best = max(scored, key=lambda r: r["test_accuracy"])

        best_clean = {k: v for k, v in best.items() if k not in ("model", "trainer")}

        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write("\n".join(log_lines))
                f.write("\n\nBest configuration found:\n")
                f.write(str(best_clean))
            print(f"Full grid search log ({len(log_lines)} configurations) written to '{self.log_file}'")

        print("Best configuration found:", best_clean)
        return best

    def summary(self) -> List[Dict[str, Any]]:
        scored = [r for r in self.results if "test_accuracy" in r]
        return sorted(scored, key=lambda r: r["test_accuracy"], reverse=True)
