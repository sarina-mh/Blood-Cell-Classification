"""Probe-then-commit grid search: train every candidate config for a short
number of epochs, then hand back the best one so the caller can retrain it
for real. Useful when a full training run is too slow to try every
combination of hyperparameters at full length.
"""
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch.nn as nn

from .trainer import Trainer, count_trainable_params


class GridSearch:
    def __init__(self, model_builder: Callable[[Dict[str, Any]], nn.Module],
                 train_loader, val_loader,
                 param_grid: List[Dict[str, Any]],
                 max_params: Optional[int] = None,
                 probe_epochs: int = 5,
                 log_file: Optional[Union[str, Path]] = None):
        self.model_builder = model_builder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_grid = param_grid
        self.max_params = max_params
        self.probe_epochs = probe_epochs
        self.log_file = log_file
        self.results: List[Dict[str, Any]] = []

    @staticmethod
    def expand_grid(options: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Cartesian product of a {param_name: [values]} dict, e.g.
        expand_grid({"lr": [1e-3, 1e-4], "opt": ["adam", "sgd"]}) -> 4 configs.
        """
        keys = list(options.keys())
        combos = list(itertools.product(*options.values()))
        return [dict(zip(keys, combo)) for combo in combos]

    def run(self) -> Dict[str, Any]:
        best = None
        log_lines: List[str] = []

        for params in self.param_grid:
            model = self.model_builder(params)
            n_params = count_trainable_params(model)

            if self.max_params is not None and n_params > self.max_params:
                self.results.append({**params, "n_params": n_params, "status": "skipped_too_large"})
                log_lines.append(f"Config {params} -> params={n_params} SKIPPED (exceeds {self.max_params})")
                continue

            trainer = Trainer(model, self.train_loader, self.val_loader,
                               epochs=self.probe_epochs, lr=params.get("lr", 1e-3),
                               optimizer_name=params.get("optimizer_name", "adam"), verbose=False)
            trainer.fit()
            acc = trainer.accuracy(self.val_loader)
            record = {**params, "n_params": n_params, "val_accuracy": acc}
            self.results.append(record)
            log_lines.append(f"Config {params} -> params={n_params}, val_acc={acc:.4f}")

            if best is None or acc > best["val_accuracy"]:
                best = {**record, "model": model, "trainer": trainer}

        best_clean = {k: v for k, v in best.items() if k not in ("model", "trainer")} if best else None

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write("\n".join(log_lines))
                if best_clean:
                    f.write("\n\nBest configuration found:\n")
                    f.write(str(best_clean))

        return best

    def summary(self) -> List[Dict[str, Any]]:
        scored = [r for r in self.results if "val_accuracy" in r]
        return sorted(scored, key=lambda r: r["val_accuracy"], reverse=True)
