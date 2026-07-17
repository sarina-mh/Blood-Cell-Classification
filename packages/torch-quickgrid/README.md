# torch-quickgrid

A small training loop and grid search for PyTorch, pulled out of a larger
image classification project once I noticed I kept writing the same
"probe a bunch of configs, retrain the winner for real" pattern from
scratch each time.

No callbacks system, no config-file DSL, no experiment tracker integration
— just two classes. If you outgrow it, you'll know exactly what you're
outgrowing, because it's about 150 lines total.

## Why this exists instead of just using Optuna/Ray Tune/Lightning

Those tools are the right call for a real hyperparameter search. This one
is for the much more common case of "I have 8-30 candidate configs, I want
to probe each for a few epochs, and I want the best one retrained for
longer" — a two-line loop plus some bookkeeping, without pulling in a
dependency that brings its own config system and vocabulary. If your search
space grows past a plain grid, switch to one of those instead.

## Install

```bash
pip install -e .
```

or just copy `src/quickgrid/` into your project — it only depends on
`torch` and `numpy`.

## Usage

```python
from quickgrid import Trainer, GridSearch

def build_model(params: dict) -> nn.Module:
    return MyModel(hidden_dim=params["hidden_dim"], dropout=params["dropout"])

grid = GridSearch.expand_grid({
    "hidden_dim": [64, 128, 256],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 5e-4],
    "optimizer_name": ["adam", "sgd"],
})

search = GridSearch(build_model, train_loader, val_loader, param_grid=grid,
                     probe_epochs=5, max_params=5_000_000, log_file="grid_log.txt")
best = search.run()
print(search.summary())  # every config, sorted by val accuracy

# retrain the winner for real
best_params = {k: v for k, v in best.items() if k not in ("model", "trainer", "n_params", "val_accuracy")}
final_model = build_model(best_params)
trainer = Trainer(final_model, train_loader, val_loader, epochs=100, lr=best_params["lr"])
trainer.fit()
print(trainer.accuracy(val_loader))
```

See [`examples/synthetic_demo.py`](examples/synthetic_demo.py) for a
complete, runnable version on synthetic data — no dataset download needed,
just `python examples/synthetic_demo.py`.

## What it assumes

- Your model is an `nn.Module` and your loaders yield `(inputs, labels)`
  batches — that's it.
- Cross-entropy loss and single-label classification. For anything else,
  subclass `Trainer` and override `_run_epoch`.
- `max_params` in `GridSearch`, if you set it, is enforced — oversized
  configs get skipped and logged as `skipped_too_large`, they don't quietly
  train anyway.

## Where this came from

This lives inside the [blood-cell-mlp-vs-cnn](../../) repo, extracted from
the training and grid-search infrastructure in
[`../../src/bloodcell`](../../src/bloodcell), where the original version was
specific to one dataset and one pair of models. This package is that same
logic generalized so it isn't tied to either — it's kept as a self-contained,
independently `pip install`-able unit (own `pyproject.toml`, own tests would
go here too) rather than folded into `src/bloodcell`, since the whole point
is that it doesn't know anything about blood cells.

## License

MIT — see [LICENSE](LICENSE).
