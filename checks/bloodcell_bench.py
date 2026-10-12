import argparse
import gc
import hashlib
import importlib
import io
import logging
import math
import os
import platform
import random
import re
import statistics
import sys
import threading
import time
import types
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SPLIT_SEED = 42
log = logging.getLogger("bench")


def fmt(v, nd=6):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if isinstance(v, (int, str)):
        return str(v)
    return f"{v:.{nd}f}"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def setup_logging(log_path):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt_ = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt_)
        log.addHandler(h)


def load_bloodcell(repo):
    sys.path.insert(0, str(repo / "src"))
    ns = types.SimpleNamespace()
    for name in ("config", "data", "models", "train", "search"):
        setattr(ns, name, importlib.import_module(f"bloodcell.{name}"))
    return ns


class PeakRSS:

    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak_mb = None
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    def _loop(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            time.sleep(self.interval)

    def __enter__(self):
        if self._proc is not None:
            self.peak_mb = self._proc.memory_info().rss / 2**20
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._stop.set()
            self._thread.join()
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)


class CountingLoader:

    def __init__(self, loader):
        self._loader = loader
        self.reset()

    def reset(self):
        self.iters = 0
        self.batches = 0

    def __iter__(self):
        self.iters += 1
        for batch in self._loader:
            self.batches += 1
            yield batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self.__dict__["_loader"], name)


def make_dm(bc, args, workers):
    cfg = bc.config.DataConfig(data_root=str(args.data_root), num_workers=workers,
                               batch_size=args.batch_size, seed=SPLIT_SEED)
    return bc.data.BloodCellDataModule(cfg)


def build_model(bc, kind, num_classes, image_size, params=None):
    c, m = bc.config, bc.models
    p = params or {}
    if kind == "mlp":
        cfg = c.MLPConfig(input_dim=image_size * image_size * 3, num_classes=num_classes,
                          **({"hidden_dims": p["hidden_dims"]} if "hidden_dims" in p else {}),
                          **({"dropout": p["dropout"]} if "dropout" in p else {}))
        return m.BloodCellMLP(cfg)
    if kind == "cnn":
        cfg = c.CNNConfig(num_classes=num_classes,
                          **({"conv_channels": p["conv_channels"]} if "conv_channels" in p else {}),
                          **({"dropout": p["dropout"]} if "dropout" in p else {}))
        return m.BloodCellCNN(cfg, input_size=image_size)
    cfg = c.TinyCNNConfig(num_classes=num_classes,
                          **({"conv_channels": p["conv_channels"]} if "conv_channels" in p else {}),
                          **({"dropout": p["dropout"]} if "dropout" in p else {}))
    return m.TinyCNN(cfg)


GRID_OPTIONS = {
    "mlp": {"hidden_dims": [[512, 256, 128], [512, 256]], "dropout": [0.2], "lr": [1e-3],
            "optimizer_name": ["adam"]},
    "cnn": {"conv_channels": [[16, 32, 64]], "dropout": [0.2, 0.4], "lr": [1e-3],
            "optimizer_name": ["adam"]},
    "tiny_cnn": {"conv_channels": [[8, 16, 32]], "dropout": [0.2, 0.4], "lr": [1e-3],
                 "optimizer_name": ["adam"]},
}


def parse_baseline_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {"split_fingerprint": None, "runs": {}, "config": {}}
    m = re.search(r"\[M01\].*?sha256=([0-9a-f]{64})", text)
    if m:
        out["split_fingerprint"] = m.group(1)
    for m in re.finditer(r"RUNDATA seed=(\d+) train_loss=(\S*) test_loss=(\S*)", text):
        out["runs"][int(m.group(1))] = {
            "train_loss": [float(x) for x in m.group(2).split(",") if x],
            "test_loss": [float(x) for x in m.group(3).split(",") if x],
        }
    m = re.search(r"CONFIG model=(\S+) epochs=(\d+) seeds=(\S+) batch_size=(\d+)", text)
    if m:
        out["config"] = {"model": m.group(1), "epochs": int(m.group(2)),
                         "seeds": [int(x) for x in m.group(3).split(",")], "batch_size": int(m.group(4))}
    return out


def split_fingerprint(dm):
    s = ",".join(str(i) for i in dm.test_indices) + "|" + ",".join(dm.classes)
    return hashlib.sha256(s.encode()).hexdigest()


def measure_setup(bc, args):
    make_dm(bc, args, 0).setup()
    times = []
    for _ in range(args.setup_repeats):
        dm = make_dm(bc, args, 0)
        t = time.perf_counter()
        dm.setup()
        times.append(time.perf_counter() - t)
    return med(times), times


def measure_throughput(bc, args):
    out = {}
    for w in args.workers:
        dm = make_dm(bc, args, w)
        dm.setup()
        train_loader, _ = dm.get_loaders()
        n = len(dm.train_dataset)
        for _ in train_loader:
            pass
        rates = []
        for _ in range(args.throughput_repeats):
            t = time.perf_counter()
            for _ in train_loader:
                pass
            rates.append(n / (time.perf_counter() - t))
        out[w] = med(rates)
        del train_loader
        gc.collect()
    return out


def instrument_epochs(trainer, device):
    durations = []
    orig = trainer._run_epoch

    def wrapped(*a, **k):
        sync(device)
        t = time.perf_counter()
        r = orig(*a, **k)
        sync(device)
        durations.append(time.perf_counter() - t)
        return r

    trainer._run_epoch = wrapped
    return durations


def capture_report(bc, trainer, loader, classes):
    mod = bc.train
    orig = getattr(mod, "classification_report", None)
    cap = {}
    if orig is not None:
        def wrapper(y_true, y_pred, *a, **k):
            cap["y_true"], cap["y_pred"] = y_true, y_pred
            return orig(y_true, y_pred, *a, **k)
        mod.classification_report = wrapper
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            trainer.report(loader, classes)
    finally:
        if orig is not None:
            mod.classification_report = orig
        plt.close("all")
    return buf.getvalue(), cap


def one_run(bc, args, device, seed):
    from sklearn.metrics import f1_score

    set_seed(seed)
    dm = make_dm(bc, args, 0)
    dm.setup()
    train_loader, test_loader = dm.get_loaders()
    test_c = CountingLoader(test_loader)
    model = build_model(bc, args.model, len(dm.classes), dm.config.image_size)
    cfg = bc.config.TrainConfig(epochs=args.epochs, verbose=False)
    trainer = bc.train.ModelTrainer(model, train_loader, test_c, cfg, device=device)
    durations = instrument_epochs(trainer, device)

    test_c.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with PeakRSS() as rss:
        t = time.perf_counter()
        history = trainer.fit()
        sync(device)
        fit_s = time.perf_counter() - t
    cuda_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    acc = float(trainer.accuracy(test_c))
    report_text, cap = capture_report(bc, trainer, test_c, dm.classes)
    f1 = float(f1_score(cap["y_true"], cap["y_pred"], average="macro")) if cap else None

    n_calls = len(durations)
    epoch_times = None
    if n_calls == 2 * args.epochs:
        epoch_times = [durations[i] + durations[i + 1] for i in range(0, n_calls, 2)]

    return {
        "seed": seed,
        "history": {"train_loss": list(history["train_loss"]), "test_loss": list(history["test_loss"])},
        "test_accuracy": acc,
        "macro_f1": f1,
        "epoch_times": epoch_times,
        "epoch_calls": n_calls,
        "peak_rss_mb": rss.peak_mb,
        "cuda_peak_mb": cuda_peak,
        "fit_s": fit_s,
        "test_batches": test_c.batches,
        "test_iters": test_c.iters,
        "test_passes": test_c.batches / max(len(test_c), 1),
        "report_text": report_text,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def loss_deviation(hist, base_hist):
    out = {}
    for key in ("train_loss", "test_loss"):
        a, b = hist[key], base_hist[key]
        if len(a) != len(b):
            out[key] = None
            continue
        out[key] = max((abs(x - y) / abs(y) if y else float("inf")) for x, y in zip(a, b))
    return out


def make_grid(bc, args, num_classes, image_size):
    def builder(params):
        return build_model(bc, args.model, num_classes, image_size, params)

    return builder, bc.search.GridSearchRunner.expand_grid(GRID_OPTIONS[args.model])


def run_grid(bc, args, device, max_params, quick_epochs):
    set_seed(args.seeds[0])
    dm = make_dm(bc, args, 0)
    dm.setup()
    builder, grid = make_grid(bc, args, len(dm.classes), dm.config.image_size)
    runner = bc.search.GridSearchRunner(model_builder=builder, data_module=dm, param_grid=grid,
                                        max_params=max_params, quick_epochs=quick_epochs, log_file=None)
    buf = io.StringIO()
    sync(device)
    t = time.perf_counter()
    outcome, detail = "none", ""
    try:
        with redirect_stdout(buf):
            runner.run()
    except Exception as e:
        outcome, detail = type(e).__name__, str(e)[:200]
    sync(device)
    wall = time.perf_counter() - t
    results = getattr(runner, "results", [])
    skipped = sum(1 for r in results if r.get("status") == "skipped_too_large")
    return {"wall_s": wall, "outcome": outcome, "detail": detail, "n_configs": len(grid),
            "n_skipped": skipped, "n_trained": len(results) - skipped, "quick_epochs": quick_epochs}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="version name, e.g. baseline, v1, exp5")
    ap.add_argument("--repo", default=".", help="repo folder to measure (default: current folder)")
    ap.add_argument("--data-root", default=None, help="default: env BLOODCELL_DATA_ROOT, else <repo>/data/raw")
    ap.add_argument("--baseline", default=None, help="the baseline run's .log file (metrics 1 and 2)")
    ap.add_argument("--out-dir", default=None, help="default: <repo>/bench_out")
    ap.add_argument("--log-file", default=None, help="exact log path (default: <out-dir>/bench_<label>_<timestamp>.log)")
    ap.add_argument("--model", choices=["mlp", "cnn", "tiny_cnn"], default="cnn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="+", default=[0], help="num_workers values for metric 6")
    ap.add_argument("--setup-repeats", type=int, default=10)
    ap.add_argument("--throughput-repeats", type=int, default=3)
    ap.add_argument("--grid-epochs", type=int, default=2, help="quick_epochs for metric 10")
    ap.add_argument("--skip-determinism", action="store_true", help="skip the repeated run for metric 14")
    return ap.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    args.data_root = Path(args.data_root or os.environ.get("BLOODCELL_DATA_ROOT") or repo / "data" / "raw")
    out_dir = Path(args.out_dir) if args.out_dir else repo / "bench_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else out_dir / f"bench_{args.label}_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    if not args.data_root.exists():
        log.error("data root not found: %s", args.data_root)
        sys.exit(1)

    bc = load_bloodcell(repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=== Blood-Cell bench | label=%s | model=%s ===", args.label, args.model)
    log.info("repo=%s", repo)
    log.info("data_root=%s", args.data_root)
    log.info("imported bloodcell from: %s", Path(bc.train.__file__).resolve())
    log.info("python=%s | torch=%s | numpy=%s | platform=%s | device=%s",
             platform.python_version(), torch.__version__, np.__version__, platform.platform(), device)
    log.info("epochs=%d seeds=%s batch_size=%d workers=%s setup_repeats=%d throughput_repeats=%d grid_epochs=%d",
             args.epochs, args.seeds, args.batch_size, args.workers, args.setup_repeats,
             args.throughput_repeats, args.grid_epochs)

    log.info("CONFIG model=%s epochs=%d seeds=%s batch_size=%d", args.model, args.epochs,
             ",".join(str(x) for x in args.seeds), args.batch_size)

    baseline = None
    if args.baseline:
        baseline = parse_baseline_log(args.baseline)
        bcfg = baseline["config"]
        for k in ("model", "epochs", "seeds", "batch_size"):
            if bcfg.get(k) != getattr(args, k):
                log.warning("baseline config differs on %s: baseline=%s this=%s", k, bcfg.get(k), getattr(args, k))

    dm0 = make_dm(bc, args, 0)
    dm0.setup()
    fp = split_fingerprint(dm0)
    m01 = None if baseline is None or baseline["split_fingerprint"] is None else int(baseline["split_fingerprint"] == fp)
    log.info("[M01] split_fingerprint_match = %s | sha256=%s | n_test=%d classes=%s",
             fmt(m01), fp, len(dm0.test_indices), dm0.classes)
    del dm0

    setup_med, setup_all = measure_setup(bc, args)
    log.info("[M05] setup_time_s = %s | median of %d | min=%s max=%s",
             fmt(setup_med), len(setup_all), fmt(min(setup_all)), fmt(max(setup_all)))

    thr = measure_throughput(bc, args)
    for w, v in thr.items():
        log.info("[M06] loader_throughput_img_s = %s | num_workers=%d | median of %d (1 untimed warm-up)",
                 fmt(v, 2), w, args.throughput_repeats)

    runs = []
    for seed in args.seeds:
        log.info("--- training run: model=%s seed=%d epochs=%d ---", args.model, seed, args.epochs)
        r = one_run(bc, args, device, seed)
        runs.append(r)
        log.info("    n_params=%d fit_s=%s test_accuracy=%s macro_f1=%s", r["n_params"], fmt(r["fit_s"], 3),
                 fmt(r["test_accuracy"]), fmt(r["macro_f1"]))
        log.info("    train_loss=%s", [round(x, 5) for x in r["history"]["train_loss"]])
        log.info("    test_loss =%s", [round(x, 5) for x in r["history"]["test_loss"]])
        log.info("RUNDATA seed=%d train_loss=%s test_loss=%s", seed,
                 ",".join(repr(float(x)) for x in r["history"]["train_loss"]),
                 ",".join(repr(float(x)) for x in r["history"]["test_loss"]))
        log.info("    epoch_calls=%d epoch_times_s=%s", r["epoch_calls"],
                 None if r["epoch_times"] is None else [round(x, 3) for x in r["epoch_times"]])
        log.info("    peak_rss_mb=%s cuda_peak_mb=%s | test_batches=%d test_iters=%d passes=%s",
                 fmt(r["peak_rss_mb"], 1), fmt(r["cuda_peak_mb"], 1), r["test_batches"], r["test_iters"],
                 fmt(r["test_passes"], 3))
        log.info("    classification report (as printed by trainer.report):\n%s", r["report_text"])

    dev_train, dev_test, m02_notes = [], [], []
    if baseline is not None:
        bruns = baseline["runs"]
        for r in runs:
            b = bruns.get(r["seed"])
            if b is None:
                m02_notes.append(f"seed {r['seed']}: not in baseline")
                continue
            d = loss_deviation(r["history"], b)
            dev_train.append(d["train_loss"])
            dev_test.append(d["test_loss"])
            log.info("[M02] seed=%d loss_trajectory_deviation train=%s test=%s", r["seed"],
                     fmt(d["train_loss"]), fmt(d["test_loss"]))
    m02_train = max([x for x in dev_train if x is not None], default=None)
    m02_test = max([x for x in dev_test if x is not None], default=None)
    if baseline is None:
        log.info("[M02] loss_trajectory_deviation = N/A (no --baseline)")

    m03 = mean([r["test_accuracy"] for r in runs])
    m04 = mean([r["macro_f1"] for r in runs])
    all_epoch_times = [t for r in runs if r["epoch_times"] for t in r["epoch_times"]]
    m07 = med(all_epoch_times)
    m08 = max([r["peak_rss_mb"] for r in runs if r["peak_rss_mb"] is not None], default=None)
    m08_cuda = max([r["cuda_peak_mb"] for r in runs if r["cuda_peak_mb"] is not None], default=None)
    m09 = [r["test_passes"] for r in runs]
    log.info("[M03] test_accuracy = %s | mean over seeds %s | per seed %s", fmt(m03), args.seeds,
             [round(r["test_accuracy"], 4) for r in runs])
    log.info("[M04] macro_f1 = %s | mean over seeds | per seed %s", fmt(m04),
             [None if r["macro_f1"] is None else round(r["macro_f1"], 4) for r in runs])
    log.info("[M07] epoch_time_s = %s | median over %d epochs (all seeds pooled)", fmt(m07), len(all_epoch_times))
    log.info("[M08] peak_memory_mb = %s | max over seeds, RSS sampled during fit() | cuda_peak_mb=%s",
             fmt(m08, 1), fmt(m08_cuda, 1))
    log.info("[M09] test_forward_passes_per_run = %s | per seed", [round(x, 3) for x in m09])

    log.info("--- grid search wall time (fixed grid: %s) ---", GRID_OPTIONS[args.model])
    g10 = run_grid(bc, args, device, max_params=10 ** 12, quick_epochs=args.grid_epochs)
    log.info("[M10] grid_search_wall_time_s = %s | configs=%d trained=%d skipped=%d quick_epochs=%d outcome=%s %s",
             fmt(g10["wall_s"], 3), g10["n_configs"], g10["n_trained"], g10["n_skipped"],
             g10["quick_epochs"], g10["outcome"], g10["detail"])

    m14 = None
    if not args.skip_determinism:
        log.info("--- determinism: re-running seed %d ---", args.seeds[0])
        again = one_run(bc, args, device, args.seeds[0])
        m14 = abs(runs[0]["test_accuracy"] - again["test_accuracy"])
        loss_gap = max(abs(a - b) for a, b in zip(runs[0]["history"]["test_loss"], again["history"]["test_loss"]))
        log.info("[M14] same_seed_accuracy_diff = %s | run1=%s run2=%s | (info) max |test_loss diff|=%s",
                 fmt(m14), fmt(runs[0]["test_accuracy"]), fmt(again["test_accuracy"]), fmt(loss_gap, 8))

    log.info("--- all configs over budget: max_params=1 ---")
    g15 = run_grid(bc, args, device, max_params=1, quick_epochs=1)
    log.info("[M15] all_skipped_outcome = %s | %s | configs=%d skipped=%d", g15["outcome"], g15["detail"],
             g15["n_configs"], g15["n_skipped"])

    res, missing = {}, 0

    def put(key, value):
        nonlocal missing
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)) and math.isfinite(value):
            res[key] = value
        else:
            missing += 1

    put("m01_split_fingerprint_match", m01)
    put("m02_loss_deviation_train", m02_train)
    put("m02_loss_deviation_test", m02_test)
    put("m03_test_accuracy", m03)
    put("m04_macro_f1", m04)
    put("m05_setup_time_s", setup_med)
    for w, v in thr.items():
        put(f"m06_loader_throughput_img_s_w{w}", v)
    put("m07_epoch_time_s", m07)
    put("m08_peak_memory_mb", m08)
    put("m09_test_forward_passes_per_run", mean(m09))
    put("m10_grid_search_wall_time_s", g10["wall_s"])
    put("m14_same_seed_accuracy_diff", m14)
    put("m15_all_skipped_raised", g15["outcome"] != "none")
    put("m15_all_skipped_is_attributeerror", g15["outcome"] == "AttributeError")
    res["missing_metric_count"] = missing

    log.info("=" * 78)
    log.info("SUMMARY  label=%s  model=%s  epochs=%d  seeds=%s  device=%s", args.label, args.model,
             args.epochs, args.seeds, device)
    log.info("M01 split_fingerprint_match       = %s", fmt(m01))
    log.info("M02 loss_trajectory_deviation     = train %s | test %s (ratio, max over epochs and seeds)",
             fmt(m02_train), fmt(m02_test))
    log.info("M03 test_accuracy                 = %s (fraction, mean over seeds)", fmt(m03))
    log.info("M04 macro_f1                      = %s (mean over seeds)", fmt(m04))
    log.info("M05 setup_time_s                  = %s (s, median)", fmt(setup_med))
    log.info("M06 loader_throughput_img_s       = %s (images/s, median)",
             ", ".join(f"workers={w}: {fmt(v, 2)}" for w, v in thr.items()))
    log.info("M07 epoch_time_s                  = %s (s, median over epochs)", fmt(m07))
    log.info("M08 peak_memory_mb                = %s (MB RSS, max over seeds)", fmt(m08, 1))
    log.info("M09 test_forward_passes_per_run   = %s (full-loader passes per run)", [round(x, 3) for x in m09])
    log.info("M10 grid_search_wall_time_s       = %s (s)", fmt(g10["wall_s"], 3))
    log.info("M14 same_seed_accuracy_diff       = %s (fraction)", fmt(m14))
    log.info("M15 all_skipped_outcome           = %s", g15["outcome"])
    log.info("=" * 78)
    log.info("RESULTS (machine-readable; one line per metric, full precision)")
    for k, v in res.items():
        log.info("RESULT %s=%r", k, v)
    log.info("log: %s", log_path)


if __name__ == "__main__":
    main()
