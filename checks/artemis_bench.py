import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
ROOT = CHECKS.parent
DATA_DIR = ROOT / "data"
BENCH = CHECKS / "bloodcell_bench.py"
BASELINE_LOG = CHECKS / "bench_baseline.log"
RUN_LOG = ROOT / "bench_out" / "bench_artemis.log"
RESULTS = ROOT / "artemis_results.json"
ARCHIVES = [DATA_DIR / f"archive{i}.zip" for i in range(1, 8)]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BENCH_ARGS = ["--model", "cnn", "--epochs", "3", "--seeds", "0", "1", "--batch-size", "64",
              "--workers", "0", "--setup-repeats", "5", "--throughput-repeats", "2", "--grid-epochs", "1"]


def describe(path):
    with open(path, "rb") as f:
        head = f.read(8)
    return f"{path.name}: {path.stat().st_size:,} bytes, first bytes {head.hex(' ')}"


def check_not_pointers(archives):
    for a in archives:
        with open(a, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs"):
            sys.exit(f"{a.name} is a Git LFS pointer file, not the data; the clone did not fetch the LFS content")
        if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
            sys.exit(f"{a.name} is an HTML page, not a zip; re-upload the real file")


def join_parts(archives, dest):
    with open(dest, "wb") as out:
        for a in archives:
            with open(a, "rb") as f:
                shutil.copyfileobj(f, out)
    if not zipfile.is_zipfile(dest):
        details = "; ".join(describe(a) for a in archives[:2])
        sys.exit(f"archive1..archive7 are not valid zips on their own or when joined in order. {details}")
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:
            sys.exit("the joined archive is corrupted (CRC check failed); a part is missing or damaged")


def build_dataset(archives):
    check_not_pointers(archives)
    sig = "|".join(f"{a.name}:{a.stat().st_size}:{int(a.stat().st_mtime)}" for a in archives)
    base = Path(tempfile.gettempdir()) / f"bloodcell_data_{hashlib.sha1(sig.encode()).hexdigest()[:12]}"
    merged = base / "merged"
    marker = base / ".ready"
    if marker.exists():
        return merged
    shutil.rmtree(base, ignore_errors=True)
    merged.mkdir(parents=True)
    valid = [zipfile.is_zipfile(a) for a in archives]
    if all(valid):
        sources = list(archives)
    elif not any(valid[:-1]):
        print("data: archive1..archive7 are not zips on their own; joining them in order into one zip")
        joined = base / "joined.zip"
        join_parts(archives, joined)
        sources = [joined]
    else:
        bad = [a.name for a, ok in zip(archives, valid) if not ok]
        sys.exit(f"not valid zip files: {', '.join(bad)}")
    for i, archive in enumerate(sources, 1):
        dest = base / "raw" / f"z{i}"
        print(f"data: unzipping {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        for d in [p for p in dest.rglob("*") if p.is_dir()]:
            if any(part.startswith(("__", ".")) for part in d.relative_to(dest).parts):
                continue
            files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
            if not files:
                continue
            out = merged / d.name
            out.mkdir(exist_ok=True)
            for f in sorted(files):
                target = out / f.name
                if target.exists():
                    target = out / f"z{i}_{f.name}"
                shutil.move(str(f), str(target))
    classes = sorted(c.name for c in merged.iterdir() if c.is_dir())
    if len(classes) < 2:
        sys.exit("fewer than 2 class folders with images were found inside the archives")
    shutil.rmtree(base / "raw", ignore_errors=True)
    if (base / "joined.zip").exists():
        (base / "joined.zip").unlink()
    marker.write_text("ok")
    print(f"data: merged classes {classes}")
    return merged


def prepare_data():
    env = os.environ.get("BLOODCELL_DATA_ROOT")
    if env and Path(env).is_dir():
        print(f"data: using BLOODCELL_DATA_ROOT={env}")
        return
    present = [a for a in ARCHIVES if a.exists()]
    if not present:
        print("data: no archive1.zip..archive7.zip in data/; the bench will use its default data root")
        return
    missing = [a.name for a in ARCHIVES if not a.exists()]
    if missing:
        sys.exit(f"missing archives in {DATA_DIR}: {', '.join(missing)}")
    root = build_dataset(ARCHIVES)
    os.environ["BLOODCELL_DATA_ROOT"] = str(root)
    print(f"data: using {root}")


def run_bench(label, log_file, baseline=None):
    cmd = [sys.executable, str(BENCH), "--repo", str(ROOT), "--label", label, "--log-file", str(log_file)] + BENCH_ARGS
    if baseline is not None:
        cmd += ["--baseline", str(baseline)]
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        sys.exit(rc)


def read_results(log_file):
    text = Path(log_file).read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, val in re.findall(r"RESULT (\w+)=(\S+)", text):
        v = float(val)
        if math.isfinite(v):
            out[key] = v
    if not out:
        sys.exit(f"no RESULT lines found in {log_file}")
    return out


def main():
    prepare_data()
    if "--make-baseline" in sys.argv:
        BASELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        run_bench("baseline", BASELINE_LOG)
        print(f"baseline log written to {BASELINE_LOG}")
        return
    if not BASELINE_LOG.exists():
        sys.exit(f"missing {BASELINE_LOG}; run: python checks/artemis_bench.py --make-baseline on the original code")
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    run_bench("artemis", RUN_LOG, baseline=BASELINE_LOG)
    results = read_results(RUN_LOG)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS} from {RUN_LOG}")


if __name__ == "__main__":
    main()
