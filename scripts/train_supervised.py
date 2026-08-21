from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sela.baseline.supervised import SupervisedConfig, TrainLogger, run_cv

_FM_MODELS = ("moment", "chronos", "timer", "timesfm")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def load_dataset(name: str):
    ds = DATA_DIR / name
    labels = defaultdict(list)
    with open(ds / "labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            sid = r["sample_id"].rsplit(".", 1)[0] \
                if r["sample_id"].endswith(".csv") else r["sample_id"]
            labels[sid].append({"className": r["class_name"],
                                "start": int(r["start"]), "end": int(r["end"])})

    sids, X, y = [], [], []
    for p in sorted((ds / "timeseries").glob("*.csv")):
        df = pd.read_csv(p)
        numeric = [c for c in df.columns
                   if df[c].dtype.kind in "iufcb" and not c.startswith("Unnamed")]
        sids.append(p.stem)
        X.append(df[numeric])
        y.append(labels.get(p.stem, []))

    class_names = sorted({e["className"] for evs in y for e in evs})
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_int = [[{"className": cls_map[e["className"]],
               "start": e["start"], "end": e["end"]} for e in evs]
             for evs in y]
    return sids, X, y_int, class_names


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised baseline trainer")
    p.add_argument("--model",   required=True,
                   choices=["cnn", "transformer", *_FM_MODELS])
    p.add_argument("--dataset", default="PressureTest")
    p.add_argument("--folds",   type=int, default=5)
    p.add_argument("--epochs",  type=int, default=25)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--head-mode", dest="head_mode", default="seq",
                   choices=["seq", "bbox"])
    p.add_argument("--num-queries", dest="num_queries", type=int, default=1)
    args = p.parse_args()

    sids, X, y, class_names = load_dataset(args.dataset)
    print(f"Dataset {args.dataset}: {len(X)} samples, classes={class_names}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "logs" / f"supervised_{ts}_{args.dataset}_{args.model}"
    log = TrainLogger(out_dir / "train.log")
    log.log(f"dataset={args.dataset}  classes={class_names}")

    if args.model in _FM_MODELS or args.head_mode == "bbox":
        from sela.baseline.supervised_fm import FMConfig, run_cv as run_cv_fm
        cfg = FMConfig(model=args.model, n_splits=args.folds,
                       epochs=args.epochs, seed=args.seed,
                       head_mode=args.head_mode, num_queries=args.num_queries)
        results = run_cv_fm(X, y, num_classes=len(class_names), cfg=cfg, log=log)
    else:
        cfg = SupervisedConfig(model=args.model, n_splits=args.folds,
                               epochs=args.epochs, seed=args.seed)
        results = run_cv(X, y, num_classes=len(class_names), cfg=cfg, log=log)
    results["dataset"] = args.dataset
    results["classes"] = class_names

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    log.log(f"results written to {out_dir / 'results.json'}")
    log.close()


if __name__ == "__main__":
    main()
