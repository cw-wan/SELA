from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sela.eval.metrics import (
    f1_iou, match_counts, coverage_counts, coverage_aggregate,
)
from sela.eval.pricing import estimate_cost
from sela.llm.config import load_llm_config

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
CONFIG_PATH = str(BASE_DIR / "config" / "llm.json")

_TRANSIENT = ("APIConnectionError", "APITimeoutError", "RateLimitError",
              "InternalServerError", "connection error", "timeout",
              "502", "503", "504", "429")


def _is_transient(exc: Exception) -> bool:
    t = f"{type(exc).__name__}: {exc}".lower()
    return any(m.lower() in t for m in _TRANSIENT)


def load_labels(ds: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    with open(ds / "labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            sid = r["sample_id"].rsplit(".", 1)[0] \
                if r["sample_id"].endswith(".csv") else r["sample_id"]
            out[sid].append({"className": r["class_name"],
                             "start": int(r["start"]), "end": int(r["end"])})
    return dict(out)


def load_ts(ds: Path, sid: str) -> pd.DataFrame:
    df = pd.read_csv(ds / "timeseries" / f"{sid}.csv")
    numeric = [c for c in df.columns
               if df[c].dtype.kind in "iufcb" and not c.startswith("Unnamed")]
    return df[numeric]


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def aggregate(counts, classes, cov=None):
    out: Dict[str, Any] = {}
    for thr in (0.5, 0.9):
        pc = {c: _f1(*counts[thr][c]) for c in classes}
        T = sum(counts[thr][c][0] for c in classes)
        P = sum(counts[thr][c][1] for c in classes)
        N = sum(counts[thr][c][2] for c in classes)
        t = f"{thr:.1f}".replace(".", "")
        out[f"micro_f1_{t}"] = _f1(T, P, N)
        out[f"macro_f1_{t}"] = sum(pc.values()) / len(classes) if classes else 0.0
        out[f"per_class_f1_{t}"] = pc
        out[f"per_class_counts_{t}"] = {c: counts[thr][c] for c in classes}
    if cov is not None:
        out.update(coverage_aggregate(cov, classes))
    return out


def build_baseline(args, cfg):
    if args.baseline == "numeric":
        from sela.baseline.numeric import NumericBaseline
        return NumericBaseline(cfg, unique=args.unique,
                               force_all_classes=args.force_all_classes)
    if args.baseline == "visual":
        from sela.baseline.visual import VisualBaseline
        return VisualBaseline(cfg, unique=args.unique,
                              force_all_classes=args.force_all_classes)
    raise ValueError(args.baseline)


async def run(args: argparse.Namespace) -> None:
    ds = DATA_DIR / args.dataset
    labels = load_labels(ds)
    desc = (ds / "desc.txt").read_text(encoding="utf-8") \
        if (ds / "desc.txt").exists() else ""
    classes = sorted({e["className"] for evs in labels.values() for e in evs})

    all_sids = sorted(p.stem for p in (ds / "timeseries").glob("*.csv"))
    sids = list(all_sids)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        sids = [s for s in sids if s in wanted]
    elif args.samples:
        sids = sids[: args.samples]

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / "logs" / \
        f"baseline_eval_{ts_str}_{args.dataset}_{args.baseline}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_llm_config(CONFIG_PATH, args.model)
    baseline = build_baseline(args, cfg)


    fs_pool: Dict[str, List[str]] = {}
    demo_cache: Dict[str, List[Dict[str, Any]]] = {}
    if args.few_shot:
        if not hasattr(baseline, "demo_messages"):
            raise SystemExit(f"--few-shot is not supported for '{args.baseline}'")
        fs_pool = {c: [s for s in all_sids
                       if any(e["className"] == c for e in labels.get(s, []))]
                   for c in classes}

    def demo_for(ex_sid: str) -> List[Dict[str, Any]]:
        if ex_sid not in demo_cache:            # render/format each example once
            demo_cache[ex_sid] = baseline.demo_messages(load_ts(ds, ex_sid),
                                                        labels[ex_sid])
        return demo_cache[ex_sid]

    def build_demos(sid: str, run_idx: int):
        if not args.few_shot:
            return [], []
        base = args.seed if args.seed is not None else 42
        rng = random.Random(f"{base}-{run_idx}-{sid}")
        demos: List[Dict[str, Any]] = []
        exs: List[str] = []
        for c in classes:
            pool = [s for s in fs_pool[c] if s != sid]
            if not pool:
                continue
            ex = rng.choice(pool)
            demos += demo_for(ex)
            exs.append(ex)
        return demos, exs

    plog = open(run_dir / "progress.log", "a", encoding="utf-8")
    lock = asyncio.Lock()

    def logline(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        plog.write(line + "\n"); plog.flush()

    logline(f"{args.baseline} eval  model={args.model}  dataset={args.dataset}")
    logline(f"samples={len(sids)}  classes={classes}  unique={args.unique}  "
            f"force_all_classes={args.force_all_classes}  "
            f"repeats={args.repeats}  concurrency={args.concurrency}")
    if args.few_shot:
        logline(f"few-shot: 1 random example/class per query, excluding the query, "
                f"resampled each run; class pools = "
                f"{ {c: len(v) for c, v in fs_pool.items()} }")
    logline(f"run dir: {run_dir}")
    logline("─" * 78)

    sem = asyncio.Semaphore(args.concurrency)
    per_run: List[Dict[str, Any]] = []          # one metrics dict per repeat
    tok_total = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def one_sample(sid, counts, done, preds_rec, cov_acc):
        async with sem:
            ts = load_ts(ds, sid)
            gt = labels.get(sid, [])
            pred, tok, err = [], {}, None
            demos, ex_used = build_demos(sid, preds_rec[0]["_run"])
            kw = {"demos": demos} if demos else {}
            for attempt in range(1, args.max_retries + 2):
                try:
                    result = await baseline.inference(ts=ts, desc=desc, events=classes, **kw)
                    pred = result.get("predictions", [])
                    tok = result.get("token_usage", {})
                    err = None
                    break
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    if _is_transient(exc) and attempt <= args.max_retries:
                        async with lock:
                            logline(f"     ↻ {sid} transient (attempt {attempt}): {err}")
                        await asyncio.sleep(min(15 * attempt, 45))
                        continue
                    break

            f05 = f1_iou(pred, gt, threshold=0.5)
            f09 = f1_iou(pred, gt, threshold=0.9)
            async with lock:
                for thr in (0.5, 0.9):
                    for c in classes:
                        p_c = [p for p in pred if p["className"] == c]
                        g_c = [g for g in gt if g["className"] == c]
                        tp, fp, fn = match_counts(p_c, g_c, threshold=thr)
                        counts[thr][c][0] += tp
                        counts[thr][c][1] += fp
                        counts[thr][c][2] += fn
                for c, v in coverage_counts(pred, gt).items():
                    s = cov_acc.setdefault(c, [0.0, 0, 0.0, 0])
                    s[0] += v[0]; s[1] += v[1]; s[2] += v[2]; s[3] += v[3]
                for k in tok_total:
                    tok_total[k] += tok.get(k, 0)
                done[0] += 1
                preds_rec.append({"sample_id": sid,
                                  "gt_class": gt[0]["className"] if gt else "—",
                                  "prediction": pred, "error": err,
                                  **({"examples": ex_used} if args.few_shot else {})})
                agg = aggregate(counts, classes, cov_acc)
                logline(
                    f"[r{preds_rec[0]['_run']}] [{done[0]:>2}/{len(sids)}] "
                    f"{sid:<14} pred={pred[0]['className'] if pred else '∅':<10} "
                    f"F1@.5={f05:.3f} F1@.9={f09:.3f}  ||  "
                    f"micro@.5={agg['micro_f1_05']:.3f} "
                    f"macro@.5={agg['macro_f1_05']:.3f}"
                    f"{'  ERROR' if err else ''}")

    for r in range(1, args.repeats + 1):
        if args.seed is not None:
            cfg.seed = args.seed               # same fixed seed on every run
            logline(f"── repeat {r}/{args.repeats} ── (seed={cfg.seed})")
        else:
            logline(f"── repeat {r}/{args.repeats} ──")
        counts = {0.5: {c: [0, 0, 0] for c in classes},
                  0.9: {c: [0, 0, 0] for c in classes}}
        cov_acc: Dict[str, List[float]] = {c: [0.0, 0, 0.0, 0] for c in classes}
        done = [0]
        preds_rec: List[Dict[str, Any]] = [{"_run": r}]
        await asyncio.gather(*(one_sample(s, counts, done, preds_rec, cov_acc)
                               for s in sids))
        agg = aggregate(counts, classes, cov_acc)
        per_run.append({"repeat": r, "metrics": agg,
                        "predictions": preds_rec[1:]})
        logline(f"── repeat {r} done: micro@.5={agg['micro_f1_05']:.4f} "
                f"macro@.5={agg['macro_f1_05']:.4f} "
                f"micro@.9={agg['micro_f1_09']:.4f} "
                f"macro@.9={agg['macro_f1_09']:.4f} "
                f"macro cov_f1={agg['macro_cov_f1']:.4f} "
                f"cov_f05={agg['macro_cov_f05']:.4f}")

    # ── aggregate mean ± std across repeats ─────────────────────────────────────
    def ms(key: str) -> Tuple[float, float]:
        vals = [pr["metrics"][key] for pr in per_run]
        return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    summary = {k: {"mean": ms(k)[0], "std": ms(k)[1]}
               for k in ("micro_f1_05", "macro_f1_05", "micro_f1_09", "macro_f1_09",
                         "micro_cov_f1", "macro_cov_f1",
                         "micro_cov_f05", "macro_cov_f05")}
    cost, price = estimate_cost(args.model, tok_total["prompt_tokens"],
                                tok_total["completion_tokens"], 0)
    (run_dir / "results.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "baseline": args.baseline,
        "unique": args.unique, "force_all_classes": args.force_all_classes,
        "few_shot": args.few_shot,
        "few_shot_protocol": ("1 random example/class per query, exclude self, "
                              "resampled each run") if args.few_shot else None,
        "repeats": args.repeats, "classes": classes,
        "n_samples": len(sids), "token_usage": tok_total,
        "cost_usd": cost, "summary": summary, "per_run": per_run,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logline("─" * 78)
    logline(f"===== {args.baseline} {args.model} on {args.dataset} "
            f"({args.repeats} repeats, {len(sids)} samples, "
            f"unique={args.unique}, force_all_classes={args.force_all_classes}) =====")
    for k, lab in (("micro_f1_05", "micro F1@0.5"), ("macro_f1_05", "macro F1@0.5"),
                   ("micro_f1_09", "micro F1@0.9"), ("macro_f1_09", "macro F1@0.9")):
        vals = [pr["metrics"][k] for pr in per_run]
        logline(f"  {lab} = {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}   "
                f"runs: {[f'{v:.3f}' for v in vals]}")
    if cost is not None:
        logline(f"tokens: {tok_total['total_tokens']:,} | est. cost @ {args.model}: "
                f"${cost:.2f} ({args.repeats} runs)")
    plog.close()


def main():
    p = argparse.ArgumentParser(description="LLM baseline evaluation")
    p.add_argument("--baseline", required=True,
                   choices=["numeric", "visual"])
    p.add_argument("--model",    required=True)
    p.add_argument("--dataset",  default="PressureTest")
    p.add_argument("--unique",   dest="unique", action="store_true", default=True)
    p.add_argument("--no-unique", dest="unique", action="store_false")
    p.add_argument("--repeats",     type=int, default=1)
    p.add_argument("--seed",        type=int, default=None)
    p.add_argument("--force-all-classes", dest="force_all_classes",
                   action="store_true", default=True)
    p.add_argument("--no-force-all-classes", dest="force_all_classes",
                   action="store_false")
    p.add_argument("--few-shot", dest="few_shot", action="store_true",
                   default=False)
    p.add_argument("--full-desc", dest="scoped_desc",
                   action="store_false", default=True)
    p.add_argument("--samples",     type=int, default=None)
    p.add_argument("--only",        default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=2, dest="max_retries")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
