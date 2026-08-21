from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sela.eval.metrics import (
    f1_iou, match_counts, coverage_counts, coverage_aggregate,
)
from sela.eval.pricing import estimate_cost
from sela.llm.config import load_llm_config
from sela.baseline.sela import SELASystem

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
CONFIG_PATH = str(BASE_DIR / "config" / "llm.json")


# ── dataset loading ───────────────────────────────────────────────────────────

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


# ── metric helpers ────────────────────────────────────────────────────────────

_TRANSIENT_MARKERS = (
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "ServiceUnavailable", "connection error",
    "timeout", "temporarily unavailable", "502", "503", "504", "429",
)


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()
    return any(m.lower() in text for m in _TRANSIENT_MARKERS)


def _f1(tp: int, fp: int, fn: int) -> float:
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0


def aggregate(counts: Dict[float, Dict[str, List[int]]],
              classes: List[str],
              cov: Dict[str, List[float]] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for thr in (0.5, 0.9):
        per_cls = {c: _f1(*counts[thr][c]) for c in classes}
        macro = sum(per_cls.values()) / len(classes) if classes else 0.0
        T = sum(counts[thr][c][0] for c in classes)
        P = sum(counts[thr][c][1] for c in classes)
        N = sum(counts[thr][c][2] for c in classes)
        tag = f"{thr:.1f}".replace(".", "")
        out[f"micro_f1_{tag}"] = _f1(T, P, N)
        out[f"macro_f1_{tag}"] = macro
        out[f"per_class_f1_{tag}"] = per_cls
        out[f"per_class_counts_{tag}"] = {c: counts[thr][c] for c in classes}
    if cov is not None:
        out.update(coverage_aggregate(cov, classes))
    return out


# ── per-sample transcript logger ──────────────────────────────────────────────

def make_sample_logger(sample_dir: Path):
    sample_dir.mkdir(parents=True, exist_ok=True)
    fh = open(sample_dir / "transcript.jsonl", "a", encoding="utf-8")
    counter = [0]

    def cb(ev: dict) -> None:
        ev = dict(ev)
        imgs = ev.pop("images", None) or []       # base64 PNG (what the LLM saw)
        svgs = ev.pop("images_svg", None) or []   # SVG twin (report-quality)
        saved = []
        for i in range(max(len(imgs), len(svgs))):
            stem = f"img_{counter[0]:03d}_{ev.get('agent', '')}"
            # Prefer SVG for report figures; keep PNG only if no SVG.
            if i < len(svgs) and svgs[i]:
                try:
                    (sample_dir / f"{stem}.svg").write_text(svgs[i], encoding="utf-8")
                    saved.append(f"{stem}.svg")
                except Exception:
                    pass
            elif i < len(imgs) and imgs[i]:
                try:
                    (sample_dir / f"{stem}.png").write_bytes(base64.b64decode(imgs[i]))
                    saved.append(f"{stem}.png")
                except Exception:
                    pass
            counter[0] += 1
        if saved:
            ev["images_saved"] = saved
        fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    return cb


# ── evaluation run ────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    ds = DATA_DIR / args.dataset
    labels = load_labels(ds)
    desc = (ds / "desc.txt").read_text(encoding="utf-8") \
        if (ds / "desc.txt").exists() else ""
    classes = sorted({e["className"] for evs in labels.values() for e in evs})

    sids = sorted(p.stem for p in (ds / "timeseries").glob("*.csv"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        sids = [s for s in sids if s in wanted]
    elif args.samples:
        sids = sids[: args.samples]

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ("_oracle" if args.oracle_schema_dir else
           "_simplified" if args.simplified else "")
    run_dir = BASE_DIR / "logs" / \
        f"sela_eval{tag}_{ts_str}_{args.dataset}_{args.model}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    schema_dir = run_dir / "schema_cache"
    schema_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_llm_config(CONFIG_PATH, args.model)
    if args.reasoning_effort:
        cfg.reasoning_effort = args.reasoning_effort

    # progress log (shared, line-buffered)
    plog = open(run_dir / "progress.log", "a", encoding="utf-8")
    lock = asyncio.Lock()

    def logline(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        plog.write(line + "\n")
        plog.flush()

    mode = (f"ORACLE (human schemas from {args.oracle_schema_dir})"
            if args.oracle_schema_dir else
            f"reparse_each_sample={args.reparse_each_sample}"
            f"{' (parse-once cache)' if not args.reparse_each_sample else ''}")
    re_str = f"reasoning_effort={cfg.reasoning_effort or 'model-default'}"
    logline(f"SELA evaluation  model={args.model}  dataset={args.dataset}")
    logline(f"samples={len(sids)}  classes={classes}  "
            f"{mode}  {re_str}  concurrency={args.concurrency}")
    logline(f"run dir: {run_dir}")
    logline("─" * 78)

    counts = {0.5: {c: [0, 0, 0] for c in classes},
              0.9: {c: [0, 0, 0] for c in classes}}
    cov_acc: Dict[str, List[float]] = {c: [0.0, 0, 0.0, 0] for c in classes}
    per_sample: List[dict] = []
    tok_total = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "cached_tokens": 0, "reasoning_tokens": 0}
    done = [0]
    sem = asyncio.Semaphore(args.concurrency)

    def checkpoint() -> None:
        payload = {
            "model": args.model, "dataset": args.dataset,
            "reparse_each_sample": args.reparse_each_sample,
            "reasoning_effort": cfg.reasoning_effort,
            "parallel_inspectors": args.parallel_inspectors,
            "n_samples_done": done[0], "n_samples_total": len(sids),
            "classes": classes,
            "token_usage": dict(tok_total),
            "metrics": aggregate(counts, classes, cov_acc),
            "per_sample": per_sample,
        }
        (run_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def one_sample(sid: str) -> None:
        async with sem:
            ts = load_ts(ds, sid)
            gt = labels.get(sid, [])
            gt_cls = gt[0]["className"] if gt else "—"
            sample_dir = run_dir / "samples" / sid

            t0 = time.time()
            pred, tok, scores, err = [], {}, {}, None
            for attempt in range(1, args.max_retries + 2):
                on_event = make_sample_logger(sample_dir)
                sela = SELASystem(cfg, on_event=on_event,
                                  reparse_each_sample=args.reparse_each_sample,
                                  schema_cache_dir=str(schema_dir),
                                  numeric_readout=args.numeric_readout,
                                  numeric_max_rows=args.numeric_max_rows,
                                  resample_mode=args.resample_mode,
                                  oracle_schema_dir=args.oracle_schema_dir,
                                  simplified_schema=args.simplified,
                                  parallel_inspectors=args.parallel_inspectors)
                try:
                    result = await sela.inference(ts=ts, desc=desc, events=classes)
                    pred = result.get("predictions", [])
                    tok = result.get("token_usage", {})
                    scores = result.get("class_scores", {})
                    err = None
                    break
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    is_transient = _is_transient(exc)
                    (sample_dir / "error.txt").write_text(
                        f"attempt {attempt}, transient={is_transient}\n\n"
                        + traceback.format_exc(), encoding="utf-8")
                    if is_transient and attempt <= args.max_retries:
                        async with lock:
                            logline(f"     ↻ {sid} transient error "
                                    f"(attempt {attempt}/{args.max_retries + 1}), "
                                    f"retrying whole sample: {err}")
                        await asyncio.sleep(min(20 * attempt, 60))
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

                pred_cls = pred[0]["className"] if pred else "∅"
                pred_iv = f"[{pred[0]['start']},{pred[0]['end']}]" if pred else "—"
                per_sample.append({
                    "sample_id": sid, "gt_class": gt_cls,
                    "ground_truth": gt, "prediction": pred,
                    "class_scores": scores, "f1_05": f05, "f1_09": f09,
                    "token_usage": tok, "error": err,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                agg = aggregate(counts, classes, cov_acc)
                emark = "  ERROR" if err else ""
                logline(
                    f"[{done[0]:>2}/{len(sids)}] {sid:<14} "
                    f"gt={gt_cls:<10} pred={pred_cls:<10} {pred_iv:<14} "
                    f"F1@.5={f05:.3f} F1@.9={f09:.3f}  ||  "
                    f"running micro@.5={agg['micro_f1_05']:.3f} "
                    f"macro@.5={agg['macro_f1_05']:.3f}"
                    f"{emark}")
                if err:
                    logline(f"     └─ {err}")
                checkpoint()

    if sids and not args.reparse_each_sample and not args.oracle_schema_dir:
        warm = SELASystem(cfg, on_event=make_sample_logger(run_dir / "_schema_warmup"),
                          reparse_each_sample=False, schema_cache_dir=str(schema_dir),
                          numeric_readout=args.numeric_readout,
                          numeric_max_rows=args.numeric_max_rows,
                          resample_mode=args.resample_mode,
                          simplified_schema=args.simplified)
        channels = [str(c) for c in load_ts(ds, sids[0]).columns]
        t_warm = time.time()
        for c in classes:
            await warm._parse_schema(c, desc, channels)
        logline(f"schema cache warmed: parsed {len(classes)} classes once "
                f"in {time.time() - t_warm:.0f}s → {schema_dir}")

    await asyncio.gather(*(one_sample(s) for s in sids))

    # ── final summary ─────────────────────────────────────────────────────────
    agg = aggregate(counts, classes, cov_acc)
    logline("─" * 78)
    logline(f"===== SELA {args.model} on {args.dataset}  "
            f"({done[0]} samples, "
            f"{'reparse each sample' if args.reparse_each_sample else 'parse-once cache'}"
            f") =====")
    logline(f"micro F1@0.5 = {agg['micro_f1_05']:.4f}    "
            f"micro F1@0.9 = {agg['micro_f1_09']:.4f}")
    logline(f"macro F1@0.5 = {agg['macro_f1_05']:.4f}    "
            f"macro F1@0.9 = {agg['macro_f1_09']:.4f}")
    logline(f"macro cov_f1 = {agg['macro_cov_f1']:.4f}    "
            f"macro cov_f05 = {agg['macro_cov_f05']:.4f}   "
            f"(micro {agg['micro_cov_f1']:.4f} / {agg['micro_cov_f05']:.4f})")
    for c in classes:
        tp5, fp5, fn5 = counts[0.5][c]
        logline(f"  {c:<12} F1@0.5={agg['per_class_f1_05'][c]:.4f}  "
                f"(tp={tp5} fp={fp5} fn={fn5})   "
                f"F1@0.9={agg['per_class_f1_09'][c]:.4f}   "
                f"cov_f1={agg['per_class_cov_f1'][c]:.4f} "
                f"cov_f05={agg['per_class_cov_f05'][c]:.4f}")
    pt, cached = tok_total["prompt_tokens"], tok_total["cached_tokens"]
    ct = tok_total["completion_tokens"]
    logline(f"tokens: {tok_total['total_tokens']:,} total "
            f"({pt:,} prompt [{cached:,} cached = "
            f"{100*cached/pt if pt else 0:.0f}%] / "
            f"{ct:,} completion "
            f"[{tok_total['reasoning_tokens']:,} reasoning])")
    cost, price = estimate_cost(args.model, pt, ct, cached)
    if cost is not None:
        logline(f"est. cost @ {args.model} pricing "
                f"(in ${price['input']}/cached ${price['cached']}/out "
                f"${price['output']} per 1M): ${cost:.2f} "
                f"({done[0]} samples, ${cost/max(done[0],1):.4f}/sample)")
    else:
        logline(f"est. cost: no pricing entry for {args.model!r} "
                f"(add it to sela/eval/pricing.py)")
    checkpoint()
    plog.close()


def main() -> None:
    p = argparse.ArgumentParser(description="SELA evaluation")
    p.add_argument("--model",       required=True)
    p.add_argument("--dataset",     default="PressureTest")
    p.add_argument("--samples",     type=int, default=None)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=2, dest="max_retries")
    p.add_argument("--numeric-readout", dest="numeric_readout",
                   action="store_true")
    p.add_argument("--numeric-max-rows", dest="numeric_max_rows",
                   type=int, default=40)
    p.add_argument("--resample-mode", dest="resample_mode",
                   choices=["decimate", "envelope"], default="decimate")
    p.add_argument("--oracle-schema-dir", dest="oracle_schema_dir", default=None)
    p.add_argument("--simplified", action="store_true")
    p.add_argument("--reparse-each-sample", dest="reparse_each_sample",
                   action="store_true", default=True)
    p.add_argument("--no-reparse-each-sample", dest="reparse_each_sample",
                   action="store_false")
    p.add_argument("--reasoning-effort", dest="reasoning_effort",
                   choices=["minimal", "low", "medium", "high"], default=None)
    p.add_argument("--parallel-inspectors", dest="parallel_inspectors",
                   action="store_true", default=False)
    p.add_argument("--only", default=None)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
