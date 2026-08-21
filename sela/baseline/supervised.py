from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from sela.eval.metrics import (
    f1_iou, match_counts, coverage_counts, coverage_aggregate,
)


@dataclass
class SupervisedConfig:
    model:             str   = "cnn"
    n_splits:          int   = 5
    epochs:            int   = 60
    window_size:       int   = 1024
    stride:            int   = 256
    seed:              int   = 42
    device:            str   = "cuda" if torch.cuda.is_available() else "cpu"

    cnn_batch_size:    int   = 32
    cnn_lr:            float = 1e-3
    cnn_hidden_dim:    int   = 128
    cnn_kernel_size:   int   = 5
    cnn_dilations:     Tuple[int, ...] = (1, 4, 16)
    cnn_dropout:       float = 0.1

    tr_batch_size:     int   = 32
    tr_lr:             float = 1e-3
    tr_warmup_epochs:  int   = 0
    tr_weight_decay:   float = 0.0
    tr_d_model:        int   = 128
    tr_nhead:          int   = 4
    tr_num_layers:     int   = 2
    tr_ffn:            int   = 256
    tr_dropout:        float = 0.1

    smooth_kernel:     int   = 5
    min_event_dur:     int   = 5


class TrainLogger:

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def znorm_features(dfs: Sequence[pd.DataFrame]) -> List[torch.Tensor]:
    out = []
    for df in dfs:
        vals = df.values.astype(np.float32)
        mean = vals.mean(axis=0, keepdims=True)
        std = vals.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        out.append(torch.tensor((vals - mean) / std))
    return out


def events_to_label_seq(events: List[Dict], seq_len: int) -> torch.Tensor:
    seq = np.zeros(seq_len, dtype=np.int64)
    for ev in events:
        c = int(ev["className"]) + 1
        s = max(0, int(ev["start"]))
        e = min(seq_len, int(ev["end"]))
        if s < e:
            seq[s:e] = c
    return torch.tensor(seq)


def seq_to_events(
    pred_seq: np.ndarray,
    num_classes: int,
    smooth_kernel: int = 0,
    min_event_dur: int = 0,
) -> List[Dict]:
    seq = np.asarray(pred_seq)
    if smooth_kernel > 1:
        seq = seq.copy()
        structure = np.ones(smooth_kernel, dtype=bool)
        for c in range(1, num_classes + 1):
            mask = binary_dilation(seq == c, structure=structure)
            mask = binary_erosion(mask, structure=structure)
            seq[mask] = c
    events: List[Dict] = []
    cur, start = 0, 0
    for i, lab in enumerate(list(seq) + [0]):
        if lab != cur:
            if cur != 0 and (i - start) >= max(min_event_dur, 1):
                events.append({"className": int(cur - 1), "start": start, "end": i})
            cur, start = lab, i
    return events


class SlidingWindowDataset(Dataset):

    def __init__(self, X: List[torch.Tensor], y: List[torch.Tensor],
                 window_size: int, stride: int) -> None:
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for x, lab in zip(X, y):
            n = x.shape[0]
            if n <= window_size:
                self.items.append((x, lab))
                continue
            for s in range(0, n, stride):
                xw, yw = x[s: s + window_size], lab[s: s + window_size]
                if xw.shape[0] > 0:
                    self.items.append((xw, yw))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x_pad, y_pad, lengths


def class_weights(y_windows: List[torch.Tensor], n_total: int) -> torch.Tensor:
    labels = torch.cat(y_windows)
    labels = labels[labels >= 0]
    counts = torch.bincount(labels, minlength=n_total).float()
    w = 1.0 / torch.sqrt(counts + 1e-6)
    return w / w.sum() * n_total


class CNNBaseline(nn.Module):

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int,
                 kernel_size: int, dilations: Sequence[int],
                 dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch_in = input_dim
        for d in dilations:
            pad = (kernel_size - 1) // 2 * d
            layers += [
                nn.Conv1d(ch_in, hidden_dim, kernel_size,
                          padding=pad, dilation=d),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            ch_in = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.classifier(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int,
                 nhead: int, num_layers: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes + 1)

    def forward(self, x: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.pos(self.proj(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.classifier(h)


def _train_epoch(model, loader, criterion, optimizer, device, is_transformer):
    model.train()
    total = 0.0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if is_transformer:
            lengths = lengths.to(device)
            mask = (torch.arange(x.size(1), device=device)
                    .expand(len(lengths), x.size(1)) >= lengths.unsqueeze(1))
            logits = model(x, pad_mask=mask)
        else:
            logits = model(x)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _predict_full(model, x: torch.Tensor, cfg: SupervisedConfig,
                  num_classes: int, is_transformer: bool) -> np.ndarray:
    model.eval()
    device = cfg.device
    n = x.shape[0]
    if not is_transformer:
        logits = model(x.unsqueeze(0).to(device))[0]
        return logits.argmax(-1).cpu().numpy()

    w, s = cfg.window_size, cfg.stride
    n_cls = num_classes + 1
    logit_sum = torch.zeros(n, n_cls)
    count = torch.zeros(n)
    if n <= w:
        logit_sum += model(x.unsqueeze(0).to(device))[0, :n].cpu()
        count += 1
    else:
        for st in range(0, n, s):
            en = min(st + w, n)
            logit_sum[st:en] += model(x[st:en].unsqueeze(0).to(device))[0, : en - st].cpu()
            count[st:en] += 1
    return (logit_sum / count.clamp(min=1).unsqueeze(1)).argmax(-1).numpy()


@torch.no_grad()
def _eval_full(model, X_val, gt_events_val, cfg, num_classes, is_transformer):
    f05, f09 = [], []
    counts = {t: {c: [0, 0, 0] for c in range(num_classes)} for t in (0.5, 0.9)}
    cov_acc = {c: [0.0, 0, 0.0, 0] for c in range(num_classes)}

    for x, gt in zip(X_val, gt_events_val):
        pred_seq = _predict_full(model, x, cfg, num_classes, is_transformer)
        pred = seq_to_events(pred_seq, num_classes,
                             cfg.smooth_kernel, cfg.min_event_dur)
        f05.append(f1_iou(pred, gt, threshold=0.5))
        f09.append(f1_iou(pred, gt, threshold=0.9))
        for thr in (0.5, 0.9):
            for c in range(num_classes):
                p_c = [e for e in pred if int(e["className"]) == c]
                g_c = [e for e in gt if int(e["className"]) == c]
                tp, fp, fn = match_counts(p_c, g_c, threshold=thr)
                acc = counts[thr][c]
                acc[0] += tp; acc[1] += fp; acc[2] += fn
        for c, v in coverage_counts(pred, gt).items():
            s = cov_acc.setdefault(int(c), [0.0, 0, 0.0, 0])
            s[0] += v[0]; s[1] += v[1]; s[2] += v[2]; s[3] += v[3]

    def macro(thr: float) -> float:
        per_cls = []
        for c in range(num_classes):
            tp, fp, fn = counts[thr][c]
            den = 2 * tp + fp + fn
            per_cls.append(2 * tp / den if den > 0 else 0.0)
        return float(np.mean(per_cls))

    cov = coverage_aggregate(cov_acc, list(range(num_classes)))
    return {
        "f05": float(np.mean(f05)), "f09": float(np.mean(f09)),
        "macro05": macro(0.5), "macro09": macro(0.9),
        "cov_f1": cov["macro_cov_f1"], "cov_f05": cov["macro_cov_f05"],
        "micro_cov_f1": cov["micro_cov_f1"], "micro_cov_f05": cov["micro_cov_f05"],
        "per_class_05": {c: tuple(counts[0.5][c]) for c in range(num_classes)},
    }


def run_cv(
    X_dfs:       List[pd.DataFrame],
    y_events:    List[List[Dict]],
    num_classes: int,
    cfg:         SupervisedConfig,
    log:         TrainLogger,
) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X = znorm_features(X_dfs)
    input_dim = X[0].shape[1]
    y_seqs = [events_to_label_seq(ev, x.shape[0]) for ev, x in zip(y_events, X)]
    strata = [ev[0]["className"] if ev else -1 for ev in y_events]

    is_tr = cfg.model == "transformer"
    log.log(f"model={cfg.model}  device={cfg.device}  samples={len(X)}  "
            f"input_dim={input_dim}  classes={num_classes}")
    log.log(f"config: {json.dumps(asdict(cfg))}")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True,
                          random_state=cfg.seed)
    fold_f05, fold_f09, fold_m05, fold_m09 = [], [], [], []
    fold_last_f05, fold_last_m05 = [], []
    fold_cov_f1, fold_cov_f05 = [], []
    fold_last_cov_f1, fold_last_cov_f05 = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(X)), strata), 1):
        t0 = time.time()
        inner = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=cfg.seed + fold)
        tr_strata = [strata[i] for i in tr_idx]
        inner_tr, inner_va = next(inner.split(np.zeros(len(tr_idx)), tr_strata))
        fit_idx = [tr_idx[i] for i in inner_tr]
        val_idx = [tr_idx[i] for i in inner_va]

        tr_X = [X[i] for i in fit_idx]
        tr_y = [y_seqs[i] for i in fit_idx]
        iv_X = [X[i] for i in val_idx]
        iv_gt = [y_events[i] for i in val_idx]
        te_X = [X[i] for i in te_idx]
        te_gt = [y_events[i] for i in te_idx]

        train_ds = SlidingWindowDataset(tr_X, tr_y, cfg.window_size, cfg.stride)
        loader = DataLoader(
            train_ds,
            batch_size=cfg.tr_batch_size if is_tr else cfg.cnn_batch_size,
            shuffle=True, collate_fn=collate)

        weights = class_weights([w for _, w in train_ds.items], num_classes + 1)
        log.log(f"── fold {fold}/{cfg.n_splits}  fit={len(fit_idx)} samples "
                f"({len(train_ds)} windows)  inner-val={len(val_idx)}  "
                f"test={len(te_idx)}  "
                f"class_weights={[f'{w:.2f}' for w in weights.tolist()]}")

        if is_tr:
            model = TransformerBaseline(
                input_dim, num_classes, cfg.tr_d_model, cfg.tr_nhead,
                cfg.tr_num_layers, cfg.tr_ffn, cfg.tr_dropout).to(cfg.device)
            optimizer = optim.AdamW(model.parameters(), lr=cfg.tr_lr,
                                    weight_decay=cfg.tr_weight_decay)
            warmup = max(cfg.tr_warmup_epochs, 1)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.1, total_iters=warmup),
                    optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(cfg.epochs - warmup, 1)),
                ],
                milestones=[warmup],
            )
        else:
            model = CNNBaseline(
                input_dim, num_classes, cfg.cnn_hidden_dim,
                cfg.cnn_kernel_size, cfg.cnn_dilations,
                cfg.cnn_dropout).to(cfg.device)
            optimizer = optim.Adam(model.parameters(), lr=cfg.cnn_lr)
            scheduler = None

        criterion = nn.CrossEntropyLoss(
            ignore_index=-100, weight=weights.to(cfg.device))

        best_iv, best_state, best_epoch = -1.0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            loss = _train_epoch(model, loader, criterion, optimizer,
                                cfg.device, is_tr)
            if scheduler is not None:
                scheduler.step()
            iv = _eval_full(model, iv_X, iv_gt, cfg, num_classes, is_tr)
            marker = ""
            if iv["f05"] > best_iv:
                best_iv, best_epoch = iv["f05"], epoch
                best_state = copy.deepcopy(model.state_dict())
                marker = "  *best"
            log.log(f"fold {fold} epoch {epoch:02d}  loss={loss:.4f}  "
                    f"inner-val F1@0.5={iv['f05']:.4f}  "
                    f"macro@0.5={iv['macro05']:.4f}{marker}")

        last = _eval_full(model, te_X, te_gt, cfg, num_classes, is_tr)
        fold_last_f05.append(last["f05"])
        fold_last_m05.append(last["macro05"])
        fold_last_cov_f1.append(last["cov_f1"])
        fold_last_cov_f05.append(last["cov_f05"])
        model.load_state_dict(best_state)
        m = _eval_full(model, te_X, te_gt, cfg, num_classes, is_tr)
        fold_f05.append(m["f05"])
        fold_f09.append(m["f09"])
        fold_m05.append(m["macro05"])
        fold_m09.append(m["macro09"])
        fold_cov_f1.append(m["cov_f1"])
        fold_cov_f05.append(m["cov_f05"])
        log.log(f"── fold {fold} done in {time.time() - t0:.0f}s  "
                f"selected epoch {best_epoch} (inner-val F1@0.5={best_iv:.4f})")
        log.log(f"   TEST(selected)  F1@0.5={m['f05']:.4f}  F1@0.9={m['f09']:.4f}  "
                f"macro@0.5={m['macro05']:.4f}  macro@0.9={m['macro09']:.4f}  "
                f"per-class(tp,fp,fn)@0.5={m['per_class_05']}")
        log.log(f"   TEST(last-ep)   F1@0.5={last['f05']:.4f}  "
                f"macro@0.5={last['macro05']:.4f}")

    results = {
        "model": cfg.model,
        "fold_f1_05": fold_f05,
        "fold_f1_09": fold_f09,
        "fold_macro_f1_05": fold_m05,
        "fold_macro_f1_09": fold_m09,
        "f1_05_mean": float(np.mean(fold_f05)),
        "f1_05_std":  float(np.std(fold_f05)),
        "f1_09_mean": float(np.mean(fold_f09)),
        "f1_09_std":  float(np.std(fold_f09)),
        "macro_f1_05_mean": float(np.mean(fold_m05)),
        "macro_f1_05_std":  float(np.std(fold_m05)),
        "macro_f1_09_mean": float(np.mean(fold_m09)),
        "macro_f1_09_std":  float(np.std(fold_m09)),
        "cov_f1_mean":  float(np.mean(fold_cov_f1)),
        "cov_f1_std":   float(np.std(fold_cov_f1)),
        "cov_f05_mean": float(np.mean(fold_cov_f05)),
        "cov_f05_std":  float(np.std(fold_cov_f05)),
        "last_f1_05_mean":  float(np.mean(fold_last_f05)),
        "last_f1_05_std":   float(np.std(fold_last_f05)),
        "last_macro_f1_05_mean": float(np.mean(fold_last_m05)),
        "last_macro_f1_05_std":  float(np.std(fold_last_m05)),
        "last_cov_f1_mean":  float(np.mean(fold_last_cov_f1)),
        "last_cov_f05_mean": float(np.mean(fold_last_cov_f05)),
    }
    log.log(f"===== {cfg.model.upper()} {cfg.n_splits}-fold CV =====")
    log.log(f"F1@0.5      = {results['f1_05_mean']:.4f} ± {results['f1_05_std']:.4f}   "
            f"folds: {[f'{v:.3f}' for v in fold_f05]}")
    log.log(f"F1@0.9      = {results['f1_09_mean']:.4f} ± {results['f1_09_std']:.4f}   "
            f"folds: {[f'{v:.3f}' for v in fold_f09]}")
    log.log(f"macroF1@0.5 = {results['macro_f1_05_mean']:.4f} ± {results['macro_f1_05_std']:.4f}   "
            f"folds: {[f'{v:.3f}' for v in fold_m05]}")
    log.log(f"macroF1@0.9 = {results['macro_f1_09_mean']:.4f} ± {results['macro_f1_09_std']:.4f}   "
            f"folds: {[f'{v:.3f}' for v in fold_m09]}")
    log.log(f"macro cov_f1= {results['cov_f1_mean']:.4f} ± {results['cov_f1_std']:.4f}   "
            f"cov_f05 = {results['cov_f05_mean']:.4f} ± {results['cov_f05_std']:.4f}")
    log.log(f"[fixed-budget reference] last-epoch  "
            f"F1@0.5 = {results['last_f1_05_mean']:.4f} ± {results['last_f1_05_std']:.4f}   "
            f"macroF1@0.5 = {results['last_macro_f1_05_mean']:.4f} ± "
            f"{results['last_macro_f1_05_std']:.4f}")
    return results
