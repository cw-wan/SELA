from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from sela.baseline.supervised import (
    SlidingWindowDataset,
    TrainLogger,
    class_weights,
    collate,
    events_to_label_seq,
    seq_to_events,
    znorm_features,
)
from sela.eval.metrics import (
    f1_iou, match_counts, coverage_counts, coverage_aggregate,
)


def _accum_cov(cov_acc, pred, gt):
    for c, v in coverage_counts(pred, gt).items():
        s = cov_acc.setdefault(int(c), [0.0, 0, 0.0, 0])
        s[0] += v[0]; s[1] += v[1]; s[2] += v[2]; s[3] += v[3]

MOMENT_ID   = "AutonLab/MOMENT-1-base"
CHRONOS_ID  = "amazon/chronos-t5-base"
TIMER_ID    = "thuml/timer-base-84m"
TIMESFM_ID  = "google/timesfm-2.5-200m-transformers"


@dataclass
class FMConfig:
    model:          str   = "moment"
    n_splits:       int   = 5
    epochs:         int   = 30
    seed:           int   = 42
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"

    head_mode:      str   = "seq"
    num_queries:    int   = 1
    bbox_cls_cost:  float = 1.0
    bbox_l1_cost:   float = 5.0

    batch_size:     int   = 8
    lr:             float = 1e-3
    moment_lr:      Optional[float] = None
    chronos_lr:     Optional[float] = None
    timer_lr:       Optional[float] = 3e-4
    timesfm_lr:     Optional[float] = 3e-4
    weight_decay:   float = 1e-2
    warmup_epochs:  int   = 2
    stride:         int   = 256
    freeze_backbone: bool = True

    smooth_kernel:  int   = 5
    min_event_dur:  int   = 5

    def probe_lr(self) -> float:
        return getattr(self, f"{self.model}_lr", None) or self.lr

    def window_size(self) -> int:
        return 1024 if self.model == "timer" else 512

    def bbox_ctx(self) -> int:
        return 960 if self.model == "timer" else 512


def _patch_dynamic_cache() -> None:
    from transformers import DynamicCache as DC
    if not hasattr(DC, "from_legacy_cache"):
        @classmethod
        def _flc(cls, past=None):
            cache = cls()
            for i, lp in enumerate(past or []):
                cache.update(lp[0], lp[1], i)
            return cache
        DC.from_legacy_cache = _flc
    if not hasattr(DC, "get_usable_length"):
        DC.get_usable_length = lambda self, n, i=0: self.get_seq_length(i)


class BBoxHead(nn.Module):

    def __init__(self, d_model: int, num_classes: int, num_queries: int,
                 nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.query = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        dec = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=layers)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.cls_head = nn.Linear(d_model, num_classes + 1)

    def forward(self, memory: torch.Tensor):
        q = self.query.unsqueeze(0).expand(memory.shape[0], -1, -1)
        h = self.decoder(q, memory)
        boxes = self.box_head(h).sigmoid()
        logits = self.cls_head(h)
        return boxes, logits


class _FMBase(nn.Module):

    def _build_head(self, d: int, num_classes: int, head_mode: str,
                    num_queries: int, patch: int) -> None:
        self.head_mode = head_mode
        self.num_classes = num_classes
        if head_mode == "bbox":
            self.bbox_head = BBoxHead(d, num_classes, num_queries)
        else:
            if patch == 1:
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, num_classes + 1))
            else:
                self.head = nn.Sequential(
                    nn.LayerNorm(d), nn.Dropout(0.1),
                    nn.Linear(d, patch * (num_classes + 1)))


class MomentSegmenter(_FMBase):
    SEQ_LEN = 512
    PATCH   = 8

    def __init__(self, input_dim, num_classes, freeze_backbone=False,
                 head_mode="seq", num_queries=1) -> None:
        super().__init__()
        from momentfm import MOMENTPipeline
        self.m = MOMENTPipeline.from_pretrained(
            MOMENT_ID, model_kwargs={"task_name": "reconstruction"})
        self.m.init()
        self._build_head(self.m.config.d_model, num_classes, head_mode, num_queries, self.PATCH)
        if freeze_backbone:
            for p in self.m.parameters():
                p.requires_grad = False

    def _encode(self, x):
        B, L, C = x.shape
        xin = x.permute(0, 2, 1)
        mask = torch.ones(B, L, device=x.device)
        xn = self.m.normalizer(x=xin, mask=mask, mode="norm")
        patches = self.m.tokenizer(x=xn)
        enc_in = self.m.patch_embedding(patches, mask=mask)
        nP, d = enc_in.shape[2], enc_in.shape[3]
        h = self.m.encoder(inputs_embeds=enc_in.reshape(B * C, nP, d)).last_hidden_state
        return h.view(B, C, nP, d).mean(dim=1)

    def forward(self, x):
        if self.head_mode == "bbox":
            return self.bbox_head(self._encode(x))
        B, L, C = x.shape
        if L < self.SEQ_LEN:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.SEQ_LEN - L))
        h = self._encode(x[:, : self.SEQ_LEN])
        nP = h.shape[1]
        logits = self.head(h).view(B, nP, self.PATCH, self.num_classes + 1
                                   ).reshape(B, nP * self.PATCH, self.num_classes + 1)
        return logits[:, :L]


class ChronosSegmenter(_FMBase):
    PATCH = 1

    def __init__(self, input_dim, num_classes, freeze_backbone=False,
                 head_mode="seq", num_queries=1) -> None:
        super().__init__()
        from chronos import ChronosPipeline
        self.pipe = ChronosPipeline.from_pretrained(CHRONOS_ID, torch_dtype=torch.float32)
        self.tok = self.pipe.tokenizer
        self.encoder = self.pipe.model.model.encoder
        self._build_head(self.encoder.config.d_model, num_classes, head_mode, num_queries, self.PATCH)
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _encode(self, x):
        B, L, C = x.shape
        per_ch = []
        for c in range(C):
            ids, attn, _s = self.tok.context_input_transform(x[:, :, c].cpu())
            ids, attn = ids.to(x.device), attn.to(x.device)
            h = self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
            per_ch.append(h[:, :L])
        return torch.stack(per_ch, 0).mean(0)

    def forward(self, x):
        h = self._encode(x)
        if self.head_mode == "bbox":
            return self.bbox_head(h)
        return self.head(h)


class TimerSegmenter(_FMBase):
    PATCH = 96

    def __init__(self, input_dim, num_classes, freeze_backbone=False,
                 head_mode="seq", num_queries=1) -> None:
        super().__init__()
        _patch_dynamic_cache()
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(TIMER_ID, trust_remote_code=True)
        self.backbone = full.model
        self._build_head(full.config.hidden_size, num_classes, head_mode, num_queries, self.PATCH)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _encode(self, x):
        B, L, C = x.shape
        xci = x.permute(0, 2, 1).reshape(B * C, L)
        h = self.backbone(input_ids=xci, use_cache=False).last_hidden_state
        d = h.shape[-1]
        return h.view(B, C, L // self.PATCH, d).mean(dim=1)

    def forward(self, x):
        if self.head_mode == "bbox":
            return self.bbox_head(self._encode(x))
        B, L, C = x.shape
        pad = (self.PATCH - L % self.PATCH) % self.PATCH
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        h = self._encode(x)
        nP = h.shape[1]
        logits = self.head(h).view(B, nP, self.PATCH, self.num_classes + 1
                                   ).reshape(B, nP * self.PATCH, self.num_classes + 1)
        return logits[:, :L]


class TimesFmSegmenter(_FMBase):
    PATCH = 32

    def __init__(self, input_dim, num_classes, freeze_backbone=False,
                 head_mode="seq", num_queries=1) -> None:
        super().__init__()
        from transformers import TimesFm2_5ModelForPrediction
        self.m = TimesFm2_5ModelForPrediction.from_pretrained(TIMESFM_ID).to(torch.float32)
        self._build_head(self.m.config.hidden_size, num_classes, head_mode, num_queries, self.PATCH)
        if freeze_backbone:
            for p in self.m.parameters():
                p.requires_grad = False

    def _encode(self, x):
        B, L, C = x.shape
        xci = x.permute(0, 2, 1).reshape(B * C, L)
        h = self.m.model(past_values=xci).last_hidden_state
        nP, d = h.shape[1], h.shape[2]
        return h.view(B, C, nP, d).mean(dim=1)

    def forward(self, x):
        if self.head_mode == "bbox":
            return self.bbox_head(self._encode(x))
        B, L, C = x.shape
        pad = (self.PATCH - L % self.PATCH) % self.PATCH
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        h = self._encode(x)
        nP = h.shape[1]
        logits = self.head(h).view(B, nP, self.PATCH, self.num_classes + 1
                                   ).reshape(B, nP * self.PATCH, self.num_classes + 1)
        return logits[:, :L]


class CnnBBox(nn.Module):

    def __init__(self, input_dim, num_classes, freeze_backbone=False,
                 head_mode="bbox", num_queries=1, hidden=128) -> None:
        super().__init__()
        from sela.baseline.supervised import CNNBaseline
        self.features = CNNBaseline(input_dim, num_classes, hidden, 5,
                                    (1, 4, 16), 0.1).features
        self.bbox_head = BBoxHead(hidden, num_classes, num_queries)

    def forward(self, x):
        h = self.features(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.bbox_head(h)


class TransformerBBox(nn.Module):

    def __init__(self, input_dim, num_classes, freeze_backbone=False,
                 head_mode="bbox", num_queries=1, d_model=64) -> None:
        super().__init__()
        from sela.baseline.supervised import TransformerBaseline
        tr = TransformerBaseline(input_dim, num_classes, d_model, 4, 2, 256, 0.3)
        self.proj, self.pos, self.encoder = tr.proj, tr.pos, tr.encoder
        self.bbox_head = BBoxHead(d_model, num_classes, num_queries)

    def forward(self, x):
        h = self.encoder(self.pos(self.proj(x)))
        return self.bbox_head(h)


def build_model(cfg: FMConfig, input_dim: int, num_classes: int) -> nn.Module:
    kind = {"moment": MomentSegmenter, "chronos": ChronosSegmenter,
            "timer": TimerSegmenter, "timesfm": TimesFmSegmenter,
            "cnn": CnnBBox, "transformer": TransformerBBox}[cfg.model]
    return kind(input_dim, num_classes, cfg.freeze_backbone,
                cfg.head_mode, cfg.num_queries).to(cfg.device)


def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _predict_full(model, x, window, stride, num_classes, device) -> np.ndarray:
    model.eval()
    n = x.shape[0]
    k = num_classes + 1
    logit_sum = torch.zeros(n, k)
    count = torch.zeros(n)
    if n <= window:
        logit_sum += model(x.unsqueeze(0).to(device))[0, :n].float().cpu()
        count += 1
    else:
        for st in range(0, n, stride):
            en = min(st + window, n)
            out = model(x[st:en].unsqueeze(0).to(device))[0, : en - st].float().cpu()
            logit_sum[st:en] += out
            count[st:en] += 1
    return (logit_sum / count.clamp(min=1).unsqueeze(1)).argmax(-1).numpy()


@torch.no_grad()
def _eval_full(model, X_val, gt_val, cfg: FMConfig, num_classes: int) -> Dict[str, Any]:
    f05, f09 = [], []
    counts = {t: {c: [0, 0, 0] for c in range(num_classes)} for t in (0.5, 0.9)}
    cov_acc = {c: [0.0, 0, 0.0, 0] for c in range(num_classes)}
    window = cfg.window_size()
    for x, gt in zip(X_val, gt_val):
        pred_seq = _predict_full(model, x, window, cfg.stride, num_classes, cfg.device)
        pred = seq_to_events(pred_seq, num_classes, cfg.smooth_kernel, cfg.min_event_dur)
        f05.append(f1_iou(pred, gt, threshold=0.5))
        f09.append(f1_iou(pred, gt, threshold=0.9))
        for thr in (0.5, 0.9):
            for c in range(num_classes):
                p_c = [e for e in pred if int(e["className"]) == c]
                g_c = [e for e in gt if int(e["className"]) == c]
                tp, fp, fn = match_counts(p_c, g_c, threshold=thr)
                acc = counts[thr][c]
                acc[0] += tp; acc[1] += fp; acc[2] += fn
        _accum_cov(cov_acc, pred, gt)

    def macro(thr: float) -> float:
        per = []
        for c in range(num_classes):
            tp, fp, fn = counts[thr][c]
            den = 2 * tp + fp + fn
            per.append(2 * tp / den if den > 0 else 0.0)
        return float(np.mean(per))

    cov = coverage_aggregate(cov_acc, list(range(num_classes)))
    return {"f05": float(np.mean(f05)), "f09": float(np.mean(f09)),
            "macro05": macro(0.5), "macro09": macro(0.9),
            "cov_f1": cov["macro_cov_f1"], "cov_f05": cov["macro_cov_f05"],
            "per_class_05": {c: tuple(counts[0.5][c]) for c in range(num_classes)}}


def _resample(x: torch.Tensor, out_len: int) -> torch.Tensor:
    xt = x.permute(1, 0).unsqueeze(0)
    xt = F.interpolate(xt, size=out_len, mode="linear", align_corners=False)
    return xt.squeeze(0).permute(1, 0)


class BBoxDataset(Dataset):

    def __init__(self, X, y_events, ctx_len: int) -> None:
        self.items = []
        for x, evs in zip(X, y_events):
            L0 = x.shape[0]
            xr = _resample(x, ctx_len)
            boxes = torch.tensor(
                [[e["start"] / L0, e["end"] / L0, int(e["className"])] for e in evs],
                dtype=torch.float32).reshape(-1, 3)
            self.items.append((xr, boxes))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _bbox_collate(batch):
    xs, gts = zip(*batch)
    return torch.stack(xs), list(gts)


def _hungarian(boxes, logits, gts, cfg: FMConfig):
    K = boxes.shape[0]
    no_obj = logits.shape[-1] - 1
    tgt = torch.full((K,), no_obj, dtype=torch.long, device=boxes.device)
    matched = []
    if gts.numel() == 0:
        return tgt, matched
    gtb, gtc = gts[:, :2], gts[:, 2].long()
    pb = torch.sort(boxes, dim=-1).values
    cost = (cfg.bbox_l1_cost * torch.cdist(pb, gtb, p=1)
            - cfg.bbox_cls_cost * logits.softmax(-1)[:, gtc])
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    for r, c in zip(row, col):
        tgt[r] = gtc[c]
        matched.append((r, gtb[c]))
    return tgt, matched


def _bbox_loss(boxes, logits, gts_list, cfg: FMConfig, weight=None):
    B = boxes.shape[0]
    cls_loss, box_loss, nb = 0.0, 0.0, 0
    for b in range(B):
        tgt, matched = _hungarian(boxes[b], logits[b], gts_list[b], cfg)
        cls_loss = cls_loss + F.cross_entropy(logits[b], tgt, weight=weight)
        for qi, gtbox in matched:
            box_loss = box_loss + F.l1_loss(torch.sort(boxes[b, qi]).values, gtbox)
            nb += 1
    return cls_loss / B + cfg.bbox_l1_cost * (box_loss / max(nb, 1))


def _bbox_class_weights(y_events, fit_idx, num_classes, device):
    cls = [int(e["className"]) for i in fit_idx for e in y_events[i]]
    cnt = torch.bincount(torch.tensor(cls), minlength=num_classes).float()
    w_ev = 1.0 / torch.sqrt(cnt + 1.0)
    w = torch.cat([w_ev, w_ev.mean().unsqueeze(0)])
    return (w / w.mean()).to(device)


@torch.no_grad()
def _decode_boxes(boxes, logits, orig_len: int, num_classes: int):
    evs, cls = [], logits.argmax(-1)
    for k in range(boxes.shape[0]):
        if int(cls[k]) == num_classes:
            continue
        a, o = torch.sort(boxes[k]).values.tolist()
        s, e = int(round(a * orig_len)), int(round(o * orig_len))
        if e > s:
            evs.append({"className": int(cls[k]), "start": s, "end": e})
    return evs


def _train_epoch_bbox(model, loader, optimizer, cfg: FMConfig, weight=None) -> float:
    model.train()
    total = 0.0
    for xr, gts in loader:
        xr = xr.to(cfg.device)
        gts = [g.to(cfg.device) for g in gts]
        optimizer.zero_grad()
        boxes, logits = model(xr)
        loss = _bbox_loss(boxes, logits, gts, cfg, weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _eval_bbox(model, X_val, gt_val, cfg: FMConfig, num_classes: int) -> Dict[str, Any]:
    model.eval()
    ctx = cfg.bbox_ctx()
    f05, f09 = [], []
    counts = {t: {c: [0, 0, 0] for c in range(num_classes)} for t in (0.5, 0.9)}
    cov_acc = {c: [0.0, 0, 0.0, 0] for c in range(num_classes)}
    for x, gt in zip(X_val, gt_val):
        xr = _resample(x, ctx).unsqueeze(0).to(cfg.device)
        boxes, logits = model(xr)
        pred = _decode_boxes(boxes[0], logits[0], x.shape[0], num_classes)
        f05.append(f1_iou(pred, gt, threshold=0.5))
        f09.append(f1_iou(pred, gt, threshold=0.9))
        for thr in (0.5, 0.9):
            for c in range(num_classes):
                p_c = [e for e in pred if int(e["className"]) == c]
                g_c = [e for e in gt if int(e["className"]) == c]
                tp, fp, fn = match_counts(p_c, g_c, threshold=thr)
                acc = counts[thr][c]; acc[0] += tp; acc[1] += fp; acc[2] += fn
        _accum_cov(cov_acc, pred, gt)

    def macro(thr):
        per = []
        for c in range(num_classes):
            tp, fp, fn = counts[thr][c]; den = 2 * tp + fp + fn
            per.append(2 * tp / den if den > 0 else 0.0)
        return float(np.mean(per))

    cov = coverage_aggregate(cov_acc, list(range(num_classes)))
    return {"f05": float(np.mean(f05)), "f09": float(np.mean(f09)),
            "macro05": macro(0.5), "macro09": macro(0.9),
            "cov_f1": cov["macro_cov_f1"], "cov_f05": cov["macro_cov_f05"],
            "per_class_05": {c: tuple(counts[0.5][c]) for c in range(num_classes)}}


def _run_cv_bbox(X_dfs, y_events, num_classes, cfg: FMConfig, log: TrainLogger) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    X = znorm_features(X_dfs)
    input_dim = X[0].shape[1]
    strata = [ev[0]["className"] if ev else -1 for ev in y_events]
    ctx = cfg.bbox_ctx()
    log.log(f"model={cfg.model}  head=bbox  queries={cfg.num_queries}  device={cfg.device}  "
            f"samples={len(X)}  input_dim={input_dim}  classes={num_classes}  ctx={ctx}")
    log.log(f"config: {json.dumps(asdict(cfg))}")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    f05s, f09s, m05s, m09s, last_f05s, last_m05s = [], [], [], [], [], []
    cf1s, cf05s, last_cf1s, last_cf05s = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(X)), strata), 1):
        t0 = time.time()
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed + fold)
        itr, iva = next(inner.split(np.zeros(len(tr_idx)), [strata[i] for i in tr_idx]))
        fit_idx = [tr_idx[i] for i in itr]; val_idx = [tr_idx[i] for i in iva]

        ds = BBoxDataset([X[i] for i in fit_idx], [y_events[i] for i in fit_idx], ctx)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_bbox_collate)
        iv_X, iv_gt = [X[i] for i in val_idx], [y_events[i] for i in val_idx]
        te_X, te_gt = [X[i] for i in te_idx], [y_events[i] for i in te_idx]
        weight = _bbox_class_weights(y_events, fit_idx, num_classes, cfg.device)
        log.log(f"── fold {fold}/{cfg.n_splits}  fit={len(fit_idx)}  inner-val={len(val_idx)}  "
                f"test={len(te_idx)}  cls_weights={[f'{w:.2f}' for w in weight.tolist()]}")

        model = build_model(cfg, input_dim, num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.probe_lr(), weight_decay=cfg.weight_decay)
        warmup = max(cfg.warmup_epochs, 1)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - warmup, 1))],
            milestones=[warmup])

        best_iv, best_state, best_ep = -1.0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            loss = _train_epoch_bbox(model, loader, optimizer, cfg, weight)
            scheduler.step()
            iv = _eval_bbox(model, iv_X, iv_gt, cfg, num_classes)
            mark = ""
            if iv["f05"] > best_iv:
                best_iv, best_ep = iv["f05"], epoch
                best_state = copy.deepcopy(model.state_dict()); mark = "  *best"
            log.log(f"fold {fold} epoch {epoch:02d}  loss={loss:.4f}  "
                    f"inner-val F1@0.5={iv['f05']:.4f}  macro@0.5={iv['macro05']:.4f}{mark}")

        last = _eval_bbox(model, te_X, te_gt, cfg, num_classes)
        last_f05s.append(last["f05"]); last_m05s.append(last["macro05"])
        last_cf1s.append(last["cov_f1"]); last_cf05s.append(last["cov_f05"])
        model.load_state_dict(best_state)
        m = _eval_bbox(model, te_X, te_gt, cfg, num_classes)
        f05s.append(m["f05"]); f09s.append(m["f09"]); m05s.append(m["macro05"]); m09s.append(m["macro09"])
        cf1s.append(m["cov_f1"]); cf05s.append(m["cov_f05"])
        log.log(f"── fold {fold} done {time.time() - t0:.0f}s  selected epoch {best_ep} (inner-val F1@0.5={best_iv:.4f})")
        log.log(f"   TEST(selected)  F1@0.5={m['f05']:.4f}  F1@0.9={m['f09']:.4f}  "
                f"macro@0.5={m['macro05']:.4f}  macro@0.9={m['macro09']:.4f}  per-class(tp,fp,fn)@0.5={m['per_class_05']}")
        log.log(f"   TEST(last-ep)   F1@0.5={last['f05']:.4f}  macro@0.5={last['macro05']:.4f}")

    def ms(v): return float(np.mean(v)), float(np.std(v))
    res = {"model": cfg.model, "head_mode": "bbox", "num_queries": cfg.num_queries,
           "fold_f1_05": f05s, "fold_f1_09": f09s, "fold_macro_f1_05": m05s, "fold_macro_f1_09": m09s,
           "f1_05_mean": ms(f05s)[0], "f1_05_std": ms(f05s)[1], "f1_09_mean": ms(f09s)[0], "f1_09_std": ms(f09s)[1],
           "macro_f1_05_mean": ms(m05s)[0], "macro_f1_05_std": ms(m05s)[1],
           "macro_f1_09_mean": ms(m09s)[0], "macro_f1_09_std": ms(m09s)[1],
           "last_f1_05_mean": ms(last_f05s)[0], "last_f1_05_std": ms(last_f05s)[1],
           "last_macro_f1_05_mean": ms(last_m05s)[0], "last_macro_f1_05_std": ms(last_m05s)[1],
           "cov_f1_mean": ms(cf1s)[0], "cov_f1_std": ms(cf1s)[1],
           "cov_f05_mean": ms(cf05s)[0], "cov_f05_std": ms(cf05s)[1],
           "last_cov_f1_mean": ms(last_cf1s)[0], "last_cov_f05_mean": ms(last_cf05s)[0]}
    log.log(f"===== {cfg.model.upper()} (bbox, K={cfg.num_queries}) {cfg.n_splits}-fold CV =====")
    log.log(f"F1@0.5      = {res['f1_05_mean']:.4f} ± {res['f1_05_std']:.4f}   folds: {[f'{v:.3f}' for v in f05s]}")
    log.log(f"F1@0.9      = {res['f1_09_mean']:.4f} ± {res['f1_09_std']:.4f}   folds: {[f'{v:.3f}' for v in f09s]}")
    log.log(f"macroF1@0.5 = {res['macro_f1_05_mean']:.4f} ± {res['macro_f1_05_std']:.4f}   folds: {[f'{v:.3f}' for v in m05s]}")
    log.log(f"macroF1@0.9 = {res['macro_f1_09_mean']:.4f} ± {res['macro_f1_09_std']:.4f}   folds: {[f'{v:.3f}' for v in m09s]}")
    log.log(f"macro cov_f1= {res['cov_f1_mean']:.4f} ± {res['cov_f1_std']:.4f}   cov_f05 = {res['cov_f05_mean']:.4f} ± {res['cov_f05_std']:.4f}")
    log.log(f"[fixed-budget reference] last-epoch  F1@0.5 = {res['last_f1_05_mean']:.4f} ± {res['last_f1_05_std']:.4f}   "
            f"macroF1@0.5 = {res['last_macro_f1_05_mean']:.4f} ± {res['last_macro_f1_05_std']:.4f}")
    return res


def run_cv(X_dfs, y_events, num_classes, cfg: FMConfig, log: TrainLogger) -> Dict[str, Any]:
    if cfg.head_mode == "bbox":
        return _run_cv_bbox(X_dfs, y_events, num_classes, cfg, log)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X = znorm_features(X_dfs)
    input_dim = X[0].shape[1]
    y_seqs = [events_to_label_seq(ev, x.shape[0]) for ev, x in zip(y_events, X)]
    strata = [ev[0]["className"] if ev else -1 for ev in y_events]
    window = cfg.window_size()

    log.log(f"model={cfg.model}  device={cfg.device}  samples={len(X)}  "
            f"input_dim={input_dim}  classes={num_classes}  window={window}")
    log.log(f"config: {json.dumps(asdict(cfg))}")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    f05s, f09s, m05s, m09s, last_f05s, last_m05s = [], [], [], [], [], []
    cf1s, cf05s, last_cf1s, last_cf05s = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(X)), strata), 1):
        t0 = time.time()
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed + fold)
        inner_tr, inner_va = next(inner.split(np.zeros(len(tr_idx)),
                                              [strata[i] for i in tr_idx]))
        fit_idx = [tr_idx[i] for i in inner_tr]
        val_idx = [tr_idx[i] for i in inner_va]

        tr_ds = SlidingWindowDataset([X[i] for i in fit_idx], [y_seqs[i] for i in fit_idx],
                                     window, cfg.stride)
        loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
        iv_X, iv_gt = [X[i] for i in val_idx], [y_events[i] for i in val_idx]
        te_X, te_gt = [X[i] for i in te_idx], [y_events[i] for i in te_idx]

        weights = class_weights([w for _, w in tr_ds.items], num_classes + 1)
        log.log(f"── fold {fold}/{cfg.n_splits}  fit={len(fit_idx)} ({len(tr_ds)} win)  "
                f"inner-val={len(val_idx)}  test={len(te_idx)}  "
                f"weights={[f'{w:.2f}' for w in weights.tolist()]}")

        model = build_model(cfg, input_dim, num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.probe_lr(), weight_decay=cfg.weight_decay)
        warmup = max(cfg.warmup_epochs, 1)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - warmup, 1))],
            milestones=[warmup])
        criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weights.to(cfg.device))

        best_iv, best_state, best_ep = -1.0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            loss = _train_epoch(model, loader, criterion, optimizer, cfg.device)
            scheduler.step()
            iv = _eval_full(model, iv_X, iv_gt, cfg, num_classes)
            mark = ""
            if iv["f05"] > best_iv:
                best_iv, best_ep = iv["f05"], epoch
                best_state = copy.deepcopy(model.state_dict())
                mark = "  *best"
            log.log(f"fold {fold} epoch {epoch:02d}  loss={loss:.4f}  "
                    f"inner-val F1@0.5={iv['f05']:.4f}  macro@0.5={iv['macro05']:.4f}{mark}")

        last = _eval_full(model, te_X, te_gt, cfg, num_classes)
        last_f05s.append(last["f05"]); last_m05s.append(last["macro05"])
        last_cf1s.append(last["cov_f1"]); last_cf05s.append(last["cov_f05"])
        model.load_state_dict(best_state)
        m = _eval_full(model, te_X, te_gt, cfg, num_classes)
        f05s.append(m["f05"]); f09s.append(m["f09"]); m05s.append(m["macro05"]); m09s.append(m["macro09"])
        cf1s.append(m["cov_f1"]); cf05s.append(m["cov_f05"])
        log.log(f"── fold {fold} done {time.time() - t0:.0f}s  selected epoch {best_ep} "
                f"(inner-val F1@0.5={best_iv:.4f})")
        log.log(f"   TEST(selected)  F1@0.5={m['f05']:.4f}  F1@0.9={m['f09']:.4f}  "
                f"macro@0.5={m['macro05']:.4f}  macro@0.9={m['macro09']:.4f}  "
                f"per-class(tp,fp,fn)@0.5={m['per_class_05']}")
        log.log(f"   TEST(last-ep)   F1@0.5={last['f05']:.4f}  macro@0.5={last['macro05']:.4f}")

    def ms(v): return float(np.mean(v)), float(np.std(v))
    res = {
        "model": cfg.model,
        "fold_f1_05": f05s, "fold_f1_09": f09s,
        "fold_macro_f1_05": m05s, "fold_macro_f1_09": m09s,
        "f1_05_mean": ms(f05s)[0], "f1_05_std": ms(f05s)[1],
        "f1_09_mean": ms(f09s)[0], "f1_09_std": ms(f09s)[1],
        "macro_f1_05_mean": ms(m05s)[0], "macro_f1_05_std": ms(m05s)[1],
        "macro_f1_09_mean": ms(m09s)[0], "macro_f1_09_std": ms(m09s)[1],
        "last_f1_05_mean": ms(last_f05s)[0], "last_f1_05_std": ms(last_f05s)[1],
        "last_macro_f1_05_mean": ms(last_m05s)[0], "last_macro_f1_05_std": ms(last_m05s)[1],
        "cov_f1_mean": ms(cf1s)[0], "cov_f1_std": ms(cf1s)[1],
        "cov_f05_mean": ms(cf05s)[0], "cov_f05_std": ms(cf05s)[1],
        "last_cov_f1_mean": ms(last_cf1s)[0], "last_cov_f05_mean": ms(last_cf05s)[0],
    }
    log.log(f"===== {cfg.model.upper()} {cfg.n_splits}-fold CV =====")
    log.log(f"F1@0.5      = {res['f1_05_mean']:.4f} ± {res['f1_05_std']:.4f}   folds: {[f'{v:.3f}' for v in f05s]}")
    log.log(f"F1@0.9      = {res['f1_09_mean']:.4f} ± {res['f1_09_std']:.4f}   folds: {[f'{v:.3f}' for v in f09s]}")
    log.log(f"macroF1@0.5 = {res['macro_f1_05_mean']:.4f} ± {res['macro_f1_05_std']:.4f}   folds: {[f'{v:.3f}' for v in m05s]}")
    log.log(f"macroF1@0.9 = {res['macro_f1_09_mean']:.4f} ± {res['macro_f1_09_std']:.4f}   folds: {[f'{v:.3f}' for v in m09s]}")
    log.log(f"macro cov_f1= {res['cov_f1_mean']:.4f} ± {res['cov_f1_std']:.4f}   cov_f05 = {res['cov_f05_mean']:.4f} ± {res['cov_f05_std']:.4f}")
    log.log(f"[fixed-budget reference] last-epoch  F1@0.5 = {res['last_f1_05_mean']:.4f} ± {res['last_f1_05_std']:.4f}   "
            f"macroF1@0.5 = {res['last_macro_f1_05_mean']:.4f} ± {res['last_macro_f1_05_std']:.4f}")
    return res
