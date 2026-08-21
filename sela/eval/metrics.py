"""Temporal event detection metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union


def interval_iou(a, b) -> float:
    """Intersection-over-Union of two 1-D closed intervals."""
    x1, y1 = a
    x2, y2 = b
    if x1 > y1 or x2 > y2:
        raise ValueError("Each interval must satisfy start <= end")

    inter_len = max(0.0, min(y1, y2) - max(x1, x2))
    len1, len2 = y1 - x1, y2 - x2

    if len1 == 0 and len2 == 0:
        return 1.0 if x1 == x2 else 0.0

    union_len = len1 + len2 - inter_len
    return inter_len / union_len


def match_counts(
    pred: List[Dict[str, Any]],
    gt:   List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """(tp, fp, fn) under the same greedy max-IoU, same-class matching as f1_iou."""
    gt_copy = [dict(seg, matched=False) for seg in gt]
    tp = 0
    for p in pred:
        try:
            p_cls = str(p["className"]).casefold()
            p_iv  = (int(p["start"]), int(p["end"]))
            best_idx: Optional[int] = None
            best_iou = -1.0
            for i, g in enumerate(gt_copy):
                if g["matched"] or str(g["className"]).casefold() != p_cls:
                    continue
                iou = interval_iou((int(g["start"]), int(g["end"])), p_iv)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx is not None and best_iou >= threshold:
                tp += 1
                gt_copy[best_idx]["matched"] = True
        except (KeyError, ValueError, TypeError):
            print(f"[metrics] bad segment: {p}")
    fp = len(pred) - tp
    fn = sum(1 for g in gt_copy if not g["matched"])
    return tp, fp, fn


def f1_iou(
    pred: List[Dict[str, Any]],
    gt:   List[Dict[str, Any]],
    threshold: float = 0.5,
    compute_drift: bool = False,
) -> Union[float, Tuple[float, List[float]]]:
    """Greedy one-to-one matching F1 for temporal segments (MAX-IoU assignment)."""
    gt_copy = [dict(seg, matched=False) for seg in gt]
    tp = 0
    drifts: List[float] = []

    for p in pred:
        try:
            p_cls = str(p["className"]).casefold()
            p_iv  = (int(p["start"]), int(p["end"]))
            best_idx: Optional[int] = None
            best_iou = -1.0

            for i, g in enumerate(gt_copy):
                if g["matched"]:
                    continue
                if str(g["className"]).casefold() != p_cls:
                    continue
                iou = interval_iou((int(g["start"]), int(g["end"])), p_iv)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx is not None and best_iou >= threshold:
                tp += 1
                gt_copy[best_idx]["matched"] = True
                if compute_drift:
                    g = gt_copy[best_idx]
                    span = max(g["end"] - g["start"], 1)
                    drift = (abs(p["start"] - g["start"]) + abs(p["end"] - g["end"])) / 2 / span
                    drifts.append(float(drift))

        except (KeyError, ValueError, TypeError):
            print(f"[metrics] bad segment: {p}")

    fp  = len(pred) - tp
    fn  = sum(1 for g in gt_copy if not g["matched"])
    den = 2 * tp + fp + fn
    f1  = 0.0 if den == 0 else 2 * tp / den
    return (f1, drifts) if compute_drift else f1


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Length of the intersection of two closed intervals (0 if disjoint)."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def coverage_pair(g: Dict[str, Any], p: Dict[str, Any]) -> Tuple[float, float]:
    """(C_g, C_p) = (range recall, range precision) for one GT/pred pair."""
    s_g, e_g = int(g["start"]), int(g["end"])
    s_p, e_p = int(p["start"]), int(p["end"])
    o = _overlap((s_g, e_g), (s_p, e_p))
    gl, pl = float(e_g - s_g), float(e_p - s_p)
    c_g = (o / gl) if gl > 0 else (1.0 if s_p <= s_g <= e_p else 0.0)
    c_p = (o / pl) if pl > 0 else (1.0 if s_g <= s_p <= e_p else 0.0)
    return c_g, c_p


def fbeta_coverage(precision: float, recall: float, beta: float = 1.0) -> float:
    """F_beta of coverage precision (C_p) and recall (C_g); beta<1 favours precision."""
    b2 = beta * beta
    den = b2 * precision + recall
    return (1.0 + b2) * precision * recall / den if den > 0 else 0.0


def coverage_counts(
    pred: List[Dict[str, Any]],
    gt:   List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Per-class coverage accumulators for ONE sample — the coverage analogue of"""
    pred = [p for p in pred
            if p.get("start") is not None and p.get("end") is not None]

    cand: List[Tuple[float, int, int, float, float]] = []
    for gi, g in enumerate(gt):
        gc = str(g["className"]).casefold()
        for pi, p in enumerate(pred):
            if str(p["className"]).casefold() != gc:
                continue
            if _overlap((int(g["start"]), int(g["end"])),
                        (int(p["start"]), int(p["end"]))) <= 0:
                continue
            c_g, c_p = coverage_pair(g, p)
            d = c_g + c_p
            cand.append((2 * c_g * c_p / d if d > 0 else 0.0, gi, pi, c_g, c_p))
    cand.sort(key=lambda x: x[0], reverse=True)

    acc: Dict[Any, List[float]] = {}

    def _slot(cls: Any) -> List[float]:
        return acc.setdefault(cls, [0.0, 0, 0.0, 0])

    used_g: set = set()
    used_p: set = set()
    for _, gi, pi, c_g, c_p in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        s = _slot(gt[gi]["className"])
        s[0] += c_g
        s[2] += c_p
    for g in gt:
        _slot(g["className"])[1] += 1
    for p in pred:
        _slot(p["className"])[3] += 1
    return acc


def coverage_aggregate(
    cov: Dict[str, List[float]],
    classes: List[str],
) -> Dict[str, Any]:
    """Finalise pooled per-class coverage accumulators (from ``coverage_counts``)"""
    zero = [0.0, 0, 0.0, 0]
    per_f1: Dict[str, float] = {}
    per_f05: Dict[str, float] = {}
    for c in classes:
        gs, gn, ps, pn = cov.get(c, zero)
        rec  = gs / gn if gn > 0 else 0.0
        prec = ps / pn if pn > 0 else 0.0
        per_f1[c]  = fbeta_coverage(prec, rec, 1.0)
        per_f05[c] = fbeta_coverage(prec, rec, 0.5)

    gs = sum(cov.get(c, zero)[0] for c in classes)
    gn = sum(cov.get(c, zero)[1] for c in classes)
    ps = sum(cov.get(c, zero)[2] for c in classes)
    pn = sum(cov.get(c, zero)[3] for c in classes)
    rec_mu  = gs / gn if gn > 0 else 0.0
    prec_mu = ps / pn if pn > 0 else 0.0

    n = len(classes)
    return {
        "micro_cov_f1":  fbeta_coverage(prec_mu, rec_mu, 1.0),
        "macro_cov_f1":  sum(per_f1.values()) / n if n else 0.0,
        "micro_cov_f05": fbeta_coverage(prec_mu, rec_mu, 0.5),
        "macro_cov_f05": sum(per_f05.values()) / n if n else 0.0,
        "per_class_cov_f1":  per_f1,
        "per_class_cov_f05": per_f05,
    }


def coverage_metrics(
    preds: List[List[Dict[str, Any]]],
    gts:   List[List[Dict[str, Any]]],
    classes: List[str],
) -> Dict[str, Any]:
    """Convenience one-shot: aligned per-sample ``preds`` / ``gts`` lists →"""
    acc: Dict[str, List[float]] = {c: [0.0, 0, 0.0, 0] for c in classes}
    for p, g in zip(preds, gts):
        for c, v in coverage_counts(p, g).items():
            s = acc.setdefault(c, [0.0, 0, 0.0, 0])
            s[0] += v[0]
            s[1] += v[1]
            s[2] += v[2]
            s[3] += v[3]
    return coverage_aggregate(acc, classes)
