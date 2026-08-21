"""Event Logic Tree (ELT) tool backend."""
from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sela.mas.tool_backend import ToolBackend, ToolResult

logger = logging.getLogger(__name__)

_V_SURFACE  = "#fcfcfb"
_V_INK      = "#0b0b0b"
_V_INK2     = "#52514e"
_V_MUTED    = "#898781"
_V_GRID     = "#e1e0d9"
_V_BASELINE = "#c3c2b7"
_V_SERIES   = [
    "#2a78d6", "#008300", "#e87ba4", "#eda100",
    "#1baf7a", "#eb6834", "#4a3aa7", "#e34948",
]
_V_OP = {
    "SEQ": "#4a3aa7", "SYNC": "#1baf7a", "GUARD": "#eb6834", "OR": "#eda100",
}


class Operator(str, Enum):
    SEQ   = "SEQ"
    SYNC  = "SYNC"
    GUARD = "GUARD"
    OR    = "OR"


DEFAULT_PARAMS: Dict[str, float] = {
    "lambda": 4.0,
    "mu":     4.0,
    "kappa":  0.4,
    "sigma":  4.0,
    "eta":    4.0,
    "omega":  4.0,
    "tau":    0.05,
}


@dataclass
class PrimitiveDef:
    alias:          str
    target_channel: str
    description:    str = ""


@dataclass
class CompositeDef:
    alias:       str
    operator:    Operator
    children:    List[str]
    description: str         = ""
    params:      Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))


def _interval_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """IoU of two closed intervals; 0 if disjoint or degenerate."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PrimitiveInstance:
    alias:      str
    start:      int
    end:        int
    confidence: float = 1.0
    channel:    str   = ""


@dataclass
class GhostInstance:
    """Placeholder for a missing primitive (keeps the tree shape intact)."""
    alias:      str
    start:      int = -1
    end:        int = -1
    confidence: float = 0.0


@dataclass
class CompositeInstance:
    alias:    str
    operator: Operator
    children: List["InstanceNode"]
    params:   Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))

    @property
    def is_ghost(self) -> bool:
        """A composite is a ghost when it has no active (non-ghost) leaves."""
        return len(_active_leaves(self)) == 0

    @property
    def start(self) -> int:
        leaves = _active_leaves(self)
        return min(l.start for l in leaves) if leaves else 0

    @property
    def end(self) -> int:
        leaves = _active_leaves(self)
        return max(l.end for l in leaves) if leaves else 0

    @property
    def confidence(self) -> float:
        if len(self.children) < 2:
            if not self.children:
                return 0.0
            c = self.children[0]
            return 0.0 if _is_ghost(c) else c.confidence

        A, B = self.children[0], self.children[1]
        a_ghost, b_ghost = _is_ghost(A), _is_ghost(B)
        s_A, s_B = A.confidence, B.confidence
        p = self.params
        t_A, t_B = _interval(A), _interval(B)

        if self.operator == Operator.OR:
            if a_ghost and b_ghost:
                return 0.0
            if a_ghost:
                return s_B
            if b_ghost:
                return s_A
            span = max(1, max(t_A[1], t_B[1]) - min(t_A[0], t_B[0]))
            diff = (abs(t_A[0] - t_B[0]) + abs(t_A[1] - t_B[1])) / span
            return max(s_A, s_B) * math.exp(-p["eta"] * diff)

        if a_ghost or b_ghost:
            return 0.0

        dur_A = max(1, t_A[1] - t_A[0])
        dur_B = max(1, t_B[1] - t_B[0])
        scale = dur_A + dur_B

        overlap    = _channel_overlap(A, B)
        over_ratio = overlap / min(dur_A, dur_B)
        p_coll     = math.exp(-p["omega"] * max(0.0, over_ratio - p["tau"]))

        if self.operator == Operator.SEQ:
            gap    = max(0.0, t_B[0] - t_A[1]) / scale
            causal = max(0.0, t_A[0] - t_B[0]) / scale
            p_struct = math.exp(-p["lambda"] * gap - p["mu"] * causal)
            return s_A * s_B * p_struct * p_coll

        if self.operator == Operator.SYNC:
            iou = _interval_iou(t_A, t_B)
            return s_A * s_B * math.exp(-(1.0 - iou) / p["kappa"]) * p_coll

        if self.operator == Operator.GUARD:
            overflow = (max(0.0, t_A[0] - t_B[0]) +
                        max(0.0, t_B[1] - t_A[1])) / scale
            return s_A * s_B * math.exp(-p["sigma"] * overflow) * p_coll

        return 0.0


InstanceNode = PrimitiveInstance | CompositeInstance | GhostInstance


def _is_ghost(node: InstanceNode) -> bool:
    if isinstance(node, GhostInstance):
        return True
    if isinstance(node, PrimitiveInstance):
        return False
    return node.is_ghost


def _active_leaves(node: InstanceNode) -> List[PrimitiveInstance]:
    """Return the primitive leaves that actually contribute to this node's"""
    if isinstance(node, GhostInstance):
        return []
    if isinstance(node, PrimitiveInstance):
        return [node]
    if node.operator == Operator.OR:
        cands = [c for c in node.children if not _is_ghost(c)]
        if not cands:
            return []
        best = max(cands, key=lambda c: c.confidence)
        return _active_leaves(best)
    out: List[PrimitiveInstance] = []
    for c in node.children:
        out.extend(_active_leaves(c))
    return out


def _interval(node: InstanceNode) -> Tuple[int, int]:
    """Union hull of a node's active leaves (its temporal footprint)."""
    if isinstance(node, PrimitiveInstance):
        return (node.start, node.end)
    if isinstance(node, GhostInstance):
        return (-1, -1)
    leaves = _active_leaves(node)
    if not leaves:
        return (-1, -1)
    return (min(l.start for l in leaves), max(l.end for l in leaves))


def _channel_overlap(a: InstanceNode, b: InstanceNode) -> float:
    """Total temporal overlap between same-channel leaves of two sub-trees."""
    total = 0.0
    for la in _active_leaves(a):
        for lb in _active_leaves(b):
            if la.channel and la.channel == lb.channel:
                overlap = min(la.end, lb.end) - max(la.start, lb.start)
                if overlap > 0:
                    total += overlap
    return total


def _count_active_leaves(node: InstanceNode) -> int:
    """K(T̂) = |L*(T̂)| — number of active primitive leaves contributing to μ_root."""
    return len(_active_leaves(node))


def normalized_confidence(root: CompositeInstance) -> float:
    """μ_norm(T̂) = μ_root^(1/K)  where K = |L*(T̂)| = active primitive leaves."""
    K = _count_active_leaves(root)
    if K == 0:
        return 0.0
    conf = root.confidence
    if conf <= 0.0:
        return 0.0
    return conf ** (1.0 / K)


def find_main_instance(root: CompositeInstance) -> Optional[InstanceNode]:
    """Find the node whose alias is exactly 'Main' (case-insensitive) in the tree."""
    def _search(node: InstanceNode) -> Optional[InstanceNode]:
        if not isinstance(node, CompositeInstance):
            return None
        for child in node.children:
            if child.alias.lower() == "main" and not _is_ghost(child):
                return child
        for child in node.children:
            found = _search(child)
            if found is not None:
                return found
        return None

    if root.alias.lower() == "main":
        return root
    return _search(root)


class ELT:
    """Event Logic Tree: compile a schema, add candidate instances, read confidence."""

    def __init__(self, df: pd.DataFrame, require_main: bool = False,
                 label: str = "", numeric_readout: bool = False,
                 numeric_max_rows: int = 40,
                 resample_mode: str = "decimate",
                 main_only: bool = False) -> None:
        self._df         = df.reset_index(drop=True)
        self._numeric_readout  = numeric_readout
        self._numeric_max_rows = max(2, int(numeric_max_rows))
        self._resample_mode    = resample_mode if resample_mode in (
            "decimate", "envelope") else "decimate"
        self._primitives: Dict[str, PrimitiveDef]  = {}
        self._composites: Dict[str, CompositeDef]  = {}
        self._root:       Optional[str]            = None
        self._candidates: Dict[str, PrimitiveInstance] = {}
        self._compiled   = False
        self._require_main = require_main
        self._main_only = main_only
        self.label = label
        self._placements: List[Tuple[int, int]] = []


    def load_schema(self, schema_dict: Dict[str, Any], root_alias: str) -> str:
        """Load a schema dict directly (skips image rendering; for orchestrators)."""
        result = self.submit_schema(
            primitives=schema_dict.get("primitives", []),
            composites=schema_dict.get("composites", []),
            root_alias=root_alias,
        )
        return result.text


    def define_primitive(
        self, alias: str, target_channel: str, description: str = ""
    ) -> None:
        self._primitives[alias] = PrimitiveDef(alias, target_channel, description)
        self._compiled = False

    def define_composite(
        self,
        alias:       str,
        operator:    Operator,
        children:    List[str],
        description: str         = "",
    ) -> None:
        self._composites[alias] = CompositeDef(alias, operator, children, description)
        self._compiled = False

    def compile(self, root_alias: str) -> str:
        """Validate the tree and set the root alias. Returns status message."""
        if root_alias not in self._primitives and root_alias not in self._composites:
            return f"Error: root alias '{root_alias}' not defined."
        errors = self._validate(root_alias)
        if errors:
            return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        self._root    = root_alias
        self._compiled = True
        return (
            f"Schema compiled. Root: '{root_alias}'. "
            f"Primitives: {list(self._primitives)}. "
            f"Composites: {list(self._composites)}."
        )

    def _validate(self, root_alias: str) -> List[str]:
        """Enforce the ELT structural axioms (ported from the original ELT):"""
        errors: List[str] = []
        prim_set    = set(self._primitives)
        comp_set    = set(self._composites)
        all_aliases = prim_set | comp_set

        for a in sorted(prim_set & comp_set):
            errors.append(
                f"Alias '{a}' is defined as BOTH a primitive and a composite — "
                f"aliases must be unique across the schema."
            )

        columns = list(self._df.columns)
        if columns:
            for alias, p in self._primitives.items():
                if p.target_channel not in columns:
                    errors.append(
                        f"Primitive '{alias}': channel '{p.target_channel}' "
                        f"does not exist in the data. Available channels: "
                        f"{columns}."
                    )

        if self._require_main:
            lowered = {a.lower() for a in all_aliases}
            need = ("main",) if self._main_only else ("pre", "main", "post")
            for required in need:
                if required not in lowered:
                    errors.append(
                        f"Missing required '{required.capitalize()}' node: the "
                        f"schema must contain nodes named exactly 'Pre', 'Main' "
                        f"and 'Post'. 'Main' carries the final event interval; "
                        f"'Pre' and 'Post' encode the context/termination "
                        f"evidence that anchors the event location. Add the "
                        f"missing node and resubmit."
                    )

        child_parent: Dict[str, str] = {}
        for alias, comp in self._composites.items():
            if len(comp.children) != 2:
                errors.append(
                    f"Composite '{alias}' has {len(comp.children)} child(ren) — "
                    f"every composite must have EXACTLY 2. Encode SEQ(A, B, C) "
                    f"as SEQ(SEQ(A, B), C), OR(A, B, C) as OR(OR(A, B), C)."
                )
            if len(set(comp.children)) != len(comp.children):
                errors.append(
                    f"Composite '{alias}' lists the same child twice — the two "
                    f"children must be distinct nodes."
                )
            for child in comp.children:
                if child not in all_aliases:
                    errors.append(f"Composite '{alias}': unknown child '{child}'.")
                    continue
                if child in child_parent and child_parent[child] != alias:
                    errors.append(
                        f"Node '{child}' has multiple parents "
                        f"('{child_parent[child]}' and '{alias}') — the schema "
                        f"must be a strict TREE: one parent per node, no shared "
                        f"subtrees or shortcuts. If two branches genuinely need "
                        f"the same pattern, define it twice under new aliases."
                    )
                child_parent[child] = alias

        if root_alias in prim_set:
            errors.append(
                f"Root '{root_alias}' is a primitive — the root must be a "
                f"composite node."
            )
        if root_alias in child_parent:
            errors.append(
                f"Root '{root_alias}' appears as a child of "
                f"'{child_parent[root_alias]}' — the root must have no parent."
            )
        reachable: set = set()
        queue = [root_alias]
        while queue:
            a = queue.pop()
            if a in reachable:
                continue
            reachable.add(a)
            comp = self._composites.get(a)
            if comp:
                queue.extend(c for c in comp.children if c in all_aliases)
        orphans = sorted(all_aliases - reachable)
        if orphans:
            errors.append(
                f"Node(s) {orphans} are not reachable from root "
                f"'{root_alias}' — remove unused definitions or wire them "
                f"into the tree."
            )

        visited: set = set()
        def dfs(a: str, stack: set) -> bool:
            if a in stack:
                return True
            if a in visited:
                return False
            stack.add(a)
            visited.add(a)
            for child in self._composites.get(a, CompositeDef("", Operator.OR, [])).children:
                if dfs(child, stack):
                    return True
            stack.discard(a)
            return False
        for alias in self._composites:
            if dfs(alias, set()):
                errors.append(f"Cycle detected involving composite '{alias}'.")
                break
        return errors


    def add_candidate(
        self, alias: str, start: int, end: int, confidence: float = 1.0
    ) -> None:
        if alias not in self._primitives:
            raise KeyError(f"Unknown primitive alias '{alias}'.")
        if float(confidence) <= 0.0:
            self._candidates.pop(alias, None)
            return
        start, end = int(start), int(end)
        if start >= end:
            raise ValueError(f"Interval [{start}, {end}] invalid: start must be < end.")
        self._candidates[alias] = PrimitiveInstance(
            alias, start, end, float(confidence),
            channel=self._primitives[alias].target_channel,
        )

    def instantiate_all(self) -> List[CompositeInstance]:
        """Build the composite instance tree. Returns [root] (or [] if not compiled)."""
        if not self._compiled or self._root is None:
            return []
        return [self._build_instance(self._root)]

    def _build_instance(self, alias: str) -> InstanceNode:
        """Build the (single) instance node for an alias — Ghost when missing."""
        if alias in self._primitives:
            return self._candidates.get(alias) or GhostInstance(alias)
        comp = self._composites[alias]
        children = [self._build_instance(c) for c in comp.children]
        return CompositeInstance(alias, comp.operator, children, params=comp.params)


    def submit_schema(
        self,
        primitives: List[Dict],
        composites: List[Dict],
        root_alias: str,
    ) -> ToolResult:
        """Load a schema definition and compile it."""
        self._primitives.clear()
        self._composites.clear()
        self._candidates.clear()
        self._placements.clear()
        self._compiled = False

        dup_errors: List[str] = []
        seen: set = set()
        for p in primitives:
            a = p["alias"]
            if a in seen:
                dup_errors.append(f"Duplicate alias '{a}' in the schema.")
            seen.add(a)
            self.define_primitive(
                alias=a,
                target_channel=p["target_channel"],
                description=p.get("description", ""),
            )
        for c in composites:
            a = c["alias"]
            if a in seen:
                dup_errors.append(f"Duplicate alias '{a}' in the schema.")
            seen.add(a)
            self.define_composite(
                alias=a,
                operator=Operator(c["operator"]),
                children=c["children"],
                description=c.get("description", ""),
            )

        msg = self.compile(root_alias)
        if dup_errors:
            self._compiled = False
            msg = ("Validation errors:\n"
                   + "\n".join(f"  - {e}" for e in dup_errors)
                   + ("\n" + msg if msg.startswith("Validation") else ""))
        if not self._compiled:
            return ToolResult(text=msg)
        return self._tool_result(msg, self._render_state())

    @staticmethod
    def _tool_result(text: str, rendered: Optional[Tuple[str, str]]) -> ToolResult:
        """Build a ToolResult with the PNG (for the LLM) + SVG (for saving)."""
        if not rendered:
            return ToolResult(text=text)
        png, svg = rendered
        return ToolResult(text=text, images=[png], images_svg=[svg])

    def _append_numeric(self, text: str,
                        interval: Optional[Tuple[int, int]]) -> str:
        csv = self._numeric_csv(interval)
        return f"{text}\n\n{csv}" if csv else text

    def view_full(self) -> ToolResult:
        """Combined full view: signals with instantiated primitive bands (left)"""
        text = self._append_numeric(self._state_text(), None)
        return self._tool_result(text, self._render_state())

    def view_window(
        self,
        interval: List[int],
        vlines:   Optional[List[int]] = None,
    ) -> ToolResult:
        """Zoomed combined view.  The Y axis is RE-NORMALISED within the window so"""
        if len(interval) != 2:
            return ToolResult(text="interval must be [start, end].")
        start, end = int(interval[0]), int(interval[1])
        n = len(self._df)
        start, end = max(0, start), min(max(n - 1, 0), end)
        if start >= end:
            return ToolResult(text=f"Invalid window [{start}, {end}].")
        rendered = self._render_state(
            interval=(start, end), vlines=vlines, normalize_in_view=True
        )
        text = (f"Window [{start}, {end}] rendered (y re-normalised in view, "
                f"{len(vlines or [])} guide lines).\n" + self._state_text())
        text = self._append_numeric(text, (start, end))
        return self._tool_result(text, rendered)

    def instantiate(self, instances: List[Dict]) -> ToolResult:
        """Register/UPDATE primitive candidates and return the refreshed combined"""
        updated, removed, errors = [], [], []
        for inst in instances:
            alias = inst.get("alias", "")
            try:
                conf = float(inst.get("confidence", 1.0))
                self.add_candidate(
                    alias=alias,
                    start=int(inst["start"]),
                    end=int(inst["end"]),
                    confidence=conf,
                )
                (removed if conf <= 0.0 else updated).append(alias)
            except (KeyError, ValueError) as exc:
                errors.append(f"  {alias}: {exc}")

        if self._candidates:
            hull = (min(c.start for c in self._candidates.values()),
                    max(c.end for c in self._candidates.values()))
            if all(_interval_iou(hull, h) < 0.5 for h in self._placements):
                self._placements.append(hull)

        lines = [f"Updated candidates: {updated}"]
        if removed:
            lines.append(f"Retracted (returned to ghost): {removed}")
        if errors:
            lines.append("Errors:\n" + "\n".join(errors))
        lines.append(self._state_text())
        lines.append(
            "Verify band by band: each band claims its definition holds over "
            "the ENTIRE interval. Check direction words literally (a downward-"
            "spike primitive can never sit on an upward excursion) and look "
            "for contradicting sub-segments inside every band (an abrupt "
            "plunge/jump inside a 'slow'/'steady' claim invalidates it). "
            "Score by the WORST-matching part; tighten, split, retract "
            "(confidence 0) or relocate any band that fails."
        )
        text = self._append_numeric("\n".join(lines), None)
        return self._tool_result(text, self._render_state())


    def _definitions_text(self) -> str:
        """Primitive definitions block — repeated in EVERY tool response so the"""
        if not self._primitives:
            return ""
        lines = ["Primitive definitions:"]
        for alias, p in self._primitives.items():
            desc = p.description or "(no description)"
            lines.append(f"  {alias} ⟨{p.target_channel}⟩: {desc}")
        return "\n".join(lines)

    def _resampled_indices(self, lo: int, hi: int) -> List[int]:
        """Original indices to read out for the window [lo, hi] (inclusive)."""
        lo = max(0, lo)
        hi = min(len(self._df) - 1, hi)
        n = hi - lo + 1
        if n <= 0:
            return []
        cap = self._numeric_max_rows
        if n <= cap:
            return list(range(lo, hi + 1))

        if self._resample_mode == "envelope":
            sub = self._df[self._visible_channels()] \
                      .iloc[lo: hi + 1].values.astype(float)
            mean = sub.mean(axis=0, keepdims=True)
            std = sub.std(axis=0, keepdims=True)
            std[std < 1e-9] = 1.0
            dev = np.abs((sub - mean) / std).sum(axis=1)
            edges = np.linspace(0, n, cap + 1).astype(int)
            picked = set()
            for b in range(cap):
                s, e = edges[b], edges[b + 1]
                if e <= s:
                    continue
                picked.add(lo + s + int(np.argmax(dev[s:e])))
            picked.add(lo); picked.add(hi)
            return sorted(picked)

        stride = math.ceil(n / cap)
        idx = list(range(lo, hi + 1, stride))
        if idx[-1] != hi:
            idx.append(hi)
        return idx

    def _numeric_csv(self, interval: Optional[Tuple[int, int]]) -> str:
        """CSV read-out of the visible window: `idx,<ch1>,<ch2>,…`."""
        if not self._numeric_readout or len(self._df) == 0:
            return ""
        lo, hi = interval if interval is not None else (0, len(self._df) - 1)
        idx = self._resampled_indices(int(lo), int(hi))
        if not idx:
            return ""
        cols = self._visible_channels()
        sub = self._df.iloc[idx]
        n_full = int(hi) - int(lo) + 1
        if n_full > len(idx):
            how = ("regular stride, original indices"
                   if self._resample_mode == "decimate"
                   else "envelope: most-deviating row per bucket kept, so "
                        "spikes survive; indices irregular")
            note = (f"(window [{lo}, {hi}]: {n_full} rows subsampled to "
                    f"{len(idx)} — {how}; values are raw)")
        else:
            note = f"(window [{lo}, {hi}]: all {len(idx)} rows, raw)"
        lines = [f"Numeric readout {note}:",
                 "idx," + ",".join(str(c) for c in cols)]
        for i, (_, row) in zip(idx, sub.iterrows()):
            vals = ",".join(f"{float(row[c]):.4g}" for c in cols)
            lines.append(f"{i},{vals}")
        return "\n".join(lines)

    def _state_text(self) -> str:
        """Indented per-node table: alias, operator, interval, confidence."""
        roots = self.instantiate_all()
        if not roots or not isinstance(roots[0], CompositeInstance):
            return "Schema not compiled yet."
        root = roots[0]
        lines: List[str] = []
        defs = self._definitions_text()
        if defs:
            lines.append(defs)
        lines.append("Tree status:")

        def walk(node: InstanceNode, depth: int) -> None:
            pad = "  " * (depth + 1)
            if isinstance(node, GhostInstance):
                lines.append(f"{pad}{node.alias}: MISSING (not instantiated)")
            elif isinstance(node, PrimitiveInstance):
                lines.append(
                    f"{pad}{node.alias}: [{node.start}, {node.end}] "
                    f"conf={node.confidence:.2f}"
                )
            else:
                iv = _interval(node)
                tag = "  << GHOST (branch inactive)" if node.is_ghost else ""
                lines.append(
                    f"{pad}{node.alias} <{node.operator.value}>: "
                    f"[{iv[0]}, {iv[1]}] conf={node.confidence:.3f}{tag}"
                )
                for c in node.children:
                    walk(c, depth + 1)

        walk(root, 0)
        K = _count_active_leaves(root)
        if K:
            lines.append(
                f"Root normalized confidence: {root.confidence:.4f}^(1/{K}) "
                f"= {normalized_confidence(root):.4f}"
            )
        else:
            lines.append("No primitives instantiated yet.")
        return "\n".join(lines)


    def _visible_channels(self) -> List[str]:
        """Channels this tree actually references (its primitives' target"""
        cols = list(self._df.columns)
        if not cols:
            return []
        used = {p.target_channel for p in self._primitives.values()
                if p.target_channel in cols}
        return [c for c in cols if c in used] if used else cols

    def _prim_color_map(self) -> Dict[str, str]:
        return {
            a: _V_SERIES[i % len(_V_SERIES)]
            for i, a in enumerate(sorted(self._primitives))
        }

    def _tree_layout(self) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
        """Leaf-ordered layout: leaves evenly spaced, parents centred above."""
        root = self._root or (
            next(reversed(list(self._composites)), None)
            or next(iter(self._primitives), None)
        )
        if root is None:
            return {}, 0, 0
        leaves: List[str] = []
        depths: Dict[str, int] = {}

        def walk(alias: str, depth: int) -> None:
            depths[alias] = max(depths.get(alias, 0), depth)
            comp = self._composites.get(alias)
            if comp is None:
                if alias not in leaves:
                    leaves.append(alias)
            else:
                for c in comp.children:
                    walk(c, depth + 1)

        walk(root, 0)
        xs: Dict[str, float] = {a: float(i) for i, a in enumerate(leaves)}

        def xcalc(alias: str) -> float:
            if alias in xs:
                return xs[alias]
            ch = self._composites[alias].children
            xs[alias] = (sum(xcalc(c) for c in ch) / len(ch)) if ch else 0.0
            return xs[alias]

        xcalc(root)
        maxd = max(depths.values()) or 1
        maxx = max(xs.values()) or 1.0
        pos = {
            a: (0.06 + 0.88 * (xv / maxx if maxx else 0.5),
                0.94 - 0.88 * depths[a] / maxd)
            for a, xv in xs.items()
        }
        return pos, len(leaves), maxd

    def _render_state(
        self,
        interval:          Optional[Tuple[int, int]] = None,
        vlines:            Optional[List[int]]       = None,
        normalize_in_view: bool                      = False,
    ) -> Optional[Tuple[str, str]]:
        """Render the combined view; return (png_base64, svg_text) or None."""
        if not self._primitives and not self._composites:
            return None

        pos, n_leaves, depth = self._tree_layout()
        has_data = len(self._df) > 0 and len(self._df.columns) > 0
        n_ch     = len(self._visible_channels())

        tree_w = min(max(1.12 * max(n_leaves, 3), 3.4), 9.0)
        tree_h = 0.9 + 0.72 * (depth + 1)
        sig_h  = 0.7 + 1.05 * max(n_ch, 1)
        fig_h  = min(max(tree_h, sig_h, 2.6), 7.0)

        if has_data:
            sig_w = 7.4
            fig = plt.figure(figsize=(sig_w + tree_w + 0.3, fig_h),
                             facecolor=_V_SURFACE)
            gs = fig.add_gridspec(
                1, 2, width_ratios=[sig_w, tree_w], wspace=0.05,
                left=0.012, right=0.995, top=0.88, bottom=0.16,
            )
            ax_sig  = fig.add_subplot(gs[0])
            ax_tree = fig.add_subplot(gs[1])
            self._draw_signals(ax_sig, interval, vlines, normalize_in_view)
        else:
            fig, ax_tree = plt.subplots(
                figsize=(tree_w + 0.6, fig_h), facecolor=_V_SURFACE)

        self._draw_tree(ax_tree, pos)

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=110, facecolor=_V_SURFACE,
                    bbox_inches="tight", pad_inches=0.12)
        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg", facecolor=_V_SURFACE,
                    bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        png_b64 = base64.b64encode(png_buf.getvalue()).decode()
        svg_txt = svg_buf.getvalue().decode("utf-8")
        return png_b64, svg_txt

    def _draw_signals(
        self,
        ax,
        interval:          Optional[Tuple[int, int]],
        vlines:            Optional[List[int]],
        normalize_in_view: bool,
    ) -> None:
        df       = self._df
        channels = self._visible_channels()
        n_ch     = len(channels)
        gain     = 0.74
        x        = np.arange(len(df))

        if normalize_in_view and interval is not None:
            norm_src = df.iloc[interval[0]: interval[1] + 1]
        else:
            norm_src = df

        ax.set_facecolor(_V_SURFACE)
        cmap = self._prim_color_map()

        per_ch: Dict[str, List[PrimitiveInstance]] = {}
        for alias, inst in self._candidates.items():
            per_ch.setdefault(self._primitives[alias].target_channel, []).append(inst)
        for lst in per_ch.values():
            lst.sort(key=lambda i: i.start)

        ch_rows: List[Tuple[float, np.ndarray, np.ndarray]] = []

        for i, ch in enumerate(channels):
            y0   = float(n_ch - 1 - i)
            data = df[ch].values.astype(float)
            lo   = float(norm_src[ch].min())
            hi   = float(norm_src[ch].max())
            norm = (data - lo) / (hi - lo + 1e-9)
            ch_rows.append((y0, data, norm))

            if i:
                ax.axhline(y0 + 0.5, color=_V_GRID, linewidth=0.7, zorder=1)

            ax.plot(x, (norm - 0.5) * gain + y0,
                    color=_V_MUTED, linewidth=1.1, alpha=0.8, zorder=2)

            tag = ch
            if normalize_in_view and interval is not None:
                g_lo, g_hi = float(df[ch].min()), float(df[ch].max())
                pct = 100.0 * (hi - lo) / max(g_hi - g_lo, 1e-12)
                tag = f"{ch}  ·  y-span {pct:.0f}% of global range"
            ax.text(0.006, y0 + 0.46, tag,
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="top", fontsize=7.2, fontweight="bold",
                    color=_V_INK2, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=_V_GRID, linewidth=0.8))

            x_lo, x_hi = (interval if interval is not None
                          else (0, max(len(df) - 1, 1)))
            span    = max(x_hi - x_lo, 1)
            per_px  = span / 820.0
            level_end = [-1e18, -1e18, -1e18]
            level_dy  = [0.26, 0.10, -0.06]

            for inst in per_ch.get(ch, []):
                c = cmap[inst.alias]
                ax.add_patch(plt.Rectangle(
                    (inst.start, y0 - 0.37), max(inst.end - inst.start, 1), 0.74,
                    facecolor=c, alpha=0.13, edgecolor=c, linewidth=1.1,
                    zorder=3,
                ))
                mask = (x >= inst.start) & (x <= inst.end)
                ax.plot(x[mask], (norm[mask] - 0.5) * gain + y0,
                        color=c, linewidth=2.0, alpha=0.95, zorder=4)

                label  = f"{inst.alias} · {inst.confidence:.2f}"
                chip_w = (4.2 * len(label) + 16) * per_px
                cx     = (inst.start + inst.end) / 2
                cx     = min(max(cx, x_lo + chip_w / 2), x_hi - chip_w / 2)
                lvl    = min(range(3), key=lambda k: (level_end[k] > cx - chip_w / 2,
                                                      level_end[k]))
                level_end[lvl] = cx + chip_w / 2
                ax.text(cx, y0 + level_dy[lvl], label,
                        ha="center", va="center", fontsize=6.6, color=_V_INK,
                        zorder=5,
                        bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                                  edgecolor=c, linewidth=1.0, alpha=0.95))

        for vl in (vlines or []):
            vl = int(vl)
            ax.axvline(vl, color=_V_BASELINE, linestyle="--",
                       linewidth=1.0, alpha=0.9, zorder=2)
            ax.text(vl, 0.005, str(vl),
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.2, color=_V_MUTED,
                    zorder=6)
            if not (0 <= vl < len(df)):
                continue
            for y0, data, norm in ch_rows:
                yv = (norm[vl] - 0.5) * gain + y0
                ax.plot(vl, yv, marker="o", markersize=3,
                        color=_V_BASELINE, zorder=7)
                ax.annotate(
                    f"{data[vl]:.4g}", xy=(vl, yv),
                    xytext=(3, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=6.0, color=_V_INK,
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=_V_BASELINE, linewidth=0.7, alpha=0.92),
                )

        if interval is not None:
            ax.set_xlim(interval[0], interval[1])
        ax.set_ylim(-0.6, n_ch - 1 + 0.6)
        ax.set_yticks([])
        ax.grid(axis="x", color=_V_GRID, linewidth=0.6,
                linestyle=(0, (3, 3)), alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=_V_MUTED, labelsize=7, length=3)
        for name, sp in ax.spines.items():
            sp.set_color(_V_BASELINE if name == "bottom" else "none")
            sp.set_visible(name == "bottom")
        suffix = "  ·  view-normalized" if (normalize_in_view and interval) else ""
        ax.set_title(f"Signals & Primitive Instances{suffix}",
                     loc="left", fontsize=9.5, fontweight="bold",
                     color=_V_INK, pad=7)

    @staticmethod
    def _tint(color: str, frac: float = 0.12) -> Tuple[float, float, float]:
        """Opaque light tint: `color` blended over the chart surface."""
        from matplotlib.colors import to_rgba
        r, g, b, _ = to_rgba(color)
        sr, sg, sb, _ = to_rgba(_V_SURFACE)
        return (r * frac + sr * (1 - frac),
                g * frac + sg * (1 - frac),
                b * frac + sb * (1 - frac))

    def _draw_tree(self, ax, pos: Dict[str, Tuple[float, float]]) -> None:

        ax.set_facecolor(_V_SURFACE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        title = f"Logic Tree · {self.label}" if self.label else "Logic Tree"
        ax.set_title(title, loc="left", fontsize=9.5,
                     fontweight="bold", color=_V_INK, pad=7)
        if not pos:
            return

        info: Dict[str, InstanceNode] = {}
        root_inst: Optional[CompositeInstance] = None
        roots = self.instantiate_all()
        if roots and isinstance(roots[0], CompositeInstance):
            root_inst = roots[0]
            def collect(node: InstanceNode) -> None:
                info[node.alias] = node
                if isinstance(node, CompositeInstance):
                    for c in node.children:
                        collect(c)
            collect(root_inst)

        if root_inst is not None:
            K = _count_active_leaves(root_inst)
            if K:
                ax.text(0.995, 1.0,
                        f"μ_root {root_inst.confidence:.3f}  ·  K {K}  ·  "
                        f"μ_norm {normalized_confidence(root_inst):.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.8, color=_V_INK2, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                                  edgecolor=_V_GRID, linewidth=0.8))

        schema_only = not self._candidates

        for alias, comp in self._composites.items():
            if alias not in pos:
                continue
            px, py = pos[alias]
            for child in comp.children:
                if child not in pos:
                    continue
                cx, cy = pos[child]
                ghost_edge = not schema_only and (
                    not info
                    or _is_ghost(info.get(alias, GhostInstance(alias)))
                    or _is_ghost(info.get(child, GhostInstance(child)))
                )
                ax.plot([px, cx], [py, cy],
                        color=_V_MUTED if ghost_edge else _V_BASELINE,
                        linestyle=(0, (3, 2)) if ghost_edge else "-",
                        linewidth=0.9 if ghost_edge else 1.4,
                        alpha=0.55 if ghost_edge else 1.0,
                        zorder=1)

        cmap = self._prim_color_map()
        for alias, (xx, yy) in pos.items():
            node = info.get(alias)
            if alias in self._primitives:
                c     = cmap[alias]
                ghost = node is None or isinstance(node, GhostInstance)
                if schema_only:
                    ax.text(xx, yy,
                            f"{alias}\n⟨{self._primitives[alias].target_channel}⟩",
                            ha="center", va="center", fontsize=6.6,
                            color=_V_INK, zorder=3,
                            bbox=dict(boxstyle="round,pad=0.32",
                                      facecolor=self._tint(c),
                                      edgecolor=c, linewidth=1.3))
                elif ghost:
                    ch_name = self._primitives[alias].target_channel
                    ax.text(xx, yy, f"{alias} ⟨{ch_name}⟩\nmissing",
                            ha="center", va="center", fontsize=6.6,
                            color=_V_MUTED, zorder=3,
                            bbox=dict(boxstyle="round,pad=0.32",
                                      facecolor=_V_SURFACE, edgecolor=_V_MUTED,
                                      linestyle=(0, (3, 2)), linewidth=1.0))
                else:
                    ch_name = self._primitives[alias].target_channel
                    ax.text(xx, yy,
                            f"{alias} ⟨{ch_name}⟩\n[{node.start}, {node.end}] · "
                            f"{node.confidence:.2f}",
                            ha="center", va="center", fontsize=6.6,
                            color=_V_INK, zorder=3,
                            bbox=dict(boxstyle="round,pad=0.32",
                                      facecolor=self._tint(c),
                                      edgecolor=c, linewidth=1.3))
            else:
                comp = self._composites[alias]
                oc   = _V_OP.get(comp.operator.value, _V_SERIES[0])
                if schema_only:
                    label = f"{comp.operator.value} · {alias}"
                elif node is not None and not _is_ghost(node):
                    iv    = _interval(node)
                    label = (f"{comp.operator.value} · {alias}\n"
                             f"{node.confidence:.3f} · ({iv[0]}, {iv[1]})")
                else:
                    label = f"{comp.operator.value} · {alias}\n—"
                ax.text(xx, yy, label,
                        ha="center", va="center", fontsize=6.8,
                        color=_V_INK, zorder=3,
                        bbox=dict(boxstyle="round,pad=0.34",
                                  facecolor="white", edgecolor=oc,
                                  linewidth=1.6))

        if not schema_only:
            ax.text(0.0, -0.03, "solid = instantiated      dashed = missing",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.2, color=_V_MUTED)


_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "submit_schema",
            "description": (
                "Load and compile an ELT schema. "
                "Call this first to define the event tree before instantiating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "primitives": {
                        "type": "array",
                        "description": "List of primitive definitions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias":          {"type": "string"},
                                "target_channel": {"type": "string"},
                                "description":    {"type": "string"},
                            },
                            "required": ["alias", "target_channel"],
                        },
                    },
                    "composites": {
                        "type": "array",
                        "description": "List of composite definitions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias":       {"type": "string"},
                                "operator":    {"type": "string", "enum": ["SEQ","SYNC","GUARD","OR"]},
                                "children":    {"type": "array", "items": {"type": "string"}},
                                "description": {"type": "string"},
                            },
                            "required": ["alias", "operator", "children"],
                        },
                    },
                    "root_alias": {
                        "type": "string",
                        "description": "Alias of the root composite node.",
                    },
                },
                "required": ["primitives", "composites", "root_alias"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_full",
            "description": (
                "Full combined view: all signal channels (stacked, normalised) "
                "with every instantiated primitive highlighted as a coloured "
                "band, plus the logic tree annotated with per-node confidence "
                "and intervals. Call this FIRST to plan, and again to verify."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_window",
            "description": (
                "Zoomed combined view of [start, end]. The Y axis is "
                "re-normalised WITHIN the window, so subtle morphology "
                "invisible in the full view becomes clear. Use it to refine "
                "primitive boundaries before/after instantiating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interval": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[start, end] index range to display.",
                    },
                    "vlines": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Row indices to mark with vertical lines.",
                    },
                },
                "required": ["interval"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "instantiate",
            "description": (
                "Register or UPDATE primitive candidates, then return the "
                "refreshed combined visualisation (signals + tree with "
                "confidences). One live candidate per alias — re-submitting an "
                "alias REPLACES its interval, so to correct a primitive just "
                "resubmit only that alias. Submitting confidence 0 RETRACTS "
                "the alias entirely (the leaf returns to missing/ghost) — use "
                "this to abandon an OR branch or an obsolete placement. "
                "Intervals must satisfy start < end."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instances": {
                        "type": "array",
                        "description": "List of primitive candidates.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias":      {"type": "string"},
                                "start":      {"type": "integer"},
                                "end":        {"type": "integer"},
                                "confidence": {
                                    "type": "number",
                                    "description": (
                                        "REQUIRED semantic-fidelity score in "
                                        "[0, 1]: how well this segment shows "
                                        "the primitive's described morphology."
                                    ),
                                },
                            },
                            "required": ["alias", "start", "end", "confidence"],
                        },
                    }
                },
                "required": ["instances"],
            },
        },
    },
]


class ELTBackend:
    """ToolBackend wrapping an ELT instance for one data sample."""

    def __init__(self, df: pd.DataFrame, require_main: bool = False,
                 main_only: bool = False) -> None:
        self._elt = ELT(df, require_main=require_main, main_only=main_only)

    @property
    def elt(self) -> ELT:
        """Direct access to the underlying ELT for external inspection."""
        return self._elt

    def get_schemas(self) -> List[Dict]:
        return list(_SCHEMAS)

    def get_callables(self) -> Dict[str, Callable]:
        e = self._elt
        return {
            "submit_schema": e.submit_schema,
            "view_full":     e.view_full,
            "view_window":   e.view_window,
            "instantiate":   e.instantiate,
        }


class MultiELTBackend:
    """ToolBackend holding one ELT per event class over the SAME data sample."""

    def __init__(
        self,
        df:    pd.DataFrame,
        trees: Dict[str, Tuple[Dict[str, Any], str]],
        numeric_readout:  bool = False,
        numeric_max_rows: int  = 40,
        resample_mode:    str  = "decimate",
    ) -> None:
        self._elts: Dict[str, ELT] = {}
        for name, (schema_dict, root_alias) in trees.items():
            elt = ELT(df, label=name, numeric_readout=numeric_readout,
                      numeric_max_rows=numeric_max_rows,
                      resample_mode=resample_mode)
            msg = elt.load_schema(schema_dict, root_alias)
            if not elt._compiled:
                logger.warning(
                    "[MultiELT] Tree %r failed to compile: %s", name, msg)
                continue
            self._elts[name] = elt

    @property
    def tree_names(self) -> List[str]:
        return list(self._elts)

    def elt(self, name: str) -> Optional[ELT]:
        """Direct access to one class's ELT for external inspection."""
        return self._elts.get(name)


    def _get(self, tree: str) -> Optional[ELT]:
        return self._elts.get(tree)

    def view_full(self, tree: str) -> ToolResult:
        elt = self._get(tree)
        if elt is None:
            return ToolResult(text=f"Unknown tree '{tree}'. Available: {self.tree_names}")
        return elt.view_full()

    def view_window(
        self, tree: str, interval: List[int],
        vlines: Optional[List[int]] = None,
    ) -> ToolResult:
        elt = self._get(tree)
        if elt is None:
            return ToolResult(text=f"Unknown tree '{tree}'. Available: {self.tree_names}")
        return elt.view_window(interval, vlines)

    def instantiate(self, tree: str, instances: List[Dict]) -> ToolResult:
        elt = self._get(tree)
        if elt is None:
            return ToolResult(text=f"Unknown tree '{tree}'. Available: {self.tree_names}")
        return elt.instantiate(instances)

    def compare_trees(self) -> ToolResult:
        """Standings of every tree: μ_root, K, μ_norm, Main interval."""
        lines = ["Tree comparison (highest normalized confidence wins):"]
        single_region = []
        for name, elt in self._elts.items():
            roots = elt.instantiate_all()
            root  = roots[0] if roots else None
            n_pl  = len(elt._placements)
            if isinstance(root, CompositeInstance):
                K = _count_active_leaves(root)
                if K:
                    m    = find_main_instance(root)
                    mtxt = f"Main [{m.start}, {m.end}]" if m is not None else "Main n/a"
                    lines.append(
                        f"  {name:<24s} μ_root {root.confidence:.4f}   K {K}   "
                        f"μ_norm {normalized_confidence(root):.4f}   {mtxt}   "
                        f"regions tested: {n_pl}"
                    )
                    if n_pl < 2:
                        single_region.append(name)
                    continue
            lines.append(f"  {name:<24s} (no primitives instantiated yet)")
        if single_region:
            lines.append(
                f"NOTE: {single_region} tested only ONE placement so far. "
                f"Placement discipline: if any other region could plausibly "
                f"host the event, instantiate there too and keep the better "
                f"placement before concluding."
            )
        return ToolResult(text="\n".join(lines))


    def get_schemas(self) -> List[Dict]:
        tree_param = {
            "type": "string",
            "enum": self.tree_names,
            "description": "Which event-class tree to operate on.",
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": "view_full",
                    "description": (
                        "Full combined view for ONE tree: the signals on that "
                        "tree's OWN channels (only the channels its primitives "
                        "reference are shown) with its instantiated bands, plus "
                        "its logic-tree status."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"tree": tree_param},
                        "required": ["tree"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "view_window",
                    "description": (
                        "Zoomed combined view of [start, end] for ONE tree. "
                        "The Y axis is re-normalised WITHIN the window so "
                        "subtle morphology becomes visible."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tree": tree_param,
                            "interval": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "[start, end] index range.",
                            },
                            "vlines": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Row indices to mark.",
                            },
                        },
                        "required": ["tree", "interval"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "instantiate",
                    "description": (
                        "Register or UPDATE primitive candidates on ONE tree, "
                        "then return that tree's refreshed visualisation. One "
                        "live candidate per alias — re-submitting an alias "
                        "REPLACES its interval. Submitting confidence 0 "
                        "RETRACTS the alias (leaf returns to missing/ghost) — "
                        "use this to abandon an OR branch or an obsolete "
                        "placement after relocating. Intervals: start < end."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tree": tree_param,
                            "instances": {
                                "type": "array",
                                "description": "List of primitive candidates.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "alias":      {"type": "string"},
                                        "start":      {"type": "integer"},
                                        "end":        {"type": "integer"},
                                        "confidence": {
                                            "type": "number",
                                            "description": (
                                                "REQUIRED semantic-fidelity "
                                                "score in [0, 1]: how well "
                                                "this segment shows the "
                                                "described morphology."
                                            ),
                                        },
                                    },
                                    "required": ["alias", "start", "end",
                                                 "confidence"],
                                },
                            },
                        },
                        "required": ["tree", "instances"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_trees",
                    "description": (
                        "Current standings of ALL trees: μ_root, K, normalized "
                        "confidence μ_norm, and Main interval. Call after "
                        "instantiating to see which tree the evidence favours."
                    ),
                    "parameters": {"type": "object", "properties": {},
                                   "required": []},
                },
            },
        ]

    def get_callables(self) -> Dict[str, Callable]:
        return {
            "view_full":     self.view_full,
            "view_window":   self.view_window,
            "instantiate":   self.instantiate,
            "compare_trees": self.compare_trees,
        }
