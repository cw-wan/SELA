"""PlotViewer tool backend."""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sela.mas.tool_backend import ToolBackend, ToolResult

logger = logging.getLogger(__name__)

_BG      = "#1e1e1e"
_BG_SIDE = "#252526"
_FG      = "#d4d4d4"
_FG_DIM  = "#858585"
_ACCENT  = "#569cd6"
_BORDER  = "#3c3c3c"
_COLORS  = ["#569cd6", "#4ec9b0", "#ce9178", "#dcdcaa", "#c586c0", "#f44747"]
_EVENT_PALETTE = ["#4ec9b0", "#ce9178", "#dcdcaa", "#c586c0", "#f44747", "#569cd6", "#b5cea8"]


class PlotViewer:
    """Stateful time-series viewer."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df      = df.reset_index(drop=True)
        n             = len(self._df)
        self._x_start = 0
        self._x_end   = max(n - 1, 0)
        self._y_lo:  Dict[str, float] = {}
        self._y_hi:  Dict[str, float] = {}


    def _render(
        self,
        df_slice:     pd.DataFrame,
        title:        str                       = "",
        extra_series: Optional[Dict[str, Any]]  = None,
        x_offset:     int                       = 0,
    ) -> str:
        """Render df_slice to a dark-themed multi-subplot figure; return base64 PNG."""
        cols = list(df_slice.columns)
        n    = len(cols)
        fig, axes = plt.subplots(
            n, 1,
            figsize=(12, max(2.4 * n, 4)),
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
        fig.patch.set_facecolor(_BG)
        if n == 1:
            axes = [axes]

        x_idx = np.arange(len(df_slice)) + x_offset

        for i, (ax, col) in enumerate(zip(axes, cols)):
            ax.set_facecolor(_BG_SIDE)
            for spine in ax.spines.values():
                spine.set_color(_BORDER)
            ax.tick_params(colors=_FG_DIM, labelsize=9, length=3,
                           width=0.5, labelcolor=_FG_DIM)
            ax.set_ylabel(col, color=_FG_DIM, fontsize=9,
                          rotation=0, ha="right", va="center", labelpad=50)

            vals = df_slice[col].values
            ax.plot(x_idx, vals, color=_COLORS[i % len(_COLORS)],
                    linewidth=0.9, alpha=0.95)

            if col in self._y_lo:
                ax.set_ylim(self._y_lo[col], self._y_hi[col])

            if extra_series and col in extra_series:
                ax.plot(x_idx, extra_series[col], color="#f44747",
                        linewidth=0.9, alpha=0.8, linestyle="--")

        if title:
            axes[0].set_title(title, color=_FG, fontsize=10, pad=4)

        axes[-1].set_xlabel("index", color=_FG_DIM, fontsize=9)
        axes[-1].tick_params(axis="x", colors=_FG_DIM)

        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=_BG)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _render_with_events(
        self,
        df_slice:  pd.DataFrame,
        events:    List[Dict],
        title:     str = "",
        x_offset:  int = 0,
    ) -> str:
        """Like _render but with coloured shaded regions for each event."""
        cols = list(df_slice.columns)
        n    = len(cols)
        fig, axes = plt.subplots(
            n, 1,
            figsize=(12, max(2.4 * n, 4)),
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
        fig.patch.set_facecolor(_BG)
        if n == 1:
            axes = [axes]

        x_idx = np.arange(len(df_slice)) + x_offset

        for i, (ax, col) in enumerate(zip(axes, cols)):
            ax.set_facecolor(_BG_SIDE)
            for spine in ax.spines.values():
                spine.set_color(_BORDER)
            ax.tick_params(colors=_FG_DIM, labelsize=9, length=3,
                           width=0.5, labelcolor=_FG_DIM)
            ax.set_ylabel(col, color=_FG_DIM, fontsize=9,
                          rotation=0, ha="right", va="center", labelpad=50)
            vals = df_slice[col].values
            ax.plot(x_idx, vals, color=_COLORS[i % len(_COLORS)],
                    linewidth=0.9, alpha=0.95, zorder=2)
            if col in self._y_lo:
                ax.set_ylim(self._y_lo[col], self._y_hi[col])

        class_names = sorted({e.get("class_name", "") for e in events})
        cmap = {cn: _EVENT_PALETTE[i % len(_EVENT_PALETTE)] for i, cn in enumerate(class_names)}
        for ev in events:
            ev_s  = ev.get("start", 0)
            ev_e  = ev.get("end",   0)
            color = cmap.get(ev.get("class_name", ""), _EVENT_PALETTE[0])
            for ax in axes:
                ax.axvspan(ev_s, ev_e, alpha=0.18, color=color, zorder=1)
            axes[0].axvline(ev_s, color=color, alpha=0.45, linewidth=0.8, zorder=3)
            axes[0].axvline(ev_e, color=color, alpha=0.45, linewidth=0.8, zorder=3)

        if class_names:
            from matplotlib.patches import Patch
            handles = [Patch(color=cmap[cn], alpha=0.55, label=cn) for cn in class_names]
            axes[0].legend(handles=handles, loc="upper right", fontsize=8,
                           facecolor=_BG_SIDE, edgecolor=_BORDER, labelcolor=_FG)

        if title:
            axes[0].set_title(title, color=_FG, fontsize=10, pad=4)
        axes[-1].set_xlabel("index", color=_FG_DIM, fontsize=9)
        axes[-1].tick_params(axis="x", colors=_FG_DIM)

        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=_BG)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _clamp_range(self, start: int, end: int) -> Tuple[int, int]:
        n = len(self._df)
        return max(0, min(start, n - 1)), max(0, min(end, n - 1))

    def _window_df(self) -> pd.DataFrame:
        s, e = self._clamp_range(self._x_start, self._x_end)
        return self._df.iloc[s : e + 1]

    def _span(self) -> int:
        return max(self._x_end - self._x_start, 1)


    def plot_all(self) -> ToolResult:
        """Plot the entire dataset."""
        self._x_start, self._x_end = 0, len(self._df) - 1
        self._y_lo.clear()
        self._y_hi.clear()
        b64 = self._render(self._df, title="Full dataset")
        return ToolResult(
            text=f"Full dataset plotted. Range: 0–{self._x_end} ({len(self._df)} rows).",
            images=[b64],
        )

    def plot_window(self, start: int, end: int) -> ToolResult:
        """Plot an explicit index window [start, end]."""
        self._x_start, self._x_end = self._clamp_range(int(start), int(end))
        b64 = self._render(
            self._window_df(),
            title=f"Window {self._x_start}–{self._x_end}",
            x_offset=self._x_start,
        )
        return ToolResult(
            text=f"Window plotted: {self._x_start}–{self._x_end}.",
            images=[b64],
        )

    def plot_window_with_window_size(self, mid_idx: int, window_size: int) -> ToolResult:
        """Plot a window centred on mid_idx with the given width."""
        half = int(window_size) // 2
        return self.plot_window(int(mid_idx) - half, int(mid_idx) + half)

    def plot_left(self) -> ToolResult:
        """Shift the current window left by 3/4 of its span."""
        shift = max(self._span() * 3 // 4, 1)
        return self.plot_window(self._x_start - shift, self._x_end - shift)

    def plot_right(self) -> ToolResult:
        """Shift the current window right by 3/4 of its span."""
        shift = max(self._span() * 3 // 4, 1)
        return self.plot_window(self._x_start + shift, self._x_end + shift)

    def plot_zoom_in_x(self) -> ToolResult:
        """Zoom in on the X axis (halve the span, centred)."""
        mid   = (self._x_start + self._x_end) // 2
        half  = max(self._span() // 4, 1)
        return self.plot_window(mid - half, mid + half)

    def plot_zoom_out_x(self) -> ToolResult:
        """Zoom out on the X axis (double the span, centred)."""
        mid  = (self._x_start + self._x_end) // 2
        half = self._span()
        return self.plot_window(mid - half, mid + half)

    def plot_zoom_in_y(self) -> ToolResult:
        """Zoom in on Y: clamp each channel to ±½ sigma around its mean in the window."""
        df = self._window_df()
        for col in df.columns:
            v    = df[col].dropna()
            mean = float(v.mean())
            std  = float(v.std()) or 1.0
            self._y_lo[col] = mean - std * 0.5
            self._y_hi[col] = mean + std * 0.5
        b64 = self._render(self._window_df(),
                           title=f"Window {self._x_start}–{self._x_end} (Y zoomed)",
                           x_offset=self._x_start)
        return ToolResult(text="Y axis zoomed in.", images=[b64])

    def plot_zoom_out_y(self) -> ToolResult:
        """Reset Y axis to auto-scale."""
        self._y_lo.clear()
        self._y_hi.clear()
        b64 = self._render(self._window_df(),
                           title=f"Window {self._x_start}–{self._x_end} (Y auto)",
                           x_offset=self._x_start)
        return ToolResult(text="Y axis reset to auto-scale.", images=[b64])

    def plot_derivative(self, channels: List[str]) -> ToolResult:
        """Plot the first derivative (diff) of the specified channels."""
        df   = self._window_df()
        cols = [c for c in channels if c in df.columns]
        if not cols:
            return ToolResult(text=f"None of {channels} found in data.")
        df_d = df[cols].diff().fillna(0)
        b64  = self._render(df_d, title="First derivative", x_offset=self._x_start)
        return ToolResult(text=f"First derivative plotted for: {cols}.", images=[b64])

    def plot_second_derivative(self, channels: List[str]) -> ToolResult:
        """Plot the second derivative of the specified channels."""
        df   = self._window_df()
        cols = [c for c in channels if c in df.columns]
        if not cols:
            return ToolResult(text=f"None of {channels} found in data.")
        df_d2 = df[cols].diff().diff().fillna(0)
        b64   = self._render(df_d2, title="Second derivative", x_offset=self._x_start)
        return ToolResult(text=f"Second derivative plotted for: {cols}.", images=[b64])

    def plot_event_region(self, start: int, end: int, context_margin: int = 100) -> ToolResult:
        """Visualise a candidate event region [start, end] with surrounding context."""
        events = [{"class_name": "candidate", "start": int(start), "end": int(end)}]
        s = max(0, int(start) - int(context_margin))
        e = min(len(self._df) - 1, int(end) + int(context_margin))
        self._x_start = s
        self._x_end   = e
        b64 = self._render_with_events(
            self._df.iloc[s : e + 1],
            events,
            title=f"Candidate region [{start}–{end}] ±{context_margin} context",
            x_offset=s,
        )
        return ToolResult(
            text=f"Event region [{start}–{end}] visualised in [{s}–{e}].",
            images=[b64],
        )


    def lookup_x(self, x_list: List[int]) -> ToolResult:
        """Return the channel values at each index in x_list."""
        n    = len(self._df)
        rows = []
        for x in x_list:
            xi = int(x)
            if 0 <= xi < n:
                vals = {c: round(float(self._df[c].iloc[xi]), 4)
                        for c in self._df.columns}
                rows.append(f"  [{xi}] {vals}")
            else:
                rows.append(f"  [{xi}] out of range (0–{n - 1})")
        return ToolResult(text="Lookup results:\n" + "\n".join(rows))

    def lookup_y(self, col: str, y_value: float) -> ToolResult:
        """Return indices where `col` crosses `y_value` in the current window."""
        if col not in self._df.columns:
            return ToolResult(text=f"Column '{col}' not found.")
        df   = self._window_df()
        vals = df[col].values
        yv   = float(y_value)
        crossings = []
        for i in range(1, len(vals)):
            if (vals[i - 1] < yv <= vals[i]) or (vals[i - 1] >= yv > vals[i]):
                crossings.append(self._x_start + i)
        if not crossings:
            return ToolResult(
                text=f"'{col}' does not cross {yv} in window "
                     f"{self._x_start}–{self._x_end}."
            )
        return ToolResult(
            text=f"'{col}' crosses {yv} at indices: {crossings}"
        )

    def get_value(self) -> ToolResult:
        """Return a downsampled tabular view (~20 rows) of the current window."""
        df      = self._window_df()
        n       = len(df)
        step    = max(n // 20, 1)
        sampled = df.iloc[::step]
        lines   = [f"{'idx':>6}  " + "  ".join(f"{c:>12}" for c in df.columns)]
        lines.append("-" * (8 + 14 * len(df.columns)))
        for i, (idx, row) in enumerate(sampled.iterrows()):
            vals = "  ".join(f"{row[c]:>12.4f}" for c in df.columns)
            lines.append(f"{self._x_start + i * step:>6}  {vals}")
        return ToolResult(text="\n".join(lines))


_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "plot_all",
            "description": "Plot the entire time series. Resets view to full range.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_window",
            "description": "Plot a specific index window [start, end].",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "description": "Start index (inclusive)."},
                    "end":   {"type": "integer", "description": "End index (inclusive)."},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_window_with_window_size",
            "description": "Plot a window centred on mid_idx with the given width.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mid_idx":     {"type": "integer", "description": "Centre index."},
                    "window_size": {"type": "integer", "description": "Total width in rows."},
                },
                "required": ["mid_idx", "window_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_left",
            "description": "Shift the current window left by 3/4 of its span.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_right",
            "description": "Shift the current window right by 3/4 of its span.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_zoom_in_x",
            "description": "Halve the X span (zoom in temporally), centred on the current window.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_zoom_out_x",
            "description": "Double the X span (zoom out temporally), centred on the current window.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_zoom_in_y",
            "description": "Clamp the Y axes to ±½σ around the channel means (zoom in vertically).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_zoom_out_y",
            "description": "Reset Y axes to automatic scaling.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_x",
            "description": "Return channel values at specific row indices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_list": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of row indices to look up.",
                    }
                },
                "required": ["x_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_y",
            "description": "Find row indices where a channel crosses a given value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col":     {"type": "string",  "description": "Channel name."},
                    "y_value": {"type": "number",  "description": "Target value."},
                },
                "required": ["col", "y_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_value",
            "description": "Return a downsampled tabular view (~20 rows) of the current window.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_event_region",
            "description": (
                "Highlight a candidate event region [start, end] with context rows on each side. "
                "Use this to visually confirm a detected event before submitting it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "description": "Candidate event start index (inclusive)."},
                    "end":   {"type": "integer", "description": "Candidate event end index (inclusive)."},
                    "context_margin": {
                        "type": "integer",
                        "description": "Rows of context before and after the region (default 100).",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_derivative",
            "description": "Plot the first derivative of the specified channels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Channel names to differentiate.",
                    }
                },
                "required": ["channels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_second_derivative",
            "description": "Plot the second derivative of the specified channels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Channel names to differentiate twice.",
                    }
                },
                "required": ["channels"],
            },
        },
    },
]


class PlotViewerBackend:
    """ToolBackend wrapping a PlotViewer for one data sample."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._viewer = PlotViewer(df)

    def get_schemas(self) -> List[Dict]:
        return list(_SCHEMAS)

    def get_callables(self) -> Dict[str, Callable]:
        v = self._viewer
        return {
            "plot_all":                    v.plot_all,
            "plot_window":                 v.plot_window,
            "plot_window_with_window_size": v.plot_window_with_window_size,
            "plot_left":                   v.plot_left,
            "plot_right":                  v.plot_right,
            "plot_zoom_in_x":              v.plot_zoom_in_x,
            "plot_zoom_out_x":             v.plot_zoom_out_x,
            "plot_zoom_in_y":              v.plot_zoom_in_y,
            "plot_zoom_out_y":             v.plot_zoom_out_y,
            "lookup_x":                    v.lookup_x,
            "lookup_y":                    v.lookup_y,
            "get_value":                   v.get_value,
            "plot_event_region":           v.plot_event_region,
            "plot_derivative":             v.plot_derivative,
            "plot_second_derivative":      v.plot_second_derivative,
        }


_ORCH_PLOT_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "plot_all",
            "description": "Plot the full time series for a global overview.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_events",
            "description": (
                "Overlay events from one or more detector agents on a single plot. "
                "The view spans the union of all event regions plus a context margin. "
                "Each event class gets a distinct colour regardless of which agent found it. "
                "Pass all planned agents' IDs before submitting to do a global sanity check."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Detector node IDs whose events to overlay (one or more).",
                    },
                    "context_margin": {
                        "type": "integer",
                        "description": "Rows of context before/after the union event region (default 200).",
                    },
                },
                "required": ["agent_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_agents",
            "description": (
                "Plot two agents' detected events side-by-side on the full dataset "
                "to compare their results visually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id_1": {"type": "string", "description": "First detector node ID."},
                    "agent_id_2": {"type": "string", "description": "Second detector node ID."},
                },
                "required": ["agent_id_1", "agent_id_2"],
            },
        },
    },
]


class OrchestratorPlotViewerBackend:
    """Restricted PlotViewer for the orchestrator agent."""

    def __init__(self, df: pd.DataFrame, slots_ref: Dict) -> None:
        self._viewer    = PlotViewer(df)
        self._df        = df
        self._slots_ref = slots_ref

    def plot_all(self) -> ToolResult:
        return self._viewer.plot_all()

    def plot_events(self, agent_ids: List[str], context_margin: int = 200) -> ToolResult:
        all_events: List[Dict] = []
        missing: List[str] = []
        for aid in agent_ids:
            slot = self._slots_ref.get(aid)
            if slot and slot.result and hasattr(slot.result, "items") and slot.result.items:
                all_events.extend(e.model_dump() for e in slot.result.items)
            else:
                missing.append(aid)
        if not all_events:
            return ToolResult(
                text=f"No events found for agent(s): {agent_ids}. "
                     + (f"Missing/empty: {missing}." if missing else "")
            )
        ev_s  = min(e["start"] for e in all_events)
        ev_e  = max(e["end"]   for e in all_events)
        s     = max(0, ev_s - int(context_margin))
        e     = min(len(self._df) - 1, ev_e + int(context_margin))
        label = ", ".join(agent_ids)
        b64   = self._viewer._render_with_events(
            self._df.iloc[s : e + 1], all_events,
            title=f"{label}  — {len(all_events)} event(s)  (±{context_margin} context)",
            x_offset=s,
        )
        note = f"  [{aid}: no events]" if missing else ""
        return ToolResult(
            text=f"{len(all_events)} event(s) from {len(agent_ids)} agent(s) shown in [{s}–{e}].{note}",
            images=[b64],
        )

    def compare_agents(self, agent_id_1: str, agent_id_2: str) -> ToolResult:
        def _events(aid: str) -> List[Dict]:
            sl = self._slots_ref.get(aid)
            return [e.model_dump() for e in sl.result.items] if (
                sl and sl.result and hasattr(sl.result, "items")
            ) else []

        evs1 = _events(agent_id_1)
        evs2 = _events(agent_id_2)
        b64_1 = self._viewer._render_with_events(
            self._df, evs1,
            title=f"{agent_id_1} — {len(evs1)} event(s)",
            x_offset=0,
        )
        b64_2 = self._viewer._render_with_events(
            self._df, evs2,
            title=f"{agent_id_2} — {len(evs2)} event(s)",
            x_offset=0,
        )
        return ToolResult(
            text=f"'{agent_id_1}': {len(evs1)} event(s)  |  '{agent_id_2}': {len(evs2)} event(s).",
            images=[b64_1, b64_2],
        )

    def get_schemas(self) -> List[Dict]:
        return list(_ORCH_PLOT_SCHEMAS)

    def get_callables(self) -> Dict[str, Callable]:
        return {
            "plot_all":       self.plot_all,
            "plot_events":    self.plot_events,
            "compare_agents": self.compare_agents,
        }
