from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from sela.llm.config import LLMConfig
from sela.llm.vllm import VLLM
from sela.tools.plot_viewer import PlotViewer
from sela.baseline._utils import parse_json_block, select_unique
from sela.baseline.numeric import _FORCE_RULE, gt_answer_json

_SYSTEM = """\
# Role & Objective
You are an AI-powered expert in signal processing and time-series analysis.
Your primary objective is to identify the starting and ending indices of
subsequences that represent specific events within time series datasets.

# Context
You will be provided with:
1. A visualisation of the time-series data (all channels, full range).
2. Pattern descriptions of each target event type.\
"""

_USER_TMPL = """\
# Task
You are a Time-Series Event Detector. Inspect the visualisation of the
time-series data and identify all occurrences of the events listed below
with precise start/end indices.

# Event Types
{events}

# Dataset Description
{desc}

# Time Series Info
{info}

# Output Format
Reply with ONLY a JSON array wrapped in ```json ... ``` fences:
```json
[
    {{"className": "event_name", "start": 123, "end": 456, "confidence": 0.95}}
]
```
className must be a string; start/end must be integers; confidence is a float in [0, 1].
{force_rule}\
"""


def _znorm(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        std = result[col].std()
        if std > 0:
            result[col] = (result[col] - result[col].mean()) / std
    return result


class VisualBaseline:

    def __init__(
        self,
        config: LLMConfig,
        unique: bool = True,
        force_all_classes: bool = True,
    ) -> None:
        self._llm = VLLM(config)
        self._unique = unique
        self._force_rule = _FORCE_RULE if force_all_classes else ""

    def demo_messages(
        self, ex_ts: pd.DataFrame, ex_gt: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        plot = PlotViewer(_znorm(ex_ts)).plot_all()
        text = (
            "# Annotated example\n"
            "The plot below is a sample from the same dataset, labelled with its "
            "ground-truth events (given right after it). Use it to learn the "
            "output format and each event type's visual morphology.\n\n"
            "# Time Series Info\n" + plot.text
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}] + [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img}"}}
            for img in plot.images
        ]
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": gt_answer_json(ex_gt)},
        ]

    async def inference(
        self,
        ts: pd.DataFrame,
        desc: str,
        events: List[str],
        instruction: str = "",
        demos: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        viewer  = PlotViewer(_znorm(ts))
        plot    = viewer.plot_all()

        user_prompt = _USER_TMPL.format(
            events="\n".join(f"- {e}" for e in events),
            desc=desc + (f"\n\n{instruction}" if instruction else ""),
            info=plot.text,
            force_rule=self._force_rule,
        )
        messages = [{"role": "system", "content": _SYSTEM}]
        messages += demos or []
        messages += [{"role": "user", "content": user_prompt}]
        raw = ""
        raw_predictions: List[Dict[str, Any]] = []
        try:
            resp = await self._llm.chat(messages, images=plot.images)
            raw = resp.content or ""
            raw_predictions = parse_json_block(raw)
            predictions = (select_unique(raw_predictions)
                           if self._unique else raw_predictions)
        except Exception as exc:
            print(f"[VisualBaseline] inference failed: {exc}")
            predictions = []

        return {
            "predictions": predictions,
            "raw_predictions": raw_predictions,
            "raw_response": raw,
            "token_usage": resp.usage if "resp" in dir() else {},
        }
