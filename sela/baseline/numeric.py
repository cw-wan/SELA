from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from sela.llm.config import LLMConfig
from sela.llm.vllm import VLLM
from sela.baseline._utils import parse_json_block, select_unique


def gt_answer_json(gt_events: List[Dict[str, Any]]) -> str:
    arr = [{"className": e["className"], "start": int(e["start"]),
            "end": int(e["end"]), "confidence": 1.0} for e in gt_events]
    return "```json\n" + json.dumps(arr, ensure_ascii=False) + "\n```"

_SYSTEM = """\
# Role & Objective
You are an AI expert in time-series analysis. \
Your primary objective is to identify the starting and ending indices of \
subsequences that represent specific events within time series datasets.

# Context
You will be provided with:
1. Pattern descriptions of each target event type.
2. Time-series data.
"""

_USER_TMPL = """\
# Task
You are a Time-Series Event Detector. Identify occurrences of the events \
listed below with start/end indices.

# Event Types
{events}

# Dataset Description
{desc}

# Output Format
Reply with ONLY a JSON array wrapped in ```json ... ``` fences:
```json
[
    {{"className": "event_name", "start": 123, "end": 456, "confidence": 0.95}}
]
```
className must be a string; start/end must be integers; confidence is a float in [0, 1].
{force_rule}
# Time Series Data
{data}
"""

_FORCE_RULE = """
# Mandatory: one candidate per event type
You MUST output exactly one best-candidate subsequence for EVERY event type listed
above — one array entry per event type, no more and no fewer. Even if an event type
appears absent, still report your single best-guess location for it and express that
doubt with a LOW confidence. The confidence values rank the event types against each
other for this sample: the type you believe is genuinely present must receive the
highest confidence.
"""


class NumericBaseline:

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
        user = (
            "# Annotated example\n"
            "Below is a time series from the same dataset, labelled with its "
            "ground-truth events (given right after the data). Use it to learn "
            "the output format and each event type's numeric signature.\n\n"
            "# Time Series Data\n" + ex_ts.to_csv().strip()
        )
        return [
            {"role": "user", "content": user},
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
        user_prompt = _USER_TMPL.format(
            events="\n".join(f"- {e}" for e in events),
            desc=desc + (f"\n\n{instruction}" if instruction else ""),
            force_rule=self._force_rule,
            data=ts.to_csv().strip()
        )
        messages = [{"role": "system", "content": _SYSTEM}]
        messages += demos or []
        messages += [{"role": "user", "content": user_prompt}]
        raw = ""
        raw_predictions: List[Dict[str, Any]] = []
        try:
            resp = await self._llm.chat(messages)
            raw = resp.content or ""
            raw_predictions = parse_json_block(raw)
            predictions = (select_unique(raw_predictions)
                           if self._unique else raw_predictions)
        except Exception as exc:
            print(f"[NumericBaseline] inference failed: {exc}")
            predictions = []

        return {
            "predictions": predictions,
            "raw_predictions": raw_predictions,
            "raw_response": raw,
            "token_usage": resp.usage if "resp" in dir() else {},
        }
