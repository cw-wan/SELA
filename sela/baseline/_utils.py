from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional


def parse_json_block(text: str) -> List[Dict[str, Any]]:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    return json.loads(raw)


def select_unique(
    predictions: List[Dict[str, Any]],
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    valid = [p for p in predictions
             if isinstance(p, dict) and "className" in p
             and "start" in p and "end" in p]
    if not valid:
        return []
    top = max(float(p.get("confidence", 0.0)) for p in valid)
    tied = [p for p in valid if float(p.get("confidence", 0.0)) == top]
    if len(tied) == 1:
        return [tied[0]]
    key = seed if seed is not None else hash(
        json.dumps([{k: t.get(k) for k in ("className", "start", "end")}
                    for t in tied], sort_keys=True)
    )
    return [random.Random(key).choice(tied)]
