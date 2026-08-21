"""LLM token pricing and cost estimation."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PRICING: Dict[str, Dict[str, float]] = {
    "gpt-5":   {"input": 1.25, "cached": 0.13, "output": 10.0},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.0},
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """Return (cost_usd, price_row) for the given token counts, or (None, None) if"""
    price = PRICING.get(model)
    if price is None:
        return None, None
    uncached = max(int(prompt_tokens) - int(cached_tokens), 0)
    cost = (
        uncached * price["input"]
        + int(cached_tokens) * price["cached"]
        + int(completion_tokens) * price["output"]
    ) / 1_000_000.0
    return cost, price
