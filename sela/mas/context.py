"""Pluggable conversation context management strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional


def _user_msg(content: str, images: Optional[List[str]] = None) -> Dict:
    """Build a user message, optionally with inline base64 PNG images."""
    if not images:
        return {"role": "user", "content": content}
    parts: List[Any] = [{"type": "text", "text": content}]
    parts += [
        {
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/png;base64,{img}",
                "detail": "high",
            },
        }
        for img in images
    ]
    return {"role": "user", "content": parts}


class ContextStrategy(ABC):
    """Manages the message list that an agent sends to the LLM on each turn."""

    @abstractmethod
    def start_task(self, prompt: str, images: Optional[List[str]] = None) -> None:
        """Initialize context for a new task. Should clear rolling state."""

    @abstractmethod
    def reset(self) -> None:
        """Full reset — called when the agent receives a new data sample."""

    @abstractmethod
    def add_assistant(
        self, content: str, tool_calls_raw: Optional[List[Dict]] = None
    ) -> None:
        """Record the model's response for this turn."""

    @abstractmethod
    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        """Record a tool execution result (text only)."""

    @abstractmethod
    def add_user(self, content: str, images: Optional[List[str]] = None) -> None:
        """Inject a user message."""

    @property
    @abstractmethod
    def messages(self) -> List[Dict]:
        """Return the full message list to send to the LLM API."""

    def rollback(self, n: int = 1) -> int:
        """Remove the last *n* messages from the rolling window."""
        return 0


class FullHistoryContext(ContextStrategy):
    """Retains all messages without pruning."""

    def __init__(self, system_prompt: str) -> None:
        self._permanent = [{"role": "system", "content": system_prompt}]
        self._history: List[Dict] = []

    def start_task(self, prompt: str, images: Optional[List[str]] = None) -> None:
        self._history = [_user_msg(prompt, images)]

    def reset(self) -> None:
        self._history = []

    def add_assistant(
        self, content: str, tool_calls_raw: Optional[List[Dict]] = None
    ) -> None:
        msg: Dict = {"role": "assistant", "content": content}
        if tool_calls_raw:
            msg["tool_calls"] = tool_calls_raw
        self._history.append(msg)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self._history.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
        )

    def add_user(self, content: str, images: Optional[List[str]] = None) -> None:
        self._history.append(_user_msg(content, images))

    def rollback(self, n: int = 1) -> int:
        removed = 0
        for _ in range(n):
            if self._history:
                self._history.pop()
                removed += 1
        return removed

    def rollback_turn(self) -> int:
        """Remove the most recently completed LLM turn from history."""
        removed = 0
        while self._history and self._history[-1].get("role") != "assistant":
            self._history.pop()
            removed += 1
        if self._history and self._history[-1].get("role") == "assistant":
            self._history.pop()
            removed += 1
        return removed

    @property
    def messages(self) -> List[Dict]:
        return self._permanent + self._history


class SlidingWindowContext(ContextStrategy):
    """Keeps system prompt + task instruction permanently; recent turns in a"""

    def __init__(self, system_prompt: str, max_window: int = 40) -> None:
        self._permanent = [{"role": "system", "content": system_prompt}]
        self._task_msg: Optional[Dict] = None
        self._window: deque[Dict] = deque(maxlen=max_window)

    def start_task(self, prompt: str, images: Optional[List[str]] = None) -> None:
        self._task_msg = _user_msg(prompt, images)
        self._window.clear()

    def reset(self) -> None:
        self._task_msg = None
        self._window.clear()

    def add_assistant(
        self, content: str, tool_calls_raw: Optional[List[Dict]] = None
    ) -> None:
        msg: Dict = {"role": "assistant", "content": content}
        if tool_calls_raw:
            msg["tool_calls"] = tool_calls_raw
        self._window.append(msg)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self._window.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
        )

    def add_user(self, content: str, images: Optional[List[str]] = None) -> None:
        self._window.append(_user_msg(content, images))

    def rollback(self, n: int = 1) -> int:
        removed = 0
        for _ in range(n):
            if self._window:
                self._window.pop()
                removed += 1
        return removed

    def rollback_turn(self) -> int:
        """Remove the most recently completed LLM turn from the rolling window."""
        removed = 0
        while self._window and self._window[-1].get("role") != "assistant":
            self._window.pop()
            removed += 1
        if self._window and self._window[-1].get("role") == "assistant":
            self._window.pop()
            removed += 1
        return removed

    @property
    def messages(self) -> List[Dict]:
        msgs = list(self._permanent)
        if self._task_msg:
            msgs.append(self._task_msg)
        window_msgs = list(self._window)
        while window_msgs and window_msgs[0].get("role") == "tool":
            window_msgs.pop(0)
        msgs.extend(window_msgs)
        return msgs


class SummarizationContext(ContextStrategy):
    """Compresses old turns into a running summary when the window fills up."""

    def __init__(self, system_prompt: str, max_window: int = 40) -> None:
        self._inner = SlidingWindowContext(system_prompt, max_window)

    def start_task(self, prompt: str, images: Optional[List[str]] = None) -> None:
        self._inner.start_task(prompt, images)

    def reset(self) -> None:
        self._inner.reset()

    def add_assistant(
        self, content: str, tool_calls_raw: Optional[List[Dict]] = None
    ) -> None:
        self._inner.add_assistant(content, tool_calls_raw)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self._inner.add_tool_result(tool_call_id, result)

    def add_user(self, content: str, images: Optional[List[str]] = None) -> None:
        self._inner.add_user(content, images)

    @property
    def messages(self) -> List[Dict]:
        return self._inner.messages
