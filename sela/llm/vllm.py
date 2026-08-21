"""Unified, stateless LLM client."""
from __future__ import annotations

from typing import Dict, List, Optional

from .config import LLMConfig, load_llm_config
from .backends import ChatResponse, OpenAIBackend, GoogleBackend


class VLLM:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        if config.api_type in ("azure_openai", "openai"):
            self._backend = OpenAIBackend(config)
        elif config.api_type == "google":
            self._backend = GoogleBackend(config)
        else:
            raise ValueError(f"Unknown api_type: {config.api_type!r}")

    @classmethod
    def from_config_file(cls, config_path: str, model_name: str) -> "VLLM":
        return cls(load_llm_config(config_path, model_name))

    async def chat(
        self,
        messages: List[Dict],
        tool_schemas: Optional[List[Dict]] = None,
        images: Optional[List[str]] = None,
    ) -> ChatResponse:
        return await self._backend.chat(
            messages=messages,
            tool_schemas=tool_schemas,
            images=images,
        )
