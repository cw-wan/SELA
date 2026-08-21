"""Low-level async LLM backends."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: List[ToolCall]
    tool_calls_raw: Optional[List[Dict]]
    usage: Dict[str, int]
    reasoning_content: str = field(default="")


def _image_parts(images: List[str]) -> List[Dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        for img in images
    ]


def _attach_images_to_last_user_msg(
    messages: List[Dict], images: List[str]
) -> List[Dict]:
    """Return a copy of messages with images appended to the last user message."""
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            content = msgs[i].get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            msgs[i] = {**msgs[i], "content": content + _image_parts(images)}
            break
    return msgs


class OpenAIBackend:
    """Covers:"""

    def __init__(self, config) -> None:
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        self.config = config
        if config.api_type == "azure_openai":
            self._client = AsyncAzureOpenAI(
                api_key=config.api_key,
                api_version=config.api_version,
                azure_endpoint=config.base_url,
            )
        else:
            self._client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )

    async def chat(
        self,
        messages: List[Dict],
        tool_schemas: Optional[List[Dict]] = None,
        images: Optional[List[str]] = None,
    ) -> ChatResponse:
        if images:
            messages = _attach_images_to_last_user_msg(messages, images)

        tokens_key = (
            "max_completion_tokens"
            if self.config.api_type == "azure_openai"
            else "max_tokens"
        )
        params: Dict[str, Any] = {
            "model": self.config.deployment_name,
            "messages": messages,
            tokens_key: self.config.max_completion_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort
        if getattr(self.config, "seed", None) is not None:
            params["seed"] = self.config.seed
        if self.config.extra_body:
            params["extra_body"] = self.config.extra_body
        if tool_schemas:
            params["tools"] = tool_schemas
            params["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**params)
        msg = response.choices[0].message

        tool_calls: List[ToolCall] = []
        tool_calls_raw: Optional[List[Dict]] = None
        if msg.tool_calls:
            tool_calls_raw = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
                tool_calls_raw.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        usage: Dict[str, int] = {}
        if response.usage:
            u = response.usage
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
            pd = getattr(u, "prompt_tokens_details", None)
            if pd is not None:
                cached = getattr(pd, "cached_tokens", None)
                if cached is not None:
                    usage["cached_tokens"] = int(cached)
            cd = getattr(u, "completion_tokens_details", None)
            if cd is not None:
                rt = getattr(cd, "reasoning_tokens", None)
                if rt is not None:
                    usage["reasoning_tokens"] = int(rt)

        extra: Dict = msg.model_extra or {}
        reasoning_content: str = (
            getattr(msg, "reasoning_content", None)
            or extra.get("reasoning_content", "")
            or extra.get("thinking_content", "")
            or ""
        )

        if self.config.reasoning_effort and not reasoning_content:
            logger.info(
                "[LLM] reasoning_effort=%r but no reasoning_content found. "
                "model_extra keys: %s",
                self.config.reasoning_effort,
                list(extra.keys()),
            )

        return ChatResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            tool_calls_raw=tool_calls_raw,
            usage=usage,
            reasoning_content=reasoning_content,
        )


class GoogleBackend:
    """Google Gemini backend via google-generativeai SDK."""

    def __init__(self, config) -> None:
        import google.generativeai as genai

        self._genai = genai
        self.config = config
        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(model_name=config.model_name)

    async def chat(
        self,
        messages: List[Dict],
        tool_schemas: Optional[List[Dict]] = None,
        images: Optional[List[str]] = None,
    ) -> ChatResponse:
        import base64, io
        import PIL.Image

        google_msgs = []
        for m in messages:
            if m["role"] == "system":
                continue
            parts: List[Any] = []
            content = m.get("content", "") or ""
            if isinstance(content, str):
                if content:
                    parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64)
                            img = PIL.Image.open(io.BytesIO(img_bytes))
                            parts.append(img)

            if parts:
                role = "model" if m["role"] == "assistant" else "user"
                google_msgs.append({"role": role, "parts": parts})

        gen_cfg = self._genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_completion_tokens,
            candidate_count=1,
        )
        response = await self._model.generate_content_async(
            contents=google_msgs, generation_config=gen_cfg
        )

        text = (response.text or "").strip()
        usage: Dict[str, int] = {}
        if hasattr(response, "usage_metadata"):
            u = response.usage_metadata
            usage = {
                "prompt_tokens": u.prompt_token_count,
                "completion_tokens": u.candidates_token_count,
                "total_tokens": u.total_token_count,
            }

        return ChatResponse(
            content=text,
            tool_calls=[],
            tool_calls_raw=None,
            usage=usage,
        )
