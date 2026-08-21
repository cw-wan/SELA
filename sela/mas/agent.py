"""Single agent with a ReAct (Reason + Act) loop."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type

from pydantic import create_model

from sela.llm.vllm import VLLM
from sela.llm.backends import ChatResponse
from .artifacts import Artifact, CollectionArtifact, NoneArtifact
from .context import ContextStrategy, SlidingWindowContext
from .tool_backend import ToolBackend, ToolResult

if TYPE_CHECKING:
    from sela.skills.base import Skill

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[Dict], Awaitable[None]]]


async def _emit(callback: EventCallback, event: Dict) -> None:
    """Fire the on_event callback if provided; handles both sync and async."""
    if callback is None:
        return
    result = callback(event)
    if asyncio.iscoroutine(result):
        await result


@dataclass
class Budget:
    max_turns:  int = 20
    max_tokens: int = 100_000


def _build_collection_output_tool(
    schema_cls: Type[CollectionArtifact],
) -> Tuple[Dict, Callable, CollectionArtifact]:
    """Generates a submit_item tool for a CollectionArtifact subclass."""
    item_type   = schema_cls.item_type()
    json_schema = item_type.model_json_schema()
    tool_schema = {
        "type": "function",
        "function": {
            "name": "submit_item",
            "description": f"Submit one detected {item_type.__name__}. Call once per item.",
            "parameters": {
                "type":       "object",
                "properties": json_schema.get("properties", {}),
                "required":   json_schema.get("required",   []),
            },
        },
    }
    buffer = schema_cls()

    def submit_item(**kwargs: Any) -> str:
        try:
            item = item_type(**kwargs)
            buffer.add(item)
            return f"Submitted. Running total: {len(buffer)}."
        except Exception as exc:
            return f"Validation error: {exc}"

    return tool_schema, submit_item, buffer


def _build_single_output_tool(
    schema_cls: Type[Artifact],
) -> Tuple[Dict, Callable, List[Optional[Artifact]]]:
    """Generates a submit_result tool for a single Artifact subclass."""
    json_schema = schema_cls.model_json_schema()
    fields      = {
        k: (v.annotation, v)
        for k, v in schema_cls.model_fields.items()
        if not v.exclude
    }
    input_model         = create_model(f"_{schema_cls.__name__}Input", **fields)
    holder: List[Optional[Any]] = [None]

    tool_schema = {
        "type": "function",
        "function": {
            "name": "submit_result",
            "description": f"Submit the final {schema_cls.__name__}.",
            "parameters": {
                "type":       "object",
                "properties": json_schema.get("properties", {}),
                "required":   json_schema.get("required",   []),
            },
        },
    }

    def submit_result(**kwargs: Any) -> str:
        try:
            holder[0] = input_model(**kwargs)
            return "Result submitted."
        except Exception as exc:
            return f"Validation error: {exc}"

    return tool_schema, submit_result, holder


_REASONING_ADDENDUM = (
    "\n\n## Reasoning requirement\n"
    "Before calling any tool, always write a brief explanation in plain text "
    "(1–3 sentences): what you observe in the current view, what you intend "
    "to do next, and why. This text must appear before the tool call in the "
    "same response — never call a tool without accompanying reasoning text."
)


class Agent:
    """General-purpose reactive agent."""

    def __init__(
        self,
        agent_id:         str,
        llm:              VLLM,
        system_prompt:    str,
        tool_backends:    Optional[List[ToolBackend]] = None,
        budget:           Optional[Budget]            = None,
        context:          Optional[ContextStrategy]   = None,
        retry_limit:      int                         = 5,
        require_reasoning: bool                       = True,
    ) -> None:
        self.agent_id   = agent_id
        self.llm        = llm
        self.budget     = budget or Budget()
        self.retry_limit = retry_limit

        effective_prompt = (
            system_prompt + _REASONING_ADDENDUM if require_reasoning else system_prompt
        )
        self.context: ContextStrategy = context or SlidingWindowContext(
            effective_prompt, max_window=40
        )
        self._tool_backends: List[ToolBackend] = list(tool_backends or [])

        self._out_schema:     Optional[Dict]     = None
        self._out_fn:         Optional[Callable] = None
        self._out_buffer:     Any                = None
        self._comment_schema: Optional[Dict]     = None
        self._comment_fn:     Optional[Callable] = None

        self._total_tokens:            int = 0
        self._total_prompt_tokens:     int = 0
        self._total_completion_tokens: int = 0
        self._total_cached_tokens:     int = 0
        self._total_reasoning_tokens:  int = 0


    @classmethod
    def from_skill(
        cls,
        skill:         "Skill",
        llm:           VLLM,
        agent_id:      str           = "agent",
        tool_backends: Optional[List[ToolBackend]] = None,
        **kwargs,
    ) -> "Agent":
        """Create an Agent pre-configured from a Skill."""
        budget = skill.budget or Budget()
        return cls(
            agent_id=agent_id,
            llm=llm,
            system_prompt=skill.system_prompt,
            tool_backends=list(tool_backends or []),
            budget=budget,
            **kwargs,
        )


    def _all_schemas(self) -> List[Dict]:
        schemas: List[Dict] = []
        for backend in self._tool_backends:
            schemas.extend(backend.get_schemas())
        if self._out_schema:
            schemas.append(self._out_schema)
        if self._comment_schema:
            schemas.append(self._comment_schema)
        return schemas

    def _all_callables(self) -> Dict[str, Callable]:
        callables: Dict[str, Callable] = {}
        for backend in self._tool_backends:
            callables.update(backend.get_callables())
        if self._out_fn and self._out_schema:
            callables[self._out_schema["function"]["name"]] = self._out_fn
        if self._comment_fn and self._comment_schema:
            callables["submit_comment"] = self._comment_fn
        return callables


    def reset(self) -> None:
        """Full reset for a new data sample."""
        self.context.reset()


    async def run(
        self,
        task_prompt:   str,
        output_schema: Type[Artifact],
        images:        Optional[List[str]] = None,
        on_event:      EventCallback       = None,
    ) -> Artifact:
        """Execute one task with a ReAct loop and return the collected Artifact."""
        self._setup_output_tool(output_schema)
        self.context.start_task(task_prompt, images=images)

        tokens_used        = 0
        prompt_tokens      = 0
        completion_tokens  = 0
        cached_tokens      = 0
        reasoning_tokens   = 0
        schemas            = self._all_schemas() or None
        last_turn          = 0

        for turn in range(self.budget.max_turns):
            last_turn = turn
            if tokens_used >= self.budget.max_tokens:
                logger.warning(
                    "[%s] Token budget exhausted after %d turns", self.agent_id, turn
                )
                break

            await _emit(on_event, {"type": "turn_start", "turn": turn + 1})

            response          = await self._call_llm_with_retry(schemas)
            tokens_used       += response.usage.get("total_tokens",       0)
            prompt_tokens     += response.usage.get("prompt_tokens",      0)
            completion_tokens += response.usage.get("completion_tokens",  0)
            cached_tokens     += response.usage.get("cached_tokens",      0)
            reasoning_tokens  += response.usage.get("reasoning_tokens",   0)

            if response.reasoning_content:
                await _emit(on_event, {
                    "type": "reasoning",
                    "content": response.reasoning_content,
                    "turn": turn + 1,
                })
            if response.content:
                await _emit(on_event, {
                    "type": "thinking",
                    "content": response.content,
                    "turn": turn + 1,
                })

            if response.tool_calls:
                self.context.add_assistant(response.content, response.tool_calls_raw)

                callables      = self._all_callables()
                pending_images: List[str] = []

                for tc in response.tool_calls:
                    await _emit(on_event, {
                        "type": "tool_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "turn": turn + 1,
                    })

                    result = await self._invoke_tool(tc.name, tc.arguments, callables)
                    logger.debug(
                        "[%s] tool %s -> %s", self.agent_id, tc.name, result.text[:120]
                    )

                    await _emit(on_event, {
                        "type": "tool_result",
                        "id": tc.id,
                        "name": tc.name,
                        "text": result.text,
                        "images": result.images,
                        "images_svg": result.images_svg,
                        "turn": turn + 1,
                    })

                    if tc.name in ("submit_item", "submit_result", "submit_comment"):
                        await _emit(on_event, {
                            "type": "output_submitted",
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "turn": turn + 1,
                        })

                    self.context.add_tool_result(tc.id, result.text)
                    pending_images.extend(result.images)

                if pending_images:
                    self.context.add_user(
                        "The tool generated the chart(s) above. Analyse them to continue.",
                        images=pending_images,
                    )

                if self._has_output() and not isinstance(self._out_buffer, CollectionArtifact):
                    break
            else:
                self.context.add_assistant(response.content)
                if self._has_output():
                    break
                if isinstance(self._out_buffer, CollectionArtifact):
                    self.context.add_user(
                        "Continue your analysis. When you have finished examining all "
                        "event classes, call submit_comment with your reasoning summary "
                        "— this is required even if no events were detected."
                    )
                else:
                    self.context.add_user(
                        "Continue reasoning. Use the available tools to make progress "
                        "and call the submit tool when you have your final answer."
                    )

        if not self._has_output():
            logger.warning("[%s] Finished without submitting output.", self.agent_id)

        self._total_tokens            += tokens_used
        self._total_prompt_tokens     += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cached_tokens     += cached_tokens
        self._total_reasoning_tokens  += reasoning_tokens

        await _emit(on_event, {
            "type":                    "done",
            "turns":                   last_turn + 1,
            "tokens":                  tokens_used,
            "prompt_tokens":           prompt_tokens,
            "completion_tokens":       completion_tokens,
            "cached_tokens":           cached_tokens,
            "reasoning_tokens":        reasoning_tokens,
            "total_tokens":            self._total_tokens,
            "total_prompt_tokens":     self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        })

        return self._collect_output(output_schema)


    async def _call_llm_with_retry(
        self, schemas: Optional[List[Dict]]
    ) -> ChatResponse:
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(self.retry_limit):
            try:
                return await self.llm.chat(
                    messages=self.context.messages,
                    tool_schemas=schemas,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] LLM call failed (attempt %d/%d): %s",
                    self.agent_id, attempt + 1, self.retry_limit, exc,
                )
                if attempt < self.retry_limit - 1:
                    exc_str = str(exc)
                    if "content_filter" in exc_str or "ResponsibleAIPolicyViolation" in exc_str:
                        rolled = self.context.rollback_turn()
                        logger.info(
                            "[%s] Content filter: rolled back %d message(s), retrying.",
                            self.agent_id, rolled,
                        )
                    else:
                        await asyncio.sleep(min(2 ** attempt, 30))
        raise last_exc

    @staticmethod
    async def _invoke_tool(
        name: str, arguments: Dict, callables: Dict[str, Callable]
    ) -> ToolResult:
        """Execute a tool call (sync or async) and normalise the return value to ToolResult."""
        if name not in callables:
            return ToolResult(text=f"Error: unknown tool '{name}'.")
        try:
            result = callables[name](**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, ToolResult):
                return result
            return ToolResult(text=str(result))
        except Exception as exc:
            return ToolResult(text=f"Error executing '{name}': {exc}")


    def _setup_output_tool(self, schema_cls: Type[Artifact]) -> None:
        self._comment_schema = None
        self._comment_fn     = None

        if schema_cls is NoneArtifact:
            self._out_schema = self._out_fn = self._out_buffer = None
            return

        if issubclass(schema_cls, CollectionArtifact):
            schema, fn, buf = _build_collection_output_tool(schema_cls)
            self._comment_schema = {
                "type": "function",
                "function": {
                    "name": "submit_comment",
                    "description": (
                        "Record a reasoning summary that supports your detections. "
                        "Call once after all submit_item calls are done. "
                        "Write 2–5 sentences describing the morphological evidence "
                        "you observed for each detected event."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "comment": {
                                "type": "string",
                                "description": "Evidence summary supporting your detections.",
                            }
                        },
                        "required": ["comment"],
                    },
                },
            }
            _buf = buf
            def _submit_comment(comment: str) -> str:
                _buf.comment = str(comment)
                return "Comment recorded."
            self._comment_fn = _submit_comment
        else:
            schema, fn, buf = _build_single_output_tool(schema_cls)

        self._out_schema = schema
        self._out_fn     = fn
        self._out_buffer = buf

    def _has_output(self) -> bool:
        buf = self._out_buffer
        if buf is None:
            return False
        if isinstance(buf, CollectionArtifact):
            return bool(buf.comment)
        if isinstance(buf, list):
            return buf[0] is not None
        return False

    def _collect_output(self, schema_cls: Type[Artifact]) -> Artifact:
        if schema_cls is NoneArtifact or self._out_buffer is None:
            return NoneArtifact()
        if isinstance(self._out_buffer, CollectionArtifact):
            return self._out_buffer
        if isinstance(self._out_buffer, list):
            held = self._out_buffer[0]
            if held is None:
                return NoneArtifact()
            return schema_cls(**held.model_dump())
        return NoneArtifact()
