"""SELA — Signal Event Logic Analysis system."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from sela.llm.config import LLMConfig
from sela.llm.vllm import VLLM
from sela.mas.agent import Agent
from sela.mas.artifacts import ELTSchemaArtifact, TSEventCollection
from sela.skills.elt_reasoner import (
    SELAComparativeInspectorSkill,
    SELAParserSkill,
    SELASingleTreeInspectorSkill,
)
from sela.tools.elt import (
    ELT,
    ELTBackend,
    MultiELTBackend,
    find_main_instance,
    normalized_confidence,
)

logger = logging.getLogger(__name__)


class SELASystem:
    """Two-agent ELT-based event detection system."""

    def __init__(
        self,
        config: LLMConfig,
        on_event: Optional[Any] = None,
        schema_cache_dir: Optional[str] = None,
        reparse_each_sample: bool = False,
        numeric_readout: bool = False,
        numeric_max_rows: int = 40,
        resample_mode: str = "decimate",
        oracle_schema_dir: Optional[str] = None,
        simplified_schema: bool = False,
        parallel_inspectors: bool = False,
    ) -> None:
        self._llm = VLLM(config)
        self._schema_cache: Dict[str, Optional[ELTSchemaArtifact]] = {}
        self._on_event = on_event
        self._schema_cache_dir = Path(schema_cache_dir) if schema_cache_dir else None
        self._reparse_each_sample = reparse_each_sample
        self._oracle_schema_dir = Path(oracle_schema_dir) if oracle_schema_dir else None
        self._simplified_schema = simplified_schema
        self._numeric_readout  = numeric_readout
        self._numeric_max_rows = numeric_max_rows
        self._resample_mode    = resample_mode
        self._parallel_inspectors = parallel_inspectors
        self._tok = self._zero_tok()

    @staticmethod
    def _zero_tok() -> Dict[str, int]:
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cached_tokens": 0, "reasoning_tokens": 0}

    def _accum_tokens(self, agent: Agent) -> None:
        """Add an agent's cumulative token counts to the per-inference tally."""
        self._tok["total_tokens"]      += agent._total_tokens
        self._tok["prompt_tokens"]     += agent._total_prompt_tokens
        self._tok["completion_tokens"] += agent._total_completion_tokens
        self._tok["cached_tokens"]     += agent._total_cached_tokens
        self._tok["reasoning_tokens"]  += agent._total_reasoning_tokens

    def _tagged_cb(self, agent_id: str):
        """Wrap the user callback so every event carries the agent identity."""
        if self._on_event is None:
            return None
        cb = self._on_event
        async def _wrapped(ev: Dict[str, Any]) -> None:
            import asyncio as _aio
            result = cb({**ev, "agent": agent_id})
            if _aio.iscoroutine(result):
                await result
        return _wrapped


    def _load_oracle_schema(
        self, class_name: str, channels: List[str]
    ) -> Optional[ELTSchemaArtifact]:
        """Load a human-authored schema `<oracle_dir>/<class>.json` and validate it."""
        fname = class_name.replace(" ", "_") + ".json"
        path = self._oracle_schema_dir / fname
        if not path.exists():
            logger.warning("[SELA-oracle] no schema for %r at %s", class_name, path)
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            prims = [
                {"alias": p["alias"],
                 "target_channel": p.get("target_channel", p.get("channel")),
                 "description": p.get("description", "")}
                for p in raw["primitives"]
            ]
            schema_dict = {"primitives": prims, "composites": raw["composites"]}
            root_alias = raw.get("root") or raw.get("root_alias")
            probe = ELT(pd.DataFrame(columns=channels), require_main=False)
            msg = probe.load_schema(schema_dict, root_alias)
            if not probe._compiled:
                logger.warning("[SELA-oracle] %r failed validation: %s",
                               class_name, msg)
                return None
            if not any(a.lower() == "main"
                       for a in [p["alias"] for p in prims]
                       + [c["alias"] for c in raw["composites"]]):
                logger.warning("[SELA-oracle] %r has no 'Main' node.", class_name)
                return None
            logger.info("[SELA-oracle] loaded human schema for %r (root=%s, "
                        "prims=%d, comps=%d)", class_name, root_alias,
                        len(prims), len(raw["composites"]))
            return ELTSchemaArtifact(schema_dict=schema_dict, root_alias=root_alias)
        except Exception as exc:
            logger.warning("[SELA-oracle] could not load %r: %s", class_name, exc)
            return None

    async def _parse_schema(
        self, class_name: str, desc: str, channels: List[str]
    ) -> Optional[ELTSchemaArtifact]:
        """Return the ELT schema for a class — oracle-loaded, cached, or parsed."""
        if self._oracle_schema_dir is not None:
            return self._load_oracle_schema(class_name, channels)

        if not self._reparse_each_sample and class_name in self._schema_cache:
            cached = self._schema_cache[class_name]
            logger.info("[SELA] Schema for %r loaded from cache.", class_name)
            return cached

        disk_path = (
            self._schema_cache_dir / f"{class_name.replace(' ', '_')}.json"
            if self._schema_cache_dir and not self._reparse_each_sample else None
        )
        if disk_path is not None and disk_path.exists():
            try:
                payload = json.loads(disk_path.read_text(encoding="utf-8"))
                art = ELTSchemaArtifact(
                    schema_dict=payload["schema_dict"],
                    root_alias=payload["root_alias"],
                )
                probe = ELT(pd.DataFrame(columns=channels), require_main=True, main_only=self._simplified_schema)
                probe.load_schema(art.schema_dict, art.root_alias)
                if probe._compiled:
                    logger.info(
                        "[SELA] Schema for %r loaded from disk cache (%s).",
                        class_name, disk_path,
                    )
                    self._schema_cache[class_name] = art
                    return art
                logger.warning(
                    "[SELA] Disk-cached schema for %r failed validation — "
                    "re-parsing.", class_name,
                )
            except Exception as exc:
                logger.warning(
                    "[SELA] Could not load disk cache for %r (%s) — re-parsing.",
                    class_name, exc,
                )

        skill = SELAParserSkill(simplified=self._simplified_schema)
        elt_backend = ELTBackend(pd.DataFrame(columns=channels),
                                 require_main=True,
                                 main_only=self._simplified_schema)
        agent = Agent.from_skill(
            skill,
            self._llm,
            agent_id=f"sela_parser_{class_name.replace(' ', '_')}",
            tool_backends=[elt_backend],
        )
        task_prompt = skill.build_task_message(description=desc, event_name=class_name)
        try:
            result = await agent.run(
                task_prompt, ELTSchemaArtifact,
                on_event=self._tagged_cb(agent.agent_id),
            )
            self._accum_tokens(agent)
            schema_art: Optional[ELTSchemaArtifact] = (
                result if isinstance(result, ELTSchemaArtifact) else None
            )
        except Exception as exc:
            logger.warning("[SELA] Parser failed for %r: %s", class_name, exc)
            schema_art = None

        if schema_art is not None:
            probe = ELT(pd.DataFrame(columns=channels), require_main=True, main_only=self._simplified_schema)
            try:
                msg = probe.load_schema(
                    schema_art.schema_dict, schema_art.root_alias
                )
            except Exception as exc:
                msg = f"load error: {exc}"
            if not probe._compiled:
                logger.warning(
                    "[SELA] Final schema for %r failed validation — "
                    "discarding.\n%s", class_name, msg,
                )
                schema_art = None

        if schema_art is None:
            logger.warning(
                "[SELA] Parser returned no schema for class %r — "
                "will produce zero-confidence prediction.",
                class_name,
            )
        elif disk_path is not None:
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                disk_path.write_text(
                    json.dumps(
                        {"schema_dict": schema_art.schema_dict,
                         "root_alias":  schema_art.root_alias},
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.info("[SELA] Schema for %r saved to %s.",
                            class_name, disk_path)
            except Exception as exc:
                logger.warning("[SELA] Failed to write schema cache: %s", exc)
        self._schema_cache[class_name] = schema_art
        logger.info(
            "[SELA] Parsed schema for %r: root=%s, prims=%d, comps=%d",
            class_name,
            getattr(schema_art, "root_alias", "?"),
            len((getattr(schema_art, "schema_dict", {}) or {}).get("primitives", [])),
            len((getattr(schema_art, "schema_dict", {}) or {}).get("composites", [])),
        )
        return schema_art


    @staticmethod
    def _read_tree(elt) -> Tuple[float, int, int]:
        """Read (norm_conf, main_start, main_end) from an instantiated ELT."""
        roots = elt.instantiate_all()
        root  = roots[0] if roots else None
        if root is None or not hasattr(root, "children"):
            return 0.0, 0, 0
        norm_conf = normalized_confidence(root)
        main_inst = find_main_instance(root)
        if main_inst is not None:
            return norm_conf, main_inst.start, main_inst.end
        return norm_conf, root.start, root.end


    async def inference(
        self,
        ts:          pd.DataFrame,
        desc:        str,
        events:      List[str],
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Detect one event in a time-series sample."""
        self._tok = self._zero_tok()

        channels = [str(c) for c in ts.columns]
        schemas: Dict[str, Optional[ELTSchemaArtifact]] = {}
        for cls in events:
            schemas[cls] = await self._parse_schema(cls, desc, channels)
        valid = {c: s for c, s in schemas.items() if s is not None}

        results: List[Dict[str, Any]] = []

        elts: Dict[str, Any] = {}
        if valid and self._parallel_inspectors:
            async def _inspect_one(cls: str, art: ELTSchemaArtifact):
                pb = MultiELTBackend(
                    ts, {cls: (art.schema_dict, art.root_alias)},
                    numeric_readout=self._numeric_readout,
                    numeric_max_rows=self._numeric_max_rows,
                    resample_mode=self._resample_mode,
                )
                if not pb.tree_names:
                    return cls, None
                skill = SELASingleTreeInspectorSkill()
                agent_id = "sela_inspector_" + cls.replace(" ", "_")
                agent = Agent.from_skill(
                    skill, self._llm, agent_id=agent_id, tool_backends=[pb],
                )
                tree_info = (
                    f"- `{cls}`: {len(art.schema_dict.get('primitives', []))} "
                    f"primitives, {len(art.schema_dict.get('composites', []))} "
                    f"composites"
                )
                await agent.run(
                    skill.build_task_message(class_name=cls, description=desc,
                                             tree_info=tree_info),
                    TSEventCollection,
                    on_event=self._tagged_cb(agent_id),
                )
                self._accum_tokens(agent)
                return cls, pb.elt(cls)

            pairs = await asyncio.gather(
                *(_inspect_one(c, s) for c, s in valid.items()))
            elts = {c: e for c, e in pairs if e is not None}
        elif valid:
            backend = MultiELTBackend(
                ts,
                {c: (s.schema_dict, s.root_alias) for c, s in valid.items()},
                numeric_readout=self._numeric_readout,
                numeric_max_rows=self._numeric_max_rows,
                resample_mode=self._resample_mode,
            )
            if backend.tree_names:
                skill = SELAComparativeInspectorSkill()
                agent = Agent.from_skill(
                    skill,
                    self._llm,
                    agent_id="sela_comparative_inspector",
                    tool_backends=[backend],
                )
                tree_info = "\n".join(
                    f"- `{c}`: {len(s.schema_dict.get('primitives', []))} "
                    f"primitives, {len(s.schema_dict.get('composites', []))} "
                    f"composites"
                    for c, s in valid.items()
                )
                task_prompt = skill.build_task_message(
                    classes=backend.tree_names,
                    description=desc,
                    tree_info=tree_info,
                )
                await agent.run(
                    task_prompt, TSEventCollection,
                    on_event=self._tagged_cb(agent.agent_id),
                )
                self._accum_tokens(agent)
            elts = {c: backend.elt(c) for c in backend.tree_names}

        for cls in events:
            elt = elts.get(cls)
            if elt is None:
                results.append({"class": cls, "norm_conf": 0.0, "start": 0,
                                "end": max(len(ts) - 1, 0)})
                continue
            norm_conf, start, end = self._read_tree(elt)
            logger.info(
                "[SELA][%s] norm_conf=%.4f  main=[%d, %d]",
                cls, norm_conf, start, end,
            )
            results.append({
                "class":     cls,
                "norm_conf": norm_conf,
                "start":     start,
                "end":       end,
            })

        winner = max(results, key=lambda r: r["norm_conf"])
        prediction = {
            "className":  winner["class"],
            "start":      int(winner["start"]),
            "end":        int(winner["end"]),
            "confidence": round(float(winner["norm_conf"]), 6),
        }

        return {
            "predictions": [prediction],
            "token_usage": dict(self._tok),
            "class_scores": {
                r["class"]: round(float(r["norm_conf"]), 6)
                for r in results
            },
        }
