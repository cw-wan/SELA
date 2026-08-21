"""Artifact types for the multi-agent framework."""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, get_args

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """Base class for all inter-node data."""
    pass


class NoneArtifact(Artifact):
    """Sentinel for nodes that produce no meaningful output."""
    pass


A = TypeVar("A", bound=Artifact)


class CollectionArtifact(Artifact, Generic[A]):
    """An ordered collection of same-typed artifacts."""

    items: List[A] = Field(default_factory=list)

    def add(self, item: A) -> None:
        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @classmethod
    def item_type(cls) -> Type[A]:
        """Return the concrete item type (e.g. TSEvent for TSEventCollection)."""
        for base in cls.__dict__.get("__orig_bases__", []):
            args = get_args(base)
            if args and not isinstance(args[0], TypeVar):
                return args[0]
        field = cls.model_fields.get("items")
        if field is not None:
            args = get_args(field.annotation)
            if args and not isinstance(args[0], TypeVar):
                return args[0]
        raise TypeError(f"Cannot resolve item type for {cls.__name__}")


class TaskContextArtifact(Artifact):
    """Carries top-level task context through the graph."""
    description:    str
    csv_path:       str
    target_classes: List[str] = Field(default_factory=list)


class TSEvent(Artifact):
    """A single detected event in a time-series sample."""
    class_name: str   = Field(description="Event class label.")
    start:      int   = Field(description="Inclusive start index (row number).")
    end:        int   = Field(description="Inclusive end index (row number).")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Detection confidence in [0, 1].",
    )


class TSEventCollection(CollectionArtifact[TSEvent]):
    """Ordered collection of TSEvent detections produced by one agent."""
    comment: str = Field(
        default="",
        description="Agent's reasoning summary — evidence supporting the detections.",
    )


class ReviewArtifact(Artifact):
    """Reviewer's verdict: which candidate (0-based) is best and how confident."""
    choice:     int   = Field(
        description="0-based index of the preferred candidate."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Reviewer's confidence in the choice, on a 0–10 scale.",
    )


class PlannerSubtask(Artifact):
    """One sub-task emitted by a planner agent."""
    description:    str            = Field(description="Natural-language task description.")
    target_classes: List[str]      = Field(default_factory=list)
    slice_start:    Optional[int]  = Field(default=None, description="Suggested window start index.")
    slice_end:      Optional[int]  = Field(default=None, description="Suggested window end index.")


class PlannerSubtaskCollection(CollectionArtifact[PlannerSubtask]):
    """All sub-tasks emitted by a planner."""
    pass


class ELTSchemaArtifact(Artifact):
    """Compiled Event Logic Tree schema produced by a parser agent."""
    schema_dict: Dict[str, Any] = Field(
        description=(
            'Serialised ELT definition. Must have keys "primitives" (list of '
            '{alias, target_channel, description}) and "composites" (list of '
            '{alias, operator, children, description}).'
        )
    )
    root_alias: str = Field(description="Alias of the root composite node (e.g. 'Root').")
