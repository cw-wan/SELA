"""Skill abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Type

from sela.mas.artifacts import Artifact, NoneArtifact
from sela.mas.agent import Budget


class Skill(ABC):
    """Abstract base for all domain skills."""


    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's role/persona definition."""

    @abstractmethod
    def build_task_message(self, **kwargs) -> str:
        """Render the task instruction from runtime inputs."""


    @property
    def output_schema(self) -> Type[Artifact]:
        """The Artifact subclass this skill produces. Default: NoneArtifact."""
        return NoneArtifact

    @property
    def required_backend_types(self) -> List[str]:
        """Identifiers of tool backends this skill expects."""
        return []

    @property
    def budget(self) -> Optional[Budget]:
        """Optional budget override. None means use the agent default."""
        return None
