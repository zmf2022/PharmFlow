"""Task graph contract for collection entry points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CollectionTask(ABC):
    """Resolve an environment, robot, and policy without coupling them."""

    name: str

    @abstractmethod
    def build_environment(self, runtime: Any) -> tuple[Any, Any, Any]:
        """Return ``(environment, env_cfg, success_term)``."""

    @abstractmethod
    def build_policy(self, runtime: Any, env: Any, success_term: Any) -> Any:
        """Return the policy/expert for this task and runtime mode."""
