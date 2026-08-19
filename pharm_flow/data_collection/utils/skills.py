"""Small contracts for composing data-collection atomic skills."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AtomicSkillContext:
    """Stable input passed to every atomic skill invocation."""

    env: Any
    stage: str
    target_name: str
    config: Any
    runtime: Any


@dataclass(frozen=True)
class AtomicSkillOutput:
    """Result returned by a skill after one environment decision step."""

    action: Any
    next_skill: str | None = None
    done: bool = False
    failed: bool = False
    info: dict[str, Any] | None = None


SkillHandler = Callable[[AtomicSkillContext], AtomicSkillOutput]


@dataclass(frozen=True)
class AtomicSkill:
    """Named, callable skill implementation used by a task pipeline."""

    name: str
    handler: SkillHandler

    def execute(self, context: AtomicSkillContext) -> AtomicSkillOutput:
        if context.stage != self.name:
            raise ValueError(
                f"Skill context stage {context.stage!r} does not match {self.name!r}."
            )
        result = self.handler(context)
        if not isinstance(result, AtomicSkillOutput):
            raise TypeError(
                f"Skill {self.name!r} must return AtomicSkillOutput, "
                f"got {type(result).__name__}."
            )
        return result


def validate_skill_pipeline(
    pipeline: Iterable[str], skills: dict[str, AtomicSkill]
) -> tuple[str, ...]:
    """Validate and normalize a task's ordered skill names."""

    names = tuple(str(name) for name in pipeline)
    if not names:
        raise ValueError("A task skill pipeline cannot be empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"A task skill pipeline cannot repeat a skill: {names}")
    unknown = tuple(name for name in names if name not in skills)
    if unknown:
        raise ValueError(
            f"Unknown atomic skills {unknown}; available skills: {tuple(skills)}"
        )
    return names


__all__ = [
    "AtomicSkill",
    "AtomicSkillContext",
    "AtomicSkillOutput",
    "validate_skill_pipeline",
]
