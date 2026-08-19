"""Small policy adapters for collection-time action sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class TeleopPolicyCfg(PolicyCfg):
    """Configuration marker for an IsaacLab teleoperation device."""


class TeleopPolicy(PolicyBase[TeleopPolicyCfg]):
    """Expose an IsaacLab device through Arena's PolicyBase contract."""

    def __init__(self, device: Any, config: TeleopPolicyCfg | None = None):
        super().__init__(config or TeleopPolicyCfg())
        self.device = device

    def reset(self, env_ids=None) -> None:
        del env_ids
        self.device.reset()

    def get_action(self, env: Any, observation: Any):
        del env, observation
        action = self.device.advance()
        if action is None:
            return None
        return action.unsqueeze(0) if action.ndim == 1 else action


__all__ = ["TeleopPolicy", "TeleopPolicyCfg"]
