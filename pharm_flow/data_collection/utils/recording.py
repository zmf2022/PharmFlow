"""IsaacLab recording helpers used by all collection entry points.

The actual HDF5 implementation remains IsaacLab's RecorderManager.  This
module only provides the project-level configuration and episode commit
boundary needed by the collection runner.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode


def _resolve_recorder_environment(env):
    """Resolve IsaacLab managers behind Gymnasium's wrapper boundary."""

    native_env = getattr(env, "unwrapped", env)
    if not hasattr(native_env, "recorder_manager"):
        raise AttributeError(
            "The unwrapped environment must expose IsaacLab recorder_manager; "
            f"got {type(native_env).__name__}."
        )
    return native_env


class RecorderSession:
    """Commit one episode through IsaacLab's public recorder API."""

    def __init__(self, env) -> None:
        self.env = _resolve_recorder_environment(env)
        self.manager = self.env.recorder_manager

    def reset(self, env_ids=None, *, capture_initial_state: bool = False) -> None:
        """Start a fresh recorder episode.

        The normal collection boundary calls this before ``env.reset()`` so
        IsaacLab's post-reset recorder terms can capture the new scene state.
        ``capture_initial_state`` is for task flows that intentionally keep the
        current physics scene and start a new logical episode.
        """

        self.manager.reset(env_ids)
        if capture_initial_state:
            self.manager.record_post_reset(env_ids)

    def commit(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        success: bool | torch.Tensor,
    ) -> None:
        if isinstance(env_ids, torch.Tensor):
            normalized_ids = env_ids.flatten().tolist()
        else:
            normalized_ids = [int(env_id) for env_id in env_ids]
        if not normalized_ids:
            return

        missing_initial_state = [
            env_id
            for env_id in normalized_ids
            if "initial_state" not in self.manager.get_episode(env_id).data
        ]
        if missing_initial_state:
            raise RuntimeError(
                "Cannot export a Mimic episode without initial_state for "
                f"environment ids {missing_initial_state}. Reset the recorder "
                "before env.reset() and keep the initial-state recorder enabled."
            )

        self.manager.record_pre_reset(normalized_ids, force_export_or_skip=False)
        if isinstance(success, torch.Tensor):
            success_values = success.to(
                device=self.env.device, dtype=torch.bool
            ).reshape(-1, 1)
        else:
            success_values = torch.full(
                (len(normalized_ids), 1),
                bool(success),
                dtype=torch.bool,
                device=self.env.device,
            )
        if success_values.shape[0] != len(normalized_ids):
            raise ValueError("The number of success values must match env_ids.")
        self.manager.set_success_to_episodes(normalized_ids, success_values)
        self.manager.export_episodes(normalized_ids)


__all__ = [
    "ActionStateRecorderManagerCfg",
    "DatasetExportMode",
    "RecorderSession",
]
