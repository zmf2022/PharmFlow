# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SyncedBatchActionScheduler: hold per-env state until every env needs a new chunk."""

from __future__ import annotations

import torch
from collections.abc import Callable

from isaaclab_arena.policy.action_scheduling.action_chunk_scheduler import ActionChunkScheduler


class SyncedBatchActionScheduler(ActionChunkScheduler):
    """ActionChunkScheduler that waits until ALL envs need a new chunk before calling inference.

    Envs that exhaust their chunk early hold their current robot state until every env is ready.
    Only then is one full-batch inference call made for all envs together.

    Tradeoff vs ActionChunkScheduler:
    - Action tensor batch is always full (N envs, never wasted)
    - Envs that reset early hold their post-reset state for up to
      (action_chunk_length - 1) steps before receiving a fresh action tensor
    """

    def get_action(
        self,
        fetch_action_tensor_fn: Callable[[], torch.Tensor],
        hold_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one action per env, fetching only when all envs need a new chunk.

        Args:
            fetch_action_tensor_fn: Callable that queries the action and returns an action tensor.
            hold_action: (num_envs, action_dim) current robot joint state; applied
                to envs that are waiting for others to catch up. Required for this
                scheduler — defaulted to ``None`` only to match the base API.
        """
        if hold_action is None:
            raise ValueError("SyncedBatchActionScheduler.get_action requires a hold_action tensor")
        if self.env_requires_new_chunk.all():
            self._n_fetch_calls += 1
            self._total_envs_needed += self.num_envs
            self._per_env_fetch_count += 1

            new_chunk = fetch_action_tensor_fn()
            self.current_action_chunk[:] = new_chunk
            self.current_action_index[:] = 0
            self.env_requires_new_chunk[:] = False

        waiting = self.env_requires_new_chunk
        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.current_action_chunk[batch_idx, self.current_action_index.clamp(min=0)]
        action[waiting] = hold_action[waiting]

        self.current_action_index[~waiting] += 1
        exhausted = (~waiting) & (self.current_action_index >= self.action_chunk_length)
        self.current_action_chunk[exhausted] = 0.0
        self.current_action_index[exhausted] = -1
        self.env_requires_new_chunk[exhausted] = True

        return action
