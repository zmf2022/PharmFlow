"""Generic single-environment collection loop.

The loop is shared by teleoperation and scripted experts.  Task-specific
composition belongs in ``data_collection/tasks``; this module only coordinates
the Arena policy lifecycle with IsaacLab recording and episode boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


def _has_any(value) -> bool:
    return bool(value.any().item()) if isinstance(value, torch.Tensor) else bool(value)


@dataclass
class CollectionRunnerCfg:
    """Controls episode boundaries; task success stays in a callback."""

    max_steps_per_episode: int
    success_hold_steps: int = 1


class CollectionRunner:
    """Drive any Arena ``PolicyBase`` and an IsaacLab recorder together."""

    def __init__(
        self,
        env,
        policy,
        config: CollectionRunnerCfg,
        success_fn: Callable[..., torch.Tensor | bool],
        *,
        should_continue: Callable[[], bool] | None = None,
        discard_requested: Callable[[], bool] | None = None,
        on_episode_complete: Callable[[bool], bool] | None = None,
        on_step: Callable[[Any], None] | None = None,
        on_idle: Callable[[], None] | None = None,
        on_reset: Callable[[], None] | None = None,
        initial_observation: tuple[Any, dict] | None = None,
    ):
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("CollectionRunner currently requires env.num_envs == 1.")
        if config.max_steps_per_episode < 1 or config.success_hold_steps < 1:
            raise ValueError("CollectionRunner limits must be positive.")
        self.env = env
        self.policy = policy
        self.config = config
        self.success_fn = success_fn
        # Keep IsaacLab imports lazy so CLI discovery and unit tests do not
        # require Kit to be initialized before the collection environment.
        from .recording import RecorderSession

        self.recorder = RecorderSession(env)
        self.should_continue = should_continue or (lambda: True)
        self.discard_requested = discard_requested
        self.on_episode_complete = on_episode_complete
        self.on_step = on_step
        self.on_idle = on_idle
        self.on_reset = on_reset
        self._initial_observation = initial_observation

    def _prepare_episode(self, observation: Any, info: dict) -> tuple[Any, dict]:
        """Reset policy state for an already-reset environment.

        Recorder reset must happen before ``env.reset()``.  IsaacLab records
        ``initial_state`` in its post-reset callback, so clearing the recorder
        here would erase the state needed by Mimic annotation.
        """

        self.policy.reset()
        task_description = getattr(self.env.unwrapped, "task_description", None)
        if task_description is None:
            task_description = getattr(self.env.unwrapped.cfg, "task_description", None)
        if task_description is not None and hasattr(self.policy, "set_task_description"):
            self.policy.set_task_description(task_description)
        if self.on_reset is not None:
            self.on_reset()
        return observation, info

    def _reset_episode(self) -> tuple[Any, dict]:
        """Discard the current recorder buffer and reset policy/environment."""

        for attempt in range(4):
            # Clear the previous logical episode before IsaacLab's env.reset()
            # records the new episode's initial_state.
            self.recorder.reset()
            observation, info = self.env.reset()
            try:
                return self._prepare_episode(observation, info)
            except RuntimeError as exc:
                self.recorder.commit([0], False)
                print(
                    f"Dataset episode discarded (reset_failed={attempt + 1}, reason={exc})",
                    flush=True,
                )
        raise RuntimeError("Collection could not reset a valid episode after 16 attempts.")

    @staticmethod
    def _policy_flag(policy, name: str) -> bool:
        value = getattr(policy, name, False)
        return bool(value() if callable(value) else value)

    def run(self, num_episodes: int | None = None) -> int:
        """Collect successful episodes until the target or runner stops."""

        if num_episodes is not None and num_episodes < 1:
            raise ValueError("num_episodes must be positive or None.")

        if self._initial_observation is None:
            obs, _ = self._reset_episode()
        else:
            initial_observation = self._initial_observation
            self._initial_observation = None
            # The caller already reset the environment in order to calibrate a
            # policy before constructing this runner.  Capture that current
            # state without resetting physics a second time.
            self.recorder.reset(capture_initial_state=True)
            obs, _ = self._prepare_episode(*initial_observation)
        completed = 0
        steps = 0
        held_success_steps = 0
        success_latched = False

        while self.should_continue() and (num_episodes is None or completed < num_episodes):
            if self.discard_requested is not None and self.discard_requested():
                self.recorder.commit([0], False)
                obs, _ = self._reset_episode()
                steps = 0
                held_success_steps = 0
                success_latched = False
                continue

            # Keep both policy and environment execution in the normal
            # autograd state.  cuRobo-backed policies need autograd for IK,
            # and project-owned environment state (for example conveyor
            # attachment flags) is reset with in-place updates.  Running
            # env.step() in inference_mode would turn those state tensors into
            # inference tensors that cannot be reset on the next episode.
            action = self.policy.get_action(self.env, obs)
            if action is None:
                if self.on_idle is not None:
                    self.on_idle()
                continue

            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.on_step is not None:
                self.on_step(obs)
            steps += 1
            success = self.success_fn(self.env, obs, reward, terminated, truncated, info)
            success = bool(success.item()) if isinstance(success, torch.Tensor) else bool(success)
            held_success_steps = held_success_steps + 1 if success else 0
            if held_success_steps >= self.config.success_hold_steps:
                success_latched = True

            policy_failed = self._policy_flag(self.policy, "failed")
            policy_done = self._policy_flag(self.policy, "done")
            has_commit_boundary = hasattr(self.policy, "ready_to_commit")
            policy_ready = (
                self._policy_flag(self.policy, "ready_to_commit")
                if has_commit_boundary
                else True
            )
            # A scripted policy may enter its semantic terminal state before
            # its return-to-home motion has physically reached the target.
            # Keep recording until that explicit boundary is reached; failed
            # policies still terminate immediately and are discarded.
            episode_done = policy_failed or (
                policy_done
                and policy_ready
            )
            if not hasattr(self.policy, "done"):
                episode_done = episode_done or success_latched
            episode_done = episode_done or _has_any(terminated) or _has_any(truncated)
            episode_done = episode_done or steps >= self.config.max_steps_per_episode
            if not episode_done:
                continue

            succeeded = success_latched and not policy_failed and policy_ready
            self.recorder.commit([0], succeeded)
            if succeeded:
                completed += 1
            progress = f"{completed}/{num_episodes}" if num_episodes is not None else str(completed)
            print(
                "Dataset episode exported" if succeeded else "Dataset episode discarded",
                f"(progress={progress}, steps={steps}, policy_failed={policy_failed}, "
                f"ready_to_commit={policy_ready})",
                flush=True,
            )
            if not self.should_continue() or (num_episodes is not None and completed >= num_episodes):
                break
            if self.on_episode_complete is not None and self.on_episode_complete(succeeded):
                # Keep the current physical scene for multi-object tasks while
                # starting a fresh recorder episode and policy cursor.
                self.recorder.reset(capture_initial_state=True)
                if self.on_reset is not None:
                    self.on_reset()
                steps = 0
                held_success_steps = 0
                success_latched = False
                continue

            obs, _ = self._reset_episode()
            steps = 0
            held_success_steps = 0
            success_latched = False

        return completed
