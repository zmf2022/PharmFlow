"""Reusable pick-and-place atomic skills.

This module contains execution behaviors only.  Target selection, pose
generation, planner setup, and task-level ordering remain in the policy that
provides the runtime context.  The skills themselves do not know which task,
scene, or object type is being collected.
"""

from __future__ import annotations

import torch

from pharm_flow.data_collection.utils.skills import (
    AtomicSkill,
    AtomicSkillContext,
    AtomicSkillOutput,
)


class PickPlaceSkills:
    """Register reusable pick-and-place stages for a compatible runtime."""

    _MOTION_SKILLS = (
        "approach",
        "grasp",
        "lift",
        "transport",
        "reorient",
        "place",
        "retreat",
        "home",
    )

    def registry(self) -> dict[str, AtomicSkill]:
        return {
            name: AtomicSkill(name, self._run_motion)
            for name in self._MOTION_SKILLS
        } | {
            "hold_grasp": AtomicSkill("hold_grasp", self._run_hold_grasp),
            "hold_release": AtomicSkill("hold_release", self._run_hold_release),
            "release_lift": AtomicSkill("release_lift", self._run_release_lift),
        }

    def _output(
        self, context: AtomicSkillContext, action: torch.Tensor
    ) -> AtomicSkillOutput:
        runtime = context.runtime
        next_skill = runtime._stage if runtime._stage != context.stage else None
        return AtomicSkillOutput(
            action=action,
            next_skill=next_skill,
            done=runtime.done,
            failed=runtime.failed,
        )

    def _run_hold_grasp(self, context: AtomicSkillContext) -> AtomicSkillOutput:
        host = context.runtime
        host._stage_steps += 1
        robot = host.env.scene["robot"]
        gripper_ids, _ = robot.find_joints(["finger_joint"])
        finger_position = float(robot.data.joint_pos.torch[0, gripper_ids][0].item())
        if host._grasp_verified():
            if (
                host._last_gripper_position is not None
                and abs(finger_position - host._last_gripper_position)
                <= host.config.grasp_position_delta_tolerance
            ):
                host._grasp_stable_steps += 1
            else:
                host._grasp_stable_steps = 0
            host._last_gripper_position = finger_position
        else:
            host._grasp_stable_steps = 0
            host._last_gripper_position = finger_position

        if host._stage_steps >= host.config.hold_steps and host._grasp_stable_steps >= 3:
            if host._grasp_verified():
                host._pre_lift_target_z = float(
                    host.env.scene[host._target_name].data.root_pos_w.torch[0, 2].item()
                )
                host._advance_stage()
                host._plan_current_stage(expected_attached_object=host._target_name)
                action = host._action_for_target(
                    host._next_planned_arm_target(), gripper=1.0
                )
                return self._output(context, action)
            host._mark_failed("grasp_not_obstructed: finger reached an empty-space stop")
        if host._stage_steps >= host.config.max_grasp_settle_steps:
            host._mark_failed("grasp_settle_timeout")

        return self._output(
            context, host._action_for_target(host._held_arm_target(), gripper=1.0)
        )

    def _run_hold_release(self, context: AtomicSkillContext) -> AtomicSkillOutput:
        host = context.runtime
        host._stage_steps += 1
        if host._stage_steps >= host.config.hold_steps:
            host._advance_stage()
            host._plan_release_lift()
            action = host._action_for_target(
                host._next_planned_arm_target(), gripper=0.0
            )
        else:
            action = host._action_for_target(host._held_arm_target(), gripper=0.0)
        return self._output(context, action)

    def _run_release_lift(self, context: AtomicSkillContext) -> AtomicSkillOutput:
        host = context.runtime
        if host._plan_positions is not None and host._plan_index < len(host._plan_positions):
            action = host._action_for_target(
                host._next_planned_arm_target(), gripper=0.0
            )
            return self._output(context, action)
        if not host._arm_reached_plan_target():
            action = host._action_for_target(host._held_arm_target(), gripper=0.0)
            return self._output(context, action)
        host._advance_stage()
        host._plan_current_stage(expected_attached_object=None)
        action = host._action_for_target(host._next_planned_arm_target(), gripper=0.0)
        return self._output(context, action)

    def _run_motion(self, context: AtomicSkillContext) -> AtomicSkillOutput:
        host = context.runtime
        host._stage_steps += 1
        planned_steps = len(host._plan_positions) if host._plan_positions is not None else 0
        stage_step_limit = max(
            host.config.max_motion_stage_steps,
            planned_steps + host.config.hold_steps + 8,
        )
        if host._stage_steps >= stage_step_limit:
            host._mark_failed(f"motion_stage_timeout: {host._stage}")
            return self._output(
                context, host._action_for_target(host._held_arm_target(), gripper=0.0)
            )

        if host._plan_positions is not None and host._plan_index < len(host._plan_positions):
            action = host._action_for_target(
                host._next_planned_arm_target(), host._stage_gripper()
            )
            return self._output(context, action)

        if host._stage == "place":
            if host._target_on_conveyor(require_release=False):
                host._advance_stage()
                host._stage_steps = 0
                action = host._action_for_target(host._held_arm_target(), gripper=0.0)
            else:
                action = host._action_for_target(host._held_arm_target(), gripper=1.0)
            return self._output(context, action)

        if not host._arm_reached_plan_target():
            if host._stage == "home":
                host._home_stable_steps = 0
            action = host._action_for_target(
                host._held_arm_target(), host._stage_gripper()
            )
            return self._output(context, action)

        if host._stage == "approach":
            host._advance_stage()
            host._plan_current_stage(expected_attached_object=None)
            action = host._action_for_target(
                host._next_planned_arm_target(), gripper=0.0
            )
            return self._output(context, action)
        if host._stage == "grasp":
            host._advance_stage()
            host._stage_steps = 0
            action = host._action_for_target(host._held_arm_target(), gripper=1.0)
            return self._output(context, action)
        if host._stage == "lift":
            current_z = float(
                host.env.scene[host._target_name].data.root_pos_w.torch[0, 2].item()
            )
            if host._pre_lift_target_z is not None:
                lift_delta = current_z - host._pre_lift_target_z
                host._lift_verified_steps = (
                    host._lift_verified_steps + 1
                    if lift_delta >= host.config.lift_verification_height
                    else 0
                )
            if (
                host._pre_lift_target_z is None
                or host._lift_verified_steps < host.config.lift_verification_steps
            ):
                if host._stage_steps >= host.config.max_motion_stage_steps:
                    host._mark_failed(
                        f"object_not_lifted: bottle did not follow the gripper "
                        f"(delta={current_z - (host._pre_lift_target_z or 0):.4f}m, "
                        f"required={host.config.lift_verification_height}m, "
                        f"verified_steps={host._lift_verified_steps})"
                    )
                    return self._output(
                        context,
                        host._action_for_target(host._held_arm_target(), gripper=0.0),
                    )
                return self._output(
                    context,
                    host._action_for_target(
                        host._next_planned_arm_target(), gripper=1.0
                    ),
                )

            host._planner.logger.info(
                f"Lift verified for {host._target_name}: "
                f"delta={current_z - host._pre_lift_target_z:.4f}m, "
                f"steps={host._lift_verified_steps}"
            )
            host._advance_stage()
            host._plan_current_stage(expected_attached_object=host._target_name)
            action = host._action_for_target(
                host._next_planned_arm_target(), gripper=1.0
            )
            return self._output(context, action)
        if host._stage in {"transport", "reorient"}:
            host._advance_stage()
            host._plan_current_stage(expected_attached_object=host._target_name)
            action = host._action_for_target(
                host._next_planned_arm_target(), gripper=1.0
            )
            return self._output(context, action)
        if host._stage == "retreat":
            host._advance_stage(terminal="home")
            host._plan_current_stage(expected_attached_object=None)
            action = host._action_for_target(
                host._next_planned_arm_target(), gripper=0.0
            )
            return self._output(context, action)
        if host._stage == "home":
            host._home_stable_steps += 1
            if host._home_stable_steps < host.config.hold_steps:
                return self._output(
                    context,
                    host._action_for_target(host._held_arm_target(), gripper=0.0),
                )
            host._home_reached = True
            host._planner.logger.info(
                f"Home pose settled for {host._target_name}; "
                f"stable_steps={host._home_stable_steps}, "
                "episode ready for export"
            )
            host._advance_stage()
            return self._output(
                context, host._action_for_target(host._held_arm_target(), gripper=0.0)
            )

        raise RuntimeError(f"Unknown biomedical collection stage: {host._stage!r}")


__all__ = ["PickPlaceSkills"]
