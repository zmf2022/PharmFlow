"""Reusable kinematic scene motion for IsaacLab environments.

The scene YAML describes motion; this module owns the runtime state needed to
apply that motion.  It is deliberately independent of RLinf, Arena, and any
particular task so the same conveyor behaviour can be used by RL, evaluation,
teleoperation, and IsaacLab Mimic environments.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.utils.math import matrix_from_quat, quat_mul


def _safe_name(name: str, index: int) -> str:
    value = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
    value = value.strip("_")
    return value or f"component_{index:02d}"


def conveyor_region_mask(
    positions: torch.Tensor,
    center: torch.Tensor,
    size: torch.Tensor,
    axis: int,
    bounds: tuple[float, float],
    margin: float = 0.0,
) -> torch.Tensor:
    """Return the shared XY mask for objects inside a conveyor region."""

    lower = float(bounds[0]) + margin
    upper = float(bounds[1]) - margin
    lateral_axis = 1 if axis == 0 else 0
    axis_inside = (positions[:, axis] >= lower) & (positions[:, axis] <= upper)
    lateral_inside = torch.abs(positions[:, lateral_axis] - center[lateral_axis]) <= max(
        float(size[lateral_axis]) / 2 - margin, 0.0
    )
    return axis_inside & lateral_inside


def conveyor_surface_mask(
    positions: torch.Tensor,
    center: torch.Tensor,
    size: torch.Tensor,
    support_heights: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Return whether object centers are resting near the conveyor surface."""

    surface_top = center[2] + size[2] / 2
    expected_center_height = surface_top + support_heights
    return torch.abs(positions[..., 2] - expected_center_height) <= float(tolerance)


def conveyor_support_height(
    root_quaternions: torch.Tensor,
    component: dict[str, Any],
) -> torch.Tensor:
    """Return the current world-Z support extent of a moving object."""

    support_extents = component.get("support_extents")
    if support_extents is None:
        support_height = float(
            component.get("support_height", component.get("size", (0.1, 0.1, 0.1))[2])
        )
        if support_height <= 0:
            raise ValueError("Conveyor support height must be positive")
        return root_quaternions.new_full(root_quaternions.shape[:-1], support_height)

    extents = torch.as_tensor(
        support_extents,
        device=root_quaternions.device,
        dtype=root_quaternions.dtype,
    )
    if extents.shape != (3,) or torch.any(extents <= 0):
        raise ValueError("Conveyor support extents must contain three positive values")
    rotation = matrix_from_quat(root_quaternions)
    return torch.sum(torch.abs(rotation[..., 2, :]) * extents, dim=-1)


class SceneMotionController:
    """Advance YAML-declared conveyors and rotating scene components.

    The controller owns only transient motion state.  Asset creation and task
    success remain in the environment/task layers, matching Arena's scene/task
    split.  ``env`` is intentionally duck-typed to keep this usable with both
    ``ManagerBasedRLEnv`` and ``ManagerBasedRLMimicEnv``.
    """

    def __init__(self, env: Any, scene_spec: dict[str, Any]):
        self.env = env
        self.scene_spec = scene_spec
        self.motion_components = [
            (index, component)
            for index, component in enumerate(scene_spec.get("components", []), start=1)
            if component.get("motion")
        ]
        self.motion_time = torch.zeros(env.num_envs, device=env.device)
        self.conveyor_attached: dict[str, torch.Tensor] = {}
        self.motion_initial_quats: dict[str, torch.Tensor] = {}

        for index, component in self.motion_components:
            motion = component.get("motion", {})
            motion = {"type": motion} if isinstance(motion, str) else motion
            if motion.get("type") == "conveyor":
                name = _safe_name(str(component.get("name", "")), index)
                self.conveyor_attached[name] = torch.zeros(
                    env.num_envs, dtype=torch.bool, device=env.device
                )
            if motion.get("type") != "rotate":
                continue
            name = _safe_name(str(component.get("name", "")), index)
            object_cfg = env.scene.rigid_objects.get(name)
            if object_cfg is not None:
                self.motion_initial_quats[name] = object_cfg.data.root_pose_w.torch[:, 3:7].clone()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.motion_time[env_ids] = 0.0
        for attached in self.conveyor_attached.values():
            attached[env_ids] = False

    def advance(self) -> None:
        if not self.motion_components:
            return
        self.motion_time += self.env.step_dt
        for index, component in self.motion_components:
            motion = component.get("motion", {})
            motion = {"type": motion} if isinstance(motion, str) else motion
            motion_type = motion.get("type")
            if motion_type == "conveyor":
                self._advance_conveyor(component, motion)
                continue

            name = _safe_name(str(component.get("name", "")), index)
            object_cfg = self.env.scene.rigid_objects.get(name)
            if object_cfg is None or motion_type != "rotate":
                continue
            angle = self.motion_time * float(motion.get("speed", 1.0))
            pose = object_cfg.data.root_pose_w.torch.clone()
            axis_name = str(component.get("axis", motion.get("axis", "Y"))).upper()
            axis = {"X": 0, "Y": 1, "Z": 2}.get(axis_name)
            if axis is None:
                raise ValueError(f"Unsupported rotation axis: {axis_name}")
            spin = torch.zeros((self.env.num_envs, 4), device=self.env.device, dtype=pose.dtype)
            spin[:, axis] = torch.sin(angle * 0.5)
            spin[:, 3] = torch.cos(angle * 0.5)
            pose[:, 3:7] = quat_mul(self.motion_initial_quats[name], spin)
            object_cfg.write_root_pose_to_sim(pose)

    def _advance_conveyor(self, conveyor: dict[str, Any], motion: dict[str, Any]) -> None:
        center = torch.as_tensor(
            conveyor.get("position", (0.0, 0.0, 0.0)), device=self.env.device, dtype=torch.float32
        )
        size = torch.as_tensor(
            conveyor.get("size", (1.0, 1.0, 0.1)), device=self.env.device, dtype=torch.float32
        )
        axis_name = str(motion.get("axis", "X")).upper()
        axis = {"X": 0, "Y": 1, "Z": 2}.get(axis_name)
        if axis not in (0, 1):
            raise ValueError("Conveyor transport supports horizontal X/Y axes only")
        bounds = motion.get("bounds")
        if bounds is None or len(bounds) != 2:
            raise ValueError("A conveyor motion requires two bounds")

        lower, upper = float(bounds[0]), float(bounds[1])
        speed = float(motion.get("speed", 0.1))
        height_tolerance = float(motion.get("surface_height_tolerance", 0.08))
        surface_clearance = float(motion.get("surface_clearance", 0.005))
        if surface_clearance < 0:
            raise ValueError("Conveyor surface clearance must be non-negative")
        surface_top = center[2] + size[2] / 2

        for index, component in enumerate(self.scene_spec.get("components", []), start=1):
            if component is conveyor or component.get("kind") not in {"rigid", "rigid_usd", "dynamic"}:
                continue
            name = _safe_name(str(component.get("name", "")), index)
            box = self.env.scene.rigid_objects.get(name)
            if box is None:
                continue
            position = box.data.root_pos_w.torch - self.env.scene.env_origins
            in_region = conveyor_region_mask(position, center, size, axis, (lower, upper))
            support_height = conveyor_support_height(box.data.root_pose_w.torch[:, 3:7], component)
            on_surface = conveyor_surface_mask(
                position, center, size, support_height, height_tolerance
            )
            attached = self.conveyor_attached.setdefault(
                name, torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            )
            newly_attached = in_region & on_surface & ~attached
            attached[in_region & on_surface] = True
            transporting = in_region & attached
            if not torch.any(transporting):
                continue
            if torch.any(newly_attached):
                pose = box.data.root_pose_w.torch.clone()
                pose[newly_attached, 2] = (
                    surface_top
                    + support_height[newly_attached] / 2
                    + surface_clearance
                    + self.env.scene.env_origins[newly_attached, 2]
                )
                box.write_root_pose_to_sim(pose)

            velocity = box.data.root_vel_w.torch.clone()
            velocity[transporting, :3] = 0.0
            transport_speed = torch.where(
                position[:, axis] < upper,
                velocity.new_full((self.env.num_envs,), speed),
                velocity.new_zeros((self.env.num_envs,)),
            )
            velocity[transporting, axis] = transport_speed[transporting]
            velocity[transporting, 3:] = 0.0
            box.write_root_velocity_to_sim(velocity)
