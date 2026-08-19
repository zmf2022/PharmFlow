"""Shared contracts for the biomedical collection task.

This module is the collection-side source of truth for the DROID gripper
convention and medicine placement predicate.  Training adapters may import
these functions later, but collection never imports the RLinf package.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat


DROID_GRIPPER_OPEN_POSITION = 0.0
DROID_GRIPPER_CLOSED_POSITION = float(torch.pi / 4)
DROID_GRIPPER_MAX_WIDTH = 0.085
DROID_BASE_TO_CLOSED_GRASP_CENTER = torch.tensor(
    (0.13108637, 0.00014421, 0.0), dtype=torch.float32
)


def droid_grasp_position(env) -> torch.Tensor:
    """Return the midpoint of the DROID inner fingers in env coordinates."""

    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)
    for left_name, right_name in (
        ("left_inner_finger", "right_inner_finger"),
        ("left_outer_finger", "right_outer_finger"),
    ):
        if left_name in body_names and right_name in body_names:
            left = robot.data.body_pos_w.torch[:, body_names.index(left_name)]
            right = robot.data.body_pos_w.torch[:, body_names.index(right_name)]
            return (left + right) / 2 - env.scene.env_origins
    for candidate in ("base_link", "panda_hand", "panda_link7", "panda_link6"):
        if candidate in body_names:
            return (
                robot.data.body_pos_w.torch[:, body_names.index(candidate)]
                - env.scene.env_origins
            )
    return robot.data.body_pos_w.torch[:, -1] - env.scene.env_origins


def _target_positions(env, target_cfgs: tuple[SceneEntityCfg, ...]) -> torch.Tensor:
    return torch.stack(
        [
            env.scene[target_cfg.name].data.root_pos_w.torch - env.scene.env_origins
            for target_cfg in target_cfgs
        ],
        dim=1,
    )


def _region_mask(
    positions: torch.Tensor,
    center: torch.Tensor,
    size: torch.Tensor,
    conveyor_axis: int,
    bounds: tuple[float, float],
    margin: float,
) -> torch.Tensor:
    lateral_axis = 1 if conveyor_axis == 0 else 0
    along = (positions[..., conveyor_axis] >= bounds[0] + margin) & (
        positions[..., conveyor_axis] <= bounds[1] - margin
    )
    lateral_half_extent = torch.clamp(size[lateral_axis] / 2 - margin, min=0.0)
    lateral = torch.abs(positions[..., lateral_axis] - center[lateral_axis]) <= lateral_half_extent
    return along & lateral


def _support_heights(
    env,
    target_cfgs: tuple[SceneEntityCfg, ...],
    target_support_extents: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    rotations = matrix_from_quat(
        torch.stack(
            [
                env.scene[target_cfg.name].data.root_pose_w.torch[:, 3:7]
                for target_cfg in target_cfgs
            ],
            dim=1,
        )
    )
    extents = torch.as_tensor(
        target_support_extents, device=env.device, dtype=rotations.dtype
    )
    return torch.sum(torch.abs(rotations[..., 2, :]) * extents.unsqueeze(0), dim=-1)


def _gripper_is_open(env) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_index = robot.data.joint_names.index("finger_joint")
    return torch.abs(
        robot.data.joint_pos.torch[:, joint_index] - DROID_GRIPPER_OPEN_POSITION
    ) <= 0.04


def medicine_on_conveyor(
    env,
    target_cfgs: tuple[SceneEntityCfg, ...],
    target_support_extents: tuple[tuple[float, float, float], ...],
    target_upright_axes: tuple[tuple[float, float, float], ...],
    conveyor_center: tuple[float, float, float],
    conveyor_size: tuple[float, float, float],
    conveyor_axis: int = 0,
    conveyor_bounds: tuple[float, float] = (0.0, 1.0),
    region_margin: float = 0.05,
    surface_height_tolerance: float = 0.04,
    require_release: bool = True,
    require_upright: bool = True,
    min_upright_cosine: float = 0.95,
) -> torch.Tensor:
    """Return whether at least one released target is correctly placed."""

    positions = _target_positions(env, target_cfgs)
    center = positions.new_tensor(conveyor_center)
    size = positions.new_tensor(conveyor_size)
    in_region = _region_mask(
        positions, center, size, conveyor_axis, conveyor_bounds, region_margin
    )
    support_heights = _support_heights(env, target_cfgs, target_support_extents)
    expected_z = center[2] + size[2] / 2 + support_heights
    on_surface = torch.abs(positions[..., 2] - expected_z) <= surface_height_tolerance

    rotations = matrix_from_quat(
        torch.stack(
            [
                env.scene[target_cfg.name].data.root_pose_w.torch[:, 3:7]
                for target_cfg in target_cfgs
            ],
            dim=1,
        )
    )
    axes = torch.as_tensor(
        target_upright_axes, device=env.device, dtype=positions.dtype
    )
    axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-6)
    world_axes = torch.einsum("bnij,nj->bni", rotations, axes)
    upright = world_axes[..., 2] >= min_upright_cosine
    placed = (in_region & on_surface & (upright if require_upright else True)).any(dim=1)
    return placed & (_gripper_is_open(env) if require_release else True)


def single_target_params(success_term: Any, index: int) -> dict[str, Any]:
    """Narrow a multi-target success term to one expert target."""

    params = dict(success_term.params)
    for key in ("target_cfgs", "target_support_extents", "target_upright_axes"):
        params[key] = (params[key][index],)
    return params


__all__ = [
    "DROID_BASE_TO_CLOSED_GRASP_CENTER",
    "DROID_GRIPPER_CLOSED_POSITION",
    "DROID_GRIPPER_MAX_WIDTH",
    "DROID_GRIPPER_OPEN_POSITION",
    "droid_grasp_position",
    "medicine_on_conveyor",
    "single_target_params",
]
