"""Project-owned cuRobo adapter for the current Arena/IsaacLab APIs.

The latest IsaacLab-Arena keeps the robot-family cuRobo registry and the
IsaacLab Mimic planner, but no longer ships the old ``planner_utils`` module.
The two small helpers used by the biomedical expert live here so the project
does not modify or depend on a removed Arena-private import path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
import yaml

import isaaclab.utils.math as math_utils
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.utils.frame_utils import world_pose_to_robot_frame
from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab.envs import ManagerBasedEnv


def make_planner_cfg(
    embodiment: EmbodimentBase,
    *,
    debug_planner: bool = False,
) -> CuroboPlannerCfg:
    """Build a Mimic ``CuroboPlannerCfg`` from Arena's robot registry."""

    curobo_cfg = get_embodiment_curobo_cfg(embodiment)
    robot_yaml = load_patched_robot_yaml(curobo_cfg)

    runtime_dir = Path(tempfile.mkdtemp(prefix="curobo_robot_cfg_"))
    robot_cfg_file = runtime_dir / "curobo_runtime.yml"
    with robot_cfg_file.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(robot_yaml, stream, sort_keys=False)

    locked_joints = dict(robot_yaml["robot_cfg"]["kinematics"]["lock_joints"])
    open_positions = {**locked_joints, **curobo_cfg.gripper_open_joint_pos}
    closed_positions = {**locked_joints, **curobo_cfg.gripper_closed_joint_pos}

    return CuroboPlannerCfg(
        robot_config_file=str(robot_cfg_file),
        robot_name=curobo_cfg.robot_name,
        ee_link_name=curobo_cfg.ee_link_name,
        gripper_joint_names=curobo_cfg.gripper_joint_names,
        gripper_open_positions=open_positions,
        gripper_closed_positions=closed_positions,
        hand_link_names=curobo_cfg.hand_link_names,
        grasp_gripper_open_val=curobo_cfg.grasp_gripper_open_val,
        approach_distance=curobo_cfg.approach_distance,
        retreat_distance=curobo_cfg.retreat_distance,
        time_dilation_factor=curobo_cfg.time_dilation_factor,
        collision_activation_distance=curobo_cfg.collision_activation_distance,
        motion_step_size=None,
        trajopt_tsteps=curobo_cfg.trajopt_tsteps,
        visualize_plan=False,
        visualize_spheres=False,
        debug_planner=debug_planner,
        world_ignore_substrings=curobo_cfg.world_ignore_substrings,
    )


def sync_object_poses_in_robot_base_frame(planner: Any) -> None:
    """Synchronize Mimic collision objects into cuRobo's robot-base frame.

    The current public Mimic planner does not expose this project-specific
    robot-base synchronization operation.  The adapter therefore uses the
    planner's stable object-world update hooks in one place instead of
    scattering Arena-private imports through the expert.
    """

    object_mappings = planner._get_object_mappings()
    world_model = planner.motion_gen.world_coll_checker.world_model
    rigid_objects = planner.env.scene.rigid_objects

    robot_pos_w = wp.to_torch(planner.robot.data.root_pos_w)[planner.env_id, :3]
    robot_quat_w_xyzw = wp.to_torch(planner.robot.data.root_quat_w)[planner.env_id, :4]
    static_objects = getattr(planner.config, "static_objects", [])
    ignored_paths = tuple(
        value
        for value in getattr(planner.config, "world_ignore_substrings", ())
        if value
    )
    updated_count = 0

    for object_name, object_path in object_mappings.items():
        if object_name not in rigid_objects:
            continue
        if any(value in object_name.lower() for value in static_objects):
            continue
        # Objects filtered from cuRobo's collision world are intentionally not
        # present in its OBB model.  Updating their poses here only produces
        # misleading "obstacle not found" warnings and cannot affect planning.
        if any(value in object_path for value in ignored_paths):
            continue

        object_asset = rigid_objects[object_name]
        object_pos_w = wp.to_torch(object_asset.data.root_pos_w)[planner.env_id, :3]
        object_quat_w_xyzw = wp.to_torch(object_asset.data.root_quat_w)[planner.env_id, :4]
        object_pos_r, object_quat_r_xyzw = world_pose_to_robot_frame(
            object_pos_w,
            object_quat_w_xyzw,
            robot_pos_w,
            robot_quat_w_xyzw,
        )
        object_pos_r = planner._to_curobo_device(object_pos_r)
        object_quat_r_xyzw = planner._to_curobo_device(object_quat_r_xyzw)
        object_quat_r_wxyz = math_utils.convert_quat(object_quat_r_xyzw, to="wxyz")
        pose_list = torch.cat((object_pos_r, object_quat_r_wxyz)).tolist()

        if planner._update_object_in_world_model(
            world_model,
            object_name,
            object_path,
            pose_list,
        ):
            curobo_pose = planner._make_pose(
                position=object_pos_r,
                quaternion=object_quat_r_wxyz,
                quat_is_xyzw=False,
            )
            planner.motion_gen.world_coll_checker.update_obstacle_pose(
                object_path,
                curobo_pose,
                update_cpu_reference=True,
            )
            updated_count += 1

    planner.logger.debug("SYNC (robot-base frame): Updated %d object poses", updated_count)
    # ``pose_list = ... .tolist()`` above already synchronizes values needed by
    # the CPU-side world-model update. An unconditional device-wide barrier on
    # every planning call makes stage transitions visibly stall; retain it only
    # for explicit planner debugging.
    if getattr(planner.config, "debug_planner", False) and torch.cuda.is_available():
        torch.cuda.synchronize()


__all__ = ["make_planner_cfg", "sync_object_poses_in_robot_base_frame"]
