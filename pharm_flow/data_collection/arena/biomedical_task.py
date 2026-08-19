"""Arena task semantics for the biomedical medicine-to-conveyor task.

This module contains task state, success predicates, metrics, and Mimic
subtask configuration only.  Scene construction is in
``biomedical_environment.py`` and robot pose/action conversion is in
``droid_mimic.py``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields
from typing import Any

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import agent_ready, register_task
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition
from isaaclab_arena.utils.configclass import make_configclass

from .biomedical_contract import (
    DROID_GRIPPER_CLOSED_POSITION,
    medicine_on_conveyor,
)


def _target_position(env, target_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[target_cfg.name].data.root_pos_w.torch - env.scene.env_origins


def _gripper_is_closed(env, threshold: float = 0.04) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_index = robot.data.joint_names.index("finger_joint")
    return robot.data.joint_pos.torch[:, joint_index] > threshold


def medicine_target_position_in_robot_frame(
    env,
    target_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the primary medicine target position in the robot root frame."""

    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    target_pos, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        target.data.root_pos_w.torch,
    )
    return target_pos


def medicine_target_quaternion_in_robot_frame(
    env,
    target_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the primary medicine target orientation in the robot root frame."""

    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    _, target_quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        target.data.root_pos_w.torch,
        target.data.root_quat_w.torch,
    )
    return target_quat


def randomize_medicine_target_layout(
    env,
    env_ids: torch.Tensor,
    target_cfgs: tuple[SceneEntityCfg, ...],
    min_count: int,
    max_count: int,
    pose_range: dict[str, tuple[float, float]],
    target_support_extents: tuple[tuple[float, float, float], ...],
    support_surface_z: float,
    surface_clearance: float,
    inactive_position_offset: tuple[float, float, float],
    required_target_name: str | None = None,
) -> None:
    """Reset the pre-created medicine pool using the task randomization contract.

    Arena/IsaacLab needs every candidate rigid object to exist when the scene is
    built.  The reset event then samples the active subset and writes each
    object's pose from its configured default pose.  This is the same lifecycle
    as the RL environment: active bottles are placed on the carton support
    plane, while inactive candidates are moved out of the workcell.
    """

    if not target_cfgs:
        return
    if min_count < 1 or max_count < min_count or max_count > len(target_cfgs):
        raise ValueError(
            "Biomedical target count range must satisfy "
            f"1 <= min_count <= max_count <= {len(target_cfgs)}"
        )
    if len(target_support_extents) != len(target_cfgs):
        raise ValueError("Target support extents must match the medicine object pool")
    required_target_index = None
    if required_target_name is not None:
        target_names = tuple(target_cfg.name for target_cfg in target_cfgs)
        if required_target_name not in target_names:
            raise ValueError(
                f"Required biomedical target {required_target_name!r} is not in {target_names!r}"
            )
        required_target_index = target_names.index(required_target_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    counts = torch.randint(min_count, max_count + 1, (len(env_ids),), device=env.device)
    offset = torch.as_tensor(
        inactive_position_offset, device=env.device, dtype=torch.float32
    )
    ranges = torch.tensor(
        [pose_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device=env.device,
        dtype=torch.float32,
    )
    yaw_range = pose_range.get("yaw", (0.0, 0.0))

    for row, env_id in enumerate(env_ids):
        active_indices = torch.randperm(len(target_cfgs), device=env.device)[: counts[row]]
        if required_target_index is not None and not bool(
            (active_indices == required_target_index).any().item()
        ):
            active_indices[0] = required_target_index
        # Candidate names are semantic identities, while their authored poses
        # are physical slots in the carton. Shuffle the slots independently so
        # the required target is not always tied to medicine_bottle_00's corner.
        slot_indices = torch.randperm(len(target_cfgs), device=env.device)[: active_indices.numel()]
        slot_by_target = torch.empty(len(target_cfgs), dtype=torch.long, device=env.device)
        slot_by_target[active_indices] = slot_indices
        active_mask = torch.zeros(len(target_cfgs), dtype=torch.bool, device=env.device)
        active_mask[active_indices] = True
        for target_index, target_cfg in enumerate(target_cfgs):
            asset = env.scene[target_cfg.name]
            default_pose = asset.data.default_root_pose.torch[env_id].clone()
            default_velocity = asset.data.default_root_vel.torch[env_id].clone()
            if bool(active_mask[target_index].item()):
                slot_index = int(slot_by_target[target_index].item())
                slot_asset = env.scene[target_cfgs[slot_index].name]
                slot_pose = slot_asset.data.default_root_pose.torch[env_id].clone()
                random_position = torch.rand(3, device=asset.device, dtype=default_pose.dtype)
                random_position = ranges.to(asset.device, dtype=default_pose.dtype)[:, 0] + (
                    ranges.to(asset.device, dtype=default_pose.dtype)[:, 1]
                    - ranges.to(asset.device, dtype=default_pose.dtype)[:, 0]
                ) * random_position
                yaw = torch.as_tensor(
                    yaw_range[0], device=asset.device, dtype=default_pose.dtype
                ) + (
                    torch.as_tensor(
                        yaw_range[1] - yaw_range[0],
                        device=asset.device,
                        dtype=default_pose.dtype,
                    )
                    * torch.rand((), device=asset.device, dtype=default_pose.dtype)
                )
                yaw_delta = quat_from_euler_xyz(
                    torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
                )
                orientation = quat_mul(default_pose[3:7], yaw_delta)
                extents = torch.as_tensor(
                    target_support_extents[target_index],
                    device=asset.device,
                    dtype=default_pose.dtype,
                )
                rotation = matrix_from_quat(orientation.unsqueeze(0))[0]
                support_height = torch.sum(torch.abs(rotation[2, :]) * extents)
                position = slot_pose[:3] + env.scene.env_origins[env_id]
                position += random_position
                position[2] = support_surface_z + support_height + surface_clearance
            else:
                orientation = default_pose[3:7]
                position = default_pose[:3] + env.scene.env_origins[env_id] + offset

            asset.write_root_pose_to_sim_index(
                root_pose=torch.cat((position, orientation)).unsqueeze(0),
                env_ids=env_id.unsqueeze(0),
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=default_velocity.unsqueeze(0),
                env_ids=env_id.unsqueeze(0),
            )


def _inside_conveyor_region(
    position: torch.Tensor,
    conveyor_center: tuple[float, float, float],
    conveyor_size: tuple[float, float, float],
    conveyor_axis: int,
    conveyor_bounds: tuple[float, float],
    margin: float,
) -> torch.Tensor:
    center = position.new_tensor(conveyor_center)
    size = position.new_tensor(conveyor_size)
    lower, upper = conveyor_bounds
    lateral_axis = 1 if conveyor_axis == 0 else 0
    along = (position[:, conveyor_axis] >= lower + margin) & (
        position[:, conveyor_axis] <= upper - margin
    )
    lateral = torch.abs(position[:, lateral_axis] - center[lateral_axis]) <= max(
        float(size[lateral_axis] / 2 - margin), 0.0
    )
    return along & lateral


def medicine_grasp_signal(
    env,
    target_cfg: SceneEntityCfg,
    support_surface_z: float,
    lift_threshold: float,
) -> torch.Tensor:
    """Signal the first stable grasp/lift transition used by Mimic."""

    position = _target_position(env, target_cfg)
    return _gripper_is_closed(env, threshold=0.15) & (
        position[:, 2] > support_surface_z + lift_threshold
    )


def medicine_place_signal(
    env,
    target_cfg: SceneEntityCfg,
    conveyor_center: tuple[float, float, float],
    conveyor_size: tuple[float, float, float],
    conveyor_axis: int,
    conveyor_bounds: tuple[float, float],
    conveyor_surface_z: float,
    target_support_height: float,
    region_margin: float,
    surface_tolerance: float,
) -> torch.Tensor:
    """Signal a released bottle resting on the conveyor acceptance region."""

    position = _target_position(env, target_cfg)
    in_region = _inside_conveyor_region(
        position,
        conveyor_center,
        conveyor_size,
        conveyor_axis,
        conveyor_bounds,
        region_margin,
    )
    expected_z = conveyor_surface_z + target_support_height
    settled = torch.abs(position[:, 2] - expected_z) <= surface_tolerance
    return in_region & settled & ~_gripper_is_closed(env, threshold=0.15)


def medicine_success(
    env,
    conveyor_center: tuple[float, float, float],
    conveyor_size: tuple[float, float, float],
    conveyor_axis: int,
    conveyor_bounds: tuple[float, float],
    region_margin: float,
    target_cfg: SceneEntityCfg | None = None,
    target_cfgs: tuple[SceneEntityCfg, ...] | None = None,
    target_support_extents: tuple[tuple[float, float, float], ...] | None = None,
    target_upright_axes: tuple[tuple[float, float, float], ...] | None = None,
    surface_height_tolerance: float = 0.04,
    require_release: bool = True,
    require_upright: bool = True,
    min_upright_cosine: float = 0.95,
) -> torch.Tensor:
    if target_cfgs is None:
        if target_cfg is None:
            raise ValueError("medicine_success requires target_cfg or target_cfgs")
        target_cfgs = (target_cfg,)
    if target_support_extents is None:
        raise ValueError("medicine_success requires target_support_extents")
    if target_upright_axes is None:
        target_upright_axes = tuple((0.0, 0.0, 1.0) for _ in target_cfgs)
    return medicine_on_conveyor(
        env=env,
        target_cfgs=target_cfgs,
        target_support_extents=target_support_extents,
        target_upright_axes=target_upright_axes,
        conveyor_center=conveyor_center,
        conveyor_size=conveyor_size,
        conveyor_axis=conveyor_axis,
        conveyor_bounds=conveyor_bounds,
        region_margin=region_margin,
        surface_height_tolerance=surface_height_tolerance,
        require_release=require_release,
        require_upright=require_upright,
        min_upright_cosine=min_upright_cosine,
    )


@configclass
class BiomedicalTerminationCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class BiomedicalMimicEnvCfg(MimicEnvCfg):
    """IsaacLab Mimic configuration for one atomic medicine pick/place task."""

    arm_mode: ArmMode = ArmMode.SINGLE_ARM
    target_object_name: str = "medicine_bottle_00"
    destination_location_name: str = "conveyor_surface"

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "embodied_fusion_biomedical_pick_place_D0"
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)
        self.datagen_config.generation_relative = True
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.subtask_configs = {
            "robot": [
                SubTaskConfig(
                    object_ref=self.target_object_name,
                    subtask_term_signal="grasp",
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.005,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                SubTaskConfig(
                    # The conveyor is a static support at a fixed world pose,
                    # so the place segment is replayed in absolute coordinates
                    # instead of being anchored to the carried bottle (whose
                    # in-hand pose would otherwise shift the target trajectory
                    # away from the conveyor).
                    object_ref=None,
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="random",
                    selection_strategy_kwargs=None,
                    action_noise=0.005,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
            ]
        }


@agent_ready
@register_task
class BiomedicalPickMedicineTask(TaskBase):
    """Atomic medicine pick/place semantics for an Arena environment.

    The environment owns a pool of candidate targets.  The success predicate
    is vectorized over that pool so a scripted policy can complete one target
    at a time without inventing a second task-specific success contract.
    """

    def __init__(
        self,
        target_object: Asset,
        destination_location: Asset,
        background_scene: Asset,
        *,
        target_objects: tuple[Asset, ...] | None = None,
        conveyor_center: tuple[float, float, float],
        conveyor_size: tuple[float, float, float],
        conveyor_axis: int,
        conveyor_bounds: tuple[float, float],
        conveyor_surface_z: float,
        target_support_height: float,
        target_support_extents: tuple[tuple[float, float, float], ...] | None = None,
        target_upright_axes: tuple[tuple[float, float, float], ...] | None = None,
        support_surface_z: float,
        target_randomization: dict[str, Any] | None = None,
        viewer_robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        region_margin: float = 0.05,
        surface_tolerance: float = 0.04,
        episode_length_s: float = 30.0,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.target_objects = target_objects or (target_object,)
        self.target_object = self.target_objects[0]
        self.destination_location = destination_location
        self.background_scene = background_scene
        self.conveyor_center = conveyor_center
        self.conveyor_size = conveyor_size
        self.conveyor_axis = conveyor_axis
        self.conveyor_bounds = conveyor_bounds
        self.conveyor_surface_z = conveyor_surface_z
        self.target_support_height = target_support_height
        self.target_support_extents = target_support_extents or tuple(
            (target_support_height, target_support_height, target_support_height)
            for _ in self.target_objects
        )
        self.target_upright_axes = target_upright_axes or tuple(
            (0.0, 0.0, 1.0) for _ in self.target_objects
        )
        self.target_randomization = dict(target_randomization or {})
        self.support_surface_z = support_surface_z
        self.viewer_robot_position = viewer_robot_position
        self.region_margin = region_margin
        self.surface_tolerance = surface_tolerance
        self.task_description = task_description or (
            f"Pick up {target_object.name}, place it upright on the conveyor, and release it."
        )
        # Success is defined by the released object's pose in the conveyor
        # acceptance region.  The conveyor is a static support, so a PhysX
        # contact sensor would not be valid here (Arena contact filters require
        # a rigid target).  Keep contact sensing opt-in for future task variants
        # instead of manufacturing an invalid sensor configuration.
        self.scene_config = None
        self.events_cfg = self._make_events_cfg()
        self.termination_cfg = BiomedicalTerminationCfg(
            success=TerminationTermCfg(
                func=medicine_success,
                params=self._predicate_params(),
            )
        )

    def make_policy_observation_cfg(self, base_policy_cfg: ObservationGroupCfg) -> ObservationGroupCfg:
        """Extend the native embodiment policy group with task state.

        Arena merges observation groups by dataclass field type.  Building the
        extension from the embodiment's native policy type keeps that merge
        contract intact while making the target visible to Robomimic.
        """

        target_cfg = SceneEntityCfg(self.target_object.name)
        robot_cfg = SceneEntityCfg("robot")

        def _configure_policy_group(group) -> None:
            group.enable_corruption = False
            group.concatenate_terms = False

        PolicyObsCfg = make_configclass(
            "BiomedicalPolicyObservationCfg",
            [
                (
                    "target_object_pos",
                    ObservationTermCfg,
                    ObservationTermCfg(
                        func=medicine_target_position_in_robot_frame,
                        params={"target_cfg": target_cfg, "robot_cfg": robot_cfg},
                    ),
                ),
                (
                    "target_object_quat",
                    ObservationTermCfg,
                    ObservationTermCfg(
                        func=medicine_target_quaternion_in_robot_frame,
                        params={"target_cfg": target_cfg, "robot_cfg": robot_cfg},
                    ),
                ),
            ],
            bases=(type(base_policy_cfg),),
            namespace={"__post_init__": _configure_policy_group},
        )
        policy_cfg = PolicyObsCfg()
        # ``make_configclass`` inherits class defaults, but the embodiment may
        # have changed terms on its concrete policy instance (for example,
        # DROID EEF observations are robot-root-frame values).  Preserve that
        # instance state while leaving the task-owned group flags above intact.
        for field in fields(base_policy_cfg):
            if field.name in {"enable_corruption", "concatenate_terms"}:
                continue
            setattr(policy_cfg, field.name, deepcopy(getattr(base_policy_cfg, field.name)))
        return policy_cfg

    def _predicate_params(self) -> dict[str, Any]:
        return {
            "target_cfgs": tuple(SceneEntityCfg(target.name) for target in self.target_objects),
            "target_support_extents": self.target_support_extents,
            "target_upright_axes": self.target_upright_axes,
            "conveyor_center": self.conveyor_center,
            "conveyor_size": self.conveyor_size,
            "conveyor_axis": self.conveyor_axis,
            "conveyor_bounds": self.conveyor_bounds,
            "region_margin": self.region_margin,
            "surface_height_tolerance": self.surface_tolerance,
            "require_release": True,
            "require_upright": True,
            "min_upright_cosine": 0.95,
        }

    def _place_observation_params(self, target_cfg: SceneEntityCfg) -> dict[str, Any]:
        """Build the single-target contract required by ``medicine_place_signal``.

        The termination predicate evaluates all sampled bottles, while the
        Mimic ``place`` subtask observation describes the task's primary
        target.  These are deliberately separate contracts: passing the
        multi-target termination parameters to the single-target observation
        function makes IsaacLab reject the observation manager configuration.
        """

        return {
            "target_cfg": target_cfg,
            "conveyor_center": self.conveyor_center,
            "conveyor_size": self.conveyor_size,
            "conveyor_axis": self.conveyor_axis,
            "conveyor_bounds": self.conveyor_bounds,
            "conveyor_surface_z": self.conveyor_surface_z,
            "target_support_height": self.target_support_height,
            "region_margin": self.region_margin,
            "surface_tolerance": self.surface_tolerance,
        }

    def apply_reachability_constraints(self) -> None:
        # The destination is a fixed conveyor support, not a movable object that
        # needs its own placement/reachability solve.  Requiring reachability for
        # it would make primitive-only destinations enter Arena's USD bounding
        # box path.  The movable target is the only object whose layout must be
        # checked against the selected embodiment.
        self._apply_reachability_constraints(list(self.target_objects))

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def _make_events_cfg(self):
        randomization = self.target_randomization
        if not bool(randomization.get("enabled", False)):
            return None

        pose_spec = dict(randomization.get("pose_range", {}))
        pose_range = {
            axis: tuple(float(value) for value in pose_spec.get(axis, (0.0, 0.0)))
            for axis in ("x", "y", "z", "yaw")
        }
        if any(len(values) != 2 or values[1] < values[0] for values in pose_range.values()):
            raise ValueError("Biomedical target pose ranges must be increasing pairs")
        count_range = tuple(randomization.get("count_range", (len(self.target_objects),) * 2))
        if len(count_range) != 2:
            raise ValueError("Biomedical target count range must be a pair")
        min_count, max_count = (int(count_range[0]), int(count_range[1]))
        inactive_offset = tuple(
            float(value)
            for value in randomization.get("inactive_position_offset", (0.0, 0.0, -10.0))
        )
        if len(inactive_offset) != 3:
            raise ValueError("Biomedical inactive target position offset must be a 3-vector")

        event = EventTermCfg(
            func=randomize_medicine_target_layout,
            mode="reset",
            params={
                "target_cfgs": tuple(SceneEntityCfg(target.name) for target in self.target_objects),
                "min_count": min_count,
                "max_count": max_count,
                "pose_range": pose_range,
                "target_support_extents": self.target_support_extents,
                "support_surface_z": float(
                    randomization.get("support_surface_z", self.support_surface_z)
                ),
                "surface_clearance": float(randomization.get("surface_clearance", 0.005)),
                "inactive_position_offset": inactive_offset,
                "required_target_name": self.target_object.name,
            },
        )
        EventsCfg = make_configclass(
            "BiomedicalTargetEventsCfg",
            [("randomize_target_layout", EventTermCfg, event)],
        )
        return EventsCfg()

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return BiomedicalMimicEnvCfg(
            arm_mode=arm_mode,
            target_object_name=self.target_object.name,
            destination_location_name=self.destination_location.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.target_object)]

    def get_observation_cfg(self):
        target_cfg = SceneEntityCfg(self.target_object.name)

        def _configure_subtask_group(group) -> None:
            # Match IsaacLab Mimic task configs: scalar subtask flags remain
            # individual terms and are not concatenated into a vector.
            group.enable_corruption = False
            group.concatenate_terms = False

        subtask_group_cfg = make_configclass(
            "BiomedicalSubtaskTermsCfg",
            [
                (
                    "grasp",
                    ObservationTermCfg,
                    ObservationTermCfg(
                        func=medicine_grasp_signal,
                        params={
                            "target_cfg": target_cfg,
                            "support_surface_z": self.support_surface_z,
                            # Threshold above the spawn height (0.8508) and
                            # above the height of a bottle held against the
                            # gripper by collision at reset (~0.863): a
                            # freshly reset bottle otherwise spuriously fires
                            # the grasp signal at step 0.
                            "lift_threshold": 0.10,
                        },
                    ),
                ),
                (
                    "place",
                    ObservationTermCfg,
                    ObservationTermCfg(
                        func=medicine_place_signal,
                        params=self._place_observation_params(target_cfg),
                    ),
                ),
            ],
            bases=(ObservationGroupCfg,),
            namespace={"__post_init__": _configure_subtask_group},
        )
        return make_configclass(
            "BiomedicalObservationCfg",
            [("subtask_terms", subtask_group_cfg, subtask_group_cfg())],
        )()

    def get_viewer_cfg(self) -> ViewerCfg:
        # Preserve the original biomedical viewport contract: it is framed
        # relative to the DROID base, not the conveyor's world coordinates.
        robot_x, robot_y, robot_z = self.viewer_robot_position
        return ViewerCfg(
            eye=(robot_x - 2.4, robot_y - 3.5, robot_z + 3.0),
            lookat=(robot_x + 1.2, robot_y - 0.25, robot_z + 0.05),
            origin_type="env",
        )

    @classmethod
    def success_state_transition(cls, target_object: str, destination_location: str, **_) -> TaskTransition:
        relation = ObjectRelationLibraryRegistry().get_object_relation_by_name("on")
        return TaskTransition(
            subject=target_object,
            effects=(Relocate(subject=target_object, relation=relation.name, target=destination_location),),
        )


__all__ = [
    "BiomedicalMimicEnvCfg",
    "BiomedicalPickMedicineTask",
    "BiomedicalTerminationCfg",
    "medicine_grasp_signal",
    "medicine_place_signal",
    "medicine_success",
]
