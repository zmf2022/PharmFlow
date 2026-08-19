"""cuRobo expert for the biomedical DROID collection task.

The planner consumes the live IsaacLab object poses, derives grasp and release
poses from the authored object extents, and executes the resulting joint
trajectory through the native DROID arm/gripper action contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
from typing import Any

import torch
import isaaclab.utils.math as PoseUtils
from isaaclab.utils.math import matrix_from_quat
from isaaclab_arena_curobo.utils.frame_utils import world_pose_to_robot_frame
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CollisionCheckerType
from isaaclab_arena.policy.policy_base import PolicyBase
from curobo.types.state import JointState

from pharm_flow.data_collection.arena.biomedical_contract import (
    DROID_BASE_TO_CLOSED_GRASP_CENTER,
    DROID_GRIPPER_CLOSED_POSITION,
    DROID_GRIPPER_MAX_WIDTH,
    DROID_GRIPPER_OPEN_POSITION,
    droid_grasp_position,
    medicine_on_conveyor,
    _target_positions,
)
from pharm_flow.data_collection.arena.curobo import (
    make_planner_cfg,
    sync_object_poses_in_robot_base_frame,
)
from pharm_flow.data_collection.utils.skills import (
    AtomicSkillContext,
    AtomicSkillOutput,
    validate_skill_pipeline,
)
from .pick_place_skills import PickPlaceSkills


DEFAULT_PICK_PLACE_PIPELINE: tuple[str, ...] = (
    "approach",
    "grasp",
    "hold_grasp",
    "lift",
    "transport",
    "reorient",
    "place",
    "hold_release",
    "release_lift",
    "home",
)


@dataclass(frozen=True)
class MedicinePickPlaceConfig:
    # Keep the planner target aligned with the target state exposed by the
    # Biomedical task observation contract.  Distractor bottles remain in the
    # scene, but they must not silently become the policy target.
    target_object_name: str | None = None
    # Dynamic bottles are written to their sampled reset pose before PhysX
    # has resolved contact with the carton floor.  Settle that reset state
    # before reading object poses for grasp planning.
    reset_settle_steps: int = 32
    approach_clearance: float = 0.08
    grasp_pose_path: str | None = None
    interaction_frame_rotation: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    max_grasp_candidates: int = 300
    feasible_grasp_candidates: int = 8
    disable_upside_down: bool = True
    minimum_grasp_width_ratio: float = 0.9
    maximum_grasp_width_ratio: float = 1.1
    max_grasp_center_offset: float = 0.005
    grasp_long_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    lift_height: float = 0.16
    place_clearance: float = 0.10
    release_clearance: float = 0.02
    retreat_clearance: float = 0.10
    release_lift_height: float = 0.08
    # Horizontal distance along the conveyor travel axis from the robot base to
    # the bottle drop centroid. Keeps the wrist in a natural straight-down grasp
    # instead of descending into the base's own workspace and hitting the belt.
    place_forward_distance: float = 0.65
    # Extra vertical clearance above the lift apex for the horizontal carry so
    # the still-flat bottle clears the carton, neighbouring bottles, and the
    # belt edge while being transported to the drop centroid.
    transport_carry_clearance: float = 0.10
    hold_steps: int = 12
    waypoint_step: float = 0.02
    transport_waypoint_step: float = 0.03
    place_waypoint_step: float = 0.03
    collision_checker: str = "PRIMITIVE"
    enable_graph: bool = False
    use_cuda_graph: bool = False
    warmup: bool = False
    joint_target_tolerance: float = 0.03
    max_motion_stage_steps: int = 240
    grasp_closure_min: float = 0.005
    empty_grasp_closed_margin: float = 0.05
    max_grasp_settle_steps: int = 120
    lift_verification_height: float = 0.02  # Reduced from 0.03 to account for physics delay
    lift_verification_steps: int = 3  # Require sustained lift over multiple steps
    grasp_position_delta_tolerance: float = 0.002
    # The semantic stage graph is task configuration, while the planner and
    # robot contracts stay in this expert implementation.  Keep the default
    # identical to the established biomedical collection order.
    pipeline: tuple[str, ...] = DEFAULT_PICK_PLACE_PIPELINE


_DEFAULT_GRASP_POSE_PATH = (
    Path(__file__).resolve().parents[3]
    / "assets/interaction/benchmark_medicine_bottle_000/grasp_pose/grasp_pose.pkl"
)


@lru_cache(maxsize=16)
def _load_authored_grasp_data(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load immutable object-local grasp poses and gripper widths once.

    These poses are the source-of-truth interaction annotations shipped with
    the object asset.  Keeping the cache on CPU avoids copying hundreds of
    poses to CUDA on every reset; the selected batch is moved by the planner's
    normal device conversion.
    """

    grasp_path = Path(path).expanduser().resolve()
    if not grasp_path.is_file():
        raise FileNotFoundError(f"Authored grasp pose file does not exist: {grasp_path}")
    with grasp_path.open("rb") as handle:
        data = pickle.load(handle)
    if "grasp_pose" not in data:
        raise ValueError(f"Authored grasp data has no 'grasp_pose' field: {grasp_path}")
    poses = torch.as_tensor(data["grasp_pose"], dtype=torch.float32).contiguous()
    widths = torch.as_tensor(
        data.get("width", torch.zeros(len(poses))), dtype=torch.float32
    ).contiguous()
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasp_pose with shape [N,4,4], got {tuple(poses.shape)}")
    if widths.shape != (poses.shape[0],):
        raise ValueError(
            f"Expected width with shape [{poses.shape[0]}], got {tuple(widths.shape)}"
        )
    return poses, widths


class _BiomedicalDroidCuroboPlanner(CuroboPlanner):
    """DROID planner using the robot-base frame for dynamic scene objects."""

    def __init__(
        self,
        *args: Any,
        empty_grasp_closed_margin: float,
        **kwargs: Any,
    ):
        # Contact planning is a task-stage property. Keep it in this local
        # adapter so the shared IsaacLab planner remains unchanged. Contact is
        # enabled only for stages that intentionally start or finish against
        # the carton/bottle/belt geometry.
        self.contact_planning = False
        self.empty_grasp_closed_margin = empty_grasp_closed_margin
        super().__init__(*args, **kwargs)

    def _plan_to_contact(
        self,
        start_state: JointState,
        goal_pose: Any,
        retreat_distance: float,
        approach_distance: float,
        contact: bool = False,
        retime_plan: bool = False,
        step_size: float | None = None,
    ) -> bool:
        """Use Mimic's official contact phases only for contact task stages."""

        return super()._plan_to_contact(
            start_state=start_state,
            goal_pose=goal_pose,
            retreat_distance=retreat_distance,
            approach_distance=approach_distance,
            contact=contact or self.contact_planning,
            retime_plan=retime_plan,
            step_size=step_size,
        )

    def _sync_object_poses_with_isaaclab(self) -> None:
        # Curobo goals are expressed in the robot base frame.  Keep collision
        # objects in that same frame; the generic Mimic implementation uses
        # the environment origin for this update.
        sync_object_poses_in_robot_base_frame(self)

    def _check_object_grasped(self, gripper_pos: torch.Tensor, object_name: str) -> bool:
        """Use the DROID gripper direction for cuRobo attachment checks.

        The shared Mimic planner assumes that a smaller joint value means a
        closed gripper.  DROID is the opposite: ``finger_joint=0`` is open
        and ``pi/4`` is closed.  Read the named drive joint directly instead
        of inheriting the generic ``[-2:]`` gripper slice, which is not the
        DROID actuator contract.
        """

        joint_index = self.robot.data.joint_names.index("finger_joint")
        finger_position = self.robot.data.joint_pos.torch[0, joint_index]
        object_grasped = bool(
            self.config.grasp_gripper_open_val < finger_position.item()
            < float(DROID_GRIPPER_CLOSED_POSITION) - self.empty_grasp_closed_margin
        )
        self.logger.info(
            f"Object {object_name} is grasped: {object_grasped} "
            f"(finger_joint={finger_position.item():.4f}, "
            f"closed={float(DROID_GRIPPER_CLOSED_POSITION):.4f})"
        )
        return object_grasped


class MedicinePickPlaceExpert(PolicyBase):
    """Plan and execute one medicine pick-and-place trajectory at a time."""

    def __init__(self, env: Any, success_term: Any, config: MedicinePickPlaceConfig):
        if config.joint_target_tolerance <= 0.0:
            raise ValueError("joint_target_tolerance must be positive")
        if config.reset_settle_steps < 0:
            raise ValueError("reset_settle_steps must be non-negative")
        if config.max_motion_stage_steps <= 0:
            raise ValueError("max_motion_stage_steps must be positive")
        if config.waypoint_step <= 0.0:
            raise ValueError("waypoint_step must be positive")
        if config.transport_waypoint_step <= 0.0:
            raise ValueError("transport_waypoint_step must be positive")
        if config.place_waypoint_step <= 0.0:
            raise ValueError("place_waypoint_step must be positive")
        if config.grasp_closure_min <= 0.0:
            raise ValueError("grasp_closure_min must be positive")
        if config.empty_grasp_closed_margin <= 0.0:
            raise ValueError("empty_grasp_closed_margin must be positive")
        if config.max_grasp_settle_steps < config.hold_steps:
            raise ValueError("max_grasp_settle_steps must be at least hold_steps")
        if not 0.0 < config.minimum_grasp_width_ratio <= 1.0:
            raise ValueError("minimum_grasp_width_ratio must be in (0, 1]")
        if config.maximum_grasp_width_ratio < 1.0:
            raise ValueError("maximum_grasp_width_ratio must be at least 1")
        if config.max_grasp_center_offset <= 0.0:
            raise ValueError("max_grasp_center_offset must be positive")
        if len(config.grasp_long_axis) != 3:
            raise ValueError("grasp_long_axis must contain three values")
        if sum(value * value for value in config.grasp_long_axis) <= 0.0:
            raise ValueError("grasp_long_axis must be non-zero")
        if config.lift_verification_height <= 0.0:
            raise ValueError("lift_verification_height must be positive")
        if config.grasp_position_delta_tolerance <= 0.0:
            raise ValueError("grasp_position_delta_tolerance must be positive")
        if config.feasible_grasp_candidates <= 0:
            raise ValueError("feasible_grasp_candidates must be positive")

        super().__init__(config)
        self.env = env
        self.config = config
        self._droid_gripper_closed_position = DROID_GRIPPER_CLOSED_POSITION
        self._droid_grasp_position = droid_grasp_position
        self._target_positions = _target_positions
        self._base_to_grasp_center = DROID_BASE_TO_CLOSED_GRASP_CENTER.to(
            device=env.device, dtype=torch.float32
        )
        self._gripper_max_width = DROID_GRIPPER_MAX_WIDTH
        interaction_quat = torch.as_tensor(
            config.interaction_frame_rotation,
            device=env.device,
            dtype=torch.float32,
        )
        if interaction_quat.shape != (4,):
            raise ValueError("interaction_frame_rotation must be an XYZW quaternion")
        self._interaction_frame_rotation = matrix_from_quat(
            interaction_quat.unsqueeze(0)
        )[0]
        self._target_cfgs = tuple(success_term.params["target_cfgs"])
        if config.target_object_name is not None and not any(
            cfg.name == config.target_object_name for cfg in self._target_cfgs
        ):
            raise ValueError(
                f"Unknown target_object_name {config.target_object_name!r}; "
                f"available targets are {[cfg.name for cfg in self._target_cfgs]}"
            )
        self._target_extents = tuple(success_term.params["target_support_extents"])
        self._target_upright_axes = torch.as_tensor(
            success_term.params["target_upright_axes"],
            device=env.device,
            dtype=torch.float32,
        )
        self._conveyor_center = torch.as_tensor(
            success_term.params["conveyor_center"], device=env.device, dtype=torch.float32
        )
        self._conveyor_size = torch.as_tensor(
            success_term.params["conveyor_size"], device=env.device, dtype=torch.float32
        )
        self._conveyor_axis = int(success_term.params["conveyor_axis"])
        self._conveyor_bounds = tuple(
            float(value) for value in success_term.params["conveyor_bounds"]
        )
        self._conveyor_region_margin = float(success_term.params["region_margin"])
        self._success_term = success_term
        # IsaacLab-Arena owns the DROID hardware description: Robotiq mimic
        # joints, ``finger_joint`` gripper semantics, base_link EE frame, URDF,
        # and cuRobo collision spheres.  Reuse that registry instead of the
        # generic Panda config, which has a different joint contract.
        class _DroidEmbodiment:
            name = "droid"

        planner_cfg = make_planner_cfg(_DroidEmbodiment())
        # Keep the planner aligned with the shared DROID actuator contract:
        # finger_joint=0 is open and finger_joint=pi/4 is closed.
        planner_cfg.gripper_open_positions["finger_joint"] = float(DROID_GRIPPER_OPEN_POSITION)
        planner_cfg.gripper_closed_positions["finger_joint"] = float(DROID_GRIPPER_CLOSED_POSITION)
        # A physical grasp obstructs the gripper shortly after it leaves the
        # open stop. Reaching the fully closed stop means the fingers closed
        # through empty space and must never be treated as an attachment.
        planner_cfg.grasp_gripper_open_val = float(
            DROID_GRIPPER_OPEN_POSITION + config.grasp_closure_min
        )
        # Resolve the robot prim from the Arena embodiment instead of assuming
        # a project-specific ``robot``/``Robot`` spelling.  The planner's
        # default is ``/World/envs/env_0/Robot``; deriving the leaf from the
        # actual scene keeps this expert compatible with every registered
        # DROID embodiment.
        robot_prim_leaf = str(env.scene["robot"].cfg.prim_path).rstrip("/").split("/")[-1]
        env_prim_path = "/World/envs/env_0"
        planner_cfg.robot_prim_path = f"{env_prim_path}/{robot_prim_leaf}"
        planner_cfg.world_ignore_substrings = [
            planner_cfg.robot_prim_path,
            f"{env_prim_path}/EmbodiedScene/background",
            f"{env_prim_path}/EmbodiedScene/medicine_bottle_",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        try:
            planner_cfg.collision_checker_type = CollisionCheckerType(
                config.collision_checker.upper()
            )
        except ValueError as exc:
            supported = ", ".join(item.value for item in CollisionCheckerType)
            raise ValueError(
                f"Unsupported automatic-collection collision_checker={config.collision_checker!r}; "
                f"choose one of: {supported}"
            ) from exc
        planner_cfg.enable_graph = config.enable_graph
        planner_cfg.use_cuda_graph = config.use_cuda_graph
        planner_cfg.warmup = config.warmup
        # ``None`` is cuRobo's explicit value for keeping graph planning off.
        # Zero would enable graph planning immediately after the first failed
        # trajectory-optimization attempt, even when ``enable_graph`` is false.
        planner_cfg.enable_graph_attempt = (
            planner_cfg.enable_graph_attempt if config.enable_graph else None
        )
        # The expert already exposes approach/grasp/lift/place waypoints.
        # Avoid adding a second internal approach waypoint that can create an
        # unnecessary IK discontinuity near the tabletop.
        planner_cfg.approach_distance = 0.0
        planner_cfg.retreat_distance = 0.0
        # The authored collection path is executed as position targets.  Use
        # the native full-speed retiming factor; the shared DROID config's
        # 0.6 factor is intended for slower interactive demonstrations.
        planner_cfg.time_dilation_factor = 1.0
        planner_cfg.debug_planner = False
        # Let IsaacLab-Mimic construct and warm up its native MotionGen.  The
        # warm-up initializes the CUDA kernels and planner caches required by
        # the first real plan; bypassing it leaves the planner incomplete and
        # can terminate the Kit application before collection starts.
        self._planner = _BiomedicalDroidCuroboPlanner(
            env=env,
            robot=env.scene["robot"],
            config=planner_cfg,
            env_id=0,
            empty_grasp_closed_margin=config.empty_grasp_closed_margin,
        )
        self._plan_positions: torch.Tensor | None = None
        self._plan_arm_indices: tuple[int, ...] = ()
        self._last_joint_target: torch.Tensor | None = None
        self._plan_index = 0
        self._skill_set = PickPlaceSkills()
        self._skill_registry = self._skill_set.registry()
        self._pipeline_names = validate_skill_pipeline(
            config.pipeline, self._skill_registry
        )
        self._last_skill_output: AtomicSkillOutput | None = None
        self._set_stage("idle")
        self._stage_steps = 0
        self._target_index = -1
        self._target_name = ""
        self._poses: dict[str, torch.Tensor] = {}
        self._grasp_candidates: torch.Tensor | None = None
        self._grasp_centers: torch.Tensor | None = None
        self._object_pose: torch.Tensor | None = None
        self._robot_position: torch.Tensor | None = None
        self._robot_rotation: torch.Tensor | None = None
        self._upright_axis: torch.Tensor | None = None
        self._grasp_center_target: torch.Tensor | None = None
        self._pre_lift_target_z: float | None = None
        self._lift_verified_steps: int = 0  # Track consecutive lift verification successes
        self._grasp_stable_steps = 0
        self._last_gripper_position: float | None = None
        self._failed = False
        self._failure_reason: str | None = None
        self._home_reached = False
        self._home_stable_steps = 0
        # A settling pass is required after each environment reset so planning
        # reads the physically settled target pose.
        self._reset_settle_pending = True
        robot_body_names = tuple(env.scene["robot"].data.body_names)
        if "base_link" not in robot_body_names:
            raise RuntimeError(
                "DROID robot must expose the cuRobo end-effector body 'base_link'"
            )
        self._ee_body_index = robot_body_names.index("base_link")
        self._curobo_to_isaac_ee = self._calibrate_ee_frames()
        self._home_pose = self._current_physical_ee_pose()

    def _set_stage(self, stage: str) -> None:
        """Set a validated semantic stage without changing stage timing."""

        if stage not in {"idle", "done", "failed", "retreat", *self._pipeline_names}:
            raise RuntimeError(
                f"Unknown biomedical skill stage {stage!r}; "
                f"configured pipeline={self._pipeline_names}"
            )
        self._stage = stage

    def _advance_stage(self, terminal: str = "done") -> str:
        """Advance according to the configured biomedical stage sequence."""

        try:
            stage_index = self._pipeline_names.index(self._stage)
        except ValueError as exc:
            raise RuntimeError(f"Cannot advance unknown stage {self._stage!r}") from exc
        next_stage = (
            self._pipeline_names[stage_index + 1]
            if stage_index + 1 < len(self._pipeline_names)
            else terminal
        )
        self._set_stage(next_stage)
        return next_stage

    @property
    def done(self) -> bool:
        return self._stage in {"done", "failed"}

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def ready_to_commit(self) -> bool:
        """Whether the recorder may export this episode.

        A successful placement is not the end of the collection episode.  The
        arm must first complete the configured safe ``home`` motion so the
        final recorded frame is not the release posture.
        """

        return self._home_reached and not self._failed

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def target_name(self) -> str:
        return self._target_name

    def _target_on_conveyor(self, *, require_release: bool) -> bool:
        """Check the current bottle's physical placement state."""
        if self._target_index < 0:
            return False
        index = self._target_index
        params = dict(self._success_term.params)
        params["target_cfgs"] = (params["target_cfgs"][index],)
        params["target_support_extents"] = (params["target_support_extents"][index],)
        params["target_upright_axes"] = (params["target_upright_axes"][index],)
        params["require_release"] = require_release
        return bool(medicine_on_conveyor(self.env, **params)[0].item())

    def target_succeeded(self) -> bool:
        """True when the current target is upright, placed, and released."""

        return self._target_on_conveyor(require_release=True)

    def start_new_episode(self) -> None:
        """Compatibility name for the Arena ``PolicyBase.reset`` lifecycle."""
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None and env_ids.numel() not in (0, 1):
            raise NotImplementedError(
                "MedicinePickPlaceExpert is episode-scoped and supports one environment."
            )
        self._reset_settle_pending = True
        self._reset_target()

    def _reset_target(self) -> None:
        """Select and plan one target in the current physical scene."""

        self._failed = False
        self._failure_reason = None
        self._home_reached = False
        self._home_stable_steps = 0
        self._pre_lift_target_z = None
        self._lift_verified_steps = 0
        if self._reset_settle_pending:
            self._settle_reset_objects()
            self._reset_settle_pending = False
        positions = self._target_positions(self.env, self._target_cfgs)[0]
        self._planner.logger.info(
            "Biomedical reset target pool: "
            f"active_bottles={int((positions[:, 2] > 0.0).sum().item())}/"
            f"{len(self._target_cfgs)}"
        )
        active = torch.where(positions[:, 2] > 0.0)[0].to(device=self.env.device)
        if len(active) == 0:
            raise RuntimeError("No active medicine bottle is available for automatic collection")

        if self.config.target_object_name is not None:
            target_index = next(
                index
                for index, cfg in enumerate(self._target_cfgs)
                if cfg.name == self.config.target_object_name
            )
            if not bool((active == target_index).any().item()):
                raise RuntimeError(
                    f"Configured target {self.config.target_object_name!r} is not active "
                    "after reset"
                )
            active = torch.as_tensor([target_index], device=self.env.device)

        # The configured target is the task's semantic target.  If no explicit
        # target is configured, retain the historical nearest-active fallback.
        robot = self.env.scene["robot"]
        eef = self._droid_grasp_position(self.env)[0]
        distances = torch.linalg.vector_norm(positions[active] - eef, dim=-1)
        self._target_index = int(active[torch.argmin(distances)].item())
        self._target_name = self._target_cfgs[self._target_index].name

        target_asset = self.env.scene[self._target_name]
        target, target_quat = world_pose_to_robot_frame(
            target_asset.data.root_pos_w.torch[0, :3],
            target_asset.data.root_quat_w.torch[0, :4],
            robot.data.root_pos_w.torch[0, :3],
            robot.data.root_quat_w.torch[0, :4],
        )
        robot_position = robot.data.root_pos_w.torch[0, :3] - self.env.scene.env_origins[0]
        robot_rotation = matrix_from_quat(robot.data.root_quat_w.torch[0].unsqueeze(0))[0]
        self._upright_axis = self._target_upright_axes[self._target_index].to(
            dtype=target.dtype
        )
        self._object_pose = torch.eye(4, device=self.env.device, dtype=target.dtype)
        self._object_pose[:3, :3] = matrix_from_quat(target_quat.unsqueeze(0))[0]
        self._object_pose[:3, 3] = target
        self._robot_position = robot_position
        self._robot_rotation = robot_rotation
        # Keep an ordered candidate set so reset-time planning can choose the
        # best initially feasible pose.  Execution failures are episode
        # failures and are handled by the collection runner's reset boundary.
        self._grasp_candidates, self._grasp_centers = self._select_authored_grasp_poses(
            target, target_quat
        )
        self._grasp_stable_steps = 0
        self._last_gripper_position = None
        self._configure_grasp_candidate(0)
        self._plan_positions = None
        self._plan_index = 0
        self._set_stage("approach")
        self._stage_steps = 0
        self._plan_current_stage(expected_attached_object=None)

    def _settle_reset_objects(self) -> None:
        """Let reset-spawned dynamic objects resolve physical contact.

        ``ManagerBasedRLEnv.reset`` writes the sampled root poses and calls
        ``sim.forward`` but intentionally does not advance physics.  The
        medicine bottles therefore move on their first physics steps as they
        settle on the carton floor.  Planning before that movement makes the
        grasp target stale in both Z and the horizontal plane.  Advance only
        the physics scene here (without ``env.step``) so no warm-up actions or
        recorder samples enter the demonstration.
        """

        if self.config.reset_settle_steps == 0:
            return
        before = torch.stack(
            [
                self.env.scene[cfg.name].data.root_pos_w.torch[0, :3].clone()
                for cfg in self._target_cfgs
            ]
        )
        active_before = before[:, 2] > 0.0
        for _ in range(self.config.reset_settle_steps):
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(self.env.physics_dt)
        self.env.sim.forward()
        after = torch.stack(
            [
                self.env.scene[cfg.name].data.root_pos_w.torch[0, :3]
                for cfg in self._target_cfgs
            ]
        )
        displacement = torch.linalg.vector_norm(after - before, dim=-1)
        active_displacement = displacement[active_before]
        self._planner.logger.info(
            "Settled reset bottle poses before grasp planning: "
            f"steps={self.config.reset_settle_steps}, "
            f"active_bottles={int(active_before.sum().item())}, "
            f"max_displacement={active_displacement.max().item():.4f} m"
        )

    def _configure_grasp_candidate(self, candidate_index: int) -> None:
        if (
            self._grasp_candidates is None
            or self._grasp_centers is None
            or self._object_pose is None
            or self._robot_position is None
            or self._robot_rotation is None
            or self._upright_axis is None
        ):
            raise RuntimeError("Biomedical grasp candidate context is not initialized")
        grasp_pose = self._grasp_candidates[candidate_index]
        self._grasp_center_target = self._grasp_centers[candidate_index].clone()
        approach_pose = grasp_pose.clone()
        approach_pose[:3, 3] += grasp_pose[:3, :3] @ grasp_pose.new_tensor(
            (-self.config.approach_clearance, 0.0, 0.0)
        )
        lift_pose = grasp_pose.clone()
        lift_pose[:3, 3] += grasp_pose.new_tensor(
            (0.0, 0.0, self.config.lift_height)
        )
        # The source asset's visual/rigid child is rotated within the object
        # root.  Its configured upright axis is therefore the source of truth
        # rather than root-frame Z.  Align that live world-space axis to +Z
        # and apply the same delta to the gripper, preserving the grasp.
        source_rotation = self._object_pose[:3, :3]
        source_axis = source_rotation @ self._upright_axis
        reorientation = self._rotation_aligning(
            source_axis, grasp_pose.new_tensor((0.0, 0.0, 1.0))
        )
        upright_object_rotation = reorientation @ source_rotation
        tool_to_object = torch.linalg.inv(grasp_pose) @ self._object_pose
        # A bottle is rotationally symmetric around its upright axis.  The
        # minimal axis alignment above leaves the remaining world-Z yaw
        # arbitrary.  That yaw is part of the actual robot goal: selecting it
        # only by rotation distance can still produce an unreachable wrist
        # orientation.  Filter the equivalent upright goals through the same
        # cuRobo IK solver used for grasp selection, then choose the shortest
        # rotation among the feasible goals.
        yaw_count = 16
        yaw = torch.arange(
            yaw_count, device=grasp_pose.device, dtype=grasp_pose.dtype
        ) * (2.0 * torch.pi / yaw_count)
        yaw_rotation = torch.zeros(
            (yaw_count, 3, 3), device=grasp_pose.device, dtype=grasp_pose.dtype
        )
        yaw_rotation[:, 0, 0] = torch.cos(yaw)
        yaw_rotation[:, 0, 1] = -torch.sin(yaw)
        yaw_rotation[:, 1, 0] = torch.sin(yaw)
        yaw_rotation[:, 1, 1] = torch.cos(yaw)
        yaw_rotation[:, 2, 2] = 1.0
        upright_candidates = yaw_rotation @ upright_object_rotation
        tool_candidates = upright_candidates @ torch.linalg.inv(tool_to_object[:3, :3])
        relative_rotation = grasp_pose[:3, :3].transpose(0, 1) @ tool_candidates
        rotation_cosine = torch.clamp(
            (torch.diagonal(relative_rotation, dim1=1, dim2=2).sum(dim=1) - 1.0)
            * 0.5,
            -1.0,
            1.0,
        )
        rotation_cost = torch.acos(rotation_cosine)
        object_extents = torch.as_tensor(
            self._target_extents[self._target_index],
            device=self.env.device,
            dtype=grasp_pose.dtype,
        )
        lift_object_pose = lift_pose @ tool_to_object
        lift_object_rotation = lift_object_pose[:3, :3]
        drop_position = self._reachable_conveyor_position(self._robot_position)
        conveyor = self._to_robot_frame(
            drop_position, self._robot_position, self._robot_rotation
        )
        conveyor_surface_z = conveyor[2] + self._conveyor_size[2] * 0.5
        horizontal_support_height = torch.sum(
            torch.abs(lift_object_rotation[2, :]) * object_extents
        )
        carry_object_z = torch.maximum(
            lift_object_pose[2, 3] + self.config.transport_carry_clearance,
            conveyor_surface_z
            + horizontal_support_height
            + self.config.release_clearance,
        )
        feasible_yaws: list[int] = []
        for yaw_index in range(yaw_count):
            # Test the actual conveyor targets, not merely the same wrist
            # rotation at the lift position.  The EE translation changes with
            # the grasp offset when the bottle is reoriented.
            candidate_object_pose = torch.eye(
                4, device=self.env.device, dtype=grasp_pose.dtype
            )
            candidate_object_pose[:3, :3] = upright_candidates[yaw_index]
            candidate_object_pose[:3, 3] = conveyor
            candidate_object_pose[2, 3] = carry_object_z
            reorient_candidate = candidate_object_pose @ torch.linalg.inv(tool_to_object)
            reorient_planner_pose = self._physical_to_planner_pose(reorient_candidate)
            reorient_goal = self._planner._make_pose(
                position=reorient_planner_pose[:3, 3],
                quaternion=PoseUtils.quat_from_matrix(reorient_planner_pose[:3, :3]),
                normalize_rotation=True,
            )
            reorient_result = self._planner.motion_gen.ik_solver.solve_single(
                reorient_goal, return_seeds=1
            )
            if not bool(reorient_result.success.reshape(-1)[0].item()):
                continue
            candidate_support_height = torch.sum(
                torch.abs(upright_candidates[yaw_index][2, :]) * object_extents
            )
            place_object_pose = candidate_object_pose.clone()
            place_object_pose[2, 3] = (
                conveyor_surface_z
                + candidate_support_height
                + self.config.release_clearance
            )
            place_candidate = place_object_pose @ torch.linalg.inv(tool_to_object)
            place_planner_pose = self._physical_to_planner_pose(place_candidate)
            place_goal = self._planner._make_pose(
                position=place_planner_pose[:3, 3],
                quaternion=PoseUtils.quat_from_matrix(place_planner_pose[:3, :3]),
                normalize_rotation=True,
            )
            place_result = self._planner.motion_gen.ik_solver.solve_single(
                place_goal, return_seeds=1
            )
            if bool(place_result.success.reshape(-1)[0].item()):
                feasible_yaws.append(yaw_index)
        if not feasible_yaws:
            raise RuntimeError(
                f"No IK-feasible conveyor reorientation/place for {self._target_name}; "
                f"candidates={yaw_count}"
            )
        feasible_yaw_tensor = torch.as_tensor(
            feasible_yaws, device=rotation_cost.device, dtype=torch.long
        )
        selected_yaw = int(
            feasible_yaw_tensor[torch.argmin(rotation_cost[feasible_yaw_tensor])].item()
        )
        self._planner.logger.info(
            f"Selected upright yaw for {self._target_name}: "
            f"candidate={selected_yaw}/{yaw_count}, "
            f"ik_feasible={feasible_yaws}, "
            f"rotation_cost={rotation_cost[selected_yaw].item():.3f} rad"
        )
        upright_object_rotation = upright_candidates[selected_yaw]
        upright_support_height = torch.sum(
            torch.abs(upright_object_rotation[2, :]) * object_extents
        )
        desired_object_pose = torch.eye(
            4, device=self.env.device, dtype=grasp_pose.dtype
        )
        desired_object_pose[:3, :3] = upright_object_rotation
        desired_object_pose[:3, 3] = conveyor
        desired_object_pose[2, 3] = (
            conveyor_surface_z
            + upright_support_height
            + self.config.release_clearance
        )
        place_pose = desired_object_pose @ torch.linalg.inv(tool_to_object)
        # Move the bottle center directly above the drop centroid while it is
        # still horizontal.  Keep it above both the lift apex and the belt,
        # then derive the EE pose from the same rigid grasp transform.
        transport_object_pose = torch.eye(
            4, device=self.env.device, dtype=grasp_pose.dtype
        )
        transport_object_pose[:3, :3] = lift_object_rotation
        transport_object_pose[:3, 3] = conveyor
        transport_object_pose[2, 3] = carry_object_z
        transport_pose = transport_object_pose @ torch.linalg.inv(tool_to_object)
        # Keep the bottle center fixed over the belt while changing only its
        # orientation. The EE position changes as required by the grasp
        # offset, instead of making the bottle orbit around the wrist.
        reorient_object_pose = transport_object_pose.clone()
        reorient_object_pose[:3, :3] = upright_object_rotation
        reorient_pose = reorient_object_pose @ torch.linalg.inv(tool_to_object)
        retreat_pose = place_pose.clone()
        retreat_pose[2, 3] += self.config.retreat_clearance

        self._poses = {
            "approach": approach_pose,
            "grasp": grasp_pose,
            "lift": lift_pose,
            "transport": transport_pose,
            "reorient": reorient_pose,
            "place": place_pose,
            "retreat": retreat_pose,
            "home": self._home_pose,
        }

    def get_action(self, env: Any, observation: object) -> torch.Tensor:
        """Expose the existing planner step through Arena's policy contract."""
        action = self.next_action()
        if action is None:
            raise RuntimeError("MedicinePickPlaceExpert produced no action for the current planner stage.")
        return action.unsqueeze(0) if action.ndim == 1 else action

    def next_action(self) -> torch.Tensor:
        """Advance the biomedical stage machine by one environment step."""

        return self._step_current_stage()

    def _step_current_stage(self) -> torch.Tensor:
        if self._stage in {"done", "failed"}:
            return self._action_for_target(self._held_arm_target(), gripper=0.0)
        skill = self._skill_registry.get(self._stage)
        if skill is None:
            raise RuntimeError(
                f"No atomic skill registered for stage {self._stage!r}; "
                f"configured pipeline={self._pipeline_names}"
            )
        context = AtomicSkillContext(
            env=self.env,
            stage=self._stage,
            target_name=self._target_name,
            config=self.config,
            runtime=self,
        )
        output = skill.execute(context)
        self._last_skill_output = output
        return output.action

    def _stage_gripper(self) -> float:
        return 0.0 if self._stage in {"approach", "grasp"} else 1.0

    def _select_authored_grasp_poses(
        self, object_position: torch.Tensor, object_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordered IK-feasible authored DROID grasp poses.

        The interaction annotations are object-local proposals, not fixed
        world poses.  Every reset transforms each retained middle-section
        candidate through the live bottle pose and the explicit annotation to
        DROID-gripper frame adapter before checking IK.
        """

        path = self.config.grasp_pose_path or str(_DEFAULT_GRASP_POSE_PATH)
        canonical, widths = _load_authored_grasp_data(path)
        canonical = canonical.to(device=object_position.device, dtype=object_position.dtype)
        widths = widths.to(device=object_position.device, dtype=object_position.dtype)
        width_compatible = widths <= self._gripper_max_width
        object_short_width = 2.0 * min(self._target_extents[self._target_index])
        width_compatible &= widths >= (
            object_short_width * self.config.minimum_grasp_width_ratio
        )
        width_compatible &= widths <= (
            object_short_width * self.config.maximum_grasp_width_ratio
        )
        canonical = canonical[width_compatible]
        widths = widths[width_compatible]
        if canonical.shape[0] == 0:
            raise RuntimeError(
                f"No authored grasp fits the DROID gripper width "
                f"({self._gripper_max_width:.3f} m)"
            )
        if canonical.shape[0] > self.config.max_grasp_candidates:
            indices = torch.linspace(
                0,
                canonical.shape[0] - 1,
                self.config.max_grasp_candidates,
                device=canonical.device,
            ).round().long()
            canonical = canonical[indices]
            widths = widths[indices]

        object_rotation = matrix_from_quat(object_quat.unsqueeze(0))[0]
        canonical_translation = (
            self._interaction_frame_rotation.unsqueeze(0)
            @ canonical[:, :3, 3].unsqueeze(-1)
        ).squeeze(-1)
        # Preserve the complete annotation asset, but only use poses located
        # at the bottle's middle circular cross-section.  The bottle is too
        # long for an end-cap grasp with this gripper.
        grasp_long_axis = object_position.new_tensor(self.config.grasp_long_axis)
        grasp_long_axis = grasp_long_axis / torch.linalg.vector_norm(grasp_long_axis)
        axial_offset = torch.sum(canonical_translation * grasp_long_axis, dim=-1)
        radial_offset = torch.linalg.vector_norm(
            canonical_translation - axial_offset.unsqueeze(-1) * grasp_long_axis,
            dim=-1,
        )
        centered = (
            (torch.abs(axial_offset) <= self.config.max_grasp_center_offset)
            & (radial_offset <= self.config.max_grasp_center_offset)
        )
        widths = widths[centered]
        canonical_translation = canonical_translation[centered]
        axial_offset = axial_offset[centered]
        radial_offset = radial_offset[centered]
        if axial_offset.shape[0] == 0:
            raise RuntimeError(
                f"No authored centered middle-section grasp fits {self._target_name}; "
                f"limit={self.config.max_grasp_center_offset:.3f} m"
            )
        object_long_axis = object_rotation @ grasp_long_axis
        object_long_axis = object_long_axis / torch.linalg.vector_norm(object_long_axis)
        approach_axis = object_position.new_tensor((0.0, 0.0, -1.0))
        closing_axis = torch.linalg.cross(object_long_axis, approach_axis)
        closing_axis = closing_axis / torch.linalg.vector_norm(closing_axis)
        grasp_frame = torch.stack((approach_axis, closing_axis, object_long_axis), dim=-1)
        grasp_rotation = grasp_frame.unsqueeze(0).repeat(axial_offset.shape[0], 1, 1)
        grasp_center = object_position.unsqueeze(0) + (
            object_long_axis.unsqueeze(0) * axial_offset.unsqueeze(-1)
        )
        base_to_center = self._base_to_grasp_center.to(dtype=object_position.dtype)
        grasp_position = grasp_center - (
            grasp_rotation @ base_to_center.unsqueeze(-1)
        ).squeeze(-1)

        current_state = self._planner._get_current_joint_state_for_curobo()
        current_position = current_state.position[0]
        # +X is the physical approach axis and points straight down.  Since
        # the frame is rebuilt above, no robot-specific upright filter is
        # needed here.
        approach_position = grasp_position + grasp_rotation @ object_position.new_tensor(
            (-self.config.approach_clearance, 0.0, 0.0)
        )
        center_offset = torch.linalg.vector_norm(
            grasp_center - object_position.unsqueeze(0), dim=-1
        )
        # The grasp must be centered on the bottle's middle circular section.
        # Opening width is a secondary criterion: prioritizing it first can
        # select a 4--5 cm axial-offset annotation even when a near-center
        # candidate is IK-feasible, producing a visible horizontal miss after
        # the bottle yaw is applied.
        candidate_order = torch.argsort(
            center_offset
            + 0.25 * torch.abs(widths - object_short_width)
        )
        feasible_indices: list[int] = []
        feasible_solutions: list[torch.Tensor] = []
        contact_failures = 0
        approach_failures = 0
        for candidate_tensor in candidate_order:
            candidate = int(candidate_tensor.item())
            contact_pose = torch.eye(
                4, device=object_position.device, dtype=object_position.dtype
            )
            contact_pose[:3, :3] = grasp_rotation[candidate]
            contact_pose[:3, 3] = grasp_position[candidate]
            contact_planner_pose = self._physical_to_planner_pose(contact_pose)
            contact_goal = self._planner._make_pose(
                position=contact_planner_pose[:3, 3],
                quaternion=PoseUtils.quat_from_matrix(contact_planner_pose[:3, :3]),
                normalize_rotation=True,
            )
            # Match the original validated collection contract: only the
            # contact pose disables the hand-link collision spheres.  The
            # approach pose must retain the hand geometry so the pre-grasp
            # path is checked against the carton and other scene geometry.
            self._planner._set_active_links(self._planner.config.hand_link_names, False)
            try:
                contact_result = self._planner.motion_gen.ik_solver.solve_single(
                    contact_goal, return_seeds=1
                )
            finally:
                self._planner._set_active_links(self._planner.config.hand_link_names, True)
            if not bool(contact_result.success.reshape(-1)[0].item()):
                contact_failures += 1
                continue
            approach_pose = contact_pose.clone()
            approach_pose[:3, 3] = approach_position[candidate]
            approach_planner_pose = self._physical_to_planner_pose(approach_pose)
            approach_goal = self._planner._make_pose(
                position=approach_planner_pose[:3, 3],
                quaternion=PoseUtils.quat_from_matrix(approach_planner_pose[:3, :3]),
                normalize_rotation=True,
            )
            approach_result = self._planner.motion_gen.ik_solver.solve_single(
                approach_goal, return_seeds=1
            )
            if not bool(approach_result.success.reshape(-1)[0].item()):
                approach_failures += 1
                continue
            feasible_indices.append(candidate)
            feasible_solutions.append(contact_result.solution.reshape(-1, contact_result.solution.shape[-1])[0])
            if len(feasible_indices) >= self.config.feasible_grasp_candidates:
                break
        if not feasible_indices:
            self._planner.logger.error(
                f"DROID grasp IK rejected all {grasp_position.shape[0]} candidates for "
                f"{self._target_name}: contact_failures={contact_failures}, "
                f"approach_failures={approach_failures}"
            )
            raise RuntimeError(
                f"No IK-feasible authored grasp pose for {self._target_name}; "
                f"candidates={grasp_position.shape[0]}"
            )
        feasible_index_tensor = torch.as_tensor(
            feasible_indices, device=grasp_position.device, dtype=torch.long
        )
        solutions = torch.stack(feasible_solutions)
        joint_distance = torch.linalg.vector_norm(
            solutions - current_position.unsqueeze(0), dim=-1
        )
        centrality = torch.linalg.vector_norm(
            grasp_center[feasible_index_tensor] - object_position.unsqueeze(0), dim=-1
        )
        width_error = torch.abs(widths[feasible_index_tensor] - object_short_width)
        # Asset annotations can contain grasps for larger grippers.  Pick the
        # contact width closest to this object's measured short diameter; use
        # arm travel only to break ties between geometrically equivalent poses.
        grasp_score = centrality + 0.25 * width_error + 1.0e-5 * joint_distance
        first_feasible = int(torch.argmin(grasp_score).item())
        remaining = [
            index for index in range(len(feasible_indices)) if index != first_feasible
        ]
        ordered_feasible = [first_feasible, *remaining]
        ordered_indices = [feasible_indices[index] for index in ordered_feasible]
        ordered_poses = torch.eye(
            4, device=object_position.device, dtype=object_position.dtype
        ).repeat(len(ordered_indices), 1, 1)
        ordered_poses[:, :3, :3] = grasp_rotation[ordered_indices]
        ordered_poses[:, :3, 3] = grasp_position[ordered_indices]
        ordered_centers = grasp_center[ordered_indices]
        tcp_offsets = ordered_centers - ordered_poses[:, :3, 3]
        tcp_length_error = torch.abs(
            torch.linalg.vector_norm(tcp_offsets, dim=-1)
            - torch.linalg.vector_norm(base_to_center)
        )
        if torch.any(tcp_length_error > 1.0e-5):
            raise RuntimeError(
                "Authored grasp composition violated the rigid DROID TCP length: "
                f"max_error={tcp_length_error.max().item():.6f} m"
            )
        selected = ordered_indices[0]
        selected_feasible = ordered_feasible[0]
        self._planner.logger.info(
            f"Selected {len(ordered_indices)} authored grasp candidates for "
            f"{self._target_name}; first={selected}/{grasp_position.shape[0]}, "
            f"approach_down={grasp_rotation[selected, 2, 0].item():.4f}, "
            f"center_offset={centrality[selected_feasible].item():.4f} m, "
            f"required_width={widths[selected].item():.4f} m, "
            f"joint_distance={joint_distance[selected_feasible].item():.4f}"
        )
        return ordered_poses, ordered_centers

    def _arm_reached_plan_target(self) -> bool:
        if self._last_joint_target is None:
            return False
        current = self._current_arm_target()
        return bool(
            torch.all(
                torch.abs(current - self._last_joint_target)
                <= self.config.joint_target_tolerance
            ).item()
        )

    def _grasp_verified(self) -> bool:
        """Require a stable, physically obstructed DROID finger position.

        A fully closed finger reached the empty-space stop. A valid primitive
        grasp therefore lies strictly between the open stop and that empty
        closure; the following lift stage verifies object motion separately.
        """

        robot = self.env.scene["robot"]
        gripper_ids, _ = robot.find_joints(["finger_joint"])
        finger_position = robot.data.joint_pos.torch[0, gripper_ids]
        return bool(
            torch.all(
                (finger_position >= self.config.grasp_closure_min)
                & (
                    finger_position
                    <= self._droid_gripper_closed_position
                    - self.config.empty_grasp_closed_margin
                )
            ).item()
        )

    def _mark_failed(self, reason: str) -> None:
        self._failed = True
        self._failure_reason = reason
        robot = self.env.scene["robot"]
        gripper_ids, _ = robot.find_joints(["finger_joint"])
        finger_position = float(robot.data.joint_pos.torch[0, gripper_ids][0].item())
        bottle_position = float(
            self.env.scene[self._target_name].data.root_pos_w.torch[0, 2].item()
        )
        self._planner.logger.error(
            f"Biomedical expert failed at stage={self._stage}: {reason}; "
            f"finger_joint={finger_position:.4f}, bottle_z={bottle_position:.4f}, "
            f"pre_lift_z={self._pre_lift_target_z}"
        )
        self._set_stage("failed")

    def _current_arm_target(self) -> torch.Tensor:
        return self.env.scene["robot"].data.joint_pos.torch[0, :7].detach().clone()

    def _held_arm_target(self) -> torch.Tensor:
        if self._last_joint_target is None:
            return self._current_arm_target()
        return self._last_joint_target.clone()

    def _next_planned_arm_target(self) -> torch.Tensor:
        if self._plan_positions is None or self._plan_index >= len(self._plan_positions):
            return self._current_arm_target()
        planned = self._plan_positions[self._plan_index]
        self._plan_index += 1
        return planned[list(self._plan_arm_indices)]

    def _action_for_target(self, target: torch.Tensor, gripper: float) -> torch.Tensor:
        action = torch.zeros(8, device=self.env.device, dtype=torch.float32)
        action[:7] = target
        action[7] = gripper
        return action

    def _plan_current_stage(self, expected_attached_object: str | None) -> None:
        self._stage_steps = 0
        if self._stage == "lift":
            self._plan_vertical_lift()
            return
        physical_pose = self._poses[self._stage]
        planner_pose = self._physical_to_planner_pose(physical_pose)
        # Keep contact-aware planning through the post-release disengagement
        # and return-home motion as well.  Without it, cuRobo can re-solve the
        # free-space retreat from a slightly stale post-release state, which
        # appears as a short reverse correction before the arm resumes home.
        self._planner.contact_planning = self._stage in {
            "grasp",
            "lift",
            "place",
            "release_lift",
            "home",
        }
        # Reorientation is an in-place wrist conversion after the bottle has
        # already been lifted.  It does not need the 20 mrad resampling used
        # for Cartesian approach and placement; retaining that density turns a
        # short, collision-safe cuRobo path into a long sequence of nearly
        # identical position targets.  A larger joint-space sample preserves
        # the planned path while allowing the position controller to execute
        # the rotation within the episode budget.
        if self._stage == "reorient":
            step_size = max(self.config.waypoint_step, 0.06)
        elif self._stage == "transport":
            step_size = self.config.transport_waypoint_step
        elif self._stage in {"home", "retreat"}:
            # Return motion is free-space motion after the bottle is already
            # released. Use the transport sampling density so the smooth
            # cuRobo trajectory is not stretched by unnecessary waypoints.
            step_size = self.config.transport_waypoint_step
        elif self._stage == "place":
            step_size = self.config.place_waypoint_step
        else:
            step_size = self.config.waypoint_step
        try:
            planned = self._planner.update_world_and_plan_motion(
                planner_pose,
                expected_attached_object=expected_attached_object,
                step_size=step_size,
                enable_retiming=True,
            )
        finally:
            self._planner.contact_planning = False
        if not planned or self._planner.current_plan is None:
            if self._plan_from_ik(planner_pose):
                return
            current_state = self._planner._get_current_joint_state_for_curobo()
            current_pose = self._planner.get_ee_pose(current_state)
            current_position = current_pose.position[0].detach().cpu().tolist()
            goal_position = planner_pose[:3, 3].detach().cpu().tolist()
            raise RuntimeError(
                f"cuRobo failed to plan biomedical stage '{self._stage}'; "
                f"current_curobo_ee={current_position}, curobo_goal={goal_position}"
            )
        self._plan_positions = self._planner.current_plan.position.detach().to(self.env.device)
        self._planner.logger.info(
            f"Biomedical stage plan: stage={self._stage}, "
            f"waypoints={len(self._plan_positions)}, "
            f"step_size={step_size:.3f}"
        )
        plan_names = tuple(self._planner.current_plan.joint_names)
        robot_names = tuple(self.env.scene["robot"].data.joint_names[:7])
        self._plan_arm_indices = tuple(plan_names.index(name) for name in robot_names)
        self._last_joint_target = self._plan_positions[-1, list(self._plan_arm_indices)].clone()
        self._plan_index = 0

    def _plan_from_ik(self, planner_pose: torch.Tensor) -> bool:
        """Use the validated bounded IK fallback when MotionGen rejects contact.

        A contact goal can be IK-feasible while cuRobo's collision trajectory
        optimizer rejects the final segment.  The original biomedical expert
        handled that case with a short joint-space segment from the live state
        to the same IK solution.  Keeping the fallback here preserves that
        execution contract without modifying IsaacLab-Mimic or cuRobo.
        """

        current_state = self._planner._get_current_joint_state_for_curobo()
        target_pose_cuda = self._planner._to_curobo_device(planner_pose)
        target_position, target_rotation = PoseUtils.unmake_pose(target_pose_cuda)
        target = self._planner._make_pose(
            position=target_position,
            quaternion=PoseUtils.quat_from_matrix(target_rotation),
        )
        ik_result = self._planner.motion_gen.ik_solver.solve_batch(
            target,
            seed_config=current_state.position.unsqueeze(0),
            return_seeds=1,
        )
        successful = ik_result.success.view(-1)
        if not bool(successful.any().item()):
            return False

        dof = self._planner.motion_gen.kinematics.dof
        goal_position = ik_result.solution[successful].reshape(-1, dof)[0]
        positions = torch.stack((current_state.position[0], goal_position), dim=0)
        plan = JointState.from_position(
            positions,
            joint_names=list(self._planner.motion_gen.kinematics.joint_names),
        )
        plan = self._planner.motion_gen.get_full_js(plan)
        plan = self._planner._linearly_retime_plan(
            step_size=(
                self.config.place_waypoint_step
                if self._stage == "place"
                else self.config.waypoint_step
            ),
            plan=plan,
        )
        self._planner._current_plan = plan
        self._plan_positions = plan.position.detach().to(self.env.device)
        plan_names = tuple(plan.joint_names)
        robot_names = tuple(self.env.scene["robot"].data.joint_names[:7])
        self._plan_arm_indices = tuple(plan_names.index(name) for name in robot_names)
        self._last_joint_target = self._plan_positions[-1, list(self._plan_arm_indices)].clone()
        self._plan_index = 0
        self._planner.logger.info(
            f"Biomedical stage fallback: stage={self._stage}, "
            f"waypoints={len(self._plan_positions)}, mode=ik_joint_segment"
        )
        return True

    def _plan_vertical_lift(self) -> None:
        """Build a monotonic Cartesian lift from the physically reached grasp.

        The lift uses the same MotionGen path as every other free-space stage.
        Its target is built from the current cuRobo FK pose, with only the
        physical Z coordinate increased.  This keeps the grasp frame aligned
        with the live joint state and avoids a separate endpoint-IK plus
        joint-interpolation path that can introduce a downward jump.
        """

        current_state = self._planner._get_current_joint_state_for_curobo()
        # The Isaac body pose and cuRobo FK can differ by a small amount while
        # the position controller is settling.  Using the Isaac pose as the
        # IK start therefore makes the first lift target inconsistent with the
        # joint state that cuRobo actually receives.  Build the physical pose
        # from that same live cuRobo state and the calibrated fixed frame.
        curobo_pose = self._planner.get_ee_pose(current_state)
        curobo_quat_wxyz = curobo_pose.quaternion[0].to(self.env.device)
        curobo_quat_xyzw = torch.cat(
            (curobo_quat_wxyz[1:], curobo_quat_wxyz[:1])
        )
        planner_pose = torch.eye(4, device=self.env.device, dtype=torch.float32)
        planner_pose[:3, :3] = matrix_from_quat(curobo_quat_xyzw.unsqueeze(0))[0]
        planner_pose[:3, 3] = curobo_pose.position[0].to(self.env.device)
        current_physical = planner_pose @ self._curobo_to_isaac_ee
        target_physical = self._poses["lift"]
        start_position = current_physical[:3, 3]
        # Keep the measured grasp X/Y exactly fixed. The lift target is a
        # vertical clearance operation, not a correction back to the authored
        # grasp position after the controller has already reached contact.
        target_position = start_position.clone()
        target_position[2] = target_physical[2, 3]
        if target_position[2] <= start_position[2]:
            raise RuntimeError(
                "Biomedical lift target is not above the physically reached grasp: "
                f"current_z={start_position[2].item():.4f}, "
                f"target_z={target_position[2].item():.4f}"
            )
        physical_goal = current_physical.clone()
        physical_goal[:3, 3] = target_position
        physical_goal[:3, :3] = current_physical[:3, :3]
        planner_goal = self._physical_to_planner_pose(physical_goal)

        self._planner.contact_planning = True
        try:
            planned = self._planner.update_world_and_plan_motion(
                planner_goal,
                expected_attached_object=self._target_name,
                step_size=self.config.waypoint_step,
                enable_retiming=True,
            )
        finally:
            self._planner.contact_planning = False
        if not planned or self._planner.current_plan is None:
            current_pose = self._planner.get_ee_pose(current_state)
            raise RuntimeError(
                "cuRobo failed to plan the biomedical vertical lift; "
                f"current_curobo_ee={current_pose.position[0].detach().cpu().tolist()}, "
                f"target_z={target_position[2].item():.4f}"
            )

        self._plan_positions = self._planner.current_plan.position.detach().to(self.env.device)
        plan_names = tuple(self._planner.current_plan.joint_names)
        robot_names = tuple(self.env.scene["robot"].data.joint_names[:7])
        self._plan_arm_indices = tuple(plan_names.index(name) for name in robot_names)
        self._last_joint_target = self._plan_positions[-1, list(self._plan_arm_indices)].clone()
        self._plan_index = 0
        self._planner.logger.info(
            "Biomedical stage plan: stage=lift, "
            f"waypoints={len(self._plan_positions)}, "
            f"step_size={self.config.waypoint_step:.3f}, mode=curobo_motion_gen"
        )

    def _plan_release_lift(self) -> None:
        """Lift the already-open gripper vertically off the placed bottle.

        After hold_release drops the bottle on the conveyor, the open gripper
        still surrounds its neck. Retreating laterally from that pose can press
        or drag the bottle, and the long free-space plan is slow. This applies
        a short pure-Z Cartesian lift from the actually-reached pose so the
        gripper clears the bottle before the lateral retreat is planned.
        """

        current_state = self._planner._get_current_joint_state_for_curobo()
        curobo_pose = self._planner.get_ee_pose(current_state)
        curobo_quat_wxyz = curobo_pose.quaternion[0].to(self.env.device)
        curobo_quat_xyzw = torch.cat((curobo_quat_wxyz[1:], curobo_quat_wxyz[:1]))
        planner_pose = torch.eye(4, device=self.env.device, dtype=torch.float32)
        planner_pose[:3, :3] = matrix_from_quat(curobo_quat_xyzw.unsqueeze(0))[0]
        planner_pose[:3, 3] = curobo_pose.position[0].to(self.env.device)
        current_physical = planner_pose @ self._curobo_to_isaac_ee
        current_position = current_physical[:3, 3]
        lift_delta = float(self.config.release_lift_height)
        target_physical = current_physical.clone()
        target_physical[:3, 3] = current_position.clone()
        target_physical[2, 3] = current_position[2] + lift_delta
        if target_physical[2, 3] <= current_position[2]:
            raise RuntimeError(
                "Biomedical release lift must rise above the reached place pose"
            )

        # The home pose is also part of the post-release safety envelope.  A
        # low home target can make cuRobo descend immediately after this lift,
        # putting the open gripper back into the conveyor/bottle travel zone
        # before the reset rotation is complete.  Keep the final home target
        # no lower than both the reached lift and the conservative top of a
        # medicine bottle on the conveyor.
        conveyor_center = self._to_robot_frame(
            self._conveyor_center, self._robot_position, self._robot_rotation
        )
        conveyor_surface_z = conveyor_center[2] + self._conveyor_size[2] * 0.5
        maximum_bottle_extent = max(
            max(float(extent) for extent in target_extents)
            for target_extents in self._target_extents
        )
        bottle_clearance_z = (
            conveyor_surface_z
            + 2.0 * maximum_bottle_extent
            + self.config.release_clearance
            + 0.03
        )
        bottle_clearance_z = bottle_clearance_z.to(
            device=target_physical.device, dtype=target_physical.dtype
        )
        safe_home_z = torch.maximum(
            target_physical[2, 3],
            bottle_clearance_z,
        )
        home_pose = self._poses["home"].clone()
        previous_home_z = float(home_pose[2, 3].item())
        home_pose[2, 3] = torch.maximum(home_pose[2, 3], safe_home_z)
        self._poses["home"] = home_pose
        # The release-lift target must reach the same clearance plane as the
        # subsequent home target.  Otherwise cuRobo correctly follows the
        # low lift target first, then has to descend/raise again while solving
        # the return-home rotation and translation.
        configured_lift_z = target_physical[2, 3].clone()
        target_physical[2, 3] = torch.maximum(
            target_physical[2, 3], home_pose[2, 3]
        )
        self._planner.logger.info(
            "Biomedical post-release home clearance: "
            f"lift_z={target_physical[2, 3].item():.4f}, "
            f"configured_lift_z={configured_lift_z.item():.4f}, "
            f"bottle_clearance_z={bottle_clearance_z:.4f}, "
            f"home_z={home_pose[2, 3].item():.4f}, "
            f"adjusted={home_pose[2, 3].item() > previous_home_z + 1.0e-6}"
        )
        planner_goal = self._physical_to_planner_pose(target_physical)
        self._planner.contact_planning = True
        try:
            planned = self._planner.update_world_and_plan_motion(
                planner_goal,
                expected_attached_object=None,
                step_size=self.config.transport_waypoint_step,
                enable_retiming=True,
            )
        finally:
            self._planner.contact_planning = False
        if not planned or self._planner.current_plan is None:
            raise RuntimeError(
                "cuRobo failed to plan the biomedical release lift; "
                f"current_curobo_ee={curobo_pose.position[0].detach().cpu().tolist()}, "
                f"target_z={target_physical[2, 3].item():.4f}"
            )
        self._plan_positions = self._planner.current_plan.position.detach().to(self.env.device)
        plan_names = tuple(self._planner.current_plan.joint_names)
        robot_names = tuple(self.env.scene["robot"].data.joint_names[:7])
        self._plan_arm_indices = tuple(plan_names.index(name) for name in robot_names)
        self._last_joint_target = self._plan_positions[-1, list(self._plan_arm_indices)].clone()
        self._plan_index = 0
        self._planner.logger.info(
            "Biomedical stage plan: stage=release_lift, "
            f"waypoints={len(self._plan_positions)}, "
            f"step_size={self.config.waypoint_step:.3f}, mode=curobo_motion_gen"
        )

    def _calibrate_ee_frames(self) -> torch.Tensor:
        """Return the fixed transform from cuRobo EE to Isaac's physical EE.

        The DROID URDF used by cuRobo and the flattened USD used by IsaacLab
        expose a link named ``base_link``, but their link frames are not
        coincident.  Calibrate that fixed URDF-to-USD transform from the same
        live joint state once, then use it for every IK and motion goal.
        """

        robot = self.env.scene["robot"]
        isaac_position, isaac_quat = world_pose_to_robot_frame(
            robot.data.body_pos_w.torch[0, self._ee_body_index, :3],
            robot.data.body_quat_w.torch[0, self._ee_body_index, :4],
            robot.data.root_pos_w.torch[0, :3],
            robot.data.root_quat_w.torch[0, :4],
        )
        isaac_pose = torch.eye(4, device=self.env.device, dtype=torch.float32)
        isaac_pose[:3, :3] = matrix_from_quat(isaac_quat.unsqueeze(0))[0]
        isaac_pose[:3, 3] = isaac_position

        current_state = self._planner._get_current_joint_state_for_curobo()
        curobo_pose = self._planner.get_ee_pose(current_state)
        curobo_quat_wxyz = curobo_pose.quaternion[0].to(self.env.device)
        curobo_quat_xyzw = torch.cat(
            (curobo_quat_wxyz[1:], curobo_quat_wxyz[:1])
        )
        planner_pose = torch.eye(4, device=self.env.device, dtype=torch.float32)
        planner_pose[:3, :3] = matrix_from_quat(curobo_quat_xyzw.unsqueeze(0))[0]
        planner_pose[:3, 3] = curobo_pose.position[0].to(self.env.device)

        curobo_to_isaac = torch.linalg.inv(planner_pose) @ isaac_pose
        reconstructed = planner_pose @ curobo_to_isaac
        if not torch.allclose(reconstructed, isaac_pose, atol=1.0e-5, rtol=0.0):
            raise RuntimeError("Failed to calibrate the DROID cuRobo-to-Isaac EE transform")
        return curobo_to_isaac

    def _physical_to_planner_pose(self, physical_pose: torch.Tensor) -> torch.Tensor:
        """Convert an Isaac USD ``base_link`` target to cuRobo's EE frame."""

        return physical_pose @ torch.linalg.inv(self._curobo_to_isaac_ee)

    def _current_physical_ee_pose(self) -> torch.Tensor:
        """Return Isaac's physical DROID EE pose in the robot-root frame."""

        robot = self.env.scene["robot"]
        root_position = robot.data.root_pos_w.torch[0, :3]
        root_rotation = matrix_from_quat(robot.data.root_quat_w.torch[0].unsqueeze(0))[0]
        ee_position = robot.data.body_pos_w.torch[0, self._ee_body_index, :3]
        ee_rotation = matrix_from_quat(
            robot.data.body_quat_w.torch[0, self._ee_body_index].unsqueeze(0)
        )[0]
        pose = torch.eye(4, device=self.env.device, dtype=torch.float32)
        pose[:3, :3] = root_rotation.transpose(0, 1) @ ee_rotation
        pose[:3, 3] = root_rotation.transpose(0, 1) @ (ee_position - root_position)
        return pose

    @staticmethod
    def _rotation_aligning(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        """Return the minimal rotation mapping one nonzero direction onto another."""

        source = source / torch.linalg.vector_norm(source)
        destination = destination / torch.linalg.vector_norm(destination)
        cross = torch.linalg.cross(source, destination)
        cosine = torch.clamp(torch.dot(source, destination), -1.0, 1.0)
        sine = torch.linalg.vector_norm(cross)
        identity = torch.eye(3, device=source.device, dtype=source.dtype)
        if sine <= 1e-6:
            if cosine >= 0.0:
                return identity
            basis = identity[torch.argmin(torch.abs(source))]
            axis = torch.linalg.cross(source, basis)
            axis = axis / torch.linalg.vector_norm(axis)
            return 2.0 * torch.outer(axis, axis) - identity
        skew = torch.zeros((3, 3), device=source.device, dtype=source.dtype)
        skew[0, 1] = -cross[2]
        skew[0, 2] = cross[1]
        skew[1, 0] = cross[2]
        skew[1, 2] = -cross[0]
        skew[2, 0] = -cross[1]
        skew[2, 1] = cross[0]
        return identity + skew + skew @ skew * ((1.0 - cosine) / sine.square())

    @staticmethod
    def _to_robot_frame(
        position: torch.Tensor,
        robot_position: torch.Tensor,
        robot_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """Convert an environment-local world position into robot-base coordinates."""

        return robot_rotation.transpose(0, 1) @ (position - robot_position)

    def _reachable_conveyor_position(self, robot_position: torch.Tensor) -> torch.Tensor:
        """Choose a drop point inside the success region at a healthy reach.

        The DROID arm faces +X toward the conveyor travel direction. Placing
        close to the arm-side edge (the default nearest-lower choice) forces
        the wrist to descend into its own workspace, so the forearm tips into
        the belt and the pose looks unnatural. Instead target a point roughly
        one arm's horizontal reach ahead of the base, clamped inside the valid
        success region so the bottle rests away from both edges.
        """

        lower = self._conveyor_bounds[0] + self._conveyor_region_margin
        upper = self._conveyor_bounds[1] - self._conveyor_region_margin
        if upper <= lower:
            raise RuntimeError("Biomedical conveyor has no valid placement region")
        if self._conveyor_axis >= robot_position.numel():
            raise RuntimeError("Biomedical conveyor axis exceeds the robot pose")
        robot_axis_position = float(robot_position[self._conveyor_axis].item())
        # Aim for a bottle centroid about this far ahead of the base along the
        # belt, giving the wrist a natural straight-down grasp without the
        # forearm colliding with the belt edge.  Clamp into the valid region.
        target = robot_axis_position + float(self.config.place_forward_distance)
        drop_axis = min(max(target, lower), upper)
        drop_position = self._conveyor_center.clone()
        drop_position[self._conveyor_axis] = drop_axis
        return drop_position

def build_medicine_pick_place(env: Any, success_term: Any, config: Any) -> Any:
    """Build the expert against IsaacLab's native environment contract.

    Arena returns the environment through Gymnasium's ``OrderEnforcing``
    wrapper.  The collection runner should keep that wrapper for ``reset`` /
    ``step`` semantics, while cuRobo needs the native ``scene``, ``sim`` and
    ``device`` attributes.  Normalize that boundary once here so every
    planner-backed skill receives the same environment type.
    """
    if isinstance(config, dict):
        config = MedicinePickPlaceConfig(**config)
    if not isinstance(config, MedicinePickPlaceConfig):
        raise TypeError(
            "medicine_pick_place expects a MedicinePickPlaceConfig or a mapping "
            f"of config values, got {type(config)!r}."
        )
    native_env = getattr(env, "unwrapped", env)
    return MedicinePickPlaceExpert(native_env, success_term, config)


__all__ = [
    "MedicinePickPlaceConfig",
    "MedicinePickPlaceExpert",
    "build_medicine_pick_place",
]
