"""Project-side DROID adapter for the official IsaacLab Mimic contract.

Arena already owns the DROID hardware description and action/camera configs.
The missing piece is the ``ManagerBasedRLMimicEnv`` adapter, so this module
only supplies that adapter and registers a project-local DROID embodiment.  It
does not modify the Arena submodule or duplicate the DROID action semantics.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import warp as wp
import isaaclab.utils.math as pose_utils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.droid.droid import (
    DroidAbsoluteJointPositionEmbodiment,
    DroidDifferentialIKEmbodiment,
)

from pharm_flow.envs.scene_motion import SceneMotionController


def _eef_pose_in_robot_frame(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the DROID gripper pose in the robot root frame.

    Arena's generic DROID observation is expressed in world coordinates.  That
    is valid for a single environment, but makes a policy trained with
    ``num_envs > 1`` learn the clone placement instead of the task geometry.
    Biomedical collection and playback use the robot-local frame, matching
    the target-object observations.
    """

    robot = env.scene[asset_cfg.name]
    body_index = robot.data.body_names.index("base_link")
    return subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        wp.to_torch(robot.data.body_pos_w)[:, body_index, :],
        wp.to_torch(robot.data.body_quat_w)[:, body_index, :],
    )


def _eef_pos_in_robot_frame(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _eef_pose_in_robot_frame(env, asset_cfg)[0]


def _eef_quat_in_robot_frame(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _eef_pose_in_robot_frame(env, asset_cfg)[1]


class EmbodiedFusionDroidMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic pose/action conversion for the Arena DROID differential-IK mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scene_spec = getattr(self.cfg, "scene_motion_spec", None)
        self._motion_controller = (
            SceneMotionController(self, scene_spec) if scene_spec is not None else None
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._motion_controller is not None:
            self._motion_controller.reset(env_ids)

    def step(self, action: torch.Tensor):
        if self._motion_controller is not None:
            self._motion_controller.advance()
        return super().step(action)

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        # Mimic requires this pose to be in the same frame as the controller.
        eef_pos, eef_quat = _eef_pose_in_robot_frame(self)
        eef_pos = eef_pos[env_ids]
        eef_quat = eef_quat[env_ids]
        return pose_utils.make_pose(eef_pos, pose_utils.matrix_from_quat(eef_quat))

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Return Mimic object poses in the DROID robot-root frame.

        The differential-IK controller and ``get_robot_eef_pose`` both use the
        articulation root frame.  Converting objects here keeps the official
        Mimic object-relative transform contract in that same frame.
        """

        object_poses = super().get_object_poses(env_ids=env_ids)
        selected_env_ids = slice(None) if env_ids is None else env_ids
        robot = self.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[selected_env_ids]
        root_quat = robot.data.root_quat_w.torch[selected_env_ids]
        env_origins = self.scene.env_origins[selected_env_ids]
        for object_name, object_pose in object_poses.items():
            object_pos_env, object_rot_env = pose_utils.unmake_pose(object_pose)
            object_pos_root, object_quat_root = subtract_frame_transforms(
                root_pos,
                root_quat,
                object_pos_env + env_origins,
                pose_utils.quat_from_matrix(object_rot_env),
            )
            object_poses[object_name] = pose_utils.make_pose(
                object_pos_root, pose_utils.matrix_from_quat(object_quat_root)
            )
        return object_poses

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = next(iter(self.cfg.subtask_configs))
        target_eef_pose = next(iter(target_eef_pose_dict.values()))
        target_pos_root, target_rot_root = pose_utils.unmake_pose(target_eef_pose)

        current_pos, current_quat = _eef_pose_in_robot_frame(self)
        current_pos = current_pos[env_id]
        current_rot = pose_utils.matrix_from_quat(current_quat[env_id])
        delta_position = target_pos_root - current_pos
        delta_rot_mat = target_rot_root.matmul(current_rot.transpose(-1, -2))
        delta_rotation = pose_utils.axis_angle_from_quat(
            pose_utils.quat_from_matrix(delta_rot_mat)
        )

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise_scale = float(action_noise_dict.get(eef_name, 0.0))
            pose_action = torch.clamp(
                pose_action + noise_scale * torch.randn_like(pose_action), -1.0, 1.0
            )

        gripper_action = next(iter(gripper_action_dict.values())).reshape(-1)
        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = next(iter(self.cfg.subtask_configs))
        arm_action = self.action_manager.get_term("arm_action")
        if isinstance(arm_action, JointPositionAction):
            # Absolute joint-position embodiment: the action is the commanded
            # joint positions, so forward-kinematics them to the base_link pose.
            target_pose = self._fk_base_link_pose(action)
        else:
            # Differential-IK embodiment: the action is a body-frame delta pose.
            target_pose = self._ik_delta_pose(action)
        return {eef_name: target_pose}

    def _ik_delta_pose(self, action: torch.Tensor) -> torch.Tensor:
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]
        current_pos, current_quat = _eef_pose_in_robot_frame(self)
        current_rot = pose_utils.matrix_from_quat(current_quat)

        target_pos = current_pos + delta_position
        delta_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_axis = delta_rotation / delta_angle.clamp_min(torch.finfo(delta_rotation.dtype).eps)
        near_zero = delta_angle.squeeze(-1) < 1e-6
        delta_axis = torch.where(near_zero.unsqueeze(-1), torch.zeros_like(delta_axis), delta_axis)
        delta_quat = pose_utils.quat_from_angle_axis(delta_angle.squeeze(-1), delta_axis)
        target_rot = pose_utils.matrix_from_quat(delta_quat).matmul(current_rot)
        return pose_utils.make_pose(target_pos, target_rot).clone()

    def _fk_base_link_pose(self, action: torch.Tensor) -> torch.Tensor:
        robot = self.scene["robot"]
        try:
            body_index = robot.data.body_names.index("base_link")
        except ValueError as exc:
            raise ValueError(
                "The DROID Mimic adapter requires the Robotiq base_link body; "
                f"available bodies: {robot.data.body_names}"
            ) from exc
        # The absolute embodiment commands the panda joints followed by the
        # finger.  Only the arm joints influence the base_link pose.
        arm_action = self.action_manager.get_term("arm_action")
        arm_joint_ids = arm_action._joint_ids
        num_arm = len(arm_joint_ids)
        commanded_pos = action[:, :num_arm]
        env_ids = torch.arange(self.num_envs, device=self.device)

        saved_pos = robot.data.joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(
            position=commanded_pos, joint_ids=arm_joint_ids, env_ids=env_ids
        )
        try:
            robot.data._physics_sim_view.update_articulations_kinematic()
            link_pose = wp.to_torch(robot.data._root_view.get_link_transforms())
            eef_pos, eef_quat = subtract_frame_transforms(
                robot.data.root_pos_w.torch[env_ids],
                robot.data.root_quat_w.torch[env_ids],
                link_pose[env_ids, body_index, :3],
                link_pose[env_ids, body_index, 3:],
            )
            return pose_utils.make_pose(eef_pos, pose_utils.matrix_from_quat(eef_quat)).clone()
        finally:
            robot.write_joint_position_to_sim_index(position=saved_pos, env_ids=env_ids)

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = next(iter(self.cfg.subtask_configs))
        return {eef_name: actions[..., -1:]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Expose Arena/IsaacLab Mimic subtask annotations from task observations."""

        if env_ids is None:
            env_ids = slice(None)
        terms = self.obs_buf["subtask_terms"]
        signals: dict[str, torch.Tensor] = {}
        for name in ("grasp", "place"):
            if name in terms:
                signals[name] = terms[name][env_ids]
        return signals


@register_asset
class EmbodiedFusionDroidMimicIKEmbodiment(DroidDifferentialIKEmbodiment):
    """Arena DROID IK embodiment with the project Mimic environment adapter."""

    name = "embodied_fusion_droid_mimic_ik"
    tags = ["embodiment", "mimic", "droid"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The Arena DROID IK controller tracks the fixed arm base (panda_link0),
        # but the project Mimic adapter computes target deltas relative to the
        # Robotiq base_link (the moving gripper frame).  Track base_link so the
        # IK control loop and the annotation frame are consistent.
        self.action_config.arm_action.body_name = "base_link"
        self.action_config.arm_action.body_offset = None
        self.observation_config.policy.eef_pos.func = _eef_pos_in_robot_frame
        self.observation_config.policy.eef_quat.func = _eef_quat_in_robot_frame
        self.mimic_env = EmbodiedFusionDroidMimicEnv


@register_asset
class EmbodiedFusionDroidMimicAbsoluteJointEmbodiment(DroidAbsoluteJointPositionEmbodiment):
    """Arena DROID embodiment for scripted joint-position collection."""

    name = "embodied_fusion_droid_mimic_absolute"
    tags = ["embodiment", "mimic", "droid", "absolute_joint_position"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_config.policy.eef_pos.func = _eef_pos_in_robot_frame
        self.observation_config.policy.eef_quat.func = _eef_quat_in_robot_frame
        self.mimic_env = EmbodiedFusionDroidMimicEnv


__all__ = [
    "EmbodiedFusionDroidMimicEnv",
    "EmbodiedFusionDroidMimicIKEmbodiment",
    "EmbodiedFusionDroidMimicAbsoluteJointEmbodiment",
]
