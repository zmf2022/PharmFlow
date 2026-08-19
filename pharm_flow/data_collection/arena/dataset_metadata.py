"""Shared dataset metadata contract for biomedical Arena environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    root = os.environ.get("PHARM_FLOW_ROOT", str(PROJECT_ROOT))
    text = str(value).replace("${PHARM_FLOW_ROOT}", root)
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def configure_biomedical_dataset_metadata(
    env_cfg: Any,
    *,
    env_name: str,
    scene_config: str | os.PathLike[str],
    scene_spec: dict[str, Any],
    include_background: bool,
    enable_cameras: bool,
    language_instruction: str | None,
    seed: int | None,
    cosmos_visual_observations: bool = False,
) -> None:
    """Attach the metadata contract consumed by IsaacLab's recorder.

    Both automatic collection and Mimic generation use the same Arena
    recorder.  Keeping this callback at the environment boundary ensures the
    exported HDF5 metadata is identical for both entry points.
    """

    task_description = (
        language_instruction
        or "Pick one medicine bottle from the open carton, grasp it securely with the DROID gripper, lift it clear of the carton, rotate it upright, place it within the valid region of the conveyor belt, release it, and return the arm to its home pose."
    )
    camera_observation_names = [
        "external_camera_rgb",
        "external_camera_2_rgb",
        "wrist_camera_rgb",
    ]
    if cosmos_visual_observations and enable_cameras:
        camera_observation_names.extend(
            [
                "external_camera_semantic_segmentation",
                "external_camera_normals",
                "external_camera_distance_to_image_plane",
            ]
        )

    static_metadata: dict[str, Any] = {
        "env_name": env_name,
        "type": 2,
        "task_name": "PickMedicineConveyor",
        "robot_name": "DROID",
        "scene_type": "biomedical",
        "scene_backend": "isaaclab",
        "task_backend": "isaaclab",
        "scene_config": str(_resolve_path(scene_config)),
        "scene_spec": scene_spec,
        "disable_background": not include_background,
        "task_description": task_description,
        "lang": task_description,
        "camera_observation_names": tuple(camera_observation_names) if enable_cameras else (),
        "policy_observation_names": (
            "eef_pos",
            "eef_quat",
            "joint_pos",
            "robot_joint_pos",
            "gripper_pos",
            "target_object_pos",
            "target_object_quat",
        ),
        "observation_coordinate_frame": "robot_root",
        "observation_contract_version": 2,
        "usd_simplify": False,
        "seed": seed,
    }
    background_path = (scene_spec.get("background") or {}).get("usd_path")
    if background_path:
        static_metadata["usd_path"] = str(_resolve_path(background_path))

    def get_ep_meta() -> dict[str, Any]:
        metadata = dict(static_metadata)
        if seed is None:
            metadata["seed"] = getattr(env_cfg, "seed", None)
        metadata["sim_args"] = {
            "dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "render_interval": int(env_cfg.sim.render_interval),
            "num_envs": int(env_cfg.scene.num_envs),
        }
        return metadata

    env_cfg.env_name = env_name
    env_cfg.get_ep_meta = get_ep_meta
