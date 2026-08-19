"""Arena factory for the configurable biomedical DROID Mimic task.

The YAML file describes the workcell layout and component assets.  Arena owns
the environment assembly contract; :mod:`biomedical_task` owns task semantics
and :mod:`droid_mimic` owns the robot-to-Mimic action contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_CONFIG = PROJECT_ROOT / "pharm_flow/config/scenes/biomedical.yaml"


def _resolve_path(value: str | os.PathLike[str]) -> str:
    root = os.environ.get("PHARM_FLOW_ROOT", str(PROJECT_ROOT))
    text = str(value).replace("${PHARM_FLOW_ROOT}", root)
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _tuple(values: Any, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if values is None:
        return default
    if len(values) != size:
        raise ValueError(f"Expected {size} values, got {values}")
    return tuple(float(value) for value in values)


_COSMOS_CAMERA_DATA_TYPES = [
    "rgb",
    "semantic_segmentation",
    "normals",
    "distance_to_image_plane",
]
_COSMOS_SEMANTIC_MAPPING = {
    "class:medicine_bottle": (220, 70, 70, 255),
    "class:medicine_carton": (180, 120, 70, 255),
    "class:conveyor": (70, 180, 220, 255),
    "class:table": (180, 150, 100, 255),
    "class:robot": (80, 220, 120, 255),
    "class:background": (130, 130, 130, 255),
    "class:UNLABELLED": (150, 150, 150, 255),
    "class:BACKGROUND": (200, 200, 200, 255),
}


def _cosmos_semantic_class(name: str) -> str:
    if name.startswith("medicine_bottle_"):
        return "medicine_bottle"
    if name == "medicine_carton_open":
        return "medicine_carton"
    if name.startswith("conveyor"):
        return "conveyor"
    if name == "workcell_table":
        return "table"
    if name == "background":
        return "background"
    return "background"


def _set_cosmos_semantic_class(asset: Any, semantic_class: str) -> None:
    object_cfg = getattr(asset, "object_cfg", None)
    spawn_cfg = getattr(object_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "semantic_tags"):
        raise RuntimeError(f"Asset {getattr(asset, 'name', asset)!r} has no semantic tag configuration")
    existing_tags = list(getattr(spawn_cfg, "semantic_tags", None) or [])
    spawn_cfg.semantic_tags = [tag for tag in existing_tags if tag[0] != "class"]
    spawn_cfg.semantic_tags.append(("class", semantic_class))


def _pose(spec: dict[str, Any]) -> Pose:
    from isaaclab_arena.utils.pose import Pose

    return Pose(
        position_xyz=_tuple(spec.get("position"), 3, (0.0, 0.0, 0.0)),
        rotation_xyzw=_tuple(spec.get("rotation"), 4, (0.0, 0.0, 0.0, 1.0)),
    )


def _component_object(component: dict[str, Any], index: int) -> Object:
    """Create one Arena Object from a YAML component declaration."""

    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType

    name = str(component["name"])
    kind = str(component.get("kind", "static")).lower()
    pose = _pose(component)
    collision_enabled = bool(component.get("collision_enabled", kind in {"dynamic", "rigid", "rigid_usd"}))
    has_mass = component.get("mass") is not None
    # A plain ``usd`` component is a static asset in the scene contract.  In
    # particular, the open carton USD contains collision geometry but no
    # rigid-body root; promoting it to ``RigidObject`` would make IsaacLab
    # enable contact sensors and fail during scene creation.  Components that
    # are meant to move must use an explicit rigid kind.
    is_rigid = kind in {"dynamic", "rigid", "rigid_usd", "medicine_box"}
    object_type = ObjectType.RIGID if is_rigid else ObjectType.BASE
    size = _tuple(component.get("size"), 3, (0.1, 0.1, 0.1))
    color = _tuple(component.get("color"), 3, (0.5, 0.5, 0.5))
    collision_cfg = CollisionPropertiesCfg(collision_enabled=collision_enabled)

    if kind == "conveyor_roller":
        spawner = sim_utils.CylinderCfg(
            radius=size[0],
            height=size[2],
            axis=str(component.get("axis", "Y")).upper(),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        return Object(name=name, object_type=ObjectType.BASE, spawner_cfg=spawner, initial_pose=pose)

    if kind in {"static", "conveyor_surface"}:
        spawner = sim_utils.CuboidCfg(
            size=size,
            collision_props=collision_cfg,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        return Object(name=name, object_type=ObjectType.BASE, spawner_cfg=spawner, initial_pose=pose)

    usd_path = component.get("usd_path")
    if not usd_path:
        raise ValueError(f"USD component '{name}' requires usd_path")
    spawn_cfg_addon: dict[str, Any] = {"collision_props": collision_cfg}
    if is_rigid:
        spawn_cfg_addon["rigid_props"] = RigidBodyPropertiesCfg(
            kinematic_enabled=bool(component.get("kinematic", False)),
            disable_gravity=bool(component.get("kinematic", False)),
        )
        if has_mass:
            spawn_cfg_addon["mass_props"] = MassPropertiesCfg(mass=float(component["mass"]))
    return Object(
        name=name,
        prim_path=f"{{ENV_REGEX_NS}}/EmbodiedScene/{name}",
        object_type=object_type,
        usd_path=_resolve_path(usd_path),
        scale=_tuple(component.get("scale"), 3, (1.0, 1.0, 1.0)),
        initial_pose=pose,
        spawn_cfg_addon=spawn_cfg_addon,
    )


def _load_scene(path: str) -> dict[str, Any]:
    scene_path = Path(_resolve_path(path))
    if not scene_path.is_file():
        raise FileNotFoundError(f"Biomedical scene config does not exist: {scene_path}")
    with scene_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return dict(payload.get("scene", payload))


@dataclass
class BiomedicalArenaEnvironmentCfg(ArenaEnvironmentCfg):
    """Select the robot, target, destination and YAML-described workcell."""

    scene_config: str = str(DEFAULT_SCENE_CONFIG)
    embodiment: str = "embodied_fusion_droid_mimic_ik"
    target_object: str = "medicine_bottle_00"
    destination_location: str = "conveyor_surface"
    include_background: bool = True
    cosmos_visual_observations: bool = False
    episode_length_s: float = 30.0
    language_instruction: str | None = None


@register_environment
class BiomedicalArenaEnvironment(ArenaEnvironmentFactory[BiomedicalArenaEnvironmentCfg]):
    """Registered Arena provider for the biomedical Mimic workcell."""

    name = "embodied_fusion_biomedical_droid_mimic"
    _legacy_argparse_cfg_type = BiomedicalArenaEnvironmentCfg

    def build(self, cfg: BiomedicalArenaEnvironmentCfg) -> IsaacLabArenaEnvironment:
        scene_spec = _load_scene(cfg.scene_config)
        import isaaclab.sim as sim_utils
        from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from pharm_flow.data_collection.arena.biomedical_task import BiomedicalPickMedicineTask
        from pharm_flow.data_collection.arena import droid_mimic as _droid_mimic  # noqa: F401

        components = [dict(component) for component in scene_spec.get("components", [])]
        component_by_name = {str(component["name"]): component for component in components}
        if cfg.target_object not in component_by_name:
            raise KeyError(f"Unknown biomedical target object: {cfg.target_object}")
        if cfg.destination_location not in component_by_name:
            raise KeyError(f"Unknown biomedical destination: {cfg.destination_location}")

        background_spec = scene_spec.get("background")
        background = None
        if cfg.include_background and background_spec is not None:
            background = _component_object({"name": "background", "kind": "usd", **background_spec}, 0)

        workcell = dict(scene_spec.get("workcell", {}))
        table = Object(
            name="workcell_table",
            prim_path="{ENV_REGEX_NS}/EmbodiedScene/workcell_table",
            object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.CuboidCfg(
                size=_tuple(workcell.get("table_size"), 3, (1.8, 1.2, 0.08)),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
            ),
            initial_pose=Pose(
                position_xyz=_tuple(workcell.get("table_position"), 3, (0.0, 0.0, 0.0)),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        scene_objects = [table]
        if background is not None:
            scene_objects.append(background)
        scene_objects.extend(_component_object(component, index) for index, component in enumerate(components, start=1))

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        if cfg.cosmos_visual_observations and not cfg.enable_cameras:
            raise ValueError("Cosmos visual observations require --enable_cameras")
        if cfg.cosmos_visual_observations:
            for asset in scene_objects:
                _set_cosmos_semantic_class(asset, _cosmos_semantic_class(asset.name))
            robot_spawn = embodiment.scene_config.robot.spawn
            robot_tags = list(getattr(robot_spawn, "semantic_tags", None) or [])
            robot_spawn.semantic_tags = [tag for tag in robot_tags if tag[0] != "class"]
            robot_spawn.semantic_tags.append(("class", "robot"))

            primary_camera = embodiment.camera_config.external_camera
            primary_camera.data_types = list(_COSMOS_CAMERA_DATA_TYPES)
            primary_camera.colorize_semantic_segmentation = True
            primary_camera.semantic_segmentation_mapping = dict(_COSMOS_SEMANTIC_MAPPING)
        if cfg.enable_cameras:
            camera_config = getattr(embodiment, "camera_config", None)
            if camera_config is None:
                raise RuntimeError(f"Embodiment {cfg.embodiment!r} has no camera configuration")
            for camera_name in camera_config.camera_names():
                camera_cfg = getattr(camera_config, camera_name)
                camera_cfg.height = 480
                camera_cfg.width = 640
        embodiment.set_initial_pose(
            Pose(
                position_xyz=_tuple(workcell.get("robot_position"), 3, (0.0, 0.0, 0.0)),
                rotation_xyzw=_tuple(workcell.get("robot_rotation"), 4, (0.0, 0.0, 0.0, 1.0)),
            )
        )

        task_spec = dict(scene_spec.get("task", {}))
        conveyor_spec = component_by_name[cfg.destination_location]
        conveyor_motion = conveyor_spec.get("motion", {})
        if isinstance(conveyor_motion, str):
            conveyor_motion = {"type": conveyor_motion}
        conveyor_axis = {"X": 0, "Y": 1}.get(str(conveyor_motion.get("axis", "X")).upper())
        if conveyor_axis is None:
            raise ValueError("Biomedical conveyor motion axis must be X or Y")
        conveyor_center = _tuple(conveyor_spec.get("position"), 3, (0.0, 0.0, 0.0))
        conveyor_size = _tuple(conveyor_spec.get("size"), 3, (1.0, 1.0, 0.1))
        bounds = conveyor_motion.get("bounds")
        if bounds is None:
            half_length = conveyor_size[conveyor_axis] / 2
            bounds = (
                conveyor_center[conveyor_axis] - half_length,
                conveyor_center[conveyor_axis] + half_length,
            )
        conveyor_bounds = _tuple(bounds, 2, (0.0, 1.0))
        conveyor_surface_z = conveyor_center[2] + conveyor_size[2] / 2
        target_spec = component_by_name[cfg.target_object]
        support_extents = _tuple(
            target_spec.get("support_extents"),
            3,
            (target_spec.get("size") or (0.1, 0.1, 0.1)),
        )
        target_support_height = float(support_extents[2])
        target_names = tuple(
            str(name) for name in task_spec.get("target_components", (cfg.target_object,))
        )
        target_objects = tuple(
            next(obj for obj in scene_objects if obj.name == name) for name in target_names
        )
        target_support_extents = tuple(
            _tuple(
                component_by_name[name].get("support_extents"),
                3,
                (component_by_name[name].get("size") or (0.1, 0.1, 0.1)),
            )
            for name in target_names
        )
        target_upright_axes = tuple(
            _tuple(component_by_name[name].get("upright_axis"), 3, (0.0, 0.0, 1.0))
            for name in target_names
        )
        randomization = dict(task_spec.get("target_randomization", {}))
        motion = dict(task_spec.get("auto_collection", {}))

        task = BiomedicalPickMedicineTask(
            target_object=next(obj for obj in scene_objects if obj.name == cfg.target_object),
            destination_location=next(obj for obj in scene_objects if obj.name == cfg.destination_location),
            background_scene=background or table,
            target_objects=target_objects,
            conveyor_center=conveyor_center,
            conveyor_size=conveyor_size,
            conveyor_axis=conveyor_axis,
            conveyor_bounds=conveyor_bounds,
            conveyor_surface_z=conveyor_surface_z,
            target_support_height=target_support_height,
            target_support_extents=target_support_extents,
            target_upright_axes=target_upright_axes,
            support_surface_z=float(randomization.get("support_surface_z", 0.0)),
            target_randomization=randomization,
            viewer_robot_position=_tuple(workcell.get("robot_position"), 3, (0.0, 0.0, 0.0)),
            region_margin=float(task_spec.get("conveyor_region_margin", 0.05)),
            surface_tolerance=float(conveyor_motion.get("surface_height_tolerance", 0.04)),
            episode_length_s=cfg.episode_length_s,
            task_description=cfg.language_instruction
            or "Pick up the medicine bottle, place it upright on the conveyor, and release it.",
        )
        # Compose task state into the DROID embodiment's native policy group.
        # Arena requires overlapping observation fields to have the same
        # dataclass type; extending the native group here preserves that
        # contract for collection, training, and playback.
        embodiment.observation_config.policy = task.make_policy_observation_cfg(
            embodiment.observation_config.policy
        )

        def _attach_scene_contract(env_cfg):
            # The controller is optional and local to the project.  It receives
            # the same YAML scene spec as the RLinf environment without making
            # the Arena config or third-party source depend on an environment
            # variable or a private global singleton.
            env_cfg.scene_motion_spec = scene_spec
            env_cfg.mimic_motion_cfg = motion
            # The biomedical collection contract is 30 Hz: 120 Hz physics
            # with four physics steps per environment/action step.  Keep the
            # camera render cadence aligned with that environment step so
            # every recorded observation has one fresh camera frame.
            env_cfg.sim.dt = 1 / 120
            env_cfg.decimation = 4
            env_cfg.sim.render_interval = env_cfg.decimation
            if hasattr(env_cfg, "num_rerenders_on_reset"):
                env_cfg.num_rerenders_on_reset = 1

            # Arena's manager config intentionally sets RTX ambient light to
            # zero and expects an environment to provide its own light.  The
            # old biomedical scene inherited this exact DomeLightCfg from
            # ObjectTableSceneCfg; add it at the project scene boundary so the
            # background USD keeps the same brightness after the migration.
            from isaaclab.assets import AssetBaseCfg

            env_cfg.scene.light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(
                    color=(0.75, 0.75, 0.75),
                    intensity=3000.0,
                ),
            )
            from pharm_flow.data_collection.arena.dataset_metadata import (
                configure_biomedical_dataset_metadata,
            )

            configure_biomedical_dataset_metadata(
                env_cfg,
                env_name=self.name,
                scene_config=cfg.scene_config,
                scene_spec=scene_spec,
                include_background=cfg.include_background,
                enable_cameras=cfg.enable_cameras,
                cosmos_visual_observations=cfg.cosmos_visual_observations,
                language_instruction=cfg.language_instruction,
                seed=getattr(env_cfg, "seed", None),
            )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=Scene(assets=scene_objects),
            embodiment=embodiment,
            task=task,
            env_cfg_callback=_attach_scene_contract,
        )


__all__ = ["BiomedicalArenaEnvironment", "BiomedicalArenaEnvironmentCfg"]
