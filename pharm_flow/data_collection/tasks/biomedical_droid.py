"""Biomedical DROID collection task composed from IsaacLab Arena."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.assets.registries import EnvironmentRegistry

from ..arena import ensure_registered
from ..arena.biomedical_environment import (
    BiomedicalArenaEnvironmentCfg,
)
from ..arena.biomedical_task import BiomedicalPickMedicineTask
from ..arena.dataset_metadata import configure_biomedical_dataset_metadata
from ..experts.medicine_pick_place import (
    MedicinePickPlaceConfig,
    build_medicine_pick_place,
)
from ..utils.policy import TeleopPolicy
from ..utils.recording import DatasetExportMode
from .base import CollectionTask


def _load_scene_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return dict(payload.get("scene", payload))


class BiomedicalDroidTask(CollectionTask):
    """Compose the biomedical scene, Arena DROID embodiment and collection policy."""

    name = "biomedical_droid"

    def __init__(self, runtime: Any, dataset_file: Path):
        self.runtime = runtime
        self.dataset_file = dataset_file
        self.scene_spec = _load_scene_config(runtime.scene_config)
        task_spec = dict(self.scene_spec.get("task", {}))
        target_components = tuple(task_spec.get("target_components", ()))
        self.target_object = str(target_components[0]) if target_components else "medicine_bottle_00"
        self.destination_location = str(
            task_spec.get("conveyor_component", "conveyor_surface")
        )
        self.arena_environment = None
        self.arena_task: BiomedicalPickMedicineTask | None = None

    @property
    def _embodiment_name(self) -> str:
        # The planner emits absolute joint targets; keyboard teleoperation uses
        # Arena's differential-IK DROID embodiment.  Both inherit the same
        # official Arena DROID hardware and camera configuration.
        if self.runtime.controller == "auto":
            return "embodied_fusion_droid_mimic_absolute"
        return "embodied_fusion_droid_mimic_ik"

    def build_environment(self, runtime: Any) -> tuple[Any, Any, Any]:
        ensure_registered()
        task_spec = dict(self.scene_spec.get("task", {}))
        workcell = dict(self.scene_spec.get("workcell", {}))
        environment_cfg = BiomedicalArenaEnvironmentCfg(
            scene_config=str(runtime.scene_config.resolve()),
            embodiment=self._embodiment_name,
            target_object=self.target_object,
            destination_location=self.destination_location,
            include_background=not runtime.disable_background,
            episode_length_s=float(task_spec.get("episode_length_s", 30.0)),
            language_instruction=task_spec.get("language_instruction"),
            enable_cameras=bool(runtime.enable_cameras),
        )
        factory_type = EnvironmentRegistry().get_component_by_name(
            "embodied_fusion_biomedical_droid_mimic"
        )
        self.arena_environment = factory_type().build(environment_cfg)
        self.arena_task = self.arena_environment.task
        if self.arena_task is None:
            raise RuntimeError("Biomedical Arena environment did not provide a task")

        builder_cfg = ArenaEnvBuilderCfg(
            num_envs=1,
            env_spacing=float(self.scene_spec.get("environment_spacing", 30.0)),
            # A fixed seed made the first reset of every collection process
            # select the same medicine count.  Keep reproducibility available
            # through --seed, but make ordinary collection runs draw a fresh
            # reset sequence.
            seed=getattr(runtime, "seed", None),
            solve_relations=True,
            mimic=True,
            device=str(runtime.device),
            language_instruction=environment_cfg.language_instruction,
        )
        builder = ArenaEnvBuilder(self.arena_environment, builder_cfg)
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        # Collection owns the episode boundary.  The task success predicate is
        # still used by the keyboard callback and the scripted expert, but it
        # must not reset the IsaacLab environment at release time: the expert
        # has to execute its return-to-home motion before the recorder commits.
        env_cfg.terminations.success = None
        self._configure_dataset_recorder(env_cfg)
        self._configure_dataset_metadata(env_cfg, runtime, environment_cfg)
        env = builder.make_registered(env_cfg, env_kwargs)
        success_term = self.arena_task.get_termination_cfg().success
        return env, env_cfg, success_term

    def _configure_dataset_recorder(self, env_cfg: Any) -> None:
        """Attach Arena's Mimic recorder to IsaacLab's active recorder field.

        Arena keeps the Mimic recorder configuration in
        ``mimic_recorder_config`` while composing a Mimic environment, but
        IsaacLab's ``ManagerBasedEnv`` records through ``cfg.recorders``.
        Connect those two official configuration points here, before the
        environment is instantiated.
        """

        recorder_cfg = getattr(env_cfg, "mimic_recorder_config", None)
        if recorder_cfg is None:
            recorder_cfg = getattr(env_cfg, "demo_recorder_config", None)
        if recorder_cfg is None:
            raise RuntimeError(
                "The biomedical Mimic environment has no recorder configuration. "
                "Enable the Arena DROID camera/recorder embodiment before collection."
            )
        recorder_cfg.dataset_export_dir_path = str(self.dataset_file.parent)
        recorder_cfg.dataset_filename = self.dataset_file.stem
        recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        env_cfg.recorders = recorder_cfg

    def _configure_dataset_metadata(
        self, env_cfg: Any, runtime: Any, environment_cfg: BiomedicalArenaEnvironmentCfg
    ) -> None:
        """Provide the complete environment contract to IsaacLab Recorder.

        RecorderManager calls ``env.cfg.get_ep_meta()`` when an episode is
        exported.  Without this task-owned hook it writes only generic
        ``env_name/type/sim_args`` metadata, which is insufficient to rebuild
        the Arena environment for replay.
        """

        configure_biomedical_dataset_metadata(
            env_cfg,
            env_name=self.arena_environment.name,
            scene_config=runtime.scene_config,
            scene_spec=self.scene_spec,
            include_background=not runtime.disable_background,
            enable_cameras=bool(runtime.enable_cameras),
            language_instruction=environment_cfg.language_instruction,
            seed=getattr(runtime, "seed", None),
        )

    def create_teleop_device(self, env: Any, sensitivity: float) -> Any:
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        native_env = getattr(env, "unwrapped", env)
        return Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=0.2 * sensitivity,
                rot_sensitivity=0.5 * sensitivity,
                sim_device=native_env.device,
            )
        )

    def build_policy(
        self,
        runtime: Any,
        env: Any,
        success_term: Any,
        teleop_device: Any | None = None,
    ) -> Any:
        if runtime.controller == "keyboard":
            return TeleopPolicy(
                teleop_device or self.create_teleop_device(env, runtime.sensitivity)
            )

        planner_values = dict(self.scene_spec.get("task", {}).get("auto_collection", {}))
        if "pipeline" in planner_values:
            planner_values["pipeline"] = tuple(planner_values["pipeline"])
        planner_values["target_object_name"] = self.target_object
        return build_medicine_pick_place(
            env,
            success_term,
            MedicinePickPlaceConfig(**planner_values),
        )

    def success_callback(self, success_term: Any, policy: Any):
        if self.runtime.controller == "auto":
            return lambda env, *_: policy.target_succeeded()

        # The runner feeds the Gymnasium ``OrderEnforcing`` wrapper env here,
        # but ``success_term.func`` is a scene-reading contract (robot/object
        # poses) that needs the native ``scene`` attribute.  Normalize the
        # boundary identically to the planner and teleop device so the success
        # predicate receives the native environment.
        def _success(env, *_) -> Any:
            native_env = getattr(env, "unwrapped", env)
            return success_term.func(native_env, **success_term.params)[0]

        return _success

__all__ = ["BiomedicalDroidTask"]
