#!/usr/bin/env python3
"""Automatic collection CLI for registered data-collection tasks.

The saved HDF5 actions are the biomedical environment's native eight-dimensional
DROID contract: seven arm targets followed by the binary gripper command
(0=open, 1=close).  The default keyboard mode is retained; ``--controller auto``
uses the IsaacLab Mimic cuRobo planner without human input.

The task adapter supplies environment and expert configuration.  Episode
lifecycle, success latching, and recorder commits are implemented once by
:mod:`pharm_flow.data_collection.utils.collection`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys
import traceback
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load vendored top-level third-party packages (isaaclab_arena, isaaclab_mimic,
# isaaclab_arena_curobo) from third_party/ before importing them.
# ``import pharm_flow`` injects third_party/ into sys.path automatically.
import pharm_flow  # noqa: E402,F401


def _parse_args() -> argparse.Namespace:
    """Parse collection and IsaacLab launcher arguments."""

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Registered collection task name; defaults to the scene YAML binding.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "pharm_flow_logs" / "datasets" / "biomedical_droid",
        help="Output directory. A timestamped HDF5 file is created for each run.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Optional explicit HDF5 path kept for compatibility; existing files are never overwritten.",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=0,
        help="Stop after this many successful demonstrations; 0 records until Ctrl+C.",
    )
    parser.add_argument(
        "--controller",
        choices=("keyboard", "auto"),
        default="keyboard",
        help="Action source: keyboard teleoperation or the scripted pick/place expert.",
    )
    parser.add_argument(
        "--step-hz",
        type=int,
        default=30,
        help="Keyboard/environment action rate in Hz. Defaults to 30 Hz.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Multiplier for the official keyboard position and rotation increments.",
    )
    parser.add_argument(
        "--success-hold-steps",
        type=int,
        default=1,
        help="Consecutive success states required before exporting an episode.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=650,
        help="Discard and reset an automatic episode after this many steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional environment seed; omit it to randomize the initial medicine count each run.",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "pharm_flow" / "config" / "scenes" / "biomedical.yaml",
        help="Biomedical scene YAML.",
    )
    parser.add_argument(
        "--disable-background",
        action="store_true",
        help="Load only the workcell instead of the outpatient-clinic background.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless and args.controller == "keyboard":
        parser.error("Keyboard collection requires the IsaacLab GUI; do not pass --headless.")
    if (
        args.step_hz <= 0
        or args.success_hold_steps <= 0
        or args.max_episode_steps <= 0
        or args.num_demos < 0
    ):
        parser.error("step, hold, and max episode limits must be positive; --num-demos cannot be negative.")
    if not args.scene_config.is_file():
        parser.error(f"Scene configuration does not exist: {args.scene_config}")
    # Preserve IsaacLab's explicit ``--visualizer none`` contract.  Only the omitted
    # option gets the keyboard GUI default; AppLauncher handles headless mode.
    if args.headless:
        args.visualizer = "none"
    elif args.visualizer is None and not getattr(args, "visualizer_explicit", False):
        args.visualizer = "kit"
    args.enable_cameras = True
    return args


def _create_keyboard_help(env):
    import omni.ui as ui

    window = ui.Window("Keyboard Controls", width=400, height=300, visible=True)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label("Biomedical DROID keyboard controls", style={"font_size": 18})
            ui.Label("W / S   move X forward / backward")
            ui.Label("A / D   move Y left / right")
            ui.Label("Q / E   move Z up / down")
            ui.Label("Z / X   roll + / -")
            ui.Label("T / G   pitch + / -")
            ui.Label("C / V   yaw + / -")
            ui.Label("K       open / close gripper")
            ui.Label("R       discard episode and reset")
    return window


class WristCameraPreview:
    """Display the first environment's wrist-camera RGB stream in Kit UI."""

    def __init__(self, env):
        import omni.ui as ui

        self._camera = env.scene["wrist_camera"]
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window("Wrist Camera Preview", width=640, height=480, visible=True)
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label("DROID wrist camera", style={"font_size": 18})
                ui.ImageWithProvider(self._provider, width=640, height=480)

    def update(self) -> None:
        import numpy as np

        rgb = self._camera.data.output.get("rgb")
        if rgb is None:
            return
        if hasattr(rgb, "detach"):
            rgb = rgb[0].detach().to(device="cpu").numpy()
        else:
            rgb = np.asarray(rgb[0])
        # The wrist camera is mounted upside down on the gripper. Rotate only
        # the preview image; policy observations and recorded data stay native.
        rgb = np.rot90(rgb, 2).copy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[-1] == 3:
            rgb = np.concatenate(
                (rgb, np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)), axis=-1
            )
        rgb = np.ascontiguousarray(rgb)
        self._provider.set_bytes_data(rgb.flatten().data, [rgb.shape[1], rgb.shape[0]])

    def close(self) -> None:
        self._window.visible = False
        self._window.destroy()


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    # Keep the launcher alive for the entire session.  Its Kit callbacks own
    # the GUI lifecycle, as in IsaacLab's official record_demos.py entrypoint.
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    env = None
    try:
        os.environ["PHARM_FLOW_ROOT"] = str(PROJECT_ROOT)
        os.environ["PHARM_FLOW_SCENE_CONFIG"] = str(args.scene_config.resolve())
        if args.disable_background:
            os.environ["PHARM_FLOW_DISABLE_BACKGROUND"] = "1"
        else:
            os.environ.pop("PHARM_FLOW_DISABLE_BACKGROUND", None)

        from isaaclab_arena.utils.rate_limiter import RateLimiter

        from pharm_flow.data_collection.utils.collection import (
            CollectionRunner,
            CollectionRunnerCfg,
        )
        from pharm_flow.data_collection.tasks.biomedical_droid import BiomedicalDroidTask

        if args.dataset_file is not None:
            dataset_file = args.dataset_file.expanduser().resolve()
            if dataset_file.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing dataset: {dataset_file}"
                )
        else:
            dataset_dir = args.dataset_dir.expanduser().resolve()
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dataset_file = dataset_dir / f"teleop-{timestamp}.hdf5"
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        scene_payload = yaml.safe_load(args.scene_config.read_text(encoding="utf-8")) or {}
        scene_payload = dict(scene_payload.get("scene", scene_payload))
        task_name = str(
            args.task
            or scene_payload.get("collection", {}).get("task", "biomedical_droid")
        )
        if task_name != BiomedicalDroidTask.name:
            raise KeyError(
                f"Unknown collection task {task_name!r}; "
                f"available tasks: {BiomedicalDroidTask.name}"
            )
        task = BiomedicalDroidTask(args, dataset_file)
        env, env_cfg, success_term = task.build_environment(args)
        help_window = (
            _create_keyboard_help(env)
            if args.controller == "keyboard" and not args.headless
            else None
        )
        keyboard = None
        if args.controller == "keyboard":
            keyboard = task.create_teleop_device(env, args.sensitivity)
        reset_requested = False

        def request_reset() -> None:
            nonlocal reset_requested
            reset_requested = True

        if keyboard is not None:
            keyboard.add_callback("R", request_reset)
        limiter = RateLimiter(period_seconds=1.0 / args.step_hz)
        # Automatic collection still records the sensor stream into HDF5, but
        # does not open an interactive preview window.  The preview is only a
        # keyboard-teleoperation aid and otherwise adds an unnecessary Kit UI
        # workload during scripted collection.
        # WristCameraPreview reads the scene sensor, but ``env`` is Arena's
        # Gymnasium ``OrderEnforcing`` wrapper, which only exposes Gymnasium
        # reset/step semantics.  Normalize the boundary identically to the
        # planner and teleop device (…/medicine_pick_place.py, …/create_teleop_device)
        # so the preview receives the native scene/sim/device attributes.
        native_env = getattr(env, "unwrapped", env)
        camera_preview = (
            WristCameraPreview(native_env)
            if args.controller == "keyboard" and not args.headless
            else None
        )

        if args.controller == "auto":
            # The expert calibrates its planner and home pose from live robot
            # state. Reset the environment before constructing it, then let the
            # runner consume this observation instead of resetting a second
            # time and invalidating that calibration boundary.
            initial_observation = env.reset()
            expert = task.build_policy(args, env, success_term)
            # Return control to Kit once after cuRobo's synchronous
            # construction/warm-up before the first CUDA planning call.
            app.update()
            print("Automatic collection started. Pick/place stages are controlled by the scripted expert.")
            print(f"Successful demonstrations will be appended to: {dataset_file}")

            def update_auto_frame(_obs) -> None:
                # Keep the scripted expert at the same fixed action cadence as
                # the original collection loop.  Without this wait, the
                # synchronous planner and Kit rendering determine the cadence
                # independently, which makes joint-position targets arrive
                # with visible timing jitter.
                limiter.sleep(wait_callback=app.update)

            runner = CollectionRunner(
                env,
                expert,
                CollectionRunnerCfg(
                    max_steps_per_episode=args.max_episode_steps,
                    success_hold_steps=args.success_hold_steps,
                ),
                success_fn=task.success_callback(success_term, expert),
                should_continue=app.is_running,
                on_step=update_auto_frame,
                on_idle=lambda: (env.sim.render(), app.update()),
                on_reset=app.update,
                initial_observation=initial_observation,
            )
            runner.run(args.num_demos or None)
            return

        if keyboard is None:
            raise RuntimeError("Keyboard policy was not initialized for keyboard collection.")

        print("Keyboard collection started. R: discard/reset episode; K: toggle gripper; Ctrl+C: finish.")
        print(f"Successful demonstrations will be appended to: {dataset_file}")

        def consume_reset_request() -> bool:
            nonlocal reset_requested
            if not reset_requested:
                return False
            reset_requested = False
            print("Episode discarded and reset.")
            return True

        def update_keyboard_frame(_obs) -> None:
            if camera_preview is not None:
                camera_preview.update()
            app.update()
            limiter.sleep(wait_callback=idle_keyboard_frame)

        def idle_keyboard_frame() -> None:
            env.sim.render()
            if camera_preview is not None:
                camera_preview.update()
            app.update()

        runner = CollectionRunner(
            env,
            task.build_policy(args, env, success_term, teleop_device=keyboard),
            CollectionRunnerCfg(
                max_steps_per_episode=args.max_episode_steps,
                success_hold_steps=args.success_hold_steps,
            ),
            success_fn=task.success_callback(success_term, keyboard),
            should_continue=app.is_running,
            discard_requested=consume_reset_request,
            on_step=update_keyboard_frame,
            on_idle=idle_keyboard_frame,
            on_reset=app.update,
        )
        runner.run(args.num_demos or None)
    except KeyboardInterrupt:
        print("Keyboard collection stopped.")
    except BaseException:
        # app.close() below terminates the process at the C level before
        # Python can print an uncaught exception's traceback. Print it first
        # so a failure is visible instead of exiting silently with code 0.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if "camera_preview" in locals() and camera_preview is not None:
            camera_preview.close()
        if "help_window" in locals() and help_window is not None:
            help_window.visible = False
            help_window.destroy()
        app.close()


if __name__ == "__main__":
    main()
