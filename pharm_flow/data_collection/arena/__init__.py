"""Arena task, embodiment, and environment adapters for data collection.

The package intentionally has no eager Isaac Sim imports.  Import the concrete
environment module after ``SimulationApp``/``AppLauncher`` has started Kit,
matching Arena's official environment modules.
"""

_REGISTERED = False


def ensure_registered() -> None:
    """Load project Arena registrations after Isaac Sim has started."""

    global _REGISTERED
    if _REGISTERED:
        return
    # Import order is intentional: the embodiment is needed while the
    # environment factory builds its scene, and the task is needed by that
    # factory as well.  Arena's registries remain the only registry layer.
    from . import droid_mimic as _droid_mimic  # noqa: F401
    from . import biomedical_task as _biomedical_task  # noqa: F401
    from . import biomedical_environment as _biomedical_environment  # noqa: F401

    _REGISTERED = True


__all__ = ["ensure_registered"]
