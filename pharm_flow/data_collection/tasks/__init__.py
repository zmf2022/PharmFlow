"""Registered collection task definitions."""

from .base import CollectionTask
from .biomedical_droid import BiomedicalDroidTask

__all__ = ["BiomedicalDroidTask", "CollectionTask"]
