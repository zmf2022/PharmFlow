"""PharmFlow - pharmaceutical task data collection based on Isaac Lab Arena.

The three vendored third-party packages (isaaclab_arena, isaaclab_arena_curobo,
isaaclab_mimic) live under ``third_party/`` and keep their top-level package
names so collection code can ``import isaaclab_arena`` unchanged.  Loading this
package prepends ``third_party/`` to ``sys.path`` so those vendored imports
resolve before any pip-installed copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vendored top-level third-party packages (keep original package names).
THIRD_PARTY_PATH = PROJECT_ROOT / "third_party"
if THIRD_PARTY_PATH.is_dir() and str(THIRD_PARTY_PATH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_PATH))

CONFIGS_PATH = PROJECT_ROOT / "pharm_flow" / "config"
ASSETS_PATH = PROJECT_ROOT / "assets"
