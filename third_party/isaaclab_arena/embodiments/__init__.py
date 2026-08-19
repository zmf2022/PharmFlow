# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# PharmFlow only uses the DROID embodiment (and its shared Franka dependency).
# Loading the other embodiments imported extra packages (e.g. isaaclab_arena_g1)
# that are not vendored / installed in PharmFlow, breaking collection startup.
from .droid.droid import *
from .franka.franka import *
