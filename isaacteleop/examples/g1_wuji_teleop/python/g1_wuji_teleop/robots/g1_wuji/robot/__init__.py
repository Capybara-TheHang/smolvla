# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot definitions used by the G1-Wuji teleop example."""

from .articulation import make_g1_wuji_robot_cfg
from .profiles import (
    available_robot_variants,
    normalize_robot_variant,
    robot_profile_from_config,
    with_robot_variant_override,
)
from .types import RobotProfile

__all__ = [
    "RobotProfile",
    "available_robot_variants",
    "make_g1_wuji_robot_cfg",
    "normalize_robot_variant",
    "robot_profile_from_config",
    "with_robot_variant_override",
]
