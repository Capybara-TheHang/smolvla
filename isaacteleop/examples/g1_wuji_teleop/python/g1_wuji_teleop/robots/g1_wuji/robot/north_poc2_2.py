# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""North POC2.2 robot profile and joint definitions."""

from __future__ import annotations

from collections.abc import Sequence

from .g1_wuji import FIXED_WUJI_HAND_CONFIG_DIR
from .types import RobotProfile

NORTH_POC2_2_ARM_JOINTS = {
    "left": tuple(f"left_arm_joint_{index}" for index in range(1, 8)),
    "right": tuple(f"right_arm_joint_{index}" for index in range(1, 8)),
}
NORTH_POC2_2_LEFT_HAND_JOINTS = (
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
NORTH_POC2_2_RIGHT_HAND_JOINTS = tuple(
    name.replace("left_", "right_", 1) for name in NORTH_POC2_2_LEFT_HAND_JOINTS
)
NORTH_POC2_2_INITIAL_WORLD_POSITION = (0.0, 0.0, 0.8)
NORTH_POC2_2_INITIAL_WORLD_ORIENTATION_XYZW = (0.0, 0.70710678, 0.70710678, 0.0)
NORTH_POC2_2_ACTUATOR_JOINT_NAMES_EXPR = {
    "base": ("lower_body_joint_.*", "neck_joint_.*"),
    "right_arm": ("right_arm_joint_.*",),
    "left_arm": ("left_arm_joint_.*",),
    "wrist": (),
    "fingers": (
        "left_thumb_.*",
        "left_index_.*",
        "left_middle_.*",
        "left_ring_.*",
        "left_pinky_.*",
        "right_thumb_.*",
        "right_index_.*",
        "right_middle_.*",
        "right_ring_.*",
        "right_pinky_.*",
    ),
}

NORTH_POC2_2_ARM_INITIAL_JOINT_POSITIONS = {
    "left_arm_joint_1": 0.0,
    "left_arm_joint_2": 0.0,
    "left_arm_joint_3": 1.57,
    "left_arm_joint_4": 1.57,
    "left_arm_joint_5": -1.57,
    "left_arm_joint_6": 0.0,
    "left_arm_joint_7": 0.0,
    "right_arm_joint_1": 0.0,
    "right_arm_joint_2": 0.0,
    "right_arm_joint_3": 1.57,
    "right_arm_joint_4": 1.57,
    "right_arm_joint_5": -1.57,
    "right_arm_joint_6": 0.0,
    "right_arm_joint_7": 0.0,
}


def _zero_joint_positions(joint_names: Sequence[str]) -> dict[str, float]:
    return {name: 0.0 for name in joint_names}


def _north_poc2_2_joint_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for side in ("left", "right"):
        pairs = (
            ("thumb_CMC_FE", "finger1_joint1"),
            ("thumb_CMC_AA", "finger1_joint2"),
            ("thumb_MCP_FE", "finger1_joint3"),
            ("thumb_IP", "finger1_joint4"),
            ("index_MCP_FE", "finger2_joint1"),
            ("index_MCP_AA", "finger2_joint2"),
            ("index_PIP", "finger2_joint3"),
            ("index_DIP", "finger2_joint4"),
            ("middle_MCP_FE", "finger3_joint1"),
            ("middle_MCP_AA", "finger3_joint2"),
            ("middle_PIP", "finger3_joint3"),
            ("middle_DIP", "finger3_joint4"),
            ("ring_MCP_FE", "finger4_joint1"),
            ("ring_MCP_AA", "finger4_joint2"),
            ("ring_PIP", "finger4_joint3"),
            ("ring_DIP", "finger4_joint4"),
            ("pinky_MCP_FE", "finger5_joint1"),
            ("pinky_MCP_AA", "finger5_joint2"),
            ("pinky_PIP", "finger5_joint3"),
            ("pinky_DIP", "finger5_joint4"),
        )
        for target_suffix, source_suffix in pairs:
            aliases[f"{side}_{target_suffix}"] = f"{side}_{source_suffix}"
    return aliases


NORTH_POC2_2_PROFILE = RobotProfile(
    variant="north_poc2_2",
    robot_prim="/World/NorthPoc2_2",
    default_usd_relpath=(
        "assets/north_poc2_2_urdf_usd/north_poc2_2_v3_1.usda"
    ),
    initial_world_position=NORTH_POC2_2_INITIAL_WORLD_POSITION,
    initial_world_orientation_xyzw=NORTH_POC2_2_INITIAL_WORLD_ORIENTATION_XYZW,
    initial_joint_positions={
        "lower_body_joint_.*": 0.0,
        "neck_joint_.*": 0.0,
        "left_arm_joint_.*": 0.0,
        "right_arm_joint_.*": 0.0,
    },
    actuator_joint_names_expr={
        name: tuple(values)
        for name, values in NORTH_POC2_2_ACTUATOR_JOINT_NAMES_EXPR.items()
    },
    arm_joint_names={
        side: tuple(values) for side, values in NORTH_POC2_2_ARM_JOINTS.items()
    },
    left_lock_joint_name="left_arm_joint_4",
    left_hand_joint_names=NORTH_POC2_2_LEFT_HAND_JOINTS,
    right_hand_joint_names=NORTH_POC2_2_RIGHT_HAND_JOINTS,
    initial_hand_joint_positions=_zero_joint_positions(
        NORTH_POC2_2_LEFT_HAND_JOINTS + NORTH_POC2_2_RIGHT_HAND_JOINTS
    ),
    hand_retarget_backend="wuji",
    hand_wuji_config_dir=FIXED_WUJI_HAND_CONFIG_DIR,
    hand_joint_aliases=_north_poc2_2_joint_aliases(),
    ik_body_names={
        "left": "left_hand_base_link",
        "right": "right_hand_base_link",
    },
    ik_body_offset_pos={
        "left": (0.0, 0.0, 0.0),
        "right": (0.0, 0.0, 0.0),
    },
    ik_body_offset_quat_xyzw={
        "left": (0.0, 0.0, 0.0, 1.0),
        "right": (0.0, 0.0, 0.0, 1.0),
    },
    ik_target_orientation_correction_quat_xyzw={
        "left": (0.0, 0.0, 0.0, 1.0),
        "right": (0.0, 0.0, 0.0, 1.0),
    },
    reference_body="head_base_link",
    reference_offset_pos=(0.0, 0.0, 0.0),
    reference_offset_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    axis_signs={},
    status_label="North POC2.2",
)
