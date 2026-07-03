# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot metadata used by the G1-Wuji RL scene."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LEFT_WUJI_JOINTS = (
    "left_finger1_joint1",
    "left_finger1_joint2",
    "left_finger1_joint3",
    "left_finger1_joint4",
    "left_finger2_joint1",
    "left_finger2_joint2",
    "left_finger2_joint3",
    "left_finger2_joint4",
    "left_finger3_joint1",
    "left_finger3_joint2",
    "left_finger3_joint3",
    "left_finger3_joint4",
    "left_finger4_joint1",
    "left_finger4_joint2",
    "left_finger4_joint3",
    "left_finger4_joint4",
    "left_finger5_joint1",
    "left_finger5_joint2",
    "left_finger5_joint3",
    "left_finger5_joint4",
)

RIGHT_WUJI_JOINTS = tuple(
    name.replace("left_", "right_", 1) for name in LEFT_WUJI_JOINTS
)


def default_wuji_initial_joint_positions() -> dict[str, float]:
    return {
        "left_finger1_joint1": 0.059,
        "left_finger1_joint2": 0.059,
        "left_finger1_joint3": 0.059,
        "left_finger1_joint4": 0.0,
        "left_finger2_joint1": 0.0,
        "left_finger2_joint2": 0.0,
        "left_finger2_joint3": 0.0,
        "left_finger2_joint4": 0.0,
        "left_finger3_joint1": 0.0,
        "left_finger3_joint2": 0.0,
        "left_finger3_joint3": 0.0,
        "left_finger3_joint4": 0.0,
        "left_finger4_joint1": 0.0,
        "left_finger4_joint2": 0.0,
        "left_finger4_joint3": 0.0,
        "left_finger4_joint4": 0.0,
        "left_finger5_joint1": 0.0,
        "left_finger5_joint2": 0.0,
        "left_finger5_joint3": 0.0,
        "left_finger5_joint4": 0.0,
        "right_finger1_joint1": 0.037,
        "right_finger1_joint2": 0.037,
        "right_finger1_joint3": 0.037,
        "right_finger1_joint4": 0.0,
        "right_finger2_joint1": 0.0,
        "right_finger2_joint2": 0.0,
        "right_finger2_joint3": 0.0,
        "right_finger2_joint4": 0.0,
        "right_finger3_joint1": 0.0,
        "right_finger3_joint2": 0.0,
        "right_finger3_joint3": 0.0,
        "right_finger3_joint4": 0.0,
        "right_finger4_joint1": 0.0,
        "right_finger4_joint2": 0.0,
        "right_finger4_joint3": 0.0,
        "right_finger4_joint4": 0.0,
        "right_finger5_joint1": 0.0,
        "right_finger5_joint2": 0.0,
        "right_finger5_joint3": 0.0,
        "right_finger5_joint4": 0.0,
    }


@dataclass(frozen=True)
class RobotProfile:
    variant: str
    robot_prim: str
    default_usd_relpath: str
    left_hand_joint_names: tuple[str, ...]
    right_hand_joint_names: tuple[str, ...]
    initial_hand_joint_positions: dict[str, float]
    status_label: str


ROBOT_PROFILES: dict[str, RobotProfile] = {
    "g1_wuji": RobotProfile(
        variant="g1_wuji",
        robot_prim="/World/G1Wuji",
        default_usd_relpath="assets/g1_wuji/g1_wuji.usd",
        left_hand_joint_names=LEFT_WUJI_JOINTS,
        right_hand_joint_names=RIGHT_WUJI_JOINTS,
        initial_hand_joint_positions=default_wuji_initial_joint_positions(),
        status_label="G1-Wuji",
    ),
}


def normalize_robot_variant(value: Any) -> str:
    raw = str(value or "g1_wuji").strip().lower()
    aliases = {
        "g1wuji": "g1_wuji",
        "g1-wuji": "g1_wuji",
        "wuji": "g1_wuji",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in ROBOT_PROFILES:
        raise ValueError(
            f"Unsupported robot.variant={value!r}. Expected one of {sorted(ROBOT_PROFILES)}"
        )
    return normalized


def robot_profile_from_config(config: Mapping[str, Any]) -> RobotProfile:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")
    return ROBOT_PROFILES[normalize_robot_variant(robot_config.get("variant"))]


def initial_hand_joint_positions_from_config(
    config: Mapping[str, Any],
    profile: RobotProfile,
) -> dict[str, float]:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    initial_positions = dict(profile.initial_hand_joint_positions)
    override = robot_config.get("initial_hand_joint_positions", {})
    if override is None:
        return initial_positions
    if not isinstance(override, Mapping):
        raise ValueError("Config field 'robot.initial_hand_joint_positions' must be a mapping")

    known_joints = set(profile.left_hand_joint_names) | set(profile.right_hand_joint_names)
    for joint_name, value in override.items():
        joint_name = str(joint_name)
        if joint_name not in known_joints:
            raise ValueError(
                "Config field 'robot.initial_hand_joint_positions' contains "
                f"unknown Wuji joint: {joint_name!r}"
            )
        initial_positions[joint_name] = float(value)
    return initial_positions


def as_torch(value: Any) -> Any:
    """Convert Warp arrays to torch tensors when needed, otherwise return the input."""

    to_torch = getattr(value, "to_torch", None)
    if callable(to_torch):
        return to_torch()

    try:
        import warp as wp
    except Exception:
        wp = None

    if wp is not None:
        try:
            array_type = wp.array
        except Exception:
            array_type = ()
        if array_type and isinstance(value, array_type):
            return wp.to_torch(value)

    return value
