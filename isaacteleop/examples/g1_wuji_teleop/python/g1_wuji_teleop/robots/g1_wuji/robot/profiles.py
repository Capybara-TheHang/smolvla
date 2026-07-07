# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot profile registry and variant selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .g1_wuji import (
    G1_WUJI_PROFILE,
    LEFT_WUJI_JOINTS,
    RIGHT_WUJI_JOINTS,
    WUJI_SKELETON21_OPENXR_NAMES,
    _default_wuji_initial_joint_positions,
)
from .north_poc2_2 import NORTH_POC2_2_PROFILE
from .types import RobotProfile


ROBOT_PROFILES: dict[str, RobotProfile] = {
    G1_WUJI_PROFILE.variant: G1_WUJI_PROFILE,
    NORTH_POC2_2_PROFILE.variant: NORTH_POC2_2_PROFILE,
}


def normalize_robot_variant(value: Any) -> str:
    raw = str(value or "g1_wuji").strip().lower()
    aliases = {
        "g1wuji": "g1_wuji",
        "g1-wuji": "g1_wuji",
        "wuji": "g1_wuji",
        "north": "north_poc2_2",
        "north-poc2": "north_poc2_2",
        "north_poc2": "north_poc2_2",
        "north-poc2-2": "north_poc2_2",
        "poc2": "north_poc2_2",
        "poc2_2": "north_poc2_2",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in ROBOT_PROFILES:
        raise ValueError(
            f"Unsupported robot.variant={value!r}. Expected one of {sorted(ROBOT_PROFILES)}"
        )
    return normalized


def available_robot_variants() -> tuple[str, ...]:
    return tuple(sorted(ROBOT_PROFILES))


def robot_profile_from_config(config: Mapping[str, Any]) -> RobotProfile:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")
    return ROBOT_PROFILES[normalize_robot_variant(robot_config.get("variant"))]


def _values_match(lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, str):
        return str(lhs) == rhs
    if isinstance(rhs, Sequence) and not isinstance(rhs, (str, bytes)):
        if not isinstance(lhs, Sequence) or isinstance(lhs, (str, bytes)):
            return False
        if len(lhs) != len(rhs):
            return False
        try:
            return all(
                abs(float(left) - float(right)) <= 1.0e-6
                for left, right in zip(lhs, rhs, strict=True)
            )
        except (TypeError, ValueError):
            return False
    return lhs == rhs


def _replace_default_value(
    mapping: dict[str, Any],
    *,
    key: str,
    old_default: Any,
    new_default: Any,
) -> None:
    if key not in mapping or _values_match(mapping[key], old_default):
        mapping[key] = (
            list(new_default) if isinstance(new_default, tuple) else new_default
        )


def with_robot_variant_override(
    config: Mapping[str, Any], robot_variant: Any | None
) -> dict[str, Any]:
    result = dict(config)
    if robot_variant is None:
        return result

    robot_config_value = result.get("robot", {})
    if robot_config_value is None:
        robot_config_value = {}
    if not isinstance(robot_config_value, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    robot_config = dict(robot_config_value)
    old_profile = robot_profile_from_config(result)
    new_profile = ROBOT_PROFILES[normalize_robot_variant(robot_variant)]
    robot_config["variant"] = new_profile.variant
    _replace_default_value(
        robot_config,
        key="initial_world_position",
        old_default=old_profile.initial_world_position,
        new_default=new_profile.initial_world_position,
    )
    _replace_default_value(
        robot_config,
        key="initial_world_orientation_xyzw",
        old_default=old_profile.initial_world_orientation_xyzw,
        new_default=new_profile.initial_world_orientation_xyzw,
    )

    binding_value = robot_config.get("avp_frame_binding", {})
    if binding_value is None:
        binding_value = {}
    if not isinstance(binding_value, Mapping):
        raise ValueError("Config field 'robot.avp_frame_binding' must be a mapping")
    binding_config = dict(binding_value)
    _replace_default_value(
        binding_config,
        key="robot_reference_body",
        old_default=old_profile.reference_body,
        new_default=new_profile.reference_body,
    )
    _replace_default_value(
        binding_config,
        key="robot_reference_offset_pos",
        old_default=old_profile.reference_offset_pos,
        new_default=new_profile.reference_offset_pos,
    )
    _replace_default_value(
        binding_config,
        key="robot_reference_offset_quat_xyzw",
        old_default=old_profile.reference_offset_quat_xyzw,
        new_default=new_profile.reference_offset_quat_xyzw,
    )
    robot_config["avp_frame_binding"] = binding_config
    result["robot"] = robot_config
    return result

