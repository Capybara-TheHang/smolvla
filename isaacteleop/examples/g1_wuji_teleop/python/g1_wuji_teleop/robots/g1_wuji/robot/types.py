# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared robot profile datatypes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotProfile:
    variant: str
    robot_prim: str
    default_usd_relpath: str
    initial_world_position: tuple[float, float, float]
    initial_world_orientation_xyzw: tuple[float, float, float, float]
    initial_joint_positions: dict[str, float]
    actuator_joint_names_expr: dict[str, tuple[str, ...]]
    arm_joint_names: dict[str, tuple[str, ...]]
    left_lock_joint_name: str | None
    left_hand_joint_names: tuple[str, ...]
    right_hand_joint_names: tuple[str, ...]
    initial_hand_joint_positions: dict[str, float]
    hand_retarget_backend: str
    hand_wuji_config_dir: str | None
    hand_joint_aliases: dict[str, str]
    ik_body_names: dict[str, str]
    ik_body_offset_pos: dict[str, tuple[float, float, float]]
    ik_body_offset_quat_xyzw: dict[str, tuple[float, float, float, float]]
    ik_target_orientation_correction_quat_xyzw: dict[
        str, tuple[float, float, float, float]
    ]
    reference_body: str
    reference_offset_pos: tuple[float, float, float]
    reference_offset_quat_xyzw: tuple[float, float, float, float]
    axis_signs: dict[str, float]
    status_label: str

