# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1-Wuji robot profile and joint definitions."""

from __future__ import annotations

from .types import RobotProfile

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

G1_WUJI_ARM_JOINTS = {
    "left": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
}

G1_WUJI_INITIAL_WORLD_POSITION = (0.0, 0.0, 0.8)
G1_WUJI_INITIAL_WORLD_ORIENTATION_XYZW = (0.0, 0.70710678, 0.70710678, 0.0)
G1_WUJI_INITIAL_JOINT_POSITIONS = {
    "right_wrist_yaw_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_roll_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "left_elbow_joint": 1.57,
}
G1_WUJI_ACTUATOR_JOINT_NAMES_EXPR = {
    "base": ("base_.*",),
    "right_arm": ("right_shoulder_.*", "right_elbow_joint"),
    "left_arm": ("left_shoulder_.*", "left_elbow_joint"),
    "wrist": (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
    "fingers": ("left_finger.*", "right_finger.*"),
}

WUJI_SKELETON21_OPENXR_NAMES = (
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)

FIXED_WUJI_HAND_CONFIG_DIR = "official/g1_wuji"


def _default_wuji_initial_joint_positions() -> dict[str, float]:
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


G1_WUJI_PROFILE = RobotProfile(
    variant="g1_wuji",
    robot_prim="/World/G1Wuji",
    default_usd_relpath="assets/g1_wuji/g1_wuji.usd",
    initial_world_position=G1_WUJI_INITIAL_WORLD_POSITION,
    initial_world_orientation_xyzw=G1_WUJI_INITIAL_WORLD_ORIENTATION_XYZW,
    initial_joint_positions=dict(G1_WUJI_INITIAL_JOINT_POSITIONS),
    actuator_joint_names_expr={
        name: tuple(values)
        for name, values in G1_WUJI_ACTUATOR_JOINT_NAMES_EXPR.items()
    },
    arm_joint_names={
        side: tuple(values) for side, values in G1_WUJI_ARM_JOINTS.items()
    },
    left_lock_joint_name="left_elbow_joint",
    left_hand_joint_names=LEFT_WUJI_JOINTS,
    right_hand_joint_names=RIGHT_WUJI_JOINTS,
    initial_hand_joint_positions=_default_wuji_initial_joint_positions(),
    hand_retarget_backend="wuji",
    hand_wuji_config_dir=FIXED_WUJI_HAND_CONFIG_DIR,
    hand_joint_aliases={},
    ik_body_names={
        "left": "left_wrist_yaw_link",
        "right": "right_wrist_yaw_link",
    },
    ik_body_offset_pos={
        "left": (0.0415, 0.003, 0.0),
        "right": (0.0415, -0.003, 0.0),
    },
    ik_body_offset_quat_xyzw={
        "left": (0.0, 0.0, 0.0, 1.0),
        "right": (0.0, 0.0, 0.0, 1.0),
    },
    ik_target_orientation_correction_quat_xyzw={
        "left": (0.5, 0.5, 0.5, 0.5),
        "right": (-0.5, 0.5, 0.5, -0.5),
    },
    reference_body="torso_link",
    reference_offset_pos=(0.0077774, 0.0000210, 0.3836842),
    reference_offset_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    axis_signs={},
    status_label="G1-Wuji",
)
