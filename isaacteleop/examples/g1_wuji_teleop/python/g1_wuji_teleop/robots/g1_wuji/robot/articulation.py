# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab articulation configuration for supported teleop robots."""

from __future__ import annotations

from typing import Any

from .north_poc2_2 import NORTH_POC2_2_ARM_INITIAL_JOINT_POSITIONS


def _xyzw_to_wxyz(
    quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = quat_xyzw
    return (w, x, y, z)


def make_g1_wuji_robot_cfg(config: Any):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    contact = config.contact_optimization
    robot_max_depenetration_velocity = (
        contact.robot_max_depenetration_velocity if contact.enabled else 1.0
    )
    hand_contact_offset = contact.hand_contact_offset_m if contact.enabled else 0.002
    hand_rest_offset = contact.hand_rest_offset_m if contact.enabled else 0.001
    robot_solver_position_iterations = (
        contact.robot_solver_position_iteration_count if contact.enabled else 16
    )
    robot_solver_velocity_iterations = (
        contact.robot_solver_velocity_iteration_count if contact.enabled else 4
    )

    profile = config.robot_profile
    joint_pos = dict(getattr(profile, "initial_joint_positions", {}) or {})
    if not joint_pos:
        joint_pos = {
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
    if getattr(profile, "variant", None) == "north_poc2_2":
        joint_pos.pop("left_arm_joint_.*", None)
        joint_pos.pop("right_arm_joint_.*", None)
        joint_pos.update(NORTH_POC2_2_ARM_INITIAL_JOINT_POSITIONS)
    joint_pos.update(
        {
            name: float(value)
            for name, value in config.initial_hand_joint_positions.items()
        }
    )

    return ArticulationCfg(
        prim_path=config.robot_prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(config.robot_usd),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                max_linear_velocity=100.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=robot_max_depenetration_velocity,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=hand_contact_offset,
                rest_offset=hand_rest_offset,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=robot_solver_position_iterations,
                solver_velocity_iteration_count=robot_solver_velocity_iterations,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=config.initial_world_position,
            # Config stores xyzw; IsaacLab InitialStateCfg.rot expects wxyz.
            rot=_xyzw_to_wxyz(config.initial_world_orientation_xyzw),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.98,
        actuators=_wuji_actuators(ImplicitActuatorCfg, contact, profile),
    )


def _wuji_actuators(ImplicitActuatorCfg: Any, contact: Any, profile: Any | None):
    if contact.enabled:
        finger_effort_limit_sim = contact.finger_effort_limit_sim
        finger_velocity_limit_sim = contact.finger_velocity_limit_sim
        finger_stiffness = contact.finger_stiffness
        finger_damping = contact.finger_damping
    else:
        finger_effort_limit_sim = 100.0
        finger_velocity_limit_sim = 6.0
        finger_stiffness = 3000.0
        finger_damping = 100.0

    expr_by_group = dict(getattr(profile, "actuator_joint_names_expr", {}) or {})
    if not expr_by_group:
        expr_by_group = {
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

    common = {}
    if expr_by_group.get("base"):
        common["base"] = ImplicitActuatorCfg(
            joint_names_expr=list(expr_by_group["base"]),
            effort_limit_sim=100000.0,
            velocity_limit_sim=1000.0,
            stiffness=1e6,
            damping=1e4,
        )
    if expr_by_group.get("right_arm"):
        common["right_arm"] = ImplicitActuatorCfg(
            joint_names_expr=list(expr_by_group["right_arm"]),
            effort_limit_sim=5.0,
            velocity_limit_sim=3.0,
            stiffness=1500.0,
            damping=300.0,
        )
    if expr_by_group.get("left_arm"):
        common["left_arm"] = ImplicitActuatorCfg(
            joint_names_expr=list(expr_by_group["left_arm"]),
            effort_limit_sim=5.0,
            velocity_limit_sim=3.0,
            stiffness=1500.0,
            damping=300.0,
        )
    if expr_by_group.get("wrist"):
        common["wrist"] = ImplicitActuatorCfg(
            joint_names_expr=list(expr_by_group["wrist"]),
            effort_limit_sim=300.0,
            velocity_limit_sim=8.0,
            stiffness=1500.0,
            damping=300.0,
        )
    if expr_by_group.get("fingers"):
        common["fingers"] = ImplicitActuatorCfg(
            joint_names_expr=list(expr_by_group["fingers"]),
            effort_limit_sim=finger_effort_limit_sim,
            velocity_limit_sim=finger_velocity_limit_sim,
            stiffness=finger_stiffness,
            damping=finger_damping,
        )
    return common
