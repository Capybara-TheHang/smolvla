# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab scene helpers for the G1-Wuji teleop app."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .robot.articulation import make_g1_wuji_robot_cfg


TABLE_TOP_SIZE_M = (1.20, 0.75, 0.05)
TABLE_TOP_CENTER_M = (0.0, 0.5, 0.74)
TABLETOP_SURFACE_Z_M = TABLE_TOP_CENTER_M[2] + TABLE_TOP_SIZE_M[2] * 0.5

BLUE_CUBE_PRIM_PATH = "/World/Props/BlueCube"
RED_CUBE_PRIM_PATH = "/World/Props/RedCube"
GRASP_CYLINDER_PRIM_PATH = "/World/Props/GraspCylinder"

CUBE_SIDE_M = 0.08
CYLINDER_HEIGHT_M = 0.10

PROP_RANDOMIZATION_SPECS = {
    BLUE_CUBE_PRIM_PATH: {
        "center": (
            TABLE_TOP_CENTER_M[0] - 0.05,
            TABLE_TOP_CENTER_M[1] - 0.1,
            TABLETOP_SURFACE_Z_M + CUBE_SIDE_M * 0.5,
        ),
        "radius_m": 0.04,
    },
    RED_CUBE_PRIM_PATH: {
        "center": (
            TABLE_TOP_CENTER_M[0] + 0.2,
            TABLE_TOP_CENTER_M[1] - 0.1,
            TABLETOP_SURFACE_Z_M + CUBE_SIDE_M * 0.5,
        ),
        "radius_m": 0.04,
    },
    GRASP_CYLINDER_PRIM_PATH: {
        "center": (
            TABLE_TOP_CENTER_M[0]+ 0.1,
            TABLE_TOP_CENTER_M[1] - 0.2,
            TABLETOP_SURFACE_Z_M + CYLINDER_HEIGHT_M * 0.5,
        ),
        "radius_m": 0.05,
    },
}


@dataclass(frozen=True)
class ContactOptimizationConfig:
    enabled: bool = True
    hand_cylinder_only: bool = True
    keep_tabletop_collision: bool = True
    keep_floor_collision: bool = False
    keep_table_leg_collision: bool = False
    disable_cylinder_gravity_without_tabletop: bool = False
    disable_left_side_collision: bool = True
    left_side_collision_keywords: tuple[str, ...] = (
        "/left",
        "left_",
    )
    hand_collision_keywords: tuple[str, ...] = (
        "hand",
        "wujihand",
        "finger",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "palm",
    )
    hand_contact_offset_m: float = 0.0035
    hand_rest_offset_m: float = 0.0
    cylinder_contact_offset_m: float = 0.0035
    cylinder_rest_offset_m: float = 0.0
    robot_max_depenetration_velocity: float = 1.5
    cylinder_max_depenetration_velocity: float = 2.5
    robot_solver_position_iteration_count: int = 24
    robot_solver_velocity_iteration_count: int = 6
    cylinder_solver_position_iteration_count: int = 24
    cylinder_solver_velocity_iteration_count: int = 6
    grasp_static_friction: float = 0.85
    grasp_dynamic_friction: float = 0.65
    grasp_friction_combine_mode: str = "average"
    table_static_friction: float = 0.9
    table_dynamic_friction: float = 0.7
    table_friction_combine_mode: str = "average"
    object_linear_damping: float = 0.04
    object_angular_damping: float = 0.04
    finger_effort_limit_sim: float = 45.0
    finger_velocity_limit_sim: float = 5.0
    finger_stiffness: float = 3000.0
    finger_damping: float = 500.0


@dataclass(frozen=True)
class G1WujiSceneConfig:
    robot_prim: str
    robot_usd: Path
    initial_world_position: tuple[float, float, float]
    initial_world_orientation_xyzw: tuple[float, float, float, float]
    initial_hand_joint_positions: Mapping[str, float]
    light_intensity: float
    robot_profile: Any | None = None
    scene_usd: Path | None = None
    scene_prim_path: str = "/World/Scene"
    contact_optimization: ContactOptimizationConfig = field(
        default_factory=ContactOptimizationConfig
    )


def _status_label(config: G1WujiSceneConfig) -> str:
    return str(getattr(config.robot_profile, "status_label", "G1-Wuji"))


def _make_shape_cfg_without_visual_material(shape_cfg_type: Any, **kwargs: Any) -> Any:
    kwargs.pop("visual_material", None)
    return shape_cfg_type(**kwargs)


def _apply_shape_display_color(
    prim_path: str, display_color: tuple[float, float, float]
) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    mesh_prim = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
    if mesh_prim is None or not mesh_prim.IsValid():
        return
    UsdGeom.Gprim(mesh_prim).CreateDisplayColorAttr(
        [Gf.Vec3f(*(float(value) for value in display_color))]
    )


def _spawn_display_color_shape(
    prim_path: str,
    cfg: Any,
    *,
    display_color: tuple[float, float, float],
    **kwargs: Any,
) -> None:
    cfg.func(prim_path, cfg, **kwargs)
    _apply_shape_display_color(prim_path, display_color)


def _collision_enabled_for_scene_part(
    contact: ContactOptimizationConfig,
    *,
    keep_when_hand_cylinder_only: bool,
) -> bool:
    if not contact.enabled or not contact.hand_cylinder_only:
        return True
    return keep_when_hand_cylinder_only


def _path_matches_any_keyword(path: str, keywords: tuple[str, ...]) -> bool:
    path_lower = path.lower()
    return any(keyword and keyword.lower() in path_lower for keyword in keywords)


def _set_collision_enabled(UsdPhysics: Any, prim: Any, enabled: bool) -> None:
    collision_api = UsdPhysics.CollisionAPI(prim)
    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
    if not collision_enabled_attr or not collision_enabled_attr.IsValid():
        collision_enabled_attr = collision_api.CreateCollisionEnabledAttr()
    collision_enabled_attr.Set(bool(enabled))


def _apply_hand_cylinder_contact_optimization(
    *,
    config: G1WujiSceneConfig,
    sim_utils: Any,
) -> None:
    contact = config.contact_optimization
    if not contact.enabled:
        return

    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    robot_root = stage.GetPrimAtPath(config.robot_prim)
    if robot_root is None or not robot_root.IsValid():
        return

    root_prefix = f"{config.robot_prim.rstrip('/')}/"
    kept_robot_colliders = 0
    disabled_left_side_colliders = 0
    disabled_robot_colliders = 0
    hand_collision_cfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=contact.hand_contact_offset_m,
        rest_offset=contact.hand_rest_offset_m,
    )

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != config.robot_prim and not prim_path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        if contact.disable_left_side_collision and _path_matches_any_keyword(
            prim_path, contact.left_side_collision_keywords
        ):
            _set_collision_enabled(UsdPhysics, prim, False)
            disabled_left_side_colliders += 1
            continue

        keep_collision = not contact.hand_cylinder_only or _path_matches_any_keyword(
            prim_path, contact.hand_collision_keywords
        )
        if keep_collision:
            if not sim_utils.modify_collision_properties(
                prim_path, hand_collision_cfg, stage=stage
            ):
                _set_collision_enabled(UsdPhysics, prim, True)
            kept_robot_colliders += 1
            continue

        _set_collision_enabled(UsdPhysics, prim, False)
        disabled_robot_colliders += 1

    if contact.hand_cylinder_only:
        print(
            f"[{_status_label(config)}] contact optimization: kept "
            f"{kept_robot_colliders} hand collider(s), disabled "
            f"{disabled_robot_colliders} non-hand robot collider(s), disabled "
            f"{disabled_left_side_colliders} left-side robot collider(s).",
            flush=True,
        )


def _spawn_external_scene_usd(config: G1WujiSceneConfig, sim_utils: Any) -> None:
    if config.scene_usd is None:
        return

    scene_cfg = sim_utils.UsdFileCfg(usd_path=str(config.scene_usd))
    scene_cfg.func(config.scene_prim_path, scene_cfg)
    print(
        f"[{_status_label(config)}] loaded external scene USD at "
        f"{config.scene_prim_path}: {config.scene_usd}",
        flush=True,
    )


def _design_g1_wuji_builtin_grasp_scene(config: G1WujiSceneConfig) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    contact = config.contact_optimization

    table_static_friction = contact.table_static_friction if contact.enabled else 1.3
    table_dynamic_friction = contact.table_dynamic_friction if contact.enabled else 1.1
    table_friction_combine_mode = (
        contact.table_friction_combine_mode if contact.enabled else "max"
    )
    grasp_static_friction = contact.grasp_static_friction if contact.enabled else 1.4
    grasp_dynamic_friction = contact.grasp_dynamic_friction if contact.enabled else 1.2
    grasp_friction_combine_mode = (
        contact.grasp_friction_combine_mode if contact.enabled else "max"
    )

    tabletop_surface_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=table_static_friction,
        dynamic_friction=table_dynamic_friction,
        restitution=0.0,
        friction_combine_mode=table_friction_combine_mode,
        restitution_combine_mode="min",
    )
    graspable_surface_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=grasp_static_friction,
        dynamic_friction=grasp_dynamic_friction,
        restitution=0.0,
        friction_combine_mode=grasp_friction_combine_mode,
        restitution_combine_mode="min",
    )

    floor_color = (0.25, 0.25, 0.25)
    floor_cfg = _make_shape_cfg_without_visual_material(
        sim_utils.CuboidCfg,
        size=(6.0, 6.0, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=_collision_enabled_for_scene_part(
                contact,
                keep_when_hand_cylinder_only=contact.keep_floor_collision,
            )
        ),
    )
    _spawn_display_color_shape(
        "/World/Floor",
        floor_cfg,
        translation=(0.0, 0.0, -0.02),
        display_color=floor_color,
    )

    table_color = (0.54, 0.42, 0.30)
    table_leg_color = (0.22, 0.22, 0.24)
    table_top_size = TABLE_TOP_SIZE_M
    table_top_center = TABLE_TOP_CENTER_M
    table_leg_size = (0.06, 0.06, 0.72)
    table_leg_x_offset = table_top_size[0] * 0.5 - 0.09
    table_leg_y_offset = table_top_size[1] * 0.5 - 0.09
    table_leg_z = table_leg_size[2] * 0.5

    table_top_cfg = _make_shape_cfg_without_visual_material(
        sim_utils.CuboidCfg,
        size=table_top_size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=_collision_enabled_for_scene_part(
                contact,
                keep_when_hand_cylinder_only=contact.keep_tabletop_collision,
            )
        ),
        physics_material=tabletop_surface_material,
    )
    _spawn_display_color_shape(
        "/World/Props/TableTop",
        table_top_cfg,
        translation=table_top_center,
        display_color=table_color,
    )

    table_leg_cfg = _make_shape_cfg_without_visual_material(
        sim_utils.CuboidCfg,
        size=table_leg_size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=_collision_enabled_for_scene_part(
                contact,
                keep_when_hand_cylinder_only=contact.keep_table_leg_collision,
            )
        ),
        physics_material=tabletop_surface_material,
    )
    for leg_index, (x_sign, y_sign) in enumerate(
        ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)), start=1
    ):
        _spawn_display_color_shape(
            f"/World/Props/TableLeg{leg_index}",
            table_leg_cfg,
            translation=(
                table_top_center[0] + x_sign * table_leg_x_offset,
                table_top_center[1] + y_sign * table_leg_y_offset,
                table_leg_z,
            ),
            display_color=table_leg_color,
        )

    cube_side = CUBE_SIDE_M
    cube_cfg = _make_shape_cfg_without_visual_material(
        sim_utils.CuboidCfg,
        size=(cube_side, cube_side, cube_side),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=(
                contact.cylinder_contact_offset_m if contact.enabled else 0.003
            ),
            rest_offset=contact.cylinder_rest_offset_m if contact.enabled else 0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=contact.object_linear_damping if contact.enabled else 0.08,
            angular_damping=contact.object_angular_damping if contact.enabled else 0.06,
            max_linear_velocity=8.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=(
                contact.cylinder_max_depenetration_velocity if contact.enabled else 3.0
            ),
            solver_position_iteration_count=(
                contact.cylinder_solver_position_iteration_count
                if contact.enabled
                else 16
            ),
            solver_velocity_iteration_count=(
                contact.cylinder_solver_velocity_iteration_count
                if contact.enabled
                else 4
            ),
            sleep_threshold=0.002,
            stabilization_threshold=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.14),
        physics_material=graspable_surface_material,
    )
    for cube_name, cube_color in (
        ("BlueCube", (0.05, 0.22, 0.95)),
        ("RedCube", (0.92, 0.06, 0.04)),
    ):
        prim_path = f"/World/Props/{cube_name}"
        center = PROP_RANDOMIZATION_SPECS[prim_path]["center"]
        _spawn_display_color_shape(
            prim_path,
            cube_cfg,
            translation=center,
            display_color=cube_color,
        )

    cylinder_radius = 0.030
    cylinder_height = CYLINDER_HEIGHT_M
    cylinder_color = (0.84, 0.18, 0.10)
    cylinder_cfg = _make_shape_cfg_without_visual_material(
        sim_utils.CylinderCfg,
        radius=cylinder_radius,
        height=cylinder_height,
        axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=(
                contact.cylinder_contact_offset_m if contact.enabled else 0.003
            ),
            rest_offset=contact.cylinder_rest_offset_m if contact.enabled else 0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=(
                contact.enabled
                and contact.hand_cylinder_only
                and not contact.keep_tabletop_collision
                and contact.disable_cylinder_gravity_without_tabletop
            ),
            linear_damping=contact.object_linear_damping if contact.enabled else 0.08,
            angular_damping=contact.object_angular_damping if contact.enabled else 0.06,
            max_linear_velocity=8.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=(
                contact.cylinder_max_depenetration_velocity if contact.enabled else 3.0
            ),
            solver_position_iteration_count=(
                contact.cylinder_solver_position_iteration_count
                if contact.enabled
                else 16
            ),
            solver_velocity_iteration_count=(
                contact.cylinder_solver_velocity_iteration_count
                if contact.enabled
                else 4
            ),
            sleep_threshold=0.002,
            stabilization_threshold=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.18),
        physics_material=graspable_surface_material,
    )
    _spawn_display_color_shape(
        GRASP_CYLINDER_PRIM_PATH,
        cylinder_cfg,
        translation=PROP_RANDOMIZATION_SPECS[GRASP_CYLINDER_PRIM_PATH]["center"],
        display_color=cylinder_color,
    )

    light_cfg = sim_utils.DomeLightCfg(
        intensity=config.light_intensity,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=make_g1_wuji_robot_cfg(config))
    _apply_hand_cylinder_contact_optimization(config=config, sim_utils=sim_utils)
    return robot


def design_g1_wuji_scene(config: G1WujiSceneConfig) -> Any:
    if config.scene_usd is None:
        return _design_g1_wuji_builtin_grasp_scene(config)

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    _spawn_external_scene_usd(config, sim_utils)

    light_cfg = sim_utils.DomeLightCfg(
        intensity=config.light_intensity,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=make_g1_wuji_robot_cfg(config))
    _apply_hand_cylinder_contact_optimization(config=config, sim_utils=sim_utils)
    return robot
