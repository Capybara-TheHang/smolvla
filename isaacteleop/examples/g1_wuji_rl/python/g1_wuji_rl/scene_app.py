# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Isaac Lab launcher for the G1-Wuji RL scene."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .paths import (
    app_root,
    default_config_path,
    ensure_import_paths,
    resolve_repo_relative_path,
)
from .robots.g1_wuji.runtime import (
    initial_hand_joint_positions_from_config,
    robot_profile_from_config,
)
from .robots.g1_wuji.scene import (
    ContactOptimizationConfig,
    G1WujiSceneConfig,
    design_g1_wuji_scene,
)


def _load_yaml_file(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _float_tuple(
    values: Any,
    *,
    name: str,
    length: int,
    default: Sequence[float],
) -> tuple[float, ...]:
    raw = default if values is None else values
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"Config field {name!r} must be a {length}-element sequence")
    if len(raw) != length:
        raise ValueError(f"Config field {name!r} must contain exactly {length} values")
    return tuple(float(value) for value in raw)


def _string_tuple(
    value: Any,
    *,
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        items = [item.strip().lower() for item in str(value).split(",")]
    elif isinstance(value, Sequence):
        items = [str(item).strip().lower() for item in value]
    else:
        raise ValueError(f"Config field '{name}' must be a sequence or comma string")

    result = tuple(item for item in items if item)
    if not result:
        raise ValueError(f"Config field '{name}' must contain at least one keyword")
    return result


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config field '{key}' must be a mapping")
    return value


def _scene_usd_path(
    config: Mapping[str, Any],
    *,
    config_dir: Path,
    override: str | None,
) -> Path | None:
    raw_value: Any = override
    field_name = "--scene-usd"
    if raw_value is None:
        scene_config = _mapping(config, "scene")
        raw_value = scene_config.get("usd_path")
        field_name = "scene.usd_path"

    if raw_value is None:
        return None

    raw_path = str(raw_value).strip()
    if not raw_path or raw_path.lower() in {"none", "null"}:
        return None
    if not raw_path.lower().endswith((".usd", ".usda", ".usdc")):
        raise ValueError(
            f"Config field '{field_name}' must be a local USD path ending in "
            ".usd, .usda, or .usdc"
        )

    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [config_dir / path, app_root() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    resolved = candidates[0].resolve()
    raise FileNotFoundError(f"Scene USD not found from {field_name}: {resolved}")


def _robot_initial_world_position(
    config: Mapping[str, Any],
) -> tuple[float, float, float]:
    robot_config = _mapping(config, "robot")
    return _float_tuple(
        robot_config.get("initial_world_position"),
        name="robot.initial_world_position",
        length=3,
        default=(0.0, 0.0, 0.0),
    )


def _robot_initial_world_orientation_xyzw(
    config: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    robot_config = _mapping(config, "robot")
    quat = _float_tuple(
        robot_config.get("initial_world_orientation_xyzw"),
        name="robot.initial_world_orientation_xyzw",
        length=4,
        default=(0.0, 0.0, 0.0, 1.0),
    )
    norm = sum(value * value for value in quat) ** 0.5
    if norm < 1.0e-8:
        raise ValueError(
            "Config field 'robot.initial_world_orientation_xyzw' must be non-zero"
        )
    return tuple(value / norm for value in quat)


def _contact_optimization_config(
    config: Mapping[str, Any],
) -> ContactOptimizationConfig:
    robot_config = _mapping(config, "robot")
    contact_config = robot_config.get("contact_optimization", {})
    if contact_config is None:
        contact_config = {}
    if not isinstance(contact_config, Mapping):
        raise ValueError("Config field 'robot.contact_optimization' must be a mapping")

    defaults = ContactOptimizationConfig()

    def nonnegative_float(key: str, default: float) -> float:
        value = float(contact_config.get(key, default))
        if value < 0.0:
            raise ValueError(
                f"Config field 'robot.contact_optimization.{key}' must be >= 0"
            )
        return value

    def positive_float(key: str, default: float) -> float:
        value = float(contact_config.get(key, default))
        if value <= 0.0:
            raise ValueError(
                f"Config field 'robot.contact_optimization.{key}' must be > 0"
            )
        return value

    def positive_int(key: str, default: int) -> int:
        value = int(contact_config.get(key, default))
        if value <= 0:
            raise ValueError(
                f"Config field 'robot.contact_optimization.{key}' must be > 0"
            )
        return value

    def friction_combine_mode(key: str, default: str) -> str:
        value = str(contact_config.get(key, default)).strip().lower()
        valid_values = {"average", "min", "multiply", "max"}
        if value not in valid_values:
            raise ValueError(
                f"Config field 'robot.contact_optimization.{key}' must be one of "
                f"{sorted(valid_values)}"
            )
        return value

    return ContactOptimizationConfig(
        enabled=bool(contact_config.get("enabled", defaults.enabled)),
        hand_cylinder_only=bool(
            contact_config.get("hand_cylinder_only", defaults.hand_cylinder_only)
        ),
        keep_tabletop_collision=bool(
            contact_config.get(
                "keep_tabletop_collision", defaults.keep_tabletop_collision
            )
        ),
        keep_floor_collision=bool(
            contact_config.get("keep_floor_collision", defaults.keep_floor_collision)
        ),
        keep_table_leg_collision=bool(
            contact_config.get(
                "keep_table_leg_collision", defaults.keep_table_leg_collision
            )
        ),
        disable_cylinder_gravity_without_tabletop=bool(
            contact_config.get(
                "disable_cylinder_gravity_without_tabletop",
                defaults.disable_cylinder_gravity_without_tabletop,
            )
        ),
        disable_left_side_collision=bool(
            contact_config.get(
                "disable_left_side_collision",
                defaults.disable_left_side_collision,
            )
        ),
        left_side_collision_keywords=_string_tuple(
            contact_config.get("left_side_collision_keywords"),
            name="robot.contact_optimization.left_side_collision_keywords",
            default=defaults.left_side_collision_keywords,
        ),
        hand_collision_keywords=_string_tuple(
            contact_config.get("hand_collision_keywords"),
            name="robot.contact_optimization.hand_collision_keywords",
            default=defaults.hand_collision_keywords,
        ),
        hand_contact_offset_m=nonnegative_float(
            "hand_contact_offset_m", defaults.hand_contact_offset_m
        ),
        hand_rest_offset_m=nonnegative_float(
            "hand_rest_offset_m", defaults.hand_rest_offset_m
        ),
        cylinder_contact_offset_m=nonnegative_float(
            "cylinder_contact_offset_m", defaults.cylinder_contact_offset_m
        ),
        cylinder_rest_offset_m=nonnegative_float(
            "cylinder_rest_offset_m", defaults.cylinder_rest_offset_m
        ),
        robot_max_depenetration_velocity=positive_float(
            "robot_max_depenetration_velocity",
            defaults.robot_max_depenetration_velocity,
        ),
        cylinder_max_depenetration_velocity=positive_float(
            "cylinder_max_depenetration_velocity",
            defaults.cylinder_max_depenetration_velocity,
        ),
        robot_solver_position_iteration_count=positive_int(
            "robot_solver_position_iteration_count",
            defaults.robot_solver_position_iteration_count,
        ),
        robot_solver_velocity_iteration_count=positive_int(
            "robot_solver_velocity_iteration_count",
            defaults.robot_solver_velocity_iteration_count,
        ),
        cylinder_solver_position_iteration_count=positive_int(
            "cylinder_solver_position_iteration_count",
            defaults.cylinder_solver_position_iteration_count,
        ),
        cylinder_solver_velocity_iteration_count=positive_int(
            "cylinder_solver_velocity_iteration_count",
            defaults.cylinder_solver_velocity_iteration_count,
        ),
        grasp_static_friction=nonnegative_float(
            "grasp_static_friction", defaults.grasp_static_friction
        ),
        grasp_dynamic_friction=nonnegative_float(
            "grasp_dynamic_friction", defaults.grasp_dynamic_friction
        ),
        grasp_friction_combine_mode=friction_combine_mode(
            "grasp_friction_combine_mode", defaults.grasp_friction_combine_mode
        ),
        table_static_friction=nonnegative_float(
            "table_static_friction", defaults.table_static_friction
        ),
        table_dynamic_friction=nonnegative_float(
            "table_dynamic_friction", defaults.table_dynamic_friction
        ),
        table_friction_combine_mode=friction_combine_mode(
            "table_friction_combine_mode", defaults.table_friction_combine_mode
        ),
        object_linear_damping=nonnegative_float(
            "object_linear_damping", defaults.object_linear_damping
        ),
        object_angular_damping=nonnegative_float(
            "object_angular_damping", defaults.object_angular_damping
        ),
        finger_effort_limit_sim=positive_float(
            "finger_effort_limit_sim", defaults.finger_effort_limit_sim
        ),
        finger_velocity_limit_sim=positive_float(
            "finger_velocity_limit_sim", defaults.finger_velocity_limit_sim
        ),
        finger_stiffness=positive_float("finger_stiffness", defaults.finger_stiffness),
        finger_damping=positive_float("finger_damping", defaults.finger_damping),
    )


def _sim_step_should_render(sim: Any) -> bool:
    is_rendering = getattr(sim, "is_rendering", None)
    if is_rendering is not None:
        return bool(is_rendering)

    render_mode = getattr(sim, "render_mode", None)
    render_mode_type = getattr(sim, "RenderMode", None)
    if render_mode is not None and render_mode_type is not None:
        return render_mode != render_mode_type.NO_GUI_OR_RENDERING

    has_gui = getattr(sim, "has_gui", None)
    if callable(has_gui):
        return bool(has_gui())

    return True


def _simulation_dt(config: Mapping[str, Any]) -> float:
    simulation_config = _mapping(config, "simulation")
    dt = float(simulation_config.get("dt", 1.0 / 60.0))
    if dt <= 0.0:
        raise ValueError("Config field 'simulation.dt' must be > 0")
    return dt


def _light_intensity(config: Mapping[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    simulation_config = _mapping(config, "simulation")
    return float(simulation_config.get("light_intensity", 3000.0))


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Load the G1-Wuji RL scene.")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--robot-usd", default=None)
    parser.add_argument("--robot-prim", default=None)
    parser.add_argument(
        "--scene-usd",
        default=None,
        help="Optional local USD scene path. Overrides scene.usd_path in the YAML.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument(
        "--robot-height",
        type=float,
        default=None,
        help="Optional override for robot.initial_world_position z.",
    )
    parser.add_argument("--light-intensity", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ensure_import_paths()
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config_path = Path(args.config).expanduser().resolve()
    config = _load_yaml_file(config_path)
    config_dir = config_path.parent

    profile = robot_profile_from_config(config)
    robot_config = _mapping(config, "robot")
    scene_config = _mapping(config, "scene")

    initial_world_position = _robot_initial_world_position(config)
    if args.robot_height is not None:
        initial_world_position = (
            initial_world_position[0],
            initial_world_position[1],
            float(args.robot_height),
        )
    initial_world_orientation_xyzw = _robot_initial_world_orientation_xyzw(config)

    robot_usd_value = args.robot_usd or robot_config.get("usd_path")
    if args.robot_usd:
        robot_usd = Path(args.robot_usd).expanduser().resolve()
    elif robot_usd_value:
        robot_usd = resolve_repo_relative_path(robot_usd_value)
    else:
        robot_usd = resolve_repo_relative_path(profile.default_usd_relpath)
    if not robot_usd.is_file():
        raise FileNotFoundError(f"{profile.status_label} USD not found: {robot_usd}")

    scene_usd = _scene_usd_path(
        config,
        config_dir=config_dir,
        override=args.scene_usd,
    )
    scene_prim_path = str(scene_config.get("prim_path", "/World/Scene"))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    sim_cfg = sim_utils.SimulationCfg(dt=_simulation_dt(config), device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, -4.0, 2.0], [0.0, 0.0, 0.9])

    robot_prim = str(args.robot_prim or robot_config.get("prim_path") or profile.robot_prim)
    robot = design_g1_wuji_scene(
        G1WujiSceneConfig(
            robot_prim=robot_prim,
            robot_usd=robot_usd,
            initial_world_position=initial_world_position,
            initial_world_orientation_xyzw=initial_world_orientation_xyzw,
            initial_hand_joint_positions=initial_hand_joint_positions_from_config(
                config,
                profile,
            ),
            light_intensity=_light_intensity(config, args.light_intensity),
            scene_usd=scene_usd,
            scene_prim_path=scene_prim_path,
            contact_optimization=_contact_optimization_config(config),
        )
    )

    sim.reset()
    sim_dt = sim.get_physics_dt()
    print(
        f"[{profile.status_label}] RL scene ready: robot={robot_usd} "
        f"scene_usd={scene_usd or 'builtin'} prim={robot.cfg.prim_path}",
        flush=True,
    )

    start_time = time.monotonic()
    while simulation_app.is_running():
        if args.duration_s > 0.0 and time.monotonic() - start_time >= args.duration_s:
            break
        sim.step(render=_sim_step_should_render(sim))
        robot.update(sim_dt)

    close_app = getattr(simulation_app, "close", None)
    if callable(close_app):
        close_app()
    return 0
