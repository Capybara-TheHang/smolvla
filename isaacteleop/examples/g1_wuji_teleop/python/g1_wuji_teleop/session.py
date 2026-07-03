# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Isaac Lab scene for AVP + MANUS teleop on G1-Wuji.

Run this script with an Isaac Sim / Isaac Lab Python environment. It intentionally
keeps the simulation side small: a ground plane, a light, the selected G1 USD,
and live hand joint targets from ``TeleopMain``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .devices.avp_manus_stream import (
    INPUT_PROFILE_CHOICES,
    INPUT_PROFILE_CONFIG,
    NullTeleopMain,
    TeleopMain,
    apply_input_profile_config,
)
from .paths import (
    app_root,
    default_config_path,
    ensure_import_paths,
    resolve_repo_relative_path,
)
from .robots.g1_wuji.runtime import (
    AvpFrameBindingRuntimeConfig,
    AvpRobotFrameBinding,
    HandTargetPostProcessor,
    WujiHandRuntimeConfig,
    WujiHandTargetBackend,
    as_torch,
    avp_frame_binding_runtime_config,
    matrix_to_quat_xyzw,
    normalize_quat_xyzw,
    openxr_pose_to_isaac_pose,
    ordered_hand_targets,
    robot_profile_from_config,
    wuji_hand_runtime_config,
)
from .robots.g1_wuji.scene import (
    BLUE_CUBE_PRIM_PATH,
    ContactOptimizationConfig,
    G1WujiSceneConfig,
    PROP_RANDOMIZATION_SPECS,
    RED_CUBE_PRIM_PATH,
    design_g1_wuji_scene,
)


os.environ.setdefault("WARP_CACHE_PATH", "/tmp/isaacteleop-warp-cache")


def _enable_kit_extension(extension_name: str) -> None:
    """Enable a packaged Kit extension by name."""

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if ext_manager is not None and not ext_manager.is_extension_enabled(extension_name):
        ext_manager.set_extension_enabled_immediate(extension_name, True)


def _inject_isaacsim_extension_root(extension_name: str) -> None:
    """Append a packaged Isaac Sim extension root from ``extscache`` or ``exts`` to ``sys.path``."""

    for path_entry in sys.path:
        if "site-packages" not in path_entry:
            continue
        isaacsim_root = Path(path_entry) / "isaacsim"
        extscache_dir = isaacsim_root / "extscache"
        if extscache_dir.is_dir():
            matches = sorted(extscache_dir.glob(f"{extension_name}-*"))
            if matches:
                ext_path = str(matches[-1])
                if ext_path not in sys.path:
                    sys.path.append(ext_path)
                return

        exts_dir = isaacsim_root / "exts" / extension_name
        if exts_dir.is_dir():
            ext_path = str(exts_dir)
            if ext_path not in sys.path:
                sys.path.append(ext_path)
            return


def _import_kit_module(
    module_name: str,
    *,
    extension_names: Sequence[str] = (),
    extension_root: str | None = None,
    error_cls: type[Exception] = RuntimeError,
    error_message: str,
):
    for extension_name in extension_names:
        _enable_kit_extension(extension_name)
    if importlib.util.find_spec(module_name) is not None:
        return importlib.import_module(module_name)

    _inject_isaacsim_extension_root(extension_root or module_name)
    for extension_name in extension_names:
        _enable_kit_extension(extension_name)
    if importlib.util.find_spec(module_name) is not None:
        return importlib.import_module(module_name)

    raise error_cls(error_message)


def _ensure_carb_input_available() -> None:
    """Load ``carb.input`` explicitly so keyboard events work across Kit variants."""

    _import_kit_module(
        "carb.input",
        error_message="Failed to load carb.input required for teleop keyboard shortcuts.",
    )


def _import_omni_appwindow():
    """Import ``omni.appwindow`` even when Kit hasn't pre-enabled the extension."""

    return _import_kit_module(
        "omni.appwindow",
        extension_names=("omni.appwindow",),
        error_message="Failed to load omni.appwindow required for teleop keyboard shortcuts.",
    )


def _import_omni_viewport_utility():
    """Import ``omni.kit.viewport.utility`` when Isaac Sim hasn't enabled it yet."""

    return _import_kit_module(
        "omni.kit.viewport.utility",
        extension_names=("omni.kit.viewport.window", "omni.kit.viewport.utility"),
        error_message=(
            "Failed to load omni.kit.viewport.utility required for head-view "
            "camera control."
        ),
    )


def _import_omni_replicator_core():
    """Import ``omni.replicator.core`` when Isaac Sim hasn't enabled it yet."""

    return _import_kit_module(
        "omni.replicator.core",
        extension_names=("omni.replicator.core",),
        error_cls=ModuleNotFoundError,
        error_message="robot.head_view_xr_display requires omni.replicator.core.",
    )


def _load_smolvla_data_sample_module():
    module_path = app_root() / "data" / "smolvla_data_sample.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"SmolVLA data sampler not found: {module_path}")

    module_name = "g1_wuji_smolvla_data_sample"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import SmolVLA data sampler: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _disable_robot_transmissive_materials(
    *, stage: Any, robot_root_prim: Any, status_label: str
) -> None:
    """Make the robot body render opaque even when the source USD uses a transmissive OmniSurface."""

    if stage is None or robot_root_prim is None or not robot_root_prim.IsValid():
        return

    root_prefix = f"{robot_root_prim.GetPath()}/"
    patched_shader_paths: list[str] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(root_prefix):
            continue
        if prim.GetTypeName() != "Shader" or not prim_path.endswith(
            "/OmniSurface/Shader"
        ):
            continue

        enable_specular_transmission = prim.GetAttribute(
            "inputs:enable_specular_transmission"
        )
        if enable_specular_transmission.IsValid():
            enable_specular_transmission.Set(False)

        specular_transmission_weight = prim.GetAttribute(
            "inputs:specular_transmission_weight"
        )
        if specular_transmission_weight.IsValid():
            specular_transmission_weight.Set(0.0)

        enable_diffuse_transmission = prim.GetAttribute(
            "inputs:enable_diffuse_transmission"
        )
        if enable_diffuse_transmission.IsValid():
            enable_diffuse_transmission.Set(False)

        enable_opacity = prim.GetAttribute("inputs:enable_opacity")
        if enable_opacity.IsValid():
            enable_opacity.Set(False)

        geometry_opacity = prim.GetAttribute("inputs:geometry_opacity")
        if geometry_opacity.IsValid():
            geometry_opacity.Set(1.0)

        patched_shader_paths.append(prim_path)

    if patched_shader_paths:
        print(
            f"[{status_label}] disabled transmissive robot material on "
            f"{len(patched_shader_paths)} OmniSurface shader(s).",
            flush=True,
        )


def _walk_child_prims(root_prim: Any):
    if not root_prim or not root_prim.IsValid():
        return
    yield root_prim
    for child in root_prim.GetChildren():
        yield from _walk_child_prims(child)


def _find_child_prim_by_name(root_prim: Any, target_name: str) -> Any | None:
    for prim in _walk_child_prims(root_prim):
        if prim.GetName() == target_name:
            return prim
    return None


def _head_like_relative_paths() -> tuple[str, ...]:
    return (
        "torso_link/head_link",
        "g1/torso_link/head_link",
        "head_link",
        "g1/head_link",
        "head",
        "g1/head",
    )


def _find_head_like_prim(stage: Any, root_prim: Any, root_path: str) -> Any | None:
    if stage is not None:
        for relative_path in _head_like_relative_paths():
            prim = stage.GetPrimAtPath(f"{root_path}/{relative_path}")
            if prim is not None and prim.IsValid():
                return prim
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if "/visuals/" not in prim_path and any(
                prim_path.endswith(f"/{suffix}")
                for suffix in _head_like_relative_paths()
            ):
                return prim

    for name in ("head_link", "head"):
        match = _find_child_prim_by_name(root_prim, name)
        if match is not None and "/visuals/" not in str(match.GetPath()):
            return match
    for prim in _walk_child_prims(root_prim):
        prim_path = str(prim.GetPath())
        if "/visuals/" not in prim_path and "head" in prim.GetName().lower():
            return prim
    return None


def _ensure_cloudxr_runtime_environment_defaults() -> None:
    """Populate common CloudXR/OpenXR env vars from ``~/.cloudxr`` when the shell has not set them."""

    cloudxr_root = Path.home() / ".cloudxr"
    manifest_candidates = (
        cloudxr_root / "openxr_cloudxr.json",
        cloudxr_root / "share" / "openxr" / "1" / "openxr_cloudxr.json",
    )
    manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    if manifest_path is not None:
        manifest_str = str(manifest_path)
        if not os.environ.get("XR_RUNTIME_JSON"):
            os.environ["XR_RUNTIME_JSON"] = manifest_str
        if not os.environ.get("OPENXR_RUNTIME_JSON"):
            os.environ["OPENXR_RUNTIME_JSON"] = manifest_str

    runtime_dir = cloudxr_root / "run"
    if runtime_dir.is_dir() and not os.environ.get("NV_CXR_RUNTIME_DIR"):
        os.environ["NV_CXR_RUNTIME_DIR"] = str(runtime_dir)


def _sim_step_should_render(sim: Any) -> bool:
    """Return whether ``SimulationContext.step`` should refresh rendering this frame."""
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


ARM_JOINTS = {
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

LEFT_ELBOW_LOCK_JOINT = "left_elbow_joint"


def _sample_xy_in_disk(
    rng: np.random.Generator, center_xy: tuple[float, float], radius_m: float
) -> tuple[float, float]:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(radius_m) * float(np.sqrt(rng.uniform(0.0, 1.0)))
    return (
        float(center_xy[0] + radius * np.cos(angle)),
        float(center_xy[1] + radius * np.sin(angle)),
    )


def _set_prim_translation(
    stage: Any,
    prim_path: str,
    translation_xyz: tuple[float, float, float],
) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Scene prop prim not found: {prim_path}")

    UsdGeom.XformCommonAPI(prim).SetTranslate(
        Gf.Vec3d(*(float(value) for value in translation_xyz))
    )


def _rigid_body_view_count(view: Any) -> int:
    count = getattr(view, "count", 0)
    if callable(count):
        count = count()
    return int(count)


def _rigid_body_view_transforms_to_torch(view: Any, *, device: str) -> Any:
    import torch
    import warp as wp

    transforms = view.get_transforms()
    if isinstance(transforms, wp.array):
        transforms = wp.to_torch(transforms)
    return torch.as_tensor(transforms, dtype=torch.float32, device=device).clone()


def _torch_to_warp_float_array(tensor: Any) -> Any:
    import warp as wp

    return wp.from_torch(tensor.contiguous(), dtype=wp.float32)


def _rigid_body_view_all_indices(view: Any, *, device: str) -> Any:
    import torch
    import warp as wp

    indices = torch.arange(
        _rigid_body_view_count(view), dtype=torch.int32, device=device
    )
    return wp.from_torch(indices.contiguous(), dtype=wp.int32)


def _set_rigid_body_view_transform_and_stop(
    view: Any,
    default_transform: Any,
    translation_xyz: tuple[float, float, float],
    *,
    device: str,
) -> None:
    import torch

    transform = default_transform.to(device=device, dtype=torch.float32).clone()
    transform[:, :3] = torch.as_tensor(
        [translation_xyz], dtype=torch.float32, device=device
    )
    indices = _rigid_body_view_all_indices(view, device=device)
    view.set_transforms(_torch_to_warp_float_array(transform), indices)
    view.set_velocities(
        _torch_to_warp_float_array(
            torch.zeros(
                (_rigid_body_view_count(view), 6),
                dtype=torch.float32,
                device=device,
            )
        ),
        indices,
    )


@dataclass(frozen=True)
class HeadViewCameraConfig:
    enabled: bool = True
    activate_on_calibration: bool = True
    prim_path: str = "/World/TeleopTargets/HeadViewCamera"
    resolution_px: tuple[int, int] = (960, 540)
    translation_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_offset_rpy_deg: tuple[float, float, float] = (90.0, 0.0, 90.0)
    horizontal_aperture_mm: float = 20.955
    focal_length_mm: float = 18.14756
    clipping_range_m: tuple[float, float] = (0.01, 1000.0)


@dataclass(frozen=True)
class TableOverheadCameraConfig:
    enabled: bool = False
    prim_path: str = "/World/TeleopTargets/TableOverheadCamera"
    resolution_px: tuple[int, int] = (640, 480)
    translation_m: tuple[float, float, float] = (0.0, 0.5, 1.55)
    orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    horizontal_aperture_mm: float = 20.955
    focal_length_mm: float = 16.0
    clipping_range_m: tuple[float, float] = (0.01, 1000.0)


@dataclass(frozen=True)
class HeadViewXrDisplayConfig:
    enabled: bool = False
    activate_on_calibration: bool = True
    app_name: str = "G1HeadViewXrDisplay"
    resolution_px: tuple[int, int] = (960, 540)
    lock_mode: str = "head"
    distance_m: float = 1.2
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    plane_width_m: float = 2.2
    near_z_m: float = 0.05
    far_z_m: float = 100.0
    clear_color_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class AuxiliaryViewportConfig:
    enabled: bool = True
    activate_on_calibration: bool = True
    window_name: str = "Teleop Default View"
    resolution_px: tuple[int, int] = (960, 540)
    position_px: tuple[int, int] = (40, 40)


@dataclass(frozen=True)
class TeleopViewportLayoutConfig:
    enabled: bool = True
    activate_on_calibration: bool = True
    head_window_name: str = "Teleop Head View"
    table_window_name: str = "Teleop Table Overhead View"
    resolution_px: tuple[int, int] = (960, 540)
    right_split_ratio: float = 0.5
    table_split_ratio: float = 0.5


@dataclass(frozen=True)
class LeftSideFreezeConfig:
    enabled_from_start: bool = False
    enabled_after_calibration: bool = False


def _rpy_deg_to_quat_xyzw(
    rpy_deg: Sequence[float],
) -> tuple[float, float, float, float]:
    roll_deg, pitch_deg, yaw_deg = rpy_deg
    roll_rad = np.deg2rad(float(roll_deg))
    pitch_rad = np.deg2rad(float(pitch_deg))
    yaw_rad = np.deg2rad(float(yaw_deg))

    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)

    quat_xyzw = normalize_quat_xyzw(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )
    return tuple(float(value) for value in quat_xyzw)


def _quat_xyzw_mul(
    lhs_xyzw: Sequence[float],
    rhs_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = normalize_quat_xyzw(lhs_xyzw)
    rx, ry, rz, rw = normalize_quat_xyzw(rhs_xyzw)
    quat_xyzw = normalize_quat_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )
    return tuple(float(value) for value in quat_xyzw)


def _quat_xyzw_rotate(
    quat_xyzw: Sequence[float], vector_xyz: Sequence[float]
) -> tuple[float, float, float]:
    qx, qy, qz, qw = normalize_quat_xyzw(quat_xyzw)
    vector = np.asarray(vector_xyz, dtype=np.float64)
    qvec = np.asarray([qx, qy, qz], dtype=np.float64)
    uv = np.cross(qvec, vector)
    uuv = np.cross(qvec, uv)
    rotated = vector + 2.0 * (qw * uv + uuv)
    return tuple(float(value) for value in rotated)


def _quat_xyzw_to_gf_quatd(Gf: Any, quat_xyzw: Sequence[float]) -> Any:
    quat = normalize_quat_xyzw(quat_xyzw)
    return Gf.Quatd(
        float(quat[3]),
        Gf.Vec3d(
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
        ),
    )


def _matrix3_to_gf_quatd(Gf: Any, matrix: np.ndarray) -> Any:
    return _quat_xyzw_to_gf_quatd(
        Gf, matrix_to_quat_xyzw(np.asarray(matrix, dtype=np.float64))
    )


class HeadViewCameraController:
    """Drive a viewport camera from the robot head pose after AVP calibration."""

    def __init__(
        self,
        *,
        stage: Any,
        UsdGeom: Any,
        Gf: Any,
        viewport_api: Any,
        config: HeadViewCameraConfig,
        head_link_prim: Any | None,
        avp_frame_binding: Any | None,
        whole_body_yaw: Any | None,
    ) -> None:
        self._config = config
        self._viewport_api = viewport_api
        self._UsdGeom = UsdGeom
        self._Gf = Gf
        self._head_link_prim = head_link_prim
        self._avp_frame_binding = avp_frame_binding
        self._whole_body_yaw = whole_body_yaw
        self._active = False
        self._announced_missing_head = False
        self._orientation_offset_quat = self._orientation_offset_quatd(config)

        camera = UsdGeom.Camera.Define(stage, config.prim_path)
        self._camera_prim = camera.GetPrim()
        self._camera_xform = UsdGeom.Xformable(self._camera_prim)
        self._translate_op = self._camera_xform.AddTranslateOp()
        # USD xform op value types are strict. Head prim world rotations come back as
        # GfQuatd here, so keep the camera orient op in double precision as well.
        self._orient_op = self._camera_xform.AddOrientOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        )
        self._scale_op = self._camera_xform.AddScaleOp()
        self._camera_path = str(self._camera_prim.GetPath())

        camera.GetHorizontalApertureAttr().Set(float(config.horizontal_aperture_mm))
        camera.GetFocalLengthAttr().Set(float(config.focal_length_mm))
        camera.GetClippingRangeAttr().Set(
            Gf.Vec2f(
                float(config.clipping_range_m[0]),
                float(config.clipping_range_m[1]),
            )
        )
        self._scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    @property
    def camera_path(self) -> str:
        return self._camera_path

    def activate(self) -> None:
        if self._viewport_api is None or self._active:
            return
        self._viewport_api.camera_path = self._camera_path
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def update(self, *, xform_cache: Any) -> None:
        head_link_prim = self._head_link_prim
        if head_link_prim is None or xform_cache is None:
            if not self._announced_missing_head:
                print(
                    "Head view camera is enabled but no robot head prim is available; "
                    "camera updates are disabled.",
                    flush=True,
                )
                self._announced_missing_head = True
            return

        xform_cache.Clear()
        world_transform = xform_cache.GetLocalToWorldTransform(head_link_prim)
        world_translation = world_transform.ExtractTranslation()
        world_rotation = world_transform.ExtractRotationQuat()
        camera_base_rotation = world_rotation * self._orientation_offset_quat
        camera_rotation = self._apply_head_pose_delta(camera_base_rotation)

        offset = self._config.translation_offset_m
        offset_vec = self._Gf.Vec3d(
            float(offset[0]), float(offset[1]), float(offset[2])
        )
        camera_translation = world_translation + world_transform.TransformDir(
            offset_vec
        )

        self._translate_op.Set(camera_translation)
        self._orient_op.Set(camera_rotation)

    def _apply_head_pose_delta(self, base_rotation: Any) -> Any:
        avp_frame_binding = self._avp_frame_binding
        if avp_frame_binding is None or not getattr(
            avp_frame_binding, "calibrated", False
        ):
            return base_rotation
        if not getattr(avp_frame_binding, "head_tilt_following_enabled", True):
            return base_rotation
        tilt_delta_rot = avp_frame_binding.latest_head_tilt_delta_rot()
        camera_rotation = base_rotation
        if tilt_delta_rot is not None:
            tilt_delta_quat = self._matrix3_to_quatd(tilt_delta_rot)
            camera_rotation = tilt_delta_quat * camera_rotation
        return camera_rotation

    def _orientation_offset_quatd(self, config: HeadViewCameraConfig) -> Any:
        roll_deg, pitch_deg, yaw_deg = config.orientation_offset_rpy_deg
        roll_rad = np.deg2rad(float(roll_deg))
        pitch_rad = np.deg2rad(float(pitch_deg))
        yaw_rad = np.deg2rad(float(yaw_deg))

        cr = np.cos(roll_rad * 0.5)
        sr = np.sin(roll_rad * 0.5)
        cp = np.cos(pitch_rad * 0.5)
        sp = np.sin(pitch_rad * 0.5)
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)

        quat_xyzw = normalize_quat_xyzw(
            (
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            )
        )
        return self._Gf.Quatd(
            float(quat_xyzw[3]),
            self._Gf.Vec3d(
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ),
        )

    def _matrix3_to_quatd(self, matrix: np.ndarray) -> Any:
        quat_xyzw = normalize_quat_xyzw(
            matrix_to_quat_xyzw(np.asarray(matrix, dtype=np.float64))
        )
        return self._Gf.Quatd(
            float(quat_xyzw[3]),
            self._Gf.Vec3d(
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ),
        )


class TableOverheadCameraController:
    """Create a fixed overhead USD camera looking at the tabletop."""

    def __init__(
        self,
        *,
        stage: Any,
        UsdGeom: Any,
        Gf: Any,
        viewport_api: Any,
        config: TableOverheadCameraConfig,
    ) -> None:
        self._config = config
        self._viewport_api = viewport_api
        self._Gf = Gf
        self._active = False
        self._orientation_quat_xyzw = _rpy_deg_to_quat_xyzw(config.orientation_rpy_deg)

        camera = UsdGeom.Camera.Define(stage, config.prim_path)
        self._camera_prim = camera.GetPrim()
        self._camera_xform = UsdGeom.Xformable(self._camera_prim)
        self._translate_op = self._camera_xform.AddTranslateOp()
        self._orient_op = self._camera_xform.AddOrientOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        )
        self._scale_op = self._camera_xform.AddScaleOp()
        self._camera_path = str(self._camera_prim.GetPath())

        camera.GetHorizontalApertureAttr().Set(float(config.horizontal_aperture_mm))
        camera.GetFocalLengthAttr().Set(float(config.focal_length_mm))
        camera.GetClippingRangeAttr().Set(
            Gf.Vec2f(
                float(config.clipping_range_m[0]),
                float(config.clipping_range_m[1]),
            )
        )
        self._scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        self._translate_op.Set(
            Gf.Vec3d(
                float(config.translation_m[0]),
                float(config.translation_m[1]),
                float(config.translation_m[2]),
            )
        )
        self._orient_op.Set(_quat_xyzw_to_gf_quatd(Gf, self._orientation_quat_xyzw))

    @property
    def camera_path(self) -> str:
        return self._camera_path

    def activate(self) -> None:
        if self._viewport_api is None or self._active:
            return
        self._viewport_api.camera_path = self._camera_path
        self._active = True

    def deactivate(self) -> None:
        self._active = False


def _head_view_camera_config(config: Mapping[str, Any]) -> HeadViewCameraConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    camera_config = robot_config.get("head_view_camera", {})
    if camera_config is None:
        camera_config = {}
    if not isinstance(camera_config, Mapping):
        raise ValueError("Config field 'robot.head_view_camera' must be a mapping")

    prim_path = str(
        camera_config.get("prim_path", "/World/TeleopTargets/HeadViewCamera")
    ).strip()
    if not prim_path:
        raise ValueError(
            "Config field 'robot.head_view_camera.prim_path' must be non-empty"
        )

    resolution_tuple = _resolution_tuple(
        camera_config.get("resolution_px", (960, 540)),
        name="robot.head_view_camera.resolution_px",
    )

    return HeadViewCameraConfig(
        enabled=bool(camera_config.get("enabled", True)),
        activate_on_calibration=bool(
            camera_config.get("activate_on_calibration", True)
        ),
        prim_path=prim_path,
        resolution_px=resolution_tuple,
        translation_offset_m=_float_tuple(
            camera_config.get("translation_offset_m"),
            name="robot.head_view_camera.translation_offset_m",
            length=3,
            default=(0.0, 0.0, 0.0),
        ),
        orientation_offset_rpy_deg=_float_tuple(
            camera_config.get("orientation_offset_rpy_deg"),
            name="robot.head_view_camera.orientation_offset_rpy_deg",
            length=3,
            default=(90.0, 0.0, 90.0),
        ),
        horizontal_aperture_mm=float(
            camera_config.get("horizontal_aperture_mm", 20.955)
        ),
        focal_length_mm=float(camera_config.get("focal_length_mm", 18.14756)),
        clipping_range_m=_float_tuple(
            camera_config.get("clipping_range_m"),
            name="robot.head_view_camera.clipping_range_m",
            length=2,
            default=(0.01, 1000.0),
        ),
    )


def _table_overhead_camera_config(
    config: Mapping[str, Any],
) -> TableOverheadCameraConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    camera_config = robot_config.get("table_overhead_camera", {})
    if camera_config is None:
        camera_config = {}
    if not isinstance(camera_config, Mapping):
        raise ValueError("Config field 'robot.table_overhead_camera' must be a mapping")

    prim_path = str(
        camera_config.get("prim_path", "/World/TeleopTargets/TableOverheadCamera")
    ).strip()
    if not prim_path:
        raise ValueError(
            "Config field 'robot.table_overhead_camera.prim_path' must be non-empty"
        )

    resolution_tuple = _resolution_tuple(
        camera_config.get("resolution_px", (640, 480)),
        name="robot.table_overhead_camera.resolution_px",
    )

    return TableOverheadCameraConfig(
        enabled=bool(camera_config.get("enabled", False)),
        prim_path=prim_path,
        resolution_px=resolution_tuple,
        translation_m=_float_tuple(
            camera_config.get("translation_m"),
            name="robot.table_overhead_camera.translation_m",
            length=3,
            default=(0.0, 0.5, 1.55),
        ),
        orientation_rpy_deg=_float_tuple(
            camera_config.get("orientation_rpy_deg"),
            name="robot.table_overhead_camera.orientation_rpy_deg",
            length=3,
            default=(0.0, 0.0, 0.0),
        ),
        horizontal_aperture_mm=float(
            camera_config.get("horizontal_aperture_mm", 20.955)
        ),
        focal_length_mm=float(camera_config.get("focal_length_mm", 16.0)),
        clipping_range_m=_float_tuple(
            camera_config.get("clipping_range_m"),
            name="robot.table_overhead_camera.clipping_range_m",
            length=2,
            default=(0.01, 1000.0),
        ),
    )


def _resolution_tuple(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Config field '{name}' must be a sequence")
    resolution_tuple = tuple(int(item) for item in value)
    if (
        len(resolution_tuple) != 2
        or resolution_tuple[0] <= 0
        or resolution_tuple[1] <= 0
    ):
        raise ValueError(f"Config field '{name}' must contain two positive integers")
    return resolution_tuple


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


def _head_view_xr_display_config(config: Mapping[str, Any]) -> HeadViewXrDisplayConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    xr_config = robot_config.get("head_view_xr_display", {})
    if xr_config is None:
        xr_config = {}
    if not isinstance(xr_config, Mapping):
        raise ValueError("Config field 'robot.head_view_xr_display' must be a mapping")

    camera_config = robot_config.get("head_view_camera", {})
    if camera_config is None:
        camera_config = {}
    if not isinstance(camera_config, Mapping):
        raise ValueError("Config field 'robot.head_view_camera' must be a mapping")

    default_resolution = camera_config.get("resolution_px", (960, 540))
    resolution_tuple = _resolution_tuple(
        xr_config.get("resolution_px", default_resolution),
        name="robot.head_view_xr_display.resolution_px",
    )

    lock_mode_raw = str(xr_config.get("lock_mode", "head")).strip().lower()
    lock_mode_aliases = {
        "head": "head",
        "head_locked": "head",
        "world": "world",
        "world_locked": "world",
    }
    lock_mode = lock_mode_aliases.get(lock_mode_raw, lock_mode_raw)
    if lock_mode not in {"head", "world"}:
        raise ValueError(
            "Config field 'robot.head_view_xr_display.lock_mode' must be 'head' or 'world'"
        )

    return HeadViewXrDisplayConfig(
        enabled=bool(xr_config.get("enabled", False)),
        activate_on_calibration=bool(xr_config.get("activate_on_calibration", True)),
        app_name=str(xr_config.get("app_name", "G1HeadViewXrDisplay")).strip()
        or "G1HeadViewXrDisplay",
        resolution_px=(resolution_tuple[0], resolution_tuple[1]),
        lock_mode=lock_mode,
        distance_m=float(xr_config.get("distance_m", 1.2)),
        offset_x_m=float(xr_config.get("offset_x_m", 0.0)),
        offset_y_m=float(xr_config.get("offset_y_m", 0.0)),
        plane_width_m=float(xr_config.get("plane_width_m", 2.2)),
        near_z_m=float(xr_config.get("near_z_m", 0.05)),
        far_z_m=float(xr_config.get("far_z_m", 100.0)),
        clear_color_rgba=_float_tuple(
            xr_config.get("clear_color_rgba"),
            name="robot.head_view_xr_display.clear_color_rgba",
            length=4,
            default=(0.0, 0.0, 0.0, 1.0),
        ),
    )


def _left_side_freeze_config(config: Mapping[str, Any]) -> LeftSideFreezeConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    freeze_config = robot_config.get("left_side_freeze", {})
    if freeze_config is None:
        freeze_config = {}
    if not isinstance(freeze_config, Mapping):
        raise ValueError("Config field 'robot.left_side_freeze' must be a mapping")

    return LeftSideFreezeConfig(
        enabled_from_start=bool(freeze_config.get("enabled_from_start", False)),
        enabled_after_calibration=bool(
            freeze_config.get("enabled_after_calibration", False)
        ),
    )


def _teleop_viewport_layout_config(
    config: Mapping[str, Any],
) -> TeleopViewportLayoutConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    layout_config = robot_config.get("teleop_viewport_layout", {})
    if layout_config is None:
        layout_config = {}
    if not isinstance(layout_config, Mapping):
        raise ValueError(
            "Config field 'robot.teleop_viewport_layout' must be a mapping"
        )

    right_split_ratio = float(layout_config.get("right_split_ratio", 0.5))
    if not 0.05 <= right_split_ratio <= 0.95:
        raise ValueError(
            "Config field 'robot.teleop_viewport_layout.right_split_ratio' "
            "must be in [0.05, 0.95]"
        )
    table_split_ratio = float(layout_config.get("table_split_ratio", 0.5))
    if not 0.05 <= table_split_ratio <= 0.95:
        raise ValueError(
            "Config field 'robot.teleop_viewport_layout.table_split_ratio' "
            "must be in [0.05, 0.95]"
        )

    return TeleopViewportLayoutConfig(
        enabled=bool(layout_config.get("enabled", True)),
        activate_on_calibration=bool(
            layout_config.get("activate_on_calibration", True)
        ),
        head_window_name=str(
            layout_config.get("head_window_name", "Teleop Head View")
        ).strip()
        or "Teleop Head View",
        table_window_name=str(
            layout_config.get("table_window_name", "Teleop Table Overhead View")
        ).strip()
        or "Teleop Table Overhead View",
        resolution_px=_resolution_tuple(
            layout_config.get("resolution_px", (960, 540)),
            name="robot.teleop_viewport_layout.resolution_px",
        ),
        right_split_ratio=right_split_ratio,
        table_split_ratio=table_split_ratio,
    )


def _contact_optimization_config(
    config: Mapping[str, Any],
) -> ContactOptimizationConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

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


def _auxiliary_viewport_config(config: Mapping[str, Any]) -> AuxiliaryViewportConfig:
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    aux_config = robot_config.get("auxiliary_viewport", {})
    if aux_config is None:
        aux_config = {}
    if not isinstance(aux_config, Mapping):
        raise ValueError("Config field 'robot.auxiliary_viewport' must be a mapping")

    resolution_values = aux_config.get("resolution_px", (960, 540))
    if isinstance(resolution_values, (str, bytes)) or not isinstance(
        resolution_values, Sequence
    ):
        raise ValueError(
            "Config field 'robot.auxiliary_viewport.resolution_px' must be a sequence"
        )
    resolution_tuple = tuple(int(value) for value in resolution_values)
    if (
        len(resolution_tuple) != 2
        or resolution_tuple[0] <= 0
        or resolution_tuple[1] <= 0
    ):
        raise ValueError(
            "Config field 'robot.auxiliary_viewport.resolution_px' must contain two positive integers"
        )

    position_values = aux_config.get("position_px", (40, 40))
    if isinstance(position_values, (str, bytes)) or not isinstance(
        position_values, Sequence
    ):
        raise ValueError(
            "Config field 'robot.auxiliary_viewport.position_px' must be a sequence"
        )
    position_tuple = tuple(int(value) for value in position_values)
    if len(position_tuple) != 2:
        raise ValueError(
            "Config field 'robot.auxiliary_viewport.position_px' must contain two integers"
        )

    return AuxiliaryViewportConfig(
        enabled=bool(aux_config.get("enabled", True)),
        activate_on_calibration=bool(aux_config.get("activate_on_calibration", True)),
        window_name=str(aux_config.get("window_name", "Teleop Default View")).strip()
        or "Teleop Default View",
        resolution_px=(resolution_tuple[0], resolution_tuple[1]),
        position_px=(position_tuple[0], position_tuple[1]),
    )


def _quat_wxyz_normalize(
    quat_wxyz: Sequence[float],
) -> tuple[float, float, float, float]:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    quat /= norm
    return tuple(float(value) for value in quat)


def _quat_wxyz_mul(
    lhs_wxyz: Sequence[float], rhs_wxyz: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = _quat_wxyz_normalize(lhs_wxyz)
    rw, rx, ry, rz = _quat_wxyz_normalize(rhs_wxyz)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_wxyz_rotate(
    quat_wxyz: Sequence[float], vector_xyz: Sequence[float]
) -> tuple[float, float, float]:
    qw, qx, qy, qz = _quat_wxyz_normalize(quat_wxyz)
    vector = np.asarray(vector_xyz, dtype=np.float64)
    qvec = np.asarray([qx, qy, qz], dtype=np.float64)
    uv = np.cross(qvec, vector)
    uuv = np.cross(qvec, uv)
    rotated = vector + 2.0 * (qw * uv + uuv)
    return tuple(float(value) for value in rotated)


def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    return (
        float(np.cos(yaw_rad * 0.5)),
        0.0,
        float(np.sin(yaw_rad * 0.5)),
        0.0,
    )


class HeadViewXrDisplayBridge:
    """Present the robot head-view camera in the AVP headset via a shared XR session."""

    _IDENTITY_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)

    def __init__(
        self,
        *,
        config: HeadViewXrDisplayConfig,
        camera_prim_path: str,
        required_xr_extensions: Sequence[str],
        sim_device: str,
    ) -> None:
        ensure_import_paths()
        import isaacteleop.viz as viz
        from isaacteleop.oxr import OpenXRSessionHandles

        if not str(sim_device).startswith("cuda"):
            raise ValueError(
                "robot.head_view_xr_display currently requires a CUDA Isaac Sim device"
            )

        rep = _import_omni_replicator_core()
        self._config = config
        self._camera_prim_path = str(camera_prim_path)
        self._rep = rep
        self._viz = viz
        self._capture_device = str(sim_device)
        self._visible = False
        self._render_product = None
        self._rgb_annotator = None
        self._render_error: BaseException | None = None
        self._stop = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._world_locked_pose: (
            tuple[
                tuple[float, float, float],
                tuple[float, float, float, float],
            ]
            | None
        ) = None
        self._plane_size_m = (
            float(config.plane_width_m),
            float(config.plane_width_m)
            * float(config.resolution_px[1])
            / float(config.resolution_px[0]),
        )

        session_cfg = viz.VizSessionConfig()
        session_cfg.mode = viz.DisplayMode.kXr
        session_cfg.app_name = config.app_name
        session_cfg.xr_near_z = float(config.near_z_m)
        session_cfg.xr_far_z = float(config.far_z_m)
        session_cfg.required_extensions = list(required_xr_extensions)
        session_cfg.clear_color = tuple(
            float(value) for value in config.clear_color_rgba
        )
        self._viz_session = viz.VizSession.create(session_cfg)

        oxr_handles_tuple = self._viz_session.get_oxr_handles()
        if oxr_handles_tuple is None:
            raise RuntimeError(
                "Televiz XR session was created, but no OpenXR handles were exposed for TeleopSession sharing."
            )
        self._oxr_handles = OpenXRSessionHandles(*oxr_handles_tuple)

        layer_cfg = viz.QuadLayerConfig()
        layer_cfg.name = "robot_head_view_xr"
        layer_cfg.resolution = viz.Resolution(
            int(config.resolution_px[0]), int(config.resolution_px[1])
        )
        layer_cfg.format = viz.PixelFormat.kRGBA8
        layer_cfg.placement = self._initial_placement()
        self._layer = self._viz_session.add_quad_layer(layer_cfg)
        self._layer.set_visible(False)
        self._create_camera_render_product()

        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="g1_head_view_xr_render",
            daemon=False,
        )
        self._render_thread.start()

    @property
    def oxr_handles(self) -> Any:
        return self._oxr_handles

    def set_visible(self, visible: bool) -> None:
        self._raise_if_failed()
        new_visible = bool(visible)
        if new_visible == self._visible:
            return
        self._layer.set_visible(new_visible)
        self._visible = new_visible

    def submit_frame(self) -> bool:
        self._raise_if_failed()
        rgba = self._read_camera_frame()
        if rgba is None:
            return False
        self._layer.submit(self._coerce_cuda_image(rgba))
        return True

    def close(self) -> None:
        self._stop.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=5.0)
            if self._render_thread.is_alive():
                print(
                    "Head-view XR display render thread did not exit within 5s; "
                    "skipping explicit Televiz destruction to avoid tearing down an active XR session.",
                    flush=True,
                )
            else:
                self._render_thread = None
        self._destroy_camera_render_product()
        if self._render_thread is None:
            viz_session = getattr(self, "_viz_session", None)
            if viz_session is not None:
                viz_session.destroy()
                self._viz_session = None
        if self._render_error is not None:
            print(
                f"Head-view XR display render loop stopped with error: {self._render_error}",
                file=sys.stderr,
                flush=True,
            )

    def _raise_if_failed(self) -> None:
        if self._render_error is not None:
            raise RuntimeError(
                f"Head-view XR display render loop failed: {self._render_error}"
            ) from self._render_error

    def _create_camera_render_product(self) -> None:
        self._render_product = self._rep.create.render_product(
            self._camera_prim_path,
            (
                int(self._config.resolution_px[0]),
                int(self._config.resolution_px[1]),
            ),
            force_new=True,
        )
        self._rgb_annotator = self._rep.AnnotatorRegistry.get_annotator(
            "LdrColor",
            device="cuda",
            do_array_copy=False,
        )
        self._rgb_annotator.attach(self._render_product)

    def _destroy_camera_render_product(self) -> None:
        annotator = getattr(self, "_rgb_annotator", None)
        if annotator is not None:
            annotator.detach()
            self._rgb_annotator = None
        render_product = getattr(self, "_render_product", None)
        if render_product is not None:
            render_product.destroy()
            self._render_product = None

    def _read_camera_frame(self) -> Any | None:
        if self._stop.is_set():
            return None
        annotator = self._rgb_annotator
        if annotator is None:
            return None
        rgba = annotator.get_data(device="cuda")
        if rgba is None:
            return None
        if isinstance(rgba, Mapping):
            rgba = rgba.get("data")
        if rgba is None:
            return None
        import warp as wp

        # Replicator commonly returns a Warp CUDA array here. Converting once
        # to Torch normalizes vec4-vs-last-dimension layouts and avoids Warp's
        # Python indexing limitations (for example ``[..., :4]``).
        if isinstance(rgba, wp.array):
            rgba = wp.to_torch(rgba)
        shape = tuple(int(dim) for dim in getattr(rgba, "shape", ()))
        # Freshly attached render products often need a few rendered frames before
        # LdrColor exposes a real HxWxC image. Treat that warm-up window as
        # "frame not ready yet" instead of tearing down the teleop session.
        if len(shape) < 3 or shape[0] <= 0 or shape[1] <= 0 or shape[-1] < 4:
            return None
        if shape[-1] == 4:
            return rgba
        return rgba[..., :4]

    def _coerce_cuda_image(self, image: Any) -> Any:
        if hasattr(image, "__cuda_array_interface__"):
            return image
        if hasattr(image, "__dlpack__"):
            import warp as wp

            converted = wp.to_torch(image)
            if hasattr(converted, "__cuda_array_interface__"):
                return converted
        if hasattr(image, "__array_interface__"):
            import torch

            return torch.as_tensor(
                np.asarray(image), device=self._capture_device
            ).contiguous()
        raise TypeError(
            "Head-view XR display received a camera frame that does not expose "
            "__cuda_array_interface__ or __array_interface__."
        )

    def _initial_placement(self) -> Any:
        return self._viz.QuadLayerPlacement(
            self._viz.Pose3D(
                (0.0, 1.5, -float(self._config.distance_m)),
                self._IDENTITY_ROT_WXYZ,
            ),
            self._plane_size_m,
        )

    def _render_loop(self) -> None:
        while not self._stop.is_set():
            self._update_layer_placement()
            self._viz_session.render()
            if self._viz_session.should_close():
                self._stop.set()

    def _update_layer_placement(self) -> None:
        head_pose = self._viz_session.head_pose_now()
        if head_pose is None:
            return
        head_position = tuple(float(value) for value in head_pose.position)
        head_orientation = tuple(float(value) for value in head_pose.orientation)
        if self._config.lock_mode == "world":
            if self._world_locked_pose is None:
                self._world_locked_pose = self._compute_world_locked_pose(
                    head_position, head_orientation
                )
            position, orientation = self._world_locked_pose
        else:
            position, orientation = self._compute_head_locked_pose(
                head_position, head_orientation
            )
        self._layer.set_placement(
            self._viz.QuadLayerPlacement(
                self._viz.Pose3D(position, orientation),
                self._plane_size_m,
            )
        )

    def _compute_head_locked_pose(
        self,
        head_position: tuple[float, float, float],
        head_orientation: tuple[float, float, float, float],
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        forward = _quat_wxyz_rotate(head_orientation, (0.0, 0.0, -1.0))
        right = _quat_wxyz_rotate(head_orientation, (1.0, 0.0, 0.0))
        up = _quat_wxyz_rotate(head_orientation, (0.0, 1.0, 0.0))
        position = (
            head_position[0]
            + forward[0] * self._config.distance_m
            + right[0] * self._config.offset_x_m
            + up[0] * self._config.offset_y_m,
            head_position[1]
            + forward[1] * self._config.distance_m
            + right[1] * self._config.offset_x_m
            + up[1] * self._config.offset_y_m,
            head_position[2]
            + forward[2] * self._config.distance_m
            + right[2] * self._config.offset_x_m
            + up[2] * self._config.offset_y_m,
        )
        orientation = _quat_wxyz_mul(head_orientation, self._IDENTITY_ROT_WXYZ)
        return position, orientation

    def _compute_world_locked_pose(
        self,
        head_position: tuple[float, float, float],
        head_orientation: tuple[float, float, float, float],
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        forward = np.asarray(
            _quat_wxyz_rotate(head_orientation, (0.0, 0.0, -1.0)),
            dtype=np.float64,
        )
        forward[1] = 0.0
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-8:
            forward = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            forward /= forward_norm
        right = np.asarray([-forward[2], 0.0, forward[0]], dtype=np.float64)
        position = (
            head_position[0]
            + float(forward[0]) * self._config.distance_m
            + float(right[0]) * self._config.offset_x_m,
            head_position[1] + self._config.offset_y_m,
            head_position[2]
            + float(forward[2]) * self._config.distance_m
            + float(right[2]) * self._config.offset_x_m,
        )
        yaw = float(
            np.arctan2(head_position[0] - position[0], head_position[2] - position[2])
        )
        return position, _quat_wxyz_mul(_yaw_quat_wxyz(yaw), self._IDENTITY_ROT_WXYZ)


class AuxiliaryViewportController:
    """Manage a secondary viewport that preserves the scene's initial camera view."""

    def __init__(
        self,
        *,
        config: AuxiliaryViewportConfig,
        viewport_utility: Any,
        initial_camera_path: str | None,
    ) -> None:
        self._config = config
        self._viewport_utility = viewport_utility
        self._initial_camera_path = (
            None if initial_camera_path is None else str(initial_camera_path)
        )
        self._window = None

    def show(self) -> None:
        if not self._initial_camera_path:
            return
        if self._window is None:
            self._window = self._viewport_utility.create_viewport_window(
                name=self._config.window_name,
                width=int(self._config.resolution_px[0]),
                height=int(self._config.resolution_px[1]),
                position_x=int(self._config.position_px[0]),
                position_y=int(self._config.position_px[1]),
                camera_path=self._initial_camera_path,
            )
            self._dock_to_main_viewport_async()
        if self._window is None:
            return
        self._window.viewport_api.camera_path = self._initial_camera_path
        self._window.visible = True

    def hide(self) -> None:
        if self._window is not None:
            self._window.visible = False

    def close(self) -> None:
        window = self._window
        self._window = None
        if window is None:
            return
        window.visible = False
        window.destroy()

    def _dock_to_main_viewport_async(self) -> None:
        window = self._window
        if window is None:
            return

        async def dock_window() -> None:
            import omni.ui as ui
            import omni.kit.app

            await omni.kit.app.get_app().next_update_async()
            main_viewport = ui.Workspace.get_window("Viewport")
            if main_viewport is not None and window is not None:
                window.dock_in(main_viewport, ui.DockPosition.RIGHT, 0.35)

        asyncio.ensure_future(dock_window())


class TeleopViewportLayoutController:
    """Manage the calibrated global/head/table teleop viewport layout."""

    def __init__(
        self,
        *,
        config: TeleopViewportLayoutConfig,
        viewport_utility: Any,
        main_viewport_api: Any,
        global_camera_path: str | None,
        head_camera_path: str | None,
        table_camera_path: str | None,
    ) -> None:
        self._config = config
        self._viewport_utility = viewport_utility
        self._main_viewport_api = main_viewport_api
        self._global_camera_path = (
            None if global_camera_path is None else str(global_camera_path)
        )
        self._head_camera_path = (
            None if head_camera_path is None else str(head_camera_path)
        )
        self._table_camera_path = (
            None if table_camera_path is None else str(table_camera_path)
        )
        self._head_window = None
        self._table_window = None
        self._layout_requested = False
        self._announced_missing_paths = False

    def show(self) -> None:
        if not self._config.enabled:
            return
        if (
            not self._global_camera_path
            or not self._head_camera_path
            or not self._table_camera_path
        ):
            missing = {
                "global": self._global_camera_path is None,
                "head": self._head_camera_path is None,
                "table": self._table_camera_path is None,
            }
            if not self._announced_missing_paths:
                print(
                    "Teleop viewport layout is enabled but camera paths are incomplete: "
                    f"{missing}",
                    flush=True,
                )
                self._announced_missing_paths = True
            return

        self._set_main_camera(self._global_camera_path)
        if self._head_window is None:
            self._head_window = self._viewport_utility.create_viewport_window(
                name=self._config.head_window_name,
                width=int(self._config.resolution_px[0]),
                height=int(self._config.resolution_px[1]),
                camera_path=self._head_camera_path,
            )
        if self._table_window is None:
            self._table_window = self._viewport_utility.create_viewport_window(
                name=self._config.table_window_name,
                width=int(self._config.resolution_px[0]),
                height=int(self._config.resolution_px[1]),
                camera_path=self._table_camera_path,
            )

        self._set_window_camera(self._head_window, self._head_camera_path)
        self._set_window_camera(self._table_window, self._table_camera_path)
        self._set_visible(True)
        if not self._layout_requested:
            self._dock_layout_async()
            self._layout_requested = True

    def hide(self) -> None:
        self._set_visible(False)
        self._set_main_camera(self._global_camera_path)

    def close(self) -> None:
        for window in (self._table_window, self._head_window):
            if window is None:
                continue
            window.visible = False
            window.destroy()
        self._table_window = None
        self._head_window = None
        self._layout_requested = False

    def _set_visible(self, visible: bool) -> None:
        if self._head_window is not None:
            self._head_window.visible = bool(visible)
        if self._table_window is not None:
            self._table_window.visible = bool(visible)

    def _set_main_camera(self, camera_path: str | None) -> None:
        if self._main_viewport_api is None or not camera_path:
            return
        self._main_viewport_api.camera_path = str(camera_path)

    def _set_window_camera(self, window: Any, camera_path: str | None) -> None:
        if window is None or not camera_path:
            return
        window.viewport_api.camera_path = str(camera_path)

    def _dock_layout_async(self) -> None:
        head_window = self._head_window
        table_window = self._table_window
        if head_window is None or table_window is None:
            return

        async def dock_windows() -> None:
            import omni.ui as ui
            import omni.kit.app

            await omni.kit.app.get_app().next_update_async()
            main_viewport = ui.Workspace.get_window("Viewport")
            if main_viewport is None:
                return
            if head_window is not None:
                head_window.dock_in(
                    main_viewport,
                    ui.DockPosition.RIGHT,
                    float(self._config.right_split_ratio),
                )
            await omni.kit.app.get_app().next_update_async()
            if table_window is not None and head_window is not None:
                table_window.dock_in(
                    head_window,
                    ui.DockPosition.BOTTOM,
                    float(self._config.table_split_ratio),
                )
            self._set_main_camera(self._global_camera_path)
            self._set_window_camera(head_window, self._head_camera_path)
            self._set_window_camera(table_window, self._table_camera_path)

        asyncio.ensure_future(dock_windows())


@dataclass(frozen=True)
class ArmIkSideConfig:
    body_name: str
    body_offset_pos: tuple[float, float, float]
    body_offset_quat_xyzw: tuple[float, float, float, float]
    target_orientation_correction_quat_xyzw: tuple[float, float, float, float]
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class ArmIkRuntimeConfig:
    status_label: str = "teleop"
    enabled: bool = True
    pose_scale: float = 1.0
    pose_z_offset: float = 0.0
    command_type: str = "pose"
    startup_relative_position: bool = True
    startup_relative_orientation: bool = True
    position_error_scale: float = 1.0
    orientation_error_scale: float = 0.35
    position_deadband_m: float = 0.003
    orientation_deadband_rad: float = 0.06
    ik_method: str = "dls"
    damping: float = 0.05
    smoothing_alpha: float = 0.35
    clamp_to_joint_limits: bool = True
    sides: dict[str, ArmIkSideConfig] = field(default_factory=dict)


def _default_config_path() -> Path:
    return default_config_path()


def _load_yaml_file(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _scene_usd_path(
    config: Mapping[str, Any],
    *,
    config_dir: Path,
    override: str | None,
) -> Path | None:
    raw_value: Any = override
    field_name = "--scene-usd"
    if raw_value is None:
        scene_config = config.get("scene", {})
        if scene_config is None:
            scene_config = {}
        if not isinstance(scene_config, Mapping):
            raise ValueError("Config field 'scene' must be a mapping")
        raw_value = scene_config.get("usd_path")
        field_name = "scene.usd_path"

    if raw_value is None:
        raw_value = config.get("scene_usd")
        field_name = "scene_usd"

    if raw_value is None:
        raw_value = config.get("layout")
        field_name = "layout"

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
    candidates = (
        [path] if path.is_absolute() else [config_dir / path, app_root() / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    resolved = candidates[0].resolve()
    raise FileNotFoundError(f"Scene USD not found from {field_name}: {resolved}")


def _robot_initial_world_position(
    config: Mapping[str, Any],
) -> tuple[float, float, float]:
    robot_config = config.get("robot", {})
    if robot_config is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    return _float_tuple(
        robot_config.get("initial_world_position"),
        name="robot.initial_world_position",
        length=3,
        default=(0.0, 0.0, 0.0),
    )


def _robot_initial_world_orientation_xyzw(
    config: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    robot_config = config.get("robot", {})
    if robot_config is None:
        return (0.0, 0.0, 0.0, 1.0)
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    # The first version of this config key was misnamed as wxyz. Keep it as a
    # short-term alias. Values are normalized xyzw; scene.py converts to wxyz for IsaacLab.
    raw_quat = robot_config.get("initial_world_orientation_xyzw")
    if raw_quat is None:
        raw_quat = robot_config.get("initial_world_orientation_wxyz")
    quat = _float_tuple(
        raw_quat,
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


def _arm_ik_runtime_config(config: Mapping[str, Any]) -> ArmIkRuntimeConfig:
    profile = robot_profile_from_config(config)
    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")

    # Keep the arm IK body bindings pinned in code rather than mirroring them in
    # the teleop YAML. These offsets are asset-specific and tuned together.
    sides = {
        "left": ArmIkSideConfig(
            body_name=profile.ik_body_names["left"],
            body_offset_pos=profile.ik_body_offset_pos["left"],
            body_offset_quat_xyzw=profile.ik_body_offset_quat_xyzw["left"],
            target_orientation_correction_quat_xyzw=(
                profile.ik_target_orientation_correction_quat_xyzw["left"]
            ),
            joint_names=ARM_JOINTS["left"],
        ),
        "right": ArmIkSideConfig(
            body_name=profile.ik_body_names["right"],
            body_offset_pos=profile.ik_body_offset_pos["right"],
            body_offset_quat_xyzw=profile.ik_body_offset_quat_xyzw["right"],
            target_orientation_correction_quat_xyzw=(
                profile.ik_target_orientation_correction_quat_xyzw["right"]
            ),
            joint_names=ARM_JOINTS["right"],
        ),
    }

    return ArmIkRuntimeConfig(
        status_label=profile.status_label,
        enabled=True,
        pose_scale=1.0,
        pose_z_offset=0.0,
        command_type="pose",
        startup_relative_position=True,
        startup_relative_orientation=True,
        position_error_scale=0.8,
        orientation_error_scale=0.8,
        position_deadband_m=0.004,
        orientation_deadband_rad=0.08,
        ik_method="dls",
        damping=0.05,
        smoothing_alpha=0.80,
        clamp_to_joint_limits=True,
        sides=sides,
    )


def _use_native_hand_targets(config: Mapping[str, Any]) -> bool:
    retargeting_config = config.get("retargeting", {})
    if retargeting_config is None:
        retargeting_config = {}
    if not isinstance(retargeting_config, Mapping):
        raise ValueError("Config field 'retargeting' must be a mapping")

    method = retargeting_config.get("method")
    if method is not None:
        normalized_method = str(method).strip().lower()
        aliases = {
            "native": "native_dex",
            "native_dex": "native_dex",
            "dex": "native_dex",
            "dexhand": "native_dex",
            "dex_hand": "native_dex",
            "official": "official_wuji",
            "official_wuji": "official_wuji",
            "wuji": "official_wuji",
        }
        normalized_method = aliases.get(normalized_method, normalized_method)
        if normalized_method == "native_dex":
            return True
        if normalized_method == "official_wuji":
            return False
        raise ValueError(
            "Config field 'retargeting.method' must be one of "
            "['native_dex', 'official_wuji']"
        )

    robot_config = config.get("robot", {})
    if robot_config is None:
        robot_config = {}
    if not isinstance(robot_config, Mapping):
        raise ValueError("Config field 'robot' must be a mapping")
    return bool(robot_config.get("use_native_hand_retargeting", False))


def _load_isaac_lab():
    from isaaclab.app import AppLauncher

    return AppLauncher


class DualArmIkController:
    """Drive G1 arm joints from left/right end-effector pose targets."""

    def __init__(
        self,
        robot: Any,
        config: ArmIkRuntimeConfig,
        *,
        device: str,
        pose_binding: AvpRobotFrameBinding | None,
    ) -> None:
        import torch
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.utils import math as math_utils

        self._robot = robot
        self._config = config
        self._pose_binding = pose_binding
        self._device = device
        self._torch = torch
        self._math_utils = math_utils
        self._controllers: dict[str, Any] = {}
        self._joint_ids: dict[str, Sequence[int] | slice] = {}
        self._joint_id_lists: dict[str, list[int]] = {}
        self._joint_names: dict[str, tuple[str, ...]] = {}
        self._body_ids: dict[str, int] = {}
        self._body_names: dict[str, str] = {}
        self._jacobi_body_ids: dict[str, int] = {}
        self._jacobi_joint_ids: dict[str, list[int]] = {}
        self._offset_pos: dict[str, Any] = {}
        self._offset_rot: dict[str, Any] = {}
        self._target_rot_correction: dict[str, Any] = {}
        self._last_targets: dict[str, Any] = {}
        self._startup_target_refs: dict[str, tuple[Any, Any, Any, Any]] = {}
        self._startup_ref_announced: set[str] = set()
        self._latest_smolvla_pose_commands: dict[str, dict[str, np.ndarray]] = {}

        ik_params = self._ik_params()
        for side, side_config in config.sides.items():
            joint_ids, resolved_joint_names = robot.find_joints(
                list(side_config.joint_names), preserve_order=True
            )
            missing = [
                name
                for name in side_config.joint_names
                if name not in resolved_joint_names
            ]
            if missing:
                raise RuntimeError(
                    f"{self._config.status_label} USD is missing {side} arm joints: {missing[:8]}"
                )

            body_ids, body_names = robot.find_bodies(side_config.body_name)
            if len(body_ids) != 1:
                raise RuntimeError(
                    f"Expected one {side} IK body match for {side_config.body_name!r}; "
                    f"got {len(body_ids)}: {body_names}"
                )

            body_idx = int(body_ids[0])
            joint_id_list = [int(joint_id) for joint_id in joint_ids]
            if getattr(robot, "is_fixed_base", False):
                jacobi_body_idx = body_idx - 1
                jacobi_joint_ids = joint_id_list
            else:
                jacobi_body_idx = body_idx
                jacobi_joint_ids = [joint_id + 6 for joint_id in joint_id_list]

            controller_cfg = DifferentialIKControllerCfg(
                command_type=config.command_type,
                use_relative_mode=False,
                ik_method=config.ik_method,
                ik_params=ik_params,
            )
            self._controllers[side] = DifferentialIKController(
                cfg=controller_cfg,
                num_envs=1,
                device=device,
            )
            self._joint_id_lists[side] = joint_id_list
            self._joint_ids[side] = (
                slice(None) if len(joint_id_list) == robot.num_joints else joint_id_list
            )
            self._joint_names[side] = tuple(str(name) for name in resolved_joint_names)
            self._body_ids[side] = body_idx
            self._body_names[side] = str(body_names[0])
            self._jacobi_body_ids[side] = int(jacobi_body_idx)
            self._jacobi_joint_ids[side] = jacobi_joint_ids
            self._offset_pos[side] = torch.tensor(
                side_config.body_offset_pos,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            self._offset_rot[side] = torch.tensor(
                side_config.body_offset_quat_xyzw,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            self._target_rot_correction[side] = self._math_utils.quat_unique(
                self._math_utils.normalize(
                    torch.tensor(
                        side_config.target_orientation_correction_quat_xyzw,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                )
            )

    def _ik_params(self) -> dict[str, float]:
        if self._config.ik_method == "dls":
            return {"lambda_val": self._config.damping}
        if self._config.ik_method == "pinv":
            return {"k_val": 1.0}
        if self._config.ik_method == "trans":
            return {"k_val": 1.0}
        if self._config.ik_method == "svd":
            return {"k_val": 1.0, "min_singular_value": 1.0e-5}
        return {}

    def _frame_pose_root(self, side: str) -> tuple[Any, Any]:
        body_idx = self._body_ids[side]
        ee_pos_w = as_torch(self._robot.data.body_pos_w)[:, body_idx]
        ee_quat_w = as_torch(self._robot.data.body_quat_w)[:, body_idx]
        root_pos_w = as_torch(self._robot.data.root_pos_w)
        root_quat_w = as_torch(self._robot.data.root_quat_w)
        ee_pos_b, ee_quat_b = self._math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        return self._math_utils.combine_frame_transforms(
            ee_pos_b,
            ee_quat_b,
            self._offset_pos[side],
            self._offset_rot[side],
        )

    def _target_pose_root(
        self, side: str, pose: Mapping[str, Any]
    ) -> tuple[Any, Any] | None:
        if self._pose_binding is not None:
            target_pose = self._pose_binding.transform_pose(side, pose)
            if target_pose is None:
                return None
            pos_w_np, quat_w_np = target_pose
        else:
            pos_w_np, quat_w_np = openxr_pose_to_isaac_pose(
                pose,
                scale=self._config.pose_scale,
                offset_z=self._config.pose_z_offset,
            )
        return self._target_pose_root_from_world_pose(
            pos_w_np,
            quat_w_np,
            side=side,
            apply_target_rot_correction=True,
        )

    def _target_pose_root_from_world_pose(
        self,
        pos_w_np: Sequence[float] | np.ndarray,
        quat_w_np: Sequence[float] | np.ndarray,
        *,
        side: str | None = None,
        apply_target_rot_correction: bool = False,
    ) -> tuple[Any, Any]:
        pos_w = self._torch.as_tensor(
            pos_w_np,
            dtype=self._torch.float32,
            device=self._device,
        ).unsqueeze(0)
        quat_w = self._torch.as_tensor(
            normalize_quat_xyzw(quat_w_np),
            dtype=self._torch.float32,
            device=self._device,
        ).unsqueeze(0)
        if apply_target_rot_correction:
            if side is None:
                raise ValueError(
                    "side is required when applying target orientation correction"
                )
            quat_w = self._math_utils.normalize(
                self._math_utils.quat_mul(quat_w, self._target_rot_correction[side])
            )
        else:
            quat_w = self._math_utils.normalize(quat_w)
        quat_w = self._math_utils.quat_unique(quat_w)
        root_pos_w = as_torch(self._robot.data.root_pos_w)
        root_quat_w = as_torch(self._robot.data.root_quat_w)
        return self._math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, pos_w, quat_w
        )

    def _startup_relative_target(
        self,
        side: str,
        target_pos: Any,
        target_quat: Any,
        ee_pos_curr: Any,
        ee_quat_curr: Any,
    ) -> tuple[Any, Any]:
        if not (
            self._config.startup_relative_position
            or self._config.startup_relative_orientation
        ):
            return target_pos, target_quat

        ref = self._startup_target_refs.get(side)
        if ref is None:
            ref = (
                target_pos.detach().clone(),
                self._math_utils.quat_unique(
                    self._math_utils.normalize(target_quat.detach().clone())
                ),
                ee_pos_curr.detach().clone(),
                self._math_utils.quat_unique(
                    self._math_utils.normalize(ee_quat_curr.detach().clone())
                ),
            )
            self._startup_target_refs[side] = ref
            self._announce_startup_relative_target(side, ref)

        target_start_pos, target_start_quat, ee_start_pos, ee_start_quat = ref
        adjusted_pos = target_pos
        adjusted_quat = target_quat

        # AVP optical wrists and the robot EE rarely share a reliable absolute
        # pose. Bind the first valid target to the current simulated EE, then
        # command startup-relative translation and rotation deltas.
        if self._config.startup_relative_position:
            adjusted_pos = ee_start_pos + (target_pos - target_start_pos)
        if self._config.startup_relative_orientation:
            target_quat = self._math_utils.quat_unique(
                self._math_utils.normalize(target_quat)
            )
            quat_delta = self._math_utils.quat_mul(
                target_quat,
                self._math_utils.quat_inv(target_start_quat),
            )
            adjusted_quat = self._math_utils.normalize(
                self._math_utils.quat_mul(quat_delta, ee_start_quat)
            )
            adjusted_quat = self._math_utils.quat_unique(adjusted_quat)
        return adjusted_pos, adjusted_quat

    def _announce_startup_relative_target(
        self,
        side: str,
        ref: tuple[Any, Any, Any, Any],
    ) -> None:
        if side in self._startup_ref_announced:
            return
        target_start_pos, target_start_quat, ee_start_pos, ee_start_quat = ref
        print(
            f"[{self._config.status_label}] {side} IK startup target reference calibrated "
            f"relative_position={self._config.startup_relative_position} "
            f"relative_orientation={self._config.startup_relative_orientation} "
            f"target_pos={np.round(target_start_pos[0].detach().cpu().numpy(), 4).tolist()} "
            f"ee_pos={np.round(ee_start_pos[0].detach().cpu().numpy(), 4).tolist()} "
            f"target_quat_xyzw={np.round(target_start_quat[0].detach().cpu().numpy(), 4).tolist()} "
            f"ee_quat_xyzw={np.round(ee_start_quat[0].detach().cpu().numpy(), 4).tolist()}",
            flush=True,
        )
        self._startup_ref_announced.add(side)

    def _compute_delta_joint_pos(self, delta_pose: Any, jacobian: Any) -> Any:
        if self._config.ik_method == "pinv":
            jacobian_pinv = self._torch.linalg.pinv(jacobian)
            return (jacobian_pinv @ delta_pose.unsqueeze(-1)).squeeze(-1)
        if self._config.ik_method == "svd":
            min_singular_value = 1.0e-5
            U, S, Vh = self._torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = self._torch.where(
                min_singular_value < S,
                S_inv,
                self._torch.zeros_like(S_inv),
            )
            jacobian_pinv = (
                self._torch.transpose(Vh, dim0=1, dim1=2)[:, :, : jacobian.shape[1]]
                @ self._torch.diag_embed(S_inv)
                @ self._torch.transpose(U, dim0=1, dim1=2)
            )
            return (jacobian_pinv @ delta_pose.unsqueeze(-1)).squeeze(-1)
        if self._config.ik_method == "trans":
            jacobian_t = self._torch.transpose(jacobian, dim0=1, dim1=2)
            return (jacobian_t @ delta_pose.unsqueeze(-1)).squeeze(-1)
        if self._config.ik_method == "dls":
            jacobian_t = self._torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_val = self._config.damping
            lambda_matrix = (lambda_val**2) * self._torch.eye(
                n=jacobian.shape[1],
                device=self._device,
            )
            return (
                jacobian_t
                @ self._torch.inverse(jacobian @ jacobian_t + lambda_matrix)
                @ delta_pose.unsqueeze(-1)
            ).squeeze(-1)
        raise ValueError(
            f"Unsupported inverse-kinematics method: {self._config.ik_method}"
        )

    def _compute_joint_targets(
        self,
        side: str,
        ee_pos_curr: Any,
        ee_quat_curr: Any,
        joint_pos: Any,
    ) -> Any:
        jacobian = self._frame_jacobian(side)
        controller = self._controllers[side]
        if self._config.command_type == "position":
            return controller.compute(
                ee_pos_curr,
                ee_quat_curr,
                jacobian,
                joint_pos,
            )

        position_error, axis_angle_error = self._math_utils.compute_pose_error(
            ee_pos_curr,
            ee_quat_curr,
            controller.ee_pos_des,
            controller.ee_quat_des,
            rot_error_type="axis_angle",
        )
        if self._config.position_deadband_m > 0.0:
            position_norm = self._torch.linalg.norm(position_error, dim=1, keepdim=True)
            position_error = self._torch.where(
                position_norm <= self._config.position_deadband_m,
                self._torch.zeros_like(position_error),
                position_error,
            )
        if self._config.orientation_deadband_rad > 0.0:
            orientation_norm = self._torch.linalg.norm(
                axis_angle_error, dim=1, keepdim=True
            )
            axis_angle_error = self._torch.where(
                orientation_norm <= self._config.orientation_deadband_rad,
                self._torch.zeros_like(axis_angle_error),
                axis_angle_error,
            )
        # Pose-mode wrist targets are much noisier in orientation than in translation.
        # Down-weight angular error so small AVP wrist rotations do not yank the whole
        # arm into a large reconfiguration before the translational target is reached.
        position_error = position_error * self._config.position_error_scale
        axis_angle_error = axis_angle_error * self._config.orientation_error_scale
        pose_error = self._torch.cat((position_error, axis_angle_error), dim=1)
        if bool(self._torch.all(self._torch.abs(pose_error) <= 1.0e-9)):
            return joint_pos
        delta_joint_pos = self._compute_delta_joint_pos(pose_error, jacobian)
        return joint_pos + delta_joint_pos

    def _frame_jacobian(self, side: str):
        jacobian = as_torch(self._robot.root_physx_view.get_jacobians())[
            :, self._jacobi_body_ids[side], :, self._jacobi_joint_ids[side]
        ].clone()
        root_quat_w = as_torch(self._robot.data.root_quat_w)
        root_rot_b_w = self._math_utils.matrix_from_quat(
            self._math_utils.quat_inv(root_quat_w)
        )
        jacobian[:, 0:3, :] = self._torch.bmm(root_rot_b_w, jacobian[:, 0:3, :])
        jacobian[:, 3:, :] = self._torch.bmm(root_rot_b_w, jacobian[:, 3:, :])
        offset_pos = self._offset_pos[side]
        offset_rot = self._offset_rot[side]
        jacobian[:, 0:3, :] += self._torch.bmm(
            -self._math_utils.skew_symmetric_matrix(offset_pos),
            jacobian[:, 3:, :],
        )
        jacobian[:, 3:, :] = self._torch.bmm(
            self._math_utils.matrix_from_quat(offset_rot),
            jacobian[:, 3:, :],
        )
        return jacobian

    def _apply_target_pose_root(
        self,
        side: str,
        target_pose_root: tuple[Any, Any],
    ) -> None:
        ee_pos_curr, ee_quat_curr = self._frame_pose_root(side)
        target_pos, target_quat = target_pose_root
        target_pos, target_quat = self._startup_relative_target(
            side,
            target_pos,
            target_quat,
            ee_pos_curr,
            ee_quat_curr,
        )
        self._latest_smolvla_pose_commands[side] = {
            "ee_current_pos": ee_pos_curr[0].detach().cpu().numpy().astype(np.float32),
            "ee_current_quat_xyzw": (
                ee_quat_curr[0].detach().cpu().numpy().astype(np.float32)
            ),
            "ee_target_pos": target_pos[0].detach().cpu().numpy().astype(np.float32),
            "ee_target_quat_xyzw": (
                target_quat[0].detach().cpu().numpy().astype(np.float32)
            ),
        }
        command = (
            target_pos
            if self._config.command_type == "position"
            else self._torch.cat((target_pos, target_quat), dim=1)
        )
        self._controllers[side].set_command(command, ee_pos_curr, ee_quat_curr)
        joint_pos = as_torch(self._robot.data.joint_pos)[:, self._joint_ids[side]]
        joint_targets = self._compute_joint_targets(
            side,
            ee_pos_curr,
            ee_quat_curr,
            joint_pos,
        )

        if self._config.clamp_to_joint_limits:
            limits = as_torch(self._robot.data.soft_joint_pos_limits)[
                :, self._joint_ids[side], :
            ]
            joint_targets = self._torch.clamp(
                joint_targets, limits[:, :, 0], limits[:, :, 1]
            )

        previous = self._last_targets.get(side)
        if previous is not None and self._config.smoothing_alpha < 1.0:
            alpha = self._config.smoothing_alpha
            joint_targets = previous * (1.0 - alpha) + joint_targets * alpha

        self._last_targets[side] = joint_targets
        self._robot.set_joint_position_target(
            target=joint_targets,
            joint_ids=self._joint_ids[side],
        )

    def update(
        self,
        sample: Mapping[str, Any],
        *,
        active_sides: Sequence[str] = ("left", "right"),
    ) -> None:
        if not self._config.enabled:
            return

        for side in active_sides:
            if side not in self._controllers:
                continue
            pose = sample["ee_poses"].get(side)
            if pose is None:
                self._latest_smolvla_pose_commands.pop(side, None)
                target = self._last_targets.get(side)
                if target is not None:
                    self._robot.set_joint_position_target(
                        target=target,
                        joint_ids=self._joint_ids[side],
                    )
                continue

            target_pose = self._target_pose_root(side, pose)
            if target_pose is None:
                self._latest_smolvla_pose_commands.pop(side, None)
                continue
            self._apply_target_pose_root(side, target_pose)

    def reset_startup_references(self) -> None:
        self._startup_target_refs.clear()
        self._startup_ref_announced.clear()
        self._latest_smolvla_pose_commands.clear()
        print(
            f"[{self._config.status_label}] arm IK startup target references cleared; "
            "waiting for re-arm calibration.",
            flush=True,
        )

    def latest_smolvla_pose_command(self, side: str) -> dict[str, np.ndarray] | None:
        command = self._latest_smolvla_pose_commands.get(side)
        if command is None:
            return None
        return {key: value.copy() for key, value in command.items()}


class WholeBodyYawController:
    """Rotate the robot base with the calibrated head yaw around the AVP z-axis."""

    def __init__(
        self,
        robot: Any,
        config: AvpFrameBindingRuntimeConfig,
        *,
        device: str,
    ) -> None:
        import torch

        self._robot = robot
        self._config = config
        self._device = device
        self._torch = torch
        self._joint_id, self._joint_name = self._find_base_yaw_joint()
        self._joint_limits = self._joint_limits_np()
        self._reference_target = self._current_joint_pos()
        self._last_target = self._reference_target
        print(
            f"[{self._config.status_label}] whole-body yaw following active: "
            f"joint={self._joint_name!r}, "
            f"deadband_rad={self._config.whole_body_yaw_deadband_rad:.3f}, "
            f"smoothing_alpha={self._config.whole_body_yaw_smoothing_alpha:.2f}.",
            flush=True,
        )

    @property
    def joint_id(self) -> int:
        return self._joint_id

    def _find_base_yaw_joint(self) -> tuple[int, str]:
        available = tuple(
            str(name) for name in (getattr(self._robot.data, "joint_names", ()) or ())
        )
        for candidate in ("base_yaw", "base_yaw_joint"):
            if candidate in available:
                return available.index(candidate), candidate

        fuzzy = [
            (index, name)
            for index, name in enumerate(available)
            if "base" in name.lower() and "yaw" in name.lower()
        ]
        if len(fuzzy) == 1:
            return fuzzy[0]

        raise RuntimeError(
            f"Could not resolve the {self._config.status_label} base yaw joint. "
            f"Candidates={fuzzy[:8]} available_first={list(available[:40])}"
        )

    def _joint_limits_np(self) -> tuple[float, float] | None:
        limits = as_torch(self._robot.data.soft_joint_pos_limits)
        values = limits[0, self._joint_id].detach().cpu().numpy()
        if np.asarray(values).shape != (2,):
            return None
        return float(values[0]), float(values[1])

    def _current_joint_pos(self) -> float:
        joint_pos = as_torch(self._robot.data.joint_pos)
        return float(joint_pos[0, self._joint_id].detach().cpu().item())

    def _apply_target(self, target_value: float) -> None:
        target_tensor = self._torch.tensor(
            [[float(target_value)]],
            dtype=self._torch.float32,
            device=self._device,
        )
        self._robot.set_joint_position_target(
            target=target_tensor,
            joint_ids=[self._joint_id],
        )

    def reset_reference(self) -> None:
        current = self._current_joint_pos()
        self._reference_target = current
        self._last_target = current
        self._apply_target(current)

    def hold_current_target(self) -> None:
        self._apply_target(self._last_target)

    def current_yaw_delta_rad(self) -> float:
        return float(self._current_joint_pos() - self._reference_target)

    def update(self, pose_binding: AvpRobotFrameBinding | None) -> None:
        if pose_binding is None or not pose_binding.whole_body_yaw_following_enabled:
            return

        target_value = self._reference_target + pose_binding.latest_head_yaw_delta_rad()
        if self._joint_limits is not None:
            target_value = float(
                np.clip(target_value, self._joint_limits[0], self._joint_limits[1])
            )

        alpha = self._config.whole_body_yaw_smoothing_alpha
        target_value = self._last_target * (1.0 - alpha) + target_value * alpha
        self._last_target = float(target_value)
        self._apply_target(self._last_target)


def _find_required_joint_ids(
    robot: Any,
    joint_names: Sequence[str],
    side: str,
    *,
    status_label: str,
) -> list[int]:
    joint_ids, resolved_names = robot.find_joints(
        list(joint_names), preserve_order=True
    )
    missing = [name for name in joint_names if name not in resolved_names]
    if missing:
        available = list(getattr(robot.data, "joint_names", []) or [])
        raise RuntimeError(
            f"{status_label} USD is missing {side} hand joints. "
            f"Missing first entries: {missing[:8]}; available count={len(available)}"
        )
    return list(joint_ids)


def _configured_initial_positions(
    joint_names: Sequence[str], config: WujiHandRuntimeConfig
) -> np.ndarray:
    return np.asarray(
        [
            config.initial_joint_positions.get(name, config.initial_joint_position)
            for name in joint_names
        ],
        dtype=np.float32,
    )


def _default_joint_positions(
    robot: Any,
    joint_ids: Sequence[int],
    joint_names: Sequence[str],
    config: WujiHandRuntimeConfig,
) -> np.ndarray:
    default_joint_pos = as_torch(robot.data.default_joint_pos)
    values = default_joint_pos[0, list(joint_ids)].detach().cpu().numpy()
    return values.astype(np.float32)


def _soft_joint_limits(robot: Any, joint_ids: Sequence[int]) -> np.ndarray | None:
    limits = as_torch(robot.data.soft_joint_pos_limits)
    values = limits[0, list(joint_ids), :].detach().cpu().numpy()
    if values.shape != (len(joint_ids), 2):
        return None
    return values.astype(np.float32)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    AppLauncher = _load_isaac_lab()
    parser = argparse.ArgumentParser(
        description="Run a minimal G1 Isaac Lab scene driven by AVP + MANUS teleop."
    )
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument(
        "--input-profile",
        choices=INPUT_PROFILE_CHOICES,
        default=INPUT_PROFILE_CONFIG,
        help=(
            "Input-source override: config keeps YAML settings, avp-manus uses "
            "OpenXR hand wrist EE poses, vr-manus uses controller aim EE poses."
        ),
    )
    parser.add_argument("--robot-usd", default=None)
    parser.add_argument("--robot-prim", default=None)
    parser.add_argument(
        "--scene-usd",
        default=None,
        help=(
            "Optional local USD scene path. Overrides scene.usd_path, scene_usd, "
            "or layout from the YAML config."
        ),
    )
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument(
        "--robot-height",
        type=float,
        default=None,
        help="Optional override for robot.initial_world_position z.",
    )
    parser.add_argument("--light-intensity", type=float, default=3000.0)
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help="Load the minimal scene and robot without starting OpenXR/MANUS teleop.",
    )
    parser.add_argument(
        "--smolvla-data-dir",
        default=None,
        help=(
            "Directory for raw SmolVLA samples. Defaults to "
            "examples/g1_wuji_teleop/data/dataset."
        ),
    )
    parser.add_argument(
        "--smolvla-task",
        default="Teleoperate the right arm and Wuji dexterous hand.",
        help="Fallback SmolVLA task string; key-selected tasks take precedence.",
    )
    parser.add_argument(
        "--smolvla-task-1",
        default="Place the cylinder on the red block.",
        help="SmolVLA task selected by keyboard 1 after reset.",
    )
    parser.add_argument(
        "--smolvla-task-2",
        default="Place the cylinder on the blue block.",
        help="SmolVLA task selected by keyboard 2 after reset.",
    )
    parser.add_argument(
        "--smolvla-sample-fps",
        type=float,
        default=15.0,
        help="Maximum raw SmolVLA sample rate while recording; <=0 records every frame.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ensure_import_paths()
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config_path = Path(args.config).expanduser().resolve()
    teleop_config = apply_input_profile_config(
        _load_yaml_file(config_path), args.input_profile
    )
    teleop_config["_config_dir"] = str(config_path.parent)
    robot_profile = robot_profile_from_config(teleop_config)
    status_label = robot_profile.status_label
    use_native_hand_targets = _use_native_hand_targets(teleop_config)
    head_view_camera_config = _head_view_camera_config(teleop_config)
    table_overhead_camera_config = _table_overhead_camera_config(teleop_config)
    head_view_xr_display_config = _head_view_xr_display_config(teleop_config)
    auxiliary_viewport_config = _auxiliary_viewport_config(teleop_config)
    left_side_freeze_config = _left_side_freeze_config(teleop_config)
    teleop_viewport_layout_config = _teleop_viewport_layout_config(teleop_config)
    contact_optimization_config = _contact_optimization_config(teleop_config)
    initial_world_position = _robot_initial_world_position(teleop_config)
    initial_world_orientation_xyzw = _robot_initial_world_orientation_xyzw(
        teleop_config
    )
    wuji_hand_config = wuji_hand_runtime_config(teleop_config)
    arm_ik_config = _arm_ik_runtime_config(teleop_config)
    avp_frame_binding_config = avp_frame_binding_runtime_config(teleop_config)
    scene_usd = _scene_usd_path(
        teleop_config,
        config_dir=config_path.parent,
        override=args.scene_usd,
    )
    if args.robot_height is not None:
        initial_world_position = (
            initial_world_position[0],
            initial_world_position[1],
            float(args.robot_height),
        )
    robot_usd = (
        Path(args.robot_usd).expanduser().resolve()
        if args.robot_usd
        else resolve_repo_relative_path(robot_profile.default_usd_relpath)
    )
    if not robot_usd.is_file():
        raise FileNotFoundError(f"{status_label} USD not found: {robot_usd}")
    robot_prim = str(args.robot_prim or robot_profile.robot_prim)

    AppLauncher = _load_isaac_lab()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    _ensure_cloudxr_runtime_environment_defaults()

    import carb
    import omni.usd
    import torch
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    _ensure_carb_input_available()
    omni_appwindow = _import_omni_appwindow()
    omni_viewport_utility = _import_omni_viewport_utility()

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, -4.0, 2.0], [0.0, 0.0, 0.9])

    scene_config = G1WujiSceneConfig(
        robot_prim=robot_prim,
        robot_usd=robot_usd,
        initial_world_position=initial_world_position,
        initial_world_orientation_xyzw=initial_world_orientation_xyzw,
        initial_hand_joint_positions=wuji_hand_config.initial_joint_positions,
        light_intensity=args.light_intensity,
        scene_usd=scene_usd,
        contact_optimization=contact_optimization_config,
    )
    robot = design_g1_wuji_scene(scene_config)

    sim.reset()
    sim_dt = sim.get_physics_dt()
    left_hand_runtime_enabled = not left_side_freeze_config.enabled_from_start
    if left_hand_runtime_enabled:
        left_joint_ids = _find_required_joint_ids(
            robot,
            wuji_hand_config.left_joint_names,
            "left",
            status_label=status_label,
        )
    else:
        left_joint_ids = []
    right_joint_ids = _find_required_joint_ids(
        robot,
        wuji_hand_config.right_joint_names,
        "right",
        status_label=status_label,
    )
    left_initial_targets = (
        _default_joint_positions(
            robot, left_joint_ids, wuji_hand_config.left_joint_names, wuji_hand_config
        )
        if left_hand_runtime_enabled
        else np.empty((0,), dtype=np.float32)
    )
    right_initial_targets = _default_joint_positions(
        robot, right_joint_ids, wuji_hand_config.right_joint_names, wuji_hand_config
    )
    left_postprocessor = (
        HandTargetPostProcessor(
            joint_names=wuji_hand_config.left_joint_names,
            initial_targets=left_initial_targets,
            joint_limits=_soft_joint_limits(robot, left_joint_ids),
            config=wuji_hand_config,
            side="left",
        )
        if left_hand_runtime_enabled
        else None
    )
    right_postprocessor = HandTargetPostProcessor(
        joint_names=wuji_hand_config.right_joint_names,
        initial_targets=right_initial_targets,
        joint_limits=_soft_joint_limits(robot, right_joint_ids),
        config=wuji_hand_config,
        side="right",
    )
    wuji_backend = (
        None if use_native_hand_targets else WujiHandTargetBackend(wuji_hand_config)
    )
    avp_frame_binding = (
        AvpRobotFrameBinding(
            robot,
            arm_ik_config,
            avp_frame_binding_config,
            pose_scale=arm_ik_config.pose_scale,
            pose_z_offset=arm_ik_config.pose_z_offset,
        )
        if avp_frame_binding_config.enabled
        else None
    )
    whole_body_yaw = (
        WholeBodyYawController(
            robot,
            avp_frame_binding_config,
            device=sim.device,
        )
        if avp_frame_binding_config.enabled
        and avp_frame_binding_config.whole_body_yaw_following
        else None
    )
    arm_ik = (
        DualArmIkController(
            robot,
            arm_ik_config,
            device=sim.device,
            pose_binding=avp_frame_binding,
        )
        if arm_ik_config.enabled
        else None
    )
    stage = omni.usd.get_context().get_stage()
    robot_root_prim = stage.GetPrimAtPath(robot_prim) if stage is not None else None
    _disable_robot_transmissive_materials(
        stage=stage,
        robot_root_prim=robot_root_prim,
        status_label=status_label,
    )
    head_link_prim = (
        _find_head_like_prim(stage, robot_root_prim, robot_prim)
        if robot_root_prim is not None and robot_root_prim.IsValid()
        else None
    )
    if head_link_prim is None:
        head_candidates = []
        if robot_root_prim is not None and robot_root_prim.IsValid():
            head_candidates = [
                str(prim.GetPath())
                for prim in _walk_child_prims(robot_root_prim)
                if "head" in prim.GetName().lower()
            ][:8]
        print(
            f"[{status_label}] warning: could not find a head-like prim under {robot_prim}; "
            f"candidates={head_candidates}.",
            flush=True,
        )
    else:
        print(
            f"[{status_label}] head camera uses prim {head_link_prim.GetPath()}",
            flush=True,
        )
    xform_cache = UsdGeom.XformCache()
    viewport_api = omni_viewport_utility.get_active_viewport()
    default_viewport_camera_path = (
        str(viewport_api.camera_path)
        if viewport_api is not None
        and getattr(viewport_api, "camera_path", None) is not None
        else None
    )
    auxiliary_viewport = AuxiliaryViewportController(
        config=auxiliary_viewport_config,
        viewport_utility=omni_viewport_utility,
        initial_camera_path=default_viewport_camera_path,
    )
    head_view_camera = (
        HeadViewCameraController(
            stage=stage,
            UsdGeom=UsdGeom,
            Gf=Gf,
            viewport_api=viewport_api,
            config=head_view_camera_config,
            head_link_prim=head_link_prim,
            avp_frame_binding=avp_frame_binding,
            whole_body_yaw=whole_body_yaw,
        )
        if head_view_camera_config.enabled and stage is not None
        else None
    )
    table_overhead_camera = (
        TableOverheadCameraController(
            stage=stage,
            UsdGeom=UsdGeom,
            Gf=Gf,
            viewport_api=viewport_api,
            config=table_overhead_camera_config,
        )
        if table_overhead_camera_config.enabled and stage is not None
        else None
    )
    if table_overhead_camera is not None:
        print(
            f"[{status_label}] table overhead camera at "
            f"{table_overhead_camera.camera_path}.",
            flush=True,
        )
    prop_randomization_rng = np.random.default_rng()
    prop_randomization_specs = {} if scene_usd is not None else PROP_RANDOMIZATION_SPECS
    prop_rigid_body_views: dict[str, Any] = {}
    prop_default_transforms: dict[str, Any] = {}
    blue_cube_color = (0.05, 0.22, 0.95)
    red_cube_color = (0.92, 0.06, 0.04)
    cube_colors_swapped = False
    cube_color_mapping = {
        "BlueCube": "blue",
        "RedCube": "red",
    }

    def get_prop_rigid_body_view(prim_path: str) -> Any:
        view = prop_rigid_body_views.get(prim_path)
        if view is not None:
            return view
        physics_sim_view = sim.physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not available.")
        view = physics_sim_view.create_rigid_body_view(prim_path)
        if _rigid_body_view_count(view) != 1:
            raise RuntimeError(
                f"Expected one rigid body for {prim_path}, "
                f"got {_rigid_body_view_count(view)}."
            )
        prop_rigid_body_views[prim_path] = view
        prop_default_transforms[prim_path] = _rigid_body_view_transforms_to_torch(
            view, device=sim.device
        )
        return view

    def set_shape_display_color(
        prim_path: str, color: tuple[float, float, float]
    ) -> None:
        if stage is None:
            return
        mesh_prim = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
        if mesh_prim is None or not mesh_prim.IsValid():
            print(
                f"[{status_label}] warning: cannot set color, mesh not found for {prim_path}",
                flush=True,
            )
            return
        color_attr = UsdGeom.Gprim(mesh_prim).CreateDisplayColorAttr()
        color_attr.Set([Gf.Vec3f(*(float(value) for value in color))])

    def randomize_cube_colors() -> None:
        nonlocal cube_colors_swapped, cube_color_mapping
        if stage is None or not prop_randomization_specs:
            return
        cube_colors_swapped = bool(prop_randomization_rng.integers(0, 2))
        if cube_colors_swapped:
            blue_path_color = red_cube_color
            red_path_color = blue_cube_color
            cube_color_mapping = {
                "BlueCube": "red",
                "RedCube": "blue",
            }
        else:
            blue_path_color = blue_cube_color
            red_path_color = red_cube_color
            cube_color_mapping = {
                "BlueCube": "blue",
                "RedCube": "red",
            }

        set_shape_display_color(BLUE_CUBE_PRIM_PATH, blue_path_color)
        set_shape_display_color(RED_CUBE_PRIM_PATH, red_path_color)
        print(
            f"[{status_label}] randomized cube colors: "
            f"BlueCube appears {cube_color_mapping['BlueCube']}, "
            f"RedCube appears {cube_color_mapping['RedCube']}",
            flush=True,
        )

    def randomize_scene_props() -> None:
        if stage is None or not prop_randomization_specs:
            return
        placements: dict[str, tuple[float, float, float]] = {}
        for prim_path, spec in prop_randomization_specs.items():
            center = tuple(float(value) for value in spec["center"])
            x, y = _sample_xy_in_disk(
                prop_randomization_rng,
                (center[0], center[1]),
                float(spec["radius_m"]),
            )
            translation = (x, y, center[2])
            _set_prim_translation(stage, prim_path, translation)
            # Dynamic rigid bodies are owned by live PhysX after ``sim.reset``;
            # USD xform edits alone can be overwritten on the next physics step.
            view = get_prop_rigid_body_view(prim_path)
            _set_rigid_body_view_transform_and_stop(
                view,
                prop_default_transforms[prim_path],
                translation,
                device=sim.device,
            )
            placements[prim_path.rsplit("/", 1)[-1]] = translation
        xform_cache.Clear()
        print(
            f"[{status_label}] randomized props xy: "
            + ", ".join(
                f"{name}=({position[0]:.3f}, {position[1]:.3f})"
                for name, position in placements.items()
            ),
            flush=True,
        )
        randomize_cube_colors()

    for prop_prim_path in prop_randomization_specs:
        get_prop_rigid_body_view(prop_prim_path)

    teleop_viewport_layout = TeleopViewportLayoutController(
        config=teleop_viewport_layout_config,
        viewport_utility=omni_viewport_utility,
        main_viewport_api=viewport_api,
        global_camera_path=default_viewport_camera_path,
        head_camera_path=(
            None if head_view_camera is None else head_view_camera.camera_path
        ),
        table_camera_path=(
            None if table_overhead_camera is None else table_overhead_camera.camera_path
        ),
    )
    smolvla_data = _load_smolvla_data_sample_module()
    smolvla_camera_specs: dict[str, dict[str, Any]] = {}
    if head_view_camera is not None:
        smolvla_camera_specs["front"] = {
            "camera_path": head_view_camera.camera_path,
            "resolution_px": head_view_camera_config.resolution_px,
        }
    if table_overhead_camera is not None:
        smolvla_camera_specs["table"] = {
            "camera_path": table_overhead_camera.camera_path,
            "resolution_px": table_overhead_camera_config.resolution_px,
        }
    smolvla_recorder = smolvla_data.SmolVLADataRecorder(
        output_root=args.smolvla_data_dir,
        task=args.smolvla_task,
        sample_fps=args.smolvla_sample_fps,
        camera_specs=smolvla_camera_specs,
        replicator_factory=_import_omni_replicator_core,
        status_label=status_label,
    )
    smolvla_tasks = {
        1: str(args.smolvla_task_1),
        2: str(args.smolvla_task_2),
    }
    teleop: TeleopMain | None = None
    teleop_started = False
    head_view_xr_display: HeadViewXrDisplayBridge | None = None
    scene_warmed_up = False
    if head_view_xr_display_config.enabled and not args.scene_only:
        if head_view_camera is None:
            raise ValueError(
                "robot.head_view_xr_display.enabled requires robot.head_view_camera.enabled "
                "so there is a camera prim to publish into AVP."
            )
    robot_default_joint_pos = as_torch(robot.data.default_joint_pos).clone()
    robot_default_joint_vel = as_torch(robot.data.default_joint_vel).clone()
    left_elbow_lock_joint_ids, left_elbow_lock_joint_names = robot.find_joints(
        [LEFT_ELBOW_LOCK_JOINT],
        preserve_order=True,
    )
    if LEFT_ELBOW_LOCK_JOINT not in left_elbow_lock_joint_names:
        available = list(getattr(robot.data, "joint_names", []) or [])
        raise RuntimeError(
            f"{status_label} USD is missing locked joint {LEFT_ELBOW_LOCK_JOINT!r}; "
            f"available count={len(available)}"
        )
    left_elbow_lock_joint_id = int(left_elbow_lock_joint_ids[0])
    left_elbow_lock_target = robot_default_joint_pos[
        :, [left_elbow_lock_joint_id]
    ].clone()

    def hold_locked_arm_joints() -> None:
        robot.set_joint_position_target(
            target=left_elbow_lock_target,
            joint_ids=[left_elbow_lock_joint_id],
        )

    if left_hand_runtime_enabled:
        robot.set_joint_position_target(
            target=torch.as_tensor(left_initial_targets, device=sim.device).unsqueeze(
                0
            ),
            joint_ids=left_joint_ids,
        )
    robot.set_joint_position_target(
        target=torch.as_tensor(right_initial_targets, device=sim.device).unsqueeze(0),
        joint_ids=right_joint_ids,
    )
    hold_locked_arm_joints()

    print(
        f"{status_label} teleop scene ready: "
        f"scene={'builtin_grasp' if scene_usd is None else str(scene_usd)}, "
        f"{len(left_joint_ids)} left hand joints, {len(right_joint_ids)} right hand joints, "
        f"hand_retarget={'native_dex' if use_native_hand_targets else wuji_hand_config.retarget_backend}, "
        f"arm_ik={arm_ik_config.enabled}, "
        f"left_elbow_lock={float(left_elbow_lock_target[0, 0].detach().cpu().item()):.4f}rad, "
        f"avp_frame_binding={avp_frame_binding_config.enabled}, "
        f"whole_body_yaw={whole_body_yaw is not None}, "
        f"head_tilt={avp_frame_binding_config.head_tilt_following}, "
        f"head_view_camera={head_view_camera is not None}, "
        f"table_overhead_camera={table_overhead_camera is not None}, "
        f"head_view_xr_display={head_view_xr_display_config.enabled and not args.scene_only}, "
        f"left_side_freeze_from_start={left_side_freeze_config.enabled_from_start}, "
        f"left_side_freeze_after_calibration={left_side_freeze_config.enabled_after_calibration}, "
        f"teleop_viewport_layout={teleop_viewport_layout_config.enabled and not args.scene_only}, "
        f"contact_optimization={contact_optimization_config.enabled}, "
        f"auxiliary_viewport={auxiliary_viewport_config.enabled and not args.scene_only and not teleop_viewport_layout_config.enabled}."
    )
    print(
        f"[{status_label}] SmolVLA task keys: "
        f"1={smolvla_tasks[1]!r}, 2={smolvla_tasks[2]!r}. "
        "D starts recording; D or R finishes and saves an active recording; "
        "G discards the active recording.",
        flush=True,
    )

    def step_scene_once() -> None:
        robot.write_data_to_sim()
        sim.step(render=_sim_step_should_render(sim))
        robot.update(sim_dt)
        if head_view_camera is not None:
            head_view_camera.update(xform_cache=xform_cache)
        if head_view_xr_display is not None:
            head_view_xr_display.submit_frame()

    if args.scene_only:
        print("Scene-only mode: OpenXR/MANUS teleop is not started.")
        start_time = time.monotonic()
        while simulation_app.is_running():
            if (
                args.duration_s > 0.0
                and time.monotonic() - start_time >= args.duration_s
            ):
                break
            step_scene_once()
        simulation_app.close()
        return 0

    teleop_armed = False
    start_waiting_announced = False
    keyboard_start_requested = False
    keyboard_reset_requested = False
    keyboard_data_requested = False
    keyboard_discard_requested = False
    keyboard_task_requested: int | None = None
    selected_smolvla_task_id: int | None = None
    selected_smolvla_task: str | None = None
    smolvla_hand_target_fallback_announced = False

    app_window = omni_appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()

    def keyboard_input_matches(value: Any, names: Sequence[str]) -> bool:
        keyboard_input = carb.input.KeyboardInput
        for name in names:
            enum_value = getattr(keyboard_input, name, None)
            if enum_value is not None and value == enum_value:
                return True
        value_name = str(value).rsplit(".", 1)[-1].upper()
        return value_name in {name.upper() for name in names}

    def on_keyboard_event(event, *args, **kwargs):
        nonlocal keyboard_data_requested, keyboard_reset_requested
        nonlocal keyboard_discard_requested
        nonlocal keyboard_start_requested, keyboard_task_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.B:
                keyboard_start_requested = True
            elif event.input == carb.input.KeyboardInput.R:
                keyboard_reset_requested = True
            elif event.input == carb.input.KeyboardInput.D:
                keyboard_data_requested = True
            elif keyboard_input_matches(event.input, ("G", "KEY_G")):
                keyboard_discard_requested = True
            elif keyboard_input_matches(
                event.input,
                ("KEY_1", "DIGIT_1", "NUM_1", "N1", "ONE", "K1", "NUMBER_1"),
            ):
                keyboard_task_requested = 1
            elif keyboard_input_matches(
                event.input,
                ("KEY_2", "DIGIT_2", "NUM_2", "N2", "TWO", "K2", "NUMBER_2"),
            ):
                keyboard_task_requested = 2
        return True

    keyboard_subscription = (
        input_interface.subscribe_to_keyboard_events(
            app_window.get_keyboard(),
            on_keyboard_event,
        )
        if app_window is not None and app_window.get_keyboard() is not None
        else None
    )

    def hold_robot_targets() -> None:
        if left_hand_runtime_enabled:
            robot.set_joint_position_target(
                target=torch.as_tensor(
                    left_initial_targets, device=sim.device
                ).unsqueeze(0),
                joint_ids=left_joint_ids,
            )
        robot.set_joint_position_target(
            target=torch.as_tensor(right_initial_targets, device=sim.device).unsqueeze(
                0
            ),
            joint_ids=right_joint_ids,
        )
        if left_hand_runtime_enabled:
            hold_locked_arm_joints()
        if whole_body_yaw is not None:
            whole_body_yaw.hold_current_target()

    def restore_robot_initial_pose() -> None:
        if hasattr(robot, "write_joint_state_to_sim"):
            robot.write_joint_state_to_sim(
                robot_default_joint_pos,
                robot_default_joint_vel,
            )
        robot.set_joint_position_target(
            target=robot_default_joint_pos,
        )
        if not left_side_freeze_config.enabled_from_start:
            hold_locked_arm_joints()
        if whole_body_yaw is not None:
            whole_body_yaw.reset_reference()
        if arm_ik is not None:
            arm_ik._last_targets.clear()

    def reset_teleop_calibration() -> None:
        if avp_frame_binding is not None:
            avp_frame_binding.reset_calibration()
        if whole_body_yaw is not None:
            whole_body_yaw.reset_reference()
        if arm_ik is not None:
            arm_ik.reset_startup_references()
        if head_view_camera is not None:
            head_view_camera.deactivate()
        if table_overhead_camera is not None:
            table_overhead_camera.deactivate()
        if teleop_viewport_layout_config.activate_on_calibration:
            teleop_viewport_layout.hide()
        if auxiliary_viewport_config.activate_on_calibration:
            auxiliary_viewport.hide()
        if (
            head_view_xr_display is not None
            and head_view_xr_display_config.activate_on_calibration
        ):
            head_view_xr_display.set_visible(False)

    def close_openxr_runtime() -> None:
        nonlocal teleop, teleop_started, head_view_xr_display
        if teleop_started and teleop is not None:
            teleop.__exit__(None, None, None)
        teleop = None
        teleop_started = False
        if head_view_xr_display is not None:
            head_view_xr_display.close()
            head_view_xr_display = None
        teleop_viewport_layout.close()
        auxiliary_viewport.close()

    def activate_null_teleop(reason: str) -> None:
        nonlocal teleop, teleop_started, head_view_xr_display
        if teleop_started and teleop is not None:
            try:
                teleop.__exit__(None, None, None)
            except Exception as close_exc:
                print(
                    f"[{status_label}] ignored teleop close error during "
                    f"device fallback: {type(close_exc).__name__}: {close_exc}",
                    flush=True,
                )
        teleop = None
        teleop_started = False
        if head_view_xr_display is not None:
            try:
                head_view_xr_display.close()
            except Exception as close_exc:
                print(
                    f"[{status_label}] ignored XR display close error during "
                    f"device fallback: {type(close_exc).__name__}: {close_exc}",
                    flush=True,
                )
            head_view_xr_display = None

        teleop = NullTeleopMain(
            teleop_config,
            config_dir=config_path.parent,
            reason=reason,
        )
        teleop.__enter__()
        teleop_started = True

    def start_openxr_runtime() -> None:
        nonlocal teleop, teleop_started, head_view_xr_display
        teleop_candidate = None
        try:
            teleop_candidate = TeleopMain(
                teleop_config,
                config_dir=config_path.parent,
            )
            if head_view_xr_display_config.enabled:
                head_view_xr_display = HeadViewXrDisplayBridge(
                    config=head_view_xr_display_config,
                    camera_prim_path=head_view_camera.camera_path,
                    required_xr_extensions=teleop_candidate.required_xr_extensions,
                    sim_device=str(args.device),
                )
                teleop_candidate.set_oxr_handles(head_view_xr_display.oxr_handles)
                if (
                    not head_view_xr_display_config.activate_on_calibration
                    or not avp_frame_binding_config.enabled
                ):
                    head_view_xr_display.set_visible(True)
            teleop_candidate.__enter__()
        except Exception as exc:
            if teleop_candidate is not None:
                try:
                    teleop_candidate.__exit__(None, None, None)
                except Exception as close_exc:
                    print(
                        f"[{status_label}] ignored partial teleop close error "
                        f"during device fallback: {type(close_exc).__name__}: "
                        f"{close_exc}",
                        flush=True,
                    )
            reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[{status_label}] OpenXR teleop startup failed; "
                f"continuing simulation with zero device inputs. reason={reason}",
                flush=True,
            )
            activate_null_teleop(reason)
            return

        teleop = teleop_candidate
        teleop_started = True
        if head_view_xr_display is not None:
            print(
                f"[{status_label}] AVP head-view XR display enabled: "
                f"resolution={head_view_xr_display_config.resolution_px[0]}x"
                f"{head_view_xr_display_config.resolution_px[1]}, "
                f"lock_mode={head_view_xr_display_config.lock_mode}, "
                f"activate_on_calibration={head_view_xr_display_config.activate_on_calibration}.",
                flush=True,
            )
        print(
            f"[{status_label}] OpenXR teleop session started.",
            flush=True,
        )

    def select_smolvla_task(task_id: int) -> None:
        nonlocal selected_smolvla_task, selected_smolvla_task_id
        if smolvla_recorder.active:
            print(
                f"[{status_label}] SmolVLA task switch ignored while recording; "
                "press D/R to finish and save, or G to discard first.",
                flush=True,
            )
            return
        task = smolvla_tasks.get(int(task_id))
        if not task:
            print(
                f"[{status_label}] unknown SmolVLA task key: {task_id}",
                flush=True,
            )
            return
        selected_smolvla_task_id = int(task_id)
        selected_smolvla_task = str(task)
        smolvla_recorder.task = selected_smolvla_task
        print(
            f"[{status_label}] SmolVLA task {selected_smolvla_task_id} selected: "
            f"{selected_smolvla_task}",
            flush=True,
        )

    def clear_smolvla_task_selection() -> None:
        nonlocal selected_smolvla_task, selected_smolvla_task_id
        selected_smolvla_task_id = None
        selected_smolvla_task = None

    def smolvla_can_record() -> bool:
        return teleop_armed and (
            avp_frame_binding is None or avp_frame_binding.calibrated
        )

    def start_smolvla_recording(sample: Mapping[str, Any]) -> None:
        if smolvla_recorder.active:
            return
        if not teleop_armed:
            print(
                f"[{status_label}] SmolVLA recording ignored: press B and finish "
                "calibration before pressing D.",
                flush=True,
            )
            return
        if avp_frame_binding is not None and not avp_frame_binding.calibrated:
            print(
                f"[{status_label}] SmolVLA recording ignored: AVP frame is still "
                "calibrating; press D after calibration completes.",
                flush=True,
            )
            return
        if selected_smolvla_task is None:
            print(
                f"[{status_label}] SmolVLA recording ignored: select task 1 "
                "or task 2 before pressing D.",
                flush=True,
            )
            return
        if arm_ik is None:
            print(
                f"[{status_label}] SmolVLA recording ignored: arm IK is disabled.",
                flush=True,
            )
            return
        smolvla_recorder.task = selected_smolvla_task
        smolvla_recorder.start(
            metadata={
                "config_path": str(config_path),
                "robot_prim": robot_prim,
                "robot_usd": str(robot_usd),
                "scene_usd": None if scene_usd is None else str(scene_usd),
                "frame_count": int(sample.get("frame_count", 0)),
                "task_id": selected_smolvla_task_id,
                "task": selected_smolvla_task,
                "available_tasks": smolvla_tasks,
                "right_hand_joint_names": list(wuji_hand_config.right_joint_names),
                "camera_specs": smolvla_camera_specs,
                "cube_colors_swapped": cube_colors_swapped,
                "cube_color_mapping": dict(cube_color_mapping),
            }
        )

    def record_smolvla_sample(
        sample: Mapping[str, Any],
        *,
        right_targets: np.ndarray | None,
    ) -> None:
        nonlocal smolvla_hand_target_fallback_announced
        if not smolvla_recorder.active:
            return
        if arm_ik is None:
            return
        if sample.get("ee_poses", {}).get("right") is None:
            return
        if selected_smolvla_task is None:
            return
        pose_command = arm_ik.latest_smolvla_pose_command("right")
        if pose_command is None:
            return

        right_hand_current = (
            as_torch(robot.data.joint_pos)[0, right_joint_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if right_targets is None:
            right_hand_targets = right_hand_current.copy()
            hand_target_source = "current_joint_fallback"
            if not smolvla_hand_target_fallback_announced:
                print(
                    f"[{status_label}] MANUS right hand targets are unavailable; "
                    "recording SmolVLA hand action as current simulated hand joints. "
                    "Fix MANUS skeleton input before collecting dexterous-hand "
                    "training demonstrations.",
                    flush=True,
                )
                smolvla_hand_target_fallback_announced = True
        else:
            right_hand_targets = np.asarray(right_targets, dtype=np.float32)
            hand_target_source = "manus_retarget"
        frame = smolvla_data.build_one_arm_hand_frame(
            ee_current_pos=pose_command["ee_current_pos"],
            ee_current_quat_xyzw=pose_command["ee_current_quat_xyzw"],
            ee_target_pos=pose_command["ee_target_pos"],
            ee_target_quat_xyzw=pose_command["ee_target_quat_xyzw"],
            hand_current_joints=right_hand_current,
            hand_target_joints=right_hand_targets,
            timestamp_s=float(sample.get("timestamp_s", time.time())),
            frame_index=int(sample.get("frame_count", 0)),
            task=selected_smolvla_task,
            side="right",
            metadata={
                "task_id": selected_smolvla_task_id,
                "hand_target_source": hand_target_source,
                "teleop_armed": teleop_armed,
                "avp_calibrated": (
                    None if avp_frame_binding is None else avp_frame_binding.calibrated
                ),
                "left_side_frozen": (
                    left_side_freeze_config.enabled_from_start
                    or left_side_freeze_config.enabled_after_calibration
                ),
                "cube_colors_swapped": cube_colors_swapped,
                "cube_color_mapping": dict(cube_color_mapping),
            },
        )
        smolvla_recorder.record(frame)

    start_time = time.monotonic()
    if not simulation_app.is_running():
        has_keyboard = bool(
            app_window is not None and app_window.get_keyboard() is not None
        )
        print(
            f"[{status_label}] simulation app is not running at teleop loop start; "
            "exiting before the first teleop frame. "
            f"headless={getattr(args, 'headless', None)} "
            f"app_window={app_window is not None} "
            f"keyboard={has_keyboard}",
            flush=True,
        )
    while simulation_app.is_running():
        if args.duration_s > 0.0 and time.monotonic() - start_time >= args.duration_s:
            break

        start_button_rising = keyboard_start_requested
        keyboard_start_requested = False
        reset_button_rising = keyboard_reset_requested
        keyboard_reset_requested = False
        data_button_rising = keyboard_data_requested
        keyboard_data_requested = False
        discard_button_rising = keyboard_discard_requested
        keyboard_discard_requested = False
        task_button_rising = keyboard_task_requested
        keyboard_task_requested = None

        if discard_button_rising:
            if smolvla_recorder.discard(reason="keyboard_g"):
                data_button_rising = False
            else:
                print(
                    f"[{status_label}] SmolVLA discard ignored: no active recording.",
                    flush=True,
                )

        if reset_button_rising:
            if smolvla_recorder.active:
                smolvla_recorder.stop(reason="reset")
            teleop_armed = False
            start_waiting_announced = False
            clear_smolvla_task_selection()
            restore_robot_initial_pose()
            randomize_scene_props()
            reset_teleop_calibration()
            print(
                f"[{status_label}] teleop reset; robot returned to its initial joint pose. "
                "Press keyboard 1 or 2 to select a SmolVLA task, then B to recalibrate. "
                "R saves an active recording unless you press G to discard it first.",
                flush=True,
            )

        if task_button_rising is not None:
            select_smolvla_task(task_button_rising)

        if not scene_warmed_up:
            hold_robot_targets()
            step_scene_once()
            scene_warmed_up = True
            continue

        if teleop is None:
            start_openxr_runtime()

        try:
            sample = teleop.step()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[{status_label}] OpenXR teleop step failed; switching to zero "
                f"device inputs and continuing simulation. reason={reason}",
                flush=True,
            )
            activate_null_teleop(reason)
            sample = teleop.step()
        if data_button_rising and smolvla_recorder.active:
            smolvla_recorder.stop(reason="keyboard_d")
            data_button_rising = False
        if not teleop_armed:
            if not start_waiting_announced:
                print(
                    f"[{status_label}] teleop is disarmed; press keyboard B to recalibrate "
                    "and start teleop. Press keyboard 1/2 to select the SmolVLA task, "
                    "then press D after calibration to start recording. Press D again "
                    "or R to finish and save; press G to discard the active recording.",
                    flush=True,
                )
                start_waiting_announced = True
            if data_button_rising:
                start_smolvla_recording(sample)
            if start_button_rising:
                teleop_armed = True
                start_waiting_announced = False
                reset_teleop_calibration()
                print(
                    f"[{status_label}] teleop armed at frame={int(sample.get('frame_count', 0))}; "
                    "starting fresh AVP head-hand calibration "
                    f"(averaging {avp_frame_binding_config.calibration_window_s:.1f}s).",
                    flush=True,
                )
            else:
                hold_robot_targets()
                step_scene_once()
                continue

        left_side_frozen = left_side_freeze_config.enabled_from_start or (
            left_side_freeze_config.enabled_after_calibration
            and avp_frame_binding is not None
            and avp_frame_binding.calibrated
        )

        if use_native_hand_targets:
            raw_left_targets = (
                None
                if left_side_frozen
                else ordered_hand_targets(
                    sample,
                    "left",
                    wuji_hand_config.left_joint_names,
                )
            )
            raw_right_targets = ordered_hand_targets(
                sample,
                "right",
                wuji_hand_config.right_joint_names,
            )
        elif left_side_frozen:
            raw_left_targets = None
            raw_right_targets = wuji_backend.right_targets(sample)
        else:
            raw_left_targets, raw_right_targets = wuji_backend.targets(sample)

        if avp_frame_binding is not None:
            avp_frame_binding.update_calibration(sample)
            avp_frame_binding.update_runtime_state(sample)

        if data_button_rising:
            start_smolvla_recording(sample)

        left_targets = (
            None
            if left_side_frozen or left_postprocessor is None
            else left_postprocessor.process(raw_left_targets)
        )
        right_targets = right_postprocessor.process(raw_right_targets)

        if left_targets is not None:
            robot.set_joint_position_target(
                target=torch.as_tensor(left_targets, device=sim.device).unsqueeze(0),
                joint_ids=left_joint_ids,
            )
        if right_targets is not None:
            robot.set_joint_position_target(
                target=torch.as_tensor(right_targets, device=sim.device).unsqueeze(0),
                joint_ids=right_joint_ids,
            )
        if avp_frame_binding is not None:
            if (
                teleop_viewport_layout_config.enabled
                and teleop_viewport_layout_config.activate_on_calibration
                and avp_frame_binding.calibrated
            ):
                teleop_viewport_layout.show()
            if (
                auxiliary_viewport_config.enabled
                and auxiliary_viewport_config.activate_on_calibration
                and not teleop_viewport_layout_config.enabled
                and avp_frame_binding.calibrated
            ):
                auxiliary_viewport.show()
            if (
                head_view_xr_display is not None
                and head_view_xr_display_config.activate_on_calibration
                and avp_frame_binding.calibrated
            ):
                head_view_xr_display.set_visible(True)
        if whole_body_yaw is not None:
            whole_body_yaw.update(avp_frame_binding)
        if arm_ik is not None:
            arm_ik.update(
                sample,
                active_sides=("right",) if left_side_frozen else ("left", "right"),
            )
        if left_hand_runtime_enabled:
            hold_locked_arm_joints()

        if smolvla_can_record():
            record_smolvla_sample(sample, right_targets=right_targets)

        step_scene_once()

    if keyboard_subscription is not None:
        unsubscribe_fn = getattr(
            input_interface, "unsubscribe_from_keyboard_events", None
        )
        if callable(unsubscribe_fn) and app_window is not None:
            unsubscribe_fn(
                app_window.get_keyboard(),
                keyboard_subscription,
            )
    smolvla_recorder.close()
    close_openxr_runtime()
    simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
