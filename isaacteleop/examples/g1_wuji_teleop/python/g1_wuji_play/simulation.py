"""Isaac simulation backend for G1-Wuji SmolVLA playback.

This backend deliberately reuses the existing G1-Wuji teleop scene and IK
helpers instead of maintaining a separate robot setup.

中文说明：
这个文件运行在 Isaac Lab 环境里，只负责仿真、相机、当前状态读取和动作执行。
模型推理不在这里发生，而是通过 ``RemoteSmolVLAClient`` 请求另一个
``lerobot-smolvla`` 进程里的 policy server。
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from .action import (
    HAND_DIM,
    decode_action,
    quat_to_rotvec_xyzw,
)
from .policy_rpc import RemoteSmolVLAClient


def add_isaac_app_launcher_args(parser: Any) -> None:
    # 只有 --backend=isaac 时才加载 Isaac Lab 的 AppLauncher 参数。
    # dry-run/推理环境不需要也不应该 import isaaclab。
    from g1_wuji_teleop.session import _load_isaac_lab

    AppLauncher = _load_isaac_lab()
    AppLauncher.add_app_launcher_args(parser)


def _make_state(*, arm_ik: Any, robot: Any, right_joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from g1_wuji_teleop.robots.g1_wuji.runtime import as_torch

    # 按训练时的数据定义构造 observation.state:
    # [ee.x, ee.y, ee.z, ee.rx, ee.ry, ee.rz, 20 个右手当前关节]
    ee_pos, ee_quat = arm_ik._frame_pose_root("right")
    hand_current = as_torch(robot.data.joint_pos)[0, right_joint_ids].detach().cpu().numpy()
    state = np.zeros((6 + HAND_DIM,), dtype=np.float32)
    state[:3] = ee_pos[0].detach().cpu().numpy().astype(np.float32)
    current_quat = ee_quat[0].detach().cpu().numpy().astype(np.float32)
    state[3:6] = quat_to_rotvec_xyzw(current_quat)
    state[6:] = hand_current.astype(np.float32)
    return state, current_quat


def _soft_limits_for_ids(robot: Any, joint_ids: list[int]) -> np.ndarray | None:
    from g1_wuji_teleop.robots.g1_wuji.runtime import as_torch

    # 右手 action 后 20 维会作为关节目标；执行前用 soft limit 做保护。
    limits = as_torch(robot.data.soft_joint_pos_limits)[0, joint_ids, :].detach().cpu().numpy()
    if limits.shape != (len(joint_ids), 2):
        return None
    return limits.astype(np.float32)


def _prompt_for_task(args: Any, *, status_label: str) -> str:
    # 进入闭环推理前明确收集语言模态。不要静默使用默认 task，
    # 否则用户会以为模型只看了图像和状态。
    preset = str(args.task).strip() if args.task is not None else ""
    print(f"[{status_label}] enter a task, then press Enter to start policy inference.", flush=True)
    while True:
        prompt = f"[{status_label}] task"
        if preset:
            prompt += f" [{preset}]"
        prompt += ": "
        try:
            value = input(prompt).strip()
        except EOFError as exc:
            if preset:
                value = preset
            else:
                raise RuntimeError(
                    f"[{status_label}] task input is required before starting policy inference"
                ) from exc

        if value:
            print(f"[{status_label}] using task: {value!r}", flush=True)
            return value
        if preset:
            print(f"[{status_label}] using task: {preset!r}", flush=True)
            return preset
        print(f"[{status_label}] task must not be empty.", flush=True)


def _keyboard_input_matches(value: Any, names: Sequence[str], *, carb_input: Any) -> bool:
    keyboard_input = carb_input.KeyboardInput
    for name in names:
        enum_value = getattr(keyboard_input, name, None)
        if enum_value is not None and value == enum_value:
            return True
    value_name = str(value).rsplit(".", 1)[-1].upper()
    return value_name in {name.upper() for name in names}


class ReplicatorImageSource:
    """Read front/table RGB frames from existing USD camera prims.

    复用数据采集时用过的 Replicator camera reader，这样训练和部署看到的
    图像来源一致：observation.images.front/table。
    """

    def __init__(self, camera_specs: dict[str, dict[str, Any]], *, status_label: str) -> None:
        from g1_wuji_teleop.session import _import_omni_replicator_core, _load_smolvla_data_sample_module

        smolvla_data = _load_smolvla_data_sample_module()
        rep = _import_omni_replicator_core()
        self._readers = {}
        self.last_shapes: dict[str, tuple[int, ...] | None] = {}
        self.last_missing_keys: list[str] = []
        try:
            for name, spec in camera_specs.items():
                camera_path = spec.get("camera_path")
                if not camera_path:
                    continue
                camera_spec = smolvla_data.CameraSpec(
                    name=str(name),
                    camera_path=str(camera_path),
                    resolution_px=tuple(int(v) for v in spec.get("resolution_px", (640, 480))),
                )
                self._readers[f"observation.images.{name}"] = smolvla_data._ReplicatorCameraReader(
                    rep=rep,
                    spec=camera_spec,
                )
            if not self._readers:
                raise RuntimeError(f"[{status_label}] no camera readers could be attached")
        except Exception:
            self.close()
            raise

    def read(self) -> dict[str, np.ndarray] | None:
        # Replicator 刚创建 render product 时可能返回空帧；遇到空帧就跳过本轮推理。
        images = {key: reader.read_rgb() for key, reader in self._readers.items()}
        self.last_shapes = {
            key: None if image is None else tuple(int(dim) for dim in image.shape)
            for key, image in images.items()
        }
        self.last_missing_keys = [key for key, image in images.items() if image is None]
        if any(image is None for image in images.values()):
            return None
        return {key: image for key, image in images.items() if image is not None}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


class IsaacActionExecutor:
    """Apply decoded policy commands to the reused IK controller and hand joints.

    模型输出先在 action.py 解码为末端目标位姿和右手关节目标：
    - 右臂：交给现有 DualArmIkController 求关节目标
    - 右手：直接 set_joint_position_target
    """

    def __init__(
        self,
        *,
        robot: Any,
        arm_ik: Any,
        right_joint_ids: list[int],
        hand_joint_limits: np.ndarray | None,
        sim_device: str,
    ) -> None:
        import torch

        self._robot = robot
        self._arm_ik = arm_ik
        self._right_joint_ids = right_joint_ids
        self._hand_joint_limits = hand_joint_limits
        self._sim_device = sim_device
        self._torch = torch

    def execute_one(
        self,
        *,
        state: np.ndarray,
        current_quat_xyzw: np.ndarray,
        raw_action: np.ndarray,
        max_position_delta_m: float | None,
        max_rotation_delta_rad: float | None,
    ) -> None:
        # raw_action 是从 policy server 返回的 26 维 action；这里再做限幅和坐标解码。
        command = decode_action(
            state=state,
            action=raw_action,
            current_quat_xyzw=current_quat_xyzw,
            max_position_delta_m=max_position_delta_m,
            max_rotation_delta_rad=max_rotation_delta_rad,
            hand_joint_limits=self._hand_joint_limits,
        )
        target_pos = self._torch.as_tensor(
            command.ee_target_pos,
            dtype=self._torch.float32,
            device=self._sim_device,
        ).unsqueeze(0)
        target_quat = self._torch.as_tensor(
            command.ee_target_quat_xyzw,
            dtype=self._torch.float32,
            device=self._sim_device,
        ).unsqueeze(0)
        self._arm_ik._apply_target_pose_root("right", (target_pos, target_quat))

        hand_targets = self._torch.as_tensor(
            command.hand_target_joints,
            dtype=self._torch.float32,
            device=self._sim_device,
        ).unsqueeze(0)
        self._robot.set_joint_position_target(
            target=hand_targets,
            joint_ids=self._right_joint_ids,
        )


def run_isaac_play(args: Any) -> int:
    # 这里大量复用 g1_wuji_teleop.session 里的场景/IK/相机 helper，
    # 避免为 play 再维护一份独立的 G1-Wuji 仿真搭建代码。
    from g1_wuji_teleop.devices.avp_manus_stream import INPUT_PROFILE_CONFIG, apply_input_profile_config
    from g1_wuji_teleop.paths import default_config_path, ensure_import_paths, resolve_repo_relative_path
    from g1_wuji_teleop.robots.g1_wuji.runtime import (
        as_torch,
        robot_profile_from_config,
        wuji_hand_runtime_config,
    )
    from g1_wuji_teleop.robots.g1_wuji.scene import (
        BLUE_CUBE_PRIM_PATH,
        PROP_RANDOMIZATION_SPECS,
        RED_CUBE_PRIM_PATH,
        G1WujiSceneConfig,
        design_g1_wuji_scene,
    )
    from g1_wuji_teleop.session import DualArmIkController
    from g1_wuji_teleop import session as teleop_session

    ensure_import_paths()
    config_path = Path(args.config or default_config_path()).expanduser().resolve()
    teleop_config = apply_input_profile_config(
        teleop_session._load_yaml_file(config_path),
        INPUT_PROFILE_CONFIG,
    )
    robot_profile = robot_profile_from_config(teleop_config)
    status_label = robot_profile.status_label
    wuji_hand_config = wuji_hand_runtime_config(teleop_config)
    # play 模式下模型已经输出机器人当前坐标系里的目标增量，不再使用 AVP 的
    # startup-relative 标定逻辑；否则会把模型目标再相对化一次。
    arm_ik_config = replace(
        teleop_session._arm_ik_runtime_config(teleop_config),
        startup_relative_position=False,
        startup_relative_orientation=False,
        smoothing_alpha=float(args.ik_smoothing_alpha),
    )
    head_view_camera_config = teleop_session._head_view_camera_config(teleop_config)
    table_overhead_camera_config = teleop_session._table_overhead_camera_config(teleop_config)
    contact_optimization_config = teleop_session._contact_optimization_config(teleop_config)
    initial_world_position = teleop_session._robot_initial_world_position(teleop_config)
    initial_world_orientation_xyzw = teleop_session._robot_initial_world_orientation_xyzw(teleop_config)
    if args.robot_height is not None:
        initial_world_position = (
            initial_world_position[0],
            initial_world_position[1],
            float(args.robot_height),
        )
    scene_usd = teleop_session._scene_usd_path(
        teleop_config,
        config_dir=config_path.parent,
        override=args.scene_usd,
    )
    robot_usd = (
        Path(args.robot_usd).expanduser().resolve()
        if args.robot_usd
        else resolve_repo_relative_path(robot_profile.default_usd_relpath)
    )
    robot_prim = str(args.robot_prim or robot_profile.robot_prim)
    if not robot_usd.is_file():
        raise FileNotFoundError(f"{status_label} USD not found: {robot_usd}")

    # 从这里开始才真正启动 Isaac Sim / Isaac Lab。
    AppLauncher = teleop_session._load_isaac_lab()
    if hasattr(args, "enable_cameras") and not bool(args.enable_cameras):
        # play 必须从 front/table 相机读取图像。Isaac Lab 需要在 AppLauncher
        # 启动前打开 enable_cameras，才能加载带 Replicator/SyntheticData 的
        # camera-enabled experience；否则后续 attach annotator 会创建 SDGPipeline 失败。
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.usd
    import torch
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    teleop_session._ensure_carb_input_available()
    import carb.input

    omni_appwindow = teleop_session._import_omni_appwindow()
    omni_viewport_utility = teleop_session._import_omni_viewport_utility()

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / float(args.sim_hz), device=args.device)
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
    robot_default_joint_pos = as_torch(robot.data.default_joint_pos).clone()
    robot_default_joint_vel = as_torch(robot.data.default_joint_vel).clone()

    # 只部署右臂 + 右手：左侧保持默认/锁定，和采集配置一致。
    right_joint_ids = teleop_session._find_required_joint_ids(
        robot,
        wuji_hand_config.right_joint_names,
        "right",
        status_label=status_label,
    )
    right_initial_targets = teleop_session._default_joint_positions(
        robot,
        right_joint_ids,
        wuji_hand_config.right_joint_names,
        wuji_hand_config,
    )
    robot.set_joint_position_target(
        target=torch.as_tensor(right_initial_targets, device=sim.device).unsqueeze(0),
        joint_ids=right_joint_ids,
    )
    left_elbow_lock_joint_ids, left_elbow_lock_joint_names = robot.find_joints(
        [teleop_session.LEFT_ELBOW_LOCK_JOINT],
        preserve_order=True,
    )
    left_elbow_lock_target = None
    if teleop_session.LEFT_ELBOW_LOCK_JOINT in left_elbow_lock_joint_names:
        left_elbow_lock_target = as_torch(robot.data.default_joint_pos)[
            :, [int(left_elbow_lock_joint_ids[0])]
        ].clone()

    arm_ik = DualArmIkController(
        robot,
        arm_ik_config,
        device=sim.device,
        pose_binding=None,
    )
    arm_ik.reset_startup_references()

    stage = omni.usd.get_context().get_stage()
    robot_root_prim = stage.GetPrimAtPath(robot_prim) if stage is not None else None
    teleop_session._disable_robot_transmissive_materials(
        stage=stage,
        robot_root_prim=robot_root_prim,
        status_label=status_label,
    )
    head_link_prim = (
        teleop_session._find_head_like_prim(stage, robot_root_prim, robot_prim)
        if robot_root_prim is not None and robot_root_prim.IsValid()
        else None
    )
    xform_cache = UsdGeom.XformCache()
    viewport_api = omni_viewport_utility.get_active_viewport()
    head_view_camera = (
        teleop_session.HeadViewCameraController(
            stage=stage,
            UsdGeom=UsdGeom,
            Gf=Gf,
            viewport_api=viewport_api,
            config=head_view_camera_config,
            head_link_prim=head_link_prim,
            avp_frame_binding=None,
            whole_body_yaw=None,
        )
        if head_view_camera_config.enabled and stage is not None
        else None
    )
    table_overhead_camera = (
        teleop_session.TableOverheadCameraController(
            stage=stage,
            UsdGeom=UsdGeom,
            Gf=Gf,
            viewport_api=viewport_api,
            config=table_overhead_camera_config,
        )
        if table_overhead_camera_config.enabled and stage is not None
        else None
    )

    # 组装和训练数据同名的两个视觉输入：front/table。
    camera_specs: dict[str, dict[str, Any]] = {}
    if head_view_camera is not None:
        camera_specs["front"] = {
            "camera_path": head_view_camera.camera_path,
            "resolution_px": head_view_camera_config.resolution_px,
        }
    if table_overhead_camera is not None:
        camera_specs["table"] = {
            "camera_path": table_overhead_camera.camera_path,
            "resolution_px": table_overhead_camera_config.resolution_px,
        }
    if not camera_specs:
        raise RuntimeError("No front/table cameras are enabled in the play scene.")

    prop_randomization_rng = np.random.default_rng()
    prop_randomization_specs = {} if scene_usd is not None else PROP_RANDOMIZATION_SPECS
    prop_rigid_body_views: dict[str, Any] = {}
    prop_default_transforms: dict[str, Any] = {}
    blue_cube_color = (0.05, 0.22, 0.95)
    red_cube_color = (0.92, 0.06, 0.04)

    def get_prop_rigid_body_view(prim_path: str) -> Any:
        view = prop_rigid_body_views.get(prim_path)
        if view is not None:
            return view
        physics_sim_view = sim.physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not available.")
        view = physics_sim_view.create_rigid_body_view(prim_path)
        if teleop_session._rigid_body_view_count(view) != 1:
            raise RuntimeError(
                f"Expected one rigid body for {prim_path}, "
                f"got {teleop_session._rigid_body_view_count(view)}."
            )
        prop_rigid_body_views[prim_path] = view
        prop_default_transforms[prim_path] = teleop_session._rigid_body_view_transforms_to_torch(
            view, device=sim.device
        )
        return view

    def set_shape_display_color(prim_path: str, color: tuple[float, float, float]) -> None:
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
        if stage is None or not prop_randomization_specs:
            return
        swapped = bool(prop_randomization_rng.integers(0, 2))
        if swapped:
            blue_path_color = red_cube_color
            red_path_color = blue_cube_color
            blue_path_name = "red"
            red_path_name = "blue"
        else:
            blue_path_color = blue_cube_color
            red_path_color = red_cube_color
            blue_path_name = "blue"
            red_path_name = "red"

        set_shape_display_color(BLUE_CUBE_PRIM_PATH, blue_path_color)
        set_shape_display_color(RED_CUBE_PRIM_PATH, red_path_color)
        print(
            f"[{status_label}] randomized cube colors: "
            f"BlueCube appears {blue_path_name}, RedCube appears {red_path_name}",
            flush=True,
        )

    def randomize_scene_props() -> None:
        if stage is None or not prop_randomization_specs:
            return
        placements: dict[str, tuple[float, float, float]] = {}
        for prim_path, spec in prop_randomization_specs.items():
            center = tuple(float(value) for value in spec["center"])
            x, y = teleop_session._sample_xy_in_disk(
                prop_randomization_rng,
                (center[0], center[1]),
                float(spec["radius_m"]),
            )
            translation = (x, y, center[2])
            teleop_session._set_prim_translation(stage, prim_path, translation)
            view = get_prop_rigid_body_view(prim_path)
            teleop_session._set_rigid_body_view_transform_and_stop(
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

    def step_scene_once() -> None:
        # Isaac 每个仿真 step 都要写目标、推进物理、刷新机器人状态和相机位姿。
        robot.write_data_to_sim()
        sim.step(render=teleop_session._sim_step_should_render(sim))
        robot.update(sim_dt)
        if head_view_camera is not None:
            head_view_camera.update(xform_cache=xform_cache)

    app_window = omni_appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard_reset_requested = False

    def on_keyboard_event(event: Any, *event_args: Any, **event_kwargs: Any) -> bool:
        nonlocal keyboard_reset_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and _keyboard_input_matches(
            event.input,
            ("R", "KEY_R"),
            carb_input=carb.input,
        ):
            keyboard_reset_requested = True
        return True

    keyboard_subscription = (
        input_interface.subscribe_to_keyboard_events(
            app_window.get_keyboard(),
            on_keyboard_event,
        )
        if app_window is not None and app_window.get_keyboard() is not None
        else None
    )
    if keyboard_subscription is None:
        print(
            f"[{status_label}] keyboard reset unavailable: no Kit keyboard was found.",
            flush=True,
        )
    else:
        print(
            f"[{status_label}] keyboard reset enabled: press R in the Kit window to reset and enter a new task.",
            flush=True,
        )

    def restore_robot_initial_pose() -> None:
        if hasattr(robot, "write_joint_state_to_sim"):
            robot.write_joint_state_to_sim(
                robot_default_joint_pos,
                robot_default_joint_vel,
            )
        robot.set_joint_position_target(target=robot_default_joint_pos)
        arm_ik._last_targets.clear()
        arm_ik.reset_startup_references()
        if left_elbow_lock_target is not None:
            robot.set_joint_position_target(
                target=left_elbow_lock_target,
                joint_ids=[int(left_elbow_lock_joint_ids[0])],
            )

    def reset_play_episode(reason: str) -> None:
        restore_robot_initial_pose()
        randomize_scene_props()
        for _ in range(max(1, int(args.camera_attach_retry_steps))):
            if left_elbow_lock_target is not None:
                robot.set_joint_position_target(
                    target=left_elbow_lock_target,
                    joint_ids=[int(left_elbow_lock_joint_ids[0])],
                )
            step_scene_once()
        print(
            f"[{status_label}] play reset by {reason}; robot returned to its initial pose. "
            "Enter a new task to resume policy inference.",
            flush=True,
        )

    def create_image_source_with_retries() -> ReplicatorImageSource:
        # Replicator/SyntheticData 的 graph 在 Isaac 刚打开 stage 后偶尔还没准备好。
        # 先推进若干帧，再失败重试，可以避开 getWrappedGraphFromNode 这类时序错误。
        retries = max(1, int(args.camera_attach_retries))
        retry_steps = max(0, int(args.camera_attach_retry_steps))
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                return ReplicatorImageSource(camera_specs, status_label=status_label)
            except Exception as exc:
                last_error = exc
                print(
                    f"[{status_label}] Replicator camera attach failed "
                    f"({attempt}/{retries}): {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt >= retries:
                    break
                for _ in range(retry_steps):
                    step_scene_once()
                time.sleep(0.2)
        raise RuntimeError(
            f"[{status_label}] failed to attach Replicator cameras after {retries} attempts"
        ) from last_error

    action_chunk: np.ndarray | None = None
    action_index = 0
    last_policy_time = 0.0
    last_action_time = 0.0
    last_status_log_time = 0.0
    # 训练数据是 15fps 左右；这里按 policy_fps 消耗 action，
    # 不按 60Hz 仿真步长消耗，避免动作被放快。
    action_period = 1.0 / max(float(args.policy_fps), 1.0e-6)
    start_time = time.monotonic()
    executed_steps = 0
    camera_wait_reads = 0
    camera_read_attempts = 0
    policy_request_count = 0
    image_source: ReplicatorImageSource | None = None
    policy: RemoteSmolVLAClient | None = None

    try:
        hand_limits = _soft_limits_for_ids(robot, right_joint_ids)
        executor = IsaacActionExecutor(
            robot=robot,
            arm_ik=arm_ik,
            right_joint_ids=right_joint_ids,
            hand_joint_limits=hand_limits,
            sim_device=sim.device,
        )

        for _ in range(max(0, int(args.warmup_steps))):
            if left_elbow_lock_target is not None:
                robot.set_joint_position_target(
                    target=left_elbow_lock_target,
                    joint_ids=[int(left_elbow_lock_joint_ids[0])],
                )
            step_scene_once()

        randomize_scene_props()
        for _ in range(max(1, int(args.camera_attach_retry_steps))):
            step_scene_once()

        image_source = create_image_source_with_retries()

        # 连接另一个终端中运行的 SmolVLA policy server。
        policy = RemoteSmolVLAClient(
            host=args.policy_host,
            port=args.policy_port,
            timeout_s=args.policy_timeout_s,
        )
        policy_info = policy.ping()

        print(
            f"[{status_label}] SmolVLA Isaac play ready: policy_server={policy.endpoint} "
            f"checkpoint={policy_info.get('checkpoint')} "
            f"scene={'builtin_grasp' if scene_usd is None else str(scene_usd)} "
            f"cameras={sorted(camera_specs)} policy_fps={args.policy_fps}",
            flush=True,
        )
        task_text = _prompt_for_task(args, status_label=status_label)
        print(
            f"[{status_label}] entering play loop: chunk_steps={args.chunk_steps} "
            f"max_steps={args.max_steps} status_log_interval_s={args.status_log_interval_s} "
            f"task={task_text!r}",
            flush=True,
        )

        while simulation_app.is_running():
            now = time.monotonic()
            if args.duration_s > 0.0 and now - start_time >= args.duration_s:
                print(
                    f"[{status_label}] stopping play loop: reached duration_s={args.duration_s}",
                    flush=True,
                )
                break
            if args.max_steps > 0 and executed_steps >= args.max_steps:
                print(
                    f"[{status_label}] stopping play loop: reached max_steps={args.max_steps}",
                    flush=True,
                )
                break

            if keyboard_reset_requested:
                keyboard_reset_requested = False
                action_chunk = None
                action_index = 0
                last_policy_time = 0.0
                last_action_time = 0.0
                camera_read_attempts = 0
                camera_wait_reads = 0
                reset_play_episode("keyboard R")
                if policy is not None:
                    policy.reset()
                task_text = _prompt_for_task(args, status_label=status_label)
                print(
                    f"[{status_label}] resumed play loop: task={task_text!r}",
                    flush=True,
                )
                continue

            state, current_quat = _make_state(
                arm_ik=arm_ik,
                robot=robot,
                right_joint_ids=right_joint_ids,
            )
            need_new_chunk = action_chunk is None or action_index >= len(action_chunk)
            if need_new_chunk and now - last_policy_time >= action_period:
                camera_read_attempts += 1
                if now - last_status_log_time >= max(float(args.status_log_interval_s), 0.1):
                    print(
                        f"[{status_label}] reading camera frames for next policy chunk: "
                        f"attempt={camera_read_attempts} previous_shapes={image_source.last_shapes}",
                        flush=True,
                    )
                    last_status_log_time = now
                images = image_source.read()
                if images is None:
                    camera_wait_reads += 1
                    if now - last_status_log_time >= max(float(args.status_log_interval_s), 0.1):
                        print(
                            f"[{status_label}] waiting for camera frames: "
                            f"missing={image_source.last_missing_keys} "
                            f"last_shapes={image_source.last_shapes} "
                            f"empty_reads={camera_wait_reads}",
                            flush=True,
                        )
                        last_status_log_time = now
                    step_scene_once()
                    continue
                # 发送一帧实时 observation 到推理进程，拿回未来一段 action chunk。
                policy_request_count += 1
                request_start = time.monotonic()
                print(
                    f"[{status_label}] requesting policy chunk #{policy_request_count}: "
                    f"image_shapes={image_source.last_shapes} state_dim={state.shape[0]} "
                    f"steps={args.chunk_steps} task={task_text!r}",
                    flush=True,
                )
                action_chunk = policy.predict_action_chunk(
                    state=state,
                    images=images,
                    task=task_text,
                    steps=args.chunk_steps,
                )
                request_latency = time.monotonic() - request_start
                print(
                    f"[{status_label}] received policy chunk #{policy_request_count}: "
                    f"shape={tuple(int(dim) for dim in action_chunk.shape)} "
                    f"latency_s={request_latency:.3f}",
                    flush=True,
                )
                action_index = 0
                last_policy_time = now

            if (
                action_chunk is not None
                and action_index < len(action_chunk)
                and now - last_action_time >= action_period
            ):
                # 每到一个 policy_fps 时间点，执行 chunk 里的下一步 action。
                executor.execute_one(
                    state=state,
                    current_quat_xyzw=current_quat,
                    raw_action=action_chunk[action_index],
                    max_position_delta_m=args.max_position_delta_m,
                    max_rotation_delta_rad=args.max_rotation_delta_rad,
                )
                action_index += 1
                executed_steps += 1
                last_action_time = now
                if int(args.action_log_every) > 0 and executed_steps % int(args.action_log_every) == 0:
                    print(
                        f"[{status_label}] executed policy action: "
                        f"executed_steps={executed_steps} "
                        f"chunk_index={action_index}/{len(action_chunk)}",
                        flush=True,
                    )

            if left_elbow_lock_target is not None:
                robot.set_joint_position_target(
                    target=left_elbow_lock_target,
                    joint_ids=[int(left_elbow_lock_joint_ids[0])],
                )
            step_scene_once()
    finally:
        if keyboard_subscription is not None:
            unsubscribe_fn = getattr(
                input_interface, "unsubscribe_from_keyboard_events", None
            )
            if callable(unsubscribe_fn) and app_window is not None:
                unsubscribe_fn(
                    app_window.get_keyboard(),
                    keyboard_subscription,
                )
        if policy is not None:
            policy.close()
        if image_source is not None:
            image_source.close()
        simulation_app.close()

    return 0
