"""Action decoding utilities for G1-Wuji SmolVLA deployment.

中文说明：
SmolVLA 输出的是训练数据里的 26 维 action，不是 Isaac 可以直接执行的
关节命令。本文件把 26 维 action 解码成“右手末端目标位姿 + 右手 20 个
手指目标关节”，再交给仿真侧 IK 和关节控制器执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATE_DIM = 26
ACTION_DIM = 26
HAND_DIM = 20


@dataclass(frozen=True)
class G1WujiActionCommand:
    """Decoded command for one policy action step.

    ee_target_pos / ee_target_quat_xyzw 给右臂 IK 使用；
    hand_target_joints 直接作为右手 20 个手指关节的位置目标。
    """

    ee_target_pos: np.ndarray
    ee_target_quat_xyzw: np.ndarray
    hand_target_joints: np.ndarray
    raw_action: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "ee_target_pos": self.ee_target_pos.tolist(),
            "ee_target_quat_xyzw": self.ee_target_quat_xyzw.tolist(),
            "hand_target_joints": self.hand_target_joints.tolist(),
            "raw_action": self.raw_action.tolist(),
        }


def _as_float_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def normalize_quat_xyzw(quat: Any) -> np.ndarray:
    array = _as_float_array(quat, shape=(4,), name="quat_xyzw")
    norm = float(np.linalg.norm(array))
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (array / norm).astype(np.float32)


def quat_mul_xyzw(lhs: Any, rhs: Any) -> np.ndarray:
    lx, ly, lz, lw = normalize_quat_xyzw(lhs)
    rx, ry, rz, rw = normalize_quat_xyzw(rhs)
    return normalize_quat_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def rotvec_to_quat_xyzw(rotvec: Any) -> np.ndarray:
    vec = _as_float_array(rotvec, shape=(3,), name="rotvec").astype(np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1.0e-8:
        quat = np.array([0.5 * vec[0], 0.5 * vec[1], 0.5 * vec[2], 1.0], dtype=np.float64)
        return normalize_quat_xyzw(quat)
    axis = vec / angle
    half = 0.5 * angle
    xyz = axis * np.sin(half)
    return normalize_quat_xyzw((xyz[0], xyz[1], xyz[2], np.cos(half)))


def quat_to_rotvec_xyzw(quat: Any) -> np.ndarray:
    q = normalize_quat_xyzw(quat).astype(np.float64)
    if q[3] < 0.0:
        q = -q
    xyz = q[:3]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1.0e-8:
        return (2.0 * xyz).astype(np.float32)
    angle = 2.0 * np.arctan2(xyz_norm, float(q[3]))
    return (xyz / xyz_norm * angle).astype(np.float32)


def state_to_ee_pose(state: Any) -> tuple[np.ndarray, np.ndarray]:
    # state 前 6 维是当前末端位姿：xyz + rotvec；后 20 维是当前手指关节。
    state_array = _as_float_array(state, shape=(STATE_DIM,), name="state")
    return state_array[:3].copy(), rotvec_to_quat_xyzw(state_array[3:6])


def clamp_action(
    action: Any,
    *,
    max_position_delta_m: float | None = 0.08,
    max_rotation_delta_rad: float | None = 0.35,
    hand_joint_limits: np.ndarray | None = None,
) -> np.ndarray:
    """Clamp one 26D action before decoding.

    The defaults are deliberately conservative for first simulation playback.
    Set a limit to ``None`` to disable that clamp.
    """

    # action 前 6 维是末端增量：dxyz + drotvec；后 20 维是手指目标关节。
    action_array = _as_float_array(action, shape=(ACTION_DIM,), name="action").copy()

    if max_position_delta_m is not None:
        pos_norm = float(np.linalg.norm(action_array[:3]))
        if pos_norm > max_position_delta_m > 0.0:
            action_array[:3] *= float(max_position_delta_m) / pos_norm

    if max_rotation_delta_rad is not None:
        rot_norm = float(np.linalg.norm(action_array[3:6]))
        if rot_norm > max_rotation_delta_rad > 0.0:
            action_array[3:6] *= float(max_rotation_delta_rad) / rot_norm

    if hand_joint_limits is not None:
        limits = np.asarray(hand_joint_limits, dtype=np.float32)
        if limits.shape != (HAND_DIM, 2):
            raise ValueError(f"hand_joint_limits must have shape {(HAND_DIM, 2)}, got {limits.shape}")
        action_array[6:] = np.clip(action_array[6:], limits[:, 0], limits[:, 1])

    return action_array


def decode_action(
    *,
    state: Any,
    action: Any,
    current_quat_xyzw: Any | None = None,
    max_position_delta_m: float | None = 0.08,
    max_rotation_delta_rad: float | None = 0.35,
    hand_joint_limits: np.ndarray | None = None,
) -> G1WujiActionCommand:
    """Decode one 26D SmolVLA action into EE target pose and hand joint targets.

    位置目标：target_pos = current_pos + dxyz
    姿态目标：target_quat = delta_quat(drotvec) * current_quat
    手部目标：直接使用 action[6:26]
    """

    current_pos, state_quat = state_to_ee_pose(state)
    current_quat = normalize_quat_xyzw(current_quat_xyzw) if current_quat_xyzw is not None else state_quat
    action_array = clamp_action(
        action,
        max_position_delta_m=max_position_delta_m,
        max_rotation_delta_rad=max_rotation_delta_rad,
        hand_joint_limits=hand_joint_limits,
    )

    # 训练数据里旋转 action 存的是相对当前姿态的 rotvec 增量。
    delta_quat = rotvec_to_quat_xyzw(action_array[3:6])
    target_quat = quat_mul_xyzw(delta_quat, current_quat)
    return G1WujiActionCommand(
        ee_target_pos=(current_pos + action_array[:3]).astype(np.float32),
        ee_target_quat_xyzw=target_quat.astype(np.float32),
        hand_target_joints=action_array[6:].astype(np.float32),
        raw_action=action_array.astype(np.float32),
    )


def decode_action_chunk(
    *,
    state: Any,
    actions: Any,
    current_quat_xyzw: Any | None = None,
    max_position_delta_m: float | None = 0.08,
    max_rotation_delta_rad: float | None = 0.35,
    hand_joint_limits: np.ndarray | None = None,
) -> list[G1WujiActionCommand]:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [T, {ACTION_DIM}], got {action_array.shape}")
    return [
        decode_action(
            state=state,
            action=step_action,
            current_quat_xyzw=current_quat_xyzw,
            max_position_delta_m=max_position_delta_m,
            max_rotation_delta_rad=max_rotation_delta_rad,
            hand_joint_limits=hand_joint_limits,
        )
        for step_action in action_array
    ]
