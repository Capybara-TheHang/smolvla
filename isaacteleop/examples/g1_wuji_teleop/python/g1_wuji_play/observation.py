"""Observation loading helpers for G1-Wuji policy playback.

中文说明：
这里只服务 dry-run 调试：从历史 ``trajectory_*.npz`` 里取一帧 observation。
真实 Isaac 闭环不会走这个文件，而是直接从仿真相机和机器人状态构造 observation。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OBS_STATE = "observation.state"
DEFAULT_TASK = "Teleoperate the right arm and Wuji dexterous hand."


@dataclass(frozen=True)
class ObservationFrame:
    state: np.ndarray
    images: dict[str, np.ndarray]
    task: str = DEFAULT_TASK

    def as_policy_observation(self, make_observation: Any) -> dict[str, Any]:
        return make_observation(state=self.state, images=self.images, task=self.task)


def load_npz_frame(path: str | Path, *, frame_index: int = 0, task: str | None = None) -> ObservationFrame:
    # npz 的字段名和训练数据保持一致：observation.state + observation.images.*
    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=True) as data:
        if OBS_STATE not in data.files:
            raise KeyError(f"{npz_path} does not contain {OBS_STATE!r}")
        frame_count = len(data[OBS_STATE])
        index = frame_index if frame_index >= 0 else frame_count + frame_index
        if index < 0 or index >= frame_count:
            raise IndexError(f"frame_index={frame_index} out of range for {frame_count} frames")

        image_keys = sorted(key for key in data.files if key.startswith("observation.images."))
        if not image_keys:
            raise KeyError(f"{npz_path} does not contain observation.images.* arrays")

        task_value = task
        if task_value is None:
            task_value = str(data["task"][index]) if "task" in data.files else DEFAULT_TASK

        return ObservationFrame(
            state=np.asarray(data[OBS_STATE][index], dtype=np.float32),
            images={key: np.asarray(data[key][index]) for key in image_keys},
            task=str(task_value),
        )
