"""Bridge from the Isaac deployment package to lerobot/examples/g1_wuji/play.py.

中文说明：
这个 bridge 只给 dry-run 使用，会在当前进程里直接加载 SmolVLA。
Isaac 仿真闭环不要用它，应该通过 policy_rpc.py 连接另一个推理进程。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .paths import lerobot_play_py


def _load_lerobot_play_module() -> ModuleType:
    # play.py 不在 Python package 里，所以这里用文件路径动态加载。
    path = lerobot_play_py()
    spec = importlib.util.spec_from_file_location("g1_wuji_lerobot_play", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmolVLAPolicyBridge:
    """Thin adapter used by the deployment runner's dry-run backend."""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None,
        device: str,
        vlm_model_name: str | None = None,
    ) -> None:
        self._play_module = _load_lerobot_play_module()
        self._player = self._play_module.G1WujiSmolVLAPlayer(
            checkpoint=checkpoint,
            device=device,
            vlm_model_name=vlm_model_name,
        )

    @property
    def checkpoint(self) -> Path:
        return self._player.checkpoint

    def make_observation(self, *, state: Any, images: dict[str, Any], task: str) -> dict[str, Any]:
        return self._play_module.make_observation(state=state, images=images, task=task)

    def predict_action_chunk(self, observation: dict[str, Any], *, steps: int | None = None) -> np.ndarray:
        return self._player.predict_action_chunk(observation, steps=steps)

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        return self._player.select_action(observation)
