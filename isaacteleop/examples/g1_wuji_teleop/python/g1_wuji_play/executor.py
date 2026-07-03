"""Action executors for G1-Wuji play deployment.

The current executor is a dry-run sink. It keeps the deployment entry runnable
before the Isaac scene loop is connected.

中文说明：
DryRunActionExecutor 不控制机器人，只把已经解码的命令写成 JSON。
它用于检查模型输出是否合理；真实仿真执行在 simulation.py 的
IsaacActionExecutor 里完成。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .action import G1WujiActionCommand


class DryRunActionExecutor:
    """Write decoded commands instead of sending them to the robot/simulator."""

    def __init__(self, output: str | Path | None = None) -> None:
        self.output = Path(output).expanduser().resolve() if output else None

    def execute(self, commands: Iterable[G1WujiActionCommand]) -> None:
        # 输出字段包括 ee_target_pos、ee_target_quat_xyzw、hand_target_joints、raw_action。
        payload = [command.to_dict() for command in commands]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if self.output is None:
            print(text)
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(text + "\n", encoding="utf-8")
