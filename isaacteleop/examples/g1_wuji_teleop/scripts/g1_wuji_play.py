#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin launcher for G1-Wuji SmolVLA playback/deployment.

中文说明：
这个脚本只是入口转发器，负责把 examples/g1_wuji_teleop/python 加入
sys.path，然后调用 g1_wuji_play.runner。真正逻辑不放在 scripts 目录里，
方便后续拆分仿真、RPC、action 解码等模块。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_app_import_path() -> None:
    # 让脚本无论从哪个工作目录启动，都能 import 到 python/g1_wuji_play。
    example_root = Path(__file__).resolve().parents[1]
    app_python_dir = example_root / "python"
    path_str = str(app_python_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    existing = sys.modules.get("g1_wuji_play")
    if existing is not None and not hasattr(existing, "__path__"):
        # 脚本本身也叫 g1_wuji_play.py；某些启动方式会先缓存同名脚本模块，
        # 这里清掉非包模块，保证下面导入的是 python/g1_wuji_play/ 包。
        del sys.modules["g1_wuji_play"]


def main() -> int:
    _ensure_app_import_path()
    from g1_wuji_play.runner import play_main

    return play_main()


if __name__ == "__main__":
    raise SystemExit(main())
