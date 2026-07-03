# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints for the G1-Wuji RL scene."""

from __future__ import annotations

from collections.abc import Sequence

from .scene_app import main as scene_main


def rl_scene_main(argv: Sequence[str] | None = None) -> int:
    return scene_main(argv)
