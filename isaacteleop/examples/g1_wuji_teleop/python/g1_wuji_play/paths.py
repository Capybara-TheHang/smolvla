"""Path helpers for G1-Wuji SmolVLA play deployment."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def app_root() -> Path:
    return package_root().parents[1]


def repo_root() -> Path:
    return app_root().parents[2]


def lerobot_g1_wuji_dir() -> Path:
    return repo_root() / "lerobot" / "examples" / "g1_wuji"


def lerobot_play_py() -> Path:
    return lerobot_g1_wuji_dir() / "play.py"


def default_checkpoint_root() -> Path:
    return repo_root() / "lerobot" / "outputs" / "train" / "g1_wuji_112339_smolvla"


def default_vlm_model_name() -> Path:
    return repo_root() / "checkpoints" / "SmolVLM2-500M-Video-Instruct"
