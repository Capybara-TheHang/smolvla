#!/usr/bin/env python
"""Merge compatible local LeRobotDataset v3 roots without re-encoding frames.

This is intended for already-converted G1-Wuji datasets that share the same
feature schema. It rewrites parquet metadata/indexes and concatenates the MP4
video shards with ffmpeg stream copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.compute_stats import aggregate_stats  # noqa: E402
from lerobot.datasets.io_utils import load_stats, write_stats  # noqa: E402


DEFAULT_VIDEO_KEYS = (
    "observation.images.front",
    "observation.images.table",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def comparable_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "dtype": feature.get("dtype"),
        "shape": list(feature.get("shape") or []),
        "names": feature.get("names"),
    }


def validate_compatible(roots: list[Path]) -> list[dict[str, Any]]:
    infos = [load_json(root / "meta" / "info.json") for root in roots]
    base = infos[0]
    for root, info in zip(roots[1:], infos[1:], strict=True):
        for key in ("fps", "robot_type", "data_path", "video_path"):
            if info.get(key) != base.get(key):
                raise ValueError(
                    f"{root} is incompatible: {key}={info.get(key)!r}, "
                    f"expected {base.get(key)!r}"
                )

        base_features = base.get("features", {})
        features = info.get("features", {})
        if set(features) != set(base_features):
            raise ValueError(f"{root} feature keys differ from {roots[0]}")
        for feature_key in base_features:
            if comparable_feature(features[feature_key]) != comparable_feature(base_features[feature_key]):
                raise ValueError(
                    f"{root} feature {feature_key!r} differs: "
                    f"{comparable_feature(features[feature_key])} != "
                    f"{comparable_feature(base_features[feature_key])}"
                )
    return infos


def read_single_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_tasks(root: Path) -> pd.DataFrame:
    tasks = read_single_parquet(root / "meta" / "tasks.parquet")
    if "task" not in tasks.columns and tasks.index.name == "task":
        tasks = tasks.reset_index()
    if "task" not in tasks.columns or "task_index" not in tasks.columns:
        raise ValueError(f"{root} meta/tasks.parquet must contain task_index and task")
    return tasks.sort_values("task_index").reset_index(drop=True)


def build_task_mapping(roots: list[Path]) -> tuple[pd.DataFrame, list[dict[int, int]]]:
    merged_tasks: list[str] = []
    task_to_index: dict[str, int] = {}
    mappings: list[dict[int, int]] = []
    for root in roots:
        tasks = load_tasks(root)
        mapping: dict[int, int] = {}
        for row in tasks.to_dict("records"):
            task = str(row["task"])
            if task not in task_to_index:
                task_to_index[task] = len(merged_tasks)
                merged_tasks.append(task)
            mapping[int(row["task_index"])] = task_to_index[task]
        mappings.append(mapping)
    merged_df = pd.DataFrame(
        {
            "task_index": np.arange(len(merged_tasks), dtype=np.int64),
            "task": merged_tasks,
        }
    ).set_index("task")
    return merged_df, mappings


def remap_data_frame(
    df: pd.DataFrame,
    *,
    episode_offset: int,
    frame_offset: int,
    task_mapping: dict[int, int],
) -> pd.DataFrame:
    result = df.copy()
    result["episode_index"] = result["episode_index"].astype("int64") + episode_offset
    result["index"] = result["index"].astype("int64") + frame_offset
    result["task_index"] = result["task_index"].astype("int64").map(task_mapping).astype("int64")
    return result


def remap_episode_frame(
    df: pd.DataFrame,
    *,
    episode_offset: int,
    frame_offset: int,
    task_mapping: dict[int, int],
    video_time_offsets: dict[str, float],
) -> pd.DataFrame:
    result = df.copy()
    result["episode_index"] = result["episode_index"].astype("int64") + episode_offset
    result["dataset_from_index"] = result["dataset_from_index"].astype("int64") + frame_offset
    result["dataset_to_index"] = result["dataset_to_index"].astype("int64") + frame_offset

    if "tasks" in result.columns:
        result["tasks"] = result["tasks"].apply(lambda value: list(value) if not isinstance(value, list) else value)

    for key, offset in video_time_offsets.items():
        from_col = f"videos/{key}/from_timestamp"
        to_col = f"videos/{key}/to_timestamp"
        if from_col in result.columns:
            result[from_col] = result[from_col].astype("float64") + offset
        if to_col in result.columns:
            result[to_col] = result[to_col].astype("float64") + offset

    # Keep task stats columns untouched. Episode-level task ids live in data.task_index.
    _ = task_mapping
    return result


def concat_videos(input_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp_dir:
        concat_file = Path(tmp_dir) / "inputs.ffconcat"
        lines = ["ffconcat version 1.0\n"]
        for path in input_paths:
            lines.append(f"file '{path.resolve()}'\n")
        concat_file.write_text("".join(lines), encoding="utf-8")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


def video_duration_from_episodes(episodes: pd.DataFrame, video_key: str) -> float:
    column = f"videos/{video_key}/to_timestamp"
    if column not in episodes.columns or episodes.empty:
        return 0.0
    return float(episodes[column].max())


def merge(args: argparse.Namespace) -> None:
    roots = [Path(path).expanduser().resolve() for path in args.inputs]
    out_root = Path(args.output).expanduser().resolve()
    if len(roots) < 2:
        raise ValueError("Provide at least two input dataset roots")
    for root in roots:
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing LeRobotDataset metadata under {root}")

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    infos = validate_compatible(roots)
    base_info = infos[0]
    video_keys = [
        key
        for key, feature in base_info["features"].items()
        if feature.get("dtype") == "video"
    ]
    if not video_keys:
        video_keys = list(DEFAULT_VIDEO_KEYS)

    merged_tasks, task_mappings = build_task_mapping(roots)
    data_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    episode_offset = 0
    frame_offset = 0
    video_time_offsets = {key: 0.0 for key in video_keys}

    for root, task_mapping in zip(roots, task_mappings, strict=True):
        data = read_single_parquet(root / "data" / "chunk-000" / "file-000.parquet")
        episodes = read_single_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

        data_frames.append(
            remap_data_frame(
                data,
                episode_offset=episode_offset,
                frame_offset=frame_offset,
                task_mapping=task_mapping,
            )
        )
        episode_frames.append(
            remap_episode_frame(
                episodes,
                episode_offset=episode_offset,
                frame_offset=frame_offset,
                task_mapping=task_mapping,
                video_time_offsets=video_time_offsets,
            )
        )

        episode_offset += int(infos[roots.index(root)]["total_episodes"])
        frame_offset += int(infos[roots.index(root)]["total_frames"])
        for key in video_keys:
            video_time_offsets[key] += video_duration_from_episodes(episodes, key)

    merged_data = pd.concat(data_frames, ignore_index=True)
    merged_episodes = pd.concat(episode_frames, ignore_index=True)

    out_data = out_root / "data" / "chunk-000" / "file-000.parquet"
    out_episodes = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_data.parent.mkdir(parents=True, exist_ok=True)
    out_episodes.parent.mkdir(parents=True, exist_ok=True)
    merged_data.to_parquet(out_data, index=False)
    merged_episodes.to_parquet(out_episodes, index=False)
    merged_tasks.to_parquet(out_root / "meta" / "tasks.parquet")

    for key in video_keys:
        concat_videos(
            [
                root
                / "videos"
                / key
                / "chunk-000"
                / "file-000.mp4"
                for root in roots
            ],
            out_root / "videos" / key / "chunk-000" / "file-000.mp4",
        )

    merged_stats = aggregate_stats([load_stats(root) for root in roots])
    write_stats(merged_stats, out_root)

    merged_info = dict(base_info)
    merged_info["total_episodes"] = int(sum(info["total_episodes"] for info in infos))
    merged_info["total_frames"] = int(sum(info["total_frames"] for info in infos))
    merged_info["total_tasks"] = int(len(merged_tasks))
    merged_info["splits"] = {"train": f"0:{merged_info['total_episodes']}"}
    write_json(out_root / "meta" / "info.json", merged_info)

    print("Merge complete")
    print(f"  output: {out_root}")
    print(f"  inputs: {', '.join(str(root) for root in roots)}")
    print(f"  total_episodes: {merged_info['total_episodes']}")
    print(f"  total_frames: {merged_info['total_frames']}")
    print(f"  total_tasks: {merged_info['total_tasks']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input LeRobotDataset roots.")
    parser.add_argument("--output", required=True, help="Merged output root.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
