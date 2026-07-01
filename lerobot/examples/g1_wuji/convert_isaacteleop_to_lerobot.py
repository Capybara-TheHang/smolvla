#!/usr/bin/env python
"""Convert G1-Wuji isaacteleop raw trajectories into a LeRobotDataset.

Expected input layout:

    <raw-dir>/
      session.json
      index.jsonl
      trajectory_000000_*.npz
      trajectory_000000_*.json
      ...

Each trajectory NPZ is expected to contain:

    observation.state          [T, state_dim]
    action                     [T, action_dim]
    task                       optional [T]
    observation.images.<name>  [T, H, W, 3]

Example:

    python lerobot/examples/g1_wuji/convert_isaacteleop_to_lerobot.py \
      --raw-dir /home/lightwheel/workspace/smolvla/isaacteleop/examples/g1_wuji_teleop/data/dataset/20260701_112339 \
      --out-root /home/lightwheel/workspace/smolvla/lerobot/outputs/g1_wuji_112339_lerobot \
      --repo-id local/g1_wuji_112339 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets import LeRobotDataset  # noqa: E402


OBS_STATE = "observation.state"
ACTION = "action"
IMAGE_PREFIX = "observation.images."


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_npz_paths(raw_dir: Path) -> list[Path]:
    direct = sorted(raw_dir.glob("trajectory_*.npz"))
    if direct:
        return direct

    nested = sorted(raw_dir.glob("*/trajectory_*.npz"))
    if nested:
        return nested

    raise FileNotFoundError(f"No trajectory_*.npz found under {raw_dir}")


def image_keys_from_npz(npz_path: Path) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        return sorted(key for key in data.files if key.startswith(IMAGE_PREFIX))


def validate_trajectory(npz_path: Path, image_keys: list[str], require_images: bool) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        missing = [key for key in (OBS_STATE, ACTION) if key not in data.files]
        if missing:
            raise ValueError(f"{npz_path} is missing required keys: {missing}")

        if require_images:
            missing_images = [key for key in image_keys if key not in data.files]
            if missing_images:
                raise ValueError(f"{npz_path} is missing image keys: {missing_images}")

        lengths = [len(data[OBS_STATE]), len(data[ACTION])]
        lengths.extend(len(data[key]) for key in image_keys if key in data.files)
        if "task" in data.files:
            lengths.append(len(data["task"]))

        frame_count = min(lengths)
        if frame_count <= 0:
            raise ValueError(f"{npz_path} has no usable frames")
        if len(set(lengths)) != 1:
            print(
                f"[warn] {npz_path.name}: length mismatch {lengths}; using first {frame_count} frames",
                flush=True,
            )
        return frame_count


def make_features(
    first_npz: Path,
    image_keys: list[str],
    session: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    with np.load(first_npz, allow_pickle=True) as data:
        if OBS_STATE not in data.files or ACTION not in data.files:
            raise ValueError(f"{first_npz} must contain {OBS_STATE!r} and {ACTION!r}")

        features: dict[str, dict[str, Any]] = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": tuple(data[OBS_STATE].shape[1:]),
                "names": session.get("state_feature_names"),
            },
            ACTION: {
                "dtype": "float32",
                "shape": tuple(data[ACTION].shape[1:]),
                "names": session.get("action_feature_names"),
            },
        }

        for key in image_keys:
            if key not in data.files:
                raise ValueError(f"{first_npz} is missing discovered image key {key!r}")
            if data[key].ndim != 4 or data[key].shape[-1] != 3:
                raise ValueError(
                    f"{first_npz}:{key} must have shape [T, H, W, 3], got {data[key].shape}"
                )
            features[key] = {
                "dtype": "video",
                "shape": tuple(data[key].shape[1:]),
                "names": ["height", "width", "channels"],
            }

    return features


def default_task(session: dict[str, Any]) -> str:
    return str(session.get("task_default") or "Teleoperate the G1-Wuji robot.")


def convert(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    session = load_json(raw_dir / "session.json")
    npz_paths = discover_npz_paths(raw_dir)

    image_keys = image_keys_from_npz(npz_paths[0])
    if not image_keys and args.require_images:
        raise ValueError(
            f"No {IMAGE_PREFIX}* arrays found in {npz_paths[0]}. "
            "This is not a visual SmolVLA dataset. Re-record with camera capture enabled, "
            "or pass --no-require-images only for debugging."
        )

    for npz_path in npz_paths[1:]:
        keys = image_keys_from_npz(npz_path)
        if keys != image_keys:
            raise ValueError(
                f"Image keys differ across trajectories. First={image_keys}, {npz_path.name}={keys}"
            )

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    fps = int(round(float(args.fps if args.fps is not None else session.get("sample_fps", 15))))
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    features = make_features(npz_paths[0], image_keys, session)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=out_root,
        fps=fps,
        features=features,
        robot_type=args.robot_type,
        use_videos=not args.images_only,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    try:
        for episode_index, npz_path in enumerate(npz_paths):
            frame_count = validate_trajectory(npz_path, image_keys, args.require_images)
            with np.load(npz_path, allow_pickle=True) as data:
                for frame_index in range(frame_count):
                    if "task" in data.files:
                        task = str(data["task"][frame_index])
                    else:
                        task = default_task(session)

                    frame = {
                        OBS_STATE: np.asarray(data[OBS_STATE][frame_index], dtype=np.float32),
                        ACTION: np.asarray(data[ACTION][frame_index], dtype=np.float32),
                        "task": task,
                    }

                    for key in image_keys:
                        image = np.asarray(data[key][frame_index])
                        if image.dtype != np.uint8:
                            image = np.clip(image, 0, 255).astype(np.uint8)
                        frame[key] = np.ascontiguousarray(image)

                    dataset.add_frame(frame)

            dataset.save_episode(parallel_encoding=not args.disable_parallel_encoding)
            total_frames += frame_count
            print(
                f"[ok] episode={episode_index} file={npz_path.name} frames={frame_count}",
                flush=True,
            )
    finally:
        dataset.finalize()

    print("\nConversion complete")
    print(f"  raw_dir: {raw_dir}")
    print(f"  out_root: {out_root}")
    print(f"  repo_id: {args.repo_id}")
    print(f"  fps: {fps}")
    print(f"  episodes: {len(npz_paths)}")
    print(f"  frames: {total_frames}")
    print(f"  image_keys: {image_keys}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="isaacteleop session dir or dataset root.")
    parser.add_argument("--out-root", required=True, help="Output LeRobotDataset root directory.")
    parser.add_argument("--repo-id", required=True, help="Local repo id, e.g. local/g1_wuji_112339.")
    parser.add_argument("--robot-type", default="g1_wuji")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS. Defaults to session.json sample_fps.")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace --out-root if it already exists.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Store visual features as image files instead of videos. Useful for debugging.",
    )
    parser.add_argument(
        "--no-require-images",
        dest="require_images",
        action="store_false",
        help="Allow conversion without observation.images.* keys. Not recommended for SmolVLA training.",
    )
    parser.add_argument(
        "--disable-parallel-encoding",
        action="store_true",
        help="Encode videos synchronously after each episode.",
    )
    parser.set_defaults(require_images=True)
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
