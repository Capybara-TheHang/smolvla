#!/usr/bin/env python
"""Generate a tiny synthetic LeRobot dataset and optionally train SmolVLA on it.

This is an end-to-end smoke test for the "one arm + one dexterous hand" setup:

    observation.state:  6D end-effector pose + 20D hand joints = 26D
    action:             6D end-effector delta + 20D hand targets = 26D

The data are deliberately simple and fake. They are only meant to verify that
dataset writing, dataset loading, SmolVLA preprocessing, and a short training
run work on this machine before real demonstrations are collected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets import LeRobotDataset  # noqa: E402


STATE_DIM = 26
ACTION_DIM = 26
TASK = "Pick up the red cube with one arm and one dexterous hand."


def build_features(image_size: int) -> dict:
    names = {
        "state": [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.rx",
            "ee.ry",
            "ee.rz",
            *[f"hand.j{i}" for i in range(20)],
        ],
        "action": [
            "ee.dx",
            "ee.dy",
            "ee.dz",
            "ee.drx",
            "ee.dry",
            "ee.drz",
            *[f"hand.target_j{i}" for i in range(20)],
        ],
    }
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": names["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": names["action"],
        },
    }


def synthetic_frame(rng: np.random.Generator, episode: int, frame: int, frames_per_episode: int, image_size: int):
    progress = frame / max(frames_per_episode - 1, 1)
    phase = np.float32(2.0 * np.pi * progress)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:6] = np.array(
        [
            -0.25 + 0.50 * progress,
            0.10 * np.sin(phase),
            0.20 + 0.05 * np.cos(phase),
            0.10 * np.sin(phase),
            0.10 * np.cos(phase),
            0.20 * progress,
        ],
        dtype=np.float32,
    )
    hand_base = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    state[6:] = np.clip(progress + 0.08 * np.sin(phase + hand_base * np.pi), 0.0, 1.0)
    state += rng.normal(0.0, 0.01, size=STATE_DIM).astype(np.float32)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:6] = np.array(
        [
            0.02,
            0.01 * np.cos(phase),
            -0.01 * np.sin(phase),
            0.005 * np.cos(phase),
            -0.005 * np.sin(phase),
            0.01,
        ],
        dtype=np.float32,
    )
    action[6:] = np.clip(state[6:] + 0.05, 0.0, 1.0)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[..., 0] = np.uint8(30 + 30 * episode)
    image[..., 1] = np.uint8(40 + 120 * progress)
    image[..., 2] = np.uint8(120)
    cube = max(4, image_size // 8)
    x = int(progress * (image_size - cube))
    y = image_size // 2 - cube // 2
    image[y : y + cube, x : x + cube] = np.array([220, 40, 40], dtype=np.uint8)

    return {
        "observation.images.front": image,
        "observation.state": state,
        "action": action,
        "task": TASK,
    }


def generate_dataset(args: argparse.Namespace) -> Path:
    root = Path(args.root).expanduser().resolve()
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(root)

    rng = np.random.default_rng(args.seed)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=build_features(args.image_size),
        robot_type="synthetic_one_arm_one_hand",
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )

    try:
        for ep in range(args.episodes):
            for fr in range(args.frames_per_episode):
                dataset.add_frame(
                    synthetic_frame(rng, ep, fr, args.frames_per_episode, args.image_size)
                )
            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()

    check_dataset = LeRobotDataset(args.repo_id, root=root, download_videos=False)
    sample = check_dataset[0]
    print(f"dataset_root={root}")
    print(f"episodes={check_dataset.num_episodes} frames={check_dataset.num_frames} fps={check_dataset.fps}")
    print(f"state_shape={tuple(sample['observation.state'].shape)} action_shape={tuple(sample['action'].shape)}")
    print(f"features={list(check_dataset.features)}")
    return root


def run_train(args: argparse.Namespace, dataset_root: Path) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite_output:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite-output to replace it.")
        shutil.rmtree(output_dir)

    patch_local_policy_preprocessor(args.policy_path, args.vlm_model_name)

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--policy.path={args.policy_path}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--steps={args.train_steps}",
        f"--batch_size={args.batch_size}",
        "--num_workers=0",
        "--persistent_workers=false",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--log_freq={args.log_freq}",
        f"--save_checkpoint={str(args.save_checkpoint).lower()}",
        "--wandb.enable=false",
    ]
    if args.vlm_model_name:
        cmd.append(f"--policy.vlm_model_name={args.vlm_model_name}")
    if args.rename_front_to_camera1:
        cmd.extend(
            [
                '--rename_map={"observation.images.front":"observation.images.camera1"}',
                "--policy.empty_cameras=2",
            ]
        )
    print("training_command=" + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{LEROBOT_SRC}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(LEROBOT_SRC)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def patch_local_policy_preprocessor(policy_path: str, tokenizer_name: str | None) -> None:
    if not tokenizer_name:
        return
    preprocessor_path = Path(policy_path).expanduser() / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        return

    with preprocessor_path.open() as f:
        config = json.load(f)

    changed = False
    for step in config.get("steps", []):
        cfg = step.get("config", {})
        if step.get("registry_name") == "tokenizer_processor" and cfg.get("tokenizer_name") != tokenizer_name:
            cfg["tokenizer_name"] = tokenizer_name
            changed = True

    if changed:
        with preprocessor_path.open("w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"patched_tokenizer={preprocessor_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/smolvla_synthetic_one_arm_hand")
    parser.add_argument("--root", default=str(REPO_ROOT / "outputs" / "synthetic_lerobot_one_arm_hand"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "train" / "smolvla_synthetic_smoke"))
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument(
        "--vlm-model-name",
        default=None,
        help="Optional local path or Hub id for HuggingFaceTB/SmolVLM2-500M-Video-Instruct.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=64)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument(
        "--rename-front-to-camera1",
        action="store_true",
        help="Use one synthetic camera as pretrained camera1 and let SmolVLA pad camera2/camera3.",
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = generate_dataset(args)
    if not args.skip_train:
        run_train(args, dataset_root)


if __name__ == "__main__":
    main()
