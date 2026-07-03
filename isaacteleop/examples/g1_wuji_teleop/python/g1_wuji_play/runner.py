"""Top-level runner for G1-Wuji SmolVLA playback.

中文说明：
这个入口支持两个后端：
- dry-run：离线读取 npz，在当前 Python 进程里直接加载模型，便于调试 action。
- isaac：启动仿真，但模型推理由另一个 lerobot-smolvla 进程提供。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .action import ACTION_DIM, decode_action_chunk
from .executor import DryRunActionExecutor
from .observation import load_npz_frame
from .paths import default_checkpoint_root, default_vlm_model_name
from .policy_bridge import SmolVLAPolicyBridge


def _backend_from_argv(argv: Sequence[str]) -> str:
    # 先解析 backend，再决定是否加载 Isaac Lab 参数；否则普通 dry-run 环境会被
    # isaaclab 依赖卡住。
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--backend", choices=("dry-run", "isaac"), default="dry-run")
    parsed, _ = pre_parser.parse_known_args(list(argv))
    return str(parsed.backend)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    backend = _backend_from_argv(argv)
    parser = argparse.ArgumentParser(
        description=(
            "Run the G1-Wuji SmolVLA play deployment entry. The current backend "
            "performs policy inference and writes decoded commands; Isaac execution "
            "can be connected behind the same action executor interface."
        )
    )
    parser.add_argument("--backend", choices=("dry-run", "isaac"), default=backend)
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint_root()),
        help=(
            "Path to training output_dir, checkpoints/last/pretrained_model, "
            "or checkpoints/<step>/pretrained_model."
        ),
    )
    parser.add_argument("--policy-device", default="cuda")
    if backend != "isaac":
        parser.add_argument("--device", dest="policy_device", help="Compatibility alias for --policy-device.")
    parser.add_argument(
        "--vlm-model-name",
        default=str(default_vlm_model_name()) if default_vlm_model_name().exists() else None,
        help="Local SmolVLM2 path used by tokenizer/backbone.",
    )
    parser.add_argument(
        "--npz",
        required=False,
        help="Raw isaacteleop trajectory_*.npz used as the initial observation source.",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Language task. Isaac backend prompts for this after the simulation is ready; "
            "dry-run uses the npz task when omitted."
        ),
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of actions to output from the chunk.")
    parser.add_argument(
        "--single-action",
        action="store_true",
        help="Use select_action and output a one-step command instead of a full chunk.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for decoded commands. Defaults to stdout.",
    )
    parser.add_argument(
        "--raw-action-output",
        default=None,
        help="Optional .npy path for raw policy actions before decoding.",
    )
    parser.add_argument("--max-position-delta-m", type=float, default=0.08)
    parser.add_argument("--max-rotation-delta-rad", type=float, default=0.35)

    if backend == "isaac":
        from .simulation import add_isaac_app_launcher_args

        parser.add_argument("--config", default=None)
        parser.add_argument("--robot-usd", default=None)
        parser.add_argument("--robot-prim", default=None)
        parser.add_argument("--scene-usd", default=None)
        parser.add_argument("--robot-height", type=float, default=None)
        parser.add_argument("--light-intensity", type=float, default=3000.0)
        parser.add_argument("--duration-s", type=float, default=0.0)
        parser.add_argument("--max-steps", type=int, default=0)
        parser.add_argument("--sim-hz", type=float, default=60.0)
        parser.add_argument("--warmup-steps", type=int, default=20)
        parser.add_argument(
            "--camera-attach-retries",
            type=int,
            default=6,
            help="Number of attempts to attach Replicator camera readers after scene warm-up.",
        )
        parser.add_argument(
            "--camera-attach-retry-steps",
            type=int,
            default=15,
            help="Simulation/render steps to advance before retrying camera reader attachment.",
        )
        parser.add_argument("--policy-fps", type=float, default=15.0)
        parser.add_argument("--chunk-steps", type=int, default=10)
        parser.add_argument("--ik-smoothing-alpha", type=float, default=0.80)
        parser.add_argument("--policy-host", default="127.0.0.1")
        parser.add_argument("--policy-port", type=int, default=5555)
        parser.add_argument("--policy-timeout-s", type=float, default=60.0)
        parser.add_argument(
            "--status-log-interval-s",
            type=float,
            default=2.0,
            help="Seconds between Isaac play status messages while waiting/running.",
        )
        parser.add_argument(
            "--action-log-every",
            type=int,
            default=10,
            help="Print one execution status message every N policy actions; <=0 disables it.",
        )
        try:
            add_isaac_app_launcher_args(parser)
        except ModuleNotFoundError as exc:
            # 在非 Isaac 环境里允许查看 --help；真正运行 isaac backend 时必须换环境。
            if any(item in {"-h", "--help"} for item in argv):
                pass
            else:
                raise SystemExit(
                    "--backend=isaac must be run with an Isaac Lab Python environment "
                    f"that can import isaaclab.app.AppLauncher. Missing module: {exc.name}"
                ) from exc

    return parser.parse_args(list(argv))


def play_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.backend == "isaac":
        # 仿真后端不会本地加载模型，只会连接 policy server。
        from .simulation import run_isaac_play

        return run_isaac_play(args)

    if not args.npz:
        raise SystemExit("--npz is required when --backend=dry-run.")

    # dry-run 后端用于离线检查：从历史轨迹取一帧 observation，直接输出 action。
    frame = load_npz_frame(args.npz, frame_index=args.frame_index, task=args.task)
    bridge = SmolVLAPolicyBridge(
        checkpoint=args.checkpoint,
        device=args.policy_device,
        vlm_model_name=args.vlm_model_name,
    )
    observation = bridge.make_observation(state=frame.state, images=frame.images, task=frame.task)

    if args.single_action:
        actions = bridge.select_action(observation)[None, :]
    else:
        actions = bridge.predict_action_chunk(observation, steps=args.steps)

    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"Expected raw actions shape [T, {ACTION_DIM}], got {actions.shape}")

    if args.raw_action_output:
        raw_path = Path(args.raw_action_output).expanduser().resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, actions.astype(np.float32, copy=False))

    commands = decode_action_chunk(
        state=frame.state,
        actions=actions,
        max_position_delta_m=args.max_position_delta_m,
        max_rotation_delta_rad=args.max_rotation_delta_rad,
    )
    DryRunActionExecutor(args.output).execute(commands)

    print(
        f"G1-Wuji play dry-run complete: checkpoint={bridge.checkpoint} "
        f"actions={len(commands)} action_dim={actions.shape[1]} task={frame.task!r}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(play_main())
