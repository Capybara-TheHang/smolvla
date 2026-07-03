#!/usr/bin/env python
"""SmolVLA policy player for the G1-Wuji example.

This module is intentionally narrow: it loads a trained SmolVLA checkpoint and
returns the next action chunk for one observation. The Isaac/IK deployment layer
should import :class:`G1WujiSmolVLAPlayer` instead of reimplementing policy
loading.

中文说明：
这个文件运行在 ``lerobot-smolvla`` 环境里，只负责模型推理。
真实部署时建议用 ``--serve`` 启动常驻推理服务；Isaac 仿真进程通过
本机 TCP 请求 action，这样 SmolVLA 依赖和 Isaac Lab 依赖不会混在一个
Python 环境里。
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import traceback
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


OBS_STATE = "observation.state"
ACTION = "action"
DEFAULT_TASK = "Teleoperate the right arm and Wuji dexterous hand."
DEFAULT_VLM_MODEL_NAME = REPO_ROOT / "checkpoints" / "SmolVLM2-500M-Video-Instruct"
DEFAULT_TRAIN_OUTPUT = REPO_ROOT / "lerobot" / "outputs" / "train" / "g1_wuji_112339_smolvla"
RPC_HEADER = struct.Struct("!Q")


def resolve_checkpoint_path(path: str | Path | None) -> Path:
    """Resolve an output/checkpoint/pretrained_model path to ``pretrained_model``.

    训练输出可能传入 ``output_dir``，也可能直接传入
    ``checkpoints/003000/pretrained_model``。这里统一解析到真正可加载的
    ``pretrained_model`` 目录，避免部署命令里路径格式不一致。
    """

    candidate = Path(path).expanduser().resolve() if path else DEFAULT_TRAIN_OUTPUT
    if (candidate / "config.json").exists():
        return candidate

    # 支持直接传入单个 step checkpoint 目录，例如：
    # .../checkpoints/022000 -> .../checkpoints/022000/pretrained_model
    step_pretrained = candidate / "pretrained_model"
    if (step_pretrained / "config.json").exists():
        return step_pretrained.resolve()

    last = candidate / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last.resolve()

    checkpoint_root = candidate / "checkpoints"
    if checkpoint_root.exists():
        checkpoint_dirs = sorted(
            (
                item
                for item in checkpoint_root.iterdir()
                if item.is_dir() and (item / "pretrained_model" / "config.json").exists()
            ),
            key=lambda item: item.name,
        )
        if checkpoint_dirs:
            return (checkpoint_dirs[-1] / "pretrained_model").resolve()

    raise FileNotFoundError(
        f"Could not find a pretrained_model checkpoint under {candidate}. "
        "Pass --checkpoint pointing to either output_dir, checkpoints/last/pretrained_model, "
        "or a numeric checkpoints/<step>/pretrained_model directory."
    )


def _resolve_vlm_model_name(config_vlm_model_name: str, override: str | None) -> str:
    if override:
        return str(Path(override).expanduser().resolve())

    config_path = Path(config_vlm_model_name).expanduser()
    if config_path.exists():
        return str(config_path.resolve())

    if DEFAULT_VLM_MODEL_NAME.exists():
        return str(DEFAULT_VLM_MODEL_NAME.resolve())

    return config_vlm_model_name


def _as_chw_float_image(image: Any) -> torch.Tensor:
    """Convert a HWC/CHW image into a float32 CHW tensor in [0, 1].

    Isaac/录制数据通常是 HWC uint8 RGB；SmolVLA processor 期望 CHW float。
    这里集中做格式转换，保证 server 和离线 npz 调试走同一条预处理路径。
    """

    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
    else:
        tensor = torch.as_tensor(np.asarray(image))

    if tensor.ndim != 3:
        raise ValueError(f"Image must be rank 3 HWC or CHW, got shape {tuple(tensor.shape)}")

    # Raw recordings are HWC RGB. LeRobot policy inputs are CHW.
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    elif tensor.shape[0] not in (1, 3):
        raise ValueError(f"Could not infer image channel dimension from shape {tuple(tensor.shape)}")

    tensor = tensor.to(dtype=torch.float32)
    if float(tensor.max().item()) > 1.5:
        tensor = tensor / 255.0
    return tensor.contiguous()


def make_observation(
    *,
    state: Any,
    images: dict[str, Any],
    task: str = DEFAULT_TASK,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        OBS_STATE: torch.as_tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32),
        "task": str(task),
    }
    for key, image in images.items():
        observation[key] = _as_chw_float_image(image)
    return observation


def load_npz_observation(
    npz_path: str | Path,
    *,
    frame_index: int = 0,
    task: str | None = None,
) -> dict[str, Any]:
    path = Path(npz_path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as data:
        if OBS_STATE not in data.files:
            raise KeyError(f"{path} does not contain {OBS_STATE!r}")

        frame_count = len(data[OBS_STATE])
        if frame_count == 0:
            raise ValueError(f"{path} contains no frames")
        index = frame_index if frame_index >= 0 else frame_count + frame_index
        if index < 0 or index >= frame_count:
            raise IndexError(f"frame_index={frame_index} out of range for {frame_count} frames")

        image_keys = sorted(key for key in data.files if key.startswith("observation.images."))
        if not image_keys:
            raise KeyError(f"{path} does not contain observation.images.* arrays")

        if task is None:
            if "task" in data.files:
                task = str(data["task"][index])
            else:
                task = DEFAULT_TASK

        return make_observation(
            state=data[OBS_STATE][index],
            images={key: data[key][index] for key in image_keys},
            task=task,
        )


def load_json_observation(path: str | Path) -> dict[str, Any]:
    """Load an observation JSON with state and image file paths.

    Expected shape:
        {
          "task": "...",
          "state": [...],
          "images": {
            "observation.images.front": "/path/front.npy",
            "observation.images.table": "/path/table.npy"
          }
        }

    Image values may also be nested lists, but file paths are preferred.
    ``.npy`` files are loaded with numpy; other image files are loaded with PIL.
    """

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if "state" not in payload:
        raise KeyError("Observation JSON must contain 'state'")
    if "images" not in payload:
        raise KeyError("Observation JSON must contain 'images'")

    images: dict[str, Any] = {}
    for key, value in payload["images"].items():
        if isinstance(value, str):
            image_path = Path(value).expanduser()
            if image_path.suffix == ".npy":
                images[key] = np.load(image_path)
            else:
                from PIL import Image

                images[key] = np.asarray(Image.open(image_path).convert("RGB"))
        else:
            images[key] = np.asarray(value)

    return make_observation(
        state=payload["state"],
        images=images,
        task=str(payload.get("task") or DEFAULT_TASK),
    )


class G1WujiSmolVLAPlayer:
    """Reusable SmolVLA inference wrapper for G1-Wuji deployment.

    这个类只存在于推理进程里。Isaac 进程不要直接 import 它，否则会把
    transformers/torch/lerobot 的运行时依赖带进 Isaac Lab 环境。
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str = "cuda",
        vlm_model_name: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.checkpoint = resolve_checkpoint_path(checkpoint)
        self.device = device

        config = PreTrainedConfig.from_pretrained(
            self.checkpoint,
            local_files_only=local_files_only,
        )
        config.device = device
        if hasattr(config, "vlm_model_name"):
            config.vlm_model_name = _resolve_vlm_model_name(config.vlm_model_name, vlm_model_name)

        # checkpoint 里保存的 tokenizer/device 可能来自另一台机器。
        # 部署时用当前命令行传入的本地路径和设备覆盖，避免路径漂移。
        processor_overrides = {
            "device_processor": {"device": device},
        }
        if hasattr(config, "vlm_model_name"):
            processor_overrides["tokenizer_processor"] = {"tokenizer_name": config.vlm_model_name}

        self.policy = SmolVLAPolicy.from_pretrained(
            self.checkpoint,
            config=config,
            local_files_only=local_files_only,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides=processor_overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.policy.reset()

    @property
    def action_dim(self) -> int:
        return int(self.policy.config.output_features[ACTION].shape[0])

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.n_action_steps)

    def reset(self) -> None:
        self.policy.reset()

    @torch.inference_mode()
    def predict_action_chunk(self, observation: dict[str, Any], *, steps: int | None = None) -> np.ndarray:
        batch = self.preprocessor(observation)
        actions = self.policy.predict_action_chunk(batch)
        actions = self.postprocessor(actions)
        actions_np = actions.detach().cpu().numpy()
        if actions_np.ndim != 3:
            raise RuntimeError(f"Expected action chunk shape [B, T, D], got {actions_np.shape}")
        chunk = actions_np[0]
        if steps is not None:
            chunk = chunk[: int(steps)]
        return chunk.astype(np.float32, copy=False)

    @torch.inference_mode()
    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        batch = self.preprocessor(observation)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        action_np = action.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]
        return action_np.astype(np.float32, copy=False)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    # TCP 是字节流，recv 不保证一次拿到完整 payload，所以需要按长度读满。
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_rpc(sock: socket.socket) -> dict[str, Any]:
    # 协议格式：8 字节大端 payload 长度 + pickle payload。
    # 这里只绑定 localhost 使用，不作为对外网络接口。
    header = _recv_exact(sock, RPC_HEADER.size)
    (size,) = RPC_HEADER.unpack(header)
    if size <= 0:
        raise ValueError(f"invalid RPC payload size: {size}")
    return pickle.loads(_recv_exact(sock, int(size)))


def _send_rpc(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(RPC_HEADER.pack(len(data)))
    sock.sendall(data)


def _handle_rpc_request(player: G1WujiSmolVLAPlayer, request: dict[str, Any]) -> dict[str, Any]:
    # Isaac 端只需要发 numpy state/images/task；server 端负责转 tensor、跑 processor、
    # 调模型、再把 action chunk 转回 numpy。
    command = str(request.get("command") or "predict_action_chunk")
    if command == "ping":
        return {"ok": True, "pong": True, "checkpoint": str(player.checkpoint)}
    if command == "reset":
        player.reset()
        return {"ok": True, "reset": True}
    if command == "predict_action_chunk":
        observation = make_observation(
            state=request["state"],
            images=request["images"],
            task=str(request.get("task") or DEFAULT_TASK),
        )
        actions = player.predict_action_chunk(observation, steps=request.get("steps"))
        return {
            "ok": True,
            "actions": actions,
            "action_dim": int(actions.shape[-1]),
            "num_actions": int(actions.shape[0]),
        }
    if command == "select_action":
        observation = make_observation(
            state=request["state"],
            images=request["images"],
            task=str(request.get("task") or DEFAULT_TASK),
        )
        action = player.select_action(observation)
        return {"ok": True, "action": action, "action_dim": int(action.shape[-1])}
    raise ValueError(f"unsupported RPC command: {command}")


def serve_policy(
    *,
    checkpoint: str | Path | None,
    device: str,
    vlm_model_name: str | None,
    host: str,
    port: int,
) -> int:
    # 推理 server 是一个常驻进程：模型只加载一次，后续每帧只处理 RPC 请求。
    # 这比 Isaac 端每次调用命令行脚本重新加载模型快得多。
    player = G1WujiSmolVLAPlayer(
        checkpoint=checkpoint,
        device=device,
        vlm_model_name=vlm_model_name,
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(8)
    print(
        f"G1-Wuji SmolVLA policy server listening on {host}:{port} "
        f"checkpoint={player.checkpoint} device={player.device}",
        flush=True,
    )
    try:
        while True:
            conn, addr = server.accept()
            print(f"G1-Wuji policy client connected: {addr}", flush=True)
            with conn:
                while True:
                    try:
                        request = _recv_rpc(conn)
                    except EOFError:
                        break
                    except Exception as exc:  # noqa: BLE001
                        _send_rpc(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                        break

                    if request.get("command") == "shutdown":
                        _send_rpc(conn, {"ok": True, "shutdown": True})
                        return 0

                    try:
                        command = str(request.get("command") or "predict_action_chunk")
                        request_start = time.monotonic()
                        if command in {"ping", "reset"}:
                            print(f"G1-Wuji policy request: command={command}", flush=True)
                        elif command == "predict_action_chunk":
                            images = request.get("images") or {}
                            image_shapes = {
                                str(key): tuple(int(dim) for dim in np.asarray(value).shape)
                                for key, value in images.items()
                            }
                            print(
                                f"G1-Wuji policy request: command={command} "
                                f"image_shapes={image_shapes} steps={request.get('steps')}",
                                flush=True,
                            )
                        response = _handle_rpc_request(player, request)
                        if command == "predict_action_chunk":
                            print(
                                f"G1-Wuji policy response: actions_shape="
                                f"{tuple(int(dim) for dim in np.asarray(response.get('actions')).shape)} "
                                f"latency_s={time.monotonic() - request_start:.3f}",
                                flush=True,
                            )
                    except Exception as exc:  # noqa: BLE001
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    _send_rpc(conn, response)
    finally:
        server.close()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_actions(path: str | Path | None, payload: dict[str, Any], *, npy: bool) -> None:
    if path is None:
        print(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2))
        return

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if npy:
        np.save(out_path, np.asarray(payload["actions"], dtype=np.float32))
        return
    out_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to output_dir, checkpoints/last/pretrained_model, or "
            "checkpoints/<step>/pretrained_model. Defaults to the local g1_wuji training output."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vlm-model-name",
        default=None,
        help="Override local SmolVLM2 path if the checkpoint was trained on another machine.",
    )
    parser.add_argument("--npz", default=None, help="Raw isaacteleop trajectory_*.npz used as observation source.")
    parser.add_argument("--observation-json", default=None, help="Observation JSON used as observation source.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--task", default=None)
    parser.add_argument("--steps", type=int, default=None, help="Limit number of action steps written.")
    parser.add_argument("--single-action", action="store_true", help="Write only policy.select_action output.")
    parser.add_argument("--output", default=None, help="Output .json or .npy path. Defaults to stdout JSON.")
    parser.add_argument("--npy", action="store_true", help="Write actions as .npy instead of JSON.")
    parser.add_argument("--serve", action="store_true", help="Run a localhost policy RPC server.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy server bind host.")
    parser.add_argument("--port", type=int, default=5555, help="Policy server bind port.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.serve:
        return serve_policy(
            checkpoint=args.checkpoint,
            device=args.device,
            vlm_model_name=args.vlm_model_name,
            host=args.host,
            port=args.port,
        )

    if bool(args.npz) == bool(args.observation_json):
        raise SystemExit("Pass exactly one of --npz or --observation-json.")

    observation = (
        load_npz_observation(args.npz, frame_index=args.frame_index, task=args.task)
        if args.npz
        else load_json_observation(args.observation_json)
    )
    player = G1WujiSmolVLAPlayer(
        checkpoint=args.checkpoint,
        device=args.device,
        vlm_model_name=args.vlm_model_name,
    )

    if args.single_action:
        actions = player.select_action(observation)[None, :]
    else:
        actions = player.predict_action_chunk(observation, steps=args.steps)

    payload = {
        "checkpoint": str(player.checkpoint),
        "device": player.device,
        "action_dim": int(actions.shape[-1]),
        "num_actions": int(actions.shape[0]),
        "actions": actions,
    }
    _write_actions(args.output, payload, npy=args.npy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
