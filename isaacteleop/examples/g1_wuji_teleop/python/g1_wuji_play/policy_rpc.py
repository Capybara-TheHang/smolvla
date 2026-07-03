"""Client for the SmolVLA policy server running in the lerobot environment.

中文说明：
这个文件运行在 Isaac Lab 环境里，只负责和另一个 ``lerobot-smolvla``
进程通信。Isaac 端不加载 SmolVLA 模型，只发送实时观测并接收 action，
从而避免两个环境依赖冲突。
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np


RPC_HEADER = struct.Struct("!Q")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    # TCP 不保消息边界；这里按指定字节数循环读取，保证 payload 完整。
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
    # 和 play.py server 保持同一协议：8 字节长度头 + pickle 数据。
    header = _recv_exact(sock, RPC_HEADER.size)
    (size,) = RPC_HEADER.unpack(header)
    if size <= 0:
        raise ValueError(f"invalid RPC payload size: {size}")
    return pickle.loads(_recv_exact(sock, int(size)))


def _send_rpc(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(RPC_HEADER.pack(len(data)))
    sock.sendall(data)


class RemoteSmolVLAClient:
    """Persistent localhost client for cross-environment policy inference.

    这个 client 持有长连接，仿真循环每次需要新 action chunk 时调用
    ``predict_action_chunk``。连接断开会自动重连一次，便于重启推理 server。
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 5555, timeout_s: float = 30.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        return sock

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 推理服务重启或连接断开时，先关闭旧 socket，再自动重连并重发一次。
        try:
            sock = self._connect()
            _send_rpc(sock, payload)
            response = _recv_rpc(sock)
        except (OSError, EOFError):
            self.close()
            sock = self._connect()
            _send_rpc(sock, payload)
            response = _recv_rpc(sock)

        if not response.get("ok", False):
            error = response.get("error", "unknown policy server error")
            trace = response.get("traceback")
            if trace:
                raise RuntimeError(f"{error}\n{trace}")
            raise RuntimeError(str(error))
        return response

    def ping(self) -> dict[str, Any]:
        return self.request({"command": "ping"})

    def reset(self) -> None:
        self.request({"command": "reset"})

    def predict_action_chunk(
        self,
        *,
        state: Any,
        images: dict[str, Any],
        task: str,
        steps: int | None = None,
    ) -> np.ndarray:
        # state: [26]，images: observation.images.front/table 的 HWC RGB，
        # 返回 actions: [T, 26]，T 由 steps/chunk_steps 控制。
        response = self.request(
            {
                "command": "predict_action_chunk",
                "state": np.asarray(state, dtype=np.float32),
                "images": {key: np.asarray(value) for key, value in images.items()},
                "task": str(task),
                "steps": steps,
            }
        )
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise RuntimeError(f"policy server returned invalid actions shape: {actions.shape}")
        return actions
