#!/usr/bin/env python3
"""Run g1_wuji training while mirroring console metrics to TensorBoard."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = SCRIPT_DIR / "tensorboard_runs"
STATE_DIR = SCRIPT_DIR / ".tensorboard"
STATE_FILE = STATE_DIR / "tensorboard_state.json"

METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_./-]*):"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?[KMBTQkmbtq]?)"
)
EVAL_LOSS_RE = re.compile(
    r"step\s+(?P<step>\d+):\s+eval_loss=(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
)

METRIC_TAGS = {
    "smpl": "progress/samples",
    "ep": "progress/episodes",
    "epch": "progress/epochs",
    "loss": "train/loss",
    "grdn": "train/grad_norm",
    "grad_norm": "train/grad_norm",
    "lr": "optimizer/lr",
    "updt_s": "time/update_s",
    "data_s": "time/dataloading_s",
    "smp/s": "throughput/samples_per_s",
    "mem_gb": "system/gpu_mem_gb",
}

HPARAM_KEYS = [
    "SMOLVLA_ROOT",
    "POLICY_PATH",
    "VLM_MODEL_NAME",
    "DATASET_REPO_ID",
    "DATASET_ROOT",
    "OUTPUT_DIR",
    "DEVICE",
    "RENAME_MAP",
    "EMPTY_CAMERAS",
    "STEPS",
    "BATCH_SIZE",
    "NUM_WORKERS",
    "SAVE_FREQ",
    "LOG_FREQ",
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--train-script", type=Path)
    parser.add_argument("--start-only", action="store_true", help="start/reuse TensorBoard and exit")
    parser.add_argument("--stop", action="store_true", help="stop the managed TensorBoard process")
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="write TensorBoard event files without starting the web server",
    )
    return parser.parse_known_args()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_tensorboard() -> int:
    if not STATE_FILE.exists():
        print("No managed TensorBoard state found.")
        return 0

    state = json.loads(STATE_FILE.read_text())
    pid = int(state.get("pid", 0))
    if pid and process_alive(pid):
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped TensorBoard pid={pid}.")
    else:
        print("Managed TensorBoard process is not running.")
    STATE_FILE.unlink(missing_ok=True)
    return 0


def port_is_free(host: str, port: int) -> bool:
    bind_host = host if host not in {"localhost"} else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 50):
        if port_is_free(host, port):
            return port
    raise RuntimeError(f"No free TensorBoard port found in {preferred_port}-{preferred_port + 49}.")


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def display_url(host: str, port: int) -> str:
    shown_host = get_lan_ip() if host in {"0.0.0.0", "::"} else host
    return f"http://{shown_host}:{port}"


def read_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None
    pid = int(state.get("pid", 0))
    if pid and process_alive(pid):
        return state
    return None


def start_tensorboard(log_root: Path, host: str, port: int, no_launch: bool) -> dict[str, Any] | None:
    if no_launch:
        return None

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    state = read_state()
    if state is not None:
        print(f"TensorBoard already running: {display_url(state['host'], int(state['port']))}")
        print(f"TensorBoard log root: {state['log_root']}")
        return state

    chosen_port = choose_port(host, port)
    log_file = STATE_DIR / "tensorboard.log"
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_root),
        "--host",
        host,
        "--port",
        str(chosen_port),
        "--reload_interval",
        "5",
    ]

    with log_file.open("ab") as out:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=os.environ.copy(),
        )

    time.sleep(1.0)
    if proc.poll() is not None:
        tail = ""
        if log_file.exists():
            tail = log_file.read_text(errors="replace")[-4000:]
        raise RuntimeError(
            "TensorBoard failed to start. Install it with `pip install tensorboard` "
            f"or check {log_file}.\n{tail}"
        )

    state = {
        "pid": proc.pid,
        "host": host,
        "port": chosen_port,
        "log_root": str(log_root),
        "log_file": str(log_file),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    print(f"TensorBoard URL: {display_url(host, chosen_port)}")
    print(f"TensorBoard log root: {log_root}")
    print(f"TensorBoard pid: {proc.pid}")
    return state


def parse_number(text: str) -> float:
    multipliers = {
        "": 1.0,
        "K": 1_000.0,
        "M": 1_000_000.0,
        "B": 1_000_000_000.0,
        "T": 1_000_000_000_000.0,
        "Q": 1_000_000_000_000_000.0,
    }
    suffix = text[-1].upper() if text[-1].isalpha() else ""
    number = text[:-1] if suffix else text
    return float(number) * multipliers.get(suffix, 1.0)


def sanitize_tag(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./-]+", "_", key).strip("_") or "metric"


def git_value(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def collect_hparams(command: list[str], cwd: Path) -> dict[str, Any]:
    hparams: dict[str, Any] = {key: os.environ[key] for key in HPARAM_KEYS if key in os.environ}
    hparams["command"] = " ".join(command)
    hparams["working_dir"] = str(cwd)
    hparams["python"] = sys.executable
    hparams["git_commit"] = git_value(["rev-parse", "HEAD"], cwd)
    hparams["git_branch"] = git_value(["branch", "--show-current"], cwd)
    return hparams


def markdown_table(items: dict[str, Any]) -> str:
    rows = ["| key | value |", "| --- | --- |"]
    for key, value in sorted(items.items()):
        rows.append(f"| `{key}` | `{value}` |")
    return "\n".join(rows)


def collect_system_text() -> str:
    lines = [f"hostname: {socket.gethostname()}", f"lan_ip: {get_lan_ip()}"]
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        output = ""
    if output:
        lines.append("gpu:")
        lines.extend(f"- {line}" for line in output.splitlines())
    return "\n".join(lines)


class TensorBoardMirror:
    def __init__(self, run_dir: Path, hparams: dict[str, Any]):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            try:
                from tensorboardX import SummaryWriter  # type: ignore[no-redef]
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise RuntimeError(
                    "TensorBoard writer is unavailable. Activate the training env first "
                    "or install it with `pip install tensorboard tensorboardX`."
                ) from exc

        self.writer = SummaryWriter(log_dir=str(run_dir), flush_secs=5)
        self.hparams = hparams
        self.last_scalars: dict[str, float] = {}
        self.writer.add_text("hparams/config", markdown_table(hparams), 0)
        self.writer.add_text("system/environment", collect_system_text(), 0)

    def observe_line(self, line: str) -> None:
        eval_match = EVAL_LOSS_RE.search(line)
        if eval_match:
            step = int(eval_match.group("step"))
            value = float(eval_match.group("value"))
            self.writer.add_scalar("eval/loss", value, step)
            self.last_scalars["eval/loss"] = value
            self.writer.flush()
            return

        step_pos = line.find("step:")
        if step_pos < 0:
            return
        metric_text = line[step_pos:]
        values = {match.group("key"): parse_number(match.group("value")) for match in METRIC_RE.finditer(metric_text)}
        if "step" not in values:
            return
        step = int(values["step"])

        for key, value in values.items():
            if key == "step":
                continue
            tag = METRIC_TAGS.get(key, f"train/{sanitize_tag(key)}")
            self.writer.add_scalar(tag, value, step)
            self.last_scalars[tag] = value
        self.writer.flush()

    def close(self) -> None:
        metrics = {}
        if "train/loss" in self.last_scalars:
            metrics["hparam/final_train_loss"] = self.last_scalars["train/loss"]
        if "eval/loss" in self.last_scalars:
            metrics["hparam/final_eval_loss"] = self.last_scalars["eval/loss"]
        if metrics:
            try:
                hparams = {key: value for key, value in self.hparams.items() if isinstance(value, (str, int, float, bool))}
                self.writer.add_hparams(hparams, metrics)
            except Exception:
                pass
        self.writer.flush()
        self.writer.close()


def run_training(args: argparse.Namespace, train_args: list[str]) -> int:
    if args.train_script is None:
        raise SystemExit("--train-script is required unless --start-only or --stop is used.")

    log_root = args.log_root.resolve()
    run_name = args.run_name or f"g1_wuji_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    command = ["bash", str(args.train_script), *train_args]
    env = os.environ.copy()
    env["G1_WUJI_TENSORBOARD_WRAPPED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    hparams = collect_hparams(command, Path.cwd())
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2, ensure_ascii=False) + "\n")
    (run_dir / "command.txt").write_text(" ".join(command) + "\n")

    mirror = TensorBoardMirror(run_dir, hparams)
    stdout_log = run_dir / "train_stdout.log"

    print(f"TensorBoard run: {run_dir}")
    print(f"Training command: {' '.join(command)}")

    proc: subprocess.Popen[str] | None = None
    return_code = 1
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None

        with stdout_log.open("a", buffering=1) as log_file:
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                mirror.observe_line(line)

        return_code = proc.wait()
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            return_code = proc.wait()
        else:
            return_code = 130
    finally:
        mirror.close()

    print(f"Training finished with exit code {return_code}.")
    print(f"Saved TensorBoard history: {run_dir}")
    return return_code


def main() -> int:
    try:
        args, train_args = parse_args()
        log_root = args.log_root.resolve()

        if args.stop:
            return stop_tensorboard()

        if not args.no_launch:
            start_tensorboard(log_root, args.host, args.port, args.no_launch)

        if args.start_only:
            return 0

        return run_training(args, train_args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
