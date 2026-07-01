#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

SMOLVLA_ROOT="${SMOLVLA_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
TENSORBOARD_LOG_ROOT="${TENSORBOARD_LOG_ROOT:-${SCRIPT_DIR}/tensorboard_runs}"

case "${1:-start}" in
  start)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_with_tensorboard.py" \
      --start-only \
      --log-root "${TENSORBOARD_LOG_ROOT}" \
      --host "${TENSORBOARD_HOST}" \
      --port "${TENSORBOARD_PORT}"
    ;;
  stop)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_with_tensorboard.py" \
      --stop \
      --log-root "${TENSORBOARD_LOG_ROOT}"
    ;;
  *)
    echo "Usage: $0 [start|stop]" >&2
    exit 2
    ;;
esac
