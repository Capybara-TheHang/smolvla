#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Repo root. Override this if checkpoints/datasets live somewhere else.
SMOLVLA_ROOT="${SMOLVLA_ROOT:-${REPO_ROOT}}"

# Prefer the active conda env libraries so torchcodec can find the matching ffmpeg/libstdc++.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# 预训练 SmolVLA checkpoint。这里用本地路径，避免训练时访问 Hugging Face。
POLICY_PATH="${POLICY_PATH:-${SMOLVLA_ROOT}/checkpoints/smolvla_base}"

# SmolVLA 使用的 VLM backbone 本地路径，避免加载 HuggingFaceTB/SmolVLM2-500M-Video-Instruct 时联网。
VLM_MODEL_NAME="${VLM_MODEL_NAME:-${SMOLVLA_ROOT}/checkpoints/SmolVLM2-500M-Video-Instruct}"

# 转换数据时使用的 repo_id，必须和 convert.sh 里的 REPO_ID 保持一致。
DATASET_REPO_ID="${DATASET_REPO_ID:-local/g1_wuji_112339}"

# 转换后的 LeRobotDataset 根目录，不是 isaacteleop 的原始 session 目录。
DATASET_ROOT="${DATASET_ROOT:-${SMOLVLA_ROOT}/lerobot/examples/g1_wuji/output/dataset1}"

# 训练输出目录。checkpoint、配置和日志会写到这里。
OUTPUT_DIR="${OUTPUT_DIR:-${SMOLVLA_ROOT}/lerobot/outputs/train/g1_wuji_112339_smolvla}"

# 训练设备。NVIDIA GPU 用 cuda；没有 GPU 可改成 cpu，但会非常慢。
DEVICE="${DEVICE:-cuda}"

# 当前数据有 front/table 两个相机；smolvla_base 期望 camera1/camera2/camera3。
# 这里把数据集相机名映射到预训练模型的相机名。
DEFAULT_RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.table":"observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-${DEFAULT_RENAME_MAP}}"

# smolvla_base 期望 3 个相机，当前只有 2 个，所以补 1 个空相机。
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

# 小数据集先用 3000 步确认流程和 loss。录更多数据后可以改到 30000 或更高。
STEPS="${STEPS:-3000}"

# 每步 batch size。显存不够就保持 1；显存充足可试 2/4。
BATCH_SIZE="${BATCH_SIZE:-10}"

# DataLoader worker 数。先设 0，便于调试；数据稳定后可改 2/4。
NUM_WORKERS="${NUM_WORKERS:-0}"

# 每隔多少 step 保存一次 checkpoint。
SAVE_FREQ="${SAVE_FREQ:-1000}"

# 每隔多少 step 打印一次训练日志。
LOG_FREQ="${LOG_FREQ:-10}"

# TensorBoard settings. History is kept under this directory and can be reopened later.
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-true}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
TENSORBOARD_LOG_ROOT="${TENSORBOARD_LOG_ROOT:-${SCRIPT_DIR}/tensorboard_runs}"
TENSORBOARD_RUN_NAME="${TENSORBOARD_RUN_NAME:-g1_wuji_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export SMOLVLA_ROOT POLICY_PATH VLM_MODEL_NAME DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR
export DEVICE RENAME_MAP EMPTY_CAMERAS STEPS BATCH_SIZE NUM_WORKERS SAVE_FREQ LOG_FREQ
export TENSORBOARD_HOST TENSORBOARD_PORT TENSORBOARD_LOG_ROOT TENSORBOARD_RUN_NAME

case "${ENABLE_TENSORBOARD}" in
  1|true|TRUE|yes|YES)
    if [[ "${G1_WUJI_TENSORBOARD_WRAPPED:-0}" != "1" ]]; then
      exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_with_tensorboard.py" \
        --log-root "${TENSORBOARD_LOG_ROOT}" \
        --run-name "${TENSORBOARD_RUN_NAME}" \
        --host "${TENSORBOARD_HOST}" \
        --port "${TENSORBOARD_PORT}" \
        --train-script "$0" \
        "$@"
    fi
    ;;
esac

lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --policy.vlm_model_name="${VLM_MODEL_NAME}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.device="${DEVICE}" \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --rename_map="${RENAME_MAP}" \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --output_dir="${OUTPUT_DIR}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --persistent_workers=false \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --wandb.enable=false \
  "$@"
