#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOLVLA_ROOT="${SMOLVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# 转换后的 LeRobotDataset 输出目录。训练时 lerobot-train --dataset.root 指向这里。
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/output/dataset4}"

# 本地训练用的数据集 repo_id，训练时 --dataset.repo_id 要保持一致。
REPO_ID="${REPO_ID:-local/g1_wuji_3}"

# isaacteleop 录制出来的原始 session 目录。
RAW_DIR="${RAW_DIR:-${SMOLVLA_ROOT}/isaacteleop/examples/g1_wuji_teleop/data/dataset/20260706_224829}"

python "${SCRIPT_DIR}/convert_isaacteleop_to_lerobot.py" \
  --raw-dir "${RAW_DIR}" \
  --out-root "${OUT_ROOT}" \
  --repo-id "${REPO_ID}" \
  --overwrite


