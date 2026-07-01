#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 转换后的 LeRobotDataset 输出目录。训练时 lerobot-train --dataset.root 指向这里。
OUT_ROOT="${SCRIPT_DIR}/output/dataset1"

# 本地训练用的数据集 repo_id，训练时 --dataset.repo_id 要保持一致。
REPO_ID="local/g1_wuji_112339"

python "${SCRIPT_DIR}/convert_isaacteleop_to_lerobot.py" \
  --raw-dir /home/lightwheel/workspace/smolvla/isaacteleop/examples/g1_wuji_teleop/data/dataset/20260701_112339 \
  --out-root "${OUT_ROOT}" \
  --repo-id "${REPO_ID}" \
  --overwrite
