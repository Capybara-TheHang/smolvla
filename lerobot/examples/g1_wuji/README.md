# G1 Wuji SmolVLA Smoke Test

This example generates a tiny synthetic LeRobot dataset for a one-arm + one-dexterous-hand setup and optionally runs a short SmolVLA fine-tuning smoke test.

```bash
cd /home/lightwheel/workspace/smolvla
conda activate lerobot-smolvla
```

Dataset-only check:

```bash
python lerobot/examples/g1_wuji/smoke_smolvla_synthetic.py --skip-train --overwrite
```

End-to-end SmolVLA training check with local checkpoints:

```bash
python lerobot/examples/g1_wuji/smoke_smolvla_synthetic.py \
  --overwrite \
  --overwrite-output \
  --policy-path /home/lightwheel/workspace/smolvla/checkpoints/smolvla_base \
  --vlm-model-name /home/lightwheel/workspace/smolvla/checkpoints/SmolVLM2-500M-Video-Instruct \
  --rename-front-to-camera1 \
  --train-steps 1 \
  --batch-size 1
```

Default outputs:

```text
/home/lightwheel/workspace/smolvla/outputs/synthetic_lerobot_one_arm_hand
/home/lightwheel/workspace/smolvla/outputs/train/smolvla_synthetic_smoke
```

## Training with TensorBoard

`train.sh` starts a background TensorBoard server automatically, mirrors training metrics into
TensorBoard event files, and keeps every run under `lerobot/examples/g1_wuji/tensorboard_runs/`.

Install TensorBoard once in the training environment if it is not already available:

```bash
conda activate lerobot-smolvla
pip install tensorboard tensorboardX
```

```bash
cd /home/lightwheel/workspace/smolvla
conda activate lerobot-smolvla
bash lerobot/examples/g1_wuji/train.sh
```

The script prints a URL like:

```text
TensorBoard URL: http://<training-machine-ip>:6006
```

Useful overrides:

```bash
TENSORBOARD_PORT=6007 LOG_FREQ=5 STEPS=3000 bash lerobot/examples/g1_wuji/train.sh
ENABLE_TENSORBOARD=false bash lerobot/examples/g1_wuji/train.sh
```

To reopen historical TensorBoard runs without starting training:

```bash
bash lerobot/examples/g1_wuji/tensorboard.sh
```

To stop the managed background TensorBoard process:

```bash
bash lerobot/examples/g1_wuji/tensorboard.sh stop
```
