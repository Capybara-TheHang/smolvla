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
