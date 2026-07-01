cd smolvla

git lfs install
mkdir -p checkpoints

git clone https://hf-mirror.com/lerobot/smolvla_base checkpoints/smolvla_base
git clone https://hf-mirror.com/HuggingFaceTB/SmolVLM2-500M-Video-Instruct checkpoints/SmolVLM2-500M-Video-Instruct

# 如果不用镜像，也可以从 Hugging Face 官方源拉：

git clone https://huggingface.co/lerobot/smolvla_base checkpoints/smolvla_base
git clone https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct checkpoints/SmolVLM2-500M-Video-Instruct

# 如果 clone 完发现模型文件是 LFS pointer，执行：

git -C checkpoints/smolvla_base lfs pull
git -C checkpoints/SmolVLM2-500M-Video-Instruct lfs pull