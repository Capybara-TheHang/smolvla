# smolvla 环境配置说明

这个仓库里包含两个源码目录，它们现在都是普通目录，不再是独立
Git 仓库：

- `isaacteleop/`：G1-Wuji AVP + MANUS 遥操作运行环境。
- `lerobot/`：LeRobot，以及本项目本地的 G1-Wuji SmolVLA 数据转换和训练脚本。

这个私有仓库快照创建时，原来嵌套在 `isaacteleop/`、`lerobot/`、
`checkpoints/` 里的 `.git` 目录已经被移除。后续使用时，把它们都当作
当前仓库的一部分即可。

下面这些大文件目录没有上传到 GitHub：

- `checkpoints/`
- `outputs/`
- `isaacteleop/examples/g1_wuji_teleop/data/`

clone 仓库后，按下面步骤恢复两个 conda 环境。

## 0. 系统依赖

以下命令假设系统是 Ubuntu，并且有 NVIDIA GPU 和可用的 conda。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git git-lfs ccache patchelf pkg-config swig \
  libx11-dev clang-format-14 \
  android-tools-adb coturn

git lfs install
```

下载私有仓库并进入目录：

```bash
git clone -b g1_wuji_grasp https://github.com/Capybara-TheHang/smolvla.git
cd smolvla
git lfs pull

export SMOLVLA_ROOT="$PWD"
```

如果你已经把 GitHub 默认分支改成了 `g1_wuji_grasp`，`git clone` 时可以不加
`-b g1_wuji_grasp`。

## 1. 恢复被忽略的大模型文件

GitHub 私有仓库里没有保存下载下来的模型 checkpoint。clone 后需要重新下载到
`checkpoints/`：

```bash
mkdir -p checkpoints

# 原始环境使用的是 hf-mirror 镜像源。
git clone https://hf-mirror.com/lerobot/smolvla_base checkpoints/smolvla_base
git clone https://hf-mirror.com/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  checkpoints/SmolVLM2-500M-Video-Instruct

git -C checkpoints/smolvla_base lfs pull
git -C checkpoints/SmolVLM2-500M-Video-Instruct lfs pull
```

如果镜像不可用，可以改用 Hugging Face 官方源：

```bash
git clone https://huggingface.co/lerobot/smolvla_base checkpoints/smolvla_base
git clone https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  checkpoints/SmolVLM2-500M-Video-Instruct
```

下面两个目录是本地采集或训练生成的内容，没有公开下载源：

- `isaacteleop/examples/g1_wuji_teleop/data/`
- `outputs/`

如果还需要这些数据，从旧机器复制回来；否则重新采集或重新训练生成。

## 2. 配置 `lerobot-smolvla` 环境

LeRobot 要求 Python 3.12 或更高版本。当前已验证的本地环境使用：

- Python 3.12.13
- LeRobot 0.5.2

创建 conda 环境：

```bash
conda create -n lerobot-smolvla python=3.12 -y
conda activate lerobot-smolvla
python -m pip install -U pip setuptools wheel
```

从本仓库的 `lerobot/` 源码目录安装 LeRobot，并安装 SmolVLA、数据集和训练相关
依赖：

```bash
cd $SMOLVLA_ROOT/lerobot

pip install -e ".[smolvla,dataset,training]"
```

Linux 下，`lerobot/pyproject.toml` 里配置了 CUDA 12.8 的 PyTorch wheel 源。
如果 `pip` 最终装成了 CPU 版 torch，可以手动重装 CUDA 版：

```bash
pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

这个项目里常用的可选依赖：

```bash
pip install onnx onnxruntime
```

检查环境是否正常：

```bash
python - <<'PY'
import torch
import lerobot
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("lerobot", getattr(lerobot, "__version__", "editable"))
PY

lerobot-train --help | head
```

运行 SmolVLA 合成数据 smoke test：

```bash
cd $SMOLVLA_ROOT

python lerobot/examples/g1_wuji/smoke_smolvla_synthetic.py \
  --overwrite \
  --overwrite-output \
  --policy-path $SMOLVLA_ROOT/checkpoints/smolvla_base \
  --vlm-model-name $SMOLVLA_ROOT/checkpoints/SmolVLM2-500M-Video-Instruct \
  --rename-front-to-camera1 \
  --train-steps 1 \
  --batch-size 1
```

如果已经复制回 Isaac Teleop 采集的原始数据，可以把它转换成 LeRobotDataset：

```bash
cd $SMOLVLA_ROOT

# 如果 session 目录不同，先编辑脚本里的 raw-dir、repo-id 和输出路径。
bash lerobot/examples/g1_wuji/convert.sh
```

在转换后的数据集上训练：

```bash
cd $SMOLVLA_ROOT

# 根据显存和数据路径，先编辑脚本里的路径、batch size 和训练步数。
bash lerobot/examples/g1_wuji/train.sh
```

## 3. 配置 `isaacteleop` 环境

G1-Wuji 遥操作示例需要在 Isaac Sim / Isaac Lab Python 环境里运行。当前已验证的
本地环境大致是：

- Python 3.12.13
- `isaacsim==6.0.1.0`
- Isaac Lab 源码包以 editable 方式安装在同一个环境里
- `isaacteleop==1.3+local`
- `torch==2.11.0`
- `numpy==2.3.1`

创建 conda 环境：

```bash
conda create -n isaacteleop python=3.12 -y
conda activate isaacteleop
python -m pip install -U pip setuptools wheel
```

从 NVIDIA Python index 安装 Isaac Sim：

```bash
pip install isaacsim==6.0.1.0 --extra-index-url https://pypi.nvidia.com
```

单独安装 Isaac Lab。当前已验证机器使用的是 IsaacLab 源码 checkout：

- IsaacLab `VERSION=3.0.0`
- commit `28a37ce`

```bash
cd /home/lightwheel/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 28a37ce

# 安装当前 isaacteleop 环境里用到的 Isaac Lab source packages。
for pkg in \
  source/isaaclab \
  source/isaaclab_assets \
  source/isaaclab_contrib \
  source/isaaclab_experimental \
  source/isaaclab_mimic \
  source/isaaclab_newton \
  source/isaaclab_ov \
  source/isaaclab_ovphysx \
  source/isaaclab_physx \
  source/isaaclab_ppisp \
  source/isaaclab_rl \
  source/isaaclab_tasks \
  source/isaaclab_tasks_experimental \
  source/isaaclab_teleop
do
  pip install -e "$pkg"
done
```

如果这个 commit 和你的 Isaac Sim 安装不兼容，按 Isaac Lab 官方文档安装与
Isaac Sim 6.0 匹配的版本，然后继续下面的步骤。

安装 Isaac Teleop。快速运行可以直接装 NVIDIA 发布的包：

```bash
pip install "isaacteleop[cloudxr,retargeters,ui]~=1.0" \
  --extra-index-url https://pypi.nvidia.com
```

如果要尽量贴近当前仓库快照，从本地 `isaacteleop/` 源码构建并安装 wheel：

```bash
cd $SMOLVLA_ROOT/isaacteleop

cmake -B build \
  -DISAAC_TELEOP_PYTHON_VERSION=3.12 \
  -DENABLE_CLANG_FORMAT_CHECK=OFF
cmake --build build --parallel
cmake --install build

pip install "isaacteleop[retargeters,cloudxr,ui]" \
  --find-links=./install/wheels \
  --force-reinstall
```

这个仓库快照里保留了 MANUS 运行时插件相关文件：

```text
isaacteleop/install/lib/libIsaacTeleopPluginsManus.so
isaacteleop/install/lib/libManusSDK_Integrated.so
isaacteleop/install/plugins/manus/plugin.yaml
```

如果干净 clone 或重新 build 后 MANUS 插件缺失，需要从旧机器复制回来，或者按你
自己的 MANUS SDK 授权和构建流程重新生成。

设置本地示例运行时路径：

```bash
cd $SMOLVLA_ROOT

export ISAAC_TELEOP_ROOT="$PWD/isaacteleop"
export LD_LIBRARY_PATH="$PWD/isaacteleop/install/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/isaacteleop/examples/g1_wuji_teleop/python:${PYTHONPATH:-}"
```

当前仓库使用的 CloudXR profile：

```bash
cat > custom.env <<'EOF'
NV_DEVICE_PROFILE=auto-webrtc
EOF
cp custom.env isaacteleop/custom.env
```

检查环境是否正常：

```bash
python - <<'PY'
import isaacteleop
import isaacsim
print("isaacteleop", getattr(isaacteleop, "__version__", "installed"))
print("isaacsim import ok")
PY

python -m isaacteleop.cloudxr --help | head
```

第一个终端启动 CloudXR，并保持它运行：

```bash
conda activate isaacteleop
cd $SMOLVLA_ROOT/isaacteleop
python -m isaacteleop.cloudxr --cloudxr-env-config=../custom.env
```

第二个终端启动 G1-Wuji teleop session：

```bash
conda activate isaacteleop
cd $SMOLVLA_ROOT/isaacteleop

source ~/.cloudxr/run/cloudxr.env 2>/dev/null || true
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/examples/g1_wuji_teleop/python:${PYTHONPATH:-}"

python examples/g1_wuji_teleop/scripts/g1_wuji_teleop_main.py \
  --config examples/g1_wuji_teleop/config/avp_manus.yml \
  --input-profile avp-manus
```

参数说明：

- `--input-profile avp-manus`：使用 AVP hand-wrist EE pose，加 MANUS 手指重定向。
- `--input-profile vr-manus`：使用 VR controller EE pose，加 MANUS 手指重定向。
- `--input-profile config`：完全使用 YAML 里的配置。

## 4. 可选：导出 requirements 快照

推荐的依赖来源是：

- `lerobot/pyproject.toml`：LeRobot 依赖。
- Isaac Teleop 的 CMake build 和 `isaacteleop[cloudxr,retargeters,ui]` extras：
  Isaac Teleop 依赖。

如果你想从一台已经跑通的机器导出精确包版本，执行：

```bash
conda activate lerobot-smolvla
python -m pip freeze > requirements-lerobot-smolvla.lock.txt

conda activate isaacteleop
python -m pip freeze > requirements-isaacteleop.lock.txt
```

不要在不同 CUDA / Isaac Sim 版本的机器上盲目安装这些 lock 文件。它们更适合用作
排查问题时的版本参考。
