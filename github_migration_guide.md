# 将含子仓库的项目整理成自己的 GitHub 私有仓库

本文记录这次把 `smolvla` 工作目录整理成 `Capybara-TheHang/smolvla`
GitHub 私有仓库的流程。适用于这种情况：

- 当前目录里包含多个别人或公司来源的 Git 仓库，例如 `isaacteleop/`、
  `lerobot/`。
- 你不想继续保留它们各自的远程源和 Git 历史，只想把当前文件内容作为一个
  新仓库快照放到自己的 GitHub。
- 目录里还有模型、训练输出、采集数据等大文件，需要决定哪些上传，哪些忽略。

## 核心原则

不要直接在原始目录里 `git init && git add .`。

如果目录里有嵌套 `.git`，父仓库直接 `git add` 时，Git 可能把它们当成
submodule/gitlink，而不是把完整文件内容纳入新仓库。正确做法是：

1. 在旁边创建一个干净副本。
2. 复制文件时排除所有内部 `.git`。
3. 在干净副本里初始化新的 Git 仓库。
4. 按需要配置 `.gitignore` 和 Git LFS。
5. 推送到自己的 GitHub 私有仓库。

## 1. 检查原始目录

假设原始目录是：

```bash
cd /home/lightwheel/workspace/smolvla
```

查看里面有哪些嵌套 Git 仓库：

```bash
find . -maxdepth 3 -name .git -type d -print
```

这次看到的结果包括：

```text
./lerobot/.git
./isaacteleop/.git
./checkpoints/smolvla_base/.git
./checkpoints/SmolVLM2-500M-Video-Instruct/.git
```

查看子仓库各自远程源：

```bash
git -C isaacteleop remote -v
git -C lerobot remote -v
```

这一步只是确认来源，不需要改它们。

## 2. 检查大文件

GitHub 普通 Git 不允许单文件超过 100 MiB。模型、ONNX、训练输出和采集数据
通常都需要用 Git LFS，或者直接忽略不上传。

检查大文件：

```bash
find . -type f -size +50M ! -path '*/.git/*' -printf '%s %p\n' | sort -nr | head -50
du -sh .
```

这次主要大文件是：

- `checkpoints/SmolVLM2-500M-Video-Instruct/`
- `checkpoints/smolvla_base/`
- `outputs/train/...`
- `isaacteleop/examples/g1_wuji_teleop/data/...`

最终决定：

- 不上传 `checkpoints/`
- 不上传 `outputs/`
- 不上传 `isaacteleop/examples/g1_wuji_teleop/data/`
- 其余源码、配置、少量二进制资源保留

## 3. 创建干净副本

在原始目录旁边创建一个新目录：

```bash
cd /home/lightwheel/workspace/smolvla
mkdir -p /home/lightwheel/workspace/smolvla_github_private
rsync -a --exclude='.git/' ./ /home/lightwheel/workspace/smolvla_github_private/
```

进入新副本：

```bash
cd /home/lightwheel/workspace/smolvla_github_private
```

确认副本里没有嵌套 `.git`：

```bash
find . -path './.git' -prune -o -name .git -type d -print
```

正常应没有输出。

## 4. 初始化新 Git 仓库

```bash
git init -b main
git lfs install --local
```

## 5. 配置 Git LFS

即使决定忽略大模型，也建议配置常见大二进制类型走 LFS，防止后续误提交。

创建 `.gitattributes`：

```bash
cat > .gitattributes <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.so filter=lfs diff=lfs merge=lfs -text
*.so.* filter=lfs diff=lfs merge=lfs -text
*.usd filter=lfs diff=lfs merge=lfs -text
*.STEP filter=lfs diff=lfs merge=lfs -text
*.step filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.whl filter=lfs diff=lfs merge=lfs -text
EOF
```

## 6. 配置忽略规则

创建 `.gitignore`，排除不上传的大目录：

```bash
cat > .gitignore <<'EOF'
checkpoints/
outputs/
isaacteleop/examples/g1_wuji_teleop/data/
EOF
```

## 7. 添加文件并提交

因为子目录自己的 `.gitignore` 可能会忽略一些你想保留的 build/install 文件，
这次使用了 `-f` 强制加入当前快照内容：

```bash
git add -A -f .
git commit -m "Initial smolvla snapshot"
```

如果你已经误把 `checkpoints/`、`outputs/` 或数据目录加进提交，可以在推送前
改写本地提交：

```bash
git rm -r --cached checkpoints outputs isaacteleop/examples/g1_wuji_teleop/data
git add .gitignore .gitattributes
git commit --amend -m "Initial smolvla snapshot"
```

注意：如果这个提交已经推送到远端，需要谨慎使用强推。

## 8. 检查提交内容

确认这些目录没有进入 Git：

```bash
git ls-tree -r --name-only HEAD | grep -E '^(checkpoints|outputs|isaacteleop/examples/g1_wuji_teleop/data)/'
```

正常应该没有任何输出。

检查 LFS 文件数量和大小：

```bash
git lfs ls-files | wc -l
git lfs ls-files --name-only | while IFS= read -r f; do
  [ -f "$f" ] && stat -c %s "$f"
done | awk '{s+=$1} END {printf "%.2f GiB\n", s/1024/1024/1024}'
```

检查是否还有超过 100 MB 但没有走 LFS 的文件：

```bash
for f in $(git ls-files); do
  if [ -f "$f" ]; then
    size=$(stat -c %s "$f")
    if [ "$size" -gt 100000000 ]; then
      attr=$(git check-attr filter -- "$f" | awk -F': ' '{print $3}')
      if [ "$attr" != lfs ]; then
        printf '%s %s\n' "$size" "$f"
      fi
    fi
  fi
done | sort -nr
```

正常应没有输出。

## 9. 创建 GitHub 私有仓库

可以在 GitHub 网页上创建：

- owner：`Capybara-TheHang`
- repo：`smolvla`
- visibility：Private

也可以用 GitHub CLI：

```bash
gh auth login
gh repo create Capybara-TheHang/smolvla --private
```

如果 GitHub 提示仓库已存在，就不用重复创建。

## 10. 添加远程源并推送

HTTPS 方式：

```bash
git remote add origin https://github.com/Capybara-TheHang/smolvla.git
git push -u origin main
```

如果远端已经有 GitHub 自动生成的 `README.md`，并且你确定要用本地快照覆盖它：

```bash
git push -u origin main --force-with-lease
```

这次实际推送到了分支 `g1_wuji_grasp`：

```bash
git push --progress -u origin g1_wuji_grasp --force
```

之后 clone 时需要指定分支：

```bash
git clone -b g1_wuji_grasp https://github.com/Capybara-TheHang/smolvla.git
```

如果想让普通 `git clone` 默认拿到这份内容，可以在 GitHub 网页把默认分支改成
`g1_wuji_grasp`，或者把本地分支推到 `main`：

```bash
git push origin g1_wuji_grasp:main --force-with-lease
```

## 11. 网络和认证问题

GitHub 私有仓库不能用账号密码拉取。HTTPS 方式需要 Personal Access Token，
或者先用 GitHub CLI 登录：

```bash
gh auth login
gh auth setup-git
```

更推荐 SSH：

```bash
git remote set-url origin git@github.com:Capybara-TheHang/smolvla.git
ssh -T git@github.com
git push
```

如果 22 端口不通，可以用 GitHub SSH 443：

```bash
git remote set-url origin ssh://git@ssh.github.com:443/Capybara-TheHang/smolvla.git
ssh -T -p 443 git@ssh.github.com
git push
```

如果 HTTPS 遇到 `gnutls_handshake()` 或连接超时，通常是代理/TLS 问题。可以尝试：

```bash
unset all_proxy ALL_PROXY
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
git config --global http.version HTTP/1.1
git push --progress
```

先用 curl 测试代理是否可用：

```bash
curl -I -x http://127.0.0.1:7897 https://github.com
```

## 12. clone 后恢复模型

因为 `checkpoints/` 没有上传，clone 后需要手动下载：

```bash
cd smolvla
git lfs install
mkdir -p checkpoints

git clone https://hf-mirror.com/lerobot/smolvla_base checkpoints/smolvla_base
git clone https://hf-mirror.com/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  checkpoints/SmolVLM2-500M-Video-Instruct

git -C checkpoints/smolvla_base lfs pull
git -C checkpoints/SmolVLM2-500M-Video-Instruct lfs pull
```

如果用官方 Hugging Face：

```bash
git clone https://huggingface.co/lerobot/smolvla_base checkpoints/smolvla_base
git clone https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  checkpoints/SmolVLM2-500M-Video-Instruct
```

注意下载位置必须是仓库根目录下的 `checkpoints/`。如果你在 `lerobot/` 目录里执行，
会下载到 `lerobot/checkpoints/`，路径就错了。

## 13. 常用最终检查

查看远程源：

```bash
git remote -v
```

查看当前分支和状态：

```bash
git status --short --branch
```

查看最新提交：

```bash
git log --oneline --decorate -1
```

查看 GitHub 仓库：

```bash
gh repo view Capybara-TheHang/smolvla --web
```

## 14. 本次结果

本次整理后的仓库特点：

- `isaacteleop/` 和 `lerobot/` 已作为普通目录纳入自己的 GitHub 私有仓库。
- 它们原来的 `.git` 和远程源没有进入新仓库。
- `checkpoints/`、`outputs/`、采集数据目录没有上传。
- 剩余二进制文件通过 Git LFS 管理。
- 远端仓库是 `Capybara-TheHang/smolvla`。
- 实际推送分支是 `g1_wuji_grasp`。



• 标记单个已跟踪文件：

git update-index --assume-unchanged 路径/文件名

标记一个目录下当前所有已修改的已跟踪文件：

git ls-files -m 路径/目录 | xargs -r git update-index --assume-unchanged

例如你的 lerobot/src：

git ls-files -m lerobot/src | xargs -r git update-index --assume-unchanged

取消标记：

git update-index --no-assume-unchanged 路径/文件名