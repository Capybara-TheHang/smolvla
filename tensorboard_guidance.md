主要改动：

- /home/lightwheel/workspace/smolvla_github_private/lerobot/examples/g1_wuji/train.sh 现在默认启动后台 TensorBoard，再运行训练。
- 新增 /home/lightwheel/workspace/smolvla_github_private/lerobot/examples/g1_wuji/train_with_tensorboard.py，负责启动/复用 TensorBoard、解析训练日志、写入事件文件。
- 新增 /home/lightwheel/workspace/smolvla_github_private/lerobot/examples/g1_wuji/tensorboard.sh，可单独打开历史 TensorBoard。
- /home/lightwheel/workspace/smolvla_github_private/lerobot/src/lerobot/scripts/lerobot_train.py 现在会把 SmolVLA 返回的额外 loss 标量也打印出来，所以 TensorBoard 能记录 losses_after_forward、
losses_after_in_ep_bound、losses_after_rm_padding 等。
- TensorBoard 历史保存在：
lerobot/examples/g1_wuji/tensorboard_runs/

使用：

cd /home/lightwheel/workspace/smolvla_github_private
conda activate lerobot-smolvla
pip install tensorboard tensorboardX
bash lerobot/examples/g1_wuji/train.sh

脚本会打印类似：

TensorBoard URL: http://<训练机器IP>:6006

只查看历史，不启动训练：

bash lerobot/examples/g1_wuji/tensorboard.sh

停止后台 TensorBoard：

bash lerobot/examples/g1_wuji/tensorboard.sh stop