# 商汤大装置：从部署到 DPO 实验

按本文顺序先跑通 SD1.5 纯 DPO，再比较多视图 B/C。以下命令在商汤已分配的单机容器中执行；先从 2 张 RTX 5090 开始，后续可使用 4/8 张或 H100。模型、数据和输出挂载到持久存储，确认新容器仍能访问。这里不提供或执行平台资源申请命令；本机 irip 集群仍遵守 Slurm 约定。

## ACP：8张5090、10000步纯 DPO

环境依赖、Pick-a-Pic 清单和128条固定监控 prompt 已按下文准备完成后，在 ACP 启动命令中填写：

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_8gpu.sh
```

启动器依次执行 `source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate`、`conda activate dpo`、进入本仓库，检查资源及8张已分配GPU，然后通过 Accelerate 启动10000步纯 DPO。每卡batch2、梯度累积16、全局batch256；BF16、LoRA rank/alpha16、LR1e-5、β5000、seed17，每200步完整保存，保留现有 TensorBoard 和固定 PickScore 监控。完整数据校验由训练入口执行。

固定文件位置：

| 内容 | 位置 |
|---|---|
| 控制台日志，包括 stderr | `/mnt/afs/L202500030/xietianyu/workspace/DPO/logs/sd15-dpo-8gpu-10000-v1.txt` |
| 冻结配置 | `configs/local-sd15-dpo-8gpu-10000-v1.json` |
| 实验、TensorBoard、完整 checkpoint | `/mnt/aoss/xietianyu/self_correct/runs/sd15-dpo-8gpu-10000-v1` |

第一次启动生成配置；重启同一命令复用配置，检测到实验目录后自动尝试 `--resume_from_checkpoint latest`，日志追加到同一 txt。缺少有效完整 checkpoint 时会停止报错，不覆盖旧实验。保持同一代码、依赖、挂载和8张GPU；启动器不自动拉取代码或安装依赖。

新实验改一个实验名，配置、日志和输出随之独立：

```bash
export SCD_RUN_NAME=sd15-dpo-8gpu-10000-v2
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_8gpu.sh
```

查看控制台日志：

```bash
tail -f /mnt/afs/L202500030/xietianyu/workspace/DPO/logs/sd15-dpo-8gpu-10000-v1.txt
```

原始 Parquet 转换不在 ACP 训练启动器中自动执行；缺少清单或监控 prompt 时先按下面的数据准备步骤完成。本启动器只使用本项目环境、资源和训练入口。

## 先更新代码

截图中的远程代码目录为下面的位置。旧实验需要续训时，保留其冻结 checkout，新版本用于新实验。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

## 环境和路径

截图中已有 `dpo` 环境，优先复用。新容器如果没有该环境，先执行 `conda create -n dpo python=3.10 -y`。不要在有活跃训练的环境里安装或更换依赖。

```bash
# 在项目独立环境中执行；不向 base 安装。
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate dpo
python --version
which python
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check

# 可选 W&B：
python -m pip install 'wandb>=0.17,<1'

# 所有资源地址从这一份配置读取。
cat configs/resources.json
# 远程私有路径可以写进 configs/resources.local.json，只填写需覆盖的键。
# 迁移后保留原有子目录结构：
# 模型根目录：/mnt/aoss/xietianyu/models
# 数据集根目录：/mnt/aoss/xietianyu/datasets
# Pick-a-Pic原始数据：/mnt/aoss/xietianyu/datasets/pickapicv2/data

# 可选：将同一份资源表导出为下面命令中的shell变量。
source paths.sh
```

`RUNS_ROOT` 必须是驱逐后仍存在的持久目录。SDXL、SD3、HPDv3 尚未下载，位置已经预留。SD3 当前仅提供路径和下载选项，训练入口仍为 SD1.5/SDXL。

本项目固定使用 `requirements.txt` 中的 PyTorch 2.7.1 + cu128、torchvision 0.22.1 + cu128 和 Accelerate 0.34.2。PyTorch 2.7 开始提供 Blackwell / CUDA 12.8 支持，参见 [官方说明](https://pytorch.org/blog/pytorch-2-7/) 和 [2.7.1 安装版本表](https://pytorch.org/get-started/previous-versions/)。容器的 `nvcc` 版本不能替代实际 PyTorch CUDA kernel 检查；不要把本机 CPU 测试环境复制到5090容器。

迁移后若旧终端执行过 `source paths.sh`，请在新终端激活环境后读取新配置；已导出的旧资源环境变量会优先于资源表。`configs/resources.local.json` 中若有旧路径，也需一起更新。迁移只改变资源前缀，SD1.5 仍在 `stable_diffusion/sd15`，PickScore 仍在 `preference/pickscore`，HPDv2/HPDv3 仍在各自的 `hpdv2` / `hpdv3` 子目录。

## 检查资源和已分配 GPU

```bash
nvidia-smi
python scripts/check_deployment.py --family sd15 --stage assets \
  --gpu --world-size 2
```

全部为 `[OK]` 再继续。脚本显示当前生效路径、包版本、剩余空间，检查 SD1.5、PickScore、processor、训练及官方 test_unique Parquet，读取下载记录；`--gpu` 在每张请求的 GPU 上实际执行很小的 BF16 卷积及反传。不加 `--gpu` 只检查 CPU 元数据。它不检验 NCCL 通信，后面的双卡训练负责验证 DDP。

如果已准备好 `pairs.jsonl` 和128条监控 prompt，这里改用 `--stage prepared --gpu --world-size 2`，跳过下一节原始数据处理。

如果路径仍指向旧位置，先处理环境变量或 `resources.local.json` 覆盖。不要重新下载已经存在的模型。未使用的 SDXL/SD3/HPDv3/HPSv3 不阻挡这条 SD1.5 + PickScore 部署路径。

## 已下载 Pick-a-Pic 的一次性处理

如果已有准备好的 `pairs.jsonl`，直接修改资源表中的 `PICKAPIC_SD15_PREPARED` / `PICKAPIC_SDXL_PREPARED`，跳过处理。原始 Parquet 不能直接作为 `--data_path`。

这一步在远程有足够 CPU、内存和存储的已分配资源上执行，不使用 GPU。脚本从原始目录或其上一级的完整 `.scd-download.json` 自动读取数据来源和版本，同时验证清单中的文件大小。第一次处理会提取原始图片，保留人类偏好标签并生成清单；后续实验复用。

```bash
source paths.sh
# 先检查原始数据和持久输出所在磁盘。
df -h "$PICKAPIC_RAW" "$(dirname "$PICKAPIC_SD15_PREPARED")"

python scripts/prepare_downloaded_data.py pickapic --family sd15 \
  --raw "$PICKAPIC_RAW" --model "$SD15_MODEL" \
  --images "$PICKAPIC_IMAGES" --out "$PICKAPIC_SD15_PREPARED"

python scripts/prepare_benchmarks.py \
  --train-pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" \
  --pickapic "$PICKAPIC_SD15_PREPARED/official-test-unique.jsonl" \
  --out "$BENCHMARKS_ROOT/sd15"

python scripts/check_deployment.py --family sd15 --stage prepared
```

如果资源检查报告没有可核实的下载版本，先查原下载记录。只有这种情况才在处理命令后加 `--revision`，传入实际的“数据仓库@下载 commit”；不要填占位符，也不要把现在的远端 main 当成当时的下载版本。

处理完成后应有 `pickapicv2/images/`、`pickapicv2/prepared-sd15/pairs.jsonl` 和 `benchmarks/sd15/pickapic-monitor.jsonl`。监控集默认128条；其余 Pick-a-Pic test prompt 另外保留。准备工具拒绝覆盖现有输出目录。`ready.json` 表示数据准备已完成；只有 `preparing.json` 时说明上次未完成，保留现场并为重新准备选择新输出目录。图像池可以复用。

## 两步纯 DPO 验证，同时检查续训

```bash
# 纯 DPO：K1，无 rollout；两卡、每卡batch2，两步预算。
python scripts/make_config.py --family sd15 --method dpo --world-size 2 \
  --smoke --train_batch_size 2 --out configs/local-sd15-dpo-smoke-v1.json

# 核对模型结构及数据文件；会读取/校验训练输入，不启动训练。
python train.py --config configs/local-sd15-dpo-smoke-v1.json \
  --out "$RUNS_ROOT/sd15-dpo-smoke-v1" --check-only

# 第1步后计划暂停，写出完整恢复点。
python train.py --config configs/local-sd15-dpo-smoke-v1.json \
  --out "$RUNS_ROOT/sd15-dpo-smoke-v1" \
  --execution standalone --run-label sd15-dpo-smoke-v1 --stop-after-step 1

# 使用同一个两步配置，从第1步恢复到第2步。
python train.py --config configs/local-sd15-dpo-smoke-v1.json \
  --out "$RUNS_ROOT/sd15-dpo-smoke-v1" \
  --execution standalone --run-label sd15-dpo-smoke-v1 \
  --resume_from_checkpoint latest
```

看到优化步从1继续到2，且 `checkpoints/latest.json` 指向第2步的完整状态，再开始正式实验。这个 smoke 保持512分辨率，关闭生成评估和诊断；只验证实际训练和恢复机制。重试新实验使用 v2 等新配置/输出名，恢复则始终使用原配置和原输出。

## 第一组正式实验：纯 DPO，2张5090

下面冻结一份2000步配置，使用 BF16、LoRA rank/alpha16、LR=1e-5、β=5000、seed17、全局batch256，每200步完整保存。每卡batch2是首次部署的起点，实际显存占用由 GPU smoke 确认；后续按单卡余量调整新实验配置。正式配置保留固定128条prompt的 PickScore 评估。

```bash
python scripts/make_config.py --family sd15 --method dpo --world-size 2 \
  --train_batch_size 2 --max_train_steps 2000 \
  --learning_rate 1e-5 --beta 5000 --seed 17 \
  --checkpointing_steps 200 --report_to tensorboard \
  --out configs/local-sd15-dpo-2gpu-v1.json

python train.py --config configs/local-sd15-dpo-2gpu-v1.json \
  --out "$RUNS_ROOT/sd15-dpo-2gpu-v1" --check-only

python train.py --config configs/local-sd15-dpo-2gpu-v1.json \
  --out "$RUNS_ROOT/sd15-dpo-2gpu-v1" \
  --execution standalone --run-label sd15-dpo-2gpu-v1
```

协调进程通过 Accelerate 启动 DDP；第0步先评估基座，之后在200步、每500步及最终步评估。第0步暂时没有训练 loss 是正常情况。训练进程退出、显存释放后再生成和评分，再从完整状态继续训练。

保持每卡batch2、全局batch256时，2/4/8卡对应梯度累积64/32/16，配置生成器自动计算。换卡数要生成新配置、新输出目录；已开始的实验恢复时保持原卡数。确认单机实际分配的卡数与 `--world-size` 一致。

被驱逐后重新取得相同资源，激活相同环境、进入同一代码版本、执行 `source paths.sh`，然后运行：

```bash
python train.py --config configs/local-sd15-dpo-2gpu-v1.json \
  --out "$RUNS_ROOT/sd15-dpo-2gpu-v1" \
  --execution standalone --run-label sd15-dpo-2gpu-v1 \
  --resume_from_checkpoint latest
```

配置文件保存在持久代码目录，输出保存在持久 AOSS。不要在驱逐后重新生成配置、改变路径/依赖/卡数或拉取新算法再续训。硬驱逐恢复最近完整保存点，通常最多重算上次保存后的不足200步；首次保存之前仍可能从头开始。

## 默认多视图 C：协调入口，自动评估

以下为完成纯 DPO 部署后的算法比较选项；不是继续上述纯 DPO run。先为选定的 K 做独立 smoke，再开新实验。

```bash
source paths.sh
RUN_NAME=sd15-C-k2-4gpu-v1

python train.py \
  --family sd15 \
  --world-size 4 \
  --views 2 \
  --launcher accelerate \
  --mixed_precision bf16 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_train_steps 2000 \
  --rank 16 --lora_alpha 16 \
  --checkpointing_steps 200 \
  --report_to tensorboard \
  --seed 17 \
  --output_dir "$RUNS_ROOT/$RUN_NAME" \
  --execution standalone --run-label "$RUN_NAME"
```

默认已经选择 `configs/multiview-C.json`。自动评估使用固定 128 条 Pick-a-Pic held-out prompt：第0步、第200步、每500步与最终步；训练进程退出后执行生成和评分，再从完整状态继续训练。

- 改为 B：增加 `--method multiview-B`，或 `--config configs/multiview-B.json`。
- 纯 DPO：增加 `--method dpo`，并删除上面显式的 `--views 2`；K=1，无 rollout。
- 改 K：`--views 3`、`--views 4` 等。
- 改卡数：使用 `--world-size 2/4/8`；若希望自动保持全局 batch，删除命令中显式的 `--gradient_accumulation_steps`。
- SDXL：使用 `--family sdxl`；删除显式 batch/accumulation，恢复 SDXL 默认每卡batch1、全局128对。先完成 SDXL 模型、清单、固定面板准备。
- 其他参数：例如 `--set training.weight_decay=0.01 --set training.warmup_steps=100`。
- 远程显卡已分配时可用 `CUDA_VISIBLE_DEVICES` 限制本次使用的 GPU，进程数须与可见资源匹配。

本集群 Slurm 分配中使用 `--execution slurm`，或省略该参数；非登录脚本需先显式加载 conda.sh 并激活项目环境。

## 直接 accelerate launch

这种方式直接启动训练 DDP，关闭协调评估；需要自动评估时使用上面的 `python train.py --launcher accelerate`。二者共享相同保存恢复实现。

```bash
source paths.sh
RUN_NAME=sd15-C-k3-2gpu-direct-v1

accelerate launch --multi_gpu --gpu_ids all --num_machines 1 --machine_rank 0 \
  --num_processes 2 --mixed_precision bf16 --dynamo_backend no \
  train.py \
  --family sd15 --views 3 --no-evaluation \
  --train_batch_size 4 --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 --max_train_steps 2000 \
  --checkpointing_steps 200 --report_to tensorboard \
  --output_dir "$RUNS_ROOT/$RUN_NAME" \
  --execution standalone --run-label "$RUN_NAME"
```

改 4/8 卡时改 `--num_processes`；只传卡数、未显式设置累积时自动保持配置中的全局 batch。启动器的 BF16 会传入训练配置，也可在 `train.py` 后显式指定 `--mixed_precision bf16`。

## 被驱逐后的续训

重新分配相同卡数后，**原命令所有参数保持一致**，追加：

```bash
--resume_from_checkpoint latest
```

例如上面的默认配置可简化为：

```bash
# 首次启动也使用相同的默认配置和覆盖参数，方可按此例恢复。
python train.py --family sd15 --world-size 4 \
  --output_dir "$RUNS_ROOT/sd15-default-v1" \
  --execution standalone --run-label sd15-default-v1 \
  --resume_from_checkpoint latest
```

也可以首次启动前冻结配置，后续一直用该文件：

```bash
python scripts/make_config.py --family sd15 --world-size 4 --views 3 \
  --learning_rate 1e-5 --out configs/local-sd15-C-k3-v1.json

# 首次启动
bash launchers/train.sh configs/local-sd15-C-k3-v1.json "$RUNS_ROOT/sd15-C-k3-v1"
# 重新分配相同资源后
bash launchers/train.sh configs/local-sd15-C-k3-v1.json "$RUNS_ROOT/sd15-C-k3-v1" \
  --resume_from_checkpoint latest
```

完整恢复点为 `checkpoints/latest.json`。`step-*` 默认仅有 LoRA，不能恢复优化器。不要改变卡数、代码、依赖或配置后强行续旧实验。硬驱逐恢复最近完整保存点，无法保证 SIGKILL 时的当前步已落盘。

## TensorBoard / W&B 与独立评估

```bash
source paths.sh
tensorboard --logdir "$RUNS_ROOT" --host 127.0.0.1 --port 6006

# 训练命令中使用以下参数即可同时记录二者。
# --report_to all --wandb-project self-correct-dpo --wandb-mode offline
# 在线 W&B：先自行 wandb login，再改 --wandb-mode online。

# 示例：在已分配GPU上评估永久保留的第1000步。
python evaluate.py --config configs/evaluate.json \
  --run "$RUNS_ROOT/sd15-C-k3-v1" --step 1000 \
  --out "$RUNS_ROOT/sd15-C-k3-v1-eval1000" \
  --execution standalone --run-label sd15-C-k3-v1-eval1000
```

独立评估默认使用 HPDv2 面板和多个评分器，先准备对应面板和权重。只比较 PickScore 时用评估入口提供的 `--metrics pickscore`；SDXL 换 `configs/evaluate-sdxl.json`。W&B offline 每次启动为独立本地会话，在线同一实验通过稳定 run ID 恢复；横轴为 `optimizer_step`。

## 本机开发到远程 main

```bash
# 本机：确认当前为开发分支，查看待提交内容。
git branch --show-current
git diff --stat

# 只运行本次改动有关的轻量检查。
python -m unittest discover -s tests -p test_joint.py -v
python -m unittest discover -s tests -p test_stage_pause.py -v

# 审查完成后只提交属于本次修改的文件，推开发分支并通过 PR 合入 main。
# 不要将尚未确认的算法改动、数据、日志或模型一起提交。

# 远程：先结束旧实验或保留旧代码快照，再更新main并记录版本。
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

本机使用 `dev/self-correct-workflow` 开发分支，远程使用已合并的 main。详细规矩见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。

## 后续 SDXL / SD3 资源（首次 SD1.5 部署跳过）

按需下载，先检查空间及已有缓存：

```bash
python scripts/download_assets.py list
python scripts/download_assets.py download sdxl
python scripts/download_assets.py download sd3
python scripts/download_assets.py download hpdv3
python scripts/download_assets.py extract-hpdv3
```

SDXL 下载后用 `--family sdxl`、`--model "$SDXL_MODEL"`、`--out "$PICKAPIC_SDXL_PREPARED"` 另行准备清单；复用同一图片池，再生成 `$BENCHMARKS_ROOT/sdxl` 面板。随后重复资源检查、同 K 小跑、配置冻结和正式训练。不要拿 SD1.5 的 batch/β 直接覆盖 SDXL 默认值。

HPDv3 的 SD3 数据可用 `prepare_downloaded_data.py hpdv3 --family unfiltered` 导出。这只准备数据；当前 SD3 训练入口尚未实现。
