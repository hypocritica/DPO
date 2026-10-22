# Self Correct DPO 命令

以下相对路径以本模块仓库根目录为工作目录。远程训练前先取得 2 / 4 / 8 张 5090 或 H100；本集群训练必须在 Slurm 分配内。本文件只提供命令，不提交作业。

## 环境和路径

```bash
# 在项目独立环境中执行；不向 base 安装。
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# 可选 W&B：
python -m pip install 'wandb>=0.17,<1'

# 所有资源地址从这一份配置读取。
cat configs/resources.json
# 远程私有路径可以写进 configs/resources.local.json，只填写需覆盖的键。
# 迁移后保留原有子目录结构：
# 模型根目录：/mnt/aoss/xietianyu/datasets/models
# 数据集根目录：/mnt/aoss/xietianyu/datasets/datasets
# Pick-a-Pic原始数据：/mnt/aoss/xietianyu/datasets/datasets/pickapicv2/data

# 可选：将同一份资源表导出为下面命令中的shell变量。
source paths.sh
```

`RUNS_ROOT` 必须是驱逐后仍存在的持久目录。SDXL、SD3、HPDv3 尚未下载，位置已经预留。SD3 当前仅提供路径和下载选项，训练入口仍为 SD1.5/SDXL。

迁移后若旧终端执行过 `source paths.sh`，请在新终端激活环境后读取新配置；已导出的旧资源环境变量会优先于资源表。`configs/resources.local.json` 中若有旧路径，也需一起更新。迁移只改变资源前缀，SD1.5 仍在 `stable_diffusion/sd15`，PickScore 仍在 `preference/pickscore`，HPDv2/HPDv3 仍在各自的 `hpdv2` / `hpdv3` 子目录。

## 已下载 Pick-a-Pic 的一次性处理

如果已有准备好的 `pairs.jsonl`，直接修改资源表中的 `PICKAPIC_SD15_PREPARED` / `PICKAPIC_SDXL_PREPARED`，跳过处理。原始 Parquet 不能直接作为 `--data_path`。

在远程有足够 CPU、内存和存储的已分配资源上执行。`--revision` 填真实下载来源与版本；若数据根目录存在完整 `.scd-download.json` 则可省略。

```bash
source paths.sh
# 先检查原始数据和持久输出所在磁盘。
df -h "$PICKAPIC_RAW" "$(dirname "$PICKAPIC_SD15_PREPARED")"

python scripts/prepare_downloaded_data.py pickapic --family sd15 \
  --raw "$PICKAPIC_RAW" --model "$SD15_MODEL" \
  --images "$PICKAPIC_IMAGES" --out "$PICKAPIC_SD15_PREPARED" \
  --revision 'REPLACE_WITH_ACTUAL_DATASET_SOURCE_AND_REVISION'

python scripts/prepare_downloaded_data.py benchmarks --family sd15 \
  --train-pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" \
  --pickapic-test "$PICKAPIC_SD15_PREPARED/official-test-unique.jsonl" \
  --out "$BENCHMARKS_ROOT/sd15"
```

SDXL 下载完成后，以 `--family sdxl`、`--model "$SDXL_MODEL"`、`--out "$PICKAPIC_SDXL_PREPARED"` 分别生成对应清单，再生成 `$BENCHMARKS_ROOT/sdxl` 面板；图像池可复用。目录已存在时工具会拒绝覆盖，重新整理数据使用新版本目录。

按需下载缺少的资源；先确认磁盘容量及已有缓存，不重复下载已有模型：

```bash
python scripts/download_assets.py list
python scripts/download_assets.py download sdxl
python scripts/download_assets.py download sd3
python scripts/download_assets.py download hpdv3
python scripts/download_assets.py extract-hpdv3
```

这些命令会写入 `configs/resources.json` 指定的位置。HPDv3 解包及数据处理也需在远程已分配资源上执行。后续 SD3 数据可以用 `prepare_downloaded_data.py hpdv3 --family unfiltered` 导出；这只是数据准备，不会启动 SD3 训练。

## 查看参数与快速验证

```bash
# 仅解析配置，不加载模型或数据，不使用GPU。
python train.py --family sd15 --world-size 4 --views 3 \
  --learning_rate 1e-5 --output_dir /tmp/scd-preview --print-config

# 生成独立的两步配置；默认 K2-C，保持模型原分辨率。
python scripts/make_config.py --family sd15 --world-size 2 --smoke \
  --out configs/local-smoke.json

# 先验证完整模型、偏好清单与配置，不启动GPU训练。
python train.py --config configs/local-smoke.json --out "$RUNS_ROOT/smoke-v1" --check-only

# 取得GPU后进行两步真实训练。
python train.py --config configs/local-smoke.json \
  --out "$RUNS_ROOT/smoke-v1" --execution standalone --run-label smoke-v1
```

不要反复覆盖 `local-smoke.json` 或旧运行目录；重试时使用新版本名。增大 K 前先用对应 K 做短跑，确认单卡显存。上述 smoke 配置关闭生成评估和诊断，只保留训练与保存机制。

## 默认多视图 C：协调入口，自动评估

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

此轮已在本机建立 `dev/self-correct-workflow`；尚未提交或推送 GitHub。详细规矩见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。
