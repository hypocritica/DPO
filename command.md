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
  --images "$PICKAPIC_IMAGES" --out "$PICKAPIC_SD15_PREPARED" \
  --workers 8 --batch-size 256

python scripts/prepare_benchmarks.py \
  --train-pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" \
  --pickapic "$PICKAPIC_SD15_PREPARED/official-test-unique.jsonl" \
  --out "$BENCHMARKS_ROOT/sd15"

python scripts/check_deployment.py --family sd15 --stage prepared
```

如果资源检查报告没有可核实的下载版本，先查原下载记录。只有这种情况才在处理命令后加 `--revision`，传入实际的“数据仓库@下载 commit”；不要填占位符，也不要把现在的远端 main 当成当时的下载版本。

处理完成后应有 `pickapicv2/images/`、`pickapicv2/prepared-sd15/pairs.jsonl` 和 `benchmarks/sd15/pickapic-monitor.jsonl`。监控集默认128条；其余 Pick-a-Pic test prompt 另外保留。`ready.json` 表示数据准备已完成；只有 `preparing.json` 时不启动训练。

### Pick-a-Pic 并行、恢复与旧任务切换

默认使用8个图片 I/O 线程，每256条原始记录提交一个原子批次。图片保留原始字节、文件名和目录；已有图片按 SHA256 校验后复用，不重写。`--workers 16` 可在存储允许时提高并发；若延迟变大则降回4或8。batch 控制内存和重做范围，不是训练 batch。

新版本中断后，在相同输出目录恢复（线程数和 batch 大小可以调整）：

```bash
python scripts/prepare_downloaded_data.py pickapic --family sd15 \
  --raw "$PICKAPIC_RAW" --model "$SD15_MODEL" \
  --images "$PICKAPIC_IMAGES" --out "$PICKAPIC_SD15_PREPARED" \
  --workers 8 --batch-size 256 --resume
```

`.preparation/` 同时保存已处理行、排除原因、图片哈希和下一个 Parquet 游标；恢复跳过已提交文件和 row group，不重新验证这些图片。若游标在一个压缩 row group 中间，Arrow 仍需解码该组的前缀，但不会再过滤、导出或校验前缀样本。新写入图片先关闭临时文件再重命名；中断时未提交的批次会重做，已完成图片复用。底层挂载须支持原子重命名及文件锁，持久性依赖存储系统。保持原始数据、图片池、tokenizer、外部测试集及代码不变；输入/代码身份变化会拒绝恢复。最终划分和清单也支持中断后重建，`ready.json` 最后写入；最终阶段仍需要足够内存。

**正在运行旧串行程序时，先停止它，再更新代码和使用 `--adopt-legacy`。旧版本不持锁，不能与新版本同时写同一个图片池。** 在容器的 Bash 终端执行以下块，输入旧 Python 进程的 PID；校验命令后发送 TERM，并等待它退出：

```bash
(
  set -e
  pgrep -af '[p]repare_downloaded_data.py.*pickapic'
  read -r -p "输入旧预处理 Python PID: " PREP_PID
  [[ "$PREP_PID" =~ ^[0-9]+$ && "$PREP_PID" -gt 1 ]]
  [[ "$(ps -p "$PREP_PID" -o comm=)" == *python* ]]
  [[ "$(ps -p "$PREP_PID" -o args=)" == *prepare_downloaded_data.py*pickapic* ]]
  kill -TERM "$PREP_PID"
  while :; do
    PREP_STATE="$(ps -p "$PREP_PID" -o stat= || true)"
    [[ -z "$PREP_STATE" || "$PREP_STATE" == *Z* ]] && break
    sleep 1
  done
  echo "旧预处理进程已停止"
)
```

只有看到“旧预处理进程已停止”后再执行下一块。若 Python 的 PID 为1，它是容器主进程，应在平台将启动命令改为下面的接管命令后停止/重建任务，并继续挂载原持久目录；不要删除数据或让平台再次自动启动旧命令。若同一 checkout 还有训练任务，先等它结束再更新。

确认旧程序退出后，在相同持久目录更新 main，再接管原图片池和原准备目录。下面从旧 `preparing.json` 读取原下载 revision；旧命令的可选外部测试集参数也需继续保留。

```bash
(
  set -e
  source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
  conda activate dpo
  cd /mnt/afs/L202500030/xietianyu/workspace/DPO
  git switch main
  git pull --ff-only origin main
  git rev-parse HEAD
  source paths.sh
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  PREP_REVISION="$(python - <<'PYTHON'
import json, os
from pathlib import Path
print(json.loads((Path(os.environ['PICKAPIC_SD15_PREPARED'])/'preparing.json').read_text())['revision'])
PYTHON
)"
  python -u scripts/prepare_downloaded_data.py pickapic --family sd15 \
    --raw "$PICKAPIC_RAW" --model "$SD15_MODEL" \
    --images "$PICKAPIC_IMAGES" --out "$PICKAPIC_SD15_PREPARED" \
    --revision "$PREP_REVISION" --workers 8 --batch-size 256 --adopt-legacy
)
```

旧版本没有保存行清单和游标，因此不能按日志中的370000直接跳过。首次接管必须从原始 Parquet 重建行元数据；图片校验/读取并行进行，原有约42万张图片均保留，校验通过的不会重写。不要伪造游标、删除图片池或移走准备目录。接管后再次中断使用 `--resume`，不再使用 `--adopt-legacy`。已完成输出拒绝覆盖；输出目录和图片池均持独占锁。

日志每约30秒或10000条输出已提交行数、总数、图片复用数、吞吐和剩余扫描时间估计；`phase=finalize` 表示扫描完成，正在做全局去重、连通分量划分和清单生成。

## 训练前数据校验：快速启动与并行完整审计

训练入口默认 `--data-validation auto --validation-workers 8 --validation-samples 64`。已有完整 `ready.json` 且其中清单 SHA256、split 数量、来源 revision 与当前清单匹配时，第一次启动即可走快速路径，不要求先重读全部图片。没有准备标记的数据先做一次有界并行完整审计；清单已含图片 SHA256 时，成功后将记录写到清单旁的 `.pairs.jsonl.validation.json`，后续启动、续训及 DDP 子进程复用匹配记录。没有图片哈希的旧清单仍需完整校验，不凭缓存补造来源身份。

每次仍会顺序读取清单并计算 SHA256，校验全部行的结构、重复 pair ID、跨 split 的场景/规范化 caption/图片路径与哈希隔离、来源身份。路径按字符串归一化，不为每张图片重复发起远程 `resolve/stat`；同一清单不会再深拷贝一次。快速路径默认并行抽查64张唯一图片，另外覆盖每个 split 的首对图片，最多70张；它不保证启动前发现未抽中的文件损坏。训练读取每张图片时，会对即将解码的同一份字节核对清单 SHA256，不额外读取一遍；缺失、损坏或被替换的训练图片会报错停止。

日志会显示 `data_manifest`、`data_schema`、`data_splits`、`data_inventory`、`data_validation`、`data_images`、`data_ready`，附进度、累计秒数和 rank。`data_validation` 中 `mode=sampled` 表示快速抽查、`mode=full` 表示完整审计。图片审计默认8线程，最多保留16个待处理任务；不会为百万图片一次创建百万 future。长文件读取/远程请求本身阻塞时，其他阶段的进度不能保证每10秒出现一次。

数据准备已完成后，普通启动器无需额外参数。要单独查看快速校验或主动做一次完整审计，用已有 `dpo` 环境执行以下 CPU 命令，不启动模型/GPU，也不改训练配置或输出：

```bash
source paths.sh
python -u scripts/check_training_data.py \
  --pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" --mode auto

# 需要排查数据损坏时才完整读取全部唯一图片；可按存储情况调整线程数。
python -u scripts/check_training_data.py \
  --pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" --mode full --workers 8
```

新实验也可通过 `train.py` / `make_config.py` 的 `--data-validation full`、`--validation-workers 4`、`--validation-samples 128` 覆盖；这些参数会进入冻结配置。不要为已有实验修改配置后续训，单独的完整审计使用上面的 CPU 工具即可。缓存和 `ready.json` 不匹配清单时不能冒用：不完整缓存会被忽略；准备标记与清单不一致直接报错。应把准备完成的数据视为只读，不能据缓存宣称所有图片当前都未改变；定期/有疑问时用 full 模式审计。

此改动不会让已运行的旧进程自动加速。先结束旧任务，再更新到包含此功能的版本并重启。若已经产生旧版训练 checkpoint，保留其原 checkout 续训，本次修改会触发训练代码一致性保护；当前仅停在输入校验且还未创建训练输出的任务可以直接使用新代码重新启动。无需重新预处理原始 Parquet 或删除已完成图片。

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

## 双卡普通高斯多视图：无 rollout，K=2/4/6/8 合批

以下启动器只读取已经准备完成的 `ready.json`、`pairs.jsonl` 和固定128条监控 prompt，不自动预处理。先等当前数据任务结束，再把多个双卡训练任务指向同一套只读数据；每个任务使用独立配置和输出目录。

在已分配的双卡 ACP 容器中填写：

```bash
# SD1.5：普通高斯 C，K=2，每卡2对，累积64，全局256对。
SCD_FAMILY=sd15 SCD_VIEWS=2 SCD_ARM=C \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh

# 另一个双卡任务：K=4，每卡1对，累积128，全局仍256对。
SCD_FAMILY=sd15 SCD_VIEWS=4 SCD_ARM=C \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh

# K=6/K=8 各自申请独立双卡任务，默认每卡1对、累积128、全局256对。
SCD_FAMILY=sd15 SCD_VIEWS=6 SCD_ARM=C \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh

SCD_FAMILY=sd15 SCD_VIEWS=8 SCD_ARM=C \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh
```

配置冻结后重启同一命令会尝试完整续训。默认实验名含模型、gaussian、B/C 和 K，避免写进旧 rollout 实验；日志追加到 `logs/<实验名>.txt`。改为 B 使用 `SCD_ARM=B`；重启时保持原环境变量、代码及依赖。修改 batch 等配置需要新的 `SCD_RUN_NAME`，启动器拒绝忽略与请求不同的冻结配置。

这是10000步正式预算，首次可追加 `--stop-after-step 2` 在第2个优化步后计划暂停。它仍包含正式基座评估和完整梯度累积；要做更快、关闭评估的显存/训练验证，使用独立的两步 smoke：

```bash
source paths.sh
python scripts/make_config.py --family sd15 --method multiview-C-gaussian \
  --views 4 --world-size 2 --train_batch_size 1 --smoke \
  --out configs/local-sd15-gaussian-C-k4-smoke-v1.json
bash launchers/train.sh configs/local-sd15-gaussian-C-k4-smoke-v1.json \
  "$RUNS_ROOT/sd15-gaussian-C-k4-smoke-v1"
```

K=2 的 smoke 改成 `--views 2 --train_batch_size 2`；K=6/K=8 改 `--views 6` / `--views 8`，每卡1对，并使用对应 K 的新配置和输出名。SDXL smoke 改 `--family sdxl`、每卡1对并使用独立名字；先准备对应模型、SDXL数据清单和监控面板，不自动下载。

SDXL 的双卡正式启动同样使用该脚本：

```bash
SCD_FAMILY=sdxl SCD_VIEWS=2 SCD_ARM=C \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh

# K=4先做上面的独立smoke，确认显存后再以SCD_VIEWS=4开新任务。
```

起步设置（均为 BF16、LoRA rank16、gradient checkpointing；不是已验证显存结论）：

| 模型/分辨率 | K | 每卡图片对 B | 单次 UNet 图片数 2×B×K | 双卡梯度累积 | 全局图片对 |
|---|---:|---:|---:|---:|---:|
| SD1.5 / 512 | 2 | 2 | 8 | 64 | 256 |
| SD1.5 / 512 | 4 | 1 | 8 | 128 | 256 |
| SD1.5 / 512 | 6 | 1 | 12 | 128 | 256 |
| SD1.5 / 512 | 8 | 1 | 16 | 128 | 256 |
| SDXL / 1024 | 2 | 1 | 4 | 64 | 128 |
| SDXL / 1024 | 4 | 1 | 8 | 64 | 128 |
| SDXL / 1024 | 6 | 1 | 12 | 64 | 128 |
| SDXL / 1024 | 8 | 1 | 16 | 64 | 128 |

本次 SD1.5 K=2 保留 B=2、累积64；K=4/6/8 默认 B=1、累积128。SDXL 继续保持模型系列自己的全局128对，不随这次 SD1.5 实验改变。更大的 K 是否可用必须实测，表格只说明启动配置。

标准 RTX 5090 为32GB，远程实际型号/显存以 `nvidia-smi` 为准（[NVIDIA规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)）。双卡 DDP 各自驻留完整模型，显存不相加。本项目 LoRA reference 复用基座，但 SDXL 还含两个文本编码器和 FP32 VAE，不能只按 LoRA 权重估显存。K 增大时优先减少 B 并增加累积，保持全局图片对数；B 已为1时，增加累积不会缩小一次 K 视图前向。SDXL K=4 若仍 OOM，当前全视图合批路径需要更大单卡显存或另行实现分块/重算，不能保证32GB可运行。

例如 SD1.5 K=2 若 B=2 仍超显存，可开新实验：

```bash
SCD_FAMILY=sd15 SCD_VIEWS=2 SCD_TRAIN_BATCH_SIZE=1 \
  SCD_RUN_NAME=sd15-gaussian-C-k2-2gpu-b1-10000-v1 \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh
```

如果显存充足，K=4/6/8 也可另开 B=2 的实验；以 K=8 为例，累积会自动降为64，全局仍为256对，单次 UNet 则处理32张带噪图片：

```bash
SCD_FAMILY=sd15 SCD_VIEWS=8 SCD_TRAIN_BATCH_SIZE=2 \
  SCD_RUN_NAME=sd15-gaussian-C-k8-2gpu-b2-10000-v1 \
  bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh
```

测显存时在同一容器的另一终端运行，记录每张卡的实际型号、总显存及占用；退出监控用 Ctrl-C，不影响训练：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv -l 2
```

启动器会以 `--effective-batch` 自动调整累积，并在日志中打印 B、K、单次 UNet 图片数、累积及全局图片对数，复用冻结配置时也核对全局 batch。SD1.5沿用β5000、512、全局256，SDXL沿用β500、1024、全局128，均不因 K 改变学习率。训练 JSONL 的 `peak_cuda_bytes` / TensorBoard 的 `resources/peak_cuda_GiB` 可查看记录进程的峰值分配，另用上面的 `nvidia-smi` 查看各卡实际占用。

相同数据清单、seed、默认 balanced 采样和全局 batch 下，B=1/2 每次优化更新使用相同一组图片对，跨卡和 microbatch 分配不同；噪声、时间步、VAE posterior 的随机数分配及浮点累加仍可能不同。因此不能承诺 B 只影响计算时间，后续补做相同 K 的 B=1/2 对照应使用新实验名。合批减少调用次数，不保证等比例提速；比较 K 时另记录 GPU 时间，不能把同图片对曝光当作同算力预算。

不使用启动脚本时，等价配置生成示例：

```bash
python scripts/make_config.py --family sd15 --method multiview-C-gaussian \
  --views 4 --world-size 2 --train_batch_size 1 --effective-batch 256 \
  --max_train_steps 10000 --out configs/local-sd15-gaussian-C-k4-v1.json
```

## 默认 rollout 多视图 C：协调入口，自动评估

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
python -m unittest discover -s tests -p test_validation.py -v
python -m unittest discover -s tests -p test_config.py -v
python -m unittest discover -s tests -p test_joint.py -v
python -m unittest discover -s tests -p test_stage_pause.py -v

# 审查完成后只提交属于本次修改的文件，推开发分支并通过 PR 合入 main。
# 不要将尚未确认的算法改动、数据、日志或模型一起提交。

# 远程：先结束旧实验或保留旧代码快照，再更新main并记录版本。
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

本机当前使用 `dev/local-development` 开发分支，远程正式实验使用已合并的 main。详细规矩见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。

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
