# 商汤大装置：从部署到 DPO 实验

按本文顺序先跑通 SD1.5 纯 DPO，再比较多视图 B/C。以下命令在商汤已分配的单机容器中执行；先从 2 张 RTX 5090 开始，后续可使用 4/8 张或 H100。模型、数据和输出挂载到持久存储，确认新容器仍能访问。这里不提供或执行平台资源申请命令；本机 irip 集群仍遵守 Slurm 约定。

## 两套 Gaussian C 对照：固定图片对 / 固定视图预算

本节是2026-10-25新增的比较入口，支持 SD1.5 512×512 和 SDXL 1024×1024。两套实验都训练10000个优化步，保留相同 seed、学习率、β、LoRA、监控面板和保存周期；区别仅在每步图片对数如何随 K 调整。普通高斯 C 仍是同一图片对的 K 个独立时间/噪声视图先平均 margin，再计算 DPO loss。

- `acp_gaussian_fixed_pairs.sh`：固定每步图片对组数，K 增大时视图计算量增加。
- `acp_gaussian_fixed_views.sh`：固定每步图片对视图数，图片对组数按 K 缩小，优化更新次数保持一致。

新实验统一基准为 **SD1.5=384、SDXL=192**，使 K=2/4/6 在2/4/8卡和默认每卡 batch 下都能精确整除。旧入口仍为256/128，不作为本组新预算的匹配基线。每个模型先运行一个新的 DPO 基线即可供两组共用；两个入口的 DPO 实际训练配置相同，仅预算标签和输出名不同，无需各跑一遍。

| 实验 | K | SD1.5 每步图片对 / 图片对视图 | SDXL 每步图片对 / 图片对视图 | 相对 DPO 的主要 UNet 样本量 |
|---|---:|---:|---:|---:|
| DPO，共用基线 | 1 | 384 / 384 | 192 / 192 | 1× |
| fixed-pairs | 2 | 384 / 768 | 192 / 384 | 2× |
| fixed-pairs | 4 | 384 / 1536 | 192 / 768 | 4× |
| fixed-pairs | 6 | 384 / 2304 | 192 / 1152 | 6× |
| fixed-views | 2 | 192 / 384 | 96 / 192 | 1× |
| fixed-views | 4 | 96 / 384 | 48 / 192 | 1× |
| fixed-views | 6 | 64 / 384 | 32 / 192 | 1× |

这里一“对视图”含 winner 和 loser；策略和参考分别计算。同视图预算对齐主要 UNet 样本量，不保证总 FLOPs 或墙钟时间完全相同，VAE/文本编码的复用和合批效率会影响速度。比较生成指标，同时报告训练时间和峰值显存；C 与 DPO 的训练 loss 聚合方式不同，不能直接据其高低判定质量。

在远程完成模型、数据准备及小规模 GPU 检查后，分别在已分配的容器运行以下命令。每行是独立实验，不自动申请或并行提交资源：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO
# SD1.5：两组共用的新 DPO 基线，384 对/步。
SCD_FAMILY=sd15 bash launchers/acp_gaussian_fixed_pairs.sh dpo
# 固定384对；分别将 SCD_VIEWS 设为2、4、6。
SCD_FAMILY=sd15 SCD_VIEWS=2 bash launchers/acp_gaussian_fixed_pairs.sh
# 固定384对视图；分别将 SCD_VIEWS 设为2、4、6。
SCD_FAMILY=sd15 SCD_VIEWS=2 bash launchers/acp_gaussian_fixed_views.sh

# SDXL：独立的新 DPO 基线，192 对/步。
SCD_FAMILY=sdxl bash launchers/acp_gaussian_fixed_pairs.sh dpo
# 分别将 SCD_VIEWS 设为2、4、6。
SCD_FAMILY=sdxl SCD_VIEWS=2 bash launchers/acp_gaussian_fixed_pairs.sh
SCD_FAMILY=sdxl SCD_VIEWS=2 bash launchers/acp_gaussian_fixed_views.sh
```

默认2卡；4/8卡通过 `SCD_WORLD_SIZE=4` 或 `8` 选择并新建实验，不能改变已有运行的卡数续训。SD1.5 的 K≤2 默认每卡2对，其余每卡1对；SDXL 默认每卡1对。梯度累积自动由预算推导，显存容许时可用 `SCD_TRAIN_BATCH_SIZE` 改每卡 batch，但必须精确整除预算。完整梯度累积后只更新一次参数，固定视图模式不会增加优化更新次数。

默认实验名明确区分，例如：

```text
sd15-dpo-fixed-pairs-p384-2gpu-10000-v1
sd15-gaussian-C-k2-fixed-pairs-p384-2gpu-10000-v1
sd15-gaussian-C-k2-fixed-views-v384-2gpu-10000-v1
sdxl-gaussian-C-k6-fixed-views-v192-2gpu-10000-v1
```

配置位于 `configs/local-<实验名>.json`，控制台日志在 `logs/<实验名>.txt`，实验输出位于 `$RUNS_ROOT/<实验名>`。可设置 `SCD_RUN_NAME` 创建独立重复实验；不要复用旧名字。重启同一命令会自动恢复 latest；首次可追加 `--stop-after-step 2` 在正式预算内暂停，然后去掉该选项继续。这仍会执行正式梯度累积与基座监控；快速 smoke 使用下节独立小跑方式。

预算保存为 `training.comparison_budget`；`training.json` 同时记录 `effective_batch_size_pairs`、`effective_batch_size_pair_views`、`planned_pair_exposures`、`planned_pair_view_exposures`。每步 JSONL/TensorBoard 记录 `pair_views_seen`，直接使用已有 `cost.prompt_pair_evaluations`，不把唯一图片数、图片对访问次数与视图次数混为一谈。完整保存和恢复沿用每200步；两模式和不同 K 使用各自恢复点。

新入口只允许追加 `--stop-after-step N`、`--resume_from_checkpoint PATH`（或 `--resume PATH`）等已列出的运行控制；训练 CLI 覆盖会被拒绝。默认是 C，可显式传 `multiview-B-gaussian` 做相同预算的 B 对照。rollout 不属于这组等视图计算比较。

需要自定义预算时，两组及 DPO 都显式设置同一个 `SCD_BUDGET_PER_UPDATE` 并使用新实验名。固定视图预算必须整除 `每卡B × GPU数 × K`，不静默取整。标准库配置预览示例（不会训练）：

```bash
python scripts/make_config.py --family sd15 --method multiview-C-gaussian \
  --views 6 --world-size 2 --train_batch_size 1 --max_train_steps 10000 \
  --budget-unit pair_views --budget-per-update 384 \
  --out configs/local-sd15-C-k6-fixed-views-preview-v1.json
```

`--budget-unit pairs` 切换到固定图片对；预算参数与 `--effective-batch` / 显式梯度累积互斥。两种模式在 `validate_config` 中复核实际预算，防止配置数值与标签不一致。

## SDXL：1024×1024 的完整入口

SDXL 通用入口为 `launchers/acp_sdxl.sh [方法]`，保留原全局128对，支持纯 DPO、rollout B/C、普通高斯 B/C。本节用于部署和已有配置；新的两套 Gaussian C 预算对照使用上一节的192基准入口。省略方法默认纯 DPO，先建立 SDXL 自己的基线。训练和评估均为1024×1024；论文依据和与原版全参 DPO 的区别见 [docs/SDXL.md](docs/SDXL.md)。模型路径、数据和监控面板来自同一份资源表，使用 SDXL 独立清单；图片池可与 SD1.5 共享。

先在远程激活项目环境并确认模型已完整下载。下载工具及 `plan` 命令见本文末尾的资源下载说明；这一步只准备命令，不自动下载。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate dpo
source paths.sh
python scripts/check_deployment.py --family sdxl --stage assets
# 在已分配 CPU 资源上准备，完成后再启动 GPU 实验。
bash launchers/acp_prepare_data.sh pickapic sdxl
```

预处理使用 SDXL 的两个 tokenizer 过滤过长 caption，保留原始图片字节；训练读取时才缩放到1024。模型下载记录自动提供 revision；手动取得的模型用配置生成命令的 `--model-revision` 指定真实版本，不能用占位值。启动检查会拒绝 refiner、不完整双 CLIP、错误 VAE scaling 和未完成的数据准备。

第一次先在已分配的双卡容器验证真实模型、保存和恢复。下面的 smoke 保持1024、每卡1对，每次只累积1个 microbatch，关闭生成评估与诊断；单独实验目录只做两次更新：

```bash
python scripts/make_config.py --family sdxl --method dpo --world-size 2 \
  --smoke --out configs/local-sdxl-dpo-smoke-v1.json
bash launchers/train.sh configs/local-sdxl-dpo-smoke-v1.json \
  "$RUNS_ROOT/sdxl-dpo-smoke-v1" --stop-after-step 1
bash launchers/train.sh configs/local-sdxl-dpo-smoke-v1.json \
  "$RUNS_ROOT/sdxl-dpo-smoke-v1" --resume_from_checkpoint latest
```

多视图在正式运行前也用上面的方式测试，将方法改为下表对应名字并加 `--views 2`，配置和输出使用独立名字。增大 K 时在相同 K 下重新测显存。

正式启动示例，以下是不同实验的独立命令，按分配资源选择执行：

```bash
# 纯 DPO；默认2卡，10000步。
bash launchers/acp_sdxl.sh dpo
# 普通高斯 K2-C（K 个视图合批）；比较 B 时改成 multiview-B-gaussian。
SCD_VIEWS=2 bash launchers/acp_sdxl.sh multiview-C-gaussian
# 独立 model rollout K2-C；比较 B 时改成 multiview-B。
SCD_VIEWS=2 bash launchers/acp_sdxl.sh multiview-C
# 已分配4卡或8卡时，显式改变卡数；新建实验，不能续接双卡恢复点。
SCD_WORLD_SIZE=4 bash launchers/acp_sdxl.sh dpo
SCD_WORLD_SIZE=8 bash launchers/acp_sdxl.sh dpo
```

| 方法 | 参数 | K | rollout | 高斯视图合批 |
|---|---|---:|---|---|
| 纯 DPO | `dpo` | 1 | 无 | 无 |
| 普通高斯 B/C | `multiview-B-gaussian` / `multiview-C-gaussian` | 2/4/6/8 | 无 | 有 |
| 独立 rollout B/C | `multiview-B` / `multiview-C` | 任意正整数，默认2 | model，original 时间采样 | 无 |

正式配置沿用 SDXL 研究默认值：LR1e-5、β500、LoRA rank/alpha16、BF16、seed17、10000优化步、每200步完整保存、每1000步保留 LoRA；TensorBoard 和 PickScore 分段监控开启。全局 batch 固定128对，每卡默认1对：

| GPU 数 | 每卡图片对 | 梯度累积 | 全局图片对 |
|---|---:|---:|---:|
| 2 | 1 | 64 | 128 |
| 4 | 1 | 32 | 128 |
| 8 | 1 | 16 | 128 |

K 不计入全局图片对 batch，但增加 UNet 计算和激活显存。普通高斯合批一次每卡处理 `2×B×K` 张带噪图片；rollout 保留各视图的策略图，还额外执行 detached rollout。32GB 5090 上的可用 K 和速度须实测，多卡 DDP 不合并显存。

默认实验名含 `sdxl`、方法、K（多视图）和卡数，例如 `sdxl-dpo-2gpu-10000-v1`、`sdxl-gaussian-C-k2-2gpu-10000-v1`、`sdxl-rollout-C-k2-2gpu-10000-v1`。需要重复实验时设置新的 `SCD_RUN_NAME`。控制台日志在 `logs/<实验名>.txt`，冻结配置在 `configs/local-<实验名>.json`，输出在 `$RUNS_ROOT/<实验名>`。

第一次执行正式入口也可追加 `--stop-after-step 2`；它保留正式累积和基座评估，耗时高于独立 smoke。重启时去掉暂停参数，原命令自动恢复 latest。续训保持相同方法、K、卡数、代码、环境、配置和资源；启动器拒绝不匹配的冻结配置。已完成实验不再重启；新代码使用新实验目录。

独立评估使用 SDXL 评估配置和训练记录中的模型版本；下面的1000步须已产生保留的 LoRA，换成实际存在的步数：

```bash
python evaluate.py --config configs/evaluate-sdxl.json \
  --run "$RUNS_ROOT/sdxl-dpo-2gpu-10000-v1" --step 1000 \
  --benchmark pickapic --metrics pickscore \
  --out "$RUNS_ROOT/sdxl-dpo-2gpu-10000-v1/eval-pickapic-step1000-v1" \
  --execution standalone --run-label sdxl-eval-step1000-v1
```

训练内监控是30步 Euler / CFG5，独立评估默认50步 Euler / CFG5；同一对照内保持相同面板、seed 和采样配置。当前本地验证覆盖配置、真实 tiny SDXL 双 CLIP/UNet/VAE/LoRA 的前后向、梯度检查点及 adapter 保存加载；完整1024模型和5090/H100显存性能待远程验证。

## 当前入口：先准备一次数据，再启动双卡实验

数据准备和训练已经分开。根据2026-10-24的运行日志，SD1.5 Pick-a-Pic 已完成导出：train=728226、validation=40815、test=37392，且部署检查已找到清单和128条监控 prompt；不需要再扫描原始 Parquet。这里记录的是当时的日志结果，当前挂载状态以启动检查为准。

| 用途 | 入口 | GPU 数 | 每卡图片对 B | 梯度累积 | 全局图片对 |
|---|---|---:|---:|---:|---:|
| 数据准备/补齐监控面板 | `bash launchers/acp_prepare_data.sh pickapic sd15` | 0 | — | — | — |
| SD1.5 纯 DPO | `bash launchers/acp_sd15_dpo_2gpu.sh` | 2 | 2 | 64 | 256 |
| SD1.5 普通高斯 C，K=2 | `SCD_VIEWS=2 bash launchers/acp_multiview_gaussian_2gpu.sh` | 2 | 2 | 64 | 256 |
| SD1.5 普通高斯 C，K=4 | `SCD_VIEWS=4 bash launchers/acp_multiview_gaussian_2gpu.sh` | 2 | 1 | 128 | 256 |
| SD1.5 普通高斯 C，K=6 | `SCD_VIEWS=6 bash launchers/acp_multiview_gaussian_2gpu.sh` | 2 | 1 | 128 | 256 |
| SD1.5 普通高斯 C，K=8（保留） | `SCD_VIEWS=8 bash launchers/acp_multiview_gaussian_2gpu.sh` | 2 | 1 | 128 | 256 |

K 是视图数，不是 GPU 数。每个训练任务独立申请两张卡，可以并行读取同一份已完成的数据；实验目录各自独立。训练脚本不会预处理或下载数据。

双卡入口共同调用 `launchers/acp_common.sh`：自动激活远程 `dpo` 环境，进入脚本所属 checkout，加载同一资源表，检查准备标记和 GPU，冻结配置并启动训练。统一10000步、BF16、LoRA rank/alpha16、LR1e-5、β5000（SD1.5）、seed17、每200步完整保存、TensorBoard 和原固定 PickScore 评估；快速数据校验见下文。重启复用冻结配置并自动尝试从 latest 恢复，缺少有效 checkpoint 时停止，不覆盖旧输出。

完成下节更新后，在 ACP 已分配双卡容器的启动命令中选择一条：

```bash
# SD1.5 纯 DPO，默认独立实验 sd15-dpo-2gpu-10000-v1。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_2gpu.sh
```

```bash
# 普通高斯 C：分别为 K=2、4、6 申请独立的双卡任务，只改此处的数字。
export SCD_VIEWS=2
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_multiview_gaussian_2gpu.sh
```

默认高斯实验名为 `sd15-gaussian-C-k<K>-2gpu-10000-v1`。需要新实验时增加 `export SCD_RUN_NAME=...`，例如旧 DPO 已有输出而本次使用新代码时用 `sd15-dpo-2gpu-10000-v2`。配置位于 `configs/local-<实验名>.json`，控制台日志位于 `logs/<实验名>.txt`，实验/TensorBoard/checkpoint 位于 `$RUNS_ROOT/<实验名>`。后续续训保持同一代码、配置、环境、实验名及卡数。

```bash
tail -f /mnt/afs/L202500030/xietianyu/workspace/DPO/logs/sd15-dpo-2gpu-10000-v1.txt
```

已有8卡入口 `launchers/acp_sd15_dpo_8gpu.sh` 保留，当前优先使用上述双卡命令。没有执行远程 GPU 训练或显存实测。

## ACP：4张5090、K=2、多视图 C

启动脚本为 `launchers/acp_sd15_c_k2_4gpu.sh`。环境激活、资源检查、日志追加和自动恢复方式与8卡纯DPO一致；使用独立实验名 `sd15-C-k2-4gpu-10000-v1`，每卡batch2、累积32、全局batch256。沿用10000步、LR1e-5、β5000、LoRA16、seed17和每200步完整保存，保留 TensorBoard / PickScore 监控。

这里的 C 是当前 `configs/multiview-C.json`：K2、model rollout，时间采样沿用 original；不是关闭 rollout 的普通高斯双视图 C。它与8卡纯DPO保持相同全局batch和优化步预算，计算成本和随机数分配不同。

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_c_k2_4gpu.sh
```

如果8卡任务已在同一个持久代码目录运行，可将这个新脚本单独保存或直接粘贴到新的4卡ACP启动命令中，使用相同版本的训练代码。不要为取得新启动脚本而更新旧任务正在使用的训练代码或环境。

```text
日志：/mnt/afs/L202500030/xietianyu/workspace/DPO/logs/sd15-C-k2-4gpu-10000-v1.txt
配置：configs/local-sd15-C-k2-4gpu-10000-v1.json
输出：/mnt/aoss/xietianyu/self_correct/runs/sd15-C-k2-4gpu-10000-v1
```

两个任务使用各自的完整恢复点；此C实验重启时仍需4张GPU。新实验可覆盖 `SCD_RUN_NAME` 为新的唯一名字；不要复用纯DPO的实验名。

## ACP：2张5090，纯 DPO / K=2-C

在已分配2张5090的容器中使用下面两个启动器。它们会激活 `dpo` 环境、进入远程仓库并检查资源。每卡batch2、梯度累积64，保持全局batch256；10000优化步、每200步完整保存，其他训练与监控参数沿用上面的版本。K2-C仍使用当前默认的 model rollout C。

数据准备已独立为 `bash launchers/acp_prepare_data.sh pickapic sd15`，应先在已分配的 CPU 资源上完成。双卡训练启动器只检查并读取准备好的图片对清单和固定128条监控提示词；输入缺失或准备未完成时停止。DPO/C 可共同读取同一套已完成的数据，准备恢复和来源版本设置见本文的数据准备章节。续训时恢复原输入和配置，不自动重建数据。

| 方法 | 脚本 | 独立实验名 |
|---|---|---|
| 纯 DPO | `launchers/acp_sd15_dpo_2gpu.sh` | `sd15-dpo-2gpu-10000-v1` |
| K=2，C | `launchers/acp_sd15_c_k2_2gpu.sh` | `sd15-C-k2-2gpu-10000-v1` |

先任选一个做两步检查；下面是两种方法各自的命令，只申请2张GPU时逐个运行：

```bash
bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_2gpu.sh --stop-after-step 2
```

```bash
bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_c_k2_2gpu.sh --stop-after-step 2
```

这里保留10000步配置及原定学习率计划，在第2个优化步额外完整保存并暂停；仍会执行固定的基线 PickScore 监控，因此不只是两步训练耗时。短跑验证的是启动、计算、保存及监控流程，不用于判断算法效果。检查成功后，在对应ACP任务的启动命令中去掉暂停参数即可继续至10000步。之后被驱逐重启也使用相同命令：

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_2gpu.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_c_k2_2gpu.sh
```

每个实验的控制台日志追加到 `/mnt/afs/L202500030/xietianyu/workspace/DPO/logs/<实验名>.txt`，配置保存在仓库 `configs/local-<实验名>.json`，完整恢复点和TensorBoard等保存在 `/mnt/aoss/xietianyu/self_correct/runs/<实验名>`。若要新开重复实验，在启动前用 `export SCD_RUN_NAME=<新名字>`，不要混用两种算法的名字。

双卡实验需从新的目录开始，不能直接用8/4卡checkpoint恢复。自动续训要求保持2张GPU、同一代码和冻结配置；有实验目录却缺少有效checkpoint或配置时会报错停止。取得启动器后无需改动现有训练代码，正在运行的其他实验继续固定原版本。

## ACP 启动时 Git 报 dubious ownership

`fatal: detected dubious ownership in repository` 表示容器当前用户与挂载仓库所有者不一致，Git 的所有者检查阻止了版本读取。旧启动器在 GPU 检查之前直接执行 `git rev-parse HEAD`，会因 `set -e` 提前退出。此时尚未开始训练，不需要删除实验目录或数据。

无需更新代码，可在ACP启动命令中先信任自己的这个仓库，再执行原启动器。纯DPO示例；C实验将最后一行的脚本名换为 `acp_sd15_c_k2_2gpu.sh`：

```bash
set -e
git config --global --add safe.directory /mnt/afs/L202500030/xietianyu/workspace/DPO
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_2gpu.sh
```

这条配置只为当前容器用户添加上述仓库的信任项，便于训练内部也记录commit；新容器不一定保留该用户配置，因此可将它留在ACP启动命令中。新版2/4/8卡启动器在记录版本时使用仅对该命令有效的 `git -c safe.directory="$PWD"`，若仍无法读取Git元信息则警告后继续资源检查；不会修改全局Git配置。训练内部读取commit失败时已有容错，不会阻断训练。

## 先更新代码

截图中的远程代码目录为下面的位置。旧实验需要续训时，保留其冻结 checkout，新版本用于新实验。

```bash
set -e
cd /mnt/afs/L202500030/xietianyu/workspace/DPO
git config --global --add safe.directory "$PWD"
git status --short --branch

# 旧容器可能手写过同名、未跟踪的双卡启动器；先备份，避免 pull 被阻挡。
SCD_LAUNCHER_BACKUP="logs/launcher-backups/$(date +%Y%m%d-%H%M%S)"
for SCD_OLD_LAUNCHER in launchers/acp_sd15_dpo_2gpu.sh launchers/acp_sd15_c_k2_2gpu.sh; do
  if [[ -f "$SCD_OLD_LAUNCHER" ]] && ! git ls-files --error-unmatch "$SCD_OLD_LAUNCHER" >/dev/null 2>&1; then
    mkdir -p "$SCD_LAUNCHER_BACKUP"
    mv "$SCD_OLD_LAUNCHER" "$SCD_LAUNCHER_BACKUP/"
  fi
done

git fetch origin
git switch main
git pull --ff-only origin main
git log -1 --oneline
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
# 独立 CPU 任务；默认8线程、原子批次256，成功后补齐缺失的固定监控面板。
bash launchers/acp_prepare_data.sh pickapic sd15

# SDXL 要单独按该模型的 tokenizer 过滤；不能直接冒用 SD1.5 清单。
# bash launchers/acp_prepare_data.sh pickapic sdxl
```

独立脚本使用同一 `dpo` 环境和资源表。已有 `ready.json` 与 `pairs.jsonl` 时跳过数据导出，已有监控 prompt 时跳过面板准备，不重新读取全部图片；训练入口再核对清单与准备记录。如果需要改变来源、过滤规则或输出位置，使用新的 `--out`，并为新清单配置单独的监控面板，不能在完成目录上覆盖。日志追加到 `logs/prepare-pickapic-sd15.txt`。

手动下载而没有下载记录时，可沿用真实 revision：

```bash
bash launchers/acp_prepare_data.sh pickapic sd15 \
  --revision liuhuohuo2/pick-a-pic-v2@f602d48e244a68f04e9bcb6ca2a9ed5bcb578576
```

这条 revision 仅适用于前面已确认的这批下载，不是其他数据的通用版本号。

HPDv3 下载并解压完成后也使用同一数据准备入口：

```bash
bash launchers/acp_prepare_data.sh hpdv3 sd15
# bash launchers/acp_prepare_data.sh hpdv3 sdxl
# 仅为独立 SD3 trainer 导出不做 tokenizer 过滤的数据；不启动 SD3 训练：
# bash launchers/acp_prepare_data.sh hpdv3 unfiltered
```

HPDv3 沿用现有实现，需官方 `train.json`、`test.json`、`images/` 和可核实的下载 revision，并使用已下载的 Pick-a-Pic held-out 做交叉排除；没有下载记录时传入其真实 `--revision`。不复用 Pick-a-Pic 的 `--workers/--batch-size/--resume`，当前 HPDv3 处理并未新增并行与中断恢复。该入口仅导出 HPDv3 清单，不改变当前 SD1.5 的 Pick-a-Pic 训练集或评估面板；未来 HPDv3 实验用 `make_config.py --dataset hpdv3` 并独立准备与该清单隔离的评估面板。

如果资源检查报告没有可核实的下载版本，先查原下载记录。只有这种情况才在处理命令后加 `--revision`，传入实际的“数据仓库@下载 commit”；不要填占位符，也不要把现在的远端 main 当成当时的下载版本。

处理完成后应有 `pickapicv2/images/`、`pickapicv2/prepared-sd15/pairs.jsonl` 和 `benchmarks/sd15/pickapic-monitor.jsonl`。监控集默认128条；其余 Pick-a-Pic test prompt 另外保留。`ready.json` 表示数据准备已完成；只有 `preparing.json` 时不启动训练。

### Pick-a-Pic 并行、恢复与旧任务切换

默认使用8个图片 I/O 线程，每256条原始记录提交一个原子批次。图片保留原始字节、文件名和目录；已有图片按 SHA256 校验后复用，不重写。`--workers 16` 可在存储允许时提高并发；若延迟变大则降回4或8。batch 控制内存和重做范围，不是训练 batch。

新版本中断后，在相同输出目录恢复（线程数和 batch 大小可以调整）：

```bash
bash launchers/acp_prepare_data.sh pickapic sd15 \
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
  bash launchers/acp_prepare_data.sh pickapic sd15 \
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

使用页首统一双卡入口，正式预算10000步，默认每卡2对、累积64、全局256对；其余超参数及评估与高斯对照一致。

```bash
bash launchers/acp_sd15_dpo_2gpu.sh
```

重启同一实验继续执行相同命令；已有有效完整 checkpoint 时自动恢复。需要在第2步计划暂停以检查保存，可追加 `--stop-after-step 2`，此时仍保留正式全局 batch 和基座评估；快速 GPU 验证使用前一节独立 smoke。新代码不能绕过旧 checkpoint 的代码身份检查，必要时使用新实验名。

保持每卡batch2、全局batch256时，2/4/8卡对应梯度累积64/32/16。换卡数需要独立实验；当前优先双卡，8卡旧入口保留。

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
