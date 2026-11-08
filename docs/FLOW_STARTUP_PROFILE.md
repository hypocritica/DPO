# SLAI Z-Image：加载诊断与 microbatch

## 连续任务入口（2026-11-08）

当前任务结束后更新SLAI代码；已有25→50续训应先在原版本完成。此次发布分支为
`feat/flow-benchmark-slai-20261108`，包含远程最新Z-Image FP32/BF16 padding修复。
以下更新不清理模型、快照、模型验证报告或已有实验输出：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
git -c safe.directory="$PWD" fetch origin feat/flow-benchmark-slai-20261108
git -c safe.directory="$PWD" switch --detach FETCH_HEAD
```

在SLAI平台创建GPU任务时，把下面一行作为任务命令。入口自动激活现有mvpo-flow，
前提为该任务可访问现有代码、Z-Image模型、128对debug快照及reports/slai中的有效模型报告。
不需要再次下载、做全量数据准备或手填冻结配置。首次整套测量可预留2小时；这只是基于旧日志的估计。
上述git命令仅对本次操作信任当前项目路径，兼容日志中的dubious ownership，不修改全局Git配置。

任务A：2×原卡型80GB GPU、8 vCPU、128GiB RAM，单任务内顺序执行：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
bash launchers/flow/benchmark.sh --tag cpu8-ram128-v1
```

任务B：2×相同卡型80GB GPU、16 vCPU、256GiB RAM，运行相同设置：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
bash launchers/flow/benchmark.sh --tag cpu16-ram256-v1
```

默认每rank固定OMP/MKL各4线程、OpenBLAS 1线程；两机相同，不因CPU数改变算法或线程设置。
实际GPU型号、可见CPU亲和性和容器CPU/内存配额写入报告。先确认两边GPU型号一致；
CPU与RAM同时增加，结果只能直接说明整体资源配置差异，不能单独证明CPU核数的因果作用。
如果两次任务共享同一物理机/存储，两边同时测量也可能相互争用，优先顺序运行。

自动执行6项，所有GPU子进程都在前一项退出后才启动：

| 顺序 | 测量 | 设置 |
|---|---|---|
| 1 | load-first | 双rank并发加载，按组件计时，无训练 |
| 2 | load-repeat | 完全相同设置再次加载，观察缓存/波动 |
| 3 | k1-b1 | DPO K1，B1、累积2、20步 |
| 4 | k1-b2 | DPO K1，B2、累积1、20步 |
| 5 | k4-b1 | Global MVPO C K4，B1、累积2、20步 |
| 6 | k4-b2 | Global MVPO C K4，B2、累积1、20步 |

同一K内两项预算一致：K1为4 pair-views/update，K4为16 pair-views/update。
不是正式128预算吞吐实验，也不测stagewise或最终图像质量。first/repeat仅表示顺序，不保证冷/热缓存；
不清系统缓存。加载与训练各自重新建进程；训练整项wall_seconds包含加载与checkpoint保存，
seconds_per_update单独从生产训练日志提取后10步中位数，不把模型加载算入更新速度。
关闭W&B/TensorBoard，保留JSONL、真实训练和原checkpoint机制。各rank显存为PyTorch allocated峰值，
不能当作nvidia-smi整卡用量；启动profile另有reserved及free信息。

某一子进程非零退出（如B2 OOM）会保留该项日志，继续后续项；整套最终返回非零以标识有失败。
TERM/INT会转发给当前子进程并停止后续项，返回75。相同命令重跑会跳过已成功项，失败项新建attempt，
使用新的run-tag从基座开始测量，旧checkpoint不删除、不混入吞吐统计。
同tag不可同时执行；变更steps、线程、入口代码或输入环境时必须换tag。

若要分成3次提交，依次运行下列命令；每一行是独立任务，使用相同资源规格和tag：

```bash
bash launchers/flow/benchmark.sh --tag cpu8-ram128-v1 --part load
bash launchers/flow/benchmark.sh --tag cpu8-ram128-v1 --part k1
bash launchers/flow/benchmark.sh --tag cpu8-ram128-v1 --part k4
```

16核任务同样可拆分，tag换为cpu16-ram256-v1。若之后需要单独比较每rank 8线程：

```bash
bash launchers/flow/benchmark.sh --tag cpu16-ram256-t8-v1 --part load --threads 8
```

缓存状态仍需结合repeat与组件数据判断，不能把后一次快直接归因于线程数。
只读预览：`bash launchers/flow/benchmark.sh --tag preview --plan-only`。
每30秒在平台日志打印当前console.log位置；完整子进程输出保留在各attempt目录。

汇总与证据位于：

```text
logs/flow-benchmarks/<tag>/plan.json            # 代码/环境身份、主机、容器配额
logs/flow-benchmarks/<tag>/summary.md           # 简短表格
logs/flow-benchmarks/<tag>/summary.json         # 加载分项、吞吐、各rank峰值、退出状态
logs/flow-benchmarks/<tag>/<stage>/<attempt>/   # 完整console、逐rank profile、结果
```

训练checkpoint仍写入资源表RUNS_ROOT，路径记录在console及summary中；可用--runs-root显式指定新根目录。
先提供两组summary.json；需要判断CPU限流/I/O时，再读取对应profile/rank-*.jsonl。
本地只完成CPU测试，以下底层命令保留给进一步有针对性的诊断。

本轮用户任务为双80GB GPU、共8 vCPU、128GiB RAM。已有日志只确认DPO B1/K1前25步与计划保存，
恢复后26–50步是否完成需看后续日志。不能从38.68GiB的PyTorch allocated峰值推断B2/K4一定可用。

本地同型号模型transformer是2个分片，合计11.46GiB；text encoder是3个分片，合计7.49GiB。
原日志2个分片耗时10分04秒，与transformer布局一致，需远程组件标签最终确认。
训练按FP32加载transformer，磁盘权重dtype由诊断读取header确认。pipeline还会加载文本编码器、
移动到GPU、插入LoRA和初始化DDP；双rank从相同AOSS路径并发读取。现有模型身份检查不全量哈希权重。
slai_common.sh默认OMP_NUM_THREADS=4，仅当用户未设置时生效；两rank共8线程，不能先假定为单线程瓶颈。

## 1. 独立加载诊断

新增scripts/profile_flow_startup.py只需复制这一个文件到SLAI现有checkout的scripts目录。
它调用原checkout的PreferenceBackend，不修改生产模型源码、配置、数据或checkpoint。
运行时在当前进程内临时包装model_index指定的组件加载类与pipeline.to；退出后恢复。
默认只读小配置与safetensors header，不import torch、不加载GPU模型。--load才加载完整后端。
报告每rank独立JSONL，记录组件begin/end、墙钟/CPU时间、进程RSS、缺页、I/O计数、cgroup限流/内存、
GPU allocated/reserved及空闲显存。I/O计数在FUSE/网络文件系统上可能不代表底层存储流量。
backend.total包含子阶段，不能把嵌套时间再次相加。该工具不测数据解码、checkpoint恢复或训练吞吐。

在SLAI任务容器激活mvpo-flow，使用原日志中的冻结配置（只读）：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
source /mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
conda activate mvpo-flow
FLOW_PROFILE_CONFIG="$PWD/configs/local-zimage-dpo-k1-pair_views128-2h100-b1-50-smoke50-predata-v1.json"
FLOW_PROFILE_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$$"
python scripts/profile_flow_startup.py --config "$FLOW_PROFILE_CONFIG" \
  --out "logs/startup-profile/$FLOW_PROFILE_TAG-metadata"
```

先等原GPU训练/恢复退出，再在同一个已分配双GPU任务内做一次并发加载，不同时占用原训练的显存：

```bash
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/profile_flow_startup.py --config "$FLOW_PROFILE_CONFIG" \
  --out "logs/startup-profile/$FLOW_PROFILE_TAG-concurrent" --load --execution standalone
```

这一步沿用当前CPU线程环境、原模型dtype与放置。实际GPU型号写入runtime记录，不仅依赖平台SKU。
只看新日志最慢阶段，再选择一种对照，避免盲目全部重跑：

- load.transformer CPU时间接近可用核数×墙钟时间，或cpu.stat限流显著增加：考虑CPU/转换瓶颈；
  在新out下显式--cpu-threads 2，与默认每rank4比较。改变CPU申请量也必须同模型同缓存条件比较。
- 每rank加载时间相差大或两rank都在读取阶段等待：可在新out下增加--serial-load，比较逐rank加载与并发；
  串行后第二rank可能命中缓存，单次变快不能单独证明竞争。比较backend.total及总墙钟时间。
- memory.current接近memory.max，memory.events的high/max/oom或swap计数增加：先处理主机内存压力。
- pipeline.to_device占主要时间：检查搬运和实际GPU状态；DDP阶段慢则检查同步/通信。
- 后续加载比首次明显快：缓存可能起作用；不要清系统缓存，不把热缓存提速归因于线程设置。

如果需要有限读取探针，在完成首次加载测量后运行；它会预热每个分片的前32MiB，不是完整冷读带宽：

```bash
python scripts/profile_flow_startup.py --config "$FLOW_PROFILE_CONFIG" \
  --out "logs/startup-profile/$FLOW_PROFILE_TAG-read" --read-mib-per-shard 32
```

六分片最多读取192MiB，不复制模型、不写权重。不要预先迁移几十GiB模型或改成BF16权重以掩盖瓶颈。
若需测试本地盘，先核实SLAI任务的可用本地挂载、容量与持久性，另做存储对照。

## 2. B1/B2吞吐试验

入口新增可选--smoke-batch-size与--smoke-accumulation-steps，默认仍1/1；旧冻结配置和输出名不变。
显式非默认探针名包含实际pair-view预算及累积数。编号正式入口新增--batch-size，自动重算累积，
原默认仍B1。更新涉及scripts/run_flow.py及两个shell入口，不修改训练算法源码；不改已有实验配置。
用新run-tag从基座开始。下面两个探针同为20更新、4 pair-views/update，学习率/数据/精度一致：

```bash
FLOW_DEBUG_STEPS=20 bash launchers/flow/debug_train.sh zimage dpo \
  --smoke-batch-size 1 --smoke-accumulation-steps 2 --run-tag throughput-b1-a2-v1
FLOW_DEBUG_STEPS=20 bash launchers/flow/debug_train.sh zimage dpo \
  --smoke-batch-size 2 --smoke-accumulation-steps 1 --run-tag throughput-b2-a1-v1
```

顺序运行。比较后10步seconds_per_update中位数及pairs_per_second、各GPU实际显存；启动时间另记。
这只测4 pair-views小预算的执行趋势，不能当作正式128 pair-views的已测吞吐。
分组改变可能改变RNG分配，不要求训练轨迹逐位相同。K2/K4需对实际最终方法另测，不从K1外推。
原debug固定数据只有112训练对，不能用于最终生成效果结论。

若B2显存与吞吐均合适，再在准备完正式数据的容器用新的正式tag，例如：

```bash
bash launchers/zimage/01_dpo_k1.sh --batch-size 2 --run-tag flow-h100-b2-v1
```

双卡正式128 pair-views/update下，B2的DPO K1累积32、MVPO K2累积16、K4累积8。
对照方法保持同预算及相同调参机会。不自动增加K、beta、学习率或修改FP32 transformer。

本工具的CPU检查不能替代SLAI真实加载或B2显存测试；本地无法直接访问该SLAI容器。
