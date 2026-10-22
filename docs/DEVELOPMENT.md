# Self Correct DPO 开发与实验约定

这份约定适用于 `self_correct_dpo`，以用户的双服务器工作流为准。更新日期：2026-10-23。

## 两台服务器的职责

- 本服务器维护单独开发分支。本次分支为 `dev/self-correct-workflow`，主要编辑代码、整理配置和做轻量 CPU 检查；当前目录不作为正式实验运行目录。
- 远程服务器使用 GitHub `main`，主要在已分配的 2 / 4 / 8 张 RTX 5090 上快速验证 SD1.5、SDXL 算法；按需要使用 H100。
- 新算法先做最小实现和小规模验证。确认可运行或有收益后，再将对应代码提交合入 GitHub `main`，远程拉取指定 commit。
- 正在运行或准备续训的实验固定代码、依赖、资源路径和超参数。先结束当前实验再更新该 checkout；并行比较不同版本可使用独立 checkout/worktree。
- Git 只管理源码、配置、文档。模型、数据、日志、checkpoint、W&B 密钥不进入仓库。每次实验使用独立输出目录，不覆盖旧结果。

## 开发风格与测试

优先简洁的研究代码：入口明确、按功能分组、默认值容易找到、命令可覆盖。不要为一个算法引入服务层、插件框架或复杂配置系统。

只保留能验证实际风险的少量检查：配置能否解析、B/C/K 的目标和梯度、保存及续训是否一致。复用现有测试，避免大量 mock、机械地照抄实现的测试或为可逆的小改动新建测试体系。正常修改不必反复跑全部测试。

新的噪声算法后续按实验需求再整理。本次只整理现有实现的导入位置，保留 `algorithms/paired_drift.py` 及原有实验修改，不将其作为默认入口。

## 资源路径只有一个入口

统一维护 `configs/resources.json`。远程机器可用同目录的 `resources.local.json` 覆盖少数值；该文件不进入 Git。优先级为：环境变量 > local 配置 > 资源配置。也可通过 `SCD_RESOURCES` 或训练命令的 `--resources` 选择另一份资源表。

| 资源 | 配置键 | 当前约定 |
|---|---|---|
| SD1.5 | `SD15_MODEL` | 沿用先前下载脚本路径 `${MODELS_ROOT}/stable_diffusion/sd15` |
| SDXL / 可选 VAE | `SDXL_MODEL` / `SDXL_VAE` | 预留原规划路径；用户尚未下载 SDXL |
| SD3 | `SD3_MODEL` | 预留 `${MODELS_ROOT}/stable_diffusion/sd3`；用户尚未下载 |
| SD3.5 | `SD35_MODEL` | 保留现有下载工具的资源项，独立于 SD3 |
| Pick-a-Pic 原始文件 | `PICKAPIC_RAW` | `${DATASETS_ROOT}/pickapicv2/data` |
| Pick-a-Pic 图片和清单 | `PICKAPIC_IMAGES` / `PICKAPIC_SD15_PREPARED` / `PICKAPIC_SDXL_PREPARED` | 放在同一数据集目录下；这些是规划的处理输出位置，不代表已生成 |
| HPDv3 数据集 | `HPDV3_RAW` / `HPDV3_SD3_PREPARED` | 为后续 SD3 训练预留；用户尚未下载 HPDv3 |
| 评分模型 | `PICKSCORE_*` / `HPSV2_*` / `HPSV3_*` / `IMAGE_REWARD_*` 等 | 沿用先前下载脚本的分类目录 |
| 固定评估面板 | `BENCHMARKS_ROOT` / `MONITOR_PROMPTS_SD15` / `MONITOR_PROMPTS_SDXL` 等 | 按模型系列保存处理后的 JSONL |
| 训练输出 | `RUNS_ROOT` | `/mnt/aoss/xietianyu/self_correct/runs`，须确认属于远程持久存储 |

按用户 2026-10-23 最终确认的安排，资源父目录 `WORKSPACE` 为 `/mnt/aoss/xietianyu`，`MODELS_ROOT` 为其下的 `models`，`DATASETS_ROOT` 为其下的 `datasets`，已去掉此前重复的目录层级。模型和数据集各自保留原子目录结构，下载、数据准备、训练和评估均使用同一份资源表；远程代码目录为 `/mnt/afs/L202500030/xietianyu/workspace/DPO`。`RUNS_ROOT` 继续使用表中的独立输出地址。本机负责代码与轻量检查；远程实际文件由部署检查和 GPU 小跑核验，训练不会自动下载模型。

环境变量和 `resources.local.json` 的优先级高于资源表。迁移后同步清理或更新旧覆盖项；之前在 shell 中导出过资源变量时，使用新终端加载迁移后的配置。

商汤大装置首次部署按根目录 `command.md` 的顺序，先验证纯 DPO 的实际训练、暂停及恢复，再开始正式基线和多视图比较。`scripts/check_deployment.py` 默认只检查 CPU 元数据，只有 `--gpu` 才实际运行 CUDA；数据处理只需要 CPU。平台申请及容器挂载通过用户实际使用的商汤入口完成，不套用本机 Slurm 命令。

HPDv3 是偏好数据集，HPSv3 是评分模型，分别记录。当前训练入口支持 **SD1.5/SDXL**；预留 SD3 路径及下载选项不等于已实现 SD3 训练。SDXL 下载后从 `.scd-download.json` 自动读取 revision；手动下载时用 `--model-revision` 指明可追溯的版本。

原始 Parquet 目录不能直接用作 `--data_path`：训练需要处理后的 `pairs.jsonl`。大规模处理放在远程已分配资源上，命令见根目录 `command.md`。

## 配置和超参数

配置按四层合成：`configs/defaults.json` → `configs/models/sd15.json` 或 `sdxl.json` → 三种方法入口 → 命令覆盖。实际加载时方法入口的 `extends` 先读公共和 SD1.5 默认值，`--family sdxl` 再应用 SDXL 默认值。

| 方法入口 | 含义 |
|---|---|
| `configs/dpo.json` | 纯 DPO，K=1，无 rollout |
| `configs/multiview-B.json` | 各个独立 rollout 视图计算 DPO loss 后平均 |
| `configs/multiview-C.json` | 独立 rollout 视图的 margin 平均后计算 DPO loss；默认入口 |

B/C 默认 K=2，`--views 3`、`--views 4` 或其他正整数均可。增大 K 会增加显存和计算量，不自动改变学习率或 batch。将多视图 K 改为 1 仍保留 rollout，纯 DPO 需选择 `dpo`。

默认 seed=17、learning_rate=1e-5、LoRA rank/alpha=16、BF16、2000 优化步、每 200 步完整保存、每 1000 步保留 LoRA。SD1.5 默认 beta=5000、分辨率512、全局256对；SDXL 默认 beta=500、分辨率1024、全局128对。

全局 batch = 每卡 batch × 梯度累积 × GPU 数。只修改卡数时自动保持默认全局 batch；显式传入梯度累积时以命令为准。使用 `--print-config` 查看最终配置；使用 `--set training.weight_decay=0.01` 等覆盖其他嵌套参数。实际配置、代码摘要、commit、依赖版本、GPU、随机种子和资源身份记录在每次运行的 `training.json`。

## Accelerate 和日志

- `python train.py ... --world-size N` 运行一个协调进程，默认用 Accelerate 启动各段 DDP 训练，在训练进程退出、显存释放后串行评估固定面板。参数覆盖先写成独立配置快照，再传给子进程。
- 直接 `accelerate launch ... train.py ... --no-evaluation` 也可运行 DDP；卡数从启动环境读取。需要自动评估时使用上面的协调入口，避免嵌套分布式启动。只支持单机，每卡一个进程。
- Accelerate 负责启动进程，训练精度与完整 checkpoint 由本项目实现管理；没有引入 DeepSpeed/FSDP 或替换为 `Accelerator.save_state`。
- TensorBoard 默认开启。`--report_to wandb` 或 `all` 可启用 W&B；先安装可选依赖 `pip install '.[tracking]'`。W&B 默认 offline，在线上报需 `--wandb-mode online` 并在远程自行登录。
- 日志包含 loss、margin、梯度范数、学习率、数据曝光、每步耗时、显存峰值及监控 PickScore。JSONL 总会保留。B/C 的训练 loss 数值不宜直接作为效果优劣依据，应比较相同 prompt、seed、生成设置下的指标。
- TensorBoard 续训会清理 checkpoint 后的旧事件。W&B 在线使用稳定 run ID 和 `optimizer_step` 横轴；离线每次启动会产生独立会话，需要自行同步，可能含被回滚区间的记录。

启动参数依据项目固定版本的 [Accelerate 0.34.2 官方说明](https://huggingface.co/docs/accelerate/v0.34.2/en/package_reference/cli)。

## 被驱逐与续训

1. 默认每 **200 个优化步**原子保存完整状态：LoRA、optimizer、LR scheduler、步数、数据曝光统计，以及每个 rank 的 Python/NumPy/PyTorch CPU/CUDA RNG。
2. 完整恢复槽使用独立临时目录，写完并校验后提交。`latest.json` 指向最近完整槽；先提交新指针再清理旧槽。恢复时先处理保存事务，再读取 latest。
3. 每 1000 步和最终步额外保留 LoRA。LoRA 快照只有权重，完整续训必须使用 `--resume_from_checkpoint latest` 或完整恢复目录。
4. TERM/INT 到来时，在完成当前优化步后尽量保存。断电、SIGKILL、突然驱逐无法保证保存当前步；恢复到最近完整提交点，常规最多重算上次保存之后不足 200 步的工作。若信号在首个 checkpoint 前且没有保存成功，则从头开始。
5. 续训保持相同 GPU 数、配置、代码、依赖和资源路径/身份。2/4/8 卡用于分别启动实验；当前不支持同一实验中途改变卡数。不要拉取新算法后直接续旧实验。
6. 完整 checkpoint 和输出必须在远程持久挂载上；容器本地临时盘被回收后无法靠代码恢复。调度器重新申请到资源后重新执行原命令并加 `--resume_from_checkpoint latest`；本工具不自动申请机器或重建容器。
7. 调整训练预算使用 `--extend-from` + 新输出目录（仅增加 max_steps，constant LR）；比较新算法使用新实验，不借续训修改算法。

## 本集群边界

遵守账号级 AGENTS 约定：只操作本人文件、环境、进程和 Slurm 作业；登录节点只做编辑与轻量 CPU 检查。GPU 或大规模处理进入 Slurm，先核实资源并按用户授权提交。预计向 `/irip` 写入超过 1GB 前检查磁盘；下载前检查共享模型/数据缓存，使用独立环境，不覆盖实验产物。

## 本次整理的验证范围

已使用项目现有 CPU 环境检查 B/C 的 K=1/2/3/4 梯度、SDXL 配置和小模型往返、200步保存及续训；额外短时检查了 Accelerate 两进程 CPU 启动、指针提交/清理中断后的恢复、TensorBoard 事件读写，以及各方法的命令覆盖。不新增复杂测试框架。

这些检查不代表 5090/H100 的实际训练、显存或性能已经验证。本机未安装 W&B 可选依赖，尚未测试其真实在线上报。远程首次使用新配置时，先按 `command.md` 进行两步 GPU smoke，再启动正式实验。
