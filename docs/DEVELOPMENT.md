# Self Correct DPO 开发与实验约定

这份约定适用于 `self_correct_dpo`，以用户的双服务器工作流为准。更新日期：2026-10-25。

## 两台服务器的职责

- 本机维护单独开发分支，当前为 `dev/self-correct-workflow`，主要编辑代码、整理配置和做轻量 CPU 检查；当前目录不作为正式实验运行目录。
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

训练输入默认轻量抽查：配置检查与清单加载并行，清单仍核对 schema、split/来源及 ready 身份；图片默认8线程抽查8张并覆盖每个 split 的首对，最多14张。已删除全图片 inventory、`.validation.json` 缓存协议、缺缓存自动完整审计及入口重复的训练行检查。实际训练读取时校验即将解码字节的 SHA256；缺 SHA 的旧清单默认报错，不静默读取全部图片，需要显式 full 或重新准备含 SHA 的清单。`scripts/check_training_data.py --mode full --pairs ...` 保留独立完整审计。ACP 训练启动器不再自动执行逐卡 CUDA 自测，`check_deployment.py --gpu` 仅首次部署或排错时手动调用。具体日志与参数见 `command.md`。

## 配置和超参数

配置按四层合成：`configs/defaults.json` → `configs/models/sd15.json` 或 `sdxl.json` → 方法入口 → 命令覆盖。实际加载时方法入口的 `extends` 先读公共和 SD1.5 默认值，`--family sdxl` 再应用 SDXL 默认值。

| 方法入口 | 含义 |
|---|---|
| `configs/dpo.json` | 纯 DPO，K=1，无 rollout |
| `configs/multiview-B.json` | 各个独立 rollout 视图计算 DPO loss 后平均 |
| `configs/multiview-C.json` | 独立 rollout 视图的 margin 平均后计算 DPO loss；默认入口 |
| `configs/multiview-B-gaussian.json` | 无 rollout，普通高斯视图的 DPO loss 平均；K 个视图合批 |
| `configs/multiview-C-gaussian.json` | 无 rollout，普通高斯视图的 margin 平均后计算 loss；K 个视图合批 |

B/C 默认 K=2，`--views 3`、`--views 4` 或其他正整数均可。增大 K 会增加显存和计算量，不自动改变学习率或 batch。在 rollout 入口中仅将 K 改为1仍保留 rollout；普通高斯入口 K=1 则退化为单视图目标。明确选择纯 DPO 使用 `dpo`。

普通高斯入口使用 `training.batch_views=true`，复用同一批 VAE latent 和文本条件；按视图顺序拼接 noisy/timestep/target 以及 SDXL 的嵌套条件，一次参考前向、一次策略前向，再还原 `[B,K,2]` 计算联合 loss。其余已有入口沿用原逐视图路径。`batch_views` 不与 rollout/paired_drift 组合。普通高斯可用 `--set training.batch_views=false` 做逐视图实现对照；两种计算顺序可能产生浮点舍入差异。

显存按每卡 `2×B×K` 张带噪图片衡量，B 是图片对数。双卡普通高斯启动器支持 K=2/4/6/8：SD1.5 K=2 保留 B=2、累积64，K=4/6/8 默认 B=1、累积128，均保持全局256对；SDXL 均从 B=1、累积64、全局128对起步。显存足够可用 `SCD_TRAIN_BATCH_SIZE=2` 在新实验中改 B，启动器自动调整累积并核对冻结配置的全局 batch。默认 balanced 下，每次更新的图片对集合不随 B 的这种拆分改变，但随机数分配和浮点累加可能变化。BF16 和梯度检查点沿用默认设置；参考复用禁用 LoRA 的基座，无第二份 UNet。DDP 不合并各卡显存，大 K 是否可容纳需要远程实测。B 已为1后，增加梯度累积不能继续缩小一次合批前向。启动器只读取准备完成的数据，避免并行任务重复预处理；命令及显存监控方式见 `command.md`。

默认 seed=17、learning_rate=1e-5、LoRA rank/alpha=16、BF16、2000 优化步、每 200 步完整保存、每 1000 步保留 LoRA。SD1.5 默认 beta=5000、分辨率512、全局256对；SDXL 默认 beta=500、分辨率1024、全局128对。

全局 batch = 每卡 batch × 梯度累积 × GPU 数。只修改卡数时自动保持默认全局 batch；显式传入梯度累积时以命令为准。使用 `--print-config` 查看最终配置；使用 `--set training.weight_decay=0.01` 等覆盖其他嵌套参数。实际配置、代码摘要、commit、依赖版本、GPU、随机种子和资源身份记录在每次运行的 `training.json`。

新增 Gaussian 对照入口 `acp_gaussian_fixed_pairs.sh` / `acp_gaussian_fixed_views.sh` 使用用户确认的 SD1.5=384、SDXL=192 基准，支持 K=2/4/6 和2/4/8卡，保持10000步；旧启动器默认值继续保留。前者固定图片对组数，后者固定图片对视图数并将图片对组数除以 K，两组每个模型共用一个匹配新预算的纯 DPO 基线。`make_config.py --budget-unit pairs|pair_views --budget-per-update N` 生成 `training.comparison_budget` 并精确推导累积，不能与 `--effective-batch` 或显式累积混用，不可整除即拒绝。新比较启动器拒绝 CLI 训练覆盖，只放行暂停/恢复；预算、K、卡数和模式写入独立命名及冻结检查。`validate_config` 也核对预算字段与实际配置一致。该入口只支持普通高斯 B/C 及 DPO，不将 rollout 的额外计算混入等视图预算。

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

2026-10-24 新增普通高斯配置、视图合批和双卡 K=2/4/6/8 启动器。本次本地仅使用标准库验证配置生成、方法切换、全局 batch、冻结配置保护及 Python/Bash 语法；没有安装依赖。`tests/test_joint.py` 已补充普通高斯合批与逐视图的误差、梯度、RNG 和 SDXL 嵌套条件对照，但因本机没有 PyTorch 尚未执行。远程 `de9eed3` 的并行预处理代码及四项 real-Parquet 测试原样保留，本次没有重跑其依赖 PyArrow/Pillow 的测试。

2026-10-25 训练前校验优化：本地10项真实小文件校验测试、6项配置测试通过，覆盖并行审计、准备记录/缓存复用、未完成准备拒绝、清单和来源变更、重复 ID / split 泄漏、缺失或损坏图片、训练读取时 SHA256 验证。没有安装 PyTorch/PyArrow/Pillow；实际 GPU 启动耗时和模型训练仍须在 SLAI 验证。改动涉及 `pairs.py`，未完成的旧预处理应保留原 checkout 完成或恢复；已完成的 `ready.json` 数据无需重新准备。

2026-10-25 统一 ACP 入口：`acp_prepare_data.sh <pickapic|hpdv3> <sd15|sdxl|unfiltered>` 独立完成数据准备，已完成输出复用；只有 HPDv3 支持数据导出的 unfiltered。训练入口只读已准备数据，纯 DPO 双卡和普通高斯 K=2/4/6/8 共用 `acp_common.sh`，正式预算统一10000步；SD1.5 纯 DPO 和 K=2 保留每卡B=2、累积64，K=4/6/8 默认B=1、累积128，均为全局256。脚本从自身位置确定 checkout，环境和资源表一致。8卡旧入口保留。本地验证不安装环境，实际 GPU、显存和数据耗时仍须在远程验证。

入口整理验证：7项配置测试、10项数据校验测试通过，Bash语法通过；临时 checkout 中实际生成9种 DPO/高斯双卡配置，并检查冻结配置冲突、缺失配置保护、续训参数、已完成数据复用及 HPDv3 自定义路径转发。此 shell 检查替代了 Conda 激活、部署检查、预处理和训练调用，不代表实际环境或 GPU 执行已经通过。

2026-10-25 SDXL 开发：新增 `acp_sdxl.sh`，将公共启动生命周期扩展到2/4/8卡和 rollout B/C，已有双卡入口仍固定2卡并保留原命名。SDXL 固定1024、每卡默认1对、全局128对，正式预算10000步；沿用每200步完整保存和既有监控/续训。论文依据、研究配置与原论文差异见 `docs/SDXL.md`，运行命令集中于根目录 `command.md`。新增验证聚焦五种方法的配置及真实 tiny SDXL 普通高斯合批/逐视图在梯度检查点下的梯度一致性；完整1024模型的5090/H100训练仍需远程验证。

本轮结果：8项配置测试和7项 SDXL 测试通过；26组启动配置及恢复/冻结配置保护通过，后者替代了 Conda 激活、GPU预检和实际训练。验证环境和测试边界记录在 `docs/SDXL.md`。

2026-10-25 两套预算对照验证：11项配置测试通过，其中新增48组预算组合及非法预算/冲突/篡改拒绝；3项现有 CPU 保存恢复测试通过，并新增视图元数据与恢复后累计数断言。临时 checkout 的60组新启动配置、26组原启动配置及默认值/自动恢复/模式冲突/CLI覆盖保护通过；这部分替代 Conda 激活、GPU部署检查和训练调用。所有 Bash 语法、变更 Python 语法和 diff 空白检查通过。未提交 GPU 作业或实测5090/H100性能。训练代码摘要已改变，旧实验续训继续使用原 checkout，新入口使用新实验目录。

2026-10-25 启动检查精简：删除自动逐卡 CUDA 自测、重复训练行检查和图片校验缓存，配置/数据并行检查，默认最多14张图片并行抽查；不再隐式全量审计。10项真实小文件测试、临时清单/模型元数据的实际 `train.py --check-only`、变更 Python/Bash 语法及 diff 检查通过。保留已有训练预算和续训约束；本轮仅轻量 CPU 验证，未提交 GPU 作业或测量远程启动耗时。

2026-10-26 SLAI 分组脚本：`launchers/sd15/`、`launchers/sdxl/` 各包含8个独立训练入口和1个双卡并行评估入口，预算分别384/192，直接复用已有配置生成和续训。补齐监控面板专用的独立评估配置，默认评估编号01/02的1000步保留快照，不轮询或提交平台任务。临时 checkout 的16组真实 shell/配置生成、自动续训、监控配置继承、两模型双评估进程并行、GPU mask分配和子进程失败返回检查通过；Conda和实际训练/评估调用被替代。19个新shell语法、2个JSON解析和diff检查通过，没有执行GPU任务。逐组命令统一见根目录 `command.md`。

2026-10-26 RTX5090依赖迁移：用户选择官方稳定组合PyTorch2.14.1/cu130、torchvision0.29.1/cu130、NCCL2.30.7，远程驱动为580.95.05；requirements和包元数据同步更新，模型相关API版本保留。远程操作保留已有dpo环境，不提供自动删除环境、隔离NCCL或LD_PRELOAD覆盖；升级前清理cu12/cu13的旧NCCL包，避免共享库路径互相覆盖。Python3.10下pip元数据解析通过，共89个发行包，采用dry-run、PEP658元数据和受限ZIP范围读取，禁止整包下载，未安装任何依赖。11项现有配置测试、依赖声明语法与diff检查通过。用户已有NCCL2.31.2通信通过记录不代表新2.30.7组合已验证；仍需远程版本核对和双卡DPO一步训练/保存，命令见command.md。

2026-10-27 AOSS存储恢复：图片发布与pending事务恢复在硬链接不支持时使用render输出锁保护的重命名，保留已刷新PNG、journal、manifest校验及目标不覆盖检查。训练、监控、评估、render的源码兼容仅放行从b766801基线到本修复的精确SHA配对；不重写旧training/monitor/evaluation身份或checkpoint，另存兼容审计记录。`acp_resume_storage_fix.sh`从training.json恢复原配置与latest，部署至独立checkout以保护正在运行的实验。9项新增CPU测试和3项既有保存/恢复测试通过，覆盖不支持硬链接、发布前后中断恢复、不覆盖图片、非兼容错误、旧版四层身份、200步恢复后的优化器/RNG一致性、1000步快照保留及原200/500评估计划。临时checkout中的真实恢复脚本保留原配置/latest/暂停参数并拒绝训练覆盖，测试替代了Conda激活和训练执行；Bash语法、Python语法和diff检查通过。实际SLAI GPU续训仍须远程执行。

2026-10-30 Stagewise SD1.5入口：新增独立 `launchers/sd15_stagewise/` 四组，固定10000步、192pair/384pair-view，双卡B=2、累积48；旧入口不变。临时checkout中执行12组真实配置生成（四目标×2/4/8卡），确认10000步和累积48/24/12、预算及128prompt基座监控；自动latest续训、阶段边界/评估采样篡改拒绝、已有输出缺冻结配置拒绝、训练CLI覆盖和非法卡数拒绝通过。检查替代了远程Conda激活与实际训练调用，未安装依赖或提交GPU作业；新增5个Bash脚本语法、help及diff空白检查通过。实际5090双卡训练、峰值显存、性能及正式两步保存/恢复仍须在SLAI已分配容器验证。

本轮算法在项目既有 `.venv`（Python3.10.19、PyTorch2.6.0）上通过5项新增CPU检查及22项原配置/联合梯度/时间采样回归。覆盖非中点边界的均匀边际、段内独立时间与噪声、pair共享、RNG重放、四目标与梯度、合批/逐视图一致性和旧Gaussian随机序列保留。四个10000步解析配置与原C-K2固定384视图预算的配置逐项比较，仅 `method` 和新增 `training.stagewise` 不同。

另通过12项既有暂停/恢复与AOSS存储兼容检查；历史版本迁移测试改用明确的original/fixed源码身份fixture，生产兼容白名单保持不变，未批准的训练源码变化仍拒绝续训。独立临时CPU小模型检查调用真实MultiViewBackend采样、误差计算和训练循环：四种目标的日志loss均与独立公式计算一致且产生不同更新；连续4步与第2步暂停后恢复的权重、optimizer、scheduler、阶段指标和各CPU RNG完全一致。这些检查未加载完整SD1.5模型，不代表GPU训练或显存验证。
