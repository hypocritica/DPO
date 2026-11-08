2026-11-08验证：连续训练/独立评估与B4探针共30项CPU回归通过；同步修正旧storage兼容测试中遗漏的base_model/zimage_compat身份fixture。未执行完整模型GPU训练或评估。
# Self Correct DPO 开发与实验约定

2026-11-08 按用户确认新增ZImage单H100 B4探针probe_zimage_b4_h100.sh/probe_zimage_b4.py，复用原SD35探针的配置/生产trainer/报告流程，保持原SD35入口默认不变。ZImage默认20步、B4/累积1/K1/1024，FP32 transformer+BF16 autocast、关闭offload、VAE chunk1与正式双H100一致；保留既有验证报告、独立tag/输出及OOM报告，只有最终保存、不自动启动正式实验。CPU检查覆盖实际新shell入口、正式精度/放置一致性及原B2/B4/OOM行为；未在H100实际加载或提交任务。

2026-11-08 用户要求正式训练默认连续、另一GPU独立评估且不影响排队任务：在独立flow-wandb checkout的feat/flow-detached-eval-20261108实现paper-v2默认evaluation.enabled=false但保留面板/采样/八点LoRA保存，输出名增加trainonly；inline-evaluation显式恢复旧流程。新evaluate_flow.py/evaluate.sh按原子发布的永久step目录评估/等待，校验run及资产身份，复用基座pipeline和PickScore、切换小PEFT adapter，独立事务/曲线/W&B身份，保留渲染与评分恢复语义；不依赖滚动latest、不锁训练进程。render/metrics仅增加内部可注入入口，普通CLI默认行为保持。真实tiny PEFT与CPU生产保存/渲染评分事务检查通过，SLAI完整GPU评估未验证。检查时65803–65806均PENDING且引用原stagewise目录，其280份源码/冻结配置指纹在开发期间保持一致，未提交/取消/修改作业。另按用户B2截图确认B2常驻6步passed，给原探针增加可选batch-size4；不修改正式batch默认。部署与限制见FLOW_DETACHED_EVAL.md。

2026-11-08 按用户要求新增单5090 SD3.5 B2常驻短测：probe_sd35_b2_5090.sh自动激活环境，probe_sd35_b2.py使用生产trainer、单GPU/B2/累积1/1024/BF16/offload=false/VAE chunk1/梯度检查点，加载一次并完成6次AdamW更新，仅最终保存，独立tag与配置/输出。报告allocated/reserved峰值、设备容量、总耗时与去掉前2步的速度；OOM保留traceback和报告并非零退出，纯文本PROBE数值避免平台导出丢字段。2项CPU检查覆盖真实shell配置、拒绝覆盖及OOM报告路径（CUDA为fixture），未在5090验证。沿用feat/flow-benchmark-slai-20261108发布；不修改训练算法源码或既有实验。

2026-11-08 按用户要求增加SLAI连续任务benchmark.sh/benchmark_flow.py：自动生成独立配置，双rank加载两次，再测K1/K4 C的B1A2/B2A1各20步，线程固定4/rank；支持--part拆分、成功跳过、失败新attempt、子进程失败继续和取消停止。按生产日志汇总后10步速度、pair-view吞吐及两个rank allocated峰值，加载分项/RSS/容器配额保留原始JSONL。CPU检查覆盖真实shell配置生成、失败隔离和取消/只读预览；整合远程修复后20项相关检查通过（含preflight回归）；本地未操作SLAI GPU。发布分支feat/flow-benchmark-slai-20261108，资源对照为双卡8CPU/128GiB与16CPU/256GiB；缓存、卡型及CPU/RAM共同变化的归因限制见FLOW_STARTUP_PROFILE.md。同步远程最新Z-Image padding修复后再发布，不更改本地正在运行的stagewise checkout。

2026-11-08 SLAI双80GB/8vCPU/128GiB启动诊断：新增独立profile_flow_startup.py，默认只读模型header与容器配额；显式--load调用原PreferenceBackend并临时计时组件加载、GPU搬运、DDP，支持逐rank加载/CPU线程对照。单进程可选每分片至多64MiB只读探针，与加载测量互斥以避免预热干扰。逐rank新JSONL拒绝覆盖，不修改生产源码或checkpoint。debug新增可选smoke microbatch/累积，默认1/1和旧输出名保持；编号正式入口放行显式batch-size并自动保持原pair-view预算。11项相关CPU测试通过（debug、stagewise、计时工具），覆盖B1A2/B2A1等预算、正式K1/K2/K4累积32/16/8、加载包装失败后恢复与报告不覆盖；Bash/Python语法和diff检查通过。本地真实Z-Image header确认2 transformer分片11.46GiB、3 encoder分片7.49GiB，均BF16磁盘权重；未加载完整GPU模型、未操作SLAI或提交Slurm，根因与B2收益待远程测量。命令见FLOW_STARTUP_PROFILE.md。

2026-11-08 发布前复核：13项preflight/receipt/真实tiny ZImage CPU检查在本地排队目录及远程分支7154f69的独立发布checkout均通过，更新monitor preflight测试fixture以包含zimage_compat.py源码身份。四个本地待运行配置及实际环境锁、模型/数据凭据绑定通过。65803/65804的两份修复前隐藏preflight.pkl在取得launcher/preflight文件锁并确认作业仍PENDING、无训练输出后改名保留；启动时重建缓存，65805/65806本来没有缓存。未取消、重提或修改Slurm作业；完整GPU训练未验证。SLAI更新后使用新run-tag，避免复用旧缓存和身份。

2026-11-08 ZImage混合精度padding修复：保留FP32 transformer/LoRA、BF16 autocast与FP32 loss，在实例上回移Diffusers v0.40.0的torch.where padding；保留0.37.1其余sequence/mask语义，训练和评估加载共同应用，>=0.40.0保留原生方法。新增zimage_compat.py纳入训练、监控、评估/生成源码身份，不添加旧checkpoint恢复白名单。实际本地Flow环境Python3.10.19/torch2.6.0/diffusers0.37.1/peft0.17.1/transformers4.57.1下，3项真实tiny ZImage CPU回归通过（原版错误复现、三种权重/激活精度、reference/policy/EMA及梯度检查点反向、padding值/梯度/掩码与FP32一致性）；4项既有Flow回归和1项暂停/恢复测试通过。7份变更Python文件AST与git diff --check通过。未更新依赖、申请/取消作业或同步SLAI容器；完整GPU训练仍未验证。精度说明见FLOW.md，CPU验证命令见command.md。

2026-11-07 根据SLAI真实日志修复debug_train.sh遗漏--run-label：模型和数据预检已过，训练前require_gpu_allocation因standalone标签空而拒绝。启动器提供family/algorithm/profile运行标签，保留分配检查；运行参数不修改冻结配置，不需清理旧产物。2项相关CPU集成检查通过，新增真实shell→run_flow.main→训练命令测试（仅拦截GPU子进程并调用真实require_gpu_allocation，非config-only），覆盖SD3.5和ZImage标签及25步暂停传递；未在远程GPU执行修复后训练。

2026-11-07 调整预检快照Only 0/128诊断：旧默认前20万条可能漏掉后续完整对，改为max_scan=0流式查到目标或EOF。先索引已完成图片名称（忽略partial/incomplete），减少远程缺失文件stat；每1万条/15秒输出计数，输出相邻availability.json记录范围、过滤、winner_only/loser_only/neither/both及路径示例。新增真实20万条前缀回归：限制前缀返回0，默认成功找到128对；一侧缺图及partial准确统计。4项debug CPU检查通过，来源/正式输出不修改。真实SLAI此次0对原因仍需新报告确认，不能保证未完成解压中必然已有128个完整对；未提交GPU任务或重跑全量准备。

2026-11-07 按用户更正将远程ZImage编号入口默认改为双H100，SD3.5仍四5090。新增prepare_flow_debug_data.py：从正在提取的正式图片中复制128对独立真实偏好快照，不读分卷，限256MiB，源数据/正式目录不写入；原始行号/标签保留，官方test caption/UID排除，缺完整test image hash审计故标记ready_for_paper=false，不生成正式passed数据凭据。debug_train.sh默认50步/1024/B1/累积1，离线W&B独立项目，25步暂停可自动latest恢复至50；无生成评分。run_flow.py新增--smoke-steps，原smoke两步及正式3200步保持。28项相关CPU回归通过（新快照/真实shell配置、旧flow/SFT/stagewise/receipts/数据）；额外实际CPU tiny PEFT生产训练50步与25→50恢复的LoRA及算法EMA逐位一致。未运行远程真实GPU或提交作业，命令见FLOW_DEBUG.md。

2026-11-07 根据SLAI实际任务诊断修正mtime纳秒精度误拒绝：分卷大小10737418240一致，mtime_ns从1793982938732000000变为1793982938000000000。仅兼容大小相同、整秒相同且一侧精确为整秒的截断；跨秒变化、非截断小数变化和大小变化仍拒绝。注释出现精度差异时重新核对既有SHA256；分卷不重复全量哈希，不宣称内容认证。日志/extraction/validation报告保留精度证据，原指纹和revision不变。21项相关CPU检查通过，包含实际数字复现、注释同大小改写拒绝及四文件精度变化后的五阶段流程/重复复用；未在SLAI真实全量数据上执行。

2026-11-07 增强HPDv3来源stat失败诊断：确定性文件顺序，错误携带expected/observed/changed_fields，准备入口输出实际任务hostname/Python/raw/module上下文；inspect_hpdv3_source.py --diagnose只读JSON检查，不创建或更新来源记录。保持mtime/大小严格检查。20项相关CPU测试通过，覆盖只改mtime拒绝、任务内错误JSON和原指纹逐字节保留。远程报错与开发机变化0的实际差异尚待任务日志确认，不宣称修复数据变化。

2026-11-07 修正SLAI读取train.json时SIGKILL/137的高内存路径：提取和metadata筛选按对象流式读取，提取仅累计图片路径；完整准备每256条校验并丢弃排除的原始注释。无新依赖，不更改筛选、源行号、标签或seed17组件划分。19项相关CPU测试通过；300条跨批次fixture的新旧pairs/excluded/ready逐字节一致。6.76MB/12001条源数据的解析+筛选tracemalloc峰值338702字节，该数字不代表完整prepare RAM；完整prepare仍保留合格元数据和组件结构。未运行远程全量数据或改动本地实验作业。

2026-11-07 HPDv3准备入口兼容HF原生下载目录：不再要求ModelScope流程的.download-manifest.json；支持.scd-download.json，缺下载记录时生成状态为inventoried的本地来源指纹，明确不宣称download complete/upstream revision verified。
prepare_slai_data.sh统一执行source/extract/prepare/audit/panel并显示阶段、日志与失败行号。提取开始前显示大注释读取阶段，来源指纹只首次哈希注释；原始分卷不重复哈希，所需图片仍经真实提取和准备验证。
源码在独立flow-wandb checkout维护，不修改本地正在使用的stagewise checkout或作业。
17项相关CPU检查通过，包括无下载标记的完整五阶段流程、重复复用、来源变更拒绝、缺卷/截断拒绝、既有数据凭据及预算回归；实际ModelScope记录只读兼容检查通过。未下载、未在真实全量数据上重跑准备、未改变GPU作业。

2026-11-07 W&B远程补齐在独立flow-wandb checkout完成，保留本地65803–65806使用的stagewise源码和配置。
同步本地未发布的offline入口及canonical面板metadata修复，新增旧面板补齐工具（新文件、保持prompt/顺序）。
新tracker每点评估记录前8张图片；SDK固定0.18.7。同步工具只上传已结束且稳定的离线段，等待基座评估后排序，跳过活动段、拒绝history重叠与目的地变更。
15项CPU/真实离线测试及两组真实编号入口配置冻结通过，未登录W&B、上传云端、安装环境或改变Slurm作业。复核摘要见FLOW_WANDB_VALIDATION.json。

2026-11-07 本地Flow比较改为DPO K1和Stagewise K2 C（65803/65804），用户授权取消旧65766/65767/65770。
两组使用新checkout，128 pair-views/update、3200步、2A800；数据审计65800和两组输入检查65802通过。
paper-v2默认W&B offline与TensorBoard，新增project/group/name/mode选项，统一mvpo-flow项目与按模型/预算/seed分组。
offline记录显式history step，训练/评估复用run ID；sync_flow_wandb.py按history起始步排序后逐段append，处理首段训练后才产生base评估的顺序。
修正prepare_flow_monitor输出缺metadata/canonical-only标记导致评估loader拒绝的问题，本地另存v2面板，旧文本和顺序不变。
9项相关CPU/真实离线记录检查通过（不上传），GPU训练仍以作业实际日志为准。命令与SLAI K4比较见docs/FLOW_WANDB.md。

2026-11-07 HPDv3本地准备加速：提取采用单路gzip读取、1MiB缓冲和4线程有界写入（最多64MiB、16个任务），每个目录只初始化一次，输出加锁，全部写入成功后原子生成extraction.json；已有同大小图片和旧日志保留。HPDv3完整准备对唯一图片分批并行校验，一次读取完成PIL verify与SHA256，原始标签、顺序、筛选和seed17划分不变。3项CPU检查通过，覆盖串并行pairs/excluded逐字节一致、跨分片gzip、旧图片复用、partial恢复、独占锁与写入失败不生成完成凭据。计算作业65773因继承Conda环境的deactivate hook遇到nounset失败；仅在Conda激活期间关闭nounset后，65774以4CPU/16GB/无GPU/4天启动在compute-1。确认旧manager提取及cat/释放控制进程退出，再开启handoff-65774.ready；由新作业完成准备并释放既有65766/65767，65770依赖保持原样。此处不代表GPU训练已启动。

2026-11-03 双H100独立入口：按用户要求新增`launchers/sdxl_h100/`的DMPO K1、DMPO K6 Stage-C、DSPO K1三个独立训练脚本，自动加载原ACP Conda及资源路径。固定双卡B1、192 pair-views，K1累积96/K6累积16，均10000步；跳过smoke、部署自检、启动图片抽查及基座生成评估，保留定期生成评估。`data.validation=none`仅关闭启动图片读取，清单/ready身份和消费时SHA仍保留，默认auto不变。运行入口增加TERM/INT转发；专用启动器恢复checkpoint事务，自动latest续训、补完最终评估、跳过完成实验；无完整状态的中断尝试保留后重新开始。原子checkpoint、200步完整保存、1000步LoRA及恢复身份保护继续沿用；新配置使用独立H100实验名。更改训练来源身份的源码后，旧实验须继续保留原checkout恢复。

本轮37项CPU检查通过（validation11、H100入口3、config19、stage_pause3、baseline恢复1）。覆盖三配方预算与冻结保护、无启动图片读取、消费时损坏拒绝、latest指针丢失恢复、首存前断电产物保留、200步恢复/1000步LoRA及目标梯度恢复。两rank CPU/Gloo生产训练流程验证关闭base后在评估中断处恢复，调用顺序为1/1/2/3且保存两个rank的RNG；生成和评分使用fixture，不代表真实PickScore/GPU。另验证TERM转发及75退出码，Bash语法/help、Python语法和diff检查通过。未安装依赖、提交GPU作业、推送main或部署远程。

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
| SD3.5 / Z-Image Base | `SD35_MODEL` / `ZIMAGE_MODEL` | 本地已有模型可复用；远程以resources.local.json覆盖，独立于旧SD3 |
| Pick-a-Pic 原始文件 | `PICKAPIC_RAW` | `${DATASETS_ROOT}/pickapicv2/data` |
| Pick-a-Pic 图片和清单 | `PICKAPIC_IMAGES` / `PICKAPIC_SD15_PREPARED` / `PICKAPIC_SDXL_PREPARED` | 放在同一数据集目录下；这些是规划的处理输出位置，不代表已生成 |
| HPDv3 数据集 | `HPDV3_RAW` / `HPDV3_SUB_PREPARED` | 本地已有原始数据；flow子集准备与审计见FLOW.md，处理输出需单独生成 |
| 评分模型 | `PICKSCORE_*` / `HPSV2_*` / `HPSV3_*` / `IMAGE_REWARD_*` 等 | 沿用先前下载脚本的分类目录 |
| 固定评估面板 | `BENCHMARKS_ROOT` / `MONITOR_PROMPTS_SD15` / `MONITOR_PROMPTS_SDXL` 等 | 按模型系列保存处理后的 JSONL |
| 训练输出 | `RUNS_ROOT` | `/mnt/aoss/xietianyu/self_correct/runs`，须确认属于远程持久存储 |

按用户 2026-10-23 最终确认的安排，资源父目录 `WORKSPACE` 为 `/mnt/aoss/xietianyu`，`MODELS_ROOT` 为其下的 `models`，`DATASETS_ROOT` 为其下的 `datasets`，已去掉此前重复的目录层级。模型和数据集各自保留原子目录结构，下载、数据准备、训练和评估均使用同一份资源表；远程代码目录为 `/mnt/afs/L202500030/xietianyu/workspace/DPO`。`RUNS_ROOT` 继续使用表中的独立输出地址。本机负责代码与轻量检查；远程实际文件由部署检查和 GPU 小跑核验，训练不会自动下载模型。

环境变量和 `resources.local.json` 的优先级高于资源表。迁移后同步清理或更新旧覆盖项；之前在 shell 中导出过资源变量时，使用新终端加载迁移后的配置。

商汤大装置首次部署按根目录 `command.md` 的顺序，先验证纯 DPO 的实际训练、暂停及恢复，再开始正式基线和多视图比较。`scripts/check_deployment.py` 默认只检查 CPU 元数据，只有 `--gpu` 才实际运行 CUDA；数据处理只需要 CPU。平台申请及容器挂载通过用户实际使用的商汤入口完成，不套用本机 Slurm 命令。

HPDv3 是偏好数据集，HPSv3 是评分模型，分别记录。当前训练入口保留 **SD1.5/SDXL**，新增 **SD3.5 Medium/Z-Image Base** 的flow入口见[FLOW.md](FLOW.md)；预留SD3路径不等于已实现旧SD3训练。SDXL 下载后从 `.scd-download.json` 自动读取 revision；手动下载时用 `--model-revision` 指明可追溯的版本。

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

2026-11-02 新增目标对比：`training.objective` 可选SFT、KTO、MaPO、DMPO，`method`继续控制视图构造。单视图基线和DMPO组合的原始来源、参数尺度、pair-derived KTO边界见 [COMPARISONS.md](COMPARISONS.md)，入口为 `scripts/compare_objectives.py`，命令仍集中在 `command.md`。默认DPO/rollout设置保留；新的objective源码纳入恢复身份，不放宽旧实验的源码一致性要求。

本轮43项CPU测试通过：objectives 5、baseline_training 4、config 16、joint 6、stage_pause 3、storage_compat 9。覆盖公式/梯度、SFT仅winner、MaPO零reference、DMPO K2/K6及Gaussian/rollout B/C和stagewise采样、四目标暂停恢复和K6恢复一致性、费用计数、配置冻结和旧存储兼容。另通过43组新入口配置生成及非法配置/冻结冲突检查；两rank CPU/Gloo的真实训练一步确认KTO与DMPO梯度及Adam一阶动量和全局拼接batch一致，KTO先跨rank求均值再clamp。数学测试使用已有 `self_correct_dpo_slai23_cpu`（Python3.10.22/PyTorch2.3.0+cpu），后端与恢复使用项目上级 `.venv`（Python3.10.19/PyTorch2.6.0）。未安装依赖、提交GPU作业、推送或部署远程；这些小模型检查不代表真实SD1.5/SDXL GPU训练、显存或生成效果已验证。

另通过12项既有暂停/恢复与AOSS存储兼容检查；历史版本迁移测试改用明确的original/fixed源码身份fixture，生产兼容白名单保持不变，未批准的训练源码变化仍拒绝续训。独立临时CPU小模型检查调用真实MultiViewBackend采样、误差计算和训练循环：四种目标的日志loss均与独立公式计算一致且产生不同更新；连续4步与第2步暂停后恢复的权重、optimizer、scheduler、阶段指标和各CPU RNG完全一致。这些检查未加载完整SD1.5模型，不代表GPU训练或显存验证。

2026-10-30 发布前复核：39项CPU测试通过（stagewise 5、joint 6、config 11、timestep 5、pause 3、storage 9）；12组stagewise启动配置及冻结/自动恢复保护复核通过。四组正式配置与旧C-K2固定384视图预算配置相比，仅method和training.stagewise不同；四组smoke及benchmark配置均可生成，分别保持既有2步/100步且关闭评估的语义。另对四种目标各执行真实1000步CPU小模型训练循环和监控协调器：确认基座/200/500/1000评估调用、200/400/500/600/800/1000完整状态保存、1000步独立LoRA保留，以及500步评分中断后从latest继续并补完评分。该检查替代模型组件、图像输入、评分资产及评分器，未运行真实生成、PickScore或GPU；环境为既有Python3.10.19/PyTorch2.6.0。检查记录在本地logs/stagewise-schedule-review-20261030.json。原SD1.5/SDXL启动脚本、公共ACP脚本和默认配置相对fcbfb37未修改；command.md补充独立smoke流程，并将旧AOSS续训部署固定到fcbfb37，防止误用含新算法的分支版本。


2026-11-02 K=4补齐：统一目标对比入口支持Gaussian B/C的K=2/4/6（SD1.5/SDXL、DPO/DMPO），SD1.5 stagewise入口也允许K=2/4/6，默认仍K=2。K4默认每卡1对，沿用384/192的pair或pair-view预算。22组K4配置/入口检查、3组非法多视图基线拒绝以及4项既有后端/恢复CPU测试通过；后端组合测试已覆盖K2/4/6。未执行GPU训练。8000步与图片/pair计数关系见COMPARISONS.md。

2026-11-02 DSPO与stagewise K4/K6：新增显式DSPO目标/配置，SD1.5采用作者score-residual配方，SDXL采用作者阈值配方；仅支持epsilon prediction。B平均逐视图loss，C共享平均margin的gate并保留各视图残差，支持Gaussian、detached rollout以及stagewise段内采样。SDXL阈值按每view跨rank归约，属于已注明的分布式适配。stagewise继续每pair先抽同一个阶段、K个view均在该段独立采样，没有改为低/高各K/2；原DPO phase/constant/reverse聚合也覆盖K4/6。

本轮43项CPU测试通过（objectives 8、baseline_training 6、config 18、joint 6、stagewise 5），另通过34组DSPO入口配置、冻结保护和两条phase K4/K6文档命令。覆盖直接向量公式的loss/梯度等价、K1/4/6、B/C、reference冻结、D=0初始化可训练、合批/逐视图一致、段内独立性及均匀边际、K6暂停恢复和费用计数。两rank CPU/Gloo生产训练测试验证SDXL逐view阈值同步：loss=0.12434701994、Adam exp_avg=0.00406438252，与拼接global batch一致。仅轻量CPU验证；未执行真实SD1.5/SDXL GPU训练、安装依赖、推送或部署。命令和方法定义见command.md与COMPARISONS.md。

2026-11-02 SDXL Stagewise-C对比入口补齐：移除 `compare_objectives.py` 中仅允许SD1.5的入口限制，开放SDXL的DMPO/DSPO及匹配采样的DPO对照，支持K2/4/6、2/4/8卡和固定pair/pair-view两种预算。复用已有模型预设和后端；SDXL仍为1024分辨率，DMPO beta=6000，DSPO为official_sdxl、beta=2000、阈值.1。每pair共享阶段、段内独立t/噪声及C聚合不变，不新增Huber组合。

本轮44项CPU测试通过（config 19、baseline_training 6、objectives 8、stagewise 5、joint 6），覆盖SDXL新统一入口、两种预算、两步smoke配置、冻结冲突拒绝，以及SDXL DSPO stagewise K1/2/4/6的合批/逐视图loss、梯度和RNG一致性。另执行42组SDXL DMPO/DSPO统一入口配置生成检查（两目标×K2/4/6×2/4/8卡×两预算的36组正式配置，及两目标×2/4/8卡的6组K6 smoke配置），全部通过；配置检查没有启动实际训练。沿用已有项目CPU环境，不安装依赖、不提交GPU作业、不推送或部署；真实1024模型GPU训练及显存仍需SLAI容器smoke验证。运行命令见command.md的“SDXL：DMPO / DSPO + Stagewise-C”。

2026-11-02 五组SD1.5基线交付：`launchers/sd15_baselines/01_kto.sh`、`02_sft.sh`、`03_mapo.sh`、`04_dmpo.sh`、`05_dspo.sh`各运行一个双5090实验。复用ACP环境加载和日志、统一objective入口及冻结/恢复流程，固定K1、B2、累积96、全局384pair、10000步；每组独立命名，支持独立两步smoke、CPU检查和暂停/自动latest续训，拒绝训练CLI覆盖。运行命令和独立checkout部署见command.md首节；发布分支为 `exp/sd15-baselines-20261102`。

发布前56项CPU测试通过（config19、objectives8、baseline_training6、joint6、stagewise5、stage_pause3、storage_compat9）。另在临时checkout执行49项真实shell/配置生命周期检查，替代远程Conda激活、日志和实际GPU训练调用，验证五组正式/独立smoke配置、固定双卡/B2/384预算、自动latest、显式恢复参数、配置变更拒绝、已有输出缺配置拒绝、smoke/正式同名拒绝、训练CLI覆盖拒绝及未完成数据准备拒绝。6个新Bash脚本语法和help、diff空白检查通过；未安装依赖、运行完整模型或提交GPU任务，5090实际训练仍需用户容器验证。

2026-11-03 最终K6 Stage对比入口：新增`launchers/final_stagewise.sh`与`sd15_final/`、`sdxl_final/`各五个包装脚本，分别运行DSPO/DMPO K1及DSPO/DMPO/DPO K6 Stage-C。SD1.5固定双RTX5090、384 pair_views；SDXL固定四RTX5090、192 pair_views。K1每步384/192 pairs，K6每步64/32 pairs，均10000步。复用已有compare配置冻结、独立smoke及自动恢复，不修改算法、默认配置或旧启动器。Stage沿用每pair随机选段后段内六视图，没有Huber/phase或3+3改动。独立发布分支`exp/stagewise-k6-20261103`，命令集中在command.md首节，论文表格与消融安排记录于COMPARISONS.md。

本轮19项既有配置测试通过；临时checkout的48项实际shell/配置生命周期检查通过（10正式、10smoke、18恢复/冻结/参数/数据边界、10help），另11个新脚本的Bash语法和可执行权限通过。保留真实Bash→compare→make_config/validate_config，仅替代远程ACP环境加载、日志与最终train调用；核对卡数、batch、累积、预算、目标参数及Stage设置，并检查自动/显式恢复、暂停、smoke/正式同名冲突、缺失冻结配置、配置篡改和训练覆盖拒绝。只修改启动脚本及文档，未安装依赖或启动GPU训练；实际完整模型的显存、速度与效果仍需远程smoke和正式实验确认。

2026-11-06 Flow开发位于独立worktree，保留源树14份未提交基线（`flow-baseline.json`）。SD3.5 Medium/ZImage Base接入既有LoRA训练循环，新增DPO/MVPO/Linear-DPO adapter-factor EMA、2/4卡预算与单卡smoke、HPDv3-sub准备及共享验证面板。数学、精度、时间采样、依赖锁和GPU未测边界详见[FLOW.md](FLOW.md)，命令统一见根command.md。旧运行checkout/环境和AOSS恢复白名单不更新。


2026-11-07 Flow Stagewise论文配方在独立self_correct_dpo-flow-stagewise checkout维护；训练入口按sd35/zimage分组，paper-v2消费一次性数据/模型审查凭据。默认4x5090/2xA800，ZImage5090仅显式BF16变体；200步滚动完整恢复、400步永久完整状态及旧八点评估并行保留。无额外policy EMA，Linear-DPO算法EMA reference保留。方法、预算、验证范围与命令见[FLOW_STAGEWISE.md](FLOW_STAGEWISE.md)，旧运行checkout及作业源码不修改。

SLAI paper-v2 部署使用独立资源表 `configs/resources-slai-flow.json` 与 `launchers/flow/slai_common.sh`；明确 SD35 为 `models/sd3.5-medium`。编号入口在旧 SLAI Conda 存在时激活独立 mvpo-flow，其他机器允许当前 Python/显式资源覆盖。模型报告保存在 AFS checkout 的 reports/slai，远程数据须独立准备并生成 passed 审计报告。完整 clone/env/prepare/run 命令见 [FLOW_SLAI.md](FLOW_SLAI.md)，不会复用本机绝对路径报告。
