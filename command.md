# 数据准备期间先验证训练（ZImage双H100）

新增独立128对真实图片快照与50步预检，不依赖正式数据完成，不读分卷；ZImage默认双H100，SD3.5四5090。
完整命令、模型一次验证、25→50步续跑和Stagewise K4见[docs/FLOW_DEBUG.md](docs/FLOW_DEBUG.md)。
正式编号入口仍需完整passed数据报告，预检不能接全量数据续训。

# HPDv3：兼容Hugging Face原生下载目录

远程数据目录没有`.download-manifest.json`时，改用统一入口，自动识别已有项目下载记录，或生成明确标识的本地来源指纹；不伪造下载完成状态。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
git pull --ff-only origin feat/flow-stagewise-slai-20261107
export HPDV3_RAW=/mnt/aoss/xietianyu/datasets/hpdv3
bash launchers/flow/prepare_slai_data.sh
```

入口显示每个阶段及日志，完整执行提取、准备、审计和64条canonical面板。
若旧版在读取train.json处Killed/137，更新后原命令重跑：已改成流式注释读取，无新增依赖，保留原数据与输出。
SLAI任务已确认存在mtime小数部分被截断的视图差异：新版接受大小相同且mtime精确整秒截断的情况，记录[source-stat-precision]，注释额外核对SHA256。保留原指纹和图片，更新后在实际任务内重跑；其他stat变化仍由[source-diagnostic]明确拒绝。
详细来源验证范围及重复运行说明见[FLOW_HPDV3_SOURCE.md](docs/FLOW_HPDV3_SOURCE.md)。

# W&B离线记录与训练中同步（2026-11-07更新）

新paper-v2默认W&B offline + TensorBoard，评估默认记录前8张图片；固定wandb==0.18.7。
两台服务器使用同一entity/project/group，以不同run ID区分实验；网络可用时在另一终端执行
`python scripts/sync_flow_wandb.py --entity YOUR_ENTITY RUN_DIR...`，反复执行会追加新完成的段，跳过正在写入的段。
加`--dry-run`只预览。完整命令及旧面板metadata补齐见[docs/FLOW_WANDB.md](docs/FLOW_WANDB.md)。
本地旧作业继续使用原checkout；本次发布在独立flow-wandb checkout完成，不替换它们的源码、配置或数据。

# 本地四组与 W&B 离线对比（2026-11-07）

最新追加：**65805（LinearDPO K1）/65806（Global MVPO K2 C，64 pairs/128 pair-views）**。
用户随后要求取消依赖，65805/65806已清除Dependency，四组直接正常排队；
相同2×A800/8CPU/96GB/4天、3200步、128 pair-views、seed17和W&B offline项目/group。
四组冻结配置已核对预算及相同模型、数据、评估面板，并复核passed凭据；脚本语法与diff检查通过。

```bash
# 已提交65805/65806，勿重复提交。
sbatch --job-name=zimage-linear-wb launchers/flow/local/train_stagewise_a800.sbatch linear-dpo
sbatch --job-name=zimage-globalk2-wb launchers/flow/local/train_stagewise_a800.sbatch global-k2
```

此前用户确认本地先跑 ZImage DPO K1、Stagewise MVPO K2 C；旧 Global MVPO/LinearDPO 作业由此替换。
新入口 `launchers/flow/local/train_stagewise_a800.sbatch dpo|stagewise-k2`，
每组2×A800/8CPU/96GB/4天，3200步、128 pair-views/update（DPO128对/K2为64对）。
独立 tag 为 `irip-wandb-20261107-v1`；旧 flow 的环境与准备数据只读复用。
一次数据审计作业65800已通过。新 paper-v2 默认 W&B offline + TensorBoard，
项目 `mvpo-flow`、group `zimage-hpdv3-pair_views128-seed17`；训练/评估共享稳定实验ID。
实际已取消旧65766/65767/65770，提交 **65803（DPO）/65804（Stagewise K2 C）**。
输入检查作业65802两组通过；原64条面板缺canonical元数据，已在新目录生成
`data/hpdv3-pilot-64-v2.jsonl`，prompt文本/顺序不变，旧产物保留。
5项receipt/预算检查、2项flow stagewise检查、1项真实离线W&B跨阶段记录检查和
1项面板生成→评估loader集成检查通过，Slurm脚本语法与diff检查通过。
提交时GPU训练尚未开始。SLAI K4入口、同面板比较、离线记录位置和显式同步命令见
[FLOW_WANDB.md](docs/FLOW_WANDB.md)。

# SLAI Flow paper-v2 部署

本轮完整 clone、独立环境、SLAI 路径核对、HPDv3 准备/一次审计与20个编号训练入口见 [docs/FLOW_SLAI.md](docs/FLOW_SLAI.md)。发布分支为 `feat/flow-stagewise-slai-20261107`；默认使用 `configs/resources-slai-flow.json`，SD3.5 路径为 `/mnt/aoss/xietianyu/models/sd3.5-medium`。编号入口在 SLAI 自动加载独立 `mvpo-flow` 环境，复用 `reports/slai/` 的远程一次模型验证报告。此前本地处理中数据快照不能用作远程 passed 凭据。

## 新实验：Flow Stagewise / Global 论文对照

本轮新实验使用独立源码目录 `/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow-stagewise`。
正在准备数据和已提交作业继续使用旧 `self_correct_dpo-flow`，下方旧记录供追溯。

- [实验矩阵、分段与硬件选项](docs/FLOW_STAGEWISE.md)：`launchers/sd35/` 默认四卡5090，`launchers/zimage/` 默认双A800；可选双H100/A800。
- [本地资源配置与启动命令](docs/LOCAL_STAGEWISE.md)：复用现有资产，新实验使用独立输出。
- [HPDv3一次性审计及可复用凭据](docs/FLOW_DATA_AUDIT.md)：数据准备完成后认证一次，后续训练只读passed报告及少量身份绑定。
- 模型验证入口为 `scripts/validate_flow_model.py`；本地静态报告为 `reports/sd35-model-validation.json` 与 `reports/zimage-model-validation.json`。

新入口统一使用 `paper-v2`，每400步永久保留完整checkpoint用于eval，每200步维护最新完整恢复状态。
不额外启用policy EMA；Linear-DPO保留算法reference EMA。ZImage四5090必须显式选择BF16权重变体，不能直接续训FP32实验。
数据处理中快照 `reports/hpdv3-review-20261107.json` 不是passed训练凭据；新脚本不会绕过尚未完成的数据准备。

## 本地双A800：Z-Image DPO / MVPO / LinearDPO

2026-11-07更新：环境已完成；数据准备从manager串行进程交接至CPU作业 **65774**（compute-1、4 CPU、16GB、无GPU）。旧提取及其cat/控制进程确认退出后才开启交接标记，已提取图片保留。当前看 `logs/local-a800/prepare-parallel-65774.log`；成功后由此作业释放65766/65767，65770依赖不变。加速实现和再次交接命令见本文末尾；下面保留先前部署记录。

交接后实测：compute-1上的两块数据盘为本地块设备挂载，绕过manager的NFS路径；38646张已有图片全部复用。新任务扫描到39000→42000张用时39.6秒，此段3000张均为新增落盘，平均约75.8张/秒（各1000张窗口63.6～99.4张/秒）；这是迁移节点与并行I/O的合计效果，不是单独线程数的对照实验。此时仍在提取，后续全量校验和GPU训练尚未完成。

2026-11-06用户确认每组两张、共四张A800并行。环境与数据均在manager tmux准备：`flow-env-a800`安装环境，`flow-data-a800`并行处理数据。DPO作业65766、MVPO作业65767各申请cluster-1、2×A800、8 CPU、96GB RAM、4天。DPO/MVPO暂时hold，由数据tmux在环境和数据都成功后自动release。LinearDPO作业65770已提交（相同资源），依赖after:65766：DPO启动后即可调度，不等DPO训练完成；实际开始时间由Slurm决定。

本地配置在`configs/local-a800-resources.json`，复用已下载Z-Image和PickScore，独立环境`.venv-flow-a800`由现有code环境克隆，保留torch2.6/cu124/torchvision0.21；旧环境不改。数据只从现有HPDv3分卷提取五来源子集及官方test隔离所需图像，输出`data/hpdv3-sub-v1`，两组共用64条validation面板。计算节点离线安装所需wheel保存在`.cache/flow-wheels`。Conda也设置`CONDA_OFFLINE=true`，复用现有包缓存。原65760因Conda联网代理失败，65761/65762随依赖失败取消，未开始训练；按用户要求，环境安装改在登录节点tmux会话`flow-env-a800`执行，日志为`logs/local-a800/environment-tmux-65768.log`；原65765已取消。用户进一步要求数据也在manager tmux执行，65768已取消。`flow-data-a800`使用只读code环境并行提取和筛选，日志为`logs/local-a800/data-tmux.log`。独立入口为`launchers/flow/local/setup_a800_env.sh`与`launchers/flow/local/prepare_a800_data.sh`。

```bash
# 查看本人队列与共用准备日志
squeue -u "$USER"
tail -f logs/local-a800/environment-tmux-65768.log
# 另一个终端查看数据进度
tail -f logs/local-a800/data-tmux.log
# 训练开始后
# tail -f logs/local-a800/train-65766.log
# tail -f logs/local-a800/train-65767.log
# tail -f logs/local-a800/train-65770.log
```

两组3200步、B2/累积32/global128 pairs；DPO K1，MVPO K2。每200步更新最新完整恢复状态，每400步永久保留完整checkpoint（含optimizer/RNG），固定点不随latest清理。八点评估不变，其他评估点继续保留LoRA；评估分段结束也保存latest以衔接训练。`checkpoints/latest.json`可指向固定完整`step-*`目录，也可指向临时滚动`resume-*`目录。

如果作业时限到达或中断，重新提交对应脚本会自动latest续训；尚在运行/排队时不要重复提交：

```bash
sbatch --job-name=zimage-dpo launchers/flow/local/train_zimage_a800.sbatch dpo
sbatch --job-name=zimage-mvpo launchers/flow/local/train_zimage_a800.sbatch mvpo
```

## 当前优先：直接做效果验证

按 [正式训练入口](docs/FLOW_QUICKSTART.md) 执行：独立 `mvpo-flow` 环境、两骨干各DPO/MVPO/Linear/SFT共八组配方，**默认跑满3200步**；评估200/400/800/1300/1800/2300/2800/3200及基座。双H100默认SD35 B4、ZImage B2，四5090 SD35 B1，均全局128对；首次预检后复用，关闭重复扫描与热循环诊断。环境一键安装见 `scripts/bootstrap_flow_env.sh`；双H100用 `launchers/flow/01_sd35_dpo.sh` 至 `08_zimage_sft.sh`，四5090用 `launchers/flow/sd35_4x5090.sh <dpo|mvpo|linear_dpo|sft>`。不要求先完成下方smoke矩阵。

新增SFT从各自基座独立训练，仅学习winner的flow速度MSE，每步128张winner，评估日程不变。Z-Image是否从HPDv3-sub获益的判断见[分析说明](docs/ZIMAGE_HPDV3_ASSESSMENT.md)。已有实验保持原checkout；新包v4用于新实验。

```bash
# 已分配的双H100容器中，每行一个独立实验
bash launchers/flow/07_sd35_sft.sh
bash launchers/flow/08_zimage_sft.sh
# 四5090，仅SD3.5M
bash launchers/flow/sd35_4x5090.sh sft
# 一个骨干的四方法配对比较；读取已有评估，无额外生成
python scripts/summarize_flow_runs.py "$SFT_RUN" "$DPO_RUN" "$MVPO_RUN" "$LINEAR_RUN" \
  --reference-run "$DPO_RUN" --out logs/flow-four-methods-v1.csv
```

下方保留通用配置与可选排障命令；其中手工环境名 `dpo-flow` 仍可使用，但当前快捷入口默认新环境名为 `mvpo-flow`。

## Flow：SD3.5 Medium / Z-Image Base（独立开发 worktree）

实现、公式、依赖来源与验证边界见 [docs/FLOW.md](docs/FLOW.md)。旧实验继续用原 checkout 和环境恢复；新flow目录为 `/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow`。以下训练/生成命令只在**已经分配的Slurm计算节点或SLAI容器**执行，本轮没有提交训练或安装依赖。

远程先克隆独立环境，保留已验证的torch/CUDA/NCCL。检查磁盘后才克隆（可能写入超过1GB）；此段由用户在目标服务器执行：

```bash
source /mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
# 先核实实际挂载及空间，目标环境应位于本人可写卷。
df -h /mnt/afs/L202500030/xietianyu/miniconda3 /mnt/aoss/xietianyu
conda create --name dpo-flow --clone dpo
conda activate dpo-flow
python -m pip freeze | rg -i '^(torch|torchvision|torchaudio|nvidia[-_])' > flow-gpu-runtime.constraints.txt
python -m pip install -r requirements-flow.txt -c flow-gpu-runtime.constraints.txt
# 在克隆环境更新本项目editable元数据/路径，不重新解析GPU依赖。
python -m pip install --no-deps -e .
python -m pip check
python scripts/check_flow_environment.py --family zimage
python -m pip freeze > flow-environment.freeze.txt
```

本地已有模型复用路径如下。把同名值写入本人 `configs/resources.local.json`；远程应按实际挂载覆盖，公共AOSS ZImage目录尚未核实。不要下载第二份模型。

```json
{
  "SD35_MODEL": "/irip_16t_0/xietianyu_2024/model/stabilityai/stable-diffusion-3.5-medium",
  "ZIMAGE_MODEL": "/irip/xietianyu_2024/workspace/sources/models/Z-Image",
  "HPDV3_RAW": "/irip_16t_0/xietianyu_2024/datasets/hpdv3",
  "HPDV3_SUB_PREPARED": "/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/data/hpdv3-sub-v1",
  "MONITOR_PROMPTS_FLOW": "/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/data/flow-monitor-v1.jsonl"
}
```

数据全量扫描/校验放计算资源。`HPDV3_REVISION`和`MODEL_REVISION`分别填实际可追溯的dataset/model源版本；模型存在完整下载receipt时可省略`--model-revision`。

```bash
# metadata-only只生成筛选记录/统计/哈希，不可用于训练；不是必需的重复预处理。
python scripts/prepare_hpdv3_sub.py --raw "$HPDV3_RAW" --out data/hpdv3-sub-selection-v1 --revision "$HPDV3_REVISION" --metadata-only
# 真正训练清单；保留原路径与path1优选标签，不按原图1024额外过滤。
python scripts/prepare_hpdv3_sub.py --raw "$HPDV3_RAW" --out data/hpdv3-sub-v1 --revision "$HPDV3_REVISION"
python scripts/prepare_flow_monitor.py --pairs data/hpdv3-sub-v1/pairs.jsonl --out data/flow-monitor-v1.jsonl --count 128 --seed 17
```

先单卡两步smoke验证真实模型加载、forward/backward和checkpoint，再在双卡新smoke run中暂停/恢复。`--smoke`保留1024分辨率但将累积降为1、每步保存、关闭评估，不能据此宣称正式预算性能已验证。

```bash
# 只生成/检查CPU配置，可在登录节点做；不加载模型。
bash launchers/flow/train.sh --family sd35_medium --algorithm mvpo --profile h100 --world-size 4 --run-tag inspect-v1 --config-only

# 以下必须在已分配GPU资源中。Slurm默认--execution slurm。
bash launchers/flow/train.sh --family sd35_medium --algorithm dpo --profile h100 --world-size 1 --smoke --run-tag sd35-single-v1 --model-revision "$MODEL_REVISION"
bash launchers/flow/train.sh --family zimage --algorithm linear_dpo --profile h100 --world-size 1 --smoke --run-tag zimage-single-v1 --model-revision "$ZIMAGE_REVISION"

# 双卡smoke暂停于step1；随后同命令/同tag去掉stop参数，自动latest恢复至step2。
bash launchers/flow/train.sh --family sd35_medium --algorithm linear_dpo --profile h100 --world-size 2 --smoke --run-tag sd35-resume-v1 --model-revision "$MODEL_REVISION" --stop-after-step 1
bash launchers/flow/train.sh --family sd35_medium --algorithm linear_dpo --profile h100 --world-size 2 --smoke --run-tag sd35-resume-v1 --model-revision "$MODEL_REVISION"
```

SLAI在上述命令末尾显式加 `--execution standalone --run-label <本次已分配资源标签>`；这不申请资源，也不绕过本集群Slurm规则。启动器默认使用当前Python；需要自动激活独立环境可设置 `SCD_FLOW_CONDA_SH=/mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh`、`SCD_FLOW_ENV=dpo-flow`，或设置`SCD_FLOW_PYTHON`。

正式pilot均默认3200更新、B1、lr5e-6、warmup200、每200完整保存、周期生成评估；首次正式运行前应完成上述GPU检查。以下只给运行入口，本轮未执行：

```bash
# 等pair-view：DPO/Linear global128 pairs；MVPO K2 global64 pairs。
bash launchers/flow/train.sh --family sd35_medium --algorithm dpo --profile 5090 --world-size 2 --beta 500 --run-tag dpo-b500-v1 --model-revision "$MODEL_REVISION"
bash launchers/flow/train.sh --family sd35_medium --algorithm mvpo --profile h100 --world-size 4 --views 2 --beta 2000 --run-tag mvpo-b2000-v1 --model-revision "$MODEL_REVISION"
bash launchers/flow/train.sh --family zimage --algorithm linear_dpo --profile h100 --world-size 2 --beta 500 --ema-decay .995 --linear-eta .01 --run-tag linear-b500-v1 --model-revision "$ZIMAGE_REVISION"
# 等pair对照global128，MVPO K2时pair-view=256。
bash launchers/flow/train.sh --family sd35_medium --algorithm mvpo --profile h100 --world-size 2 --budget-unit pairs --budget 128 --run-tag mvpo-pairs128-v1 --model-revision "$MODEL_REVISION"
# 原定义stagewise：每pair选段、K个view段内独立；index边界不是sigma=.5。
bash launchers/flow/train.sh --family sd35_medium --algorithm mvpo --profile h100 --world-size 2 --views 2 --stagewise --stage-boundary 500 --run-tag mvpo-stage-v1 --model-revision "$MODEL_REVISION"
```

LR/seed可在新run用`--learning-rate 1e-6 --seed 29`覆盖；调试可`--no-evaluation`。同tag任何配置改变都会拒绝，续训保持相同卡数、配置、源码、依赖与路径/模型身份。每服务器用独立run/tag。外部驱逐后重新申请到资源，再执行原命令即自动latest；没有完整checkpoint的旧尝试会保留为interrupted目录。源码更新后不拿新代码续旧实验。

固定生成入口（在已分配GPU上；`FROZEN_CONFIG`指run_flow输出的实际config路径，`ADAPTER_DIR`指完整checkpoint或独立快照内部的`adapter`目录）：

```bash
python scripts/generate_flow.py --training-config "$FROZEN_CONFIG" --prompts data/flow-monitor-v1.jsonl --out runs/flow-base-eval-v1 --seed 17
python scripts/generate_flow.py --training-config "$FROZEN_CONFIG" --adapter "$ADAPTER_DIR" --prompts data/flow-monitor-v1.jsonl --out runs/flow-aligned-eval-v1 --seed 17
# 中断后同参数加--resume。支持--steps、--guidance-scale以及CPU --check-only。
```

生成输出沿用原manifest/journal/AOSS原子发布与恢复；Linear生成只加载policy LoRA，不使用EMA reference。训练自动评估默认使用同面板PickScore，1000步保留图；实际评分模型兼容性仍需新环境GPU验证。


## SDXL：双H100分别直接训练及中断恢复

三个独立入口，每次在一个已分配两张H100的容器中选择一条；脚本自动激活原来的`dpo` Conda环境并加载资源表。DMPO K6沿用前述K6 Stage-C。直接正式训练，不自动运行smoke、部署自检或启动图片抽查；按用户要求关闭基座生成评估，保留200、500、1000……步及最终步的生成评估。清单加载、配置/恢复身份和实际消费图片时的校验仍保留；这不是一次完整图片审计。

```bash
# SDXL DMPO K1，两张H100。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/sdxl_h100/01_dmpo.sh
```

```bash
# SDXL DMPO + K6 Stage-C，两张H100。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/sdxl_h100/02_dmpo_k6_stage.sh
```

```bash
# SDXL DSPO K1，两张H100。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/sdxl_h100/03_dspo.sh
```

| 方法 | 每卡B | 累积 | 每步pairs | 每步pair-views | 输出目录名 |
|---|---:|---:|---:|---:|---|
| DMPO K1 | 1 | 96 | 192 | 192 | `sdxl-dmpo-pv192-2h100-10000-v1` |
| DMPO K6 Stage-C | 1 | 16 | 32 | 192 | `sdxl-dmpo-k6-stage-pv192-2h100-10000-v1` |
| DSPO K1 | 1 | 96 | 192 | 192 | `sdxl-dspo-pv192-2h100-10000-v1` |

三组均1024、BF16、seed17、10000优化步、LR1e-5、LoRA rank/alpha16；沿用DMPO beta6000/alpha.01及DSPO official_sdxl/beta2000/阈值.1。每200步原子保存完整状态（LoRA、optimizer、scheduler、各rank RNG），每1000步额外保留LoRA。输出位于资源表的`RUNS_ROOT`，默认`/mnt/aoss/xietianyu/self_correct/runs`；冻结配置在checkout的`configs/local-<输出目录名>.json`，控制台日志在`logs/sdxl-h100-<方法>.txt`。

**中断或断电后重新分配两张H100，直接重跑同一条命令。** 自动恢复未完成的checkpoint提交事务，再从`checkpoints/latest.json`继续。训练已完成、最终评估未完成时继续补评估；全部完成则退出。首个完整checkpoint之前断电时无法恢复尚未保存的更新，脚本保留原尝试为`.interrupted-时间戳-随机标识`目录并从基座重新开始，不覆盖原产物。TERM/INT沿启动链转发给子进程；平台立即强杀或断电仍只能恢复最近完整保存点，常规最多重算不足200步，checkpoint写入期间被打断可能回到前一个保存点。

启动新重复实验用`--run-tag v2`（恢复时继续带相同tag）；计划暂停可追加`--stop-after-step 2`，恢复时去掉。`--config-only`只生成冻结配置，不加载模型训练。H100入口使用新名字，不能承接原四卡5090实验；同一实验固定双卡、代码、环境和资源身份。新代码需部署到远程DPO目录后才能使用本节命令，不更新仍需恢复的旧实验checkout。

## main统一部署：后续实验使用远程DPO目录

`main`已整合`exp/sd15-baselines-20261102`的SFT/KTO/MaPO/DMPO/DSPO与`exp/stagewise-k6-20261103`的SD1.5双5090、SDXL四5090 K6 Stage-C启动器。后续新实验统一使用`/mnt/afs/L202500030/xietianyu/workspace/DPO`。下方独立exp目录的部署记录保留供旧实验追溯；新实验可把对应命令的checkout前缀换成本节DPO路径，脚本参数与预算相同。

目录尚不存在时，首次克隆：

```bash
git clone --branch main https://github.com/hypocritica/DPO.git \
  /mnt/afs/L202500030/xietianyu/workspace/DPO
```

已有DPO目录使用以下更新命令，不要向非空目录再次clone。先结束仍使用该目录的训练/评估；若旧任务还需恢复，保留其原checkout、冻结配置和环境。旧main训练源码与本次版本不同，不能仅更新代码后直接承接旧任务。正常Git更新保留未跟踪且被忽略的`configs/resources.local.json`、`configs/local-*.json`及日志/输出。

```bash
bash <<'BASH'
set -e
cd /mnt/afs/L202500030/xietianyu/workspace/DPO
if [[ -n "$(git status --porcelain)" ]]; then
  echo '存在未提交文件，已停止更新；请先保留并处理这些改动。'
  exit 1
fi
git fetch origin refs/heads/main:refs/remotes/origin/main
if git show-ref --verify --quiet refs/heads/main; then
  git switch main
else
  git switch -c main --track origin/main
fi
git merge --ff-only origin/main
git log -1 --format='%H %s'
BASH
```

显式fetch main也支持之前只克隆了某个exp分支的目录；分支分叉时`--ff-only`会停止，不覆盖本地提交。此流程不更换或安装环境。每个实验记录启动时SHA，运行和续训期间固定代码。旧基线提交9573c1e到K6启动器提交697b132只改启动器/文档，保持原冻结配置等条件时训练源码身份相同；不要将它们与旧main源码混为一谈。

新实验入口（每个容器选择一条）：

```bash
# SD1.5两张5090：KTO；同目录02_sft、03_mapo、04_dmpo、05_dspo。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/sd15_baselines/01_kto.sh
# SD1.5两张5090：DPO K6 Stage。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/sd15_final/05_dpo_k6_stage.sh
# SDXL四张5090：DPO K6 Stage。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/sdxl_final/05_dpo_k6_stage.sh
```

两套final目录均包含01_dspo、02_dspo_k6_stage、03_dmpo、04_dmpo_k6_stage、05_dpo_k6_stage，完整参数与smoke/恢复方式见下一节。

## 最终对比：SD1.5 双5090 / SDXL 四5090，K6 Stage-C

每个基座运行五组：DSPO K1、DSPO + K6 Stage-C、DMPO K1、DMPO + K6 Stage-C、DPO + K6 Stage-C。使用分支 `exp/stagewise-k6-20261103` 的独立checkout。全部为普通高斯视图、fixed pair_views预算；Stage为每个pair每次重新随机选择low/high、六个视图在该段内独立采样，边界500。此处不启用phase/Huber，也不是low/high各三个视图。DMPO仍计算反向KL，DSPO仍保留自己的score损失配方。

### 部署一次

```bash
set -e
SCD_FINAL_REPO=/mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103
git clone --single-branch --branch exp/stagewise-k6-20261103 \
  https://github.com/hypocritica/DPO.git "$SCD_FINAL_REPO"
git -C "$SCD_FINAL_REPO" checkout --detach
git -C "$SCD_FINAL_REPO" rev-parse HEAD
SCD_LOCAL=/mnt/afs/L202500030/xietianyu/workspace/DPO/configs/resources.local.json
if [[ -f "$SCD_LOCAL" ]]; then
  cp "$SCD_LOCAL" "$SCD_FINAL_REPO/configs/resources.local.json"
fi
```

首次核对发布的完整SHA；所有容器使用这一固定checkout。复用原有`dpo`环境、模型、已准备数据、评分模型及monitor面板。若有效私有资源覆盖保存在其他旧checkout，应从该位置复制。脚本自动加载资源和Conda，不提交平台作业、不下载或准备数据。每个命令放进一个已分配对应卡数的SLAI容器，五条命令是五个独立任务。

| 模型/方法 | GPU数 | 每卡pair B | 累积次数 | 全局pair/更新 | pair_views/更新 |
|---|---:|---:|---:|---:|---:|
| SD1.5 DSPO/DMPO K1 | 2 | 2 | 96 | 384 | 384 |
| SD1.5 三组K6 Stage-C | 2 | 1 | 32 | 64 | 384 |
| SDXL DSPO/DMPO K1 | 4 | 1 | 48 | 192 | 192 |
| SDXL 三组K6 Stage-C | 4 | 1 | 8 | 32 | 192 |

所有组10000优化步、seed17、LR1e-5、BF16、LoRA rank/alpha16；SD1.5为512，SDXL为1024。沿用每200步完整保存、每1000步LoRA及既有生成监控。DPO beta为5000/500，DMPO beta为2000/6000且alpha=.01，DSPO为SD1.5 score_residual/beta=.01和SDXL official_sdxl/beta=2000/阈值.1（按SD1.5/SDXL顺序）。同一objective的K1和K6共享这些参数。参数是已实现的研究起点，不能视为已验证最优。SDXL DSPO阈值按每view跨rank的microbatch归约，四卡K1/K6均B1；旧双卡DSPO结果不能只凭全局预算相同当作完全相同配方。

### SD1.5：每组两张RTX5090

```bash
# 各行分别放入一个双卡容器；不要把整个代码块作为一条任务执行。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sd15_final/01_dspo.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sd15_final/02_dspo_k6_stage.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sd15_final/03_dmpo.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sd15_final/04_dmpo_k6_stage.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sd15_final/05_dpo_k6_stage.sh
```

### SDXL：每组四张RTX5090

```bash
# 各行分别放入一个四卡容器。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sdxl_final/01_dspo.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sdxl_final/02_dspo_k6_stage.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sdxl_final/03_dmpo.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sdxl_final/04_dmpo_k6_stage.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sdxl_final/05_dpo_k6_stage.sh
```

### 检查与恢复

每组先用对应脚本运行独立两步smoke；保持正式分辨率、K、卡数和每卡B，累积1、关闭生成评估、每步保存，使用独立输出。例如四卡SDXL DSPO K6：

```bash
SCD_MODE=smoke bash /mnt/afs/L202500030/xietianyu/workspace/DPO-final-stagewise-20261103/launchers/sdxl_final/02_dspo_k6_stage.sh
```

四卡DDP不会合并单卡显存，K6合批仍每卡12张noisy图；真实1024模型显存须由这一步确认。smoke通过后运行正式脚本。可先用正式命令加`--stop-after-step 2`完成两次实际预算的更新并保存，再重跑原命令自动恢复latest。所有脚本支持`--config-only`、`--check-only`、`--stop-after-step N`、`--resume PATH`及`--resume_from_checkpoint PATH`；拒绝训练超参覆盖。

`SCD_RUN_NAME`可指定新实验名；默认输出名包含family、objective、k1或k6-stage、pv384或pv192、卡数、10000或smoke及final-v1。日志位于checkout的`logs/<实验名>.txt`，冻结配置为`configs/local-<实验名>.json`，输出为`$RUNS_ROOT/<实验名>`。保持同一代码、配置、环境及卡数续训。

已运行的SD1.5双卡K1 DSPO/DMPO基线，若数据、参数、步数与评估相同，可以直接复用结果；K1时旧pairs384与本次pair_views384实际相等。继续训练旧任务仍用旧checkout和原脚本，不能把旧目录换成本次新命名继续。无需为了新启动器重复训练相同基线。

论文主表与K消融、去掉Stage、去掉phase的安排见[COMPARISONS.md](docs/COMPARISONS.md#最终k6-stage实验与论文表格)。本次五组是当前待补实验，主表仍须包含Base、DPO K1及已计划的SFT/MaPO/KTO等适用基线。

## SD1.5：五组双 RTX 5090 单视图基线

分支 `exp/sd15-baselines-20261102` 包含当前SFT/KTO/MaPO/DMPO/DSPO、SDXL Stagewise-C及五组基线启动脚本。使用独立checkout；旧实验仍固定原代码，不在正在续训的目录切换分支。

### 远程部署一次

```bash
set -e
SCD_BASELINE_REPO=/mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102
git clone --single-branch --branch exp/sd15-baselines-20261102 \
  https://github.com/hypocritica/DPO.git "$SCD_BASELINE_REPO"
git -C "$SCD_BASELINE_REPO" checkout --detach
git -C "$SCD_BASELINE_REPO" rev-parse HEAD
SCD_LOCAL=/mnt/afs/L202500030/xietianyu/workspace/DPO/configs/resources.local.json
if [[ -f "$SCD_LOCAL" ]]; then
  cp "$SCD_LOCAL" "$SCD_BASELINE_REPO/configs/resources.local.json"
fi
```

首次部署记录输出的完整SHA，并与本次发布SHA核对；后续容器继续使用该checkout，不随分支更新。复用已有 `dpo` 环境、模型、准备完成的Pick-a-Pic及评分资产。脚本自动激活 `/mnt/afs/L202500030/xietianyu/miniconda3` 中的 `dpo` 环境，不安装或下载。资源由 `configs/resources.json` / `resources.local.json` / 环境变量提供。

五组都从同一SD1.5基座开始，K=1、512、BF16、seed17、LoRA rank/alpha16、LR1e-5，每卡B=2、双卡累积96次，即每个优化步384个pair记录，训练10000步；SFT每条记录只训练winner。沿用每200步完整保存、每1000步LoRA、基座及200/500/...步固定面板监控。SFT/MaPO不执行reference前向，等pair数不等于等FLOPs；KTO为pair-derived标签，其KL基准依赖microbatch划分。

| 脚本 | 目标及起始超参 | 默认正式输出目录名 |
|---|---|---|
| `launchers/sd15_baselines/01_kto.sh` | KTO，beta1000、两个lambda均1 | `sd15-kto-k1-p384-2gpu-10000-slai-v1` |
| `launchers/sd15_baselines/02_sft.sh` | winner-only SFT | `sd15-sft-k1-p384-2gpu-10000-slai-v1` |
| `launchers/sd15_baselines/03_mapo.sh` | MaPO，mapo_beta=.1、mapo_snr=.5 | `sd15-mapo-k1-p384-2gpu-10000-slai-v1` |
| `launchers/sd15_baselines/04_dmpo.sh` | DMPO，本地beta2000、alpha=.01 | `sd15-dmpo-k1-p384-2gpu-10000-slai-v1` |
| `launchers/sd15_baselines/05_dspo.sh` | DSPO，score_residual、beta=.01 | `sd15-dspo-k1-p384-2gpu-10000-slai-v1` |

DMPO的本地logit系数为beta，SDPO仓库为beta/2；这里沿用当前项目配方，不将同名beta视为跨实现同尺度。以上均是原目标基线，不包含SDPO的winner-preserving机制，也不包含多视图或stagewise。超参来源及适配边界见 [COMPARISONS.md](docs/COMPARISONS.md)。

### 每个双卡容器运行一组

五条命令分别填入五个已分配双5090容器；不要在同一个双卡容器同时启动五条。

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/01_kto.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/02_sft.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/03_mapo.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/04_dmpo.sh
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/05_dspo.sh
```

先在各自容器用 `SCD_MODE=smoke bash <对应脚本>` 做独立两步检查：512分辨率、每卡B=2、累积1、关闭生成评估、每步完整保存；独立输出名为 `sd15-<目标>-k1-2gpu-smoke-slai-v1`。通过后使用上面的正式命令。smoke不替代384pair正式更新及评分资产检查。

```bash
# 以KTO为例，其他四组替换脚本名。
SCD_MODE=smoke bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/01_kto.sh

# 正式预算内先跑2个优化步并保存，后续原命令自动从latest继续。
bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/01_kto.sh --stop-after-step 2
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-baselines-20261102/launchers/sd15_baselines/01_kto.sh
```

原命令重复启动自动恢复latest；也支持 `--resume latest` / `--resume_from_checkpoint latest`，完整恢复要求原代码、配置、依赖和双卡设置一致。`--config-only`仅生成配置，`--check-only`做既有CPU模型/数据预检。`SCD_RUN_NAME`可指定独立输出名；训练超参覆盖被拒绝，避免意外混用配方。控制台日志在checkout内 `logs/<实验名>.txt`，冻结配置为 `configs/local-<实验名>.json`，checkpoint和监控在 `$RUNS_ROOT/<实验名>`。已有输出缺失冻结配置、配置变化或smoke/正式模式共用名称会拒绝启动。

## SDXL 纯 DPO：双 H100 80GB 的 batch 对照与正式实验

在商汤已分配 **2 张 H100 80GB** 的容器内执行。沿用当前 AOSS 修复 checkout、`dpo` 环境、模型和数据；以下命令复用已有入口，无需更新正在使用的训练源码。每个 H100 实验使用新名称，从基座开始，不更改旧5090实验的冻结配置或续训状态。

保持 K=1、1024、BF16、全局192对/优化步、seed17、LR1e-5、beta500、LoRA rank/alpha16和梯度检查点。B 表示每卡图片对数，UNet 每次处理 `2 × B` 张图片。这里先只改变 B 和相应累积次数；VAE 仍使用原来的 FP32、`vae_batch_size=1`，不会随 B 自动扩大编码批量。

| 每卡 B | 每卡 UNet 图片数 | 累积次数 | 全局图片对/步 |
|---|---:|---:|---:|
| 4 | 8 | 24 | 192 |
| 8 | 16 | 12 | 192 |
| 12 | 24 | 8 | 192 |

这些是待实测的候选，尚未验证 H100 显存或速度。先顺序测试 B=4、8，再根据显存余量考虑12；选择稳定且秒/优化步较低的配置，不能按累积次数减少的比例推算提速。固定全局192对时保留学习率；batch 拆分变化会改变随机数分配和浮点累积，不承诺逐位复现 B=1 的轨迹。

### 五步吞吐对照

下面整段可直接粘贴到容器终端。每个 B 独立生成配置、运行5个完整优化步，关闭生成评估与诊断，最后仍保存 checkpoint 和 LoRA。打印第2–5步训练循环耗时的中位数，以及两个 rank 各自的 PyTorch 显存峰值；峰值不包含全部驱动/NCCL显存，另用 `nvidia-smi` 查看整卡占用。若任一短测报错，命令停止，不自动继续正式训练。

```bash
bash <<'BASH'
set -eo pipefail
SCD_H100_REPO=/mnt/afs/L202500030/xietianyu/workspace/DPO-aoss-storage-fix-v1
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate dpo
cd "$SCD_H100_REPO"
source paths.sh
mkdir -p logs
SCD_H100_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$$"
for SCD_H100_BATCH in 4 8; do
  SCD_H100_BENCH="sdxl-dpo-h100-2gpu-b${SCD_H100_BATCH}-p192-bench-${SCD_H100_TAG}"
  SCD_H100_CONFIG="configs/local-${SCD_H100_BENCH}.json"
  python -u scripts/make_config.py --family sdxl --method dpo --views 1 \
    --world-size 2 --train_batch_size "$SCD_H100_BATCH" \
    --budget-unit pairs --budget-per-update 192 \
    --mixed_precision bf16 --launcher accelerate \
    --benchmark --max_train_steps 5 --out "$SCD_H100_CONFIG"
  bash launchers/train.sh "$SCD_H100_CONFIG" "$RUNS_ROOT/$SCD_H100_BENCH" \
    2>&1 | tee "logs/${SCD_H100_BENCH}.txt"
  python - "$RUNS_ROOT/$SCD_H100_BENCH" <<'PY'
import json
from pathlib import Path
import statistics
import sys
out = Path(sys.argv[1])
print("Benchmark:", out.name)
for pattern in ("metrics-*.jsonl", "rank-1-*.jsonl"):
    path = sorted(out.glob(pattern))[-1]
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    seconds = [b["cost"]["training_seconds"] - a["cost"]["training_seconds"]
               for a, b in zip(rows, rows[1:]) if b["step"] == a["step"] + 1]
    peak = max(row["peak_cuda_bytes"] for row in rows) / 1024**3
    print(f"{path.name}: median={statistics.median(seconds):.2f} s/update, "
          f"PyTorch allocated peak={peak:.2f} GiB")
PY
done
BASH
```

短测目录保留用于核对，不作为正式实验的恢复源。要测试 B=12，将循环中的 `4 8` 改成 `12` 再执行，会生成新的时间戳目录。只有当前任务独占这两张卡时才做吞吐对照，避免与其他训练并发而干扰结果。

### 正式10000步

下例为 B=4；如果 B=8 的短测更快且显存稳定，将 batch 和实验名中的 `b4` 一起改为8。首次使用独立名称，后续重新执行相同命令会自动从该 H100 实验的 latest 恢复。保留每200步完整保存、每1000步 LoRA、基座和200/500/...步评估。

```bash
export SCD_FAMILY=sdxl SCD_WORLD_SIZE=2 SCD_VIEWS=1
export SCD_TRAIN_BATCH_SIZE=4 SCD_BUDGET_PER_UPDATE=192
export SCD_RUN_NAME=sdxl-dpo-h100-2gpu-b4-p192-v1
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-aoss-storage-fix-v1/launchers/acp_gaussian_fixed_pairs.sh dpo
```

应看到 `Frozen config: B=4, K=1, UNet images/GPU=8, accumulation=24, global pairs=192`。不要用 `launchers/sdxl/01_dpo.sh` 承接这些环境变量，该分组入口会把 B 固定回1。通用入口以 `dpo` 参数选择纯 DPO，脚本文件名中的 `gaussian` 不会将方法切成多视图。

验证边界：本地标准库实际生成了 B=4/8/12 的正式和五步短测共6组配置，确认预算、累积、精度、评估与保存设置；未连接商汤 H100、未提交作业、未测量性能。

## 新四组 Stagewise SD1.5：独立 checkout 与 SLAI 命令

本节使用 `launchers/sd15_stagewise/`，只运行新实验。旧实验继续使用原 checkout；不要在正在训练或准备续训的目录拉取新代码。新算法已在本地提交，远程部署前先确认该版本已通过 GitHub 插件推送。用已确认的远程提交完整 SHA 创建独立工作树，代码和配置都固定在该提交：

```bash
SCD_STAGE_REV='替换为已确认推送的stagewise提交完整SHA'
git -C /mnt/afs/L202500030/xietianyu/workspace/DPO fetch origin
git -C /mnt/afs/L202500030/xietianyu/workspace/DPO worktree add --detach /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030 "$SCD_STAGE_REV"
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030
```

先将上面的 SHA 占位符替换为实际提交。`fetch` 不切换或更新旧实验工作文件；新工作树在这些实验完成前也不再更新。

确认新 checkout 使用相同 `configs/resources.json`；若原目录使用 `configs/resources.local.json`，将本人这份私有覆盖复制到新 checkout 的相同位置，并核对模型、数据、128条 monitor panel与持久 `RUNS_ROOT`。复用既有 `dpo` 环境与已准备的数据；训练不会重新下载或准备。以下各命令分别填入 SLAI **已分配2张5090** 的容器启动命令，每个容器执行一组，不含平台提交或本机 Slurm 操作：

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030/launchers/sd15_stagewise/01_stage_dpo.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030/launchers/sd15_stagewise/02_stage_huber.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030/launchers/sd15_stagewise/03_constant_mix.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030/launchers/sd15_stagewise/04_reverse.sh
```

四组固定 K=2、全局192 pair/384 pair-view、10000优化步；双卡每卡B=2、累积48。其余保持 seed17、lr1e-5、beta5000、LoRA rank/alpha16、512 BF16、weight decay0.01、constant LR/warmup200。监控保持同一128 prompt、seed17、DDIM30、CFG7.5，包括基座评估，200步首次评估之后每500步；完整checkpoint每200步，LoRA每1000步。四组目标公式见 `docs/METHOD.md`。所有组从同一基座开启新训练，第一组也采用按pair选择阶段的采样，不能视为旧全域iid-t DPO的续训。

首次部署或排错时仍使用既有 `python scripts/check_deployment.py --gpu`；它只在已分配 GPU 的容器中手动运行，正式启动器不重复逐卡自测。训练数据仍默认8线程抽查8张并覆盖各 split 首对，不自动全量预处理或审计。

低成本快速验证复用既有 `make_config.py --smoke` 流程。先在新 checkout 激活同一 `dpo` 环境并 `source paths.sh`，然后执行下例（在已分配双卡容器中运行）；它关闭评估、warmup和诊断，每卡1对、累积1、只训练2步，每步保存，输出与正式实验分开：

```bash
python scripts/make_config.py --family sd15 --method stagewise-C-gaussian \
  --world-size 2 --smoke --set training.stagewise.objective=phase \
  --out configs/local-stagewise-phase-smoke-v1.json
python train.py --config configs/local-stagewise-phase-smoke-v1.json \
  --out "$RUNS_ROOT/sd15-stagewise-phase-smoke-v1" --check-only
python train.py --config configs/local-stagewise-phase-smoke-v1.json \
  --out "$RUNS_ROOT/sd15-stagewise-phase-smoke-v1" \
  --execution standalone --run-label stagewise-phase-smoke-v1 --stop-after-step 1
python train.py --config configs/local-stagewise-phase-smoke-v1.json \
  --out "$RUNS_ROOT/sd15-stagewise-phase-smoke-v1" \
  --execution standalone --run-label stagewise-phase-smoke-v1 --resume_from_checkpoint latest
```

其他三组将 objective 改为 `dpo`、`constant`、`reverse`，并分别更换配置文件、输出和 run-label；配置不覆盖已有文件。`--benchmark` 也沿用既有100步、关闭评估的独立流程。smoke/benchmark结果不续接正式实验。

快速验证通过后，再对每组做正式预算下的两步暂停检查；以下为第二组示例，保留正式累积和基座评估。暂停后检查输出保存，再在同一代码/环境/资源与卡数下去掉暂停参数继续：

```bash
bash /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030/launchers/sd15_stagewise/02_stage_huber.sh --stop-after-step 2
bash /mnt/afs/L202500030/xietianyu/workspace/DPO-stagewise-20261030/launchers/sd15_stagewise/02_stage_huber.sh --resume_from_checkpoint latest
```

`--stop-after-step` 是绝对优化步数，不改变10000步预算。重启原命令自动恢复latest；只放行该参数和 `--resume_from_checkpoint PATH`/`--resume PATH`，拒绝训练覆盖。默认实验名分别为 `sd15-stagewise-{dpo,phase,constant,reverse}-k2-p192-2gpu-10000-slai-v1`。冻结配置 `configs/local-<实验名>.json`、控制台 `logs/<实验名>.txt`、输出 `$RUNS_ROOT/<实验名>`；缺失冻结配置但已有输出时拒绝启动。每次启动重建预期配置并全量比较，防止改目标、阶段边界、优化参数或评估后误续训。

新重复实验可前置 `SCD_RUN_NAME=新的独立实验名`；seed仍为17。独立新实验可设置 `SCD_WORLD_SIZE=4` 或 `8`，保持每卡B=2、全局192pair，自动得到累积24或12；不支持同一实验中途改卡数。只允许2/4/8卡，并使用精确整除，绝不向下取整。旧入口保持原10000步默认值。

# 商汤大装置：从部署到 DPO 实验

按本文顺序先跑通 SD1.5 纯 DPO，再比较多视图 B/C。以下命令在商汤已分配的单机容器中执行；先从 2 张 RTX 5090 开始，后续可使用 4/8 张或 H100。模型、数据和输出挂载到持久存储，确认新容器仍能访问。这里不提供或执行平台资源申请命令；本机 irip 集群仍遵守 Slurm 约定。

## AOSS 图片保存修复与原实验续训

`os.link()` 在部分 `/mnt/aoss` 挂载上返回 `Errno 38`。图片提交和待恢复图片事务均在硬链接不支持时使用同一输出锁下的重命名；已有目标图片不覆盖，PNG、journal及manifest校验继续保留。此修复不调整训练目标、预算或监控时间：200、500、1000……步评估（含基座对照），每1000步保留LoRA，每200步更新完整续训点。

为保留旧实验进度，训练、监控、评估与render的源码检查仅接受 `b766801` 对应源码到本修复的精确SHA256配对；其他源码、配置、数据、模型、评分器、依赖或卡数变化仍拒绝。已有 `training.json`、监控身份、评估身份、run ID和checkpoint元数据保持原样，实际修复版源码及兼容策略SHA写入各输出目录的 `storage-compatibility-*.json`。未完成的评估PNG可从pending事务恢复，不需要删除评估目录。

在远程准备独立代码目录一次，保留其他正在训练的checkout和dpo环境：

```bash
set -e
SCD_FIX=/mnt/afs/L202500030/xietianyu/workspace/DPO-aoss-storage-fix-v1
git clone --single-branch --branch fix/aoss-render-resume \
  https://github.com/hypocritica/DPO.git "$SCD_FIX"
git -C "$SCD_FIX" checkout --detach fcbfb3757a5194e84a6bdf1b747f97360383d39f
SCD_LOCAL=/mnt/afs/L202500030/xietianyu/workspace/DPO/configs/resources.local.json
if [[ -f "$SCD_LOCAL" ]]; then
  cp "$SCD_LOCAL" "$SCD_FIX/configs/resources.local.json"
fi
```

旧实验续训固定在上述 AOSS 修复提交；不能仅跟随分支最新版本，因为后续 stagewise 算法修改了训练源码，不属于旧实验获准的存储兼容修复。已存在的旧 checkout 继续保留原版本。

在已分配、与旧实验相同的双卡容器中续训01：

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-aoss-storage-fix-v1/launchers/acp_resume_storage_fix.sh \
  sd15-dpo-p384-cu130-v1
```

此入口自动激活原dpo环境，从旧 `training.json` 提取完全相同的配置，并使用原输出目录的latest续训。已有200步续训点时，先补完基座和200步评估，再继续到500步评估，直至10000步。可追加 `--stop-after-step 201` 做一次受限的真实恢复，再用无暂停参数的原命令继续；这是绝对步数，不改变max_steps。其他中断实验仅替换实验名；原本关闭评估的实验仍保持关闭，不借恢复切换监控配置。没有完整checkpoint的实验无法通过此入口恢复。

需要独立评估旧快照时，也从修复目录调用 `launchers/sd15/05_evaluate.sh STEP RUN_A RUN_B`，不要修改正在训练的旧目录。实际GPU恢复和远程挂载行为仍需在SLAI验证，本机只做CPU测试。

## SLAI 分组脚本：SD1.5 / SDXL

`launchers/sd15/` 和 `launchers/sdxl/` 各自提供可直接填入容器启动命令的编号脚本。每个训练脚本固定2张GPU、seed17、10000个优化步，沿用每200步完整保存和自动续训；SD1.5 为512×512、基准384，SDXL 为1024×1024、基准192。脚本固定模型、K、每卡 batch 和预算，默认生成独立的 `*-slai-v1` 实验名。共享环境、资源路径、冻结配置和训练逻辑继续使用现有公共入口，启动时不新增逐卡自测或全图扫描。

| 脚本（两个模型文件夹内同名） | 用途 | SD1.5 每步图片对 / 对视图 | SDXL 每步图片对 / 对视图 |
|---|---|---:|---:|
| `01_dpo.sh` | 两种预算共用的 DPO 基线 | 384 / 384 | 192 / 192 |
| `02_c_k2_fixed_views.sh` | C，K=2，固定视图 | 192 / 384 | 96 / 192 |
| `03_b_k2_fixed_views.sh` | B，K=2，固定视图 | 192 / 384 | 96 / 192 |
| `04_c_k2_fixed_pairs.sh` | C，K=2，固定图片对 | 384 / 768 | 192 / 384 |
| `05_evaluate.sh` | 两张卡各评估一个模型 | — | — |
| `06_c_k4_fixed_views.sh` | C，K=4，固定视图 | 96 / 384 | 48 / 192 |
| `07_c_k6_fixed_views.sh` | C，K=6，固定视图 | 64 / 384 | 32 / 192 |
| `08_c_k4_fixed_pairs.sh` | C，K=4，固定图片对 | 384 / 1536 | 192 / 768 |
| `09_c_k6_fixed_pairs.sh` | C，K=6，固定图片对 | 384 / 2304 | 192 / 1152 |

第一批使用01–04训练和05评估，第六组资源运行06；07–09用于后续补齐K对照。当前先运行 SD1.5，SDXL入口按需使用。准备完成的对应模型、数据和监控面板由资源表提供，脚本不下载或预处理数据。

先在远程准备独立 checkout，只执行一次；旧实验继续使用原目录：

```bash
set -e
SCD_CODE=/mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1
git clone --single-branch --branch dev/self-correct-workflow \
  https://github.com/hypocritica/DPO.git "$SCD_CODE"
SCD_LOCAL=/mnt/afs/L202500030/xietianyu/workspace/DPO/configs/resources.local.json
if [[ -f "$SCD_LOCAL" ]]; then
  cp "$SCD_LOCAL" "$SCD_CODE/configs/resources.local.json"
fi
```

SLAI 每个已分配双卡的容器选择一条独立启动命令。脚本根据自身位置进入 checkout，无需手动 `cd` 或设置环境变量。例如 SD1.5 的前四组和可选第六组分别为：

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1/launchers/sd15/01_dpo.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1/launchers/sd15/02_c_k2_fixed_views.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1/launchers/sd15/03_b_k2_fixed_views.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1/launchers/sd15/04_c_k2_fixed_pairs.sh
```

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1/launchers/sd15/06_c_k4_fixed_views.sh
```

SDXL 使用相同文件名，将路径中的 `launchers/sd15/` 改成 `launchers/sdxl/` 即可。训练控制台日志在该 checkout 的 `logs/<实验名>.txt`，输出在 `$RUNS_ROOT/<实验名>`。默认名称示例：`sd15-dpo-fixed-pairs-p384-2gpu-10000-slai-v1`、`sd15-gaussian-C-k2-fixed-views-v384-2gpu-10000-slai-v1`。重启同一脚本自动恢复；首次可追加 `--stop-after-step 2` 在正式预算中暂停，后续去掉暂停参数恢复。新重复实验可以用 `SCD_RUN_NAME` 指定新名字，但改变名字不改变seed，不能算独立随机种子。

第5组在01、02都已完整保存1000步快照后启动，默认同时评估这两个实验：

```bash
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-slai-v1/launchers/sd15/05_evaluate.sh
```

评估读取监控面板而非剩余测试面板，使用50个采样步和PickScore，两个进程分别绑定容器可见的两张GPU。SD1.5 沿用独立评估的PNDM/CFG7.5，SDXL沿用Euler/CFG5；训练内监控仍为30步，分别报告。没有后台轮询或自动提交；默认1000步快照不存在时明确失败，待保存完成后重跑。评估脚本不改变训练任务原有的分段监控。

可用位置参数选择已保存的步数和任意两个同模型实验名，或用 `SCD_EVAL_STEP`、`SCD_EVAL_A`、`SCD_EVAL_B` 指定。输出目录和单模型日志会区分实验与步数；相同命令支持继续未完成的评估：

```bash
bash launchers/sd15/05_evaluate.sh 2000 \
  sd15-gaussian-B-k2-fixed-views-v384-2gpu-10000-slai-v1 \
  sd15-gaussian-C-k2-fixed-views-v384-2gpu-10000-slai-v1
```

单模型日志为 `logs/eval-<实验名>-step<步数>.txt`，结果为 `$RUNS_ROOT/<实验名>/eval-monitor-50steps-step<步数>-v1/summary.json`。`bash launchers/sd15/05_evaluate.sh --help` 可直接查看参数，不加载模型或Conda。

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

双卡入口共同调用 `launchers/acp_common.sh`：自动激活远程 `dpo` 环境，进入脚本所属 checkout，加载同一资源表，检查准备标记、冻结配置并启动训练。GPU 可用性由训练入口检查，启动器不再重复执行逐卡 CUDA 自测。统一10000步、BF16、LoRA rank/alpha16、LR1e-5、β5000（SD1.5）、seed17、每200步完整保存、TensorBoard 和原固定 PickScore 评估；快速数据校验见下文。重启复用冻结配置并自动尝试从 latest 恢复，缺少有效 checkpoint 时停止，不覆盖旧输出。

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

使用已有 `dpo` 环境，以下步骤不删除或重建环境。新容器确实没有该环境时才执行 `conda create -n dpo python=3.11 -y`。在没有训练占用该环境时更新依赖。

2026-10-26 用户选择官方稳定配套：PyTorch2.14.1/cu130、torchvision0.29.1/cu130、NCCL2.30.7；已提供的远程驱动为580.95.05，支持CUDA13.0。这是与用户已验证的NCCL2.31.2不同的新组合，安装成功不代表双卡通信已经验证。旧PyTorch2.7.1/cu128固定依赖NCCL2.26.2，不能将2.31.2直接与它放进同一份可解析的requirements。本次使用官方配套，不设置LD_PRELOAD，也不隔离安装NCCL。模型相关库保留已有API版本，避免同时迁移研究实现。

```bash
set -e
cd /mnt/afs/L202500030/xietianyu/workspace/DPO
source /mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
conda activate dpo
python --version
which python
# 检查真实环境/输出挂载的空间；安装阶段可能写入数GB。
df -h "$CONDA_PREFIX" /mnt/aoss/xietianyu

# 清除上一次临时NCCL库覆盖；后续直接使用官方wheel自带依赖。
unset LD_PRELOAD
# cu12/cu13的NCCL包可能拥有相同库文件路径，先清理再统一安装。
# 本项目不使用torchaudio，移除它可避免旧音频包固定旧torch版本。
python -m pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13 torchaudio
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --upgrade -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check

python - <<'PY'
import torch, torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA runtime:', torch.version.cuda)
print('NCCL runtime:', torch.cuda.nccl.version())
assert torch.__version__ == '2.14.1+cu130'
assert torchvision.__version__ == '0.29.1+cu130'
assert torch.version.cuda == '13.0'
assert torch.cuda.nccl.version() == (2, 30, 7)
PY

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

本项目GPU环境以 `requirements.txt` 为准：PyTorch2.14.1/cu130、torchvision0.29.1/cu130、NCCL2.30.7和Accelerate0.34.2。CUDA13.0构建要求兼容的R580及以上驱动，参考 [官方版本说明](https://pytorch.org/blog/pytorch-2-14-release-blog/)；实际NCCL版本以 `torch.cuda.nccl.version()` 为准，`nvidia-smi` 右上角的CUDA Version只是驱动能力。依赖解析与导入通过后，在SLAI已分配双卡容器做一次纯DPO优化步，验证DDP参数同步、前后向和保存，不自动执行逐卡串行检查：

```bash
source paths.sh
python scripts/make_config.py --family sd15 --method dpo --world-size 2 \
  --smoke --out configs/local-sd15-dpo-cu130-smoke-v1.json
bash launchers/train.sh configs/local-sd15-dpo-cu130-smoke-v1.json \
  "$RUNS_ROOT/sd15-dpo-cu130-smoke-v1" --stop-after-step 1
```

此smoke关闭生成评估、每卡1对、只累积1次，单独目录只做通信/训练/保存验证，不改变正式实验预算。若这个名字已存在，使用新的v2等名字。完整训练或续训依旧固定同一套依赖；原环境训练产生的checkpoint继续使用原环境，新依赖使用新的实验目录。

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

## 训练前检查：默认少量并行抽查，完整审计手动执行

训练入口默认 `--data-validation auto --validation-workers 8 --validation-samples 8`。配置检查和数据加载并行执行；普通执行路径去掉重复的 `check_rows`，由实际训练保留一次语义检查。ACP 启动器已删除逐卡串行 CUDA 自测，首次部署或排错时才手动运行前文的 `check_deployment.py --gpu`。

每个进程仍需读取训练清单，并核对清单 SHA256、行结构、重复 ID、跨 split 隔离和已有 `ready.json` 的来源/计数。路径只做字符串归一化；已删除全图/全 shard inventory、缓存匹配和缓存写入。默认按图片位置抽8张并加上各 split 首对，最多14张，8线程并行读取；不保证启动前发现未抽中图片损坏。实际训练会校验即将解码的原始字节 SHA256，缺失或损坏时停止。清单解析与模型加载仍需时间，不承诺零启动开销。

`auto` 始终抽样，不因缺 ready/旧缓存而自动完整读取图片；旧 `.pairs.jsonl.validation.json` 被忽略，不新建或改写。没有图片 SHA 的旧清单会明确报错，应重新准备含 SHA 的清单，或显式选择 full。已有准备标记与清单不符仍会拒绝。日志保留阶段、累计秒数、rank 和实际 `selected_images`；并行队列最多保留16个任务。

数据准备已完成后，普通启动器无需额外参数。要单独查看快速校验或主动做一次完整审计，用已有 `dpo` 环境执行以下 CPU 命令，不启动模型/GPU，也不改训练配置或输出：

```bash
source paths.sh
python -u scripts/check_training_data.py \
  --pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" --mode auto

# 需要排查数据损坏时才完整读取全部唯一图片；可按存储情况调整线程数。
python -u scripts/check_training_data.py \
  --pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" --mode full --workers 8
```

新实验也可通过 `train.py` / `make_config.py` 的 `--data-validation full`、`--validation-workers 4`、`--validation-samples 16` 显式覆盖并写入冻结配置。full 会读取所有唯一图片；Arrow 也实际检查行和图片字节，不只核对 shard 元数据。完整审计只返回校验结果，不生成缓存或重写原清单。已有实验不修改配置续训，独立排错使用上述 CPU 命令。

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

## SFT、KTO、MaPO、DMPO、DSPO 与现有多视图组合

实现、原文公式和适配边界见 [docs/COMPARISONS.md](docs/COMPARISONS.md)。本节使用包含新增代码的独立checkout；不在正在续训的旧checkout更新源码。代码尚未自动推送或部署到SLAI。进入部署了这些代码的目录并激活已有 `dpo` 环境后运行。配置仍通过 `configs/resources.json` / `resources.local.json` 读取模型、清单和持久输出；启动器不安装环境、不下载模型、不提交平台任务。

统一入口是 `python scripts/compare_objectives.py`。默认双卡，`--world-size 4|8`可改卡数，自动保持相应预算。`--config-only`只生成冻结配置；`--check-only`运行既有CPU数据/模型预检。实际训练必须在SLAI已分配GPU容器中运行，本机登录节点只用上述两种检查模式。

### 先生成配置，再独立两步验证

```bash
# 只生成配置，适合在开发机查看：不会访问GPU或启动训练。
python scripts/compare_objectives.py --family sd15 --objective sft --config-only
python scripts/compare_objectives.py --family sdxl --objective dmpo --config-only
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 6 --budget-unit pair_views --config-only

# 下列实际smoke只在已分配双卡容器运行；使用当前激活的项目Python。
# 两步、梯度累积1、无生成评估、每步完整保存；输出与正式训练分开。
python scripts/compare_objectives.py --family sd15 --objective sft --mode smoke
python scripts/compare_objectives.py --family sdxl --objective mapo --mode smoke
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 6 --mode smoke
```

每个新方法/模型实际运行前先做对应的 `--mode smoke`；以上只是示例，不代表所有组合的GPU检查已完成。SDXL保持1024分辨率，默认每卡1对；K6的前向批量较大，实际显存需验证，不能从CPU测试推断5090一定可容纳。

检查暂停与恢复，用一个新smoke名称运行以下两条命令：

```bash
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 2 --mode smoke \
  --run-name sd15-dmpo-k2-resume-smoke-v1 --stop-after-step 1
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 2 --mode smoke \
  --run-name sd15-dmpo-k2-resume-smoke-v1 --resume latest
```

### 主表：单视图基线

各行是一个独立实验。以下默认10000步、SD1.5每步384对、SDXL每步192对，seed17、rank/alpha16、LR1e-5；每200步完整保存、每1000步保留LoRA，正常监控评估继续启用。SFT每对只使用winner，MaPO不使用reference。保留已有同预算DPO和Base结果；若没有对应DPO，可把objective换成dpo新跑匹配对照。

```bash
python scripts/compare_objectives.py --family sd15 --objective sft
python scripts/compare_objectives.py --family sd15 --objective kto
python scripts/compare_objectives.py --family sd15 --objective mapo
python scripts/compare_objectives.py --family sd15 --objective dmpo

python scripts/compare_objectives.py --family sdxl --objective sft
python scripts/compare_objectives.py --family sdxl --objective mapo
python scripts/compare_objectives.py --family sdxl --objective dmpo
```

默认DMPO的beta为2000/6000（SD1.5/SDXL），alpha=.01；KTO beta1000；MaPO独立使用mapo_beta=.1、mapo_snr=.5。这些是起始配置，论文需披露验证选择过程。KTO来自pair-derived二元标签，详见方法说明。

### DMPO + 当前Gaussian C

```bash
# 固定pair数：与主表DMPO保持每步同样多的训练pair，计算量随K增加。
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 2
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 4
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 6
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method multiview-C-gaussian --views 4
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method multiview-C-gaussian --views 6

# 固定pair-view数：SD15 K4每步96对，SDXL K4每步48对。
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 4 --budget-unit pair_views
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method multiview-C-gaussian --views 4 --budget-unit pair_views

# 固定pair-view数：SD15 K6每步64对，SDXL K6每步32对。
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-C-gaussian --views 6 --budget-unit pair_views
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method multiview-C-gaussian --views 6 --budget-unit pair_views

# 聚合消融：各视图先算DMPO loss再平均。
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method multiview-B-gaussian --views 2 --budget-unit pair_views

# 分阶段采样 + DMPO-C，对应旧stagewise全程DPO组的采样规则。
# 与旧stagewise预算匹配：每步192对 / 384 pair-views。
python scripts/compare_objectives.py --family sd15 --objective dmpo \
  --method stagewise-C-gaussian --views 2 --budget-unit pair_views
```

每组DMPO组合都与相同method、K、预算的DPO比较（将 `--objective dmpo` 改为 `--objective dpo`）。默认输出名称包含family/objective/method/K/预算/卡数/模式，因此单视图DMPO与DMPO+多视图不会覆盖彼此。K2/K4/K6按主方法的设置选择。分阶段Huber混合暂不开放DMPO，配置显式拒绝。

### DSPO：单视图、多视图与stagewise

SD1.5默认 `dspo_recipe=score_residual`、beta=0.01；SDXL默认 `official_sdxl`、beta=2000、阈值0.1，分别对应作者代码的两种配方。DSPO-B平均各视图完整loss；DSPO-C先平均margin形成共享gate，再平均各视图残差loss。C是本项目扩展，公式和SDXL跨rank阈值适配见 [COMPARISONS.md](docs/COMPARISONS.md)。DSPO仅支持epsilon prediction。

```bash
# 开发机可做：只冻结配置，不加载模型或启动训练。
python scripts/compare_objectives.py --family sd15 --objective dspo --config-only
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method multiview-C-gaussian --views 4 --config-only
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views --config-only

# 已分配双卡容器：先用独立smoke目录训练两步。
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views --mode smoke
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method multiview-C-gaussian --views 4 --mode smoke

# 正式单视图基线，默认10000步、每步384/192对。
python scripts/compare_objectives.py --family sd15 --objective dspo
python scripts/compare_objectives.py --family sdxl --objective dspo

# 普通Gaussian融合，固定pair预算；K可改2/4/6，B可替换C作消融。
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method multiview-C-gaussian --views 4
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method multiview-C-gaussian --views 6
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method multiview-B-gaussian --views 4
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method multiview-C-gaussian --views 6

# SD1.5阶段内融合，固定384 pair-view：K4=96对/步，K6=64对/步。
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method stagewise-C-gaussian --views 4 --budget-unit pair_views
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views

# 同一smoke实验暂停后恢复；正式实验也用同样的参数保持方式。
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method multiview-C-gaussian --views 4 --mode smoke \
  --run-name sd15-dspo-c-k4-resume-smoke-v1 --stop-after-step 1
python scripts/compare_objectives.py --family sd15 --objective dspo \
  --method multiview-C-gaussian --views 4 --mode smoke \
  --run-name sd15-dspo-c-k4-resume-smoke-v1 --resume latest

# 显式超参会参与完整冻结检查；调整时选择新run-name。
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --dspo-recipe official_sdxl --beta 3000 --dspo-sdxl-threshold 0.1 \
  --run-name sdxl-dspo-beta3000-v1 --config-only
```

固定pair-view的Gaussian对照只需同样加 `--budget-unit pair_views`。上述两步smoke使用累积1，不保持正式预算。默认beta和10000步是本框架的研究起点，应使用验证集选择超参及checkpoint。每个DSPO组合与相同采样、K和预算的DPO/DMPO对比，不能直接用不同预算的旧stagewise结果替代。

DSPO也可使用已有独立rollout：

```bash
python scripts/make_config.py --family sd15 --objective dspo --method multiview-C \
  --views 4 --world-size 2 --batch-size 1 --effective-batch 384 \
  --max_train_steps 10000 --out configs/local-sd15-dspo-rollout-c-k4-v1.json
python train.py --config configs/local-sd15-dspo-rollout-c-k4-v1.json \
  --out /mnt/aoss/xietianyu/self_correct/runs/sd15-dspo-rollout-c-k4-v1 \
  --execution standalone --run-label sd15-dspo-rollout-c-k4-v1
```

rollout使用detached状态和已有有效target，属于项目扩展；其额外UNet开销不计作普通Gaussian等计算量。

### SDXL：DMPO / DSPO + Stagewise-C

以下命令在包含新增代码的独立checkout根目录、已激活项目环境的SLAI已分配GPU容器运行。两个目标均支持K=2/4/6、2/4/8卡、固定pair或pair-view预算；默认1024分辨率、每卡1对、10000步。stagewise仍为每pair先选一次阶段，全部K个视图在该段独立抽t和噪声；没有改成低/高各K/2。

```bash
# 开发机：只生成配置，不访问GPU。
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views --config-only
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views --config-only

# 已分配容器：各自独立两步smoke，保持1024分辨率、累积1、关闭评估。
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method stagewise-C-gaussian --views 6 --mode smoke
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method stagewise-C-gaussian --views 6 --mode smoke

# 正式固定pair-view预算：192 pair-view/步，K2/4/6分别为96/48/32对/步。
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views

# 匹配采样、K和预算的DPO对照。
python scripts/compare_objectives.py --family sdxl --objective dpo \
  --method stagewise-C-gaussian --views 6 --budget-unit pair_views

# 固定pair预算：192对/步，视图数随K增加。
python scripts/compare_objectives.py --family sdxl --objective dmpo \
  --method stagewise-C-gaussian --views 4 --budget-unit pairs
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method stagewise-C-gaussian --views 4 --budget-unit pairs

# 同一独立smoke实验暂停后恢复。
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method stagewise-C-gaussian --views 6 --mode smoke \
  --run-name sdxl-dspo-stagewise-k6-resume-smoke-v1 --stop-after-step 1
python scripts/compare_objectives.py --family sdxl --objective dspo \
  --method stagewise-C-gaussian --views 6 --mode smoke \
  --run-name sdxl-dspo-stagewise-k6-resume-smoke-v1 --resume latest
```

DMPO默认beta=6000、alpha=.01；DSPO默认 `official_sdxl`、beta=2000、阈值.1，保留逐视图跨rank阈值判断。两者仅组合 `stagewise.objective=dpo` 的阶段采样，不加入Huber混合或额外DPO项。正式训练默认启用原SDXL监控与每200步完整保存；输出名包含模型、目标、方法、K、预算、卡数和模式。改变这些参数需要独立实验。K6在1024分辨率的实际显存和吞吐仍需远程smoke验证。

### Stagewise K4/K6的原四种目标

当前每个pair先抽low/high阶段，全部K个view在该阶段独立采样；K4平均4个margin，K6平均6个，再应用该段目标。不是K4低/高各2个，也不是K6各3个。若做两段各半的额外消融，须先段内聚合再加权两段loss，详见方法说明；当前未实现该新模式。

`compare_objectives.py`开放stagewise采样与DPO/DMPO/DSPO的组合。原DPO的phase/constant/reverse实验使用底层配置入口，例如：

```bash
# 改成6可生成独立K6实验；固定384 pair-view预算。
SCD_STAGE_VIEWS=4
python scripts/make_config.py --family sd15 --objective dpo \
  --method stagewise-C-gaussian --views "$SCD_STAGE_VIEWS" \
  --world-size 2 --batch-size 1 --budget-unit pair_views --budget-per-update 384 \
  --max_train_steps 10000 --set training.stagewise.objective=phase \
  --out "configs/local-sd15-stagewise-phase-k${SCD_STAGE_VIEWS}-v1.json"
python train.py --config "configs/local-sd15-stagewise-phase-k${SCD_STAGE_VIEWS}-v1.json" \
  --out "/mnt/aoss/xietianyu/self_correct/runs/sd15-stagewise-phase-k${SCD_STAGE_VIEWS}-v1" \
  --execution standalone --run-label "sd15-stagewise-phase-k${SCD_STAGE_VIEWS}-v1"
```

phase为low段Huber、高段DPO；constant/reverse改对应配置值并使用独立文件/输出名。DMPO/DSPO只开放 `stagewise.objective=dpo` 的阶段采样；尚未定义的Huber混合配置会被拒绝。

### 可选rollout组合和超参覆盖

普通Gaussian统一入口只接受无rollout预算比较。若当前方法使用独立rollout，用底层配置入口复用原来的状态构造；rollout额外前向开销必须另报：

```bash
python scripts/make_config.py --family sd15 --objective dmpo --method multiview-C \
  --views 2 --world-size 2 --batch-size 1 --effective-batch 384 \
  --max_train_steps 10000 --out configs/local-sd15-dmpo-rollout-c-k2-v1.json
python train.py --config configs/local-sd15-dmpo-rollout-c-k2-v1.json \
  --out /mnt/aoss/xietianyu/self_correct/runs/sd15-dmpo-rollout-c-k2-v1 \
  --execution standalone --run-label sd15-dmpo-rollout-c-k2-v1

# 调参使用独立冻结配置/输出。此例只是匹配旧DPO beta5000的margin尺度。
python scripts/make_config.py --family sd15 --objective dmpo --method dpo \
  --world-size 2 --batch-size 2 --budget-unit pairs --budget-per-update 384 \
  --max_train_steps 10000 --beta 2500 --set training.dmpo_alpha=0.01 \
  --seed 17 --out configs/local-sd15-dmpo-beta2500-v1.json
```

`--set training.mapo_beta=...`、`--set training.mapo_snr=...`、`--set training.kto_desirable_weight=...` 和 `--set training.kto_undesirable_weight=...` 同样由底层 `make_config.py` 支持。冻结后通过 `train.py --config ... --out ...` 运行，不在同一输出中改超参。只有明确切换objective或实际切换模型family才加载相应beta预设；相同family重复解析冻结配置保留已设置beta，命令 `--beta` / `--set` 优先级最高。

已有未完成输出默认从latest恢复，也可显式 `--resume latest`。相同名称的配置发生变化、已有输出缺失冻结配置都会拒绝；已完成的实验如需重跑，使用新的 `--run-name`。`--mode smoke` 完成后，正式训练使用默认train模式从base重新开始。

### 统一生成评估

沿用 `evaluate.py --run <训练输出> --step <保留步数>`，SD1.5使用 `configs/evaluate.json`，SDXL使用 `configs/evaluate-sdxl.json`。`--benchmarks hpdv2 pickapic partiprompts`、`--metrics pickscore hpsv2 image_reward`可选择同一评估集合，实际可用键以 `python evaluate.py --help` 和评分配置为准。固定生成seed、prompt、scheduler和CFG；单独保存每个方法的结果。新目标使用相同LoRA保存格式，不需要另写生成器。

本节代码检查仅在CPU完成；不代表真实大模型训练、生成评估或性能已验证，也未自动启动正式实验。

# 本地 HPDv3 数据准备加速（2026-11-07）

`extract_hpdv3_sub.py --workers 4` 单路读取拼接 gzip，使用 1MiB tar 缓冲、最多 64MiB/16 个待写任务和4个写入线程；已完成且大小匹配的图片复用，目录只创建一次，最终凭据在所有写入成功后原子保存。`prepare_hpdv3_sub.py --workers 4` 对唯一图片分批并行校验，一次读取完成格式验证和 SHA256；筛选、标签和 seed17 划分不变。每1000张或30秒记录进度及近期速度。gzip 恢复仍需要从头扫描，不会从压缩分片直接跳转。

计算节点接管已有串行提取的入口（4 CPU、16GB、无GPU、4天）：

```bash
cd /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow
sbatch launchers/flow/local/prepare_parallel_a800.sbatch 65766 65767
# 仅在新作业 RUNNING、旧提取及其 cat 子进程均已退出后创建交接标记：
# touch logs/local-a800/handoff-<新作业ID>.ready
```

新作业最多等待交接10分钟；未获得交接标记不会写数据。成功完成提取、筛选及64条评估面板后，核实环境已就绪且训练作业仍属于本账号并处于hold，再释放参数指定的训练作业。LinearDPO继续沿用已有的DPO启动依赖。旧日志保留；新日志为 `logs/local-a800/prepare-parallel-<jobid>.log`。重复运行前检查是否已有准备任务，不能与旧版无锁提取器同时写同一目录。
