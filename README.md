# Self Correct DPO

SD1.5 / SDXL LoRA DPO。默认使用独立 rollout 多视图 C；另提供无 rollout 的普通高斯 B/C、视图合批和纯 DPO。支持任意正整数 K、单机 2/4/8 卡、Accelerate 启动、TensorBoard/W&B，以及每 200 步完整续训保存。

- [开发与双服务器实验约定](docs/DEVELOPMENT.md)
- [安装、数据准备、启动与续训命令](command.md)
- [统一资源地址](configs/resources.json)
- [公共默认超参数](configs/defaults.json)
- [数据说明](docs/DATA.md)、[方法公式](docs/METHOD.md)、[评估说明](docs/EVALUATION.md)
- [算法审查与逐轮改进计划](docs/ALGORITHM_REVIEW_AND_PLAN.md)
- [SDXL 论文依据与验证范围](docs/SDXL.md)

新增 [SD1.5 stagewise 四组入口](launchers/sd15_stagewise)：按pair选择低/高噪声阶段、阶段内各视图独立t和ε，比较全部DPO、低阶段Huber、全程固定混合及反向分配。四组均10000步、K=2、192pair/384pair-view，默认双卡B=2、累积48；公式与独立checkout部署命令见 [方法](docs/METHOD.md) 和 [command.md](command.md)。

Gaussian B/C 新比较入口分别为 [固定图片对](launchers/acp_gaussian_fixed_pairs.sh) 和 [固定图片对视图](launchers/acp_gaussian_fixed_views.sh)：两组统一10000步，SD1.5基准384、SDXL基准192，支持 K=2/4/6 和2/4/8卡，各模型共用同预算的新 DPO 基线。前者保留每步图片对数，后者将图片对数除以 K；配置、日志和输出名包含模式与预算。完整实验表和命令见 [command.md](command.md#两套-gaussian-c-对照固定图片对--固定视图预算)。下文原有256/128入口继续保留。

资源路径在 `configs/resources.json` 中统一修改；私有覆盖用 `configs/resources.local.json`，无需 `source paths.sh`。模型根目录为 `/mnt/aoss/xietianyu/models`，数据集根目录为 `/mnt/aoss/xietianyu/datasets`，各资源保留原子目录结构。训练需要已处理的 `pairs.jsonl`；原始 Pick-a-Pic 地址为 `/mnt/aoss/xietianyu/datasets/pickapicv2/data`。SDXL、SD3、HPDv3 的位置已预留，SD3 训练后续实现。

SDXL 的1024×1024入口为 [launchers/acp_sdxl.sh](launchers/acp_sdxl.sh)：纯 DPO、rollout B/C、普通高斯 B/C 共用同一启动器，支持单机2/4/8卡、全局128对、10000步和自动续训。完整数据准备、两步训练/恢复及评估命令见 [command.md](command.md#sdxl10241024-的完整入口)。

SLAI 分组实验脚本分别放在 [launchers/sd15](launchers/sd15) 和 [launchers/sdxl](launchers/sdxl)：01–04为主训练、05为双卡并行评估、06–09补齐K=4/6。SD1.5基准384、SDXL基准192，每个训练脚本直接运行一个双卡实验；逐组启动命令见 [command.md](command.md#slai-分组脚本sd15--sdxl)。

RTX5090依赖采用官方PyTorch2.14.1/cu130、torchvision0.29.1/cu130和NCCL2.30.7，要求兼容CUDA13.0的驱动；已确认的远程驱动为580.95.05。已有 `dpo` 环境的升级步骤及双卡DPO一步验证见 [command.md](command.md#环境和路径)，使用普通pip依赖安装，不需要LD_PRELOAD或单独的NCCL安装目录。

训练启动并行执行配置检查与清单加载，默认8线程抽查8张图片并覆盖各 split 首对，最多14张；实际训练读取时继续核对 SHA256。已移除自动逐卡 CUDA 自测、重复的训练行检查、全图 inventory 和校验缓存协议，默认不会自动完整读取所有图片。需要完整审计时显式运行 `python scripts/check_training_data.py --pairs /实际路径/pairs.jsonl --mode full`；具体说明见 [command.md](command.md)。

商汤大装置首次部署按 [command.md](command.md) 从上到下执行：更新 main → 激活 dpo 环境 → 资源/CUDA检查 → 一次性数据准备 → 两步纯 DPO 暂停及恢复 → 正式纯 DPO → 多视图 B/C 对照。资源检查入口为 `python scripts/check_deployment.py --stage assets --gpu --world-size 2`，仅在远程已分配 GPU 的容器里加 `--gpu`。

数据准备独立使用 [launchers/acp_prepare_data.sh](launchers/acp_prepare_data.sh)，例如 `bash launchers/acp_prepare_data.sh pickapic sd15` 或 `hpdv3 sd15`；已完成输出自动复用。SD1.5 纯 DPO 双卡正式训练使用 [launchers/acp_sd15_dpo_2gpu.sh](launchers/acp_sd15_dpo_2gpu.sh)，10000步、每卡2对、累积64、全局256对。两个双卡训练入口共用 `acp_common.sh` 的环境、日志、资源检查、冻结配置和恢复流程。GPU 启动命令及旧容器脚本备份见 [command.md](command.md)。原8卡入口保留。

4张卡的 K2-C 对照使用 [launchers/acp_sd15_c_k2_4gpu.sh](launchers/acp_sd15_c_k2_4gpu.sh)，同为10000步、全局batch256；独立保存配置、日志和checkpoint。此版本使用当前默认 model rollout C。并行实验保持训练代码不变，具体说明见 [command.md](command.md)。

2张5090版本分别为 [纯DPO](launchers/acp_sd15_dpo_2gpu.sh) 和 [K2-C](launchers/acp_sd15_c_k2_2gpu.sh)：每卡batch2、梯度累积64、全局batch256、10000步，每200步完整保存。首次可追加 `--stop-after-step 2` 检查训练和保存，然后不加该参数重启续训。各自使用新的双卡实验目录，不从8/4卡实验恢复；命令见 [command.md](command.md)。

```bash
# 在远程已分配GPU上运行；本集群登录节点不可运行训练。
python train.py --family sd15 --world-size 4 \
  --output_dir /mnt/aoss/xietianyu/self_correct/runs/sd15-C-k2-v1 \
  --execution standalone --run-label sd15-C-k2-v1

# 被驱逐并重新分配到相同卡数后，使用原参数加：
# --resume_from_checkpoint latest
```

纯 DPO 和 rollout B/C 分别在 `configs/dpo.json`、`configs/multiview-B.json`、`configs/multiview-C.json`；省略 `--config` 默认 rollout C。普通高斯多视图使用 `--method multiview-C-gaussian` 或 `multiview-B-gaussian`（对应同名配置），默认 K=2，`--views 4` 改为 K=4，并将 K 个视图合批计算。命令参数覆盖默认设置，`--print-config` 可只查看最终配置。

数据准备完成后，双卡普通高斯 K=2/4/6/8 使用 [launchers/acp_multiview_gaussian_2gpu.sh](launchers/acp_multiview_gaussian_2gpu.sh)，例如 `SCD_VIEWS=8 bash launchers/acp_multiview_gaussian_2gpu.sh`。SD1.5 的 K=2 默认每卡2对、累积64，K=4/6/8 默认每卡1对、累积128，全局均为256对；显存允许时可用 `SCD_TRAIN_BATCH_SIZE=2` 配新实验名，累积自动调整。SDXL 加 `SCD_FAMILY=sdxl`，默认每卡1对、全局128对。完整命令和未实测的显存边界见 [command.md](command.md)。

```text
src/self_correct_dpo/
  train.py, training.py       训练入口与优化循环
  config.py, resources.py    参数合成与统一资源地址
  models.py, architecture.py 模型、LoRA、SD1.5/SDXL组件
  algorithms/                DPO、B/C、rollout与实验算法
  data/                      偏好对、数据读取与固定评估面板
  runtime/                   DDP、保存恢复、评估调度与日志
  evaluation/                图像生成、评分模型与指标
```

历史配置归档到 `docs/legacy/configs/`，旧实验仍使用冻结源码复现。当前版本对代码、依赖、配置和数据做续训一致性检查；更新代码前先结束旧实验或保留其 checkout。

常规只运行与改动有关的现有轻量检查，例如：

```bash
python -m unittest discover -s tests -p test_stage_pause.py -v
python -m unittest discover -s tests -p test_joint.py -v
```
