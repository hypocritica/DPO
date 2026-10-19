# SLAI / SenseCore 部署

RTX5090（Blackwell）请优先使用 [SLAI_SD15_5090_GUIDE.md](SLAI_SD15_5090_GUIDE.md) 的torch2.7.1/cu128独立环境与完整迁移步骤；不要使用下文torch2.3镜像中的GPU构建直接训练5090。

统一路径与在线下载准备：先编辑根目录 [paths.sh](../paths.sh)，按 [SLAI_DATA.md](SLAI_DATA.md) 下载Pick-a-Pic v2、HPDv3及所需权重，生成清单和配置。下文原有环境与GPU资源说明仍适用。

依据用户提供、核对日期为2026-10-03的平台说明整理；这里只写项目部署所需信息，不复制团队账号、公网端口、密钥或私人项目路径。资源余量、镜像和挂载以实际控制台为准。

UU进入的是Windows电脑。训练先在SenseCore申请CCI GPU容器，或创建ACP训练任务。启动脚本只使用已经分配的GPU，不申请资源、不创建实例、不重复提交任务。

1. 首次使用CCI，复制团队已成功运行的兼容GPU镜像和挂载配置，使用自己的实例名和目录。核对实际GPU型号、显存、卡数、CPU、内存、自动停止时间。
2. 代码及恢复checkpoint建议放在自己可写的AFS子目录，例如`/mnt/afs/<your-directory>/self_correct_dpo`。AFS/AOSS挂载及写权限需核实；不要把别人环境路径当通用配置。滚动checkpoint使用原子rename及普通文件语义，放在支持这些操作的文件系统上。
3. 当前用户选择的镜像为 `nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5`。requirements已针对Python3.10、torch2.3、Diffusers0.33.1、Transformers4.45.2、PEFT0.10.0、Accelerate0.34.2整理。旧实验的torch2.6/Transformers4.53.3环境保持原样，不因迁移修改。
4. 核对`nvidia-smi`及PyTorch。GPU名称包含5090或80g不代表实际卡型/显存；新GPU可能需要不同PyTorch构建。独立环境重新冒烟，不改团队共享环境。

```bash
hostname
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

默认一个GPU进程，也支持同一节点2/4/8卡DDP。SD1.5模板为全局batch256；`make_config.py --world-size 2`、`4` 或 `8` 会保持全局batch并调整累积。SDXL保持全局batch128。资源配置选择一个worker副本、节点内N张GPU；协调入口只启动一次，由它管理torchrun阶段。启动与恢复见[DISTRIBUTED.md](DISTRIBUTED.md)。

## 所选镜像的环境

官方NGC24.04使用Python3.10、CUDA12.4.1、cuDNN9.1及定制PyTorch2.3.0a0构建，见[NVIDIA说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html)。平台名称写torch2.3.0，进入后仍需核对实际版本/tag。官方普通torch2.3 wheel只有CUDA11.8/12.1和CPU构建，不能使用不存在的cu124安装命令代替该镜像，见[PyTorch版本表](https://pytorch.org/get-started/previous-versions/#v230)。

在所选镜像中，为本项目建隔离venv并继承现有GPU torch/torchvision：

```bash
cd /mnt/afs/<your-directory>/self_correct_dpo
python -m venv --system-site-packages .venv-slai23
source .venv-slai23/bin/activate
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install --no-deps -e .
python -c "import torch, torchvision, diffusers, transformers, peft, accelerate; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(diffusers.__version__, transformers.__version__, peft.__version__, accelerate.__version__)"
```

requirements接受NGC的torch2.3.0a0/torchvision0.18.0a0版本，不要求升级或替换其GPU构建。包元数据也支持torch2.3。新版训练只读取显式本地权重；HPSv3/ImageReward仍使用各自可选环境。

本地另建了 `self_correct_dpo_slai23_cpu` Conda兼容环境，Python3.10、torch2.3.0+cpu、torchvision0.18.0+cpu与上述库版本，用于真实多进程Gloo和小模型测试。CPU环境没有CUDA/NCCL，不能当作SLAI镜像实机验证，也不能用于GPU训练。SLAI GPU镜像、驱动、互联及显存仍需在实际容器内核实。

SDXL组件也使用同一项目环境，`make_config.py --family sdxl` 选择1024px、batch1/累积128的独立模板；内置VAE训练用FP32，可显式提供本地FP16-fix VAE。它与SD1.5的显存和速度不同，不复用既有显存估计；在已经分配的GPU上先运行SDXL两步暂停/恢复检查。完整命令见[SDXL.md](SDXL.md)。本次没有申请CCI/ACP或提交SDXL作业。

## CCI

在已分配GPU的Linux容器内：

```bash
cd /mnt/afs/<your-directory>/self_correct_dpo
# 使用你自己的环境；若需Conda，source你实际的conda.sh再activate。
export MODEL_PATH=/mnt/afs/<your-directory>/models/sd15
export PAIRS_PATH=/mnt/afs/<your-directory>/data/pickapic/pairs.jsonl
export SCD_PYTHON=/path/to/your/env/bin/python
"$SCD_PYTHON" scripts/make_config.py --model "$MODEL_PATH" --pairs "$PAIRS_PATH" \
  --smoke --out configs/local-smoke.json
bash launchers/train.sh configs/local-smoke.json runs/slai-smoke-001 --stop-after-step 1
bash launchers/train.sh configs/local-smoke.json runs/slai-smoke-001 \
  --resume runs/slai-smoke-001/checkpoints/latest.json
```

尖括号是占位内容。确认日志、有限非零梯度、adapter保存与恢复后，再创建正式配置。CPU实例只做检查，不执行GPU训练。

## ACP

在控制台选团队授权工作空间、单副本、主方法1张GPU、兼容镜像及相同AFS挂载。正式CPU/内存/时限依资源规格决定；原实验参考4–8 CPU、32–64GB主存，不能将此当平台固定要求。首次小任务不要设置反复重试。

启动命令示例（先在CCI准备好本地配置，替换个人路径）：

```bash
cd /mnt/afs/<your-directory>/self_correct_dpo
export SCD_PYTHON=/path/to/your/env/bin/python
bash launchers/train.sh configs/local-formal.json runs/slai-model-K2-C-001
```

脚本以exec交接给Python，TERM能进入checkpoint处理。时限需给当前完整update与保存留余量；闲时抢占的具体信号/宽限期按平台核实。恢复同一任务时保留配置、源码、数据、模型清单及依赖，显式传`--resume .../latest.json`；重复启动同一输出会被拒绝。

CCI与ACP是两条资源路线，同一个实验不要同时启动两份。资源申请、任务ID和GPU冒烟结果在实际SLAI执行后再记录，本次没有连接或提交SLAI任务。

## 新默认监控（2026-10-18）

正式配置默认自动监控Pick-a-Pic test子集的PickScore，需配置`MONITOR_PROMPTS`、`SCORERS_CONFIG`及PickScore本地模型路径。每200步滚动保存最新LoRA及完整优化器/LR/各rank随机状态，恢复使用`checkpoints/latest.json`；每1000步与最终步永久保存LoRA和图片，两条保存计划独立运行。训练在200步及每500步退出进程评估，再恢复；200步滚动保存不额外生成图片或评分。`--smoke`关闭监控。指定步数的HPDv2四指标评估见[EVALUATION.md](EVALUATION.md)，同样先获得GPU资源再运行，不自动申请CCI/ACP。
