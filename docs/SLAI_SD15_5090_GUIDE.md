# 私有GitHub → SLAI → RTX 5090：SD1.5普通DPO与纯K=2对照

这份指南对应2026-10-19（北京时间）的代码，环境步骤于2026-10-20按用户要求更新为独立Conda。用户确认第一轮比较 **普通DPO（K=1）与纯K=2（不加model/oracle扰动）**。两组都从同一个SD1.5基座重新初始化LoRA，使用相同Pick-a-Pic训练清单、全局batch256、2000次优化器更新、seed17和固定监控面板。后续SDXL/SD3与HPDv3训练不挤入第一轮。命令中的 `<你的目录>`、`<EIP地址>` 等占位符须先替换为自己的实际值。

历史平台记录来自《回顾 SLAI 显卡申请流程》和《检查 SLAI 机器的服务器连接方式》：UU连接的是Windows；调试进入CCI，训练进入ACP；原有AFS挂载在 `/mnt/afs`。历史文档在另一台Mac，当前工作区只保留其已核实流程。团队入口、可用资源、镜像、挂载权限、EIP地址和SSH端口，以你当前控制台为准，本文不复制历史私人连接信息。

## 1. 准备上传的完整代码

上传的是本 `self_correct_dpo` 模块的全部运行代码、配置、依赖、测试与文档，并让它们成为私有 `hypocritica/DPO` 仓库的根目录。无需上传父目录LMDPO中的历史实验、图片、模型、环境或运行结果。

在当前集群的模块根目录生成代码包：

```bash
python scripts/package_code.py --out dist/DPO-code-20261019.zip
```

包内结构是 `DPO/train.py`、`DPO/src/self_correct_dpo/`、`DPO/paths.sh` 等；附带每个文件SHA256的 `CODE_MANIFEST.json`。打包器明确排除环境、模型、数据、运行输出、`.env`、本地实验配置和Git历史，不进行任何远端操作。

如果从Mac操作，可把包下载到Mac，例如在Mac终端执行（`irip`须是你已有的SSH配置别名）：

```bash
scp irip:/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo/dist/DPO-code-20261019.zip ~/Downloads/
```

GitHub网页确认 `https://github.com/hypocritica/DPO` 的可见性为Private。推荐先克隆远端，再复制代码，保留远端已有的README/提交历史关系：

```bash
# 在Mac本地；先用你已有的GitHub认证方式完成认证。
mkdir -p ~/work/slai-upload
cd ~/work/slai-upload
unzip ~/Downloads/DPO-code-20261019.zip
git clone https://github.com/hypocritica/DPO.git DPO-git
rsync -av --exclude='.git' DPO/ DPO-git/
cd DPO-git
git status --short
git add .
git commit -m "Prepare standalone SD1.5 DPO experiments for SLAI"
git push -u origin HEAD
```

以上假设远端默认分支为main；在GitHub确认代码所在分支。不要使用force push。若网页界面上传，上传解压后DPO目录中的内容，而不是只上传zip，也不要多套一层 `self_correct_dpo/`。本文后续均假设仓库根目录直接有 `train.py`。Git操作参考 [GitHub官方说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 2. 先准备SLAI的持久存储和CPU容器

1. Mac打开UU，连接你收藏的SLAI Windows电脑。
2. 在该Windows的Chrome打开团队SenseCore控制台。历史使用“中国上海一区”、`share-space`；当前仍应选择你有权限的工作空间。
3. 进入 **CCI云容器实例 → 创建应用**。第一阶段用CPU容器准备代码、环境、下载和预处理，避免下载几百GB时占着4/8张GPU。
4. CPU预处理先选约8–16 vCPU、64GB主存，按实际内存峰值调整。全量分组可能需要更多内存，这不是已验证的全量资源保证。
5. 挂载团队授权AFS到 `/mnt/afs`，只使用自己可写的子目录。代码、环境、数据和checkpoint放AFS；不要放Windows D盘或容器临时根目录，也不要把AOSS的对象存储挂载当成训练恢复状态的普通文件系统。
6. 使用团队允许的Linux开发镜像。GPU阶段须兼容Blackwell；不要沿用旧torch2.3镜像里的GPU构建，详见第5节。CPU/GPU实例切换后要挂载同一AFS，并核对Python/系统兼容性。
7. CCI需要交互连接时，沿用团队已验证的SSH开发镜像/启动配置，让容器保持运行并监听SSH端口。若镜像没有SSH服务入口，选择有该能力的授权镜像；不在团队共享主机上改sshd或安装驱动。

全量磁盘预算包括官方Parquet、导出的去重图片、HPDv3分卷及解压图片、模型、pip环境和实验结果。应按第6节plan输出计算，预留TB级容量；1.5–2TB可作为全量初始预算，但不是已核实的上限或配额。两个数据集只需在AFS准备一次。

## 3. 取得SSH连接信息并登录

历史使用的是CCI的 **EIP/DNAT连接路线**，不是把UU Windows当成GPU服务器。

1. 等CCI实例状态为运行中，打开应用详情，找到实例、容器和网络信息。
2. 在应用端口中确认SSH服务监听的容器端口，通常为22。
3. 使用团队授权的EIP，在 **EIP → DNAT规则** 中把外部端口映射到这个实例的SSH端口；已有规则可直接从实例详情查看。没有EIP管理权限时由团队管理员绑定，不自行改别人的规则。
4. 记下当前实例的 **EIP地址、DNAT外部端口、容器登录用户**。外部端口不一定是22。每次新建/更换实例都重新核对，不能复用别人的旧端口。
5. 首次可先使用控制台的Web终端登录容器，把自己本地公钥追加到该容器用户的 `~/.ssh/authorized_keys`，不要覆盖已有条目。

在发起SSH的电脑上生成专用密钥；若该文件已存在则复用，不覆盖。Mac终端示例：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/slai_dpo
cat ~/.ssh/slai_dpo.pub
```

把输出的 `.pub` 一整行复制到CCI Web终端。Web终端执行：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/authorized_keys
# 粘贴刚才的公钥一整行，换行后按 Ctrl-D 结束输入。
chmod 600 ~/.ssh/authorized_keys
```

然后在Mac连接：

```bash
ssh -i ~/.ssh/slai_dpo -p <DNAT外部端口> <容器用户>@<EIP地址>
```

Mac的 `~/.ssh/config` 可保存：

```sshconfig
Host slai-dpo
    HostName <EIP地址>
    Port <DNAT外部端口>
    User <容器用户>
    IdentityFile ~/.ssh/slai_dpo
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

之后 `ssh slai-dpo`，或VS Code安装Remote-SSH后连接该Host。若团队网络只允许从UU Windows访问EIP，则在Windows PowerShell执行同样的ssh步骤，密钥/配置在 `%USERPROFILE%\.ssh`；不要把Mac私钥粘到服务器。

连不上时先确认实例仍运行、SSH监听、DNAT指向当前实例、公钥属于当前用户、访问网络已授权。若团队改用IAP开发机，按控制台提供的 `sco iap ssh 用户@节点标识` 路线，不把CCI DNAT与IAP混用。[CCI官方SSH流程](https://console.sensecore.cn/micro/help/docs/cloud-foundation/compute/cci/#开发环境使用)、[IAP官方命令](https://www.sensecore.cn/help/docs/CLI/SCO_Refrence/IAP/ssh)。

## 4. 在SLAI克隆代码并统一设置路径

在自己的AFS子目录内克隆：

```bash
mkdir -p /mnt/afs/<你的目录>/code
cd /mnt/afs/<你的目录>/code
git clone https://github.com/hypocritica/DPO.git
cd DPO
git rev-parse HEAD
```

私有仓库可用GitHub CLI的交互认证或该仓库的只读deploy key。不要把访问token写进URL、脚本或paths.sh。若CCI能开网页却无法克隆，需要核对该Linux实例的GitHub网络与认证；Windows能联网不证明容器也能。

编辑 `paths.sh`：

```bash
export SCD_STORAGE_ROOT="/mnt/afs/<你的目录>/self-correct-assets"
```

其余SD1.5、SDXL、Pick-a-Pic、HPDv3、评分器、缓存和run目录由此推导，也可以在文件中单独修改。每次新终端和每次ACP启动都先：

```bash
source paths.sh
mkdir -p "$SCD_STORAGE_ROOT"
df -h "$SCD_STORAGE_ROOT" /tmp
```

同时检查AFS控制台中自己的配额。后续持久资产统一在该目录下；代码仍在 `code/DPO`。不要在已有run开始后修改其数据/模型身份、并行卡数或源码再假称精确恢复。

## 5. RTX 5090 32GB的独立运行环境

**5090属于Blackwell/sm_120；旧torch2.3 + CUDA12.4方案不适用。** PyTorch从2.7提供Blackwell支持及CUDA12.8 wheel。本文固定torch2.7.1 + torchvision0.22.1/cu128，以减少和已有Diffusers/PEFT代码的版本跨度，而不是安装“最新”所有包。[PyTorch发布说明](https://pytorch.org/blog/pytorch-2-7/)、[官方安装命令](https://pytorch.org/get-started/previous-versions/#v271)。

控制台选择团队允许、宿主驱动支持5090的资源，独立环境使用Python3.10。**镜像自带CUDA12.4也可以保留，在Conda环境内安装torch的cu128 wheel；无需先把镜像Toolkit升级为12.8。** 这不保证任意旧镜像均可运行，宿主驱动和实际CUDA计算必须通过第8节检查。CUDA12.8 GA的开发驱动版本表为Linux570.26或以上，具体5090平台驱动由SLAI管理，Conda安装包不会升级驱动。[NVIDIA说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)、[Conda与镜像版本说明](SLAI.md#独立conda环境)。

在CPU容器也可先把环境安装到AFS，切换GPU实例后再验证：

```bash
source paths.sh
source "$SCD_CONDA_BASE/etc/profile.d/conda.sh"
python --version
conda create --prefix "$SCD_5090_ENV" python=3.10 pip -y
conda activate "$SCD_5090_ENV"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-5090.txt -r requirements-eval.txt
python -m pip install --no-deps -e .
python -m pip check
export SCD_PYTHON="$SCD_5090_ENV/bin/python"
```

先在自己的持久目录安装Miniconda，并把 `paths.sh` 中的 `SCD_CONDA_BASE` 改为它的实际根目录。上述环境prefix必须是新目录；已有venv使用新prefix重新创建Conda环境，不能把Conda激活命令用于旧venv。`requirements.txt`保留旧NGC/torch2.3方案；5090使用 `requirements-5090.txt`，不要再安装前者把torch降回2.3。各profile复用 `requirements-common.txt`。此5090组合尚未在SLAI实机验证。

## 6. 验证下载通路，下载与预处理

先使用CPU容器验证真实下载，而不是只curl网页首页。HF认证按资产权限配置，下载阶段不要继承训练的离线标志：

```bash
source paths.sh
source "$SCD_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SCD_5090_ENV"
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python scripts/download_assets.py plan pickapic-v2 hpdv3 sd15 pickscore clip-h14
# 先下载小型测试metadata，验证实际文件下载通路。
python scripts/download_assets.py download hpdv2
# 这一项仅用于验证网络并顺便准备可选400-prompt旧评估，不是第二份训练数据。
python scripts/download_assets.py download sd15 pickscore clip-h14
python scripts/download_assets.py download pickapic-v2 hpdv3
python scripts/download_assets.py extract-hpdv3
python scripts/prepare_downloaded_data.py pickapic --family sd15
python scripts/prepare_downloaded_data.py benchmarks --family sd15
```

`hpdv2/test.json`实际很小；其成功不能保证Xet大文件通路成功，随后SD1.5真实权重下载也必须成功才算模型通路可用。若外网/重定向被平台阻断，停下检查团队允许的出口、官方镜像或用另一台联网机器下载后传到同一AFS；不关闭证书校验，不凭空假设ModelScope上同名仓库等价。先完成下载再申请正式多卡训练。

下载器记录并固定实际commit，处理脚本保留原人类胜负、caption和图像哈希，排除官方held-out重叠后按连通分量进行新的本地90/5/5划分。图片不提前缩放，512px变换在训练时执行。新下载版本不是旧151418对清单，训练曝光轮数需根据新的 `ready.json` 计算。详细规则见 [SLAI_DATA.md](SLAI_DATA.md)。

本轮SD1.5只需要Pick-a-Pic训练清单。HPDv3先下载解压用于held-out图片/文本隔离及后续评估，之后SD3训练再准备 `hpdv3 --family unfiltered`。不需要此时下载SDXL/SD3或HPSv3大权重。默认训练监控只需PickScore两项资产。

## 7. 申请1卡冒烟，再申请4/8卡单机

准备完数据后在CCI申请 **1张RTX5090 32GB** 做小检查，再申请4/8卡进行多卡冒烟和训练。可以新建实例，也可按平台允许方式调整原实例。都挂同一AFS、使用兼容镜像；新实例重新取得SSH地址。

正式ACP路线：**高性能算力池ACP → 创建任务 → 授权workspace/集群**。历史 `computing_cluster_5090_01g` 有1/2/4/8卡规格，当前需选中集群后查看真实GPU型号和显存。

| 项目 | 第一轮四卡 | 八卡方案 |
|---|---|---|
| worker副本数/节点数 | 1 | 1 |
| 每个副本GPU数 | 4 × RTX5090 32GB | 8 × RTX5090 32GB |
| CPU/主存起点 | 16–32 vCPU、64–128GB | 32–64 vCPU、128GB，按全量预检峰值调整 |
| 每卡训练pair batch | 8 | 8 |
| 梯度累积 | 8 | 4 |
| 全局pair batch | 256 | 256 |

CPU/主存是代码运行的建议起点，不是平台固定规格；全量清单会被每个rank加载，不能只看GPU显存来选主存。选择团队允许的优先级和标准/闲时资源；闲时资源可能抢占，保留完整恢复状态。按第12节的宽区间预留运行时限，并确认自动停止设置。初次推荐4卡2000步对照；不必为最小检查直接申请8卡。

平台框架可用其PyTorch选项，但启动命令只运行一次本项目协调器。不能让平台/脚本额外为同一节点再次包一层torchrun。当前代码只支持单机多卡：**副本数1、每副本4或8卡**；副本数4×每副本1卡是多机布局，不是本文方案。[ACP官方创建说明](https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpUserGuide/acpCreateJob)。

## 8. 在实际GPU上验证环境与小数据恢复

在已分配GPU的CCI/ACP容器中：

```bash
source paths.sh
source "$SCD_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SCD_5090_ENV"
export PYTHONNOUSERSITE=1
export SCD_PYTHON="$SCD_5090_ENV/bin/python"
nvidia-smi
nvidia-smi topo -m
python - <<'PY'
import torch
print('torch',torch.__version__,'runtime',torch.version.cuda,'arch',torch.cuda.get_arch_list())
print('visible GPUs',torch.cuda.device_count())
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i),
          torch.cuda.get_device_properties(i).total_memory/1024**3)
    x=torch.randn(128,128,device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True)
    (x@x.T).float().square().mean().backward()
    torch.cuda.synchronize(i)
    assert torch.isfinite(x.grad).all()
print('BF16 forward/backward passed')
PY
python scripts/make_smoke_data.py \
  --pairs "$PICKAPIC_SD15_PREPARED/pairs.jsonl" \
  --out "$SCD_PREPARED_ROOT/smoke-sd15-v1"
```

普通DPO的单卡两步测试：

```bash
python scripts/make_config.py --family sd15 --method dpo --smoke \
  --model "$SD15_MODEL" --pairs "$SCD_PREPARED_ROOT/smoke-sd15-v1/pairs.jsonl" \
  --out configs/local-smoke-dpo-1gpu.json
bash launchers/train.sh configs/local-smoke-dpo-1gpu.json "$SCD_RUNS_ROOT/smoke-dpo-1gpu-v1" --stop-after-step 1
bash launchers/train.sh configs/local-smoke-dpo-1gpu.json "$SCD_RUNS_ROOT/smoke-dpo-1gpu-v1" \
  --resume "$SCD_RUNS_ROOT/smoke-dpo-1gpu-v1/checkpoints/latest.json"
```

在四卡实例上验证纯K2的DDP和恢复（八卡改两处4为8，并使用新的配置/run名）：

```bash
python scripts/make_config.py --family sd15 --method k2-noise --arm C --smoke --world-size 4 \
  --model "$SD15_MODEL" --pairs "$SCD_PREPARED_ROOT/smoke-sd15-v1/pairs.jsonl" \
  --out configs/local-smoke-k2-4gpu.json
bash launchers/train_ddp.sh configs/local-smoke-k2-4gpu.json "$SCD_RUNS_ROOT/smoke-k2-4gpu-v1" --stop-after-step 1
bash launchers/train_ddp.sh configs/local-smoke-k2-4gpu.json "$SCD_RUNS_ROOT/smoke-k2-4gpu-v1" \
  --resume "$SCD_RUNS_ROOT/smoke-k2-4gpu-v1/checkpoints/latest.json"
```

检查loss/梯度有限、LoRA保存、第二进程恢复到第2步，且各rank退出正常。初始LoRA等同基座时loss接近0.693属正常，不能据此判断训练无效。遇到OOM先减每卡batch并保持全局batch256；配置改变应新开实验，不在原run中改。

## 9. 先测100步，校准你自己的5090速度

以下使用完整训练清单、正式每卡batch8与全局batch256，但关闭评估/诊断，最后保存一次。测速run不作为效果对照，也不从它恢复正式实验。

```bash
python scripts/make_config.py --family sd15 --method dpo --portable-paths --benchmark \
  --world-size 4 --batch-size 8 --effective-batch 256 --out configs/local-bench-dpo-4gpu.json
python scripts/make_config.py --family sd15 --method k2-noise --arm C --portable-paths --benchmark \
  --world-size 4 --batch-size 8 --effective-batch 256 --out configs/local-bench-k2-4gpu.json
bash launchers/train_ddp.sh configs/local-bench-dpo-4gpu.json "$SCD_RUNS_ROOT/bench-dpo-4gpu-v1"
bash launchers/train_ddp.sh configs/local-bench-k2-4gpu.json "$SCD_RUNS_ROOT/bench-k2-4gpu-v1"
python scripts/estimate_runtime.py --run "$SCD_RUNS_ROOT/bench-dpo-4gpu-v1" --steps 2000 6000
python scripts/estimate_runtime.py --run "$SCD_RUNS_ROOT/bench-k2-4gpu-v1" --steps 2000 6000
```

八卡重新测速，不从四卡run恢复。首次可把 `--steps 30` 加到benchmark命令缩短测速。脚本排除前10次更新，用累计 `cost.training_seconds` 差分的中位数估算；峰值是rank0已分配显存，不是整机显存。它不把数据全量校验、模型加载、checkpoint或生成评分算进训练ETA。

## 10. 正式普通DPO与纯K=2对照

四卡配置：

```bash
python scripts/make_config.py --family sd15 --method dpo --portable-paths \
  --steps 2000 --world-size 4 --batch-size 8 --effective-batch 256 \
  --out configs/local-sd15-dpo-4gpu.json
python scripts/make_config.py --family sd15 --method k2-noise --arm C --portable-paths \
  --steps 2000 --world-size 4 --batch-size 8 --effective-batch 256 \
  --out configs/local-sd15-k2-4gpu.json
python train.py --config configs/local-sd15-dpo-4gpu.json --out "$SCD_RUNS_ROOT/sd15-dpo-4gpu-v1" --check-only
python train.py --config configs/local-sd15-k2-4gpu.json --out "$SCD_RUNS_ROOT/sd15-k2-4gpu-v1" --check-only
```

八卡用 `--world-size 8`，并把配置/run文件名中的4gpu改为8gpu；每卡batch8与全局batch256不变，工具自动将累积改为4。保持两组每卡batch相同便于比较。若batch8的纯K2超出32GB，改为batch4，则4卡累积16、8卡累积8；同时为两组生成新的配置与输出名。

两组默认都为512px、BF16、LoRA rank=alpha16、lr1e-5、beta5000、warmup200、seed17、原caption。区别仅在每对图片的普通噪声视图数1或2；**两份正式JSON都没有 `independent_rollout`**。K2-C先平均同一pair的两个margin再计算loss；没有model/oracle扰动，不能把它记录成现有model-K2-C实验。两个视图都反传，但每个全局batch只执行一次optimizer.step，K2的2000步不是4000步。

每组2000步均曝光512000对原始图片；K2多的是随机视图。全量新数据的epoch数由512000 / 新训练对数决定，不能沿用原151418对数据的3.38轮。B/C两组不能从彼此adapter开始，普通DPO与K2都从同一冻结基座初始化。

在交互CCI中，可以用镜像已有的tmux启动并保留日志；若镜像没有tmux，正式运行改用ACP，避免SSH断开导致进程终止。不要向共享系统安装软件。直接启动命令为：

```bash
mkdir -p "$SCD_RUNS_ROOT/logs"
set -o pipefail
bash launchers/train_ddp.sh configs/local-sd15-dpo-4gpu.json "$SCD_RUNS_ROOT/sd15-dpo-4gpu-v1" \
  2>&1 | tee "$SCD_RUNS_ROOT/logs/sd15-dpo-4gpu-v1.log"
# 上一组完成后，再运行第二组；不要在同一4卡分配上并发两组。
bash launchers/train_ddp.sh configs/local-sd15-k2-4gpu.json "$SCD_RUNS_ROOT/sd15-k2-4gpu-v1" \
  2>&1 | tee "$SCD_RUNS_ROOT/logs/sd15-k2-4gpu-v1.log"
```

ACP任务中的启动命令示例（每个方法一个任务，依次运行，或在额度允许时各自申请独立资源）：

```bash
bash -lc 'set -euo pipefail
cd /mnt/afs/<你的目录>/code/DPO
source paths.sh
source "$SCD_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SCD_5090_ENV"
export PYTHONNOUSERSITE=1
export SCD_PYTHON="$SCD_5090_ENV/bin/python"
exec bash launchers/train_ddp.sh configs/local-sd15-dpo-4gpu.json "$SCD_RUNS_ROOT/sd15-dpo-4gpu-v1"'
```

第二组将config和run同时改为k2。ACP必须挂相同AFS，并保留 `configs/local-*.json`：它们不在Git仓库中，但在持久代码目录内。不要依赖训练任务重新git clone后自动存在本地配置。自动重试次数先设0，出现失败先查原因。

本模块协调器自动在训练阶段启动torchrun；外部不要再 `accelerate launch` 或 `torchrun train_ddp.sh`。训练launcher同时设置离线模式，不会申请GPU、下载权重或创建云实例。

## 11. 看效果、恢复和释放资源

默认先生成基座step0分数，再监控200、500、1000、1500、2000步；128条固定Pick-a-Pic test_unique caption，DDIM30/CFG7.5、seed17，PickScore raw cosine。模型、生成和评分分进程运行；**监控只用第0张GPU，其余卡此时空闲**，因此8卡端到端不会必然快一倍。

每200步滚动保存完整续训状态，每1000步及最终步永久保存LoRA。查看：

```text
<run>/pickscore-curve.csv/.json/.png
<run>/metrics-*.jsonl
<run>/training.json
<run>/checkpoints/latest.json
<run>/checkpoints/step-0001000/adapter/
<run>/evaluations/step-0001000/
<run>/evaluation-complete.json
```

两组使用同一面板和采样seed；比较相对各自基座的变化、同一步数的差异和图片，不用训练loss大小替代生成质量判断。训练数据和监控面板身份必须一致。

中断后先确认原进程已退出，再用相同GPU数、配置、代码、环境与挂载路径恢复：

```bash
bash launchers/train_ddp.sh configs/local-sd15-k2-4gpu.json "$SCD_RUNS_ROOT/sd15-k2-4gpu-v1" \
  --resume "$SCD_RUNS_ROOT/sd15-k2-4gpu-v1/checkpoints/latest.json"
```

评估失败同样从latest恢复，流程先补当前点评估。永久adapter不是完整optimizer checkpoint。不能用4卡的latest直接冒充8卡精确续训，也不要在运行中git pull覆盖源文件。

最终评估选未参与监控的Pick-a-Pic剩余test、HPDv3，或者已准备的HPDv2。仅PickScore时不需其他模型：

```bash
python evaluate.py --config configs/evaluate.json --run "$SCD_RUNS_ROOT/sd15-dpo-4gpu-v1" \
  --step 2000 --benchmarks pickapic hpdv3 --metrics pickscore \
  --out "$SCD_RUNS_ROOT/final-dpo-4gpu-v1" --execution standalone --run-label sd15-final-dpo
python evaluate.py --config configs/evaluate.json --run "$SCD_RUNS_ROOT/sd15-k2-4gpu-v1" \
  --step 2000 --benchmarks pickapic hpdv3 --metrics pickscore \
  --out "$SCD_RUNS_ROOT/final-k2-4gpu-v1" --execution standalone --run-label sd15-final-k2
```

完整剩余test可能很大，先查看selection.json的实际prompt数量并预留推理时间。需要四项评分时再按SLAI_DATA下载HPSv2/CLIP-L/aesthetic并准备官方OpenCLIP源码，使用 `--metrics pickscore clip hpsv2 aesthetic`，保证两方法协议相同。

用完在SenseCore **停止CCI实例或结束ACP任务**，检查GPU已释放。退出SSH、关闭VS Code或UU都不会释放实例。保留AFS、代码版本、模型与数据下载记录、配置、训练日志和结果；不要删除仍需恢复的latest/state.pt。

## 12. 四卡/八卡5090的时间预算

当前没有RTX5090实测，不应承诺具体完成时间。现有参考：旧SD1.5普通DPO在2×A800 80GB、每卡32对/累积4/全局256时，2000步的优化耗时39813.8秒，平均约19.9秒/步；旧K2扰动实现单A800的连续日志约66.7秒/步。后一项包含扰动及旧重复前向，**不能当成这里纯K2的测速**。旧MIG上的两步smoke也不能外推整卡吞吐。

以下是按当前纯K2代码、512px、LoRA16、BF16、每卡8对、全局batch256给出的**宽松规划假设**，不是由硬件规格推导出的可靠基准，也不是SLAI实测：

| 单机GPU | 方法 | 假设秒/优化步 | 2000步纯优化预算 | 6000步纯优化预算 |
|---|---|---:|---:|---:|
| 4×5090 32GB | 普通DPO | 5–15 | 约3–9小时 | 约9–25小时 |
| 4×5090 32GB | 纯K2-C，无扰动 | 8–25 | 约5–14小时 | 约14–42小时 |
| 8×5090 32GB | 普通DPO | 3–10 | 约2–6小时 | 约5–17小时 |
| 8×5090 32GB | 纯K2-C，无扰动 | 5–16 | 约3–9小时 | 约9–27小时 |

第一轮两组各2000步顺序执行：四卡可先按约8–23小时纯优化、八卡约5–15小时纯优化安排。**不包括**排队、下载、解压、预处理、全量数据校验、进程反复启动、保存、生成评分和最终大面板评估。这些项可能使端到端耗时明显更长，尤其AFS慢或节点共享时；上述范围也可能被突破。

当前每个训练进程都会校验完整清单和图片身份，自动监控会退出并重启进程；全量新数据、8个rank及频繁阶段切换会放大这部分CPU/AFS成本。不能把小样本MIG日志或峰值TFLOPS直接用于端到端估算。先做100步实际测速；需要精确规划时，再测一段包含200步监控的正式配置并暂停恢复，记录完整wall time。

纯K2共享图像读取、VAE和文本编码，但逐视图串行执行policy/reference并保留两份图，实际倍率取决于UNet利用率、显存、通信及IO；8卡也受CPU/AFS和单卡评估限制。若100步实测普通DPO为10秒/步、纯K2为16秒/步，则2000步纯优化分别约5.6/8.9小时；用 `estimate_runtime.py` 计算自己的机器，而不是固定套用本表。

## 13. 本次验证范围

已通过24项CPU测试：迁移/数据准备13项、独立目录迁移5项、SDXL回归6项；覆盖真实Parquet/图片/分卷夹具、原标签与图片身份、K1/K2配置、打包排除与SHA256、测速计算，以及微型SDXL前向和LoRA保存加载。Python与Bash语法检查通过。这些测试使用现有torch2.3 CPU环境，不能替代5090验证。

5090的torch2.7.1/cu128环境没有在这里安装，也没有连接SLAI、上传GitHub或提交GPU任务。实际完整UNet/NCCL/显存与下载网络，按第8–9节在分配的资源上验证。原torch2.3环境与冻结历史实验不被替换。
