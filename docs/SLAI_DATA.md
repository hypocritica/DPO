# SLAI：统一路径、下载与数据准备

本流程采用用户确认的 **Pick-a-Pic v2 + HPDv3**。HPDv2 仅保留为可选的历史400-prompt评估集。只迁移本代码目录，在SLAI直接下载；无需复制旧集群的Arrow缓存。所有命令在 `self_correct_dpo` 根目录执行。

## 唯一的路径入口

编辑根目录 `paths.sh`，将第一行存储根目录改成你自己的持久、可写AFS目录。默认其他路径都由它推导；模型、数据、缓存和实验结果在代码目录外。每次新开终端、CCI或ACP启动脚本中，先执行：

```bash
source paths.sh
```

路径可以单独覆盖，例如把已有SDXL目录直接填入 `SDXL_MODEL`。不要把凭据写进该文件。需要认证的模型先使用平台允许的 Hugging Face 登录方式，下载器沿用已有认证，不打印token。

```text
<SCD_STORAGE_ROOT>/
  models/                         SD1.5、SDXL、评分模型等
  data/raw/pickapic-v2/            官方Parquet及下载清单
  data/raw/hpdv3/                  官方JSON、图片分卷、解压后的images/
  data/images/pickapic-v2/         按SHA256去重的原始图片字节
  data/prepared/pickapic-v2/
    sd15-v1/                      pairs.jsonl、排除记录、audit、ready
    sdxl-v1/                      独立文本长度筛选，共用图片
  data/prepared/hpdv3/
    sd15-v1/、sdxl-v1/             相对路径直接引用raw/hpdv3/images，不复制图片
    unfiltered-v1/                给独立SD3实现的数据清单，不应用SD1.5的77-token过滤
  data/prepared/evaluation-v1/     固定监控与最终评估prompt
  sources/                        可选评分器上游源码
  cache/                          HF、datasets、torch缓存
  runs/                           每个实验使用独立目录
```

`paths.sh`统一控制位置；下载revision分别由其中的 `*_REVISION` 和下载器 `--revision` 控制。数据样本范围、置信度阈值等实验选择由命令显式给出，并保留审计记录。

## 先查看，再下载

使用项目自己的环境；基础依赖见 `requirements.txt`，包含Parquet处理所需库。下载阶段不要设置 `HF_HUB_OFFLINE=1`；训练launcher才会开启离线模式。

```bash
python scripts/download_assets.py list
# 只查询元数据、显示总下载量/还需下载量/目的目录空闲空间，不下载大文件。
python scripts/download_assets.py plan pickapic-v2 hpdv3 sd15 sdxl pickscore clip-h14
# 确认容量与网络后，在SLAI下载。
python scripts/download_assets.py download pickapic-v2 hpdv3
python scripts/download_assets.py download sd15 sdxl pickscore clip-h14
```

默认下载完整Pick-a-Pic v2，**不是旧集群约20万对缓存的副本**。官方PickScore项目说明v2超过100万条，图像位于 `jpg_0/jpg_1` 字节列：[来源](https://github.com/yuvalkirstain/PickScore#download-the-pick-a-pic-dataset)。不要使用只有图片URL的版本。

HPDv3来自 [MizzenAI/HPDv3](https://huggingface.co/datasets/MizzenAI/HPDv3/tree/main)，核对时仓库总量约141 GB；默认只选 `train.json`、`test.json` 和 `images.tar.gz.*`。`all.json`、补充Pick-a-Pic/ImageReward的pair信息和benchmark生成图不默认下载。精确选中文件大小以 `plan` 实时输出为准。

第一次下载将分支解析为commit，写入目的目录 `.scd-download.json`。后续重跑固定同一commit并复用Hub的部分下载；不把移动的main混进已有目录。不覆盖其他来源的非空目录。模型只选一套Diffusers组件权重，尽量避免同时下载根目录整合checkpoint、ONNX和FP16副本。SD1.5下载源是明确标示的 [Runway旧仓库镜像](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)，并不冒充原Runway revision。

如果只想先下载少量Pick-a-Pic训练分片，另设 `PICKAPIC_RAW` 到新目录，再显式使用 `--train-shards 1`。该子集会记录在下载清单中，不能称为全量或原151418对训练数据。正式数据换回另一独立目录，不能在准备好训练清单后原地扩充源文件。

容量预算还应加上：Pick-a-Pic导出的去重图片、HPDv3解压后的图片、模型与checkpoints。下载器显示的大小**不包括**这些额外空间；不要把141 GB当成HPDv3的完整运行磁盘预算。保留压缩分卷有利于校验和重建；脚本不自动删除它们。SLAI需自行核实AFS容量/配额，`disk_usage`只能看到文件系统报告的剩余空间。当前集群任何大写入前仍遵守 `df -h /irip` 等本地约定。

## 解压与预处理

大规模处理在SLAI CPU worker/容器内执行；若在Slurm集群运行，必须使用CPU计算节点，不在登录节点处理全量。首次先用小分片验证，不需要为Parquet和JPEG导出申请GPU。全量Python分组需要较多主存，可从32–64GB主存的CPU任务检查实际峰值后调整；这不是已测量的全量内存保证。

```bash
python scripts/download_assets.py extract-hpdv3
python scripts/prepare_downloaded_data.py pickapic --family sd15
python scripts/prepare_downloaded_data.py pickapic --family sdxl
# 准备HPDv3用于偏好训练；训练SD1.5/SDXL时分别使用对应清单。
python scripts/prepare_downloaded_data.py hpdv3 --family sd15
python scripts/prepare_downloaded_data.py hpdv3 --family sdxl
# SD3的数据准备：保留长caption，后续由实际SD3训练器检查其编码器限制。
python scripts/prepare_downloaded_data.py hpdv3 --family unfiltered
```

HPDv3解压器直接流式拼接分卷，避免额外落一个约140GB的合并压缩包。拒绝越界路径、链接、重复文件和覆盖已有images目录。解压中断保留原分卷和部分images；先由用户检查、处理不完整输出，再重试，脚本不擅自删除大目录。

Pick-a-Pic处理保留caption和原始硬偏好标签；图像只按SHA256去重落盘，不缩放、不重新编码。SD1.5检查自身CLIP tokenizer，SDXL检查两个CLIP tokenizer，不截断原训练caption。图片在训练阶段才转RGB、缩放到配置分辨率。不会预先生成两份512px/1024px图片。

HPDv3按官方约定将 `path1` 作为winner、`path2`作为loser，并保留完整原记录，包括 `choice_dist`、`confidence`、`model1/model2`；不根据奖励分数或票数重排。默认不额外按置信度筛选；若确实需要，显式传 `--min-confidence 0.95` 并更换输出版本，该选项也会排除confidence为空的记录。[官方格式](https://github.com/MizzenAI/HPSv3#pairwise-training-data-format)。补充的 `pickapic.json`、`imagereward.json` 不包含对应图片，不自动混入本流程。

两条处理路径先排除两个官方数据集的validation/test原文与图片泄漏，再对剩余train按归一化caption、图片UID/哈希连通分量划分，seed17，约90%/5%/5%，去除平局、重复对、冲突投票及相同图片。这里的validation/test是新建的本地划分，**不是官方test**。只保证可验证的字符串/图片隔离；未知语义同义改写未经过完整人工审核。独立实验之间不会自动混合两份训练数据。

每个输出目录包含 `audit.json`、`excluded.jsonl`、`ready.json` 和 `pairs.jsonl`。只有存在 `ready.json` 才表示准备成功。处理拒绝覆盖已有版本；中断时图片池可以复用，修改 `paths.sh` 中的prepared版本目录后重试。原始数据和共享图片移动时保持整个data目录的相对布局。

## 固定监控和评估面板

```bash
python scripts/prepare_downloaded_data.py benchmarks --family sd15
```

从官方Pick-a-Pic `test_unique`冻结128条监控caption，同时输出剩余test与HPDv3 test prompt。所有基线和方法共用该面板。若本轮改用HPDv3训练，可在第一次准备面板时添加 `--dataset hpdv3`；准备脚本已从两份训练数据排除这些官方held-out项。面板不足128条会报错，不会回退到训练集。

HPDv2可选：`download_assets.py download hpdv2`只下载test metadata；在准备面板之前执行，会额外输出原400条HPDv2 prompt。`hpdv2-images`才下载其训练/测试图片包，不是当前默认训练数据。HPDv2、HPDv3以及HPSv3评分模型是不同资产。

旧独立评估入口仍默认HPDv2；本流程使用HPDv3时显式选择它，例如训练产生永久1000步LoRA后执行：

```bash
python evaluate.py --config configs/evaluate.json \
  --run "$SCD_RUNS_ROOT/sd15-C-001" --step 1000 --benchmarks hpdv3 \
  --metrics pickscore --out "$SCD_RUNS_ROOT/sd15-hpdv3-step1000" \
  --execution standalone --run-label slai-hpdv3
```

SDXL对应使用 `configs/evaluate-sdxl.json` 与其run目录；准备完整四项模型后可传 `--metrics pickscore clip hpsv2 aesthetic`。这里的 `hpsv2` 是评分权重，不需要HPDv2训练图片。

## 生成路径可集中修改的训练配置

```bash
python scripts/make_config.py --family sd15 --portable-paths \
  --steps 6000 --out configs/local-slai-sd15.json
python scripts/make_config.py --family sdxl --portable-paths \
  --steps 6000 --out configs/local-slai-sdxl.json
# 也可以给SDXL选择HPDv3训练清单：
python scripts/make_config.py --family sdxl --dataset hpdv3 --portable-paths \
  --out configs/local-slai-sdxl-hpdv3.json
```

生成配置保留 `${SD15_MODEL}` / `${SDXL_MODEL}`、各模型对应的prepared路径、`${MONITOR_PROMPTS}` 和 `${SCORERS_CONFIG}`。之后改 `paths.sh` 并重新source即可解析到新位置，不需逐个改JSON。基座revision从下载记录读取并固定；变更模型、数据版本或恢复时的路径身份不能伪装成原实验，另开实验目录。旧的绝对路径生成方式和冻结实验不变。

```bash
source paths.sh
python train.py --config configs/local-slai-sd15.json \
  --out "$SCD_RUNS_ROOT/sd15-C-001" --check-only
# 已获得GPU的CCI/ACP容器内：
bash launchers/train.sh configs/local-slai-sd15.json "$SCD_RUNS_ROOT/sd15-C-001"
```

`--check-only`仍会验证全量图片身份，属于完整CPU数据预检，不是GPU训练验证。这里只提供SD1.5/SDXL配置生成；`download_assets.py ... sd35`和HPDv3 unfiltered只是SD3.5资产/数据准备，不表示现有B/C训练器已完成SD3适配或GPU显存验证。

## 评分模型按需下载

训练自动监控只需要 `pickscore clip-h14`。完整四项评价额外准备：

```bash
python scripts/download_assets.py plan hpsv2 clip-l14
python scripts/download_assets.py download hpsv2 clip-l14
bash scripts/download_sources.sh hpsv2 aesthetic
```

源码脚本下载官方HPSv2 OpenCLIP实现和aesthetic predictor（其Git仓库含MLP权重），记录checkout commit，拒绝覆盖已有源码；不把上游旧依赖整份安装进训练环境。`paths.sh`已指向对应文件。评分环境仍按 [EVALUATION.md](EVALUATION.md) 安装项目自己的依赖。

可选 `hpsv3 hpsv3-base`、`image-reward image-reward-tokenizer` 资产也支持plan/download；配套源码用 `download_sources.sh hpsv3 image-reward`，环境Python位置集中填在 `paths.sh`。必须按 [EVALUATION.md](EVALUATION.md) 核对源码配置、底座、奖励权重组合，不能仅因权重下载完成就声称评分已验证。它们不因使用HPDv3训练数据而自动启用。

本流程已用小型本地Parquet、图像、投票和分卷压缩包做CPU测试。未在当前服务器下载真实全量数据/权重，未连接SLAI，未提交GPU任务。首次SLAI实跑应验证少量真实数据、两步训练及暂停恢复，再启动正式预算。

本次完整CPU回归运行59项测试，新增8项数据/下载/迁移测试通过；已有3个多进程测试方法的6个分布式场景因沙箱禁止localhost socket而报 `Operation not permitted`，未在这里验证Gloo/DDP。其余测试通过。该限制不作为真实SLAI GPU/NCCL验证的替代。
