# Self Correct DPO

首次从私有GitHub迁移并使用RTX5090，按 [完整SD1.5部署与对照指南](docs/SLAI_SD15_5090_GUIDE.md) 执行。5090使用独立 `requirements-5090.txt`；下面的torch2.3/NGC24.04环境说明用于旧GPU方案。

迁入SLAI后，统一编辑根目录 [paths.sh](paths.sh)。Pick-a-Pic v2、HPDv3与模型的下载、预处理和可迁移配置命令见 [SLAI数据准备](docs/SLAI_DATA.md)。下载与预处理是显式步骤，训练和评估入口继续只读取本地资产。

独立的 **SD1.5 / SDXL** LoRA 偏好训练与评估代码。当前只启用 **B/C，默认 C**：B 先算每个视图的 loss 再平均；C 先平均 margin 再算 loss。每个视图可使用 model/oracle 的停止梯度 DDIM 扰动状态。D 已从训练和损失入口移除。

省略模型类型时仍使用原SD1.5配置。SDXL通过 `make_config.py --family sdxl` 创建配置，使用双文本编码器、1024px训练和独立评估模板，具体命令见 [SDXL.md](docs/SDXL.md)。

单机2/4/8卡通过 `make_config.py --world-size 2`、`4` 或 `8` 与 `launchers/train_ddp.sh` 启动，支持扰动训练、自动评估切换和各rank的恢复，见[多卡说明](docs/DISTRIBUTED.md)。默认模板仍为单卡，生成配置时默认保持原全局batch。

旧实验的源码、环境、作业及输出不受本目录修改影响。仓库不包含历史实验结果、模型或训练数据。历史 A/D 配置仅留在 `docs/legacy/` 作来源记录，当前入口拒绝运行。

## 默认实验节奏

| 项目 | 默认设置 |
|---|---|
| 训练监控 | 第 0、200、500、1000、1500、2000…步，以及最终步 |
| 监控数据 | 固定随机抽取 128 条去重的 Pick-a-Pic test caption；排除与训练 caption 重合的条目 |
| 监控生成 | 每条 1 个固定 seed=17；512px、BF16、DDIM30、CFG7.5 |
| 监控指标 | PickScore raw cosine；每个点保存逐图分数、均值及相对基座的配对变化 |
| 永久 LoRA | 每 1000 步及最终步，不按数量淘汰 |
| 生成图保留 | 第 0 步、每 1000 步及最终步；其余点评分成功后清理生成 PNG，保留分数和清单 |
| 最新模型与恢复状态 | 每 200 步滚动保存 LoRA、优化器、LR 和各 rank RNG；只保留最新一份；评估前、正常中断及最终步也保存 |
| 独立评估 | 任意保留步数；默认 HPDv2 test 400 prompts，PickScore、CLIP、HPSv2、LAION aesthetic 四项 |

训练会自动分阶段：保存退出训练进程 → 生成进程 → 逐个评分进程 → 恢复训练。这样训练模型、生成模型和奖励模型不会同时占用显存。第一次训练阶段创建 run 后会补测冻结基座的第 0 步分数；该分数与训练后的 adapter 无关。

## 代码结构

```text
train.py / launchers/train*.sh     单卡/DDP训练与自动监控入口
render.py / score.py               原图片生成、旧式单项评分入口
evaluate.py                       按 checkpoint 评估指定 benchmark
src/self_correct_dpo/
  models.py / training.py          SD1.5/SDXL + LoRA、优化循环
  architecture.py                 SDXL base、双编码器与本地VAE检查
  independent.py / joint.py       扰动、有效标签与多视图构造
  formal.py / losses.py           B/C 目标和 margin
  pipeline.py                    训练暂停、评估、恢复、曲线
  benchmarks.py                  离线 prompt 提取、固定子集
  evaluation.py / metrics.py      生成与多指标评分调度、断点恢复
  scorers.py / optional_scorers.py 四项默认指标、可选 HPSv3 / ImageReward
  checkpoints.py                 每千步 LoRA 和滚动恢复管理
configs/model-K2-C.json            默认主方法（B 有对应模板）
configs/sdxl/model-K2-C.json        SDXL 主方法（B 有对应模板）
configs/evaluate.json              默认 HPDv2 独立评估
configs/evaluate-sdxl.json         SDXL 1024px / Euler50 独立评估
configs/scorers.json               本地评价模型路径
scripts/                          数据迁移、prompt 和配置准备
tests/ / docs/                    CPU 检查、方法和部署说明
```

## 环境

先按 [SLAI.md](docs/SLAI.md) 获得 CCI/ACP GPU 资源，使用自己的环境与目录：

```bash
# 在所选 NGC24.04 镜像内，继承其 NVIDIA torch/torchvision 构建。
python -m venv --system-site-packages .venv-slai23
source .venv-slai23/bin/activate
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install --no-deps -e .
```

模型、数据和评分权重均从本地加载。训练器和评估器不自动下载。`requirements-eval.txt` 适配官方 HPSv2 的本地 OpenCLIP 源码，不要整份安装 HPSv2 仓库的旧依赖以免改变本项目的 protobuf 等版本。

截图镜像的CUDA12.4来自NVIDIA定制torch构建，不存在对应的官方torch2.3/cu124 pip安装组合。上述项目venv保留GPU构建，再安装本项目依赖，不直接替换原镜像的torch。其他机器新建环境时先安装与实际GPU/驱动匹配的torch/torchvision。独立Conda兼容测试使用torch2.3 CPU包，不能当作SLAI GPU镜像的逐项复刻。

## 准备监控和完整评估的 prompt

输入官方 Pick-a-Pic `test_unique` 的本地 Parquet/HF 目录或 caption 导出，以及 HPDv2 `test.json` / Linear-DPO `hpsv2_test400.txt`：

```bash
python scripts/prepare_benchmarks.py \
  --pickapic /data/pickapic/test_unique.parquet \
  --hpdv2 /data/HPDv2/test.json \
  --train-pairs /data/training/pairs.jsonl \
  --out data/evaluation-v1
```

输出固定的 `pickapic-monitor.jsonl`、完整 Pick-a-Pic test、排除监控子集后的剩余 test、`hpdv2-test.jsonl` 和带 SHA256/seed 的选择记录。可选 `--parti` 加入 PartiPrompt TSV，`--hpdv3` 加入 HPDv3 test JSON，`--imagereward` 加入 ImageReward test prompt 导出；这些输入也能单独准备。HPDv2 必须核对为 400 条测试 caption，不能换成 HPS 官方另一套 3200 条 benchmark。只做 caption 提取，不复制偏好图像。

`test_unique` 与本项目原先自行划分的 test 是不同来源。工具不从训练 split 抽监控 prompt；归一化字符串去重不等于语义去重，未知同义改写需另审查。反复使用的 test 子集在实验中应称监控集，最终报告使用 HPDv2 或未用于监控的 Pick-a-Pic 部分。

## 训练

模型需为完整本地 Diffusers SD1.5；偏好数据格式和迁移见 [DATA.md](docs/DATA.md)。先准备 PickScore：

```bash
export MODEL_PATH=/your/models/stable-diffusion-v1-5
export PAIRS_PATH=/your/data/pickapic/pairs.jsonl
export MONITOR_PROMPTS="$PWD/data/evaluation-v1/pickapic-monitor.jsonl"
export SCORERS_CONFIG="$PWD/configs/scorers.json"
export PICKSCORE_MODEL=/your/models/PickScore
export PICKSCORE_PROCESSOR=/your/models/CLIP-ViT-H-14-laion2B-s32B-b79K

python scripts/make_config.py --model "$MODEL_PATH" --pairs "$PAIRS_PATH" \
  --monitor-prompts "$MONITOR_PROMPTS" --scorers-config "$SCORERS_CONFIG" \
  --steps 6000 --out configs/local-train.json
python train.py --config configs/local-train.json --out runs/C-seed17-001 --check-only
# 在已分配的 GPU 容器或 Slurm allocation 中执行：
bash launchers/train.sh configs/local-train.json runs/C-seed17-001
```

默认 K=2，512px，batch8 × accumulation32 = 有效 batch256，LoRA rank=alpha=16，beta5000，lr1e-5，warmup200，seed17。B 使用 `--arm B`。调整显存时保持 batch×accumulation 并记录变化。`--smoke` 生成两步小配置，会关闭监控，不需要评分权重。

当前训练保留所有视图的策略计算图，C 先平均同一图片对的 margin，再计算 DPO loss 并统一反传；B 则平均各视图的 loss 后反传。每个视图不再额外进行一轮无梯度 policy 评分，rollout 与 reference 仍停止梯度。该方式增加激活显存，各视图前向仍逐个执行；速度与峰值显存需在目标GPU测量。step仍表示完成梯度累积后的一次参数更新。旧实验由冻结源码继续负责，不能用新源码直接恢复旧run。

恢复整套训练/评估流程：

```bash
bash launchers/train.sh configs/local-train.json runs/C-seed17-001 \
  --resume runs/C-seed17-001/checkpoints/latest.json
```

两条保存计划独立运行：`training.checkpoint_every=200` 更新 `checkpoints/latest.json` 指向的完整模型与续训状态，新状态提交后才清理旧恢复槽；`training.lora_checkpoint_every=1000` 永久保留每千步及最终步的 LoRA。第1000/2000…步两者都会保存，后续滚动更新不删除永久快照。步数均指完成梯度累积后的全局优化步，单卡和2/4/8卡一致。

上述默认适用于新模板及新生成的配置；已冻结的旧实验继续使用原配置和保存节奏。

评估失败时停在该点，不跨过去继续训练；修复后同样用 `--resume`，先补完当前点评估再继续。若最终训练完成但评分尚未完成，也可用同一命令补完。保持配置、源码、数据及评价权重身份不变。TERM/INT 会向当前子进程转发；强制 SIGKILL 只能恢复到最近已提交的完整状态。

输出包含：

```text
run/
  training.json, metrics-*.jsonl, tensorboard/
  checkpoints/step-0001000/adapter/ ...
  checkpoints/latest.json          # 每200步更新，指向最新完整恢复槽
  checkpoints/resume-<step>-<id>/  # adapter/ + state.pt；滚动只保留一份
  evaluations/step-0000200/        # 分数/配置/清单，非保留点不留 PNG
  evaluations/step-0001000/        # 分数/配置/清单及生成 PNG
  pickscore-curve.csv/.json/.png
  evaluation-complete.json         # 全部训练监控完成后才写入
```

曲线的配对置信区间按 prompt bootstrap，先平均同 prompt 的 seeds；不覆盖训练 seed 的不确定性。监控分数上涨是偏好/对齐指标的变化，不单独证明一般知识能力提升。

## 指定步数的独立评估

完整路径设置、版本口径见 [EVALUATION.md](docs/EVALUATION.md)。配置好四个 scorer 后：

```bash
export HPDV2_PROMPTS="$PWD/data/evaluation-v1/hpdv2-test.jsonl"
python evaluate.py --config configs/evaluate.json \
  --run runs/C-seed17-001 --step 1000 --out runs/hpdv2-C-step1000 \
  --execution standalone --run-label slai-hpdv2
```

默认 HPDv2 400 prompts、每条 1 图、PNDM50、CFG7.5、BF16、seed17，计算 PickScore、ViT-H/14 CLIP、HPS v2.0、improved LAION aesthetic。`--step 0` 评估基座；任何永久保留的 step 都可评估；只测一项可加 `--metrics pickscore`。Slurm 上省略 standalone 参数。

可选评估集为 `pickapic`、`partiprompts`、`hpdv3`、`imagereward`；可选指标为 `hpsv3`、`image_reward`。数据集与指标独立选择，默认均不额外运行。例如配置好相应本地路径后：

```bash
python evaluate.py --config configs/evaluate.json --run runs/C-seed17-001 \
  --step 1000 --benchmarks pickapic partiprompts hpdv3 \
  --metrics pickscore clip hpsv2 aesthetic hpsv3 image_reward \
  --out runs/optional-C-step1000 --execution standalone --run-label slai-optional
```

多个评估集各自生成图片和保存分数，同一评估集的所有指标复用同一批图片。HPSv3/ImageReward 使用官方本地实现，可各自指定独立 Python；环境与权重配置见 [EVALUATION.md](docs/EVALUATION.md)。未选用的路径不会被解析。

新评估的 PickScore/CLIP/HPS 是 raw cosine，无 softmax、无 ×100。旧 `score.py --kind pickscore` 是 scaled PickScore 兼容入口，不能把两种曲线直接混合。HPS v2.0 与 v2.1 也分开标注。此包实现指标和可控协议，不声称与 Linear-DPO 论文数值或其全局 RNG 流逐位一致。

## 验证与历史定义

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -v
```

测试不需要真实模型/数据/GPU；本次不提交训练或推理作业。迁移后仍需在实际 GPU 上验证两步训练、暂停恢复及少量生成评分。

历史 12 份配置保留来源记录：9 份 B/C 配置在 `configs/historical`，旧普通 A、rollout A、D 在 `docs/legacy`。它们保留原超参数，不自动改成新监控节奏。新实验使用上面的主模板。来源见 [PROVENANCE.md](docs/PROVENANCE.md)，旧定义见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。本包不能接管旧冻结作业的 optimizer checkpoint，已有作业仍由原代码负责。
