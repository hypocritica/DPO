# Self Correct DPO

SD1.5 / SDXL 的 LoRA DPO 训练。支持普通 DPO、两个噪声视图、停止梯度的 DDIM 扰动，以及 B/C 两种聚合目标；默认 C。支持单机 1/2/4/8 卡 DDP。

## 安装

自行创建并激活 Python 3.10/3.11 环境，然后：

```bash
pip install -r requirements.txt
```

这份依赖包含 RTX 5090 使用的 PyTorch 2.7.1 / torchvision 0.22.1 / cu128，以及训练、数据读取和默认评估所需的包。不管理你的 Conda、目录或缓存。

## 配置路径

直接修改 JSON 配置中的路径即可，不需要 `source paths.sh`：

| 文件 | 需要填写的内容 |
|---|---|
| `configs/model-K2-C.json` | `model.path`：完整 Diffusers 模型目录；`data.pairs`：偏好对 JSONL；`evaluation.prompts`：监控 prompt JSONL |
| `configs/scorers.json` | 默认训练监控只需填写 `pickscore.model` 和 `pickscore.processor` 两个本地目录 |
| `configs/sdxl/model-K2-C.json` | 使用 SDXL 时选择这个模板，并填写自己的基座来源 `model.revision` |

相对路径按所在 JSON 文件的位置解析。训练输出目录由 `--out` 指定。`paths.sh` 只保留四个可选快捷变量，不参与环境安装。

## 训练

可以直接编辑模板后运行：

```bash
python train.py --config configs/model-K2-C.json --out runs/rollout-C --execution standalone
```

也可以用下面的命令生成普通 DPO 配置。路径和卡数由你指定，不需要预设目录结构：

```bash
python scripts/make_config.py --method dpo \
  --model /models/sd15 --pairs /data/pairs.jsonl \
  --monitor-prompts /data/pickapic-monitor.jsonl \
  --world-size 4 --out configs/local-dpo.json
python train.py --config configs/local-dpo.json --out runs/dpo --execution standalone
```

| `--method` | 训练方式 |
|---|---|
| `dpo` | 普通 DPO，K=1，无扰动 |
| `k2-noise` | 两个独立的普通噪声视图，无扰动 |
| `k2-rollout` | 两个独立的 DDIM 扰动视图 |

`--arm B` 为逐视图 loss 平均；默认 `--arm C` 为先平均 margin 再计算 loss。SDXL 增加 `--family sdxl --model-revision <你的基座版本>`。省略 `--method` 沿用模板中的扰动设置，K=1 时 B/C 等价。

多卡仍只运行一次 `train.py`，程序按 `training.world_size` 启动本机 DDP。`make_config.py --world-size` 会保持模板的全局 batch；手动改配置时，全局 batch 为 `batch_size × gradient_accumulation_steps × world_size`。Slurm 分配内省略 `--execution standalone`。

## 保存与评估

- 每 200 个优化步滚动保存最新 LoRA、优化器、LR 和各 rank 随机状态，只保留最新完整恢复槽。
- 每 1000 步及最终步永久保留 LoRA，独立于滚动保存。
- 在第 0、200、每 500 步及最终步，用固定 128 条 Pick-a-Pic test prompt 监控 PickScore。
- 每 1000 步及最终步保留监控图片，其余点保留分数、清单和曲线。

恢复：

```bash
python train.py --config configs/local-dpo.json --out runs/dpo \
  --resume runs/dpo/checkpoints/latest.json --execution standalone
```

任意保留步数的独立评估：编辑 `configs/evaluate.json` 的 prompt 路径和 `configs/scorers.json` 的权重路径，然后运行：

```bash
python evaluate.py --config configs/evaluate.json --run runs/dpo \
  --step 1000 --out results/hpdv2-step1000 --execution standalone
```

默认 HPDv2 与 PickScore、CLIP、HPSv2、LAION aesthetic。可选 Pick-a-Pic、PartiPrompt、HPDv3、ImageReward prompt，以及 HPSv3 / ImageReward 指标；详见 [评估说明](docs/EVALUATION.md)。

## 数据与方法

模型、数据和评分权重自行准备；训练不会自动下载。数据格式与可选处理工具见 [DATA.md](docs/DATA.md)，B/C 与扰动公式见 [METHOD.md](docs/METHOD.md)。[EXPERIMENTS.md](docs/EXPERIMENTS.md) 和归档配置保留已有实验定义，不包含实验结果。

`tests/` 仅保留 B/C、扰动时间采样、SDXL、保存恢复和 DDP 的核心 CPU 检查：

```bash
python -m unittest discover -s tests -v
```
