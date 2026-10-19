# 评估

训练监控默认使用固定的 128 条 Pick-a-Pic test caption：排除训练 caption，第 0、200、每 500 步和最终步计算 PickScore。完整独立评估默认使用 HPDv2 的 400 条 prompt，与监控集分开。

## 本地路径

直接编辑 `configs/evaluate.json`（SDXL 用 `configs/evaluate-sdxl.json`）的 `benchmarks`，以及 `configs/scorers.json` 中所选指标的模型/权重路径。仅解析被选中的数据集和指标，没有使用的项可以不配置。相对路径按 JSON 所在目录解析。

训练监控只需要 `scorers.pickscore.model`（PickScore 模型）和 `processor`（CLIP-ViT-H/14 processor）。CLIP/HPSv2 的 `open_clip_source` 指向本地官方 HPSv2 checkout 中包含 `open_clip/` 的目录。HPSv2 默认 v2.0，对应 `HPS_v2_compressed.pt`；不能混用 v2.1 权重。

`configs/scorers.json` 已填写 SLAI 的分类路径，模型根目录是 `/mnt/afs/L202500030/xietianyu/workspace/models`：

| 子目录 | 用途 |
|---|---|
| `stable_diffusion/sd15` | SD1.5 基座，包含 VAE 和文本编码器；DPO reference 复用此基座 |
| `preference/pickscore` | PickScore 权重 |
| `clip/clip-h14` | PickScore processor，以及独立 CLIP 指标的 ViT-H/14 权重 |
| `preference/hpsv2` | HPS v2.0 权重，`source/` 存官方代码 |
| `clip/clip-l14` | LAION aesthetic 使用的 OpenAI CLIP ViT-L/14 |
| `aesthetic/improved-aesthetic-predictor` | LAION aesthetic 的 MLP 评分头及官方代码 |

已有 SD1.5 时，下载四项默认评估需要的模型和官方代码：

```bash
python scripts/download_assets.py download pickscore clip-h14 hpsv2 clip-l14
bash scripts/download_sources.sh hpsv2 /mnt/afs/L202500030/xietianyu/workspace/models/preference/hpsv2/source
bash scripts/download_sources.sh aesthetic /mnt/afs/L202500030/xietianyu/workspace/models/aesthetic/improved-aesthetic-predictor
```

美学评分头是官方仓库内的 `sac+logos+ava1-l14-linearMSE.pth`，没有它仅下载 CLIP-L 还不能算美学分。若暂时仅运行训练监控，可只下载 `pickscore pickscore-processor`；后者只含预处理/tokenizer 小文件，完整 CLIP 指标仍需 `clip-h14`。HPSv3 和 ImageReward 放在 `preference/`，它们的 Qwen/BERT 底座放在 `backbones/`，均保持可选。

## 运行

```bash
python evaluate.py --config configs/evaluate.json --run runs/dpo \
  --step 1000 --out results/hpdv2-step1000 --execution standalone
```

`--step 0` 评估基座，其他步数必须仍有永久 LoRA 或当前滚动模型。默认指标为 PickScore、CLIP、HPSv2、LAION aesthetic；只测 PickScore 加 `--metrics pickscore`。

可选面板和指标可组合：

```bash
python evaluate.py --config configs/evaluate.json --run runs/dpo \
  --step 1000 --benchmarks pickapic partiprompts hpdv3 \
  --metrics pickscore clip hpsv2 aesthetic \
  --out results/comparison --execution standalone
```

可选指标 `hpsv3`、`image_reward` 使用官方本地实现，额外依赖分别见 `requirements-hpsv3.txt`、`requirements-imagereward.txt`。如果使用独立环境，可以给对应 scorer 配置增加 `python` 字段，填写你自己的 Python 路径。

## Prompt 准备

已有 caption 导出文件时：

```bash
python scripts/prepare_benchmarks.py \
  --pickapic /data/pickapic/test_unique.parquet \
  --hpdv2 /data/hpdv2/test.json --train-pairs /data/pairs.jsonl \
  --out /data/evaluation
```

可加 `--parti`、`--hpdv3`、`--imagereward`；这些输入也可以独立准备。输出包含固定监控子集、剩余 Pick-a-Pic test、完整面板和选择记录。HPDv2 的 400 条测试 caption 与官方另一套 3200 条 HPS style benchmark 不同。

## 分数与恢复

PickScore、CLIP、HPSv2 使用 raw cosine，不做 softmax 或乘 100；LAION 使用 improved-aesthetic MLP 输出。HPSv3 使用官方 mu，ImageReward 使用官方标准化 reward。旧 `score.py` 的 scaled PickScore 不能直接与新曲线混合。

同一面板的指标复用同一批图像。结果包含逐图分数、均值和清单；训练目录另外输出 `pickscore-curve.csv/.json/.png`。生成、评分分进程运行，训练多卡监控仅使用第 0 张 GPU。

相同命令、配置和输出路径可以继续未完成的评估。监控失败会暂停训练，训练 `--resume` 先补齐当前点评估。更改模型、prompt 或评分器时使用新的输出目录。

固定监控集用于观察训练过程，不替代独立测试。B/C 的 loss 数值不同，不能仅凭训练 loss 判断图像质量。
