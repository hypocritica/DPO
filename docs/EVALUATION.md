# 评估

训练监控默认使用固定的 128 条 Pick-a-Pic test caption：排除训练 caption，第 0、200、每 500 步和最终步计算 PickScore。完整独立评估默认使用 HPDv2 的 400 条 prompt，与监控集分开。

## 本地路径

直接编辑 `configs/evaluate.json`（SDXL 用 `configs/evaluate-sdxl.json`）的 `benchmarks`，以及 `configs/scorers.json` 中所选指标的模型/权重路径。仅解析被选中的数据集和指标，没有使用的项可以不配置。相对路径按 JSON 所在目录解析。

训练监控只需要 `scorers.pickscore.model`（PickScore 模型）和 `processor`（CLIP-ViT-H/14 processor）。CLIP/HPSv2 的 `open_clip_source` 指向本地官方 HPSv2 checkout 中包含 `open_clip/` 的目录。HPSv2 默认 v2.0，对应 `HPS_v2_compressed.pt`；不能混用 v2.1 权重。

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
