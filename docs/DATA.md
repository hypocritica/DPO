# 模型与数据迁移

资源路径统一在 `configs/resources.json`；迁移后的数据集根目录为 `/mnt/aoss/xietianyu/datasets/datasets`，Pick-a-Pic 原始文件为其下的 `pickapicv2/data`。原始文件需先生成 `pairs.jsonl` 和固定评估面板，准备命令见根目录 [command.md](../command.md)。

读取本地完整Diffusers SD1.5目录：至少包括model_index、UNet、VAE、scheduler、tokenizer和text_encoder。优先复用已验证模型，记录revision；单个`.ckpt`文件不是本包的输入。

训练清单每行显式标明胜负，不从奖励分数推断原人类偏好。图片路径相对清单所在目录，或是绝对路径。原Arrow清单记录本机shard路径与mtime，不能直接复制后修改一个路径就假装身份校验仍然有效。

## 保留已有实验的数据划分

旧实验使用本地 20 万条、98 个 Arrow 分片缓存，先按规范化 caption、保守词序/同义词族及图片 UID/SHA 的连通分量划分，seed17；lighthouse/bottle/armchair/vase 诊断域固定在 test。去除无效偏好对和超过 CLIP 长度的 caption 后保留 169,069 对：train 151,418、validation 8,689、test 8,962。缓存的最初下载仓库没有完整记录，不能据内容吻合断言下载自某个仓库。

原数据迁移最稳妥的方式是导出图片，保留原split、caption、pair_id、偏好标签和图像SHA256：

```bash
# 只在获得资源的计算节点上运行大规模导出；不会在本次整理中执行。
# 原集群上先检查 df -h /irip 与目标存储空间，复用已有数据。
python scripts/export_data.py --manifest /original/pairs.jsonl --out /new/storage/pickapic-portable
```

命令拒绝覆盖已有目录，输出`images/`、相对路径`pairs.jsonl`、`export.json`。文件扩展名`.img`保留原始图片字节，Pillow按内容解码；不重新编码图片。输出图片总量可能很大，应在迁移前自行估算容量；工具会检查目标剩余空间。导出中断会保留不完整目录，重试使用新目录。

原Arrow字段通过datasets读取，不依赖相邻实验源码。导出前后校验图像SHA256，并验证所有split。数据集和模型不上传到代码GitHub仓库。单独迁移原diagnostic目录，保留`prompts.jsonl`及`suite.json`；评估prompt也应保留原内容。

## 新数据集的离线整理

将图片和原始投票导出为JSONL，例如：

```json
{"pair_id":"p1","caption":"a red cube","image_0":"images/a.jpg","image_1":"images/b.jpg","label_0":1,"label_1":0,"image_0_uid":"a","image_1_uid":"b","split":"train"}
```

提供JSON语义组映射，例如`{"a red cube":"red-cube"}`，将已知同义改写映射到同组：

```bash
python scripts/prepare_pickapic.py --raw /data/raw.jsonl --groups /data/groups.json \
  --revision YOUR_DATASET_REVISION --out /data/prepared-v1
```

这一步会**新建划分**：按caption组、图像UID/哈希的连通分量分组，seed17，约90%/5%/5%；保留原始source_split作为来源记录，不沿用它作为最终split。去除平局、重复图像对、冲突投票，禁止跨split共享图像/已知prompt组。未知语义同义仍需要人工检查。这条路径适合新实验，复现已有实验应使用上面的原清单导出方式。

训练配置默认要求原始人类偏好证据及非空validation。不要用合成标签替代正式训练数据。轻量测试中的合成图像只用于验证数据工具，不随仓库提供训练数据。

## 可选下载与处理工具

已经有模型和 `pairs.jsonl` 时无需执行这些步骤。需要下载时，逐个指定输出目录：

现在 Pick-a-Pic 下载源使用替代仓库 [liuhuohuo2/pick-a-pic-v2](https://huggingface.co/datasets/liuhuohuo2/pick-a-pic-v2)，其中保留 train、validation/test 及 test_unique 分片，图片位于 jpg_0/jpg_1。重新构建沿用下面的新流程，不保证与旧 20 万条缓存或 151,418 对训练集相同。下载工具记录替代仓库的 commit；全量约 335 GB，可先用 `--train-shards N` 下载明确记录的训练分片子集（这不是原 20 万条缓存）。

```bash
python scripts/download_assets.py download sd15 --out /models/sd15
python scripts/download_assets.py download pickapic-v2 --out /data/pickapic-raw
python scripts/download_assets.py download hpdv3 --out /data/hpdv3
python scripts/download_assets.py extract-hpdv3 --out /data/hpdv3
```

将原始 Parquet 或 HPDv3 标注转成训练清单：

```bash
python scripts/prepare_downloaded_data.py pickapic --family sd15 \
  --raw /data/pickapic-raw --model /models/sd15 \
  --images /data/pickapic-images --out /data/pickapic-prepared
python scripts/prepare_downloaded_data.py hpdv3 --family sd15 \
  --raw /data/hpdv3 --model /models/sd15 --out /data/hpdv3-prepared
```

如果跨 Pick-a-Pic / HPDv3 做比较，处理 Pick-a-Pic 时传 `--hpdv3 /data/hpdv3`，处理 HPDv3 时传 `--pickapic /data/pickapic-raw`，排除另一数据集官方 held-out 的重合。未提供另一数据集时仅执行本数据集的隔离规则。自己下载且没有下载记录的数据须用 `--revision` 写明来源版本。

处理输出包含 `pairs.jsonl`、`ready.json` 和审计清单。保留原人类胜负，过滤平局和冲突，不用奖励模型重标标签。大规模下载和预处理在已分配资源上执行。
