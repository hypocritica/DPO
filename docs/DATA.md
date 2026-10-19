# 模型与数据迁移

读取本地完整Diffusers SD1.5目录：至少包括model_index、UNet、VAE、scheduler、tokenizer和text_encoder。优先复用已验证模型，记录revision；单个`.ckpt`文件不是本包的输入。

训练清单每行显式标明胜负，不从奖励分数推断原人类偏好。图片路径相对清单所在目录，或是绝对路径。原Arrow清单记录本机shard路径与mtime，不能直接复制后修改一个路径就假装身份校验仍然有效。

## 保留已有实验的数据划分

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
