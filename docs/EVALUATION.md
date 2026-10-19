# 训练监控与独立评估

SD1.5使用 `configs/evaluate.json`；SDXL使用 `configs/evaluate-sdxl.json`（1024px、Euler50、CFG5.0）。SDXL的基座/VAE从训练记录恢复，组件与部署见[SDXL.md](SDXL.md)。两种模型共享下文的数据集和指标选择，默认仍仅HPDv2加四项指标。

## 来源与默认指标

已核对 [Linear-DPO 的 validate.py](https://github.com/Whynot0101/Linear-DPO/blob/663179c7adbbbd2d77b97b5841534447eb291ebd/validation/validate.py) 及同一 commit 的 rewards 实现。此处独立实现离线适配和可恢复流程，没有复制上游训练框架。

| 名称 | 权重/实现 | 本包输出 |
|---|---|---|
| pickscore | PickScore_v1，LAION ViT-H/14 processor | 配对归一化图文 embedding 的 cosine |
| clip | 原 LAION CLIP ViT-H/14，官方 HPSv2 OpenCLIP 预处理 | 图文 cosine |
| hpsv2 | 默认 `HPS_v2_compressed.pt`，明确 v2.0 | 图文 cosine，不乘100 |
| aesthetic | OpenAI CLIP ViT-L/14 + `sac+logos+ava1-l14-linearMSE.pth` | improved-LAION MLP 原始输出 |

Linear-DPO 的 [PickScore 实现](https://github.com/Whynot0101/Linear-DPO/blob/663179c7adbbbd2d77b97b5841534447eb291ebd/rewards/pick_score.py) 默认不乘 logit_scale；其 [HPS 实现](https://github.com/Whynot0101/Linear-DPO/blob/663179c7adbbbd2d77b97b5841534447eb291ebd/rewards/hpsv2.py) 默认 v2.0。本包保持这两个口径。审查过的 scaler 源文件 SHA 已记录在 `evaluation_sources.json`。

Aesthetic 使用 [improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)，结构是归一化768维 CLIP 特征 → 1024 → 128 → 64 → 16 → 1，带原 dropout、无额外激活，eval 模式关闭 dropout；不是 LAION v1 的单 Linear 权重。CLIP/HPS 使用官方 HPSv2 OpenCLIP 的 `light_augmentation=True` 测试预处理，支持全量 state_dict 严格加载，避免上游 wrapper 隐式下载。

## 本地模型路径

先复用已有模型；此包不下载大权重。独立评估前设置：

```bash
export PICKSCORE_MODEL=/your/models/PickScore
export PICKSCORE_PROCESSOR=/your/models/CLIP-ViT-H-14-laion2B-s32B-b79K
export HPS_OPEN_CLIP_SOURCE=/your/sources/HPSv2/hpsv2/src
export CLIP_H14_CHECKPOINT=/your/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
export HPSV2_CHECKPOINT=/your/models/HPSv2/HPS_v2_compressed.pt
export AESTHETIC_CLIP_MODEL=/your/models/openai/clip-vit-large-patch14
export AESTHETIC_CHECKPOINT=/your/models/aesthetic/sac+logos+ava1-l14-linearMSE.pth
export SCORERS_CONFIG="$PWD/configs/scorers.json"
export HPDV2_PROMPTS="$PWD/data/evaluation-v1/hpdv2-test.jsonl"
```

`HPS_OPEN_CLIP_SOURCE` 必须包含 `open_clip/__init__.py`，来自 [官方 HPSv2 仓库](https://github.com/tgxs002/HPSv2)。不需要把整个 HPSv2 的旧依赖安装进来；`requirements-eval.txt` 提供当前用到的可选依赖。四个默认模型均以 FP32 评分；生成模型先退出，再一次加载一个 scorer。

仅训练监控或 `--metrics pickscore` 时只需要 PickScore 两个路径；未使用 scorer 的环境变量不会被解析。所有依赖安装到自己的项目环境。

需要 v2.1 时另存本地 scorer 配置，同时将 `version` 改成 `v2.1`，并指向明确命名的 `HPS_v2.1_compressed.pt`，使用新输出目录。默认不做这项变更；不同版本分数不能直接混作一条曲线。

## Prompt 输入

`prepare_benchmarks.py` 从明确提供的 Pick-a-Pic test_unique 和 HPDv2 test 输入构建面板。HPDv2 是 `ymhao/HPDv2` 的 `test.json` / Linear-DPO 的 `data/hpsv2_test400.txt` 对应的400条 caption。不能用 HPSv2 的3200条 style benchmark替代。

Pick-a-Pic 工具支持本地 Parquet/HF目录及TXT/JSON/JSONL导出。Parquet只读取 caption 等小列，不读取图像列。Linear-DPO 的提取脚本也使用 test_unique caption；上游仓库当前未随同提交 `pickapicv2_test500.txt`，应从自己的已下载 test_unique 数据导出，而不是从训练集凑500条。

冻结128条监控caption和selection seed=17，排除训练caption、去重，保留来源索引和SHA256。若不足128条会报错，不会退回训练集。HPDv2清洗后必须仍有400条；存在训练重合或不对的输入时应核对来源。

可以通过 `--monitor-count` 调整数量，但要同时修改训练配置 `evaluation.expected_prompts` 并另存实验配置。未知同义改写不能由字符串去重证明隔离。监控集重复用于挑checkpoint，最终质量报告应使用独立HPDv2或剩余Pick-a-Pic test。

## 调度、保留和恢复

训练配置：`evaluation.first_step=200`、`every=500`、`include_base=true`，另测最终步；`training.lora_checkpoint_every=1000`、`max_lora_checkpoints=null` 保留每千步及最终LoRA，不淘汰早期快照。`training.checkpoint_every=200` 另行滚动保存最新LoRA及完整优化器/LR/各rank随机状态，只保留最新一份，通过 `checkpoints/latest.json` 恢复。两条计划同时执行，滚动更新不清理永久LoRA；评估暂停、正常中断及最终步也提交完整恢复状态。每200步保存不会额外触发评估或图像生成。

监控默认每条一个seed，避免小面板加多seed过度占用时间。所有步骤、B/C方法共用同一面板和seed。每点评分保存均值、逐图分数和prompt层配对bootstrap变化；0/1000/2000…及最终点保存图像，其余点只在评分完整提交后清理清单列出的PNG，不删除无关文件。曲线在 `pickscore-curve.csv/.json/.png`。

完整评估默认400条、一个seed，保存全部图像、逐图四项分数、各类均值与总体平均。生成和评分分别使用子进程，串行释放显存。生成保留原renderer的恢复校验，评分按prompt/seed提交记录，断点不会重复已提交项。评估未完成则不继续训练；同一输出有独占锁，配置、源文件、prompt、scorer身份变化会拒绝续写。

大模型权重记录路径、大小及mtime清单，**不是完整权重内容哈希**；小checkpoint/源码、prompt和结果文件有SHA256。来源revision也应另行固定。换服务器路径或环境后，另开输出并重新验证，不伪称逐位恢复。

## 独立评估命令

```bash
# CPU预检：只检查输入、checkpoint、模型清单，不加载GPU模型。
python evaluate.py --config configs/evaluate.json --run runs/C-seed17-001 \
  --step 1000 --out runs/hpdv2-C-1000 --check-only

# 已分配的CCI/ACP GPU容器内运行；Slurm省略standalone参数。
python evaluate.py --config configs/evaluate.json --run runs/C-seed17-001 \
  --step 1000 --out runs/hpdv2-C-1000 --execution standalone --run-label slai-hpdv2

# 原checkpoint和配置不变时，同一命令可补完未完成的生成或评分。
# 基座使用 --step 0；仅PickScore使用 --metrics pickscore。
```

`configs/evaluate.json` 已列出所有可选评估集。只解析命令中选中的路径，默认只需 `HPDV2_PROMPTS`；切换时使用下面的对应变量。

默认独立评估采样PNDM50、512px、BF16、CFG7.5、seed17；训练监控用DDIM30节约时间。基线与候选必须用相同独立评估配置。此实现与Linear-DPO的度量定义对齐，但BF16、逐prompt重设seed和PNG保存等细节不保证复现其FP16/全局随机流/JPEG流程的论文数值。benchmark采样明确关闭SD1.5安全检查器，长文本按CLIP长度截断并记录生成端截断条目；训练的原caption长度检查保持严格。

所有模型和prompt准备好后，首次在实际分配的GPU上用本地小面板验证生成及四项评分；本次只实现与CPU测试，不提交GPU实验。

## 可选评估集与指标

评估集选择 prompt，指标为生成图评分，两者独立组合。HPDv3 是数据集，HPSv3 是评分模型。以下可选项目都不会因新增代码而自动启用。

| `--benchmark` | 配置变量 | 准备工具输入 | 默认运行 |
|---|---|---|---|
| hpdv2 | `HPDV2_PROMPTS` | `--hpdv2`，400条测试 caption | 是 |
| pickapic | `PICKAPIC_TEST_PROMPTS` | `--pickapic`，test_unique；最终报告建议使用 remaining 面板 | 否 |
| partiprompts | `PARTI_PROMPTS` | `--parti`，官方 `PartiPrompts.tsv` 或 caption 导出 | 否 |
| hpdv3 | `HPDV3_PROMPTS` | `--hpdv3`，官方 HPDv3 `test.json` | 否 |
| imagereward | `IMAGE_REWARD_PROMPTS` | `--imagereward`，已有官方 test prompt 导出 | 否 |

```bash
# 可以只准备额外数据集，不要求同时提供 Pick-a-Pic 和 HPDv2。
python scripts/prepare_benchmarks.py \
  --parti /data/PartiPrompts.tsv --hpdv3 /data/HPDv3/test.json \
  --train-pairs /data/training/pairs.jsonl --out data/extra-evaluation-v1
export PICKAPIC_TEST_PROMPTS="$PWD/data/evaluation-v1/pickapic-test-remaining.jsonl"
export PARTI_PROMPTS="$PWD/data/extra-evaluation-v1/partiprompts.jsonl"
export HPDV3_PROMPTS="$PWD/data/extra-evaluation-v1/hpdv3-test.jsonl"
```

PartiPrompt 原集1632条；Pick-a-Pic 使用实际 test_unique 输入，保持实际清洗后数量，不硬凑论文500条。HPDv3 按[官方 test.json 的 prompt 字段](https://github.com/MizzenAI/HPSv3/blob/bd0c5fcb5f587617b0169c07222ab78d01e2f3c2/README.md#test-set-testjson)提取并去重，不读取原偏好图片。本包测新生成图片的平均分，不执行奖励模型对原图对的偏好分类准确率。所有面板排除本项目训练 caption，输出实际数量、来源索引及哈希。

| `--metrics` 可选项 | 输出口径 | 默认运行 |
|---|---|---|
| pickscore / clip / hpsv2 / aesthetic | 上文四项 | 是 |
| hpsv3 | 官方 HPSv3 输出的 mu，即 `logits[:,0]`；不加 sigmoid | 否 |
| image_reward | 官方 ImageReward v1.0 标准化 reward；不转排名或概率 | 否 |

HPSv3 接[官方本地推理 API](https://github.com/MizzenAI/HPSv3/blob/bd0c5fcb5f587617b0169c07222ab78d01e2f3c2/hpsv3/inference.py)，使用原配置中的 Qwen2-VL 模型和奖励头，仅将模型路径替换为显式本地路径、输出写入临时目录，默认关闭 FlashAttention2 而使用 SDPA。原配置文件不被修改。检查点显式传入，离线模式禁止隐式模型下载，batch 默认1，模型 dtype 跟随官方 YAML（默认 BF16）。

ImageReward 接[官方模型的 score 方法](https://github.com/zai-org/ImageReward/blob/78f8051059b34fef8e0032ccb40561c23a2ddbbb/ImageReward/ImageReward.py)，按官方均值/标准差输出 reward，BLIP + reward head 全部严格加载、FP32，batch 默认1。构造时暂时将上游硬编码的 BERT tokenizer 加载替换为本地目录，并保留 `[DEC]`/`[ENC]` 特殊 token；随后恢复上游工厂。只兼容忽略或补齐确定性的 BERT position_ids buffer，缺失模型权重仍报错。

## 可选评分环境

两个评分器均通过 `configs/scorers.json` 指定本地源码、模型和 Python。HPSv3 官方依赖 Transformers 4.45.2；旧训练环境是4.53.3，当前SLAI适配也选择4.45.2，但HPSv3仍有独立的TRL/PEFT等依赖，应保留独立评分环境。`requirements-hpsv3.txt` 的核心版本按[官方 environment.yaml](https://github.com/MizzenAI/HPSv3/blob/bd0c5fcb5f587617b0169c07222ab78d01e2f3c2/environment.yaml)整理，并补齐上游推理模块间接导入的 matplotlib、pandas、tensorboard。它与 `requirements-imagereward.txt` 用于各自环境，**不与训练 requirements 合并安装**。本次未安装这些可选环境，完整依赖和GPU执行需迁移后验证。

```bash
# 配置已准备好的官方源码和本地资产。
export HPSV3_SOURCE=/your/sources/HPSv3
export HPSV3_BASE_MODEL=/your/models/Qwen2-VL-7B-Instruct
export HPSV3_CHECKPOINT=/your/models/HPSv3/HPSv3.safetensors
export HPSV3_CONFIG="$HPSV3_SOURCE/hpsv3/config/HPSv3_7B.yaml"
export HPSV3_PYTHON=/your/envs/hpsv3/bin/python

export IMAGE_REWARD_SOURCE=/your/sources/ImageReward
export IMAGE_REWARD_CHECKPOINT=/your/models/ImageReward/ImageReward.pt
export IMAGE_REWARD_CONFIG=/your/models/ImageReward/med_config.json
export IMAGE_REWARD_TOKENIZER=/your/models/bert-base-uncased
export IMAGE_REWARD_PYTHON=/your/envs/imagereward/bin/python

# 仅在各自新环境已安装匹配的 torch/torchvision 后执行。
"$HPSV3_PYTHON" -m pip install -r requirements-hpsv3.txt -e "$HPSV3_SOURCE"
"$IMAGE_REWARD_PYTHON" -m pip install -r requirements-imagereward.txt -e "$IMAGE_REWARD_SOURCE"
```

源码 revision 记录在 `evaluation_sources.json`，应使用对应版本；HPSv3 底座、奖励检查点和配置必须配套。评分子进程会继承本包的 `src` 路径，不需向评分环境安装本包的训练依赖。在自建 scorer JSON 中可省略 `python`，表示使用当前 Python；提供的模板要求显式设置对应变量。仅使用默认四项时不需要任何上述可选变量或环境。

```bash
# 单个额外数据集，选择需要的指标；也可加 --check-only 做CPU预检。
python evaluate.py --config configs/evaluate.json --run runs/C-seed17-001 \
  --step 1000 --benchmark hpdv3 --metrics pickscore hpsv3 image_reward \
  --out runs/hpdv3-C-1000 --execution standalone --run-label slai-hpdv3

# 多个评估集：每个保存独立图像和逐图分数，顶层 summary.json 汇总。
python evaluate.py --config configs/evaluate.json --run runs/C-seed17-001 \
  --step 1000 --benchmarks pickapic partiprompts hpdv3 \
  --metrics pickscore clip hpsv2 aesthetic hpsv3 image_reward \
  --out runs/full-C-1000 --execution standalone --run-label slai-full
```

同一评估集的所有指标复用相同 PNG，按单个评分进程串行运行；更换评分器不会触发另一套随机图像。多评估集的结果分别位于 `out/pickapic/`、`out/partiprompts/` 等目录，不跨数据集合成一个含糊的平均分。原命令可恢复未完成的图像/评分；变更选项请使用新输出目录。默认训练200步/每500步的PickScore监控与每1000步的LoRA、图像保留节奏保持不变。
