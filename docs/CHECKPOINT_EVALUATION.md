# SD1.5 checkpoint 批量评估

入口 `evaluate_batch.py` 默认评估截图中勾选的九个实验，步数为 5000、6000、7000、9000。配置见 `configs/evaluate-selected-sd15.json`。五个指标固定为 PickScore、HPS v2.0、LAION aesthetic、CLIP、ImageReward；每组图片只生成一次，随后用独立进程依次加载评分器，释放上一模型的显存。

这是独立评估 checkout。训练 run、checkpoint、训练配置均只读，输出目录必须位于 `RUNS_ROOT` 之外。不要将此分支拉入正在训练或准备续训的 checkout。

## 分支与安装

开发分支 `feat/checkpoint-benchmark-eval` 基于 `fix/aoss-render-resume` 的 `9730dab`，包含已有 AOSS 修复和 stagewise checkpoint 支持；不改变 `main` 或原训练分支。

在 SLAI 持久 workspace 新建 checkout（已存在时直接使用，勿覆盖）：

```bash
git clone --branch feat/checkpoint-benchmark-eval --single-branch https://github.com/hypocritica/DPO.git /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v1
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v1
git rev-parse HEAD
conda create -n eval python=3.10 -y
conda activate eval
python -m pip install setuptools==80.9.0 wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130 nvidia-nccl-cu13==2.30.7
python -m pip install --no-build-isolation -r requirements-eval.txt
python -m pip check
```

若环境已创建，跳过 `conda create`。无需 editable install。不要在该环境安装旧的 `requirements-imagereward.txt` 或另一个 `image-reward` distribution；本项目从资源配置指定的官方源码加载 ImageReward。预装 setuptools/Torch 后再安装依赖，是为了兼容 OpenAI CLIP 与 fairscale 的源码构建。

资源沿用 `configs/resources.json`，默认训练输出为 `/mnt/aoss/xietianyu/self_correct/runs`。机器私有配置写入新 checkout 的 `configs/resources.local.json`，或通过 `--resources /absolute/path/resources.json` 指定。若旧 checkout 有私有覆盖，核对后复制到新 checkout。五项评分应使用同一个 eval Python，清除旧 scorer 配置中的其他环境 `python` 覆盖。

## 评分资产

```bash
python scripts/check_eval_environment.py
```

这个 CPU 预检检查依赖、官方源码导入和资源文件，不加载大权重，不下载模型。缺失资产按输出补齐。默认路径对应下表；完整清单和机器覆盖仍以资源表为准。

| 指标 | 资源 |
|---|---|
| PickScore | `PICKSCORE_MODEL` 与 `PICKSCORE_PROCESSOR` |
| CLIP | `PICKSCORE_PROCESSOR/open_clip_pytorch_model.bin`，HPSv2 官方 open_clip 源码 |
| HPS v2.0 | `HPSV2_MODEL_DIR/HPS_v2_compressed.pt`，`HPSV2_MODEL_DIR/source/hpsv2/src/open_clip` |
| Aesthetic | `AESTHETIC_CLIP_MODEL`，`AESTHETIC_CHECKPOINT` |
| ImageReward | `IMAGE_REWARD_MODEL_DIR/{ImageReward.pt,med_config.json,source}`，`IMAGE_REWARD_TOKENIZER` |

现有下载入口 `scripts/download_assets.py download ASSET` 支持 `pickscore`、`clip-h14`、`clip-l14`、`hpsv2`、`image-reward`、`image-reward-tokenizer`。只补缺失项；下载前用 `plan` 查看空间需求。`scripts/download_sources.sh` 支持 `hpsv2`、`aesthetic` 和固定版本的 `image-reward` 源码，已有目录会拒绝覆盖。完整下载用法仍见 `command.md`。评分过程中禁止隐式下载或用其他指标代替缺失指标。

## 准备三个提示词集

```bash
python scripts/prepare_eval_prompts.py --download-sdpo
```

只下载约 0.9 MB 的 JSON，默认写入 `/mnt/aoss/xietianyu/datasets/benchmarks/sd15/paper-eval`。也支持离线输入：

```bash
python scripts/prepare_eval_prompts.py --source-dir /path/to/Diffusion-SDPO/prompts
```

来源固定为 [Diffusion-SDPO 官方仓库](https://github.com/ATH-MaaS/Diffusion-SDPO/tree/84fb241c1b89705a247da8b0d6047798ca49830d/prompts) commit `84fb241c1b89705a247da8b0d6047798ca49830d`，逐文件校验 SHA256，保留原始文件、行顺序及文本。

| CLI 名称 | 官方文件 | 条目数 |
|---|---|---:|
| `pickv2` | `papv2.json` | 487 |
| `hpsv2` | `hpsv2.json` | 3200 |
| `partiprompts` | `partiprompts.json` | 1632 |

HPSv2 的 3200 条中有重复文本；按原行保留，不去重。这与旧配置中的 HPDv2 test400 不同。PickV2 使用完整发布面板，没有排除此前选出的监控提示词，因此不能把它称为完全未用于模型选择的独立测试集。准备脚本不会重新过滤训练重合或语义近似，`selection.json` 明确记录这些约定。

默认评估协议为 SD1.5 512×512、EulerDiscreteScheduler、50 步、标准 Diffusers CFG 7.5、seed 0、空负提示词、BF16 pipeline、关闭 safety checker。各方法使用相同提示词及初始随机 seed。更改协议需复制评估配置并使用新输出目录。

该协议用于公平比较本项目 checkpoint，不声称数值精确复现 SDPO 表格。官方 `test_sd15.py` 使用 FP32 VAE，而且 CFG 写成 `conditional + cfg * (conditional - unconditional)`：其参数 7.5 等效标准 Diffusers CFG 8.5；这里明确采用标准 CFG 7.5。评分器采用相同类型和量纲，但软件版本及实现路径不同，尚未逐图对齐。

## 检查待评估 checkpoint

```bash
python evaluate_batch.py --benchmarks pickv2 --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-pickv2-v1 --plan-only
python evaluate_batch.py --benchmarks pickv2 --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-pickv2-v1 --check-only
```

`--plan-only` 只检查提示词及 checkpoint 清单，不要求评分模型或基座资产存在，不写输出；`--check-only` 进一步检查五个评分器资产、基座和 adapter 元数据。检查结果按实验、步数和提示词集列出。默认缺失步数标为 `missing_checkpoint`，不替换为 latest；损坏的已发布 checkpoint 标为 `invalid_checkpoint` 并返回非零。`--strict-missing` 让缺失也返回非零。

## 两张图的五指标 smoke

在已分配 GPU 的 SLAI 容器内激活 eval，然后执行：

```bash
python scripts/check_eval_environment.py --gpu --execution standalone
python scripts/prepare_eval_prompts.py --source-dir /mnt/aoss/xietianyu/datasets/benchmarks/sd15/paper-eval/sources --smoke-prompts 2 --out /mnt/aoss/xietianyu/datasets/benchmarks/sd15/paper-eval-smoke
python evaluate_batch.py --runs sd15-dpo-p384-cu130-v1 --steps 5000 --benchmarks pickv2 --prompts-root /mnt/aoss/xietianyu/datasets/benchmarks/sd15/paper-eval-smoke --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-smoke-v1 --gpus 0 --strict-missing
```

若 5000 步尚未保留，选择已有的目标步数。smoke 只验证模型加载、生成、五指标评分和重复运行；其分数不能放入正式实验表。再次执行相同命令应直接复用完成结果。

## 正式批量评分

```bash
bash launchers/sd15/evaluate_checkpoints.sh --benchmarks pickv2 --gpus 0,1 --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-pickv2-v1
```

上面默认运行全部九个实验及 5000、6000、7000、9000 步。切换 HPSv2 或 PartiPrompts 时用新输出目录：

```bash
bash launchers/sd15/evaluate_checkpoints.sh --benchmarks hpsv2 --gpus 0,1 --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-hpsv2-v1
bash launchers/sd15/evaluate_checkpoints.sh --benchmarks partiprompts --gpus 0,1 --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-partiprompts-v1
```

也可 `--benchmarks pickv2 hpsv2 partiprompts` 一次选择三套，使用另一输出目录。单 seed、36 个 checkpoint 全部存在时，三套合计生成 **191484 张图片**；仅 PickV2 为 17532 张。每个 worker 一张卡，独立分配 checkpoint，不使用 DDP。`--gpus` 必须选择已分配且位于 `CUDA_VISIBLE_DEVICES` 中的设备 ID；不指定时使用当前可见 GPU 的默认设备。

用 `--runs NAME ...` 和 `--steps 5000 6000` 缩小范围，用 `--seeds 0 1 2` 增加生成 seeds；这些都改变评估身份，应使用新输出目录。初次记录缺失的 checkpoint 后，用完全相同命令重跑可以补齐新出现的 checkpoint，不会重新训练。

启动器使用已激活环境的 `python`，也可显式设置 `SCD_PYTHON=/path/to/envs/eval/bin/python`。它不自动提交作业。若需粘贴到 SLAI 启动命令，参考 `command.md` 的完整块。

## 输出与恢复

输出根目录有 `summary.csv`、`summary.json`，每行是 run × step × benchmark，包含五项均值、各指标图像数量、状态和失败原因。多卡运行中，各 worker 持续更新 `summary-worker-N.csv/json`，主进程结束后生成总表。单卡持续更新总表。

每组明细位于 `RUN/step-0005000/PANEL/`：`render/manifest.jsonl` 对应生成图，`scores/METRIC/scores.jsonl` 保留逐图分数，`summary.json` 和 `complete.json` 保存结果及校验。图片默认保留。

重复相同命令继续未完成的生成或评分，已完成且校验通过的结果复用。输出身份绑定提示词、采样设置、checkpoint、评分资产及实现版本，拒绝混入另一协议。修改代码/依赖/评分资产后应创建新版本输出目录；完成评估后保留对应 checkout commit。

只汇总、不执行生成：

```bash
python evaluate_batch.py --benchmarks pickv2 --out /mnt/aoss/xietianyu/self_correct/evaluations/selected-sd15-pickv2-v1 --summarize-only
```

汇总请求的 runs、steps、panels、seeds 必须与原请求一致。缺失和失败的分数留空，不按 0 计入结果。

## 本地验证边界

完成标准库/CPU 的 checkpoint 校验、精确步数选择、多进程 CLI、资源覆盖、五评分器顺序、输出恢复与篡改拒绝、错误持久化及提示词导入测试。实际下载并校验了上述三份小型官方提示词文件。

已核对官方 CUDA wheel 元数据和非 GPU 依赖解析；ImageReward 的官方 BLIP 核心模块通过微型 CPU 前向。没有安装完整 eval GPU 环境、加载五套真实评分模型或执行 5090 生成。因此远程正式运行前应先完成上述 smoke。
