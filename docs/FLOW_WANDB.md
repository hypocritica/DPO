# Flow 的 W&B 离线对比

[实现验证摘要（JSON）](FLOW_WANDB_VALIDATION.json)。15项相关CPU/真实离线记录检查通过；没有执行真实云端上传。

`paper-v2` 编号入口默认 `--report-to all --wandb-mode offline`，同时保留
TensorBoard、原始 JSONL 和 W&B 离线记录。训练不登录、不上传、不需要 API key。
W&B 固定为 `0.18.7`，离线文件、读取器和 `sync --append` 使用同一已测试版本。
每点评估默认将前8张已生成图片和prompt/seed caption写入离线媒体目录，不增加生成任务；
`--wandb-images 0` 可关闭，`--wandb-images 16` 可提高数量（最多64）。
旧 paper-pilot 默认不变。项目为 `mvpo-flow`；本轮三组统一 group：
`zimage-hpdv3-pair_views128-seed17`。

| 方法 | 图片对/更新 | pair-views/更新 | 本轮位置 |
|---|---:|---:|---|
| DPO K1 | 128 | 128 | irip，2×A800 |
| Stagewise K2 C | 64 | 128 | irip，2×A800 |
| LinearDPO K1 | 128 | 128 | irip，2×A800 |
| Global K2 C | 64 | 128 | irip，2×A800 |
| Stagewise K4 C | 32 | 128 | SLAI，使用已分配 GPU |

固定 seed17、3200 步、beta500、学习率5e-6；Stagewise boundary500。
两段按每对抽取，段内 K 个独立视图，平均 margin 后计算 DPO。
每200步更新恢复状态、每400步永久保留完整 checkpoint；基座及八点评估保留。

## 本地

从 stagewise 项目根目录提交，每组 2×A800 / 8 CPU / 96GB / 4天：

```bash
sbatch --job-name=zimage-dpo-wb launchers/flow/local/train_stagewise_a800.sbatch dpo
sbatch --job-name=zimage-stagek2-wb launchers/flow/local/train_stagewise_a800.sbatch stagewise-k2
# 追加两组，直接正常排队，不设置依赖。
sbatch --job-name=zimage-linear-wb launchers/flow/local/train_stagewise_a800.sbatch linear-dpo
sbatch --job-name=zimage-globalk2-wb launchers/flow/local/train_stagewise_a800.sbatch global-k2
```

已经提交/运行时不要重复提交。实际作业号见 command.md 首节。
环境复用旧 flow 的 `.venv-flow-a800`，源码/配置/输出使用新 stagewise 目录。

## SLAI K4

先同步包含本轮 W&B 入口改动的源码，在 SLAI 完成环境和数据准备并取得本机的
passed 数据/模型报告，再在已分配的 GPU 容器中启动。例如双 H100：

```bash
bash launchers/zimage/08_stagewise_mvpo_k4_views128.sh \
  --profile h100 --run-tag slai-wandb-20261107-v1 \
  --report-to all --wandb-mode offline --wandb-project mvpo-flow \
  --wandb-group zimage-hpdv3-pair_views128-seed17 \
  --wandb-name zimage-stagewise-k4-C-slai-h100
```

双 A800 使用 `--profile a800`。四5090使用 `--profile 5090 --weight-dtype bfloat16`，
需要单独标记 BF16 精度；本地 A800 默认 FP32 transformer，不能将精度差异解释为纯 K 差异。
脚本不会替用户申请 SLAI GPU。

为比较评估效果，SLAI 必须复用相同的64条 prompt、生成 seed、采样参数、
基座模型和 PickScore 权重。当前本地面板为 stagewise 目录下
`data/hpdv3-pilot-64-v2.jsonl`；它保留旧面板的全部 prompt 和顺序，补齐 canonical 元数据。
可复制此小文件到 SLAI，通过
`--monitor-prompts /SLAI/实际路径/hpdv3-pilot-64-v2.jsonl` 指定。
本地图片路径和本地 passed 报告不可复制后充当 SLAI 的数据认证。

## 汇总到 W&B

每个实验的 W&B 文件保存在 `RUN_DIR/wandb/offline-run-*`。
训练阶段、评估和续训共享同一稳定 run ID；不同方法有各自的 ID。
使用 `optimizer_step` 作为横轴，效果主要比较 `evaluation/pickscore`；
不同 K/采样方式的训练 loss 不代表直接的效果排序。

训练进行中也可在另一个终端开启网络代理，显式登录同一 W&B entity，再逐段串行追加。
脚本只选择已有结束记录且大小/mtime稳定的段（兼容core的exit及Python后端的final/footer），正在写入的段自动跳过；
不需要停止训练。训练本来就会在既有评估边界关闭记录器，网页更新到最近已结束的段，
不是每一步实时上传。首段训练之后才产生step0基座评估，因此首次同步等待该评估结束，
再按history step排序上传，防止把基座或较早的指标追加到较晚的数据后面。

已有独立 `mvpo-flow` 环境可补装固定版本；新bootstrap已自动安装：

```bash
python -m pip install 'wandb==0.18.7'
WANDB_MODE=online wandb login
# 只预览；不登录、不上传、不写同步标记。
python scripts/sync_flow_wandb.py --dry-run --entity YOUR_ENTITY /实际路径/DPO_RUN /实际路径/K2_RUN
# 网络连通时执行；两台服务器均使用相同entity/project，可反复执行追加新完成的段。
python scripts/sync_flow_wandb.py --entity YOUR_ENTITY /实际路径/DPO_RUN /实际路径/K2_RUN /实际路径/K4_RUN
```

默认project为 `mvpo-flow`；不同方法各有独立run ID，训练/评估/正常续训使用同一实验ID。
本地现有65803–65806源码/冻结配置继续保留，使用新checkout的同步脚本读取它们的RUN_DIR即可；
已有离线scalar记录可同步，新增图片记录适用于使用本次新tracker启动的实验，不自动补造旧记录。
启动后不要通过更改旧源码或冻结配置给正在跑的实验加图片。

脚本只在被执行时上传，本轮没有执行上传。按离线history起始步排序后串行使用 `--append`，
已标记synced的段跳过；保存上传entity/project并拒绝误换目的地，同一目录的上传器互斥。
若断电回退产生重叠history，脚本报错保留原记录，避免无提示覆盖已上传曲线。
离线 history step 为 `2*optimizer_step`，同一步评估为 `2*optimizer_step+1`，
从而保持分段记录的单调性；图表横轴仍为原始优化步数。

如果代理只在本机可用，也可把完整 `RUN_DIR` 日志拷到能联网的机器同步。
必须包括 `training.json` 和整个 `wandb/offline-run-*` 目录（尤其 `files/media/images`），
不能仅复制 `.wandb` 文件，否则图片缺失。一个实验由一个同步目录维护标记，避免两处同时上传同一份记录。

## 修复远程旧面板的 metadata

此前发布的生成器遗漏 `metadata` 和 `canonical-only-v1` 标记；本地v2面板已修正。
本次GitHub更新包含同样生成器修复及实际 `load_suite` 回归测试。
新生成面板直接可用；已生成的远程旧面板不会因git pull自动改变。
在训练启动前，用以下命令生成新文件，保留原prompt、顺序和旧文件，无需重跑HPDv3提取/数据准备：

```bash
export SCD_RESOURCES="$PWD/configs/resources-slai-flow.json"
source paths.sh
python scripts/repair_flow_monitor.py --source "$MONITOR_PROMPTS_FLOW_PILOT" \
  --out /mnt/aoss/xietianyu/datasets/benchmarks/flow-paper-v2/hpdv3-monitor-64-v2.jsonl
# 新实验指定修正后的面板和新tag；已冻结/运行中的旧实验不要原地修改。
bash launchers/zimage/08_stagewise_mvpo_k4_views128.sh --profile a800 \
  --monitor-prompts /mnt/aoss/xietianyu/datasets/benchmarks/flow-paper-v2/hpdv3-monitor-64-v2.jsonl \
  --run-tag slai-wandb-panelv2 --wandb-name zimage-stagewise-k4-C-slai-a800
```

这次缺字段涉及评估prompt面板，不是训练pairs.jsonl；修复不会改变训练pairs预算或数据标签。
要严格比较两台服务器，优先直接复制本地64条v2面板，并比对SHA256；重新选择另一组prompt不等价。
