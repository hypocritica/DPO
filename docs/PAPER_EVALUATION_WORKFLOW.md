# SLAI：从实验目录到论文表格

这是本分支的操作主线。沿用五套面板与现有分组指标，先冻结方法和 checkpoint，
再生成图片，之后分别评分，最后导出带来源的论文表格和原图。具体环境安装、资产下载
命令见 [command.md 第一节](../command.md) 与 [部署说明](SLAI_EVALUATION_DEPLOYMENT.md)。

## 1. 同步代码、复用资源

在 SLAI 新建独立的 `DPO-eval-v2` checkout，分支为
`feat/unified-evaluation-slai-20261111`。不在运行中的训练 checkout 内更新代码。

```bash
git clone --branch feat/unified-evaluation-slai-20261111 --single-branch \
  https://github.com/hypocritica/DPO.git \
  /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
git rev-parse HEAD
```

编辑 `configs/resources.local.json`，复用真实已有模型、数据集、评分器、source 和解释器。
旧 inventory 中 Flow 基座是 `/mnt/aoss/xietianyu/models/sd3.5-medium` 和
`/mnt/aoss/xietianyu/models/Z-Image`；资源表默认目录可能不同，须以当前文件为准。
`catalog` 会直接保留 training.json 中的模型路径、revision、精度控制；不要为了路径
好看而换一个基座或凭空填写 revision。`configs/evaluation-experiments.example.json`
只是格式示例，里面的占位目录不能直接运行。

环境按职责独立：legacy 负责 SD1.5/SDXL 和原五指标，flow 负责 SD3.5 Medium/Z-Image
生成，hpsv3、unifiedreward、geneval、dpg 分别评分。已有兼容环境可直接使用绝对
`--python /path/to/env/bin/python`；新环境使用 setup 脚本先预览，再显式 `--apply`。
GenEval/DPG 的旧 CUDA/MMCV 组合是 H100 部署候选，需要实际 GPU smoke；不能作为
5090 配方。当前没有在 SLAI 执行完整安装或真实模型推理。

## 2. 实时目录登记：哪种方法、哪个 run、哪几个 step

2026-11-11 的 inventory 是历史快照，不能判断后来启动的 SFT 等是否完成。
请在远程读取当前的小型元数据，生成可审阅的登记表；不会读取 optimizer tensor。

```bash
python evaluate_suite.py catalog \
  --runs-root /mnt/aoss/xietianyu/self_correct/runs \
              /mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs \
  --families sd35_medium \
  --out configs/experiments.sd35.local.json
```

对 SD1.5、SDXL 分别将 `--families` 改为 `sd15`、`sdxl`，用不同输出文件；Z-Image
使用 `zimage`。建议每个 backbone 一个 catalog 和一个正式输出目录，便于并行推进。

导入包含全部能识别的候选 run 和已发布 checkpoint，不自动选“最新最优”。打开 JSON：

1. `models` 核对基座；Base 自动从这里生成，无须找到一个 Base 训练目录。
2. `experiments` 每项核对 `id / family / method / root / run`，删除不参与这轮比较的
   smoke、backup 和旧重复候选。重叠的同方法同 step 会报错，必须明确选择。
3. 将每项 `steps` 改为本次要测的精确步数，例如 DPO `[2000, 3200]`、SFT `[3200]`。
   步数必须已发布；缺失就报错，不回退到 latest/rolling。训练还未保存的 step 以后
   重新导入、建立新版本计划。
4. 导入时如同 family 有不同模型身份，按实际 run 路径用 `--run-glob` 缩小范围。
   不要为了通过检查而修改训练产物。

方法标签来自训练配置核验，不仅由目录名猜测。SD3.5 Medium 的八项：

| 论文方法 | `--methods` 值 |
|---|---|
| Base | `base` |
| SFT | `sft` |
| DPO | `dpo` |
| LinearDPO | `linear_dpo` |
| Global K2 | `global_k2` |
| Stagewise K2 | `stagewise_k2` |
| Global K4 | `global_k4` |
| Stagewise K4 | `stagewise_k4` |

另外支持 SD1.5/SDXL 已有的 DMPO、DSPO、KTO、MAPO、K6 和阶段消融，完整枚举见
`python evaluate_suite.py plan --help`。无法可靠分类的候选会给出原因，不贴错误标签。

```bash
python evaluate_suite.py list --catalog configs/experiments.sd35.local.json \
  --families sd35_medium --json
```

`catalog --run-glob` 筛实际训练目录；`list/plan --catalog --run-glob` 筛登记表的
experiment `id`。冻结计划后执行阶段的 `--run-glob` 筛 task.run（登记的 id）；
建议优先使用 `--families`、`--methods`、`--steps`，减少目录名混淆。
登记表只选择已保存快照，不是实时训练进程监控器。

## 3. 冻结五面板计划，审阅图片总量

先按 command.md 准备固定官方 prompts。登记表内已写好各方法的步数，`plan --catalog`
不再接受 `--steps` 或 `--latest`；执行 generate/score 时可用 `--steps` 筛选计划子集。

```bash
source paths.sh
SCD_EVAL_OUT="$EVALUATIONS_ROOT/sd35-paper-v3"
python evaluate_suite.py plan --catalog configs/experiments.sd35.local.json \
  --families sd35_medium \
  --methods base sft dpo linear_dpo global_k2 stagewise_k2 global_k4 stagewise_k4 \
  --benchmarks pickv2 hpsv2 partiprompts geneval dpg \
  --out "$SCD_EVAL_OUT" --dry-run
```

检查输出中的方法、步数、prompt 数和图片数后，执行同一命令去掉 `--dry-run`，写入
`plan.json` 和 prompts 快照。同一基座只安排一次 Base；所有方法共用该 family 的采样
协议。后续更改方法、step、seed 或 prompt 必须用新 `--out`，不混入旧结果。

| 面板 | prompts | SD1.5/SDXL 每 prompt | SD3.5M/Z-Image 每 prompt | 默认指标 |
|---|---:|---:|---:|---|
| PickV2 | 487 | 1 | 4 | 按模型分组 |
| HPSv2 | 3200 | 1 | 4 | 按模型分组 |
| PartiPrompts | 1632 | 1 | 4 | 按模型分组 |
| GenEval | 553 | 4 | 4 | GenEval |
| DPG | 1065 | 4 | 4 | DPG |

原五指标：PickScore、HPSv2、ImageReward、CLIP、Aesthetic。
现代组：HPSv3、UnifiedReward 2.0 ACS mean、PickScore、ImageReward、CLIP、Aesthetic。
HPSv2 面板名与 HPSv2 评分器是两回事。沿用现有协议，不擅自更改 seed 或指标尺度。

单个 SD3.5M/Z-Image checkpoint 全五面板为 **27,748 张**，八方法各一个 checkpoint
是 **221,984 张**；SD1.5/SDXL 每 checkpoint 为 **11,791 张**。先用独立 smoke 计划
验证环境，然后分方法、step 和面板推进。不要在五天窗口中无意生成所有历史 checkpoint。

## 4. 先生成，再独立评分

下面使用已激活的 flow 环境，GPU 必须是 SLAI 已分配设备。只有一张卡时省略
`--gpus`；按平台实际 `CUDA_VISIBLE_DEVICES` 设置设备。

```bash
python evaluate_suite.py generate --out "$SCD_EVAL_OUT" --methods base dpo --check-only
python evaluate_suite.py generate --out "$SCD_EVAL_OUT" --methods base dpo --gpus 0,1
# 后续生成剩余方法，已完成任务核验后复用：
python evaluate_suite.py generate --out "$SCD_EVAL_OUT" --gpus 0,1
```

也可以加 `--steps 3200` 或 `--benchmarks pickv2`。Base 是 step 0，筛选 `--steps 3200`
时不包含 Base。中断后重复同一命令恢复，不删除产物。计划冻结后训练继续写入新 checkpoint
不影响已经选定的快照；不要移动或删除当前生成需要的 checkpoint。

评分只读已生成图片，不再需要训练目录、LoRA 或生成模型。相同图片可以后续追加指标。
对 SD3.5M 的默认组，分环境执行：

```bash
# legacy / 原eval环境：四项共享reward，不需要加载Flow模型
python evaluate_suite.py score --out "$SCD_EVAL_OUT" \
  --benchmarks pickv2 hpsv2 partiprompts \
  --metrics pickscore image_reward clip aesthetic --gpus 0,1
# hpsv3环境
python evaluate_suite.py score --out "$SCD_EVAL_OUT" \
  --benchmarks pickv2 hpsv2 partiprompts --metrics hpsv3 --gpus 0,1
# unifiedreward环境
python evaluate_suite.py score --out "$SCD_EVAL_OUT" \
  --benchmarks pickv2 hpsv2 partiprompts --metrics unifiedreward --gpus 0,1
# geneval环境
python evaluate_suite.py score --out "$SCD_EVAL_OUT" --benchmarks geneval --metrics geneval --gpus 0,1
# dpg环境
python evaluate_suite.py score --out "$SCD_EVAL_OUT" --benchmarks dpg --metrics dpg --gpus 0,1
```

每一条先追加 `--check-only` 做预检，再在已分配 GPU 上执行实际评分。
SD1.5/SDXL 三个偏好面板使用 `--metric-group legacy` 即原五指标。
新模型若另需原五指标对照，也可显式运行 legacy 组，并单独命名汇总/论文导出。
自动运行时可使用 `--python` 指定评分环境，不依赖交互式 conda activate。

## 5. 状态表、论文表和可搬走的图片

```bash
python evaluate_suite.py summarize --out "$SCD_EVAL_OUT"
python export_paper.py --out "$SCD_EVAL_OUT" --paper-name main-v1
python evaluate_suite.py gallery --out "$SCD_EVAL_OUT" \
  --benchmarks partiprompts --gallery-name preview-v1 --limit 200
```

`summary.csv/json` 标出完成、缺失和错误，未全部完成时退出码为 1。
论文导出默认拒绝不完整结果；只看草稿可加 `--allow-incomplete`，会标记
`INCOMPLETE DRAFT` 并返回 1。可用同样的 family/method/step/benchmark 筛选先导出
已完成部分。不完整项为空值，不填 0；不自动选择最优 checkpoint，也不声称显著性。

```text
<out>/paper/main-v1/
  results.csv           每模型/方法/step/面板/指标一行，含样本数、均值、标准误、身份
  results.json          同时保存冻结task、评分器身份与完整summary来源
  <family>-<panel>-<protocol>.tex / .md
  README.md             表格索引、范围和口径
  export.json           导出文件SHA256和完成标记
```

LaTeX 表可用 `\input{...tex}` 放进论文；路径从 README 中选择。不同基座、prompt、采样和
评分协议会分表。数值保持原始尺度，UnifiedReward 分项留在 CSV/JSON；prompt 标准误
不是置信区间。更新分数或改变筛选时换 `--paper-name`，既有版本不可覆盖。
smoke面板保留`smoke-only`标记；没有可核验selection来源的自定义面板标为
`custom-unverified`，不会称为正式完整benchmark。没有提供论文源码的位置，所以此处
生成可引用的表格文件，不改写论文结论。

从预览 `selection.json` 找实际 scene_id，再固定同 prompt/seed 导出展示图：

```bash
python evaluate_suite.py gallery --out "$SCD_EVAL_OUT" \
  --benchmarks partiprompts --methods base dpo global_k2 stagewise_k2 \
  --scene-id REPLACE_WITH_REAL_SCENE_ID --seed 17 \
  --gallery-name figure-v1 --copy-images --limit 32
```

将整个 `<out>/galleries/figure-v1/` 下载到本地即可打开 `index.html`。`images/` 是原 PNG
的逐字节副本，不重新压缩；`selection.json` 保留方法、step、checkpoint_id、prompt、
seed、原路径和 SHA256，可据此排版论文。图片挑选不会改变评分样本或汇总分母。
最终展示如有精选应说明规则，定量比较仍包含全部预定样本。

## 6. 返回什么以便继续审阅

第一轮可以返回编辑后的 catalog、`plan --dry-run` 输出和 `summary.json/csv`。
这些小文件足以检查方法是否齐全、checkpoint 是否选对、哪些评分尚未完成。
完整测试后再提供 `paper/<name>/results.json` 与需要讨论的 portable gallery。
旧 inventory/report 继续用于历史追踪，当前产物以新登记表、plan 和完成标记为准。
