# SLAI：SD3.5 Medium 八方法 checkpoint 曲线

这条流程在相同的 **128 条 development prompts、seed 17** 上比较 Base、SFT、DPO、
LinearDPO、Global K2、Stagewise K2、Global K4、Stagewise K4。每个已发布永久
checkpoint 生成128张图，然后仅计算 PickScore，保存图片、逐图分数和曲线。
新计划默认每GPU生成batch4，固定批边界随计划保存，评分batch独立配置。
Base生成一次，图中显示为水平参考线。它用于观察训练过程，不替代五套正式benchmark。

入口是 `evaluate_curves.py`，复制执行时可用 `scripts/run_sd35_curves.sh`。
所有GPU命令只在 **SLAI已分配2或4张5090的容器** 内执行；本脚本不会申请资源、
调用Slurm、安装依赖或下载模型。训练目录只读，评估结果另存。

## 1. 同步独立 checkout，复用 smoke 环境

已有独立 `DPO-eval-v2` 时，在该目录同步本分支，保留运行中的训练checkout。
下面克隆示例要求目标目录尚不存在；已有评估checkout可直接进入其目录。

```bash
git clone --branch feat/unified-evaluation-slai-20261111 --single-branch \
  https://github.com/hypocritica/DPO.git \
  /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-curves-v1
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-curves-v1
git rev-parse HEAD

export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-flow-smoke-7818ade/bin/python
"$SCD_FLOW_PY" --version
"$SCD_FLOW_PY" -m pip check
"$SCD_FLOW_PY" scripts/check_eval_environment.py --profile flow --skip-assets

# 仅画曲线额外需要matplotlib；先检查，已有兼容版本可以保留。
"$SCD_FLOW_PY" -m pip show matplotlib
# 缺失时显式安装到这个独立环境，脚本本身不会安装：
"$SCD_FLOW_PY" -m pip install -r requirements-eval-curves.txt
"$SCD_FLOW_PY" -m pip check
```

若这台SLAI机器没有上述smoke环境，显式创建新前缀，复用已有环境时不用创建：

```bash
python scripts/setup_eval_environment.py --profile flow --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-flow-curves-v1 --dry-run
# 确认目标盘空间与平台runtime后，执行相同配方：
python scripts/setup_eval_environment.py --profile flow --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-flow-curves-v1 --apply
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-flow-curves-v1/bin/python
"$SCD_FLOW_PY" -m pip install -r requirements-eval-curves.txt
"$SCD_FLOW_PY" -m pip check
```

setup拒绝覆盖已有前缀；不要向base或运行中的训练环境安装。正式开始后固定代码、
依赖和资源身份，GPU数量可改变。环境已做过smoke也不等于本轮全部checkpoint已验证。

## 2. 核对训练根目录、评分资源与持久存储

以下默认值对应此前Flow实验路径。实际训练目录不同时修改这一个变量。
同一次评估使用固定prepared和output，不把输出放在训练目录下面。

```bash
export SCD_CURVES_RUNS_ROOT=/mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs
export SCD_CURVES_PREP=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-v1-prepared
export SCD_CURVES_OUT=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-v1
df -h /mnt/aoss/xietianyu /mnt/afs/L202500030/xietianyu
```

`configs/resources.local.json` 覆盖机器路径；也可以设置 `SCD_RESOURCES` 指向另一份
资源表。主要核对 `PICKSCORE_MODEL`、`PICKSCORE_PROCESSOR` 和已有benchmark路径。
SD3.5基座直接使用各run的 `training.json` 记录；已有训练基座无需再下载。
所有选定方法必须使用一致的基座身份。

若PickScore已经在本机缓存，将上面两个资源键指向完整本地目录即可。下载操作独立于
评估执行；只有确实缺失时才显式执行下面的下载，先看plan中的目标和空间：

```bash
"$SCD_FLOW_PY" scripts/download_assets.py plan pickscore pickscore-processor
"$SCD_FLOW_PY" scripts/download_assets.py download pickscore pickscore-processor
"$SCD_FLOW_PY" scripts/download_assets.py check pickscore pickscore-processor
```

新目录通过 `--out` 另存时也要同步资源键。手动缓存可能没有下载receipt，download
`check`会提示这一点；生成/评分阶段的 `--check-only`仍可检查实际本地资产。
不要为了这条PickScore曲线下载HPSv3、UnifiedReward、GenEval detector或DPG mPLUG。

## 3. 一次准备并冻结128条prompt和全部checkpoint

```bash
bash scripts/run_sd35_curves.sh prepare --generation-batch-size 4
bash scripts/run_sd35_curves.sh plan
```

`prepare`默认选择计划训练3200步的七种非Base方法，要求每种方法有唯一匹配run，
读取各run已发布的永久checkpoint；不以rolling/latest代替，也不读取optimizer tensor。
它从配置的caption清单里选取validation文本，排除训练split文本和可用正式面板文本，
固定128条互异prompt。数量不足就报错，不回退到训练或test数据。
初次扫描大caption清单可能需要时间；成功后复用准备文件，不反复全量扫描。

准备结果保存在 `$SCD_CURVES_PREP`：

```text
prepare.json                  # request、训练来源、缺失排除面板、文件hash
catalog.json                  # 七方法及各自全部永久checkpoint
suite.json / scorers.json     # 本轮固定生成设置与PickScore路径
prompts/pickv2.jsonl           # 128条开发集prompt，沿用panel格式，不是正式PickV2
prompts/selection.json         # 文本归一化、split排除与来源记录
```

自动排除使用资源配置中真实存在的五套面板：PickV2、HPSv2、PartiPrompts、GenEval、
DPG。缺失面板会在prepare输出和manifest中记录，`independence=not-established`；
可先画开发曲线，但不能宣称已经与全部正式测试集独立。五套文件都在时，仅保证与配置
的面板文本去重。额外排除文件可重复传入，不会自动下载任何benchmark源：

```bash
bash scripts/run_sd35_curves.sh prepare \
  --exclude-prompts /absolute/path/additional-panel.jsonl
```

上面是另一份输入配方；若已执行过默认prepare，改变输入要同时换新的prepared/output。
需要显式准备三套偏好面板供排除时，可使用约0.9MB的固定官方prompt下载：

```bash
"$SCD_FLOW_PY" scripts/prepare_eval_prompts.py --download-sdpo
```

已有GenEval/DPG的suite文件可通过 `GENEVAL_PROMPTS` / `DPG_PROMPTS` 资源键复用，
无需下载其模型或完整source。缺失时这条流程仍会保留明确的未核验状态。

同方法有重复候选时，prepare会拒绝猜选。先用run路径过滤，多个过滤条件重复传：

```bash
bash scripts/run_sd35_curves.sh prepare --run-glob '*YOUR_FORMAL_TAG*'
```

或者先导入、审阅登记表，删去重复的smoke/backup/旧run，然后传给prepare。登记表必须
覆盖七个训练方法；曲线准备仍枚举选中run的全部永久checkpoint，Base自动单独加入：

```bash
"$SCD_FLOW_PY" evaluate_suite.py catalog --runs-root "$SCD_CURVES_RUNS_ROOT" \
  --families sd35_medium --out /mnt/aoss/xietianyu/self_correct/evaluations/sd35-curves-candidates.json
# 审阅登记表后，再用尚不存在的新prepared/output：
bash scripts/run_sd35_curves.sh prepare \
  --catalog /mnt/aoss/xietianyu/self_correct/evaluations/sd35-curves-candidates.json
```

用过 `--catalog`、`--run-glob`、`--exclude-prompts` 或 `--target-steps` 的prepare，
之后重复prepare时保留相同参数。单独运行后续plan/generate/score无需这些选择参数。
完成prepare后新增训练checkpoint不会偷偷加入；评估新快照建立v2 prepared和v2 output。

## 4. 两卡或四卡并行生成，再独立PickScore

先读取平台分配的设备列表：

```bash
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-UNSET}"
```

`--gpus`使用平台列表中的实际token，不重新解释为列表下标。例如环境为`2,3`就传`2,3`；
环境为`0,1,2,3`时可用其中两张或全部四张。GPU UUID同样原样传递。每个worker内部
只看到自己的一张卡，并按checkpoint任务分工；不是把一个模型跨多卡拆分。

若`CUDA_VISIBLE_DEVICES`已设置，launcher默认使用整个列表；它明确为空或`-1`时拒绝
启动GPU阶段。变量未设置时默认`0,1`，这仅是命令默认值，仍须先确认平台确实已分配这
两张卡。不要手工解除平台设备隔离。

```bash
# 示例：平台分配了物理token 0,1，先预检再实际运行。
bash scripts/run_sd35_curves.sh generate --gpus 0,1 --check-only
bash scripts/run_sd35_curves.sh generate --gpus 0,1
bash scripts/run_sd35_curves.sh score --gpus 0,1 --check-only
bash scripts/run_sd35_curves.sh score --gpus 0,1
bash scripts/run_sd35_curves.sh summarize

# 四卡分配且token确实为0,1,2,3时：
# bash scripts/run_sd35_curves.sh generate --gpus 0,1,2,3
# bash scripts/run_sd35_curves.sh score --gpus 0,1,2,3
```

无需给PickScore再加载基座、adapter或训练状态；只消费完整生成图片。
同一checkpoint始终保存128张PNG；图片总量为`128 × (1 + 七方法checkpoint总数)`。
先查看plan总量和目标盘空间，生成耗时以本轮session记录为准。

顺序自动执行全部阶段可用下面一条；它仍是先完成生成阶段，再执行评分阶段：

```bash
bash scripts/run_sd35_curves.sh all --gpus 0,1
```

`all`默认在独立任务出错后继续其余任务和summary，最终保持非零退出状态；准备/计划
失败会跳过GPU阶段，但尝试汇总已存在结果。退出130/143表示中断，立即停止后续阶段。
单独运行summary可以随时更新当前已完成点；缺失或失败点显示为缺口，不填0，也不当成功。

## 5. 恢复、看图和保存记录

中断后重跑相同generate/score命令，已完成图片和分数经核验后复用。重新分配GPU以后，
可以从两张改成四张或反向改变，继续用相同prepared/output：任务重新分片，不需要删图。
改变prompt、seed、采样协议、checkpoint集合、模型或评分身份应新建版本目录。
不要同时启动两个覆盖相同GPU和同一输出的launcher实例。

```bash
# 重新分配到四张卡后，保留原输出路径恢复：
bash scripts/run_sd35_curves.sh generate --gpus 0,1,2,3
bash scripts/run_sd35_curves.sh score --gpus 0,1,2,3
bash scripts/run_sd35_curves.sh summarize
```

结果保存在：

```text
$SCD_CURVES_OUT/
  plan.json / prompts/...
  <method>/step-.../pickv2/render/images/*.png
  <method>/step-.../pickv2/scores/pickscore/...
  curves/points.csv / points.json
  curves/pickscore.png / pickscore.pdf
  curves/status.json
$SCD_CURVES_OUT/logs/<UTC时间>-<PID>-<阶段>-<随机后缀>.log
```

实际任务相对路径以plan为准。曲线显示各checkpoint的prompt均值及标准误，阴影不是
置信区间；Base是水平线。`status.json`保存完整状态、计数和文件hash。保留全部PNG、
per-image分数、manifest和prepared目录；选展示图不改变统计分母。
脚本日志使用`tee`且保留pipeline失败状态，不覆盖已有日志。

仅预览shell将执行的命令可加`--dry-run`，不执行阶段、不创建日志：

```bash
bash scripts/run_sd35_curves.sh all --gpus 0,1 --dry-run
```

本次新增launcher通过CPU替代进程检查：普通失败仍汇总、130/143立即停止、空GPU分配
拒绝、catalog/含空格路径正确转发和dry-run无写入。没有在本机执行完整128图GPU推理；
真实GPU数值、速度和峰值显存以SLAI运行记录为准。
