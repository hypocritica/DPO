# 两张 H100：先恢复 Z-Image 分段 PickScore，再运行 SD3.5M 精简正式评测

本方案按2026-11-15确认的范围执行；只在平台已分配两张H100 80GB的容器中运行。
脚本不申请资源、不安装依赖、不下载模型。先等待旧evaluation worker退出，再更新checkout。

## 评测范围

| 模型 | 方法 | PartiPrompts | Pick-a-Pic v2（pickv2） | GenEval | 正式图片总数 |
|---|---|---|---|---|---:|
| SD3.5M | Base0 + 七方法1600 | 1632×2 seeds | 487×2 seeds | 553×4 seeds | 51,600 |
| Z-Image | Base、DPO、Stagewise K4 | 1632×1 seed | 487×1 seed | 553×4 seeds | 12,993 |

SD3.5偏好seed为17、29，Z-Image为17，GenEval统一17、29、43、71。
各面板使用完整prompt集，同模型不同方法共享采样协议。PickV2这里只用prompt生成图片，
并不是重新评测原始人类偏好图片对。两个生成seed不代表重复训练实验。

主表偏好指标为PickScore、HPSv3、ImageReward；GenEval使用专用评分器。
UnifiedReward不进入核心评分/导出依赖，仍可对Base、DPO、Stagewise K4的Parti图片追加评分。
不安排HPSv2/DPG面板，不额外生成CLIP/Aesthetic图片。

SD3.5沿用已确认的1600步主表和精确曲线catalog。DPO800保留在交接selection里，
默认`selected_report=main`不生成；需要补充对照时在独立输出目录使用`plan --report all`。
旧九版本/四seed计划、旧selection和旧曲线均不改写。新协议在独立compact目录冻结。

## 最终串行命令

默认复用用户已运行的Z-Image `curves-h800-b4-v1` 和对应prepared，batch4。
已完成PNG逐张核验复用，缺失部分用BF16 Transformer autocast；原FP32图片及身份保留。
不修改renderer或放宽上一版的精确SHA兼容策略，混合精度记录沿用原审计和逐图字段。

脚本依次完成：Z-Image prepare→plan→generate→PickScore→曲线/计时；随后
SD3.5 init/check→三面板计划→batch8生成→PickScore/ImageReward→HPSv3→GenEval→汇总/论文导出。
任一阶段失败立即停止，原命令重跑恢复；重复执行不会覆盖论文导出。
Z-Image正式表需要先根据曲线填写两个选定步数，不在此串行命令中自动选择或运行。

```bash
(
set -eo pipefail
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
git -c safe.directory="$PWD" pull --ff-only origin feat/unified-evaluation-slai-20261111
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$PWD"
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate eval

# 默认取平台CUDA_VISIBLE_DEVICES；未设置时用用户原任务的0,1。
export SCD_EVAL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_EVAL_PY="$SCD_FLOW_PY"
export SCD_RESOURCES="$PWD/configs/resources.json"
# 必须是已有可用环境；如果实际环境路径不同，在这里修改，不自动安装。
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python
export SCD_GENEVAL_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-geneval-v1/bin/python

bash scripts/run_zimage_then_sd35_compact.sh
)
```

safe.directory仅作用于本命令及子进程，不修改global Git配置。优先保持已有全局设置时
也可沿用用户原来的safe.directory命令。脚本校验指定Python存在，但不把这当作GPU/资产验证。
如现代环境尚未准备，可先单独运行原Z曲线五条阶段命令；评分环境/资产准备仍参考
H800_SELECTED_EVALUATION.md。所有GPU命令只在已分配资源中执行。

SD3.5新输出为 `/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/compact-h100-v1/core`。
`image-index.csv/json`索引所有原图，`core-v1.csv/json`为评分状态与汇总，
`paper/core-<plan_id前12位>/`保存各面板的CSV/JSON、Markdown与LaTeX表。
HPSv3/GenEval/legacy worker各自使用独立环境，不需要UnifiedReward环境。

## 单独推进或补充

```bash
export SCD_SELECTED_FAMILY=sd35_medium
export SCD_SELECTED_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/compact-h100-v1
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/core"
bash scripts/run_compact_evaluation.sh plan --dry-run
bash scripts/run_compact_evaluation.sh score-legacy
bash scripts/run_compact_evaluation.sh score-hpsv3
bash scripts/run_compact_evaluation.sh score-geneval
# 可选，仅三方法Parti，复用现有PNG；设置已有SCD_UNIFIEDREWARD_PY后执行。
bash scripts/run_compact_evaluation.sh score-unifiedreward
```

Z-Image正式表初始化也使用`run_compact_evaluation.sh init --from-prepared "$SCD_Z_PREP"`，
新根目录取Z-Image的compact路径；手填两个精确selected_step后check/plan。不要覆盖已封存选择。
新核心表使用通用`paper`阶段导出三面板结果，旧`export_zimage_table.py`专供历史四偏好指标表。

## 时间预算与测量边界

本机没有此H100任务的实测日志，也不访问远端GPU；以上命令是在用户容器里实际执行的入口。
先前估算不能当作实测吞吐。SD3.5纯生成51,600张，在单卡batch8平均1/2/4秒每图时，
两卡分别约7.2/14.3/28.7小时（不含加载、存储和评分）。
评分图片为33,904张偏好图和17,696张GenEval图；每增加1秒/图的单卡评分开销，
对应阶段两卡约增加4.7小时和2.46小时。没有昂贵UnifiedReward的整套核心评分，
初步可按SD3.5约12–36小时预算，但真实性能更慢时会超出，不作完成时间保证。

Z曲线共`128 × (1 + DPO永久点数 + K4永久点数)`张，已完成部分跳过。
若BF16单卡batch4平均6–12秒/图，剩余每1000图约0.83–1.67小时，另加加载/读写和PickScore；
实际checkpoint数和缺失图数以冻结plan/manifest为准，不猜一个固定总时长。
串行总时长为Z剩余曲线时间加SD3.5时间，不包括Z-Image正式三面板。
每个生成session保存实际GPU型号、batch、forward秒数、图片数和峰值显存，串行脚本自动打印
`report_generation_sessions.py`结果；用`forward_seconds/images_computed`校准以上每图假设。
