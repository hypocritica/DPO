# SDXL 四行补评：HPS 一组，Parti + PickV2 一组

这是开发监控曲线选点的额外对照，保留原2000步主表。DSPO与Stagewise DSPO K6固定3000，
DMPO固定10000，Stagewise DMPO K6只选最新永久LoRA（step-*，最多10000步）：有10000
就用10000，否则例如9000。忽略latest.json、resume-*与9525滚动checkpoint，不复制或提升滚动权重。
初始化核对run/checkpoint身份与所选权重哈希；若最新永久权重损坏就报错，不静默回退。
真实模型加载在GPU生成时验证；初始化成功不等于实际加载已验证。
首次成功生成inputs/catalog.json后，重跑固定该选择，不随训练进度更新。改选步数使用新root。

协议直接读取原evaluation-suite-sdxl-paper-v1.json：1024²、Euler50、CFG5、BF16、seed17、
每卡batch4，五指标PickScore/HPSv2/ImageReward/CLIP/Aesthetic。两个组共用selection短锁；
组内先完成全部生成，再评分、汇总、导出。A/B各使用独立四卡5090任务，全部产物保存AOSS。

| 组 | 面板 | 最多图片数 |
|---|---|---:|
| A | HPSv2 | 12800 |
| B | PartiPrompts + PickV2 | 8476 |
| 合计 | 三面板、四checkpoint | 21276 |

生成前检查原paper-2000-5090-v1的main和supplement目录，仅在模型、准确权重、完整sampling和
prompt哈希一致时导入完成结果。DSPO两行3000 Parti若已完成，可少生成3264张，总新增18012张。
导入复制原图及已完成且可验证的分数，记录reuse-source.json；会占一份新存储，不改旧结果。
正常generate/score恢复仍会核对图片和评分器身份，配置不符报错，不冒用旧分数。

## 更新和固定选择：SSH终端

先在没有worker使用当前checkout时pull分支。若原任务还在运行，可fetch目标提交，再只用
git show导出本次新增scripts/prepare_sdxl_peak.py、scripts/run_sdxl_peak.sh；旧核心代码不需要改。
四个准确run沿用configs/evaluation-sdxl-parti-supplement.json的历史run_id核对。

```bash
(
set -eo pipefail
git config --global --add safe.directory /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate eval
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
git pull --ff-only origin feat/unified-evaluation-slai-20261111
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_EVAL_PY="$SCD_SD15_PY"
export SCD_SDXL_PEAK_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/peak-5090-v1
bash scripts/run_sdxl_peak.sh A init
)
```

记录终端打印的四个step。也可在SSH依次执行A plan和B plan，提前检查面板和完整数量。
init核验小manifest和权重文件哈希，使用CPU；不提交任务、不启动GPU、不重新下载模型。

## 两个独立任务

下面命令用于A组HPS。B组使用同一段，只把最后一行A改为B，各申请自己的4张5090。
已有eval环境、资源和模型路径全部复用。SCD_SDXL_PEAK_ROOT两组必须相同。

```bash
(
set -eo pipefail
git config --global --add safe.directory /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate eval
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$PWD"
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_EVAL_PY="$SCD_SD15_PY"
export SCD_RESOURCES="$PWD/configs/resources.json"
export SCD_EVAL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SCD_SDXL_BATCH=4
export SCD_SDXL_CONFIG=configs/evaluation-suite-sdxl-paper-v1.json
export SCD_SDXL_PEAK_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/peak-5090-v1
export SCD_SDXL_PREVIOUS_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/paper-2000-5090-v1
bash scripts/run_sdxl_peak.sh A all
)
```

重复相同命令恢复。单独执行generate/score/summarize/paper支持分阶段重跑，评分阶段不重新
读取训练目录；all会先再次核对固定选择。原结果位于hpsv2、partiprompts、pickv2各自目录，
论文导出在每面板paper/peak-v1，包含真实step和checkpoint_id；图片索引为image-index.csv/json。
本地只做CPU检查；远程实际权重加载、5090显存和速度需要任务日志确认。
