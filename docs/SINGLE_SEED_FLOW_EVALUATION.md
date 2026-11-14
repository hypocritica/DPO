# Parti/HPSv2单seed与四张5090独立评测

按2026-11-15后续要求，使用`evaluation-suite-compact-v2.json`：

| 面板 | SD3.5M seeds | Z-Image seeds | SD3.5M八版本图片数 |
|---|---|---|---:|
| PartiPrompts（1632 prompts） | 17 | 17 | 13,056 |
| Pick-a-Pic v2（487 prompts） | 17、29 | 17 | 7,792 |
| GenEval（553 prompts） | 17、29、43、71 | 同左 | 17,696 |
| HPSv2（3200 prompts） | 17 | 17 | 25,600 |

两个H100继续先跑原Z-Image batch4 BF16续跑/PickScore曲线，之后跑SD3.5M前三套面板，
总38,544图、batch8/offload=false。新输出根目录默认`sd35_medium/compact-h100-v2`。
原`compact-h100-v1`的双seed Parti计划不改写，旧曲线和已生成PNG保持原身份。
Parti减少seed不会改变PickV2或GenEval；四seed GenEval仍用原评分协议。

四张5090独立生成/评测SD3.5M主表八版本的HPSv2面板，共25,600图。
各方法沿用同一曲线catalog与已确认Base0/七方法1600，DPO800仍为可选补充。
HPSv2在这里是prompt面板，评分保持PickScore、HPSv3、ImageReward，不切换成HPSv2 scorer。
不额外运行GenEval/DPG或UnifiedReward。

## 双H100入口

停止旧evaluation worker后，在原checkout拉取本分支，再执行原命令：

```bash
bash scripts/run_zimage_then_sd35_compact.sh
```

环境与GPU设置沿用[双H100文档](COMPACT_H100_EVALUATION.md)。不要把
`SCD_SD35_COMPACT_ROOT`指向旧v1目录；如果之前显式设置过它，改为新的v2路径。
原Z-Image曲线路径继续保持不变。

## 四张5090独立完整命令

在平台已分配的4×RTX5090容器中执行；使用独立checkout，以便与H100任务并行。
目录已存在时先确认没有该checkout的worker仍在运行，再pull。

```bash
(
set -eo pipefail
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate eval
cd /mnt/afs/L202500030/xietianyu/workspace
if [ ! -d DPO-eval-hpsv2-5090-v2 ]; then
  git clone --single-branch --branch feat/unified-evaluation-slai-20261111 \
    https://github.com/hypocritica/DPO.git DPO-eval-hpsv2-5090-v2
fi
cd DPO-eval-hpsv2-5090-v2
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$PWD"
git pull --ff-only origin feat/unified-evaluation-slai-20261111

export SCD_EVAL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_EVAL_PY="$SCD_FLOW_PY"
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python
# 复用原机器资源表及其resources.local.json；不向运行中的checkout写入。
export SCD_RESOURCES=/mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2/configs/resources.json

bash scripts/run_hpsv2_5090.sh all
)
```

四卡脚本严格校验四个不同GPU token，一卡一模型worker，不把四张卡显存合并。
每卡默认batch1、模型CPU offload=true，适应5090的显存预算；不是已经实测通过的显存结论。
需要已有支持5090的PyTorch/CUDA版Flow、legacy eval和HPSv3环境，脚本不安装依赖或下载模型。
`--cpu-offload`与batch一起冻结在独立selection/plan里；不得在已有计划中途改变。
需要其他batch时在新`SCD_HPS_ROOT`下通过init显式设置，后续所有阶段使用同一个根目录。

脚本顺序为init/check/plan→generate→原图索引/吞吐→PickScore/ImageReward→HPSv3→summary→论文导出。
失败立即停止，重复运行同一命令恢复；不会写入H100的core目录。
HPSv2源文件必须是既有完整3200条面板，不会隐式下载或替换成HPDv2 test400。

输出根目录：`/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/hpsv2-5090-seed17-v2`。
`hpsv2/image-index.csv/json`用于原图查找，`hpsv2/hpsv2-seed17.csv/json`为汇总，
`hpsv2/paper/hpsv2-seed17/`保存CSV/JSON、Markdown/LaTeX论文表。日志位于`hpsv2/logs`。

## 分段检查与恢复

```bash
# 同一组环境变量下，不需要GPU的计划预览：
bash scripts/run_hpsv2_5090.sh init  # 仅首次创建选择
bash scripts/run_hpsv2_5090.sh plan --dry-run
# 独立阶段，按需恢复：
bash scripts/run_hpsv2_5090.sh generate
bash scripts/run_hpsv2_5090.sh score
bash scripts/run_hpsv2_5090.sh summarize --summary-name hpsv2-seed17
bash scripts/run_hpsv2_5090.sh paper --report main --paper-name hpsv2-seed17
```

本机只做CPU与shell验证，未提交资源或运行5090实测。脚本结束打印实际秒数，并读取生成session中的
GPU型号、batch、每图forward时间和峰值显存。纯生图时间约为`25600 × 单卡秒/图 ÷ 4`；
例如3/6/10秒每图对应5.3/10.7/17.8小时，模型加载/CPU offload/存储/评分另计。
这些是计算情景，不是5090实测速度；H100和5090两组任务可以独立并行。
