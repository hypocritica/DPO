# Z-Image：先用固定 128 prompts 选 checkpoint

只比较 **Base、DPO、Stagewise K4**。这里的 128 是每个 checkpoint 的图片数，
不是训练步数。所有已保留的永久 checkpoint 共用 128 条 validation prompts、seed 17，
Base 只生成一次。默认先看已具备资产的 PickScore；可对同一批 PNG 追加 HPSv3。
不增加 SFT、LinearDPO、K2，不自动选最好或最新步数。

图片、逐图分数、来源清单、曲线和正式结果都保留在 AOSS。生成不要求评分模型已下载。
本机仅完成 CPU 测试；以下命令在 SLAI 已分配 GPU 后执行，不申请平台资源。

## 直接执行：两个SSH阶段和一个GPU任务

先在SSH更新checkout，再启动GPU任务；不要在它运行期间pull这个checkout。
SDXL盘点只读取training/checkpoint小文件，不加载tensor；结果不代表权重内容已通过hash验证，
也不据此断言训练进程当前状态。可在训练继续时重新盘点，每次保存独立快照。

**第一组：SSH更新与SDXL checkpoint盘点。**

```bash
(
set -euo pipefail
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
git pull --ff-only origin feat/unified-evaluation-slai-20261111
git log -1 --oneline
eval_py=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
inventory_root=/mnt/aoss/xietianyu/self_correct/evaluations/inventory
mkdir -p "$inventory_root"
inventory_out=$(mktemp -d "$inventory_root/sdxl-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")
"$eval_py" evaluate_suite.py list --families sdxl \
  --runs-root /mnt/aoss/xietianyu/self_correct/runs --json > "$inventory_out/checkpoints.json"
"$eval_py" - "$inventory_out/checkpoints.json" <<'PY'
import json, sys
for run in json.load(open(sys.argv[1])):
    print(run['name'])
    print('  永久 checkpoint 步数:', [c['step'] for c in run['checkpoints']])
print('完整路径及身份记录:', sys.argv[1])
PY
)
```

**第二组：SSH下载SD3.5M/Z-Image共用的新评分资产。** 先补HPSv3、Qwen2-VL骨干、
UnifiedReward及固定源码；已有PickScore、ImageReward、CLIP、Aesthetic继续复用。
本轮曲线只需已安装PickScore，不等待这些新资产。这里只下载，不新建或修改环境。
GenEval/DPG资产仍后置，正式HPSv3/UnifiedReward评分使用各自环境，见H800说明。

```bash
(
set -euo pipefail
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
eval_py=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_RESOURCES="$PWD/configs/resources.json"
export HF_HOME=/mnt/aoss/xietianyu/cache/huggingface
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DIFFUSERS_OFFLINE
df -h /mnt/aoss/xietianyu
"$eval_py" scripts/prepare_eval_assets.py plan --profile modern
"$eval_py" scripts/prepare_eval_assets.py download --profile modern
"$eval_py" scripts/prepare_eval_assets.py sources --profile modern
"$eval_py" scripts/prepare_eval_assets.py check --profile modern
)
```

默认下载到AOSS models/preference与models/backbones；resources.local.json/env覆盖时以plan
打印位置为准。已有完整文件由下载器复用。离线check不加载评分模型，不等于实际GPU评分通过。

**第三组：SLAI GPU任务的启动命令。** 使用两张已分配H800/H100 80GB；按平台正常方式提交，
把下面填入任务启动脚本。不要在SSH登录节点执行GPU阶段。这里只安排Z-Image曲线，SDXL
盘点/下载在另两个终端继续。GPU变量无需填写：优先使用已有SCD_Z_GPUS、SCD_EVAL_GPUS、
CUDA_VISIBLE_DEVICES，未设置或为空时默认0,1，与已跑通的H100双卡脚本一致。

```bash
set -euo pipefail
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_EVAL_PY="$SCD_FLOW_PY"
export SCD_RESOURCES="$PWD/configs/resources.json"
export SCD_Z_CURVES=/mnt/aoss/xietianyu/self_correct/evaluations/zimage/curves-h800-b1-v1
export SCD_Z_PREP="${SCD_Z_CURVES}-prepared"
bash scripts/run_zimage_curves.sh all
```

重复任务命令恢复缺失生成/评分；冻结后新增的Z-Image checkpoint需要新prepared/output版本。
所有曲线PNG/PDF/CSV在 `$SCD_Z_CURVES/curves/`，原图在方法/step面板下，由
`image-index.csv/json`索引，阶段日志在`logs/`。没有额外强制环境预检。
若只想先生成，末行改为`generate`（首次先执行`prepare`、`plan`）；之后执行`score`和
`summarize`即可。可选`score-hpsv3`需要SCD_HPSV3_PY，`summarize-hpsv3`复用原eval绘图，
另存`curves-hpsv3/`。同一GPU不要同时启动另一项生图/评分任务。

## 1. 固定输入和计划

从独立 evaluation checkout 执行；复用已有 Flow 和 legacy eval 环境。
下面两个 run 名来自历史盘点，代码会现场验证 training.json 的方法语义及永久快照，
不存在就报错，不替换成其他 run。若远程已更名，使用真实目录名。

```bash
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_Z_CURVES=/mnt/aoss/xietianyu/self_correct/evaluations/zimage/curves-h800-b1-v1
export SCD_Z_PREP="${SCD_Z_CURVES}-prepared"
export SCD_RESOURCES="$PWD/configs/resources.json"

"$SCD_FLOW_PY" evaluate_curves.py prepare \
  --family zimage \
  --runs-root /mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs \
  --run-glob \
    zimage-dpo-k1-pair_views128-2h100-b4-3200-C-float32-trainonly-paper-v2-flow-b4-stat-v1 \
    zimage-mvpo-k4-pair_views128-2h100-b1-3200-C-float32-trainonly-stage500-paper-v2-flow-microviews4-v1 \
  --selection-metric pickscore \
  --generation-batch-size 1 --cpu-offload false \
  --out "$SCD_Z_PREP"

"$SCD_FLOW_PY" evaluate_curves.py plan --prepared "$SCD_Z_PREP" \
  --out "$SCD_Z_CURVES" --dry-run
"$SCD_FLOW_PY" evaluate_curves.py plan --prepared "$SCD_Z_PREP" \
  --out "$SCD_Z_CURVES"
```

数量为 `128 × (1 + DPO 的快照数 + K4 的快照数)`，以计划输出为准。
默认保留训练记录里的 Z-Image Base、transformer float32、1024²、50 步、CFG 4；
每卡 batch 1、CPU offload=false 是未实测的起点。改变 batch/offload 要用新 prepared/output，
不能把不同生成设置的图片拼进同一条曲线。

prepare 会按规范化文本去重、排除训练文本和已准备的正式面板。若 GenEval 等排除面板
尚未准备，会明确记录独立性尚未建立；128 图应称为开发/选点面板，不能直接报成完整
PickV2 或正式测试集。准备了缺失面板后若要重新抽样，使用新版本，不改旧冻结输入。

## 2. 两卡先生成、保存，再打分

下方底层CLI命令同样沿用原H100脚本的默认双卡逻辑；快捷脚本已内置，无需额外设置。

```bash
export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
"$SCD_FLOW_PY" evaluate_curves.py generate \
  --out "$SCD_Z_CURVES" --gpus "$SCD_EVAL_GPUS" --python "$SCD_FLOW_PY"
"$SCD_FLOW_PY" evaluate_suite.py index --out "$SCD_Z_CURVES"

# 生成完成后再使用原 eval 环境；不会重生成 PNG。
"$SCD_SD15_PY" evaluate_curves.py score \
  --out "$SCD_Z_CURVES" --gpus "$SCD_EVAL_GPUS" \
  --python "$SCD_SD15_PY" --metrics pickscore
"$SCD_SD15_PY" evaluate_curves.py summarize --out "$SCD_Z_CURVES"
```

主曲线输出为 `curves/points.csv`、`curves/points.json`、`curves/pickscore.png` 和 PDF。
缺失/失败点保持空缺，不填 0；误差带是 prompt 标准误。完全相同命令可断点恢复。
不需要重新运行完整环境预检；权重、prompt、生成与评分产物的身份核验仍保留。

若先按部分方法或步数生成，可对 generate/score 加 `--methods dpo stagewise_k4` 或
`--steps ...` 筛选冻结计划。只有完整 checkpoint 点才参与其曲线均值。

## 3. 可选 HPSv3，仍用上述图片

HPSv3 环境/权重准备见 [H800 正式评测说明](H800_SELECTED_EVALUATION.md)。下载与生成可以分开。

```bash
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python
"$SCD_HPSV3_PY" evaluate_curves.py score \
  --out "$SCD_Z_CURVES" --gpus "$SCD_EVAL_GPUS" \
  --python "$SCD_HPSV3_PY" --metrics hpsv3
# 绘图可继续用已有包含 matplotlib 的 eval 环境：
"$SCD_SD15_PY" evaluate_curves.py summarize --out "$SCD_Z_CURVES" --metric hpsv3
```

追加曲线在 `curves-hpsv3/`，不会覆盖 PickScore 曲线。默认事先固定 PickScore 为选点指标，
HPSv3 作为辅助观察；不要看完正式结果后为每个指标分别挑 checkpoint。
如果从一开始决定按 HPSv3 选点，应在首次 prepare 使用 `--selection-metric hpsv3`
及独立版本目录；此时主曲线在 `curves/`，辅助 PickScore 在 `curves-pickscore/`。

## 4. 将选定步数交给正式评测

```bash
export SCD_SELECTED_FAMILY=zimage
export SCD_SELECTED_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/zimage/formal-h800-v1
export SCD_SELECTED_SELECTION="$SCD_SELECTED_ROOT/selection.json"
bash scripts/run_h800_selected.sh init --from-prepared "$SCD_Z_PREP"
```

打开 selection.json，只填写 DPO 和 Stagewise K4 的 selected_step。Base 固定为 0。
后续按 [正式评测说明](H800_SELECTED_EVALUATION.md) 分阶段执行 Parti、GenEval；
两边选点必须相同。三行五指标的组合表使用 export_zimage_table.py，全部图片保留以供论文展示。
