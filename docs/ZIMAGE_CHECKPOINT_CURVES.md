# Z-Image：先用固定 128 prompts 选 checkpoint

只比较 **Base、DPO、Stagewise K4**。这里的 128 是每个 checkpoint 的图片数，
不是训练步数。所有已保留的永久 checkpoint 共用 128 条 validation prompts、seed 17，
Base 只生成一次。默认先看已具备资产的 PickScore；可对同一批 PNG 追加 HPSv3。
不增加 SFT、LinearDPO、K2，不自动选最好或最新步数。

图片、逐图分数、来源清单、曲线和正式结果都保留在 AOSS。生成不要求评分模型已下载。
本机仅完成 CPU 测试；以下命令在 SLAI 已分配 GPU 后执行，不申请平台资源。

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

`SCD_EVAL_GPUS` 必须是平台已分配的真实 GPU token；已有 CUDA_VISIBLE_DEVICES 时直接继承。
若平台未设置该变量，先根据容器实际分配显式设置 SCD_EVAL_GPUS，例如确为 0、1 时才填 0,1。

```bash
export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:?Set allocated GPU tokens}}"
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
