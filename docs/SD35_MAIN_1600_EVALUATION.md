# SD3.5M：1600 步主表与 DPO 800 步额外对照

按用户 2026-11-14 确认的交接清单执行。主表为 Base，以及 SFT、DPO、LinearDPO、
Global K2、Stagewise K2、Global K4、Stagewise K4 的1600步；额外生成 DPO800，
标注 development-selected。共9个模型版本，主表8个，DPO步数对照表包含800与1600。
同一版本的图片生成一次，多个评分器及两个报告复用。

3200步保留已有完整开发曲线，本轮不安排正式benchmark，不把开发曲线分数填入正式表。
1600是在查看开发曲线后确定的共同优化更新数，不称为事前预注册，也不称所有方法等计算量。
各方法主表步数不再按正式测试分数调整，DPO800不能取代主表DPO1600。

## 1. 更新并导入已确认选择

在 SLAI 的独立 evaluation checkout 内执行。已有进程完成后再更新这个checkout。
已完成的 SD1.5 结果和旧 SD3.5M 开发曲线不需要重跑。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
git pull --ff-only origin feat/unified-evaluation-slai-20261111
git log -1 --oneline
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_EVAL_PY="$SCD_FLOW_PY"
export SCD_RESOURCES="$PWD/configs/resources.json"
export SCD_SELECTED_FAMILY=sd35_medium
export SCD_SELECTED_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/main1600-dpo800-h800-v1
export SCD_SELECTED_SELECTION="$SCD_SELECTED_ROOT/selection.json"
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/parti"

bash scripts/run_h800_selected.sh init \
  --from-prepared /mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-h100-b8-v1-prepared \
  --handoff configs/evaluation-sd35-main-selection.json
bash scripts/run_h800_selected.sh check
```

`init` 自动填写确认的步数，不需要手工改 JSON。来源是实际冻结曲线 catalog，SFT必须有
唯一 sft 条目且保留1600步；六个旧run和DPO800须匹配交接中的路径、run_id及checkpoint_id。
缺失或不匹配就报错，不猜目录、不回退latest。生成前沿用原框架核验adapter内容hash。
初始化只写新选择文件；已经成功init后不再重复init，直接check/plan即可。

新输出根目录避免接管先前的待填或已封存计划。协议沿用当前正式suite：1024²、28步、
CFG4.5、BF16、seed17/29/43/71，现代六指标；CPU offload=false、每卡batch8。
交接JSON中的sampling_reference保留历史cpu_offload=true，仅作来源记录，不覆盖当前
执行设置。实际协议全部写入新selection和plan，不改变旧曲线的生成身份。

## 2. 先生成 Parti，之后分别评分

```bash
bash scripts/run_h800_selected.sh plan --benchmarks partiprompts --dry-run
# 此处应显示9个任务、58752张图片：Base0、七方法1600、额外DPO800。
bash scripts/run_h800_selected.sh plan --benchmarks partiprompts

# 两张已分配GPU的真实token。若平台未设置CUDA_VISIBLE_DEVICES，先显式设置SCD_EVAL_GPUS。
export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:?Set allocated GPU tokens}}"
bash scripts/run_h800_selected.sh generate

# 生成成功后已自动写image-index.csv/json；以下只读同一批PNG。
bash scripts/run_h800_selected.sh score-legacy
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python
export SCD_UNIFIEDREWARD_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-unifiedreward-v1/bin/python
bash scripts/run_h800_selected.sh score-hpsv3
bash scripts/run_h800_selected.sh score-unifiedreward
```

score-legacy 在SD3.5M上运行PickScore、ImageReward、CLIP、Aesthetic四项；另两项使用
各自环境。缺失现代权重时可先生成并做现有评分，准备命令见
[H800环境与资产说明](H800_SELECTED_EVALUATION.md)。不为六个指标分别生图。

```bash
bash scripts/run_h800_selected.sh summarize --summary-name all-nine-variants
# 固定分为主表和DPO步数对照；不会按分数挑步数。
bash scripts/run_h800_selected.sh paper --report all \
  --paper-name main1600-v1 --comparison-name dpo800-vs1600-v1
bash scripts/run_h800_selected.sh gallery --gallery-name parti-v1 --limit 80 --copy-images
```

`paper` 默认只导出main；`--report all`同时导出两个报告。主表排除DPO800，步数对照
只包含DPO800和DPO1600。未完成评分会拒绝正式导出；显式`--allow-incomplete`可导出
缺失标记的草稿，但返回非零状态。补齐后用新的paper-name/comparison-name保留报告版本。

## 3. 后续面板沿用同一选择

同一selection已封存，后面的计划不能改步骤或基座。每个面板可使用独立目录推进：

```bash
# 第二批PickV2；第三批将pickv2替换成hpsv2。
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/pickv2"
bash scripts/run_h800_selected.sh plan --benchmarks pickv2 --dry-run
bash scripts/run_h800_selected.sh plan --benchmarks pickv2
bash scripts/run_h800_selected.sh generate
# 然后重复上一节score-legacy、score-hpsv3、score-unifiedreward及paper命令。
```

| 面板 | 9个版本图片数 |
|---|---:|
| PartiPrompts | 58,752 |
| PickV2 | 17,532 |
| HPSv2 | 115,200 |
| GenEval（后置） | 19,908 |
| DPG（后置） | 38,340 |
| 全五面板 | 249,732 |

前三套合计191,484图。GenEval/DPG仍用专用prompts和各自评分器，后置执行；只需改变
输出目录和`plan --benchmarks geneval`/`dpg`，生成后分别使用score-geneval/score-dpg。
不会默认启动全部五面板。各面板保留原图、采样身份、逐图分数、图片索引和论文导出。
