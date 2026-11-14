# H800：单 checkpoint 正式评测

**SD3.5M 已于2026-11-14确定步数，执行请使用
[1600步主表与DPO800对照命令](SD35_MAIN_1600_EVALUATION.md)。** 该入口自动核验并导入
已确认清单，共9个版本，Parti58,752图；本页下面的八版本规模和手填流程保留为通用说明。
Z-Image继续按独立曲线选择DPO与Stagewise K4的步数。

这套入口只执行独立阶段，不提交任务、不安装环境、不下载资产。使用已分配的两张 H800，
保留原 H100 启动器、renderer 和历史曲线。SD3.5 八项为 Base、SFT、DPO、LinearDPO、
Global K2、Stagewise K2、Global K4、Stagewise K4；Z-Image 只比较 Base、DPO、Stagewise K4。
未提供已确认handoff时，每个训练方法只填一个精确永久 checkpoint；Base 为 step 0。
先建立待填选择表，再根据独立128图开发曲线填入步数。正式 benchmark 分数不参与 checkpoint 选择，
不自动挑 best/latest，不以缺失方法或快照替代。Z-Image 使用其独立选点证据，不冒称来自
SD3.5 曲线。selection 中冻结曲线来源及精确 run/step；选点理由写在相邻 checkpoint-selection-notes.md，
不修改 selection 的 provenance。开始 plan 后不修改选择。

## 资源与协议

启动器默认输出 `/mnt/aoss/xietianyu/self_correct/evaluations/<family>/formal-h800-v1`。
每个 family 使用独立 selection，Parti、preference、composition 子目录分别冻结计划，
都读取同一 selection。新版本用新目录；不覆盖旧实验。PNG、manifest、评分、index、
gallery 和 paper 来源均保留在相应版本下。

| family | 面板 | seeds | 规模 |
|---|---|---|---:|
| SD3.5 Medium | Parti、PickV2、HPSv2 | 17,29,43,71 | Parti 八方法 52,224 图 |
| SD3.5 Medium | GenEval、DPG | 17,29,43,71 | 全五面板八方法 221,984 图 |
| Z-Image Base | Parti | 17 | 三方法 4,896 图 |
| Z-Image Base | GenEval | 17,29,43,71 | 三方法 6,636 图 |

Z-Image 总计 11,532 图；不增加 PickScore/ImageReward 专用图片面板，二者复用 Parti PNG。
SD3.5 保留既有五面板和 modern 六指标：HPSv3、UnifiedReward ACS mean、PickScore、
ImageReward、CLIP、Aesthetic。先 Parti，之后 PickV2/HPSv2，最后 GenEval/DPG。
Z-Image Parti 只评 HPSv3、UnifiedReward、PickScore、ImageReward，GenEval 单独评分。
GenEval 的官方使用先例支持面板选择，不表示本项目协议应复现论文数字。

默认关闭 CPU offload。SD3.5 每卡 batch=8 常驻；先前 16/32 在 VAE 阶段 OOM，不自动
升批或降批。Z-Image batch=1 是尚未实测的起点；保留训练 metadata 的 transformer float32，
不能为省显存静默换成 BF16。更改采样/批量须新计划和版本。Z-Image 两训练方法的实际
pairs_id 和 pair_views=128 预算须核对；K 不同会改变 pair 曝光，选点步数不同也不代表
相同训练更新数，不把此比较描述为相同数据曝光。

## 通用手填初始化（已确认的SD3.5M请用上方专用命令）

在独立 eval checkout 中运行。`SCD_EVAL_PY` 只需能执行 CPU 编排，不等于生成或评分环境。
绝对 Python 路径应替换为已有可用环境；不要重装已有 Flow/legacy eval。

```bash
export SCD_EVAL_PY=python3
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
# 如有机器资源表，在所有阶段保持同一个：
export SCD_RESOURCES="$PWD/configs/resources.json"
export SCD_SELECTED_FAMILY=sd35_medium
export SCD_SELECTED_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/formal-h800-v1
export SCD_SELECTED_SELECTION="$SCD_SELECTED_ROOT/selection.json"
bash scripts/run_h800_selected.sh init --from-prepared /mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-h100-b8-v1-prepared
```

打开 selection，Base 隐式固定 step 0；`choices` 中每方法 `selected_step: null` 待填。
只将 selected_step 改为 available_steps 中一个精确整数；来源/理由另存相邻说明文件。
其余字段参与冻结身份校验，不编辑它们或训练目录。`check` 在 pending 时应
明确拒绝正式计划；补齐后核对冻结身份和 checkpoint。没有每阶段重复全量预检。

```bash
bash scripts/run_h800_selected.sh check
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/parti"
bash scripts/run_h800_selected.sh plan --benchmarks partiprompts --dry-run
# 看清数量、方法、steps、seeds 后发布：
bash scripts/run_h800_selected.sh plan --benchmarks partiprompts
```

Z-Image 先按 [128 图曲线说明](ZIMAGE_CHECKPOINT_CURVES.md) 完成独立选点准备，
再继承它自己的冻结清单，不借用 SD3.5 selection：

```bash
export SCD_SELECTED_FAMILY=zimage
export SCD_SELECTED_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/zimage/formal-h800-v1
export SCD_SELECTED_SELECTION="$SCD_SELECTED_ROOT/selection.json"
bash scripts/run_h800_selected.sh init --from-prepared \
  /mnt/aoss/xietianyu/self_correct/evaluations/zimage/curves-h800-b1-v1-prepared
# 填 DPO、Stagewise K4 精确 step 后：
bash scripts/run_h800_selected.sh check
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/parti"
bash scripts/run_h800_selected.sh plan --benchmarks partiprompts --dry-run
bash scripts/run_h800_selected.sh plan --benchmarks partiprompts
```

不使用曲线时也支持 init --catalog FILE，或 --runs-root 配合 --run-glob 指定两个精确
目录；未筛选的 runs 根目录可能有多个 smoke/旧候选，会拒绝歧义，不自行挑一个。

## 两卡生成、独立评分、保留原图

只在平台已分配两张 H800 后执行。保留实际 `CUDA_VISIBLE_DEVICES` 中的两个 token，或显式
设置 `SCD_EVAL_GPUS` 为真实设备编号/UUID；没有隐式 `0,1`。无需在本机申请 irip Slurm。
重复命令恢复缺失图片，已完成图片按冻结协议核验复用。环境/资产预检可按需单独执行；
启动器不强制全套检查，冻结输入、checkpoint 和事务校验始终保留。

```bash
# 仅首次部署/排错需要，可跳过，不替代真实 H800 推理：
"$SCD_FLOW_PY" scripts/check_eval_environment.py --profile flow --skip-assets --gpu --execution standalone
# 不打印或重写完整环境，先确认实际可见的两个已分配 token：
printf '%s\n' "${CUDA_VISIBLE_DEVICES:?Use two already allocated GPUs}"
bash scripts/run_h800_selected.sh generate
# generate 成功后自动 index；中断后也可独立建索引：
bash scripts/run_h800_selected.sh index
# image-index.csv/json 在评分前即可按模型、方法、step、prompt、seed 找 PNG。
```

评分不会重新生成图片，也不要求生成模型/训练目录仍在。每次只检查并加载请求的 scorer。
启动器的独立 score 阶段直接用 `--python` 指定该 scorer 环境，避免 mixed metric group
把所有评分器塞进一个 Python；无需为了只跑 PickScore 而准备 HPSv3/GenEval。

```bash
# 两 family 都可先复用原 legacy eval 和同一批 Parti PNG：
bash scripts/run_h800_selected.sh score-legacy --metrics pickscore image_reward
# SD3.5 完成 modern 六指标时再补 CLIP/Aesthetic；Z-Image 不需要此条：
bash scripts/run_h800_selected.sh score-legacy --metrics clip aesthetic
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python
export SCD_UNIFIEDREWARD_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-unifiedreward-v1/bin/python
bash scripts/run_h800_selected.sh score-hpsv3
bash scripts/run_h800_selected.sh score-unifiedreward
bash scripts/run_h800_selected.sh summarize --summary-name parti-current
bash scripts/run_h800_selected.sh gallery --gallery-name parti-v1 --limit 80 --copy-images
bash scripts/run_h800_selected.sh paper --paper-name parti-draft-v1 --allow-incomplete
# 全指标齐备后用新的 paper-name，去掉 --allow-incomplete；缺失项从不填 0。
```

后置面板使用同一 selection。SD3.5 先切 `SCD_SELECTED_OUT` 到 `$SCD_SELECTED_ROOT/preference`，
`plan --benchmarks pickv2 hpsv2 --dry-run`，审阅后去掉 dry-run，然后 generate 和上述评分。
GenEval/DPG 再切到 `$SCD_SELECTED_ROOT/composition`；SD3.5 规划 `--benchmarks geneval dpg`，
Z-Image 只规划 `--benchmarks geneval`。先 dry-run，再正式 plan/generate。

```bash
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/composition"
# Z-Image 示例；SD3.5 在 benchmarks 后增加 dpg：
bash scripts/run_h800_selected.sh plan --benchmarks geneval --dry-run
bash scripts/run_h800_selected.sh plan --benchmarks geneval
bash scripts/run_h800_selected.sh generate
export SCD_GENEVAL_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-geneval-v1/bin/python
bash scripts/run_h800_selected.sh score-geneval
# SD3.5 DPG 同理：设置 SCD_DPG_PY 后 score-dpg。
bash scripts/run_h800_selected.sh summarize --summary-name composition-current
```

首次正式 plan 写入 selection.json.frozen.json；之后所有阶段目录必须使用同一组
selected_step，改选会被拒绝。要改变选点，复制选择表到新的版本根目录，不能改旧版本。
selection 不变，分阶段计划不影响方法和选点。gallery 的图片筛选不改变统计分母。
导出以各自冻结计划为单位保留 protocol/hash/selection 来源，论文中按面板组合表格，
不要把不同计划的目录强行合并或覆盖 summary。

Z-Image 两阶段完成后，用独立汇总工具按三个方法合并四个 Parti 指标和 GenEval：

```bash
"$SCD_EVAL_PY" export_zimage_table.py --parti-out "$SCD_SELECTED_ROOT/parti" \
  --geneval-out "$SCD_SELECTED_ROOT/composition" --out "$SCD_SELECTED_ROOT/zimage-table-v1"
```

工具读取既有评分和冻结选择，不生成图片、不重新评分；待缺失分数完成后使用新版本目录。

## 缺失现代评分环境与资产：只准备缺项

以下都是用户在远程显式执行的准备命令，本次没有执行安装或下载。先核对
`configs/resources.local.json` 及 `~/workspace/sources/models`、`~/workspace/sources/datasets`
可复用缓存。下载前对实际目标挂载点执行 `df -h`；若目标在 `/irip`，必须 `df -h /irip`。
已有完整手动 legacy/PickScore 缓存无需因为没有下载 receipt 而重下。

```bash
# metadata/离线阶段不下载权重：
"$SCD_EVAL_PY" scripts/prepare_eval_assets.py check --profile modern
"$SCD_EVAL_PY" scripts/prepare_eval_assets.py plan --profile modern
# 确认路径、空间后，显式下载固定权重并准备固定 source：
"$SCD_EVAL_PY" scripts/prepare_eval_assets.py download --profile modern
"$SCD_EVAL_PY" scripts/prepare_eval_assets.py sources --profile modern
# 只缺单项则将 modern 换成 hpsv3 或 unifiedreward。
# 新 prefix；先预览，再决定是否实际 apply：
"$SCD_EVAL_PY" scripts/setup_eval_environment.py --profile hpsv3 --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1 --dry-run
"$SCD_EVAL_PY" scripts/setup_eval_environment.py --profile unifiedreward --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-eval-unifiedreward-v1 --dry-run
# 实际安装：同一命令将 --dry-run 改为 --apply；不覆盖已有 prefix。
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python
export SCD_UNIFIEDREWARD_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-unifiedreward-v1/bin/python
"$SCD_HPSV3_PY" scripts/check_eval_environment.py --profile hpsv3 --skip-assets --gpu --execution standalone
"$SCD_UNIFIEDREWARD_PY" scripts/check_eval_environment.py --profile unifiedreward --skip-assets --gpu --execution standalone
```

GenEval 文本、detector、CLIP、MMDetection source 分开准备。资源表中的 `GENEVAL_REPO`、
`GENEVAL_MODEL_DIR`、`GENEVAL_MODEL_CONFIG`、`GENEVAL_CLIP_MODEL`、`GENEVAL_PROMPTS`
必须对应实际下载目录；下列路径来自默认资源表，机器 local/env 覆盖存在时应同步替换。已有 source 用脚本 `--check`
核验而非重新 clone。文本可先准备用于计划，无需先安装 detector 环境。

```bash
bash scripts/download_sources.sh geneval /mnt/aoss/xietianyu/sources/geneval
bash scripts/download_sources.sh mmdetection /mnt/aoss/xietianyu/sources/mmdetection-2.28.2
"$SCD_EVAL_PY" scripts/download_assets.py plan geneval-detector geneval-clip
"$SCD_EVAL_PY" scripts/download_assets.py download geneval-detector geneval-clip
"$SCD_EVAL_PY" scripts/prepare_compositional_prompts.py --benchmark geneval \
  --source-dir /mnt/aoss/xietianyu/sources/geneval --out /mnt/aoss/xietianyu/datasets/benchmarks/geneval
# 将 GENEVAL_PROMPTS 指向上面生成的 suite.jsonl；GENEVAL_MODEL_CONFIG 指向固定 mmdetection config。
"$SCD_EVAL_PY" scripts/setup_eval_environment.py --profile geneval --runtime h100-cu118 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-eval-geneval-v1 --dry-run
# 此 runtime 名保留旧命名；torch2.0/cu118 + MMCV1.7.2 是 H800 候选，尚未实际验证。
# 审阅 recipe 后可显式 --apply；只有实际 H800 smoke 通过才能视为可用。
export SCD_GENEVAL_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-geneval-v1/bin/python
"$SCD_GENEVAL_PY" scripts/check_eval_environment.py --profile geneval --skip-assets --gpu --execution standalone
bash scripts/run_h800_selected.sh score-geneval --check-only
```

`--check-only` 不能证明真实检测模型可运行。先对独立小面板做 H800 smoke，再用同一冻结
正式面板评分；不将 smoke 数字写入论文。DPG 延后，固定 source、mPLUG 与独立环境的
现有说明见 [SLAI 部署说明](SLAI_EVALUATION_DEPLOYMENT.md)。准备环境时不改 base、旧 Flow
或 legacy eval；正式评分的配置/权重/source 改动会产生新身份，不能混入已发布分数。
