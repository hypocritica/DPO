# SLAI 首次部署：SD1.5 正式表与 SD3.5 checkpoint 曲线

这份流程适用于**尚未克隆评估代码、已有旧 `eval` Conda环境**的SLAI机器。
分两条轨道执行，每条都是**先生成全部计划图片，再独立评分**。

| 轨道 | 生成设置 | 默认评分 | AOSS目录 |
|---|---|---|---|
| SD1.5 正式比较 | 4张5090，每GPU batch32 | 原五指标 | `evaluations/sd15/main-v1` |
| SD3.5 Medium 曲线 | 4张5090，每GPU batch4，固定128条、seed17 | 仅PickScore | `evaluations/sd35_medium/curves-v1` |

表中目录均在 `/mnt/aoss/xietianyu/self_correct/` 下。源码和独立Python环境放AFS；
plan、PNG、分数、曲线、gallery和日志放AOSS。生成batch是**每个GPU一次前向的图片数**，
与评分batch、训练batch无关；OOM会报错，不会静默降batch。
GPU命令只在SLAI已经分配的容器内执行；不需要Slurm。

## 1. 首次克隆并核对旧环境

目标目录应尚不存在；不要在训练checkout里拉取评估分支。

```bash
git clone --branch feat/unified-evaluation-slai-20261111 --single-branch \
  https://github.com/hypocritica/DPO.git \
  /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
git rev-parse HEAD

export HF_HOME=/mnt/aoss/xietianyu/cache/huggingface
export PIP_CACHE_DIR=/mnt/aoss/xietianyu/cache/pip
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

conda activate eval
export SCD_SD15_PY="$(command -v python)"
"$SCD_SD15_PY" --version
"$SCD_SD15_PY" -m pip check
"$SCD_SD15_PY" scripts/check_eval_environment.py --profile legacy --skip-assets
```

已有 `requirements-eval.txt` 建好的环境在检查通过后可继续给SD1.5生图及原五指标评分。
旧环境并不自动保证5090/CUDA兼容；缺失或检查不通过时，可用下列新前缀配方替代，
保留原eval。运行setup的Python只需标准库，不向其自身环境安装任何依赖。

```bash
python scripts/setup_eval_environment.py --profile legacy --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-legacy-eval-v1 --dry-run
python scripts/setup_eval_environment.py --profile legacy --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-legacy-eval-v1 --apply
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/envs/scd-legacy-eval-v1/bin/python
```

新模型环境另建，不覆盖eval，也不向base或训练环境安装。GPU算子仍需在实际分配设备上验证。

新clone没有私有资源文件时，建立下面这份最小覆盖；命令拒绝覆盖已存在文件。
模型仍沿用 `/mnt/aoss/xietianyu/models` 的既有分类目录。

```bash
(
  set -o noclobber
  cat > configs/resources.local.json <<'JSON'
{
  "WORKSPACE": "/mnt/aoss/xietianyu",
  "MODELS_ROOT": "/mnt/aoss/xietianyu/models",
  "DATASETS_ROOT": "/mnt/aoss/xietianyu/datasets",
  "EVALUATIONS_ROOT": "/mnt/aoss/xietianyu/self_correct/evaluations"
}
JSON
)
df -h /mnt/aoss/xietianyu /mnt/afs/L202500030/xietianyu
```

默认旧评分资源依次为 `models/preference/pickscore`、`models/clip/clip-h14`、
`models/preference/hpsv2`、`models/clip/clip-l14`、`models/aesthetic/...`、
`models/preference/image-reward` 和 `models/backbones/bert-base-uncased`。
实际缓存位置不同时，在私有文件中覆盖对应资源键；完整键名见
[configs/resources.json](../configs/resources.json)。也可用 `SCD_RESOURCES` 选择另一份资源表。
旧shell导出的同名环境变量优先级更高，迁移后须核对。

## 2. 先检查复用已有旧权重，缺少时才显式下载

```bash
"$SCD_SD15_PY" scripts/prepare_eval_assets.py check --profile legacy
"$SCD_SD15_PY" scripts/prepare_eval_assets.py check --profile pickscore
```

这两个检查不联网、不加载GPU模型。完整的手动下载目录可以复用，**不要求存在本项目
download receipt**。结构检查和实际评分预检通过后即可使用，不要因为缺receipt重下权重。

只有上面检查确有缺失时，执行对应profile的独立准备阶段。`plan`仅查元数据和空间，
`download`才下载；完整legacy/PickScore本地payload会跳过。已有不完整目录无法安全
补齐时会拒绝覆盖，应选新目录并更新资源键，保留原文件。

```bash
# 只缺SD3.5曲线的PickScore时，限定这个profile：
"$SCD_SD15_PY" scripts/prepare_eval_assets.py plan --profile pickscore
"$SCD_SD15_PY" scripts/prepare_eval_assets.py download --profile pickscore
"$SCD_SD15_PY" scripts/prepare_eval_assets.py check --profile pickscore

# 若SD1.5旧五指标有其他缺失，使用legacy；已有部分仍复用：
"$SCD_SD15_PY" scripts/prepare_eval_assets.py plan --profile legacy
"$SCD_SD15_PY" scripts/prepare_eval_assets.py download --profile legacy
"$SCD_SD15_PY" scripts/prepare_eval_assets.py sources --profile legacy
"$SCD_SD15_PY" scripts/prepare_eval_assets.py check --profile legacy
```

`sources`显式准备HPSv2、ImageReward和aesthetic source；已有目录保留，缺失目录才克隆。
aesthetic小型head位于其官方source仓库中。检查报缺某个source/payload时，应先核对
配置路径，不把另一个目录重命名覆盖上去。脚本不隐式安装依赖。

## 3. 准备三套固定官方prompt，再建立Flow环境

首次clone尚无benchmark文本时，先显式下载约0.9MB的固定官方prompt，准备
PartiPrompts、PickV2、HPSv2三套完整面板。已有同版文本会核验后复用。

```bash
"$SCD_SD15_PY" scripts/prepare_eval_prompts.py --download-sdpo
```

输出默认位于 `${BENCHMARKS_ROOT}/sd15/paper-eval`，由资源表解析；三套文本分别为
1632、487、3200条。这一步是SD1.5 plan的输入，也应在SD3.5 prepare之前执行，
使曲线prompt能排除它们。GenEval/DPG文本若已有，通过资源键复用；缺失会明确标记
排除覆盖不全，不会为了本轮工作自动下载专项模型。

机器只有旧eval时，需要另建这个环境。先看配方，再去掉dry-run并显式apply。
新前缀存在时setup拒绝覆盖；已有匹配Flow环境可以直接设置其绝对Python路径。

```bash
"$SCD_SD15_PY" scripts/setup_eval_environment.py --profile flow --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1 --dry-run
"$SCD_SD15_PY" scripts/setup_eval_environment.py --profile flow --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1 --apply
export SCD_FLOW_PY=/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python
"$SCD_FLOW_PY" -m pip install -r requirements-eval-curves.txt
"$SCD_FLOW_PY" -m pip check
"$SCD_FLOW_PY" scripts/check_eval_environment.py --profile flow --skip-assets
```

`requirements-eval-curves.txt`仅增加画图所需Matplotlib。已有SD3.5训练基座从training.json
读取，不再下载一份。**完成这一步及PickScore准备，即可执行SD3.5曲线**；不必等待
HPSv3、UnifiedReward、GenEval或DPG。

## 4. SD1.5：四卡batch32先生成，再评原五指标

默认输出和日志全部在AOSS。下面的日志函数保留pipeline失败状态，使用新日志文件名。

```bash
export SCD_SD15_OUT=/mnt/aoss/xietianyu/self_correct/evaluations/sd15/main-v1
run_sd15() {
  local phase="$1"
  shift
  local log
  mkdir -p "$SCD_SD15_OUT/logs"
  log="$(mktemp "$SCD_SD15_OUT/logs/$(date -u +%Y%m%dT%H%M%S)-${phase}-XXXXXX.log")"
  ( set -o pipefail
    bash scripts/run_sd15_main.sh "$phase" --out "$SCD_SD15_OUT" "$@" 2>&1 | tee "$log"
  )
}
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-UNSET}"

# 假设平台真实分配的四个token为0,1,2,3；不同则改成实际列表。
run_sd15 plan
run_sd15 generate --gpus 0,1,2,3 --check-only
run_sd15 generate --gpus 0,1,2,3 --keep-going
"$SCD_SD15_PY" evaluate_suite.py index --out "$SCD_SD15_OUT"

# 生成阶段结束后，才对已保存图片评分；无需重新生成。
run_sd15 score --gpus 0,1,2,3 --check-only
run_sd15 score --gpus 0,1,2,3 --keep-going
run_sd15 summarize
run_sd15 export
run_sd15 gallery --copy-images --limit 200
```

SD1.5选择、冻结配置与checkpoint来源由专用入口核验；每GPU的生图batch固定32。
本轮11个固定版本×3面板共33个render，`score`要求这33个render全部完成才允许启动。
即使给评分传`--benchmarks`只选一个面板，也仍检查全计划已生成；生成失败应先恢复
generate，不能把局部评分误当整轮完成。这是默认先全生成、再评分的执行边界。
每个GPU各自加载一个完整模型，任务按checkpoint分片。`--gpus`填写
`CUDA_VISIBLE_DEVICES`列表里的实际token，例如环境为`4,5,6,7`就传该列表，不填`0,1,2,3`。
请确认图片总量及磁盘余量后启动正式生成。原五指标为PickScore、HPSv2、ImageReward、
CLIP和Aesthetic，评分batch由各评分配置决定。

## 5. SD3.5：固定128、每GPU batch4，分阶段跑曲线

```bash
export SCD_CURVES_RUNS_ROOT=/mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs
export SCD_CURVES_PREP=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-v1-prepared
export SCD_CURVES_OUT=/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-v1

bash scripts/run_sd35_curves.sh prepare --generation-batch-size 4
bash scripts/run_sd35_curves.sh plan
# 平台真实分配的四个token为0,1,2,3时：
bash scripts/run_sd35_curves.sh generate --gpus 0,1,2,3 --check-only
bash scripts/run_sd35_curves.sh generate --gpus 0,1,2,3
"$SCD_FLOW_PY" evaluate_suite.py index --out "$SCD_CURVES_OUT"

# 生图结束后再评分，仍然只需要PickScore：
bash scripts/run_sd35_curves.sh score --gpus 0,1,2,3 --check-only
bash scripts/run_sd35_curves.sh score --gpus 0,1,2,3
bash scripts/run_sd35_curves.sh summarize
```

准备阶段冻结Base及七个训练方法的全部已发布永久checkpoint。128条开发集prompt、
seed17和batch4写入准备身份。图片和scores在该轨道下，曲线位于
`curves/pickscore.png`、`curves/pickscore.pdf`，完整点和缺失状态在`curves/points.csv/json`。
日志在`$SCD_CURVES_OUT/logs/`。prepare列出的缺失正式排除面板必须保留说明，不能因此
声称与五套正式benchmark完全独立。细节见[曲线说明](SD35_CHECKPOINT_CURVES.md)。

若batch4在实际显存边界下OOM，需要明确建立新版本prepared/output并改batch，不能
直接在冻结计划上静默改变batch。部分恢复会重算固定批，只有缺失图片会提交。

## 6. 后置准备HPSv3和UnifiedReward独立环境及权重

这些评分器供后续SD3.5正式五面板使用，通过通用`evaluate_suite.py score --metrics ...`
接口消费其正式计划图片；本轮SD1.5仍固定旧五指标，SD3.5曲线仍固定只评PickScore。
不要往curves-v1计划追加modern指标。这部分不是前两条轨道生图或PickScore的前置要求。
每个profile保持独立环境，不升级旧eval或Flow。先打印配方，确认后执行apply。

```bash
"$SCD_SD15_PY" scripts/setup_eval_environment.py --profile hpsv3 --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-hpsv3-v1 --dry-run
"$SCD_SD15_PY" scripts/setup_eval_environment.py --profile hpsv3 --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-hpsv3-v1 --apply
export SCD_HPSV3_PY=/mnt/afs/L202500030/xietianyu/envs/scd-hpsv3-v1/bin/python

"$SCD_SD15_PY" scripts/setup_eval_environment.py --profile unifiedreward --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-unifiedreward-v1 --dry-run
"$SCD_SD15_PY" scripts/setup_eval_environment.py --profile unifiedreward --runtime slai-cu130 \
  --prefix /mnt/afs/L202500030/xietianyu/envs/scd-unifiedreward-v1 --apply
export SCD_UNIFIEDREWARD_PY=/mnt/afs/L202500030/xietianyu/envs/scd-unifiedreward-v1/bin/python

df -h /mnt/aoss/xietianyu/models
"$SCD_SD15_PY" scripts/prepare_eval_assets.py plan --profile modern
# 确认目标与剩余空间后，先下载weights，再准备source：
"$SCD_SD15_PY" scripts/prepare_eval_assets.py download --profile modern
"$SCD_SD15_PY" scripts/prepare_eval_assets.py sources --profile modern
"$SCD_HPSV3_PY" scripts/prepare_eval_assets.py check --profile hpsv3
"$SCD_UNIFIEDREWARD_PY" scripts/prepare_eval_assets.py check --profile unifiedreward
"$SCD_HPSV3_PY" scripts/check_eval_environment.py --profile hpsv3
"$SCD_UNIFIEDREWARD_PY" scripts/check_eval_environment.py --profile unifiedreward
```

`modern`只选HPSv3、完整Qwen2-VL base和UnifiedReward 2.0；三个权重包合计约50GB，
以plan实时大小为准，不包括环境、输出图片或缓存余量。默认模型目录仍在
`models/preference/hpsv3`、`models/backbones/Qwen2-VL-7B-Instruct` 和
`models/preference/UnifiedReward-2.0-qwen-7b`。
如只需要其中一种，可以将profile改为`hpsv3`或`unifiedreward`。

HPSv3取官方mu；UnifiedReward固定2.0 ACS并保存三个分项，均值是项目明确命名的
ACS mean。固定来源、完整shards和source是协议的一部分；modern手动缓存若无法核实
身份会明确失败，脚本不会覆盖它或伪造来源记录。通过配置选择正确的完整固定版本。
这些旧DeepSpeed/视觉语言依赖与5090组合仍需在已分配GPU上做小规模实际评分核验。

GenEval/DPG后置到专项阶段。这里只准备和使用文本面板时不必下载它们的detector或VQA
模型；正式专项评分需要独立旧依赖环境和实际GPU验证，不能把H100候选配方直接套到5090。
下载与协议细节见[资产说明](EVALUATION_ASSETS.md)。

## 恢复与结果边界

每条轨道生图后执行的`evaluate_suite.py index`会在对应AOSS输出根目录写
`image-index.csv`和`image-index.json`。CSV包含family、method、step、benchmark、
scene_id、seed、完整prompt、原PNG绝对/相对路径和hash，可按prompt查找原图。
索引只消费完整render，缺失任务标明状态并返回非零；它不改变评分或选样分母。

```bash
# 示例：查SD1.5的DPO在PartiPrompts面板中的前5张图，打印prompt和原图路径。
"$SCD_SD15_PY" - <<'PY'
import csv, os
from itertools import islice
from pathlib import Path
with (Path(os.environ['SCD_SD15_OUT']) / 'image-index.csv').open() as stream:
    selected = (r for r in csv.DictReader(stream)
                if r['method'] == 'dpo' and r['benchmark'] == 'partiprompts')
    for row in islice(selected, 5):
        print(row['step'], row['seed'], row['prompt'], row['image_path'], sep='\t')
PY
```

重复相同generate或score命令可恢复；完整PNG与评分产物核验后复用。重新分配卡以后
SD3.5可以在2/4卡间切换，SD1.5专用入口保持4卡；实际GPU token可以变化，保留原out。
不要同时对同一输出启动重叠的任务。两轨若共用同一组四卡，请依次运行。
变更prompt、checkpoint选择、采样/batch或评分身份要使用新版本目录。
输入变化不会通过删除receipt强行绕过。

SD1.5正式结果、SD3.5开发曲线分开存储，缺失分数留空并返回非零，不填0。gallery复制
原图后可独立搬走；没有`--copy-images`时HTML仍依赖对应原图目录。保存PNG、manifest、
逐图scores、plan、prepared及日志，以便追溯。这里提供的是部署和执行命令；本次没有
在本机安装新环境、下载模型或执行GPU任务。
