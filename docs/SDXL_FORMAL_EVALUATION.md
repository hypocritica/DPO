# SDXL：四卡生成、三组资源并行、之后独立评分

此入口使用现有 SDXL renderer 和 legacy eval，不提交任务、不安装环境、不自动选择
checkpoint。正式方法名单和精确 steps 尚需用户提供；从真实训练 metadata 导入 catalog，
审阅后明确选择 run/steps。不会猜测方法数量、训练是否完成或使用 latest 替代缺失快照。
已选 catalog 可显式包含多个 steps，图片数量会相应增加；单 checkpoint 主表请只留一个。

使用三组彼此独立、每组四张已分配 RTX 5090 的资源：分别跑 PartiPrompts、PickV2、HPSv2。
每组可在不同机器上，均挂载相同 AOSS 根目录。三组共享冻结 checkpoint catalog 和采样协议，
面板分别保存 PNG、manifest、日志、图片索引和后续分数。无需等某一面板评分才能生成另一面板。
目前仅 CPU 验证，未实际运行 5090、申请资源或确认 batch=4 显存。

## 先审阅真实 catalog

```bash
cd /absolute/independent-eval-checkout
export SCD_EVAL_PY=python3
export SCD_SD15_PY=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
export SCD_RESOURCES="$PWD/configs/resources.json"
# 必要时先在 resources.local.json 中设置实际 SDXL 模型、面板和旧评分器缓存。
"$SCD_EVAL_PY" evaluate_suite.py catalog --families sdxl \
  --runs-root /absolute/actual-sdxl-runs-root --out /absolute/reviewed-sdxl.json
```

打开 JSON，核对模型 revision/可选 VAE、每个 method 的训练配置、唯一 run 和永久 steps。
删除不参与的方法/旧重复候选，把 steps 改为本次精确选择。不改训练产物。模型 Base 独立于
训练目录；同一模型身份只出现一次。未知方法/重叠 step/缺失快照会报错，不自动回退。
不要把示例 catalog 中的占位路径当成现成实验。

```bash
export SCD_SDXL_CATALOG=/absolute/reviewed-sdxl.json
export SCD_SDXL_ROOT=/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/formal-5090-v1
# 可选：仅当需要改实际面板路径时，提供自己的 suite 配置。
# export SCD_SDXL_CONFIG=/absolute/evaluation-suite-sdxl.local.json
export SCD_SDXL_BATCH=4
"$SCD_EVAL_PY" evaluate_suite.py list --catalog "$SCD_SDXL_CATALOG" --families sdxl --json
for panel in partiprompts pickv2 hpsv2; do
  bash scripts/run_sdxl_formal.sh plan --panel "$panel" --dry-run
done
```

保留既有 SDXL 协议：1024×1024、EulerDiscreteScheduler、BF16、50 steps、CFG5、seed17。
默认每 GPU forward batch4、整套模型常驻、cpu_offload=false；这是保守但未经真实 5090
验证的起点，不能承诺显存足够。渲染不自动降低 batch。要改 batch/seed/模型/VAE/协议，
在首次正式 plan 前改配置并用新输出根目录。每 checkpoint：Parti 1,632、PickV2 487、
HPSv2 3,200 图，合计 5,319 图。实际总量以 dry-run 的明确方法/steps 和计数为准。

审阅数量后正式冻结；三组也可分别执行各自 plan。共享锁只覆盖小型身份写入，生成和评分
不占用它。`shared/selection.json` 记录原 catalog 字节 SHA、解析后的绝对模型/run 路径、
checkpoint 身份和完整公共采样配置；三个面板重复调用必须一致。原 catalog 的注释/格式
变化也视为改变，使用新版本目录。相对路径在复制前解析，不会因为移动冻结副本改变含义。

```bash
for panel in partiprompts pickv2 hpsv2; do
  bash scripts/run_sdxl_formal.sh plan --panel "$panel"
done
```

## 三组四卡分别生成

先通过资源平台取得三组各四张 5090；下列命令不申请资源、不在登录节点运行 GPU。
每组加载同一个代码版本、上述共享路径和现有 legacy Python，并保留平台实际分配的
`CUDA_VISIBLE_DEVICES` 四个 token。也可显式 `SCD_EVAL_GPUS` 指定实际设备/UUID，
启动器拒绝缺失、重复或不是四个 token 的配置，绝不默认假定 0,1,2,3。

```bash
# 资源组 A：四卡 Parti
bash scripts/run_sdxl_formal.sh generate --panel partiprompts
# 资源组 B：另外四卡 PickV2（可与 A 同时进行）
bash scripts/run_sdxl_formal.sh generate --panel pickv2
# 资源组 C：另外四卡 HPSv2（可与 A/B 同时进行）
bash scripts/run_sdxl_formal.sh generate --panel hpsv2
```

每个组只生成自身面板并自动写 `image-index.csv/json`，不评分。用同一命令恢复缺图；
不删除已完成 PNG/manifest，不修改已有批边界。运行日志带时间戳分别位于
`$SCD_SDXL_ROOT/<panel>/logs`，模型/方法/step/prompt/seed 可在评分前用 index 定位。
可选的首次环境检查用原工具单独执行，启动器没有每阶段全量资产/checkpoint 预检；
冻结身份、checkpoint 和事务校验始终保留。

```bash
# 首次部署或排错，必须在已分配 GPU 内：
"$SCD_SD15_PY" scripts/check_eval_environment.py --profile legacy --gpu --execution standalone
# 只校验某计划不生图，可选：
bash scripts/run_sdxl_formal.sh generate --panel partiprompts --check-only
```

## 之后只读 PNG 独立评分

图已完整发布后，在后续四卡资源组运行评分；不需要保留生成模型、adapter、训练目录或
原 catalog。评分类别只用现有 legacy 五指标：PickScore、HPSv2、ImageReward、CLIP、
Aesthetic。HPSv2 面板名称不意味着只计算 HPSv2 指标；五指标复用同一批 PNG，不增面板。

```bash
bash scripts/run_sdxl_formal.sh score --panel partiprompts
bash scripts/run_sdxl_formal.sh score --panel pickv2
bash scripts/run_sdxl_formal.sh score --panel hpsv2
# 也可先只评一项，后续同一面板补齐：
bash scripts/run_sdxl_formal.sh score --panel partiprompts --metrics pickscore
for panel in partiprompts pickv2 hpsv2; do
  bash scripts/run_sdxl_formal.sh summarize --panel "$panel" --summary-name formal-current
  bash scripts/run_sdxl_formal.sh gallery --panel "$panel" --gallery-name formal-v1 --limit 80 --copy-images
  bash scripts/run_sdxl_formal.sh paper --panel "$panel" --paper-name formal-v1
done
```

汇总/画廊/论文导出不需要 GPU。缺分数时用独立新名字加 `--allow-incomplete` 导出带缺失
标记的草稿，缺失不补0；补齐后新建正式 paper 版本。不同面板保留自己的协议/身份记录，
论文按面板比较，gallery 筛图不改变统计分母。不要将三组资源的 GPU token 混为一次跨机
12卡进程；每个入口只使用本组四卡，并只写自己面板目录。
