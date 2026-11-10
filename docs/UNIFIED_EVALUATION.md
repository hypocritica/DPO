# 分阶段 checkpoint 评估

统一入口为 `evaluate_suite.py`，支持 **SD1.5、SDXL、SD3.5 Medium、Z-Image Base**。
运行命令全部放在根目录 [command.md](../command.md) 的第一节。
本目录是独立评估 checkout；不修改正在训练的 Flow checkout、训练配置或 checkpoint。

## 从很多 checkpoint 中选择

`list` 递归发现指定 runs 目录中的 `training.json` 和已发布的
`checkpoints/step-N/checkpoint.json`。可用 `--run-glob` 筛选 run 相对路径或目录名，
用 `--families` 筛选模型。列表读取小型元数据，不加载 PyTorch、生成模型或 reward，
也不扫描 optimizer tensor 的内容；`files_verified=false` 明确标注这一点。

`plan` 支持以下互斥选择方式：

- 指定 `--steps`：必须全部存在，缺步数立即报错，绝不以 latest 替代。
- 区间 `--min-step` / `--max-step`，可叠加 `--every`（整除步数）和 `--latest N`。
- 没有 step 过滤时枚举全部永久快照；务必先看 `--dry-run` 的图片总量。
- `--include-base` 为每个选中的 run 加 step 0 的未微调模型；相同基座若在多个 run
  下重复出现，会重复生成。需要节省这部分计算时单独选择一个 run 生成基座。

只读目录中未提交的 pending、rolling 槽和 `latest.json` 均不参与选择。
正式生成前再次核验 training/manifest 身份以及 checkpoint 文件 SHA256。
记录的基座路径来自该 run 的训练配置，不悄悄替换成另一个资源目录的同名模型。

## 计划、生成、评分分离

`plan` 不检查 reward 是否已下载，也不要求生成模型可加载。它保存只读的选择及采样
计划 `plan.json`，并将 prompt 原样复制到 `prompts/`，记录 SHA256。
`--dry-run` 只打印，不创建计划或输出目录。实际生成前必须执行一次不带该选项的 `plan`。

`generate` 只加载基座与所选 adapter，保留每一张 PNG、prompt/seed 和 hash。
`score` 只读取已完成图片，不访问训练目录、LoRA 或生成模型。HPSv3 等资产可在生成后
才准备；之后追加指标不触发生图。每个 scorer 可用独立 Python 环境，执行后释放进程。

默认评分：旧模型的三套偏好面板使用原五指标；新模型使用 HPSv3、UnifiedReward、
PickScore、ImageReward、CLIP、Aesthetic。GenEval 和 DPG 面板默认仅运行其专用
evaluator。`--metrics` 可指定一项或多项；专用 evaluator 拒绝错误的 prompt 面板。
`--metric-group legacy` 可以让新模型的三套偏好面板也生成原表五指标。

`generate`、`score` 均支持按 family/run/step/benchmark 筛选已冻结计划。
`--gpus` 仅使用 SLAI 已分配且在 `CUDA_VISIBLE_DEVICES` 中的设备，一卡一个独立
worker，按生成任务或评分任务分片；不发起平台资源申请。也可手工使用
`--shard-index` / `--shard-count`。多个 scorer 不会在单 worker 内同时驻留显存。
`--python` 选择本次执行阶段的解释器；没有该参数时使用 scorer config 的 `python`
或当前解释器。不要用一个旧环境运行所有新模型。

`--check-only` 做对应阶段的资产/图片/依赖元数据检查，不执行 GPU forward。
生成预检会保存小型 render-config，便于审查，但不写图片；评分预检不会产出分数。
这不是完整 CUDA、MMCV ABI 或数值正确性的验证。

## 每个 prompt 生成多少张

| 面板 / 模型 | 默认 seed | 每个 prompt 图片数 |
|---|---|---:|
| 原表三套面板 / SD1.5 | 0 | 1 |
| 原表三套面板 / SDXL | 17 | 1 |
| 原表三套面板 / SD3.5 Medium、Z-Image Base | 17, 29, 43, 71 | 4 |
| GenEval / 全部模型 | 17, 29, 43, 71 | 4 |
| DPG-Bench / 全部模型 | 17, 29, 43, 71 | 4 |

原表协议只保留项目已有配置，并不声称复现截图中第三方方法的数值：SD1.5 取
`evaluate-selected-sd15.json` 的 Euler50/CFG7.5/512/BF16/seed0；SDXL 取
`evaluate-sdxl.json` 的 Euler50/CFG5/1024/BF16/seed17。旧入口、旧配置不变。
新模型使用对应 Flow 配置的采样参数：SD3.5 Medium 28步/CFG4.5，Z-Image Base
50步/CFG4，均1024；这不是 Z-Image Turbo 协议。

同一 backbone 的基座和各方法严格使用相同 prompt、seed、尺寸、steps 和 CFG。
不同 backbone 的原生设置分别记录，不把相同 seed 解释成相同 latent。
若要旧模型最终也做4或5个seed，使用 `plan --seeds ...` 并换新版本输出；不要覆盖
旧单seed表，或把不同seed数量的结果混在一起。GenEval/DPG固定要求4个不同seed。

三套面板总计487+3200+1632=5319条；每个 checkpoint 单seed产生5319张，四seed
产生21276张。GenEval为553×4=2212张，DPG为1065×4=4260张。
先在独立验证集筛选 checkpoint，再对少量已选 checkpoint 做完整正式测试；不要用
正式测试成绩反复选择最优 checkpoint。完整PickV2仍可能含既有monitor prompt，
因此不能改称“完全独立测试集”；需要独立held-out结论时另建明确标记的过滤面板。

## 输出、恢复和展示图

```text
evaluation-tag/
  plan.json
  prompts/<benchmark>.jsonl
  <run>/step-0005000/<benchmark>/
    render-config.json
    render/run.json
    render/manifest.jsonl
    render/images/*.png
    render/complete.json
    scores/<metric>/...          # 独立身份、逐图/逐题结果、汇总、complete
    failures/                   # 失败原因，成功重跑后移除对应记录
  summary.csv / summary.json
  galleries/<name>/index.html
  galleries/<name>/selection.json
```

重复完全相同的命令可恢复，已完成图片/评分会验证后复用。输入、权重、评分模型、
代码身份发生变化时拒绝混入已有结果；新增checkpoint或改变seed需建立新计划。
评分错误、非有限值、UnifiedReward无法解析等不会填0或漏样本。默认遇错停止；
`--keep-going` 允许继续独立任务，最终仍非零退出并记录失败。

`summarize` 验证完成标记、来源及结果hash，写CSV和JSON；缺失分数留空并标记
pending，真实失败有error，不会当作0。只要选定表格未完整，汇总命令退出码为1。
不同子集/指标表使用 `--summary-name`，避免误读前一次筛选留下的表。
通用 reward 结果保留每图均值、prompt均值和prompt标准误；目前未自动实现跨方法
配对bootstrap置信区间。GenEval采用任务类别宏平均，DPG采用官方依赖评分，详见
[专项评测说明](COMPOSITIONAL_EVALUATION.md)。

`gallery` 生成本地HTML预览；同一prompt/seed的多个run/checkpoint排在一起。
可用 `--scene-id`、`--seed` 选择展示候选，并在 `selection.json` 保存来源。
HTML按相对路径引用原图；在SLAI查看或传输HTML时需保持相应图片目录可访问。
默认最多200图，预览数量不影响评分数量。变更展示选择使用新gallery name。

可以挑好看的图片用于展示，但所有预定样本仍必须进入定量结果；不将best-of-4
与随机单图结果比较。并排比较方法时优先固定相同prompt和seed，展示精选样例时
说明选择规则。不要删掉失败样例来提高统计分数。

## 资产与部署边界

资源仍集中在 `configs/resources.json`，SLAI私有覆盖为 `resources.local.json`。
模型、source checkout、benchmark prompts、解释器分开配置。
HPSv3需要官方source/YAML、HPS checkpoint和完整Qwen2-VL base；UnifiedReward
固定使用官方2.0 Qwen2.5-VL-7B，报告ACS三维及本项目显式定义的均值。
下载及协议细节见 [资产说明](EVALUATION_ASSETS.md)。

本次仅做本机轻量验证，没有下载模型或提交GPU评测。SLAI首次使用每个独立环境
时应先运行小面板smoke，再决定是否启动正式表格；GenEval旧MMCV栈尤其需要在
实际GPU验证。旧评估产物继续用其原checkout恢复，不扩大AOSS源码兼容白名单。
