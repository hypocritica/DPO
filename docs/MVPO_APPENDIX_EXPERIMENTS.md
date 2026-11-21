# MVPO 附录实验台账

更新于 2026-11-21。正式实验复用已有 checkpoint，共 14 个 SLAI 作业。每行对应一个可单独领取的容器。状态按已收到的记录填写；目前尚未收到 SLAI 作业号或正式结果。

## 启动前

首次部署及 4 条 CPU 准备命令见[部署与 CPU 准备](../command.md#部署与-cpu-准备)。每个新容器先运行以下命令，再复制表中对应的一行。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-mvpo-appendix
export SCD_RESOURCES="$PWD/configs/resources.json"
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate eval
source paths.sh
```

生成作业用 2×H100 或 2/4×5090，具体型号与冻结的生成计划一致。若统一用 5090 重新生图，先按[分片与生成协议](../command.md#正式作业与-gpu-分片)准备独立输出根。梯度与阶段前向可各用 2/4×5090 或 2×H100。输出路径以下均相对 `$MVPO_APPENDIX_ROOT`。

## 截稿前的执行顺序

| 优先级 | 作业 | 复盘依据 |
|---|---|---|
| P0 | G1、D pilot → D1–D4；CPU 配对统计、A 配置核验、C.2 解析推论与 VRPO 文献定位 | 三线可并行。G1 首先报告三方法的四指标；独立 LossAvg K4 与 Global K4 均可核验时，才隔离聚合顺序。VRPO 定位属于稿件工作。 |
| P1 | G2–G3、S1–S2 | 三个生成种子按 prompt 配对；复用两个 96-view banks 在 CPU 估计有限 K4 共享权重。 |
| P2 | F1–F2、G4–G6 | 结合 D 的实测结果决定是否做 8 对样本的 Flow 趋势复验；checkpoint 配对曲线补充已有 B.1。 |

呈现时，B 放生成分差、胜率与区间；E1 展示原始梯度的均值偏移，E2 分解权重–梯度关联，E3 并列阶段抵消与 K4 的监督差。D 的 16 对与 F 的 8 对均限于指定 checkpoint 和预选样本。

## 正式作业

| 编号 | 实验描述 | SLAI 容器内启动命令 | 状态 | SLAI 作业号 |
|---|---|---|---|---|
| G1 | 1,600 步，seed 17，1,632 条 PartiPrompts。比较 Flow-DPO、Global K4、Stage-wise K4 四项评分 | `python3 scripts/appendix_jobs.py direct` | 待回填 | — |
| G2 | 1,600 步，seed 23，固定 512 条。配对比较 Flow-DPO 与 Stage-wise K4 | `python3 scripts/appendix_jobs.py seed23` | 待回填 | — |
| G3 | 与 G2 同一面板，生成 seed 42。三种子分析复用 G1 的 seed 17 子集 | `python3 scripts/appendix_jobs.py seed42` | 待回填 | — |
| G4 | 800 步，seed 17，同一批 512 条。与 G1 的 1,600 步子集组成四个 checkpoint 的差值曲线 | `python3 scripts/appendix_jobs.py step800` | 待回填 | — |
| G5 | 2,400 步，seed 17，同一批 512 条。与 G1 的 1,600 步子集组成四个 checkpoint 的差值曲线 | `python3 scripts/appendix_jobs.py step2400` | 待回填 | — |
| G6 | 3,200 步，seed 17，同一批 512 条。与 G1 的 1,600 步子集组成四个 checkpoint 的差值曲线 | `python3 scripts/appendix_jobs.py step3200` | 待回填 | — |
| D1 | SD1.5 DPO 1,000 步，16 对、t=499。比较五个 K 的原始梯度并检验权重–梯度关联（分片 1/4） | `python3 scripts/appendix_jobs.py sd15-gradient --group 1/4` | 待回填 | — |
| D2 | SD1.5 DPO 1,000 步，16 对、t=499。比较五个 K 的原始梯度并检验权重–梯度关联（分片 2/4） | `python3 scripts/appendix_jobs.py sd15-gradient --group 2/4` | 待回填 | — |
| D3 | SD1.5 DPO 1,000 步，16 对、t=499。比较五个 K 的原始梯度并检验权重–梯度关联（分片 3/4） | `python3 scripts/appendix_jobs.py sd15-gradient --group 3/4` | 待回填 | — |
| D4 | SD1.5 DPO 1,000 步，16 对、t=499。比较五个 K 的原始梯度并检验权重–梯度关联（分片 4/4） | `python3 scripts/appendix_jobs.py sd15-gradient --group 4/4` | 待回填 | — |
| F1 | SD3.5-M Flow-DPO 1,600 步，8 对。分两个噪声阶段复验 K=1/2/4 的梯度机制（分片 1/2） | `python3 scripts/appendix_jobs.py sd35-gradient --group 1/2` | 待回填 | — |
| F2 | SD3.5-M Flow-DPO 1,600 步，8 对。分两个噪声阶段复验 K=1/2/4 的梯度机制（分片 2/2） | `python3 scripts/appendix_jobs.py sd35-gradient --group 2/2` | 待回填 | — |
| S1 | 同一 Flow-DPO，128 对，每对每阶段 96 个前向视图。测跨阶段抵消和负阶段监督（分片 1/2） | `python3 scripts/appendix_jobs.py sd35-stage --group 1/2` | 待回填 | — |
| S2 | 同一 Flow-DPO，128 对，每对每阶段 96 个前向视图。测跨阶段抵消和负阶段监督（分片 2/2） | `python3 scripts/appendix_jobs.py sd35-stage --group 2/2` | 待回填 | — |

## 结果与待补证据

在同一输出根执行[汇总与回传命令](../command.md#slai-汇总回传与本地重绘)。生成线单独汇总时，表会写在 `generation/report/direct/`、`seeds/` 或 `curve/`；下表列出全量汇总路径。

| 研究问题 | 依赖作业 | 复盘时查看 | 当前缺口 / 后续决策 |
|---|---|---|---|
| 三种方法的生成效果如何？ | G1 | `generation/report/direct-comparison.csv` | 核对四指标与实际 adapter。独立 LossAvg K4 权重尚待查证；若缺失，聚合顺序的最终生成效果归因仍待补。 |
| 优势对生成种子是否稳定？ | G1–G3 | `generation/report/paired-differences.csv` | 回填各 seed 与三 seed 平均分差、三个 seed 各自平均分差的样本标准差、胜率、按 prompt 配对的 95% CI。 |
| 优势在哪些训练步出现？ | G1、G4–G6 | `generation/report/checkpoint-differences.png` | 记录 PickScore 与 HPSv3 是否同向以及哪些步数的区间跨零。 |
| 原始平均梯度是否改变？权重与梯度关联贡献多少？ | D1–D4 | `sd15-gradient/report/pair-metrics.csv` | 回填 source 校正的平方偏移、audit 分半稳定性、方差、下降方向、有符号投影和去耦合差异。 |
| 上述现象在 Flow 的两个阶段能否复现？ | F1–F2 | `sd35-gradient/report/pair-metrics.csv` | 分阶段记录方向、估计误差及与 SD1.5 不一致的观察。 |
| 两阶段何时相互抵消？ | S1–S2 | `sd35-stage/report/summary.json`、`stage-weights.csv` | 回填异号比例、全局为正仍有负阶段的比例，以及阶段均值权重差、固定一个负阶段 view 后的有限 K4 平均权重差和区间；与 G1 并列解读。 |

## Pilot 与补充实验

| 实验 | 启动入口 | 当前记录 |
|---|---|---|
| SD1.5 梯度 pilot，4 对样本 | `python3 scripts/appendix_jobs.py sd15-gradient --smoke` | 核对 `repeat-view-check.json`、耗时和峰值显存，输出在 `smoke/sd15-gradient/`。 |
| Flow 梯度 pilot，4 对样本 | `python3 scripts/appendix_jobs.py sd35-gradient --smoke` | 核对 `repeat-view-check.json`、耗时和峰值显存，输出在 `smoke/sd35-gradient/`。 |
| Flow 阶段前向 pilot，4 对样本 | `python3 scripts/appendix_jobs.py sd35-stage --smoke` | 待回填耗时和峰值显存，输出在 `smoke/sd35-stage/`。 |
| SD3.5-M 同噪声定性图，4×5090 | [定性图启动命令](../command.md#长版计划中的低成本补充) | 可选第 15 个 GPU 作业；归档 Base / DPO / Stage-wise K4 三列图。 |
| SDXL、SD1.5 DSPO、Z-Image 历史逐图配对 | [三个独立 CPU 命令](../command.md#长版计划中的低成本补充) | 待核对两种方法共同保存的逐图评分，归档分差、胜率和区间。 |
| 训练开销表与精确似然图 | [两条本地 CPU 命令](../command.md#长版计划中的低成本补充) | 已在本地生成。开销表覆盖 9 个配置，缺失的计时和显存字段保留为空。 |

## 后续维护

领取作业时更新上表的状态与作业号。汇总后在下表追加一行，记录代码提交、输出根或报告链接、读到的数值，以及该数值支持继续补哪个实验。重跑时保留之前的记录。

| 日期 | 实验编号 / SLAI 作业号 | 代码提交 / 结果位置 | 实测结果与不确定性 | 决定补什么、为什么 |
|---|---|---|---|---|
| 待填写 | | | | |
