# MVPO 梯度实验：逐项监督

本轮入口是 `scripts/gradient_experiment.py`，实验名 `sd15` / `flow`。每次只执行一个动作，没有 `all`。运行命令统一见 [command.md 首节](../command.md#本轮梯度实验逐项监督2026-11-22)。先阅读已完成旧 D 的报告，再决定是否启动新 pilot 或扩大样本；本轮未读取到 SLAI 实际产物，也未运行 GPU。

## 研究问题与模块

| 用户顺序 | 附录 | 数据与计算 | 主要读数 |
|---|---|---|---|
| ④ 原始期望梯度 | E.1 | SD1.5 DPO step1000，训练 split 中固定 pair，固定后验 latent、canonical text、t=499，仅对高斯噪声取样 | 原始均值梯度的距离、source 校正平方距离、方差、下降余弦、有符号投影 |
| ⑤ 权重–梯度去耦合 | E.2 | 复用④的 source/audit 和 Gram；CPU 重算，不增加 GPU 采集 | coupled 与 decoupled 差、协方差的平行/垂直分量、下降余弦变化 |
| ⑥ flow 复验 | E.1 的跨模型复验，并复用 E.2 诊断 | SD3.5-M Flow-DPO step1600，分别在 scheduler index 0–499 和500–999均匀取样，K=1/2/4 | 两阶段分别报告以上读数，不把阶段混为一条总体结论 |

`scripts/mvpo_appendix_probe.py` / `evaluation/appendix_probe.py` 负责固定 checkpoint/数据身份、训练模式逐 view 反向与两项检查；`appendix_backend.py` 复用归档训练误差参数化。`appendix_gradients.py` 仅保留 block 均值的 Gram、margin 和梯度平方范数和；临时完整 block 均值在 scratch 上用后删除，不持久保存每 view 全梯度。`appendix_gradient_report.py` / `appendix_gradient_reporting.py` 用这些统计独立重画报告。本轮未改采集、后端、确定性设置或旧协议。

SD1.5 保留严格确定性算法、cuDNN 设置、cuBLAS workspace 和 math SDPA；不得用关闭检查来绕过此前同 view 梯度复算失败。Flow 的模型常驻方式保持既有实现，5090 显存和耗时由其独立 pilot 实测，尚未验证。

## 阶段与冻结

`--output ROOT` 必须显式给出。每个实验的数据分别落在 `ROOT/sd15/pilot`、`ROOT/sd15/formal` 和 `ROOT/flow/pilot`、`ROOT/flow/formal`。默认新根来自资源表的 `MVPO_GRADIENT_ROOT`。旧 `d-repeat-v2` 只通过原报告入口读取，不允许由新入口接管或更改其 protocol/prepared。

| 动作 | 做什么 | 依赖与限制 |
|---|---|---|
| `plan` | 输出目的、参数、K、pair-view 反向数及实际 prepare/worker 命令 | 纯标准库；不读模型、不导入 torch、不创建文件。未给 GPU 时只展示一个占位 worker |
| `prepare` | 核验 checkpoint/训练 manifest 并冻结选中 pair 与新入口参数 | CPU；需原训练数据依赖但不加载生成模型；`--phase pilot` 与 `formal` 分开准备 |
| `pilot` | 只运行独立 pilot 输出 | 先 prepare pilot；默认4对、2×24 source、2×8 audit，仅作工程检查 |
| `run` | 运行一个指定正式 group | 先 prepare formal；对应 pilot 全部记录、初始化及同 view 复算检查必须有效且通过；核验 checkpoint、数据、解释器、源码、资源路径与解析到的checkpoint/数据身份对应 |
| `status` | 列出完成、缺失、损坏/身份失败、两项检查、协议以及 worker 历史状态 | 只读；完整数值/Gram检查仍由 collect 执行。`running` 是启动记录，需结合日志判断进程是否仍存活 |
| `collect` | 深验全部 pair-condition、view bank、Gram、协议身份和两项检查，保存文件 hash 清单 | 不运行GPU、不复制全梯度；正式结果同样要求有效 pilot |
| `report` | 从 collect 清单重建 CSV/JSON/PNG/PDF | 不运行GPU；记录或检查在 collect 后变化则拒绝，须重新 collect |

新协议默认：正式128对；每对每条件8×96 source和8×128 audit；SD1.5 K=1/2/4/6/8，Flow K=1/2/4。两个模型默认正式采集分别是 **229,376** 和 **458,752 pair-view 反向**，另有 shard0 的初始化2次和每条件同 view复算2次；不是同成本实验。K 共用相同 view bank，不为每个 K 再跑模型。

`--pair-count`、`--source-blocks`、`--source-views-per-block`、`--audit-blocks`、`--audit-views-per-block`、`--pair-seed`、`--view-seed` 可公开覆盖；prepare 和 pilot/run 必须给同样参数。source views 必须能被所有 K 整除，audit blocks 为至少2的偶数。更改已冻结参数、源码或解释器须用新输出根，不接管已有结果。`--scratch-dir` 可指向容器已有的快速本地盘；其位置不进入实验统计身份。

固定 `--group 1/N` 至 `N/N` 按 `pair_index % N` 分组，每组再分给当前分配的 GPU，各卡一次处理一个 pair 的所有条件。各组可有不同卡数；groups 的 N 必须一致。换 group 总数前确认没有旧 worker 仍在运行，避免并发重复处理同一 pair。GPU token来自已分配的 `CUDA_VISIBLE_DEVICES`，不会申请GPU。新入口从自身文件定位checkout，不沿用硬编码旧目录的SLAI shell。

## 报告读法

先核对 `protocol.json`、`status` 和两项 passed 检查，再读报告。`aggregate.csv` / `pair-metrics.csv` 保持原始距离、source 校正平方距离、方差、alignment/coupling 和两种区间。view Monte Carlo区间在固定pair内独立重采source/audit blocks；pair bootstrap区间重采固定面板的pair且保留同pair方法配对。图中聚合区间默认使用pair bootstrap，两类区间没有合并成一个区间。

`*-gradients.png/pdf` 和 `*-coupling.png/pdf` 是汇总图。先平均带幅度的原始梯度，再求范数、距离和方向；不先把每个 view 梯度单位化。每个 pair 得出自己的余弦后再对 pair 做平均，含义是典型 pair 几何，不是跨 pair 总梯度的余弦。audit 是有限独立参照，不是真实期望的精确梯度。source correction只减去 source 均值的估计方差，audit误差仍在；负平方距离估计保留，不截成0。

`paired-changes.csv` 与 `*-paired-changes.png/pdf` 显示**同pair、同K，MVPO − LossAvg**。距离/方差变化更负表示更小，下降余弦或有符号投影变化更正表示更大。散点是各pair，橙线是中位数；这些图用于看配对变化的异质性，不代表全数据反转率。CSV保留每pair view MC区间；不要把散点正负直接升级成总体显著性结论。去耦合的下降余弦变化也列于CSV。

`audit-reliability.csv` 与 `*-audit-reliability.png/pdf` 给出独立audit半bank方向余弦、半bank margin/梯度范数和同pair方法差的两半结果。若方向余弦低，或两半参照下配对距离变化明显异号，需先评估audit误差，再在新输出协议中增加audit views；这是一项敏感度诊断，没有人为设定的自动科学结论阈值。pilot半bank很小，不能据其决定统计结论。

两个轻量一致性参照：K=1时MVPO和LossAvg完全重合；同一source bank下，LossAvg的经验平均梯度对所有K相同（分组方差可以变化）。去耦合使用全source中不同view的有序交叉项 `mean_{i != j}(a_i h_j)`，其中 `a=-sigmoid(-margin)`、`h=∇margin`；由归档均值精确重算，不重新训练。

这些结果限定为归档checkpoint和所选训练pair、SD1.5固定t499或Flow各阶段的条件分布。不能单凭这套图声称全训练分布反转率、跨训练seed一致性、完整训练稳定性，或最终生成效果。④报告与⑤重算来自同一次采集，按用户监督顺序分别阅读后再决定是否进入⑥。

## 本地检查范围

使用既有 `ate` 环境中的 NumPy/Matplotlib 做小向量与Gram等价检查、配对/去耦合/audit诊断和图像QA；launcher测试验证无依赖plan、固定group分片和缺pilot时拒绝正式worker。无可用CPU PyTorch，原collector autograd测试跳过。本轮没有安装依赖、下载权重、提交GPU、访问SLAI或实测5090显存/速度。
