# 算法审查与逐轮改进计划

审查日期：2026-10-22。对象为当前开发分支的 SD1.5 / SDXL DPO、普通多视图 B/C、独立 rollout，以及已有的 SD1.5 paired drift 分支。依据为源码、公式推导和本地历史实验记录；本次没有启动训练，也没有修改算法或默认配置。下面的新变体是待验证计划，不代表已经实现。

目前的判断：主路径的 DPO 符号、参考模型冻结、停止梯度和所有 K 视图联合反传是自洽的，没有发现这些环节的确定错误。现有较有力的对照收益来自**普通高斯双视图 C**；当前 model rollout 的额外收益尚未被隔离证明。优先保留这个有效基线，再逐项验证聚合目标、时间分布和扰动方向。

## 1. 当前算法到底在优化什么

### DPO 与多视图

令 winner / loser 的策略误差为 `eθ,w / eθ,l`，固定参考误差为 `eref,w / eref,l`，每个误差是 latent 元素上的平均 MSE：

```text
m_k = -β/2 × [(eθ,w - eref,w) - (eθ,l - eref,l)]
φ(m) = softplus(-m)
B = mean_k φ(m_k)
C = φ(mean_k m_k)
```

降低 winner 相对误差会增大 margin，方向正确。参考通过关闭 LoRA 得到，不参与梯度；每个视图的 policy 图都保留，最终联合反传。实现见 [losses.py](../src/self_correct_dpo/algorithms/losses.py)、[formal.py](../src/self_correct_dpo/algorithms/formal.py)、[joint.py](../src/self_correct_dpo/algorithms/joint.py) 和 [training.py](../src/self_correct_dpo/training.py)。

这里的“视图”是同一 caption、同一对干净 latent 上的独立时间/噪声/状态采样，**没有做语义改写或图像裁剪多视图**。VAE posterior 每张图只采样一次，所有视图复用。目标沿用去噪 MSE 构造的偏好代理目标，不应直接称为精确图像 likelihood；基础背景见 [DiffusionDPO 原论文](https://arxiv.org/abs/2311.12908)。

B/C 的区别不仅是 loss 写法：

- 固定参数、相同视图边缘分布时，B 增大 K 不改变期望目标，主要改变估计方差和计算成本。
- C 把随机 margin 的平均放进非线性函数，有限 K 的期望目标本身随 K 改变。
- 由凸性，逐样本有 `C ≤ B`。因此 C 的训练 loss 更低不能作为质量提高的证据。
- B 对第 k 个视图的梯度系数是 `-sigmoid(-m_k)/K`，C 则是共同的 `-sigmoid(-mean(m))/K`。C 允许好视图抵消坏视图。例如 `m=[20,-10]` 时，C loss 约 0.0067，B loss 约 5.0000，第二个视图仍严重排错。

C 的聚合具有合理动机，但不等于“每个视图都正确”，也不是显式的一致性正则。下一步适合验证是否需要保留一部分逐视图约束。

### rollout 的等价形式与边界

[independent.py](../src/self_correct_dpo/algorithms/independent.py) 的 DDIM 降噪和再加噪可精确化简。记 `ρt=σt/αt`，模型在 `u` 上的预测为 `ε̂θ`，则 epsilon 参数化下：

```text
κ(u,s,r) = (ρu - ρs) / ρr
ε* = ε - κ(u,s,r) × (ε̂θ - ε)
x_r = αr × x0 + σr × ε*
```

这说明它是**由模型预测误差决定的自适应噪声扰动**。`oracle` 令 `ε̂=ε`，严格退化为相同 r 上的普通前向加噪。

有四个需要实验证明的假设：

1. 模型误差方向是否比普通随机扰动更有用，不能仅由“自纠正”这个名字保证。
2. 扰动强度同时依赖时间组合和预测误差；`stride=33` 不是固定的噪声 RMS。当前没有有效噪声扰动上限。
3. winner / loser 共享原始 ε 和时间，但各自的 `ε̂θ` 不同，最终 `ε*` 通常不同。它改变了普通 DPO 的共同噪声对照结构。
4. 生成状态和标签被 detach，优化的是固定当前状态后的半梯度目标；没有计算状态生成分布对参数的导数。它不能自动继承标准高斯前向分布下的全部 DPO 理论解释。

以上是算法定义与适用边界，不是建议直接取消 detach。直接穿过 rollout 反传会改变目标和显存成本，需要作为另一个算法单独论证。

### 默认时间分布是一个主要混杂因素

当前 B/C 模板没有显式设置 `timestep_sampling`，因此使用 `original`；历史 calibrated 实验显式使用 `uniform_final`。两者不能混称同一算法配置。

对于 `T=1000, stride=33, min_anchor_t=20`，原采样有 `s∈[20,966]` 均匀，`r|s∈[s,999]` 均匀：

```text
q_original(r) = (1/947) × sum_{s=20}^{min(r,966)} 1/(1000-s)
```

对这一离散分布直接求和得到：

| 时间采样 | 支持范围 | 平均 r | P(r<200) | P(r≥800) |
|---|---|---:|---:|---:|
| 普通 DPO | 0…999 | 499.5 | 20.00% | 20.00% |
| rollout original | 20…999 | 746.0 | 1.87% | 51.16% |
| rollout uniform_final | 20…999 | 509.5 | 18.37% | 20.41% |

所以 model rollout 对普通 DPO 的变化同时包含时间分布、状态构造和 K/聚合。`uniform_final` 修正了 r 的偏斜，但没有把模型相关的 `ε*` 变回标准高斯。

还有一个数值对照细节：默认 BF16 下，普通分支用 latent dtype 的噪声及 `scheduler.add_noise`；rollout 的 oracle/model 分支用 FP32 构造噪声和状态。因此直接用“普通 C 对 oracle C”不能声称只测了时间截断。做严格时间消融时应统一状态构造精度；检验模型扰动时优先用同实现的 oracle/model 配对。

### 已有 paired drift 分支

[paired_drift.py](../src/self_correct_dpo/algorithms/paired_drift.py) 从第一个普通视图构造：

```text
v = 0.5 × [(predθ,w - predref,w) - (predθ,l - predref,l)]
δ = random_sign × RMS_clip(strength × v)
第二视图：ε' = ε + δ，x' = x + σt × δ
```

输入和 target 同步修改，且第二视图的两张图共享 δ，代数一致。随机正负号使条件平均扰动为零，更适合解释为沿特定方向的对称平滑，而非保证沿偏好改善方向前进。

已有 `direction=random` 会在当前步骤匹配每对的扰动 RMS，是必要对照。但两条训练轨迹随后模型不同，不能声称全过程逐样本等幅；需要比较实际 RMS 和 clipping 比例。`strength=0` 不消耗 sign 随机数，非零会消耗，因此同 seed 也不保证后续基础噪声完全相同。第一视图与第二视图时间独立，方向能否跨时间迁移也尚待验证。

## 2. 历史结果支持到哪一步

证据存于本机项目同级 `experiments/`，不随源码仓库发布。下表区间是原分析中的配对 prompt bootstrap 95% 区间，不包含多训练 seed 的不确定性。

| 实验与比较 | 结果 | 可以支持的判断 |
|---|---|---|
| HPD256：普通高斯 K2-C2000 − DPO2000 | scaled PickScore +0.08748，[0.04343, 0.13048]；HPSv2.1 +0.23457，[0.05143, 0.41706] | 普通多视图 C 值得作为核心基线；不是 model rollout 的净收益 |
| HPD256：model rollout K1 R600 − DPO600 | scaled PickScore −0.01427，[-0.06157, 0.02976] | 这组条件下没有清楚的正收益证据；也不能据此否定所有 rollout |
| calibrated model K2-C，4000步对基座 | raw PickScore .204588 → .207806；增益 +.003218，[.002281, .004131] | 该训练相对基座有提升，但缺同面板 DPO / oracle 对照 |
| 同一 calibrated run，4500步 | raw PickScore .207608；较基座 +.003020，[.002058, .003911] | 截至读取时最新完成点为4500，计划暂停、未完成6000；小幅回落不足以认定过拟合 |

HPD256 的 C 来自 `sd15-cd-single-20261011T190700Z/C`，冻结源码是普通 `scheduler.add_noise`，没有 `independent_rollout`。其有效 batch 与 DPO 都是256，但卡数、累积方式、随机数分配和计算成本不同；它不是同算力预算的优势证明。

旧 model-K2-B/C 的72图面板实际为8个语义场景 × 3种表达 × 3个生成 seed，unseen 只有4个场景。B600 对 DPO 的整体增益区间跨零，C600 / C1200 也跨零；不能据此稳定排序 B/C，更不能拿 B600 和 C1200 直接比较。

calibrated 的128条本地 held-out、旧72图面板、HPD256 的提示词和部分评分口径不同；raw cosine 与 scaled PickScore 不可直接拼接作差。单个训练 seed 的 prompt bootstrap 也不能替代多次训练复现。

核查入口：

- [HPD256 实验身份](../../experiments/hpd256-rewards-20261015T064830Z/experiment.json)、[指标与对照结果](../../experiments/hpd256-rewards-20261015T064830Z/full/analysis/summary.json)。
- [calibrated 实际配置](../../experiments/sd15-selfcorrect-calibrated-k2c-6000-20261018T185200Z/train/training.json)、[曲线](../../experiments/sd15-selfcorrect-calibrated-k2c-6000-20261018T185200Z/train/pickscore-curve.csv)、[暂停状态](../../experiments/sd15-selfcorrect-calibrated-k2c-6000-20261018T185200Z/train/stage-result.json)。
- [旧 B600 对照](../../experiments/sd15-joint-repair-two80-20261015T002300Z/model-K2-B/quality-comparison-0000600/comparison.json)、[旧 C1200 对照](../../experiments/sd15-joint-repair-two80-20261015T002300Z/model-K2-C/quality-comparison-0001200/comparison.json)。

## 3. 下一轮开始，按这个顺序改算法

以下是逐轮路线，不要求一次跑完整个网格。每轮只保留得到支持的分支；不能把所有改动同时加入后再解释收益。

### 第0轮：拆开三种收益，确定后续母版

先在当前实现和同一评估协议下补齐四臂：

| 臂 | K / 目标 | 状态 | 主要回答的问题 |
|---|---|---|---|
| A | K1 DPO | 普通高斯，均匀0…999 | 标准基线 |
| G | K2 C | 普通高斯，均匀0…999 | 多视图聚合整体是否值得；不单独归因于降方差 |
| O | K2 C | oracle，uniform_final 20…999 | 与 model 同时间、同 FP32 构造的对照 |
| M | K2 C | model，uniform_final 20…999 | M−O 检验模型相关扰动的净收益 |

固定模型、数据、batch、LR、β、LoRA、GPU数量、评估 prompt/seed/采样器。G−O 还含精度和时间支持差异，除非先统一构造实现，否则只报告组合差异。

需要解释默认 `original` 时，再补 **original oracle / original model** 两臂，与 O/M 成对分析。不要只增加 original-model 然后与 uniform-oracle 相减。

**决策：** G 有效而 M 不胜 O，就优先在普通多视图 C 上改聚合；M 稳定胜 O，才有较充分理由重点投入 self-correct。差异不确定时先补复现，不自动把复杂方案设为胜者。

### 第1轮：给 C 加可控的逐视图约束

这是建议优先实现的第一个新算法，改动小、不增加模型前向：

```text
Lλ = C + λ × (B-C) = (1-λ)C + λB
首轮 λ = 0 / 0.25 / 1；K 固定2，状态分布固定。
```

`B-C` 是非负 Jensen gap。这个变体检验：保留 C 的聚合，同时适度惩罚被平均值掩盖的困难视图，是否改善生成质量。λ=0 是 C，λ=1 是 B，不需要另外维护一套目标。单个随机噪声视图排错未必对应生成质量问题，λ 过大也可能追逐噪声，因此不能预设中间值一定胜出。

观察现有 per-view margin、方差、逐视图/聚合 accuracy，再增加两个简单统计即可：`B-C` 和“同一对内 margin 有正有负”的比例。无需为每个 batch 计算多份梯度余弦或添加复杂测试框架。

**保留条件：** 与相同状态的 C/B 相比，固定评估上的生成质量有可复现提升，且不是只降低训练 loss 或冲突率。若 B 或 C 端点已最好，保留端点即可。

### 第2轮：把 rollout 的强度与时间分开

在选定的时间分布和 λ 下，只改变有效噪声扰动强度：

```text
Δw,l = ε*w,l - ε
c_pair = min(1, τ / max(RMS(Δw), RMS(Δl), 1e-12))
εγ,w,l = ε + γ × c_pair × Δw,l
xγ,w,l = αr × x0,w,l + σr × εγ,w,l
```

winner / loser 共用缩放系数 `c_pair`，但保留各自扰动向量。输入和训练标签必须同步更新；v-prediction 的标签使用最终时间系数：`target = αr×εγ - σr×x0`。

先关闭 cap，仅试 `γ=0 / 0.5 / 1`，其中 γ=0 应与同构 oracle 同分布；之后只有观察到扰动尾部过大时，才独立加入一个基于开发集统计预先固定的 τ。不要在同一轮同时改 stride、γ、τ、时间采样和 β。

记录最终 `εγ-ε` 的 RMS、两分支差异及按 r 分桶的 margin。当前 `rollout_displacement_rms` 测的是中间 `z_s` 位移，不是最终 `x_r` 位移；不能用它直接替代有效噪声扰动强度。

如需要解释“动态自纠正”，增加一次固定 reference teacher 与 current policy teacher 的对照，保持其余设置一致。reference teacher 可复用禁用 LoRA 的模型，无需复制整个 UNet。两者都不胜 oracle 时暂缓该分支；这只是当前条件下的资源决策，不是对整个方向的否定。

### 第3轮：验证已有 paired drift 的方向是否有价值

这轮单独使用普通双视图，不叠加独立 rollout。先比较已有的三个条件：

- 普通 K2-C，扰动为0。
- pair contrast 方向，沿用现有 RMS cap。
- random 方向，采用现有按对匹配 RMS 的实现。

先固定 K=2、原有强度和上限；查看实际扰动 RMS / clipping 分布，以及相同协议下生成质量。若 contrast 和 random 都优于零扰动且效果相当，收益更支持通用扰动/平滑机制，不能归因于偏好方向；仅两者相当不能证明扰动有效。若 contrast 同时胜过零扰动和 random，再单独测试 `t1=t2` 对照当前独立 `t1,t2`，检查跨时间方向迁移。

需要严格复用基础噪声时，用独立扰动 RNG；仅使用相同 seed 还不够。跨训练轨迹的幅度分布仍可能不同，可另做同一冻结 checkpoint 上的方向诊断，避免把方向和幅度混为一谈。

### 第4轮：验证 K 的效率和跨模型泛化

对于胜出的普通多视图或独立 rollout 方案，再比较 K=2/4；K=3 可用于显存/速度折中，不必一开始扫所有 K。当前 paired drift 只支持 K=2，若它胜出则先保持 K=2；扩 K 需要另行定义由哪些视图产生方向、向哪些视图施加扰动，并单独验证，不能直接修改 `--views`。

比较两种预算：同图片对曝光，以及同 GPU 时间。普通 K2 比 K1 多计算，rollout 又多出无梯度模型预测；相同步数不等于相同成本。

若不同时间区间的波动是主要问题，再单独试分层时间采样：先对 uniform 时间做等概率分层，随机置换各层到视图的对应关系，保证每个视图仍具有原来的 uniform 边缘分布。对于 B 这可保持期望目标；对于 C，改变视图间耦合仍会改变有限 K 目标，应作为独立算法消融，不能直接声称目标完全不变。

SD1.5 入选方案与其直接对照至少用3个训练 seed，例如17/42/123，再迁移 SDXL。SDXL 的 β 等模型尺度参数按其自身基线固定，不直接照搬 SD1.5 的最优数值。没有 SD1.5 净收益证据时，暂不扩展到 SD3。

## 4. 每次算法迭代的固定记录

每轮实验只写清下面六项，避免不断叠功能却无法解释结果：

1. **假设：** 当前哪个机制不足，预期在哪些指标上表现出来。
2. **唯一改动：** 一条公式或一个采样规则；注明其退化回父算法的参数。
3. **直接对照：** 与父算法及必要的随机/零强度对照比较，其他条件相同。
4. **预算：** 首轮可先看1000步筛查明显失败，再按预先固定的2000步预算比较；有希望才延长，不能用早期没有收益证明最终无效。保留同曝光和同 GPU 时间结果。
5. **机制与质量：** 现有训练 margin/MSE 是机制线索；固定 prompt 的生成指标和配对图像才是效果证据。默认4对 ordinary-noise diagnostic 只是健康检查。
6. **决策：** 保留、补复现、放弃当前设置或转向；记录失败结果，下一轮从明确父版本开始。

监控128条 prompt 可用于开发和选 checkpoint；最终比较使用独立冻结评估集，统一采样器、CFG、生成 seed 和评分尺度。优先报告相对父算法的配对差值与区间，而不是只报相对基座提升。主指标与最终 checkpoint 的选择规则在实验前写明，避免从大量 checkpoint/指标中挑最好看的一项。历史 HPD256 已被查看，后续若反复据此调参，应把它视为开发参考，另留最终测试集。

2/4/8张5090用于改变吞吐时，每个对照实验保持相同卡数和全局 batch；GPU时间比较使用同型号。保持已有每200步完整 checkpoint 机制，算法变化开新实验，不能在旧 checkpoint 续训中偷偷换目标。

## 5. 文献对优先级的启发

[Rethinking Direct Preference Optimization in Diffusion Models](https://arxiv.org/abs/2505.18736) 讨论时间步奖励尺度不平衡，以及带正则的参考模型更新；[SIPO](https://proceedings.mlr.press/v306/yang26cw.html) 研究时间相关的梯度不稳定、裁剪/屏蔽与重加权。这些工作支持优先检查时间分布和优化尺度，但不能代替本项目的配对消融。

本计划中的 B/C 插值、限幅和分层采样是针对当前代码提出的实验假设，尚未做完整新颖性检索，也没有实验保证。暂不优先加入 EMA 参考、多步 rollout、额外 reward loss 或可学习视图权重，以免同时改变优化锚点、训练分布与计算预算。仅对时间乘 `p(r)/q(r)` 也不能自动修正模型相关的非高斯状态分布；对于 C 的非线性有限 K 目标，还需另行推导权重放置及其目标。

本次验证仅包括源码交叉审查、历史配置/结果核对、离散时间分布精确求和及 rollout 等价式的轻量数值检查；没有新增测试框架，没有运行 GPU 实验。
