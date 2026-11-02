# SFT 与偏好目标对比

这组实现用于同一训练框架中的受控对比：复用本项目的数据划分、LoRA、预处理、采样、优化器、保存恢复和评估。它不宣称逐项复现各论文的完整训练配方。运行命令统一见根目录 [command.md](../command.md)。

## 论文实验如何安排

建议保留 SFT。它回答「仅用较好的图片继续训练，是否已经足以解释收益」。输入仍是同一份训练集 `pairs.jsonl`；每对只取 winner 与原 caption，用标准扩散去噪 MSE 更新 LoRA。SFT 不需要单独生成数据，也不把 loser 当训练目标，不执行 reference 前向。训练入口沿用偏好对的来源和 split 校验。

| 模型 | 已有对照 | 新增单视图基线 | 组合实验 |
|---|---|---|---|
| SD1.5 | Base、DPO、当前多视图方法 | SFT、Diffusion-KTO、MaPO、DMPO、DSPO | DMPO/DSPO + 当前多视图 C；B 可作聚合消融 |
| SDXL | Base、DPO、当前多视图方法 | SFT、MaPO、DMPO、DSPO | DMPO/DSPO + 当前多视图 C；B 可作聚合消融 |

主表保留 DPO 与未训练的 Base，不能只留下新增算法。DMPO 单独成行，DMPO + 多视图另成行；组合结果不能当作原 DMPO 的复现结果。最小组合验证是 `DPO K1 / DPO + 当前视图 / DMPO K1 / DMPO + 相同视图` 四格。K 与最终主方法保持一致；入口提供 K=2、4、6，不能按测试集分数替每个方法挑不同 K。

默认固定 pair 预算：SD1.5 每步384对，SDXL 每步192对，10000个优化步。另支持固定 pair-view 预算384/192，K增大时相应减少每步pair数。SFT的一个pair对应一个winner；其余单视图目标对应winner和loser两张图。SFT/MaPO没有reference前向，rollout还有状态构造成本，**相同pair或pair-view预算不等于相同FLOPs**。同时报告实际更新数、样本曝光、policy/reference图像前向数量、耗时和峰值显存。

所有方法使用相同训练划分、base、LoRA rank、生成prompt和seed以及模型系列内相同分辨率、scheduler、CFG、采样步数。先用独立验证面板选择学习率/目标超参和checkpoint，再对锁定checkpoint跑测试集；不要用最终测试指标反复选checkpoint。默认研究起点沿用本项目 seed=17、LoRA rank/alpha=16、学习率1e-5、BF16；正式论文应记录超参选择范围和随机种子，不能将尚未训练的默认值写成最优设置。

## 最终K6 Stage实验与论文表格

2026-11-03确认的当前待跑矩阵：SD1.5用两张RTX5090、SDXL用四张RTX5090；每个基座分别跑DSPO K1、DSPO + K6 Stage-C、DMPO K1、DMPO + K6 Stage-C、DPO + K6 Stage-C。启动器显式选择fixed pair_views，SD1.5每更新384、SDXL每更新192，均10000更新；普通compare入口的默认pairs不会自动变更。此处Stage是随机共享阶段采样，不含phase/Huber，不采用3+3分段覆盖。

建议按以下方式报告；K=6应由验证集选择，不为每个测试指标另选K：

| 内容 | 放置位置 | 回答的问题 |
|---|---|---|
| Base、DPO K1、适用的SFT/KTO/MaPO/DMPO/DSPO；DPO/DMPO/DSPO + K6 Stage-C | 主结果表，可将三种目标的组合放主文单独小表 | 总体效果及组合对不同目标的可迁移性 |
| K=2/4/6，固定同一objective、Stage设置、pair_views和步数 | 主文消融表或曲线，完整数字可放附录 | 收益如何随视图数变化 |
| K6 Gaussian-C / K6 Stage-C，分别对DPO/DMPO/DSPO比较 | Stage消融，至少主文报告关键结果 | 固定K后，阶段采样本身是否贡献收益 |
| DPO K6 Stage-C / DPO K6 Stage-C + phase（low Huber/high DPO） | 目标函数消融 | Huber阶段目标是否提供额外收益 |

“没有phase”须区分：去掉Stage是全时间域Gaussian-C；保留Stage但关闭Huber/phase，就是本次DPO K6 Stage-C。DMPO/DSPO此次均保留各自损失，不支持照搬DPO phase。若后续最终方法选择phase版本，主表应展示该验证集选定版本，并在消融表保留无phase对照。

只比较DSPO/DMPO K1与K6 Stage-C可以支持整个组合有效，不能独立归因于Stage；需要各自的K6无Stage对照。没有必要把K2/K4的全部组合塞进主结果表，但正文应提供K趋势或关键消融，而不只展示最优K。若3+3版本以后实现，应单列为采样消融，不能混用当前同阶段六视图的名称。

报告每种模型内相同数据/评估协议、预算、实际耗时和显存；fixed pair_views不等于精确等FLOPs。最终checkpoint按验证集规则选择，测试集只作锁定后的报告；关键结论补不同训练随机种子。SD1.5旧K1双卡baseline符合本次配方时可复用；SDXL DSPO阈值依赖跨rank microbatch，不能把双卡B1结果视作本次四卡B1的完全等价替代。

## 损失与视图构造分离

`training.objective` / `--objective` 选择损失；`method` / `--method` 选择视图构造。为兼容旧接口，`method=dpo` 在新增目标下表示K=1的普通高斯视图，真实损失由 `objective` 决定。仅指定 `--objective` 会选择这一单视图构造；组合多视图时同时显式指定 `--method`。

新目标集中在 `algorithms/objectives.py`，训练适配位于 `algorithms/formal.py` 和 `algorithms/joint.py`，既有DPO默认公式保留。SFT/MaPO/KTO目前只开放标准K=1基线，KTO仅开放SD1.5。DMPO/DSPO可以复用高斯B/C和独立rollout B/C。分阶段高斯C可以搭配DMPO/DSPO，但只开放 `stagewise.objective=dpo` 的阶段采样：原来的phase/constant/reverse Huber混合尚未定义这些组合，配置会拒绝，避免静默改变目标。旧paired-drift研究分支不在这些组合范围内。

这里的多视图是同一caption/latent pair的时间步与噪声/状态采样，不是多相机或图像裁剪。

统一对比入口在SD1.5和SDXL均开放 `--method stagewise-C-gaussian`，可选DPO/DMPO/DSPO及K=2/4/6。SDXL保持1024分辨率、每卡默认1对、10000步；固定pair预算为每步192对，固定pair-view预算为每步192个pair-view（K2/4/6对应96/48/32对）。DMPO沿用本项目beta=6000、alpha=.01；DSPO沿用 `official_sdxl`、beta=2000及阈值.1，不使用SD1.5的score-residual配方。采样仍是每pair先选阶段、全部K个视图在该段独立抽t与噪声；DMPO聚合margin后算KL，DSPO聚合margin产生共享gate并保留各视图的残差和阈值分支。

## 原始来源与适配边界

以下 `e_w,e_l` 是每张图的去噪误差均值；不是精确图像对数似然。不同方法的beta定义不同，不能仅因同名就复用数值。

### SFT

`L = mean(e_w)`。训练只读取、编码和预测winner。清单级预检仍遵守原有pair数据契约。此定义与 [Diffusion-KTO论文的SFT基线](https://arxiv.org/html/2404.04465v2)一致。

### DMPO

依据 [DMPO论文v1](https://arxiv.org/html/2507.07510v1)公式14及附录B.3伪代码。设 `r = e_reference - e_policy`、`u = beta * (r_w-r_l)`、`p = sigmoid(u)`，目标为：

`L = p log(p/(1-alpha)) + (1-p) log((1-p)/alpha)`。

它是从预测Bernoulli分布到平滑偏好分布的反向KL。有限最优margin为 `log((1-alpha)/alpha)`。实现用稳定的log-sigmoid，不能替换成label-smoothed DPO交叉熵。初始配置 `dmpo_alpha=0.01`，beta为SD1.5=2000、SDXL=6000；这是论文报告值对应的研究起点，仍须在本框架的mean-MSE和LoRA设定下验证。

旧DPO使用 `beta/2` 乘误差差；DMPO按原伪代码使用 `beta`。比较目标本身时若希望使用相同margin尺度，需要显式设置 `beta_DMPO = beta_DPO/2`，并标明这是匹配尺度的消融。不要把该换算当成论文推荐beta。

DMPO-B：先对每个视图计算上述KL再平均。DMPO-C：先平均各视图的 `u` 再计算KL；非线性次序与现有C一致。K=1时B/C相同。这些多视图/rollout/分阶段组合是本项目扩展，论文没有替它们提供收益保证。

### DSPO

依据 [DSPO作者代码固定版本](https://github.com/huaishengzhu/DSPO/blob/8b4e09b919c118710fb0429dea4776d41def40d7/train.py) 的实际损失；仅支持原实现使用的epsilon预测。原代码的SD1.5和SDXL配方不同，因此配置显式保存 `dspo_recipe`。保留本项目LoRA和数据/预算设置，不能称为原论文完整复现。

每个视图定义 `z_k = -beta/2 * [(e_policy,w,k-e_policy,l,k)-(e_ref,w,k-e_ref,l,k)]`、`g_k=sigmoid(-z_k)`。令 `a_k=epsilon_policy,w,k-epsilon_k`、`d_k=epsilon_policy,w,k-epsilon_ref,w,k`，向量平方范数均表示分量均值。

- SD1.5 `score_residual`：`L_k = ||a_k-beta*g_k*d_k||²`，默认beta=0.01。
- SDXL `official_sdxl`：每个视图计算当前microbatch的 `mean_B ||d_k||²`；达到阈值0.1时使用 `g_k²||d_k||²`，否则使用 `g_k²`，再对样本平均。默认beta=2000。

beta是作者README给出的搜索范围内的起点，不是本项目已验证的最优值。原SDXL代码按rank本地microbatch判断分支；本项目按每个视图跨rank归约后判断，保证各rank一致，是明确的分布式适配。梯度累积的每个microbatch分别判断，不能把任意microbatch划分视为完全等价。K>1仍逐视图判断阈值，不将所有视图的gap混成一个标量。

**DSPO-B** 是各视图完整原损失的平均。**DSPO-C** 使用 `g_C=sigmoid(-mean_k z_k)` 替换所有视图的gate，随后平均各视图损失。SD1.5对应 `mean_k ||a_k-beta*g_C*d_k||²`；SDXL仍保留各视图的阈值分支。K=1时两种聚合相同。C、rollout和stagewise组合是本项目提出的扩展，不能写成DSPO论文原定义；也不先平均不同时间/噪声下的向量残差再平方。

后端额外提供可微 `D_k=||d_k||²`。利用恒等式，SD1.5每视图损失为 `A-c*(A-R+D)+c²*D`，其中 `A=e_policy,w`、`R=e_ref,w`、`c=beta*g`。这样接口只传标量统计，无需将完整预测向量跨视图堆叠到loss模块。reference及阈值判断停止梯度，policy误差、D和gate均保留梯度；autograd仍需保存反向所需张量。MSE/gap以FP32归约，小统计量的展开式以FP64计算后返回FP32，以减少消去误差；不声称与直接向量公式逐bit一致。

### MaPO

依据 [MaPO论文](https://arxiv.org/abs/2406.06424)及[作者代码固定版本](https://github.com/mapo-t2i/mapo/blob/374b3d3dd21d1c15a511547df46d355c9a6b8931/utils.py)。复现其 `compute_loss` 的分数变换与winner项：

`phi(x) = x/(exp(x)-1)`，`u = T * (phi(s*e_w)-phi(s*e_l))`，`L = mean(e_w) + beta_mapo * mean(softplus(-u))`。

默认 `mapo_snr=0.5`、`mapo_beta=0.1`，T来自训练scheduler的时间步数。`mapo_snr`是该实现中的常数误差缩放，不是每步动态计算的SNR。数值实现处理x接近0和很大的情况。MaPO不使用reference，也不使用DPO/DMPO的beta作为ranking系数。SD1.5版本是对作者SDXL实现的架构适配。

### Diffusion-KTO

依据 [论文](https://arxiv.org/abs/2404.04465)及[作者代码固定版本](https://github.com/jacklishufan/diffusion-kto/blob/226bbd02580a557f5f516c9185619de8216b5500/train_kto_sd_v1.5.py)中的sigmoid utility。

每对拆为winner/desirable、loser/undesirable两个二元样本；设 `r=e_reference-e_policy`，`z=max(mean(r.detach()),0)`。winner loss为 `lambda_d * sigmoid(-beta*(r_w-z))`，loser loss为 `lambda_u * sigmoid(beta*(r_l-z))`，对全部样本平均。初始beta=1000，两个lambda均为1。DDP先归约所有rank当前microbatch的r均值，再clamp并停止梯度；梯度累积时每个microbatch单独估计z，因此改变microbatch大小可能改变该估计器，不能宣称仅全局batch相同就完全等价。

原论文使用每图二元反馈。本项目由pair构造相对winner/loser标签，并保留同pair的时间/噪声耦合及成对批次，以匹配现有数据曝光。它属于pair-derived Diffusion-KTO对照，不应声称拥有独立的人类绝对like/dislike标签，也不是原始unpaired数据配方的完整复现。

## 运行与验证范围

新增配置和入口复用现有原子保存、各rank RNG与源码身份校验。新算法文件纳入训练源码摘要。每个方法和组合创建独立输出，不从旧DPO checkpoint直接改目标续训；旧实验继续使用原checkout。

SFT/MaPO/KTO/DMPO/DSPO默认关闭旧的DPO误差诊断（它硬编码DPO目标）；训练目标日志和正常生成评估仍保留。生成评估复用现有adapter格式及 `evaluate.py`。

本机只做轻量CPU数学、配置、后端和保存恢复检查。完整SD1.5/SDXL权重的真实GPU训练、显存、吞吐与效果需在SLAI已分配容器执行命令中的独立smoke，再开始正式实验。


## Stagewise在K=4/6时如何聚合

当前stagewise按**扩散时间步**分段，不是将训练迭代的前半/后半阶段分开。设总时间步T=1000，边界b=500，low是t=0..499，high是t=500..999。每个pair先以b/T的概率抽low，否则high；这个pair的全部K个视图在同一段内独立抽时间和噪声。winner/loser共享该视图的时间和噪声。

| K | 一个pair抽到low时 | 一个pair抽到high时 | 段内margin |
|---|---|---|---|
| 2 | 2个low视图 | 2个high视图 | `(z1+z2)/2` |
| 4 | 4个low视图 | 4个high视图 | `(z1+z2+z3+z4)/4` |
| 6 | 6个low视图 | 6个high视图 | `(z1+...+z6)/6` |

先得到该pair的段内均值，再应用该段目标，最后对batch中的pair平均。阶段概率已经体现在抽样里，不额外给每个pair乘0.5。边界不在中点时使用b/T和1-b/T，保持每个视图的全时间边际均匀。原四种DPO消融均按此规则：dpo两段都用DPO；phase为low段Huber、高段DPO；reverse反过来；constant两段都用相同的DPO/Huber混合。K4/6不改变原目标参数，梯度中的每视图系数自然除以K。

DMPO-C在段内平均margin后计算DMPO；DSPO-C在段内平均margin后产生共享gate，各视图仍计算自己的残差。配置中的 `stagewise.objective=dpo` 在这两种组合下仅选择既有分段采样模式，最终损失仍由 `training.objective` 决定；不会额外加DPO。日志 `stage_low_fraction` 用于观察实际抽中low的pair比例。

建议主实验延续这个定义，使K2/4/6比较只改变同阶段观测数。增大K减少段内Monte Carlo噪声的空间更大，但不增加独立阶段选择次数，也不保证指标单调；固定pair-view预算时pair数随K减少，阶段比例的抽样波动可能更大。

另一种可独立消融的设计是K4按low/high各2个、K6各3个，分别计算两段margin和loss，再用 `b/T * L_low + (1-b/T) * L_high` 合并。不能先跨段平均全部margin再分配一个阶段目标。这个设计每pair覆盖两段，但每段只有K/2个视图；由于loss/gate非线性，其有限K目标与当前“同段K个”不同，不能把它当等价实现。本轮没有引入该额外采样模式。

## 8000步与数据遍历次数

以清单的实际训练pair数为分母：`遍历次数 = 8000 × 全局每步pair数 / training_pairs`，其中 `全局每步pair数 = 每卡batch × 累积次数 × 卡数`。`training.json`记录 `training_pairs`，训练指标中的 `data_passes`直接给出当前累计遍历次数。

若80万张照片恰好组成40万条不重复图片的训练pair，当前预算对应：

| 预算 | SD1.5 | SDXL |
|---|---:|---:|
| 固定pair，K=1/2/4/6 | 7.68遍 | 3.84遍 |
| 固定pair-view，K=2 | 3.84遍 | 1.92遍 |
| 固定pair-view，K=4 | 1.92遍 | 0.96遍 |
| 固定pair-view，K=6 | 1.28遍 | 0.64遍 |

若80万指的是pair记录数，上表全部除以2。若图片跨pair复用，或80万包含过滤掉的样本/验证测试集，不能从照片数直接除以2；应使用实际训练清单的pair数。K个视图是同一pair的重复噪声观测，不算K份新数据，reference前向也不计数据遍历。SFT仅训练winner，按同一pair清单记录数计算winner子集的遍历。
