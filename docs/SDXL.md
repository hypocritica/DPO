# SDXL 开发依据与验证边界

本模块在 SDXL 上沿用 SD1.5 的纯 DPO、普通高斯多视图 B/C、独立 rollout B/C、LoRA 保存续训和配对生成评估。运行命令统一见根目录 [command.md](../command.md#sdxl10241024-的完整入口)。

## 论文和官方实现

- [SDXL 原论文 §2.3、§2.5](https://arxiv.org/html/2307.01952v1)：最终采用约1024×1024像素面积的多宽高比训练；两个文本编码器、pooled text embedding 和尺寸/裁剪条件是 SDXL 的组成部分。本项目先固定1024×1024正方形，以便和现有训练数据及对照保持一致。
- [Diffusion-DPO 原论文](https://arxiv.org/html/2311.12908v1)及[官方训练实现](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py)：对 winner/loser 共享时间与噪声，在相同输入上比较策略和冻结参考的误差差；代码使用 `-beta/2` 缩放策略/参考 gap。SDXL 默认分辨率为1024。
- [Diffusion-DPO SDXL 启动配置](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/launchers/sdxl.sh)采用完整 UNet 更新和另一套优化预算。本项目使用 LoRA、已有 β500 / LR1e-5 / 全局128对的研究配置；这组数值是本项目的起点，不能表述成原论文超参数复现。原版的 `--scale_lr`、优化器和全局 batch 也须一并考虑，不能直接比较学习率字符串。
- [Diffusers 0.33.1 LoRA 文档](https://huggingface.co/docs/diffusers/v0.33.1/training/lora)：实现保持当前固定依赖下的 SDXL/PEFT 接口，不迁移环境。多视图 B/C 和 detached rollout 是本项目研究扩展，其效果须由生成评估确认。

## 已有组件和本轮补齐

训练组件已使用两个冻结 CLIP，将 token embeddings 拼接成2048维，并使用第二个编码器的1280维 pooled projection。固定全图缩放、无随机裁剪时，time IDs 为 `[1024,1024,0,0,1024,1024]`。VAE 从 FP32 权重直接加载，posterior 采样一次后在各视图复用，latent scaling 为0.13025；不训练文本编码器和 VAE。参考通过禁用 LoRA 复用基座，只有策略 LoRA 有梯度。

本轮补齐单一 SDXL ACP 入口，支持单机2/4/8卡和五种方法，复用原有配置生成、Accelerate、每200步原子完整保存、各 rank RNG 和续训一致性检查。SDXL 数据、监控面板和输出命名单独使用 SDXL 路径；不从 SD1.5 恢复点加载训练状态。部署预检增加 SDXL base 架构和数据准备标记检查。

普通高斯合批将嵌套 SDXL 条件按视图顺序复制，与 `[K×B×2,...]` 输入对应，再还原 `[B,K,2]` 计算 B/C。rollout 保留原 detached 构造和逐视图路径。没有改变现有算法定义、β、学习率或 SD1.5 的实验配置。

## 各方法的步数和比较预算

2026-10-25 额外核对51组实际合成配置：纯 DPO 与四种 B/C 方法、2/4/8卡、多视图 K=2/4/6/8。使用原 `acp_sdxl.sh` 正式入口时，默认全部为10000个 optimizer step、全局每步128对，即128万次图片对曝光（允许重复，不是128万条独立数据）。在379对的轻量数据样例上，逐 rank、逐累积批次检查前4个优化步，方法、K 和卡数改变后，每步的图片对顺序仍一致；噪声流和浮点计算不要求相同。下表说明这一原始预算；新增两套比较入口另见本节末尾。

一个 step 在完成所有梯度累积后只调用一次 optimizer.step。双卡默认累积64次，四卡32次，八卡16次；B/C 的 K 个视图合成一个目标再反传，不产生 K 次优化更新。固定 K 时 B 和 C 的逻辑计算量相同。

以下按一次全局优化更新计算 UNet 前向样本量，一张图片在不同视图/策略/参考下分别计数；不包含反向、梯度检查点重算、诊断或生成评估，不能直接当作 FLOPs 或耗时：

| 方法 | K | 优化步数 | 图片对曝光总量 | 每步策略训练前向 | 每步参考前向 | 每步 rollout 前向 | 每步前向样本合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 纯 DPO | 1 | 10000 | 128万 | 256 | 256 | 0 | 512 |
| 普通高斯 B/C | 2 | 10000 | 128万 | 512 | 512 | 0 | 1024 |
| model rollout B/C | 2 | 10000 | 128万 | 512 | 512 | 512 | 1536 |
| 普通高斯 B/C | K | 10000 | 128万 | 256K | 256K | 0 | 512K |
| model rollout B/C | K | 10000 | 128万 | 256K | 256K | 256K | 768K |

实际 scoring 路径的轻量计数确认：每卡每个 microbatch，纯 DPO 调用2次 UNet；高斯合批也调用2次，但每次输入扩大 K 倍；model rollout 调用3K次，其中 K 次策略训练、K 次参考和 K 次 detached rollout。因此高斯合批减少的是调用次数，不能把它记成只用了一个视图。rollout 的 stride=33 是单次 DDIM 跳跃的索引间隔，不是额外迭代33步。

评估节点对齐为基座0步、200步，以及500/1000/1500/.../10000步。每200步完整保存，评估分段结束也会额外保存完整恢复点；每1000步保留 LoRA。训练内生成监控统一30个采样步，独立 SDXL 评估默认50步；这两种生成步数与优化步数分开记录，同一对照的采样设置必须一致。

上述设计比较相同优化更新和图片对访问次数下的效果。固定图片对视图总量也可以做合理对照：保留更新次数而将每步图片对数除以 K，或保留每步图片对数而减少更新次数；两者回答不同问题，需要明确记录图片对访问次数与视图次数。本次新增入口采用前者。GPU 时间和峰值显存另外报告，同视图数量不等于所有训练开销完全相等。

直接使用 Python 入口且未覆盖预算时，公共配置仍默认2000步；ACP 正式入口显式设置10000步。CLI 的 `--max_train_steps` / `--steps`、`--set`、手动冻结配置也可能改变预算，比较实际实验应读取各自 `training.json` 的最终配置与 metrics/checkpoint 中的已完成 step，不能只看带有10000的实验名。

日志 `cost.policy_image_evaluations` 已包含 model rollout 的策略前向，`cost.rollout_image_evaluations` 是它的子集。统计全部前向样本量时用 policy + reference，不能再加 rollout，否则重复计数。

新增 `acp_gaussian_fixed_pairs.sh` / `acp_gaussian_fixed_views.sh` 使用独立实验名，SDXL 两组均以192为基准并保持10000步。固定图片对模式的 K=2/4/6 都每步192对；固定视图模式分别每步96/48/32对，均为192对视图。192可在默认每卡B=1、2/4/8卡下精确整除所有 K；每个模型需一个匹配新预算的 DPO 基线，不能直接复用原128对的旧基线。SD1.5 对应基准384。完整表和运行命令见根目录 command.md。

两模式仅调整预算与梯度累积，不改变 C 的 margin 平均目标及一次联合反传。`training.comparison_budget` 固定预算口径和每步目标，配置验证拒绝不可整除、预算不匹配以及 rollout/drift 混入。训练记录新增每步图片对视图数、计划总视图曝光和实际累计视图曝光；实际累计直接读取已有 `cost.prompt_pair_evaluations`，不改变恢复状态中的计数结构。

## 验证范围

轻量检查使用现有项目环境，不安装依赖、下载权重或提交训练。配置验证覆盖五种方法在2/4/8卡下的1024分辨率、全局128对和独立 SDXL 路径；真实 tiny Diffusers/CLIP/PEFT CPU 组件验证双编码器、time IDs、rollout、LoRA 梯度和 adapter 生成往返，并比较开启梯度检查点后普通高斯合批与逐视图的 B/C 误差和梯度。tiny 模型使用16像素的小输入，仅验证组件接线和计算，不代表完整1024模型的速度、显存或生成质量。

2026-10-25 验证结果：`test_config.py` 8项、`test_sdxl.py` 7项全部通过；Bash/Python语法及 diff 空白检查通过。临时 checkout 实际执行26组 shell/配置生成组合，覆盖 SDXL 五种方法×2/4/8卡、较大高斯 K、原 SD1.5 双卡入口，以及默认纯 DPO、自动恢复、卡数/分辨率不匹配和冻结配置丢失保护。shell 验证仅替代 Conda 激活、GPU 预检和训练调用，真实配置生成与校验原样执行。

验证环境为项目现有 Python3.10.19、PyTorch2.6.0、Diffusers0.33.1、Transformers4.53.3、PEFT0.10.0；远程 requirements 中的 PyTorch2.7.1/CUDA12.8 和 Transformers4.45.2 组合仍需真实 GPU 检查。

部署前按 command.md 在真实已分配 GPU 上运行1024两步训练及暂停恢复；普通高斯增加 K 时先在相同 K 下检查显存。SDXL 模型更大，多卡 DDP 各自保留完整基座，不能把显存相加。正式比较同时记录生成指标、图片对曝光、K、GPU 时间和峰值显存。

两套预算入口追加验证：11项配置测试（含48组预算组合）和3项 CPU 保存恢复测试通过；新增日志字段在暂停恢复后继续累计，并与已有 cost 对齐。60组新启动配置及26组原入口检查通过，后者为临时副本中的真实 shell/配置生成，替代了环境激活、GPU预检与实际训练。没有执行完整模型训练或测量GPU性能。
