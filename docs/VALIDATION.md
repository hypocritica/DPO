# 本次整理验证

## 每200步滚动保存与每千步永久LoRA（2026-10-18）

SD1.5/SDXL的B/C主模板将 `training.checkpoint_every` 设为200，沿用已有原子滚动恢复机制，保存最新LoRA、AdamW、LR与各rank随机状态。`lora_checkpoint_every=1000`、永久LoRA保留与评估节奏不变；不增加独立保存框架，也不修改旧实验或归档配置。

50项CPU测试全部通过，耗时180.852秒，环境为既有独立torch2.3.0+cpu调试环境。新增测试实际执行小模型1200个优化步，在完成1050步后模拟训练异常，检查滚动提交步数为200/400/600/800/1000、仅有一个完整恢复槽且千步LoRA另存；从1000步恢复至1200后，永久1000步LoRA字节不变，最终模型、AdamW、LR、训练generator和曝光计数与连续训练精确一致。

2/4/8rank测试使用缩小为2/10步的独立保存计划，覆盖B/C同步、全rank恢复、阶段评估、单rank信号与worker故障；配置生成检查两种模型的2/4/8卡设置均继承200/1000步计划并保持全局batch。原37份源码/配置、本包训练源码、历史归档和requirements哈希未变。未安装依赖或申请GPU作业，真实CUDA/NCCL与SLAI镜像仍需实机验证。

## 单机8卡覆盖（2026-10-18）

在既有独立torch2.3.0+cpu调试环境中，扩展后的49项CPU测试全部通过，耗时约189秒。torchrun实际启动8个Gloo rank：B/C各自对比直接全局目标梯度，全部rank参数同步，连续训练与暂停恢复的参数、AdamW、LR、各rank RNG和随机探针逐项一致。8rank pipeline验证单份评估、评估中断补齐、rank7收到TERM后共同保存，以及rank7异常后保留最近完整checkpoint。2/4rank梯度与恢复回归同时通过。

SD1.5/SDXL的2/4/8卡配置生成均保持原全局batch。8卡默认分别为每卡batch8、累积4、全局256，以及每卡batch1、累积16、全局128；模板仍为单卡。两种模型的8卡配置另通过实际 `train_ddp.sh --check-only` 入口预检。原37份源码/配置、本包既有配置、历史归档、timestep采样实现和requirements哈希未变化。

本次只修改独立包的启动提示、配置帮助、测试和文档，未改动已有实验或环境，也未提交GPU作业。CPU/Gloo使用小型denoiser与受控评估器；真实8卡UNet、CUDA/NCCL通信、显存和SLAI镜像仍需实机验证。启动与恢复命令见[DISTRIBUTED.md](DISTRIBUTED.md)。

## 单机DDP与SLAI torch2.3适配（2026-10-18）

49项CPU测试在独立Conda环境通过，耗时约101秒。环境为Python3.10.22、torch2.3.0+cpu、torchvision0.18.0+cpu、Diffusers0.33.1、Transformers4.45.2、PEFT0.10.0、Accelerate0.34.2；pip check通过。没有修改旧实验环境的依赖。

多进程测试实际用torchrun启动2/4个Gloo rank，执行生产的JointRolloutBackend、B/C流式反传、AdamW与checkpoint循环，小型denoiser替代UNet。对比直接全局目标autograd与DDP的首步AdamW一阶矩，验证全局梯度正确；连续与暂停恢复后的参数、AdamW、LR、各rank generator/CPU RNG和Python/NumPy随机探针一致，各rank最终参数一致。单个rank收到TERM时全部保存同一更新；worker异常时最新完整checkpoint不变。

pipeline集成测试启动真实多进程训练阶段，再用受控评估器验证阶段退出、单份评估、中断后补齐和继续训练。stage_id区分每次启动的提交结果，不将旧结果或torchrun统一错误码误认为阶段完成。配置工具对2/4卡保持原SD1.5/SDXL全局batch。既有timestep_sampling测试及SDXL随机小组件的前向/梯度、LoRA保存加载和小图生成均通过。

requirements接受用户所选NGC24.04镜像的torch2.3.0a0/torchvision0.18.0a0构建，并固定兼容库。包元数据同步支持torch2.3，避免安装包时要求升级torch2.6。本次DDP改造未安装或替换原镜像、未连接SLAI、未申请或提交GPU作业、未下载模型/数据；真实CUDA12.4/NCCL、驱动与SDXL 1024px GPU运行仍待实机验证。CPU Conda环境不等价于NGC定制GPU镜像。

现有实验目录、37份原始源码/配置以及独立包原有训练模板、历史配置与timestep采样实现的哈希未变化。新增改动仅在self_correct_dpo及单独新环境中。多卡入口、恢复要求见[DISTRIBUTED.md](DISTRIBUTED.md)，镜像部署见[SLAI.md](SLAI.md)。

## 最终时间均匀采样（2026-10-18）

新增 `timestep_sampling=uniform_final`，旧默认original保持随机流逐位一致。44项CPU测试通过，包含5项新增采样测试：均匀最终时间、原条件锚点分布、支持集边界、K2配对共享/视图独立和重放、oracle退化。

真实GPU smoke作业65126在A800 MIG 3g.40gb上完成512px/BF16、batch8、K2-C/model rollout两步训练，第一步暂停后恢复第二步，并完成0/1/2步各2条提示词的DDIM30生成与PickScore评分。梯度有限且非零，训练峰值显存约6.94GiB。该smoke使用accumulation=1，只验证运行链路，不代表正式accumulation=32的完整步速或图像质量结论。

证据位于LMDPO的 `experiments/sd15-selfcorrect-uniform-k2c-6000-20261017T203100Z/smoke-complete.json`。正式6000步实验已冻结独立源码，作业65127/65128串行提交；提交时处于排队状态。正式监控沿用0/200/每500步节奏，但输入为本地独立test划分的固定128条提示词，**不是官方test_unique**。旧实验未修改。

## SDXL组件扩展（2026-10-18）

39项CPU测试通过，包含原33项回归。新增验证SDXL B/C模板、标准base双编码器/UNet/VAE/scheduler预检及错误架构拒绝、原SD1.5默认值、外置VAE路径与身份、adapter/VAE版本一致性、配置生成与隔离模式训练check-only，以及嵌套条件的停止梯度、K1单/联合扰动精确一致。

使用随机初始化的真实小型Diffusers UNet/VAE、两个Transformers CLIP编码器和PEFT LoRA，在CPU上验证普通多噪声与扰动前向、非零有限梯度、冻结参考/编码器、LoRA保存/加载结果逐位一致。还通过保存完整小pipeline和VAE后，调用本项目renderer重新加载SDXL+LoRA+外置VAE，实际生成16px小图。全部使用现有环境，未下载权重或安装依赖。

本地通用模型目录未发现SDXL权重，未运行真实SDXL 1024px/BF16 GPU训练、暂停恢复、显存或奖励评分。没有提交、取消或修改任何集群作业，原实验源码/配置与SD1.5主模板保持不变。修改只在独立包内；新版本使用新实验目录，不接管旧代码运行中的作业。

## 2026-10-18 更新

可选评估扩展后共33项CPU测试通过（约7秒）。新增覆盖：默认仍仅HPDv2/四指标且忽略未选中的环境变量；独立Parti TSV、HPDv3 JSON、ImageReward prompt导出及训练重合排除；六指标复用同一PNG面板并分派到各自Python；多数据集分别输出、拒绝改配置覆盖；HPSv3显式本地构造、临时YAML覆盖、mu列选择与输出形状检查；ImageReward本地tokenizer、特殊token、原工厂恢复、官方score调用配对与严格权重加载；可选大评分器默认batch1及显式覆盖。测试使用小张量和受控的官方API替身，未加载真实HPSv3/ImageReward模型。

新增可选源文件、配置和部署文档；B/C主模板与pipeline文件保持字节不变。原始37份源码/配置哈希仍保持不变。未向当前环境安装可选依赖，未下载数据/模型，未提交或修改GPU作业。可选环境的依赖文件供部署使用，尚未在新环境完成依赖解析或GPU验证。

新版本只启用B/C，默认C。25项CPU测试通过，包含第200步/每500步评估调度、每1000步及最终LoRA保留、超过原4份快照上限仍保留、固定128条test子集、训练caption排除、单seed与PNDM评估配置、长prompt显式截断、配对cosine尺度、HPS版本检查、LAION MLP结构、逐图评分中断恢复、四个串行评分worker、完成后清理图片、未完成评估先补齐再继续训练，以及实际CSV/PNG曲线导出。B/C连续训练与暂停恢复的参数、AdamW、学习率和随机流仍逐项一致。

模型加载和四项评分在测试中使用小张量/受控替身，不代表真实GPU模型已验证。未提交GPU作业、未安装依赖、未下载模型或数据；25项检查在已有环境用1个CPU线程运行。原始37个源码/配置文件哈希未变化；12份历史配置内容未变，3份A/D定义仅在新包内移到只读用途的归档目录。

当前命令为README中的unittest、Bash语法检查与离线wheel构建。真实GPU训练/生成/四项模型评分、SLAI镜像和显存仍待部署后验证。以下为首次整理的历史记录，包含当时尚启用的A/D测试，不代表当前启用范围。

日期：2026-10-16（北京时间）。验证仅使用轻量CPU检查，没有申请、提交、修改或取消GPU作业，没有安装或修改原环境，没有复制已有实验结果。

运行命令（在独立仓库根目录、已有依赖环境中）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -v
bash -n launchers/train.sh
python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /tmp/self_correct_dpo_wheels .
```

14项CPU测试通过，覆盖：

- B/C完整梯度与逐视图重算梯度；D均匀抽样的梯度期望及不额外乘1/K。
- rollout正负分支共享随机量、各自保留原图、停止梯度、epsilon/v oracle退化和随机状态重放。
- K1联合状态与独立rollout的一致性。
- A/C/D训练连续执行与暂停恢复：最终参数、AdamW状态、学习率状态、随机流及曝光计数一致；信号中断不误标complete。
- 12份历史配置验证；非法rollout+D和多进程rollout明确拒绝。
- 环境变量与相对路径解析；数据导出后搬迁、保留原标签与split、损坏图片校验失败。
- 仅复制本目录至临时位置、Python隔离模式下执行训练check-only，无需原项目导入，也不创建训练输出。
- 同一输出目录的单写入锁。

另外通过所有Python文件AST解析、JSON解析、三个顶层CLI的help、Bash语法检查和离线wheel构建。整理时记录的37个原始源码/配置文件SHA256均未变化；12份归档配置的training部分与来源逐项相等。联合训练器及模型后端的来源与当前6000步实验冻结源码内容一致。

本次未验证SLAI实际镜像、GPU内核、显存、真实数据IO或生成评分，也不声明本包已在SLAI完成训练。部署后应在实际分配的GPU上执行README的两步训练、第一步暂停、第二个进程恢复，并检查有限梯度、checkpoint及配置；随后再运行正式预算。更换PyTorch/CUDA、GPU型号或batch时记录差异。
