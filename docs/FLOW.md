# Flow LoRA 偏好训练

2026-11-07 新论文实验使用 [FLOW_STAGEWISE.md](FLOW_STAGEWISE.md) 的 `paper-v2`：分模型启动入口、Stagewise K2/K4、一次性JSON凭据、明确5090精度变体及200/400保存。下文旧paper-pilot设置作为历史配方保留。

在现有 `self_correct_dpo` 中接入 SD3.5 Medium 与 Z-Image Base；继续使用同一训练循环、资源表、原子checkpoint、每rank RNG及分段生成评估。开发位于独立worktree，原目录未修改。`flow-baseline.json`记录从`c2cdb78`源工作树复制的14份既有未提交文件；它们不是本轮新增功能。当前flow快捷配方按用户要求使用首次预检复用，后续不重复核验外部资产和源码；旧SD15/SDXL默认路径保持原行为。

## 模型与共同数学定义

`PreferenceBackend` 处理本地加载、文本/VAE编码、LoRA及 reference；`MultiViewBackend` 复用一次图像 posterior 和条件，再生成独立噪声视图。没有引入 Flow-Factory 的 trainer、插件注册表或分布式运行时，也不需要安装整个 Flow-Factory。

每个 image 的误差都在 FP32 以像素/latent 元素均值计算：`P=mean((velocity-target)^2)`，target 为 `noise-clean_latent`。同 view 的 winner/loser 共享 t 和 noise；不同 view 独立抽样，不跨 batch 混合 margin。

- DPO：`z=-0.5*beta*((P_w-P_l)-(R_w-R_l))`，`loss=softplus(-z)`。
- MVPO：先对同一 pair 的 K 个 z 求均值，再算 `softplus(-mean_K(z))`，即既有 C 普通高斯定义。K=1退化为 DPO；已有 B 仍为逐 view loss 平均，未改成概率平均。
- Linear-DPO：K=1，`w=clamp(0.2*beta*((P_w-P_l)-(R_w-R_l))+0.5,eta,1-eta).detach()`，`loss=mean(w*(P_w-P_l))`。采用官方代码的上界 `1-eta`；论文记作上界1，本项目明确选择代码口径。

训练从 checkpoint 的静态 FlowMatch schedule **均匀抽取 index**，再查相应 timestep/sigma，构造 `(1-sigma)*x+sigma*noise`。它不是 uniform sigma，也没有引入 loss weighting；本地 SD35 shift=3，ZImage shift=6，实际以冻结模型 scheduler 配置为准。只有静态 shifting 模型可用。保存的 error parameterization 为 `flow_velocity_mse_uniform_schedule_index`。

MVPO 的 `--stagewise` 延续原采样：先按每 pair 抽一个 index 段，再令全部 K 个 view 在该段独立抽样。boundary 默认500，表示 schedule **index**，不表示 native timestep=500/sigma=.5；前半 index 通常对应较高噪声。它不是低/高各K/2分层采样。flow阶段只开放DPO损失，Huber/phase/reverse留给旧扩散实验。

SD35保留CLIP-L/CLIP-G/T5三分支及架构校验。ZImage使用Qwen3 chat template（thinking开启）、hidden_states[-2]与attention-mask去padding的ragged条件，transformer输入list `[C,1,H,W]`、`t=(1000-native_t)/1000`，原生输出取负后才作为共同的noise-x速度；不借用SD3 forward。VAE `(latent-shift_factor)*scaling_factor`，不硬编码比例；本地ZImage scale=.3611、shift=.1159。

flow预设显式 `model.prompt_truncation=truncate`：保留原caption/标签，仅按各模型官方tokenizer长度截断，SD35 CLIP77/T5256、ZImage Qwen chat256。训练JSONL记录各encoder每microbatch截断序列数（包含winner/loser重复条件，跨累积/rank取均值），生成审计记录prompt/negative prompt截断。可在新冻结配置改为error；原SD15/SDXL仍严格拒绝训练截断。

## SFT对照

两个骨干均提供`configs/flow/<family>-sft.json`及07/08快捷入口。SFT只读取winner，按同一静态flow schedule构造带噪latent，优化`mean((v_theta - (noise-clean))**2)`，不进行reference预测或EMA更新。每步128个数据行对应128张winner；双H100 SD35 B4/累积16、ZImage B2/累积32，四5090 SD35 B1/累积32。默认从基座独立训练3200步，首检复用及八点评估不变。它是模仿对照，不是其他方法的预训练步骤。

比较工具支持`--reference-run`，给出同一步相对DPO的配对差值及区间，且区分SFT winner预算。Z-Image与HPDv3-sub的适配风险及效果判据见[ZIMAGE_HPDV3_ASSESSMENT.md](ZIMAGE_HPDV3_ASSESSMENT.md)。

## LoRA 与 EMA

只优化FP32 LoRA参数，默认rank/alpha=16/16、dropout0；denoiser activation用BF16 autocast。SD35 transformer以BF16加载，ZImage默认**直接FP32加载**transformer（与BF16权重加载后转FP32不同）；可在新配置显式改 `model.transformer_dtype=bfloat16`，这属于新的精度实验。

2026-11-08 ZImage padding兼容修复：继续采用FP32 transformer/LoRA、BF16 autocast和FP32误差计算。锁定的Diffusers 0.37.1在`_prepare_sequence`中直接把FP32 pad参数索引写入BF16激活，会在reference前向报dtype不匹配。`zimage_compat.py`将v0.40.0的`torch.where`处理回移到模型实例，padding处按类型提升到FP32；保留0.37.1的RoPE、attention/noise mask行为及梯度。训练与生成加载均安装修复，Diffusers>=0.40.0使用原生实现。没有降低pad参数精度或修改已安装依赖；兼容代码纳入训练/评估源码身份，旧checkpoint不自动放宽恢复限制。真实tiny ZImage的CPU BF16前向/反向验证命令见`command.md`；完整GPU训练仍需实际作业验证。

两模型VAE直接FP32加载。`offload_frozen=true`使denoiser常驻GPU、VAE和文本encoder仅编码时移到GPU、随后返回CPU；减少峰值显存但会增加传输耗时，不是已测的5090吞吐保证。生成使用Diffusers model CPU offload；SD35在decode入口显式把latent cast为VAE dtype，解决BF16 denoiser/FP32 VAE接口差异。ZImage pipeline原生有此cast。ZImage仅提供H100/A800 profile，SD35可选5090/H100/A800。

当前快捷配方关闭训练offload，四5090以B1/VAE逐图编码起步，双H100的SD35 B4、ZImage B2。冻结文本条件每个microbatch按独立caption只编码一次，再还原winner/loser顺序，截断计数仍按原重复条件加权。EMA形状/dtype/有限性只在构造与加载检查，不再每次更新逐参数触发GPU同步。

固定reference禁用policy LoRA，复用冻结base。Linear reference为 **LoRA因子EMA**：分别对A/B参数做EMA，不是合并dense BA更新的EMA，也不是官方full-weight EMA的等价重现。这一适配明确作为LoRA实验选择。shadow仅保存LoRA FP32 tensor；reference以未包装DDP的PEFT model通过`torch.func.functional_call`替换这些tensor，no_grad/eval前向并恢复policy mode，不替换policy Parameter或optimizer绑定。

每次真实 `optimizer.step()` 之后更新一次EMA；梯度累积microstep/评估不更新。完整checkpoint的`state.pt`保存shadow、decay、update count并由已有manifest SHA覆盖；恢复严格核对kind、名称、形状、FP32、有限性、decay和updates=optimizer step，缺EMA拒绝恢复，不从policy伪造shadow。LoRA快照只保存policy供生成。

## 配方与预算

使用 `--recipe paper-pilot`，步骤见 [FLOW_QUICKSTART.md](FLOW_QUICKSTART.md)：默认同pairs128、MVPO K2、64条validation面板、直接训练3200步，评估基座及200/400/800/1300/1800/2300/2800/3200。显式评估点同时保留LoRA。快捷配方使用`preflight=once`，复用首检结果，停用逐pair曝光诊断和消费图片重复SHA；每次优化更新才汇总指标至CPU。以下通用等pair-view/B1配置保留为对照，快捷入口不要求先跑smoke矩阵。

这些是pilot起点，尚未验证新骨干GPU效果。共同1024、AdamW、lr=5e-6、warmup200后constant、3200 updates、每200完整保存、每1000额外保留LoRA。初始beta=500；DPO/MVPO建议试500/2000/5000，Linear建议100/500/2000，不将不同方法beta直接视作已调优值。

Linear pilot eta=.01、EMA decay=.995来自[论文§5.3](https://arxiv.org/html/2605.21123v1#S5.SS3)的SD3-M full-weight实验；源码`AdapterEMA`公共默认.9999只是历史代码默认，flow配置明确覆盖为.995。这些数值还未在LoRA/SD3.5/ZImage上验证。论文SD3-M不是SD3.5；作者shell脚本未传启用Linear/EMA flag，不能当作有效Linear配方证据。

| 预算 | 方法 | K | global pairs | 2卡累积(B1) | 4卡累积(B1) |
|---|---|---:|---:|---:|---:|
| pair_views=128 | DPO / Linear | 1 | 128 | 64 | 32 |
| pair_views=128 | MVPO | 2 | 64 | 32 | 16 |
| pairs=128 | MVPO | 2 | 128 | 64 | 32 |

`run_flow.py`支持views/beta/LR/seed/EMA/eta覆盖并写入冻结配置。每个服务器独立run，不跨站点DDP。新flow预设用torchrun `--standalone`独立rendezvous，避免多个实验共享Accelerate默认端口。正式只支持2/4卡；smoke允许1卡、2步、累积1、每步完整保存、关闭评估，和正式run独立命名。

## 数据与资源

HPDv3-sub只保留model1 **和** model2均在 `flux,kolors,sd3,hunyuan,real_images` 的pair，官方path1为preferred/path2为rejected，保留原record与relative image path，不重打分，不额外按原图1024过滤。不保证筛出210008行；最终数量由真实metadata、heldout image/prompt排除及重复检查决定。

`prepare_hpdv3_sub.py --metadata-only`输出原记录selection及filter统计/来源与输出hash，不产出可训练ready；完整模式复用原prepare/adapt流程，校验图片字节，排除官方test prompt/image overlap，按prompt及共享image组件隔离90/5/5 split并生成`pairs.jsonl/audit.json/ready.json`。完整prepare/全量metadata扫描放计算资源；本轮只跑了100对小图fixture，没有对真实数据全量解码。`prepare_flow_monitor.py`从独立validation split固定128条prompt，拒绝与train规范化caption重叠，两骨干共用面板。

`resources.json`新增ZIMAGE_MODEL、HPDV3_SUB_PREPARED、MONITOR_PROMPTS_FLOW，延续环境变量 > resources.local.json >公共资源表。公共AOSS ZImage路径是可覆盖的规划路径，远程精确路径未核实，禁止把它当作已有模型证明。本地模型/HPDv3均可复用，无下载步骤；必须提供模型来源revision或complete下载receipt。实际命令见根目录`command.md`。

## 环境迁移与验证范围

保留当前已验证torch2.14.1/cu130、torchvision、CUDA、NCCL；旧实验继续固定原dpo环境。需要新flow依赖时克隆为dpo-flow，磁盘预计超过1GB前检查目标卷；不在运行旧任务的环境中直接升级。不执行完整Flow-Factory安装。

`requirements-flow.txt`锁定模型API八包：diffusers.37.1、transformers4.57.1、PEFT.17.1、Accelerate1.14.0、hub.36.0、tokenizers.22.2、safetensors.7.0、sentencepiece.2.1。使用官方v.37.1 ZImage源码核对上述forward/tokenizer/LoRA API，依赖元数据记录在`flow-package-metadata.json`，Python3.10与八包间约束一致。项目`pyproject.toml`已放宽模型库支持范围并提供相同`flow` optional pins，原`requirements.txt`仍保留旧实验锁；克隆环境需要`pip install --no-deps -e .`更新本项目元数据和editable路径。宽版本范围不代表任意交叉组合均经GPU验证。这里是**目标模型栈的版本锁和元数据兼容检查**，不是完整环境所有transitive包的已安装freeze；未安装/import验证新栈。选择.37.1避免diffusers.40（hub>=1.23）与transformers4.57（hub<1）的冲突。`check_flow_environment.py`检查准确版本及模型类，输出版本，不替代CUDA/分布式smoke。旧PickScore/HPS等评分器仍需独立核验，可沿既有独立scorer环境使用；本轮没有验证新依赖下实际评分模型。

本轮现有CPU环境：Python3.10.22、torch2.3.0+cpu、diffusers.33.1、transformers4.45.2、PEFT.10、Accelerate.34.2。数学、真实tiny PEFT及生产训练/保存恢复检查使用它，未加载完整flow模型。测试覆盖K1退化、pair共享/独立视图、flow静态schedule/stagewise、joint/sequential梯度、Linear权重、EMA和CPU/Gloo恢复、预算及HPDv3准备。实际SD35/ZImage GPU训练、显存、生成、新依赖与真实PickScore尚未验证；按用户要求直接用正式入口运行，从正常训练日志观察速度，不再要求先跑分布式测试矩阵。

## 来源与许可证

- [Flow-Factory fbb038e](https://github.com/X-GenGroup/Flow-Factory/tree/fbb038e3570cb8f0e19cf74d910d6ea1d722ffd2)：选择性借鉴adapter边界、offline DPO误差/配对思想。其offline_dpo逐view loss平均，与本项目MVPO的mean-margin定义不同。Apache-2.0，保留于`licenses/Apache-2.0.txt`。
- [Diffusers v0.37.1 ZImage pipeline](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/z_image/pipeline_z_image.py)与[transformer](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/models/transformers/transformer_z_image.py)：Qwen条件、ragged输入与时间/符号遵循其公开接口。Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team，Apache-2.0。项目修改为偏好batch、truncation审计与LoRA reference。
- [Linear-DPO 663179c](https://github.com/Whynot0101/Linear-DPO/tree/663179c7adbbbd2d77b97b5841534447eb291ebd)与[论文](https://arxiv.org/html/2605.21123v1)：公式及超参来源；本项目自行实现目标和adapter EMA，不搬运作者trainer。该revision根LICENSE链接404，未假定其源码采用Apache许可证。
