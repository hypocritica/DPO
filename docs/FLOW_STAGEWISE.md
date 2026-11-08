# Flow stagewise 实验（paper-v2）

2026-11-08：当前默认已更新为每卡B4、resident与连续训练；完整128 pair-views主实验及消融见[FLOW_EXPERIMENT_SUITE.md](FLOW_EXPERIMENT_SUITE.md)。下文旧B1/在线评估说明保留为历史配置记录。

本轮使用独立 `self_correct_dpo-flow-stagewise` checkout。旧 flow 目录的 HPDv3 准备和已提交训练继续使用原源码；本轮不修改或提交这些作业。旧 `paper-pilot` 入口保留，新增论文比较使用 `paper-v2` 和独立输出名。

Stagewise 保持现有全局采样的单视图边际：先为每对抽取 schedule index 段，概率为 `boundary/1000` 与 `1-boundary/1000`，再让同一对的全部 K 个视图在该段独立抽 index 和高斯噪声。winner/loser 共享同视图的 index/noise，K 个 margin 先平均后计算一次 DPO 损失。默认 boundary=500，阶段概率各 50%；250/750 边界消融自动使用 25%/75% 或 75%/25%，不会将新边界仍强制各抽一半。

实际训练读取模型的静态 FlowMatch schedule。SD3.5 的 shift=3，ZImage 的 shift=6；默认 index 中点对应 sigma 约为 .75 和 .857，不是 sigma=.5。低 index 实际噪声较高，新 flow 日志使用 `stage_high_noise_*` / `stage_low_noise_*`。内部历史 `low_stage` 字段仍表示 index<boundary，保留原 RNG 顺序和旧 diffusion 语义；flow 新增 `high_noise_stage` 明确含义。当前只支持两段同 DPO 损失、同 beta，没有 phase/Huber 组合。

| 每个模型的脚本 | K | 全局图片对/更新 | pair-views/更新 |
|---|---:|---:|---:|
| 01 DPO | 1 | 128 | 128 |
| 02 Linear-DPO | 1 | 128 | 128 |
| 03 SFT | 1 | 128 张 winner | 不与偏好训练等计算 |
| 04 Global MVPO fixed pairs | 2 | 128 | 256 |
| 05 Global MVPO fixed views | 2 | 64 | 128 |
| 06 Global MVPO fixed views | 4 | 32 | 128 |
| 07 Stagewise MVPO fixed views | 2 | 64 | 128 |
| 08 Stagewise MVPO fixed views | 4 | 32 | 128 |
| 09 Global B 聚合对照 | 2 | 64 | 128 |

`launchers/sd35/` 和 `launchers/zimage/` 分别提供以上九个独立入口。SD3.5 另有 10/11 两个 K2 boundary250/750 脚本，属于可选消融。所有入口默认每卡 microbatch=1、3200 步，梯度累积随 K/卡数/预算精确调整，不可整除即拒绝；学习率5e-6、warmup200、beta500、seed17。它们是待 GPU 验证的研究起点，不能视为已调好的最优参数。

SD3.5 默认四卡 5090；ZImage 默认双卡 A800。`--profile h100` 或 `--profile a800` 使用双卡；`--profile 5090` 使用四卡。5090 默认开启 frozen encoder/VAE CPU offload，80GB profile 默认常驻。GPU 型号是配方选项，脚本不申请资源。5090/H100 主要用于外部已分配 GPU 服务器，本集群 Slurm 当前可用型号以实时配置为准。

ZImage 默认 FP32 transformer，5090 必须显式加 `--weight-dtype bfloat16`。此精度变体写入冻结配置、输出目录名、启动日志，不能续训 A800 FP32 实验；比较方法应在同一精度内重新训练。四卡 DDP 不合并显存；该入口尚未实测32GB显存或吞吐。SD3.5 默认 BF16 transformer；全部方法 LoRA 参数仍 FP32，VAE直接FP32加载。

不增加用于生成的 policy EMA，全部方法评估当前 policy。Linear-DPO 的 EMA reference 是算法的一部分，继续保留 decay=.995 的 LoRA 因子 EMA，并在完整恢复状态保存。它不等价于 full-weight EMA，也不等于给所有方法额外启用生成 EMA。

训练每200步保存一次滚动 latest 完整状态，提交新状态后清理上一滚动槽；每400步永久保留完整 `step-*` 状态，可用于独立 eval 或恢复审查。400步的状态同时作为 latest，避免重复写入。仍保留原八点评估：200/400/800/1300/1800/2300/2800/3200，另评基座；非400评估点保留 adapter，不能单独当完整恢复点。400步永久保留不表示每400步自动生成评分。信号中断和评估阶段边界也会保存；因此 latest 不只可能位于200的倍数。权重快照和完整恢复状态应区别使用。

paper-v2 必须消费通过的一次性数据和模型 JSON 报告，缺失、处理中、失败或绑定变更直接报错，不回退到全量扫描。HPDv3 审计命令与当前准备状态见 [FLOW_DATA_AUDIT.md](FLOW_DATA_AUDIT.md)。模型一次验证例子：

```bash
python scripts/validate_flow_model.py --family sd35_medium --model /实际模型路径 --out reports/sd35-model-validation.json
python scripts/validate_flow_model.py --family zimage --model /实际模型路径 --out reports/zimage-model-validation.json
```

模型验证失败会原子写入 failed 报告，撤销同路径的旧 passed 凭据。模型报告核对必要 pipeline/transformer/VAE/text encoder/tokenizer 配置、safetensors header/形状/偏移/文件长度及分片索引，不读取完整 tensor payload，不证明数值质量或 CUDA 兼容。存在完整下载凭据时绑定上游 revision；无凭据时允许 `local-inventory:<digest>` 本地身份，并明确 `upstream_revision_verified=false`。重新运行验证不会将默认报告自身纳入模型身份。

首次训练只检查报告 schema、passed、模型path/revision、数据清单/ready/audit的路径和stat以及关键小配置哈希，解析清单供dataloader使用；跳过已在准备阶段完成的schema/跨split/标签全量检查和图片抽查。模型 inventory 与清单身份复用报告，权重仅按已有清单核对size/mtime而不读取payload；之后各训练进程/评估阶段复用 run preflight，只重新检查廉价绑定。实际消费图片仍核对读取字节的SHA256。该设计信任本地报告与文件元数据，不能检测恶意保留size/mtime的清单或全部模型payload改动；修改外部资产后应重新验证并使用新实验名。

SLAI 编号入口默认使用 `configs/resources-slai-flow.json`，模型报告放在 checkout 的 `reports/slai/`；完整部署、准备和启动命令见 [FLOW_SLAI.md](FLOW_SLAI.md)，根目录 `command.md` 提供入口。本轮 CPU 检查与实际资产报告见 `reports/flow-stagewise-implementation-review.json`；CPU 检查不代表正式 GPU 训练已开始或已成功。
