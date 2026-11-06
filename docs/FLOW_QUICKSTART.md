# Flow正式训练：四5090 / 双H100

两骨干的 DPO / MVPO / Linear-DPO / SFT 默认直接训练3200步。首次预检通过后，分段评估后的训练与断点续训复用结果，不再重复扫描数据、模型和评分器。不要求先完成单卡、双卡和四卡的测试矩阵。

## 1. 独立代码与环境

使用新目录，例如 `/mnt/afs/L202500030/xietianyu/workspace/mvpo-flow`。不要解压到仍在运行旧实验的 DPO checkout。交付包包含源码、配置、测试和文档，不包含模型/数据/Conda环境或本机资源覆盖；直接复用AOSS已有资产。安装一次即可，多个SLAI容器访问同一个环境时，不要并发安装。

上传 `mvpo-flow-paper-pilot-20261106-v4.tar.gz` 到SLAI后，在一个尚不存在`mvpo-flow`子目录的位置解压。`--keep-old-files`会拒绝覆盖已有文件；若目录已有旧实验，选择另一个新父目录。包内 `RELEASE_MANIFEST.json` 记录源码SHA256，不用于每步训练检查。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace
tar --keep-old-files -xzf /实际上传路径/mvpo-flow-paper-pilot-20261106-v4.tar.gz
```

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/mvpo-flow
bash scripts/bootstrap_flow_env.sh \
  --conda-root /mnt/afs/L202500030/xietianyu/miniconda3 \
  --source-prefix /mnt/afs/L202500030/xietianyu/miniconda3/envs/dpo
source /mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
conda activate mvpo-flow
```

这会克隆旧dpo并仅在新环境安装flow依赖，保存GPU依赖constraints和环境freeze。实际安装未从当前机器执行：这里无法连接SLAI。[环境说明](FLOW_ENVIRONMENT.md)列出细节；完整环境是否可用由目标服务器执行结果确认。

## 2. 复用模型与数据，只准备一次共享面板

在新checkout的 `configs/resources.local.json` 设置实际存在的路径。原self_correct资源表可以复用，至少检查 `SD35_MODEL`、`ZIMAGE_MODEL`、`HPDV3_RAW`、`HPDV3_SUB_PREPARED`、`PICKSCORE_MODEL`、`PICKSCORE_PROCESSOR`。所有容器使用同一份prepared manifest与面板。

公共表默认路径如下，ZImage的具体远程目录仍需按实际下载目录覆盖：

```json
{
  "SD35_MODEL": "/mnt/aoss/xietianyu/models/stable_diffusion/sd35",
  "ZIMAGE_MODEL": "/mnt/aoss/xietianyu/models/Z-Image",
  "HPDV3_RAW": "/mnt/aoss/xietianyu/datasets/hpdv3",
  "HPDV3_SUB_PREPARED": "/mnt/aoss/xietianyu/datasets/hpdv3/prepared-sub",
  "MONITOR_PROMPTS_FLOW_PILOT": "/mnt/aoss/xietianyu/datasets/benchmarks/flow/hpdv3-pilot-64.jsonl"
}
```

若已有有效HPDv3-sub `ready.json`和`pairs.jsonl`，直接复用，不重新扫描图片。尚未准备时在分配的计算资源中运行一次：

```bash
# HPDV3_REVISION使用实际数据来源版本；以下路径需与资源覆盖一致。
python scripts/prepare_hpdv3_sub.py \
  --raw /mnt/aoss/xietianyu/datasets/hpdv3 \
  --out /mnt/aoss/xietianyu/datasets/hpdv3/prepared-sub \
  --revision "$HPDV3_REVISION"

# 此64条validation面板用于快速筛选，不能作为最终测试集。
python scripts/prepare_flow_monitor.py \
  --pairs /mnt/aoss/xietianyu/datasets/hpdv3/prepared-sub/pairs.jsonl \
  --out /mnt/aoss/xietianyu/datasets/benchmarks/flow/hpdv3-pilot-64.jsonl \
  --count 64 --seed 17
```

## 3. 分容器启动，每个容器一个实验

八组实验全部准备好，可以并行启动：两种骨干各运行DPO、MVPO、Linear-DPO、SFT。下面每行占用一个独立的**双H100**容器，八组同时运行默认共16张H100。不要在同一组卡上同时执行多行。已经分配4卡的容器加 `--world-size 4`，偏好方法全局pairs仍为128，SFT为128张winner。

```bash
bash launchers/flow/01_sd35_dpo.sh
bash launchers/flow/02_sd35_mvpo.sh
bash launchers/flow/03_sd35_linear_dpo.sh
bash launchers/flow/04_zimage_dpo.sh
bash launchers/flow/05_zimage_mvpo.sh
bash launchers/flow/06_zimage_linear_dpo.sh
bash launchers/flow/07_sd35_sft.sh
bash launchers/flow/08_zimage_sft.sh
```

SD3.5M改用四5090服务器时，分别使用以下四个入口。默认沿用原SLAI Conda路径；其他服务器通过 `SCD_FLOW_CONDA_SH` / `SCD_FLOW_ENV` 指向该服务器的独立flow环境。

```bash
bash launchers/flow/sd35_4x5090.sh dpo
bash launchers/flow/sd35_4x5090.sh mvpo
bash launchers/flow/sd35_4x5090.sh linear_dpo
bash launchers/flow/sd35_4x5090.sh sft
```

这些命令从模型的完整 `.scd-download.json` 读取revision。若模型是手动下载且没有receipt，附加 `--model-revision <实际来源版本>`，不能保留默认占位符；路径也可用 `--model` 指定。新实验默认tag为`paper-v1`，改变任何超参需指定新的 `--run-tag`。数据/模型准备问题只处理一次，不为每个方法重复做准备。

八个入口沿用同一个 `mvpo-flow` 环境，各自的冻结配置与输出名包含骨干、算法、GPU和B，不会互相覆盖。**默认跑满3200步，不在800步暂停**。Linear-DPO已启用EMA，不需要另传开关；ZImage使用自己的Qwen条件、时间转换和原生transformer接口。脚本只启动所在已分配容器的实验，不自动申请资源。

三个偏好方法的第一轮设置：

| 参数 | DPO | MVPO | Linear-DPO |
|---|---|---|---|
| 分辨率 / LoRA rank / alpha | 1024 / 16 / 16 | 同左 | 同左 |
| 学习率 / warmup / schedule | 5e-6 / 200 / constant | 同左 | 同左 |
| beta | 500 | 500 | 500 |
| 每步图像对数 | 128 | 128 | 128 |
| K / 每步pair-views | 1 / 128 | 2 / 256 | 1 / 128 |
| reference | 固定基座 | 固定基座 | LoRA因子EMA，decay=.995，eta=.01 |
| 默认总步数 | 3200，不自动暂停 | 同左 | 同左 |

SFT独立从基座训练，仅使用同批数据行中的winner，目标为flow速度MSE；不使用loser、reference、beta或EMA。默认同样lr5e-6、warmup200、LoRA16、1024、3200步，每步128张winner，沿用下表的每卡B与累积。这里SFT的B表示winner张数；相同128行预算不代表与DPO等计算量，也不先用SFT初始化其他方法。

两个骨干使用相同64条validation prompt和seed17。除基座外，评估时间表固定为 **200、400、800、1300、1800、2300、2800、3200**。每个评估点保留LoRA和图片，完整状态每200步及评估分段末尾更新latest；新增每400步永久保留完整状态（含optimizer/各rank RNG），不随latest滚动清理。200步恰好结束warmup，不据此单点判定失败。

硬件配置如下；B指每卡图像对数，**每对包含winner/loser两张图**。H100/A800的预设按80GB级卡设计，实机显存/吞吐尚未测量。

| 配置 | 每卡B | 累积 | 全局pairs/update | MVPO K2 pair-views/update |
|---|---:|---:|---:|---:|
| SD3.5M，4×5090 | 1 | 32 | 128 | 256 |
| SD3.5M，2×H100 | 4 | 16 | 128 | 256 |
| ZImage，2×H100 | 2 | 32 | 128 | 256 |

四个方法按同骨干采用相同B；`--batch-size N`可覆盖，累积自动重算以保持128对。增大B不自动改变LR/beta。改变B使用新run-tag，不能中途改变同一run的批次划分。

当前配方全部让模型与冻结编码器常驻GPU；5090以B1和VAE逐图编码控制峰值，保留BF16、梯度检查点和逐view前向。按本地SD3.5M safetensors header（排除重复fp16变体）统计，主干/CLIP/T5以BF16、VAE以FP32计静态权重约15.3GiB，其中T5约8.87GiB。标准[RTX5090是32GB](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)，这个容量支持尝试B1 LoRA，但未包含激活/LoRA/allocator，不能当成显存实测；四卡DDP也不合并单卡显存。为速度不默认在每个microbatch搬运T5权重。显存不足时可用新tag加 `--frozen-placement offload`，但会明显增加传输量，不能保证其速度。

训练里还去掉了winner/loser重复文本编码、逐microbatch逐指标GPU→CPU同步、逐参数EMA有限性重扫、逐pair曝光统计和重复图片SHA。日志直接打印 `seconds_per_update`、`pairs_per_second`、`peak_cuda_GiB`，正常训练开始后就能观察实际吞吐，无需另跑一套benchmark。未测量真实GPU，因此不承诺具体秒/步或GPU间加速比。

MVPO在此对照中使用两倍pair-view计算，不将结果宣称为等计算量优势。另开 `--budget-unit pair_views --budget 128 --run-tag compute-v1` 可做等计算量对照，MVPO每步改为64对，需分别报告。

## 4. 看效果并直接续训

每个输出目录包含 `pickscore-curve.csv/json/png`，记录相对基座的paired差值与区间。生成图在 `evaluations/step-0000400/render` 等目录。主要看：相对基座是否改善；相同步数MVPO对DPO是否有一致改善；视觉质量、文字与构图是否退化。不要用训练loss跨方法大小判断谁更好。

汇总已完成评估：

```bash
python scripts/summarize_flow_runs.py "$SFT_RUN" "$DPO_RUN" "$MVPO_RUN" "$LINEAR_RUN" \
  --reference-run "$DPO_RUN" --out logs/flow-comparison-v1.csv
```

`delta_base`及其区间比较基座，`delta_reference`及其区间比较同一步的DPO；参考步尚未完成时留空。SFT单独标识winner图像预算，不计为偏好pair-views。工具验证完成记录，并拒绝把不同数据、面板、评分器、采样配置、基座或随机种子的同骨干run混成同一比较。64条prompt的区间只反映该面板的不确定性，不是训练seed重复实验，也不是最终论文结论。

中断后直接重跑原脚本，自动latest续训到3200，不追加暂停参数。示例：

```bash
bash launchers/flow/02_sd35_mvpo.sh
```

`training.preflight=once`首次把已检查的清单、模型/源码身份和评分器信息存为run旁的预检快照；后续stage、rank和续训直接读取，不重新审计。保留配置相同、数值失败报错、原子checkpoint提交及必要保存/加载。模型和数据在同一run中保持不变；新配方开新tag。v4增加SFT；已有v1/v2/v3实验继续使用原checkout，新增实验使用新目录。Z-Image的可学习性及如何解释四组结果见[分析说明](ZIMAGE_HPDV3_ASSESSMENT.md)。

## 超参证据与适用边界

[Linear-DPO论文附录D](https://arxiv.org/html/2605.21123v1#A4)给出SD3-M的lr5e-6、global batch128、warmup200、beta500和HPDv3-sub两epochs；[§5.3](https://arxiv.org/html/2605.21123v1#S5.SS3)支持eta=.01和EMA=.995。3200更新是项目采用的预算近似，不等于本项目过滤后数据的精确两epochs。

已发表配方对应SD3-M；本项目是SD3.5M/ZImage、LoRA、BF16 activation和LoRA因子EMA。因此称为“有论文依据的迁移起点”，不称为这两个骨干已经调好的超参。DPO/MVPO先复用共同LR/beta以尽快建立对照，不能由Linear论文推断这两个目标也已达到最佳设置。Flow-Factory继续提供模型接口/工程参考，其小型offline示例不作为收敛步数的证据。
