# Flow B4 / 128 pair-views 实验入口

2026-11-08 新版默认：**所有方法每卡microbatch=4，主实验固定128 pair-views/optimizer update**。
这里B4是每卡偏好对数；梯度累积、全局pairs、pair-views是不同量。
SD3.5默认4×5090，ZImage默认2×H100。所有编号入口默认关闭offload，沿用1024、梯度检查点、VAE chunk1。
SD3.5为BF16权重/activation，ZImage为FP32 transformer权重+BF16 autocast。
这是用户指定的待实测默认，不代表B4已经在全部K/算法/多卡配置上验证通过。

| K | 每步pairs | 每步pair-views | SD3.5累积（4卡×B4） | ZImage累积（2卡×B4） |
|---|---:|---:|---:|---:|
| 1（DPO/LinearDPO） | 128 | 128 | 8 | 16 |
| 2 | 64 | 128 | 4 | 8 |
| 4 | 32 | 128 | 2 | 4 |
| 8 | 16 | 128 | 1 | 2 |

固定pair-view是代理计算预算，实际GPU时间、启动、保存和评估成本仍须另报。
SFT是独立winner-only基线，每步128张winner，不把它宣称为与DPO等计算量。

## 更新与主实验

在上次新建的SLAI目录更新；已运行/准备续训的实验保留原checkout及版本。
本地65803–65806引用的stagewise目录不在本次修改范围内。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-trainonly
git pull --ff-only
```

如果首次部署，见[独立目录与验证报告复用](FLOW_DETACHED_EVAL.md)。不需要重下模型/数据或重建环境。
以下命令只在已分配的对应GPU任务里运行，不提交Slurm或平台作业。

两模型目录提供相同编号，主实验优先级如下：

| 方法 | SD3.5入口 | ZImage入口 |
|---|---|---|
| DPO K1 | `launchers/sd35/01_dpo_k1.sh` | `launchers/zimage/01_dpo_k1.sh` |
| Stagewise MVPO K2 C | `launchers/sd35/07_stagewise_mvpo_k2_views128.sh` | `launchers/zimage/07_stagewise_mvpo_k2_views128.sh` |
| Stagewise MVPO K4 C | `launchers/sd35/08_stagewise_mvpo_k4_views128.sh` | `launchers/zimage/08_stagewise_mvpo_k4_views128.sh` |
| LinearDPO K1 | `launchers/sd35/02_linear_dpo_k1.sh` | `launchers/zimage/02_linear_dpo_k1.sh` |

单独运行任意一项，例如：

```bash
bash launchers/zimage/07_stagewise_mvpo_k2_views128.sh
```

或者在一个任务内按DPO→StageK2→StageK4→LinearDPO依次运行：

```bash
bash launchers/zimage/run_suite.sh main
# SD3.5使用自己的4卡任务：
bash launchers/sd35/run_suite.sh main
```

默认tag统一为flow-b4-v1；编号入口与suite中的同一配方映射到同一输出。
重跑相同代码、配置、tag会通过原启动器跳过完整实验、恢复未完成实验；失败/暂停/取消时停止整个序列。
每个独立实验从同一基座初始化自己的LoRA/optimizer/RNG，因此每项启动加载一次；不会接着前一方法的权重训练。
单项内部持续训练，保存checkpoint不重启、不为在线评估加载模型。另GPU评估见FLOW_DETACHED_EVAL.md。

共同seed17、3200更新、lr5e-6、warmup200、beta500；DPO/MVPO reference=base，LinearDPO reference=LoRA EMA .995、eta .01。
保留每200步完整恢复、每400步永久完整checkpoint及原八点评估LoRA；W&B默认offline与TensorBoard。

## 已配好的消融组

每个case支持单独运行，也可以按组执行；默认不运行消融组。

| 组名 | 配方数 | 内容 |
|---|---:|---|
| main | 4 | DPO、Stage C K2/K4、LinearDPO |
| baselines | 3 | DPO、LinearDPO、winner-only SFT |
| aggregation | 8 | K2/K4 × Global/Stagewise × LossAvg B/MarginAvg C |
| views | 7 | DPO K1，Global C/Stage C的K2/K4/K8，均128 pair-views |
| boundary | 6 | Stage C K2/K4，各boundary250/500/750 |
| beta | 21 | 7种方法各beta100/500/2000，相同调参网格 |
| seeds | 21 | 同7种方法各训练seed17/42/123 |
| pairs | 14 | 可选固定128 pairs，计算量随K增加；不属于主结果的固定计算预算 |
| all | 61 | 全部唯一配方，去掉重复项；资源有限时不要整组启动 |

beta/seeds的7种方法：DPO、Global B/C K2、Stage B/C K2、Stage C K4、LinearDPO。
这些值是预先写好的研究网格，不声称是已调好的最佳超参数；训练seed与生成seed分开。

只读查看（不加载模型、不写配置）：

```bash
bash launchers/zimage/run_suite.sh all --list
bash launchers/zimage/run_suite.sh main --plan-only
```

运行一组或单个case：

```bash
bash launchers/zimage/run_suite.sh aggregation
bash launchers/zimage/run_suite.sh boundary
bash launchers/zimage/run_suite.sh stage-b-k2
bash launchers/zimage/run_suite.sh stage-c-k4-beta2000
bash launchers/zimage/run_suite.sh stage-c-k2-seed42
# 可用逗号选择多个case/组；同一条命令内去重。
bash launchers/sd35/run_suite.sh global-b-k2,global-c-k2
```

beta/seed变体自动在tag后增加对应后缀，避免与默认配方同名；其余算法/K/聚合/边界/预算已编码在原输出名中。
默认网格之外可透传现有运行参数，用新tag固定身份，无需再改代码或Git：

```bash
bash launchers/zimage/run_suite.sh stage-c-k2 --beta 1000 --run-tag beta1000-v1
bash launchers/zimage/run_suite.sh main --batch-size 2 --run-tag b2-views128-v1
bash launchers/sd35/run_suite.sh main --profile h100 --run-tag sd35-h100-v1
bash launchers/sd35/run_suite.sh stage-c-k4 --frozen-placement offload --run-tag offload-k4-v1
```

batch覆盖会自动重算累积以维持128 pair-views；不自动改变K、LR、beta或权重精度，不静默OOM降配。
改配置必须用新tag；冻结检查会拒绝在同一输出中混用超参数。单卡DPO K1 B4通过并不能保证多视图或DDP显存足够。

`--config-only`可生成所选配方的冻结配置，不读取模型权重或启动GPU；`--check-only`走原一次性资产校验。
评估仍从每个训练输出的training.json读取冻结面板与采样参数，case选择不改变评估协议。

## 更多单独编号入口

下列文件在两个模型目录都存在，均默认B4；10/11为原SD3.5边界脚本并补齐ZImage对应入口。

| 文件名 | 含义 |
|---|---|
| 05_global_mvpo_k2_views128.sh | Global C K2 |
| 06_global_mvpo_k4_views128.sh | Global C K4 |
| 09_global_mvpo_B_k2_views128.sh | Global LossAvg B K2 |
| 10_stagewise_mvpo_k2_boundary250.sh / 11_…boundary750.sh | Stage C K2边界消融 |
| 12_global_mvpo_B_k4_views128.sh | Global LossAvg B K4 |
| 13_global_mvpo_k8_views128.sh / 14_stagewise_mvpo_k8_views128.sh | Global/Stage C K8 |
| 15_global_mvpo_B_k8_views128.sh | Global LossAvg B K8 |
| 16_stagewise_lossavg_k2_views128.sh / 17_…k4… / 18_…k8… | Stage LossAvg B K2/K4/K8 |
| 19_stagewise_mvpo_k4_boundary250.sh / 20_…boundary750.sh | Stage C K4边界消融 |
| 04_global_mvpo_k2_pairs128.sh / 21_global_mvpo_k4_pairs128.sh | 固定pairs的Global C K2/K4补充 |
| 22_stagewise_mvpo_k2_pairs128.sh / 23_stagewise_mvpo_k4_pairs128.sh | 固定pairs的Stage C K2/K4补充 |

Stage B只支持Flow DPO：与Stage C采用完全相同的阶段/时间/噪声采样，只把softplus(-mean margin)改为mean(softplus(-margin))。
它用于补齐采样×聚合的2×2消融，不改变已有Stage C或diffusion Huber目标。
CPU检查覆盖两模型全部61配方的真实shell参数与配置构造、预算/输出身份、Stage B损失/梯度与真实tiny PEFT保存恢复。
完整模型GPU训练及新增Stage B的实际生成效果仍须在SLAI验证。
