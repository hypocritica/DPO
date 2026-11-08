# Flow正式训练与独立GPU评估

2026-11-08起，本分支的paper-v2默认连续训练，不在评估点结束训练进程。
正常不中断的一次训练，每个rank只构造一次模型；保存checkpoint继续沿用当前进程。
进程退出、抢占或主动暂停后的恢复仍然需要重新加载。

## 保持已有任务不变

本地65803–65806在检查时均PENDING，WorkDir/Command引用self_correct_dpo-flow-stagewise。
本次修改仅在self_correct_dpo-flow-wandb内的新分支feat/flow-detached-eval-20261108进行。
不修改旧checkout、冻结配置、预检快照、环境或Slurm作业。旧排队任务仍按原流程执行。
不要把新版源码覆盖到这些作业的目录，也不要用新版续训旧输出；新任务使用独立checkout及新tag。

SLAI也建议新建源码目录，复用已有模型、数据和环境。以下在任务外准备即可，不加载模型：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace
git clone --depth 1 --single-branch --branch feat/flow-detached-eval-20261108 \
  https://github.com/hypocritica/DPO.git DPO-flow-trainonly
cd DPO-flow-trainonly
mkdir -p reports/slai
cp ../DPO-flow-paper-v2/reports/slai/*-model-validation.json reports/slai/
```

只复制现有验证报告，不下载模型或重建数据。资源表仍指向原AOSS模型、数据和输出根目录。
如有私有资源覆盖，启动命令用--resources显式指定原配置；不要改旧作业使用的文件。

## 连续训练

在已分配的训练GPU任务里运行原编号入口即可，不再需要手加--no-evaluation：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-trainonly
bash launchers/zimage/01_dpo_k1.sh --run-tag flow-trainonly-v1
# SD3.5对应：bash launchers/sd35/01_dpo_k1.sh --run-tag flow-trainonly-v1
```

新输出名自动增加-trainonly，避免与旧inline评估实验同名；原冻结配置继续拒绝不一致覆盖。
两模型、DPO/MVPO B/C/Stagewise/LinearDPO/SFT的编号入口都通过paper-v2配置继承此默认。
训练预算、精度、batch、seed、优化器、W&B/TensorBoard训练日志保持原配方。
--batch-size可在完成显存/吞吐测量后自行选择，不因本次评估解耦自动改变。

保存仍为：

- 每200步更新完整恢复状态；每400步永久保留完整checkpoint。
- 保留原LoRA评估点200、400、800、1300、1800、2300、2800、3200。
- 保存时不退出训练进程。无需评分器或监控面板就绪才能训练。

evaluation.enabled=false，但面板路径、采样参数、seed和评估时间点仍保留在training.json中。
--no-evaluation继续有效；如确需旧分段方式，可显式--inline-evaluation。
paper-pilot等旧配方默认不在本次改动范围内。

## 另一张GPU上的评估

训练启动并发布training.json后，将训练入口打印的Output完整路径作为--run。
评估任务可晚于训练启动，也可等训练结束后再运行；--watch会等待training.json和后续快照。
只需一个进程及一张已分配的评估GPU，不用torchrun、不占用训练作业的GPU。

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-trainonly
bash launchers/flow/evaluate.sh \
  --run /mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs/实际训练输出目录 \
  --watch --report-to all
```

单进程加载一次生成pipeline、一次PickScore，随后只读取并切换小LoRA。
基座评估在禁用LoRA的上下文运行；各快照之间不合并权重，清理旧adapter，重建采样scheduler。
默认沿用冻结采样设置（包括cpu_offload），显存紧时可用--scorer-device cpu，另开--out区分该变体。
GPU模型仍可能在CPU/GPU之间移动，但不会为每个检查点从AOSS重新构造基座。

只读取已原子发布的checkpoints/step-N目录，验证checkpoint manifest、run ID和文件校验和。
不读.pending目录，不依赖会轮换删除的latest/resume目录，不取得训练器的锁，不修改其checkpoint。
启动独立校验模型、面板/评分器来源与训练caption隔离；检查点内容变化会拒绝混用结果。
每个点完成后立刻提交结果，重跑跳过完整结果并恢复未完成的图片/评分。
重启评估进程仍需加载一次基座；--watch保持进程存活即可跨多个评估点复用模型。

不加--watch时，仅处理当前可用的连续评估点并报告后续pending；训练完成但缺指定快照会明确报错。
可用--steps 0 400 800只测指定点。只读计划（无GPU加载、无输出写入）：

```bash
bash launchers/flow/evaluate.sh --run /实际训练输出目录 --check-only
```

默认输出是RUN/offline-eval-v1，可用--out指定独立新目录。包含：

```text
evaluation-job.json             # 原训练ID、冻结设置、资产/代码/依赖身份
checkpoint-bindings.json        # 各永久快照的checkpoint ID
status.json                     # 已完成状态、待出现快照、本进程基座加载次数
evaluations/step-N/             # 原生成/评分事务、完整图片与PickScore
pickscore-curve.csv/json/png    # 相对基座的配对比较曲线
```

--report-to all/ wandb/ tensorboard可选，默认none。独立评估使用单独W&B run ID及名字后缀-evaluation，
继承原训练group，记录optimizer_step、PickScore和默认前8张图片；避免异步评估覆盖训练的history。
没有登录或自动上传。评估进程结束后可用现有W&B CLI同步其wandb/offline-run-*目录。
训练的离线段在连续训练期间保持活动，现有安全同步工具会等该段结束才上传，不会因checkpoint自动分段。

## 5090 B4单卡探针

用户B2截图已显示6步passed，allocated18.45GiB、reserved19.26GiB、总容量31.36GiB。
B4尚未实测，更新后的原入口可显式选择：

```bash
bash launchers/flow/probe_sd35_b2_5090.sh --batch-size 4
```

仍为SD3.5 K1/1024/BF16/关闭offload/VAE chunk1，默认6步，每次新tag及独立输出。
这个单卡探针是单独进程，与上面的正式训练入口无关；它启动时仍需加载一次模型。

CPU验证覆盖真实编号入口的连续/inline配置、一次worker启动、生产训练的多次保存只构造一次backend、
真实tiny PEFT切换两份LoRA及基座复现、原渲染/评分事务在单进程执行和重复跳过、检查点篡改拒绝。
未在SLAI实际GPU验证新的独立评估入口；不把CPU fixture结果宣称为完整模型生成通过。

## ZImage H100 B4单卡探针

在新源码目录、已分配的H100任务里执行：

```bash
bash launchers/flow/probe_zimage_b4_h100.sh
```

默认20步，单卡B4/累积1/DPO K1/1024；沿用正式ZImage的FP32 transformer权重、BF16 autocast、关闭offload、VAE chunk1与梯度检查点，不以降低权重精度换取显存。
复用原debug偏好数据与zimage-model-validation.json；可用--pairs/--model-report覆盖，不重新运行模型验证。
启动只构造一次模型，20步连续更新、最后保存；速度丢弃前2步，报告allocated/reserved峰值、设备容量及包含加载/保存的总耗时。
默认日志logs/zimage-b4-probe/TAG/summary.json、训练输出zimage-b4-resident-1gpu-TAG，OOM保留报告并非零退出。
可用--steps 6缩短更新数；每次单独调用仍需加载一次模型。实际H100是否能容纳B4须由本探针验证。

检查PROBE RESULT=passed、显存余量和稳定吞吐后，双H100正式DPO入口为：

```bash
bash launchers/zimage/01_dpo_k1.sh --batch-size 4 --run-tag h100-b4-trainonly-v1
```

正式配置自动使用累积16，保持2×4×16=128 pair-views/update。单卡探针不包含DDP通信显存，多视图算法还需分别验证自己的峰值。
探针不会自动启动或复用为正式实验；正式入口独立加载一次，之后连续训练保存不重启。
