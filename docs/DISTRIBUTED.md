# 单机多卡 DDP

SD1.5/SDXL 的 B/C、K=1/2、model/oracle 扰动和普通噪声视图均使用同一训练入口。`training.world_size` 为单节点GPU数，每卡一个进程，支持2/4/8卡。CPU/Gloo测试覆盖多进程同步、阶段切换、恢复与故障处理，具体记录见[VALIDATION.md](VALIDATION.md)。默认模板仍是单卡。真实GPU/NCCL与SLAI镜像需要在实际分配的资源上另做小规模验证。

## 全局 batch 与目标

```text
global_pair_batch = world_size * batch_size_per_rank * gradient_accumulation_steps
```

`make_config.py --world-size` 默认保留模板的全局batch，自动调整累积步数；显式 `--accumulation` 则使用所给值，也可用 `--effective-batch` 指定全局batch。两者不能同时指定。`--smoke` 使用每卡batch1、累积1、两步和关闭监控。

| 模板 | GPU数 | 每卡batch | 累积 | 全局pair batch |
|---|---:|---:|---:|---:|
| SD1.5 | 1 / 2 / 4 / 8 | 8 | 32 / 16 / 8 / 4 | 256 |
| SDXL | 1 / 2 / 4 / 8 | 1 | 128 / 64 / 32 / 16 | 128 |

每个rank抽取不同图片对，使用 `seed + 1 + rank` 的独立训练generator；同一pair的正负图仍共享噪声和时间，各视图保持独立。C只平均同一pair的视图margin，不把不同rank的margin混合后再算loss。DDP对各rank梯度取均值，从而对应全局batch上的原B/C目标。每个microbatch保留所有视图的策略计算图，对联合loss反传一次；梯度累积期间使用no_sync，最后microbatch完成同步，随后裁剪梯度并更新相同的LoRA。参考冻结与FP32状态/标签定义不因卡数改变。

## 启动

在项目根目录、已分配的同一台GPU机器上运行一次：

```bash
python scripts/make_config.py --family sdxl \
  --model /your/models/sdxl-base --model-revision '<verified-base-revision>' \
  --pairs /your/data/pairs.jsonl --world-size 8 \
  --monitor-prompts /your/data/evaluation/pickapic-monitor.jsonl \
  --scorers-config "$PWD/configs/scorers.json" --out configs/local-sdxl-8gpu.json

export SCD_PYTHON=/your/environment/bin/python
bash launchers/train_ddp.sh configs/local-sdxl-8gpu.json runs/sdxl-C-8gpu-001
```

两卡或四卡分别改成 `--world-size 2` 或 `4`；SD1.5改为 `--family sd15`。`train.sh` 也会按配置自动选择多卡；`train_ddp.sh` 额外检查卡数至少为2。可按已分配资源设置CUDA_VISIBLE_DEVICES，但可见GPU数必须至少等于world_size。脚本不申请资源。

不要对 `train_ddp.sh` 或带自动监控的协调入口再次执行torchrun/accelerate。一个协调进程自行使用 `python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=N --max-restarts=0` 启动训练worker。当前只支持单机，SLAI选择一个worker副本、节点内2/4/8张GPU。

## 训练与评估切换

```text
一个协调进程
  -> torchrun启动N个rank，训练到200/500/1000/...目标步
  -> 完成同步更新、收集所有rank RNG，rank0原子保存完整checkpoint
  -> 全部训练worker退出，释放模型显存
  -> 单个生成进程、逐个评分进程，使用可见的第0张GPU
  -> 提交分数与曲线
  -> 用相同world_size重启全部rank，从checkpoint恢复训练
```

默认监控步数、128条prompt、PickScore定义与LoRA/图像保留节奏不变。评估每个checkpoint只运行一份，其余GPU在此阶段空闲；不会让每个rank重复生成整套prompt。训练worker退出后才开始生成和评分。子进程清除旧RANK/WORLD_SIZE/MASTER/TORCHELASTIC环境，保留原GPU可见范围。

评估失败时不会继续下一训练阶段；重跑相同resume命令先补齐当前评估，再继续训练。生成和评分仍使用各自的提交清单、校验和与断点逻辑。

## checkpoint与恢复

```bash
bash launchers/train_ddp.sh configs/local-sdxl-8gpu.json runs/sdxl-C-8gpu-001 \
  --resume runs/sdxl-C-8gpu-001/checkpoints/latest.json
```

rank0保存一份共享LoRA、AdamW和LR状态，`state.pt.rank_states` 包含全部rank的训练generator、CPU RNG、当前本卡CUDA RNG、Python/NumPy RNG及rank/world身份。DDP参数和optimizer状态在各rank一致，恢复时共享权重/optimizer映射到各自设备，各rank再恢复自己的随机流。日志按rank分文件，checkpoint与曲线只有一个写入者。

默认每200个全局优化步更新完整恢复槽（`training.checkpoint_every=200`），所有rank一起收集状态，只有rank0写入。`checkpoints/latest.json` 始终指向最近提交成功的模型与状态；新checkpoint提交后才清理旧槽。每1000步及最终步另存永久LoRA（`lora_checkpoint_every=1000`），不会因200步滚动清理而删除。评估暂停、正常中断和最终步同样保存，不改变评估频率。

严格恢复要求卡数、每卡batch、累积步数、源码、环境版本、模型、数据和评分配置不变。切换2/4/8卡或改变全局batch应另开实验；不把改变并行布局描述为原实验的精确续跑。永久LoRA不能恢复optimizer，使用latest.json指向的完整恢复槽。

单rank收到TERM/INT后完成当前全局update，通过collective通知所有rank并保存。协调进程向整个子进程组转发信号；每次训练阶段有独立stage_id，`stage-result.json`只在正常返回后记录完成/计划暂停/中断状态，避免把旧状态或torchrun的通用退出码误认成新阶段完成。异常退出、OOM或SIGKILL不提交半个update，保留最近完整checkpoint；torchrun不自动重启并重复消耗实验预算。

## 调试与实际GPU检查

CPU测试通过本项目实际 `JointRolloutBackend`、B/C联合loss反传、DDP、checkpoint与pipeline运行；小型denoiser替代真实UNet。测试对比直接全局目标梯度，覆盖2/4/8rank参数同步、暂停恢复后参数/AdamW/RNG精确一致；8rank同时覆盖B/C，以及评估中断补齐、rank7收到TERM与rank7异常。SDXL双CLIP/UNet/VAE与LoRA的随机小组件测试也包含在完整suite中。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

在已分配的GPU上先做 `--smoke --world-size 2` 的两步暂停/恢复，再检查4卡和8卡；随后启用小面板监控确认GPU训练/评估切换。这里没有申请或提交集群/SLAI GPU作业，也没有测试真实NCCL互联、驱动、显存或5090。不要预设多卡显存可以合并，或预设多卡获得线性加速。
