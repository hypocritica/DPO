# Self Correct DPO

SD1.5 / SDXL LoRA DPO。默认使用独立 rollout 多视图 C；也提供多视图 B 和纯 DPO。支持任意正整数 K、单机 2/4/8 卡、Accelerate 启动、TensorBoard/W&B，以及每 200 步完整续训保存。

- [开发与双服务器实验约定](docs/DEVELOPMENT.md)
- [安装、数据准备、启动与续训命令](command.md)
- [统一资源地址](configs/resources.json)
- [公共默认超参数](configs/defaults.json)
- [数据说明](docs/DATA.md)、[方法公式](docs/METHOD.md)、[评估说明](docs/EVALUATION.md)
- [算法审查与逐轮改进计划](docs/ALGORITHM_REVIEW_AND_PLAN.md)

资源路径在 `configs/resources.json` 中统一修改；私有覆盖用 `configs/resources.local.json`，无需 `source paths.sh`。训练需要已处理的 `pairs.jsonl`。原始 Pick-a-Pic 地址已设置为 `/mnt/aoss/xietianyu/datasets/pickapicv2/data`。SDXL、SD3、HPDv3 的位置已预留，SD3 训练后续实现。

```bash
# 在远程已分配GPU上运行；本集群登录节点不可运行训练。
python train.py --family sd15 --world-size 4 \
  --output_dir /mnt/aoss/xietianyu/self_correct/runs/sd15-C-k2-v1 \
  --execution standalone --run-label sd15-C-k2-v1

# 被驱逐并重新分配到相同卡数后，使用原参数加：
# --resume_from_checkpoint latest
```

三种方法分别在 `configs/dpo.json`、`configs/multiview-B.json`、`configs/multiview-C.json`；省略 `--config` 默认 C，`--views 3` 改为 K=3。命令参数覆盖默认设置，`--print-config` 可只查看最终配置。

```text
src/self_correct_dpo/
  train.py, training.py       训练入口与优化循环
  config.py, resources.py    参数合成与统一资源地址
  models.py, architecture.py 模型、LoRA、SD1.5/SDXL组件
  algorithms/                DPO、B/C、rollout与实验算法
  data/                      偏好对、数据读取与固定评估面板
  runtime/                   DDP、保存恢复、评估调度与日志
  evaluation/                图像生成、评分模型与指标
```

历史配置归档到 `docs/legacy/configs/`，旧实验仍使用冻结源码复现。当前版本对代码、依赖、配置和数据做续训一致性检查；更新代码前先结束旧实验或保留其 checkout。

常规只运行与改动有关的现有轻量检查，例如：

```bash
python -m unittest discover -s tests -p test_stage_pause.py -v
python -m unittest discover -s tests -p test_joint.py -v
```
