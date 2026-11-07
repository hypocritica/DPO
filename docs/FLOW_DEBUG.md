# 数据准备期间的Flow训练预检

ZImage按用户更正使用双H100，SD3.5使用四5090。正式编号入口的ZImage默认也改为H100；
仍可用--profile a800选择双A800。这里的GPU必须通过SLAI平台申请，入口不提交任务。

完整HPDv3准备未完成时，编号paper-v2入口会正确拒绝缺少passed数据报告的输入。
可以同时创建独立小快照，提前检查真实模型加载、1024分辨率前反向、数值、DDP、checkpoint和恢复。
这些预检使用真实原始偏好标签，但不是论文结果，不能接到全量数据上续训。

## 1. 在mvpo-flow环境准备小快照

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
git pull --ff-only origin feat/flow-stagewise-slai-20261107
python scripts/prepare_flow_debug_data.py \
  --raw /mnt/aoss/xietianyu/datasets/hpdv3 \
  --images /mnt/aoss/xietianyu/datasets/hpdv3/flow-sub-images-v2 \
  --out /mnt/aoss/xietianyu/datasets/hpdv3/flow-debug-v1
```

默认128对，流式扫描源注释直到找到足够配对或读到末尾；最多复制256MiB图片，不读取或解压分卷。
旧版默认只扫前20万条，可能漏掉后面的可用对；现在--max-scan 0表示不限前缀，正数可显式限制。
先一次列出已完成图片名称，再用集合匹配注释，避免扫描中对大量缺失路径反复访问AOSS。
跳过缺任一正式图片的样本，不读.partial；从已完成图片复制独立字节并verify，保留原始偏好记录
和selection.jsonl中的原始行号，不写原始文件、现有提取目录或正式准备目录。
筛掉官方test归一化prompt及图片UID；没有全量test图片hash审计，因此报告明确ready_for_paper=false，
不生成可冒充正式数据认证的validation-report.json。准备调用已有组件划分/图片哈希实现。
每1万条或15秒打印debug_scan计数；结果写入与输出目录相邻的flow-debug-v1.availability.json，
成功/不足都保留scanned_rows、stop_reason、skips、图片可用性计数及路径示例。
completed_files_at_start是开始时的图片名称快照，不包含扫描期间新完成的图片。

若扫描后仍不足，先看报告而不是反复提交GPU：

- completed_files_at_start为0或缺images目录：核对--images所在挂载/路径。
- stop_reason=scan_limit：本次只检查了指定前缀，可改--max-scan 0。
- winner_only/loser_only较多：这些候选对的一侧尚未解出；图片数量大不代表完整对数量大。
- neither_image较多：对照examples里的路径与completed_files样例/当前提取范围。

不足时让提取继续后重跑，不需要重跑大解压；失败不会留下可被误用的完成目录。
诊断计数是在模型子集、heldout和重复过滤之后统计，不能当作整个数据集的图片配对统计。
快照完成后不可变，重复命令复用；换数量或重新采样需新--out，并通过FLOW_DEBUG_PAIRS指定新清单。

## 2. 每模型验证一次

使用训练容器的mvpo-flow环境验证真实模型。已有有效报告可以复用。

```bash
mkdir -p reports/slai
python scripts/validate_flow_model.py --family sd35_medium \
  --model /mnt/aoss/xietianyu/models/sd3.5-medium \
  --out reports/slai/sd35_medium-model-validation.json
python scripts/validate_flow_model.py --family zimage \
  --model /mnt/aoss/xietianyu/models/Z-Image \
  --out reports/slai/zimage-model-validation.json
```

## 3. 在对应已分配GPU任务运行50步

两组各自进入项目根目录，使用对应卡型/数量。启动器自动激活既有mvpo-flow。

```bash
# 4×5090：先跑到25步，计划暂停并保存完整状态。
bash launchers/flow/debug_train.sh sd35_medium dpo --stop-after-step 25
# 同一输出继续到50步，验证恢复；不要同时提交这两个命令。
bash launchers/flow/debug_train.sh sd35_medium dpo
```

```bash
# 2×H100：先25步，再恢复至50步。
bash launchers/flow/debug_train.sh zimage dpo --stop-after-step 25
bash launchers/flow/debug_train.sh zimage dpo
```

也可省略暂停参数直接50步。FLOW_DEBUG_STEPS=100可改为100步，范围1..200；这会改变预检身份，
不能在同一轮25步暂停后再改。单轮checkpoint每10步滚动保存，计划暂停和最终步也完整保存。
默认B1、梯度累积1、分辨率1024；SD3.5为BF16训练并offload冻结模块，ZImage保留FP32 transformer、
H100冻结模块resident，与对应正式入口的放置一致。实际GPU显存/数值需要本次远程任务确认。

如要测试K4最大视图开销，另开一个实验：

```bash
bash launchers/flow/debug_train.sh zimage mvpo --views 4 --stagewise --run-tag predata-stagek4-v1
```

算法支持dpo/mvpo/linear_dpo/sft。新的run-tag会创建独立输出；正式与预检输出名不同。
底层run_flow.py保留旧--smoke两步默认，新增--smoke-steps，不改变paper-v2的3200步预算。

默认W&B offline + TensorBoard + JSONL，项目mvpo-flow-debug；只记录训练指标。
本轮不做图片生成、PickScore或64条面板评估，因而不依赖尚在准备的正式评估输入。
需要确认loss/梯度无NaN/Inf、GPU内存、迭代速度、25→50恢复、W&B离线文件确实产生。
它不验证正式128 pair-views的梯度累积吞吐或长训练收益；全量完成后另开正式实验。
不要更改已冻结预检的数据或源码后继续原输出。
