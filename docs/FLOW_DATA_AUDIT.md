# HPDv3 准备报告与训练凭据

本地开发时的整理由旧 checkout 的 Slurm 作业 `65774` 完成；新代码不改变它或其输出。此处是本地处理记录，SLAI 首次准备请使用 [FLOW_SLAI.md](FLOW_SLAI.md)。
当时的本地进度快照为 `reports/hpdv3-review-20261107.json`，该机器专属产物不提交 Git，也不能用来认证另一台服务器的数据。可随仓库查看的实现验证摘要见 [FLOW_RELEASE_VALIDATION.json](FLOW_RELEASE_VALIDATION.json)。
`in_progress` / `passed=false` 不是可启动训练的凭据。最终 usable pairs 和 split 数量应等待 `audit.json` / `ready.json` 产生后读取。

准备完成后，在 CPU 计算分配中执行一次审计。以下绝对路径复用当前整理产物；使用已具备项目依赖的 Python 环境：

```bash
cd /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow-stagewise
python scripts/audit_flow_data.py \
  --pairs /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/data/hpdv3-sub-v1/pairs.jsonl \
  --extraction /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/data/hpdv3-sub-images-v1/extraction.json \
  --raw /irip_16t_0/xietianyu_2024/datasets/hpdv3 \
  --preparation-log /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/logs/local-a800/prepare-parallel-65774.log \
  --job-id 65774
```

默认原子写入 `pairs.jsonl` 同目录的 `validation-report.json`，新 flow 启动器自动使用它。
也可通过 `--out /absolute/report.json` 指定报告位置，再向启动器传入 `--data-report /absolute/report.json`。
退出码为 `0=passed`、`2=in_progress`、`1=failed`。
无准备完成产物时只汇总小凭据和已有日志，不扫描图片；完成后的首次审计需要解析一次完整 manifest。

报告核对清单 SHA256、schema、图片对去重/split 隔离、来源 revision、准备数量、官方 test caption/UID 隔离以及 path1=winner 标签顺序。图片字节完整性和官方 test 图片 hash 排除复用准备过程证据；不会重读全部图片，也不会重读完整原始 train.json 验证每个 original_record。报告明确记录这些边界。当前本地准备没有传入 Pick-a-Pic heldout，所以报告不声称已完成该排除；未知语义改写也没有被认证隔离。

`passed` 报告绑定 pairs / ready / audit 的绝对路径、大小、mtime_ns、SHA256。训练启动仅检查文件 stat 和小 ready 文件的 SHA256，然后加载必需训练行；不重新计算完整清单 hash、不重新做全局 schema/split 审计，也不启动图片抽查。训练消费图片时继续校验各图片 SHA256。普通修改会使报告失效；这套本地凭据依赖准备文件保持不变，不提供对刻意保留 stat 的篡改认证。
