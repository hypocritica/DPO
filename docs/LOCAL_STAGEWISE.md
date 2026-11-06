# 本地 Stagewise 实验入口

本轮源码位于 `/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow-stagewise`。
旧 `self_correct_dpo-flow` 的数据准备和已提交作业继续使用原源码；不要中途替换旧作业的 checkout。
新目录复用已有模型和数据，不复制权重或图片。`configs/local-stagewise-resources.json`
将新实验输出放到本目录 `runs/local-stagewise`，数据和模型仍引用原绝对路径。

## 验证凭据

- `reports/sd35-model-validation.json`：本地 SD3.5M 的静态验证结果。
- `reports/zimage-model-validation.json`：本地 Z-Image 的静态验证结果。
- `reports/hpdv3-review-20261107.json`：数据准备进度审查快照。`in_progress` 不能作为训练通过凭据。
- 完成准备后，按 [数据审计说明](FLOW_DATA_AUDIT.md) 在 CPU 分配中运行一次审计，生成
  `/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/data/hpdv3-sub-v1/validation-report.json`。

启动读取 passed 凭据和少量身份绑定，不重扫图片或权重。数据清单仍需解析用于取样；训练实际消费图片时核对 SHA256。
修改数据、模型或关键配置后应重新生成凭据。模型静态验证不等于已验证 GPU 显存、CUDA 兼容性或权重数值。
SD3.5M 没有完整上游下载凭据，使用 `local-inventory:` 身份；该身份不声称认证了所有权重的上游 commit。

## 启动示例

所有训练命令仅在已经分配的 GPU 节点/容器执行；脚本本身不申请 Slurm 资源。
下面使用本地 A800 环境。远程 5090/H100 服务器应激活各自匹配的环境，并使用该服务器自己的资源配置和验证凭据。

```bash
cd /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow-stagewise
export SCD_FLOW_PYTHON=/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/.venv-flow-a800/bin/python

# 本地 ZImage：默认 2×A800，Stagewise K2，128 pair-views/update。
bash launchers/zimage/07_stagewise_mvpo_k2_views128.sh \
  --resources configs/local-stagewise-resources.json \
  --model-report reports/zimage-model-validation.json \
  --execution slurm

# 本地 SD3.5M：显式改用 2×A800；不传 profile 时默认 4×5090。
bash launchers/sd35/07_stagewise_mvpo_k2_views128.sh \
  --profile a800 \
  --resources configs/local-stagewise-resources.json \
  --model-report reports/sd35-model-validation.json \
  --execution slurm
```

加 `--config-only --run-tag inspect-v1` 只冻结配置，不启动训练；它不表示数据已经 ready。
正式启动前可用独立 tag 加 `--check-only` 检查凭据与输入是否可用。
新训练使用自己的 `--run-tag`，原参数和 tag 重启时自动从最新完整状态续训；更换精度、预算或分段必须使用新实验。

## 硬件与保存

`--profile 5090` 对应四卡，`--profile h100` / `--profile a800` 对应双卡。
ZImage 的 5090 选项要求同时显式传 `--weight-dtype bfloat16`；它与默认 A800/H100 的 FP32 权重属于不同精度实验，实际显存尚未验证。

每 400 步永久保留可用于评估的完整 checkpoint；每 200 步更新滚动的最新完整恢复状态。
400 步交点复用永久完整 checkpoint，避免重复写入。原有八点评估日程和对应 LoRA 仍保留。
全部方法不额外使用 policy EMA；Linear-DPO 保留其算法规定的 reference EMA。
