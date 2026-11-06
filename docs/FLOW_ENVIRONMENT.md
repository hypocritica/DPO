# 独立 mvpo-flow 环境

`bootstrap_flow_env.sh`只从**用户明确指定的既有GPU dpo环境**克隆，默认目标为`<conda-root>/envs/mvpo-flow`。必须显式给Conda根目录和source prefix；本地只有已核实的CPU环境`self_correct_dpo_slai23_cpu`，脚本不会猜测本地存在dpo，也不接受CPU-only源环境。用户确认SLAI仍使用此前self_correct的同一位置；下面直接给既有路径，实际存在性由脚本检查。当前无法连接SLAI，本轮只编辑和dry-run，未安装环境或提交训练。

在目标服务器、本人可写的项目checkout内执行：

```bash
# 用户确认沿用此前SLAI位置；脚本会核对，不使用本地CPU环境。
FLOW_CONDA_ROOT=/mnt/afs/L202500030/xietianyu/miniconda3
FLOW_SOURCE_PREFIX=/mnt/afs/L202500030/xietianyu/miniconda3/envs/dpo

# 无写入预览；不验证远程路径存在或实际空闲空间。
bash scripts/bootstrap_flow_env.sh \
  --conda-root "$FLOW_CONDA_ROOT" --source-prefix "$FLOW_SOURCE_PREFIX" --dry-run

# 确认路径后执行一次；不需要激活或修改base/source。
bash scripts/bootstrap_flow_env.sh \
  --conda-root "$FLOW_CONDA_ROOT" --source-prefix "$FLOW_SOURCE_PREFIX"
```

目标盘不同，可增加`--target-prefix /absolute/owned/path/mvpo-flow`。默认要求该盘至少30GiB空闲，`--min-free-gib N`可按真实环境大小调整；这是最低空闲阈值，不是环境精确容量估计。脚本先执行目标盘`df`，本机存在`/irip`时也检查它。克隆和依赖安装可能超过1GB；空间不足立即停止。新Conda cache使用target旁的`.conda-pkgs`目录，pip cache位于target内，避免向source/base缓存写入。

脚本依次执行：读取源环境包版本及torch编译CUDA版本，保存torch/torchvision/torchaudio、Triton、nvidia CUDA/NCCL等已安装包的精确constraints；克隆至独立prefix；安装`requirements-flow.txt`八项pins；安装本项目及`arrow,evaluation` extras并解析所需依赖（GPU runtime精确constraints继续生效）；`pip check`；SD35及ZImage两个family的离线API/version preflight；再次核对GPU runtime版本完全一致；保存`pip freeze --all`与Conda清单。它不会下载模型、分配GPU、运行CUDA训练或提交Slurm任务。

所有安装仅调用target的Python。source不运行pip，禁用bytecode和user-site写入；source和target不允许重叠或嵌套，也不允许使用Conda base作源或目标。脚本只读源的包元数据和编译CUDA信息，不执行GPU计算。

成功记录写在target的`.mvpo-flow-bootstrap.json`，日志/freeze放于项目`logs/flow-environment/<UTC时间-PID>/`。同一source、源包指纹、项目位置、依赖锁和pyproject身份可重复调用；安装中途出错停止，修正网络等外部问题后可重跑已完成clone的target。来源/锁变化或未知既有target会拒绝，不删除或覆盖已有环境。若clone本身中断，残留目录可能没有有效marker，需检查后选择新prefix；脚本不自动删除它。创建完成不代表GPU训练或生成效果已验证。

## 评分依赖边界

| 评分器 | 本仓库依赖审查结论 |
|---|---|
| PickScore / CLIP aesthetic | 现有AutoModel/AutoProcessor、CLIP特征API在新栈仍有对应接口；此次没有加载真实权重或运行GPU评分，不能据preflight声称评分已验证。 |
| HPSv2 | 使用独立的作者open_clip源码；实际torchvision/timm/权重兼容性仍须首次评分核验。 |
| HPSv3 | `requirements-hpsv3.txt`明确固定transformers4.45.2、PEFT.10、diffusers.33.1、safetensors.5.3，并使用numpy2/Pillow11，和flow训练栈冲突。必须保留独立scorer环境。 |
| ImageReward | `requirements-imagereward.txt`固定transformers4.27.4等旧依赖，和flow的4.57.1冲突；继续独立scorer环境。 |

bootstrap不安装两份scorer requirements，也不自动卸载源环境已有评分包。若源克隆带来的包与flow产生metadata冲突，`pip check`会停止；应查看日志、选择适当训练源环境或在独立target中明确处理，不能为通过检查改旧dpo/base。HPSv3/ImageReward可沿用评分配置的`python`字段调用其独立环境。本次既未验证这些scorer环境，也未运行真实PickScore/HPS/ImageReward；后续把训练与效果检查放到已分配GPU资源即可，无需重跑大量分布式检查。

本轮脚本检查：`bash -n`、默认/自定义target的无写入dry-run、缺失参数和source/target/base边界拒绝。没有执行实际clone、pip安装或GPU检查。
