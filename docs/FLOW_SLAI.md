# SLAI 部署与运行 Flow paper-v2

在新 checkout 操作。编号脚本、公共 Bash 入口、`scripts/run_flow.py` 和实际训练/评估主体均在仓库内；脚本在已分配的 SLAI 容器运行，不申请平台资源。旧 `self_correct_dpo-flow` 的准备作业、环境和实验继续使用原 checkout。

已知远程 shell 起点为 `/mnt/afs/L202500030/xietianyu`；用户列出的资源父目录为 `/mnt/aoss/xietianyu/datasets/{benchmarks,hpdv3,pickapicv2}` 与 `/mnt/aoss/xietianyu/models/{aesthetic,backbones,clip,preference,sd3.5-medium,stable_diffusion,Z-Image}`。这里只确认父目录；尚未连接该服务器核实子目录、模型完整性、HPDv3 文件、评分器或可用空间。下面的输出目录是新规划位置，不表示已经存在。

## 1. Clone 到持久 AFS

本轮发布分支为 `feat/flow-stagewise-slai-20261107`。从用户当前 shell 执行；记录完整 SHA，各训练容器固定使用该 checkout，不在运行中更新源码。

```bash
cd /mnt/afs/L202500030/xietianyu
mkdir -p workspace
git clone --single-branch --branch feat/flow-stagewise-slai-20261107 \
  https://github.com/hypocritica/DPO.git workspace/DPO-flow-paper-v2
cd workspace/DPO-flow-paper-v2
git rev-parse HEAD
test -f launchers/flow/paper_experiment.sh
test -f scripts/run_flow.py
```

默认资源表为 `configs/resources-slai-flow.json`，SD3.5 明确使用 `models/sd3.5-medium`，ZImage 使用 `models/Z-Image`。默认独立产物位置：

| 内容 | 位置 |
|---|---|
| 训练输出、checkpoint、生成评估 | `/mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs/<实验名>` |
| HPDv3 已准备清单 | `/mnt/aoss/xietianyu/datasets/hpdv3/prepared-sub-flow-v2/pairs.jsonl` |
| 两模型共享64条评估面板 | `/mnt/aoss/xietianyu/datasets/benchmarks/flow-paper-v2/hpdv3-monitor-64.jsonl` |
| 模型一次验证报告 | `<checkout>/reports/slai/{sd35_medium,zimage}-model-validation.json` |
| 数据一次审计报告 | `pairs.jsonl` 同目录的 `validation-report.json` |
| 冻结配置 | `<checkout>/configs/local-<实验名>.json` |

若实际模型或原始数据在父目录下另有子目录，先查看后通过 `configs/resources-slai-flow.local.json` 覆盖对应键。该文件不会进入 Git。环境变量优先于 local 及资源表；不要从旧实验复制整份私有配置或在导出旧路径的 shell 中直接启动新实验。实际 AOSS 挂载需支持本项目的锁、原子 rename 和 checkpoint 文件语义，旧 `self_correct` 的 AOSS 兼容代码仍保留。

## 2. 独立环境安装一次

沿用旧 `self_correct/launchers/acp_common.sh` 的已记录 Conda 安装位置与 `dpo` 源环境；脚本会检查真实存在性。检查空间后从只读源环境克隆，不升级旧训练环境。bootstrap 自带目标盘空间门槛、CUDA 包版本 constraints、模型 API 导入及 `pip check`；不重新下载模型。

```bash
df -h /mnt/afs/L202500030/xietianyu /mnt/aoss/xietianyu
bash scripts/bootstrap_flow_env.sh \
  --conda-root /mnt/afs/L202500030/xietianyu/miniconda3 \
  --source-prefix /mnt/afs/L202500030/xietianyu/miniconda3/envs/dpo

source /mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
conda activate mvpo-flow
which python
python --version
```

安装内容包括固定 flow 模型依赖与本项目、数据/评估 extras；torch、torchvision 和源环境已有 CUDA/NCCL 包精确保持。日志与完整 freeze 在 `logs/flow-environment/<时间-PID>/`。若源环境包含与新栈冲突的评分包，安装/`pip check` 会失败并保留日志，应在新 target 环境解决，不修改旧源环境。详见 [FLOW_ENVIRONMENT.md](FLOW_ENVIRONMENT.md)。HPSv3/ImageReward 使用独立 scorer 环境；默认 flow 评估只使用 PickScore，其实际 `preference/pickscore` 和 `clip/clip-h14` 子路径也须核对。

编号入口在上述 SLAI Conda 存在时自动加载并激活 `mvpo-flow`；其他机器使用当前 Python，可显式传入本机 `--resources FILE`。自定义 prefix 时设 `SCD_FLOW_PYTHON=/absolute/prefix/bin/python`；或设置 `SCD_FLOW_CONDA_SH` / `SCD_FLOW_ENV`。`--config-only` 使用当前 Python，不自动激活默认远程环境。独立环境缺失时应先完成 bootstrap，不能将当前 Python 的启动成功视作环境验证通过。

## 3. 检查实际路径与模型一次验证

在新 shell 激活环境后加载新资源表。只有下列检查真实成功才能认为服务器资源已验证；仓库中 `/irip...` 的本地报告不能复用到 SLAI。

```bash
export SCD_RESOURCES="$PWD/configs/resources-slai-flow.json"
source paths.sh
ls -ld "$SD35_MODEL" "$ZIMAGE_MODEL" "$HPDV3_RAW"
find "$HPDV3_RAW" -maxdepth 2 -type f \
  \( -name 'train.json' -o -name 'test.json' -o -name 'images.tar.gz.*' -o -name '.*download*.json' \) -print
test -f "$SD35_MODEL/model_index.json"
test -f "$ZIMAGE_MODEL/model_index.json"
test -d "$PICKSCORE_MODEL"
test -d "$PICKSCORE_PROCESSOR"
mkdir -p reports/slai
python scripts/validate_flow_model.py --family sd35_medium \
  --model "$SD35_MODEL" --out reports/slai/sd35_medium-model-validation.json
python scripts/validate_flow_model.py --family zimage \
  --model "$ZIMAGE_MODEL" --out reports/slai/zimage-model-validation.json
```

验证读取配置、tokenizer、safetensors header/分片索引与文件 stat，不载入完整权重，不运行 CUDA。缺少可核实上游下载凭据时生成明确标识的 `local-inventory` 身份；不冒充上游 revision 已验证。每模型只需一次 passed 报告，多组训练复用；资产变化后重新验证并用新实验名。模型可只读，报告写在 AFS 项目目录。

## 4. HPDv3 子集准备与数据审计一次

完整解包、图片校验和数据扫描放在已分配 CPU/GPU 容器，不能在本集群登录节点执行。先查看上一步原始文件与下载凭据：当前提取脚本明确要求 HPDv3 根下有 `train.json`、`test.json`、`images.tar.gz.00` 等分卷及 `.download-manifest.json`，不能仅用 `.scd-download.json` 代替该文件。父目录存在不代表这些文件齐备；若服务器采用别的布局，先调整 raw 路径或整理实际文件，不填造 revision 或完成凭据。

以下命令先读取真实 `.download-manifest.json` 并检查 complete 状态，将 `repo_id@revision` 作为准备来源身份；不能只传裸 SHA，否则最终审计拒绝。缺少完成凭据时在大规模处理前退出。图片选择与正式训练清单属于新输出，不覆盖原始数据。准备耗时和存储可能较大，应先检查空间；脚本有提取容量/剩余空间门槛。

```bash
set -eo pipefail
FLOW_IMAGE_VIEW="$HPDV3_RAW/flow-sub-images-v2"
HPDV3_REVISION="$(python - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ['HPDV3_RAW']) / '.download-manifest.json'
if not path.is_file():
    raise SystemExit(f'Missing real download receipt: {path}')
receipt = json.loads(path.read_text())
if not (receipt.get('status') == 'complete' or receipt.get('complete') is True):
    raise SystemExit('HPDv3 download receipt is incomplete; stop before extraction')
repo, revision = receipt.get('repo_id'), receipt.get('revision')
if not isinstance(repo, str) or not repo or not isinstance(revision, str) or not revision:
    raise SystemExit('HPDv3 download receipt lacks repo_id/revision')
print(repo + '@' + revision)
PY
)"
df -h "$HPDV3_RAW"
mkdir -p logs/slai-data
python -u scripts/extract_hpdv3_sub.py --raw "$HPDV3_RAW" \
  --out "$FLOW_IMAGE_VIEW" --workers 4 \
  2>&1 | tee -a logs/slai-data/extract-flow-v2.log
python -u scripts/prepare_hpdv3_sub.py --raw "$FLOW_IMAGE_VIEW" \
  --out "$HPDV3_SUB_PREPARED" --revision "$HPDV3_REVISION" --workers 4 \
  2>&1 | tee -a logs/slai-data/prepare-flow-v2.log
python scripts/audit_flow_data.py --pairs "$HPDV3_SUB_PREPARED/pairs.jsonl" \
  --extraction "$FLOW_IMAGE_VIEW/extraction.json" --raw "$HPDV3_RAW" \
  --preparation-log logs/slai-data/prepare-flow-v2.log
python scripts/prepare_flow_monitor.py --pairs "$HPDV3_SUB_PREPARED/pairs.jsonl" \
  --out "$MONITOR_PROMPTS_FLOW_PILOT" --count 64 --seed 17
```

只需准备一份，两个骨干共用。已完成的提取可直接复用；若 `pairs.jsonl/audit.json/ready.json` 与面板已经存在，不重复执行 prepare/monitor（它们拒绝覆盖），直接对实际产物路径审计，并将资源表指向它们。完整审计一次解析 manifest，复用准备时图片校验/官方 test 排除证据，报告明确核验范围。退出码 `0=passed`、`2=in_progress`、`1=failed`；只有 passed 可启动新实验。本地65774正在准备的数据及其处理中快照与本服务器无关，不能代替该审计。

## 5. 检查配置与启动

CPU 配置检查不会加载模型、审计图片或训练。可写临时 runs/config 路径，避免生成正式冻结配置后才补报告 revision：

```bash
bash launchers/sd35/07_stagewise_mvpo_k2_views128.sh --config-only \
  --runs-root /tmp/flow-config-preview/runs --config-dir /tmp/flow-config-preview/configs
bash launchers/zimage/07_stagewise_mvpo_k2_views128.sh --config-only \
  --runs-root /tmp/flow-config-preview/runs --config-dir /tmp/flow-config-preview/configs
```

以下命令各自填入一个已分配对应 GPU 的 SLAI 容器。启动器按自身位置进入 checkout，自动环境、资源表及报告路径；不在每次启动重做完整扫描。默认直接进行3200步正式实验，从正常日志观察速度。

```bash
# SD3.5：默认4×5090；也可追加 --profile h100 或 --profile a800 使用双卡。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2/launchers/sd35/07_stagewise_mvpo_k2_views128.sh
```

```bash
# ZImage：默认2×A800；可追加 --profile h100 使用双卡。
exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2/launchers/zimage/07_stagewise_mvpo_k2_views128.sh
```

每模型九组编号如下，同一脚本名在 `launchers/sd35/` 和 `launchers/zimage/` 均存在：

| 编号脚本 | 方法 | K | 每步图片对 | 每步 pair-views |
|---|---|---:|---:|---:|
| `01_dpo_k1.sh` | DPO | 1 | 128 | 128 |
| `02_linear_dpo_k1.sh` | Linear-DPO | 1 | 128 | 128 |
| `03_sft_k1.sh` | winner-only SFT | 1 | 128张winner | — |
| `04_global_mvpo_k2_pairs128.sh` | Global C固定pair | 2 | 128 | 256 |
| `05_global_mvpo_k2_views128.sh` | Global C固定view | 2 | 64 | 128 |
| `06_global_mvpo_k4_views128.sh` | Global C固定view | 4 | 32 | 128 |
| `07_stagewise_mvpo_k2_views128.sh` | Stagewise C | 2 | 64 | 128 |
| `08_stagewise_mvpo_k4_views128.sh` | Stagewise C | 4 | 32 | 128 |
| `09_global_mvpo_B_k2_views128.sh` | Global B消融 | 2 | 64 | 128 |

SD3.5另有 `10_stagewise_mvpo_k2_boundary250.sh`、`11_stagewise_mvpo_k2_boundary750.sh`。将上面启动命令末尾的文件名替换成表中对应脚本，每组是独立任务。ZImage 四5090必须显式 `--profile 5090 --weight-dtype bfloat16`，作为独立精度实验；默认 A800/H100 transformer 为 FP32，未实测显存/性能。所有正式入口 microbatch=1，累积自动匹配128预算；四5090 K1/K2/K4固定view累积32/16/8，双卡为64/32/16，固定pair K2分别32/64。

本轮20个入口仅含 K1/K2/K4 及 SD3.5 边界消融；K6 不列入固定128 pair-view配方，因为128不能整除 K6 与2/4卡的预算因子，脚本也不会取整凑预算。

相同命令重启自动续 latest；暂停可加 `--stop-after-step 2`，恢复去掉该参数，仍使用原冻结配置和预算。不同 seed 或重复实验用新 `--run-tag`。不能在旧输出中改 profile、权重精度、数据/报告路径、配置或源码。

每200步滚动完整状态、每400步永久完整状态，基座及200/400/800/1300/1800/2300/2800/3200评估；非400评估点的adapter不是完整恢复点。没有额外 policy EMA，只有 Linear-DPO 算法内 EMA reference。启动器写明最终 Config/Output，训练日志与评估结果落在独立输出目录；平台控制台日志由平台保存。复用报告仍会廉价核对绑定及消费图片 SHA256；不会把历史 `/irip` 报告改路径后当作远程 passed 凭据。

本地检查已通过20入口×3 profile的60组真实 shell 参数映射与 Python 配置冻结、2组默认 profile、显式本机资源/报告覆盖、冻结 seed 篡改拒绝、4组非法参数拒绝及 Conda hook 的 nounset 兼容；全部相关 Bash 语法、bootstrap 无写入 dry-run 与 diff 空白检查通过。60组检查通过轻量解释器记录 shell 参数，再调用真实 `run_flow.main()`，没有运行训练；另直接调用两模型 Stagewise K2 的 Bash→Python `--config-only`。尚未在 SLAI 安装环境、验证真实资产或运行 GPU；不据 CPU 检查承诺完整模型显存和吞吐。
