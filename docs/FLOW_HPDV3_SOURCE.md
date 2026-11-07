# HPDv3：Hugging Face目录与来源记录

`.download-manifest.json` 和 `.scd-download.json` 都是项目下载工具额外生成的文件，
不是原始HPDv3内容，也不是Hugging Face和ModelScope统一要求的格式。
`hf download` / `snapshot_download(local_dir=...)` 可只留下原始文件及 `.cache/huggingface/`，
缺少项目标记不等于图片下载失败。

## 直接准备

在已安装mvpo-flow环境、分配了资源的SLAI容器执行：

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-paper-v2
git pull --ff-only origin feat/flow-stagewise-slai-20261107
export HPDV3_RAW=/mnt/aoss/xietianyu/datasets/hpdv3
bash launchers/flow/prepare_slai_data.sh
```

入口自动使用SLAI资源表和独立环境，按顺序显示source/extract/prepare/audit/panel五个阶段，
出错时打印阶段、行号和退出码，不再由无输出的 `test -f` 终止。
输出及日志路径会在开头打印。默认位置与resources-slai-flow.json一致：

- 图片：`$HPDV3_RAW/flow-sub-images-v2/images`
- 清单：`$HPDV3_SUB_PREPARED/pairs.jsonl`
- 凭据：`$HPDV3_SUB_PREPARED/validation-report.json`
- 面板：`$MONITOR_PROMPTS_FLOW_PILOT`
- 日志：`<checkout>/logs/slai-data/prepare-<UTC时间>-<PID>.log`

已完成提取与准备可复用，再次审计并验证面板。未完成的prepared目录保留并报错；
可通过 `FLOW_PREPARED=/新的目录` 另开一次准备，不删除已有产物。
自定义布局用 `FLOW_IMAGES` / `FLOW_PREPARED` / `FLOW_PANEL`，完成后启动器使用打印的
`--pairs ... --monitor-prompts ...`。每次准备默认4线程，不执行GPU、不下载内容。

## 读取train.json被Killed / exit 137

137表示进程收到SIGKILL；若停在读取train.json且没有Python traceback，优先检查容器RAM限制与OOM事件。
旧实现使用read_text加json.loads同时保留全量文本和解析对象。现提取/metadata筛选均逐条读取JSON数组，
提取只累计所需图片路径；完整准备每256条校验图片，不再保留全部源注释或全部筛选原因列表。
不需要安装新依赖，筛选、原始行号、标签和seed17组件划分保持不变。
解析器每64Ki字符读取，单条注释上限16Mi字符，拒绝截断/尾部内容，每15秒输出扫描进度。

更新后直接重跑上面的入口，保留原始数据和输出目录。本次若停在注释读取，尚未开始提取图片；
若以前已提取部分图片，重跑会重新扫描gzip，复用同大小文件。
完整prepare仍保留合格样本、图片哈希、排除记录与组件划分结构，因此总RAM随数据规模增长，
不是整个流程固定0.34MB。若再次137，查看容器实际限制，按平台方式增加CPU内存；减少图片线程
不能解决旧版全量JSON读取峰值，也不涉及GPU显存或BF16。

可选只读诊断（cgroup v2常见路径，路径不存在时跳过）：

```bash
free -h
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
  if [ -f "$f" ]; then echo "$f"; cat "$f"; fi
done
```

memory.events的oom_kill非零说明该cgroup曾有OOM杀进程，但累计数本身不能唯一归因这一次失败。
实现验证摘要见FLOW_HPDV3_SOURCE_VALIDATION.json；未在该SLAI容器重跑真实全量数据。

## 来源身份与验证范围

优先使用完整且字段有效的已有 `.download-manifest.json` 或 `.scd-download.json`；
已有记录明确标为未完成时拒绝静默绕过。`.scd-download.json` 中记录的文件大小也会核对。

两者都没有时，第一次生成 `.flow-source-inventory.json`：

- SHA256绑定train.json与test.json；分卷记录名称、大小和mtime，要求从00连续编号且非空。
- 来源身份为 `local/HPDv3@local-inventory:<digest>`，状态为 `inventoried`。
- 明确记录 `upstream_revision_verified=false`、`download_complete_verified=false`，不写虚假的complete标记。
- 如有HF缓存commit元数据，只作为观察记录，不把它当作分卷内容或上游版本验证。
- 后续复用仅检查已有inventory的stat；注释文件和分卷变化时拒绝旧身份。

指纹本身不保证所有分卷下载完整。随后真实顺序读取压缩分卷、检查所需图片齐全，
准备阶段进行PIL验证与SHA256，最终审计通过才生成passed训练凭据。
原始下载完成与子集可训练性分别记录：子集验证passed不等于已核验整个上游仓库。

仅匹配 `images.tar.gz.[0-9][0-9]`，`*.incomplete` 残留不读取、不删除。
若缺中间分卷、文件被截断或所需图片缺失，会在相应阶段失败。
train.json首次读取可能较大，现在会先打印读取阶段；第一次注释哈希每15秒报告读取字节进度。

Hugging Face对local_dir缓存元数据的说明：
[Download files to a local folder](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)。
